//! Sandboxes reached by wrapping a command: your own, SSH, or a Kubernetes pod.
//!
//! [`CommandSandbox`] is the escape hatch. Anything that can run `sh -c <script>`
//! somewhere — nsjail, bubblewrap, firejail, `lxc exec`, `nerdctl exec`, a wrapper
//! script of your own — becomes a sandbox by saying what to put in front:
//!
//! ```no_run
//! # use agent_harness::sandboxes::sandbox;
//! # use serde_json::json;
//! let ws = sandbox(&json!({
//!     "sandbox": "command",
//!     "wrap": ["bwrap", "--ro-bind", "/", "/", "--bind", "/tmp/w", "/work", "--unshare-all", "sh", "-c"],
//!     "workdir": "/work",
//!     "isolated": true,
//! }));
//! ```
//!
//! All three run every script through a local command ([`Runner`]), so tests can
//! hand them a fake one and check the exact command lines without the CLIs.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use super::base::{ExecResult, ProcessExec, Runner, SandboxBackend, q};
use crate::errors::{Error, Result};
use crate::types::new_id;

pub const COMMAND_URL_OPTION: &str = "";
pub const SSH_URL_OPTION: &str = "host";
pub const KUBERNETES_URL_OPTION: &str = "image";

// ---------------------------------------------------------------------------
// reading a backend's options (shared by the backends in this folder)
// ---------------------------------------------------------------------------

/// A backend's options, checked against the names it takes. The shared
/// lifecycle and workspace options (`workdir`, `keep`, `env`, ...) are always
/// allowed: a backend that can use them reads them, the rest ignore them.
pub(crate) struct Opts<'a> {
    name: &'static str,
    map: &'a Map<String, Value>,
}

fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

impl<'a> Opts<'a> {
    /// Refuse an option the backend does not take, as Python's constructor does.
    pub(crate) fn new(name: &'static str, map: &'a Map<String, Value>, known: &[&str]) -> Result<Self> {
        for k in map.keys() {
            if !known.contains(&k.as_str()) && !super::is_shared_option(k) {
                return Err(Error::config(format!("{name} sandbox: got an unexpected keyword argument '{k}' (it takes {})", known.join(", "))));
            }
        }
        Ok(Self { name, map })
    }

    fn get(&self, k: &str) -> Option<&Value> {
        self.map.get(k).filter(|v| !v.is_null())
    }

    fn wrong(&self, k: &str, what: &str) -> Error {
        Error::config(format!("{} sandbox: `{k}` must be {what}", self.name))
    }

    /// A string (a number is taken as its text). Absent or null: `None`.
    pub(crate) fn text(&self, k: &str) -> Result<Option<String>> {
        match self.get(k) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(Value::Number(n)) => Ok(Some(n.to_string())),
            Some(_) => Err(self.wrong(k, "a string")),
        }
    }

    /// As [`Self::text`], with `default` when the key is not there at all.
    /// Given as null, it is `None` — Python's `memory=None`.
    pub(crate) fn text_or(&self, k: &str, default: Option<&str>) -> Result<Option<String>> {
        if self.map.contains_key(k) { self.text(k) } else { Ok(default.map(str::to_string)) }
    }

    pub(crate) fn flag(&self, k: &str, default: bool) -> Result<bool> {
        match self.get(k) {
            None => Ok(default),
            Some(Value::Bool(b)) => Ok(*b),
            Some(_) => Err(self.wrong(k, "true or false")),
        }
    }

    pub(crate) fn number(&self, k: &str) -> Result<Option<f64>> {
        match self.get(k) {
            None => Ok(None),
            Some(v) => v.as_f64().map(Some).ok_or_else(|| self.wrong(k, "a number")),
        }
    }

    /// A whole number of seconds or the like; `default` when absent, `None` when null.
    pub(crate) fn int_or(&self, k: &str, default: Option<u64>) -> Result<Option<u64>> {
        if !self.map.contains_key(k) {
            return Ok(default);
        }
        Ok(self.number(k)?.map(|n| n.max(0.0) as u64))
    }

    pub(crate) fn list(&self, k: &str) -> Result<Option<Vec<String>>> {
        match self.get(k) {
            None => Ok(None),
            Some(Value::Array(items)) => Ok(Some(items.iter().map(text_of).collect())),
            Some(_) => Err(self.wrong(k, "a list of strings")),
        }
    }

    /// A mapping of strings, in the order given.
    pub(crate) fn mapping(&self, k: &str) -> Result<Option<Map<String, Value>>> {
        match self.get(k) {
            None => Ok(None),
            Some(Value::Object(m)) => Ok(Some(m.iter().map(|(k, v)| (k.clone(), Value::String(text_of(v)))).collect())),
            Some(_) => Err(self.wrong(k, "a mapping of strings")),
        }
    }

    /// The shared `env`, when the options carry it.
    pub(crate) fn env(&self) -> BTreeMap<String, String> {
        match self.get("env") {
            Some(Value::Object(m)) => m.iter().map(|(k, v)| (k.clone(), text_of(v))).collect(),
            _ => BTreeMap::new(),
        }
    }

    /// An option that holds a live SDK object in Python, which Rust has no use for.
    pub(crate) fn refuse_objects(&self, keys: &[&str]) -> Result<()> {
        for k in keys {
            if self.get(k).is_some() {
                return Err(Error::config(format!("{} sandbox: `{k}` is an SDK object in Python; the Rust backend talks to the service itself", self.name)));
            }
        }
        Ok(())
    }
}

/// Where a command given no directory runs: the folder above `workdir`, so a
/// sandbox told no working directory settles on `workdir` itself (it is
/// worked out at start as `pwd` + `/workspace`).
pub(crate) fn above(workdir: &str) -> String {
    match workdir.trim_end_matches('/').rsplit_once('/') {
        Some(("", _)) | None => "/".to_string(),
        Some((p, _)) => p.to_string(),
    }
}

/// Which sandbox to pick up at start: the one the lifecycle asks for, or —
/// the first time only — the one named in the backend's own options
/// (`container=`, `pod=`, `sandbox_id=`). One named in the options is not
/// ours: stopping leaves it running, as Python's `keep` does for it.
pub(crate) struct Given {
    id: Option<String>,
    used: AtomicBool,
}

impl Given {
    pub(crate) fn new(id: Option<String>) -> Self {
        Self { id: id.filter(|s| !s.is_empty()), used: AtomicBool::new(false) }
    }

    /// `(id to pick up, adopted from the options)`, or `None` to create one.
    pub(crate) fn target(&self, attach: Option<&str>) -> Option<(String, bool)> {
        let first = !self.used.swap(true, Ordering::SeqCst);
        match attach {
            Some(a) => Some((a.to_string(), false)),
            None if first => self.id.clone().map(|g| (g, true)),
            None => None,
        }
    }
}

/// What a command that outran its deadline did.
pub(crate) fn timed_out(limit: f64) -> ExecResult {
    ExecResult { returncode: 124, stdout: String::new(), stderr: format!("timed out after {limit:.0}s"), timed_out: true }
}

/// The HTTP client the hosted sandboxes use. Each call sets its own timeout.
pub(crate) fn http_client() -> reqwest::Client {
    crate::llm_providers::base::install_tls();
    reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(20)).build().unwrap_or_default()
}

/// The first `n` characters of a service's answer, for an error message.
pub(crate) fn said(body: &Value, n: usize) -> String {
    let text = match body.get("message").and_then(Value::as_str) {
        Some(m) => m.to_string(),
        None => match body {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        },
    };
    text.trim().chars().take(n).collect()
}

fn lossy(bytes: &[u8], n: usize) -> String {
    String::from_utf8_lossy(bytes).trim().chars().take(n).collect()
}

// ---------------------------------------------------------------------------
// command
// ---------------------------------------------------------------------------

/// Runs every script as `[*wrap, script]`. With the default wrap that is a
/// plain local shell — contained by nothing, and it says so.
pub struct CommandSandbox {
    pub wrap: Vec<String>,
    pub workdir: String,
    isolated: bool,
    proc: ProcessExec,
}

impl CommandSandbox {
    pub fn new(options: &Map<String, Value>, runner: Option<Runner>) -> Result<Self> {
        let o = Opts::new("command", options, &["wrap", "isolated"])?;
        let wrap = o.list("wrap")?.unwrap_or_else(|| vec!["sh".into(), "-c".into()]);
        if wrap.is_empty() {
            return Err(Error::config("a command sandbox needs a `wrap` — the command a shell script is appended to"));
        }
        let Some(workdir) = o.text("workdir")? else {
            return Err(Error::config("a command sandbox needs a `workdir` — the directory, as the wrapped shell sees it, that the agent works in"));
        };
        // What the wrap contains is yours to vouch for.
        let isolated = o.flag("isolated", false)?;
        Ok(Self { wrap, workdir, isolated, proc: ProcessExec::new(runner, o.number("timeout")?.unwrap_or(120.0)) })
    }

    fn argv(&self, script: &str) -> Vec<String> {
        let mut argv = self.wrap.clone();
        argv.push(script.to_string());
        argv
    }
}

#[async_trait]
impl SandboxBackend for CommandSandbox {
    fn name(&self) -> &str {
        "command"
    }

    fn isolated(&self) -> bool {
        self.isolated
    }

    async fn start(&self, _attach_id: Option<&str>) -> Result<String> {
        self.proc.probe(&|s| self.argv(s)).await;
        // The same place every time, so a conversation finds its files again.
        Ok(self.workdir.clone())
    }

    async fn exec(&self, command: &str, cwd: Option<&str>, env: &BTreeMap<String, String>, timeout: f64) -> Result<ExecResult> {
        self.proc.exec(&|s| self.argv(s), command, cwd, env, timeout).await
    }

    async fn read_bytes(&self, path: &str, max_bytes: u64) -> Option<Result<Vec<u8>>> {
        Some(self.proc.read_bytes(&|s| self.argv(s), path, max_bytes).await)
    }

    async fn write_bytes(&self, path: &str, data: &[u8], append: bool) -> Option<Result<()>> {
        Some(self.proc.write_bytes(&|s| self.argv(s), path, data, append).await)
    }
}

/// `sandbox("command", wrap=[...], workdir=...)`.
pub fn build_command(options: &Map<String, Value>) -> Result<Arc<dyn SandboxBackend>> {
    Ok(Arc::new(CommandSandbox::new(options, None)?))
}

// ---------------------------------------------------------------------------
// ssh
// ---------------------------------------------------------------------------

fn expand_user(path: &str) -> String {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_default();
    match path.strip_prefix('~') {
        Some(rest) if !home.is_empty() && (rest.is_empty() || rest.starts_with('/') || rest.starts_with('\\')) => format!("{home}{rest}"),
        _ => path.to_string(),
    }
}

/// A machine over SSH — a VM you already have, a build box, a lab host.
///
/// ```no_run
/// # use agent_harness::sandboxes::sandbox;
/// # use serde_json::json;
/// let ws = sandbox(&json!({"sandbox": "ssh", "host": "agent@build-7", "key": "~/.ssh/agent_ed25519"}));
/// ```
///
/// Authentication is your SSH configuration's business: keys, agents and
/// `~/.ssh/config` all apply, and a password prompt is refused rather than
/// waited on. The machine is a real one, so this does not claim to be isolated
/// unless you say it is (`isolated: true`, for a throwaway VM).
pub struct SshSandbox {
    pub host: String,
    base: Vec<String>,
    isolated: bool,
    start_timeout: f64,
    proc: ProcessExec,
}

impl SshSandbox {
    pub fn new(options: &Map<String, Value>, runner: Option<Runner>) -> Result<Self> {
        let o = Opts::new("ssh", options, &["host", "port", "key", "options", "isolated", "ssh"])?;
        let host = o.text("host")?.unwrap_or_default();
        if host.is_empty() {
            return Err(Error::config("an ssh sandbox needs a `host` — user@machine"));
        }
        let ssh = o.text("ssh")?.unwrap_or_else(|| "ssh".into());
        let mut base = vec![ssh, "-o".into(), "BatchMode=yes".into(), "-o".into(), "ConnectTimeout=20".into()];
        if let Some(port) = o.text("port")?.filter(|p| !p.is_empty() && p != "0") {
            base.extend(["-p".into(), port]);
        }
        if let Some(key) = o.text("key")?.filter(|k| !k.is_empty()) {
            base.extend(["-i".into(), expand_user(&key)]);
        }
        for option in o.list("options")?.unwrap_or_default() {
            base.extend(["-o".into(), option]);
        }
        Ok(Self {
            host,
            base,
            isolated: o.flag("isolated", false)?,
            start_timeout: o.number("start_timeout")?.unwrap_or(300.0),
            proc: ProcessExec::new(runner, o.number("timeout")?.unwrap_or(120.0)),
        })
    }

    /// ssh joins its arguments with spaces and hands them to the remote shell,
    /// so the script has to arrive as one quoted word.
    pub fn wrap(&self, script: &str) -> Vec<String> {
        let mut argv = self.base.clone();
        argv.push(self.host.clone());
        argv.push(format!("sh -lc {}", q(script)));
        argv
    }
}

#[async_trait]
impl SandboxBackend for SshSandbox {
    fn name(&self) -> &str {
        "ssh"
    }

    fn isolated(&self) -> bool {
        self.isolated
    }

    async fn start(&self, _attach_id: Option<&str>) -> Result<String> {
        match (self.proc.runner)(self.wrap("true"), None, self.start_timeout).await? {
            Some((0, _, _)) => {}
            None => return Err(Error::tool(format!("could not reach {} over ssh: timed out after {:.0}s", self.host, self.start_timeout), "sandbox")),
            Some((_, _, err)) => return Err(Error::tool(format!("could not reach {} over ssh: {}", self.host, lossy(&err, 300)), "sandbox")),
        }
        self.proc.probe(&|s| self.wrap(s)).await;
        Ok(self.host.clone())
    }

    async fn exec(&self, command: &str, cwd: Option<&str>, env: &BTreeMap<String, String>, timeout: f64) -> Result<ExecResult> {
        self.proc.exec(&|s| self.wrap(s), command, cwd, env, timeout).await
    }

    /// The machine is not ours to switch off.
    async fn stop(&self, _id: &str) -> Result<()> {
        Ok(())
    }

    async fn read_bytes(&self, path: &str, max_bytes: u64) -> Option<Result<Vec<u8>>> {
        Some(self.proc.read_bytes(&|s| self.wrap(s), path, max_bytes).await)
    }

    async fn write_bytes(&self, path: &str, data: &[u8], append: bool) -> Option<Result<()>> {
        Some(self.proc.write_bytes(&|s| self.wrap(s), path, data, append).await)
    }
}

/// `sandbox("ssh://agent@build-7")`.
pub fn build_ssh(options: &Map<String, Value>) -> Result<Arc<dyn SandboxBackend>> {
    Ok(Arc::new(SshSandbox::new(options, None)?))
}

// ---------------------------------------------------------------------------
// kubernetes
// ---------------------------------------------------------------------------

/// A pod, driven with `kubectl`.
///
/// ```no_run
/// # use agent_harness::sandboxes::sandbox;
/// # use serde_json::json;
/// let ws = sandbox(&json!({"sandbox": "kubernetes", "image": "python:3.12-slim", "namespace": "agents"}));
/// let ws = sandbox(&json!({"sandbox": "kubernetes", "pod": "agent-box-7", "namespace": "agents"})); // one you run
/// ```
///
/// A pod it creates has no service-account token mounted, a CPU and memory
/// limit, and a lifetime (`ttl`) after which it ends on its own. Whether it can
/// reach the network is the cluster's NetworkPolicy to decide, not ours.
pub struct KubernetesSandbox {
    pub image: String,
    pub container: Option<String>,
    pub cpu: String,
    pub memory: String,
    pub ttl: Option<u64>,
    pub labels: Map<String, Value>,
    pub service_account: Option<String>,
    pub workdir: String,
    base: Vec<String>,
    start_timeout: f64,
    given: Given,
    /// The pod in use, and whether it was named in the options (and so is not ours).
    pod: std::sync::Mutex<(Option<String>, bool)>,
    proc: ProcessExec,
}

impl KubernetesSandbox {
    pub fn new(options: &Map<String, Value>, runner: Option<Runner>) -> Result<Self> {
        let o = Opts::new(
            "kubernetes",
            options,
            &["image", "pod", "namespace", "container", "context", "cpu", "memory", "ttl", "labels", "service_account", "kubectl"],
        )?;
        let mut labels = Map::new();
        labels.insert("app.kubernetes.io/managed-by".into(), json!("agent-harness"));
        labels.extend(o.mapping("labels")?.unwrap_or_default());
        let mut base = vec![o.text("kubectl")?.unwrap_or_else(|| "kubectl".into())];
        if let Some(context) = o.text("context")?.filter(|c| !c.is_empty()) {
            base.extend(["--context".into(), context]);
        }
        if let Some(namespace) = o.text("namespace")?.filter(|n| !n.is_empty()) {
            base.extend(["-n".into(), namespace]);
        }
        let given = o.text("id")?.or(o.text("pod")?);
        Ok(Self {
            image: o.text("image")?.unwrap_or_else(|| "python:3.12-slim".into()),
            container: o.text("container")?,
            cpu: o.text("cpu")?.unwrap_or_else(|| "2".into()),
            memory: o.text("memory")?.unwrap_or_else(|| "2Gi".into()),
            ttl: o.int_or("ttl", Some(14_400))?.filter(|t| *t > 0),
            labels,
            service_account: o.text("service_account")?,
            workdir: o.text("workdir")?.unwrap_or_else(|| "/workspace".into()),
            base,
            start_timeout: o.number("start_timeout")?.unwrap_or(300.0),
            given: Given::new(given),
            pod: std::sync::Mutex::new((None, false)),
            proc: ProcessExec::new(runner, o.number("timeout")?.unwrap_or(120.0)),
        })
    }

    /// The pod in use; `None` before start.
    pub fn pod(&self) -> Option<String> {
        self.pod.lock().unwrap_or_else(|p| p.into_inner()).0.clone()
    }

    pub fn wrap(&self, script: &str) -> Vec<String> {
        let mut argv = self.base.clone();
        argv.extend(["exec".into(), "-i".into(), self.pod().unwrap_or_default()]);
        if let Some(c) = &self.container {
            argv.extend(["-c".into(), c.clone()]);
        }
        argv.extend(["--".into(), "sh".into(), "-c".into(), script.to_string()]);
        argv
    }

    /// The pod it creates, as `kubectl run --overrides` takes it.
    pub fn manifest(&self) -> Value {
        let name = self.pod().unwrap_or_default();
        let command = match self.ttl {
            Some(t) => json!(["sleep", t.to_string()]),
            None => json!(["tail", "-f", "/dev/null"]),
        };
        let mut spec = json!({
            "automountServiceAccountToken": false,
            "restartPolicy": "Never",
            "containers": [{
                "name": name, "image": self.image,
                "command": command,
                "workingDir": "/",
                "resources": {"limits": {"cpu": self.cpu, "memory": self.memory}},
                "securityContext": {"allowPrivilegeEscalation": false},
            }],
        });
        if let Some(t) = self.ttl {
            spec["activeDeadlineSeconds"] = json!(t);
        }
        if let Some(sa) = &self.service_account {
            spec["serviceAccountName"] = json!(sa);
        }
        json!({"metadata": {"labels": self.labels}, "spec": spec})
    }

    async fn kubectl(&self, args: &[String], timeout: f64) -> Result<String> {
        let mut argv = self.base.clone();
        argv.extend(args.iter().cloned());
        match (self.proc.runner)(argv, None, timeout).await? {
            None => Err(Error::tool(format!("kubectl {} did not finish in {timeout:.0}s", args[0]), "sandbox")),
            Some((0, out, _)) => Ok(String::from_utf8_lossy(&out).into_owned()),
            Some((_, _, err)) => Err(Error::tool(format!("kubectl {} failed: {}", args[0], lossy(&err, 400)), "sandbox")),
        }
    }
}

#[async_trait]
impl SandboxBackend for KubernetesSandbox {
    fn name(&self) -> &str {
        "kubernetes"
    }

    async fn start(&self, attach_id: Option<&str>) -> Result<String> {
        let pod = match self.given.target(attach_id) {
            Some((pod, adopted)) => {
                *self.pod.lock().unwrap_or_else(|p| p.into_inner()) = (Some(pod.clone()), adopted);
                pod
            }
            None => {
                let pod = format!("agent-harness-{}", new_id(""));
                *self.pod.lock().unwrap_or_else(|p| p.into_inner()) = (Some(pod.clone()), false);
                let overrides = format!("--overrides={}", self.manifest());
                self.kubectl(&["run".into(), pod.clone(), format!("--image={}", self.image), "--restart=Never".into(), overrides], 60.0).await?;
                pod
            }
        };
        let wait = ["wait".into(), "--for=condition=Ready".into(), format!("pod/{pod}"), format!("--timeout={}s", self.start_timeout as i64)];
        self.kubectl(&wait, self.start_timeout + 10.0).await?;
        self.proc.probe(&|s| self.wrap(s)).await;
        Ok(pod)
    }

    async fn exec(&self, command: &str, cwd: Option<&str>, env: &BTreeMap<String, String>, timeout: f64) -> Result<ExecResult> {
        let home = above(&self.workdir);
        self.proc.exec(&|s| self.wrap(s), command, Some(cwd.unwrap_or(&home)), env, timeout).await
    }

    async fn stop(&self, _id: &str) -> Result<()> {
        let (pod, adopted) = self.pod.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if let (Some(pod), false) = (pod, adopted) {
            let args = ["delete", "pod", &pod, "--grace-period=0", "--force", "--wait=false", "--ignore-not-found"];
            self.kubectl(&args.map(str::to_string), 60.0).await?;
        }
        Ok(())
    }

    async fn read_bytes(&self, path: &str, max_bytes: u64) -> Option<Result<Vec<u8>>> {
        Some(self.proc.read_bytes(&|s| self.wrap(s), path, max_bytes).await)
    }

    async fn write_bytes(&self, path: &str, data: &[u8], append: bool) -> Option<Result<()>> {
        Some(self.proc.write_bytes(&|s| self.wrap(s), path, data, append).await)
    }
}

/// `sandbox("k8s://python:3.13")`.
pub fn build_kubernetes(options: &Map<String, Value>) -> Result<Arc<dyn SandboxBackend>> {
    Ok(Arc::new(KubernetesSandbox::new(options, None)?))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::runtime::workspace::Workspace;
    use crate::sandboxes::{Sandbox, SandboxOptions, SandboxWorkspace, WrapOptions, sandbox};

    /// A runner that records every command line and answers with the reply
    /// whose needle appears in it (exit 0, empty output otherwise).
    #[derive(Clone, Default)]
    pub(crate) struct Recorder {
        pub calls: Arc<Mutex<Vec<Vec<String>>>>,
        pub replies: Arc<Vec<(String, Vec<u8>)>>,
    }

    impl Recorder {
        pub(crate) fn with(replies: &[(&str, &str)]) -> Self {
            Self { calls: Arc::default(), replies: Arc::new(replies.iter().map(|(n, r)| (n.to_string(), r.as_bytes().to_vec())).collect()) }
        }

        pub(crate) fn runner(&self) -> Runner {
            let me = self.clone();
            Arc::new(move |argv: Vec<String>, _data, _timeout| {
                let joined = argv.join(" ");
                me.calls.lock().unwrap().push(argv);
                let reply = me.replies.iter().find(|(n, _)| joined.contains(n)).map(|(_, r)| r.clone()).unwrap_or_default();
                Box::pin(async move { Ok(Some((0, reply, Vec::new()))) })
            })
        }

        pub(crate) fn calls(&self) -> Vec<Vec<String>> {
            self.calls.lock().unwrap().clone()
        }
    }

    fn opts(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    #[tokio::test]
    async fn ssh_carries_the_script_as_one_quoted_word() {
        let run = Recorder::default();
        let backend = SshSandbox::new(
            &opts(json!({"host": "agent@build-7", "port": 2222, "key": "~/.ssh/id", "options": ["StrictHostKeyChecking=accept-new"]})),
            Some(run.runner()),
        )
        .unwrap();
        assert!(!backend.isolated()); // a real machine, unless you say otherwise
        let sb = Sandbox::new(Arc::new(backend), SandboxOptions { workdir: Some("/srv/w".into()), ..Default::default() }).unwrap();
        let result = sb.exec("echo 'it works'", None, None, None).await.unwrap();
        sb.stop(false).await.unwrap();

        assert!(result.ok());
        let calls = run.calls();
        let first = &calls[0];
        assert_eq!(first[..5], ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]);
        assert_eq!(first[first.iter().position(|a| a == "-p").unwrap() + 1], "2222");
        assert_eq!(first[first.iter().position(|a| a == "-i").unwrap() + 1], expand_user("~/.ssh/id"));
        assert!(!expand_user("~/.ssh/id").starts_with('~') || std::env::var("HOME").is_err());
        assert!(first.contains(&"StrictHostKeyChecking=accept-new".to_string()));
        assert_eq!(first[first.len() - 2..], ["agent@build-7", "sh -lc true"]);
        let sent = calls.last().unwrap().last().unwrap().clone();
        assert!(sent.starts_with("sh -lc ") && sent.contains("cd /srv/w") && sent.contains("it works"), "{sent}");

        assert!(SshSandbox::new(&opts(json!({"host": "h", "isolated": true})), None).unwrap().isolated());
        assert!(SshSandbox::new(&Map::new(), None).err().unwrap().to_string().contains("needs a `host`"));
    }

    #[tokio::test]
    async fn an_unreachable_host_says_so() {
        let runner: Runner = Arc::new(|_argv, _data, _t| Box::pin(async { Ok(Some((255, Vec::new(), b"Permission denied (publickey)".to_vec()))) }));
        let backend = SshSandbox::new(&opts(json!({"host": "h"})), Some(runner)).unwrap();
        let err = backend.start(None).await.unwrap_err().to_string();
        assert_eq!(err, "could not reach h over ssh: Permission denied (publickey)");
    }

    #[tokio::test]
    async fn kubernetes_makes_a_pod_with_nothing_it_does_not_need() {
        let run = Recorder::default();
        let backend =
            Arc::new(KubernetesSandbox::new(&opts(json!({"image": "python:3.13", "namespace": "agents", "context": "dev"})), Some(run.runner())).unwrap());
        let sb = Sandbox::new(backend.clone(), SandboxOptions::default()).unwrap();
        sb.exec("echo hi", None, None, None).await.unwrap();
        // No working directory named: commands without one run at `/`, so it is /workspace.
        assert_eq!(sb.workdir().await, "/workspace");
        sb.stop(false).await.unwrap();

        let calls = run.calls();
        let pod = backend.pod().unwrap();
        let created = &calls[0];
        assert_eq!(created[..6], ["kubectl", "--context", "dev", "-n", "agents", "run"]);
        assert!(created.contains(&"--image=python:3.13".to_string()) && created.contains(&"--restart=Never".to_string()));
        let overrides = created.iter().find_map(|a| a.strip_prefix("--overrides=")).unwrap();
        let spec = serde_json::from_str::<Value>(overrides).unwrap()["spec"].clone();
        assert_eq!(spec["automountServiceAccountToken"], false);
        assert_eq!(spec["activeDeadlineSeconds"], 14_400);
        let container = &spec["containers"][0];
        assert_eq!(container["name"], json!(pod));
        assert_eq!(container["command"], json!(["sleep", "14400"]));
        assert_eq!(container["resources"]["limits"], json!({"cpu": "2", "memory": "2Gi"}));
        assert_eq!(container["securityContext"], json!({"allowPrivilegeEscalation": false}));

        assert_eq!(calls[1][5..8], ["wait".to_string(), "--for=condition=Ready".into(), format!("pod/{pod}")]);
        assert_eq!(calls[1][8], "--timeout=300s");
        let executed = calls.iter().find(|c| c.last().unwrap().contains("echo hi")).unwrap();
        assert_eq!(executed[5..9], ["exec".to_string(), "-i".into(), pod.clone(), "--".into()]);
        assert_eq!(calls.last().unwrap()[5..8], ["delete".to_string(), "pod".into(), pod]);
    }

    #[tokio::test]
    async fn a_pod_it_was_pointed_at_is_not_deleted() {
        let run = Recorder::default();
        let backend = KubernetesSandbox::new(&opts(json!({"pod": "mine", "container": "main"})), Some(run.runner())).unwrap();
        let sb = Sandbox::new(Arc::new(backend), SandboxOptions::default()).unwrap();
        sb.exec("true", None, None, None).await.unwrap();
        sb.stop(false).await.unwrap();
        let calls = run.calls();
        assert!(!calls.iter().any(|c| c.contains(&"run".to_string()) || c.contains(&"delete".to_string())));
        assert_eq!(calls.last().unwrap()[..6], ["kubectl", "exec", "-i", "mine", "-c", "main"]);
        assert_eq!(sb.id().await, "mine");
    }

    #[tokio::test]
    async fn a_command_sandbox_wraps_and_needs_a_workdir() {
        let run = Recorder::with(&[("command -v timeout", "yes\n")]);
        let backend =
            CommandSandbox::new(&opts(json!({"wrap": ["bwrap", "--unshare-all", "sh", "-c"], "workdir": "/work", "isolated": true})), Some(run.runner()))
                .unwrap();
        assert!(backend.isolated());
        assert_eq!(backend.start(None).await.unwrap(), "/work");
        backend.exec("ls", Some("/work"), &BTreeMap::new(), 7.0).await.unwrap();
        let last = run.calls().last().unwrap().clone();
        assert_eq!(last[..4], ["bwrap", "--unshare-all", "sh", "-c"]);
        // `timeout` was found in the sandbox, so an overrun is killed inside it.
        assert_eq!(last[4], "timeout -s KILL 7 sh -c 'cd /work && ls'");

        let bare = CommandSandbox::new(&opts(json!({"workdir": "/w"})), None).unwrap();
        assert!(!bare.isolated() && bare.wrap == ["sh", "-c"]);
        assert!(CommandSandbox::new(&Map::new(), None).err().unwrap().to_string().contains("needs a `workdir`"));
        assert!(CommandSandbox::new(&opts(json!({"wrap": [], "workdir": "/w"})), None).err().unwrap().to_string().contains("needs a `wrap`"));
    }

    #[test]
    fn options_are_checked_by_name() {
        let err = build_kubernetes(&opts(json!({"no_such_option": 1}))).err().unwrap().to_string();
        assert!(err.starts_with("kubernetes sandbox: got an unexpected keyword argument 'no_such_option'"), "{err}");
        // The shared ones are always allowed.
        assert!(build_ssh(&opts(json!({"host": "h", "workdir": "/w", "read_only": true}))).is_ok());
        assert!(sandbox(&json!("command://x")).err().unwrap().to_string().contains("takes no URL"));
        assert!(build_ssh(&opts(json!({"host": "h", "options": "BatchMode=no"}))).err().unwrap().to_string().contains("a list of strings"));
        assert_eq!(above("/workspace"), "/");
        assert_eq!(above("/home/user/workspace/"), "/home/user");
    }

    #[tokio::test]
    async fn a_local_command_sandbox_moves_bytes_and_times_out() {
        if !crate::runtime::workspace::on_path("sh") {
            return; // needs a POSIX shell (Git for Windows has one)
        }
        let dir = std::env::temp_dir().join(format!("ah_cmd_{}", new_id("")));
        let root = dir.to_string_lossy().replace('\\', "/");
        let backend = build_command(&opts(json!({"workdir": format!("{root}/ws")}))).unwrap();
        let sb = Sandbox::new(backend, SandboxOptions { workdir: Some(format!("{root}/ws")), ..Default::default() }).unwrap();
        let ws = SandboxWorkspace::new(sb, WrapOptions::default());
        // Files travel over the command's stdin and stdout: binary-safe.
        ws.write("a.bin", &[0, 255, 10, 13], false).await.unwrap();
        assert_eq!(ws.read_bytes("a.bin").await.unwrap(), vec![0, 255, 10, 13]);
        assert!(ws.read("gone.txt").await.unwrap_err().to_string().contains("no such file"));
        if cfg!(unix) {
            // (Windows has a `timeout.exe` of its own that a probe may find.)
            let err = ws.shell("sleep 20", Some(1.0)).await.unwrap_err().to_string();
            assert!(err.contains("timed out"), "{err}");
        }
        ws.close().await;
        let _ = std::fs::remove_dir_all(dir);
    }
}
