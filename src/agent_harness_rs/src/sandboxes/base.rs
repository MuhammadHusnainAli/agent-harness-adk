//! The sandbox contract, and the workspace that sits on top of one.
//!
//! A sandbox is somewhere else to run things: a container, a micro-VM, a pod,
//! a machine over SSH. The contract ([`SandboxBackend`]) is deliberately one
//! method wide — `exec` — because everything else can be said in shell.
//! Reading a file is `base64`, writing one is `base64 -d`, listing is `ls`, and
//! "what changed" is `find` and `stat`, all written to run on busybox as well
//! as GNU. A backend with a real file API overrides `read_bytes` and
//! `write_bytes` and gets faster; one without still works on day one.
//!
//! [`SandboxWorkspace`] is what an agent holds. It is a [`Workspace`], so the
//! file tools, the shell tool, `run_python`, `parse_document` and cowork's file
//! collection work on it unchanged.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use base64::Engine;
use futures::future::BoxFuture;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::errors::{Error, Result};
use crate::runtime::workspace::{SCRATCH, Snapshot, Workspace};
use crate::types::{new_id, now};

const MISSING: &str = "__AH_MISSING__";
const TOO_BIG: &str = "__AH_TOO_BIG__";
/// One argument to exec may not exceed 128 KiB on Linux, and the whole command
/// travels as one. 48 KiB of bytes is 64 KiB of base64, comfortably inside it.
const CHUNK: usize = 48 * 1024;
/// How long past its own deadline a command is given before we stop waiting.
const GRACE: f64 = 10.0;

fn sandbox_error(message: impl Into<String>, tool: &str) -> Error {
    Error::tool(message, tool)
}

/// Quote for a POSIX shell, as Python's `shlex.quote`.
pub fn q(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "@%+=:,./-_".contains(c)) {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

/// What a command did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExecResult {
    pub returncode: i32,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

impl ExecResult {
    pub fn new(returncode: i32, stdout: impl Into<String>, stderr: impl Into<String>) -> Self {
        Self { returncode, stdout: stdout.into(), stderr: stderr.into(), timed_out: false }
    }

    pub fn ok(&self) -> bool {
        self.returncode == 0 && !self.timed_out
    }

    fn timed_out(limit: f64) -> Self {
        Self { returncode: 124, stdout: String::new(), stderr: format!("timed out after {limit:.0}s"), timed_out: true }
    }
}

/// `command` with its directory and environment said in shell — for a backend
/// whose exec call takes neither.
pub fn script(command: &str, cwd: Option<&str>, env: &BTreeMap<String, String>) -> String {
    let mut parts = Vec::new();
    if let Some(c) = cwd.filter(|c| !c.is_empty()) {
        parts.push(format!("cd {}", q(c)));
    }
    for (k, v) in env {
        parts.push(format!("export {k}={}", q(v)));
    }
    parts.push(command.to_string());
    parts.join(" && ")
}

/// Somewhere else to run commands. One method is required; the rest have
/// defaults that work over `exec`.
#[async_trait]
pub trait SandboxBackend: Send + Sync {
    /// What this sandbox is called: `docker`, `e2b`, ...
    fn name(&self) -> &str;
    /// Commands here cannot touch the machine the harness runs on.
    fn isolated(&self) -> bool {
        true
    }

    /// Create the sandbox — or, with `attach_id`, connect to that one and fail
    /// if it is not there. Returns its id ("" to have one made up).
    async fn start(&self, _attach_id: Option<&str>) -> Result<String> {
        Ok(String::new())
    }

    /// Run a shell command and wait for it. `cwd = None` is the sandbox's own.
    async fn exec(&self, command: &str, cwd: Option<&str>, env: &BTreeMap<String, String>, timeout: f64) -> Result<ExecResult>;

    /// Destroy the sandbox.
    async fn stop(&self, _id: &str) -> Result<()> {
        Ok(())
    }

    /// A faster way to read a file, if the platform has one. `None`: use shell.
    async fn read_bytes(&self, _path: &str, _max_bytes: u64) -> Option<Result<Vec<u8>>> {
        None
    }

    /// A faster way to write a file, if the platform has one. `None`: use shell.
    async fn write_bytes(&self, _path: &str, _data: &[u8], _append: bool) -> Option<Result<()>> {
        None
    }
}

/// How a sandbox behaves. Every field has a working default.
#[derive(Debug, Clone)]
pub struct SandboxOptions {
    /// The sandbox to pick back up instead of creating one — whatever a run
    /// reported as `result.sandbox_id`.
    pub id: Option<String>,
    /// Where the agent works, inside the sandbox. `None`: a `workspace` folder
    /// under wherever the sandbox starts.
    pub workdir: Option<String>,
    pub env: BTreeMap<String, String>,
    pub timeout: f64,
    pub start_timeout: f64,
    /// Leave it running at `stop()`, to be picked back up later by its id.
    /// One that was itself picked up is kept unless told otherwise.
    pub keep: Option<bool>,
    /// When the sandbox to pick up is gone: `new` starts one in its place, `error` fails.
    pub on_missing: String,
    pub python: String,
}

impl Default for SandboxOptions {
    fn default() -> Self {
        Self {
            id: None,
            workdir: None,
            env: BTreeMap::new(),
            timeout: 120.0,
            start_timeout: 300.0,
            keep: None,
            on_missing: "new".into(),
            python: "python3".into(),
        }
    }
}

#[derive(Debug, Default)]
struct State {
    attach_id: Option<String>,
    id: String,
    workdir: Option<String>,
    keep: bool,
    keep_before: bool,
    replaced: String,
    missing_reason: String,
    started: bool,
    stopped: bool,
    started_at: f64,
}

/// A sandbox with its lifecycle: started on first use, stopped once, picked
/// back up by id. Files and listings are said in shell unless the backend has
/// faster ways. Cheap to clone; clones are the same sandbox.
#[derive(Clone)]
pub struct Sandbox {
    pub backend: Arc<dyn SandboxBackend>,
    pub env: BTreeMap<String, String>,
    pub timeout: f64,
    pub start_timeout: f64,
    pub on_missing: String,
    pub python: String,
    state: Arc<Mutex<State>>,
}

impl std::fmt::Debug for Sandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Sandbox {}>", self.backend.name())
    }
}

impl Sandbox {
    pub fn new(backend: Arc<dyn SandboxBackend>, options: SandboxOptions) -> Result<Self> {
        if options.on_missing != "new" && options.on_missing != "error" {
            return Err(Error::config(format!("on_missing must be \"new\" or \"error\" — got '{}'", options.on_missing)));
        }
        let attach = options.id.filter(|s| !s.is_empty());
        let keep = options.keep.unwrap_or(attach.is_some());
        let state = State { attach_id: attach, workdir: options.workdir, keep, keep_before: keep, ..Default::default() };
        Ok(Self {
            backend,
            env: options.env,
            timeout: options.timeout,
            start_timeout: options.start_timeout,
            on_missing: options.on_missing,
            python: options.python,
            state: Arc::new(Mutex::new(state)),
        })
    }

    pub fn name(&self) -> &str {
        self.backend.name()
    }

    /// The id to pick this sandbox back up with. "" until it has started.
    pub async fn id(&self) -> String {
        self.state.lock().await.id.clone()
    }

    pub async fn workdir(&self) -> String {
        self.state.lock().await.workdir.clone().unwrap_or_else(|| "workspace".into())
    }

    pub async fn running(&self) -> bool {
        self.state.lock().await.started
    }

    pub async fn keep(&self) -> bool {
        self.state.lock().await.keep
    }

    /// Pick up the sandbox with this id when this one starts, and leave it
    /// running afterwards. Only before the first use.
    pub async fn attach(&self, id: &str) -> Result<()> {
        let mut s = self.state.lock().await;
        if s.started {
            return Err(sandbox_error(format!("this {} sandbox is already running as {}; it cannot become another one", self.name(), s.id), "sandbox"));
        }
        s.attach_id = Some(id.into());
        s.keep_before = s.keep;
        s.keep = true;
        Ok(())
    }

    /// Start it, with whatever went wrong said as one kind of error.
    async fn begin(&self, attach: Option<&str>) -> Result<String> {
        match tokio::time::timeout(Duration::from_secs_f64(self.start_timeout), self.backend.start(attach)).await {
            Err(_) => Err(sandbox_error(format!("the {} sandbox did not start within {:.0}s", self.name(), self.start_timeout), "sandbox")),
            Ok(Ok(id)) => Ok(id),
            Ok(Err(e @ (Error::Configuration(_) | Error::Tool { .. }))) => Err(e),
            Ok(Err(e)) => {
                let what = match attach {
                    Some(a) => format!("{} sandbox {a} could not be picked up", self.name()),
                    None => format!("the {} sandbox could not start", self.name()),
                };
                Err(sandbox_error(format!("{what}: {e}"), "sandbox"))
            }
        }
    }

    pub async fn start(&self) -> Result<()> {
        let mut s = self.state.lock().await;
        if s.started {
            return Ok(());
        }
        if s.stopped {
            return Err(sandbox_error(format!("this {} sandbox has been stopped", self.name()), "sandbox"));
        }
        let attach = s.attach_id.clone();
        let id = match self.begin(attach.as_deref()).await {
            Ok(id) => id,
            Err(e @ Error::Configuration(_)) => return Err(e), // a missing key or driver: a new one would fail too
            Err(e) if attach.is_none() || self.on_missing == "error" => return Err(e),
            Err(e) => {
                // The one to pick up is gone. Work goes on in a new one, and the
                // caller is told which.
                s.replaced = attach.clone().unwrap_or_default();
                s.attach_id = None;
                s.missing_reason = e.to_string().chars().take(300).collect();
                s.keep = s.keep_before;
                self.begin(None).await?
            }
        };
        s.id = if !id.is_empty() {
            id
        } else {
            match s.attach_id.clone() {
                Some(a) => a,
                None => new_id("sbx"),
            }
        };
        if s.workdir.is_none() {
            let here = self.backend.exec("pwd", None, &BTreeMap::new(), 30.0).await?;
            let base = here.stdout.trim().lines().last().unwrap_or("").to_string();
            let base = if base.is_empty() { "/".to_string() } else { base };
            s.workdir = Some(format!("{}/workspace", base.trim_end_matches('/')));
        }
        let workdir = s.workdir.clone().unwrap_or_default();
        let made = self.backend.exec(&format!("mkdir -p {}", q(&workdir)), None, &BTreeMap::new(), 30.0).await?;
        if !made.ok() {
            let why: String = if made.stderr.trim().is_empty() { made.stdout.trim() } else { made.stderr.trim() }.chars().take(300).collect();
            return Err(sandbox_error(format!("the {} sandbox started but {workdir} could not be created: {why}", self.name()), "sandbox"));
        }
        s.started_at = now();
        s.started = true;
        Ok(())
    }

    /// Destroy the sandbox, unless it was asked to be kept. Safe to repeat.
    /// `destroy` ends a kept one too — the conversation is over.
    pub async fn stop(&self, destroy: bool) -> Result<()> {
        let mut s = self.state.lock().await;
        if s.stopped {
            return Ok(());
        }
        s.stopped = true;
        let was = s.started;
        s.started = false;
        if was && (destroy || !s.keep) {
            self.backend.stop(&s.id).await?;
        }
        Ok(())
    }

    /// Run a shell command in the sandbox. A timeout is a result, not a failure.
    pub async fn exec(&self, command: &str, cwd: Option<&str>, env: Option<&BTreeMap<String, String>>, timeout: Option<f64>) -> Result<ExecResult> {
        self.start().await?;
        let limit = timeout.filter(|t| *t > 0.0).unwrap_or(self.timeout);
        let workdir = self.workdir().await;
        let cwd = cwd.map(str::to_string).unwrap_or(workdir);
        let mut merged = self.env.clone();
        if let Some(e) = env {
            merged.extend(e.clone());
        }
        match tokio::time::timeout(Duration::from_secs_f64(limit + GRACE), self.backend.exec(command, Some(&cwd), &merged, limit)).await {
            Ok(r) => r,
            Err(_) => Ok(ExecResult::timed_out(limit)),
        }
    }

    async fn sh(&self, script: &str, what: &str) -> Result<ExecResult> {
        let r = self.exec(script, None, None, None).await?;
        if r.timed_out {
            return Err(sandbox_error(format!("{what}: {}", r.stderr), "sandbox"));
        }
        Ok(r)
    }

    fn why(r: &ExecResult) -> String {
        let t = if r.stderr.trim().is_empty() { r.stdout.trim() } else { r.stderr.trim() };
        t.chars().take(300).collect()
    }

    /// The bytes of a file; `Err(Tool "no such file")`-style errors are the caller's to word.
    pub async fn read_bytes(&self, path: &str, max_bytes: Option<u64>) -> Result<Option<Vec<u8>>> {
        self.start().await?;
        let limit = max_bytes.unwrap_or(1 << 62);
        if let Some(fast) = self.backend.read_bytes(path, limit).await {
            return match fast {
                Ok(b) => Ok(Some(b)),
                Err(Error::Io(m)) if m == "missing" => Ok(None),
                Err(e) => Err(e),
            };
        }
        let r = self
            .sh(
                &format!(
                    "if [ ! -f {p} ]; then echo {MISSING}; elif [ \"$(wc -c < {p})\" -gt {limit} ]; then echo {TOO_BIG}; else base64 < {p} 2>/dev/null; fi",
                    p = q(path)
                ),
                &format!("reading {path}"),
            )
            .await?;
        let text = r.stdout.trim();
        if text.contains(MISSING) {
            return Ok(None);
        }
        if text.contains(TOO_BIG) {
            return Err(sandbox_error(format!("{path} is larger than the {limit}-byte limit"), "fs_read"));
        }
        let joined: String = text.split_whitespace().collect();
        base64::engine::general_purpose::STANDARD
            .decode(joined)
            .map(Some)
            .map_err(|_| sandbox_error(format!("could not read {path}: {}", Self::why(&r)), "fs_read"))
    }

    pub async fn write_bytes(&self, path: &str, data: &[u8], append: bool) -> Result<()> {
        self.start().await?;
        if let Some(fast) = self.backend.write_bytes(path, data, append).await {
            return fast;
        }
        let parent = match path.rsplit_once('/') {
            Some(("", _)) | None => "/".to_string(),
            Some((p, _)) => p.to_string(),
        };
        let first = if append { ">>" } else { ">" };
        if data.is_empty() {
            let r = self.sh(&format!("mkdir -p {} && : {first} {}", q(&parent), q(path)), &format!("writing {path}")).await?;
            if !r.ok() {
                return Err(sandbox_error(format!("could not write {path}: {}", Self::why(&r)), "fs_write"));
            }
            return Ok(());
        }
        for (i, chunk) in data.chunks(CHUNK).enumerate() {
            let encoded = base64::engine::general_purpose::STANDARD.encode(chunk);
            let redirect = if i == 0 { first } else { ">>" };
            let prefix = if i == 0 { format!("mkdir -p {} && ", q(&parent)) } else { String::new() };
            let r = self.sh(&format!("{prefix}printf %s {} | base64 -d {redirect} {}", q(&encoded), q(path)), &format!("writing {path}")).await?;
            if !r.ok() {
                return Err(sandbox_error(format!("could not write {path}: {}", Self::why(&r)), "fs_write"));
            }
        }
        Ok(())
    }

    /// `None` when it is not a directory.
    pub async fn list_dir(&self, path: &str) -> Result<Option<Vec<String>>> {
        let r = self.sh(&format!("if [ -d {p} ]; then ls -1Ap {p}; else echo {MISSING}; fi", p = q(path)), &format!("listing {path}")).await?;
        if r.stdout.contains(MISSING) {
            return Ok(None);
        }
        let mut names: Vec<String> = r.stdout.lines().filter(|l| !l.trim().is_empty()).map(str::to_string).collect();
        names.sort();
        Ok(Some(names))
    }

    pub async fn remove(&self, path: &str) -> Result<()> {
        let r = self.sh(&format!("rm -rf -- {}", q(path)), &format!("deleting {path}")).await?;
        if !r.ok() {
            return Err(sandbox_error(format!("could not delete {path}: {}", Self::why(&r)), "fs_delete"));
        }
        Ok(())
    }

    pub async fn exists(&self, path: &str) -> Result<bool> {
        let r = self.sh(&format!("test -e {} && echo yes || echo no", q(path)), &format!("checking {path}")).await?;
        Ok(r.stdout.trim().ends_with("yes"))
    }

    /// Every file under `root` as (a stamp of its modified time, its size) —
    /// the stamp is the CRC-32 of `stat`'s time text, as Python makes it.
    pub async fn snapshot(&self, root: &str) -> Result<Snapshot> {
        let r =
            self.sh(&format!("cd {} && find . -type f ! -path '*/.*' -exec stat -c '%s|%y|%n' {{}} + 2>/dev/null", q(root)), "looking at the files").await?;
        let mut seen = Snapshot::new();
        for line in r.stdout.lines() {
            let (size, rest) = line.split_once('|').unwrap_or((line, ""));
            let (stamp, name) = rest.split_once('|').unwrap_or((rest, ""));
            let Ok(size) = size.trim().parse::<u64>() else { continue };
            if name.is_empty() {
                continue;
            }
            let name = name.strip_prefix("./").unwrap_or(name);
            seen.insert(name.to_string(), (crc32(stamp.as_bytes()) as i64, size));
        }
        Ok(seen)
    }

    pub async fn info(&self) -> Value {
        let s = self.state.lock().await;
        json!({"sandbox": self.name(), "id": s.id, "workdir": s.workdir, "isolated": self.backend.isolated(), "running": s.started, "kept": s.keep, "replaced": s.replaced})
    }

    /// The id this sandbox will pick up when it starts, if any.
    pub async fn attach_id(&self) -> Option<String> {
        self.state.lock().await.attach_id.clone()
    }

    /// The sandbox this one replaced because it was gone, and why.
    pub async fn replaced(&self) -> (String, String) {
        let s = self.state.lock().await;
        (s.replaced.clone(), s.missing_reason.clone())
    }
}

/// CRC-32 (IEEE), as Python's `zlib.crc32`.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for b in data {
        crc ^= u32::from(*b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

// ---------------------------------------------------------------------------
// sandboxes reached through a command line
// ---------------------------------------------------------------------------

/// Runs a local command to completion: `(argv, stdin) -> (exit code, stdout, stderr)`.
/// `Ok(None)` means it outlived its deadline and was killed.
pub type Runner = Arc<dyn Fn(Vec<String>, Option<Vec<u8>>, f64) -> BoxFuture<'static, Result<Option<(i32, Vec<u8>, Vec<u8>)>>> + Send + Sync>;

/// Run a local command to completion: `Some((exit code, stdout, stderr))`, or
/// `None` when it outlived `timeout` and was killed.
pub async fn run_process(argv: Vec<String>, data: Option<Vec<u8>>, timeout: f64) -> Result<Option<(i32, Vec<u8>, Vec<u8>)>> {
    use tokio::io::AsyncWriteExt;
    let mut cmd = tokio::process::Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .stdin(if data.is_some() { std::process::Stdio::piped() } else { std::process::Stdio::null() })
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let mut child = cmd.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound { Error::config(format!("{} is not on PATH", argv[0])) } else { Error::Io(format!("{}: {e}", argv[0])) }
    })?;
    if let (Some(bytes), Some(mut stdin)) = (data, child.stdin.take()) {
        tokio::spawn(async move {
            let _ = stdin.write_all(&bytes).await;
        });
    }
    match tokio::time::timeout(Duration::from_secs_f64(timeout.max(0.1)), child.wait_with_output()).await {
        Ok(Ok(out)) => Ok(Some((out.status.code().unwrap_or(-1), out.stdout, out.stderr))),
        Ok(Err(e)) => Err(Error::Io(format!("{}: {e}", argv[0]))),
        Err(_) => Ok(None),
    }
}

/// The default [`Runner`]: [`run_process`].
pub fn default_runner() -> Runner {
    Arc::new(|argv, data, timeout| Box::pin(run_process(argv, data, timeout)))
}

/// The machinery a sandbox driven through a CLI (`docker exec`, `ssh`,
/// `kubectl exec`) shares: a backend says how a shell script is carried
/// inside (`wrap`), and gets `exec` and binary-safe file transfer over the
/// command's own stdin and stdout.
pub struct ProcessExec {
    pub runner: Runner,
    /// Whether the sandbox has `timeout(1)`, found out at start. With it a
    /// command that overruns is killed inside the sandbox, not just here.
    has_timeout: AtomicBool,
    pub timeout: f64,
}

impl ProcessExec {
    pub fn new(runner: Option<Runner>, timeout: f64) -> Self {
        Self { runner: runner.unwrap_or_else(default_runner), has_timeout: AtomicBool::new(false), timeout }
    }

    /// Find out whether `timeout` exists in the sandbox. Call once it is up.
    pub async fn probe(&self, wrap: &(dyn Fn(&str) -> Vec<String> + Sync)) {
        if let Ok(Some((code, out, _))) = (self.runner)(wrap("command -v timeout >/dev/null 2>&1 && echo yes"), None, 30.0).await {
            self.has_timeout.store(code == 0 && String::from_utf8_lossy(&out).contains("yes"), Ordering::SeqCst);
        }
    }

    pub async fn exec(
        &self,
        wrap: &(dyn Fn(&str) -> Vec<String> + Sync),
        command: &str,
        cwd: Option<&str>,
        env: &BTreeMap<String, String>,
        timeout: f64,
    ) -> Result<ExecResult> {
        let mut body = script(command, cwd, env);
        let killer = self.has_timeout.load(Ordering::SeqCst);
        if killer {
            body = format!("timeout -s KILL {} sh -c {}", (timeout as i64).max(1), q(&body));
        }
        let started = Instant::now();
        let Some((code, out, err)) = (self.runner)(wrap(&body), None, timeout + GRACE / 2.0).await? else {
            return Ok(ExecResult::timed_out(timeout));
        };
        let killed = killer && code == 137 && started.elapsed().as_secs_f64() >= timeout - 0.5;
        Ok(ExecResult {
            returncode: code,
            stdout: String::from_utf8_lossy(&out).into_owned(),
            stderr: if killed { format!("timed out after {timeout:.0}s") } else { String::from_utf8_lossy(&err).into_owned() },
            timed_out: killed,
        })
    }

    pub async fn read_bytes(&self, wrap: &(dyn Fn(&str) -> Vec<String> + Sync), path: &str, limit: u64) -> Result<Vec<u8>> {
        let script = format!("if [ ! -f {p} ]; then exit 44; elif [ \"$(wc -c < {p})\" -gt {limit} ]; then exit 45; else cat {p}; fi", p = q(path));
        match (self.runner)(wrap(&script), None, self.timeout).await? {
            None => Err(sandbox_error(format!("reading {path}: timed out after {:.0}s", self.timeout), "fs_read")),
            Some((44, _, _)) => Err(Error::Io("missing".into())),
            Some((45, _, _)) => Err(sandbox_error(format!("{path} is larger than the {limit}-byte limit"), "fs_read")),
            Some((0, out, _)) => Ok(out),
            Some((_, _, err)) => {
                Err(sandbox_error(format!("could not read {path}: {}", String::from_utf8_lossy(&err).trim().chars().take(300).collect::<String>()), "fs_read"))
            }
        }
    }

    pub async fn write_bytes(&self, wrap: &(dyn Fn(&str) -> Vec<String> + Sync), path: &str, data: &[u8], append: bool) -> Result<()> {
        let parent = match path.rsplit_once('/') {
            Some(("", _)) | None => "/".to_string(),
            Some((p, _)) => p.to_string(),
        };
        let script = format!("mkdir -p {} && cat {} {}", q(&parent), if append { ">>" } else { ">" }, q(path));
        match (self.runner)(wrap(&script), Some(data.to_vec()), self.timeout).await? {
            Some((0, _, _)) => Ok(()),
            None => Err(sandbox_error(format!("writing {path}: timed out after {:.0}s", self.timeout), "fs_write")),
            Some((_, _, err)) => Err(sandbox_error(
                format!("could not write {path}: {}", String::from_utf8_lossy(&err).trim().chars().take(300).collect::<String>()),
                "fs_write",
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// the workspace an agent holds
// ---------------------------------------------------------------------------

/// A workspace whose files and commands live in a sandbox.
///
/// Paths are jailed to the sandbox's working directory by name — `..` cannot
/// climb out of it. That is tidiness rather than security here: the sandbox
/// itself is the wall, and nothing inside it can reach this machine.
pub struct SandboxWorkspace {
    pub sandbox: Sandbox,
    id: String,
    pub read_only: bool,
    allow_shell: AtomicBool,
    pub timeout: f64,
    pub max_file_bytes: u64,
    export: std::sync::Mutex<Option<PathBuf>>,
}

impl std::fmt::Debug for SandboxWorkspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<SandboxWorkspace {}>", self.sandbox.name())
    }
}

/// The options that belong to the workspace wrapped round a sandbox, not to it.
#[derive(Debug, Clone)]
pub struct WrapOptions {
    pub read_only: bool,
    pub allow_shell: bool,
    pub max_file_bytes: u64,
    pub export_dir: Option<PathBuf>,
}

impl Default for WrapOptions {
    fn default() -> Self {
        Self { read_only: false, allow_shell: true, max_file_bytes: 8_000_000, export_dir: None }
    }
}

fn posix_normalise(path: &str) -> String {
    let absolute = path.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|p| *p != "..") {
                    parts.pop();
                } else if !absolute {
                    parts.push("..");
                }
            }
            s => parts.push(s),
        }
    }
    let joined = parts.join("/");
    if absolute {
        format!("/{joined}")
    } else if joined.is_empty() {
        ".".into()
    } else {
        joined
    }
}

impl SandboxWorkspace {
    pub fn new(sandbox: Sandbox, options: WrapOptions) -> Self {
        let timeout = sandbox.timeout;
        Self {
            sandbox,
            id: new_id("ws"),
            read_only: options.read_only,
            allow_shell: AtomicBool::new(options.allow_shell),
            timeout,
            max_file_bytes: options.max_file_bytes,
            export: std::sync::Mutex::new(options.export_dir),
        }
    }

    /// The working directory inside the sandbox.
    pub async fn root(&self) -> String {
        self.sandbox.workdir().await
    }

    /// Resolve inside the working directory, by name.
    pub async fn resolve(&self, path: &str) -> Result<String> {
        let root = posix_normalise(&self.root().await);
        let joined = if path.starts_with('/') { path.to_string() } else { format!("{root}/{path}") };
        let target = posix_normalise(&joined);
        if target != root && !target.starts_with(&format!("{}/", root.trim_end_matches('/'))) {
            return Err(sandbox_error(format!("path '{path}' is outside the workspace"), "workspace"));
        }
        Ok(target)
    }

    /// Where a file is, for something that cannot be handed a local path.
    pub async fn uri(&self, path: &str) -> Result<String> {
        let target = if path.is_empty() { self.root().await } else { self.resolve(path).await? };
        Ok(format!("{}://{}{target}", self.sandbox.name(), self.sandbox.id().await))
    }

    fn writable(&self) -> Result<()> {
        if self.read_only { Err(sandbox_error("this workspace is read-only", "workspace")) } else { Ok(()) }
    }

    /// Start the sandbox now rather than on first use.
    pub async fn start(&self) -> Result<()> {
        self.sandbox.start().await
    }

    /// End the sandbox even if it was being kept for later.
    pub async fn destroy(&self) -> Result<()> {
        self.sandbox.stop(true).await
    }

    /// The id to pick this sandbox back up with. "" until it has started.
    pub async fn sandbox_id(&self) -> String {
        self.sandbox.id().await
    }

    /// Start the sandbox and prove each thing an agent will ask of it.
    /// Never fails: every step and whether it held is in the answer.
    pub async fn check(&self) -> Value {
        let started = Instant::now();
        let mut steps: Vec<Value> = Vec::new();
        let probe = format!(".ah-check-{}", new_id(""));
        let payload: Vec<u8> = (0..4).flat_map(|_| 0u8..=255).collect();
        macro_rules! step {
            ($name:expr, $body:expr) => {{
                let began = Instant::now();
                let outcome: Result<String> = async { $body }.await;
                let ok = outcome.is_ok();
                let detail = match outcome {
                    Ok(d) => d,
                    Err(e) => format!("{}: {e}", e.kind_name()),
                };
                steps.push(json!({"step": $name, "ok": ok, "detail": detail, "seconds": (began.elapsed().as_secs_f64() * 100.0).round() / 100.0}));
                ok
            }};
        }
        let sb = &self.sandbox;
        if step!("start", {
            sb.start().await?;
            Ok(sb.id().await)
        }) {
            let root = self.root().await;
            let dir = format!("{root}/{probe}");
            step!("exec", {
                let r = sb.exec("echo sandbox-ok", None, None, None).await?;
                if !r.stdout.contains("sandbox-ok") {
                    return Err(sandbox_error(format!("exit {}: {}", r.returncode, Sandbox::why(&r).chars().take(200).collect::<String>()), "sandbox"));
                }
                Ok(format!("in {root}"))
            });
            step!("write and read", {
                sb.write_bytes(&format!("{dir}/dir/bytes.bin"), &payload, false).await?;
                let back = sb.read_bytes(&format!("{dir}/dir/bytes.bin"), None).await?.unwrap_or_default();
                if back != payload {
                    return Err(sandbox_error(format!("wrote {} bytes, read back {}", payload.len(), back.len()), "sandbox"));
                }
                Ok(format!("{} bytes, binary-safe", payload.len()))
            });
            step!("list", {
                let names = sb.list_dir(&dir).await?.unwrap_or_default();
                if names != vec!["dir/".to_string()] {
                    return Err(sandbox_error(format!("expected ['dir/'], got {names:?}"), "sandbox"));
                }
                Ok("directories are marked".to_string())
            });
            step!("detect changes", {
                let before = sb.snapshot(&dir).await?;
                sb.write_bytes(&format!("{dir}/new.txt"), b"x", false).await?;
                let after = sb.snapshot(&dir).await?;
                let fresh: Vec<&String> = after.keys().filter(|k| !before.contains_key(*k)).collect();
                if fresh != vec!["new.txt"] || !before.contains_key("dir/bytes.bin") {
                    return Err(sandbox_error(format!("expected ['new.txt'], saw {fresh:?}"), "sandbox"));
                }
                Ok("new files are seen".to_string())
            });
            step!("delete", {
                sb.remove(&dir).await?;
                if sb.exists(&dir).await? {
                    return Err(sandbox_error("the probe directory is still there", "sandbox"));
                }
                Ok(String::new())
            });
        }
        let mut out = sb.info().await;
        out["ok"] = json!(steps.iter().all(|s| s["ok"] == true));
        out["steps"] = json!(steps);
        out["seconds"] = json!((started.elapsed().as_secs_f64() * 100.0).round() / 100.0);
        out
    }
}

#[async_trait]
impl Workspace for SandboxWorkspace {
    fn id(&self) -> &str {
        &self.id
    }

    fn isolated(&self) -> bool {
        self.sandbox.backend.isolated()
    }

    fn read_only(&self) -> bool {
        self.read_only
    }

    fn allow_shell(&self) -> bool {
        self.allow_shell.load(Ordering::SeqCst)
    }

    fn set_allow_shell(&self, on: bool) {
        self.allow_shell.store(on, Ordering::SeqCst);
    }

    fn python(&self) -> String {
        self.sandbox.python.clone()
    }

    fn location(&self) -> String {
        format!("{} sandbox", self.sandbox.name())
    }

    async fn read_bytes(&self, path: &str) -> Result<Vec<u8>> {
        self.sandbox.start().await?;
        let target = self.resolve(path).await?;
        self.sandbox.read_bytes(&target, Some(self.max_file_bytes)).await?.ok_or_else(|| sandbox_error(format!("no such file: {path}"), "fs_read"))
    }

    async fn write(&self, path: &str, content: &[u8], append: bool) -> Result<usize> {
        self.writable()?;
        self.sandbox.start().await?;
        let target = self.resolve(path).await?;
        self.sandbox.write_bytes(&target, content, append).await?;
        Ok(String::from_utf8_lossy(content).chars().count())
    }

    async fn listdir(&self, path: &str) -> Result<Vec<String>> {
        self.sandbox.start().await?;
        let target = self.resolve(path).await?;
        self.sandbox.list_dir(&target).await?.ok_or_else(|| sandbox_error(format!("not a directory: {path}"), "fs_list"))
    }

    async fn remove(&self, path: &str) -> Result<()> {
        self.writable()?;
        self.sandbox.start().await?;
        let target = self.resolve(path).await?;
        if target == posix_normalise(&self.root().await) {
            return Err(sandbox_error("refusing to delete the workspace root", "fs_delete"));
        }
        self.sandbox.remove(&target).await
    }

    async fn exists(&self, path: &str) -> bool {
        if self.sandbox.start().await.is_err() {
            return false;
        }
        match self.resolve(path).await {
            Ok(t) => self.sandbox.exists(&t).await.unwrap_or(false),
            Err(_) => false,
        }
    }

    async fn snapshot(&self) -> Result<Snapshot> {
        self.sandbox.start().await?;
        self.sandbox.snapshot(&self.root().await).await
    }

    async fn changed(&self, since: &Snapshot) -> Result<Vec<String>> {
        let now = self.snapshot().await?;
        Ok(now.into_iter().filter(|(n, s)| since.get(n) != Some(s) && !SCRATCH.contains(&n.as_str())).map(|(n, _)| n).collect())
    }

    /// Download `path` and return where it landed on this machine.
    async fn materialize(&self, path: &str) -> Result<PathBuf> {
        let target = self.resolve(path).await?;
        let data = self.read_bytes(path).await?;
        let dir = {
            let mut e = self.export.lock().unwrap_or_else(|p| p.into_inner());
            e.get_or_insert_with(|| std::env::temp_dir().join(format!("agent-harness-sandbox-{}", new_id("")))).clone()
        };
        let root = posix_normalise(&self.root().await);
        let rel = target.strip_prefix(&root).unwrap_or(&target).trim_start_matches('/');
        let local = dir.join(rel);
        if let Some(p) = local.parent() {
            std::fs::create_dir_all(p).map_err(|e| Error::Io(format!("{}: {e}", p.display())))?;
        }
        std::fs::write(&local, data).map_err(|e| Error::Io(format!("{}: {e}", local.display())))?;
        Ok(local)
    }

    async fn shell(&self, command: &str, timeout: Option<f64>) -> Result<Value> {
        let limit = timeout.unwrap_or(self.timeout);
        let r = self.sandbox.exec(command, None, None, Some(limit)).await?;
        if r.timed_out {
            return Err(sandbox_error(format!("command timed out after {limit}s"), "shell"));
        }
        let tail = |s: &str, n: usize| -> String {
            let c = s.chars().count();
            s.chars().skip(c.saturating_sub(n)).collect()
        };
        Ok(json!({"returncode": r.returncode, "stdout": tail(&r.stdout, 20_000), "stderr": tail(&r.stderr, 8_000)}))
    }

    async fn close(&self) {
        let _ = self.sandbox.stop(false).await;
    }

    fn as_sandbox(&self) -> Option<&SandboxWorkspace> {
        Some(self)
    }

    fn sandbox_ref(&self) -> Option<Value> {
        let s = self.sandbox.state.try_lock().ok()?;
        Some(json!({"sandbox": self.sandbox.name(), "id": s.id, "workdir": s.workdir.clone().unwrap_or_default()}))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A sandbox that is a temporary folder on this machine, run through `sh`
    /// — enough to prove the shell-only contract end to end where `sh` exists.
    pub(crate) struct LocalSh {
        pub root: PathBuf,
    }

    #[async_trait]
    impl SandboxBackend for LocalSh {
        fn name(&self) -> &str {
            "localsh"
        }

        async fn start(&self, _attach: Option<&str>) -> Result<String> {
            std::fs::create_dir_all(&self.root).map_err(|e| Error::Io(e.to_string()))?;
            Ok(String::new())
        }

        async fn exec(&self, command: &str, cwd: Option<&str>, env: &BTreeMap<String, String>, timeout: f64) -> Result<ExecResult> {
            let body = script(command, cwd.or(Some(self.root.to_str().unwrap_or("/"))), env);
            match run_process(vec!["sh".into(), "-c".into(), body], None, timeout).await? {
                Some((code, out, err)) => Ok(ExecResult::new(code, String::from_utf8_lossy(&out), String::from_utf8_lossy(&err))),
                None => Ok(ExecResult::timed_out(timeout)),
            }
        }
    }

    #[test]
    fn quoting_and_paths_match_python() {
        assert_eq!(q("plain/path.txt"), "plain/path.txt");
        assert_eq!(q("it's"), "'it'\"'\"'s'");
        assert_eq!(q(""), "''");
        assert_eq!(posix_normalise("/w/a/../b/./c"), "/w/b/c");
        assert_eq!(posix_normalise("/w/../../x"), "/x");
        // zlib.crc32(b"hello")
        assert_eq!(crc32(b"hello"), 0x3610_a686);
        assert_eq!(script("ls", Some("/w"), &[("A".to_string(), "b c".to_string())].into_iter().collect()), "cd /w && export A='b c' && ls");
    }

    #[tokio::test]
    async fn a_shell_only_sandbox_passes_its_own_check() {
        if !crate::runtime::workspace::on_path("sh") || !crate::runtime::workspace::on_path("base64") || !crate::runtime::workspace::on_path("stat") {
            return; // needs a POSIX userland (Git for Windows has one)
        }
        let dir = std::env::temp_dir().join(format!("ah_sbx_{}", uuid::Uuid::new_v4().simple()));
        let root = dir.to_string_lossy().replace('\\', "/");
        let sb =
            Sandbox::new(Arc::new(LocalSh { root: dir.clone() }), SandboxOptions { workdir: Some(format!("{root}/workspace")), ..Default::default() }).unwrap();
        let ws = SandboxWorkspace::new(sb, WrapOptions::default());
        let report = ws.check().await;
        assert_eq!(report["ok"], true, "{report:#}");
        ws.write("a/b.txt", "héllo".as_bytes(), false).await.unwrap();
        assert_eq!(ws.read("a/b.txt").await.unwrap(), "héllo");
        assert!(ws.resolve("../../etc/passwd").await.is_err());
        assert!(ws.read("missing.txt").await.unwrap_err().to_string().contains("no such file: missing.txt"));
        ws.close().await;
        let _ = std::fs::remove_dir_all(dir);
    }
}
