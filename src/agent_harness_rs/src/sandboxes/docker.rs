//! A long-lived container: Docker, Podman, or anything with the same CLI.
//!
//! ```no_run
//! # use agent_harness::sandboxes::sandbox;
//! # use serde_json::json;
//! let ws = sandbox(&json!({"sandbox": "docker", "image": "python:3.12-slim"}));
//! let ws = sandbox(&json!({"sandbox": "docker", "image": "node:22", "mount": "./project", "network": true}));
//! let ws = sandbox(&json!({"sandbox": "podman", "image": "python:3.12-slim"}));
//! let ws = sandbox(&json!({"sandbox": "docker", "image": "python:3.12-slim", "runtime": "runsc"})); // gVisor
//! ```
//!
//! Unlike `DockerWorkspace`, which starts a fresh container for every command,
//! the container here lives as long as the sandbox does — so what the agent
//! installs in one step is still there in the next.
//!
//! Python also removes a forgotten container when the process exits (`atexit`).
//! Rust has no such hook; the container's lifetime (`ttl`) is what ends one
//! orphaned by a crash.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use super::base::{ExecResult, ProcessExec, Runner, SandboxBackend, q};
use super::command::{Given, Opts, above};
use crate::errors::{Error, Result};
use crate::runtime::workspace::on_path;
use crate::types::new_id;

pub const URL_OPTION: &str = "image";

/// One container, kept running, with every command `exec`ed into it.
///
/// Defaults lean closed: no network, 2 CPUs, 2 GB, a process limit, and no way
/// to gain privileges. The container removes itself when it stops, is stopped
/// when the sandbox is, and has a lifetime (`ttl`) so that one orphaned by a
/// crash does not run for ever.
///
/// Options:
/// - `image`: the image to run. Pulled on first use if it is not present.
/// - `mount`: a directory on this machine to work in. It appears at `workdir`,
///   and what the agent writes there stays after the container is gone. The
///   container then runs as you, so the files are yours. `mount_read_only`.
/// - `network`: let the container reach the network.
/// - `container`: pick up a container that already exists — one of your own,
///   or one an earlier run was told to `keep` — and leave it afterwards.
/// - `keep`: leave the container behind at the end, so a later conversation
///   can pick it up by its id. It is not removed when it exits, and is started
///   again if it has.
/// - `memory`, `cpus`, `pids`, `user`: limits and who runs; null lifts one.
/// - `runtime`: an OCI runtime — "runsc" for gVisor, "kata" for Kata.
/// - `ttl`: seconds until the container ends by itself. null is no limit.
/// - `args`: anything else `docker run` should be given.
/// - `cli`: "docker", "podman", "nerdctl" — whichever is on PATH.
pub struct DockerSandbox {
    pub image: String,
    pub mount: Option<PathBuf>,
    pub mount_read_only: bool,
    pub network: bool,
    pub memory: Option<String>,
    pub cpus: Option<String>,
    pub pids: Option<u64>,
    pub user: Option<String>,
    pub runtime: Option<String>,
    pub ttl: Option<u64>,
    pub args: Vec<String>,
    pub cli: String,
    pub workdir: String,
    pub keep: bool,
    pub env: BTreeMap<String, String>,
    start_timeout: f64,
    given: Given,
    /// The container in use, and whether it was named in the options (and so is not ours).
    container: std::sync::Mutex<(String, bool)>,
    starts: std::sync::atomic::AtomicUsize,
    /// Whether commands really run (the default runner), so the CLI must be on PATH.
    real: bool,
    proc: ProcessExec,
}

fn expand(path: &str) -> PathBuf {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_default();
    let p = match path.strip_prefix('~') {
        Some(rest) if !home.is_empty() && (rest.is_empty() || rest.starts_with('/') || rest.starts_with('\\')) => format!("{home}{rest}"),
        _ => path.to_string(),
    };
    std::path::absolute(&p).unwrap_or_else(|_| PathBuf::from(p))
}

/// `uid:gid` of this process, as Python's `os.getuid()`/`os.getgid()`.
#[cfg(unix)]
fn host_user(mount: &std::path::Path) -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    // /proc/self belongs to whoever this process runs as; elsewhere the mount,
    // which this process made, does.
    let m = std::fs::metadata("/proc/self").or_else(|_| std::fs::metadata(mount)).ok()?;
    Some(format!("{}:{}", m.uid(), m.gid()))
}

#[cfg(not(unix))]
fn host_user(_mount: &std::path::Path) -> Option<String> {
    None
}

impl DockerSandbox {
    pub fn new(options: &Map<String, Value>, runner: Option<Runner>) -> Result<Self> {
        let o = Opts::new(
            "docker",
            options,
            &["image", "mount", "mount_read_only", "network", "container", "memory", "cpus", "pids", "user", "runtime", "ttl", "args", "cli"],
        )?;
        let given = o.text("id")?.or(o.text("container")?);
        let name = given.clone().unwrap_or_else(|| format!("agent-harness-{}", new_id("")));
        Ok(Self {
            image: o.text("image")?.unwrap_or_else(|| "python:3.12-slim".into()),
            mount: o.text("mount")?.filter(|m| !m.is_empty()).map(|m| expand(&m)),
            mount_read_only: o.flag("mount_read_only", false)?,
            network: o.flag("network", false)?,
            memory: o.text_or("memory", Some("2g"))?.filter(|m| !m.is_empty() && m != "0"),
            cpus: o.text_or("cpus", Some("2"))?.filter(|c| !c.is_empty() && c != "0"),
            pids: o.int_or("pids", Some(512))?.filter(|p| *p > 0),
            user: o.text("user")?.filter(|u| !u.is_empty()),
            runtime: o.text("runtime")?.filter(|r| !r.is_empty()),
            ttl: o.int_or("ttl", Some(14_400))?.filter(|t| *t > 0),
            args: o.list("args")?.unwrap_or_default(),
            cli: o.text("cli")?.unwrap_or_else(|| "docker".into()),
            workdir: o.text("workdir")?.unwrap_or_else(|| "/workspace".into()),
            keep: o.flag("keep", false)?,
            env: o.env(),
            start_timeout: o.number("start_timeout")?.unwrap_or(300.0),
            given: Given::new(given),
            container: std::sync::Mutex::new((name, false)),
            starts: Default::default(),
            real: runner.is_none(),
            proc: ProcessExec::new(runner, o.number("timeout")?.unwrap_or(120.0)),
        })
    }

    /// The container's name.
    pub fn container(&self) -> String {
        self.container.lock().unwrap_or_else(|p| p.into_inner()).0.clone()
    }

    fn set_container(&self, name: String, adopted: bool) {
        *self.container.lock().unwrap_or_else(|p| p.into_inner()) = (name, adopted);
    }

    pub fn wrap(&self, script: &str) -> Vec<String> {
        vec![self.cli.clone(), "exec".into(), "-i".into(), self.container(), "sh".into(), "-c".into(), script.to_string()]
    }

    /// The `docker run` this sandbox starts its container with.
    pub fn run_argv(&self) -> Vec<String> {
        let s = |x: &str| x.to_string();
        let mut argv = vec![
            self.cli.clone(),
            s("run"),
            s("-d"),
            s("--name"),
            self.container(),
            s("--label"),
            s("agent-harness=sandbox"),
            s("-w"),
            self.workdir.clone(),
            s("--security-opt"),
            s("no-new-privileges"),
        ];
        if !self.keep {
            // One to be picked up later must outlive its own exit.
            argv.insert(3, s("--rm"));
        }
        if !self.network {
            argv.extend([s("--network"), s("none")]);
        }
        if let Some(m) = &self.memory {
            argv.extend([s("--memory"), m.clone()]);
        }
        if let Some(c) = &self.cpus {
            argv.extend([s("--cpus"), c.clone()]);
        }
        if let Some(p) = self.pids {
            argv.extend([s("--pids-limit"), p.to_string()]);
        }
        if let Some(r) = &self.runtime {
            argv.extend([s("--runtime"), r.clone()]);
        }
        let mut user = self.user.clone();
        if let Some(mount) = &self.mount {
            let mode = if self.mount_read_only { ":ro" } else { "" };
            argv.extend([s("-v"), format!("{}:{}{mode}", mount.display(), self.workdir)]);
            // Otherwise everything written lands on the host owned by root.
            if user.is_none() && self.cli == "docker" {
                user = host_user(mount);
            }
        }
        if let Some(u) = user {
            argv.extend([s("--user"), u, s("-e"), s("HOME=/tmp")]);
        }
        for (k, v) in &self.env {
            argv.extend([s("-e"), format!("{k}={v}")]);
        }
        argv.extend(self.args.iter().cloned());
        argv.push(self.image.clone());
        match self.ttl {
            Some(t) => argv.extend([s("sleep"), t.to_string()]),
            None => argv.extend([s("tail"), s("-f"), s("/dev/null")]),
        }
        argv
    }

    async fn cli_run(&self, argv: Vec<String>, timeout: f64, what: &str) -> Result<String> {
        match (self.proc.runner)(argv, None, timeout).await? {
            None => Err(Error::tool(format!("{} {what} did not finish in {timeout:.0}s", self.cli), "sandbox")),
            Some((0, out, _)) => Ok(String::from_utf8_lossy(&out).trim().to_string()),
            Some((_, _, err)) => Err(Error::tool(
                format!("{} {what} failed: {}", self.cli, String::from_utf8_lossy(&err).trim().chars().take(400).collect::<String>()),
                "sandbox",
            )),
        }
    }
}

#[async_trait]
impl SandboxBackend for DockerSandbox {
    /// "docker" whichever CLI drives it, as in Python.
    fn name(&self) -> &str {
        "docker"
    }

    async fn start(&self, attach_id: Option<&str>) -> Result<String> {
        if self.real && !on_path(&self.cli) {
            return Err(Error::config(format!("{} is not on PATH — install it, or pick another sandbox", self.cli)));
        }
        let first = self.starts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0;
        if let Some((name, adopted)) = self.given.target(attach_id) {
            self.set_container(name.clone(), adopted);
            let state = self.cli_run(vec![self.cli.clone(), "inspect".into(), "-f".into(), "{{.State.Running}}".into(), name.clone()], 30.0, "inspect").await?;
            if state != "true" {
                // Kept, and since exited: its files are still in it.
                self.cli_run(vec![self.cli.clone(), "start".into(), name], self.start_timeout, "start").await?;
            }
        } else {
            if !first {
                // Taking the place of one that could not be picked up.
                self.set_container(format!("agent-harness-{}", new_id("")), false);
            }
            if let Some(m) = &self.mount {
                std::fs::create_dir_all(m).map_err(|e| Error::Io(format!("{}: {e}", m.display())))?;
            }
            self.cli_run(self.run_argv(), self.start_timeout, "run").await?;
            if let (Some(user), None) = (&self.user, &self.mount) {
                // Docker made the working directory as root; hand it over.
                let fix = format!("mkdir -p {w} && chown {} {w}", q(user), w = q(&self.workdir));
                self.cli_run(vec![self.cli.clone(), "exec".into(), "-u".into(), "0".into(), self.container(), "sh".into(), "-c".into(), fix], 30.0, "exec")
                    .await?;
            }
        }
        self.proc.probe(&|s| self.wrap(s)).await;
        Ok(self.container())
    }

    async fn exec(&self, command: &str, cwd: Option<&str>, env: &BTreeMap<String, String>, timeout: f64) -> Result<ExecResult> {
        let home = above(&self.workdir);
        self.proc.exec(&|s| self.wrap(s), command, Some(cwd.unwrap_or(&home)), env, timeout).await
    }

    async fn stop(&self, _id: &str) -> Result<()> {
        let (name, adopted) = self.container.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if !adopted {
            // A timeout here leaves the container to its own lifetime.
            (self.proc.runner)(vec![self.cli.clone(), "rm".into(), "-f".into(), name], None, 30.0).await?;
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

/// `sandbox("docker://node:22")`; podman is this with `"cli": "podman"`.
pub fn build(options: &Map<String, Value>) -> Result<Arc<dyn SandboxBackend>> {
    Ok(Arc::new(DockerSandbox::new(options, None)?))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::sandboxes::command::tests::Recorder;
    use crate::sandboxes::{Sandbox, SandboxOptions, SandboxWorkspace, WrapOptions, sandbox};

    fn opts(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    fn after<'a>(argv: &'a [String], flag: &str) -> &'a str {
        &argv[argv.iter().position(|a| a == flag).unwrap_or_else(|| panic!("{flag} not in {argv:?}")) + 1]
    }

    #[test]
    fn a_container_is_started_closed_by_default() {
        let argv = DockerSandbox::new(&opts(json!({"image": "python:3.12-slim"})), None).unwrap().run_argv();
        assert_eq!(argv[..4], ["docker", "run", "-d", "--rm"]);
        for (flag, value) in [
            ("--network", "none"),
            ("--memory", "2g"),
            ("--cpus", "2"),
            ("--pids-limit", "512"),
            ("--security-opt", "no-new-privileges"),
            ("-w", "/workspace"),
            ("--label", "agent-harness=sandbox"),
        ] {
            assert_eq!(after(&argv, flag), value);
        }
        assert!(after(&argv, "--name").starts_with("agent-harness-"));
        assert_eq!(argv[argv.len() - 3..], ["python:3.12-slim", "sleep", "14400"]);
        assert!(!argv.contains(&"--user".to_string()) && !argv.contains(&"-v".to_string()));
    }

    #[test]
    fn a_container_opens_up_only_as_far_as_it_is_told() {
        let dir = std::env::temp_dir();
        let mount = std::path::absolute(&dir).unwrap();
        let argv = DockerSandbox::new(
            &opts(json!({"image": "node:22", "network": true, "runtime": "runsc", "ttl": null, "mount": dir.to_string_lossy(),
                         "env": {"CI": "1"}, "memory": null, "args": ["--gpus", "all"]})),
            None,
        )
        .unwrap()
        .run_argv();
        assert!(!argv.contains(&"--network".to_string()) && !argv.contains(&"--memory".to_string()));
        assert_eq!(after(&argv, "--runtime"), "runsc");
        assert_eq!(after(&argv, "-v"), format!("{}:/workspace", mount.display()));
        // A mounted folder is written as you, not as root.
        if cfg!(unix) {
            assert!(after(&argv, "--user").contains(':'));
            assert_eq!(after(&argv, "-e"), "HOME=/tmp");
        }
        assert!(argv.contains(&"CI=1".to_string()));
        assert_eq!(argv[argv.len() - 6..], ["--gpus", "all", "node:22", "tail", "-f", "/dev/null"]);

        let read_only = DockerSandbox::new(&opts(json!({"image": "a", "mount": dir.to_string_lossy(), "mount_read_only": true})), None).unwrap().run_argv();
        assert!(after(&read_only, "-v").ends_with(":/workspace:ro"));
        // Kept for later: it must not remove itself when it exits.
        let kept = DockerSandbox::new(&opts(json!({"keep": true, "cpus": 1.5})), None).unwrap().run_argv();
        assert_eq!(kept[3], "--name");
        assert_eq!(after(&kept, "--cpus"), "1.5");
    }

    #[tokio::test]
    async fn a_container_it_did_not_start_is_left_running() {
        let run = Recorder::with(&[("inspect", "true\n")]);
        let backend = Arc::new(DockerSandbox::new(&opts(json!({"container": "mine"})), Some(run.runner())).unwrap());
        let sb = Sandbox::new(backend.clone(), SandboxOptions::default()).unwrap();
        sb.start().await.unwrap();
        sb.stop(false).await.unwrap();
        assert_eq!(sb.id().await, "mine");
        let calls = run.calls();
        assert!(!calls.iter().any(|c| c[..2].contains(&"run".to_string()) || c[..2].contains(&"rm".to_string())), "{calls:?}");
        assert_eq!(calls[0], ["docker", "inspect", "-f", "{{.State.Running}}", "mine"]);
        assert_eq!(calls[1][..4], ["docker", "exec", "-i", "mine"]);
        // No working directory was named: the default one, as in Python.
        assert_eq!(sb.workdir().await, "/workspace");
    }

    #[tokio::test]
    async fn a_stopped_container_is_started_again_and_a_new_one_removed_at_the_end() {
        let run = Recorder::with(&[("inspect", "false\n")]);
        let backend = Arc::new(DockerSandbox::new(&opts(json!({"cli": "podman", "user": "1000"})), Some(run.runner())).unwrap());
        assert_eq!(backend.cli, "podman");
        backend.start(Some("kept-1")).await.unwrap();
        assert_eq!(run.calls()[1], ["podman", "start", "kept-1"]);

        let run = Recorder::default();
        let backend = Arc::new(DockerSandbox::new(&opts(json!({"user": "1000"})), Some(run.runner())).unwrap());
        let name = backend.start(None).await.unwrap();
        let calls = run.calls();
        assert_eq!(calls[0][..2], ["docker", "run"]);
        assert_eq!(after(&calls[0], "--name"), name);
        // Docker made the working directory as root; it is handed over.
        assert_eq!(calls[1], ["docker", "exec", "-u", "0", &name, "sh", "-c", "mkdir -p /workspace && chown 1000 /workspace"]);
        backend.stop(&name).await.unwrap();
        assert_eq!(run.calls().last().unwrap(), &["docker", "rm", "-f", &name]);
    }

    #[tokio::test]
    async fn a_cli_that_is_not_installed_says_so() {
        let backend = DockerSandbox::new(&opts(json!({"image": "a", "cli": "definitely-not-a-container-cli"})), None).unwrap();
        let err = backend.start(None).await.unwrap_err();
        assert!(matches!(err, Error::Configuration(_)) && err.to_string().contains("is not on PATH"), "{err}");
    }

    #[test]
    fn a_sandbox_is_named_by_word_url_or_mapping() {
        assert!(sandbox(&json!("docker://node:22")).is_ok());
        assert!(sandbox(&json!("podman")).is_ok());
        assert!(sandbox(&json!({"sandbox": "docker", "image": "a", "read_only": true})).unwrap().read_only);
        let err = sandbox(&json!({"sandbox": "docker", "no_such_option": 1})).err().unwrap().to_string();
        assert!(err.starts_with("docker sandbox: got an unexpected keyword argument 'no_such_option'"), "{err}");
        assert!(build(&opts(json!({"image": "a", "network": "yes"}))).err().unwrap().to_string().contains("true or false"));
    }

    /// Real docker, with an image already on this machine (nothing is pulled).
    /// `AH_LIVE_DOCKER=1`, and `AH_LIVE_DOCKER_IMAGE` to pick it (alpine:latest).
    #[tokio::test]
    async fn live_docker_passes_the_self_check_and_cleans_up_after_itself() {
        if std::env::var("AH_LIVE_DOCKER").as_deref() != Ok("1") {
            return;
        }
        let image = std::env::var("AH_LIVE_DOCKER_IMAGE").unwrap_or_else(|_| "alpine:latest".into());
        let present = crate::sandboxes::run_process(vec!["docker".into(), "image".into(), "inspect".into(), image.clone()], None, 20.0).await;
        if !matches!(present, Ok(Some((0, _, _)))) {
            eprintln!("skipping: {image} is not present locally");
            return;
        }
        let backend = Arc::new(DockerSandbox::new(&opts(json!({"image": image})), None).unwrap());
        let ws = SandboxWorkspace::new(Sandbox::new(backend.clone(), SandboxOptions::default()).unwrap(), WrapOptions::default());
        let report = ws.check().await;
        assert_eq!(report["ok"], true, "{report:#}");
        use crate::runtime::workspace::Workspace;
        let state = ws.shell("echo kept > /tmp/state && ls /sys/class/net", None).await.unwrap();
        assert_eq!(state["stdout"].as_str().unwrap().split_whitespace().collect::<Vec<_>>(), ["lo"]);
        assert_eq!(ws.shell("cat /tmp/state", None).await.unwrap()["stdout"], "kept\n");
        let err = ws.shell("sleep 300", Some(1.0)).await.unwrap_err().to_string();
        assert!(err.contains("timed out"), "{err}");
        let name = backend.container();
        ws.close().await;
        let left = crate::sandboxes::run_process(vec!["docker".into(), "ps".into(), "-a".into(), "--format".into(), "{{.Names}}".into()], None, 20.0).await;
        let listed = left.ok().flatten().map(|(_, out, _)| String::from_utf8_lossy(&out).into_owned()).unwrap_or_default();
        assert!(!listed.lines().any(|l| l == name));
    }
}
