//! Workspace broker: an isolated place for an agent to work.
//!
//! Every path a workspace tool touches is resolved inside the workspace root —
//! a symlink or a `../..` that escapes is refused, not clamped. Shell access is
//! off unless you ask for it, and the shell tool asks for approval even then.
//!
//! Two backends ship here: [`LocalWorkspace`] (a directory per agent) and
//! [`DockerWorkspace`] (the same directory bind-mounted into a container, so
//! the command itself is contained). For a workspace that lives somewhere else
//! entirely — a long-lived container, an E2B, Daytona or Modal sandbox, a pod,
//! a machine over SSH — see [`crate::sandboxes`].
//!
//! Every operation is async, so a workspace whose files are a network call
//! away behaves exactly like one on local disk.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use indexmap::IndexMap;
use schemas::{Delete, List, Read, Shell, Write};
use serde_json::{Value, json};

use crate::errors::{Error, Result};
use crate::tools::{Permission, Tool};
use crate::types::new_id;

/// Every file in a workspace, as (a stamp of when it changed, its size), by
/// relative path.
pub type Snapshot = BTreeMap<String, (i64, u64)>;

/// Files the harness writes for its own use — never something a run produced.
pub const SCRATCH: [&str; 1] = ["_snippet.py"];

fn ws_error(message: impl Into<String>, tool: &str) -> Error {
    Error::tool(message, tool)
}

/// Somewhere an agent's files live and its commands run.
#[async_trait]
pub trait Workspace: Send + Sync {
    fn id(&self) -> &str;
    /// Commands here cannot touch the machine the harness runs on. A workspace
    /// that is its own machine says so, and its shell does not need asking for.
    fn isolated(&self) -> bool {
        false
    }
    fn read_only(&self) -> bool;
    fn allow_shell(&self) -> bool;
    fn set_allow_shell(&self, on: bool);
    /// The interpreter a snippet run in this workspace is given to.
    fn python(&self) -> String;
    /// Where it is, for messages.
    fn location(&self) -> String;

    async fn read(&self, path: &str) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.read_bytes(path).await?).into_owned())
    }
    async fn read_bytes(&self, path: &str) -> Result<Vec<u8>>;
    async fn write(&self, path: &str, content: &[u8], append: bool) -> Result<usize>;
    async fn listdir(&self, path: &str) -> Result<Vec<String>>;
    async fn remove(&self, path: &str) -> Result<()>;
    async fn exists(&self, path: &str) -> bool;
    async fn snapshot(&self) -> Result<Snapshot>;
    /// The files created or modified since `since` was taken.
    async fn changed(&self, since: &Snapshot) -> Result<Vec<String>> {
        let now = self.snapshot().await?;
        Ok(now.into_iter().filter(|(name, stamp)| since.get(name) != Some(stamp) && !SCRATCH.contains(&name.as_str())).map(|(n, _)| n).collect())
    }
    /// A local file holding `path` — for anything that must open it here. On
    /// local disk that is the file itself; a remote workspace downloads it.
    async fn materialize(&self, path: &str) -> Result<PathBuf>;
    /// Run a command with the workspace as its working directory:
    /// `{"returncode", "stdout", "stderr"}`.
    async fn shell(&self, command: &str, timeout: Option<f64>) -> Result<Value>;
    /// Release whatever the workspace holds. Safe to call twice.
    async fn close(&self) {}
    /// Which sandbox this is, as plain data — what a session remembers. `None`
    /// for a workspace on this machine.
    fn sandbox_ref(&self) -> Option<Value> {
        None
    }

    /// The sandbox behind this workspace, if it is one.
    fn as_sandbox(&self) -> Option<&crate::sandboxes::SandboxWorkspace> {
        None
    }
}

/// Filesystem tools bound to a workspace (plus `shell`, if allowed).
pub fn workspace_tools(ws: Arc<dyn Workspace>) -> Vec<Tool> {
    let w = ws.clone();
    let fs_read = Tool::new("fs_read", move |a: Read, _| {
        let w = w.clone();
        async move { w.read(&a.path).await }
    })
    .tags(["builtin", "fs"]);
    let w = ws.clone();
    let fs_write = Tool::new("fs_write", move |a: Write, _| {
        let w = w.clone();
        async move {
            let written = w.write(&a.path, a.content.as_bytes(), a.append).await?;
            Ok(format!("wrote {written} characters to {}", a.path))
        }
    })
    .tags(["builtin", "fs"])
    .permission(Permission::Allow);
    let w = ws.clone();
    let fs_list = Tool::new("fs_list", move |a: List, _| {
        let w = w.clone();
        async move { w.listdir(&a.path).await }
    })
    .tags(["builtin", "fs"]);
    let w = ws.clone();
    let fs_delete = Tool::new("fs_delete", move |a: Delete, _| {
        let w = w.clone();
        async move {
            w.remove(&a.path).await?;
            Ok(format!("deleted {}", a.path))
        }
    })
    .tags(["builtin", "fs"])
    .permission(Permission::Ask);
    let mut built = vec![fs_read, fs_write, fs_list, fs_delete];
    if ws.allow_shell() {
        // Inside its own machine a command can only hurt the sandbox, and asking
        // before each one would make the sandbox pointless. On this machine it
        // asks, every time.
        let w = ws.clone();
        let shell = Tool::new("shell", move |a: Shell, _| {
            let w = w.clone();
            async move { w.shell(&a.command, Some(a.timeout)).await }
        })
        .tags(["builtin", "exec"])
        .permission(if ws.isolated() { Permission::Allow } else { Permission::Ask });
        built.push(shell);
    }
    built
}

mod schemas {
    use schemars::JsonSchema;
    use serde::Deserialize;

    fn here() -> String {
        ".".into()
    }

    fn sixty() -> f64 {
        60.0
    }

    /// Read a text file from the workspace.
    #[derive(Deserialize, JsonSchema)]
    pub struct Read {
        /// path relative to the workspace root.
        pub path: String,
    }

    /// Write a text file in the workspace.
    #[derive(Deserialize, JsonSchema)]
    pub struct Write {
        /// path relative to the workspace root.
        pub path: String,
        /// the full text to write.
        pub content: String,
        /// append instead of replacing the file.
        #[serde(default)]
        pub append: bool,
    }

    /// List a directory in the workspace.
    #[derive(Deserialize, JsonSchema)]
    pub struct List {
        /// directory relative to the workspace root.
        #[serde(default = "here")]
        pub path: String,
    }

    /// Delete a file or directory in the workspace.
    #[derive(Deserialize, JsonSchema)]
    pub struct Delete {
        /// path relative to the workspace root.
        pub path: String,
    }

    /// Run a shell command inside the workspace.
    #[derive(Deserialize, JsonSchema)]
    pub struct Shell {
        /// the command line to run.
        pub command: String,
        /// seconds before the command is killed.
        #[serde(default = "sixty")]
        pub timeout: f64,
    }
}

/// How a local workspace behaves. Every field has a working default.
#[derive(Debug, Clone)]
pub struct WorkspaceOptions {
    pub id: Option<String>,
    pub network: bool,
    pub read_only: bool,
    pub allow_shell: bool,
    pub env: BTreeMap<String, String>,
    pub timeout: f64,
    pub max_file_bytes: u64,
    /// Remove the folder when the workspace is closed.
    pub ephemeral: bool,
}

impl Default for WorkspaceOptions {
    fn default() -> Self {
        Self {
            id: None,
            network: false,
            read_only: false,
            allow_shell: false,
            env: BTreeMap::new(),
            timeout: 120.0,
            max_file_bytes: 8_000_000,
            ephemeral: false,
        }
    }
}

/// Lexically normalise a path: `a/./b/../c` → `a/c`.
fn normalise(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// The tail of a command's output, at most `n` characters.
pub(crate) fn tail(bytes: &[u8], n: usize) -> String {
    let text = String::from_utf8_lossy(bytes);
    let count = text.chars().count();
    text.chars().skip(count.saturating_sub(n)).collect()
}

/// A jailed directory plus the tools that operate inside it.
#[derive(Debug)]
pub struct LocalWorkspace {
    id: String,
    pub root: PathBuf,
    pub options: WorkspaceOptions,
    allow_shell: AtomicBool,
}

impl LocalWorkspace {
    pub fn new(root: impl Into<PathBuf>, options: WorkspaceOptions) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(|e| Error::Io(format!("{}: {e}", root.display())))?;
        let root = root.canonicalize().map_err(|e| Error::Io(format!("{}: {e}", root.display())))?;
        let id = options.id.clone().unwrap_or_else(|| new_id("ws"));
        let shell = options.allow_shell;
        Ok(Self { id, root, options, allow_shell: AtomicBool::new(shell) })
    }

    /// Resolve inside the jail. Anything that escapes — by `..` or by a
    /// symlink — is refused.
    pub fn resolve(&self, path: &str) -> Result<PathBuf> {
        let candidate = Path::new(path);
        let joined = if candidate.is_absolute() { candidate.to_path_buf() } else { self.root.join(candidate) };
        // Follow symlinks for the part that exists; normalise the rest.
        let mut existing = joined.clone();
        let mut rest = Vec::new();
        while !existing.exists() {
            match (existing.file_name().map(|n| n.to_os_string()), existing.parent().map(Path::to_path_buf)) {
                (Some(name), Some(parent)) => {
                    rest.push(name);
                    existing = parent;
                }
                _ => break,
            }
        }
        let mut target = existing.canonicalize().unwrap_or_else(|_| normalise(&existing));
        for name in rest.into_iter().rev() {
            target.push(name);
        }
        let target = normalise(&target);
        if target != self.root && !target.starts_with(&self.root) {
            return Err(ws_error(format!("path '{path}' is outside the workspace"), "workspace"));
        }
        Ok(target)
    }

    fn writable(&self) -> Result<()> {
        if self.options.read_only { Err(ws_error("this workspace is read-only", "workspace")) } else { Ok(()) }
    }

    fn walk(dir: &Path, root: &Path, seen: &mut Snapshot) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let path = e.path();
            if e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let Ok(meta) = std::fs::symlink_metadata(&path) else { continue };
            if meta.is_dir() {
                Self::walk(&path, root, seen);
            } else if meta.is_file() {
                let rel =
                    path.strip_prefix(root).unwrap_or(&path).components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/");
                let modified = meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos() as i64);
                seen.insert(rel, (modified, meta.len()));
            }
        }
    }

    /// Remove the folder if it is temporary.
    pub fn cleanup(&self) {
        if self.options.ephemeral {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    pub(crate) async fn run_shell(&self, argv: Vec<String>, timeout: f64, tool: &str) -> Result<Value> {
        let mut cmd = tokio::process::Command::new(&argv[0]);
        cmd.args(&argv[1..]).current_dir(&self.root).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).kill_on_drop(true);
        if argv[0] != "docker" && argv[0] != "podman" {
            cmd.env_clear();
            for key in ["PATH", "SYSTEMROOT", "COMSPEC", "PATHEXT", "TEMP", "TMP"] {
                if let Ok(v) = std::env::var(key) {
                    cmd.env(key, v);
                }
            }
            cmd.env("HOME", &self.root).env("LANG", std::env::var("LANG").unwrap_or_else(|_| "C.UTF-8".into()));
            cmd.envs(&self.options.env);
        }
        let child = cmd.spawn().map_err(|e| ws_error(format!("could not start {}: {e}", argv[0]), tool))?;
        match tokio::time::timeout(std::time::Duration::from_secs_f64(timeout.max(0.1)), child.wait_with_output()).await {
            Ok(Ok(out)) => Ok(json!({"returncode": out.status.code().unwrap_or(-1), "stdout": tail(&out.stdout, 20_000), "stderr": tail(&out.stderr, 8_000)})),
            Ok(Err(e)) => Err(ws_error(format!("{}: {e}", argv[0]), tool)),
            Err(_) => {
                let what = if tool == "shell" && (argv[0] == "docker" || argv[0] == "podman") { "container" } else { "command" };
                Err(ws_error(format!("{what} timed out after {timeout}s"), "shell"))
            }
        }
    }
}

#[async_trait]
impl Workspace for LocalWorkspace {
    fn id(&self) -> &str {
        &self.id
    }

    fn read_only(&self) -> bool {
        self.options.read_only
    }

    fn allow_shell(&self) -> bool {
        self.allow_shell.load(Ordering::SeqCst)
    }

    fn set_allow_shell(&self, on: bool) {
        self.allow_shell.store(on, Ordering::SeqCst);
    }

    fn python(&self) -> String {
        if cfg!(windows) { "python".into() } else { "python3".into() }
    }

    fn location(&self) -> String {
        self.root.display().to_string()
    }

    async fn read(&self, path: &str) -> Result<String> {
        let target = self.resolve(path)?;
        let meta = std::fs::metadata(&target).ok().filter(|m| m.is_file()).ok_or_else(|| ws_error(format!("no such file: {path}"), "fs_read"))?;
        if meta.len() > self.options.max_file_bytes {
            return Err(ws_error(format!("{path} is larger than the {}-byte limit", self.options.max_file_bytes), "fs_read"));
        }
        let bytes = tokio::fs::read(&target).await.map_err(|e| ws_error(format!("could not read {path}: {e}"), "fs_read"))?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    async fn read_bytes(&self, path: &str) -> Result<Vec<u8>> {
        let target = self.resolve(path)?;
        if !target.is_file() {
            return Err(ws_error(format!("no such file: {path}"), "fs_read"));
        }
        tokio::fs::read(&target).await.map_err(|e| ws_error(format!("could not read {path}: {e}"), "fs_read"))
    }

    async fn write(&self, path: &str, content: &[u8], append: bool) -> Result<usize> {
        use tokio::io::AsyncWriteExt;
        self.writable()?;
        let target = self.resolve(path)?;
        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| ws_error(format!("could not write {path}: {e}"), "fs_write"))?;
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(&target)
            .await
            .map_err(|e| ws_error(format!("could not write {path}: {e}"), "fs_write"))?;
        file.write_all(content).await.map_err(|e| ws_error(format!("could not write {path}: {e}"), "fs_write"))?;
        // Python counts characters of the text it was given.
        Ok(String::from_utf8_lossy(content).chars().count())
    }

    async fn listdir(&self, path: &str) -> Result<Vec<String>> {
        let target = self.resolve(path)?;
        if !target.is_dir() {
            return Err(ws_error(format!("not a directory: {path}"), "fs_list"));
        }
        let mut names: Vec<String> = std::fs::read_dir(&target)
            .map_err(|e| ws_error(format!("{path}: {e}"), "fs_list"))?
            .flatten()
            .map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if e.path().is_dir() { format!("{name}/") } else { name }
            })
            .collect();
        names.sort();
        Ok(names)
    }

    async fn remove(&self, path: &str) -> Result<()> {
        self.writable()?;
        let target = self.resolve(path)?;
        if target == self.root {
            return Err(ws_error("refusing to delete the workspace root", "fs_delete"));
        }
        if target.is_dir() {
            tokio::fs::remove_dir_all(&target).await.map_err(|e| ws_error(format!("could not delete {path}: {e}"), "fs_delete"))
        } else {
            let _ = tokio::fs::remove_file(&target).await;
            Ok(())
        }
    }

    async fn exists(&self, path: &str) -> bool {
        self.resolve(path).is_ok_and(|p| p.exists())
    }

    async fn snapshot(&self) -> Result<Snapshot> {
        let mut seen = Snapshot::new();
        Self::walk(&self.root, &self.root, &mut seen);
        Ok(seen)
    }

    async fn materialize(&self, path: &str) -> Result<PathBuf> {
        self.resolve(path)
    }

    async fn shell(&self, command: &str, timeout: Option<f64>) -> Result<Value> {
        let argv: Vec<String> = if cfg!(windows) { vec!["cmd".into(), "/C".into(), command.into()] } else { vec!["sh".into(), "-c".into(), command.into()] };
        self.run_shell(argv, timeout.unwrap_or(self.options.timeout), "shell").await
    }

    async fn close(&self) {
        self.cleanup();
    }
}

/// The same directory, but commands run inside a container.
///
/// Requires the `docker` CLI on PATH: if it is missing the constructor fails
/// rather than quietly running on the host.
#[derive(Debug)]
pub struct DockerWorkspace {
    pub local: LocalWorkspace,
    pub image: String,
    pub docker_args: Vec<String>,
}

/// Is `program` on PATH?
pub fn on_path(program: &str) -> bool {
    let exts: Vec<String> = if cfg!(windows) { vec![".exe".into(), ".cmd".into(), ".bat".into(), String::new()] } else { vec![String::new()] };
    std::env::var_os("PATH").is_some_and(|paths| std::env::split_paths(&paths).any(|dir| exts.iter().any(|e| dir.join(format!("{program}{e}")).is_file())))
}

impl DockerWorkspace {
    pub fn new(root: impl Into<PathBuf>, image: &str, docker_args: Vec<String>, options: WorkspaceOptions) -> Result<Self> {
        if !on_path("docker") {
            return Err(ws_error("docker is not on PATH; use the local workspace backend", "workspace"));
        }
        Ok(Self { local: LocalWorkspace::new(root, options)?, image: image.into(), docker_args })
    }
}

#[async_trait]
impl Workspace for DockerWorkspace {
    fn id(&self) -> &str {
        self.local.id()
    }

    fn read_only(&self) -> bool {
        self.local.read_only()
    }

    fn allow_shell(&self) -> bool {
        self.local.allow_shell()
    }

    fn set_allow_shell(&self, on: bool) {
        self.local.set_allow_shell(on);
    }

    fn python(&self) -> String {
        "python3".into()
    }

    fn location(&self) -> String {
        format!("docker:{} at {}", self.image, self.local.location())
    }

    async fn read(&self, path: &str) -> Result<String> {
        self.local.read(path).await
    }

    async fn read_bytes(&self, path: &str) -> Result<Vec<u8>> {
        self.local.read_bytes(path).await
    }

    async fn write(&self, path: &str, content: &[u8], append: bool) -> Result<usize> {
        self.local.write(path, content, append).await
    }

    async fn listdir(&self, path: &str) -> Result<Vec<String>> {
        self.local.listdir(path).await
    }

    async fn remove(&self, path: &str) -> Result<()> {
        self.local.remove(path).await
    }

    async fn exists(&self, path: &str) -> bool {
        self.local.exists(path).await
    }

    async fn snapshot(&self) -> Result<Snapshot> {
        self.local.snapshot().await
    }

    async fn materialize(&self, path: &str) -> Result<PathBuf> {
        self.local.materialize(path).await
    }

    async fn shell(&self, command: &str, timeout: Option<f64>) -> Result<Value> {
        let mut argv: Vec<String> = ["docker", "run", "--rm", "-i", "-v"].iter().map(|s| s.to_string()).collect();
        argv.push(format!("{}:/work", self.local.root.display()));
        argv.extend(["-w", "/work", "--memory", "2g", "--cpus", "2"].iter().map(|s| s.to_string()));
        if !self.local.options.network {
            argv.extend(["--network".into(), "none".into()]);
        }
        if self.local.options.read_only {
            argv.push("--read-only".into());
        }
        argv.extend(self.docker_args.iter().cloned());
        argv.extend([self.image.clone(), "sh".into(), "-lc".into(), command.into()]);
        self.local.run_shell(argv, timeout.unwrap_or(self.local.options.timeout), "shell").await
    }

    async fn close(&self) {
        self.local.cleanup();
    }
}

struct BrokerInner {
    root: Option<PathBuf>,
    temporary: bool,
    open: IndexMap<String, Arc<dyn Workspace>>,
}

/// Hands out workspaces: one shared, one per isolated sub-agent.
///
/// - `WorkspaceBroker::new(None)` — a folder each, under a temporary root.
/// - `.backend("docker")` — a container each, over the same folders.
/// - `.sandbox(json!({"sandbox": "e2b", "template": "base"}))` — a sandbox
///   each, built with these options (see [`crate::sandboxes`]).
///
/// Cheap to clone; clones hand out the same workspaces.
#[derive(Clone)]
pub struct WorkspaceBroker {
    inner: Arc<Mutex<BrokerInner>>,
    pub backend: String,
    pub sandbox: Option<Value>,
    pub image: String,
    pub network: bool,
    pub allow_shell: bool,
    /// Remove what was temporary on cleanup (the default when no root was given).
    pub ephemeral: bool,
}

impl std::fmt::Debug for WorkspaceBroker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<WorkspaceBroker {} {} open>", self.backend, self.lock().open.len())
    }
}

impl Default for WorkspaceBroker {
    fn default() -> Self {
        Self::new(None::<PathBuf>)
    }
}

impl WorkspaceBroker {
    pub fn new(root: Option<impl Into<PathBuf>>) -> Self {
        let root = root.map(Into::into);
        let ephemeral = root.is_none();
        Self {
            inner: Arc::new(Mutex::new(BrokerInner { root, temporary: false, open: IndexMap::new() })),
            backend: "local".into(),
            sandbox: None,
            image: "python:3.12-slim".into(),
            network: false,
            allow_shell: false,
            ephemeral,
        }
    }

    /// `local`, `docker`, or the name of a sandbox (`e2b`, `ssh`, ...).
    pub fn backend(mut self, backend: &str) -> Self {
        if backend == "local" || backend == "docker" {
            self.backend = backend.into();
        } else {
            self.sandbox = Some(json!({"sandbox": backend}));
        }
        self
    }

    /// Every workspace handed out is its own sandbox, built with these options:
    /// a name, a URL (`docker://node:22`) or a mapping with `sandbox:`.
    pub fn sandbox(mut self, spec: Value) -> Self {
        self.sandbox = Some(spec);
        self
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BrokerInner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Where local workspaces live. Made on first use.
    pub fn root(&self) -> Result<PathBuf> {
        let mut i = self.lock();
        if i.root.is_none() {
            let dir = std::env::temp_dir().join(format!("agent-harness-{}", new_id("")));
            i.root = Some(dir);
            i.temporary = true;
        }
        let root = i.root.clone().expect("set above");
        std::fs::create_dir_all(&root).map_err(|e| Error::Io(format!("{}: {e}", root.display())))?;
        Ok(root)
    }

    fn make(&self, name: &str) -> Result<Arc<dyn Workspace>> {
        if let Some(spec) = &self.sandbox {
            let mut spec = match spec {
                Value::String(s) => json!({"sandbox": s}),
                other => other.clone(),
            };
            if let Value::Object(m) = &mut spec {
                if self.image != "python:3.12-slim" {
                    m.entry("image").or_insert(json!(self.image));
                }
                if self.network {
                    m.entry("network").or_insert(json!(true));
                }
            }
            return Ok(Arc::new(crate::sandboxes::sandbox(&spec)?));
        }
        let safe: String = name.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
        let path = self.root()?.join(safe);
        let options = WorkspaceOptions { network: self.network, allow_shell: self.allow_shell, ..WorkspaceOptions::default() };
        Ok(if self.backend == "docker" {
            Arc::new(DockerWorkspace::new(path, &self.image, vec![], options)?)
        } else {
            Arc::new(LocalWorkspace::new(path, options)?)
        })
    }

    /// An isolated workspace per name, or the shared one that files pass through.
    pub fn acquire(&self, name: &str, isolated: bool) -> Result<Arc<dyn Workspace>> {
        let key = if isolated { name.to_string() } else { "shared".into() };
        if let Some(w) = self.lock().open.get(&key) {
            return Ok(w.clone());
        }
        let w = self.make(&key)?;
        self.lock().open.insert(key, w.clone());
        Ok(w)
    }

    /// Take charge of a workspace built elsewhere, so it is closed with the rest.
    pub fn adopt(&self, workspace: Arc<dyn Workspace>) -> Arc<dyn Workspace> {
        let mut i = self.lock();
        if !i.open.values().any(|w| Arc::ptr_eq(w, &workspace)) {
            i.open.insert(format!("adopted:{}", workspace.id()), workspace.clone());
        }
        workspace
    }

    /// Where parallel sub-agents hand files to each other.
    pub fn shared(&self) -> Result<Arc<dyn Workspace>> {
        self.acquire("shared", false)
    }

    pub async fn release(&self, name: &str) {
        let held = self.lock().open.shift_remove(name);
        if let Some(w) = held {
            w.close().await;
        }
    }

    /// Stop every sandbox and remove what was temporary. One that will not
    /// stop does not keep the others running.
    pub async fn close(&self) {
        let open: Vec<Arc<dyn Workspace>> = self.lock().open.drain(..).map(|(_, w)| w).collect();
        for w in open {
            w.close().await;
        }
        let (root, temporary) = {
            let i = self.lock();
            (i.root.clone(), i.temporary)
        };
        if (self.ephemeral || temporary)
            && let Some(r) = root
        {
            let _ = std::fs::remove_dir_all(r);
        }
    }

    pub fn open_count(&self) -> usize {
        self.lock().open.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn the_jail_holds_and_changes_are_seen() {
        let dir = std::env::temp_dir().join(format!("ah_ws_{}", uuid::Uuid::new_v4().simple()));
        let ws = LocalWorkspace::new(&dir, WorkspaceOptions { allow_shell: true, ..Default::default() }).unwrap();
        let before = ws.snapshot().await.unwrap();
        assert_eq!(ws.write("notes/a.txt", "héllo".as_bytes(), false).await.unwrap(), 5);
        ws.write("notes/a.txt", b" again", true).await.unwrap();
        assert_eq!(ws.read("notes/a.txt").await.unwrap(), "héllo again");
        assert_eq!(ws.listdir(".").await.unwrap(), vec!["notes/"]);
        assert_eq!(ws.changed(&before).await.unwrap(), vec!["notes/a.txt"]);
        assert!(ws.resolve("../escape.txt").unwrap_err().to_string().contains("outside the workspace"));
        assert!(ws.resolve("notes/../../x").is_err());
        assert!(ws.remove(".").await.unwrap_err().to_string().contains("refusing to delete the workspace root"));
        let out = ws.shell("echo hi", Some(10.0)).await.unwrap();
        assert_eq!(out["returncode"], 0);
        assert!(out["stdout"].as_str().unwrap().contains("hi"));

        let shared: Arc<dyn Workspace> = Arc::new(ws);
        let tools = workspace_tools(shared.clone());
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["fs_read", "fs_write", "fs_list", "fs_delete", "shell"]);
        assert_eq!(tools[4].permission, Permission::Ask, "a shell on this machine asks first");
        assert_eq!(tools[3].permission, Permission::Ask);
        let ro = LocalWorkspace::new(dir.join("ro"), WorkspaceOptions { read_only: true, ..Default::default() }).unwrap();
        assert!(ro.write("x", b"y", false).await.unwrap_err().to_string().contains("read-only"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn the_broker_hands_out_and_cleans_up() {
        let broker = WorkspaceBroker::new(None::<PathBuf>);
        let a = broker.acquire("researcher", true).unwrap();
        let again = broker.acquire("researcher", true).unwrap();
        assert!(Arc::ptr_eq(&a, &again));
        let shared = broker.shared().unwrap();
        assert!(!Arc::ptr_eq(&a, &shared));
        a.write("x.txt", b"1", false).await.unwrap();
        let root = broker.root().unwrap();
        assert!(root.exists());
        broker.close().await;
        assert!(!root.exists(), "a temporary root is removed");
    }
}
