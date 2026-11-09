//! Sandboxes: somewhere other than this machine for an agent to work.
//!
//! ```no_run
//! # use agent_harness::sandboxes::sandbox;
//! # use serde_json::json;
//! let ws = sandbox(&json!("docker"));
//! let ws = sandbox(&json!("docker://node:22"));                       // the URL names the image
//! let ws = sandbox(&json!({"sandbox": "e2b", "template": "base"}));
//! let ws = sandbox(&json!("ssh://agent@build-7"));
//! ```
//!
//! A sandbox is a [`Workspace`](crate::runtime::workspace::Workspace), so
//! everything that works on a local folder — the file tools, the shell,
//! `run_python`, `parse_document`, cowork handing back the files it wrote —
//! works inside one, unchanged.
//!
//! | name | what it is | needs |
//! |---|---|---|
//! | `docker` | a long-lived container | the `docker` CLI |
//! | `podman` | the same, with Podman | the `podman` CLI |
//! | `kubernetes` | a pod, created or attached to | `kubectl` |
//! | `ssh` | a machine you can already reach | `ssh` |
//! | `e2b` | a Firecracker micro-VM | `E2B_API_KEY` |
//! | `daytona` | a sandbox from an image or snapshot | `DAYTONA_API_KEY` |
//! | `modal` | a container on Modal | a Modal token |
//! | `command` | anything you can put in front of `sh -c` | — |
//!
//! A sandbox that is not ready says exactly what is missing. Your own is one
//! trait method — see [`SandboxBackend`] — and [`register_sandbox`] gives it a name.

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, RwLock};

use serde_json::{Map, Value, json};

use crate::errors::{Error, Result};
use crate::runtime::workspace::on_path;

pub mod base;
pub mod command;
pub mod daytona;
pub mod docker;
pub mod e2b;
pub mod modal;

pub use base::{ExecResult, ProcessExec, Runner, Sandbox, SandboxBackend, SandboxOptions, SandboxWorkspace, WrapOptions, run_process};

/// Builds a backend from its own options (what is left once the shared
/// sandbox and workspace options are taken out).
pub type BackendFactory = Arc<dyn Fn(&Map<String, Value>) -> Result<Arc<dyn SandboxBackend>> + Send + Sync>;

struct Entry {
    factory: BackendFactory,
    /// What a URL's remainder means: `docker://python:3.12` sets `image`.
    url_option: &'static str,
    binary: Vec<&'static str>,
    env: Vec<&'static str>,
    summary: String,
}

/// A backend's own constructor, before defaults are merged in.
type Build = fn(&Map<String, Value>) -> Result<Arc<dyn SandboxBackend>>;

fn with(defaults: Value, build: Build) -> BackendFactory {
    Arc::new(move |options| {
        let mut merged = defaults.as_object().cloned().unwrap_or_default();
        merged.extend(options.clone());
        build(&merged)
    })
}

static REGISTRY: LazyLock<RwLock<BTreeMap<String, Entry>>> = LazyLock::new(|| {
    let mut m = BTreeMap::new();
    let mut add = |name: &str, factory: BackendFactory, url: &'static str, binary: Vec<&'static str>, env: Vec<&'static str>, summary: &str| {
        m.insert(name.to_string(), Entry { factory, url_option: url, binary, env, summary: summary.into() });
    };
    add("docker", with(json!({}), docker::build), docker::URL_OPTION, vec!["docker"], vec![], "a long-lived container on this machine");
    add("podman", with(json!({"cli": "podman"}), docker::build), docker::URL_OPTION, vec!["podman"], vec![], "a long-lived container, with Podman");
    add("kubernetes", with(json!({}), command::build_kubernetes), command::KUBERNETES_URL_OPTION, vec!["kubectl"], vec![], "a pod, created or attached to");
    add("ssh", with(json!({}), command::build_ssh), command::SSH_URL_OPTION, vec!["ssh"], vec![], "a machine reached over SSH");
    add("command", with(json!({}), command::build_command), command::COMMAND_URL_OPTION, vec![], vec![], "anything you can put in front of `sh -c`");
    add("e2b", with(json!({}), e2b::build), e2b::URL_OPTION, vec![], vec!["E2B_API_KEY"], "a Firecracker micro-VM on E2B");
    add("daytona", with(json!({}), daytona::build), daytona::URL_OPTION, vec![], vec!["DAYTONA_API_KEY"], "a Daytona sandbox, from an image or a snapshot");
    add("modal", with(json!({}), modal::build), modal::URL_OPTION, vec![], vec!["MODAL_TOKEN_ID"], "a sandbox container on Modal");
    RwLock::new(m)
});

const ALIASES: [(&str, &str); 4] = [("k8s", "kubernetes"), ("kube", "kubernetes"), ("container", "docker"), ("pod", "kubernetes")];

/// Options that belong to the sandbox lifecycle, shared by every backend.
const SHARED: [&str; 8] = ["id", "workdir", "env", "timeout", "start_timeout", "keep", "on_missing", "python"];
/// Options that belong to the workspace wrapped round a sandbox, not to it.
const WRAPPING: [&str; 4] = ["read_only", "allow_shell", "max_file_bytes", "export_dir"];

/// Add your own: `register_sandbox("mine", factory, "", &[], &[], "my platform")`.
pub fn register_sandbox(name: &str, factory: BackendFactory, url_option: &'static str, binary: &[&'static str], env: &[&'static str], summary: &str) {
    REGISTRY
        .write()
        .unwrap_or_else(|p| p.into_inner())
        .insert(name.trim().to_lowercase(), Entry { factory, url_option, binary: binary.to_vec(), env: env.to_vec(), summary: summary.into() });
}

fn key(name: &str) -> Result<String> {
    let k = name.trim().to_lowercase();
    let k = ALIASES.iter().find(|(a, _)| *a == k).map(|(_, t)| t.to_string()).unwrap_or(k);
    let r = REGISTRY.read().unwrap_or_else(|p| p.into_inner());
    if r.contains_key(&k) {
        Ok(k)
    } else {
        Err(Error::config(format!("unknown sandbox '{name}'; known: {}", r.keys().cloned().collect::<Vec<_>>().join(", "))))
    }
}

/// What stands between this sandbox and starting. Empty if nothing does.
pub fn missing(name: &str) -> Result<Vec<String>> {
    let k = key(name)?;
    let r = REGISTRY.read().unwrap_or_else(|p| p.into_inner());
    let e = &r[&k];
    let mut gaps: Vec<String> = e.binary.iter().filter(|b| !on_path(b)).map(|b| format!("`{b}` on PATH")).collect();
    if !e.env.is_empty() && !e.env.iter().any(|v| std::env::var(v).is_ok_and(|x| !x.is_empty())) {
        gaps.push(format!("set {}", e.env.join(" or ")));
    }
    Ok(gaps)
}

/// Which sandboxes could start right now, without installing or setting
/// anything. `true` is not a promise the service will answer — for that,
/// `sandbox(..)?.check().await`.
pub fn available_sandboxes() -> BTreeMap<String, bool> {
    let names: Vec<String> = REGISTRY.read().unwrap_or_else(|p| p.into_inner()).keys().cloned().collect();
    names.into_iter().map(|n| (n.clone(), missing(&n).is_ok_and(|m| m.is_empty()))).collect()
}

/// Every sandbox, what it is, and what it still needs.
pub fn describe_sandboxes() -> Vec<Value> {
    let rows: Vec<(String, String)> = REGISTRY.read().unwrap_or_else(|p| p.into_inner()).iter().map(|(k, e)| (k.clone(), e.summary.clone())).collect();
    rows.into_iter()
        .map(|(name, summary)| {
            let gaps = missing(&name).unwrap_or_default();
            json!({"name": name, "summary": summary, "ready": gaps.is_empty(), "missing": gaps})
        })
        .collect()
}

fn shared_options(o: &mut Map<String, Value>) -> SandboxOptions {
    let mut s = SandboxOptions::default();
    let take = |o: &mut Map<String, Value>, k: &str| o.remove(k).filter(|v| !v.is_null());
    if let Some(v) = take(o, "id") {
        s.id = v.as_str().map(str::to_string);
    }
    if let Some(v) = take(o, "workdir") {
        s.workdir = v.as_str().map(str::to_string);
    }
    if let Some(Value::Object(env)) = take(o, "env") {
        s.env = env.into_iter().map(|(k, v)| (k, v.as_str().map_or_else(|| v.to_string(), str::to_string))).collect();
    }
    if let Some(v) = take(o, "timeout").and_then(|v| v.as_f64()) {
        s.timeout = v;
    }
    if let Some(v) = take(o, "start_timeout").and_then(|v| v.as_f64()) {
        s.start_timeout = v;
    }
    if let Some(v) = take(o, "keep") {
        s.keep = v.as_bool();
    }
    if let Some(v) = take(o, "on_missing").and_then(|v| v.as_str().map(str::to_string)) {
        s.on_missing = v;
    }
    if let Some(v) = take(o, "python").and_then(|v| v.as_str().map(str::to_string)) {
        s.python = v;
    }
    s
}

fn wrap_options(o: &mut Map<String, Value>) -> WrapOptions {
    let mut w = WrapOptions::default();
    if let Some(v) = o.remove("read_only").and_then(|v| v.as_bool()) {
        w.read_only = v;
    }
    if let Some(v) = o.remove("allow_shell").and_then(|v| v.as_bool()) {
        w.allow_shell = v;
    }
    if let Some(v) = o.remove("max_file_bytes").and_then(|v| v.as_u64()) {
        w.max_file_bytes = v;
    }
    if let Some(v) = o.remove("export_dir").and_then(|v| v.as_str().map(std::path::PathBuf::from)) {
        w.export_dir = Some(v);
    }
    w
}

/// The backend a name builds, from the options left for it.
pub fn sandbox_backend(name: &str, options: &Map<String, Value>) -> Result<Arc<dyn SandboxBackend>> {
    let k = key(name)?;
    let factory = REGISTRY.read().unwrap_or_else(|p| p.into_inner())[&k].factory.clone();
    factory(options)
}

/// A sandbox workspace from a name, a URL or a mapping.
///
/// It starts on first use and is stopped when the harness's workspaces are
/// closed, or sooner with `close()`.
pub fn sandbox(spec: &Value) -> Result<SandboxWorkspace> {
    let mut options: Map<String, Value> = match spec {
        Value::String(s) => {
            let mut m = Map::new();
            m.insert("sandbox".into(), json!(s));
            m
        }
        Value::Object(m) => m.clone(),
        other => return Err(Error::config(format!("a sandbox is a name, a URL or a mapping with `sandbox:` — got {other}"))),
    };
    let name = options.remove("sandbox").or_else(|| options.remove("name")).and_then(|v| v.as_str().map(str::to_string));
    let Some(name) = name.filter(|n| !n.is_empty()) else {
        return Err(Error::config("a sandbox given as a mapping needs `sandbox:` — its name"));
    };
    let (name, rest) = name.split_once("://").map_or((name.clone(), String::new()), |(a, b)| (a.to_string(), b.to_string()));
    let k = key(&name)?;
    if !rest.is_empty() {
        let option = REGISTRY.read().unwrap_or_else(|p| p.into_inner())[&k].url_option;
        if option.is_empty() {
            return Err(Error::config(format!("the {k} sandbox takes no URL — pass its options by name")));
        }
        options.entry(option).or_insert(json!(rest));
    }
    let wrap = wrap_options(&mut options);
    let shared = shared_options(&mut options);
    let backend = sandbox_backend(&k, &options)?;
    Ok(SandboxWorkspace::new(Sandbox::new(backend, shared)?, wrap))
}

/// Keys every backend's options may hold that are not its own.
pub fn is_shared_option(key: &str) -> bool {
    SHARED.contains(&key) || WRAPPING.contains(&key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_urls_and_needs() {
        assert!(sandbox(&json!("nope")).unwrap_err().to_string().starts_with("unknown sandbox 'nope'"));
        assert!(sandbox(&json!({"image": "x"})).unwrap_err().to_string().contains("needs `sandbox:`"));
        assert_eq!(key("K8S").unwrap(), "kubernetes");
        let all = describe_sandboxes();
        assert_eq!(all.len(), 8);
        assert!(available_sandboxes()["command"]);
        let e2b = missing("e2b").unwrap();
        if std::env::var("E2B_API_KEY").is_err() {
            assert_eq!(e2b, vec!["set E2B_API_KEY"]);
        }
    }
}
