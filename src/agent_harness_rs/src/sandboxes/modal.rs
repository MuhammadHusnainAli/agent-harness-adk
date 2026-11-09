//! Modal: a sandbox container on Modal.
//!
//! **Not available from Rust.** Python drives Modal through its SDK, and the
//! SDK speaks to Modal over gRPC (protobuf over HTTP/2, `api.modal.com`) —
//! creating an app, a sandbox, running a command in it and terminating it are
//! all gRPC calls. Modal publishes no REST equivalent, and this crate carries
//! no gRPC or protobuf stack. So the options are read and checked as Python
//! reads them (a misspelt one is still refused at once), and starting says
//! plainly — as a configuration error, so no run is spent finding out — that
//! this backend cannot start here, and what to use instead.
//!
//! Modal authenticates with its own token — `modal token new`, or
//! `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use super::base::{ExecResult, SandboxBackend};
use super::command::Opts;
use crate::errors::{Error, Result};

pub const URL_OPTION: &str = "image";

const UNAVAILABLE: &str = "the modal sandbox is not available in the Rust crate: Modal's API is gRPC-only (no public HTTP API), and this crate \
                           carries no gRPC stack. Use e2b, daytona, docker or kubernetes — or register a backend of your own with register_sandbox";

/// A Modal sandbox, as Python configures one.
///
/// Options:
/// - `image`: a registry image. None is Modal's debian-slim.
/// - `app`: the Modal app the sandbox is created under; made if missing.
/// - `ttl`: seconds the sandbox lives before Modal ends it.
/// - `network`: false blocks all network access.
/// - `cpu`, `memory`: cores, and MiB.
/// - `sandbox_id`: pick up an existing sandbox, and leave it afterwards.
pub struct ModalSandbox {
    pub image: Option<String>,
    pub app: String,
    pub ttl: u64,
    pub network: bool,
    pub cpu: Option<f64>,
    pub memory: Option<u64>,
    pub sandbox_id: Option<String>,
    pub workdir: String,
}

impl ModalSandbox {
    pub fn new(options: &Map<String, Value>) -> Result<Self> {
        let o = Opts::new("modal", options, &["image", "app", "ttl", "network", "cpu", "memory", "sandbox_id", "box"])?;
        o.refuse_objects(&["box"])?;
        Ok(Self {
            image: o.text("image")?.filter(|i| !i.is_empty()),
            app: o.text("app")?.unwrap_or_else(|| "agent-harness".into()),
            ttl: o.int_or("ttl", Some(3600))?.unwrap_or(3600),
            network: o.flag("network", true)?,
            cpu: o.number("cpu")?,
            memory: o.int_or("memory", None)?,
            sandbox_id: o.text("id")?.or(o.text("sandbox_id")?),
            workdir: o.text("workdir")?.unwrap_or_else(|| "/workspace".into()),
        })
    }
}

#[async_trait]
impl SandboxBackend for ModalSandbox {
    fn name(&self) -> &str {
        "modal"
    }

    async fn start(&self, _attach_id: Option<&str>) -> Result<String> {
        Err(Error::config(UNAVAILABLE))
    }

    async fn exec(&self, _command: &str, _cwd: Option<&str>, _env: &BTreeMap<String, String>, _timeout: f64) -> Result<ExecResult> {
        Err(Error::config(UNAVAILABLE))
    }
}

/// `sandbox("modal://python:3.12-slim")`: options checked, start refused.
pub fn build(options: &Map<String, Value>) -> Result<Arc<dyn SandboxBackend>> {
    Ok(Arc::new(ModalSandbox::new(options)?))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::runtime::workspace::Workspace;
    use crate::sandboxes::sandbox;

    #[tokio::test]
    async fn modal_reads_its_options_and_says_why_it_cannot_start() {
        let b =
            ModalSandbox::new(json!({"image": "python:3.12-slim", "app": "jobs", "ttl": 600, "network": false, "cpu": 2, "memory": 4096}).as_object().unwrap())
                .unwrap();
        assert_eq!(
            (b.image.as_deref(), b.app.as_str(), b.ttl, b.network, b.cpu, b.memory),
            (Some("python:3.12-slim"), "jobs", 600, false, Some(2.0), Some(4096))
        );
        let plain = ModalSandbox::new(&Map::new()).unwrap();
        assert_eq!((plain.image, plain.app.as_str(), plain.ttl), (None, "agent-harness", 3600));

        let err = sandbox(&json!({"sandbox": "modal", "no_such": 1})).err().unwrap().to_string();
        assert!(err.starts_with("modal sandbox: got an unexpected keyword argument 'no_such'"), "{err}");

        // The run ends before it costs anything, with the reason.
        let ws = sandbox(&json!("modal://python:3.12-slim")).unwrap();
        let err = ws.start().await.unwrap_err();
        assert!(matches!(err, Error::Configuration(_)) && err.to_string().contains("gRPC-only"), "{err}");
        assert!(!ws.exists("a").await);
    }
}
