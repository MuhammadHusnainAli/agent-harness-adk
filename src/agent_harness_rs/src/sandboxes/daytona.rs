//! Daytona: a sandbox from an image or a snapshot. Needs an API key
//! (`DAYTONA_API_KEY`).
//!
//! Python drives Daytona through its SDK; this talks to Daytona's REST API
//! (`https://app.daytona.io/api`, bearer key) with `reqwest`, as the SDK does:
//!
//! - `POST /sandbox` creates one — from a snapshot, or from an image as a
//!   one-line build (`FROM <image>`), which is what the SDK sends for an image
//!   given by name — then `GET /sandbox/{id}` is polled until it is `started`;
//! - `GET /sandbox/{id}` picks one up, and `POST /sandbox/{id}/start` wakes one
//!   Daytona stopped for idleness;
//! - commands go to the sandbox's toolbox, `POST .../process/execute`, at the
//!   address `GET /sandbox/{id}/toolbox-proxy-url` gives (or the API's own
//!   `/toolbox/{id}/toolbox` route where that is not offered). The command is
//!   wrapped exactly as the SDK wraps it, environment included;
//! - `DELETE /sandbox/{id}` deletes it.
//!
//! `DAYTONA_API_URL` and `DAYTONA_TARGET` apply as they do to the SDK.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use reqwest::Method;
use serde_json::{Map, Value, json};

use super::base::{ExecResult, SandboxBackend};
use super::command::{Given, Opts, http_client, said, timed_out};
use crate::errors::{Error, Result};

pub const URL_OPTION: &str = "image";

const TIMED_OUT: &str = "daytona: timed out";

/// A Daytona sandbox.
///
/// ```no_run
/// # use agent_harness::sandboxes::sandbox;
/// # use serde_json::json;
/// let ws = sandbox(&json!("daytona"));                                                   // DAYTONA_API_KEY
/// let ws = sandbox(&json!({"sandbox": "daytona", "image": "python:3.12-slim", "network": false}));
/// let ws = sandbox(&json!({"sandbox": "daytona", "snapshot": "my-snapshot"}));
/// let ws = sandbox(&json!({"sandbox": "daytona", "sandbox_id": "..."}));                 // one already running
/// ```
///
/// Options:
/// - `image`: build the sandbox from this image.
/// - `snapshot`: or start it from this snapshot. Neither is Daytona's default.
/// - `api_key`, `api_url`, `target`: default to `DAYTONA_API_KEY`,
///   `DAYTONA_API_URL` and `DAYTONA_TARGET`.
/// - `ttl`: seconds of idleness after which Daytona stops the sandbox itself.
/// - `network`: false blocks all network access.
/// - `sandbox_id`: pick up an existing sandbox, and leave it afterwards. One
///   that Daytona has stopped for idleness is started again, files and all.
///   The same as `id`.
/// - `keep`: do not delete it at the end, so a later conversation can pick it
///   up by its id.
/// - `labels`: labels on the sandbox.
///
/// Daytona returns a command's stdout and stderr as one stream, so both arrive
/// in `stdout`.
pub struct DaytonaSandbox {
    pub image: Option<String>,
    pub snapshot: Option<String>,
    pub api_key: Option<String>,
    pub api_url: String,
    pub target: Option<String>,
    pub ttl: u64,
    pub network: bool,
    pub labels: Option<Map<String, Value>>,
    /// Set in the sandbox at creation, when the options carry the shared `env`.
    pub env: BTreeMap<String, String>,
    /// How long to wait between looks at a sandbox that is still starting.
    pub poll: Duration,
    given: Given,
    http: reqwest::Client,
    live: std::sync::Mutex<Option<Live>>,
}

#[derive(Clone)]
struct Live {
    id: String,
    toolbox: String,
    adopted: bool,
}

/// The command as the SDK sends it: the script base64-encoded and piped to
/// `sh`, each variable exported from base64 too, so no quoting can break it.
pub fn wrap_command(command: &str, env: &BTreeMap<String, String>) -> String {
    let enc = |s: &str| base64::engine::general_purpose::STANDARD.encode(s.as_bytes());
    let mut cmd = format!("echo '{}' | base64 -d | sh", enc(command));
    if !env.is_empty() {
        let exports: Vec<String> = env.iter().map(|(k, v)| format!("export {k}=$(echo '{}' | base64 -d)", enc(v))).collect();
        cmd = format!("{}; {cmd}", exports.join(";"));
    }
    format!("sh -c \"{cmd}\"")
}

impl DaytonaSandbox {
    pub fn new(options: &Map<String, Value>) -> Result<Self> {
        let o =
            Opts::new("daytona", options, &["image", "snapshot", "api_key", "api_url", "target", "ttl", "network", "sandbox_id", "labels", "client", "box"])?;
        o.refuse_objects(&["client", "box"])?;
        let image = o.text("image")?.filter(|i| !i.is_empty());
        let snapshot = o.text("snapshot")?.filter(|s| !s.is_empty());
        if image.is_some() && snapshot.is_some() {
            return Err(Error::config("a daytona sandbox starts from an image or a snapshot, not both"));
        }
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        Ok(Self {
            image,
            snapshot,
            api_key: o.text("api_key")?.filter(|k| !k.is_empty()),
            api_url: o.text("api_url")?.or_else(|| env("DAYTONA_API_URL")).unwrap_or_else(|| "https://app.daytona.io/api".into()),
            target: o.text("target")?.or_else(|| env("DAYTONA_TARGET")),
            ttl: o.int_or("ttl", Some(3600))?.unwrap_or(3600),
            network: o.flag("network", true)?,
            labels: o.mapping("labels")?.filter(|m| !m.is_empty()),
            env: o.env(),
            poll: Duration::from_secs(1),
            given: Given::new(o.text("id")?.or(o.text("sandbox_id")?)),
            http: http_client(),
            live: std::sync::Mutex::new(None),
        })
    }

    /// What `POST /sandbox` is sent.
    pub fn create_request(&self) -> Value {
        // Daytona counts idleness in minutes.
        let mut body = json!({"autoStopInterval": (self.ttl / 60).max(1)});
        if !self.env.is_empty() {
            body["env"] = json!(self.env);
        }
        if let Some(l) = &self.labels {
            body["labels"] = json!(l);
        }
        if !self.network {
            body["networkBlockAll"] = json!(true);
        }
        if let Some(t) = &self.target {
            body["target"] = json!(t);
        }
        match (&self.image, &self.snapshot) {
            (Some(image), _) => body["buildInfo"] = json!({"dockerfileContent": format!("FROM {image}\n"), "contextHashes": []}),
            (None, Some(snap)) => body["snapshot"] = json!(snap),
            (None, None) => {}
        }
        body
    }

    fn key(&self) -> Result<String> {
        self.api_key
            .clone()
            .or_else(|| std::env::var("DAYTONA_API_KEY").ok().filter(|k| !k.is_empty()))
            .ok_or_else(|| Error::config("the daytona sandbox needs an API key — set DAYTONA_API_KEY, or pass api_key="))
    }

    fn live(&self) -> Option<Live> {
        self.live.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    async fn call(&self, method: Method, url: String, body: Option<Value>, key: &str, timeout: f64) -> Result<(u16, Value)> {
        let mut req = self.http.request(method, url).bearer_auth(key).timeout(Duration::from_secs_f64(timeout.max(1.0)));
        if let Some(b) = body {
            req = req.json(&b);
        }
        let failed = |e: reqwest::Error| if e.is_timeout() { Error::Io(format!("{TIMED_OUT}: {e}")) } else { Error::Io(format!("daytona: {e}")) };
        let resp = req.send().await.map_err(failed)?;
        let status = resp.status().as_u16();
        let text = resp.text().await.map_err(failed)?;
        let value = serde_json::from_str(&text).unwrap_or(Value::String(text));
        if status == 401 || status == 403 {
            return Err(Error::config(format!("the daytona API refused the key ({status}): {}", said(&value, 300))));
        }
        Ok((status, value))
    }

    async fn api(&self, method: Method, path: &str, body: Option<Value>, key: &str, what: &str) -> Result<Value> {
        let (status, value) = self.call(method, format!("{}{path}", self.api_url.trim_end_matches('/')), body, key, 120.0).await?;
        if !(200..300).contains(&status) {
            return Err(Error::tool(format!("{what} ({status}): {}", said(&value, 300)), "sandbox"));
        }
        Ok(value)
    }

    /// Wait until the sandbox says it has started.
    async fn started(&self, mut state: Value, key: &str) -> Result<()> {
        let id = state["id"].as_str().unwrap_or("").to_string();
        loop {
            match state["state"].as_str().unwrap_or("").to_lowercase().as_str() {
                "started" => return Ok(()),
                s @ ("error" | "build_failed" | "destroyed") => {
                    let why = state["errorReason"].as_str().unwrap_or("");
                    return Err(Error::tool(format!("daytona sandbox {id} is {s}: {why}"), "sandbox"));
                }
                _ => {}
            }
            tokio::time::sleep(self.poll).await;
            state = self.api(Method::GET, &format!("/sandbox/{id}"), None, key, "daytona could not look at the sandbox").await?;
        }
    }
}

#[async_trait]
impl SandboxBackend for DaytonaSandbox {
    fn name(&self) -> &str {
        "daytona"
    }

    async fn start(&self, attach_id: Option<&str>) -> Result<String> {
        let key = self.key()?;
        let target = self.given.target(attach_id);
        let (box_, adopted) = match &target {
            Some((id, adopted)) => {
                let (status, mut found) = self.call(Method::GET, format!("{}/sandbox/{id}", self.api_url.trim_end_matches('/')), None, &key, 120.0).await?;
                if status == 404 {
                    return Err(Error::tool(format!("daytona sandbox {id} was not found: {}", said(&found, 300)), "sandbox"));
                }
                if !(200..300).contains(&status) {
                    return Err(Error::tool(format!("daytona could not pick up {id} ({status}): {}", said(&found, 300)), "sandbox"));
                }
                let state = found["state"].as_str().unwrap_or("").to_lowercase();
                if state.contains("stopped") || state.contains("archived") {
                    found = self.api(Method::POST, &format!("/sandbox/{id}/start"), None, &key, &format!("daytona could not start {id}")).await?;
                    if found["id"].is_null() {
                        found = json!({"id": id, "state": "starting"});
                    }
                }
                (found, *adopted)
            }
            None => (self.api(Method::POST, "/sandbox", Some(self.create_request()), &key, "daytona could not create a sandbox").await?, false),
        };
        let id = box_["id"].as_str().map(str::to_string).or_else(|| target.map(|t| t.0)).unwrap_or_default();
        let mut box_ = box_;
        box_["id"] = json!(id);
        // Recorded first, so a sandbox that never starts is still deleted at stop.
        let api = self.api_url.trim_end_matches('/').to_string();
        *self.live.lock().unwrap_or_else(|p| p.into_inner()) = Some(Live { id: id.clone(), toolbox: format!("{api}/toolbox/{id}/toolbox"), adopted });
        self.started(box_, &key).await?;
        // Newer Daytona serves the toolbox from a proxy of its own.
        if let Ok((200, proxy)) = self.call(Method::GET, format!("{api}/sandbox/{id}/toolbox-proxy-url"), None, &key, 30.0).await
            && let Some(url) = proxy["url"].as_str().filter(|u| !u.is_empty())
        {
            let toolbox = format!("{}/{id}", url.trim_end_matches('/'));
            if let Some(live) = self.live.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
                live.toolbox = toolbox;
            }
        }
        Ok(id)
    }

    async fn exec(&self, command: &str, cwd: Option<&str>, env: &BTreeMap<String, String>, timeout: f64) -> Result<ExecResult> {
        let live = self.live().ok_or_else(|| Error::tool("the daytona sandbox is not running", "sandbox"))?;
        let key = self.key()?;
        let mut body = json!({"command": wrap_command(command, env), "timeout": (timeout as u64).max(1)});
        if let Some(c) = cwd.filter(|c| !c.is_empty()) {
            body["cwd"] = json!(c);
        }
        let (status, done) = match self.call(Method::POST, format!("{}/process/execute", live.toolbox), Some(body), &key, timeout + 10.0).await {
            Err(Error::Io(m)) if m.starts_with(TIMED_OUT) => return Ok(timed_out(timeout)),
            other => other?,
        };
        if status == 408 || status == 504 {
            return Ok(timed_out(timeout));
        }
        if !(200..300).contains(&status) {
            return Err(Error::tool(format!("daytona exec failed ({status}): {}", said(&done, 300)), "sandbox"));
        }
        let code = done["exitCode"].as_i64().or_else(|| done["exit_code"].as_i64()).unwrap_or(0) as i32;
        Ok(ExecResult::new(code, done["result"].as_str().unwrap_or(""), ""))
    }

    async fn stop(&self, _id: &str) -> Result<()> {
        let Some(live) = self.live() else { return Ok(()) };
        if live.adopted {
            return Ok(());
        }
        let key = self.key()?;
        let (status, body) = self.call(Method::DELETE, format!("{}/sandbox/{}", self.api_url.trim_end_matches('/'), live.id), None, &key, 120.0).await?;
        if !(200..300).contains(&status) && status != 404 {
            return Err(Error::tool(format!("daytona could not delete {} ({status}): {}", live.id, said(&body, 300)), "sandbox"));
        }
        Ok(())
    }
}

/// `sandbox("daytona://python:3.12-slim")`.
pub fn build(options: &Map<String, Value>) -> Result<Arc<dyn SandboxBackend>> {
    Ok(Arc::new(DaytonaSandbox::new(options)?))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::Router;
    use axum::extract::{Path, State};
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::IntoResponse;
    use axum::routing::{get, post};

    use super::*;
    use crate::runtime::workspace::{Workspace, on_path};
    use crate::sandboxes::{Sandbox, SandboxOptions, SandboxWorkspace, WrapOptions, run_process};

    #[derive(Default)]
    struct Seen {
        base: String,
        /// Where commands run when nothing else is said: the sandbox's home.
        home: String,
        auth: Vec<String>,
        created: Vec<Value>,
        looked: Vec<String>,
        started: Vec<String>,
        deleted: Vec<String>,
        execs: Vec<Value>,
        real: bool,
    }

    type Shared = Arc<Mutex<Seen>>;

    fn auth(s: &Shared, h: &HeaderMap) {
        s.lock().unwrap().auth.push(h.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("").to_string());
    }

    async fn create(State(s): State<Shared>, h: HeaderMap, body: axum::Json<Value>) -> impl IntoResponse {
        auth(&s, &h);
        s.lock().unwrap().created.push(body.0);
        axum::Json(json!({"id": "dt-1", "state": "creating"}))
    }

    async fn look(State(s): State<Shared>, Path(id): Path<String>, h: HeaderMap) -> axum::response::Response {
        auth(&s, &h);
        s.lock().unwrap().looked.push(id.clone());
        match id.as_str() {
            "gone" => (StatusCode::NOT_FOUND, axum::Json(json!({"message": "Sandbox with ID gone not found"}))).into_response(),
            "dt-9" => {
                let state = if s.lock().unwrap().started.is_empty() { "stopped" } else { "started" };
                axum::Json(json!({"id": id, "state": state})).into_response()
            }
            _ => axum::Json(json!({"id": id, "state": "started"})).into_response(),
        }
    }

    async fn wake(State(s): State<Shared>, Path(id): Path<String>) -> impl IntoResponse {
        s.lock().unwrap().started.push(id.clone());
        axum::Json(json!({"id": id, "state": "starting"}))
    }

    async fn remove(State(s): State<Shared>, Path(id): Path<String>) -> StatusCode {
        s.lock().unwrap().deleted.push(id);
        StatusCode::OK
    }

    async fn proxy(State(s): State<Shared>) -> impl IntoResponse {
        let base = s.lock().unwrap().base.clone();
        axum::Json(json!({"url": format!("{base}/proxy")}))
    }

    /// The toolbox: runs the wrapped command in a local shell when told to.
    async fn execute(State(s): State<Shared>, Path(id): Path<String>, body: axum::Json<Value>) -> impl IntoResponse {
        let (real, home) = {
            let mut s = s.lock().unwrap();
            s.execs.push(json!({"id": id, "body": body.0.clone()}));
            (s.real, s.home.clone())
        };
        if !real {
            return axum::Json(json!({"exitCode": 0, "result": "/home/daytona\n"}));
        }
        let cwd = body.0["cwd"].as_str().unwrap_or(&home).to_string();
        let line = format!("cd '{cwd}' && {}", body.0["command"].as_str().unwrap());
        let (code, out, err) = run_process(vec!["sh".into(), "-c".into(), line], None, 30.0).await.unwrap().unwrap();
        let mut result = String::from_utf8_lossy(&out).into_owned();
        result.push_str(&String::from_utf8_lossy(&err));
        axum::Json(json!({"exitCode": code, "result": result}))
    }

    async fn serve() -> Shared {
        let seen: Shared = Arc::default();
        let app = Router::new()
            .route("/sandbox", post(create))
            .route("/sandbox/{id}", get(look).delete(remove))
            .route("/sandbox/{id}/start", post(wake))
            .route("/sandbox/{id}/toolbox-proxy-url", get(proxy))
            .route("/proxy/{id}/process/execute", post(execute))
            .with_state(seen.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        seen.lock().unwrap().base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        seen
    }

    fn backend(seen: &Shared, options: Value) -> Arc<DaytonaSandbox> {
        let mut options = options.as_object().cloned().unwrap();
        options.insert("api_url".into(), json!(seen.lock().unwrap().base.clone()));
        options.insert("api_key".into(), json!("dk"));
        let mut b = DaytonaSandbox::new(&options).unwrap();
        b.poll = Duration::from_millis(10);
        Arc::new(b)
    }

    #[tokio::test]
    async fn daytona_is_created_from_an_image_or_a_snapshot_and_deleted() {
        let seen = serve().await;
        let b = backend(&seen, json!({"image": "python:3.12-slim", "ttl": 1800, "network": false, "target": "eu", "labels": {"team": "x"}}));
        let sb = Sandbox::new(b, SandboxOptions::default()).unwrap();
        let ws = SandboxWorkspace::new(sb.clone(), WrapOptions::default());
        ws.start().await.unwrap();
        {
            let s = seen.lock().unwrap();
            assert_eq!(s.auth[0], "Bearer dk");
            assert_eq!(
                s.created[0],
                json!({"autoStopInterval": 30, "labels": {"team": "x"}, "networkBlockAll": true, "target": "eu",
                       "buildInfo": {"dockerfileContent": "FROM python:3.12-slim\n", "contextHashes": []}})
            );
            // Created as "creating": looked at until it had started.
            assert_eq!(s.looked, ["dt-1"]);
            // No working directory was named, so it is a folder under where it starts.
            assert_eq!(s.execs[0]["body"]["command"], wrap_command("pwd", &BTreeMap::new()));
            assert_eq!(s.execs[0]["id"], "dt-1");
        }
        assert_eq!(sb.workdir().await, "/home/daytona/workspace");
        assert_eq!(sb.id().await, "dt-1");
        sb.exec("ls", None, None, Some(45.0)).await.unwrap();
        {
            let s = seen.lock().unwrap();
            let last = &s.execs.last().unwrap()["body"];
            assert_eq!(last["cwd"], "/home/daytona/workspace");
            assert_eq!(last["timeout"], 45);
        }
        ws.close().await;
        assert_eq!(seen.lock().unwrap().deleted, ["dt-1"]);

        let snap = backend(&seen, json!({"snapshot": "base-snap"}));
        snap.start(None).await.unwrap();
        assert_eq!(seen.lock().unwrap().created[1], json!({"autoStopInterval": 60, "snapshot": "base-snap"}));
    }

    #[tokio::test]
    async fn one_already_running_is_woken_and_left_running() {
        let seen = serve().await;
        let sb = Sandbox::new(backend(&seen, json!({"sandbox_id": "dt-9"})), SandboxOptions::default()).unwrap();
        sb.start().await.unwrap();
        sb.stop(false).await.unwrap();
        {
            let s = seen.lock().unwrap();
            assert_eq!(s.started, ["dt-9"]);
            assert!(s.deleted.is_empty() && s.created.is_empty());
        }
        assert_eq!(sb.id().await, "dt-9");
        assert_eq!(sb.replaced().await.0, "");

        let sb = Sandbox::new(backend(&seen, json!({})), SandboxOptions { id: Some("gone".into()), ..Default::default() }).unwrap();
        sb.start().await.unwrap();
        assert_eq!(sb.replaced().await.0, "gone");
        assert_eq!(sb.id().await, "dt-1");
    }

    #[test]
    fn options_and_keys_are_checked() {
        let both = DaytonaSandbox::new(json!({"image": "a", "snapshot": "b"}).as_object().unwrap()).err().unwrap();
        assert!(both.to_string().contains("not both"));
        if std::env::var("DAYTONA_API_KEY").is_err() {
            let err = DaytonaSandbox::new(&Map::new()).unwrap().key().unwrap_err();
            assert!(matches!(err, Error::Configuration(_)) && err.to_string().contains("set DAYTONA_API_KEY"));
        }
        assert!(DaytonaSandbox::new(json!({"box": 1}).as_object().unwrap()).err().unwrap().to_string().contains("SDK object"));
        let mut env = BTreeMap::new();
        env.insert("A".to_string(), "it's".to_string());
        assert_eq!(wrap_command("ls", &env), "sh -c \"export A=$(echo 'aXQncw==' | base64 -d); echo 'bHM=' | base64 -d | sh\"");
    }

    #[tokio::test]
    async fn daytona_passes_the_self_check() {
        if !on_path("sh") || !on_path("base64") || !on_path("stat") {
            return; // the fake toolbox runs commands in a local shell
        }
        let seen = serve().await;
        let dir = std::env::temp_dir().join(format!("ah_dt_{}", crate::types::new_id("")));
        std::fs::create_dir_all(&dir).unwrap();
        let root = dir.to_string_lossy().replace('\\', "/");
        {
            let mut s = seen.lock().unwrap();
            s.real = true;
            s.home = root.clone();
        }
        let ws = SandboxWorkspace::new(Sandbox::new(backend(&seen, json!({})), SandboxOptions::default()).unwrap(), WrapOptions::default());
        let report = ws.check().await;
        assert_eq!(report["ok"], true, "{report:#}");
        let mut env = BTreeMap::new();
        env.insert("GREETING".to_string(), "hi 'there'".to_string());
        let r = ws.sandbox.exec("echo $GREETING; exit 3", None, Some(&env), None).await.unwrap();
        assert_eq!((r.returncode, r.stdout.as_str()), (3, "hi 'there'\n"));
        // Settled at start as where it starts + /workspace (Git's sh on Windows says /c/...).
        assert!(ws.root().await.ends_with("/workspace"));
        assert!(dir.join("workspace").is_dir());
        ws.close().await;
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Real Daytona: `AH_LIVE_DAYTONA=1` and `DAYTONA_API_KEY`.
    #[tokio::test]
    async fn live_daytona_passes_the_self_check() {
        if std::env::var("AH_LIVE_DAYTONA").as_deref() != Ok("1") {
            return;
        }
        let ws = crate::sandboxes::sandbox(&json!("daytona")).unwrap();
        let report = ws.check().await;
        ws.destroy().await.unwrap();
        assert_eq!(report["ok"], true, "{report:#}");
    }
}
