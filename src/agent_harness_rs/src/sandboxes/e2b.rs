//! E2B: a Firecracker micro-VM per sandbox. Needs an API key (`E2B_API_KEY`).
//!
//! Python drives E2B through its SDK; this talks to the same two HTTP services
//! the SDK does, with `reqwest`:
//!
//! - the control API (`https://api.e2b.app`, `X-API-Key`): `POST /sandboxes` to
//!   create, `POST /sandboxes/{id}/connect` to pick one up (which renews its
//!   lifetime and wakes a paused one), `DELETE /sandboxes/{id}` to kill;
//! - the sandbox's own daemon, envd (`https://49983-{id}.e2b.app`): commands
//!   through `process.Process/Start` — a Connect streaming call, spoken in its
//!   JSON form, so no protobuf is needed — and whole files through
//!   `GET`/`POST /files`.
//!
//! `E2B_DOMAIN` and `E2B_API_URL` point it elsewhere, as they do the SDK.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use reqwest::Method;
use serde_json::{Map, Value, json};

use super::base::{ExecResult, SandboxBackend};
use super::command::{Given, Opts, above, http_client, said, timed_out};
use crate::errors::{Error, Result};

pub const URL_OPTION: &str = "template";

const ENVD_PORT: u16 = 49983;
/// envd takes the user to run as from HTTP basic auth: `user`, no password.
const AS_USER: &str = "Basic dXNlcjo=";

/// An E2B sandbox.
///
/// ```no_run
/// # use agent_harness::sandboxes::sandbox;
/// # use serde_json::json;
/// let ws = sandbox(&json!("e2b"));                                              // E2B_API_KEY from the environment
/// let ws = sandbox(&json!({"sandbox": "e2b", "template": "my-template", "ttl": 1800}));
/// let ws = sandbox(&json!({"sandbox": "e2b", "sandbox_id": "i1a2b3..."}));       // one that is already running
/// ```
///
/// Options:
/// - `template`: the sandbox template. None is E2B's base image.
/// - `api_key`: defaults to `E2B_API_KEY`.
/// - `ttl`: seconds the sandbox lives. E2B ends it after that whatever happens
///   here, so one we lose track of does not bill for ever.
/// - `network`: false cuts the sandbox off from the internet.
/// - `sandbox_id`: pick up a sandbox that is already running instead of
///   creating one, and leave it running afterwards. The same as `id`.
/// - `keep`: do not kill it at the end, so that a later conversation can pick
///   it up by its id — within `ttl`, which picking it up renews.
/// - `metadata`: labels shown in the E2B dashboard.
pub struct E2bSandbox {
    pub template: Option<String>,
    pub api_key: Option<String>,
    pub ttl: u64,
    pub network: bool,
    pub metadata: Option<Map<String, Value>>,
    /// Set in the sandbox at creation, when the options carry the shared `env`.
    pub env: BTreeMap<String, String>,
    pub workdir: String,
    /// The control API: `E2B_API_URL`, else `https://api.<domain>`.
    pub api_url: String,
    /// `E2B_DOMAIN`, else `e2b.app`.
    pub domain: String,
    /// Where envd answers, instead of `https://49983-<id>.<domain>` (tests, proxies).
    pub envd_url: Option<String>,
    given: Given,
    http: reqwest::Client,
    live: std::sync::Mutex<Option<Live>>,
}

#[derive(Clone)]
struct Live {
    id: String,
    envd: String,
    token: Option<String>,
    /// Named in the options: not ours to kill.
    adopted: bool,
}

/// One Connect message: a flags byte, a big-endian length, the JSON.
fn envelope(flags: u8, message: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(message).unwrap_or_default();
    let mut out = Vec::with_capacity(body.len() + 5);
    out.push(flags);
    out.extend((body.len() as u32).to_be_bytes());
    out.extend(body);
    out
}

/// The messages of a Connect stream, in order.
fn frames(mut body: &[u8]) -> Vec<(u8, Value)> {
    let mut out = Vec::new();
    while body.len() >= 5 {
        let flags = body[0];
        let len = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
        let Some(message) = body.get(5..5 + len) else { break };
        out.push((flags, serde_json::from_slice(message).unwrap_or(Value::Null)));
        body = &body[5 + len..];
    }
    out
}

fn b64(v: &Value) -> Vec<u8> {
    v.as_str().and_then(|s| base64::engine::general_purpose::STANDARD.decode(s).ok()).unwrap_or_default()
}

impl E2bSandbox {
    pub fn new(options: &Map<String, Value>) -> Result<Self> {
        let o = Opts::new("e2b", options, &["template", "api_key", "ttl", "network", "sandbox_id", "metadata", "client"])?;
        o.refuse_objects(&["client"])?;
        let domain = std::env::var("E2B_DOMAIN").ok().filter(|d| !d.is_empty()).unwrap_or_else(|| "e2b.app".into());
        let api_url = std::env::var("E2B_API_URL").ok().filter(|d| !d.is_empty()).unwrap_or_else(|| format!("https://api.{domain}"));
        Ok(Self {
            template: o.text("template")?.filter(|t| !t.is_empty()),
            api_key: o.text("api_key")?.filter(|k| !k.is_empty()),
            ttl: o.int_or("ttl", Some(3600))?.unwrap_or(3600),
            network: o.flag("network", true)?,
            metadata: o.mapping("metadata")?.filter(|m| !m.is_empty()),
            env: o.env(),
            workdir: o.text("workdir")?.unwrap_or_else(|| "/home/user/workspace".into()),
            api_url,
            domain,
            envd_url: None,
            given: Given::new(o.text("id")?.or(o.text("sandbox_id")?)),
            http: http_client(),
            live: std::sync::Mutex::new(None),
        })
    }

    /// The sandbox's id; "" before start.
    pub fn sandbox_id(&self) -> String {
        self.live().map(|l| l.id).unwrap_or_default()
    }

    fn live(&self) -> Option<Live> {
        self.live.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    fn running(&self) -> Result<Live> {
        self.live().ok_or_else(|| Error::tool("the e2b sandbox is not running", "sandbox"))
    }

    fn key(&self) -> Result<String> {
        self.api_key
            .clone()
            .or_else(|| std::env::var("E2B_API_KEY").ok().filter(|k| !k.is_empty()))
            .ok_or_else(|| Error::config("the e2b sandbox needs an API key — set E2B_API_KEY, or pass api_key="))
    }

    async fn api(&self, method: Method, path: &str, body: Option<Value>, key: &str) -> Result<(u16, Value)> {
        let mut req =
            self.http.request(method, format!("{}{path}", self.api_url.trim_end_matches('/'))).header("X-API-Key", key).timeout(Duration::from_secs(120));
        if let Some(b) = body {
            req = req.json(&b);
        }
        let resp = req.send().await.map_err(|e| Error::Io(format!("e2b: {e}")))?;
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        let value = serde_json::from_str(&text).unwrap_or(Value::String(text));
        if status == 401 || status == 403 {
            return Err(Error::config(format!("the e2b API refused the key ({status}): {}", said(&value, 300))));
        }
        Ok((status, value))
    }

    fn envd(&self, live: &Live, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let req = req.header(reqwest::header::AUTHORIZATION, AS_USER);
        match &live.token {
            Some(t) => req.header("X-Access-Token", t),
            None => req,
        }
    }
}

#[async_trait]
impl SandboxBackend for E2bSandbox {
    fn name(&self) -> &str {
        "e2b"
    }

    async fn start(&self, attach_id: Option<&str>) -> Result<String> {
        let key = self.key()?;
        let target = self.given.target(attach_id);
        let (status, body, adopted) = match &target {
            Some((id, adopted)) => {
                // Connecting renews its lifetime, and wakes one that was paused.
                let (status, body) = self.api(Method::POST, &format!("/sandboxes/{id}/connect"), Some(json!({"timeout": self.ttl})), &key).await?;
                (status, body, *adopted)
            }
            None => {
                let mut create = json!({"templateID": self.template.as_deref().unwrap_or("base"), "timeout": self.ttl, "secure": true});
                if let Some(m) = &self.metadata {
                    create["metadata"] = json!(m);
                }
                if !self.env.is_empty() {
                    create["envVars"] = json!(self.env);
                }
                if !self.network {
                    create["allow_internet_access"] = json!(false);
                }
                let (status, body) = self.api(Method::POST, "/sandboxes", Some(create), &key).await?;
                (status, body, false)
            }
        };
        if !(200..300).contains(&status) {
            let what = match &target {
                Some((id, _)) if status == 404 => format!("e2b sandbox {id} was not found"),
                Some((id, _)) => format!("e2b could not pick up {id} ({status})"),
                None => format!("e2b could not create a sandbox ({status})"),
            };
            return Err(Error::tool(format!("{what}: {}", said(&body, 300)), "sandbox"));
        }
        let id = body["sandboxID"].as_str().map(str::to_string).or_else(|| target.map(|t| t.0)).unwrap_or_default();
        let domain = body["domain"].as_str().filter(|d| !d.is_empty()).unwrap_or(&self.domain);
        let envd = self.envd_url.clone().unwrap_or_else(|| format!("https://{ENVD_PORT}-{id}.{domain}"));
        let token = body["envdAccessToken"].as_str().filter(|t| !t.is_empty()).map(str::to_string);
        *self.live.lock().unwrap_or_else(|p| p.into_inner()) = Some(Live { id: id.clone(), envd, token, adopted });
        Ok(id)
    }

    async fn exec(&self, command: &str, cwd: Option<&str>, env: &BTreeMap<String, String>, timeout: f64) -> Result<ExecResult> {
        let live = self.running()?;
        let home = above(&self.workdir);
        let mut process = json!({"cmd": "/bin/bash", "args": ["-l", "-c", command], "cwd": cwd.filter(|c| !c.is_empty()).unwrap_or(&home)});
        if !env.is_empty() {
            process["envs"] = json!(env);
        }
        let req = self
            .envd(&live, self.http.post(format!("{}/process.Process/Start", live.envd.trim_end_matches('/'))))
            .header(reqwest::header::CONTENT_TYPE, "application/connect+json")
            .header("Connect-Protocol-Version", "1")
            .header("Connect-Timeout-Ms", ((timeout.max(1.0) * 1000.0) as u64).to_string())
            .body(envelope(0, &json!({"process": process})));
        let sent = tokio::time::timeout(Duration::from_secs_f64(timeout.max(1.0) + 5.0), async {
            let resp = req.send().await?;
            let status = resp.status().as_u16();
            Ok::<_, reqwest::Error>((status, resp.bytes().await?))
        })
        .await;
        let Ok(sent) = sent else { return Ok(timed_out(timeout)) };
        let (status, bytes) = sent.map_err(|e| Error::Io(format!("e2b exec: {e}")))?;
        if status != 200 {
            let err: Value = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
            if err["code"] == "deadline_exceeded" {
                return Ok(timed_out(timeout));
            }
            return Err(Error::tool(format!("e2b exec failed ({status}): {}", said(&err, 300)), "sandbox"));
        }
        let (mut out, mut errs, mut code) = (Vec::new(), Vec::new(), None);
        for (flags, message) in frames(&bytes) {
            if flags & 0x02 != 0 {
                // The end of the stream: trailers, and an error if there was one.
                let e = &message["error"];
                if e["code"] == "deadline_exceeded" {
                    return Ok(timed_out(timeout));
                }
                if !e.is_null() && code.is_none() {
                    return Err(Error::tool(format!("e2b exec failed: {}", said(e, 300)), "sandbox"));
                }
                continue;
            }
            let event = &message["event"];
            if let Some(data) = event.get("data") {
                out.extend(b64(&data["stdout"]));
                errs.extend(b64(&data["stderr"]));
            }
            if let Some(end) = event.get("end") {
                // proto3 leaves a zero exit code out.
                code = Some(end["exitCode"].as_i64().unwrap_or(0) as i32);
            }
        }
        let Some(code) = code else {
            return Err(Error::tool("e2b exec ended without saying how the command exited", "sandbox"));
        };
        // A non-zero exit is an answer, not a failure of the sandbox.
        Ok(ExecResult::new(code, String::from_utf8_lossy(&out), String::from_utf8_lossy(&errs)))
    }

    async fn stop(&self, _id: &str) -> Result<()> {
        let Some(live) = self.live() else { return Ok(()) };
        if live.adopted {
            return Ok(());
        }
        let key = self.key()?;
        let (status, body) = self.api(Method::DELETE, &format!("/sandboxes/{}", live.id), None, &key).await?;
        if !(200..300).contains(&status) && status != 404 {
            return Err(Error::tool(format!("e2b could not kill {} ({status}): {}", live.id, said(&body, 300)), "sandbox"));
        }
        Ok(())
    }

    /// Its own file API for whole files.
    async fn read_bytes(&self, path: &str, max_bytes: u64) -> Option<Result<Vec<u8>>> {
        let live = self.live()?;
        let url = format!("{}/files", live.envd.trim_end_matches('/'));
        let read = async {
            let resp = self
                .envd(&live, self.http.get(url).query(&[("path", path), ("username", "user")]))
                .timeout(Duration::from_secs(300))
                .send()
                .await
                .map_err(|e| Error::Io(format!("e2b: {e}")))?;
            let status = resp.status().as_u16();
            if status == 404 || status == 400 {
                return Err(Error::Io("missing".into()));
            }
            let too_big = || Error::tool(format!("{path} is larger than the {max_bytes}-byte limit"), "fs_read");
            if resp.content_length().is_some_and(|n| n > max_bytes) && (200..300).contains(&status) {
                return Err(too_big());
            }
            let bytes = resp.bytes().await.map_err(|e| Error::Io(format!("e2b: {e}")))?;
            if !(200..300).contains(&status) {
                return Err(Error::tool(
                    format!("could not read {path}: {}", String::from_utf8_lossy(&bytes).trim().chars().take(300).collect::<String>()),
                    "fs_read",
                ));
            }
            if bytes.len() as u64 > max_bytes {
                return Err(too_big());
            }
            Ok(bytes.to_vec())
        };
        Some(read.await)
    }

    /// Whole files through its file API; appending is left to the shell.
    async fn write_bytes(&self, path: &str, data: &[u8], append: bool) -> Option<Result<()>> {
        if append {
            return None;
        }
        let live = self.live()?;
        let boundary = format!("ah-{}", crate::types::new_id(""));
        let filename = path.rsplit('/').next().unwrap_or("file").replace('"', "%22");
        let mut body =
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n")
                .into_bytes();
        body.extend_from_slice(data);
        body.extend(format!("\r\n--{boundary}--\r\n").into_bytes());
        let url = format!("{}/files", live.envd.trim_end_matches('/'));
        let write = async {
            let resp = self
                .envd(&live, self.http.post(url).query(&[("path", path), ("username", "user")]))
                .header(reqwest::header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
                .timeout(Duration::from_secs(300))
                .body(body)
                .send()
                .await
                .map_err(|e| Error::Io(format!("e2b: {e}")))?;
            let status = resp.status().as_u16();
            if (200..300).contains(&status) {
                return Ok(());
            }
            let text = resp.text().await.unwrap_or_default();
            Err(Error::tool(format!("could not write {path}: {}", text.trim().chars().take(300).collect::<String>()), "fs_write"))
        };
        Some(write.await)
    }
}

/// `sandbox("e2b://my-template")`.
pub fn build(options: &Map<String, Value>) -> Result<Arc<dyn SandboxBackend>> {
    Ok(Arc::new(E2bSandbox::new(options)?))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use axum::Router;
    use axum::body::Bytes;
    use axum::extract::{Path, Query, State};
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::IntoResponse;
    use axum::routing::{delete, get, post};

    use super::*;
    use crate::runtime::workspace::{Workspace, on_path};
    use crate::sandboxes::base::script;
    use crate::sandboxes::{Sandbox, SandboxOptions, SandboxWorkspace, WrapOptions, run_process};

    #[derive(Default)]
    struct Seen {
        create: Vec<(String, Value)>,
        connect: Vec<(String, Value)>,
        killed: Vec<String>,
        runs: Vec<(HeaderMap, Value)>,
        files: Vec<String>,
        /// Run commands in a local shell.
        real: bool,
    }

    type Shared = Arc<Mutex<Seen>>;

    fn key_of(h: &HeaderMap) -> String {
        h.get("x-api-key").and_then(|v| v.to_str().ok()).unwrap_or("").to_string()
    }

    async fn create(State(s): State<Shared>, h: HeaderMap, body: axum::Json<Value>) -> impl IntoResponse {
        s.lock().unwrap().create.push((key_of(&h), body.0));
        axum::Json(json!({"sandboxID": "sbx-123", "templateID": "t", "clientID": "c", "envdVersion": "0.2.0", "envdAccessToken": "tok-1", "domain": null}))
    }

    async fn connect(State(s): State<Shared>, Path(id): Path<String>, h: HeaderMap, body: axum::Json<Value>) -> axum::response::Response {
        s.lock().unwrap().connect.push((format!("{id} {}", key_of(&h)), body.0));
        if id == "gone" {
            return (StatusCode::NOT_FOUND, axum::Json(json!({"code": 404, "message": "sandbox \"gone\" doesn't exist"}))).into_response();
        }
        axum::Json(json!({"sandboxID": id, "envdAccessToken": "tok-2"})).into_response()
    }

    async fn kill(State(s): State<Shared>, Path(id): Path<String>) -> StatusCode {
        s.lock().unwrap().killed.push(id);
        StatusCode::NO_CONTENT
    }

    /// envd's process service: runs what it is sent in a local shell when
    /// told to, else answers with the command as its output (and `exit N` as N).
    async fn start_process(State(s): State<Shared>, h: HeaderMap, body: Bytes) -> impl IntoResponse {
        let message = frames(&body).remove(0).1;
        let real = {
            let mut s = s.lock().unwrap();
            s.runs.push((h, message.clone()));
            s.real
        };
        let p = &message["process"];
        let command = p["args"][2].as_str().unwrap_or("").to_string();
        let env: BTreeMap<String, String> = serde_json::from_value(p["envs"].clone()).unwrap_or_default();
        let (code, out, err) = if real {
            let ran = run_process(vec!["sh".into(), "-c".into(), script(&command, p["cwd"].as_str(), &env)], None, 30.0).await.unwrap().unwrap();
            (ran.0, ran.1, ran.2)
        } else {
            (command.strip_prefix("exit ").and_then(|n| n.parse().ok()).unwrap_or(0), command.into_bytes(), Vec::new())
        };
        let enc = |b: &[u8]| base64::engine::general_purpose::STANDARD.encode(b);
        let mut stream = envelope(0, &json!({"event": {"start": {"pid": 7}}}));
        stream.extend(envelope(0, &json!({"event": {"data": {"stdout": enc(&out)}}})));
        stream.extend(envelope(0, &json!({"event": {"data": {"stderr": enc(&err)}}})));
        let end = if code == 0 { json!({"exited": true, "status": "exit status 0"}) } else { json!({"exitCode": code, "exited": true}) };
        stream.extend(envelope(0, &json!({"event": {"end": end}})));
        stream.extend(envelope(2, &json!({})));
        ([("content-type", "application/connect+json")], stream)
    }

    async fn read_file(Query(q): Query<HashMap<String, String>>) -> axum::response::Response {
        match std::fs::read(&q["path"]) {
            Ok(b) => b.into_response(),
            Err(_) => (StatusCode::NOT_FOUND, "no such file").into_response(),
        }
    }

    async fn write_file(State(s): State<Shared>, Query(q): Query<HashMap<String, String>>, h: HeaderMap, body: Bytes) -> StatusCode {
        let ct = h["content-type"].to_str().unwrap().to_string();
        let boundary = ct.split("boundary=").nth(1).unwrap();
        let start = body.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        let end = body.len() - format!("\r\n--{boundary}--\r\n").len();
        let path = std::path::PathBuf::from(&q["path"]);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &body[start..end]).unwrap();
        s.lock().unwrap().files.push(q["path"].clone());
        StatusCode::OK
    }

    async fn serve() -> (String, Shared) {
        let seen: Shared = Arc::default();
        let app = Router::new()
            .route("/sandboxes", post(create))
            .route("/sandboxes/{id}/connect", post(connect))
            .route("/sandboxes/{id}", delete(kill))
            .route("/process.Process/Start", post(start_process))
            .route("/files", get(read_file).post(write_file))
            .with_state(seen.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), seen)
    }

    fn backend(base: &str, options: Value) -> Arc<E2bSandbox> {
        let mut b = E2bSandbox::new(options.as_object().unwrap()).unwrap();
        b.api_url = base.to_string();
        b.envd_url = Some(base.to_string());
        Arc::new(b)
    }

    #[tokio::test]
    async fn e2b_is_created_with_what_was_asked_and_killed_at_the_end() {
        let (base, seen) = serve().await;
        let b =
            backend(&base, json!({"template": "my-template", "ttl": 900, "network": false, "env": {"A": "1"}, "metadata": {"job": "7"}, "api_key": "key-1"}));
        let sb = Sandbox::new(b.clone(), SandboxOptions { workdir: Some("/w".into()), ..Default::default() }).unwrap();
        let ws = SandboxWorkspace::new(sb.clone(), WrapOptions::default());
        ws.start().await.unwrap();
        assert_eq!(sb.id().await, "sbx-123");
        {
            let s = seen.lock().unwrap();
            assert_eq!(s.create[0].0, "key-1");
            assert_eq!(
                s.create[0].1,
                json!({"templateID": "my-template", "timeout": 900, "secure": true, "metadata": {"job": "7"}, "envVars": {"A": "1"}, "allow_internet_access": false})
            );
            // Commands go to envd as a Connect stream, as the user `user`, with the sandbox's token.
            let (h, message) = &s.runs[0];
            assert_eq!(h["content-type"], "application/connect+json");
            assert_eq!(h["connect-protocol-version"], "1");
            assert_eq!(h["x-access-token"], "tok-1");
            assert_eq!(h["authorization"], AS_USER);
            assert_eq!(message["process"]["cmd"], "/bin/bash");
            assert_eq!(message["process"]["args"], json!(["-l", "-c", "mkdir -p /w"]));
        }
        let mut env = BTreeMap::new();
        env.insert("B".to_string(), "2".to_string());
        sb.exec("exit 7", None, Some(&env), Some(45.0)).await.unwrap();
        {
            let s = seen.lock().unwrap();
            let (h, message) = s.runs.last().unwrap();
            assert_eq!(message["process"]["cwd"], "/w");
            assert_eq!(message["process"]["envs"], json!({"B": "2"}));
            assert_eq!(h["connect-timeout-ms"], "45000");
        }
        ws.destroy().await.unwrap();
        assert_eq!(seen.lock().unwrap().killed, ["sbx-123"]);

        // Nothing asked for, nothing passed.
        let plain = backend(&base, json!({"api_key": "key-1"}));
        plain.start(None).await.unwrap();
        assert_eq!(seen.lock().unwrap().create[1].1, json!({"templateID": "base", "timeout": 3600, "secure": true}));
    }

    #[tokio::test]
    async fn one_already_running_is_picked_up_and_left_running() {
        let (base, seen) = serve().await;
        let b = backend(&base, json!({"sandbox_id": "sbx-9", "api_key": "other"}));
        let sb = Sandbox::new(b.clone(), SandboxOptions { workdir: Some("/w".into()), ..Default::default() }).unwrap();
        sb.start().await.unwrap();
        sb.stop(false).await.unwrap();
        {
            let s = seen.lock().unwrap();
            assert_eq!(s.connect, [("sbx-9 other".to_string(), json!({"timeout": 3600}))]);
            assert!(s.killed.is_empty() && s.create.is_empty());
            assert_eq!(s.runs.last().unwrap().0["x-access-token"], "tok-2");
        }

        // Gone: a new one in its place, and the caller told which.
        let b = backend(&base, json!({"api_key": "k"}));
        let sb = Sandbox::new(b, SandboxOptions { id: Some("gone".into()), workdir: Some("/w".into()), ..Default::default() }).unwrap();
        sb.start().await.unwrap();
        let (replaced, why) = sb.replaced().await;
        assert_eq!(replaced, "gone");
        assert!(why.contains("e2b sandbox gone was not found"), "{why}");
        assert_eq!(sb.id().await, "sbx-123");
    }

    #[tokio::test]
    async fn e2b_passes_the_self_check_through_its_own_file_api() {
        if !on_path("sh") || !on_path("base64") || !on_path("stat") {
            return; // the fake envd runs commands in a local shell
        }
        let (base, seen) = serve().await;
        seen.lock().unwrap().real = true;
        let dir = std::env::temp_dir().join(format!("ah_e2b_{}", crate::types::new_id("")));
        let root = dir.to_string_lossy().replace('\\', "/");
        let b = backend(&base, json!({"api_key": "k"}));
        let sb = Sandbox::new(b, SandboxOptions { workdir: Some(format!("{root}/w")), ..Default::default() }).unwrap();
        let ws = SandboxWorkspace::new(sb, WrapOptions::default());
        let report = ws.check().await;
        assert_eq!(report["ok"], true, "{report:#}");
        assert!(!seen.lock().unwrap().files.is_empty());
        let r = ws.shell("echo out; echo err >&2; exit 2", Some(30.0)).await.unwrap();
        assert_eq!(r, json!({"returncode": 2, "stdout": "out\n", "stderr": "err\n"}));
        ws.write("log.txt", b"one\n", false).await.unwrap();
        ws.write("log.txt", b"two\n", true).await.unwrap();
        assert_eq!(ws.read("log.txt").await.unwrap(), "one\ntwo\n");
        assert!(ws.read("nope.txt").await.unwrap_err().to_string().contains("no such file"));
        ws.close().await;
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn without_its_key_it_says_which() {
        let b = E2bSandbox::new(&Map::new()).unwrap();
        if std::env::var("E2B_API_KEY").is_err() {
            let err = b.key().unwrap_err();
            assert!(matches!(err, Error::Configuration(_)) && err.to_string().contains("set E2B_API_KEY"));
        }
        assert!(E2bSandbox::new(json!({"client": {}}).as_object().unwrap()).err().unwrap().to_string().contains("SDK object"));
        assert!(
            E2bSandbox::new(json!({"image": "x"}).as_object().unwrap())
                .err()
                .unwrap()
                .to_string()
                .starts_with("e2b sandbox: got an unexpected keyword argument 'image'")
        );
        let framed = envelope(2, &json!({"a": 1}));
        assert_eq!(frames(&framed), [(2, json!({"a": 1}))]);
    }

    /// Real E2B: `AH_LIVE_E2B=1` and `E2B_API_KEY`.
    #[tokio::test]
    async fn live_e2b_passes_the_self_check() {
        if std::env::var("AH_LIVE_E2B").as_deref() != Ok("1") {
            return;
        }
        let ws = crate::sandboxes::sandbox(&json!("e2b")).unwrap();
        let report = ws.check().await;
        ws.destroy().await.unwrap();
        assert_eq!(report["ok"], true, "{report:#}");
    }
}
