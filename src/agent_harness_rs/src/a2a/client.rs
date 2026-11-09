//! Call an agent that is served over A2A — whatever it was built with.
//!
//! ```no_run
//! use agent_harness::a2a::{A2AClient, RemoteAgent, SendOptions};
//! use futures::StreamExt;
//! # async fn demo() -> Result<(), agent_harness::a2a::A2AError> {
//! let client = A2AClient::new("https://agents.example.com/pricing").token("sk-…");
//! let task = client.send("What does the gold plan cost?", &SendOptions::default()).await?;
//! println!("{} ({})", task.text, task.state);
//!
//! let mut events = client.stream("And for 40 seats?", &SendOptions::default().context_id(&task.context_id)).await?;
//! while let Some(event) = events.next().await {
//!     print!("{}", event?.text);
//! }
//!
//! // As a tool of one of your own agents:
//! let pricing = RemoteAgent::connect("https://agents.example.com/pricing").await?;
//! let tool = pricing.as_tool(Some("ask_pricing"), None);
//! # Ok(()) }
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::stream::BoxStream;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use super::protocol::{self as wire, A2AError};
use crate::agent::RunOptions;
use crate::harness::Harness;
use crate::tools::Tool;
use crate::types::{Artifact, ContentBlock, Message, RunResult, StopReason};

const RETRY_STATUS: [u16; 4] = [429, 502, 503, 504];

/// A task on somebody else's agent, as it stood when it was last heard of.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RemoteTask {
    pub id: String,
    pub context_id: String,
    pub state: String,
    /// What it came to, as text.
    pub text: String,
    pub artifacts: Vec<Value>,
    /// The task exactly as the server sent it.
    pub raw: Value,
}

fn s(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

impl RemoteTask {
    pub fn of(result: &Value) -> Self {
        if result.get("kind").and_then(Value::as_str) == Some("message") || result.get("parts").is_some() {
            // An agent may answer a simple request with a message and no task.
            return Self {
                id: s(result, "taskId"),
                context_id: s(result, "contextId"),
                state: "completed".into(),
                text: wire::text_of(result.get("parts")),
                artifacts: vec![],
                raw: result.clone(),
            };
        }
        let state = result.get("status").and_then(|st| st.get("state")).and_then(Value::as_str).unwrap_or("unknown").to_string();
        Self {
            id: s(result, "id"),
            context_id: s(result, "contextId"),
            state,
            text: wire::answer_of(result),
            artifacts: result.get("artifacts").and_then(Value::as_array).cloned().unwrap_or_default(),
            raw: result.clone(),
        }
    }

    pub fn done(&self) -> bool {
        wire::is_terminal(&self.state)
    }

    pub fn ok(&self) -> bool {
        self.state == "completed"
    }

    /// Why it did not complete, in the server's words.
    pub fn error(&self) -> Option<String> {
        if self.ok() || !self.done() {
            return None;
        }
        let said = wire::text_of(self.raw.get("status").and_then(|st| st.get("message")).and_then(|m| m.get("parts")));
        Some(if said.is_empty() { format!("the task was {}", self.state) } else { said })
    }
}

/// One event of a streamed task. `text` is what was just written.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct A2AEvent {
    /// `task`, `message`, `status-update` or `artifact-update`.
    pub kind: String,
    pub text: String,
    pub state: String,
    /// The last event of the task (Python's `final`).
    pub is_final: bool,
    pub task_id: String,
    pub context_id: String,
    pub raw: Value,
}

/// What to send: some text, or a whole A2A message object of your own.
#[derive(Debug, Clone)]
pub enum Outgoing {
    Text(String),
    Message(Map<String, Value>),
}

impl From<&str> for Outgoing {
    fn from(t: &str) -> Self {
        Self::Text(t.into())
    }
}

impl From<String> for Outgoing {
    fn from(t: String) -> Self {
        Self::Text(t)
    }
}

impl From<&String> for Outgoing {
    fn from(t: &String) -> Self {
        Self::Text(t.clone())
    }
}

impl From<Value> for Outgoing {
    fn from(v: Value) -> Self {
        match v {
            Value::Object(m) => Self::Message(m),
            Value::String(s) => Self::Text(s),
            other => Self::Text(other.to_string()),
        }
    }
}

/// How to send a message.
#[derive(Debug, Clone)]
pub struct SendOptions {
    /// Carry a conversation on: the context id a task came back with.
    pub context_id: Option<String>,
    pub task_id: Option<String>,
    pub attachments: Vec<ContentBlock>,
    /// A resent message with the same id is the same task on the server.
    pub message_id: Option<String>,
    pub metadata: Option<Map<String, Value>>,
    /// Wait for the task to finish (the default), or come back once accepted.
    pub blocking: bool,
    pub history_length: Option<i64>,
    /// A webhook config (`{"url": ..., "token": ...}`) or just its URL.
    pub push: Option<Value>,
}

impl Default for SendOptions {
    fn default() -> Self {
        Self { context_id: None, task_id: None, attachments: vec![], message_id: None, metadata: None, blocking: true, history_length: None, push: None }
    }
}

impl SendOptions {
    pub fn context_id(mut self, id: &str) -> Self {
        self.context_id = Some(id.into()).filter(|s: &String| !s.is_empty());
        self
    }

    pub fn task_id(mut self, id: &str) -> Self {
        self.task_id = Some(id.into());
        self
    }

    pub fn message_id(mut self, id: &str) -> Self {
        self.message_id = Some(id.into());
        self
    }

    pub fn attachments(mut self, attachments: Vec<ContentBlock>) -> Self {
        self.attachments = attachments;
        self
    }

    /// Come back as soon as the task is accepted.
    pub fn no_wait(mut self) -> Self {
        self.blocking = false;
        self
    }

    pub fn history_length(mut self, n: i64) -> Self {
        self.history_length = Some(n);
        self
    }

    pub fn push(mut self, config: Value) -> Self {
        self.push = Some(config);
        self
    }
}

/// A client for one A2A agent.
///
/// `url` is the agent's base URL — where its card is — or its endpoint.
/// `timeout` is how long one request may take; a blocking `send` waits for the
/// whole task, so this is the longest a task may run. `retries` is how many
/// times a request that could not be delivered, or was told to come back
/// later (429/502/503/504), is sent again: a resent message carries the same
/// message id, so the server runs it once.
pub struct A2AClient {
    url: String,
    headers: IndexMap<String, String>,
    timeout: f64,
    retries: u32,
    http: reqwest::Client,
    own_http: bool,
    card: Mutex<Option<Value>>,
    endpoint: Mutex<Option<String>>,
    ids: AtomicU64,
}

impl std::fmt::Debug for A2AClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("A2AClient").field("url", &self.url).finish()
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn unreachable_err(at: &str, e: impl std::fmt::Display) -> A2AError {
    A2AError::http(format!("{at} could not be reached: {e}"), wire::INTERNAL_ERROR, 503)
}

impl A2AClient {
    pub fn new(url: &str) -> Self {
        let mut url = url.trim_end_matches('/').to_string();
        for suffix in [wire::CARD_PATH, wire::LEGACY_CARD_PATH] {
            if let Some(stripped) = url.strip_suffix(suffix) {
                url = stripped.to_string();
            }
        }
        Self {
            url,
            headers: IndexMap::new(),
            timeout: 600.0,
            retries: 2,
            http: crate::llm_providers::base::default_client(600.0, 10.0),
            own_http: true,
            card: Mutex::new(None),
            endpoint: Mutex::new(None),
            ids: AtomicU64::new(0),
        }
    }

    /// A bearer token, if the agent asks for one.
    pub fn token(self, token: &str) -> Self {
        if token.is_empty() { self } else { self.header("Authorization", &format!("Bearer {token}")) }
    }

    /// A header every request carries.
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    pub fn timeout(mut self, seconds: f64) -> Self {
        self.timeout = seconds;
        if self.own_http {
            self.http = crate::llm_providers::base::default_client(seconds, 10.0);
        }
        self
    }

    pub fn retries(mut self, n: u32) -> Self {
        self.retries = n;
        self
    }

    /// A `reqwest` client of your own.
    pub fn http(mut self, client: reqwest::Client) -> Self {
        self.http = client;
        self.own_http = false;
        self
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn headers(&self) -> &IndexMap<String, String> {
        &self.headers
    }

    fn with_headers(&self, mut request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        for (k, v) in &self.headers {
            request = request.header(k, v);
        }
        request
    }

    /// The agent's card: who it is, what it can do, where to send requests.
    pub async fn card(&self, refresh: bool) -> Result<Value, A2AError> {
        if !refresh && let Some(c) = lock(&self.card).clone() {
            return Ok(c);
        }
        let mut problem = String::new();
        for path in [wire::CARD_PATH, wire::LEGACY_CARD_PATH] {
            let response = self.with_headers(self.http.get(format!("{}{path}", self.url))).send().await.map_err(|e| unreachable_err(&self.url, e))?;
            let status = response.status().as_u16();
            if status == 200 {
                match response.json::<Value>().await {
                    Ok(card) => {
                        *lock(&self.card) = Some(card.clone());
                        return Ok(card);
                    }
                    Err(_) => {
                        problem = "its agent card is not JSON".into();
                        continue;
                    }
                }
            }
            problem = format!("HTTP {status}");
        }
        Err(A2AError::http(format!("no A2A agent card at {} ({problem})", self.url), wire::INTERNAL_ERROR, 404))
    }

    /// The endpoint requests go to: what the card says, else the URL given.
    async fn endpoint(&self) -> String {
        if let Some(e) = lock(&self.endpoint).clone() {
            return e;
        }
        let found = match self.card(false).await {
            Ok(card) => card.get("url").and_then(Value::as_str).filter(|u| !u.is_empty()).map(str::to_string).unwrap_or_else(|| self.url.clone()),
            Err(_) => self.url.clone(), // no card: treat the URL as the endpoint
        };
        *lock(&self.endpoint) = Some(found.clone());
        found
    }

    fn request(&self, method: &str, params: Value) -> Value {
        let id = self.ids.fetch_add(1, Ordering::SeqCst) + 1;
        json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
    }

    /// One JSON-RPC call. Its result; an `A2AError` for an error.
    pub async fn call(&self, method: &str, params: Value) -> Result<Value, A2AError> {
        let payload = self.request(method, params);
        let endpoint = self.endpoint().await;
        let mut attempt = 0u32;
        loop {
            let mut retry_after: Option<f64> = None;
            let sent = self.with_headers(self.http.post(&endpoint)).header("content-type", "application/json").body(payload.to_string()).send().await;
            match sent {
                Err(e) => {
                    if attempt >= self.retries {
                        return Err(unreachable_err(&endpoint, e));
                    }
                }
                Ok(response) => {
                    let status = response.status().as_u16();
                    if !RETRY_STATUS.contains(&status) || attempt >= self.retries {
                        let body = response.text().await.unwrap_or_default();
                        return result_of(status, &body);
                    }
                    retry_after = response.headers().get("retry-after").and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse::<f64>().ok());
                }
            }
            let wait = retry_after.unwrap_or(0.5 * f64::from(2u32.saturating_pow(attempt))).min(30.0);
            tokio::time::sleep(Duration::from_secs_f64(wait.max(0.0))).await;
            attempt += 1;
        }
    }

    fn send_params(&self, message: Outgoing, opts: &SendOptions, blocking: Option<bool>, history_length: Option<i64>, push: Option<&Value>) -> Value {
        let mut message = match message {
            Outgoing::Message(m) => {
                let mut out = Map::new();
                out.insert("kind".into(), json!("message"));
                out.insert("role".into(), json!("user"));
                out.extend(m);
                if !out.contains_key("messageId") {
                    out.insert("messageId".into(), json!(opts.message_id.clone().unwrap_or_else(wire::new_uuid)));
                }
                Value::Object(out)
            }
            Outgoing::Text(t) => wire::message("user", wire::parts_of(&t, &opts.attachments), "", "", opts.message_id.as_deref().unwrap_or(""), None),
        };
        if let Some(c) = opts.context_id.as_ref().filter(|c| !c.is_empty()) {
            message["contextId"] = json!(c);
        }
        if let Some(t) = opts.task_id.as_ref().filter(|t| !t.is_empty()) {
            message["taskId"] = json!(t);
        }
        let mut params = json!({"message": message});
        let mut configuration = Map::new();
        if let Some(b) = blocking {
            configuration.insert("blocking".into(), json!(b));
        }
        if let Some(n) = history_length {
            configuration.insert("historyLength".into(), json!(n));
        }
        if let Some(p) = push.filter(|p| !p.is_null()) {
            let config = if let Value::String(url) = p { json!({"url": url}) } else { p.clone() };
            configuration.insert("pushNotificationConfig".into(), config);
        }
        if !configuration.is_empty() {
            params["configuration"] = Value::Object(configuration);
        }
        if let Some(m) = opts.metadata.as_ref().filter(|m| !m.is_empty()) {
            params["metadata"] = Value::Object(m.clone());
        }
        params
    }

    /// Send a message. Blocking, the task comes back finished; otherwise as
    /// soon as it is accepted — ask after it with `get` or `wait`.
    pub async fn send(&self, message: impl Into<Outgoing>, opts: &SendOptions) -> Result<RemoteTask, A2AError> {
        let blocking = if opts.blocking { None } else { Some(false) };
        let params = self.send_params(message.into(), opts, blocking, opts.history_length, opts.push.as_ref());
        Ok(RemoteTask::of(&self.call("message/send", params).await?))
    }

    /// Send a message and hear the task as it happens.
    pub async fn stream(&self, message: impl Into<Outgoing>, opts: &SendOptions) -> Result<BoxStream<'static, Result<A2AEvent, A2AError>>, A2AError> {
        let params = self.send_params(message.into(), &SendOptions { task_id: None, ..opts.clone() }, None, None, None);
        self.events("message/stream", params).await
    }

    /// Pick the events of a running task back up after a dropped connection.
    pub async fn resubscribe(&self, task_id: &str) -> Result<BoxStream<'static, Result<A2AEvent, A2AError>>, A2AError> {
        self.events("tasks/resubscribe", json!({"id": task_id})).await
    }

    async fn events(&self, method: &str, params: Value) -> Result<BoxStream<'static, Result<A2AEvent, A2AError>>, A2AError> {
        let endpoint = self.endpoint().await;
        let response = self
            .with_headers(self.http.post(&endpoint))
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .body(self.request(method, params).to_string())
            .send()
            .await
            .map_err(|e| unreachable_err(&endpoint, e))?;
        let sse = response.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|c| c.contains("text/event-stream"));
        if !sse {
            // Refused before any stream began: the body is the error — or a
            // server that answered without streaming.
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            let one = result_of(status, &body).and_then(|r| event_of(&r));
            return Ok(futures::stream::once(async move { one }).boxed());
        }
        let mut body = response.bytes_stream();
        Ok(Box::pin(async_stream::stream! {
            let mut buffer: Vec<u8> = Vec::new();
            let mut data: Vec<String> = Vec::new();
            let emit = |data: &mut Vec<String>| -> Result<A2AEvent, A2AError> {
                let joined = data.join("\n");
                data.clear();
                let parsed: Value = serde_json::from_str(&joined)
                    .map_err(|_| A2AError::new("the agent sent an event that is not JSON", wire::INVALID_AGENT_RESPONSE))?;
                event_of(&unwrap(&parsed)?)
            };
            while let Some(chunk) = body.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        yield Err(A2AError::http(format!("the stream was cut off: {e}"), wire::INTERNAL_ERROR, 503));
                        return;
                    }
                };
                buffer.extend_from_slice(&chunk);
                while let Some(at) = buffer.iter().position(|b| *b == b'\n') {
                    let raw: Vec<u8> = buffer.drain(..=at).collect();
                    let line = String::from_utf8_lossy(&raw);
                    let line = line.trim_end_matches(['\n', '\r']);
                    if let Some(rest) = line.strip_prefix("data:") {
                        data.push(rest.trim_start().to_string());
                    } else if line.is_empty() && !data.is_empty() {
                        yield emit(&mut data);
                    }
                }
            }
            if !data.is_empty() {
                yield emit(&mut data);
            }
        }))
    }

    pub async fn get(&self, task_id: &str, history_length: Option<i64>) -> Result<RemoteTask, A2AError> {
        let mut params = json!({"id": task_id});
        if let Some(n) = history_length {
            params["historyLength"] = json!(n);
        }
        Ok(RemoteTask::of(&self.call("tasks/get", params).await?))
    }

    pub async fn cancel(&self, task_id: &str) -> Result<RemoteTask, A2AError> {
        Ok(RemoteTask::of(&self.call("tasks/cancel", json!({"id": task_id})).await?))
    }

    /// Ask after a task until it is over (or needs input), at `interval`
    /// seconds and growing. An `A2AError` if it is not by `timeout`.
    pub async fn wait(&self, task_id: &str, timeout: f64, interval: f64) -> Result<RemoteTask, A2AError> {
        let deadline = Instant::now() + Duration::from_secs_f64(timeout);
        let mut interval = interval;
        loop {
            let task = self.get(task_id, None).await?;
            if task.done() || task.state == "input-required" || task.state == "auth-required" {
                return Ok(task);
            }
            if Instant::now() >= deadline {
                let secs = if timeout.fract() == 0.0 { format!("{}", timeout as i64) } else { format!("{timeout}") };
                return Err(A2AError::http(format!("task {task_id} was still {} after {secs}s", task.state), wire::INTERNAL_ERROR, 504));
            }
            tokio::time::sleep(Duration::from_secs_f64(interval)).await;
            interval = (interval * 1.5).min(5.0);
        }
    }

    /// Ask to be told at `url` when a task is over.
    pub async fn set_push(&self, task_id: &str, url: &str, token: Option<&str>) -> Result<Value, A2AError> {
        let mut config = json!({"url": url});
        if let Some(t) = token.filter(|t| !t.is_empty()) {
            config["token"] = json!(t);
        }
        self.call("tasks/pushNotificationConfig/set", json!({"taskId": task_id, "pushNotificationConfig": config})).await
    }
}

fn unwrap(response: &Value) -> Result<Value, A2AError> {
    let Value::Object(r) = response else { return Err(A2AError::new("the agent's reply is not a JSON-RPC response", wire::INVALID_AGENT_RESPONSE)) };
    if let Some(error) = r.get("error").filter(|e| !e.is_null() && e.as_object().is_none_or(|o| !o.is_empty())) {
        let message = error.get("message").and_then(Value::as_str).filter(|m| !m.is_empty()).unwrap_or("the agent returned an error");
        let code = error.get("code").and_then(Value::as_i64).filter(|c| *c != 0).unwrap_or(wire::INTERNAL_ERROR);
        let mut e = A2AError::new(message, code);
        if let Some(d) = error.get("data").filter(|d| !d.is_null()) {
            e.data = Some(Box::new(d.clone()));
        }
        return Err(e);
    }
    Ok(r.get("result").cloned().unwrap_or(Value::Null))
}

fn result_of(status: u16, body: &str) -> Result<Value, A2AError> {
    let Ok(parsed) = serde_json::from_str::<Value>(body) else {
        let shown: String = body.chars().take(200).collect();
        return Err(A2AError::http(
            format!("the agent answered HTTP {status} with something that is not JSON: {shown:?}"),
            wire::INVALID_AGENT_RESPONSE,
            status,
        ));
    };
    unwrap(&parsed).map_err(|mut e| {
        e.status = status;
        e
    })
}

fn event_of(result: &Value) -> Result<A2AEvent, A2AError> {
    if !result.is_object() {
        return Err(A2AError::new("the agent sent an event that is not an object", wire::INVALID_AGENT_RESPONSE));
    }
    let kind = match result.get("kind").and_then(Value::as_str).filter(|k| !k.is_empty()) {
        Some(k) => k.to_string(),
        None if result.get("status").is_some() && result.get("id").is_some() => "task".into(),
        None => "message".into(),
    };
    let base = A2AEvent { kind: kind.clone(), task_id: s(result, "taskId"), context_id: s(result, "contextId"), raw: result.clone(), ..Default::default() };
    Ok(match kind.as_str() {
        "artifact-update" => A2AEvent { text: wire::text_of(result.get("artifact").and_then(|a| a.get("parts"))), ..base },
        "status-update" => A2AEvent {
            state: result.get("status").and_then(|st| st.get("state")).and_then(Value::as_str).unwrap_or("").to_string(),
            is_final: result.get("final").and_then(Value::as_bool).unwrap_or(false),
            ..base
        },
        "message" => A2AEvent { text: wire::text_of(result.get("parts")), state: "completed".into(), is_final: true, ..base },
        _ => {
            let task = RemoteTask::of(result);
            A2AEvent { kind: "task".into(), state: task.state.clone(), is_final: task.done(), task_id: task.id, context_id: task.context_id, ..base }
        }
    })
}

/// A card's display name as something a model can name in a tool call.
fn identifier(name: &str) -> Result<String, A2AError> {
    let cleaned: String = name.trim().chars().map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
    let cleaned = cleaned.split('_').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("_");
    if cleaned.is_empty() {
        return Err(A2AError::new(format!("the agent card's name '{name}' cannot be used"), wire::INVALID_AGENT_RESPONSE));
    }
    Ok(cleaned)
}

/// Builds a [`RemoteAgent`].
pub struct RemoteAgentBuilder {
    url: String,
    name: String,
    description: String,
    token: Option<String>,
    headers: Vec<(String, String)>,
    timeout: f64,
    retries: Option<u32>,
    client: Option<A2AClient>,
    harness: Option<Harness>,
    identity: Option<Value>,
}

impl RemoteAgentBuilder {
    /// What your agents know it as. `connect` reads it from the card otherwise.
    pub fn name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }

    pub fn description(mut self, description: &str) -> Self {
        self.description = description.into();
        self
    }

    pub fn token(mut self, token: &str) -> Self {
        self.token = Some(token.into());
        self
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// The longest one task may take, in seconds.
    pub fn timeout(mut self, seconds: f64) -> Self {
        self.timeout = seconds;
        self
    }

    pub fn retries(mut self, n: u32) -> Self {
        self.retries = Some(n);
        self
    }

    /// A client of your own (its URL wins).
    pub fn client(mut self, client: A2AClient) -> Self {
        self.client = Some(client);
        self
    }

    /// A harness to record the calls on — and, if it is governed, to register
    /// this agent with.
    pub fn harness(mut self, harness: Harness) -> Self {
        self.harness = Some(harness);
        self
    }

    /// Its governance identity, as for an `Agent`.
    pub fn identity(mut self, identity: Value) -> Self {
        self.identity = Some(identity);
        self
    }

    fn client_of(&mut self) -> A2AClient {
        let mut client = match self.client.take() {
            Some(c) => c,
            None => A2AClient::new(&self.url).timeout(self.timeout),
        };
        if let Some(t) = &self.token {
            client = client.token(t);
        }
        for (k, v) in &self.headers {
            client = client.header(k, v);
        }
        if let Some(n) = self.retries {
            client = client.retries(n);
        }
        client
    }

    fn finish(self, client: A2AClient, version: String) -> Result<RemoteAgent, A2AError> {
        let url = client.url().to_string();
        let guessed = url.trim_end_matches('/').rsplit('/').next().unwrap_or("").split(':').next().unwrap_or("").to_string();
        let name = if !self.name.is_empty() {
            self.name
        } else if !guessed.is_empty() {
            guessed
        } else {
            "remote".into()
        };
        let description = if self.description.is_empty() { format!("A remote agent at {url}.") } else { self.description };
        let agent = RemoteAgent {
            inner: Arc::new(RemoteInner {
                client,
                url,
                name,
                description,
                harness: self.harness,
                identity: self.identity,
                timeout: self.timeout,
                version,
                context_id: Mutex::new(None),
            }),
        };
        agent.register()?;
        Ok(agent)
    }

    /// The remote agent, without reading its card: the name is the one given,
    /// or guessed from the URL.
    pub fn build(mut self) -> Result<RemoteAgent, A2AError> {
        let client = self.client_of();
        self.finish(client, String::new())
    }

    /// Read the agent's card, and call it by its own name and description
    /// (unless they were given).
    pub async fn connect(mut self) -> Result<RemoteAgent, A2AError> {
        let client = self.client_of();
        let card = client.card(false).await?;
        if self.name.is_empty()
            && let Some(n) = card.get("name").and_then(Value::as_str).filter(|n| !n.is_empty())
        {
            self.name = identifier(n)?;
        }
        if self.description.is_empty() {
            let skills: Vec<String> = card
                .get("skills")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|sk| sk.get("description").and_then(Value::as_str).filter(|d| !d.is_empty()).map(str::to_string).unwrap_or_else(|| s(sk, "name")))
                .collect();
            let skills = skills.join("; ");
            self.description = card.get("description").and_then(Value::as_str).filter(|d| !d.is_empty()).map(str::to_string).unwrap_or(skills);
        }
        let version = match card.get("version") {
            Some(Value::String(v)) => v.clone(),
            Some(Value::Null) | None => String::new(),
            Some(other) => other.to_string(),
        };
        self.finish(client, version)
    }
}

struct RemoteInner {
    client: A2AClient,
    url: String,
    name: String,
    description: String,
    harness: Option<Harness>,
    identity: Option<Value>,
    timeout: f64,
    version: String,
    context_id: Mutex<Option<String>>,
}

/// An agent somewhere else, used as if it were one of yours.
///
/// It runs a task and hands back a [`RunResult`], as an agent's own run does.
/// Give it to an agent as a tool ([`Self::as_tool`]), or several at once as a
/// delegation tool ([`Self::delegate_tool`]).
///
/// Run directly, it keeps the conversation going between runs
/// ([`Self::new_session`] starts another). As a tool it starts clean each time.
/// Cheap to clone; clones share the conversation.
#[derive(Clone)]
pub struct RemoteAgent {
    inner: Arc<RemoteInner>,
}

impl std::fmt::Debug for RemoteAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<RemoteAgent {} {}>", self.inner.name, self.inner.url)
    }
}

impl RemoteAgent {
    pub fn builder(url: &str) -> RemoteAgentBuilder {
        RemoteAgentBuilder {
            url: url.into(),
            name: String::new(),
            description: String::new(),
            token: None,
            headers: vec![],
            timeout: 600.0,
            retries: None,
            client: None,
            harness: None,
            identity: None,
        }
    }

    /// Read the agent's card at `url`, and call it by its own name.
    pub async fn connect(url: &str) -> Result<RemoteAgent, A2AError> {
        Self::builder(url).connect().await
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }

    pub fn description(&self) -> &str {
        &self.inner.description
    }

    pub fn url(&self) -> &str {
        &self.inner.url
    }

    /// The version its card gave, if it was read.
    pub fn version(&self) -> &str {
        &self.inner.version
    }

    /// What the harness reads off any agent: always `a2a`.
    pub fn model(&self) -> &str {
        "a2a"
    }

    pub fn client(&self) -> &A2AClient {
        &self.inner.client
    }

    /// On a governed harness, an agent has to be known before anything is
    /// delegated to it — a remote one as much as a local one.
    fn register(&self) -> Result<(), A2AError> {
        if let Some(gov) = self.inner.harness.as_ref().and_then(|h| h.governance.as_ref()) {
            let info = crate::governance::AgentInfo {
                name: self.inner.name.clone(),
                description: self.inner.description.clone(),
                model: "a2a".into(),
                tools: vec![],
                identity: self.inner.identity.clone(),
            };
            gov.register_agent(&info)?;
        }
        Ok(())
    }

    /// Start a fresh conversation: the next run does not continue the last.
    pub fn new_session(&self) {
        *lock(&self.inner.context_id) = None;
    }

    /// Send the task and wait for it, carrying the conversation on.
    pub async fn run(&self, task: impl Into<Message>) -> RunResult {
        self.run_with(task, RunOptions::default()).await
    }

    /// Send the task and wait for it. Never fails: like an agent's own run,
    /// a failure is `result.error` (`"A2AError: ..."`). `options.messages`
    /// set (even empty) marks a one-off — a sub-agent's or a tool's task — that
    /// neither continues nor keeps the conversation; `options.attachments` go
    /// with the task. Other options do not apply to a remote agent.
    pub async fn run_with(&self, task: impl Into<Message>, options: RunOptions) -> RunResult {
        let task: Message = task.into();
        let text = task.text();
        let mut files: Vec<ContentBlock> = task.media().into_iter().map(|(k, m)| ContentBlock::media(k, m.clone())).collect();
        files.extend(options.attachments);
        let one_off = options.messages.is_some();
        let inner = &self.inner;
        let mut result = RunResult { agent: inner.name.clone(), ..Default::default() };
        let context = if one_off { None } else { lock(&inner.context_id).clone() };
        let opts = SendOptions { context_id: context, attachments: files, ..Default::default() };
        let sent = async {
            let remote = inner.client.send(text.as_str(), &opts).await?;
            if !remote.done() && !remote.id.is_empty() {
                return inner.client.wait(&remote.id, inner.timeout, 0.5).await;
            }
            Ok(remote)
        }
        .await;
        let remote = match sent {
            Ok(r) => r,
            Err(e) => {
                result.error = Some(format!("A2AError: {e}"));
                result.stop_reason = StopReason::Error;
                self.record("error", &e.message);
                return result;
            }
        };
        if !one_off && !remote.context_id.is_empty() {
            *lock(&inner.context_id) = Some(remote.context_id.clone());
        }
        result.output = remote.text.clone();
        result.session_id = remote.context_id.clone();
        if !remote.id.is_empty() {
            result.run_id = remote.id.clone();
        }
        result.steps = 1;
        if !remote.ok() {
            result.error = Some(format!("A2AError: {}", remote.error().unwrap_or_else(|| remote.state.clone())));
            result.stop_reason = if remote.state == "canceled" { StopReason::Stopped } else { StopReason::Error };
        }
        for artifact in &remote.artifacts {
            let name = s(artifact, "name");
            if !name.is_empty() && name != wire::ANSWER {
                let mut a = Artifact::new(name, wire::text_of(artifact.get("parts")));
                a.produced_by = inner.name.clone();
                result.artifacts.push(a);
            }
        }
        result.messages = vec![Message::user(text), Message::assistant(remote.text.clone())];
        self.record(&remote.state, &remote.id);
        result
    }

    fn record(&self, decision: &str, target: &str) {
        if let Some(h) = &self.inner.harness {
            let mut detail = Map::new();
            detail.insert("url".into(), json!(self.inner.url));
            h.audit.record(&self.inner.name, "a2a_call", target, decision, "", detail);
        }
    }

    /// Expose the remote agent as a tool another agent can call. Each call is a
    /// one-off task; a failure comes back as `[failed] <why>`.
    pub fn as_tool(&self, name: Option<&str>, description: Option<&str>) -> Tool {
        let agent = self.clone();
        let schema = json!({"type": "object", "properties": {"task": {"type": "string", "description": "The task, stated in full."}}, "required": ["task"]});
        Tool::raw(name.unwrap_or(&self.inner.name), description.unwrap_or(&self.inner.description), schema, move |args, _ctx| {
            let agent = agent.clone();
            async move {
                let task = args.get("task").and_then(Value::as_str).unwrap_or("").to_string();
                let result = agent.run_with(task, RunOptions { messages: Some(vec![]), ..Default::default() }).await;
                Ok(json!(match result.error {
                    Some(e) => format!("[failed] {e}"),
                    None => result.output,
                }))
            }
        })
        .tags(["agent", "a2a"])
    }

    /// One tool for handing tasks to any of several remote agents, with the
    /// arguments of the built-in `delegate` tool (`agent_name`, `task`,
    /// `context`). Name it `delegate` on an agent with no local sub-agents, or
    /// anything else (say `delegate_remote`) beside them.
    pub fn delegate_tool(name: &str, agents: &[RemoteAgent]) -> Tool {
        let mut names: Vec<String> = agents.iter().map(|a| a.name().to_string()).collect();
        names.sort();
        let catalogue: Vec<String> = agents.iter().map(|a| format!("- {}: {}", a.name(), a.description())).collect();
        let schema = json!({
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "enum": names, "description": catalogue.join("\n")},
                "task": {"type": "string", "description": "The complete task, stated on its own."},
                "context": {"type": "string", "description": "Facts the agent needs and cannot look up itself."},
            },
            "required": ["agent_name", "task"],
        });
        let agents: Vec<RemoteAgent> = agents.to_vec();
        Tool::raw(
            name,
            "Hand one self-contained task to a remote agent and get its result back. Remote agents start with no memory of this conversation, so put everything they need in `task` and `context`.",
            schema,
            move |args, _ctx| {
                let agents = agents.clone();
                async move {
                    let arg = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                    let wanted = arg("agent_name");
                    let Some(agent) = agents.iter().find(|a| a.name() == wanted) else {
                        let mut known: Vec<&str> = agents.iter().map(|a| a.name()).collect();
                        known.sort();
                        return Ok(json!(format!("No remote agent named '{wanted}'. Available: {}.", known.join(", "))));
                    };
                    let (task, context) = (arg("task"), arg("context"));
                    let brief = if context.is_empty() { task } else { format!("{task}\n\n## Context you were given\n{context}") };
                    let result = agent.run_with(brief, RunOptions { messages: Some(vec![]), ..Default::default() }).await;
                    Ok(json!(match result.error {
                        Some(e) => format!("[failed] {e}"),
                        None => result.output,
                    }))
                }
            },
        )
        .tags(["agent", "a2a", "delegation"])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks_and_events_are_read() {
        let msg = json!({"kind": "message", "role": "agent", "contextId": "c1", "parts": [{"kind": "text", "text": "echo"}]});
        let t = RemoteTask::of(&msg);
        assert!(t.ok() && t.done() && t.text == "echo" && t.context_id == "c1");
        let failed = json!({"id": "t", "contextId": "c", "status": {"state": "failed", "message": {"parts": [{"kind": "text", "text": "Boom."}]}}});
        let t = RemoteTask::of(&failed);
        assert_eq!(t.error().as_deref(), Some("Boom."));
        assert_eq!(RemoteTask::of(&json!({"id": "t", "status": {"state": "canceled"}})).error().as_deref(), Some("the task was canceled"));
        assert_eq!(RemoteTask::of(&json!({"id": "t", "status": {"state": "working"}})).error(), None);

        let e = event_of(&json!({"kind": "status-update", "taskId": "t", "status": {"state": "working"}, "final": false})).unwrap();
        assert_eq!((e.kind.as_str(), e.state.as_str(), e.is_final), ("status-update", "working", false));
        let e = event_of(&json!({"kind": "artifact-update", "artifact": {"parts": [{"kind": "text", "text": "Hi"}]}})).unwrap();
        assert_eq!(e.text, "Hi");
        let e = event_of(&failed).unwrap();
        assert_eq!((e.kind.as_str(), e.is_final), ("task", true));
        assert!(event_of(&json!(3)).is_err());
    }

    #[test]
    fn replies_are_unwrapped() {
        assert_eq!(result_of(200, r#"{"jsonrpc":"2.0","id":1,"result":{"x":1}}"#).unwrap(), json!({"x": 1}));
        let e = result_of(401, r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"not authorised"}}"#).unwrap_err();
        assert_eq!((e.status, e.code, e.message.as_str()), (401, -32600, "not authorised"));
        let e = result_of(502, "bad gateway").unwrap_err();
        assert_eq!((e.status, e.code), (502, wire::INVALID_AGENT_RESPONSE));
        assert!(e.message.contains("HTTP 502"));
        assert_eq!(identifier("Echo Agent!").unwrap(), "Echo_Agent");
        assert!(identifier("!!").is_err());
    }

    #[test]
    fn urls_and_params() {
        let c = A2AClient::new("https://h.test/agent/.well-known/agent-card.json").token("sk-1");
        assert_eq!(c.url(), "https://h.test/agent");
        assert_eq!(c.headers()["Authorization"], "Bearer sk-1");
        let opts = SendOptions::default().context_id("c1").message_id("m1").no_wait().push(json!("https://hook.test"));
        let p = c.send_params(Outgoing::from("hi"), &opts, Some(false), Some(2), opts.push.as_ref());
        assert_eq!(p["message"]["messageId"], "m1");
        assert_eq!(p["message"]["contextId"], "c1");
        assert_eq!(p["configuration"], json!({"blocking": false, "historyLength": 2, "pushNotificationConfig": {"url": "https://hook.test"}}));
        let r = RemoteAgent::builder("http://127.0.0.1:9999/").build().unwrap();
        assert_eq!(r.name(), "127.0.0.1");
        assert_eq!(r.description(), "A remote agent at http://127.0.0.1:9999.");
    }
}
