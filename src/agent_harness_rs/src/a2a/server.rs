//! Serve an agent over A2A, so any other framework's agent can call it.
//!
//! ```no_run
//! use agent_harness::a2a::{A2AServer, Principal};
//! # async fn demo(agent: agent_harness::Agent) -> agent_harness::Result<()> {
//! let server = A2AServer::builder(agent)
//!     .token_for("sk-live-1", Principal::user("ada").tenant("acme"))
//!     .build()?;
//! server.serve("127.0.0.1:8000").await?;   // or mount `server.router()` in your own axum app
//! # Ok(()) }
//! ```
//!
//! It speaks the JSON-RPC binding of A2A 0.3: the agent card at
//! `/.well-known/agent-card.json`, `message/send`, `message/stream` over SSE,
//! `tasks/get`, `tasks/cancel`, `tasks/resubscribe`, and push notifications.
//!
//! Built to run as many replicas behind a load balancer:
//!
//! * a replica keeps nothing another one needs. Tasks and conversations live in
//!   the harness's session store — name a database there and every replica
//!   shares them (`Harness::on("postgresql://…")`);
//! * a message sent twice (a client retrying) is one task, not two;
//! * a replica that is full says so (`429`, `Retry-After`) rather than queueing
//!   without end;
//! * a task can be asked about, cancelled or re-subscribed to on any replica,
//!   whichever one is running it; and one whose worker died is reported failed
//!   rather than working for ever.
//!
//! One agent is shared by every conversation it serves: each conversation is a
//! session of its own (`a2a_<agent>_<contextId>`), stamped with the caller.
//! For an agent instance per caller and conversation, serve a factory
//! ([`Served::factory`]) — it is called once per conversation, with the caller.

use std::collections::HashMap;
use std::convert::Infallible;
use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use futures::stream::BoxStream;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tokio::sync::{Semaphore, mpsc, watch};

use super::protocol::{self as wire, A2AError, is_terminal};
use super::store::{SessionTaskStore, TaskRecord, TaskStore};
use crate::agent::{Agent, RunOptions, SessionRef};
use crate::errors::{Error, Result};
use crate::harness::Harness;
use crate::runtime::Session;
use crate::types::{ContentBlock, RunResult, StopReason, StreamEventType};

const INPUT_MODES: [&str; 5] = ["text/plain", "application/json", "application/pdf", "image/png", "image/jpeg"];

/// Who a request is from. Their tasks and conversations are theirs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Principal {
    pub user_id: Option<String>,
    pub tenant_id: Option<String>,
    /// Anything else the caller's credentials said about them.
    pub claims: Map<String, Value>,
}

impl Principal {
    /// A caller known by a user id.
    pub fn user(user_id: &str) -> Self {
        Self { user_id: Some(user_id.into()), ..Default::default() }
    }

    pub fn tenant(mut self, tenant_id: &str) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// From JSON, as Python's `Principal.of`: a string is a user id; an object
    /// has `user_id`, `tenant_id`, and anything else as claims.
    pub fn of(value: &Value) -> Self {
        match value {
            Value::String(s) => Self::user(s),
            Value::Object(m) => Self {
                user_id: m.get("user_id").and_then(Value::as_str).map(str::to_string),
                tenant_id: m.get("tenant_id").and_then(Value::as_str).map(str::to_string),
                claims: m.iter().filter(|(k, _)| *k != "user_id" && *k != "tenant_id").map(|(k, v)| (k.clone(), v.clone())).collect(),
            },
            _ => Self::default(),
        }
    }

    /// `tenant/user`, the key a caller's tasks and agents are kept under.
    pub fn key(&self) -> String {
        format!("{}/{}", self.tenant_id.as_deref().unwrap_or(""), self.user_id.as_deref().unwrap_or(""))
    }
}

/// Builds the agent that serves one caller's conversation.
pub type AgentFactory = Arc<dyn Fn(&Principal) -> Result<Agent> + Send + Sync>;
/// From request headers to who is calling; `None` refuses (401).
pub type A2AAuthFn = Arc<dyn Fn(&HeaderMap) -> Pin<Box<dyn Future<Output = Option<Principal>> + Send>> + Send + Sync>;
/// Decides whether a webhook URL may be called.
pub type PushFilter = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// What a server serves: an agent, or a function from the caller to one.
#[derive(Clone)]
pub enum Served {
    Agent(Agent),
    Factory(AgentFactory),
}

impl Served {
    /// A new agent for every conversation, built for its caller.
    pub fn factory<F: Fn(&Principal) -> Result<Agent> + Send + Sync + 'static>(f: F) -> Self {
        Self::Factory(Arc::new(f))
    }
}

impl From<Agent> for Served {
    fn from(a: Agent) -> Self {
        Self::Agent(a)
    }
}

/// Who may send webhooks where.
#[derive(Clone)]
enum Push {
    Off,
    /// Any public https URL.
    PublicHttps,
    Allow(PushFilter),
}

struct Mount {
    name: String,
    prefix: String,
    agent: Option<Agent>,
    factory: Option<AgentFactory>,
}

/// Builds an [`A2AServer`].
pub struct A2AServerBuilder {
    mounts: Vec<(String, String, Served)>,
    url: Option<String>,
    tokens: Vec<(String, Principal)>,
    auth: Option<A2AAuthFn>,
    store: Option<Arc<dyn TaskStore>>,
    max_concurrency: usize,
    max_queue: usize,
    task_timeout: f64,
    push: Push,
    version: String,
    provider: Option<Value>,
    documentation_url: Option<String>,
    skills: Option<Vec<Value>>,
    security_schemes: Option<Map<String, Value>>,
    max_body_bytes: usize,
    max_agents: usize,
    poll_interval: f64,
    keepalive: f64,
}

impl A2AServerBuilder {
    fn new(mounts: Vec<(String, String, Served)>) -> Self {
        Self {
            mounts,
            url: None,
            tokens: vec![],
            auth: None,
            store: None,
            max_concurrency: 64,
            max_queue: 256,
            task_timeout: 600.0,
            push: Push::Off,
            version: "1.0.0".into(),
            provider: None,
            documentation_url: None,
            skills: None,
            security_schemes: None,
            max_body_bytes: 10_000_000,
            max_agents: 1024,
            poll_interval: 1.0,
            keepalive: 15.0,
        }
    }

    /// The public base URL, for the agent card. Left out, it is read from each
    /// request — which is right behind most proxies.
    pub fn url(mut self, url: &str) -> Self {
        self.url = Some(url.trim_end_matches('/').to_string());
        self
    }

    /// A bearer token that may call, as nobody in particular.
    pub fn token(self, token: &str) -> Self {
        self.token_for(token, Principal::default())
    }

    /// A bearer token, and who it is.
    pub fn token_for(mut self, token: &str, who: Principal) -> Self {
        self.tokens.push((token.into(), who));
        self
    }

    /// Decide who is calling yourself, from the request headers.
    pub fn auth<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(&HeaderMap) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<Principal>> + Send + 'static,
    {
        self.auth = Some(Arc::new(move |h| Box::pin(f(h))));
        self
    }

    /// Where tasks are kept. Left out, the harness's session store.
    pub fn store(mut self, store: Arc<dyn TaskStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// How many tasks run at once on this replica.
    pub fn max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self
    }

    /// How many more may wait. Past that, callers are told to come back (`429`).
    pub fn max_queue(mut self, n: usize) -> Self {
        self.max_queue = n;
        self
    }

    /// The longest one task may run, in seconds.
    pub fn task_timeout(mut self, seconds: f64) -> Self {
        self.task_timeout = seconds;
        self
    }

    /// Let callers register a webhook to be told when a task is over: any
    /// public https URL.
    pub fn push_notifications(mut self, on: bool) -> Self {
        self.push = if on { Push::PublicHttps } else { Push::Off };
        self
    }

    /// Push notifications, to the URLs `allowed` says yes to.
    pub fn push_notifications_to<F: Fn(&str) -> bool + Send + Sync + 'static>(mut self, allowed: F) -> Self {
        self.push = Push::Allow(Arc::new(allowed));
        self
    }

    /// The agent's version, as the card says it.
    pub fn version(mut self, version: &str) -> Self {
        self.version = version.into();
        self
    }

    /// `{"organization": ..., "url": ...}` for the card.
    pub fn provider(mut self, provider: Value) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn documentation_url(mut self, url: &str) -> Self {
        self.documentation_url = Some(url.into());
        self
    }

    /// The card's skills, instead of the ones read off the agent.
    pub fn skills(mut self, skills: Vec<Value>) -> Self {
        self.skills = Some(skills);
        self
    }

    /// The card's security schemes, instead of plain bearer.
    pub fn security_schemes(mut self, schemes: Map<String, Value>) -> Self {
        self.security_schemes = Some(schemes);
        self
    }

    pub fn max_body_bytes(mut self, n: usize) -> Self {
        self.max_body_bytes = n;
        self
    }

    /// How many per-conversation agents a factory's are kept, newest first.
    pub fn max_agents(mut self, n: usize) -> Self {
        self.max_agents = n.max(1);
        self
    }

    /// How often, in seconds, a task on another replica is looked at.
    pub fn poll_interval(mut self, seconds: f64) -> Self {
        self.poll_interval = seconds;
        self
    }

    /// How often, in seconds, a quiet stream says it is still there.
    pub fn keepalive(mut self, seconds: f64) -> Self {
        self.keepalive = seconds;
        self
    }

    pub fn build(self) -> Result<A2AServer> {
        let mut mounts: IndexMap<String, Mount> = IndexMap::new();
        for (name, prefix, served) in self.mounts {
            let (agent, factory) = match served {
                Served::Agent(a) => (Some(a), None),
                Served::Factory(f) => (None, Some(f)),
            };
            mounts.insert(name.clone(), Mount { name, prefix, agent, factory });
        }
        let Some(first) = mounts.values().next() else { return Err(Error::config("an A2A server needs at least one agent")) };
        let harness = first.agent.as_ref().map(|a| a.harness().clone());
        let store = match (self.store, &harness) {
            (Some(s), _) => s,
            (None, Some(h)) => Arc::new(SessionTaskStore::new(h.sessions.clone())) as Arc<dyn TaskStore>,
            (None, None) => {
                return Err(Error::config("an A2A server built from a factory needs `store(...)` — SessionTaskStore::new(harness.sessions), say"));
            }
        };
        let auth = match (self.auth, self.tokens.is_empty()) {
            (Some(f), _) => Auth::Func(f),
            (None, false) => Auth::Tokens(self.tokens),
            (None, true) => Auth::Open,
        };
        let stats = ["accepted", "completed", "failed", "canceled", "rejected", "deduplicated"].into_iter().map(|k| (k.to_string(), 0u64)).collect();
        Ok(A2AServer {
            inner: Arc::new(Inner {
                mounts,
                harness,
                store,
                url: self.url,
                auth,
                max_concurrency: self.max_concurrency,
                max_queue: self.max_queue,
                task_timeout: self.task_timeout,
                push: self.push,
                version: self.version,
                provider: self.provider,
                documentation_url: self.documentation_url,
                skills: self.skills,
                security_schemes: self.security_schemes,
                max_body_bytes: self.max_body_bytes,
                max_agents: self.max_agents,
                poll_interval: self.poll_interval,
                keepalive: self.keepalive,
                gate: Semaphore::new(self.max_concurrency),
                running: Mutex::new(HashMap::new()),
                buses: Mutex::new(HashMap::new()),
                agents: Mutex::new(IndexMap::new()),
                pushes: Mutex::new(vec![]),
                http: OnceLock::new(),
                closing: AtomicBool::new(false),
                stats: Mutex::new(stats),
            }),
        })
    }
}

enum Auth {
    Open,
    Tokens(Vec<(String, Principal)>),
    Func(A2AAuthFn),
}

/// The events of one running task, for everyone listening to it here.
type Bus = Arc<Mutex<Vec<mpsc::Sender<Option<Value>>>>>;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn listen(bus: &Bus) -> mpsc::Receiver<Option<Value>> {
    let (tx, rx) = mpsc::channel(2000);
    lock(bus).push(tx);
    rx
}

fn publish(bus: &Bus, event: Option<Value>) {
    // A listener that cannot keep up is dropped, not waited for.
    lock(bus).retain(|tx| tx.try_send(event.clone()).is_ok());
}

struct Worker {
    cancel: watch::Sender<bool>,
    done: watch::Receiver<bool>,
}

struct Inner {
    mounts: IndexMap<String, Mount>,
    harness: Option<Harness>,
    store: Arc<dyn TaskStore>,
    url: Option<String>,
    auth: Auth,
    max_concurrency: usize,
    max_queue: usize,
    task_timeout: f64,
    push: Push,
    version: String,
    provider: Option<Value>,
    documentation_url: Option<String>,
    skills: Option<Vec<Value>>,
    security_schemes: Option<Map<String, Value>>,
    max_body_bytes: usize,
    max_agents: usize,
    poll_interval: f64,
    keepalive: f64,
    gate: Semaphore,
    running: Mutex<HashMap<String, Worker>>,
    buses: Mutex<HashMap<String, Bus>>,
    agents: Mutex<IndexMap<(String, String), Agent>>,
    pushes: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    http: OnceLock<reqwest::Client>,
    closing: AtomicBool,
    stats: Mutex<IndexMap<String, u64>>,
}

/// What `handle` answers: one JSON-RPC response, or (for `message/stream` and
/// `tasks/resubscribe`) a stream of them.
pub enum Reply {
    Json(Value),
    Stream(BoxStream<'static, Value>),
}

impl std::fmt::Debug for Reply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(v) => write!(f, "Reply::Json({v})"),
            Self::Stream(_) => f.write_str("Reply::Stream(..)"),
        }
    }
}

impl Reply {
    /// The response, if it is one.
    pub fn json(self) -> Option<Value> {
        match self {
            Self::Json(v) => Some(v),
            Self::Stream(_) => None,
        }
    }
}

/// Why a run did not come back with a result.
enum Interrupt {
    Cancelled,
    Failed(String),
}

/// One or more agents, served over A2A. Cheap to clone.
#[derive(Clone)]
pub struct A2AServer {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for A2AServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("A2AServer").field("agents", &self.inner.mounts.keys().collect::<Vec<_>>()).finish()
    }
}

fn py_s(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) | Some(Value::Bool(false)) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => crate::governance::py_str(other),
    }
}

/// Python's `f"{x:g}"` for the seconds we print.
fn seconds(s: f64) -> String {
    if s.fract() == 0.0 { format!("{}", s as i64) } else { format!("{s}") }
}

/// Constant time in the contents: says nothing about which byte differed.
fn same(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn context_session(mount: &str, context_id: &str) -> String {
    format!("a2a_{mount}_{context_id}")
}

fn without_bytes(part: &Value) -> Value {
    match part.get("file") {
        Some(Value::Object(file)) if file.contains_key("bytes") => {
            let mut out = part.clone();
            out["file"] = Value::Object(file.iter().filter(|(k, _)| *k != "bytes").map(|(k, v)| (k.clone(), v.clone())).collect());
            out
        }
        _ => part.clone(),
    }
}

fn to_state(state: &'static str) -> impl FnMut(&mut TaskRecord) -> bool + Send {
    move |record| {
        if is_terminal(record.state()) {
            return false;
        }
        record.task["status"] = wire::status(state, "", "", "");
        record.updated = crate::types::now();
        true
    }
}

impl A2AServer {
    /// Serve one agent (or factory) at the root: the card at
    /// `/.well-known/agent-card.json`, requests at `/`.
    pub fn builder(served: impl Into<Served>) -> A2AServerBuilder {
        let served = served.into();
        let name = match &served {
            Served::Agent(a) => a.name().to_string(),
            Served::Factory(_) => "agent".into(),
        };
        A2AServerBuilder::new(vec![(name, String::new(), served)])
    }

    /// Serve several, each under `/<name>`.
    pub fn builder_many<I, S>(served: I) -> A2AServerBuilder
    where
        I: IntoIterator<Item = (S, Served)>,
        S: Into<String>,
    {
        A2AServerBuilder::new(
            served
                .into_iter()
                .map(|(n, s)| {
                    let n = n.into();
                    (n.clone(), format!("/{n}"), s)
                })
                .collect(),
        )
    }

    /// The names of the agents served.
    pub fn agent_names(&self) -> Vec<String> {
        self.inner.mounts.keys().cloned().collect()
    }

    /// Where tasks are kept.
    pub fn store(&self) -> &Arc<dyn TaskStore> {
        &self.inner.store
    }

    fn mount(&self, name: Option<&str>) -> std::result::Result<&Mount, A2AError> {
        match name {
            None => Ok(self.inner.mounts.values().next().expect("at least one mount")),
            Some(n) => self.inner.mounts.get(n).ok_or_else(|| A2AError::http(format!("no agent '{n}' is served"), wire::INVALID_PARAMS, 404)),
        }
    }

    fn bump(&self, stat: &str) {
        *lock(&self.inner.stats).entry(stat.to_string()).or_insert(0) += 1;
    }

    fn is_running(&self, task_id: &str) -> bool {
        lock(&self.inner.running).contains_key(task_id)
    }

    // ------------------------------------------------------------------
    // the agent card
    // ------------------------------------------------------------------
    /// The agent card: what this agent is, where it is, how to call it.
    pub fn card(&self, name: Option<&str>, base_url: &str) -> Value {
        let Ok(mount) = self.mount(name) else { return Value::Null };
        let inner = &self.inner;
        let agent = mount.agent.as_ref();
        let described = agent.map(|a| a.description().to_string()).filter(|d| !d.is_empty()).unwrap_or_else(|| format!("{} agent", mount.name));
        let skills = match &inner.skills {
            Some(s) => s.clone(),
            None => {
                let mut skills: Vec<Value> = agent
                    .and_then(|a| a.skills())
                    .map(|r| {
                        r.iter()
                            .map(|s| {
                                let tags = if s.tags.is_empty() { vec!["skill".to_string()] } else { s.tags.clone() };
                                let description = if s.description.is_empty() { s.name.clone() } else { s.description.clone() };
                                json!({"id": s.name, "name": s.name, "description": description, "tags": tags})
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if skills.is_empty() {
                    let mode = agent.and_then(|a| a.mode()).map(|m| m.name.clone()).filter(|m| !m.is_empty()).unwrap_or_else(|| "agent".into());
                    skills.push(json!({"id": mount.name, "name": mount.name, "description": described, "tags": [mode]}));
                }
                skills
            }
        };
        let base = inner.url.clone().unwrap_or_else(|| base_url.to_string());
        let mut card = json!({
            "protocolVersion": wire::PROTOCOL_VERSION,
            "name": mount.name,
            "description": described,
            "url": format!("{base}{}/", mount.prefix),
            "preferredTransport": "JSONRPC",
            "version": inner.version,
            "capabilities": {"streaming": true, "pushNotifications": !matches!(inner.push, Push::Off), "stateTransitionHistory": false},
            "defaultInputModes": INPUT_MODES,
            "defaultOutputModes": ["text/plain"],
            "skills": skills,
            "supportsAuthenticatedExtendedCard": false,
        });
        if let Some(p) = &inner.provider {
            card["provider"] = p.clone();
        }
        if let Some(d) = &inner.documentation_url {
            card["documentationUrl"] = json!(d);
        }
        if let Some(schemes) = inner.security_schemes.as_ref().filter(|s| !s.is_empty()) {
            card["securitySchemes"] = Value::Object(schemes.clone());
            card["security"] = Value::Array(schemes.keys().map(|k| json!({k.clone(): []})).collect());
        } else if !matches!(inner.auth, Auth::Open) {
            card["securitySchemes"] = json!({"bearer": {"type": "http", "scheme": "bearer"}});
            card["security"] = json!([{"bearer": []}]);
        }
        card
    }

    // ------------------------------------------------------------------
    // who is calling
    // ------------------------------------------------------------------
    /// Who this request is from. An `A2AError` (401) if nobody we know.
    pub async fn authenticate(&self, headers: &HeaderMap) -> std::result::Result<Principal, A2AError> {
        let refused = || A2AError::http("not authorised", wire::INVALID_REQUEST, 401);
        match &self.inner.auth {
            Auth::Open => Ok(Principal::default()),
            Auth::Func(f) => f(headers).await.ok_or_else(refused),
            Auth::Tokens(known) => {
                let authz = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).unwrap_or("");
                let (scheme, token) = authz.split_once(' ').unwrap_or((authz, ""));
                let token = token.trim();
                if !scheme.eq_ignore_ascii_case("bearer") || token.is_empty() {
                    return Err(refused());
                }
                // Every token is compared, in constant time, so how long this
                // takes says nothing about which one nearly matched.
                let mut found: Option<Principal> = None;
                for (candidate, who) in known {
                    if same(candidate.as_bytes(), token.as_bytes()) && found.is_none() {
                        found = Some(who.clone());
                    }
                }
                found.ok_or_else(refused)
            }
        }
    }

    // ------------------------------------------------------------------
    // the agent for a conversation
    // ------------------------------------------------------------------
    fn agent_for(&self, mount: &Mount, context_id: &str, principal: &Principal) -> Result<Agent> {
        let Some(factory) = &mount.factory else { return Ok(mount.agent.clone().expect("a mount has an agent or a factory")) };
        let key = (mount.name.clone(), format!("{}/{context_id}", principal.key()));
        {
            let mut agents = lock(&self.inner.agents);
            if let Some(held) = agents.shift_remove(&key) {
                agents.insert(key, held.clone());
                return Ok(held);
            }
        }
        let built = factory(principal)?;
        let mut agents = lock(&self.inner.agents);
        agents.insert(key, built.clone());
        while agents.len() > self.inner.max_agents {
            agents.shift_remove_index(0);
        }
        Ok(built)
    }

    // ------------------------------------------------------------------
    // JSON-RPC
    // ------------------------------------------------------------------
    /// Answer one JSON-RPC request, from `principal` (nobody, if `None`), to
    /// the agent named `agent` (the first, if `None`).
    ///
    /// An error about the task is a JSON-RPC error in the reply; one about the
    /// request itself (401, 429, 503) is an `Err` with its HTTP status. This
    /// is the whole protocol without HTTP around it: mount it in a framework of
    /// your own, or call it in a test.
    pub async fn handle(&self, payload: Value, principal: Option<&Principal>, agent: Option<&str>) -> std::result::Result<Reply, A2AError> {
        let nobody = Principal::default();
        let principal = principal.unwrap_or(&nobody);
        let mount = self.mount(agent)?;
        let request_id = payload.get("id").cloned().unwrap_or(Value::Null);
        let frame = |id: Value, r: std::result::Result<Value, A2AError>| match r {
            Ok(result) => Ok(json!({"jsonrpc": "2.0", "id": id, "result": result})),
            Err(e) if e.status != 200 => Err(e),
            Err(e) => Ok(json!({"jsonrpc": "2.0", "id": id, "error": e.body()})),
        };
        let method = payload.get("method").and_then(Value::as_str);
        let (Some(method), true) = (method, payload.get("jsonrpc").and_then(Value::as_str) == Some("2.0")) else {
            return frame(request_id, Err(A2AError::new("not a JSON-RPC 2.0 request", wire::INVALID_REQUEST))).map(Reply::Json);
        };
        let params = match payload.get("params") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(m)) => m.clone(),
            Some(Value::Array(a)) if a.is_empty() => Map::new(),
            Some(_) => return frame(request_id, Err(A2AError::new("params must be an object", wire::INVALID_PARAMS))).map(Reply::Json),
        };
        let framed = |events: BoxStream<'static, Value>, id: Value| -> Reply {
            Reply::Stream(events.map(move |e| json!({"jsonrpc": "2.0", "id": id, "result": e})).boxed())
        };
        let result = match method {
            "message/stream" => match self.send_stream(mount, &params, principal).await {
                Ok(events) => return Ok(framed(events, request_id)),
                Err(e) => Err(e),
            },
            "tasks/resubscribe" => match self.task(params.get("id"), principal).await {
                Ok(record) => return Ok(framed(self.follow(record, true), request_id)),
                Err(e) => Err(e),
            },
            "message/send" => self.send(mount, &params, principal).await,
            "tasks/get" => self.get(&params, principal).await,
            "tasks/cancel" => self.cancel(&params, principal).await,
            "tasks/pushNotificationConfig/set" => self.push_set(&params, principal).await,
            "tasks/pushNotificationConfig/get" => self.push_get(&params, principal).await,
            "tasks/pushNotificationConfig/list" => self.push_list(&params, principal).await,
            "tasks/pushNotificationConfig/delete" => self.push_delete(&params, principal).await,
            "agent/getAuthenticatedExtendedCard" => Err(A2AError::new("this agent has no extended card", wire::NO_EXTENDED_CARD)),
            other => Err(A2AError::new(format!("method '{other}' is not supported"), wire::METHOD_NOT_FOUND)),
        };
        frame(request_id, result).map(Reply::Json)
    }

    // ---- tasks: finding one -------------------------------------------------
    /// The caller's task. Somebody else's is "not found", like one that never
    /// existed, so an id cannot be probed.
    async fn task(&self, task_id: Option<&Value>, principal: &Principal) -> std::result::Result<TaskRecord, A2AError> {
        let Some(task_id) = task_id.and_then(Value::as_str).filter(|t| !t.is_empty()) else {
            return Err(A2AError::new("a task id is required", wire::INVALID_PARAMS));
        };
        let record = self.inner.store.load(task_id).await?;
        let Some(record) = record.filter(|r| r.owned_by(principal.user_id.as_deref(), principal.tenant_id.as_deref())) else {
            return Err(A2AError::new(format!("no task '{task_id}'"), wire::TASK_NOT_FOUND));
        };
        if !is_terminal(record.state()) && !self.is_running(task_id) && crate::types::now() - record.updated > self.inner.task_timeout + 60.0 {
            // Nobody here is working on it, and whoever was has had longer than
            // any task is allowed: its worker is gone.
            let changed = self
                .inner
                .store
                .change(task_id, &mut |r| {
                    if is_terminal(r.state()) {
                        return false;
                    }
                    let context = r.task["contextId"].as_str().unwrap_or("").to_string();
                    r.task["status"] = wire::status("failed", "The worker handling this task was lost before it finished.", &r.id, &context);
                    r.updated = crate::types::now();
                    true
                })
                .await?;
            return Ok(changed.unwrap_or(record));
        }
        Ok(record)
    }

    // ---- message/send, message/stream ---------------------------------------
    /// Turn a message into a task — or find the task it already became.
    async fn accept(
        &self,
        mount: &Mount,
        params: &Map<String, Value>,
        principal: &Principal,
    ) -> std::result::Result<(TaskRecord, bool, String, Vec<ContentBlock>), A2AError> {
        let Some(Value::Object(incoming)) = params.get("message") else { return Err(A2AError::new("`message` is required", wire::INVALID_PARAMS)) };
        if incoming.get("role").is_some_and(|r| r.as_str() != Some("user")) {
            return Err(A2AError::new("a message sent to an agent has the role `user`", wire::INVALID_PARAMS));
        }
        let (text, files) = wire::read_parts(incoming.get("parts"))?;
        if text.is_empty() && files.is_empty() {
            return Err(A2AError::new("the message is empty", wire::INVALID_PARAMS));
        }
        let earlier_id = py_s(incoming.get("taskId"));
        if !earlier_id.is_empty() {
            let earlier = self.task(Some(&json!(earlier_id)), principal).await?;
            if is_terminal(earlier.state()) {
                return Err(A2AError::new(
                    format!(
                        "task {} is {} and takes no more messages; send a new message with its contextId to carry on the conversation",
                        earlier.id,
                        earlier.state()
                    ),
                    wire::INVALID_PARAMS,
                ));
            }
            return Err(A2AError::new(
                format!("task {} is still {}; this agent does not take input part-way through a task", earlier.id, earlier.state()),
                wire::UNSUPPORTED_OPERATION,
            ));
        }
        let message_id = Some(py_s(incoming.get("messageId"))).filter(|s| !s.is_empty()).unwrap_or_else(wire::new_uuid);
        let given_context = py_s(incoming.get("contextId"));
        let context_id = if given_context.is_empty() { wire::new_uuid() } else { given_context.clone() };
        // The same message from the same caller is the same task: a client that
        // retries a request it never saw the answer to does not pay twice.
        let digest: String =
            Sha256::digest(format!("{}\0{}\0{message_id}", mount.name, principal.key()).as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
        let task_id = format!("{}-{}-{}-{}-{}", &digest[..8], &digest[8..12], &digest[12..16], &digest[16..20], &digest[20..32]);
        if let Some(existing) = self.inner.store.load(&task_id).await? {
            self.bump("deduplicated");
            return Ok((existing, false, text, files));
        }
        if let Some(agent) = &mount.agent
            && !given_context.is_empty()
        {
            // A conversation that is somebody else's is refused before any work
            // is started on it — in the words used for one that does not exist.
            if let Ok(held) = agent.harness().sessions.load(&context_session(&mount.name, &context_id)).await
                && !held.owned_by(principal.user_id.as_deref(), principal.tenant_id.as_deref())
            {
                return Err(A2AError::new(format!("no conversation '{context_id}'"), wire::INVALID_PARAMS));
            }
        }
        if self.inner.closing.load(Ordering::SeqCst) {
            return Err(A2AError::http("the server is shutting down", wire::SERVER_BUSY, 503).with_retry_after(5.0));
        }
        if lock(&self.inner.running).len() >= self.inner.max_concurrency + self.inner.max_queue {
            self.bump("rejected");
            return Err(A2AError::http("the server is at capacity; try again shortly", wire::SERVER_BUSY, 429).with_retry_after(2.0));
        }
        let configuration = params.get("configuration").cloned().unwrap_or_else(|| json!({}));
        let mut push: Vec<Value> = Vec::new();
        if let Some(config) = configuration.get("pushNotificationConfig").filter(|c| !c.is_null()) {
            push.push(self.push_checked(Some(config))?);
        }
        // Kept without the bytes of what was attached: the record is read far
        // more often than a file would ever be wanted back.
        let mut said = incoming.clone();
        said.insert("kind".into(), json!("message"));
        said.insert("role".into(), json!("user"));
        said.insert("messageId".into(), json!(message_id));
        said.insert("taskId".into(), json!(task_id));
        said.insert("contextId".into(), json!(context_id));
        let parts: Vec<Value> = incoming.get("parts").and_then(Value::as_array).into_iter().flatten().map(without_bytes).collect();
        said.insert("parts".into(), Value::Array(parts));
        let mut task = json!({"kind": "task", "id": task_id, "contextId": context_id, "status": wire::status("submitted", "", "", ""),
                              "history": [said], "artifacts": []});
        if let Some(m) = params.get("metadata").filter(|m| m.is_object()) {
            task["metadata"] = m.clone();
        }
        let mut record = TaskRecord {
            id: task_id.clone(),
            agent: mount.name.clone(),
            user_id: principal.user_id.clone(),
            tenant_id: principal.tenant_id.clone(),
            task,
            push,
            ..Default::default()
        };
        if !self.inner.store.create(&mut record).await? {
            // Another replica got the same message a moment sooner.
            self.bump("deduplicated");
            let held = self.inner.store.load(&task_id).await?.unwrap_or(record);
            return Ok((held, false, text, files));
        }
        self.bump("accepted");
        Ok((record, true, text, files))
    }

    fn start(&self, mount: &Mount, record: &TaskRecord, principal: &Principal, text: String, files: Vec<ContentBlock>, bus: Bus) {
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let (done_tx, done_rx) = watch::channel(false);
        lock(&self.inner.buses).insert(record.id.clone(), bus.clone());
        lock(&self.inner.running).insert(record.id.clone(), Worker { cancel: cancel_tx, done: done_rx });
        let (server, mount, record, principal) = (self.clone(), mount.name.clone(), record.clone(), principal.clone());
        tokio::spawn(async move {
            let _settle = Settle { server: server.clone(), task_id: record.id.clone(), done: Some(done_tx) };
            server.work(&mount, record, principal, text, files, bus, cancel_rx).await;
        });
    }

    async fn send(&self, mount: &Mount, params: &Map<String, Value>, principal: &Principal) -> std::result::Result<Value, A2AError> {
        let (mut record, created, text, files) = self.accept(mount, params, principal).await?;
        let configuration = params.get("configuration").cloned().unwrap_or_else(|| json!({}));
        if created {
            self.start(mount, &record, principal, text, files, Bus::default());
        }
        if configuration.get("blocking") != Some(&Value::Bool(false)) {
            record = self.settled(record).await?;
        }
        Ok(wire::trimmed(&record.task, configuration.get("historyLength")))
    }

    /// Wait for a task to be over — here, or on whichever replica has it.
    async fn settled(&self, record: TaskRecord) -> std::result::Result<TaskRecord, A2AError> {
        let limit = Duration::from_secs_f64(self.inner.task_timeout + 5.0);
        let deadline = Instant::now() + limit;
        let done = lock(&self.inner.running).get(&record.id).map(|w| w.done.clone());
        if let Some(mut done) = done {
            let _ = tokio::time::timeout(limit, done.wait_for(|d| *d)).await;
        }
        loop {
            let latest = self.inner.store.load(&record.id).await?.unwrap_or_else(|| record.clone());
            if is_terminal(latest.state()) || Instant::now() >= deadline || self.is_running(&record.id) {
                return Ok(latest);
            }
            tokio::time::sleep(Duration::from_secs_f64(self.inner.poll_interval)).await;
        }
    }

    async fn send_stream(&self, mount: &Mount, params: &Map<String, Value>, principal: &Principal) -> std::result::Result<BoxStream<'static, Value>, A2AError> {
        let (record, created, text, files) = self.accept(mount, params, principal).await?;
        if !created {
            return Ok(self.follow(record, true));
        }
        // Listening before the work starts, so not one event is missed.
        let bus = Bus::default();
        let rx = listen(&bus);
        self.start(mount, &record, principal, text, files, bus);
        Ok(listened(record.task, rx, true))
    }

    /// The events of a task that is already under way: from the worker if it
    /// is here, from the store if it is on another replica.
    fn follow(&self, record: TaskRecord, first: bool) -> BoxStream<'static, Value> {
        let server = self.clone();
        Box::pin(async_stream::stream! {
            let bus = lock(&server.inner.buses).get(&record.id).cloned();
            if let Some(bus) = bus.filter(|_| !is_terminal(record.state())) {
                let rx = listen(&bus);
                let latest = server.inner.store.load(&record.id).await.ok().flatten().unwrap_or_else(|| record.clone());
                let mut events = listened(latest.task, rx, first);
                while let Some(e) = events.next().await {
                    yield e;
                }
                return;
            }
            if first {
                yield record.task.clone();
            }
            let mut state = record.state().to_string();
            let deadline = Instant::now() + Duration::from_secs_f64(server.inner.task_timeout + 60.0);
            while !is_terminal(&state) && Instant::now() < deadline {
                tokio::time::sleep(Duration::from_secs_f64(server.inner.poll_interval)).await;
                let Ok(Some(latest)) = server.inner.store.load(&record.id).await else { return };
                if latest.state() != state {
                    state = latest.state().to_string();
                    let over = is_terminal(&state);
                    if over {
                        for artifact in latest.task.get("artifacts").and_then(Value::as_array).into_iter().flatten() {
                            yield wire::artifact_event(&latest.task, artifact, false, true);
                        }
                    }
                    yield wire::status_event(&latest.task, over);
                }
            }
        })
    }

    // ---- the work ---------------------------------------------------------------
    /// Run the agent on one task, and write down how it went.
    #[allow(clippy::too_many_arguments)]
    async fn work(
        &self,
        mount_name: &str,
        record: TaskRecord,
        principal: Principal,
        text: String,
        files: Vec<ContentBlock>,
        bus: Bus,
        mut cancel: watch::Receiver<bool>,
    ) {
        let mount = &self.inner.mounts[mount_name];
        let task_id = record.id.clone();
        let context_id = record.task["contextId"].as_str().unwrap_or("").to_string();
        let answer_id = wire::new_uuid();
        let timeout = self.inner.task_timeout;
        let run = async {
            let _permit = self.inner.gate.acquire().await.map_err(|_| Interrupt::Cancelled)?;
            let working = self.inner.store.change(&task_id, &mut to_state("working")).await.map_err(|e| Interrupt::Failed(e.to_string()))?;
            let working = match working {
                Some(w) if !w.cancel_requested => w,
                _ => return Err(Interrupt::Cancelled),
            };
            publish(&bus, Some(wire::status_event(&working.task, false)));
            match tokio::time::timeout(Duration::from_secs_f64(timeout), self.run(mount, &working, &principal, &text, &files, &bus, &answer_id)).await {
                Ok(outcome) => outcome,
                Err(_) => Err(Interrupt::Failed(format!("The task ran longer than {}s and was stopped.", seconds(timeout)))),
            }
        };
        let stopped = async {
            if cancel.wait_for(|c| *c).await.is_err() {
                std::future::pending::<()>().await;
            }
        };
        let outcome = tokio::select! {
            o = run => o,
            _ = stopped => Err(Interrupt::Cancelled),
        };
        let (state, note, result, streamed) = match outcome {
            Ok((result, streamed)) => {
                let (state, note) = if result.stop_reason == StopReason::Stopped {
                    ("canceled", result.error.clone().unwrap_or_else(|| "stopped".into()))
                } else if let Some(e) = &result.error {
                    ("failed", e.clone())
                } else {
                    ("completed", String::new())
                };
                (state, note, Some(result), streamed)
            }
            Err(Interrupt::Cancelled) => {
                let note = if self.inner.closing.load(Ordering::SeqCst) { "The server shut down before this task finished." } else { "Canceled." };
                ("canceled", note.to_string(), None, String::new())
            }
            Err(Interrupt::Failed(note)) => ("failed", note, None, String::new()),
        };
        let artifacts = match (&result, state) {
            (Some(r), "completed") => wire::artifacts_of(r, &answer_id),
            _ => vec![],
        };
        let mut finish = |held: &mut TaskRecord| {
            if is_terminal(held.state()) {
                return false; // already settled, by a cancel say
            }
            held.task["artifacts"] = Value::Array(artifacts.clone());
            held.task["status"] = wire::status(state, &note, &task_id, &context_id);
            if let Some(r) = &result {
                if !r.output.is_empty()
                    && state == "completed"
                    && let Some(h) = held.task["history"].as_array_mut()
                {
                    h.push(wire::text_message("agent", &r.output, &task_id, &context_id));
                }
                if !r.agent.is_empty() {
                    if !held.task.get("metadata").is_some_and(Value::is_object) {
                        held.task["metadata"] = json!({});
                    }
                    held.task["metadata"]["agent"] = json!(r.agent);
                }
            }
            held.updated = crate::types::now();
            true
        };
        let fin = match self.inner.store.change(&task_id, &mut finish).await {
            Ok(Some(f)) => f,
            Ok(None) => return,
            Err(e) => {
                let reason: String = e.to_string().chars().take(300).collect();
                self.audit(mount, "a2a_task", &task_id, "error", vec![("reason", json!(reason))]);
                return;
            }
        };
        self.bump(fin.state());
        self.audit(
            mount,
            "a2a_task",
            &task_id,
            fin.state(),
            vec![("user", json!(principal.user_id)), ("tenant", json!(principal.tenant_id)), ("context", json!(context_id))],
        );
        if fin.state() == "completed" {
            let all = fin.task["artifacts"].as_array().cloned().unwrap_or_default();
            let answer = all.iter().position(|a| a.get("name").and_then(Value::as_str) == Some(wire::ANSWER));
            for (i, artifact) in all.iter().enumerate() {
                if Some(i) == answer && !streamed.is_empty() {
                    // Already sent as it was written. If what was kept differs —
                    // a guardrail redacted it, a mode added its sources — the
                    // whole of it replaces what was streamed.
                    let first = artifact["parts"].as_array().map(|p| Value::Array(p.iter().take(1).cloned().collect()));
                    let same = wire::text_of(first.as_ref()) == streamed;
                    let closing = if same {
                        let mut c = artifact.clone();
                        c["parts"] = json!([wire::text_part("")]);
                        c
                    } else {
                        artifact.clone()
                    };
                    publish(&bus, Some(wire::artifact_event(&fin.task, &closing, same, true)));
                } else {
                    publish(&bus, Some(wire::artifact_event(&fin.task, artifact, false, true)));
                }
            }
        }
        publish(&bus, Some(wire::status_event(&fin.task, true)));
        if !fin.push.is_empty() && is_terminal(fin.state()) {
            self.notify(fin);
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn run(
        &self,
        mount: &Mount,
        record: &TaskRecord,
        principal: &Principal,
        text: &str,
        files: &[ContentBlock],
        bus: &Bus,
        answer_id: &str,
    ) -> std::result::Result<(RunResult, String), Interrupt> {
        let failed = |e: Error| Interrupt::Failed(e.to_string());
        let context_id = record.task["contextId"].as_str().unwrap_or("");
        let agent = self.agent_for(mount, context_id, principal).map_err(failed)?;
        // The conversation is a session of the agent's own, found by the
        // context id — on whichever replica the last turn ran.
        let session_id = context_session(&mount.name, context_id);
        let session = match agent.harness().sessions.load(&session_id).await {
            Ok(s) => s,
            Err(Error::Configuration(_)) => {
                Session { id: session_id, user_id: principal.user_id.clone(), tenant_id: principal.tenant_id.clone(), ..Session::new(agent.name()) }
            }
            Err(e) => return Err(failed(e)),
        };
        let task = if text.is_empty() { "(see the attached files)" } else { text };
        let options = RunOptions { session: Some(SessionRef::Session(Box::new(session))), attachments: files.to_vec(), ..Default::default() };
        let mut events = agent.stream(task, options).await.map_err(failed)?;
        let mut result: Option<RunResult> = None;
        let mut streamed: Vec<String> = Vec::new();
        while let Some(event) = events.next().await {
            match event.kind {
                StreamEventType::Text if !event.text.is_empty() => {
                    let artifact = json!({"artifactId": answer_id, "name": wire::ANSWER, "parts": [wire::text_part(&event.text)]});
                    publish(bus, Some(wire::artifact_event(&record.task, &artifact, !streamed.is_empty(), false)));
                    streamed.push(event.text);
                }
                StreamEventType::Handoff => streamed.clear(), // another agent answers from here
                StreamEventType::StepStart if event.step > 1 => {
                    // Between steps: has another replica been asked to cancel this?
                    if let Ok(Some(held)) = self.inner.store.load(&record.id).await
                        && held.cancel_requested
                    {
                        return Err(Interrupt::Cancelled);
                    }
                }
                StreamEventType::RunEnd => {
                    result = event.data.get("result").and_then(|r| serde_json::from_value(r.clone()).ok());
                }
                _ => {}
            }
        }
        let Some(result) = result else { return Err(Interrupt::Failed("the agent produced no result".into())) };
        Ok((result, streamed.concat()))
    }

    fn audit(&self, mount: &Mount, action: &str, target: &str, decision: &str, detail: Vec<(&str, Value)>) {
        let harness = mount.agent.as_ref().map(|a| a.harness()).or(self.inner.harness.as_ref());
        if let Some(h) = harness {
            let detail: Map<String, Value> = detail.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
            h.audit.record(&mount.name, action, target, decision, "", detail);
        }
    }

    // ---- tasks/get, tasks/cancel ----------------------------------------------
    async fn get(&self, params: &Map<String, Value>, principal: &Principal) -> std::result::Result<Value, A2AError> {
        let record = self.task(params.get("id"), principal).await?;
        Ok(wire::trimmed(&record.task, params.get("historyLength")))
    }

    async fn cancel(&self, params: &Map<String, Value>, principal: &Principal) -> std::result::Result<Value, A2AError> {
        let record = self.task(params.get("id"), principal).await?;
        if is_terminal(record.state()) {
            return Err(A2AError::new(format!("task {} is already {}", record.id, record.state()), wire::TASK_NOT_CANCELABLE));
        }
        let here = lock(&self.inner.running).get(&record.id).map(|w| {
            let _ = w.cancel.send(true);
            w.done.clone()
        });
        if let Some(mut done) = here {
            let _ = tokio::time::timeout(Duration::from_secs(10), done.wait_for(|d| *d)).await;
            return Ok(self.inner.store.load(&record.id).await?.unwrap_or(record).task);
        }
        // Another replica has it. Mark it; that replica stops at its next step.
        let changed = self
            .inner
            .store
            .change(&record.id, &mut |held| {
                if is_terminal(held.state()) {
                    return false;
                }
                held.cancel_requested = true;
                if held.state() == "submitted" {
                    held.task["status"] = wire::status("canceled", "Canceled.", "", "");
                    held.updated = crate::types::now();
                }
                true
            })
            .await?;
        Ok(changed.unwrap_or(record).task)
    }

    // ---- push notifications ---------------------------------------------------
    fn push_checked(&self, config: Option<&Value>) -> std::result::Result<Value, A2AError> {
        if matches!(self.inner.push, Push::Off) {
            return Err(A2AError::new("this agent does not send push notifications", wire::PUSH_NOT_SUPPORTED));
        }
        let Some(Value::Object(config)) = config else { return Err(A2AError::new("a push notification config needs a `url`", wire::INVALID_PARAMS)) };
        let Some(url) = config.get("url").and_then(Value::as_str) else {
            return Err(A2AError::new("a push notification config needs a `url`", wire::INVALID_PARAMS));
        };
        let allowed = match &self.inner.push {
            Push::Allow(f) => f(url),
            _ => public_https(url),
        };
        if !allowed {
            return Err(A2AError::new(format!("push notifications may not be sent to '{url}'"), wire::INVALID_PARAMS));
        }
        let mut out = config.clone();
        if out.get("id").and_then(Value::as_str).is_none_or(str::is_empty) {
            out.insert("id".into(), json!(wire::new_uuid()));
        }
        Ok(Value::Object(out))
    }

    async fn push_set(&self, params: &Map<String, Value>, principal: &Principal) -> std::result::Result<Value, A2AError> {
        let record = self.task(params.get("taskId"), principal).await?;
        let config = self.push_checked(params.get("pushNotificationConfig"))?;
        self.inner
            .store
            .change(&record.id, &mut |held| {
                held.push.retain(|c| c.get("id") != config.get("id"));
                held.push.push(config.clone());
                true
            })
            .await?;
        Ok(json!({"taskId": record.id, "pushNotificationConfig": config}))
    }

    async fn push_get(&self, params: &Map<String, Value>, principal: &Principal) -> std::result::Result<Value, A2AError> {
        let record = self.task(params.get("id"), principal).await?;
        let wanted = params.get("pushNotificationConfigId").filter(|v| !v.is_null());
        for config in &record.push {
            if wanted.is_none() || config.get("id") == wanted {
                return Ok(json!({"taskId": record.id, "pushNotificationConfig": config}));
            }
        }
        Err(A2AError::new("no push notification config for this task", wire::TASK_NOT_FOUND))
    }

    async fn push_list(&self, params: &Map<String, Value>, principal: &Principal) -> std::result::Result<Value, A2AError> {
        let record = self.task(params.get("id"), principal).await?;
        Ok(Value::Array(record.push.iter().map(|c| json!({"taskId": record.id, "pushNotificationConfig": c})).collect()))
    }

    async fn push_delete(&self, params: &Map<String, Value>, principal: &Principal) -> std::result::Result<Value, A2AError> {
        let record = self.task(params.get("id"), principal).await?;
        let wanted = params.get("pushNotificationConfigId").cloned().unwrap_or(Value::Null);
        self.inner
            .store
            .change(&record.id, &mut |held| {
                held.push.retain(|c| c.get("id").unwrap_or(&Value::Null) != &wanted);
                true
            })
            .await?;
        Ok(Value::Null)
    }

    fn notify(&self, record: TaskRecord) {
        let server = self.clone();
        let job = tokio::spawn(async move { server.deliver(record).await });
        let mut pushes = lock(&self.inner.pushes);
        pushes.retain(|j| !j.is_finished());
        pushes.push(job);
    }

    /// Tell each webhook the task is over. Tried three times, then dropped:
    /// the task is still there to be asked about.
    async fn deliver(&self, record: TaskRecord) {
        let http = self.inner.http.get_or_init(|| {
            crate::install_tls();
            reqwest::Client::builder().timeout(Duration::from_secs(10)).redirect(reqwest::redirect::Policy::none()).build().expect("reqwest client")
        });
        for config in &record.push {
            let Some(url) = config.get("url").and_then(Value::as_str) else { continue };
            for attempt in 0..3u32 {
                let mut request = http.post(url).header("content-type", "application/json").body(record.task.to_string());
                if let Some(token) = config.get("token").filter(|t| !t.is_null() && t.as_str() != Some("")) {
                    request = request.header("X-A2A-Notification-Token", crate::governance::py_str(token));
                }
                if let Some(credentials) = config.get("authentication").and_then(|a| a.get("credentials")).filter(|c| !c.is_null() && c.as_str() != Some("")) {
                    request = request.header("Authorization", format!("Bearer {}", crate::governance::py_str(credentials)));
                }
                if let Ok(response) = request.send().await
                    && response.status().as_u16() < 500
                {
                    break;
                }
                tokio::time::sleep(Duration::from_secs_f64(0.5 * f64::from(2u32.pow(attempt)))).await;
            }
        }
    }

    /// Wait for the push notifications already on their way.
    pub async fn flush_notifications(&self) {
        let jobs: Vec<_> = std::mem::take(&mut *lock(&self.inner.pushes));
        for job in jobs {
            let _ = job.await;
        }
    }

    // ------------------------------------------------------------------
    // health, and going away
    // ------------------------------------------------------------------
    pub fn health(&self) -> Value {
        let running = lock(&self.inner.running).len();
        let mut agents = self.agent_names();
        agents.sort();
        let mut out = json!({
            "status": if self.inner.closing.load(Ordering::SeqCst) { "draining" } else { "ok" },
            "agents": agents,
            "running": running.min(self.inner.max_concurrency),
            "queued": running.saturating_sub(self.inner.max_concurrency),
            "capacity": self.inner.max_concurrency + self.inner.max_queue,
        });
        for (k, v) in lock(&self.inner.stats).iter() {
            out[k] = json!(v);
        }
        out
    }

    /// Stop taking work. Give what is running `drain` seconds to finish, then
    /// cancel it — each task records that it was cancelled.
    pub async fn close(&self, drain: f64) {
        self.inner.closing.store(true, Ordering::SeqCst);
        let workers: Vec<watch::Receiver<bool>> = lock(&self.inner.running).values().map(|w| w.done.clone()).collect();
        let all_done = |mut rxs: Vec<watch::Receiver<bool>>| async move {
            for rx in &mut rxs {
                let _ = rx.wait_for(|d| *d).await;
            }
        };
        if !workers.is_empty() && drain > 0.0 {
            let _ = tokio::time::timeout(Duration::from_secs_f64(drain), all_done(workers.clone())).await;
        }
        for w in lock(&self.inner.running).values() {
            let _ = w.cancel.send(true);
        }
        all_done(workers).await;
        self.flush_notifications().await;
    }

    // ------------------------------------------------------------------
    // HTTP
    // ------------------------------------------------------------------
    /// The `axum` router: the cards, `POST /` (or `/<agent>/`), `GET /healthz`.
    pub fn router(&self) -> Router {
        Router::new().fallback(http_entry).with_state(self.clone())
    }

    /// Serve on `addr` until the process ends. Enough to develop against and
    /// to put behind a proxy; for more, mount [`Self::router`] in your own server.
    pub async fn serve(&self, addr: &str) -> Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        self.serve_on(listener).await
    }

    /// Serve on a listener you bound (port 0 for any free one).
    pub async fn serve_on(&self, listener: tokio::net::TcpListener) -> Result<()> {
        let served = axum::serve(listener, self.router()).await;
        self.close(0.0).await;
        served.map_err(|e| Error::Io(e.to_string()))
    }

    /// Bind any free port on localhost and serve in the background. Returns
    /// the base URL (`http://127.0.0.1:<port>`).
    pub async fn spawn_local(&self) -> Result<(String, tokio::task::JoinHandle<()>)> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr: SocketAddr = listener.local_addr()?;
        let server = self.clone();
        let handle = tokio::spawn(async move {
            let _ = server.serve_on(listener).await;
        });
        Ok((format!("http://{addr}"), handle))
    }

    /// Which agent a path belongs to, and what is left of the path.
    fn route(&self, path: &str) -> Option<(&Mount, String)> {
        let mut best: Option<&Mount> = None;
        for m in self.inner.mounts.values() {
            if (path == m.prefix || path.starts_with(&format!("{}/", m.prefix))) && best.is_none_or(|b| m.prefix.len() > b.prefix.len()) {
                best = Some(m);
            }
        }
        let best = best?;
        if best.prefix.is_empty() && self.inner.mounts.len() > 1 {
            return None;
        }
        let rest = &path[best.prefix.len()..];
        Some((best, if rest.is_empty() { "/".into() } else { rest.to_string() }))
    }
}

/// Settles a task when its worker ends, however it ends.
struct Settle {
    server: A2AServer,
    task_id: String,
    done: Option<watch::Sender<bool>>,
}

impl Drop for Settle {
    fn drop(&mut self) {
        lock(&self.server.inner.running).remove(&self.task_id);
        if let Some(bus) = lock(&self.server.inner.buses).remove(&self.task_id) {
            publish(&bus, None);
        }
        if let Some(d) = self.done.take() {
            let _ = d.send(true);
        }
    }
}

fn listened(task: Value, mut rx: mpsc::Receiver<Option<Value>>, first: bool) -> BoxStream<'static, Value> {
    Box::pin(async_stream::stream! {
        if first {
            yield task;
        }
        while let Some(Some(event)) = rx.recv().await {
            let last = event.get("final") == Some(&Value::Bool(true));
            yield event;
            if last {
                break;
            }
        }
    })
}

/// A webhook the server will call unasked: https, and not this network.
pub(crate) fn public_https(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else { return false };
    if parsed.scheme() != "https" {
        return false;
    }
    let host = parsed.host_str().unwrap_or("").to_ascii_lowercase();
    if host.is_empty() || host == "localhost" || host.ends_with(".local") || host.ends_with(".internal") || host.ends_with(".localhost") {
        return false;
    }
    match host.trim_start_matches('[').trim_end_matches(']').parse::<IpAddr>() {
        Ok(ip) => is_global(ip),
        Err(_) => true,
    }
}

/// Python's `ipaddress.ip_address(host).is_global`, for the ranges that matter.
fn is_global(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            !(v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                || o[0] == 0
                || (o[0] == 100 && (o[1] & 0xc0) == 64)
                || (o[0] == 198 && (o[1] & 0xfe) == 18)
                || o[0] >= 240)
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_global(IpAddr::V4(v4));
            }
            let s = v6.segments();
            !(v6.is_loopback() || v6.is_unspecified() || (s[0] & 0xfe00) == 0xfc00 || (s[0] & 0xffc0) == 0xfe80 || (s[0] == 0x2001 && s[1] == 0x0db8))
        }
    }
}

fn base_url(headers: &HeaderMap) -> String {
    let get = |h: &str| headers.get(h).and_then(|v| v.to_str().ok()).map(str::to_string);
    let scheme = get("x-forwarded-proto").unwrap_or_else(|| "http".into());
    let scheme = scheme.split(',').next().unwrap_or("http").trim().to_string();
    let host = get("x-forwarded-host").or_else(|| get("host")).unwrap_or_else(|| "localhost:80".into());
    format!("{scheme}://{}", host.split(',').next().unwrap_or("").trim())
}

fn json_reply(status: StatusCode, body: &Value, extra: &[(&str, String)]) -> Response {
    let mut r = (status, [(header::CONTENT_TYPE, "application/json")], body.to_string()).into_response();
    for (k, v) in extra {
        if let (Ok(name), Ok(value)) = (header::HeaderName::from_bytes(k.as_bytes()), v.parse()) {
            r.headers_mut().insert(name, value);
        }
    }
    r
}

fn refusal(e: &A2AError) -> Response {
    let mut extra: Vec<(&str, String)> = Vec::new();
    if e.status == 401 {
        extra.push(("www-authenticate", "Bearer realm=\"a2a\"".into()));
    }
    if let Some(r) = e.retry_after {
        extra.push(("retry-after", seconds(r)));
    }
    let status = StatusCode::from_u16(if e.status == 200 { 400 } else { e.status }).unwrap_or(StatusCode::BAD_REQUEST);
    json_reply(status, &json!({"jsonrpc": "2.0", "id": null, "error": e.body()}), &extra)
}

async fn http_entry(State(server): State<A2AServer>, method: Method, uri: Uri, headers: HeaderMap, body: Body) -> Response {
    let raw = uri.path();
    let path = if raw.trim_end_matches('/').is_empty() { "/".to_string() } else { raw.trim_end_matches('/').to_string() };
    if method == Method::GET && path == "/healthz" {
        let status = if server.inner.closing.load(Ordering::SeqCst) { StatusCode::SERVICE_UNAVAILABLE } else { StatusCode::OK };
        return json_reply(status, &server.health(), &[]);
    }
    let Some((mount, rest)) = server.route(&path) else {
        if method == Method::GET && path == "/" && server.inner.mounts.len() > 1 {
            let base = server.inner.url.clone().unwrap_or_else(|| base_url(&headers));
            let agents: Map<String, Value> =
                server.inner.mounts.iter().map(|(n, m)| (n.clone(), json!(format!("{base}{}{}", m.prefix, wire::CARD_PATH)))).collect();
            return json_reply(StatusCode::OK, &json!({"agents": agents}), &[]);
        }
        return json_reply(StatusCode::NOT_FOUND, &json!({"error": format!("nothing at {path}")}), &[]);
    };
    if rest == wire::CARD_PATH || rest == wire::LEGACY_CARD_PATH {
        if method != Method::GET {
            return json_reply(StatusCode::METHOD_NOT_ALLOWED, &json!({"error": "the agent card is read with GET"}), &[]);
        }
        let card = server.card(Some(&mount.name), &base_url(&headers));
        return json_reply(StatusCode::OK, &card, &[("cache-control", "public, max-age=300".into())]);
    }
    if rest != "/" {
        return json_reply(StatusCode::NOT_FOUND, &json!({"error": format!("nothing at {path}")}), &[]);
    }
    if method != Method::POST {
        return json_reply(StatusCode::METHOD_NOT_ALLOWED, &json!({"error": "A2A requests are sent with POST"}), &[("allow", "POST".into())]);
    }
    let principal = match server.authenticate(&headers).await {
        Ok(p) => p,
        Err(e) => return refusal(&e),
    };
    let limit = server.inner.max_body_bytes;
    let Ok(bytes) = axum::body::to_bytes(body, limit).await else {
        return refusal(&A2AError::http(format!("the request is larger than {limit} bytes"), wire::INVALID_REQUEST, 413));
    };
    let Ok(payload) = serde_json::from_slice::<Value>(&bytes) else {
        return json_reply(
            StatusCode::OK,
            &json!({"jsonrpc": "2.0", "id": null, "error": {"code": wire::PARSE_ERROR, "message": "the body is not JSON"}}),
            &[],
        );
    };
    let name = mount.name.clone();
    let events = match server.handle(payload, Some(&principal), Some(&name)).await {
        Ok(Reply::Json(answer)) => return json_reply(StatusCode::OK, &answer, &[]),
        Ok(Reply::Stream(events)) => events,
        Err(e) => return refusal(&e),
    };
    // Events as they come; when the caller hangs up the body is dropped, and
    // the task carries on — it can be asked about, or re-subscribed to.
    let keepalive = Duration::from_secs_f64(server.inner.keepalive.max(0.05));
    let stream = async_stream::stream! {
        let mut events = events;
        loop {
            tokio::select! {
                next = events.next() => match next {
                    Some(event) => yield Ok::<Bytes, Infallible>(Bytes::from(format!("data: {event}\n\n"))),
                    None => break,
                },
                // Nothing to say yet: a comment keeps a proxy from closing it.
                _ = tokio::time::sleep(keepalive) => yield Ok(Bytes::from_static(b": keep-alive\n\n")),
            }
        }
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("x-accel-buffering", "no")
        .body(Body::from_stream(stream))
        .expect("valid response")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webhooks_only_go_out_to_public_https() {
        assert!(public_https("https://hooks.example.com/a2a"));
        assert!(public_https("https://8.8.8.8/x"));
        for url in [
            "http://hooks.example.com",
            "https://localhost/x",
            "https://10.0.0.8/x",
            "https://127.0.0.1/x",
            "https://db.internal/x",
            "https://printer.local/x",
            "https://[::1]/x",
            "https://[fd00::1]/x",
            "https://100.64.0.1/x",
            "not a url",
        ] {
            assert!(!public_https(url), "{url}");
        }
    }

    #[test]
    fn principals_and_helpers() {
        let p = Principal::of(&json!({"user_id": "ada", "tenant_id": "acme", "plan": "gold"}));
        assert_eq!(p.key(), "acme/ada");
        assert_eq!(p.claims["plan"], "gold");
        assert_eq!(Principal::of(&json!("bob")), Principal::user("bob"));
        assert_eq!(Principal::default().key(), "/");
        assert!(same(b"sk-1", b"sk-1") && !same(b"sk-1", b"sk-2") && !same(b"sk", b"sk-1"));
        assert_eq!(seconds(2.0), "2");
        assert_eq!(seconds(0.05), "0.05");
        let part = json!({"kind": "file", "file": {"name": "n", "bytes": "AA=="}});
        assert_eq!(without_bytes(&part), json!({"kind": "file", "file": {"name": "n"}}));
        assert_eq!(py_s(Some(&json!(7))), "7");
    }
}
