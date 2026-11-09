//! Expose agents as an MCP server, so any MCP client can use them as tools.
//!
//! ```no_run
//! use agent_harness::mcp::McpAgentServer;
//! # async fn demo(billing: agent_harness::Agent, orders: agent_harness::Agent) -> agent_harness::Result<()> {
//! let server = McpAgentServer::builder([billing, orders])
//!     .agent_key("billing", "sk-billing-0001")
//!     .agent_key("orders", "sk-orders-0001")
//!     .build()?;
//! server.serve("127.0.0.1:8000").await?;   // streamable HTTP at /mcp and /<agent>/mcp
//! # Ok(()) }
//! ```
//!
//! Each agent is one tool named after it: the client sends a `task`, the agent
//! runs with its own model, tools and budget, and its answer comes back.
//!
//! **Every agent can have its own key.** A key opens only the agents it was
//! issued for; to a caller holding the billing key, the orders agent does not
//! exist. `api_key` opens all of them. Keys are sent as `Authorization: Bearer`
//! or `X-API-Key`, and compared in constant time. The HTTP transport is
//! stateless, so any replica answers any request. [`McpAgentServer::router`]
//! hands back the `axum` router to mount in a server of your own.
//!
//! Not yet ported: a separate agent instance per caller (Python's `isolate`,
//! which needs the A2A principal from M5) and the audit trail (M4).

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use futures::channel::mpsc;
use indexmap::IndexMap;
use regex::Regex;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{Notify, Semaphore};

use crate::agent::{Agent, RunOptions, SessionRef};
use crate::errors::{Error, Result};
use crate::runtime::Session;
use crate::tools::fnmatch;
use crate::types::{RunResult, StreamEventType};

/// The MCP revisions this server speaks, newest first.
pub const PROTOCOL_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;
pub const SERVER_BUSY: i64 = -32000;

const HIDDEN_TAGS: [&str; 4] = ["delegation", "memory", "mode", "handoff"];

/// One API key, and whose calls it makes.
#[derive(Debug, Clone)]
pub struct ApiKey {
    pub key: String,
    pub user_id: Option<String>,
    pub tenant_id: Option<String>,
}

impl From<&str> for ApiKey {
    fn from(key: &str) -> Self {
        Self { key: key.into(), user_id: None, tenant_id: None }
    }
}

impl From<String> for ApiKey {
    fn from(key: String) -> Self {
        Self { key, user_id: None, tenant_id: None }
    }
}

/// Who a custom `auth` function says is calling.
#[derive(Debug, Clone, Default)]
pub struct Claims {
    pub user_id: Option<String>,
    pub tenant_id: Option<String>,
    /// The agents they may use; `None` for all.
    pub agents: Option<Vec<String>>,
}

/// From request headers to who is calling; `None` refuses (401).
pub type AuthFn = Arc<dyn Fn(&HeaderMap) -> Pin<Box<dyn Future<Output = Option<Claims>> + Send>> + Send + Sync>;

/// Also serve the agents' own tools?
#[derive(Debug, Clone, Default)]
pub enum ExposeTools {
    #[default]
    None,
    All,
    /// Globs of tool names.
    Matching(Vec<String>),
}

#[derive(Debug, Clone)]
struct Key {
    id: String,
    /// `None`: every agent.
    agents: Option<HashSet<String>>,
    user_id: Option<String>,
    tenant_id: Option<String>,
}

/// What one request may reach.
#[derive(Debug, Clone, Default)]
pub struct Access {
    pub agents: Vec<String>,
    pub key_id: String,
    pub user_id: Option<String>,
    pub tenant_id: Option<String>,
}

impl Access {
    fn owner(&self) -> String {
        format!("{}/{}", self.tenant_id.as_deref().unwrap_or(""), self.user_id.as_deref().unwrap_or(""))
    }
}

#[derive(Debug, Clone)]
struct Exposed {
    name: String,
    agent: String,
    /// "" — the agent itself.
    tool: String,
    definition: Value,
}

struct RpcError(i64, String);

/// A progress note to send while a call runs.
type Notifier = Arc<dyn Fn(Value) + Send + Sync>;

fn sha_hex(text: &str) -> String {
    Sha256::digest(text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

fn key_id(key: &str) -> String {
    format!("key_{}", &sha_hex(key)[..12])
}

/// Constant time in the contents: says nothing about which byte differed.
fn same(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

pub(crate) fn slug(name: &str) -> String {
    let re = Regex::new(r"[^A-Za-z0-9_-]+").expect("valid regex");
    let s = re.replace_all(name, "_").trim_matches('_').to_string();
    if s.is_empty() { "agent".into() } else { s }
}

fn text(text: &str, error: bool) -> Value {
    json!({"content": [{"type": "text", "text": text}], "isError": error})
}

fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn seconds(s: f64) -> String {
    if s.fract() == 0.0 { format!("{}", s as i64) } else { format!("{s}") }
}

/// Builds an [`McpAgentServer`].
pub struct McpServerBuilder {
    agents: IndexMap<String, Agent>,
    api_keys: Vec<ApiKey>,
    agent_keys: Vec<(String, ApiKey)>,
    public: Vec<String>,
    keys_configured: bool,
    auth: Option<AuthFn>,
    expose_tools: ExposeTools,
    conversations: bool,
    name: String,
    version: String,
    instructions: Option<String>,
    max_concurrency: usize,
    max_queue: usize,
    timeout: f64,
    allowed_origins: Option<HashSet<String>>,
    max_body_bytes: usize,
    keepalive: f64,
}

impl McpServerBuilder {
    /// A key that opens every agent.
    pub fn api_key(mut self, key: impl Into<ApiKey>) -> Self {
        self.api_keys.push(key.into());
        self.keys_configured = true;
        self
    }

    /// A key that opens only `agent`.
    pub fn agent_key(mut self, agent: &str, key: impl Into<ApiKey>) -> Self {
        self.agent_keys.push((agent.into(), key.into()));
        self.keys_configured = true;
        self
    }

    /// Leave `agent` open to callers with no key at all.
    pub fn public(mut self, agent: &str) -> Self {
        self.public.push(agent.into());
        self.keys_configured = true;
        self
    }

    /// Decide who is calling yourself, instead of keys.
    pub fn auth<F, Fut>(mut self, f: F) -> Self
    where
        F: Fn(&HeaderMap) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Option<Claims>> + Send + 'static,
    {
        self.auth = Some(Arc::new(move |h| Box::pin(f(h))));
        self
    }

    pub fn expose_tools(mut self, expose: ExposeTools) -> Self {
        self.expose_tools = expose;
        self
    }

    /// Let a caller pass `conversation_id` to carry a conversation on.
    pub fn conversations(mut self, on: bool) -> Self {
        self.conversations = on;
        self
    }

    pub fn name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }

    pub fn version(mut self, version: &str) -> Self {
        self.version = version.into();
        self
    }

    pub fn instructions(mut self, text: &str) -> Self {
        self.instructions = Some(text.into());
        self
    }

    pub fn max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self
    }

    pub fn max_queue(mut self, n: usize) -> Self {
        self.max_queue = n;
        self
    }

    /// The longest one call may run, in seconds.
    pub fn timeout(mut self, seconds: f64) -> Self {
        self.timeout = seconds;
        self
    }

    /// Refuse browser requests from any other `Origin`.
    pub fn allowed_origins<I: IntoIterator<Item = S>, S: Into<String>>(mut self, origins: I) -> Self {
        self.allowed_origins = Some(origins.into_iter().map(Into::into).collect());
        self
    }

    pub fn max_body_bytes(mut self, n: usize) -> Self {
        self.max_body_bytes = n;
        self
    }

    pub fn keepalive(mut self, seconds: f64) -> Self {
        self.keepalive = seconds;
        self
    }

    pub fn build(self) -> Result<McpAgentServer> {
        if self.agents.is_empty() {
            return Err(Error::config("an MCP server needs at least one agent"));
        }
        let inner = ServerInner {
            agents: self.agents,
            name: self.name,
            version: self.version,
            instructions: self.instructions,
            conversations: self.conversations,
            timeout: self.timeout,
            max_queue: self.max_queue,
            max_body_bytes: self.max_body_bytes,
            keepalive: self.keepalive,
            allowed_origins: self.allowed_origins,
            auth: self.auth.clone(),
            keys: RwLock::new(HashMap::new()),
            public: self.public.iter().cloned().collect(),
            open: AtomicBool::new(!self.keys_configured && self.auth.is_none()),
            tools: IndexMap::new(),
            gate: Semaphore::new(self.max_concurrency),
            waiting: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            calls: Mutex::new(HashMap::new()),
            closing: AtomicBool::new(false),
            stats: [("calls", 0), ("errors", 0), ("refused", 0), ("busy", 0)].into_iter().map(|(k, v)| (k, AtomicU64::new(v))).collect(),
        };
        let mut server = McpAgentServer { inner: Arc::new(inner) };
        for name in &self.public {
            if !server.inner.agents.contains_key(name) {
                return Err(Error::config(format!("api_keys names '{name}', which is not served; the agents are: {}", server.agent_names().join(", "))));
            }
        }
        for key in self.api_keys {
            server.add_key(key, None)?;
        }
        for (agent, key) in self.agent_keys {
            if !server.inner.agents.contains_key(&agent) {
                return Err(Error::config(format!("api_keys names '{agent}', which is not served; the agents are: {}", server.agent_names().join(", "))));
            }
            server.add_key(key, Some(&agent))?;
        }
        if !server.inner.open.load(Ordering::SeqCst) && server.inner.auth.is_none() {
            let keys = server.inner.keys.read().unwrap_or_else(|p| p.into_inner());
            let master = keys.values().any(|k| k.agents.is_none());
            let locked: Vec<&String> = server
                .inner
                .agents
                .keys()
                .filter(|n| !server.inner.public.contains(*n) && !master && !keys.values().any(|k| k.agents.as_ref().is_some_and(|a| a.contains(*n))))
                .collect();
            if !locked.is_empty() {
                let names: Vec<&str> = locked.iter().map(|s| s.as_str()).collect();
                return Err(Error::config(format!(
                    "{} would be served with no key that opens it. Give each a key with agent_key(), pass api_key() for one that opens every agent, or mark it public()",
                    names.join(", ")
                )));
            }
        }
        // ---- tools ----------------------------------------------------------
        let inner = Arc::get_mut(&mut server.inner).expect("not yet shared");
        let mut tools: IndexMap<String, Exposed> = IndexMap::new();
        for (agent_name, agent) in &inner.agents {
            let name = unique(&tools, &slug(agent_name));
            tools.insert(name.clone(), expose_agent(&name, agent_name, agent, inner.conversations));
            let patterns = match &self.expose_tools {
                ExposeTools::None => continue,
                ExposeTools::All => None,
                ExposeTools::Matching(p) => Some(p),
            };
            for tool in agent.tools().iter() {
                if tool.tags.iter().any(|t| HIDDEN_TAGS.contains(&t.as_str())) {
                    continue;
                }
                if patterns.is_some_and(|ps| !ps.iter().any(|p| fnmatch(&tool.name, p))) {
                    continue;
                }
                let name = unique(&tools, &slug(&format!("{agent_name}_{}", tool.name)));
                let params = if tool.parameters.is_object() { tool.parameters.clone() } else { json!({"type": "object", "properties": {}}) };
                let definition = json!({"name": name, "description": tool.description, "inputSchema": params});
                tools.insert(name.clone(), Exposed { name, agent: agent_name.clone(), tool: tool.name.clone(), definition });
            }
        }
        inner.tools = tools;
        Ok(server)
    }
}

fn unique(tools: &IndexMap<String, Exposed>, name: &str) -> String {
    let cut = |s: &str, n: usize| s.chars().take(n).collect::<String>();
    let mut candidate = cut(name, 64);
    let mut n = 2;
    while tools.contains_key(&candidate) {
        let suffix = format!("_{n}");
        candidate = format!("{}{suffix}", cut(name, 64 - suffix.len()));
        n += 1;
    }
    candidate
}

fn expose_agent(name: &str, agent_name: &str, agent: &Agent, conversations: bool) -> Exposed {
    let mut properties = json!({"task": {
        "type": "string",
        "description": "What you want done, stated in full. The agent does not see your conversation — include everything it needs.",
    }});
    if conversations {
        properties["conversation_id"] = json!({
            "type": "string",
            "description": "Optional. Any id of your choosing: calls that share it are one conversation, and the agent remembers the earlier ones.",
        });
    }
    let said = format!("Ask the {agent_name} agent. {}. It runs the task with its own tools and answers in text.", agent.description().trim_end_matches('.'));
    Exposed {
        name: name.into(),
        agent: agent_name.into(),
        tool: String::new(),
        definition: json!({"name": name, "title": agent_name, "description": said,
                           "inputSchema": {"type": "object", "properties": properties, "required": ["task"]}}),
    }
}

struct ServerInner {
    agents: IndexMap<String, Agent>,
    name: String,
    version: String,
    instructions: Option<String>,
    conversations: bool,
    timeout: f64,
    max_queue: usize,
    max_body_bytes: usize,
    keepalive: f64,
    allowed_origins: Option<HashSet<String>>,
    auth: Option<AuthFn>,
    keys: RwLock<HashMap<String, Key>>,
    public: HashSet<String>,
    open: AtomicBool,
    tools: IndexMap<String, Exposed>,
    gate: Semaphore,
    waiting: AtomicUsize,
    running: AtomicUsize,
    calls: Mutex<HashMap<(String, String), Arc<Notify>>>,
    closing: AtomicBool,
    stats: HashMap<&'static str, AtomicU64>,
}

/// Agents, served as MCP tools. Cheap to clone.
#[derive(Clone)]
pub struct McpAgentServer {
    inner: Arc<ServerInner>,
}

impl std::fmt::Debug for McpAgentServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpAgentServer").field("name", &self.inner.name).field("agents", &self.agent_names()).finish()
    }
}

impl McpAgentServer {
    pub fn builder<I: IntoIterator<Item = Agent>>(agents: I) -> McpServerBuilder {
        McpServerBuilder {
            agents: agents.into_iter().map(|a| (a.name().to_string(), a)).collect(),
            api_keys: vec![],
            agent_keys: vec![],
            public: vec![],
            keys_configured: false,
            auth: None,
            expose_tools: ExposeTools::None,
            conversations: true,
            name: "agent-harness".into(),
            version: "1.0.0".into(),
            instructions: None,
            max_concurrency: 32,
            max_queue: 128,
            timeout: 600.0,
            allowed_origins: None,
            max_body_bytes: 4_000_000,
            keepalive: 15.0,
        }
    }

    pub fn agent_names(&self) -> Vec<String> {
        self.inner.agents.keys().cloned().collect()
    }

    fn bump(&self, stat: &str) {
        if let Some(s) = self.inner.stats.get(stat) {
            s.fetch_add(1, Ordering::SeqCst);
        }
    }

    // ---- keys ---------------------------------------------------------
    /// Issue a key: for one agent, or (with none) for all. Returns the id it
    /// is logged under — never the key. Safe to call while serving.
    pub fn add_key(&self, key: impl Into<ApiKey>, agent: Option<&str>) -> Result<String> {
        let key = key.into();
        if key.key.chars().count() < 8 {
            let for_agent = agent.map(|a| format!(" (for {a})")).unwrap_or_default();
            return Err(Error::config(format!("an API key must be a string of at least 8 characters{for_agent}")));
        }
        if let Some(a) = agent
            && !self.inner.agents.contains_key(a)
        {
            return Err(Error::config(format!("no agent '{a}' is served")));
        }
        let mut keys = self.inner.keys.write().unwrap_or_else(|p| p.into_inner());
        let held = keys.entry(key.key.clone()).or_insert_with(|| Key {
            id: key_id(&key.key),
            agents: agent.map(|_| HashSet::new()),
            user_id: key.user_id.clone(),
            tenant_id: key.tenant_id.clone(),
        });
        match agent {
            None => held.agents = None,
            Some(a) => {
                if let Some(set) = &mut held.agents {
                    set.insert(a.into());
                }
            }
        }
        self.inner.open.store(false, Ordering::SeqCst);
        Ok(held.id.clone())
    }

    /// Stop a key working, now. True if there was such a key.
    pub fn revoke_key(&self, key: &str) -> bool {
        self.inner.keys.write().unwrap_or_else(|p| p.into_inner()).remove(key).is_some()
    }

    /// Which agents this request may reach, or a refusal (status, message).
    async fn who(&self, headers: &HeaderMap) -> std::result::Result<Access, (StatusCode, String)> {
        let everything = self.agent_names();
        if let Some(auth) = &self.inner.auth {
            let Some(claims) = auth(headers).await else { return Err((StatusCode::UNAUTHORIZED, "not authorised".into())) };
            let agents = everything.into_iter().filter(|a| claims.agents.as_ref().is_none_or(|allowed| allowed.contains(a))).collect();
            return Ok(Access { agents, key_id: "auth".into(), user_id: claims.user_id, tenant_id: claims.tenant_id });
        }
        if self.inner.open.load(Ordering::SeqCst) {
            return Ok(Access { agents: everything, ..Default::default() });
        }
        let get = |h: &str| headers.get(h).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
        let authz = get("authorization");
        let (scheme, bearer) = authz.split_once(' ').unwrap_or((authz.as_str(), ""));
        let mut offered = if scheme.eq_ignore_ascii_case("bearer") { bearer.trim().to_string() } else { String::new() };
        if offered.is_empty() {
            offered = get("x-api-key").trim().to_string();
        }
        if offered.is_empty() {
            if !self.inner.public.is_empty() {
                return Ok(Access { agents: everything.into_iter().filter(|a| self.inner.public.contains(a)).collect(), ..Default::default() });
            }
            return Err((StatusCode::UNAUTHORIZED, "an API key is required".into()));
        }
        let keys = self.inner.keys.read().unwrap_or_else(|p| p.into_inner());
        let mut matched: Option<Key> = None;
        // Every key is compared, so the time taken says nothing about which nearly matched.
        for (secret, key) in keys.iter() {
            if same(secret.as_bytes(), offered.as_bytes()) {
                matched = Some(key.clone());
            }
        }
        let Some(k) = matched else { return Err((StatusCode::UNAUTHORIZED, "that API key is not valid".into())) };
        let agents = everything.into_iter().filter(|a| k.agents.as_ref().is_none_or(|set| set.contains(a)) || self.inner.public.contains(a)).collect();
        Ok(Access { agents, key_id: k.id.clone(), user_id: Some(k.user_id.clone().unwrap_or(k.id)), tenant_id: k.tenant_id })
    }

    /// The tool definitions a caller who may reach `agents` is shown.
    pub fn tools(&self, agents: Option<&[String]>) -> Vec<Value> {
        self.inner.tools.values().filter(|t| agents.is_none_or(|a| a.contains(&t.agent))).map(|t| t.definition.clone()).collect()
    }

    // ---- JSON-RPC -------------------------------------------------------
    /// Answer one JSON-RPC message, as a caller who may reach every agent.
    /// `None` for a notification. The protocol with no transport around it.
    pub async fn handle(&self, payload: Value) -> Option<Value> {
        let access = Access { agents: self.agent_names(), ..Default::default() };
        self.handle_as(payload, &access, None).await
    }

    async fn handle_as(&self, payload: Value, access: &Access, notify: Option<Notifier>) -> Option<Value> {
        if payload.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Some(rpc_error(Value::Null, INVALID_REQUEST, "not a JSON-RPC 2.0 message"));
        }
        let method = payload.get("method").and_then(Value::as_str)?.to_string(); // a response to something we never asked
        let params = payload.get("params").cloned().unwrap_or_else(|| json!({}));
        let Some(id) = payload.get("id").cloned() else {
            if method == "notifications/cancelled"
                && let Some(rid) = params.get("requestId")
            {
                let key = (access.owner(), rid.to_string());
                if let Some(n) = self.inner.calls.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
                    n.notify_one();
                }
            }
            return None;
        };
        let result: std::result::Result<Value, RpcError> = async {
            let Value::Object(params) = params else { return Err(RpcError(INVALID_PARAMS, "params must be an object".into())) };
            match method.as_str() {
                "initialize" => Ok(self.initialize(&params)),
                "ping" => Ok(json!({})),
                "tools/list" => Ok(json!({"tools": self.tools(Some(&access.agents))})),
                "tools/call" => self.tracked(&id, &params, access, notify).await,
                "resources/list" => Ok(json!({"resources": []})),
                "prompts/list" => Ok(json!({"prompts": []})),
                "resources/templates/list" => Ok(json!({"resourceTemplates": []})),
                other => Err(RpcError(METHOD_NOT_FOUND, format!("method '{other}' is not supported"))),
            }
        }
        .await;
        Some(match result {
            Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
            Err(RpcError(code, message)) => rpc_error(id, code, &message),
        })
    }

    fn initialize(&self, params: &Map<String, Value>) -> Value {
        let asked = params.get("protocolVersion").and_then(Value::as_str).unwrap_or("");
        let version = if PROTOCOL_VERSIONS.contains(&asked) { asked } else { PROTOCOL_VERSIONS[0] };
        let mut result = json!({"protocolVersion": version, "capabilities": {"tools": {"listChanged": false}},
                                "serverInfo": {"name": self.inner.name, "version": self.inner.version}});
        if let Some(i) = &self.inner.instructions {
            result["instructions"] = json!(i);
        }
        result
    }

    /// A call that the `notifications/cancelled` naming it can stop.
    async fn tracked(&self, id: &Value, params: &Map<String, Value>, access: &Access, notify: Option<Notifier>) -> std::result::Result<Value, RpcError> {
        let key = (access.owner(), id.to_string());
        let stop = Arc::new(Notify::new());
        self.inner.calls.lock().unwrap_or_else(|p| p.into_inner()).insert(key.clone(), stop.clone());
        let outcome = tokio::select! {
            r = self.call(params, access, notify) => r,
            _ = stop.notified() => Ok(text("Cancelled.", true)),
        };
        let mut calls = self.inner.calls.lock().unwrap_or_else(|p| p.into_inner());
        if calls.get(&key).is_some_and(|n| Arc::ptr_eq(n, &stop)) {
            calls.remove(&key);
        }
        outcome
    }

    async fn call(&self, params: &Map<String, Value>, access: &Access, notify: Option<Notifier>) -> std::result::Result<Value, RpcError> {
        let name = params.get("name").and_then(Value::as_str);
        // A tool the caller's key does not open is, to them, not there.
        let exposed = name.and_then(|n| self.inner.tools.get(n)).filter(|e| access.agents.contains(&e.agent)).cloned();
        let Some(exposed) = exposed else {
            let shown = name.map(|n| format!("'{n}'")).unwrap_or_else(|| "None".into());
            return Err(RpcError(INVALID_PARAMS, format!("Unknown tool: {shown}")));
        };
        let arguments = match params.get("arguments") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(a)) => a.clone(),
            Some(_) => return Err(RpcError(INVALID_PARAMS, "arguments must be an object".into())),
        };
        if self.inner.closing.load(Ordering::SeqCst) {
            return Err(RpcError(SERVER_BUSY, "the server is shutting down".into()));
        }
        if self.inner.gate.available_permits() == 0 && self.inner.waiting.load(Ordering::SeqCst) >= self.inner.max_queue {
            self.bump("busy");
            return Err(RpcError(SERVER_BUSY, "the server is at capacity; try again shortly".into()));
        }
        let token = params.get("_meta").and_then(|m| m.get("progressToken")).filter(|t| !t.is_null()).cloned();
        self.bump("calls");
        let agent = self.inner.agents[&exposed.agent].clone();
        self.inner.waiting.fetch_add(1, Ordering::SeqCst);
        let permit = self.inner.gate.acquire().await;
        self.inner.waiting.fetch_sub(1, Ordering::SeqCst);
        let _permit = permit.map_err(|_| RpcError(SERVER_BUSY, "the server is shutting down".into()))?;
        self.inner.running.fetch_add(1, Ordering::SeqCst);
        let limit = Duration::from_secs_f64(self.inner.timeout);
        let answer = if !exposed.tool.is_empty() {
            match tokio::time::timeout(limit, agent.call_tool(&exposed.tool, arguments)).await {
                Ok(outcome) => Ok(text(&outcome.content, outcome.is_error)),
                Err(_) => Ok(text(&format!("{} ran longer than {}s and was stopped.", exposed.name, seconds(self.inner.timeout)), true)),
            }
        } else {
            match tokio::time::timeout(limit, self.ask(&exposed, &agent, &arguments, access, token, notify)).await {
                Ok(Ok(a)) => Ok(a),
                Ok(Err(AskError::Rpc(e))) => Err(e),
                Ok(Err(AskError::Run(e))) => Ok(text(&e.to_string(), true)),
                Err(_) => Ok(text(&format!("{} ran longer than {}s and was stopped.", exposed.name, seconds(self.inner.timeout)), true)),
            }
        };
        self.inner.running.fetch_sub(1, Ordering::SeqCst);
        if let Ok(a) = &answer
            && a["isError"] == true
        {
            self.bump("errors");
        }
        answer
    }

    /// Run the agent on one task and hand back what it said.
    async fn ask(
        &self,
        exposed: &Exposed,
        agent: &Agent,
        arguments: &Map<String, Value>,
        access: &Access,
        token: Option<Value>,
        notify: Option<Notifier>,
    ) -> std::result::Result<Value, AskError> {
        let task = arguments.get("task").and_then(Value::as_str).filter(|t| !t.trim().is_empty());
        let Some(task) = task else { return Err(AskError::Rpc(RpcError(INVALID_PARAMS, "`task` is required: what should the agent do?".into()))) };
        let conversation = if self.inner.conversations { arguments.get("conversation_id") } else { None };
        let conversation = match conversation {
            None | Some(Value::Null) => None,
            Some(Value::String(c)) if !c.is_empty() && c.chars().count() <= 128 => Some(c.clone()),
            Some(_) => return Err(AskError::Rpc(RpcError(INVALID_PARAMS, "`conversation_id` is a short string".into()))),
        };
        let options = match &conversation {
            Some(c) => {
                // A session of the agent's own, found by the caller and the id
                // they chose — on whichever replica the last call ran.
                let digest = &sha_hex(&format!("{}\0{c}", access.owner()))[..24];
                let session_id = format!("mcp_{}_{digest}", slug(&exposed.agent));
                let session = match agent.harness().sessions.load(&session_id).await {
                    Ok(s) => s,
                    Err(_) => Session { id: session_id, ..Session::new(agent.name()) },
                };
                RunOptions { session: Some(SessionRef::Session(Box::new(session))), ..Default::default() }
            }
            None => RunOptions { messages: Some(vec![]), ..Default::default() },
        };
        let mut events = agent.stream(task, options).await.map_err(AskError::Run)?;
        let mut result: Option<RunResult> = None;
        let mut done = 0u64;
        while let Some(event) = events.next().await {
            if event.kind == StreamEventType::RunEnd {
                result = event.data.get("result").and_then(|r| serde_json::from_value(r.clone()).ok());
                continue;
            }
            let said = match event.kind {
                StreamEventType::StepStart => format!("step {}", event.step),
                StreamEventType::ToolCall => format!("using {}", event.text),
                StreamEventType::Handoff => format!("handed to {}", event.text),
                _ => continue,
            };
            if let (Some(token), Some(notify)) = (&token, &notify) {
                done += 1;
                notify(json!({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progressToken": token, "progress": done, "message": said}}));
            }
        }
        let Some(result) = result else { return Err(AskError::Run(Error::config("the agent produced no result"))) };
        let mut answer = match &result.error {
            Some(e) => text(e, true),
            None => {
                let mut a = text(&result.output, false);
                if result.data.is_object() {
                    a["structuredContent"] = result.data.clone();
                }
                for produced in &result.artifacts {
                    if !produced.content.is_empty() && produced.content.chars().count() <= 200_000 {
                        let mime = if produced.media_type.is_empty() { "text/plain" } else { &produced.media_type };
                        a["content"].as_array_mut().expect("array").push(json!({"type": "resource", "resource": {
                            "uri": format!("artifact:///{}", produced.name), "mimeType": mime, "text": produced.content}}));
                    }
                }
                a
            }
        };
        let mut meta = json!({"agent": result.agent, "steps": result.steps});
        if let Some(c) = conversation {
            meta["conversation_id"] = json!(c);
        }
        answer["_meta"] = meta;
        Ok(answer)
    }

    // ---- health, and going away ---------------------------------------
    pub fn health(&self) -> Value {
        let stat = |k: &str| self.inner.stats.get(k).map_or(0, |s| s.load(Ordering::SeqCst));
        let mut agents = self.agent_names();
        agents.sort();
        json!({
            "status": if self.inner.closing.load(Ordering::SeqCst) { "draining" } else { "ok" },
            "agents": agents, "tools": self.inner.tools.len(),
            "running": self.inner.running.load(Ordering::SeqCst), "waiting": self.inner.waiting.load(Ordering::SeqCst),
            "keys": self.inner.keys.read().unwrap_or_else(|p| p.into_inner()).len(),
            "calls": stat("calls"), "errors": stat("errors"), "refused": stat("refused"), "busy": stat("busy"),
        })
    }

    /// Stop taking calls, and cancel the ones running.
    pub fn close(&self) {
        self.inner.closing.store(true, Ordering::SeqCst);
        for n in self.inner.calls.lock().unwrap_or_else(|p| p.into_inner()).values() {
            n.notify_one();
        }
    }

    // ---- HTTP: streamable, stateless ------------------------------------
    /// The `axum` router: `POST /mcp`, `POST /<agent>/mcp`, `GET /healthz`.
    pub fn router(&self) -> Router {
        Router::new().fallback(http_entry).with_state(self.clone())
    }

    /// Serve on `addr` until the process ends. Enough to develop against;
    /// for production mount [`Self::router`] in your own server.
    pub async fn serve(&self, addr: &str) -> Result<()> {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        self.serve_on(listener).await
    }

    /// Serve on a listener you bound (port 0 for any free one).
    pub async fn serve_on(&self, listener: tokio::net::TcpListener) -> Result<()> {
        let served = axum::serve(listener, self.router()).await;
        self.close();
        served.map_err(|e| Error::Io(e.to_string()))
    }

    /// Bind any free port on localhost and serve in the background.
    /// Returns the URL of the MCP endpoint.
    pub async fn spawn_local(&self) -> Result<(String, tokio::task::JoinHandle<()>)> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr: SocketAddr = listener.local_addr()?;
        let server = self.clone();
        let handle = tokio::spawn(async move {
            let _ = server.serve_on(listener).await;
        });
        Ok((format!("http://{addr}/mcp"), handle))
    }

    fn scope_of(&self, path: &str) -> Option<String> {
        let path = path.trim_end_matches('/');
        let path = if path.is_empty() { "/" } else { path };
        if path == "/mcp" || path == "/" {
            return Some(String::new());
        }
        let name = path.strip_suffix("/mcp")?.strip_prefix('/')?;
        self.inner.agents.contains_key(name).then(|| name.to_string())
    }

    // ---- stdio: for a client that starts the server itself ---------------
    /// Serve one client over a reader and a writer, a JSON message to a line.
    /// The client owns both ends, so there is no key to check. `agents`
    /// narrows what it is offered.
    pub async fn serve_stdio<R, W>(&self, reader: R, writer: W, agents: Option<Vec<String>>) -> Result<()>
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let access = Access {
            agents: self.agent_names().into_iter().filter(|a| agents.as_ref().is_none_or(|only| only.contains(a))).collect(),
            key_id: "stdio".into(),
            ..Default::default()
        };
        let (tx, mut rx) = mpsc::unbounded::<Value>();
        let mut writer = writer;
        let writing = tokio::spawn(async move {
            while let Some(message) = rx.next().await {
                let mut line = message.to_string();
                line.push('\n');
                if writer.write_all(line.as_bytes()).await.is_err() || writer.flush().await.is_err() {
                    break;
                }
            }
        });
        let mut running = tokio::task::JoinSet::new();
        let mut lines = BufReader::new(reader).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if line.trim().is_empty() {
                continue;
            }
            let Ok(payload) = serde_json::from_str::<Value>(&line) else {
                let _ = tx.unbounded_send(rpc_error(Value::Null, PARSE_ERROR, "the line is not JSON"));
                continue;
            };
            // Each on its own, so a long task does not hold up a ping.
            let (server, access, tx) = (self.clone(), access.clone(), tx.clone());
            running.spawn(async move {
                let notes = tx.clone();
                let notify: Notifier = Arc::new(move |m| {
                    let _ = notes.unbounded_send(m);
                });
                if let Some(answer) = server.handle_as(payload, &access, Some(notify)).await {
                    let _ = tx.unbounded_send(answer);
                }
            });
        }
        running.abort_all();
        while running.join_next().await.is_some() {}
        drop(tx);
        let _ = writing.await;
        self.close();
        Ok(())
    }
}

enum AskError {
    Rpc(RpcError),
    Run(Error),
}

fn json_reply(status: StatusCode, body: Option<Value>) -> Response {
    let data = body.map(|b| b.to_string()).unwrap_or_default();
    (status, [(header::CONTENT_TYPE, "application/json")], data).into_response()
}

fn sse(message: &Value) -> Bytes {
    Bytes::from(format!("data: {message}\n\n"))
}

/// Aborts the spawned work if the client hangs up mid-stream.
struct AbortOnDrop(tokio::task::JoinHandle<Vec<Value>>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

async fn http_entry(State(server): State<McpAgentServer>, method: Method, uri: Uri, headers: HeaderMap, body: Body) -> Response {
    let path = uri.path().to_string();
    if method == Method::GET && path.trim_end_matches('/') == "/healthz" {
        let status = if server.inner.closing.load(Ordering::SeqCst) { StatusCode::SERVICE_UNAVAILABLE } else { StatusCode::OK };
        return json_reply(status, Some(server.health()));
    }
    let Some(only) = server.scope_of(&path) else {
        return json_reply(StatusCode::NOT_FOUND, Some(json!({"error": format!("nothing at {path}")})));
    };
    if method != Method::POST {
        // No stream of the server's own to open, and no session to end.
        let mut r = json_reply(StatusCode::METHOD_NOT_ALLOWED, Some(json!({"error": "this MCP endpoint takes POST"})));
        r.headers_mut().insert(header::ALLOW, "POST".parse().expect("valid header"));
        return r;
    }
    let origin = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok());
    if let (Some(allowed), Some(origin)) = (&server.inner.allowed_origins, origin)
        && !allowed.contains(origin)
    {
        return json_reply(StatusCode::FORBIDDEN, Some(json!({"error": format!("requests from {origin} are not accepted")})));
    }
    let access = match server.who(&headers).await {
        Ok(mut a) => {
            if !only.is_empty() {
                if !a.agents.contains(&only) {
                    server.bump("refused");
                    return json_reply(StatusCode::FORBIDDEN, Some(rpc_error(Value::Null, INVALID_REQUEST, &format!("this key does not open {only}"))));
                }
                a.agents = vec![only.clone()];
            }
            a
        }
        Err((status, message)) => {
            server.bump("refused");
            let mut r = json_reply(status, Some(rpc_error(Value::Null, INVALID_REQUEST, &message)));
            if status == StatusCode::UNAUTHORIZED {
                r.headers_mut().insert(header::WWW_AUTHENTICATE, "Bearer realm=\"mcp\"".parse().expect("valid header"));
            }
            return r;
        }
    };
    let Ok(bytes) = axum::body::to_bytes(body, server.inner.max_body_bytes).await else {
        let limit = server.inner.max_body_bytes;
        return json_reply(StatusCode::PAYLOAD_TOO_LARGE, Some(rpc_error(Value::Null, INVALID_REQUEST, &format!("the request is larger than {limit} bytes"))));
    };
    let Ok(payload) = serde_json::from_slice::<Value>(&bytes) else {
        return json_reply(StatusCode::BAD_REQUEST, Some(rpc_error(Value::Null, PARSE_ERROR, "the body is not JSON")));
    };
    let batch = payload.is_array();
    let messages: Vec<Value> = match payload {
        Value::Array(items) => items,
        one => vec![one],
    };
    let accepts_sse = headers.get(header::ACCEPT).and_then(|v| v.to_str().ok()).is_some_and(|a| a.contains("text/event-stream"));
    let wants_progress = messages
        .iter()
        .any(|m| m.get("method").and_then(Value::as_str) == Some("tools/call") && m["params"]["_meta"].get("progressToken").is_some_and(|t| !t.is_null()));

    if !(accepts_sse && wants_progress) {
        let mut answers = Vec::new();
        for m in messages {
            if let Some(a) = server.handle_as(m, &access, None).await {
                answers.push(a);
            }
        }
        return match (answers.is_empty(), batch) {
            (true, _) => json_reply(StatusCode::ACCEPTED, None), // only notifications
            (false, true) => json_reply(StatusCode::OK, Some(Value::Array(answers))),
            (false, false) => json_reply(StatusCode::OK, answers.into_iter().next()),
        };
    }

    // Progress as it happens, then the answers — one event each.
    let (tx, mut rx) = mpsc::unbounded::<Value>();
    let worker = server.clone();
    let work = tokio::spawn(async move {
        let notify: Notifier = Arc::new(move |m| {
            let _ = tx.unbounded_send(m);
        });
        let mut answers = Vec::new();
        for m in messages {
            if let Some(a) = worker.handle_as(m, &access, Some(notify.clone())).await {
                answers.push(a);
            }
        }
        answers
    });
    let keepalive = Duration::from_secs_f64(server.inner.keepalive.max(0.05));
    let stream = async_stream::stream! {
        let mut work = AbortOnDrop(work);
        loop {
            tokio::select! {
                Some(note) = rx.next() => yield Ok::<Bytes, Infallible>(sse(&note)),
                done = &mut work.0 => {
                    while let Ok(note) = rx.try_recv() {
                        yield Ok(sse(&note));
                    }
                    for answer in done.unwrap_or_default() {
                        yield Ok(sse(&answer));
                    }
                    break;
                }
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
    fn helpers() {
        assert_eq!(slug("billing team!"), "billing_team");
        assert_eq!(slug("!!!"), "agent");
        assert!(key_id("sk-abcdefgh").starts_with("key_"));
        assert_eq!(key_id("sk-abcdefgh").len(), 16);
        assert!(same(b"abc", b"abc"));
        assert!(!same(b"abc", b"abd"));
        assert!(!same(b"abc", b"abcd"));
        assert_eq!(seconds(600.0), "600");
        assert_eq!(seconds(0.5), "0.5");
    }

    #[test]
    fn sha_matches_python() {
        // hashlib.sha256(b"abc").hexdigest()
        assert_eq!(sha_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
