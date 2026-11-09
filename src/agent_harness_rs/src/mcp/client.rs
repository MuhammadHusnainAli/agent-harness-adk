//! MCP client: connect to Model Context Protocol servers and use their tools.
//!
//! ```no_run
//! use agent_harness::mcp::{McpManager, McpServer};
//! use agent_harness::Agent;
//!
//! # async fn demo() -> agent_harness::Result<()> {
//! let files = McpServer::stdio("files", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"]);
//! let mcp = McpManager::new([files]).connect().await;
//! let agent = Agent::builder("assistant").tools(mcp.tools()).build()?;
//! # Ok(()) }
//! ```
//!
//! Both transports: `stdio` (a subprocess speaking line-delimited JSON-RPC)
//! and `http` (streamable HTTP). No SDK required.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;

use crate::errors::{Error, Result};
use crate::tools::{Tool, render_result};

pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// stdio or streamable HTTP.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    #[default]
    Stdio,
    Http,
}

/// How to reach one MCP server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct McpServer {
    pub name: String,
    pub transport: Transport,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub url: Option<String>,
    pub headers: HashMap<String, String>,
    /// Seconds to wait for one reply.
    pub timeout: f64,
    /// Expose tools as `<server>_<tool>`.
    pub prefix: bool,
    pub allowed_tools: Option<Vec<String>>,
}

impl Default for McpServer {
    fn default() -> Self {
        Self {
            name: String::new(),
            transport: Transport::Stdio,
            command: None,
            args: vec![],
            env: HashMap::new(),
            url: None,
            headers: HashMap::new(),
            timeout: 60.0,
            prefix: true,
            allowed_tools: None,
        }
    }
}

impl McpServer {
    /// A server this process starts and talks to over its stdin and stdout.
    pub fn stdio<I: IntoIterator<Item = S>, S: Into<String>>(name: &str, command: &str, args: I) -> Self {
        Self { name: name.into(), command: Some(command.into()), args: args.into_iter().map(Into::into).collect(), ..Default::default() }
    }

    /// A server at a URL, over streamable HTTP.
    pub fn http(name: &str, url: &str) -> Self {
        Self { name: name.into(), transport: Transport::Http, url: Some(url.into()), ..Default::default() }
    }

    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    pub fn env(mut self, name: &str, value: &str) -> Self {
        self.env.insert(name.into(), value.into());
        self
    }

    /// The transport actually used: a URL with no command means HTTP.
    pub fn effective_transport(&self) -> Transport {
        if self.url.is_some() && self.transport == Transport::Stdio && self.command.is_none() { Transport::Http } else { self.transport }
    }
}

struct StdioPipe {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

enum Conn {
    None,
    Stdio(Box<StdioPipe>),
    Http(reqwest::Client),
}

/// One connection, JSON-RPC 2.0 either way. Share it behind an `Arc`.
pub struct McpClient {
    pub server: McpServer,
    conn: Mutex<Conn>,
    session_id: StdMutex<Option<String>>,
    next_id: AtomicU64,
    tools_cache: StdMutex<Vec<Value>>,
    connected: AtomicBool,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpClient").field("server", &self.server.name).field("connected", &self.is_connected()).finish()
    }
}

fn mcp_err(message: impl Into<String>) -> Error {
    Error::Mcp(message.into())
}

impl McpClient {
    pub fn new(server: McpServer) -> Arc<Self> {
        Arc::new(Self {
            server,
            conn: Mutex::new(Conn::None),
            session_id: StdMutex::new(None),
            next_id: AtomicU64::new(0),
            tools_cache: StdMutex::new(vec![]),
            connected: AtomicBool::new(false),
        })
    }

    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    /// Start (or reach) the server and run the MCP handshake.
    pub async fn connect(&self) -> Result<()> {
        if self.is_connected() {
            return Ok(());
        }
        let name = &self.server.name;
        {
            let mut conn = self.conn.lock().await;
            *conn = match self.server.effective_transport() {
                Transport::Stdio => {
                    let command = self.server.command.as_deref().ok_or_else(|| mcp_err(format!("{name}: stdio transport needs a command")))?;
                    let mut child = tokio::process::Command::new(command)
                        .args(&self.server.args)
                        .envs(&self.server.env)
                        .stdin(Stdio::piped())
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .kill_on_drop(true)
                        .spawn()
                        .map_err(|e| mcp_err(format!("{name}: could not start {command}: {e}")))?;
                    let stdin = child.stdin.take().ok_or_else(|| mcp_err("no stdin"))?;
                    let stdout = BufReader::new(child.stdout.take().ok_or_else(|| mcp_err("no stdout"))?);
                    Conn::Stdio(Box::new(StdioPipe { child, stdin, stdout }))
                }
                Transport::Http => {
                    if self.server.url.is_none() {
                        return Err(mcp_err(format!("{name}: http transport needs a url")));
                    }
                    Conn::Http(crate::llm_providers::base::default_client(self.server.timeout, 10.0))
                }
            };
        }
        self.request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "clientInfo": {"name": "agent-harness", "version": crate::VERSION},
            }),
        )
        .await?;
        self.notify("notifications/initialized", json!({})).await?;
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn send_stdio(&self, pipe: &mut StdioPipe, payload: &Value) -> Result<Option<Value>> {
        let name = &self.server.name;
        let mut line = payload.to_string();
        line.push('\n');
        pipe.stdin.write_all(line.as_bytes()).await.map_err(|e| mcp_err(format!("{name}: write failed: {e}")))?;
        pipe.stdin.flush().await.map_err(|e| mcp_err(format!("{name}: write failed: {e}")))?;
        let Some(id) = payload.get("id") else { return Ok(None) };
        let wait = Duration::from_secs_f64(self.server.timeout);
        loop {
            let mut raw = String::new();
            let read = tokio::time::timeout(wait, pipe.stdout.read_line(&mut raw))
                .await
                .map_err(|_| mcp_err(format!("{name}: no reply within {}s", self.server.timeout)))?
                .map_err(|e| mcp_err(format!("{name}: read failed: {e}")))?;
            if read == 0 {
                let mut err = vec![0u8; 2000];
                let n = match pipe.child.stderr.as_mut() {
                    Some(s) => tokio::time::timeout(Duration::from_millis(500), s.read(&mut err)).await.ok().and_then(|r| r.ok()).unwrap_or(0),
                    None => 0,
                };
                let said: String = String::from_utf8_lossy(&err[..n]).chars().take(500).collect();
                return Err(mcp_err(format!("{name} closed the connection: {said}")));
            }
            // Servers sometimes log to stdout; skip the noise.
            let Ok(message) = serde_json::from_str::<Value>(raw.trim()) else { continue };
            if message.get("id") == Some(id) {
                return Ok(Some(message));
            }
            // Anything else is a notification or a server-initiated request.
        }
    }

    async fn send_http(&self, http: &reqwest::Client, payload: &Value) -> Result<Option<Value>> {
        let name = &self.server.name;
        let url = self.server.url.as_deref().unwrap_or_default();
        let mut req = http.post(url).header("content-type", "application/json").header("accept", "application/json, text/event-stream");
        for (k, v) in &self.server.headers {
            req = req.header(k, v);
        }
        let sid = self.session_id.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if let Some(sid) = sid {
            req = req.header("mcp-session-id", sid);
        }
        let resp = tokio::time::timeout(Duration::from_secs_f64(self.server.timeout), req.body(payload.to_string()).send())
            .await
            .map_err(|_| mcp_err(format!("{name}: no reply within {}s", self.server.timeout)))?
            .map_err(|e| mcp_err(format!("{name}: {e}")))?;
        let status = resp.status().as_u16();
        if let Some(sid) = resp.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
            *self.session_id.lock().unwrap_or_else(|p| p.into_inner()) = Some(sid.to_string());
        }
        let is_sse = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|c| c.starts_with("text/event-stream"));
        let body = resp.text().await.map_err(|e| mcp_err(format!("{name}: {e}")))?;
        if status >= 300 {
            return Err(mcp_err(format!("{name} returned {status}: {}", body.chars().take(500).collect::<String>())));
        }
        let Some(id) = payload.get("id") else { return Ok(None) };
        let body = body.trim();
        if is_sse {
            for line in body.lines() {
                if let Some(data) = line.strip_prefix("data:")
                    && let Ok(message) = serde_json::from_str::<Value>(data.trim())
                    && message.get("id") == Some(id)
                {
                    return Ok(Some(message));
                }
            }
            return Err(mcp_err(format!("{name}: no response in the event stream")));
        }
        if body.is_empty() {
            return Ok(None);
        }
        serde_json::from_str(body).map(Some).map_err(|e| mcp_err(format!("{name}: bad JSON reply: {e}")))
    }

    async fn send(&self, payload: &Value) -> Result<Option<Value>> {
        let mut conn = self.conn.lock().await;
        match &mut *conn {
            Conn::Stdio(pipe) => self.send_stdio(pipe, payload).await,
            Conn::Http(http) => {
                let http = http.clone();
                drop(conn);
                self.send_http(&http, payload).await
            }
            Conn::None => Err(mcp_err(format!("{} is not connected", self.server.name))),
        }
    }

    /// One JSON-RPC request; its `result`, or an `Error::Mcp`.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst) + 1;
        let payload = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let name = &self.server.name;
        let message = self.send(&payload).await?.ok_or_else(|| mcp_err(format!("{name}: no reply to {method}")))?;
        if let Some(err) = message.get("error") {
            let text = err.get("message").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| err.to_string());
            let code = err.get("code").map(Value::to_string).unwrap_or_else(|| "None".into());
            return Err(mcp_err(format!("{name} {method}: {text} ({code})")));
        }
        Ok(message.get("result").cloned().unwrap_or_else(|| json!({})))
    }

    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.send(&json!({"jsonrpc": "2.0", "method": method, "params": params})).await.map(|_| ())
    }

    // ---- the MCP surface ----------------------------------------------
    pub async fn list_tools(&self, refresh: bool) -> Result<Vec<Value>> {
        {
            let cache = self.tools_cache.lock().unwrap_or_else(|p| p.into_inner());
            if !cache.is_empty() && !refresh {
                return Ok(cache.clone());
            }
        }
        let result = self.request("tools/list", json!({})).await?;
        let mut tools: Vec<Value> = result.get("tools").and_then(Value::as_array).cloned().unwrap_or_default();
        if let Some(allowed) = &self.server.allowed_tools {
            tools.retain(|t| t.get("name").and_then(Value::as_str).is_some_and(|n| allowed.iter().any(|a| a == n)));
        }
        *self.tools_cache.lock().unwrap_or_else(|p| p.into_inner()) = tools.clone();
        Ok(tools)
    }

    pub async fn call_tool(&self, name: &str, arguments: Map<String, Value>) -> Result<String> {
        let result = self.request("tools/call", json!({"name": name, "arguments": arguments})).await?;
        Ok(Self::render(&result))
    }

    pub async fn list_resources(&self) -> Result<Vec<Value>> {
        Ok(self.request("resources/list", json!({})).await?.get("resources").and_then(Value::as_array).cloned().unwrap_or_default())
    }

    pub async fn read_resource(&self, uri: &str) -> Result<String> {
        let result = self.request("resources/read", json!({"uri": uri})).await?;
        let parts: Vec<&str> = result
            .get("contents")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|c| c.get("text").and_then(Value::as_str))
            .filter(|t| !t.is_empty())
            .collect();
        Ok(parts.join("\n"))
    }

    pub async fn list_prompts(&self) -> Result<Vec<Value>> {
        Ok(self.request("prompts/list", json!({})).await?.get("prompts").and_then(Value::as_array).cloned().unwrap_or_default())
    }

    pub async fn get_prompt(&self, name: &str, arguments: Map<String, Value>) -> Result<String> {
        let result = self.request("prompts/get", json!({"name": name, "arguments": arguments})).await?;
        let mut out = Vec::new();
        for message in result.get("messages").and_then(Value::as_array).into_iter().flatten() {
            match message.get("content") {
                Some(Value::Object(c)) => out.push(c.get("text").and_then(Value::as_str).unwrap_or("").to_string()),
                Some(Value::Array(items)) => out.extend(items.iter().filter_map(|c| c.get("text").and_then(Value::as_str)).map(str::to_string)),
                _ => {}
            }
        }
        Ok(out.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n"))
    }

    /// MCP content blocks → the string the model reads.
    pub fn render(result: &Value) -> String {
        let prefix = if result.get("isError").and_then(Value::as_bool) == Some(true) { "Error: " } else { "" };
        let mut chunks: Vec<String> = Vec::new();
        for block in result.get("content").and_then(Value::as_array).into_iter().flatten() {
            let s = |v: Option<&Value>| v.and_then(Value::as_str).unwrap_or("").to_string();
            match block.get("type").and_then(Value::as_str) {
                Some("text") => chunks.push(s(block.get("text"))),
                Some("resource") => {
                    let r = block.get("resource").cloned().unwrap_or_default();
                    let text = s(r.get("text"));
                    chunks.push(if text.is_empty() { s(r.get("uri")) } else { text });
                }
                Some(kind @ ("image" | "audio")) => {
                    let mime = block.get("mimeType").and_then(Value::as_str).unwrap_or("binary");
                    chunks.push(format!("[{kind}: {mime}]"));
                }
                _ => {}
            }
        }
        if chunks.is_empty()
            && let Some(structured) = result.get("structuredContent").filter(|v| !v.is_null())
        {
            chunks.push(render_result(structured, 40_000));
        }
        format!("{prefix}{}", chunks.into_iter().filter(|c| !c.is_empty()).collect::<Vec<_>>().join("\n"))
    }

    // ---- tools ---------------------------------------------------------
    /// The server's tools as harness tools. Call after `list_tools()`.
    pub fn as_tools(self: &Arc<Self>) -> Vec<Tool> {
        let specs = self.tools_cache.lock().unwrap_or_else(|p| p.into_inner()).clone();
        specs.iter().map(|spec| self.make_tool(spec)).collect()
    }

    fn make_tool(self: &Arc<Self>, spec: &Value) -> Tool {
        let remote = spec.get("name").and_then(Value::as_str).unwrap_or("tool").to_string();
        let local = if self.server.prefix { format!("{}_{remote}", self.server.name) } else { remote.clone() };
        let schema = spec.get("inputSchema").cloned().filter(Value::is_object).unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        let description =
            spec.get("description").and_then(Value::as_str).filter(|d| !d.is_empty()).map(str::to_string).unwrap_or_else(|| format!("MCP tool {remote}"));
        let client = self.clone();
        Tool::raw(local, description, schema, move |args, _ctx| {
            let client = client.clone();
            let remote = remote.clone();
            async move { client.call_tool(&remote, args).await.map(Value::String) }
        })
        .tags(["mcp".to_string(), self.server.name.clone()])
    }

    pub async fn close(&self) {
        self.connected.store(false, Ordering::SeqCst);
        let mut conn = self.conn.lock().await;
        if let Conn::Stdio(pipe) = &mut *conn {
            let _ = pipe.stdin.shutdown().await;
            if tokio::time::timeout(Duration::from_secs(5), pipe.child.wait()).await.is_err() {
                let _ = pipe.child.kill().await;
            }
        }
        *conn = Conn::None;
    }
}

/// Several servers at once, with one tool list across all of them.
#[derive(Debug, Default)]
pub struct McpManager {
    pub clients: IndexMap<String, Arc<McpClient>>,
    /// Servers that failed to connect: `"name: reason"`.
    pub errors: Vec<String>,
}

impl McpManager {
    pub fn new<I: IntoIterator<Item = McpServer>>(servers: I) -> Self {
        let mut m = Self::default();
        for s in servers {
            m.add(s);
        }
        m
    }

    pub fn add(&mut self, server: McpServer) -> Arc<McpClient> {
        let client = McpClient::new(server);
        self.clients.insert(client.server.name.clone(), client.clone());
        client
    }

    /// Connect every server in parallel; a server that fails is reported in
    /// `errors`, not fatal.
    pub async fn connect(mut self) -> Self {
        let results = futures::future::join_all(self.clients.values().map(|c| async move {
            c.connect().await?;
            c.list_tools(false).await
        }))
        .await;
        self.errors = self.clients.keys().zip(results).filter_map(|(name, r)| r.err().map(|e| format!("{name}: {e}"))).collect();
        self
    }

    pub fn tools(&self) -> Vec<Tool> {
        self.clients.values().filter(|c| c.is_connected()).flat_map(|c| c.as_tools()).collect()
    }

    pub fn get(&self, name: &str) -> Result<&Arc<McpClient>> {
        self.clients.get(name).ok_or_else(|| mcp_err(format!("no MCP server named '{name}'")))
    }

    pub async fn close(&self) {
        futures::future::join_all(self.clients.values().map(|c| c.close())).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_content_blocks() {
        let r = json!({"isError": true, "content": [
            {"type": "text", "text": "a"},
            {"type": "resource", "resource": {"uri": "file:///x"}},
            {"type": "image", "mimeType": "image/png"}
        ]});
        assert_eq!(McpClient::render(&r), "Error: a\nfile:///x\n[image: image/png]");
        assert_eq!(McpClient::render(&json!({"structuredContent": {"n": 1}})), "{\n  \"n\": 1\n}");
    }

    #[test]
    fn transport_follows_the_url() {
        let s = McpServer { name: "x".into(), url: Some("http://h/mcp".into()), ..Default::default() };
        assert_eq!(s.effective_transport(), Transport::Http);
        assert_eq!(McpServer::stdio("f", "npx", ["a"]).effective_transport(), Transport::Stdio);
        let parsed: McpServer = serde_json::from_value(json!({"name": "n", "url": "http://h"})).unwrap();
        assert!(parsed.prefix);
        assert_eq!(parsed.timeout, 60.0);
    }

    #[tokio::test]
    async fn manager_reports_failures() {
        let m = McpManager::new([McpServer::stdio("broken", "definitely-not-a-real-command-xyz", Vec::<String>::new())]).connect().await;
        assert_eq!(m.errors.len(), 1);
        assert!(m.errors[0].starts_with("broken: broken: could not start"), "{}", m.errors[0]);
        assert!(m.tools().is_empty());
        assert!(m.get("nope").is_err());
    }
}
