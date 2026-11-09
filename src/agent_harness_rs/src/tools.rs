//! Tools: plain Rust functions the model is allowed to call.
//!
//! Python reads a function's signature and docstring. In Rust the arguments are
//! a struct that derives `Deserialize` and `JsonSchema`; its doc comments become
//! the parameter descriptions the model sees, and serde validates what comes
//! back before your function is called.
//!
//! ```
//! use agent_harness::tools::{Tool, ToolContext};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! /// Look up the status of a customer order.
//! #[derive(Deserialize, JsonSchema)]
//! struct OrderStatus {
//!     /// the order number, digits only.
//!     order_id: String,
//! }
//!
//! let tool = Tool::new("order_status", |args: OrderStatus, _ctx: ToolContext| async move {
//!     Ok(format!("order {} shipped", args.order_id))
//! });
//! assert_eq!(tool.description, "Look up the status of a customer order.");
//! assert_eq!(tool.parameters["required"][0], "order_id");
//! ```

use std::collections::HashSet;
use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::FutureExt;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::errors::{Error, Result};
use crate::types::{ContentBlock, ToolOutcome};

/// What the policy gate does with a call: run it, ask a person, or refuse.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Permission {
    #[default]
    Allow,
    Ask,
    Deny,
}

/// A callback a tool can use to report progress: `(message, data)`.
pub type Emit = Arc<dyn Fn(&str, Value) + Send + Sync>;

/// Everything a tool may want to know about the run calling it.
/// The model never sees it.
#[derive(Clone, Default)]
pub struct ToolContext {
    pub agent: String,
    pub run_id: String,
    pub session_id: String,
    pub step: u32,
    /// Shared scratch state for the run's tools.
    pub state: Arc<Mutex<Map<String, Value>>>,
    pub emit: Option<Emit>,
    /// The run this call belongs to: its budget, the sub-agent results it
    /// collects, the conversation relay. Set by the agent loop.
    pub run: Option<Arc<RunScope>>,
    /// This call's span: what a sub-agent it starts nests under.
    pub(crate) span: Option<crate::runtime::Span>,
}

/// What a tool call can reach of the run it is part of. Opaque: the agent
/// loop fills it in, and its own tools (`delegate`, `handoff`) read it.
#[derive(Default)]
pub struct RunScope {
    pub(crate) guard: Option<Arc<crate::runtime::BudgetGuard>>,
    pub(crate) children: Mutex<Vec<crate::types::RunResult>>,
    pub(crate) relay: Option<Arc<Mutex<crate::handoff::Relay>>>,
    pub(crate) notebook: Option<Arc<crate::modes::Notebook>>,
    /// The run's span: what a tool's or a sub-agent's span nests under.
    pub(crate) span: Option<crate::runtime::Span>,
    /// The stored approval this run is being finished from.
    pub(crate) approval: Option<crate::runtime::Approval>,
    /// May a call that needs a person stop the run and wait?
    pub(crate) pausable: bool,
}

impl ToolContext {
    pub fn log(&self, message: &str, data: Value) {
        if let Some(emit) = &self.emit {
            emit(message, data);
        }
    }
}

impl fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolContext")
            .field("agent", &self.agent)
            .field("run_id", &self.run_id)
            .field("session_id", &self.session_id)
            .field("step", &self.step)
            .finish_non_exhaustive()
    }
}

/// Arguments for a tool that takes none.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
// Its doc comment is for readers of this crate, not for the model.
#[schemars(description = "")]
pub struct NoArgs {}

/// A tool as the model sees it: a name, a reason to call it, and a shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "empty_object_schema")]
    pub parameters: Value,
}

pub(crate) fn empty_object_schema() -> Value {
    json!({"type": "object", "properties": {}})
}

type Handler = Arc<dyn Fn(Map<String, Value>, ToolContext) -> BoxFuture<'static, Result<Value>> + Send + Sync>;

/// A callable exposed to the model, with a validated schema around it.
#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments.
    pub parameters: Value,
    pub permission: Permission,
    pub tags: HashSet<String>,
    pub cacheable: bool,
    pub max_output_chars: usize,
    /// End the run with this tool's output as the answer.
    pub returns_direct: bool,
    handler: Handler,
}

impl fmt::Debug for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<Tool {}>", self.name)
    }
}

/// The JSON Schema for `A`, cleaned the way the Python harness cleans it.
/// Returns `(parameters, struct doc comment)`.
pub fn schema_for<A: JsonSchema>() -> (Value, String) {
    let mut schema = serde_json::to_value(schemars::schema_for!(A)).unwrap_or_else(|_| empty_object_schema());
    let mut summary = String::new();
    if let Value::Object(map) = &mut schema {
        map.remove("$schema");
        map.remove("title");
        if let Some(Value::String(d)) = map.remove("description") {
            summary = d;
        }
        if let Some(Value::Object(props)) = map.get_mut("properties") {
            for prop in props.values_mut() {
                if let Value::Object(p) = prop {
                    p.remove("title");
                }
            }
        }
        map.entry("type").or_insert_with(|| json!("object"));
        map.entry("properties").or_insert_with(|| json!({}));
        // A struct with no fields allows nothing extra; the model needs no such rule.
        if map.get("additionalProperties") == Some(&Value::Bool(false)) && map.get("properties").and_then(Value::as_object).is_some_and(Map::is_empty) {
            map.remove("additionalProperties");
        }
    }
    (schema, summary)
}

impl Tool {
    /// A tool from a typed async function. The description is the doc comment
    /// on the argument struct, unless you set one with [`Tool::describe`].
    pub fn new<A, R, F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        A: DeserializeOwned + JsonSchema + Send + 'static,
        R: Serialize + Send + 'static,
        F: Fn(A, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<R>> + Send + 'static,
    {
        let name = name.into();
        let (parameters, summary) = schema_for::<A>();
        let f = Arc::new(f);
        let tool_name = name.clone();
        let handler: Handler = Arc::new(move |args, ctx| {
            let parsed: std::result::Result<A, _> = serde_json::from_value(Value::Object(args));
            let f = f.clone();
            let tool_name = tool_name.clone();
            async move {
                let args = parsed.map_err(|e| Error::tool(format!("invalid arguments for {tool_name} — {e}"), &tool_name))?;
                let out = f(args, ctx).await?;
                serde_json::to_value(out).map_err(|e| Error::tool(format!("{tool_name} returned something that is not JSON: {e}"), &tool_name))
            }
            .boxed()
        });
        let description = if summary.is_empty() { format!("Call {name}.") } else { summary };
        Self::build(name, description, parameters, handler)
    }

    /// A tool from a schema you supply and a function taking the raw argument
    /// object. The remote side owns validation — this is how MCP tools are wired.
    pub fn raw<F, Fut>(name: impl Into<String>, description: impl Into<String>, parameters: Value, f: F) -> Self
    where
        F: Fn(Map<String, Value>, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        let f = Arc::new(f);
        let handler: Handler = Arc::new(move |args, ctx| f(args, ctx).boxed());
        Self::build(name.into(), description.into(), parameters, handler)
    }

    fn build(name: String, description: String, parameters: Value, handler: Handler) -> Self {
        Self {
            name,
            description,
            parameters,
            permission: Permission::Allow,
            tags: HashSet::new(),
            cacheable: false,
            max_output_chars: 40_000,
            returns_direct: false,
            handler,
        }
    }

    // ---- builder ------------------------------------------------------
    pub fn describe(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn permission(mut self, permission: Permission) -> Self {
        self.permission = permission;
        self
    }

    pub fn tags<I: IntoIterator<Item = S>, S: Into<String>>(mut self, tags: I) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    pub fn cacheable(mut self, yes: bool) -> Self {
        self.cacheable = yes;
        self
    }

    pub fn max_output_chars(mut self, limit: usize) -> Self {
        self.max_output_chars = limit;
        self
    }

    pub fn returns_direct(mut self, yes: bool) -> Self {
        self.returns_direct = yes;
        self
    }

    pub fn rename(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    // ---- execution ----------------------------------------------------
    pub fn to_schema(&self) -> ToolSchema {
        ToolSchema { name: self.name.clone(), description: self.description.clone(), parameters: self.parameters.clone() }
    }

    /// Validate, call, and return the tool's value. Failures are `Error::Tool`.
    pub async fn invoke(&self, args: Map<String, Value>, ctx: ToolContext) -> Result<Value> {
        match (self.handler)(args, ctx).await {
            Ok(v) => Ok(v),
            Err(e @ (Error::Tool { .. } | Error::ToolNotFound { .. })) => Err(e),
            Err(e) => Err(Error::tool(format!("{} failed: {e}", self.name), &self.name)),
        }
    }

    /// Invoke and package the result for the transcript. Never fails.
    pub async fn run(&self, call_id: &str, args: Map<String, Value>, ctx: ToolContext) -> ToolOutcome {
        let started = Instant::now();
        let ms = |s: Instant| s.elapsed().as_secs_f64() * 1000.0;
        match self.invoke(args, ctx).await {
            Ok(value) => {
                let (readable, media) = split_media(&value);
                ToolOutcome {
                    call_id: call_id.into(),
                    name: self.name.clone(),
                    content: render_result(&readable, self.max_output_chars),
                    duration_ms: ms(started),
                    value: Some(value),
                    media,
                    ..Default::default()
                }
            }
            Err(e) => ToolOutcome {
                call_id: call_id.into(),
                name: self.name.clone(),
                content: format!("Error: {e}"),
                is_error: true,
                duration_ms: ms(started),
                ..Default::default()
            },
        }
    }
}

/// Is this JSON value a serialised media block?
fn as_media(v: &Value) -> Option<ContentBlock> {
    let obj = v.as_object()?;
    let kind = obj.get("type")?.as_str()?;
    if !matches!(kind, "image" | "audio" | "video" | "document") || !obj.contains_key("media_type") {
        return None;
    }
    serde_json::from_value(v.clone()).ok()
}

/// What a tool returned, as what is read and what is looked at.
///
/// A tool may return an image — on its own or in a list beside its text. The
/// image reaches the model as an image, not as base64 in the prompt.
pub fn split_media(value: &Value) -> (Value, Vec<ContentBlock>) {
    if let Some(block) = as_media(value) {
        let (_, m) = block.as_media().expect("media block");
        let label = if m.name.is_empty() { m.media_type.clone() } else { m.name.clone() };
        return (Value::String(format!("[{}: {label}]", block.type_name())), vec![block]);
    }
    if let Value::Array(items) = value {
        let media: Vec<ContentBlock> = items.iter().filter_map(as_media).collect();
        if media.is_empty() {
            return (value.clone(), vec![]);
        }
        let rest: Vec<&Value> = items.iter().filter(|v| as_media(v).is_none()).collect();
        if rest.is_empty() {
            let plural = if media.len() > 1 { "s" } else { "" };
            return (Value::String(format!("[{} {}{plural}]", media.len(), media[0].type_name())), media);
        }
        if rest.iter().all(|v| v.is_string()) {
            let text: Vec<&str> = rest.iter().filter_map(|v| v.as_str()).collect();
            return (Value::String(text.join("\n")), media);
        }
        let readable = if rest.len() == 1 { rest[0].clone() } else { Value::Array(rest.into_iter().cloned().collect()) };
        return (readable, media);
    }
    (value.clone(), vec![])
}

/// Turn whatever a tool returned into something a model can read.
pub fn render_result(value: &Value, limit: usize) -> String {
    let text = match value {
        Value::Null => return "(no output)".into(),
        Value::String(s) => s.clone(),
        // Python's str(): the model sees the same text from either harness.
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => n.to_string(),
        other => serde_json::to_string_pretty(other).unwrap_or_else(|_| other.to_string()),
    };
    truncate(&text, limit)
}

pub(crate) fn truncate(text: &str, limit: usize) -> String {
    let count = text.chars().count();
    if count <= limit {
        return text.to_string();
    }
    let head: String = text.chars().take(limit).collect();
    format!("{head}\n... [truncated {} characters]", count - limit)
}

/// Shell-style glob match, as Python's `fnmatch`.
pub(crate) fn fnmatch(name: &str, pattern: &str) -> bool {
    glob::Pattern::new(pattern).map(|p| p.matches(name)).unwrap_or(name == pattern)
}

/// The set of tools an agent may reach for. Glob-filterable for allowlists.
#[derive(Debug, Clone, Default)]
pub struct ToolRegistry {
    tools: IndexMap<String, Tool>,
}

impl ToolRegistry {
    pub fn new<I: IntoIterator<Item = Tool>>(tools: I) -> Self {
        let mut reg = Self::default();
        reg.extend(tools);
        reg
    }

    /// Add a tool, replacing any with the same name.
    pub fn add(&mut self, tool: Tool) -> &Tool {
        let name = tool.name.clone();
        self.tools.insert(name.clone(), tool);
        &self.tools[&name]
    }

    /// Add a tool, failing if the name is taken.
    pub fn try_add(&mut self, tool: Tool) -> Result<&Tool> {
        if self.tools.contains_key(&tool.name) {
            return Err(Error::tool(format!("tool '{}' is already registered", tool.name), &tool.name));
        }
        Ok(self.add(tool))
    }

    pub fn extend<I: IntoIterator<Item = Tool>>(&mut self, tools: I) -> &mut Self {
        for t in tools {
            self.add(t);
        }
        self
    }

    pub fn remove(&mut self, name: &str) {
        self.tools.shift_remove(name);
    }

    pub fn get(&self, name: &str) -> Result<&Tool> {
        self.tools.get(name).ok_or_else(|| {
            let known = if self.tools.is_empty() { "none".to_string() } else { self.names().join(", ") };
            Error::ToolNotFound { message: format!("no tool named '{name}'; available: {known}"), tool: name.into() }
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Tool> {
        self.tools.values()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Sorted tool names.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// A sub-registry matching glob patterns or tags — this is the tool
    /// allowlist. `None` keeps everything; an empty list keeps nothing.
    pub fn select(&self, patterns: Option<&[String]>) -> ToolRegistry {
        let Some(patterns) = patterns else { return self.clone() };
        ToolRegistry::new(self.tools.values().filter(|t| patterns.iter().any(|p| fnmatch(&t.name, p) || t.tags.contains(p))).cloned())
    }

    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(Tool::to_schema).collect()
    }

    pub fn merge(&mut self, other: &ToolRegistry) -> &mut Self {
        self.extend(other.iter().cloned())
    }
}

impl FromIterator<Tool> for ToolRegistry {
    fn from_iter<I: IntoIterator<Item = Tool>>(iter: I) -> Self {
        Self::new(iter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MediaBlock, attach_bytes};

    /// Add two numbers.
    #[derive(Deserialize, JsonSchema)]
    struct Add {
        /// the first
        a: i64,
        /// the second
        #[serde(default = "one")]
        b: i64,
    }

    fn one() -> i64 {
        1
    }

    fn add_tool() -> Tool {
        Tool::new("add", |args: Add, _ctx| async move { Ok(args.a + args.b) })
    }

    fn obj(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn schema_from_struct() {
        let t = add_tool();
        assert_eq!(t.description, "Add two numbers.");
        assert_eq!(t.parameters["type"], "object");
        assert_eq!(t.parameters["properties"]["a"]["description"], "the first");
        assert_eq!(t.parameters["required"], json!(["a"]));
        assert!(t.parameters.get("title").is_none());
        assert!(t.parameters.get("$schema").is_none());
    }

    #[test]
    fn no_args_schema_is_empty_object() {
        let t = Tool::new("ping", |_: NoArgs, _| async { Ok("pong") });
        assert_eq!(t.parameters["properties"], json!({}));
        assert_eq!(t.description, "Call ping.");
    }

    #[tokio::test]
    async fn runs_and_validates() {
        let t = add_tool();
        let ok = t.run("c1", obj(json!({"a": 2})), ToolContext::default()).await;
        assert_eq!(ok.content, "3");
        assert!(!ok.is_error);
        let bad = t.run("c2", obj(json!({"a": "x"})), ToolContext::default()).await;
        assert!(bad.is_error);
        assert!(bad.content.starts_with("Error: invalid arguments for add"));
    }

    #[tokio::test]
    async fn failures_are_wrapped() {
        let t = Tool::new("boom", |_: NoArgs, _| async { Err::<(), _>(Error::config("nope")) });
        let out = t.run("c", Map::new(), ToolContext::default()).await;
        assert_eq!(out.content, "Error: boom failed: nope");
    }

    #[tokio::test]
    async fn context_is_passed() {
        let t = Tool::new("who", |_: NoArgs, ctx: ToolContext| async move { Ok(ctx.agent) });
        let ctx = ToolContext { agent: "a1".into(), ..Default::default() };
        assert_eq!(t.run("c", Map::new(), ctx).await.content, "a1");
    }

    #[test]
    fn render_matches_python() {
        assert_eq!(render_result(&Value::Null, 100), "(no output)");
        assert_eq!(render_result(&json!(true), 100), "True");
        assert_eq!(render_result(&json!({"a": 1}), 100), "{\n  \"a\": 1\n}");
        assert_eq!(render_result(&json!("abcdef"), 3), "abc\n... [truncated 3 characters]");
    }

    #[test]
    fn media_is_split_out() {
        let img = serde_json::to_value(ContentBlock::Image(MediaBlock::from_bytes(b"png", "image/png"))).unwrap();
        let (text, media) = split_media(&json!(["clicked", img.clone()]));
        assert_eq!(text, json!("clicked"));
        assert_eq!(media.len(), 1);
        let (text, media) = split_media(&img);
        assert_eq!(text, json!("[image: image/png]"));
        assert_eq!(media.len(), 1);
        let audio = serde_json::to_value(attach_bytes(b"x", "audio/wav")).unwrap();
        assert_eq!(split_media(&json!([audio.clone(), audio])).0, json!("[2 audios]"));
    }

    #[test]
    fn registry_select_and_errors() {
        let echo = Tool::new("echo", |_: NoArgs, _| async { Ok("") }).tags(["builtin"]);
        let reg = ToolRegistry::new([add_tool(), echo]);
        assert_eq!(reg.names(), vec!["add", "echo"]);
        assert_eq!(reg.select(Some(&["a*".into()])).names(), vec!["add"]);
        assert_eq!(reg.select(Some(&["builtin".into()])).names(), vec!["echo"]);
        assert!(reg.select(Some(&[])).is_empty());
        assert_eq!(reg.select(None).len(), 2);
        let err = reg.get("nope").unwrap_err();
        assert_eq!(err.to_string(), "no tool named 'nope'; available: add, echo");
        let mut reg = reg;
        assert!(reg.try_add(add_tool()).is_err());
    }
}
