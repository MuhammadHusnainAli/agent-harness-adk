//! Hook engine: pre and post automation around every model call and tool call.
//!
//! ```
//! use agent_harness::runtime::hooks::HookEngine;
//!
//! let mut hooks = HookEngine::new();
//! hooks.on("pre_tool", |ctx| {
//!     if ctx.data["tool"] == "shell" && ctx.data["args"].to_string().contains("rm -rf") {
//!         ctx.block("no recursive deletes");
//!     }
//! });
//! ```
//!
//! A `pre_tool` handler can block the call or rewrite its arguments;
//! `post_tool` can rewrite the result before the model sees it.
//! `model_egress` fires once per provider actually tried; blocking it skips
//! that backend. `handoff` fires when an agent asks to hand the conversation
//! on. `workflow_start` / `workflow_step` / `workflow_end` wrap workflows.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use serde_json::{Map, Value};

/// Every event the harness fires.
pub const EVENTS: [&str; 17] = [
    "run_start",
    "run_end",
    "step_start",
    "step_end",
    "pre_model",
    "post_model",
    "model_egress",
    "pre_tool",
    "post_tool",
    "subagent_start",
    "subagent_end",
    "handoff",
    "memory_write",
    "error",
    "workflow_start",
    "workflow_step",
    "workflow_end",
];

/// What a handler is given, and how it answers back.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HookContext {
    pub event: String,
    pub agent: String,
    pub run_id: String,
    pub step: u32,
    pub data: Map<String, Value>,
    pub blocked: bool,
    pub reason: String,
    pub replacement: Option<Value>,
}

impl HookContext {
    pub fn block(&mut self, reason: &str) {
        self.blocked = true;
        self.reason = if reason.is_empty() { "blocked by hook".into() } else { reason.into() };
    }

    /// Swap the arguments (pre_tool) or the result (post_tool / post_model).
    pub fn replace(&mut self, value: Value) {
        self.replacement = Some(value);
    }

    pub fn replaced(&self) -> bool {
        self.replacement.is_some()
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }
}

type SyncHandler = Arc<dyn Fn(&mut HookContext) + Send + Sync>;
type AsyncHandler = Arc<dyn Fn(HookContext) -> BoxFuture<'static, HookContext> + Send + Sync>;

/// One registered handler.
#[derive(Clone)]
pub enum Handler {
    Sync(SyncHandler),
    /// Takes the context and hands it back, possibly changed.
    Async(AsyncHandler),
}

/// Registry of handlers, fired in registration order.
#[derive(Clone, Default)]
pub struct HookEngine {
    handlers: HashMap<String, Vec<Handler>>,
}

impl fmt::Debug for HookEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let counts: HashMap<&str, usize> = self.handlers.iter().filter(|(_, v)| !v.is_empty()).map(|(k, v)| (k.as_str(), v.len())).collect();
        f.debug_struct("HookEngine").field("handlers", &counts).finish()
    }
}

impl HookEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a synchronous handler.
    pub fn on(&mut self, event: &str, f: impl Fn(&mut HookContext) + Send + Sync + 'static) -> &mut Self {
        self.add(event, Handler::Sync(Arc::new(f)))
    }

    /// Register an async handler: it takes the context and returns it.
    pub fn on_async<F, Fut>(&mut self, event: &str, f: F) -> &mut Self
    where
        F: Fn(HookContext) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = HookContext> + Send + 'static,
    {
        let f = Arc::new(f);
        self.add(event, Handler::Async(Arc::new(move |ctx| f(ctx).boxed())))
    }

    pub fn add(&mut self, event: &str, handler: Handler) -> &mut Self {
        self.handlers.entry(event.into()).or_default().push(handler);
        self
    }

    pub fn clear(&mut self, event: &str) {
        self.handlers.remove(event);
    }

    pub fn count(&self, event: &str) -> usize {
        self.handlers.get(event).map_or(0, Vec::len)
    }

    pub fn is_empty(&self) -> bool {
        self.handlers.values().all(Vec::is_empty)
    }

    /// Fire every handler for `event`; return the (possibly changed) context.
    /// Stops at the first handler that blocks.
    pub async fn emit(&self, event: &str, agent: &str, run_id: &str, step: u32, data: Map<String, Value>) -> HookContext {
        let mut ctx = HookContext { event: event.into(), agent: agent.into(), run_id: run_id.into(), step, data, ..Default::default() };
        for handler in self.handlers.get(event).into_iter().flatten() {
            match handler {
                Handler::Sync(f) => f(&mut ctx),
                Handler::Async(f) => ctx = f(ctx).await,
            }
            if ctx.blocked {
                break;
            }
        }
        ctx
    }

    /// Combine two engines — an agent's own hooks plus the harness-wide ones.
    pub fn merge(&self, other: Option<&HookEngine>) -> HookEngine {
        let mut merged = self.clone();
        if let Some(other) = other {
            for (event, handlers) in &other.handlers {
                merged.handlers.entry(event.clone()).or_default().extend(handlers.iter().cloned());
            }
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn data(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap_or_default()
    }

    #[tokio::test]
    async fn block_stops_the_chain() {
        let mut hooks = HookEngine::new();
        hooks.on("pre_tool", |c| c.block(""));
        hooks.on("pre_tool", |c| c.replace(json!("never")));
        let ctx = hooks.emit("pre_tool", "a", "r", 1, data(json!({"tool": "x"}))).await;
        assert!(ctx.blocked);
        assert_eq!(ctx.reason, "blocked by hook");
        assert!(!ctx.replaced());
    }

    #[tokio::test]
    async fn async_and_merge() {
        let mut a = HookEngine::new();
        a.on("post_tool", |c| c.replace(json!(1)));
        let mut b = HookEngine::new();
        b.on_async("post_tool", |mut c: HookContext| async move {
            let n = c.replacement.as_ref().and_then(Value::as_i64).unwrap_or(0);
            c.replace(json!(n + 1));
            c
        });
        let merged = a.merge(Some(&b));
        assert_eq!(merged.count("post_tool"), 2);
        let ctx = merged.emit("post_tool", "", "", 0, Map::new()).await;
        assert_eq!(ctx.replacement, Some(json!(2)));
        assert_eq!(EVENTS.len(), 17);
    }
}
