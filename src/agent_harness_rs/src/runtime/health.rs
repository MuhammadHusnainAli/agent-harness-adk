//! Service health: latency, failure rate, saturation. Is the platform itself healthy?
//!
//! Every model call and tool call is timed and counted here, so when a run gets
//! slow you can tell whether it is the model, one tool, or your own concurrency cap.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use serde_json::{Value, json};

use crate::types::now;

fn percentile(values: &VecDeque<f64>, pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut ordered: Vec<f64> = values.iter().copied().collect();
    ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((ordered.len() as f64 * pct) as usize).min(ordered.len() - 1);
    (ordered[index] * 100.0).round() / 100.0
}

/// Rolling stats for one component — a provider, a tool, an MCP server.
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    pub name: String,
    pub kind: String,
    pub window: usize,
    pub calls: u64,
    pub failures: u64,
    pub inflight: u64,
    pub last_error: String,
    pub last_ts: f64,
    pub durations: VecDeque<f64>,
}

impl ComponentHealth {
    pub fn new(name: &str, kind: &str, window: usize) -> Self {
        Self {
            name: name.into(),
            kind: kind.into(),
            window,
            calls: 0,
            failures: 0,
            inflight: 0,
            last_error: String::new(),
            last_ts: 0.0,
            durations: VecDeque::new(),
        }
    }

    pub fn record(&mut self, duration_ms: f64, ok: bool, error: &str) {
        self.calls += 1;
        self.last_ts = now();
        if self.durations.len() >= self.window.max(1) {
            self.durations.pop_front();
        }
        self.durations.push_back(duration_ms);
        if !ok {
            self.failures += 1;
            self.last_error = error.chars().take(300).collect();
        }
    }

    pub fn failure_rate(&self) -> f64 {
        if self.calls == 0 { 0.0 } else { (self.failures as f64 / self.calls as f64 * 10_000.0).round() / 10_000.0 }
    }

    pub fn p50(&self) -> f64 {
        percentile(&self.durations, 0.50)
    }

    pub fn p95(&self) -> f64 {
        percentile(&self.durations, 0.95)
    }

    /// healthy | degraded | unhealthy | unknown
    pub fn status(&self, degraded_rate: f64, unhealthy_rate: f64) -> &'static str {
        if self.calls == 0 {
            "unknown"
        } else if self.failure_rate() >= unhealthy_rate {
            "unhealthy"
        } else if self.failure_rate() >= degraded_rate {
            "degraded"
        } else {
            "healthy"
        }
    }

    pub fn snapshot(&self) -> Value {
        json!({
            "name": self.name, "kind": self.kind, "status": self.status(0.1, 0.5),
            "calls": self.calls, "failures": self.failures, "failure_rate": self.failure_rate(),
            "inflight": self.inflight, "p50_ms": self.p50(), "p95_ms": self.p95(),
            "last_error": if self.last_error.is_empty() { Value::Null } else { json!(self.last_error) },
        })
    }
}

struct Inner {
    window: usize,
    degraded_rate: f64,
    unhealthy_rate: f64,
    saturation_warn: f64,
    components: IndexMap<String, ComponentHealth>,
    started: f64,
}

/// The health of every component the harness talks to, in one place. Cheap
/// to clone; clones share the numbers.
#[derive(Clone)]
pub struct ServiceHealth {
    inner: Arc<Mutex<Inner>>,
}

impl Default for ServiceHealth {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                window: 200,
                degraded_rate: 0.1,
                unhealthy_rate: 0.5,
                saturation_warn: 0.9,
                components: IndexMap::new(),
                started: now(),
            })),
        }
    }
}

impl std::fmt::Debug for ServiceHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<ServiceHealth {}>", self.status())
    }
}

impl ServiceHealth {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn with<T>(&self, name: &str, kind: &str, f: impl FnOnce(&mut ComponentHealth) -> T) -> T {
        let mut i = self.lock();
        let window = i.window;
        let entry = i.components.entry(format!("{kind}:{name}")).or_insert_with(|| ComponentHealth::new(name, kind, window));
        f(entry)
    }

    pub fn component(&self, name: &str, kind: &str) -> ComponentHealth {
        self.with(name, kind, |c| c.clone())
    }

    pub fn record(&self, name: &str, kind: &str, duration_ms: f64, ok: bool, error: &str) {
        self.with(name, kind, |c| c.record(duration_ms, ok, error));
    }

    pub fn begin(&self, name: &str, kind: &str) {
        self.with(name, kind, |c| c.inflight += 1);
    }

    pub fn end(&self, name: &str, kind: &str, duration_ms: f64, ok: bool, error: &str) {
        self.with(name, kind, |c| {
            c.inflight = c.inflight.saturating_sub(1);
            c.record(duration_ms, ok, error);
        });
    }

    /// The worst status across every component.
    pub fn status(&self) -> &'static str {
        let i = self.lock();
        let seen: Vec<&str> = i.components.values().map(|c| c.status(i.degraded_rate, i.unhealthy_rate)).collect();
        if seen.iter().all(|s| *s == "unknown") {
            "unknown"
        } else if seen.contains(&"unhealthy") {
            "unhealthy"
        } else if seen.contains(&"degraded") {
            "degraded"
        } else {
            "healthy"
        }
    }

    /// `running + queued` against a concurrency limit, 0.0 to 1.0.
    pub fn saturation(running: usize, queued: usize, limit: usize) -> f64 {
        (((running + queued) as f64 / limit.max(1) as f64).min(1.0) * 1000.0).round() / 1000.0
    }

    /// Everything, with saturation when a scheduler's `(running, queued, limit)` is given.
    pub fn snapshot(&self, scheduler: Option<(usize, usize, usize)>) -> Value {
        let status = self.status();
        let i = self.lock();
        let mut out = json!({
            "status": status,
            "uptime_s": ((now() - i.started) * 10.0).round() / 10.0,
            "components": i.components.values().map(ComponentHealth::snapshot).collect::<Vec<_>>(),
        });
        if let Some((running, queued, limit)) = scheduler {
            let s = Self::saturation(running, queued, limit);
            out["saturation"] = json!(s);
            out["saturated"] = json!(s >= i.saturation_warn);
        }
        out
    }

    pub fn reset(&self) {
        let mut i = self.lock();
        i.components.clear();
        i.started = now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_and_percentiles() {
        let h = ServiceHealth::new();
        assert_eq!(h.status(), "unknown");
        for ms in [10.0, 20.0, 30.0, 40.0] {
            h.record("search", "tool", ms, true, "");
        }
        assert_eq!(h.status(), "healthy");
        h.begin("claude", "model");
        h.end("claude", "model", 900.0, false, "overloaded");
        assert_eq!(h.status(), "unhealthy");
        let c = h.component("search", "tool");
        assert_eq!((c.p50(), c.p95()), (30.0, 40.0));
        let snap = h.snapshot(Some((9, 1, 10)));
        assert_eq!((snap["saturation"].as_f64(), snap["saturated"].as_bool()), (Some(1.0), Some(true)));
    }
}
