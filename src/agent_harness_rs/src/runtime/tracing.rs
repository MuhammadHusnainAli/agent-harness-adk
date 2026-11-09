//! End-to-end tracing: one span per run, step, tool call, model call and sub-agent.
//!
//! Python nests spans through a context variable. Here the parent is passed:
//! [`Tracer::start`] takes the parent span, and the agent hands its run span
//! to its tools and sub-agents, so a sub-agent three levels down still lands
//! under the run that started it. The JSONL export is the Python one.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::types::{new_id, now};

/// One timed unit of work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub name: String,
    /// run | step | tool | model | subagent | memory
    pub kind: String,
    pub trace_id: String,
    pub id: String,
    pub parent_id: Option<String>,
    pub start: f64,
    pub end: Option<f64>,
    pub status: String,
    pub error: Option<String>,
    pub attrs: Map<String, Value>,
}

impl Span {
    pub fn duration_ms(&self) -> f64 {
        (self.end.unwrap_or_else(now) - self.start) * 1000.0
    }

    pub fn set(&mut self, key: &str, value: impl Into<Value>) -> &mut Self {
        self.attrs.insert(key.into(), value.into());
        self
    }

    /// Mark the span failed.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.status = "error".into();
        self.error = Some(error.into());
    }

    /// The exported form — the Python `Span.to_dict()`.
    pub fn to_json(&self) -> Value {
        json!({
            "trace_id": self.trace_id, "span_id": self.id, "parent_id": self.parent_id,
            "name": self.name, "kind": self.kind, "status": self.status,
            "duration_ms": (self.duration_ms() * 100.0).round() / 100.0, "start": self.start,
            "error": self.error, "attrs": self.attrs,
        })
    }
}

/// Receives each finished span.
pub type Exporter = Arc<dyn Fn(&Span) + Send + Sync>;

/// Human-readable spans on stderr, indented by depth.
pub fn console_exporter() -> Exporter {
    let depths: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
    Arc::new(move |span: &Span| {
        let mut d = depths.lock().unwrap_or_else(|p| p.into_inner());
        let depth = span.parent_id.as_ref().and_then(|p| d.get(p).copied()).unwrap_or(0);
        d.insert(span.id.clone(), depth + 1);
        let status = if span.status == "ok" { String::new() } else { format!(" [{}: {}]", span.status, span.error.clone().unwrap_or_default()) };
        let extra = span.attrs.get("cost_usd").and_then(Value::as_f64).filter(|c| *c != 0.0).map(|c| format!(" ${c:.4}")).unwrap_or_default();
        eprintln!("{}{}:{} {:.0}ms{extra}{status}", "  ".repeat(depth), span.kind, span.name, span.duration_ms());
    })
}

struct Inner {
    exporters: Vec<Exporter>,
    files: Vec<PathBuf>,
    enabled: bool,
    keep: usize,
    spans: Vec<Span>,
}

/// Creates spans and hands finished ones to its exporters. Cheap to clone;
/// clones are the same tracer.
#[derive(Clone)]
pub struct Tracer {
    inner: Arc<Mutex<Inner>>,
}

impl Default for Tracer {
    fn default() -> Self {
        Self { inner: Arc::new(Mutex::new(Inner { exporters: vec![], files: vec![], enabled: true, keep: 1000, spans: vec![] })) }
    }
}

impl std::fmt::Debug for Tracer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Tracer {} spans>", self.lock().spans.len())
    }
}

impl Tracer {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn exporter(self, exporter: Exporter) -> Self {
        self.lock().exporters.push(exporter);
        self
    }

    /// One JSON object per span, appended to `path` — the record you debug from.
    pub fn jsonl(self, path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        if let Some(p) = path.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        let file = path.clone();
        let export: Exporter = Arc::new(move |span: &Span| {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&file) {
                let _ = writeln!(f, "{}", crate::pyjson::dumps(&span.to_json()));
            }
        });
        {
            let mut i = self.lock();
            i.exporters.push(export);
            i.files.push(path);
        }
        self
    }

    pub fn enabled(self, on: bool) -> Self {
        self.lock().enabled = on;
        self
    }

    /// How many finished spans to keep in memory.
    pub fn keep(self, n: usize) -> Self {
        self.lock().keep = n;
        self
    }

    /// Open a span under `parent` (a new trace when there is none).
    pub fn start(&self, name: &str, kind: &str, parent: Option<&Span>) -> Span {
        Span {
            name: name.into(),
            kind: kind.into(),
            trace_id: parent.map_or_else(|| new_id("trace"), |p| p.trace_id.clone()),
            id: new_id("span"),
            parent_id: parent.map(|p| p.id.clone()),
            start: now(),
            end: None,
            status: "ok".into(),
            error: None,
            attrs: Map::new(),
        }
    }

    /// Close a span and export it. An exporter can never break a run.
    pub fn finish(&self, mut span: Span) -> Span {
        span.end = Some(now());
        let exporters = {
            let mut i = self.lock();
            if i.spans.len() < i.keep {
                i.spans.push(span.clone());
            }
            if !i.enabled {
                return span;
            }
            i.exporters.clone()
        };
        for e in exporters {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| e(&span)));
        }
        span
    }

    pub fn spans(&self) -> Vec<Span> {
        self.lock().spans.clone()
    }

    /// Drop these traces from memory and from every JSONL export. Returns how
    /// many spans were dropped from memory.
    pub fn forget(&self, trace_ids: &HashSet<String>) -> usize {
        let mut i = self.lock();
        let before = i.spans.len();
        i.spans.retain(|s| !trace_ids.contains(&s.trace_id));
        for path in &i.files {
            if let Ok(text) = std::fs::read_to_string(path) {
                let kept: String = text
                    .lines()
                    .filter(|l| serde_json::from_str::<Value>(l).ok().and_then(|v| v["trace_id"].as_str().map(|t| trace_ids.contains(t))) != Some(true))
                    .map(|l| format!("{l}\n"))
                    .collect();
                let _ = std::fs::write(path, kept);
            }
        }
        before - i.spans.len()
    }

    /// The collected spans as an indented tree — handy in tests and the CLI.
    pub fn tree(&self) -> String {
        let spans = self.spans();
        let mut lines = Vec::new();
        fn walk(spans: &[Span], parent: Option<&str>, depth: usize, lines: &mut Vec<String>) {
            for s in spans.iter().filter(|s| s.parent_id.as_deref() == parent) {
                lines.push(format!("{}{}:{} {:.0}ms", "  ".repeat(depth), s.kind, s.name, s.duration_ms()));
                walk(spans, Some(&s.id), depth + 1, lines);
            }
        }
        walk(&spans, None, 0, &mut lines);
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_nest_and_export() {
        let path = std::env::temp_dir().join(format!("ah_trace_{}.jsonl", uuid::Uuid::new_v4().simple()));
        let t = Tracer::new().jsonl(&path);
        let run = t.start("bot", "run", None);
        let mut tool = t.start("search", "tool", Some(&run));
        tool.fail("ToolError: nope");
        let tool = t.finish(tool);
        let run = t.finish(run);
        assert_eq!(tool.trace_id, run.trace_id);
        assert!(t.tree().starts_with("run:bot ") && t.tree().contains("\n  tool:search "));
        let line: Value = serde_json::from_str(std::fs::read_to_string(&path).unwrap().lines().next().unwrap()).unwrap();
        assert_eq!((line["status"].as_str(), line["parent_id"].as_str()), (Some("error"), Some(run.id.as_str())));
        assert_eq!(t.forget(&[run.trace_id.clone()].into_iter().collect()), 2);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");
        let _ = std::fs::remove_file(path);
    }
}
