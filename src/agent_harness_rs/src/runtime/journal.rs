//! Run journal: what each agent was asked, and what it returned.
//!
//! The append-only record you debug a multi-agent run from. Unlike the audit
//! trail it can be trimmed: erasing a person's runs from it is a rewrite.

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::types::{new_id, now};

fn jrn() -> String {
    new_id("jrn")
}

fn event() -> String {
    "event".into()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JournalEntry {
    #[serde(default = "jrn")]
    pub id: String,
    #[serde(default = "now")]
    pub ts: f64,
    #[serde(default)]
    pub run_id: String,
    #[serde(default)]
    pub trace_id: String,
    #[serde(default)]
    pub agent: String,
    /// assignment | handback | tool | model | decision | error
    #[serde(default = "event")]
    pub kind: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub data: Map<String, Value>,
}

impl JournalEntry {
    pub fn line(&self) -> String {
        let clock = &crate::runtime::audit::stamp(self.ts)[11..];
        format!("[{clock}] {} {}: {}", if self.agent.is_empty() { "-" } else { &self.agent }, self.kind, self.text)
    }
}

struct Inner {
    path: Option<PathBuf>,
    echo: bool,
    entries: Vec<JournalEntry>,
}

/// In memory by default; give it a path and it survives the process. Cheap to
/// clone; clones are the same journal.
#[derive(Clone)]
pub struct RunJournal {
    inner: Arc<Mutex<Inner>>,
}

impl Default for RunJournal {
    fn default() -> Self {
        Self { inner: Arc::new(Mutex::new(Inner { path: None, echo: false, entries: vec![] })) }
    }
}

impl std::fmt::Debug for RunJournal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<RunJournal {} entries>", self.lock().entries.len())
    }
}

impl RunJournal {
    pub fn new() -> Self {
        Self::default()
    }

    /// Kept in a JSONL file. Entries already there are not read back — use [`RunJournal::load`].
    pub fn at(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        if let Some(p) = path.parent() {
            let _ = std::fs::create_dir_all(p);
        }
        let j = Self::default();
        j.lock().path = Some(path);
        j
    }

    /// A journal file, read back.
    pub fn load(path: impl Into<PathBuf>) -> Self {
        let j = Self::at(path);
        let entries = {
            let i = j.lock();
            std::fs::read_to_string(i.path.as_ref().expect("set above"))
                .map(|t| t.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str(l).ok()).collect())
                .unwrap_or_default()
        };
        j.lock().entries = entries;
        j
    }

    /// Print every entry as it is written.
    pub fn echo(self, on: bool) -> Self {
        self.lock().echo = on;
        self
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn write(&self, kind: &str, text: &str, agent: &str, run_id: &str, trace_id: &str, data: Map<String, Value>) -> JournalEntry {
        let entry = JournalEntry {
            id: jrn(),
            ts: now(),
            run_id: run_id.into(),
            trace_id: trace_id.into(),
            agent: agent.into(),
            kind: kind.into(),
            text: text.into(),
            data,
        };
        let mut i = self.lock();
        if i.echo {
            println!("{}", entry.line());
        }
        if let Some(path) = &i.path
            && let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path)
        {
            let _ = writeln!(f, "{}", serde_json::to_string(&entry).unwrap_or_default());
        }
        i.entries.push(entry.clone());
        entry
    }

    pub fn assignment(&self, agent: &str, task: &str, run_id: &str, trace_id: &str) -> JournalEntry {
        self.write("assignment", task, agent, run_id, trace_id, Map::new())
    }

    pub fn handback(&self, agent: &str, summary: &str, run_id: &str, trace_id: &str) -> JournalEntry {
        self.write("handback", summary, agent, run_id, trace_id, Map::new())
    }

    /// Drop every entry for these runs, on disk too. Returns how many went.
    pub fn forget(&self, run_ids: &HashSet<String>) -> usize {
        let mut i = self.lock();
        let before = i.entries.len();
        i.entries.retain(|e| !run_ids.contains(&e.run_id));
        if let Some(path) = &i.path
            && let Ok(text) = std::fs::read_to_string(path)
        {
            let kept: String = text
                .lines()
                .filter(|l| serde_json::from_str::<Value>(l).ok().and_then(|v| v["run_id"].as_str().map(|r| run_ids.contains(r))) != Some(true))
                .map(|l| format!("{l}\n"))
                .collect();
            let _ = std::fs::write(path, kept);
        }
        before - i.entries.len()
    }

    pub fn entries(&self) -> Vec<JournalEntry> {
        self.lock().entries.clone()
    }

    pub fn for_agent(&self, agent: &str) -> Vec<JournalEntry> {
        self.lock().entries.iter().filter(|e| e.agent == agent).cloned().collect()
    }

    pub fn render(&self, limit: Option<usize>) -> String {
        let i = self.lock();
        let skip = limit.map_or(0, |n| i.entries.len().saturating_sub(n));
        i.entries.iter().skip(skip).map(JournalEntry::line).collect::<Vec<_>>().join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_forgets_and_reloads() {
        let path = std::env::temp_dir().join(format!("ah_jrn_{}.jsonl", uuid::Uuid::new_v4().simple()));
        let j = RunJournal::at(&path);
        j.assignment("bot", "do it", "r1", "");
        j.handback("bot", "done", "r2", "");
        assert_eq!(RunJournal::load(&path).entries().len(), 2);
        assert_eq!(j.forget(&["r1".to_string()].into_iter().collect()), 1);
        let back = RunJournal::load(&path);
        assert_eq!(back.entries().len(), 1);
        assert!(back.render(None).ends_with("bot handback: done"));
        let _ = std::fs::remove_file(path);
    }
}
