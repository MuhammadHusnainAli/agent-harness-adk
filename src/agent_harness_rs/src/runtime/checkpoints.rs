//! Checkpoints and time travel: durable state at every step, re-runnable.
//!
//! No work is lost on failure, and you can reproduce a run from any prior step
//! instead of guessing what the agent was holding at the time. The files are
//! Python's: one JSONL per run.

use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::errors::{Error, Result};
use crate::types::{Message, Usage, now};

/// A run frozen mid-flight.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub run_id: String,
    #[serde(default)]
    pub step: u32,
    #[serde(default = "now")]
    pub ts: f64,
    #[serde(default)]
    pub agent: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub state: Map<String, Value>,
}

impl Checkpoint {
    pub fn new(run_id: &str, step: u32) -> Self {
        Self {
            run_id: run_id.into(),
            step,
            ts: now(),
            agent: String::new(),
            session_id: String::new(),
            label: String::new(),
            messages: vec![],
            usage: Usage::default(),
            state: Map::new(),
        }
    }
}

struct Inner {
    path: Option<PathBuf>,
    every: u32,
    keep: usize,
    mem: HashMap<String, Vec<Checkpoint>>,
}

/// Step snapshots, in memory or on disk. `every = 0` disables snapshotting
/// (the default on a plain harness). Cheap to clone; clones are the same store.
#[derive(Clone)]
pub struct Checkpointer {
    inner: Arc<Mutex<Inner>>,
}

impl Default for Checkpointer {
    fn default() -> Self {
        Self::new(0)
    }
}

impl std::fmt::Debug for Checkpointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Checkpointer every={}>", self.lock().every)
    }
}

impl Checkpointer {
    /// In memory, a snapshot every `every` steps.
    pub fn new(every: u32) -> Self {
        Self { inner: Arc::new(Mutex::new(Inner { path: None, every, keep: 100, mem: HashMap::new() })) }
    }

    /// On disk under `path`, a snapshot every `every` steps.
    pub fn at(path: impl Into<PathBuf>, every: u32) -> Self {
        let path = path.into();
        let _ = std::fs::create_dir_all(&path);
        let c = Self::new(every);
        c.lock().path = Some(path);
        c
    }

    /// How many snapshots to keep in memory per run.
    pub fn keep(self, n: usize) -> Self {
        self.lock().keep = n.max(1);
        self
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn enabled(&self) -> bool {
        self.lock().every > 0
    }

    pub fn should_save(&self, step: u32) -> bool {
        let every = self.lock().every;
        every > 0 && step.is_multiple_of(every)
    }

    fn file(root: &std::path::Path, run_id: &str) -> PathBuf {
        let safe: String = run_id.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
        root.join(format!("{safe}.jsonl"))
    }

    pub fn save(&self, checkpoint: Checkpoint) -> Checkpoint {
        let mut i = self.lock();
        let keep = i.keep;
        let rows = i.mem.entry(checkpoint.run_id.clone()).or_default();
        rows.push(checkpoint.clone());
        if rows.len() > keep {
            rows.drain(..rows.len() - keep);
        }
        if let Some(root) = &i.path
            && let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(Self::file(root, &checkpoint.run_id))
        {
            let _ = writeln!(f, "{}", serde_json::to_string(&checkpoint).unwrap_or_default());
        }
        checkpoint
    }

    pub fn history(&self, run_id: &str) -> Vec<Checkpoint> {
        let i = self.lock();
        if let Some(rows) = i.mem.get(run_id) {
            return rows.clone();
        }
        let Some(root) = &i.path else { return vec![] };
        std::fs::read_to_string(Self::file(root, run_id))
            .map(|t| t.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str(l).ok()).collect())
            .unwrap_or_default()
    }

    /// The checkpoint at `step` (the last one at or before it), or the latest.
    pub fn load(&self, run_id: &str, step: Option<u32>) -> Result<Checkpoint> {
        let rows = self.history(run_id);
        let Some(last) = rows.last() else {
            return Err(Error::config(format!("no checkpoints for run '{run_id}'")));
        };
        match step {
            None => Ok(last.clone()),
            Some(s) => {
                rows.iter().rev().find(|r| r.step <= s).cloned().ok_or_else(|| Error::config(format!("run '{run_id}' has no checkpoint at or before step {s}")))
            }
        }
    }

    pub fn runs(&self) -> Vec<String> {
        let i = self.lock();
        let mut known: BTreeSet<String> = i.mem.keys().cloned().collect();
        if let Some(root) = &i.path
            && let Ok(entries) = std::fs::read_dir(root)
        {
            known.extend(
                entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
                    .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned())),
            );
        }
        known.into_iter().collect()
    }

    pub fn delete(&self, run_id: &str) {
        let mut i = self.lock();
        i.mem.remove(run_id);
        if let Some(root) = &i.path {
            let _ = std::fs::remove_file(Self::file(root, run_id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saves_loads_and_travels() {
        let dir = std::env::temp_dir().join(format!("ah_ckpt_{}", uuid::Uuid::new_v4().simple()));
        let c = Checkpointer::at(&dir, 2);
        assert!(!c.should_save(1) && c.should_save(2));
        for step in [2, 4, 6] {
            c.save(Checkpoint { messages: vec![Message::user(format!("at {step}"))], ..Checkpoint::new("run_1", step) });
        }
        assert_eq!(c.load("run_1", Some(5)).unwrap().step, 4);
        let fresh = Checkpointer::at(&dir, 2);
        assert_eq!(fresh.load("run_1", None).unwrap().messages[0].text(), "at 6");
        assert_eq!(fresh.runs(), vec!["run_1"]);
        assert!(fresh.load("run_1", Some(1)).is_err());
        fresh.delete("run_1");
        assert!(Checkpointer::at(&dir, 2).history("run_1").is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }
}
