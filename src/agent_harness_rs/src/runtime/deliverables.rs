//! Deliverable store: the documents, reports and artefacts a run produced.
//!
//! The output of the run, kept apart from the transcript that produced it.
//! Versioned by name, so a second pass at `report.md` does not destroy the
//! first. On disk the layout is Python's: `<root>/<name>/v<n><ext>` and an
//! `index.jsonl`.

use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::types::{Artifact, now};

fn safe(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_alphanumeric() || "-_.".contains(c) { c } else { '_' }).collect();
    if s.is_empty() { "artifact".into() } else { s }
}

struct Inner {
    root: Option<PathBuf>,
    items: IndexMap<String, Vec<Artifact>>,
}

/// Versioned artefacts, in memory or on disk. Cheap to clone; clones are the
/// same store.
#[derive(Clone)]
pub struct DeliverableStore {
    inner: Arc<Mutex<Inner>>,
}

impl Default for DeliverableStore {
    fn default() -> Self {
        Self { inner: Arc::new(Mutex::new(Inner { root: None, items: IndexMap::new() })) }
    }
}

impl std::fmt::Debug for DeliverableStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<DeliverableStore {} versions>", self.len())
    }
}

impl DeliverableStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Kept under a folder, and read back from its index.
    pub fn at(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let _ = std::fs::create_dir_all(&root);
        let mut items: IndexMap<String, Vec<Artifact>> = IndexMap::new();
        if let Ok(text) = std::fs::read_to_string(root.join("index.jsonl")) {
            for a in text.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str::<Artifact>(l).ok()) {
                items.entry(a.name.clone()).or_default().push(a);
            }
        }
        Self { inner: Arc::new(Mutex::new(Inner { root: Some(root), items })) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Store a new version of an artefact and return it with its path set.
    pub fn put(&self, artifact: &Artifact, run_id: &str) -> Artifact {
        let mut i = self.lock();
        let n = i.items.get(&artifact.name).map_or(0, Vec::len) + 1;
        let mut stored = artifact.clone();
        if stored.ts == 0.0 {
            stored.ts = now();
        }
        if !run_id.is_empty() {
            stored.extra.insert("run_id".into(), json!(run_id));
        }
        stored.extra.insert("version".into(), json!(n));
        // An artefact with no text but a file behind it — an image, a
        // spreadsheet, something too large to carry — is kept by its bytes.
        let source = stored.path.as_ref().filter(|_| stored.content.is_empty()).map(PathBuf::from).filter(|p| p.is_file());
        let payload = source.and_then(|p| std::fs::read(p).ok()).unwrap_or_else(|| stored.content.as_bytes().to_vec());
        let digest: String = Sha256::digest(&payload).iter().map(|b| format!("{b:02x}")).collect();
        stored.extra.insert("digest".into(), json!(digest[..16]));
        if let Some(root) = &i.root {
            let folder = root.join(safe(&stored.name));
            let _ = std::fs::create_dir_all(&folder);
            let ext = Path::new(&stored.name).extension().map_or_else(|| ".txt".to_string(), |e| format!(".{}", e.to_string_lossy()));
            let path = folder.join(format!("v{n}{ext}"));
            if std::fs::write(&path, &payload).is_ok() {
                stored.path = Some(path.display().to_string());
            }
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(root.join("index.jsonl")) {
                let _ = writeln!(f, "{}", serde_json::to_string(&stored).unwrap_or_default());
            }
        }
        i.items.entry(stored.name.clone()).or_default().push(stored.clone());
        stored
    }

    pub fn extend(&self, artifacts: &[Artifact], run_id: &str) -> Vec<Artifact> {
        artifacts.iter().map(|a| self.put(a, run_id)).collect()
    }

    /// The latest version, or a given one (from 1).
    pub fn get(&self, name: &str, version: Option<usize>) -> Option<Artifact> {
        let i = self.lock();
        let versions = i.items.get(name)?;
        match version {
            None => versions.last().cloned(),
            Some(v) if v >= 1 => versions.get(v - 1).cloned(),
            Some(_) => None,
        }
    }

    pub fn versions(&self, name: &str) -> Vec<Artifact> {
        self.lock().items.get(name).cloned().unwrap_or_default()
    }

    /// The newest version of every artefact.
    pub fn latest(&self) -> Vec<Artifact> {
        self.lock().items.values().filter_map(|v| v.last().cloned()).collect()
    }

    pub fn for_run(&self, run_id: &str) -> Vec<Artifact> {
        self.lock().items.values().flatten().filter(|a| a.extra_str("run_id") == run_id).cloned().collect()
    }

    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<String> = self.lock().items.keys().cloned().collect();
        n.sort();
        n
    }

    pub fn contains(&self, name: &str) -> bool {
        self.lock().items.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.lock().items.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// What the run produced, one row per artefact.
    pub fn manifest(&self) -> Vec<Value> {
        self.latest()
            .iter()
            .map(|a| {
                json!({
                    "name": a.name, "version": a.extra.get("version").cloned().unwrap_or(json!(1)),
                    "produced_by": a.produced_by, "media_type": a.media_type,
                    "chars": a.content.chars().count(), "digest": a.extra_str("digest"), "path": a.path,
                })
            })
            .collect()
    }

    pub fn clear(&self) {
        self.lock().items.clear();
    }

    /// Delete every version produced by these runs, files and index included.
    /// Returns the names of what was removed. Used when a person is erased.
    pub fn forget_runs(&self, run_ids: &HashSet<String>) -> Vec<String> {
        let mut i = self.lock();
        let mut removed = Vec::new();
        for versions in i.items.values_mut() {
            versions.retain(|a| {
                if run_ids.contains(a.extra_str("run_id")) {
                    removed.push(a.name.clone());
                    if let Some(p) = &a.path {
                        let _ = std::fs::remove_file(p);
                    }
                    false
                } else {
                    true
                }
            });
        }
        i.items.retain(|_, v| !v.is_empty());
        if !removed.is_empty()
            && let Some(root) = &i.root
        {
            let body: String = i.items.values().flatten().map(|a| format!("{}\n", serde_json::to_string(a).unwrap_or_default())).collect();
            let _ = std::fs::write(root.join("index.jsonl"), body);
        }
        removed.sort();
        removed.dedup();
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_survive_and_runs_are_forgotten() {
        let dir = std::env::temp_dir().join(format!("ah_dlv_{}", uuid::Uuid::new_v4().simple()));
        let store = DeliverableStore::at(&dir);
        store.put(&Artifact::new("report.md", "first"), "r1");
        let second = store.put(&Artifact::new("report.md", "second"), "r2");
        assert_eq!(second.extra["version"], 2);
        assert!(second.path.as_deref().unwrap().ends_with("v2.md"));
        let again = DeliverableStore::at(&dir);
        assert_eq!(again.get("report.md", Some(1)).unwrap().content, "first");
        assert_eq!(again.manifest()[0]["digest"].as_str().unwrap().len(), 16);
        assert_eq!(again.forget_runs(&["r1".to_string()].into_iter().collect()), vec!["report.md"]);
        assert_eq!(DeliverableStore::at(&dir).versions("report.md").len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }
}
