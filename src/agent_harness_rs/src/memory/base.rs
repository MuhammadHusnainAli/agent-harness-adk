//! Storage behind every memory scope: records in, records out.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::errors::{Error, Result};
use crate::memory::trace::Trace;
use crate::types::{new_id, now};

fn mem_id() -> String {
    new_id("mem")
}

fn user_scope() -> String {
    "user".into()
}

fn note_kind() -> String {
    "note".into()
}

/// One remembered thing. `kind` is free-form: fact, decision, finding, plan...
///
/// Serialises exactly as the Python `MemoryRecord` does, so one store can be
/// shared by both.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    #[serde(default = "mem_id")]
    pub id: String,
    #[serde(default = "user_scope")]
    pub scope: String,
    #[serde(default = "note_kind")]
    pub kind: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub data: Map<String, Value>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub source: String,
    // Whose memory this is. Flat rather than nested so every backend can index
    // them — a column in SQL, a field in Mongo, a prefix in an object store.
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub tenant_id: Option<String>,
    #[serde(default = "now")]
    pub ts: f64,
    /// A precomputed embedding. Never stored with the record.
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

impl Default for MemoryRecord {
    fn default() -> Self {
        Self::new("user", "note", "")
    }
}

impl MemoryRecord {
    pub fn new(scope: &str, kind: &str, text: &str) -> Self {
        Self {
            id: mem_id(),
            scope: scope.into(),
            kind: kind.into(),
            text: text.into(),
            data: Map::new(),
            tags: vec![],
            source: String::new(),
            user_id: None,
            session_id: None,
            tenant_id: None,
            ts: now(),
            embedding: None,
        }
    }

    pub fn tags<I: IntoIterator<Item = S>, S: Into<String>>(mut self, tags: I) -> Self {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    pub fn data(mut self, data: Map<String, Value>) -> Self {
        self.data = data;
        self
    }

    /// The record as a bullet: `- [decision] text`, or `- text` for a note.
    pub fn line(&self) -> String {
        if self.kind != "note" { format!("- [{}] {}", self.kind, self.text) } else { format!("- {}", self.text) }
    }

    /// The value on one identity axis (`user_id`, `session_id`, `tenant_id`).
    pub fn axis(&self, field: &str) -> Option<&str> {
        match field {
            "user_id" => self.user_id.as_deref(),
            "session_id" => self.session_id.as_deref(),
            "tenant_id" => self.tenant_id.as_deref(),
            _ => None,
        }
    }
}

/// Which records to read back. Every field narrows; the default is all of them.
#[derive(Debug, Clone, Default)]
pub struct RecordQuery {
    pub scope: Option<String>,
    /// The most recent `limit`.
    pub limit: Option<usize>,
    pub kind: Option<String>,
    pub trace: Option<Trace>,
}

impl RecordQuery {
    pub fn scope(scope: &str) -> Self {
        Self { scope: Some(scope.into()), ..Self::default() }
    }

    pub fn traced(mut self, trace: Option<&Trace>) -> Self {
        self.trace = trace.cloned();
        self
    }

    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    pub fn kind(mut self, kind: &str) -> Self {
        self.kind = Some(kind.into());
        self
    }

    /// Does a record pass every narrowing but the limit?
    pub fn admits(&self, r: &MemoryRecord) -> bool {
        self.scope.as_ref().is_none_or(|s| &r.scope == s) && self.kind.as_ref().is_none_or(|k| &r.kind == k) && self.trace.as_ref().is_none_or(|t| t.matches(r))
    }
}

/// The most recent `limit` of `rows` (all of them with no limit).
pub fn last(mut rows: Vec<MemoryRecord>, limit: Option<usize>) -> Vec<MemoryRecord> {
    if let Some(n) = limit.filter(|n| *n > 0)
        && rows.len() > n
    {
        rows.drain(..rows.len() - n);
    }
    rows
}

/// Where a document lives for this trace — `u-alice/user.md`, say.
pub fn doc_key(name: &str, trace: Option<&Trace>) -> String {
    match trace {
        Some(t) if !t.is_empty() => format!("{}/{name}", t.slug()),
        _ => name.to_string(),
    }
}

/// The keyword fallback every store has: records sharing the query's words,
/// most shared first, newest first among equals.
pub fn keyword_search(records: Vec<MemoryRecord>, query: &str, limit: usize) -> Vec<MemoryRecord> {
    let lowered = query.to_lowercase();
    let terms: Vec<&str> = lowered.split_whitespace().filter(|t| t.chars().count() > 2).collect();
    let mut scored: Vec<(usize, MemoryRecord)> = records
        .into_iter()
        .filter_map(|rec| {
            let hay = format!("{} {} {}", rec.text, rec.kind, rec.tags.join(" ")).to_lowercase();
            let hits = terms.iter().filter(|t| hay.contains(*t)).count();
            (hits > 0).then_some((hits, rec))
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.ts.partial_cmp(&a.1.ts).unwrap_or(std::cmp::Ordering::Equal)));
    scored.into_iter().take(limit).map(|(_, r)| r).collect()
}

/// Append-only records plus a handful of named documents (like `user.md`).
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn append(&self, record: MemoryRecord) -> Result<MemoryRecord>;
    async fn all(&self, query: RecordQuery) -> Result<Vec<MemoryRecord>>;
    async fn clear(&self, scope: Option<&str>, trace: Option<&Trace>) -> Result<()>;
    async fn read_doc(&self, name: &str, trace: Option<&Trace>) -> Result<String>;
    async fn write_doc(&self, name: &str, text: &str, trace: Option<&Trace>) -> Result<()>;

    /// Keyword fallback. `SemanticMemory` overrides this with real retrieval.
    async fn search(&self, query: &str, scope: Option<&str>, limit: usize, trace: Option<&Trace>) -> Result<Vec<MemoryRecord>> {
        let records = self.all(RecordQuery { scope: scope.map(str::to_string), trace: trace.cloned(), ..Default::default() }).await?;
        Ok(keyword_search(records, query, limit))
    }

    async fn extend(&self, records: Vec<MemoryRecord>) -> Result<()> {
        for r in records {
            self.append(r).await?;
        }
        Ok(())
    }

    /// Release whatever the backend holds open. Safe to call twice.
    async fn close(&self) {}

    /// What the store is, for messages.
    fn describe(&self) -> String {
        "memory store".into()
    }

    /// A session store on this backend's own connections, if it can hold
    /// sessions. `options` are the session store's own (`table`, `ttl`, ...).
    fn sessions(&self, _options: &Map<String, Value>) -> Option<Result<std::sync::Arc<dyn crate::runtime::session::SessionStore>>> {
        None
    }
}

/// The default. Fast, and gone when the process exits.
#[derive(Debug, Default)]
pub struct InMemoryStore {
    state: Mutex<(Vec<MemoryRecord>, HashMap<String, String>)>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl MemoryStore for InMemoryStore {
    async fn append(&self, record: MemoryRecord) -> Result<MemoryRecord> {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).0.push(record.clone());
        Ok(record)
    }

    async fn all(&self, query: RecordQuery) -> Result<Vec<MemoryRecord>> {
        let s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        Ok(last(s.0.iter().filter(|r| query.admits(r)).cloned().collect(), query.limit))
    }

    async fn clear(&self, scope: Option<&str>, trace: Option<&Trace>) -> Result<()> {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        s.0.retain(|r| scope.is_some_and(|sc| r.scope != sc) || trace.is_some_and(|t| !t.matches(r)));
        match trace {
            None if scope.is_none() => s.1.clear(),
            Some(t) => {
                let prefix = format!("{}/", t.slug());
                s.1.retain(|k, _| !k.starts_with(&prefix));
            }
            None => {}
        }
        Ok(())
    }

    async fn read_doc(&self, name: &str, trace: Option<&Trace>) -> Result<String> {
        Ok(self.state.lock().unwrap_or_else(|p| p.into_inner()).1.get(&doc_key(name, trace)).cloned().unwrap_or_default())
    }

    async fn write_doc(&self, name: &str, text: &str, trace: Option<&Trace>) -> Result<()> {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).1.insert(doc_key(name, trace), text.to_string());
        Ok(())
    }

    fn describe(&self) -> String {
        "in-memory".into()
    }
}

/// Durable memory on disk: one JSONL per scope, plain files for documents.
/// The same layout the Python `FileStore` writes.
///
/// Writes are small and append-only, so this stays cheap even on a hot loop.
#[derive(Debug)]
pub struct FileStore {
    pub root: PathBuf,
    lock: tokio::sync::Mutex<()>,
}

fn io(e: std::io::Error, path: &Path) -> Error {
    Error::Io(format!("{}: {e}", path.display()))
}

impl FileStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(|e| io(e, &root))?;
        Ok(Self { root, lock: tokio::sync::Mutex::new(()) })
    }

    fn path(&self, scope: &str) -> PathBuf {
        let safe: String = scope.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
        self.root.join(format!("{safe}.jsonl"))
    }

    fn files(&self, scope: Option<&str>) -> Vec<PathBuf> {
        if let Some(s) = scope {
            return vec![self.path(s)];
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&self.root)
            .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "jsonl")).collect())
            .unwrap_or_default();
        files.sort();
        files
    }

    /// Does this stored line belong to the trace? (Owned by it on some axis,
    /// and by nobody else on any.)
    fn belongs(line: &str, trace: &Trace) -> bool {
        let Ok(blob) = serde_json::from_str::<Value>(line) else { return false };
        let filters = trace.filters();
        filters.iter().all(|(f, v)| blob.get(*f).is_none_or(|x| x.is_null() || x.as_str() == Some(v.as_str())))
            && filters.iter().any(|(f, v)| blob.get(*f).and_then(Value::as_str) == Some(v.as_str()))
    }

    fn doc_path(&self, name: &str, trace: Option<&Trace>) -> PathBuf {
        self.root.join(doc_key(name, trace))
    }
}

#[async_trait]
impl MemoryStore for FileStore {
    async fn append(&self, record: MemoryRecord) -> Result<MemoryRecord> {
        use tokio::io::AsyncWriteExt;
        let line = format!("{}\n", serde_json::to_string(&record).unwrap_or_default());
        let path = self.path(&record.scope);
        let _held = self.lock.lock().await;
        let mut file = tokio::fs::OpenOptions::new().create(true).append(true).open(&path).await.map_err(|e| io(e, &path))?;
        file.write_all(line.as_bytes()).await.map_err(|e| io(e, &path))?;
        Ok(record)
    }

    async fn all(&self, query: RecordQuery) -> Result<Vec<MemoryRecord>> {
        let mut rows = Vec::new();
        for file in self.files(query.scope.as_deref()) {
            let Ok(text) = tokio::fs::read_to_string(&file).await else { continue };
            // A half-written line never kills a run.
            rows.extend(text.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str::<MemoryRecord>(l).ok()));
        }
        rows.retain(|r| query.kind.as_ref().is_none_or(|k| &r.kind == k) && query.trace.as_ref().is_none_or(|t| t.matches(r)));
        rows.sort_by(|a, b| a.ts.partial_cmp(&b.ts).unwrap_or(std::cmp::Ordering::Equal));
        Ok(last(rows, query.limit))
    }

    async fn clear(&self, scope: Option<&str>, trace: Option<&Trace>) -> Result<()> {
        let _held = self.lock.lock().await;
        let Some(trace) = trace else {
            for file in self.files(scope) {
                if file.exists() {
                    tokio::fs::remove_file(&file).await.map_err(|e| io(e, &file))?;
                }
            }
            return Ok(());
        };
        // Selective delete: rewrite each file without this trace's rows.
        for file in self.files(scope) {
            let Ok(text) = tokio::fs::read_to_string(&file).await else { continue };
            let kept: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty() && !Self::belongs(l, trace)).collect();
            let body = if kept.is_empty() { String::new() } else { format!("{}\n", kept.join("\n")) };
            tokio::fs::write(&file, body).await.map_err(|e| io(e, &file))?;
        }
        let folder = self.root.join(trace.slug());
        if let Ok(mut entries) = tokio::fs::read_dir(&folder).await {
            while let Ok(Some(e)) = entries.next_entry().await {
                let _ = tokio::fs::remove_file(e.path()).await;
            }
        }
        Ok(())
    }

    async fn read_doc(&self, name: &str, trace: Option<&Trace>) -> Result<String> {
        let path = self.doc_path(name, trace);
        match tokio::fs::read_to_string(&path).await {
            Ok(t) => Ok(t),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
            Err(e) => Err(io(e, &path)),
        }
    }

    async fn write_doc(&self, name: &str, text: &str, trace: Option<&Trace>) -> Result<()> {
        let path = self.doc_path(name, trace);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| io(e, parent))?;
        }
        tokio::fs::write(&path, text).await.map_err(|e| io(e, &path))
    }

    fn describe(&self) -> String {
        format!("files at {}", self.root.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn exercise(store: &dyn MemoryStore) {
        let alice = Trace::user("alice");
        let mut a = MemoryRecord::new("user", "decision", "Bill in EUR always");
        alice.stamp(&mut a);
        store.append(a).await.unwrap();
        let mut b = MemoryRecord::new("user", "fact", "Bob likes tea");
        Trace::user("bob").stamp(&mut b);
        store.append(b).await.unwrap();
        store.append(MemoryRecord::new("project", "note", "Shared euro note")).await.unwrap();

        let mine = store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap();
        assert_eq!(mine.len(), 2, "her own and the shared one");
        let found = store.search("bill EUR always", None, 5, Some(&alice)).await.unwrap();
        assert_eq!(found[0].text, "Bill in EUR always");
        assert_eq!(store.all(RecordQuery::scope("project")).await.unwrap().len(), 1);
        assert_eq!(store.all(RecordQuery::default().limit(1)).await.unwrap().len(), 1);

        store.write_doc("user.md", "- likes EUR", Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "- likes EUR");
        assert_eq!(store.read_doc("user.md", Some(&Trace::user("bob"))).await.unwrap(), "");
        store.clear(None, Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "");
        let left = store.all(RecordQuery::default()).await.unwrap();
        assert!(left.iter().all(|r| r.user_id.as_deref() != Some("alice")));
        assert!(left.iter().any(|r| r.text == "Bob likes tea"));
        store.clear(None, None).await.unwrap();
        assert!(store.all(RecordQuery::default()).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn stores_scope_by_trace() {
        exercise(&InMemoryStore::new()).await;
        let dir = std::env::temp_dir().join(format!("ah_mem_{}", uuid::Uuid::new_v4().simple()));
        exercise(&FileStore::new(&dir).unwrap()).await;
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn records_read_python_json() {
        let r: MemoryRecord = serde_json::from_str(r#"{"id":"mem_1","scope":"user","kind":"fact","text":"t","ts":1.5,"user_id":null}"#).unwrap();
        assert_eq!((r.id.as_str(), r.line().as_str(), r.ts), ("mem_1", "- [fact] t", 1.5));
        let back = serde_json::to_value(&r).unwrap();
        assert!(back.get("embedding").is_none());
        assert_eq!(MemoryRecord::new("user", "note", "x").line(), "- x");
    }
}
