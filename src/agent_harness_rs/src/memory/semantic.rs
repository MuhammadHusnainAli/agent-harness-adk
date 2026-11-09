//! Semantic recall: answer from precedent instead of from scratch.
//!
//! Embeddings come from whatever you give it. With no embedder configured it
//! falls back to a deterministic hashing embedder — offline, free, no extra
//! dependency, and good enough to surface records that share vocabulary with
//! the query. Swap in `ProviderEmbedder::new(openai, ...)` for real semantics.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::errors::{Error, Result};
use crate::knowledge::base::{VectorRecord, VectorStore};
use crate::llm_providers::base::Provider;
use crate::llm_providers::fake::hash_embedding;
use crate::memory::base::{MemoryRecord, MemoryStore, RecordQuery};
use crate::memory::trace::Trace;

pub use crate::knowledge::base::cosine;

/// Turns text into vectors.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Zero-dependency, deterministic, offline. The default — and the same
/// vectors the Python `HashEmbedder` makes.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    pub dims: usize,
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self { dims: 256 }
    }
}

#[async_trait]
impl Embedder for HashEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| hash_embedding(t, self.dims)).collect())
    }
}

/// Real embeddings from a provider that has an embeddings endpoint.
pub struct ProviderEmbedder {
    pub provider: Arc<dyn Provider>,
    pub model: Option<String>,
    pub batch: usize,
}

impl ProviderEmbedder {
    pub fn new(provider: Arc<dyn Provider>, model: Option<&str>) -> Self {
        Self { provider, model: model.map(str::to_string), batch: 64 }
    }
}

#[async_trait]
impl Embedder for ProviderEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch.max(1)) {
            out.extend(self.provider.embed(chunk, self.model.as_deref()).await?);
        }
        Ok(out)
    }
}

/// A brute-force cosine index. Exact, dependency-free, fine to ~50k rows.
///
/// With a path, every row is appended to a JSONL file (the record plus its
/// `_vector`) and read back on start — the Python `VectorStore` file format.
#[derive(Debug, Default)]
pub struct LocalIndex {
    path: Option<PathBuf>,
    rows: Mutex<Vec<(Vec<f32>, MemoryRecord)>>,
}

impl LocalIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn at(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let mut rows = Vec::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let Ok(Value::Object(mut blob)) = serde_json::from_str::<Value>(line) else { continue };
                let vector: Vec<f32> = blob.remove("_vector").and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default();
                if let Ok(rec) = serde_json::from_value(Value::Object(blob)) {
                    rows.push((vector, rec));
                }
            }
        }
        Self { path: Some(path), rows: Mutex::new(rows) }
    }

    pub fn add(&self, vector: Vec<f32>, record: MemoryRecord) -> Result<()> {
        if let Some(path) = &self.path {
            use std::io::Write;
            let mut blob = serde_json::to_value(&record).unwrap_or_else(|_| json!({}));
            blob["_vector"] = json!(vector);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
            writeln!(file, "{blob}").map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        }
        self.rows.lock().unwrap_or_else(|p| p.into_inner()).push((vector, record));
        Ok(())
    }

    pub fn search(&self, vector: &[f32], limit: usize, scope: Option<&str>, min_score: f64, trace: Option<&Trace>) -> Vec<(f64, MemoryRecord)> {
        let rows = self.rows.lock().unwrap_or_else(|p| p.into_inner());
        let mut scored: Vec<(f64, MemoryRecord)> = rows
            .iter()
            .filter(|(_, r)| scope.is_none_or(|s| r.scope == s) && trace.is_none_or(|t| t.matches(r)))
            .map(|(v, r)| (cosine(vector, v), r.clone()))
            .filter(|(s, _)| *s > min_score)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored
    }

    pub fn len(&self) -> usize {
        self.rows.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        self.rows.lock().unwrap_or_else(|p| p.into_inner()).clear();
        if let Some(p) = &self.path {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Memory's index in a vector database — past what one process can scan.
///
/// Each record is stored with its scope and its owner, and a search is held to
/// them in the database, not after the fact — so a tenant's memory is never
/// even fetched for another. A record with no owner on an axis is shared, as
/// it is everywhere else.
pub struct DatabaseIndex {
    pub store: Arc<dyn VectorStore>,
    ready: AtomicBool,
}

impl DatabaseIndex {
    pub fn new(store: Arc<dyn VectorStore>) -> Self {
        Self { store, ready: AtomicBool::new(false) }
    }

    async fn ready(&self, dimension: usize) -> Result<()> {
        if !self.ready.load(Ordering::SeqCst) {
            self.store.ensure(dimension).await?;
            self.ready.store(true, Ordering::SeqCst);
        }
        Ok(())
    }

    pub async fn add(&self, vector: Vec<f32>, record: &MemoryRecord) -> Result<()> {
        self.ready(vector.len()).await?;
        let meta = json!({
            "scope": record.scope, "kind": record.kind,
            // "" for "nobody in particular", so it can be asked for by name.
            "user_id": record.user_id.clone().unwrap_or_default(),
            "tenant_id": record.tenant_id.clone().unwrap_or_default(),
            "session_id": record.session_id.clone().unwrap_or_default(),
            "record": serde_json::to_string(record).unwrap_or_default(),
        });
        self.store.upsert(vec![VectorRecord::new(record.id.clone(), vector, record.text.clone(), meta)]).await?;
        Ok(())
    }

    pub async fn search(&self, vector: &[f32], limit: usize, scope: Option<&str>, min_score: f64, trace: Option<&Trace>) -> Result<Vec<(f64, MemoryRecord)>> {
        self.ready(vector.len()).await?;
        let mut wanted = serde_json::Map::new();
        if let Some(s) = scope {
            wanted.insert("scope".into(), json!(s));
        }
        for (axis, value) in trace.map(Trace::filters).unwrap_or_default() {
            wanted.insert(axis.into(), json!({"$in": [value, ""]}));
        }
        let filter = (!wanted.is_empty()).then_some(Value::Object(wanted));
        let hits = self.store.query(vector, limit, filter.as_ref()).await?;
        Ok(hits
            .into_iter()
            .filter(|h| h.score > min_score)
            .filter_map(|h| {
                let rec = h.metadata.get("record")?.as_str()?;
                serde_json::from_str(rec).ok().map(|r| (h.score, r))
            })
            .collect())
    }

    pub async fn clear(&self) -> Result<()> {
        self.store.clear().await?;
        self.ready.store(false, Ordering::SeqCst);
        Ok(())
    }
}

/// Where semantic memory keeps its vectors.
pub enum MemoryIndex {
    /// In this process: exact, good to about fifty thousand records.
    Local(LocalIndex),
    /// In a vector database: good to as many as the database is.
    Database(DatabaseIndex),
}

impl From<Arc<dyn VectorStore>> for MemoryIndex {
    fn from(store: Arc<dyn VectorStore>) -> Self {
        Self::Database(DatabaseIndex::new(store))
    }
}

impl From<LocalIndex> for MemoryIndex {
    fn from(index: LocalIndex) -> Self {
        Self::Local(index)
    }
}

/// A `MemoryStore` that also retrieves by meaning. Wraps any other store.
pub struct SemanticMemory {
    pub store: Arc<dyn MemoryStore>,
    pub embedder: Arc<dyn Embedder>,
    pub index: MemoryIndex,
    pub min_score: f64,
}

impl SemanticMemory {
    /// Over `store`, with the hashing embedder and an in-process index.
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        Self { store, embedder: Arc::new(HashEmbedder::default()), index: MemoryIndex::Local(LocalIndex::new()), min_score: 0.05 }
    }

    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = embedder;
        self
    }

    pub fn index(mut self, index: impl Into<MemoryIndex>) -> Self {
        self.index = index.into();
        self
    }

    pub fn min_score(mut self, score: f64) -> Self {
        self.min_score = score;
        self
    }

    /// The index is a database: it is asked, not scanned.
    pub fn remote(&self) -> bool {
        matches!(self.index, MemoryIndex::Database(_))
    }

    async fn hits(&self, query: &str, scope: Option<&str>, limit: usize, trace: Option<&Trace>) -> Result<Vec<(f64, MemoryRecord)>> {
        let vector = self.embedder.embed(&[query.to_string()]).await?.into_iter().next().unwrap_or_default();
        match &self.index {
            MemoryIndex::Local(i) => Ok(i.search(&vector, limit, scope, self.min_score, trace)),
            MemoryIndex::Database(d) => d.search(&vector, limit, scope, self.min_score, trace).await,
        }
    }

    /// Hits with their similarity, best first.
    pub async fn search_scored(&self, query: &str, scope: Option<&str>, limit: usize, trace: Option<&Trace>) -> Result<Vec<(f64, MemoryRecord)>> {
        self.hits(query, scope, limit, trace).await
    }
}

#[async_trait]
impl MemoryStore for SemanticMemory {
    async fn append(&self, record: MemoryRecord) -> Result<MemoryRecord> {
        let record = self.store.append(record).await?;
        if !record.text.trim().is_empty() {
            let vector = match &record.embedding {
                Some(v) => v.clone(),
                None => self.embedder.embed(std::slice::from_ref(&record.text)).await?.into_iter().next().unwrap_or_default(),
            };
            match &self.index {
                MemoryIndex::Local(i) => i.add(vector, record.clone())?,
                MemoryIndex::Database(d) => d.add(vector, &record).await?,
            }
        }
        Ok(record)
    }

    async fn all(&self, query: RecordQuery) -> Result<Vec<MemoryRecord>> {
        self.store.all(query).await
    }

    async fn clear(&self, scope: Option<&str>, trace: Option<&Trace>) -> Result<()> {
        self.store.clear(scope, trace).await?;
        if scope.is_none() && trace.is_none() {
            match &self.index {
                MemoryIndex::Local(i) => i.clear(),
                MemoryIndex::Database(d) => d.clear().await?,
            }
        }
        Ok(())
    }

    async fn read_doc(&self, name: &str, trace: Option<&Trace>) -> Result<String> {
        self.store.read_doc(name, trace).await
    }

    async fn write_doc(&self, name: &str, text: &str, trace: Option<&Trace>) -> Result<()> {
        self.store.write_doc(name, text, trace).await
    }

    async fn close(&self) {
        self.store.close().await;
    }

    async fn search(&self, query: &str, scope: Option<&str>, limit: usize, trace: Option<&Trace>) -> Result<Vec<MemoryRecord>> {
        let empty = matches!(&self.index, MemoryIndex::Local(i) if i.is_empty());
        if query.trim().is_empty() || empty {
            return self.store.search(query, scope, limit, trace).await;
        }
        let hits = self.hits(query, scope, limit, trace).await?;
        if hits.is_empty() {
            return self.store.search(query, scope, limit, trace).await;
        }
        Ok(hits.into_iter().map(|(_, r)| r).collect())
    }

    fn describe(&self) -> String {
        format!("semantic over {}", self.store.describe())
    }

    /// The sessions go underneath the semantic layer, on the real store.
    fn sessions(&self, options: &serde_json::Map<String, Value>) -> Option<Result<Arc<dyn crate::runtime::session::SessionStore>>> {
        self.store.sessions(options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::stores::MemoryVectorStore;
    use crate::memory::base::InMemoryStore;

    async fn recall_with(mem: SemanticMemory) {
        let alice = Trace::user("alice");
        for (text, who) in [("The invoice currency is EUR", "alice"), ("Deploys happen on Tuesday", "alice"), ("The invoice currency is GBP", "bob")] {
            let mut r = MemoryRecord::new("project", "fact", text);
            Trace::user(who).stamp(&mut r);
            mem.append(r).await.unwrap();
        }
        let hits = mem.search("which currency for the invoice", None, 2, Some(&alice)).await.unwrap();
        assert_eq!(hits[0].text, "The invoice currency is EUR");
        assert!(hits.iter().all(|h| h.user_id.as_deref() == Some("alice")));
        let scored = mem.search_scored("deploys tuesday", None, 1, Some(&alice)).await.unwrap();
        assert!(scored[0].0 > 0.3, "{scored:?}");
        mem.clear(None, None).await.unwrap();
        assert!(mem.search("currency", None, 5, None).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn recalls_by_meaning_in_process_and_in_a_database() {
        recall_with(SemanticMemory::new(Arc::new(InMemoryStore::new()))).await;
        let db: Arc<dyn VectorStore> = Arc::new(MemoryVectorStore::new());
        recall_with(SemanticMemory::new(Arc::new(InMemoryStore::new())).index(db)).await;
    }

    #[tokio::test]
    async fn local_index_persists() {
        let path = std::env::temp_dir().join(format!("ah_idx_{}.jsonl", uuid::Uuid::new_v4().simple()));
        let idx = LocalIndex::at(&path);
        idx.add(vec![1.0, 0.0], MemoryRecord::new("user", "fact", "a")).unwrap();
        let again = LocalIndex::at(&path);
        assert_eq!(again.len(), 1);
        assert_eq!(again.search(&[1.0, 0.0], 5, None, 0.0, None)[0].1.text, "a");
        again.clear();
        assert!(!path.exists());
    }
}
