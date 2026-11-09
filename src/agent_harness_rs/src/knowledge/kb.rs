//! A knowledge base: documents in, the passages that answer a question out.
//!
//! ```no_run
//! # async fn demo() -> agent_harness::Result<()> {
//! use agent_harness::knowledge::{KnowledgeBase, NewDocument};
//!
//! let kb = KnowledgeBase::builder().store("qdrant://localhost:6333/handbook").build()?;
//! kb.add(NewDocument::path("handbook/refunds.md")).await?;
//! kb.add(NewDocument::text("Orders ship within two working days.").id("shipping").title("Shipping")).await?;
//! let passages = kb.search("how long do refunds take?", 5, None).await?;
//! let tool = kb.as_tool("search_knowledge");
//! # Ok(()) }
//! ```
//!
//! Adding a document cuts it into overlapping passages at paragraph and
//! sentence boundaries, embeds them in batches, and stores them under the
//! document's id. Adding it again replaces it — and when nothing in it has
//! changed, nothing is embedded or written. Removing it removes every passage.
//!
//! A search embeds the question, takes more candidates than were asked for,
//! and orders them by meaning and by the words they share with the question,
//! so an exact term — a product code, a name — is not lost to a near-synonym.
//!
//! With no embedder a hashing one is used: free and offline, and blind to
//! meaning — fine for tests, not for answers.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use futures::future::BoxFuture;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::errors::{Error, Result};
use crate::knowledge::base::{VectorHit, VectorRecord, VectorStore, conditions, store_error};
use crate::knowledge::stores::vector_store;
use crate::memory::semantic::{Embedder, HashEmbedder};
use crate::tools::Tool;

static PARAGRAPH: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n\s*\n").expect("valid"));
static HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(#{1,6})\s+(.+?)\s*#*$").expect("valid"));
static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[0-9A-Za-z\u{C0}-\u{FFFF}]+").expect("valid"));
/// Metadata the knowledge base writes itself. A document's own may not use them.
pub const RESERVED: [&str; 7] = ["doc", "chunk", "title", "source", "doc_hash", "doc_chunks", "namespace"];

fn clen(s: &str) -> usize {
    s.chars().count()
}

/// Split after `.!?。！？` followed by whitespace — the sentence boundaries.
fn sentences(block: &str) -> Vec<String> {
    let chars: Vec<char> = block.chars().collect();
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        if i > 0 && chars[i].is_whitespace() && matches!(chars[i - 1], '.' | '!' | '?' | '。' | '！' | '？') {
            out.push(chars[start..i].iter().collect());
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            start = i;
            continue;
        }
        i += 1;
    }
    out.push(chars[start..].iter().collect());
    out
}

/// Cut text into passages of about `size` characters.
///
/// Cuts fall between paragraphs where they can, between sentences where a
/// paragraph is too long, and inside a sentence only when the sentence itself
/// is. Each passage starts with the tail of the one before, so a fact that
/// straddles a cut is whole in one of them — and under the headings it sits
/// beneath, so "it takes five days" still says *what* takes five days.
pub fn chunk_text(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let size = size.max(200);
    let overlap = overlap.min(size / 2);
    let mut pieces: Vec<(String, String)> = Vec::new();
    let mut trail: Vec<(usize, String)> = Vec::new();
    let normalised = text.replace("\r\n", "\n");
    for block in PARAGRAPH.split(&normalised) {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let first = block.split('\n').next().unwrap_or("");
        if let Some(h) = HEADING.captures(first) {
            let level = h[1].len();
            trail.retain(|(lv, _)| *lv < level);
            trail.push((level, h[2].to_string()));
        }
        let context = trail.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>().join(" > ");
        if clen(block) <= size {
            pieces.push((context, block.to_string()));
            continue;
        }
        for sentence in sentences(block) {
            let mut sentence: Vec<char> = sentence.chars().collect();
            while sentence.len() > size {
                // One sentence longer than a passage: cut at a space in its second half.
                let cut = (size / 2..size).rev().find(|&i| sentence[i] == ' ').filter(|&c| c > 0).unwrap_or(size);
                pieces.push((context.clone(), sentence[..cut].iter().collect::<String>().trim().to_string()));
                sentence = sentence[cut..].iter().collect::<String>().trim().chars().collect();
            }
            if !sentence.is_empty() {
                pieces.push((context.clone(), sentence.into_iter().collect()));
            }
        }
    }

    let mut chunks: Vec<String> = Vec::new();
    let (mut current, mut place, mut tail) = (String::new(), String::new(), String::new());
    for (context, piece) in pieces {
        if !current.is_empty() && (clen(&current) + clen(&piece) + 2 > size || context != place) {
            chunks.push(current.clone());
            tail = if overlap > 0 && context == place {
                let n = clen(&current);
                current.chars().skip(n.saturating_sub(overlap)).collect()
            } else {
                String::new()
            };
            // Start the overlap at a word, not in the middle of one.
            tail = match tail.find(' ') {
                Some(i) => tail[i + 1..].to_string(),
                None => String::new(),
            };
            current.clear();
        }
        if current.is_empty() {
            // Every passage says where in the document it is from.
            let label = if !context.is_empty() && !piece.trim_start().starts_with('#') { format!("[{context}]\n") } else { String::new() };
            current = if tail.is_empty() { format!("{label}{piece}") } else { format!("{label}{tail} {piece}").trim().to_string() };
            tail.clear();
        } else {
            current = format!("{current} {piece}");
        }
        place = context;
    }
    if !current.trim().is_empty() {
        chunks.push(current);
    }
    chunks.into_iter().map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect()
}

/// One stretch of a document that a search found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Passage {
    pub text: String,
    /// Cosine similarity between the question and this passage.
    pub score: f64,
    #[serde(default)]
    pub document: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub chunk: usize,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default)]
    pub id: String,
}

impl Passage {
    pub fn to_json(&self) -> Value {
        let mut out = json!({"text": self.text, "score": (self.score * 1000.0).round() / 1000.0});
        for (k, v) in [("title", &self.title), ("source", &self.source), ("document", &self.document)] {
            if !v.is_empty() {
                out[k] = json!(v);
            }
        }
        out
    }
}

/// What adding a document came to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub chunks: usize,
    pub title: String,
    pub source: String,
    /// True when it was already there, unchanged, and nothing was written.
    pub unchanged: bool,
}

/// A document to add: its text, a file, or a URL, and what it is called.
#[derive(Debug, Clone, Default)]
pub struct NewDocument {
    pub text: Option<String>,
    pub path: Option<std::path::PathBuf>,
    pub url: Option<String>,
    pub id: Option<String>,
    pub title: String,
    pub source: String,
    pub metadata: Map<String, Value>,
}

impl NewDocument {
    pub fn text(text: impl Into<String>) -> Self {
        Self { text: Some(text.into()), ..Default::default() }
    }

    /// A text file: plain text, Markdown, HTML, CSV, JSON.
    pub fn path(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: Some(path.into()), ..Default::default() }
    }

    pub fn url(url: impl Into<String>) -> Self {
        Self { url: Some(url.into()), ..Default::default() }
    }

    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    pub fn meta(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// `(query, passages) -> one score per passage` — a cross-encoder, a rerank API.
pub type Reranker = Arc<dyn Fn(String, Vec<Passage>) -> BoxFuture<'static, Result<Vec<f64>>> + Send + Sync>;

/// Something that does its own embedding and searching — a Bedrock knowledge
/// base, say. A knowledge base built on one reads from it and cannot add to it.
#[async_trait]
pub trait Retriever: Send + Sync {
    async fn retrieve(&self, query: &str, k: usize, filter: Option<&Value>) -> Result<Vec<VectorHit>>;
}

/// Builds a [`KnowledgeBase`].
#[derive(Default)]
pub struct KnowledgeBaseBuilder {
    store: Option<Arc<dyn VectorStore>>,
    store_spec: Option<String>,
    retriever: Option<Arc<dyn Retriever>>,
    embedder: Option<Arc<dyn Embedder>>,
    namespace: String,
    chunk_size: Option<usize>,
    chunk_overlap: Option<usize>,
    hybrid: Option<bool>,
    reranker: Option<Reranker>,
    min_score: f64,
    candidates: Option<usize>,
    batch: Option<usize>,
    concurrency: Option<usize>,
}

impl KnowledgeBaseBuilder {
    /// A store by name or URL — `"qdrant://host/handbook"`.
    pub fn store(mut self, spec: &str) -> Self {
        self.store_spec = Some(spec.into());
        self
    }

    pub fn vector_store(mut self, store: Arc<dyn VectorStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn retriever(mut self, retriever: Arc<dyn Retriever>) -> Self {
        self.retriever = Some(retriever);
        self
    }

    pub fn embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Keeps one knowledge base apart from another in the same store: every
    /// write is marked with it and every search held to it.
    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    pub fn chunk_size(mut self, chars: usize) -> Self {
        self.chunk_size = Some(chars);
        self
    }

    pub fn chunk_overlap(mut self, chars: usize) -> Self {
        self.chunk_overlap = Some(chars);
        self
    }

    /// Order results by shared words as well as by meaning (on by default).
    pub fn hybrid(mut self, on: bool) -> Self {
        self.hybrid = Some(on);
        self
    }

    pub fn reranker(mut self, reranker: Reranker) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Drop passages less similar than this.
    pub fn min_score(mut self, score: f64) -> Self {
        self.min_score = score;
        self
    }

    /// How many candidates per result to fetch before re-ordering.
    pub fn candidates(mut self, n: usize) -> Self {
        self.candidates = Some(n);
        self
    }

    pub fn batch(mut self, n: usize) -> Self {
        self.batch = Some(n);
        self
    }

    pub fn concurrency(mut self, n: usize) -> Self {
        self.concurrency = Some(n);
        self
    }

    pub fn build(self) -> Result<KnowledgeBase> {
        if self.retriever.is_some() && (self.store.is_some() || self.store_spec.is_some()) {
            return Err(Error::config("a knowledge base has a store or a retriever, not both"));
        }
        let store = match (self.store, self.store_spec, &self.retriever) {
            (_, _, Some(_)) => None,
            (Some(s), _, _) => Some(s),
            (None, Some(spec), _) => Some(vector_store(&spec)?),
            (None, None, _) => Some(vector_store("memory")?),
        };
        Ok(KnowledgeBase {
            inner: Arc::new(KbInner {
                store,
                retriever: self.retriever,
                embedder: self.embedder.unwrap_or_else(|| Arc::new(HashEmbedder::default())),
                namespace: self.namespace,
                chunk_size: self.chunk_size.unwrap_or(1200),
                chunk_overlap: self.chunk_overlap.unwrap_or(150),
                hybrid: self.hybrid.unwrap_or(true),
                reranker: self.reranker,
                min_score: self.min_score,
                candidates: self.candidates.unwrap_or(4).max(1),
                batch: self.batch.unwrap_or(64).max(1),
                concurrency: self.concurrency.unwrap_or(4).max(1),
                ready: AtomicBool::new(false),
                dimension: AtomicUsize::new(0),
            }),
        })
    }
}

struct KbInner {
    store: Option<Arc<dyn VectorStore>>,
    retriever: Option<Arc<dyn Retriever>>,
    embedder: Arc<dyn Embedder>,
    namespace: String,
    chunk_size: usize,
    chunk_overlap: usize,
    hybrid: bool,
    reranker: Option<Reranker>,
    min_score: f64,
    candidates: usize,
    batch: usize,
    concurrency: usize,
    ready: AtomicBool,
    dimension: AtomicUsize,
}

/// Documents, searchable by meaning, in any vector database. Cheap to clone.
#[derive(Clone)]
pub struct KnowledgeBase {
    inner: Arc<KbInner>,
}

impl std::fmt::Debug for KnowledgeBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<KnowledgeBase {}>", self.inner.store.as_ref().map_or("retriever", |s| s.name()))
    }
}

fn kb_error(message: impl Into<String>) -> Error {
    Error::tool(message, "knowledge")
}

impl KnowledgeBase {
    pub fn builder() -> KnowledgeBaseBuilder {
        KnowledgeBaseBuilder::default()
    }

    /// A knowledge base in a store, with the hashing embedder.
    pub fn new(store: &str) -> Result<Self> {
        Self::builder().store(store).build()
    }

    pub fn store(&self) -> Option<&Arc<dyn VectorStore>> {
        self.inner.store.as_ref()
    }

    pub fn namespace(&self) -> &str {
        &self.inner.namespace
    }

    // ---- embedding ---------------------------------------------------------
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let me = &self.inner;
        let mut out = Vec::with_capacity(texts.len());
        for part in texts.chunks(me.batch) {
            let vectors = me.embedder.embed(part).await?;
            if vectors.len() != part.len() {
                return Err(store_error("", format!("the embedder returned {} vectors for {} texts", vectors.len(), part.len())));
            }
            out.extend(vectors);
        }
        if !out.is_empty() {
            let mut sizes: Vec<usize> = out.iter().map(Vec::len).collect::<HashSet<_>>().into_iter().collect();
            sizes.sort();
            let held = me.dimension.load(Ordering::SeqCst);
            if sizes.len() != 1 || (held != 0 && sizes[0] != held) {
                let also = if held != 0 { format!("; this knowledge base holds {held}") } else { String::new() };
                return Err(store_error("", format!("the embedder returned vectors of {sizes:?} dimensions{also}")));
            }
            me.dimension.store(sizes[0], Ordering::SeqCst);
        }
        Ok(out)
    }

    async fn prepare(&self) -> Result<&Arc<dyn VectorStore>> {
        let Some(store) = &self.inner.store else {
            return Err(store_error("", "this knowledge base reads from a retriever; documents are added where the retriever keeps them"));
        };
        if !self.inner.ready.load(Ordering::SeqCst) {
            if self.inner.dimension.load(Ordering::SeqCst) == 0 {
                self.embed(&["dimension probe".to_string()]).await?;
            }
            store.ensure(self.inner.dimension.load(Ordering::SeqCst)).await?;
            self.inner.ready.store(true, Ordering::SeqCst);
        }
        Ok(store)
    }

    fn scope(&self, filter: Option<&Value>) -> Option<Value> {
        if self.inner.namespace.is_empty() {
            return filter.filter(|f| f.as_object().is_some_and(|m| !m.is_empty())).cloned();
        }
        let mut f = filter.and_then(Value::as_object).cloned().unwrap_or_default();
        f.insert("namespace".into(), json!(self.inner.namespace));
        Some(Value::Object(f))
    }

    fn chunk_id(&self, document: &str, number: usize) -> String {
        if self.inner.namespace.is_empty() { format!("{document}#{number}") } else { format!("{}/{document}#{number}", self.inner.namespace) }
    }

    // ---- documents ---------------------------------------------------------
    /// What is stored of a document already: its hash and how many passages.
    async fn existing(&self, store: &Arc<dyn VectorStore>, document: &str, vector: &[f32]) -> Map<String, Value> {
        match store.query(vector, 1, self.scope(Some(&json!({"doc": document}))).as_ref()).await {
            Ok(hits) => hits.into_iter().next().map(|h| h.metadata).unwrap_or_default(),
            Err(_) => Map::new(),
        }
    }

    /// Add a document — or replace the one with the same id.
    pub async fn add(&self, doc: NewDocument) -> Result<Document> {
        let given = [doc.text.is_some(), doc.path.is_some(), doc.url.is_some()].iter().filter(|x| **x).count();
        if given != 1 {
            return Err(kb_error("add takes one of: text, path, url"));
        }
        let (mut title, mut source) = (doc.title.clone(), doc.source.clone());
        let text = if let Some(path) = &doc.path {
            let text = tokio::fs::read(path).await.map_err(|e| kb_error(format!("could not read {}: {e}", path.display())))?;
            if source.is_empty() {
                source = path.display().to_string();
            }
            if title.is_empty() {
                title = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            }
            let text = String::from_utf8_lossy(&text).into_owned();
            if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("html") || e.eq_ignore_ascii_case("htm")) { html_text(&text) } else { text }
        } else if let Some(url) = &doc.url {
            crate::llm_providers::base::install_tls();
            let response = reqwest::get(url).await.map_err(|e| kb_error(format!("could not fetch {url}: {e}")))?;
            let html = response.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|v| v.contains("html"));
            let body = response.text().await.map_err(|e| kb_error(format!("could not read {url}: {e}")))?;
            if source.is_empty() {
                source = url.clone();
            }
            if title.is_empty() {
                title = url.clone();
            }
            if html { html_text(&body) } else { body }
        } else {
            doc.text.clone().unwrap_or_default()
        };
        let text = text.replace('\0', "").trim().to_string();
        if text.is_empty() {
            return Err(kb_error("there is no text in that document"));
        }
        let clash: Vec<&str> = doc.metadata.keys().filter(|k| RESERVED.contains(&k.as_str())).map(String::as_str).collect();
        if !clash.is_empty() {
            return Err(kb_error(format!("metadata may not use the names {} — the knowledge base writes those itself", clash.join(", "))));
        }
        let document = doc
            .id
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| if source.is_empty() { hex(&Sha256::digest(text.as_bytes()))[..16].to_string() } else { source.clone() });
        let me = &self.inner;
        let chunks = chunk_text(&text, me.chunk_size, me.chunk_overlap);
        let mut extra: Vec<(&String, &Value)> = doc.metadata.iter().collect();
        extra.sort_by(|a, b| a.0.cmp(b.0));
        let fingerprint = json!([text, extra, title, source, me.chunk_size, me.chunk_overlap]).to_string();
        let digest = hex(&Sha256::digest(fingerprint.as_bytes()))[..24].to_string();

        let store = self.prepare().await?.clone();
        let first = self.embed(&chunks[..1]).await?.remove(0);
        let held = self.existing(&store, &document, &first).await;
        let held_chunks = held.get("doc_chunks").and_then(Value::as_u64).unwrap_or(0) as usize;
        if held.get("doc_hash").and_then(Value::as_str) == Some(digest.as_str()) && held_chunks == chunks.len() {
            return Ok(Document { id: document, chunks: chunks.len(), title, source, unchanged: true });
        }
        let mut vectors = vec![first];
        if chunks.len() > 1 {
            vectors.extend(self.embed(&chunks[1..]).await?);
        }
        let mut base = doc.metadata.clone();
        base.insert("doc".into(), json!(document));
        base.insert("title".into(), json!(title));
        base.insert("source".into(), json!(source));
        base.insert("doc_hash".into(), json!(digest));
        base.insert("doc_chunks".into(), json!(chunks.len()));
        if !me.namespace.is_empty() {
            base.insert("namespace".into(), json!(me.namespace));
        }
        let records = chunks
            .iter()
            .zip(vectors)
            .enumerate()
            .map(|(n, (chunk, vector))| {
                let mut meta = base.clone();
                meta.insert("chunk".into(), json!(n));
                VectorRecord { id: self.chunk_id(&document, n), vector, text: chunk.clone(), metadata: meta }
            })
            .collect();
        store.upsert(records).await?;
        // It was longer before: the passages past its new end are no longer its.
        if held_chunks > chunks.len() {
            store.delete(Some((chunks.len()..held_chunks).map(|n| self.chunk_id(&document, n)).collect()), None).await?;
        }
        Ok(Document { id: document, chunks: chunks.len(), title, source, unchanged: false })
    }

    /// Add several, a few at a time.
    pub async fn add_many(&self, documents: Vec<NewDocument>) -> Result<Vec<Document>> {
        use futures::stream::{self, StreamExt, TryStreamExt};
        self.prepare().await?; // once, not once per document
        stream::iter(documents.into_iter().map(|d| self.add(d))).buffered(self.inner.concurrency).try_collect().await
    }

    /// Remove a document and every passage of it.
    pub async fn delete(&self, document: &str) -> Result<()> {
        let store = self.prepare().await?.clone();
        if store.delete(None, self.scope(Some(&json!({"doc": document}))).as_ref()).await.is_ok() {
            return Ok(());
        }
        // A store that deletes by id only: ask how many passages there are.
        let probe = self.embed(&[document.to_string()]).await?.remove(0);
        let held = self.existing(&store, document, &probe).await;
        let count = held.get("doc_chunks").and_then(Value::as_u64).unwrap_or(0) as usize;
        if count > 0 {
            store.delete(Some((0..count).map(|n| self.chunk_id(document, n)).collect()), None).await?;
        }
        Ok(())
    }

    /// How many passages are stored (in the whole store, not this namespace).
    pub async fn count(&self) -> Result<usize> {
        self.prepare().await?.count().await
    }

    // ---- searching ---------------------------------------------------------
    /// How much each candidate shares the question's words — rare ones most (BM25).
    fn lexical(query: &str, hits: &[VectorHit]) -> Vec<f64> {
        let terms: HashSet<String> = WORD.find_iter(query).map(|m| m.as_str().to_lowercase()).filter(|w| clen(w) > 1).collect();
        if terms.is_empty() || hits.is_empty() {
            return vec![0.0; hits.len()];
        }
        let bags: Vec<HashMap<String, usize>> = hits
            .iter()
            .map(|h| {
                let mut bag = HashMap::new();
                for w in WORD.find_iter(&h.text) {
                    *bag.entry(w.as_str().to_lowercase()).or_insert(0) += 1;
                }
                bag
            })
            .collect();
        let total: usize = bags.iter().map(|b| b.values().sum::<usize>()).sum();
        let average = if total == 0 { 1.0 } else { total as f64 / bags.len() as f64 };
        bags.iter()
            .map(|bag| {
                let length = bag.values().sum::<usize>().max(1) as f64;
                terms
                    .iter()
                    .filter_map(|term| {
                        let seen = *bag.get(term)? as f64;
                        let having = bags.iter().filter(|o| o.contains_key(term)).count() as f64;
                        let idf = (1.0 + (bags.len() as f64 - having + 0.5) / (having + 0.5)).ln();
                        Some(idf * seen * 2.2 / (seen + 1.2 * (0.25 + 0.75 * length / average)))
                    })
                    .sum()
            })
            .collect()
    }

    /// The passages that best answer `query`, best first.
    pub async fn search(&self, query: &str, k: usize, filter: Option<&Value>) -> Result<Vec<Passage>> {
        let query = query.trim();
        if query.is_empty() {
            return Err(kb_error("the query is empty — say what to look for"));
        }
        let k = k.max(1);
        conditions(filter)?; // a bad filter fails before any call
        let me = &self.inner;
        let scoped = self.scope(filter);
        let mut hits = if let Some(r) = &me.retriever {
            r.retrieve(query, if me.reranker.is_some() { k * me.candidates } else { k }, scoped.as_ref()).await?
        } else {
            let store = self.prepare().await?.clone();
            let vector = self.embed(&[query.to_string()]).await?.remove(0);
            let wide = me.hybrid || me.reranker.is_some();
            store.query(&vector, if wide { store.max_k().min(k * me.candidates) } else { k }, scoped.as_ref()).await?
        };
        hits.retain(|h| h.score >= me.min_score && !h.text.trim().is_empty());

        let mut order: Vec<usize> = (0..hits.len()).collect();
        if me.hybrid && hits.len() > 1 {
            let lexical = Self::lexical(query, &hits);
            let mut by_words = order.clone();
            by_words.sort_by(|a, b| lexical[*b].partial_cmp(&lexical[*a]).unwrap_or(std::cmp::Ordering::Equal));
            let rank: HashMap<usize, usize> = by_words.iter().enumerate().map(|(place, n)| (*n, place)).collect();
            // Reciprocal rank fusion: a passage near the top of either list rises.
            let fused: Vec<f64> = order.iter().map(|&n| 1.0 / (60 + n) as f64 + if lexical[n] > 0.0 { 1.0 / (60 + rank[&n]) as f64 } else { 0.0 }).collect();
            order.sort_by(|a, b| fused[*b].partial_cmp(&fused[*a]).unwrap_or(std::cmp::Ordering::Equal));
        }

        let mut seen = HashSet::new();
        let mut passages = Vec::new();
        for n in order {
            let hit = &hits[n];
            let key: String = hit.text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(400).collect();
            if !seen.insert(key) {
                continue; // the same words twice add nothing
            }
            let mut meta = hit.metadata.clone();
            let mut take = |k: &str| match meta.remove(k) {
                Some(Value::String(s)) => s,
                Some(Value::Null) | None => String::new(),
                Some(other) => other.to_string(),
            };
            let (document, title, source) = (take("doc"), take("title"), take("source"));
            let chunk = meta.remove("chunk").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            meta.retain(|k, _| !RESERVED.contains(&k.as_str()));
            passages.push(Passage { text: hit.text.clone(), score: hit.score, document, title, source, chunk, metadata: meta, id: hit.id.clone() });
        }

        if let Some(rerank) = &me.reranker
            && passages.len() > 1
        {
            let scores = rerank(query.to_string(), passages.clone()).await?;
            if scores.len() != passages.len() {
                return Err(store_error("", format!("the reranker returned {} scores for {} passages", scores.len(), passages.len())));
            }
            let mut paired: Vec<(f64, Passage)> = scores.into_iter().zip(passages).collect();
            paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            passages = paired.into_iter().map(|(_, p)| p).collect();
        }
        passages.truncate(k);
        Ok(passages)
    }

    // ---- as a tool ---------------------------------------------------------
    /// The knowledge base as a tool an agent can search, returning five
    /// passages by default, at most ten.
    pub fn as_tool(&self, name: &str) -> Tool {
        self.as_tool_with(name, None, 5, 10, None, 1500)
    }

    /// The knowledge base as a tool. `filter` is applied to every search the
    /// agent makes and cannot be removed by it — the way to give one agent
    /// one tenant's documents.
    pub fn as_tool_with(&self, name: &str, description: Option<&str>, k: usize, max_k: usize, filter: Option<Value>, max_chars: usize) -> Tool {
        let kb = self.clone();
        let description = description.map(str::to_string).unwrap_or_else(|| {
            "Search the knowledge base and return the passages that best answer the query.\n\nEach passage comes with the document it is from: say which you used. Search again with different words when the passages miss.".into()
        });
        let parameters = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what you want to know, as a question or a few key terms."},
                "limit": {"type": "integer", "description": "how many passages to return.", "default": k},
            },
            "required": ["query"],
        });
        Tool::raw(name, description, parameters, move |args, _ctx| {
            let kb = kb.clone();
            let filter = filter.clone();
            async move {
                let query = args.get("query").and_then(Value::as_str).unwrap_or("").to_string();
                let limit = args.get("limit").and_then(Value::as_u64).map(|n| n as usize).filter(|n| *n > 0).unwrap_or(k);
                let found = kb.search(&query, limit.clamp(1, max_k.max(1)), filter.as_ref().filter(|f| f.as_object().is_some_and(|m| !m.is_empty()))).await?;
                if found.is_empty() {
                    return Ok(json!(format!("Nothing in the knowledge base matches {}. Try other words, or say that it is not covered.", py_repr(&query))));
                }
                let rows: Vec<Value> = found
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let mut row = p.to_json();
                        if clen(&p.text) > max_chars {
                            let cut: String = p.text.chars().take(max_chars).collect();
                            let cut = cut.rsplit_once(' ').map_or(cut.as_str(), |(a, _)| a).to_string();
                            row["text"] = json!(format!("{cut}…"));
                        }
                        let mut out = Map::new();
                        out.insert("n".into(), json!(i + 1));
                        out.extend(row.as_object().cloned().unwrap_or_default());
                        Value::Object(out)
                    })
                    .collect();
                Ok(Value::Array(rows))
            }
        })
        .tags(["builtin", "knowledge", "search"])
    }

    pub async fn close(&self) {
        if let Some(s) = &self.inner.store {
            s.close().await;
        }
    }
}

/// Python's `repr()` of a string, for messages that quote what was asked.
pub(crate) fn py_repr(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') { format!("\"{s}\"") } else { format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")) }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The readable text of an HTML page: no scripts, styles or tags.
fn html_text(html: &str) -> String {
    static DROP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<(script|style|noscript)[^>]*>.*?</(script|style|noscript)>").expect("valid"));
    static BREAK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)<(br|/p|/div|/h[1-6]|/li|/tr)[^>]*>").expect("valid"));
    static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").expect("valid"));
    let text = DROP.replace_all(html, "");
    let text = BREAK.replace_all(&text, "\n\n");
    let text = TAG.replace_all(&text, "");
    text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_keep_headings_and_overlap() {
        let text = "# Refunds\n\nRefunds take five days. They go to the card you paid with.\n\n## Exceptions\n\nGift cards are never refunded.";
        // The same passages the Python harness cuts.
        let chunks = chunk_text(text, 200, 50);
        assert_eq!(chunks, vec!["# Refunds Refunds take five days. They go to the card you paid with.", "## Exceptions Gift cards are never refunded."]);
        let long = "word ".repeat(400);
        let parts = chunk_text(&long, 200, 40);
        assert_eq!((parts.len(), parts.iter().map(|p| clen(p)).max().unwrap()), (10, 239));
        assert!(parts[1].starts_with("word word word word word word "));
        let t2 = format!(
            "{}

{}",
            "Alpha beta gamma. ".repeat(30),
            "Delta epsilon. ".repeat(20)
        );
        let r = chunk_text(&t2, 200, 60);
        assert_eq!(r.len(), 6);
        assert_eq!(r[0], "Alpha beta gamma. ".repeat(11).trim());
        assert_eq!(r[1], r[0]);
        assert_eq!(sentences("One. Two!  Three"), vec!["One.", "Two!", "Three"]);
    }
}
