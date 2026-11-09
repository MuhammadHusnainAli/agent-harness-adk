//! MongoDB memory, on the official async driver.
#![cfg(feature = "mongodb")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures::TryStreamExt;
use mongodb::bson::{Bson, Document, doc};
use mongodb::options::IndexOptions;
use mongodb::{Client, Collection, IndexModel};
use serde_json::{Map, Value};

use crate::errors::{Error, Result};
use crate::memory::base::{MemoryRecord, MemoryStore, RecordQuery, doc_key};
use crate::memory::trace::Trace;
use crate::runtime::session::SessionStore;
use crate::types::now;

/// A MongoDB failure, as the error every backend raises.
pub(crate) fn mongo_error(e: impl std::fmt::Display) -> Error {
    let message: String = format!("mongodb: {e}").chars().take(400).collect();
    Error::VectorStore { message, store: "mongodb".into(), status: None, retryable: false }
}

/// A JSON value as BSON, the way pymongo encodes the same Python value: an
/// integer that fits 32 bits is an `int`, a bigger one a `long`, a fraction a
/// `double`.
pub(crate) fn to_bson(value: &Value) -> Bson {
    match value {
        Value::Null => Bson::Null,
        Value::Bool(b) => Bson::Boolean(*b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => i32::try_from(i).map_or(Bson::Int64(i), Bson::Int32),
            None => Bson::Double(n.as_f64().unwrap_or(0.0)),
        },
        Value::String(s) => Bson::String(s.clone()),
        Value::Array(items) => Bson::Array(items.iter().map(to_bson).collect()),
        Value::Object(map) => Bson::Document(to_document(map)),
    }
}

/// A JSON object as a BSON document, keys in order.
pub(crate) fn to_document(map: &Map<String, Value>) -> Document {
    map.iter().map(|(k, v)| (k.clone(), to_bson(v))).collect()
}

/// BSON back to plain JSON. Types JSON has no word for (an ObjectId, a date)
/// come back as relaxed Extended JSON.
pub(crate) fn to_json(value: &Bson) -> Value {
    match value {
        Bson::Null | Bson::Undefined => Value::Null,
        Bson::Boolean(b) => Value::Bool(*b),
        Bson::Int32(i) => Value::from(*i),
        Bson::Int64(i) => Value::from(*i),
        Bson::Double(f) => serde_json::Number::from_f64(*f).map_or(Value::Null, Value::Number),
        Bson::String(s) => Value::String(s.clone()),
        Bson::Array(items) => Value::Array(items.iter().map(to_json).collect()),
        Bson::Document(d) => Value::Object(doc_to_json(d)),
        other => other.clone().into_relaxed_extjson(),
    }
}

/// A BSON document as a JSON object.
pub(crate) fn doc_to_json(document: &Document) -> Map<String, Value> {
    document.iter().map(|(k, v)| (k.clone(), to_json(v))).collect()
}

/// A connection string split the way the Python harness splits it: a path
/// names the database, and is taken off the URL with the query string.
///
/// `mongodb://host:27017/agents?x=1` is `("mongodb://host:27017", Some("agents"))`;
/// `mongodb://host:27017/?authSource=admin` is kept whole, with no database.
pub fn split_database(url: &str) -> (String, Option<String>) {
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    let before_query = rest.split(['?', '#']).next().unwrap_or("");
    let path = before_query.find('/').map_or("", |i| &before_query[i..]);
    let database = path.trim_matches('/');
    if database.is_empty() {
        return (url.to_string(), None);
    }
    let bare = url.split('?').next().unwrap_or(url);
    let base = bare.rsplit_once('/').map_or(bare, |(b, _)| b);
    (base.to_string(), Some(database.to_string()))
}

/// One MongoDB client, made on first use and shared by everything built on
/// it — memory, sessions, a vector store.
pub struct MongoLink {
    url: String,
    client: tokio::sync::Mutex<Option<Client>>,
}

impl std::fmt::Debug for MongoLink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The URL may carry a password, so it is not printed.
        f.write_str("MongoLink")
    }
}

impl MongoLink {
    /// A link to `mongodb://…` or `mongodb+srv://…`. Does not connect.
    pub fn new(url: &str) -> Result<Self> {
        let lower = url.to_ascii_lowercase();
        if !(lower.starts_with("mongodb://") || lower.starts_with("mongodb+srv://")) {
            return Err(Error::config(format!("not a MongoDB URL — it starts mongodb:// or mongodb+srv://: '{}'", lower.split("://").next().unwrap_or(""))));
        }
        Ok(Self { url: url.to_string(), client: tokio::sync::Mutex::new(None) })
    }

    /// The client, made now if this is the first call. (Making one reads the
    /// URL, and for `mongodb+srv` looks the hosts up; it does not wait for a server.)
    pub async fn client(&self) -> Result<Client> {
        let mut held = self.client.lock().await;
        if let Some(c) = held.as_ref() {
            return Ok(c.clone());
        }
        let made = Client::with_uri_str(&self.url).await.map_err(mongo_error)?;
        *held = Some(made.clone());
        Ok(made)
    }

    /// Let the client go. The next call makes a new one.
    pub async fn close(&self) {
        if let Some(client) = self.client.lock().await.take() {
            client.shutdown().await;
        }
    }
}

/// Records and documents in two collections, indexed on the trace.
///
/// ```no_run
/// # use agent_harness::memory::providers::mongo::MongoMemory;
/// let store = MongoMemory::from_url("mongodb://localhost:27017/agents").unwrap();
/// ```
///
/// A record is a document shaped like the record's JSON, with its id as `_id`;
/// a named document (like `user.md`) is `{doc_key, name, namespace, text,
/// updated}` in `{collection}_docs`. The same collections the Python harness
/// writes.
#[derive(Debug, Clone)]
pub struct MongoMemory {
    link: Arc<MongoLink>,
    /// Default `agent_harness`, or the path of the URL.
    pub database: String,
    /// Default `memory`.
    pub collection: String,
    /// Default `{collection}_docs`.
    pub docs_collection: String,
    ready: Arc<AtomicBool>,
}

impl MongoMemory {
    /// A store on `mongodb://host:27017/agents`; the path names the database
    /// (`agent_harness` without one). Connects on first use.
    pub fn from_url(url: &str) -> Result<Self> {
        let (url, database) = split_database(url);
        Ok(Self {
            link: Arc::new(MongoLink::new(&url)?),
            database: database.unwrap_or_else(|| "agent_harness".into()),
            collection: "memory".into(),
            docs_collection: "memory_docs".into(),
            ready: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Keep records in `collection` and documents in `docs` (`{collection}_docs` when empty).
    pub fn with_collections(mut self, collection: &str, docs: &str) -> Self {
        self.collection = collection.into();
        self.docs_collection = if docs.is_empty() { format!("{collection}_docs") } else { docs.into() };
        self
    }

    /// Use another database.
    pub fn with_database(mut self, database: &str) -> Self {
        self.database = database.into();
        self
    }

    /// The client this store uses, to share with a session store.
    pub fn link(&self) -> Arc<MongoLink> {
        self.link.clone()
    }

    async fn ensure(&self) -> Result<Client> {
        let client = self.link.client().await?;
        if !self.ready.load(Ordering::Acquire) {
            let db = client.database(&self.database);
            // Matches how the harness reads: trace first, then scope, newest first.
            let records = IndexModel::builder().keys(doc! {"user_id": 1, "session_id": 1, "tenant_id": 1, "scope": 1, "ts": -1}).build();
            db.collection::<Document>(&self.collection).create_index(records).await.map_err(mongo_error)?;
            let docs = IndexModel::builder().keys(doc! {"doc_key": 1}).options(IndexOptions::builder().unique(true).build()).build();
            db.collection::<Document>(&self.docs_collection).create_index(docs).await.map_err(mongo_error)?;
            self.ready.store(true, Ordering::Release);
        }
        Ok(client)
    }

    async fn records(&self) -> Result<Collection<Document>> {
        Ok(self.ensure().await?.database(&self.database).collection(&self.collection))
    }

    async fn docs(&self) -> Result<Collection<Document>> {
        Ok(self.ensure().await?.database(&self.database).collection(&self.docs_collection))
    }

    /// The filter for a read or a clear.
    pub(crate) fn query(scope: Option<&str>, kind: Option<&str>, trace: Option<&Trace>) -> Document {
        let mut query = Document::new();
        if let Some(s) = scope {
            query.insert("scope", s);
        }
        if let Some(k) = kind {
            query.insert("kind", k);
        }
        if let Some(t) = trace {
            // A record with no value on an axis is shared, not hidden.
            for (field, value) in t.filters() {
                query.insert(field, doc! {"$in": [value, Bson::Null]});
            }
        }
        query
    }

    /// A record as it is stored: its JSON, with its id as `_id`.
    pub(crate) fn record_document(record: &MemoryRecord) -> Document {
        let mut blob = match serde_json::to_value(record) {
            Ok(Value::Object(m)) => to_document(&m),
            _ => Document::new(),
        };
        blob.insert("_id", record.id.clone());
        blob
    }

    fn record_from(mut row: Document) -> Option<MemoryRecord> {
        row.remove("_id");
        serde_json::from_value(Value::Object(doc_to_json(&row))).ok()
    }
}

#[async_trait]
impl MemoryStore for MongoMemory {
    async fn append(&self, record: MemoryRecord) -> Result<MemoryRecord> {
        self.records().await?.insert_one(Self::record_document(&record)).await.map_err(mongo_error)?;
        Ok(record)
    }

    async fn extend(&self, records: Vec<MemoryRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let blobs: Vec<Document> = records.iter().map(Self::record_document).collect();
        self.records().await?.insert_many(blobs).await.map_err(mongo_error)?;
        Ok(())
    }

    async fn all(&self, query: RecordQuery) -> Result<Vec<MemoryRecord>> {
        let coll = self.records().await?;
        let filter = Self::query(query.scope.as_deref(), query.kind.as_deref(), query.trace.as_ref());
        let limit = query.limit.filter(|n| *n > 0).unwrap_or(10_000);
        let cursor = coll.find(filter).sort(doc! {"ts": -1}).limit(limit as i64).await.map_err(mongo_error)?;
        let rows: Vec<Document> = cursor.try_collect().await.map_err(mongo_error)?;
        let mut out: Vec<MemoryRecord> = rows.into_iter().filter_map(Self::record_from).collect();
        out.reverse(); // back to chronological order
        Ok(out)
    }

    async fn clear(&self, scope: Option<&str>, trace: Option<&Trace>) -> Result<()> {
        self.records().await?.delete_many(Self::query(scope, None, trace)).await.map_err(mongo_error)?;
        if let Some(t) = trace {
            self.docs().await?.delete_many(doc! {"namespace": t.slug()}).await.map_err(mongo_error)?;
        } else if scope.is_none() {
            self.docs().await?.delete_many(doc! {}).await.map_err(mongo_error)?;
        }
        Ok(())
    }

    async fn read_doc(&self, name: &str, trace: Option<&Trace>) -> Result<String> {
        let found = self.docs().await?.find_one(doc! {"doc_key": doc_key(name, trace)}).await.map_err(mongo_error)?;
        Ok(found.and_then(|d| d.get_str("text").ok().map(str::to_string)).unwrap_or_default())
    }

    async fn write_doc(&self, name: &str, text: &str, trace: Option<&Trace>) -> Result<()> {
        let key = doc_key(name, trace);
        let namespace = trace.map_or_else(|| "_shared".to_string(), Trace::slug);
        let update = doc! {"$set": {"doc_key": &key, "name": name, "namespace": namespace, "text": text, "updated": now()}};
        self.docs().await?.update_one(doc! {"doc_key": &key}, update).upsert(true).await.map_err(mongo_error)?;
        Ok(())
    }

    async fn close(&self) {
        self.link.close().await;
        self.ready.store(false, Ordering::Release);
    }

    fn describe(&self) -> String {
        format!("mongodb ({}.{})", self.database, self.collection)
    }

    /// Sessions on this store's client. `options`: `collection` (default `sessions`).
    fn sessions(&self, options: &Map<String, Value>) -> Option<Result<Arc<dyn SessionStore>>> {
        let collection = options.get("collection").and_then(Value::as_str).unwrap_or("sessions");
        Some(Ok(Arc::new(crate::sessions::mongo::MongoSessionStore::new(self, collection))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn urls_split_like_python() {
        assert_eq!(split_database("mongodb://h:27017/agents?retryWrites=true"), ("mongodb://h:27017".to_string(), Some("agents".to_string())));
        assert_eq!(split_database("mongodb://h:27017"), ("mongodb://h:27017".to_string(), None));
        assert_eq!(split_database("mongodb://h:27017/?authSource=admin"), ("mongodb://h:27017/?authSource=admin".to_string(), None));
        assert_eq!(split_database("mongodb+srv://u:p@c.example.net/kb").1.as_deref(), Some("kb"));
        let m = MongoMemory::from_url("mongodb://localhost:27017/agents").unwrap();
        assert_eq!((m.database.as_str(), m.collection.as_str(), m.docs_collection.as_str()), ("agents", "memory", "memory_docs"));
        assert_eq!(MongoMemory::from_url("mongodb://localhost").unwrap().database, "agent_harness");
        assert!(MongoMemory::from_url("redis://x").is_err());
    }

    #[test]
    fn documents_and_queries_match_python() {
        let mut r = MemoryRecord::new("user", "fact", "x");
        r.user_id = Some("alice".into());
        r.data = json!({"n": 1, "big": 5_000_000_000i64, "f": 1.5}).as_object().cloned().unwrap();
        let d = MongoMemory::record_document(&r);
        assert_eq!(d.get_str("_id").unwrap(), r.id);
        assert_eq!(d.get("session_id"), Some(&Bson::Null));
        assert_eq!(d.get_document("data").unwrap().get("n"), Some(&Bson::Int32(1)));
        assert_eq!(d.get_document("data").unwrap().get("big"), Some(&Bson::Int64(5_000_000_000)));
        assert!(matches!(d.get("ts"), Some(Bson::Double(_))));
        assert_eq!(MongoMemory::record_from(d).unwrap(), r);

        let q = MongoMemory::query(Some("user"), None, Some(&Trace::user("alice").tenant("acme")));
        assert_eq!(q, doc! {"scope": "user", "tenant_id": {"$in": ["acme", Bson::Null]}, "user_id": {"$in": ["alice", Bson::Null]}});
        assert_eq!(MongoMemory::query(None, Some("fact"), None), doc! {"kind": "fact"});
    }

    #[tokio::test]
    async fn live_round_trip() {
        let Ok(url) = std::env::var("AH_LIVE_MONGO_URL") else { return };
        let suffix = &uuid::Uuid::new_v4().simple().to_string()[..8];
        let store = MongoMemory::from_url(&url).unwrap().with_database("ah_test").with_collections(&format!("memory_{suffix}"), "");
        let alice = Trace::user("alice");
        let mut a = MemoryRecord::new("user", "decision", "Bill in EUR always");
        alice.stamp(&mut a);
        store.append(a).await.unwrap();
        let mut b = MemoryRecord::new("user", "fact", "Bob likes tea");
        Trace::user("bob").stamp(&mut b);
        store.extend(vec![b]).await.unwrap();
        store.append(MemoryRecord::new("project", "note", "Shared euro note")).await.unwrap();
        assert_eq!(store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap().len(), 2);
        assert_eq!(store.all(RecordQuery::default().limit(1)).await.unwrap()[0].text, "Shared euro note");
        store.write_doc("user.md", "- likes EUR", Some(&alice)).await.unwrap();
        store.write_doc("user.md", "- likes EUR!", Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "- likes EUR!");
        store.clear(None, Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "");
        let left = store.all(RecordQuery::default()).await.unwrap();
        assert!(left.iter().all(|r| r.user_id.as_deref() != Some("alice")));
        let client = store.link().client().await.unwrap();
        let db = client.database("ah_test");
        db.collection::<Document>(&store.collection).drop().await.unwrap();
        db.collection::<Document>(&store.docs_collection).drop().await.unwrap();
        store.close().await;
    }
}
