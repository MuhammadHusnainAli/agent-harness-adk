//! Sessions in MongoDB (and what speaks its protocol: Cosmos DB, DocumentDB).
//!
//! One document per conversation, keyed by its id. A save replaces the
//! document only where the version still matches — the database makes that
//! comparison and the write one atomic step.
#![cfg(feature = "mongodb")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use futures::TryStreamExt;
use mongodb::bson::{Bson, Document, doc};
use mongodb::error::{ErrorKind, WriteFailure};
use mongodb::{Collection, IndexModel};

use crate::errors::{Error, Result};
use crate::memory::providers::mongo::{MongoLink, MongoMemory, mongo_error};
use crate::runtime::session::{DurableBackend, ListFilter, Session};

/// Sessions in a collection, over the MongoDB memory backend's client.
///
/// ```no_run
/// # use agent_harness::memory::providers::mongo::MongoMemory;
/// # use agent_harness::sessions::mongo::MongoSessionStore;
/// let memory = MongoMemory::from_url("mongodb://host:27017/agents").unwrap();
/// let store = MongoSessionStore::new(&memory, "sessions");
/// ```
///
/// The conversation is stored as one JSON string, not as nested fields: tool
/// arguments are arbitrary and may hold keys Mongo will not take. A document
/// is limited to 16 MB, which is a very long conversation.
#[derive(Debug, Clone)]
pub struct MongoSessionStore {
    link: Arc<MongoLink>,
    pub database: String,
    pub collection: String,
    ready: Arc<AtomicBool>,
}

impl MongoSessionStore {
    /// On `mongo`'s client and database, in `collection` (`sessions` by default).
    pub fn new(mongo: &MongoMemory, collection: &str) -> Self {
        let collection = if collection.is_empty() { "sessions" } else { collection };
        Self { link: mongo.link(), database: mongo.database.clone(), collection: collection.into(), ready: Arc::new(AtomicBool::new(false)) }
    }

    async fn coll(&self) -> Result<Collection<Document>> {
        let coll = self.link.client().await?.database(&self.database).collection::<Document>(&self.collection);
        if !self.ready.load(Ordering::Acquire) {
            let index = IndexModel::builder().keys(doc! {"tenant_id": 1, "user_id": 1, "updated": -1}).build();
            coll.create_index(index).await.map_err(mongo_error)?;
            self.ready.store(true, Ordering::Release);
        }
        Ok(coll)
    }

    /// A session as it is stored: the fields a listing filters and sorts on,
    /// and the whole conversation as JSON in `payload`.
    pub(crate) fn document(session: &Session) -> Result<Document> {
        let payload = serde_json::to_string(session).map_err(|e| Error::Io(e.to_string()))?;
        let opt = |v: &Option<String>| v.clone().map_or(Bson::Null, Bson::String);
        let version = i64::try_from(session.version).unwrap_or(i64::MAX);
        Ok(doc! {
            "_id": &session.id, "agent": &session.agent,
            "user_id": opt(&session.user_id), "tenant_id": opt(&session.tenant_id),
            "title": &session.title, "created": session.created,
            "updated": session.updated,
            "version": i32::try_from(version).map_or(Bson::Int64(version), Bson::Int32),
            "payload": payload,
        })
    }

    /// A session from its document: the payload, with the version the document holds.
    pub(crate) fn session(document: Option<Document>) -> Option<Session> {
        let document = document?;
        let mut found: Session = serde_json::from_str(document.get_str("payload").ok()?).ok()?;
        let held = match document.get("version") {
            Some(Bson::Int32(v)) => u64::try_from(*v).ok(),
            Some(Bson::Int64(v)) => u64::try_from(*v).ok(),
            Some(Bson::Double(v)) if *v >= 0.0 => Some(*v as u64),
            _ => None,
        };
        if let Some(v) = held.filter(|v| *v > 0) {
            found.version = v;
        }
        Some(found)
    }

    /// The filter for a listing.
    pub(crate) fn list_query(filter: &ListFilter) -> Document {
        let mut query = Document::new();
        for (key, value) in [("tenant_id", &filter.tenant_id), ("user_id", &filter.user_id), ("agent", &filter.agent)] {
            if let Some(v) = value {
                query.insert(key, v.clone());
            }
        }
        query
    }
}

fn duplicate_key(e: &mongodb::error::Error) -> bool {
    matches!(e.kind.as_ref(), ErrorKind::Write(WriteFailure::WriteError(w)) if w.code == 11000)
}

#[async_trait]
impl DurableBackend for MongoSessionStore {
    async fn read(&self, session_id: &str) -> Result<Option<Session>> {
        let found = self.coll().await?.find_one(doc! {"_id": session_id}).await.map_err(mongo_error)?;
        Ok(Self::session(found))
    }

    async fn write(&self, session: &Session, expect: u64) -> Result<bool> {
        let coll = self.coll().await?;
        let document = Self::document(session)?;
        if expect == 0 {
            return match coll.insert_one(document).await {
                Ok(_) => Ok(true),
                Err(e) if duplicate_key(&e) => Ok(false),
                Err(e) => Err(mongo_error(e)),
            };
        }
        let expect = i64::try_from(expect).unwrap_or(i64::MAX);
        let done = coll.replace_one(doc! {"_id": &session.id, "version": expect}, document).await.map_err(mongo_error)?;
        Ok(done.matched_count == 1)
    }

    async fn list_sessions(&self, filter: &ListFilter) -> Result<Vec<Session>> {
        let coll = self.coll().await?;
        let limit = filter.limit.max(1) as i64;
        let cursor = coll.find(Self::list_query(filter)).sort(doc! {"updated": -1}).limit(limit).await.map_err(mongo_error)?;
        let rows: Vec<Document> = cursor.try_collect().await.map_err(mongo_error)?;
        Ok(rows.into_iter().filter_map(|d| Self::session(Some(d))).collect())
    }

    async fn remove(&self, session_id: &str) -> Result<()> {
        self.coll().await?.delete_one(doc! {"_id": session_id}).await.map_err(mongo_error)?;
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "MongoSessionStore"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::session::SessionStore;

    #[test]
    fn documents_match_python() {
        let s = Session { title: "chat".into(), user_id: Some("u".into()), version: 3, ..Session::new("bot") };
        let d = MongoSessionStore::document(&s).unwrap();
        let keys: Vec<&str> = d.keys().map(String::as_str).collect();
        assert_eq!(keys, ["_id", "agent", "user_id", "tenant_id", "title", "created", "updated", "version", "payload"]);
        assert_eq!(d.get("tenant_id"), Some(&Bson::Null));
        assert_eq!(d.get("version"), Some(&Bson::Int32(3)));
        let back = MongoSessionStore::session(Some(d)).unwrap();
        assert_eq!(back, s);
        assert!(MongoSessionStore::session(Some(doc! {"_id": "x"})).is_none());
        let q = MongoSessionStore::list_query(&ListFilter { limit: 5, user_id: Some("u".into()), tenant_id: None, agent: Some("bot".into()) });
        assert_eq!(q, doc! {"user_id": "u", "agent": "bot"});
    }

    #[tokio::test]
    async fn live_check_and_conflict() {
        let Ok(url) = std::env::var("AH_LIVE_MONGO_URL") else { return };
        let memory = MongoMemory::from_url(&url).unwrap().with_database("ah_test");
        let collection = format!("sessions_{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        let store = MongoSessionStore::new(&memory, &collection);
        let report = store.check().await;
        assert_eq!(report["ok"], true, "{report}");
        let mut s = Session::new("bot");
        store.save(&mut s).await.unwrap();
        let mut stale = store.load(&s.id).await.unwrap();
        store.save(&mut s).await.unwrap();
        assert!(matches!(store.save(&mut stale).await, Err(Error::SessionConflict { .. })));
        assert_eq!(store.load(&s.id).await.unwrap().version, 2);
        let client = memory.link().client().await.unwrap();
        client.database("ah_test").collection::<Document>(&collection).drop().await.unwrap();
    }
}
