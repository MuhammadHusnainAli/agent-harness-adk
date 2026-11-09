//! Sessions in object storage: an S3 bucket, an Azure storage account, GCS.
//!
//! One object per conversation, and a tiny marker per owner so that listing
//! one person's chats is a prefix listing rather than a read of every
//! conversation:
//!
//! ```text
//! {prefix}/sessions/{id}.json
//! {prefix}/session-index/{tenant}/{user}/{id}
//! ```
//!
//! Cheap, durable, and the right place for chats kept for years. Listing reads
//! the markers and sorts them here, so it is for a person's history, not for
//! querying millions.
//!
//! The version check reads before it writes, as the Python store does. Where
//! Python stops there (and two saves landing in the same instant can both
//! pass), this store also makes the write itself conditional — `If-Match` on
//! the ETag for S3 and Azure, `ifGenerationMatch` for GCS, `If-None-Match: *`
//! for a new session — so exactly one of a race wins. A bucket that refuses
//! conditional writes can turn them off with `conditional=false`. The objects
//! are the Python harness's either way.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::errors::{Error, Result};
use crate::memory::providers::objects::{ObjectBackend, PutIf};
use crate::runtime::session::{DurableBackend, ListFilter, Session};

/// Sessions as objects, over any of the object-store memory backends.
///
/// ```no_run
/// # use agent_harness::sessions::session_provider;
/// let store = session_provider("s3://my-bucket/chats");
/// ```
pub struct ObjectSessionStore {
    objects: Arc<dyn ObjectBackend>,
    /// The memory backend's prefix; sessions live under `{prefix}/sessions/`.
    pub prefix: String,
    /// How many markers are read at once when listing.
    pub concurrency: usize,
    /// Make the write conditional on what was read (see the module docs).
    pub conditional: bool,
}

impl std::fmt::Debug for ObjectSessionStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<ObjectSessionStore {} {}>", self.objects.describe(), self.prefix)
    }
}

impl ObjectSessionStore {
    pub fn new(objects: Arc<dyn ObjectBackend>, prefix: &str) -> Self {
        Self { objects, prefix: prefix.trim_matches('/').to_string(), concurrency: 16, conditional: true }
    }

    fn base(&self, rest: &str) -> String {
        if self.prefix.is_empty() { rest.to_string() } else { format!("{}/{rest}", self.prefix) }
    }

    pub fn key(&self, session_id: &str) -> String {
        self.base(&format!("sessions/{session_id}.json"))
    }

    pub fn marker(&self, session: &Session) -> String {
        let or = |v: &Option<String>| v.as_deref().filter(|s| !s.is_empty()).unwrap_or("_").to_string();
        self.base(&format!("session-index/{}/{}/{}", or(&session.tenant_id), or(&session.user_id), session.id))
    }

    fn parse(raw: &[u8]) -> Option<Session> {
        if raw.is_empty() {
            return None;
        }
        serde_json::from_slice(raw).ok()
    }
}

#[async_trait]
impl DurableBackend for ObjectSessionStore {
    async fn read(&self, session_id: &str) -> Result<Option<Session>> {
        Ok(self.objects.get(&self.key(session_id)).await?.and_then(|f| Self::parse(&f.body)))
    }

    async fn write(&self, session: &Session, expect: u64) -> Result<bool> {
        let key = self.key(&session.id);
        let fetched = self.objects.get(&key).await?;
        let held = fetched.as_ref().and_then(|f| Self::parse(&f.body));
        if held.as_ref().map_or(0, |h| h.version) != expect {
            return Ok(false);
        }
        let condition = match (&fetched, self.conditional) {
            (_, false) => PutIf::Always,
            (None, true) => PutIf::Absent,
            (Some(f), true) => f.tag.clone().map_or(PutIf::Always, PutIf::Matches),
        };
        let body = serde_json::to_vec(session).map_err(|e| Error::Io(e.to_string()))?;
        if !self.objects.put(&key, body, "application/json", &condition).await? {
            return Ok(false);
        }
        let marker = self.marker(session);
        if let Some(h) = &held
            && self.marker(h) != marker
        {
            self.objects.delete(&[self.marker(h)]).await?;
        }
        let stamp = serde_json::to_vec(&json!({"updated": session.updated, "agent": session.agent})).unwrap_or_default();
        self.objects.put(&marker, stamp, "application/json", &PutIf::Always).await?;
        Ok(true)
    }

    async fn list_sessions(&self, filter: &ListFilter) -> Result<Vec<Session>> {
        let mut base = self.base("session-index/");
        if let Some(t) = &filter.tenant_id {
            base.push_str(&format!("{t}/"));
            if let Some(u) = &filter.user_id {
                base.push_str(&format!("{u}/"));
            }
        }
        let markers = self.objects.list(&base).await?;
        let objects = &self.objects;
        let stamped: Vec<Result<(f64, String, String)>> = futures::stream::iter(markers)
            .map(|key| async move {
                let raw = objects.get(&key).await?.map(|f| f.body).unwrap_or_default();
                let blob: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
                let updated = blob.get("updated").and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).unwrap_or(0.0);
                let agent = blob.get("agent").and_then(Value::as_str).unwrap_or("").to_string();
                let id = key.rsplit('/').next().unwrap_or("").to_string();
                Ok((updated, id, agent))
            })
            .buffer_unordered(self.concurrency.max(1))
            .collect()
            .await;
        let mut rows = stamped.into_iter().collect::<Result<Vec<_>>>()?;
        // Newest first, as Python sorts the (updated, id, agent) tuples.
        rows.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| b.1.cmp(&a.1)).then_with(|| b.2.cmp(&a.2)));
        let mut out = Vec::new();
        for (_, session_id, owner_agent) in rows {
            if filter.agent.as_ref().is_some_and(|a| a != &owner_agent) {
                continue;
            }
            if let Some(found) = self.read(&session_id).await?
                && filter.user_id.as_ref().is_none_or(|u| found.user_id.as_ref() == Some(u))
                && filter.tenant_id.as_ref().is_none_or(|t| found.tenant_id.as_ref() == Some(t))
            {
                out.push(found);
                if out.len() >= filter.limit {
                    break;
                }
            }
        }
        Ok(out)
    }

    async fn remove(&self, session_id: &str) -> Result<()> {
        // Some stores (Azure) object to deleting what is not there.
        let Some(held) = self.read(session_id).await? else { return Ok(()) };
        self.objects.delete(&[self.key(session_id), self.marker(&held)]).await
    }

    fn backend_name(&self) -> &'static str {
        "ObjectSessionStore"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::providers::objects::testing::MapBucket;

    #[test]
    fn keys_match_python() {
        let store = ObjectSessionStore::new(Arc::new(MapBucket::default()), "/chats/");
        let s = Session { id: "ses_1".into(), user_id: Some("alice".into()), ..Session::new("a") };
        assert_eq!(store.key("ses_1"), "chats/sessions/ses_1.json");
        assert_eq!(store.marker(&s), "chats/session-index/_/alice/ses_1");
    }

    #[tokio::test]
    async fn a_bucket_without_conditional_writes_still_refuses_a_stale_save() {
        let mut store = ObjectSessionStore::new(Arc::new(MapBucket::default()), "chats");
        store.conditional = false;
        crate::memory::providers::objects::testing::exercise_sessions(&store, false).await;
    }
}
