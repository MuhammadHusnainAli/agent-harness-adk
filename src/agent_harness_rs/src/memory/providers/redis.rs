//! Redis memory. Fast, and the right fit for session-scoped memory with a TTL.
#![cfg(feature = "redis")]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use redis::aio::{ConnectionManager, ConnectionManagerConfig};
use serde_json::{Map, Value};

use crate::errors::{Error, Result};
use crate::memory::base::{MemoryRecord, MemoryStore, RecordQuery, doc_key, last};
use crate::memory::trace::Trace;
use crate::runtime::session::SessionStore;
use crate::types::now;

/// A Redis failure, as the error every backend raises.
pub(crate) fn redis_error(e: impl std::fmt::Display) -> Error {
    Error::VectorStore { message: format!("redis: {e}"), store: "redis".into(), status: None, retryable: false }
}

/// One Redis connection, opened on first use and shared by everything built on
/// it — memory, sessions, a vector store.
///
/// The URL is checked when the link is made; nothing connects until a command
/// is sent. The connection reconnects by itself after a network failure.
pub struct RedisLink {
    client: redis::Client,
    conn: tokio::sync::Mutex<Option<ConnectionManager>>,
}

impl std::fmt::Debug for RedisLink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The URL may carry a password, so it is not printed.
        f.write_str("RedisLink")
    }
}

impl RedisLink {
    /// A link to `redis://…` or `rediss://…` (TLS). Does not connect.
    pub fn open(url: &str) -> Result<Self> {
        let client = redis::Client::open(url).map_err(|e| Error::config(format!("not a Redis URL: {e}")))?;
        Ok(Self { client, conn: tokio::sync::Mutex::new(None) })
    }

    /// The connection, made now if this is the first call.
    pub async fn conn(&self) -> Result<ConnectionManager> {
        let mut held = self.conn.lock().await;
        if let Some(c) = held.as_ref() {
            return Ok(c.clone());
        }
        let config = ConnectionManagerConfig::new()
            .set_connection_timeout(Some(Duration::from_secs(10)))
            .set_response_timeout(Some(Duration::from_secs(60)))
            .set_number_of_retries(2);
        let made = self.client.get_connection_manager_with_config(config).await.map_err(redis_error)?;
        *held = Some(made.clone());
        Ok(made)
    }

    /// Let the connection go. The next command opens a new one.
    pub async fn close(&self) {
        self.conn.lock().await.take();
    }

    /// Every key matching `pattern`, by `SCAN` (never `KEYS`, which blocks the server).
    pub async fn scan(&self, pattern: &str) -> Result<Vec<String>> {
        let mut conn = self.conn().await?;
        let mut cursor: u64 = 0;
        let mut out = Vec::new();
        loop {
            let (next, keys): (u64, Vec<String>) =
                redis::cmd("SCAN").arg(cursor).arg("MATCH").arg(pattern).arg("COUNT").arg(500).query_async(&mut conn).await.map_err(redis_error)?;
            out.extend(keys);
            if next == 0 {
                break;
            }
            cursor = next;
        }
        Ok(out)
    }
}

/// Records in a sorted set per namespace, documents in plain keys.
///
/// ```no_run
/// # use agent_harness::memory::providers::redis::RedisMemory;
/// let store = RedisMemory::from_url("redis://localhost:6379/0").unwrap().with_ttl(Some(86_400));
/// ```
///
/// Scored by timestamp, so a range read is ordered and a `limit` is served by
/// Redis rather than by us. The keys are the Python harness's:
/// `{prefix}:rec:{namespace}:{scope}` (a sorted set of record JSON),
/// `{prefix}:doc:{doc key}` (a string) and `{prefix}:doc_meta` (a hash of when
/// each document was written).
#[derive(Debug, Clone)]
pub struct RedisMemory {
    link: Arc<RedisLink>,
    /// Every key starts with this. Default `agent-memory`.
    pub prefix: String,
    /// Seconds a record set lives after its last write. `None` keeps it.
    pub ttl: Option<u64>,
}

impl RedisMemory {
    /// A store on `redis://host:6379/0` (or `rediss://` for TLS). Connects on first use.
    pub fn from_url(url: &str) -> Result<Self> {
        Ok(Self { link: Arc::new(RedisLink::open(url)?), prefix: "agent-memory".into(), ttl: None })
    }

    /// Put every key under another prefix.
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.trim_end_matches(':').to_string();
        self
    }

    /// Let record sets expire this many seconds after their last write.
    /// Documents outlive records by design and never expire.
    pub fn with_ttl(mut self, ttl: Option<u64>) -> Self {
        self.ttl = ttl.filter(|t| *t > 0);
        self
    }

    /// The connection this store uses, to share with a session store.
    pub fn link(&self) -> Arc<RedisLink> {
        self.link.clone()
    }

    // ---- keys -----------------------------------------------------------------
    pub(crate) fn records_key(&self, namespace: &str, scope: &str) -> String {
        format!("{}:rec:{namespace}:{scope}", self.prefix)
    }

    pub(crate) fn doc_key(&self, name: &str, trace: Option<&Trace>) -> String {
        format!("{}:doc:{}", self.prefix, doc_key(name, trace))
    }

    /// Whose record set this is: the trace's, or the record's own owner's.
    pub(crate) fn namespace(trace: Option<&Trace>, record: Option<&MemoryRecord>) -> String {
        if let Some(t) = trace {
            return t.slug();
        }
        if let Some(r) = record {
            return Trace { user_id: r.user_id.clone(), session_id: r.session_id.clone(), tenant_id: r.tenant_id.clone(), ..Trace::default() }.slug();
        }
        "_shared".into()
    }

    async fn scopes(&self, namespace: &str) -> Result<Vec<String>> {
        let keys = self.link.scan(&format!("{}:rec:{namespace}:*", self.prefix)).await?;
        Ok(keys.into_iter().map(|k| k.rsplit(':').next().unwrap_or_default().to_string()).collect())
    }
}

#[async_trait]
impl MemoryStore for RedisMemory {
    async fn append(&self, record: MemoryRecord) -> Result<MemoryRecord> {
        let mut conn = self.link.conn().await?;
        let key = self.records_key(&Self::namespace(None, Some(&record)), &record.scope);
        let blob = serde_json::to_string(&record).map_err(|e| Error::Io(e.to_string()))?;
        let mut pipe = redis::pipe();
        pipe.cmd("ZADD").arg(&key).arg(record.ts).arg(blob).ignore();
        if let Some(ttl) = self.ttl {
            pipe.cmd("EXPIRE").arg(&key).arg(ttl).ignore();
        }
        pipe.query_async::<()>(&mut conn).await.map_err(redis_error)?;
        Ok(record)
    }

    async fn all(&self, query: RecordQuery) -> Result<Vec<MemoryRecord>> {
        let mut conn = self.link.conn().await?;
        let namespace = Self::namespace(query.trace.as_ref(), None);
        let scopes = match &query.scope {
            Some(s) if !s.is_empty() => vec![s.clone()],
            _ => self.scopes(&namespace).await?,
        };
        let limit = query.limit.filter(|n| *n > 0);
        let mut rows: Vec<MemoryRecord> = Vec::new();
        for one in scopes {
            // Newest first from Redis, so a limit is applied server-side.
            let stop: i64 = limit.map_or(-1, |n| n as i64 - 1);
            let raw: Vec<String> =
                redis::cmd("ZREVRANGE").arg(self.records_key(&namespace, &one)).arg(0).arg(stop).query_async(&mut conn).await.map_err(redis_error)?;
            rows.extend(raw.iter().filter_map(|blob| serde_json::from_str::<MemoryRecord>(blob).ok()));
        }
        rows.retain(|r| query.kind.as_ref().is_none_or(|k| &r.kind == k) && query.trace.as_ref().is_none_or(|t| t.matches(r)));
        rows.sort_by(|a, b| a.ts.total_cmp(&b.ts));
        Ok(last(rows, limit))
    }

    async fn clear(&self, scope: Option<&str>, trace: Option<&Trace>) -> Result<()> {
        let namespace = Self::namespace(trace, None);
        let scopes = match scope {
            Some(s) if !s.is_empty() => vec![s.to_string()],
            _ => self.scopes(&namespace).await?,
        };
        let mut keys: Vec<String> = scopes.iter().map(|s| self.records_key(&namespace, s)).collect();
        if trace.is_some() || scope.is_none() {
            let pattern = match trace {
                Some(t) => format!("{}:doc:{}/*", self.prefix, t.slug()),
                None => format!("{}:doc:*", self.prefix),
            };
            keys.extend(self.link.scan(&pattern).await?);
        }
        if !keys.is_empty() {
            let mut conn = self.link.conn().await?;
            redis::cmd("DEL").arg(&keys).query_async::<()>(&mut conn).await.map_err(redis_error)?;
        }
        Ok(())
    }

    async fn read_doc(&self, name: &str, trace: Option<&Trace>) -> Result<String> {
        let mut conn = self.link.conn().await?;
        let found: Option<String> = redis::cmd("GET").arg(self.doc_key(name, trace)).query_async(&mut conn).await.map_err(redis_error)?;
        Ok(found.unwrap_or_default())
    }

    async fn write_doc(&self, name: &str, text: &str, trace: Option<&Trace>) -> Result<()> {
        let mut conn = self.link.conn().await?;
        let key = self.doc_key(name, trace);
        // Documents outlive records by design, so the TTL does not apply here.
        redis::cmd("SET").arg(&key).arg(text).query_async::<()>(&mut conn).await.map_err(redis_error)?;
        redis::cmd("HSET").arg(format!("{}:doc_meta", self.prefix)).arg(&key).arg(now().to_string()).query_async::<()>(&mut conn).await.map_err(redis_error)?;
        Ok(())
    }

    async fn close(&self) {
        self.link.close().await;
    }

    fn describe(&self) -> String {
        format!("redis ({})", self.prefix)
    }

    /// Sessions on this store's connection. `options`: `ttl` (seconds a
    /// conversation lives after its last save), `prefix`.
    fn sessions(&self, options: &Map<String, Value>) -> Option<Result<Arc<dyn SessionStore>>> {
        let ttl = options.get("ttl").and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64)).or_else(|| v.as_str().and_then(|s| s.parse().ok())));
        let prefix = options.get("prefix").and_then(Value::as_str);
        Some(Ok(Arc::new(crate::sessions::redis::RedisSessionStore::new(self, prefix, ttl))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_match_python() {
        let store = RedisMemory::from_url("redis://localhost:6379/0").unwrap();
        let alice = Trace::user("alice").tenant("acme");
        assert_eq!(store.records_key(&RedisMemory::namespace(Some(&alice), None), "user"), "agent-memory:rec:t-acme__u-alice:user");
        assert_eq!(store.doc_key("user.md", Some(&alice)), "agent-memory:doc:t-acme__u-alice/user.md");
        assert_eq!(store.doc_key("user.md", None), "agent-memory:doc:user.md");
        let mut r = MemoryRecord::new("user", "fact", "x");
        r.user_id = Some("bob".into());
        r.session_id = Some("s1".into());
        assert_eq!(RedisMemory::namespace(None, Some(&r)), "u-bob", "a record's namespace is at user depth, like Trace(...).slug");
        assert_eq!(RedisMemory::namespace(None, None), "_shared");
        let other = store.with_prefix("chats:");
        assert_eq!(other.records_key("_shared", "project"), "chats:rec:_shared:project");
        assert!(RedisMemory::from_url("nope://x").is_err());
    }

    #[tokio::test]
    async fn live_round_trip() {
        let Ok(url) = std::env::var("AH_LIVE_REDIS_URL") else { return };
        let prefix = format!("ah-test-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        let store = RedisMemory::from_url(&url).unwrap().with_prefix(&prefix).with_ttl(Some(600));
        let alice = Trace::user("alice");
        let mut a = MemoryRecord::new("user", "decision", "Bill in EUR always");
        alice.stamp(&mut a);
        store.append(a).await.unwrap();
        let mut b = MemoryRecord::new("user", "fact", "Bob likes tea");
        Trace::user("bob").stamp(&mut b);
        store.append(b).await.unwrap();
        let mine = store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap();
        assert_eq!(mine.len(), 1);
        assert_eq!(mine[0].text, "Bill in EUR always");
        let found = store.search("bill EUR", None, 5, Some(&alice)).await.unwrap();
        assert_eq!(found.len(), 1);
        store.write_doc("user.md", "- likes EUR", Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "- likes EUR");
        assert_eq!(store.read_doc("user.md", Some(&Trace::user("bob"))).await.unwrap(), "");
        store.clear(None, Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "");
        assert!(store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap().is_empty());
        assert_eq!(store.all(RecordQuery::default().traced(Some(&Trace::user("bob")))).await.unwrap().len(), 1);
        store.clear(None, Some(&Trace::user("bob"))).await.unwrap();
        let mut conn = store.link().conn().await.unwrap();
        redis::cmd("DEL").arg(format!("{prefix}:doc_meta")).query_async::<()>(&mut conn).await.unwrap();
        store.close().await;
    }
}
