//! Sessions in Redis. Fast, and the natural home for chats with a lifetime.
//!
//! A conversation is a hash; a sorted set per owner, scored by when it last
//! changed, is what makes "newest first" a single call. The version check and
//! the write happen inside one Lua script, so they are atomic on the server.
#![cfg(feature = "redis")]

use std::sync::Arc;

use async_trait::async_trait;

use crate::errors::{Error, Result};
use crate::memory::providers::redis::{RedisLink, RedisMemory, redis_error};
use crate::runtime::session::{DurableBackend, ListFilter, Session};

/// KEYS: the session, the index of every session, the owner's index.
/// ARGV: expected version, new version, payload, updated, id, ttl, agent.
const SAVE: &str = r#"
local held = redis.call('HGET', KEYS[1], 'version')
if (held == false and ARGV[1] == '0') or (held == ARGV[1]) then
  redis.call('HSET', KEYS[1], 'version', ARGV[2], 'payload', ARGV[3],
             'owner', KEYS[3], 'agent', ARGV[7])
  redis.call('ZADD', KEYS[2], ARGV[4], ARGV[5])
  redis.call('ZADD', KEYS[3], ARGV[4], ARGV[5])
  if tonumber(ARGV[6]) > 0 then redis.call('EXPIRE', KEYS[1], ARGV[6]) end
  return 1
end
return 0
"#;

/// Sessions in Redis, over the Redis memory backend's connection.
///
/// ```no_run
/// # use agent_harness::memory::providers::redis::RedisMemory;
/// # use agent_harness::sessions::redis::RedisSessionStore;
/// let memory = RedisMemory::from_url("redis://host:6379/0").unwrap();
/// let store = RedisSessionStore::new(&memory, None, Some(30 * 86_400));
/// ```
///
/// `ttl` lets a conversation expire that long after it was last saved; `None`
/// keeps it until it is deleted. The keys of one save live under one prefix but
/// not one hash slot, so on Redis Cluster wrap the prefix in braces —
/// `{chats}` — to keep them together.
#[derive(Debug, Clone)]
pub struct RedisSessionStore {
    link: Arc<RedisLink>,
    pub prefix: String,
    pub ttl: Option<u64>,
}

impl RedisSessionStore {
    /// On `redis`'s connection, under `prefix` (the memory's own by default).
    pub fn new(redis: &RedisMemory, prefix: Option<&str>, ttl: Option<u64>) -> Self {
        let prefix = prefix.unwrap_or(&redis.prefix).trim_end_matches(':').to_string();
        Self { link: redis.link(), prefix, ttl }
    }

    pub(crate) fn key(&self, session_id: &str) -> String {
        format!("{}:session:{session_id}", self.prefix)
    }

    /// The index of every session.
    pub(crate) fn everything(&self) -> String {
        format!("{}:sessions:_all", self.prefix)
    }

    /// One owner's index.
    pub(crate) fn index(&self, tenant_id: Option<&str>, user_id: Option<&str>) -> String {
        let or = |v: Option<&str>| v.filter(|s| !s.is_empty()).unwrap_or("_").to_string();
        format!("{}:sessions:{}:{}", self.prefix, or(tenant_id), or(user_id))
    }

    /// A session from its hash: the payload, with the version the hash holds.
    fn parse(row: &std::collections::HashMap<String, String>) -> Option<Session> {
        let mut found: Session = serde_json::from_str(row.get("payload")?).ok()?;
        if let Some(v) = row.get("version").and_then(|v| v.parse::<u64>().ok()) {
            found.version = v;
        }
        Some(found)
    }
}

#[async_trait]
impl DurableBackend for RedisSessionStore {
    async fn read(&self, session_id: &str) -> Result<Option<Session>> {
        let mut conn = self.link.conn().await?;
        let row: std::collections::HashMap<String, String> =
            redis::cmd("HGETALL").arg(self.key(session_id)).query_async(&mut conn).await.map_err(redis_error)?;
        Ok(Self::parse(&row))
    }

    async fn write(&self, session: &Session, expect: u64) -> Result<bool> {
        let mut conn = self.link.conn().await?;
        let payload = serde_json::to_string(session).map_err(|e| Error::Io(e.to_string()))?;
        let done: i64 = redis::cmd("EVAL")
            .arg(SAVE)
            .arg(3)
            .arg(self.key(&session.id))
            .arg(self.everything())
            .arg(self.index(session.tenant_id.as_deref(), session.user_id.as_deref()))
            .arg(expect.to_string())
            .arg(session.version.to_string())
            .arg(payload)
            .arg(session.updated.to_string())
            .arg(&session.id)
            .arg(self.ttl.unwrap_or(0).to_string())
            .arg(&session.agent)
            .query_async(&mut conn)
            .await
            .map_err(redis_error)?;
        Ok(done == 1)
    }

    async fn list_sessions(&self, filter: &ListFilter) -> Result<Vec<Session>> {
        let mut conn = self.link.conn().await?;
        let (user_id, tenant_id, agent) = (filter.user_id.as_deref(), filter.tenant_id.as_deref(), filter.agent.as_deref());
        // One owner's index when both are named; otherwise everything, filtered.
        let index = if user_id.is_some() && tenant_id.is_some() { self.index(tenant_id, user_id) } else { self.everything() };
        let limit = filter.limit.max(1);
        let page = (limit * 2).max(50);
        let mut out: Vec<Session> = Vec::new();
        let mut start = 0usize;
        while out.len() < limit {
            let ids: Vec<String> = redis::cmd("ZREVRANGE").arg(&index).arg(start).arg(start + page - 1).query_async(&mut conn).await.map_err(redis_error)?;
            if ids.is_empty() {
                break;
            }
            for session_id in ids {
                let Some(found) = self.read(&session_id).await? else {
                    // Expired: tidy up.
                    redis::cmd("ZREM").arg(&index).arg(&session_id).query_async::<()>(&mut conn).await.map_err(redis_error)?;
                    continue;
                };
                if user_id.is_none_or(|u| found.user_id.as_deref() == Some(u))
                    && tenant_id.is_none_or(|t| found.tenant_id.as_deref() == Some(t))
                    && agent.is_none_or(|a| found.agent == a)
                {
                    out.push(found);
                    if out.len() >= limit {
                        break;
                    }
                }
            }
            start += page;
        }
        Ok(out)
    }

    async fn remove(&self, session_id: &str) -> Result<()> {
        let mut conn = self.link.conn().await?;
        let key = self.key(session_id);
        let owner: Option<String> = redis::cmd("HGET").arg(&key).arg("owner").query_async(&mut conn).await.map_err(redis_error)?;
        let mut pipe = redis::pipe();
        pipe.atomic();
        pipe.cmd("DEL").arg(&key).ignore();
        pipe.cmd("ZREM").arg(self.everything()).arg(session_id).ignore();
        if let Some(owner) = owner.filter(|o| !o.is_empty()) {
            pipe.cmd("ZREM").arg(owner).arg(session_id).ignore();
        }
        pipe.query_async::<()>(&mut conn).await.map_err(redis_error)?;
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "RedisSessionStore"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::session::SessionStore;

    #[test]
    fn keys_match_python() {
        let memory = RedisMemory::from_url("redis://localhost:6379/0").unwrap();
        let store = RedisSessionStore::new(&memory, None, None);
        assert_eq!(store.key("ses_1"), "agent-memory:session:ses_1");
        assert_eq!(store.everything(), "agent-memory:sessions:_all");
        assert_eq!(store.index(Some("acme"), Some("alice")), "agent-memory:sessions:acme:alice");
        assert_eq!(store.index(None, Some("")), "agent-memory:sessions:_:_");
        let braced = RedisSessionStore::new(&memory, Some("{chats}:"), Some(60));
        assert_eq!(braced.key("x"), "{chats}:session:x");

        let s = Session { version: 1, ..Session::new("bot") };
        let row: std::collections::HashMap<String, String> =
            [("payload".to_string(), serde_json::to_string(&s).unwrap()), ("version".to_string(), "4".to_string())].into_iter().collect();
        let back = RedisSessionStore::parse(&row).unwrap();
        assert_eq!((back.id.as_str(), back.version), (s.id.as_str(), 4));
        assert!(RedisSessionStore::parse(&Default::default()).is_none());
    }

    #[tokio::test]
    async fn live_check_and_conflict() {
        let Ok(url) = std::env::var("AH_LIVE_REDIS_URL") else { return };
        let prefix = format!("ah-test-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        let memory = RedisMemory::from_url(&url).unwrap().with_prefix(&prefix);
        let store = RedisSessionStore::new(&memory, None, Some(600));
        let report = store.check().await;
        assert_eq!(report["ok"], true, "{report}");

        let mut s = Session { user_id: Some("u".into()), tenant_id: Some("t".into()), ..Session::new("bot") };
        store.save(&mut s).await.unwrap();
        let mut stale = store.load(&s.id).await.unwrap();
        store.save(&mut s).await.unwrap();
        assert!(matches!(store.save(&mut stale).await, Err(Error::SessionConflict { .. })));
        let listed = store.list(&ListFilter { limit: 5, user_id: Some("u".into()), tenant_id: Some("t".into()), agent: None }).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].version, 2);
        store.delete(&s.id).await.unwrap();
        assert!(store.load(&s.id).await.is_err());
        let mut conn = memory.link().conn().await.unwrap();
        redis::cmd("DEL").arg(store.everything()).arg(store.index(Some("t"), Some("u"))).query_async::<()>(&mut conn).await.unwrap();
    }
}
