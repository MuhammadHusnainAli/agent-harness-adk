//! Where conversations are kept.
//!
//! ```no_run
//! # use agent_harness::sessions::session_provider;
//! let store = session_provider("postgresql://user:pass@host/agents");
//! let store = session_provider("sqlite:///./chats.db");
//! let store = session_provider("s3://my-bucket/chats");
//! ```
//!
//! | store | URL | Cargo feature |
//! |---|---|---|
//! | in process | `memory://` | — |
//! | files | `file:///path` or a bare path | — |
//! | SQLite | `sqlite:///chats.db` | `sqlite` (default) |
//! | PostgreSQL | `postgresql://…` | `postgres` |
//! | MySQL / MariaDB | `mysql://…` | `mysql` |
//! | MongoDB, Cosmos DB (Mongo API) | `mongodb://…` | `mongodb` |
//! | Redis | `redis://…` | `redis` |
//! | DynamoDB | `dynamodb://table` | — |
//! | Amazon S3 | `s3://bucket/prefix` | — |
//! | Azure Blob | `azure://container` | — |
//! | Google Cloud Storage | `gs://bucket/prefix` | — |
//!
//! The URLs, the drivers and the connections are the memory backends' own, so
//! there is one place where a database is configured — and
//! [`session_store_on`] keeps chats on the connections memory already holds.
//! Every one keeps a session with its owner and its version, and refuses a
//! stale save. The rows and objects are the Python harness's, so a session
//! saved by one is read by the other.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::errors::{Error, Result};
use crate::memory::base::MemoryStore;
use crate::memory::providers::{available, backend_for, memory_provider};
use crate::runtime::session::{FileSessionStore, InMemorySessionStore, SessionStore};

pub mod dynamodb;
pub mod mongo;
pub mod objects;
pub mod redis;
pub mod sql;

/// Memory backends that can hold sessions.
pub const SESSION_BACKENDS: [&str; 9] = ["sqlite", "postgres", "mysql", "mongo", "redis", "dynamodb", "s3", "azure", "gcs"];

/// Which stores can be used in this build.
pub fn session_backends() -> BTreeMap<String, bool> {
    let ready = available();
    let mut out: BTreeMap<String, bool> = SESSION_BACKENDS.iter().map(|n| (n.to_string(), ready.get(*n).copied().unwrap_or(false))).collect();
    out.insert("memory".into(), true);
    out.insert("file".into(), true);
    out
}

/// A session store on a memory backend you already have — the same pool.
///
/// `options` are the session store's own: `table` (SQL), `collection`
/// (MongoDB), `ttl` seconds (Redis), `concurrency` (object stores).
pub fn session_store_on(store: &Arc<dyn MemoryStore>, options: &Map<String, Value>) -> Result<Arc<dyn SessionStore>> {
    store.sessions(options).unwrap_or_else(|| {
        Err(Error::config(format!("{} cannot hold sessions; use SQL, MongoDB, Redis, DynamoDB or an object store — or write a SessionStore", store.describe())))
    })
}

/// A session store from a connection string.
pub fn session_provider(url: &str) -> Result<Arc<dyn SessionStore>> {
    session_provider_with(url, &Map::new())
}

/// A session store from a connection string, with the store's own options.
pub fn session_provider_with(url: &str, options: &Map<String, Value>) -> Result<Arc<dyn SessionStore>> {
    let scheme = url.split_once("://").map(|(s, _)| s.to_ascii_lowercase()).unwrap_or_default();
    if scheme == "memory" {
        return Ok(Arc::new(InMemorySessionStore::new()));
    }
    if scheme.is_empty() || scheme == "file" {
        let path = url.split_once("://").map_or(url, |(_, r)| r);
        return Ok(Arc::new(FileSessionStore::new(if path.is_empty() { ".harness/sessions" } else { path })?));
    }
    let name = backend_for(url)?;
    if !SESSION_BACKENDS.contains(&name.as_str()) {
        return Err(Error::config(format!("no session store for '{scheme}'; known: memory, file, {}", SESSION_BACKENDS.join(", "))));
    }
    session_store_on(&memory_provider(url)?, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_pick_stores() {
        assert!(session_provider("memory://").is_ok());
        assert!(session_provider("https://example.com").is_err());
        assert!(session_backends()["file"]);
        let mem: Arc<dyn MemoryStore> = Arc::new(crate::memory::InMemoryStore::new());
        assert!(session_store_on(&mem, &Map::new()).is_err());
    }
}
