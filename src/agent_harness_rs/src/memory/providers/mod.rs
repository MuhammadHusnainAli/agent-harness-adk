//! Where memory is stored.
//!
//! Every backend implements [`MemoryStore`], so they are interchangeable — the
//! agent loop never learns which one it is talking to. Each is scoped by
//! [`Trace`](crate::memory::Trace), so one store serves many users without
//! them seeing each other.
//!
//! ```no_run
//! # use agent_harness::memory::memory_provider;
//! let store = memory_provider("postgresql://user:pass@host/agents");
//! let store = memory_provider("s3://my-bucket/memory");
//! let store = memory_provider("sqlite:///./memory.db");
//! ```
//!
//! | backend | URL | Cargo feature |
//! |---|---|---|
//! | in-process | `memory://` | — |
//! | files | `file:///path` or a bare path | — |
//! | SQLite | `sqlite:///memory.db` | `sqlite` (default) |
//! | PostgreSQL | `postgresql://…` | `postgres` |
//! | MySQL / MariaDB | `mysql://…` | `mysql` |
//! | MongoDB | `mongodb://…` | `mongodb` |
//! | Redis | `redis://…` | `redis` |
//! | DynamoDB | `dynamodb://table` | — |
//! | Elasticsearch / OpenSearch | `es://host:9200` | — |
//! | Amazon S3 (and MinIO) | `s3://bucket/prefix` | — |
//! | Azure Blob | `azure://container` | — |
//! | Google Cloud Storage | `gs://bucket/prefix` | — |
//! | your own API | `https://…` | — |
//!
//! The ones without a feature speak HTTP and are always built in.

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, RwLock};

use crate::errors::{Error, Result};
use crate::memory::base::{FileStore, InMemoryStore, MemoryStore};

pub mod azure_blob;
pub mod dynamodb;
pub mod elasticsearch;
pub mod gcs;
pub mod http;
pub mod mongo;
pub mod mysql;
pub mod objects;
pub mod postgres;
pub mod redis;
pub mod s3;
pub mod sql;
pub mod sqlite;

/// Builds a store from its URL.
pub type Factory = Arc<dyn Fn(&str) -> Result<Arc<dyn MemoryStore>> + Send + Sync>;

/// URL scheme → backend name.
pub const SCHEMES: [(&str, &str); 27] = [
    ("memory", "memory"),
    ("", "file"),
    ("file", "file"),
    ("sqlite", "sqlite"),
    ("sqlite3", "sqlite"),
    ("postgres", "postgres"),
    ("postgresql", "postgres"),
    ("mysql", "mysql"),
    ("mariadb", "mysql"),
    ("mongodb", "mongo"),
    ("mongodb+srv", "mongo"),
    ("mongo", "mongo"),
    ("redis", "redis"),
    ("rediss", "redis"),
    ("dynamodb", "dynamodb"),
    ("es", "elasticsearch"),
    ("elasticsearch", "elasticsearch"),
    ("opensearch", "elasticsearch"),
    ("s3", "s3"),
    ("minio", "s3"),
    ("azure", "azure"),
    ("abfs", "azure"),
    ("blob", "azure"),
    ("gs", "gcs"),
    ("gcs", "gcs"),
    ("http", "http"),
    ("https", "http"),
];

/// backend → the Cargo feature it needs ("" for none).
pub const FEATURES: [(&str, &str); 13] = [
    ("memory", ""),
    ("file", ""),
    ("sqlite", "sqlite"),
    ("postgres", "postgres"),
    ("mysql", "mysql"),
    ("mongo", "mongodb"),
    ("redis", "redis"),
    ("dynamodb", ""),
    ("elasticsearch", ""),
    ("s3", ""),
    ("azure", ""),
    ("gcs", ""),
    ("http", ""),
];

struct Registry {
    factories: BTreeMap<String, Factory>,
    schemes: BTreeMap<String, String>,
}

static REGISTRY: LazyLock<RwLock<Registry>> =
    LazyLock::new(|| RwLock::new(Registry { factories: BTreeMap::new(), schemes: SCHEMES.iter().map(|(s, n)| (s.to_string(), n.to_string())).collect() }));

/// Add your own backend: `register_backend("cassandra", &["cassandra"], factory)`.
/// It takes over its schemes, and a name that ships is replaced.
pub fn register_backend(name: &str, schemes: &[&str], factory: Factory) {
    let mut r = REGISTRY.write().unwrap_or_else(|p| p.into_inner());
    r.factories.insert(name.into(), factory);
    let list: Vec<&str> = if schemes.is_empty() { vec![name] } else { schemes.to_vec() };
    for s in list {
        r.schemes.insert(s.to_ascii_lowercase(), name.into());
    }
}

#[allow(clippy::match_like_matches_macro)] // a match on cfg! reads better per feature
fn compiled(feature: &str) -> bool {
    match feature {
        "" => true,
        "sqlite" => cfg!(feature = "sqlite"),
        "postgres" => cfg!(feature = "postgres"),
        "mysql" => cfg!(feature = "mysql"),
        "mongodb" => cfg!(feature = "mongodb"),
        "redis" => cfg!(feature = "redis"),
        _ => false,
    }
}

/// Which backends this build can use — each needs its Cargo feature compiled in.
pub fn available() -> BTreeMap<String, bool> {
    let r = REGISTRY.read().unwrap_or_else(|p| p.into_inner());
    let mut out: BTreeMap<String, bool> = FEATURES.iter().map(|(n, f)| (n.to_string(), compiled(f))).collect();
    for name in r.factories.keys() {
        out.insert(name.clone(), true);
    }
    out
}

/// What to turn on for a backend that is not built in. Empty if it is.
pub fn missing_feature(name: &str) -> String {
    match FEATURES.iter().find(|(n, _)| *n == name) {
        Some((_, f)) if !compiled(f) => format!("enable the `{f}` feature: agent-harness-adk = {{ version = \"*\", features = [\"{f}\"] }}"),
        _ => String::new(),
    }
}

/// The backend a URL names, by its scheme.
pub fn backend_for(url: &str) -> Result<String> {
    let scheme = url.split_once("://").map(|(s, _)| s.to_ascii_lowercase()).unwrap_or_default();
    let r = REGISTRY.read().unwrap_or_else(|p| p.into_inner());
    r.schemes.get(&scheme).cloned().ok_or_else(|| {
        let known: Vec<&str> = r.schemes.keys().map(String::as_str).filter(|s| !s.is_empty()).collect();
        Error::config(format!("no memory backend for '{}'; known schemes: {}", if scheme.is_empty() { "a bare path" } else { &scheme }, known.join(", ")))
    })
}

/// The file a `sqlite://` URL names, after the scheme.
///
/// `sqlite://:memory:` is in memory, `sqlite:///chats.db` is `chats.db` here,
/// `sqlite:///./data/chats.db` is `data/chats.db` here, and
/// `sqlite:////var/lib/chats.db` (or `sqlite:///var/lib/chats.db`) is absolute.
pub fn sqlite_path(rest: &str) -> String {
    let rest = rest.split('?').next().unwrap_or("");
    if matches!(rest, "" | ":memory:" | "/:memory:") {
        return ":memory:".into();
    }
    if rest.starts_with("//") {
        return format!("/{}", rest.trim_start_matches('/'));
    }
    if rest.starts_with("/./") || rest.starts_with("/../") || (rest.starts_with('/') && !rest[1..].contains('/')) {
        return rest[1..].to_string();
    }
    rest.to_string()
}

#[allow(dead_code)]
fn needs(name: &str) -> Error {
    Error::config(format!("the '{name}' memory backend is not built in: {}", missing_feature(name)))
}

/// Build a store from a connection string.
///
/// ```no_run
/// # use agent_harness::memory::memory_provider;
/// memory_provider("sqlite:///./memory.db");
/// memory_provider("postgresql://user:pass@host/agents");
/// memory_provider("s3://my-bucket/agent-memory");
/// memory_provider("mongodb://localhost:27017/agents");
/// ```
pub fn memory_provider(url: &str) -> Result<Arc<dyn MemoryStore>> {
    let name = backend_for(url)?;
    let custom = REGISTRY.read().unwrap_or_else(|p| p.into_inner()).factories.get(&name).cloned();
    if let Some(f) = custom {
        return f(url);
    }
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    Ok(match name.as_str() {
        "memory" => Arc::new(InMemoryStore::new()),
        "file" => {
            let path = if url.contains("://") { rest.split('?').next().unwrap_or("") } else { url };
            Arc::new(FileStore::new(if path.is_empty() { ".harness/memory" } else { path })?)
        }
        "sqlite" => {
            #[cfg(feature = "sqlite")]
            {
                Arc::new(sqlite::SqliteMemory::open(&sqlite_path(rest))?)
            }
            #[cfg(not(feature = "sqlite"))]
            {
                return Err(needs("sqlite"));
            }
        }
        "postgres" => {
            #[cfg(feature = "postgres")]
            {
                Arc::new(postgres::PostgresMemory::from_url(url)?)
            }
            #[cfg(not(feature = "postgres"))]
            {
                return Err(needs("postgres"));
            }
        }
        "mysql" => {
            #[cfg(feature = "mysql")]
            {
                Arc::new(mysql::MySqlMemory::from_url(url)?)
            }
            #[cfg(not(feature = "mysql"))]
            {
                return Err(needs("mysql"));
            }
        }
        "redis" => {
            #[cfg(feature = "redis")]
            {
                Arc::new(redis::RedisMemory::from_url(url)?)
            }
            #[cfg(not(feature = "redis"))]
            {
                return Err(needs("redis"));
            }
        }
        "mongo" => {
            #[cfg(feature = "mongodb")]
            {
                Arc::new(mongo::MongoMemory::from_url(url)?)
            }
            #[cfg(not(feature = "mongodb"))]
            {
                return Err(needs("mongo"));
            }
        }
        "dynamodb" => Arc::new(dynamodb::DynamoDbMemory::from_url(url)?),
        "elasticsearch" => Arc::new(elasticsearch::ElasticsearchMemory::from_url(url)?),
        "s3" => Arc::new(s3::S3Memory::from_url(url)?),
        "azure" => Arc::new(azure_blob::AzureBlobMemory::from_url(url)?),
        "gcs" => Arc::new(gcs::GcsMemory::from_url(url)?),
        "http" => Arc::new(http::HttpMemory::from_url(url)?),
        other => return Err(Error::config(format!("unknown memory backend '{other}'"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_pick_backends() {
        assert_eq!(sqlite_path(":memory:"), ":memory:");
        assert_eq!(sqlite_path("/chats.db"), "chats.db");
        assert_eq!(sqlite_path("/./data/chats.db"), "./data/chats.db");
        assert_eq!(sqlite_path("//var/lib/chats.db"), "/var/lib/chats.db");
        assert_eq!(sqlite_path("/var/lib/chats.db"), "/var/lib/chats.db");
        assert_eq!(backend_for("postgresql://x/y").unwrap(), "postgres");
        assert_eq!(backend_for("gs://bucket").unwrap(), "gcs");
        assert!(backend_for("ftp://x").is_err());
        assert_eq!(memory_provider("memory://").unwrap().describe(), "in-memory");
        assert!(available()["memory"]);
    }
}
