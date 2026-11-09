//! The vector databases that ship.
//!
//! Stores spoken to over HTTP are always built in. The ones that need a
//! database driver sit behind a Cargo feature: `postgres` (pgvector), `redis`,
//! `mongodb`; `sqlite` is on by default.

use std::sync::Arc;

use indexmap::IndexMap;
use serde_json::{Value, json};

use crate::errors::{Error, Result};
use crate::knowledge::base::VectorStore;

pub mod aws;
pub mod azure;
pub mod chroma;
pub mod cloudflare;
pub mod local;
pub mod milvus;
pub mod mongodb;
pub mod opensearch;
pub mod pgvector;
pub mod pinecone;
pub mod qdrant;
pub mod redis;
pub mod upstash;
pub mod vertex;
pub mod weaviate;

pub use local::MemoryVectorStore;
#[cfg(feature = "sqlite")]
pub use local::SqliteVectorStore;

/// name → (type, the Cargo feature it needs beyond the defaults)
pub const STORES: [(&str, &str, &str); 17] = [
    ("memory", "MemoryVectorStore", ""),
    ("sqlite", "SqliteVectorStore", "sqlite"),
    ("qdrant", "QdrantStore", ""),
    ("chroma", "ChromaStore", ""),
    ("weaviate", "WeaviateStore", ""),
    ("milvus", "MilvusStore", ""),
    ("pinecone", "PineconeStore", ""),
    ("opensearch", "OpenSearchStore", ""),
    ("elasticsearch", "ElasticsearchStore", ""),
    ("pgvector", "PgVectorStore", "postgres"),
    ("redis", "RedisVectorStore", "redis"),
    ("mongodb", "MongoVectorStore", "mongodb"),
    ("azure-ai-search", "AzureAISearchStore", ""),
    ("cloudflare-vectorize", "VectorizeStore", ""),
    ("upstash", "UpstashStore", ""),
    ("vertex-vector-search", "VertexVectorStore", ""),
    ("s3-vectors", "S3VectorsStore", ""),
];

const ALIASES: [(&str, &str); 24] = [
    ("zilliz", "milvus"),
    ("postgres", "pgvector"),
    ("postgresql", "pgvector"),
    ("supabase", "pgvector"),
    ("neon", "pgvector"),
    ("elastic", "elasticsearch"),
    ("es", "elasticsearch"),
    ("aoss", "opensearch"),
    ("atlas", "mongodb"),
    ("mongo", "mongodb"),
    ("mongodb+srv", "mongodb"),
    ("azure", "azure-ai-search"),
    ("azure-search", "azure-ai-search"),
    ("vectorize", "cloudflare-vectorize"),
    ("cloudflare", "cloudflare-vectorize"),
    ("vertex", "vertex-vector-search"),
    ("vertex-ai", "vertex-vector-search"),
    ("s3vectors", "s3-vectors"),
    ("s3", "s3-vectors"),
    ("rediss", "redis"),
    ("file", "sqlite"),
    ("chromadb", "chroma"),
    ("opensearch-serverless", "opensearch"),
    ("pg", "pgvector"),
];

/// Every store that ships, and the Cargo feature it needs, if it needs one.
pub fn vector_stores() -> Vec<Value> {
    STORES.iter().map(|(name, ty, needs)| json!({"name": name, "class": ty, "needs": needs})).collect()
}

fn canonical(name: &str) -> Result<&'static str> {
    let lower = name.to_ascii_lowercase();
    let key = ALIASES.iter().find(|(a, _)| *a == lower).map(|(_, k)| *k).unwrap_or(lower.as_str());
    STORES
        .iter()
        .find(|(n, _, _)| *n == key)
        .map(|(n, _, _)| *n)
        .ok_or_else(|| Error::config(format!("no vector store named '{name}'; there are: {}", STORES.iter().map(|s| s.0).collect::<Vec<_>>().join(", "))))
}

/// A store URL, taken apart the way every store reads it.
#[derive(Debug, Clone, Default)]
pub struct UrlParts {
    /// The scheme as written, before any `+https`: `qdrant`, `mongodb+srv`.
    pub scheme: String,
    /// `https` when the URL asked for TLS with `+https` or `?tls=1`.
    pub secure: bool,
    /// `user:pass@host:port`.
    pub netloc: String,
    /// The path without its leading and trailing slashes, decoded.
    pub path: String,
    /// The path exactly as written, leading slash and all.
    pub raw_path: String,
    /// Everything after `?`, in order.
    pub given: IndexMap<String, String>,
}

impl UrlParts {
    /// `http(s)://netloc`.
    pub fn web(&self) -> String {
        format!("{}://{}", if self.secure { "https" } else { "http" }, self.netloc)
    }

    /// Take an option out of the query string.
    pub fn take(&mut self, key: &str) -> Option<String> {
        self.given.shift_remove(key)
    }

    /// What is left of the query string, as `a=b&c=d`.
    pub fn rest_query(&self) -> String {
        self.given.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&")
    }

    pub fn parse(spec: &str) -> Result<Self> {
        let (scheme, rest) = spec.split_once("://").ok_or_else(|| Error::config(format!("'{spec}' is not a URL")))?;
        let (rest, query) = rest.split_once('?').unwrap_or((rest, ""));
        let (netloc, raw_path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, ""),
        };
        let mut given: IndexMap<String, String> =
            query.split('&').filter(|p| !p.is_empty()).map(|p| p.split_once('=').unwrap_or((p, ""))).map(|(k, v)| (decode(k), decode(v))).collect();
        // `qdrant+https://…` — but `mongodb+srv` is a scheme of its own.
        let (base, plus) = scheme.split_once('+').unwrap_or((scheme, ""));
        let tls = given.shift_remove("tls").is_some_and(|v| v == "1" || v == "true");
        let (scheme, secure) = if plus == "https" || plus == "http" { (base.to_string(), plus == "https" || tls) } else { (scheme.to_string(), tls) };
        Ok(Self { scheme, secure, netloc: netloc.into(), path: decode(raw_path.trim_matches('/')), raw_path: raw_path.into(), given })
    }
}

/// `%xx` decoding, enough for a URL's path and query.
pub fn decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(b) = u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("zz"), 16)
        {
            out.push(b);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[allow(dead_code)] // unused when every feature is on
fn needs(feature: &str, name: &str) -> Error {
    Error::config(format!("the '{name}' vector store needs the `{feature}` feature: agent-harness-adk = {{ version = \"*\", features = [\"{feature}\"] }}"))
}

/// A vector store from its name, or from a URL.
///
/// ```no_run
/// # use agent_harness::knowledge::vector_store;
/// let store = vector_store("qdrant://localhost:6333/handbook").unwrap();
/// let store = vector_store("pgvector://user:pass@host/db?table=handbook");
/// let store = vector_store("opensearch+https://host:9200/handbook");
/// let store = vector_store("sqlite:///knowledge.db");
/// ```
///
/// In a URL the path names the collection, `+https` asks for TLS, and anything
/// after `?` is passed to the store as an option. A bare name builds the store
/// with its defaults (and its keys from the environment).
pub fn vector_store(spec: &str) -> Result<Arc<dyn VectorStore>> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err(Error::config("a vector store is a name or a URL — got nothing"));
    }
    let parts = if spec.contains("://") { UrlParts::parse(spec)? } else { UrlParts { scheme: spec.into(), ..Default::default() } };
    let bare = !spec.contains("://");
    let key = canonical(&parts.scheme)?;
    let p = parts;
    Ok(match key {
        "memory" => Arc::new(MemoryVectorStore::new()),
        "sqlite" => {
            #[cfg(feature = "sqlite")]
            {
                let mut p = p;
                let table = p.take("table").unwrap_or_else(|| "vectors".into());
                let file = if bare {
                    "knowledge.db".to_string()
                } else if p.netloc.is_empty() {
                    decode(p.raw_path.strip_prefix('/').unwrap_or(&p.raw_path))
                } else {
                    decode(&format!("{}{}", p.netloc, p.raw_path))
                };
                Arc::new(SqliteVectorStore::new(if file.is_empty() { "knowledge.db" } else { &file }, &table)?)
            }
            #[cfg(not(feature = "sqlite"))]
            {
                let _ = (p, bare);
                return Err(needs("sqlite", "sqlite"));
            }
        }
        "qdrant" => Arc::new(qdrant::QdrantStore::from_url(p, bare)?),
        "chroma" => Arc::new(chroma::ChromaStore::from_url(p, bare)?),
        "weaviate" => Arc::new(weaviate::WeaviateStore::from_url(p, bare)?),
        "milvus" => Arc::new(milvus::MilvusStore::from_url(p, bare)?),
        "pinecone" => Arc::new(pinecone::PineconeStore::from_url(p, bare)?),
        "opensearch" => Arc::new(opensearch::OpenSearchStore::from_url(p, bare, false)?),
        "elasticsearch" => Arc::new(opensearch::OpenSearchStore::from_url(p, bare, true)?),
        "azure-ai-search" => Arc::new(azure::AzureAISearchStore::from_url(p, bare)?),
        "cloudflare-vectorize" => Arc::new(cloudflare::VectorizeStore::from_url(p, bare)?),
        "upstash" => Arc::new(upstash::UpstashStore::from_url(p, bare)?),
        "vertex-vector-search" => Arc::new(vertex::VertexVectorStore::from_url(p, bare)?),
        "s3-vectors" => Arc::new(aws::S3VectorsStore::from_url(p, bare)?),
        "pgvector" => {
            #[cfg(feature = "postgres")]
            {
                Arc::new(pgvector::PgVectorStore::from_url(p, bare)?)
            }
            #[cfg(not(feature = "postgres"))]
            {
                let _ = p;
                return Err(needs("postgres", "pgvector"));
            }
        }
        "redis" => {
            #[cfg(feature = "redis")]
            {
                Arc::new(redis::RedisVectorStore::from_url(p, bare)?)
            }
            #[cfg(not(feature = "redis"))]
            {
                let _ = p;
                return Err(needs("redis", "redis"));
            }
        }
        "mongodb" => {
            #[cfg(feature = "mongodb")]
            {
                Arc::new(mongodb::MongoVectorStore::from_url(p, bare)?)
            }
            #[cfg(not(feature = "mongodb"))]
            {
                let _ = p;
                return Err(needs("mongodb", "mongodb"));
            }
        }
        other => return Err(Error::config(format!("'{other}' cannot be given as a URL — build it directly"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_come_apart() {
        let p = UrlParts::parse("qdrant+https://user:pw@host:6333/hand%20book?api_key=k&tls=0").unwrap();
        assert_eq!((p.scheme.as_str(), p.secure, p.netloc.as_str(), p.path.as_str()), ("qdrant", true, "user:pw@host:6333", "hand book"));
        assert_eq!(p.given.get("api_key").map(String::as_str), Some("k"));
        assert_eq!(p.web(), "https://user:pw@host:6333");
        let m = UrlParts::parse("mongodb+srv://c.example.net/db").unwrap();
        assert_eq!((m.scheme.as_str(), m.secure), ("mongodb+srv", false));
        assert_eq!(canonical("Zilliz").unwrap(), "milvus");
        assert!(vector_store("nope").is_err());
        assert_eq!(vector_store("memory").unwrap().name(), "memory");
        assert_eq!(vector_stores().len(), 17);
    }
}
