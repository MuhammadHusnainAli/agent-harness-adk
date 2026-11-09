//! Chroma — a Chroma server, or Chroma Cloud.

use std::sync::Mutex;

use async_trait::async_trait;
use reqwest::Method;
use serde_json::{Map, Value, json};

use super::UrlParts;
use super::qdrant::{Settings, floats, no_more, number, object, or_env, py_dumps};
use crate::errors::Result;
use crate::knowledge::base::{Condition, HttpClient, VectorHit, VectorRecord, VectorStore, store_error};
use crate::llm_providers::resilience::RetryPolicy;

/// ```ignore
/// ChromaStore::new("http://localhost:8000").collection("handbook");
/// ChromaStore::new("https://api.trychroma.com").api_key("…").tenant("…").database("…");
/// ```
///
/// Chroma's metadata is flat and holds no lists, so a list is kept as its
/// members joined by a separator and is not filterable there.
#[derive(Debug)]
pub struct ChromaStore {
    pub url: String,
    pub collection: String,
    pub tenant: String,
    pub database: String,
    pub api_key: String,
    settings: Settings,
    http: HttpClient,
    /// The collection's id, once it has been asked for.
    id: Mutex<String>,
}

impl Default for ChromaStore {
    fn default() -> Self {
        Self::new("http://localhost:8000")
    }
}

impl ChromaStore {
    /// The key comes from `$CHROMA_API_KEY` unless one is given.
    pub fn new(url: &str) -> Self {
        let settings = Settings::default();
        let mut store = Self {
            url: url.trim_end_matches('/').to_string(),
            collection: "knowledge".into(),
            tenant: "default_tenant".into(),
            database: "default_database".into(),
            api_key: or_env("", "CHROMA_API_KEY"),
            http: settings.client("chroma", &[], &[]),
            settings,
            id: Mutex::new(String::new()),
        };
        store.rebuild();
        store
    }

    pub fn collection(mut self, collection: &str) -> Self {
        self.collection = collection.into();
        self
    }

    pub fn tenant(mut self, tenant: &str) -> Self {
        self.tenant = tenant.into();
        self
    }

    pub fn database(mut self, database: &str) -> Self {
        self.database = database.into();
        self
    }

    pub fn api_key(mut self, key: &str) -> Self {
        self.api_key = or_env(key, "CHROMA_API_KEY");
        self.rebuild();
        self
    }

    /// Seconds before a request is given up on (30 by default).
    pub fn timeout(mut self, seconds: f64) -> Self {
        self.settings.timeout = seconds;
        self.rebuild();
        self
    }

    /// How many times a call that may succeed later is tried again (3 by default).
    pub fn retries(mut self, retries: u32) -> Self {
        self.settings.retries = retries;
        self.rebuild();
        self
    }

    /// The whole retry policy, in place of `retries`.
    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.settings.retry = Some(policy);
        self.rebuild();
        self
    }

    /// A header sent with every request.
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.settings.headers.push((name.into(), value.into()));
        self.rebuild();
        self
    }

    /// `chroma://host:8000/collection?tenant=…&database=…&api_key=…`, or the
    /// defaults for a bare `chroma`.
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        if bare {
            return Ok(Self::default());
        }
        let settings = Settings::from_url(&mut p, "chroma")?;
        let named = p.take("collection");
        let tenant = p.take("tenant");
        let database = p.take("database");
        let key = p.take("api_key").unwrap_or_default();
        no_more(&p, "chroma")?;
        let mut store = Self::new(&p.web()).api_key(&key);
        store.settings = settings;
        store.rebuild();
        if !p.path.is_empty() {
            store.collection = p.path.clone();
        } else if let Some(c) = named {
            store.collection = c;
        }
        if let Some(t) = tenant {
            store.tenant = t;
        }
        if let Some(d) = database {
            store.database = d;
        }
        Ok(store)
    }

    fn rebuild(&mut self) {
        let own: Vec<(&str, String)> = if self.api_key.is_empty() { vec![] } else { vec![("x-chroma-token", self.api_key.clone())] };
        self.http = self.settings.client("chroma", &own, &[&self.api_key]);
    }

    fn collections(&self) -> String {
        format!("{}/api/v2/tenants/{}/databases/{}/collections", self.url, self.tenant, self.database)
    }

    fn known(&self) -> String {
        self.id.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The collection's address, created (cosine) the first time it is needed.
    async fn address(&self) -> Result<String> {
        let mut id = self.known();
        if id.is_empty() {
            let answer = self
                .http
                .json(
                    Method::POST,
                    &self.collections(),
                    Some(&json!({"name": self.collection, "get_or_create": true, "configuration": {"hnsw": {"space": "cosine"}}})),
                    &[],
                )
                .await?;
            id = answer.body.get("id").and_then(Value::as_str).ok_or_else(|| store_error("chroma", "chroma answered without the collection's id"))?.to_string();
            *self.id.lock().unwrap_or_else(|p| p.into_inner()) = id.clone();
        }
        Ok(format!("{}/{id}", self.collections()))
    }

    /// The filter in Chroma's terms: one condition, or `$and` of several.
    pub fn filter(wanted: &[Condition]) -> Option<Value> {
        let mut parts: Vec<Value> = wanted.iter().map(|c| json!({c.field.clone(): {c.op.clone(): c.value}})).collect();
        match parts.len() {
            0 => None,
            1 => parts.pop(),
            _ => Some(json!({"$and": parts})),
        }
    }

    /// Metadata as Chroma keeps it: no nulls, and a list or object written as
    /// JSON under `<name>__json`.
    pub fn flat(metadata: &Map<String, Value>) -> Value {
        let mut out = Map::new();
        for (key, value) in metadata {
            match value {
                Value::Null => {}
                Value::Array(_) | Value::Object(_) => {
                    out.insert(format!("{key}__json"), Value::String(py_dumps(value)));
                }
                other => {
                    out.insert(key.clone(), other.clone());
                }
            }
        }
        if out.is_empty() { Value::Null } else { Value::Object(out) }
    }

    /// The metadata as it was given.
    pub fn unflat(metadata: Option<&Value>) -> Map<String, Value> {
        let mut out = Map::new();
        for (key, value) in object(metadata) {
            if let Some(name) = key.strip_suffix("__json")
                && let Some(parsed) = value.as_str().and_then(|s| serde_json::from_str::<Value>(s).ok())
            {
                out.insert(name.to_string(), parsed);
                continue;
            }
            out.insert(key, value);
        }
        out
    }
}

/// The first list of a Chroma answer's list of lists.
fn first(answer: &Value, key: &str) -> Vec<Value> {
    answer.get(key).and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_array).cloned().unwrap_or_default()
}

#[async_trait]
impl VectorStore for ChromaStore {
    fn name(&self) -> &str {
        "chroma"
    }

    fn batch_size(&self) -> usize {
        500
    }

    async fn ensure(&self, _dimension: usize) -> Result<()> {
        self.address().await?;
        Ok(())
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        let body = json!({
            "ids": records.iter().map(|r| r.id.clone()).collect::<Vec<_>>(),
            "embeddings": records.iter().map(|r| floats(&r.vector)).collect::<Vec<_>>(),
            "documents": records.iter().map(|r| r.text.clone()).collect::<Vec<_>>(),
            "metadatas": records.iter().map(|r| Self::flat(&r.metadata)).collect::<Vec<_>>(),
        });
        self.http.json(Method::POST, &format!("{}/upsert", self.address().await?), Some(&body), &[]).await?;
        Ok(())
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let mut body = json!({"query_embeddings": [floats(vector)], "n_results": k, "include": ["documents", "metadatas", "distances"]});
        if let Some(found) = Self::filter(wanted) {
            body["where"] = found;
        }
        let answer = self.http.json(Method::POST, &format!("{}/query", self.address().await?), Some(&body), &[]).await?.body;
        let (ids, distances, texts, metas) = (first(&answer, "ids"), first(&answer, "distances"), first(&answer, "documents"), first(&answer, "metadatas"));
        Ok(ids
            .iter()
            .enumerate()
            .map(|(n, id)| VectorHit {
                id: id.as_str().unwrap_or_default().to_string(),
                score: 1.0 - number(distances.get(n)),
                text: texts.get(n).and_then(Value::as_str).unwrap_or_default().to_string(),
                metadata: Self::unflat(metas.get(n)),
            })
            .collect())
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        let mut body = json!({});
        if let Some(ids) = ids {
            body["ids"] = json!(ids);
        }
        if let Some(found) = Self::filter(wanted) {
            body["where"] = found;
        }
        self.http.json(Method::POST, &format!("{}/delete", self.address().await?), Some(&body), &[]).await?;
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let answer = self.http.json(Method::GET, &format!("{}/count", self.address().await?), None, &[]).await?.body;
        answer.as_u64().map(|n| n as usize).ok_or_else(|| store_error("chroma", format!("chroma answered a count with {answer}")))
    }

    async fn clear(&self) -> Result<()> {
        self.http.json(Method::DELETE, &format!("{}/{}", self.collections(), self.collection), None, &[404]).await?;
        self.id.lock().unwrap_or_else(|p| p.into_inner()).clear();
        self.address().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::qdrant::fake;
    use super::*;
    use crate::knowledge::base::conditions;

    #[test]
    fn filters_and_metadata() {
        let one = conditions(Some(&json!({"a": 1}))).unwrap();
        assert_eq!(ChromaStore::filter(&one), Some(json!({"a": {"$eq": 1}})));
        let two = conditions(Some(&json!({"$and": [{"a": 1}, {"b": {"$in": ["x"]}}]}))).unwrap();
        assert_eq!(ChromaStore::filter(&two), Some(json!({"$and": [{"a": {"$eq": 1}}, {"b": {"$in": ["x"]}}]})));
        assert_eq!(ChromaStore::filter(&[]), None);
        let meta = object(Some(&json!({"tags": ["a", "b"], "gone": null, "n": 1})));
        let flat = ChromaStore::flat(&meta);
        assert_eq!(flat, json!({"tags__json": "[\"a\", \"b\"]", "n": 1}));
        assert_eq!(ChromaStore::unflat(Some(&flat)), object(Some(&json!({"tags": ["a", "b"], "n": 1}))));
        assert_eq!(ChromaStore::flat(&Map::new()), Value::Null);
        assert_eq!(ChromaStore::unflat(Some(&json!({"x__json": "not json"}))), object(Some(&json!({"x__json": "not json"}))));
    }

    #[test]
    fn urls() {
        let c = ChromaStore::from_url(UrlParts::parse("chromadb+https://c.example/kb?tenant=t&database=d&api_key=key1").unwrap(), false).unwrap();
        assert_eq!(c.collections(), "https://c.example/api/v2/tenants/t/databases/d/collections");
        assert_eq!((c.collection.as_str(), c.api_key.as_str()), ("kb", "key1"));
        let c = ChromaStore::from_url(UrlParts::default(), true).unwrap();
        assert_eq!((c.url.as_str(), c.collection.as_str()), ("http://localhost:8000", "knowledge"));
    }

    #[tokio::test]
    async fn requests_are_what_chroma_expects() {
        let fake = fake::serve(|s| match (s.method.as_str(), s.path.as_str()) {
            ("POST", p) if p.ends_with("/collections") => (200, json!({"id": "col-1"})),
            (_, p) if p.ends_with("/query") => {
                (200, json!({"ids": [["doc#0"]], "distances": [[0.2]], "documents": [["T"]], "metadatas": [[{"year": 2026, "tags__json": "[\"a\"]"}]]}))
            }
            (_, p) if p.ends_with("/count") => (200, json!(4)),
            ("DELETE", _) => (404, json!({})),
            _ => (200, json!({})),
        })
        .await;
        let store = ChromaStore::new(&fake.url).collection("kb").api_key("chroma-key").retries(0);
        store.upsert(vec![VectorRecord::new("doc#0", vec![1.0, 0.0], "T", json!({"year": 2026, "tags": ["a"]}))]).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen[0].path, "/api/v2/tenants/default_tenant/databases/default_database/collections");
        assert_eq!(seen[0].body, json!({"name": "kb", "get_or_create": true, "configuration": {"hnsw": {"space": "cosine"}}}));
        assert_eq!(seen[0].header("x-chroma-token"), "chroma-key");
        assert!(seen[1].path.ends_with("/collections/col-1/upsert"));
        assert_eq!(seen[1].body["metadatas"], json!([{"year": 2026, "tags__json": "[\"a\"]"}]));
        assert_eq!(seen[1].body["documents"], json!(["T"]));

        let hits = store.query(&[1.0, 0.0], 10, Some(&json!({"source": "faq.md", "year": {"$gte": 2025}}))).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen.len(), 3, "the collection's id is asked for once");
        let parts = seen[2].body["where"]["$and"].as_array().unwrap();
        assert!(parts.contains(&json!({"source": {"$eq": "faq.md"}})) && parts.contains(&json!({"year": {"$gte": 2025}})));
        assert_eq!(seen[2].body["n_results"], json!(10));
        assert!((hits[0].score - 0.8).abs() < 1e-9);
        assert_eq!(hits[0].metadata, object(Some(&json!({"year": 2026, "tags": ["a"]}))));

        store.delete(Some(vec!["a".into()]), None).await.unwrap();
        assert_eq!(fake.seen()[3].body, json!({"ids": ["a"]}));
        assert_eq!(store.count().await.unwrap(), 4);

        store.clear().await.unwrap();
        let seen = fake.seen();
        assert_eq!((seen[5].method.as_str(), seen[5].path.as_str()), ("DELETE", "/api/v2/tenants/default_tenant/databases/default_database/collections/kb"));
        assert!(seen[6].path.ends_with("/collections"), "made again");
    }

    #[tokio::test]
    async fn live() {
        // e.g. http://localhost:8000; the key, if any, from $CHROMA_API_KEY.
        let Ok(url) = std::env::var("AH_LIVE_CHROMA_URL") else { return };
        let report = ChromaStore::new(&url).collection("ah_check").check(8).await;
        assert_eq!(report["ok"], json!(true), "{report}");
    }
}
