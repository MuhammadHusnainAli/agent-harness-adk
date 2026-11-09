//! Pinecone.

use std::sync::Mutex;

use async_trait::async_trait;
use reqwest::Method;
use serde_json::{Map, Value, json};

use super::UrlParts;
use super::qdrant::{Settings, floats, no_more, number, object, or_env, text_of};
use crate::errors::{Error, Result};
use crate::knowledge::base::{Condition, HttpClient, VectorHit, VectorRecord, VectorStore, store_error};
use crate::llm_providers::resilience::RetryPolicy;

/// The API version every request asks for.
pub const API_VERSION: &str = "2025-01";

/// ```ignore
/// PineconeStore::new("handbook")?;                                    // $PINECONE_API_KEY
/// PineconeStore::new("handbook")?.namespace("acme").cloud("aws").region("us-east-1");
/// PineconeStore::with_host("https://handbook-abc123.svc.aped-4627-b74a.pinecone.io")?;
/// ```
///
/// Named by `index`, its address is looked up — and the index created, as a
/// serverless one, if it is not there. Given a `host`, that lookup is skipped.
///
/// Pinecone shows a write a moment after it was made, not at once.
#[derive(Debug)]
pub struct PineconeStore {
    pub index: String,
    pub api_key: String,
    pub namespace: String,
    pub cloud: String,
    pub region: String,
    pub control_url: String,
    settings: Settings,
    http: HttpClient,
    /// The index's data-plane address, once known.
    host: Mutex<String>,
    /// The vectors' length, once known.
    dimension: Mutex<usize>,
}

impl PineconeStore {
    /// An index by name. The key comes from `$PINECONE_API_KEY` unless one is given.
    pub fn new(index: &str) -> Result<Self> {
        Self::build(index, "")
    }

    /// An index by its address: no lookup, no creating.
    pub fn with_host(host: &str) -> Result<Self> {
        Self::build("", host)
    }

    fn build(index: &str, host: &str) -> Result<Self> {
        if index.is_empty() && host.is_empty() {
            return Err(store_error("pinecone", "Pinecone needs the index's name, or its host"));
        }
        let settings = Settings::default();
        let mut store = Self {
            index: index.into(),
            api_key: or_env("", "PINECONE_API_KEY"),
            namespace: String::new(),
            cloud: "aws".into(),
            region: "us-east-1".into(),
            control_url: "https://api.pinecone.io".into(),
            http: settings.client("pinecone", &[], &[]),
            settings,
            host: Mutex::new(host.trim_end_matches('/').to_string()),
            dimension: Mutex::new(0),
        };
        store.rebuild();
        Ok(store)
    }

    /// The index's address, which skips the lookup.
    pub fn host(self, host: &str) -> Self {
        *self.host.lock().unwrap_or_else(|p| p.into_inner()) = host.trim_end_matches('/').to_string();
        self
    }

    pub fn api_key(mut self, key: &str) -> Self {
        self.api_key = or_env(key, "PINECONE_API_KEY");
        self.rebuild();
        self
    }

    pub fn namespace(mut self, namespace: &str) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// Where a new serverless index is made (`aws` by default).
    pub fn cloud(mut self, cloud: &str) -> Self {
        self.cloud = cloud.into();
        self
    }

    /// Where a new serverless index is made (`us-east-1` by default).
    pub fn region(mut self, region: &str) -> Self {
        self.region = region.into();
        self
    }

    pub fn control_url(mut self, url: &str) -> Self {
        self.control_url = url.trim_end_matches('/').to_string();
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

    /// `pinecone://index?api_key=…&namespace=…&host=…&cloud=…&region=…`. A bare
    /// `pinecone` names no index, which Pinecone cannot do without.
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        if bare {
            return Self::new("");
        }
        let settings = Settings::from_url(&mut p, "pinecone")?;
        let host = p.take("host").unwrap_or_default();
        let key = p.take("api_key").unwrap_or_default();
        let namespace = p.take("namespace").unwrap_or_default();
        let cloud = p.take("cloud");
        let region = p.take("region");
        let control = p.take("control_url");
        let index = p.take("index");
        no_more(&p, "pinecone")?;
        let index = if p.netloc.is_empty() { index.unwrap_or_default() } else { p.netloc.clone() };
        let mut store = Self::build(&index, &host)?.api_key(&key).namespace(&namespace);
        store.settings = settings;
        store.rebuild();
        if let Some(c) = cloud {
            store.cloud = c;
        }
        if let Some(r) = region {
            store.region = r;
        }
        if let Some(c) = control {
            store = store.control_url(&c);
        }
        Ok(store)
    }

    fn rebuild(&mut self) {
        let own = [("api-key", self.api_key.clone()), ("x-pinecone-api-version", API_VERSION.to_string())];
        self.http = self.settings.client("pinecone", &own, &[&self.api_key]);
    }

    /// The data-plane address, if it is known yet.
    pub fn known_host(&self) -> String {
        self.host.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    fn remember(&self, found: &Value) {
        let host = text_of(found.get("host").unwrap_or(&Value::Null));
        *self.host.lock().unwrap_or_else(|p| p.into_inner()) = format!("https://{}", host.strip_prefix("https://").unwrap_or(&host));
    }

    async fn data(&self) -> Result<String> {
        if self.known_host().is_empty() {
            let found = self.http.json(Method::GET, &format!("{}/indexes/{}", self.control_url, self.index), None, &[]).await?.body;
            self.remember(&found);
        }
        Ok(self.known_host())
    }

    /// The filter in Pinecone's terms (Mongo-like): one condition, or `$and` of several.
    pub fn filter(wanted: &[Condition]) -> Option<Value> {
        let mut parts: Vec<Value> = wanted.iter().map(|c| json!({c.field.clone(): {c.op.clone(): c.value}})).collect();
        match parts.len() {
            0 => None,
            1 => parts.pop(),
            _ => Some(json!({"$and": parts})),
        }
    }

    fn scoped(&self, mut body: Value) -> Value {
        if !self.namespace.is_empty() {
            body["namespace"] = json!(self.namespace);
        }
        body
    }

    /// The ids a filter matches, asked for with a vector that favours none.
    async fn matching(&self, found: &Value, limit: usize) -> Result<Vec<String>> {
        let mut dimension = *self.dimension.lock().unwrap_or_else(|p| p.into_inner());
        if dimension == 0 {
            let stats = self.http.json(Method::POST, &format!("{}/describe_index_stats", self.data().await?), Some(&json!({})), &[]).await?.body;
            dimension = number(stats.get("dimension")) as usize;
            *self.dimension.lock().unwrap_or_else(|p| p.into_inner()) = dimension;
        }
        if dimension == 0 {
            return Err(store_error("pinecone", "the index's dimension is not known yet, so it cannot be searched by filter alone"));
        }
        let body = self.scoped(json!({"vector": vec![1.0f32; dimension], "topK": limit, "filter": found, "includeMetadata": false}));
        let answer = self.http.json(Method::POST, &format!("{}/query", self.data().await?), Some(&body), &[]).await?.body;
        Ok(answer.get("matches").and_then(Value::as_array).into_iter().flatten().map(|m| text_of(m.get("id").unwrap_or(&Value::Null))).collect())
    }
}

#[async_trait]
impl VectorStore for PineconeStore {
    fn name(&self) -> &str {
        "pinecone"
    }

    fn batch_size(&self) -> usize {
        100
    }

    fn max_k(&self) -> usize {
        10_000
    }

    async fn ensure(&self, dimension: usize) -> Result<()> {
        if !self.known_host().is_empty() {
            return Ok(());
        }
        let at = format!("{}/indexes/{}", self.control_url, self.index);
        let mut found = self.http.json(Method::GET, &at, None, &[404]).await?;
        if found.status == 404 {
            let body =
                json!({"name": self.index, "dimension": dimension, "metric": "cosine", "spec": {"serverless": {"cloud": self.cloud, "region": self.region}}});
            found = self.http.json(Method::POST, &format!("{}/indexes", self.control_url), Some(&body), &[409]).await?;
            if found.status == 409 {
                found = self.http.json(Method::GET, &at, None, &[]).await?;
            }
        } else if let Some(held) = found.body.get("dimension").filter(|d| !d.is_null())
            && held.as_f64() != Some(dimension as f64)
        {
            return Err(store_error(
                "pinecone",
                format!("the Pinecone index '{}' holds {}-dimension vectors; {dimension} was asked for", self.index, text_of(held)),
            ));
        }
        self.remember(&found.body);
        Ok(())
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        if let Some(first) = records.first() {
            *self.dimension.lock().unwrap_or_else(|p| p.into_inner()) = first.vector.len();
        }
        let vectors: Vec<Value> = records
            .iter()
            .map(|r| {
                let mut metadata: Map<String, Value> = r.metadata.iter().filter(|(_, v)| !v.is_null()).map(|(k, v)| (k.clone(), v.clone())).collect();
                metadata.insert("_text".into(), json!(r.text));
                json!({"id": r.id, "values": floats(&r.vector), "metadata": metadata})
            })
            .collect();
        let body = self.scoped(json!({"vectors": vectors}));
        self.http.json(Method::POST, &format!("{}/vectors/upsert", self.data().await?), Some(&body), &[]).await?;
        Ok(())
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let mut body = json!({"vector": floats(vector), "topK": k, "includeMetadata": true});
        if let Some(found) = Self::filter(wanted) {
            body["filter"] = found;
        }
        let body = self.scoped(body);
        let answer = self.http.json(Method::POST, &format!("{}/query", self.data().await?), Some(&body), &[]).await?.body;
        Ok(answer
            .get("matches")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|m| {
                let mut metadata = object(m.get("metadata"));
                let text = metadata.remove("_text").and_then(|t| t.as_str().map(String::from)).unwrap_or_default();
                VectorHit { id: text_of(m.get("id").unwrap_or(&Value::Null)), score: number(m.get("score")), text, metadata }
            })
            .collect())
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        let found = Self::filter(wanted);
        let url = format!("{}/vectors/delete", self.data().await?);
        let mut ids: Vec<String> = match (ids, &found) {
            (None, Some(found)) => {
                match self.http.json(Method::POST, &url, Some(&self.scoped(json!({"filter": found}))), &[]).await {
                    Ok(_) => return Ok(()),
                    // Serverless indexes do not delete by filter: find what matches,
                    // and remove it by id, until nothing matches.
                    Err(Error::VectorStore { status: Some(400), .. }) => {}
                    Err(e) => return Err(e),
                }
                for _ in 0..10_000 {
                    let matched = self.matching(found, 1000).await?;
                    if matched.is_empty() {
                        return Ok(());
                    }
                    let n = matched.len();
                    self.http.json(Method::POST, &url, Some(&self.scoped(json!({"ids": matched}))), &[]).await?;
                    if n < 1000 {
                        return Ok(());
                    }
                }
                return Ok(());
            }
            (ids, _) => ids.map(<[String]>::to_vec).unwrap_or_default(),
        };
        if let Some(found) = &found {
            // Ids and a filter: only the ids the filter also matches.
            let keep = self.matching(found, 10_000).await?;
            ids.retain(|i| keep.contains(i));
        }
        for chunk in ids.chunks(1000) {
            self.http.json(Method::POST, &url, Some(&self.scoped(json!({"ids": chunk}))), &[404]).await?;
        }
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let stats = self.http.json(Method::POST, &format!("{}/describe_index_stats", self.data().await?), Some(&json!({})), &[]).await?.body;
        if !self.namespace.is_empty() {
            let space = stats.get("namespaces").and_then(|n| n.get(&self.namespace));
            return Ok(number(space.and_then(|s| s.get("vectorCount"))) as usize);
        }
        Ok(number(stats.get("totalVectorCount")) as usize)
    }

    async fn clear(&self) -> Result<()> {
        let body = self.scoped(json!({"deleteAll": true}));
        self.http.json(Method::POST, &format!("{}/vectors/delete", self.data().await?), Some(&body), &[404]).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::super::qdrant::fake;
    use super::*;
    use crate::knowledge::base::conditions;

    #[test]
    fn filters_and_urls() {
        let c = conditions(Some(&json!({"source": "faq.md", "lang": {"$in": ["en"]}}))).unwrap();
        let f = PineconeStore::filter(&c).unwrap();
        let parts = f["$and"].as_array().unwrap();
        assert!(parts.contains(&json!({"source": {"$eq": "faq.md"}})) && parts.contains(&json!({"lang": {"$in": ["en"]}})));
        let source: Vec<_> = c.iter().filter(|x| x.field == "source").cloned().collect();
        assert_eq!(PineconeStore::filter(&source), Some(json!({"source": {"$eq": "faq.md"}})));
        let p = PineconeStore::from_url(UrlParts::parse("pinecone://handbook?api_key=k&namespace=acme").unwrap(), false).unwrap();
        assert_eq!((p.index.as_str(), p.api_key.as_str(), p.namespace.as_str()), ("handbook", "k", "acme"));
        assert!(PineconeStore::from_url(UrlParts::default(), true).is_err(), "no index, no host");
        assert!(PineconeStore::new("").is_err());
    }

    #[tokio::test]
    async fn finds_its_index_and_creates_it_when_there_is_none() {
        let fake = fake::serve(|s| match (s.method.as_str(), s.path.as_str()) {
            ("GET", "/indexes/handbook") => (404, json!({})),
            ("POST", "/indexes") => (200, json!({"host": "handbook-abc.svc.pinecone.io", "dimension": 2})),
            _ => (500, json!({})),
        })
        .await;
        let store = PineconeStore::new("handbook").unwrap().api_key("pc-secret").namespace("acme").control_url(&fake.url).retries(0);
        store.ensure(2).await.unwrap();
        let seen = fake.seen();
        assert_eq!(
            seen[1].body,
            json!({"name": "handbook", "dimension": 2, "metric": "cosine", "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}}})
        );
        assert_eq!(seen[0].header("api-key"), "pc-secret");
        assert_eq!(seen[0].header("x-pinecone-api-version"), "2025-01");
        assert_eq!(store.known_host(), "https://handbook-abc.svc.pinecone.io");

        let fake = fake::serve(|_| (200, json!({"host": "h", "dimension": 3}))).await;
        let store = PineconeStore::new("handbook").unwrap().control_url(&fake.url).retries(0);
        let e = store.ensure(2).await.unwrap_err();
        assert!(e.to_string().contains("holds 3-dimension vectors; 2 was asked for"), "{e}");
    }

    #[tokio::test]
    async fn writes_and_reads_at_the_host() {
        let fake = fake::serve(|s| match s.path.as_str() {
            "/query" => (200, json!({"matches": [{"id": "doc#0", "score": 0.88, "metadata": {"year": 2026, "_text": "T"}}]})),
            "/describe_index_stats" => (200, json!({"namespaces": {"acme": {"vectorCount": 5}}, "totalVectorCount": 9})),
            _ => (200, json!({})),
        })
        .await;
        let store = PineconeStore::with_host(&fake.url).unwrap().api_key("k").namespace("acme").retries(0);
        store.ensure(2).await.unwrap();
        assert!(fake.seen().is_empty(), "a host needs no lookup");
        let record = VectorRecord::new("doc#0", vec![0.6, 0.8], "Refunds take five days.", json!({"source": "faq.md", "gone": null}));
        store.upsert(vec![record]).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen[0].path, "/vectors/upsert");
        assert_eq!(seen[0].body["namespace"], json!("acme"));
        assert_eq!(seen[0].body["vectors"][0]["metadata"], json!({"source": "faq.md", "_text": "Refunds take five days."}));

        let hits = store.query(&[1.0, 0.0], 3, Some(&json!({"year": {"$gte": 2025}}))).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen[1].body, json!({"vector": [1.0, 0.0], "topK": 3, "includeMetadata": true, "filter": {"year": {"$gte": 2025}}, "namespace": "acme"}));
        assert_eq!(hits, vec![VectorHit { id: "doc#0".into(), score: 0.88, text: "T".into(), metadata: object(Some(&json!({"year": 2026}))) }]);
        assert_eq!(store.count().await.unwrap(), 5);
        store.clear().await.unwrap();
        assert_eq!(fake.seen().last().unwrap().body, json!({"deleteAll": true, "namespace": "acme"}));
    }

    #[tokio::test]
    async fn deletes_by_filter_even_where_the_index_cannot() {
        let matches = Mutex::new(vec![json!({"matches": []}), json!({"matches": [{"id": "a"}, {"id": "b"}]})]);
        let fake = fake::serve(move |s| {
            if s.path == "/vectors/delete" {
                if s.body.get("filter").is_some() {
                    (400, json!({"message": "Serverless indexes do not support deleting with metadata filtering"}))
                } else {
                    (200, json!({}))
                }
            } else if s.path == "/describe_index_stats" {
                (200, json!({"dimension": 3}))
            } else {
                (200, matches.lock().unwrap().pop().unwrap())
            }
        })
        .await;
        let store = PineconeStore::with_host(&fake.url).unwrap().api_key("k").retries(0);
        store.delete(None, Some(&json!({"doc": "handbook"}))).await.unwrap();
        let seen = fake.seen();
        let deletes: Vec<Value> = seen.iter().filter(|s| s.path == "/vectors/delete").map(|s| s.body.clone()).collect();
        assert_eq!(deletes, vec![json!({"filter": {"doc": {"$eq": "handbook"}}}), json!({"ids": ["a", "b"]})]);
        assert_eq!(seen[2].body["vector"], json!([1.0, 1.0, 1.0]));
        assert_eq!(seen[2].body["topK"], json!(1000));
    }

    #[tokio::test]
    async fn live() {
        // An index name; the key from $PINECONE_API_KEY.
        let Ok(index) = std::env::var("AH_LIVE_PINECONE_INDEX") else { return };
        let report = PineconeStore::new(&index).unwrap().check(8).await;
        assert_eq!(report["ok"], json!(true), "{report}");
    }
}
