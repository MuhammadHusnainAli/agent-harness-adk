//! Elasticsearch and OpenSearch memory — full-text recall without an embedder.
//!
//! Spoken to over the REST API — no client library. The indices and documents
//! are the Python `ElasticsearchMemory`'s, so one cluster serves both.
//!
//! ```no_run
//! # use agent_harness::memory::providers::elasticsearch::ElasticsearchMemory;
//! let store = ElasticsearchMemory::from_url("es://localhost:9200");
//! let cloud = ElasticsearchMemory::from_url("es://https://my-cluster.es.io:443?api_key=...");
//! ```
//!
//! **Options** on the URL: `index` (default `agent-memory`), `docs_index`
//! (default `{index}-docs`), `refresh` (`true` in tests; costly in
//! production), `api_key`. A user and password in the URL are sent as basic
//! auth; otherwise `ELASTICSEARCH_API_KEY`, else nothing.

use async_trait::async_trait;
use base64::Engine as _;
use reqwest::Method;
use serde_json::{Value, json};
use tokio::sync::OnceCell;

use crate::errors::{Error, Result};
use crate::knowledge::base::HttpClient;
use crate::memory::base::{MemoryRecord, MemoryStore, RecordQuery, doc_key};
use crate::memory::providers::objects::{encode, env, option, split_url, truthy};
use crate::memory::trace::Trace;
use crate::types::now;

/// Records in one index, documents in another.
///
/// `search()` uses the engine's own text scoring rather than the keyword
/// fallback, so this is the one backend where recall is good without an
/// embedder.
#[derive(Debug)]
pub struct ElasticsearchMemory {
    /// `http://localhost:9200`.
    pub host: String,
    pub index: String,
    pub docs_index: String,
    /// Make every write visible to search at once. True in tests; costly in production.
    pub refresh: bool,
    http: HttpClient,
    ready: OnceCell<()>,
}

impl ElasticsearchMemory {
    pub fn new(host: &str) -> Self {
        let mut me = Self {
            host: host.trim_end_matches('/').to_string(),
            index: "agent-memory".into(),
            docs_index: "agent-memory-docs".into(),
            refresh: false,
            http: HttpClient::new("elasticsearch", 30.0, 3),
            ready: OnceCell::new(),
        };
        if let Some(key) = env("ELASTICSEARCH_API_KEY") {
            me = me.api_key(&key);
        }
        me
    }

    /// Records in `index`, documents in `{index}-docs`.
    pub fn index(mut self, index: &str) -> Self {
        self.index = index.into();
        self.docs_index = format!("{index}-docs");
        self
    }

    pub fn api_key(mut self, key: &str) -> Self {
        self.http.headers.retain(|(k, _)| k != "authorization");
        self.http = self.http.header("authorization", format!("ApiKey {key}")).secret(key);
        self
    }

    pub fn basic_auth(mut self, user: &str, password: &str) -> Self {
        let token = base64::engine::general_purpose::STANDARD.encode(format!("{user}:{password}"));
        self.http.headers.retain(|(k, _)| k != "authorization");
        self.http = self.http.header("authorization", format!("Basic {token}")).secret(token).secret(password);
        self
    }

    pub fn retries(mut self, retries: u32) -> Self {
        self.http.retry.max_retries = retries;
        self
    }

    /// `es://host:9200`, `elasticsearch://…`, `opensearch://…`. Plain HTTP
    /// unless the URL nests one (`es://https://host`), as Python reads it.
    pub fn from_url(url: &str) -> Result<Self> {
        let inner = url.split_once("://").map_or(url, |(_, r)| r);
        let target = if inner.contains("://") { inner.to_string() } else { format!("http://{inner}") };
        let parts = split_url(&target);
        if parts.host.is_empty() {
            return Err(Error::config(format!("ElasticsearchMemory needs a host: es://host:9200 (got '{url}')")));
        }
        let path = parts.path.trim_end_matches('/');
        let host = if path.is_empty() { format!("{}://{}", parts.scheme, parts.host) } else { format!("{}://{}/{path}", parts.scheme, parts.host) };
        let o = &parts.query;
        let mut me = Self::new(&host);
        if let Some(i) = option(o, "index") {
            me = me.index(i);
        }
        if let Some(d) = option(o, "docs_index") {
            me.docs_index = d.into();
        }
        me.refresh = option(o, "refresh").is_some_and(truthy);
        if let Some(k) = option(o, "api_key") {
            me = me.api_key(k);
        } else if let (Some(u), Some(p)) = (&parts.user, &parts.password) {
            me = me.basic_auth(u, p);
        }
        Ok(me)
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{path}", self.host)
    }

    fn refresh_param(&self) -> &'static str {
        if self.refresh { "true" } else { "false" }
    }

    async fn ensure(&self) -> Result<()> {
        self.ready
            .get_or_try_init(|| async {
                let mapping = json!({"mappings": {"properties": {
                    "scope": {"type": "keyword"}, "kind": {"type": "keyword"},
                    "text": {"type": "text"}, "source": {"type": "keyword"},
                    "user_id": {"type": "keyword"}, "session_id": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"}, "ts": {"type": "double"},
                }}});
                for (name, body) in [(&self.index, Some(mapping)), (&self.docs_index, None)] {
                    let index_url = self.url(&encode(name, false));
                    if self.http.json(Method::HEAD, &index_url, None, &[404]).await?.status == 404 {
                        // Another process may get there first: that is fine.
                        let made = self.http.json(Method::PUT, &index_url, body.as_ref(), &[400]).await?;
                        if made.status == 400 && !made.body.to_string().contains("resource_already_exists_exception") {
                            return Err(Error::VectorStore {
                                message: format!("elasticsearch: could not create {name}: {}", made.body),
                                store: "elasticsearch".into(),
                                status: Some(400),
                                retryable: false,
                            });
                        }
                    }
                }
                Ok(())
            })
            .await
            .map(|_| ())
    }

    /// The bool query for these narrowings: a record with no value on a trace
    /// axis is shared, so it passes.
    pub fn query(scope: Option<&str>, kind: Option<&str>, trace: Option<&Trace>, text: &str) -> Value {
        let must: Vec<Value> = if text.is_empty() { vec![json!({"match_all": {}})] } else { vec![json!({"match": {"text": text}})] };
        let mut filters: Vec<Value> = Vec::new();
        if let Some(s) = scope {
            filters.push(json!({"term": {"scope": s}}));
        }
        if let Some(k) = kind {
            filters.push(json!({"term": {"kind": k}}));
        }
        if let Some(t) = trace {
            for (field, value) in t.filters() {
                // Either it is this trace's, or it predates traces entirely.
                filters.push(json!({"bool": {"should": [
                    {"term": {field: value}},
                    {"bool": {"must_not": {"exists": {"field": field}}}},
                ], "minimum_should_match": 1}}));
            }
        }
        json!({"bool": {"must": must, "filter": filters}})
    }

    fn hits(found: &Value) -> Vec<MemoryRecord> {
        found["hits"]["hits"].as_array().into_iter().flatten().filter_map(|h| serde_json::from_value(h["_source"].clone()).ok()).collect()
    }
}

#[async_trait]
impl MemoryStore for ElasticsearchMemory {
    async fn append(&self, record: MemoryRecord) -> Result<MemoryRecord> {
        self.ensure().await?;
        let url = self.url(&format!("{}/_doc/{}?refresh={}", encode(&self.index, false), encode(&record.id, false), self.refresh_param()));
        self.http.json(Method::PUT, &url, Some(&serde_json::to_value(&record).unwrap_or_default()), &[]).await?;
        Ok(record)
    }

    async fn extend(&self, records: Vec<MemoryRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        self.ensure().await?;
        let mut lines = String::new();
        for r in &records {
            lines.push_str(&json!({"index": {"_index": self.index, "_id": r.id}}).to_string());
            lines.push('\n');
            lines.push_str(&serde_json::to_string(r).unwrap_or_default());
            lines.push('\n');
        }
        let reply = self
            .http
            .send(Method::POST, &self.url(&format!("_bulk?refresh={}", self.refresh_param())), Some(lines.into_bytes()), Some("application/x-ndjson"), &[], &[])
            .await?;
        if reply.body["errors"] == true {
            return Err(Error::VectorStore {
                message: "elasticsearch: some records were not indexed".into(),
                store: "elasticsearch".into(),
                status: None,
                retryable: false,
            });
        }
        Ok(())
    }

    async fn all(&self, query: RecordQuery) -> Result<Vec<MemoryRecord>> {
        self.ensure().await?;
        let body = json!({
            "query": Self::query(query.scope.as_deref(), query.kind.as_deref(), query.trace.as_ref(), ""),
            "sort": [{"ts": "desc"}],
            "size": query.limit.filter(|n| *n > 0).unwrap_or(1000),
        });
        let found = self.http.json(Method::POST, &self.url(&format!("{}/_search", encode(&self.index, false))), Some(&body), &[]).await?;
        let mut rows = Self::hits(&found.body);
        rows.reverse();
        Ok(rows)
    }

    /// Ranked by the engine, which is why you would choose this backend.
    async fn search(&self, query: &str, scope: Option<&str>, limit: usize, trace: Option<&Trace>) -> Result<Vec<MemoryRecord>> {
        if query.trim().is_empty() {
            return self.all(RecordQuery { scope: scope.map(str::to_string), limit: Some(limit), trace: trace.cloned(), ..Default::default() }).await;
        }
        self.ensure().await?;
        let body = json!({"query": Self::query(scope, None, trace, query), "size": limit});
        let found = self.http.json(Method::POST, &self.url(&format!("{}/_search", encode(&self.index, false))), Some(&body), &[]).await?;
        Ok(Self::hits(&found.body))
    }

    async fn clear(&self, scope: Option<&str>, trace: Option<&Trace>) -> Result<()> {
        self.ensure().await?;
        let records = json!({"query": Self::query(scope, None, trace, "")});
        let url = |index: &str| self.url(&format!("{}/_delete_by_query?refresh={}", encode(index, false), self.refresh_param()));
        self.http.json(Method::POST, &url(&self.index), Some(&records), &[]).await?;
        if trace.is_some() || scope.is_none() {
            // `namespace` is mapped dynamically (text + keyword), so match the
            // keyword too: Python's bare term on the text field misses "u-alice".
            let docs = match trace {
                Some(t) => {
                    json!({"query": {"bool": {"should": [{"term": {"namespace": t.slug()}}, {"term": {"namespace.keyword": t.slug()}}], "minimum_should_match": 1}}})
                }
                None => json!({"query": {"match_all": {}}}),
            };
            self.http.json(Method::POST, &url(&self.docs_index), Some(&docs), &[]).await?;
        }
        Ok(())
    }

    async fn read_doc(&self, name: &str, trace: Option<&Trace>) -> Result<String> {
        self.ensure().await?;
        let url = self.url(&format!("{}/_doc/{}", encode(&self.docs_index, false), encode(&doc_key(name, trace), false)));
        let found = self.http.json(Method::GET, &url, None, &[404]).await?;
        if found.status == 404 {
            return Ok(String::new());
        }
        Ok(found.body["_source"]["text"].as_str().unwrap_or("").to_string())
    }

    async fn write_doc(&self, name: &str, text: &str, trace: Option<&Trace>) -> Result<()> {
        self.ensure().await?;
        let url = self.url(&format!("{}/_doc/{}?refresh={}", encode(&self.docs_index, false), encode(&doc_key(name, trace), false), self.refresh_param()));
        let namespace = trace.map_or_else(|| "_shared".to_string(), Trace::slug);
        self.http.json(Method::PUT, &url, Some(&json!({"name": name, "text": text, "updated": now(), "namespace": namespace})), &[]).await?;
        Ok(())
    }

    fn describe(&self) -> String {
        format!("elasticsearch {} ({})", self.host, self.index)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use axum::body::Bytes;
    use axum::http::{Method as AxMethod, StatusCode, Uri};
    use axum::response::{IntoResponse, Response};

    use super::*;
    use crate::memory::providers::objects::decode;
    use crate::memory::providers::objects::testing::serve;

    type Indices = Arc<Mutex<BTreeMap<String, BTreeMap<String, Value>>>>;

    /// Does a stored document pass this (small) query language?
    fn passes(doc: &Value, q: &Value) -> bool {
        if let Some(b) = q.get("bool") {
            let all = |k: &str| b[k].as_array().into_iter().flatten().all(|c| passes(doc, c));
            let none = match &b["must_not"] {
                Value::Null => true,
                Value::Array(a) => a.iter().all(|c| !passes(doc, c)),
                one => !passes(doc, one),
            };
            let should = b["should"].as_array().is_none_or(|a| a.iter().any(|c| passes(doc, c)));
            return all("must") && all("filter") && none && should;
        }
        if q.get("match_all").is_some() {
            return true;
        }
        if let Some(m) = q.get("match").and_then(Value::as_object) {
            return m.iter().all(|(f, v)| {
                let hay = doc[f].as_str().unwrap_or("").to_lowercase();
                v.as_str().unwrap_or("").to_lowercase().split_whitespace().any(|w| hay.contains(w))
            });
        }
        if let Some(t) = q.get("term").and_then(Value::as_object) {
            return t.iter().all(|(f, v)| doc[f.trim_end_matches(".keyword")] == *v);
        }
        if let Some(e) = q.get("exists") {
            return !doc[e["field"].as_str().unwrap_or("")].is_null();
        }
        false
    }

    async fn fake_es(state: Indices, method: AxMethod, uri: Uri, body: Bytes) -> Response {
        let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        let parts: Vec<String> = uri.path().trim_start_matches('/').split('/').map(decode).collect();
        let mut s = state.lock().unwrap();
        match (method, parts.iter().map(String::as_str).collect::<Vec<_>>().as_slice()) {
            (AxMethod::HEAD, [index]) => if s.contains_key(*index) { StatusCode::OK } else { StatusCode::NOT_FOUND }.into_response(),
            (AxMethod::PUT, [index]) => {
                s.entry(index.to_string()).or_default();
                axum::Json(json!({"acknowledged": true})).into_response()
            }
            (AxMethod::PUT, [index, "_doc", id]) => {
                s.get_mut(*index).unwrap().insert(id.to_string(), body);
                axum::Json(json!({"result": "created"})).into_response()
            }
            (AxMethod::GET, [index, "_doc", id]) => match s.get(*index).and_then(|i| i.get(*id)) {
                Some(d) => axum::Json(json!({"_source": d})).into_response(),
                None => StatusCode::NOT_FOUND.into_response(),
            },
            (AxMethod::POST, [index, "_search"]) => {
                let mut hits: Vec<&Value> = s.get(*index).into_iter().flat_map(|i| i.values()).filter(|d| passes(d, &body["query"])).collect();
                hits.sort_by(|a, b| b["ts"].as_f64().unwrap_or(0.0).total_cmp(&a["ts"].as_f64().unwrap_or(0.0)));
                let size = body["size"].as_u64().unwrap_or(10) as usize;
                let hits: Vec<Value> = hits.into_iter().take(size).map(|d| json!({"_source": d})).collect();
                axum::Json(json!({"hits": {"hits": hits}})).into_response()
            }
            (AxMethod::POST, [index, "_delete_by_query"]) => {
                if let Some(i) = s.get_mut(*index) {
                    i.retain(|_, d| !passes(d, &body["query"]));
                }
                axum::Json(json!({"deleted": 1})).into_response()
            }
            _ => StatusCode::BAD_REQUEST.into_response(),
        }
    }

    #[test]
    fn the_filter_treats_an_untagged_record_as_shared_and_scores_the_text() {
        let q = ElasticsearchMemory::query(Some("user"), None, Some(&Trace::user("alice")), "refund window");
        assert_eq!(q["bool"]["must"], json!([{"match": {"text": "refund window"}}]));
        assert!(q["bool"]["filter"].as_array().unwrap().contains(&json!({"term": {"scope": "user"}})));
        let should = &q["bool"]["filter"].as_array().unwrap().last().unwrap()["bool"]["should"];
        assert!(should.as_array().unwrap().contains(&json!({"term": {"user_id": "alice"}})));
        let store = ElasticsearchMemory::from_url("es://elastic:pw@localhost:9200?index=mem&refresh=true").unwrap();
        assert_eq!((store.host.as_str(), store.docs_index.as_str(), store.refresh), ("http://localhost:9200", "mem-docs", true));
        assert_eq!(ElasticsearchMemory::from_url("opensearch://https://node:443").unwrap().host, "https://node:443");
    }

    #[tokio::test]
    async fn records_and_documents_round_trip_through_the_rest_api() {
        let state: Indices = Arc::default();
        let shared = state.clone();
        let router = axum::Router::new().fallback(move |m: AxMethod, u: Uri, b: Bytes| fake_es(shared.clone(), m, u, b));
        let base = serve(router).await;
        let store = ElasticsearchMemory::from_url(&format!("es://{}?refresh=true", base.trim_start_matches("http://"))).unwrap().retries(0);
        let (alice, bob) = (Trace::user("alice"), Trace::user("bob"));
        for (t, text, ts) in [(Some(&alice), "the refund window is 30 days", 1.0), (Some(&bob), "bob likes tea", 2.0), (None, "shared refund note", 3.0)] {
            let mut r = MemoryRecord::new("user", "fact", text);
            r.ts = ts;
            if let Some(t) = t {
                t.stamp(&mut r);
            }
            store.append(r).await.unwrap();
        }
        let mine: Vec<String> = store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap().into_iter().map(|r| r.text).collect();
        assert_eq!(mine, ["the refund window is 30 days", "shared refund note"]);
        assert_eq!(store.search("refund", None, 5, Some(&bob)).await.unwrap().len(), 1);
        assert_eq!(store.search("", None, 1, Some(&alice)).await.unwrap()[0].text, "shared refund note");

        store.write_doc("user.md", "- alice", Some(&alice)).await.unwrap();
        assert!(state.lock().unwrap()["agent-memory-docs"].contains_key("u-alice/user.md"));
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "- alice");
        assert_eq!(store.read_doc("user.md", Some(&bob)).await.unwrap(), "");
        store.clear(None, Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "");
        let left: Vec<String> = store.all(RecordQuery::default()).await.unwrap().into_iter().map(|r| r.text).collect();
        assert_eq!(left, ["bob likes tea"], "her own and the untagged ones match her trace");
        store.extend(vec![MemoryRecord::new("project", "note", "bulk")]).await.unwrap_err(); // the fake has no _bulk
    }

    /// `AH_LIVE_ES_URL` (e.g. `es://localhost:9200`) to run against a real cluster.
    #[tokio::test]
    async fn a_real_cluster_holds_the_contract() {
        let Some(url) = env("AH_LIVE_ES_URL") else { return };
        let index = format!("ah-rust-test-{}", crate::types::new_id(""));
        let sep = if url.contains('?') { '&' } else { '?' };
        let store = ElasticsearchMemory::from_url(&format!("{url}{sep}index={index}&refresh=true")).unwrap();
        let alice = Trace::user("alice");
        let mut r = MemoryRecord::new("user", "fact", "the refund window is 30 days");
        alice.stamp(&mut r);
        store.extend(vec![r]).await.unwrap();
        assert_eq!(store.search("refund", None, 5, Some(&alice)).await.unwrap().len(), 1);
        store.write_doc("user.md", "- alice", Some(&alice)).await.unwrap();
        store.clear(None, Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "");
    }
}
