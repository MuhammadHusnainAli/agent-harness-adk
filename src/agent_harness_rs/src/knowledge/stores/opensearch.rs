//! OpenSearch and Elasticsearch — self-hosted, Elastic Cloud, Amazon OpenSearch
//! Service, and Amazon OpenSearch Serverless.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use reqwest::Method;
use serde_json::{Map, Value, json};

use super::UrlParts;
use crate::errors::{Error, Result};
use crate::knowledge::base::{Condition, HttpClient, Signer, VectorHit, VectorRecord, VectorStore, normalise, store_error};
use crate::llm_providers::sigv4::{self, AwsCredentials};

/// The options of an [`OpenSearchStore`] — the same as the Python constructor's.
#[derive(Clone)]
pub struct OpenSearchOptions {
    /// The index records live in. `knowledge` by default.
    pub index: String,
    /// Basic auth. Falls back to `OPENSEARCH_USERNAME` / `OPENSEARCH_PASSWORD`
    /// (`ELASTICSEARCH_USERNAME` / `ELASTICSEARCH_PASSWORD` for Elasticsearch).
    pub username: Option<String>,
    pub password: Option<String>,
    /// Sent as `ApiKey …`; wins over basic auth. Elasticsearch falls back to
    /// `ELASTICSEARCH_API_KEY`.
    pub api_key: Option<String>,
    /// Set for Amazon OpenSearch: every request is signed with SigV4 for this region.
    pub aws_region: Option<String>,
    pub aws_access_key: Option<String>,
    pub aws_secret_key: Option<String>,
    pub aws_session_token: Option<String>,
    /// OpenSearch Serverless. `None` decides from the URL (`.aoss.` in it).
    pub serverless: Option<bool>,
    /// `lucene` or `faiss`. `None`: faiss on Serverless, lucene elsewhere.
    pub engine: Option<String>,
    /// Make a write visible to the next search. Serverless does not take it.
    pub refresh: bool,
    pub timeout: f64,
    pub retries: u32,
    /// Sent with every request.
    pub headers: Vec<(String, String)>,
}

impl Default for OpenSearchOptions {
    fn default() -> Self {
        Self {
            index: "knowledge".into(),
            username: None,
            password: None,
            api_key: None,
            aws_region: None,
            aws_access_key: None,
            aws_secret_key: None,
            aws_session_token: None,
            serverless: None,
            engine: None,
            refresh: true,
            timeout: 30.0,
            retries: 3,
            headers: vec![],
        }
    }
}

/// OpenSearch, or Elasticsearch (`OpenSearchStore::elasticsearch`).
///
/// ```no_run
/// # use agent_harness::knowledge::stores::opensearch::{OpenSearchStore, OpenSearchOptions};
/// let local = OpenSearchStore::new("http://localhost:9200", OpenSearchOptions { index: "handbook".into(), ..Default::default() });
/// // Amazon OpenSearch Service, signed with SigV4:
/// let aws = OpenSearchStore::new("https://search-x.eu-west-1.es.amazonaws.com",
///     OpenSearchOptions { index: "handbook".into(), aws_region: Some("eu-west-1".into()), ..Default::default() });
/// // OpenSearch Serverless (`serverless` is also inferred from `.aoss.` in the URL):
/// let aoss = OpenSearchStore::new("https://x.eu-west-1.aoss.amazonaws.com",
///     OpenSearchOptions { index: "handbook".into(), aws_region: Some("eu-west-1".into()), serverless: Some(true), ..Default::default() });
/// ```
///
/// AWS credentials come from the options, then the environment
/// (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`). Python
/// also asks botocore (instance roles, SSO, config files) when it is installed;
/// here those have to be exported into the environment first.
///
/// OpenSearch Serverless does not let a document choose its own id and takes a
/// few seconds to show what it was given, so there an upsert first removes the
/// record it replaces.
pub struct OpenSearchStore {
    name: &'static str,
    elastic: bool,
    url: String,
    index: String,
    serverless: bool,
    engine: String,
    refresh: bool,
    http: HttpClient,
}

/// Elasticsearch is the same store with a different mapping and search body.
///
/// ```no_run
/// # use agent_harness::knowledge::stores::opensearch::{ElasticsearchStore, OpenSearchOptions};
/// let store = ElasticsearchStore::elasticsearch("https://x.es.europe-west1.gcp.cloud.es.io",
///     OpenSearchOptions { api_key: Some("…".into()), index: "handbook".into(), ..Default::default() });
/// ```
pub type ElasticsearchStore = OpenSearchStore;

impl std::fmt::Debug for OpenSearchStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<{} {} {}>", if self.elastic { "ElasticsearchStore" } else { "OpenSearchStore" }, self.url, self.index)
    }
}

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

fn given(value: Option<String>) -> Option<String> {
    value.filter(|v| !v.is_empty())
}

/// `1`, `true`, `yes`, `on` — or their opposites.
pub(crate) fn flag(key: &str, value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" | "" => Ok(false),
        _ => Err(Error::config(format!("'{key}' is true or false, not '{value}'"))),
    }
}

/// Signs every request with SigV4, resolving credentials each time — explicit,
/// then the environment — and saying so plainly when there are none.
pub(crate) fn aws_signer(store: &'static str, label: &'static str, region: String, service: &'static str, keys: [Option<String>; 3]) -> Signer {
    Arc::new(move |method: &str, url: &str, body: &[u8], headers: &mut reqwest::header::HeaderMap| {
        let credentials = AwsCredentials::resolve(keys[0].as_deref(), keys[1].as_deref(), keys[2].as_deref());
        if !credentials.is_set() {
            return Err(store_error(
                store,
                format!("no AWS credentials were found for {label} — pass them, or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"),
            ));
        }
        sigv4::signer(credentials, region.clone(), service, true)(method, url, body, headers)
    })
}

const RANGE: [(&str, &str); 4] = [("$gt", "gt"), ("$gte", "gte"), ("$lt", "lt"), ("$lte", "lte")];

/// A filter as the inside of a `bool` query: `filter` for what must hold,
/// `must_not` for `$ne`. `None` when there is nothing to filter on.
pub fn bool_clause(wanted: &[Condition]) -> Option<Map<String, Value>> {
    let mut must = vec![];
    let mut must_not = vec![];
    for c in wanted {
        let key = format!("meta.{}", c.field);
        match c.op.as_str() {
            "$eq" => must.push(json!({"term": {key: c.value}})),
            "$ne" => must_not.push(json!({"term": {key: c.value}})),
            "$in" => must.push(json!({"terms": {key: c.value}})),
            op => {
                let range = RANGE.iter().find(|(o, _)| *o == op).map_or("gte", |(_, r)| *r);
                must.push(json!({"range": {key: {range: c.value}}}));
            }
        }
    }
    if must.is_empty() && must_not.is_empty() {
        return None;
    }
    let mut out = Map::new();
    if !must.is_empty() {
        out.insert("filter".into(), Value::Array(must));
    }
    if !must_not.is_empty() {
        out.insert("must_not".into(), Value::Array(must_not));
    }
    Some(out)
}

impl OpenSearchStore {
    /// An OpenSearch store at `url` (`http://localhost:9200` in Python's default).
    pub fn new(url: &str, options: OpenSearchOptions) -> Self {
        Self::build(url, options, false)
    }

    /// An Elasticsearch store: a `dense_vector` mapping, the top-level `knn`
    /// search, never serverless, always lucene.
    pub fn elasticsearch(url: &str, options: OpenSearchOptions) -> Self {
        Self::build(url, options, true)
    }

    fn build(url: &str, o: OpenSearchOptions, elastic: bool) -> Self {
        let name = if elastic { "elasticsearch" } else { "opensearch" };
        let (username, password, api_key) = if elastic {
            (
                given(o.username).or_else(|| env("ELASTICSEARCH_USERNAME")).unwrap_or_default(),
                given(o.password).or_else(|| env("ELASTICSEARCH_PASSWORD")).unwrap_or_default(),
                given(o.api_key).or_else(|| env("ELASTICSEARCH_API_KEY")).unwrap_or_default(),
            )
        } else {
            (
                given(o.username).or_else(|| env("OPENSEARCH_USERNAME")).unwrap_or_default(),
                given(o.password).or_else(|| env("OPENSEARCH_PASSWORD")).unwrap_or_default(),
                given(o.api_key).unwrap_or_default(),
            )
        };
        let serverless = !elastic && o.serverless.unwrap_or_else(|| url.contains(".aoss."));
        // Serverless runs faiss, and scores an inner product; vectors are sent
        // at unit length so that is a cosine.
        let engine = if elastic { "lucene".to_string() } else { given(o.engine).unwrap_or_else(|| if serverless { "faiss".into() } else { "lucene".into() }) };
        let aws_region = given(o.aws_region);

        let mut http =
            HttpClient::new(name, o.timeout, o.retries).secret(password.clone()).secret(api_key.clone()).secret(o.aws_secret_key.clone().unwrap_or_default());
        for (k, v) in o.headers {
            http = http.header(&k, v);
        }
        if !api_key.is_empty() {
            http = http.header("authorization", format!("ApiKey {api_key}"));
        } else if !username.is_empty() && aws_region.is_none() {
            let token = base64::engine::general_purpose::STANDARD.encode(format!("{username}:{password}"));
            http = http.header("authorization", format!("Basic {token}"));
        }
        if let Some(region) = aws_region {
            let label = if elastic { "Elasticsearch" } else { "OpenSearch" };
            http.signer =
                Some(aws_signer(name, label, region, if serverless { "aoss" } else { "es" }, [o.aws_access_key, o.aws_secret_key, o.aws_session_token]));
        }
        Self { name, elastic, url: url.trim_end_matches('/').to_string(), index: o.index, serverless, engine, refresh: o.refresh && !serverless, http }
    }

    /// As `vector_store("opensearch+https://host:9200/handbook?username=…")`
    /// builds it: the path is the index, the query string the options.
    pub fn from_url(mut p: UrlParts, bare: bool, elastic: bool) -> Result<Self> {
        let mut o = OpenSearchOptions::default();
        let url = if bare { "http://localhost:9200".to_string() } else { p.web() };
        if !bare && !p.path.is_empty() {
            o.index = p.path.clone();
        }
        for (key, value) in std::mem::take(&mut p.given) {
            match key.as_str() {
                "index" => o.index = value,
                "username" => o.username = Some(value),
                "password" => o.password = Some(value),
                "api_key" => o.api_key = Some(value),
                "aws_region" => o.aws_region = Some(value),
                "aws_access_key" => o.aws_access_key = Some(value),
                "aws_secret_key" => o.aws_secret_key = Some(value),
                "aws_session_token" => o.aws_session_token = Some(value),
                "serverless" if !elastic => o.serverless = Some(flag(&key, &value)?),
                "engine" if !elastic => o.engine = Some(value),
                "refresh" => o.refresh = flag(&key, &value)?,
                "timeout" => o.timeout = value.parse().map_err(|_| Error::config(format!("timeout is a number of seconds, not '{value}'")))?,
                "retries" => o.retries = value.parse().map_err(|_| Error::config(format!("retries is a whole number, not '{value}'")))?,
                other => return Err(Error::config(format!("{} takes no option '{other}'", if elastic { "ElasticsearchStore" } else { "OpenSearchStore" }))),
            }
        }
        Ok(Self::build(&url, o, elastic))
    }

    /// Is this the Serverless flavour (no ids of our own, no refresh, no delete-by-query)?
    pub fn serverless(&self) -> bool {
        self.serverless
    }

    // ---- the shape of the index -----------------------------------------------
    fn mapping(&self, dimension: usize) -> Value {
        let strings = json!([{"metadata_strings": {"path_match": "meta.*", "match_mapping_type": "string", "mapping": {"type": "keyword"}}}]);
        if self.elastic {
            return json!({"mappings": {
                "dynamic_templates": strings,
                "properties": {
                    "vector": {"type": "dense_vector", "dims": dimension, "index": true, "similarity": "cosine"},
                    "ref": {"type": "keyword"}, "text": {"type": "text"}}}});
        }
        let space = if self.engine == "faiss" { "innerproduct" } else { "cosinesimil" };
        json!({
            "settings": {"index": {"knn": true}},
            "mappings": {
                // Strings in metadata are exact values to filter on, not prose.
                "dynamic_templates": strings,
                "properties": {
                    "vector": {"type": "knn_vector", "dimension": dimension,
                               "method": {"name": "hnsw", "space_type": space, "engine": self.engine}},
                    "ref": {"type": "keyword"},
                    "text": {"type": "text"}}}})
    }

    fn search(&self, vector: Vec<f32>, k: usize, found: Option<Map<String, Value>>) -> Value {
        if self.elastic {
            let mut knn = json!({"field": "vector", "query_vector": vector, "k": k, "num_candidates": (k * 10).max(100)});
            if let Some(f) = found {
                knn["filter"] = json!({"bool": f});
            }
            return json!({"size": k, "_source": ["ref", "text", "meta"], "knn": knn});
        }
        let mut knn = json!({"vector": vector, "k": k});
        if let Some(f) = found {
            knn["filter"] = json!({"bool": f});
        }
        json!({"size": k, "_source": ["ref", "text", "meta"], "query": {"knn": {"vector": knn}}})
    }

    /// The engine's score as a cosine similarity.
    fn similarity(&self, score: f64) -> f64 {
        if !self.elastic && self.engine == "faiss" {
            // An inner product p is scored p + 1 when it is not negative, and
            // 1 / (1 - p) when it is.
            return if score >= 1.0 {
                score - 1.0
            } else if score != 0.0 {
                1.0 - 1.0 / score
            } else {
                0.0
            };
        }
        2.0 * score - 1.0 // lucene and Elasticsearch: (1 + cosine) / 2
    }

    fn prepare(&self, vector: &[f32]) -> Vec<f32> {
        if !self.elastic && self.engine == "faiss" { normalise(vector) } else { vector.to_vec() }
    }

    fn at(&self, path: &str) -> String {
        format!("{}/{}{path}", self.url, self.index)
    }

    async fn bulk(&self, lines: Vec<Value>) -> Result<()> {
        let mut body = String::new();
        for line in &lines {
            body.push_str(&serde_json::to_string(line).unwrap_or_default());
            body.push('\n');
        }
        let url = format!("{}/_bulk{}", self.url, if self.refresh { "?refresh=true" } else { "" });
        let answer = self.http.send(Method::POST, &url, Some(body.into_bytes()), Some("application/x-ndjson"), &[], &[]).await?;
        if answer.body.get("errors").is_some_and(truthy) {
            for item in answer.body["items"].as_array().into_iter().flatten() {
                let result = item.as_object().and_then(|m| m.values().next()).cloned().unwrap_or(Value::Null);
                let error = &result["error"];
                if truthy(error) && result["status"].as_u64() != Some(404) {
                    let shown: String = error.to_string().chars().take(300).collect();
                    return Err(store_error(self.name, format!("{} refused a record: {shown}", self.name)));
                }
            }
        }
        Ok(())
    }

    async fn remove(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        if let Some(ids) = ids
            && wanted.is_empty()
            && !self.serverless
        {
            return self.bulk(ids.iter().map(|i| json!({"delete": {"_index": self.index, "_id": i}})).collect()).await;
        }
        let mut clause = bool_clause(wanted).unwrap_or_default();
        if let Some(ids) = ids {
            let entry = clause.entry("filter").or_insert_with(|| Value::Array(vec![]));
            if let Value::Array(list) = entry {
                list.push(json!({"terms": {"ref": ids}}));
            }
        }
        let query = if clause.is_empty() { json!({"match_all": {}}) } else { json!({"bool": clause}) };
        if self.serverless {
            // No delete-by-query there: find what matches, then remove it by id.
            for _ in 0..1000 {
                let answer = self.http.json(Method::POST, &self.at("/_search"), Some(&json!({"size": 500, "_source": false, "query": query})), &[404]).await?;
                if answer.status == 404 {
                    return Ok(());
                }
                let hits: Vec<Value> = answer.body["hits"]["hits"].as_array().cloned().unwrap_or_default();
                if hits.is_empty() {
                    return Ok(());
                }
                let n = hits.len();
                self.bulk(hits.iter().map(|h| json!({"delete": {"_index": self.index, "_id": h["_id"]}})).collect()).await?;
                if n < 500 {
                    return Ok(());
                }
            }
            return Ok(());
        }
        let params = if self.refresh { "?refresh=true&conflicts=proceed" } else { "?conflicts=proceed" };
        self.http.json(Method::POST, &self.at(&format!("/_delete_by_query{params}")), Some(&json!({"query": query})), &[]).await?;
        Ok(())
    }
}

/// Python's truthiness, for a JSON value.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|x| x != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[async_trait]
impl VectorStore for OpenSearchStore {
    fn name(&self) -> &str {
        self.name
    }

    fn batch_size(&self) -> usize {
        500
    }

    async fn ensure(&self, dimension: usize) -> Result<()> {
        let found = self.http.json(Method::HEAD, &self.at(""), None, &[404]).await?;
        if found.status != 404 {
            return Ok(());
        }
        match self.http.json(Method::PUT, &self.at(""), Some(&self.mapping(dimension)), &[]).await {
            Ok(_) => Ok(()),
            // Another replica won.
            Err(e) if e.to_string().contains("resource_already_exists") => Ok(()),
            Err(e) => Err(e),
        }
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        if self.serverless {
            let ids: Vec<String> = records.iter().map(|r| r.id.clone()).collect();
            self.remove(Some(&ids), &[]).await?;
        }
        let mut lines = Vec::with_capacity(records.len() * 2);
        for r in records {
            let mut action = json!({"_index": self.index});
            if !self.serverless {
                action["_id"] = json!(r.id);
            }
            lines.push(json!({"index": action}));
            lines.push(json!({"vector": self.prepare(&r.vector), "ref": r.id, "text": r.text, "meta": r.metadata}));
        }
        self.bulk(lines).await
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let body = self.search(self.prepare(vector), k, bool_clause(wanted));
        let answer = self.http.json(Method::POST, &self.at("/_search"), Some(&body), &[]).await?;
        let mut out = vec![];
        for hit in answer.body["hits"]["hits"].as_array().into_iter().flatten() {
            let source = &hit["_source"];
            let id = source["ref"].as_str().filter(|s| !s.is_empty()).or_else(|| hit["_id"].as_str()).unwrap_or_default();
            out.push(VectorHit {
                id: id.to_string(),
                score: self.similarity(hit["_score"].as_f64().unwrap_or(0.0)),
                text: source["text"].as_str().unwrap_or_default().to_string(),
                metadata: source["meta"].as_object().cloned().unwrap_or_default(),
            });
        }
        Ok(out)
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        self.remove(ids, wanted).await
    }

    async fn count(&self) -> Result<usize> {
        let answer = self.http.json(Method::GET, &self.at("/_count"), None, &[404]).await?;
        if answer.status == 404 {
            return Ok(0);
        }
        Ok(answer.body["count"].as_u64().unwrap_or(0) as usize)
    }

    async fn clear(&self) -> Result<()> {
        self.remove(None, &[]).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::knowledge::base::conditions;

    type Seen = Arc<Mutex<Vec<(String, String, reqwest::header::HeaderMap, String)>>>;
    type Answer = Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync>;

    /// A local server that records every request and answers with `answer`.
    async fn fake(answer: Answer) -> (String, Seen) {
        let seen: Seen = Arc::default();
        let s = seen.clone();
        let app = axum::Router::new().fallback(move |req: axum::extract::Request| {
            let (s, answer) = (s.clone(), answer.clone());
            async move {
                let (parts, body) = req.into_parts();
                let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap_or_default();
                let text = String::from_utf8_lossy(&bytes).to_string();
                let path = parts.uri.path_and_query().map(|p| p.to_string()).unwrap_or_default();
                s.lock().unwrap().push((parts.method.to_string(), path.clone(), parts.headers.clone(), text.clone()));
                let (status, out) = answer(parts.method.as_str(), &path, &text);
                (axum::http::StatusCode::from_u16(status).unwrap(), [("content-type", "application/json")], out)
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), seen)
    }

    fn opts() -> OpenSearchOptions {
        OpenSearchOptions { index: "handbook".into(), retries: 0, ..Default::default() }
    }

    #[test]
    fn filters_become_bool_clauses() {
        let wanted = conditions(Some(&json!({"a": "x", "b": {"$in": [1, 2]}, "c": {"$gte": 3, "$lt": 9}, "d": {"$ne": true}}))).unwrap();
        let clause = Value::Object(bool_clause(&wanted).unwrap());
        assert_eq!(
            clause,
            json!({
                "filter": [{"term": {"meta.a": "x"}}, {"terms": {"meta.b": [1, 2]}}, {"range": {"meta.c": {"gte": 3}}}, {"range": {"meta.c": {"lt": 9}}}],
                "must_not": [{"term": {"meta.d": true}}]})
        );
        assert!(bool_clause(&[]).is_none());
    }

    #[test]
    fn scores_become_cosines() {
        let lucene = OpenSearchStore::new("http://x", opts());
        assert!((lucene.similarity(1.0) - 1.0).abs() < 1e-9);
        assert!((lucene.similarity(0.5) - 0.0).abs() < 1e-9);
        let faiss = OpenSearchStore::new("https://x.eu-west-1.aoss.amazonaws.com", opts());
        assert!(faiss.serverless() && faiss.engine == "faiss" && !faiss.refresh);
        assert!((faiss.similarity(2.0) - 1.0).abs() < 1e-9);
        assert!((faiss.similarity(0.5) - -1.0).abs() < 1e-9);
        assert_eq!(faiss.similarity(0.0), 0.0);
        let es = OpenSearchStore::elasticsearch("http://x", opts());
        assert_eq!(es.name(), "elasticsearch");
        assert!((es.similarity(0.75) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn urls_are_read_like_python() {
        let p = UrlParts::parse("opensearch+https://host:9200/hand?refresh=false&username=u&password=p").unwrap();
        let s = OpenSearchStore::from_url(p, false, false).unwrap();
        assert_eq!((s.url.as_str(), s.index.as_str(), s.refresh), ("https://host:9200", "hand", false));
        let auth = s.http.headers.iter().find(|(k, _)| k == "authorization").unwrap();
        assert_eq!(auth.1, "Basic dTpw");
        let bare = OpenSearchStore::from_url(UrlParts { scheme: "opensearch".into(), ..Default::default() }, true, false).unwrap();
        assert_eq!((bare.url.as_str(), bare.index.as_str()), ("http://localhost:9200", "knowledge"));
        assert!(OpenSearchStore::from_url(UrlParts::parse("es://h/i?serverless=1").unwrap(), false, true).is_err());
        assert!(OpenSearchStore::from_url(UrlParts::parse("opensearch://h/i?nope=1").unwrap(), false, false).is_err());
    }

    #[tokio::test]
    async fn opensearch_requests_have_the_python_shape() {
        let (url, seen) = fake(Arc::new(|method, path, _| match (method, path) {
            ("HEAD", "/handbook") => (404, String::new()),
            ("POST", p) if p.starts_with("/_bulk") => (200, r#"{"errors": false, "items": []}"#.into()),
            ("POST", "/handbook/_search") => {
                (200, r#"{"hits": {"hits": [{"_id": "x", "_score": 1.0, "_source": {"ref": "a", "text": "hello", "meta": {"n": 1}}}]}}"#.into())
            }
            ("GET", "/handbook/_count") => (200, r#"{"count": 7}"#.into()),
            _ => (200, "{}".into()),
        }))
        .await;
        let store = OpenSearchStore::new(&url, OpenSearchOptions { api_key: Some("secret-key".into()), ..opts() });
        store.ensure(3).await.unwrap();
        store.upsert(vec![VectorRecord::new("a", vec![1.0, 0.0, 0.0], "hello", json!({"n": 1}))]).await.unwrap();
        let hits = store.query(&[1.0, 0.0, 0.0], 2, Some(&json!({"n": {"$gte": 1}}))).await.unwrap();
        assert_eq!(hits[0].id, "a");
        assert!((hits[0].score - 1.0).abs() < 1e-9);
        assert_eq!(hits[0].metadata["n"], 1);
        store.delete(Some(vec!["a".into()]), None).await.unwrap();
        store.delete(None, Some(&json!({"n": 1}))).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 7);

        let seen = seen.lock().unwrap();
        let calls: Vec<(String, String)> = seen.iter().map(|(m, p, _, _)| (m.clone(), p.clone())).collect();
        assert_eq!(calls[0], ("HEAD".into(), "/handbook".into()));
        assert_eq!(calls[1], ("PUT".into(), "/handbook".into()));
        let mapping: Value = serde_json::from_str(&seen[1].3).unwrap();
        assert_eq!(mapping["mappings"]["properties"]["vector"]["method"], json!({"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}));
        assert_eq!(mapping["settings"], json!({"index": {"knn": true}}));
        assert_eq!(seen[1].2["authorization"], "ApiKey secret-key");
        assert_eq!(calls[2], ("POST".into(), "/_bulk?refresh=true".into()));
        assert_eq!(seen[2].2["content-type"], "application/x-ndjson");
        let lines: Vec<Value> = seen[2].3.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines[0], json!({"index": {"_index": "handbook", "_id": "a"}}));
        assert_eq!(lines[1], json!({"vector": [1.0, 0.0, 0.0], "ref": "a", "text": "hello", "meta": {"n": 1}}));
        let search: Value = serde_json::from_str(&seen[3].3).unwrap();
        assert_eq!(
            search,
            json!({"size": 2, "_source": ["ref", "text", "meta"], "query": {"knn": {"vector": {"vector": [1.0, 0.0, 0.0], "k": 2, "filter": {"bool": {"filter": [{"range": {"meta.n": {"gte": 1}}}]}}}}}})
        );
        let deleted: Value = serde_json::from_str(seen[4].3.trim()).unwrap();
        assert_eq!(deleted, json!({"delete": {"_index": "handbook", "_id": "a"}}));
        assert_eq!(calls[5], ("POST".into(), "/handbook/_delete_by_query?refresh=true&conflicts=proceed".into()));
        assert_eq!(serde_json::from_str::<Value>(&seen[5].3).unwrap(), json!({"query": {"bool": {"filter": [{"term": {"meta.n": 1}}]}}}));
    }

    #[tokio::test]
    async fn elasticsearch_and_serverless_shapes() {
        let (url, seen) = fake(Arc::new(|method, path, _| match (method, path) {
            ("HEAD", _) => (404, String::new()),
            ("POST", "/handbook/_search") => (200, r#"{"hits": {"hits": [{"_id": "gen-1", "_score": 0.9}]}}"#.into()),
            ("POST", p) if p.starts_with("/_bulk") => (200, r#"{"errors": true, "items": [{"delete": {"status": 404, "error": {"type": "nf"}}}]}"#.into()),
            _ => (200, "{}".into()),
        }))
        .await;
        let es = OpenSearchStore::elasticsearch(&url, opts());
        es.ensure(4).await.unwrap();
        es.query(&[1.0, 0.0, 0.0, 0.0], 3, Some(&json!({"lang": "en"}))).await.unwrap();
        {
            let seen = seen.lock().unwrap();
            let mapping: Value = serde_json::from_str(&seen[1].3).unwrap();
            assert_eq!(mapping["mappings"]["properties"]["vector"], json!({"type": "dense_vector", "dims": 4, "index": true, "similarity": "cosine"}));
            assert!(mapping.get("settings").is_none());
            let search: Value = serde_json::from_str(&seen[2].3).unwrap();
            assert_eq!(
                search["knn"],
                json!({"field": "vector", "query_vector": [1.0, 0.0, 0.0, 0.0], "k": 3, "num_candidates": 100, "filter": {"bool": {"filter": [{"term": {"meta.lang": "en"}}]}}})
            );
        }
        seen.lock().unwrap().clear();
        // Serverless: no ids of our own, so an upsert removes what it replaces first.
        let aoss = OpenSearchStore::new(&url, OpenSearchOptions { serverless: Some(true), ..opts() });
        aoss.upsert(vec![VectorRecord::new("a", vec![3.0, 4.0], "t", json!({}))]).await.unwrap();
        let seen = seen.lock().unwrap();
        let calls: Vec<String> = seen.iter().map(|(m, p, _, _)| format!("{m} {p}")).collect();
        assert_eq!(calls, ["POST /handbook/_search", "POST /_bulk", "POST /_bulk"]);
        assert_eq!(
            serde_json::from_str::<Value>(&seen[0].3).unwrap(),
            json!({"size": 500, "_source": false, "query": {"bool": {"filter": [{"terms": {"ref": ["a"]}}]}}})
        );
        assert_eq!(serde_json::from_str::<Value>(seen[1].3.trim()).unwrap(), json!({"delete": {"_index": "handbook", "_id": "gen-1"}}));
        let lines: Vec<Value> = seen[2].3.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines[0], json!({"index": {"_index": "handbook"}}));
        let v = lines[1]["vector"].as_array().unwrap();
        assert!((v[0].as_f64().unwrap() - 0.6).abs() < 1e-6, "faiss vectors go at unit length");
    }

    #[tokio::test]
    async fn signed_requests_carry_sigv4() {
        let (url, seen) = fake(Arc::new(|_, _, _| (200, r#"{"count": 2}"#.into()))).await;
        let store = OpenSearchStore::new(
            &url,
            OpenSearchOptions {
                aws_region: Some("eu-west-1".into()),
                aws_access_key: Some("AKIDEXAMPLE".into()),
                aws_secret_key: Some("secret".into()),
                username: Some("ignored".into()),
                ..opts()
            },
        );
        assert_eq!(store.count().await.unwrap(), 2);
        let seen = seen.lock().unwrap();
        let auth = seen[0].2["authorization"].to_str().unwrap();
        assert!(auth.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/") && auth.contains("/eu-west-1/es/aws4_request"), "{auth}");
        assert!(seen[0].2.contains_key("x-amz-content-sha256"));
    }

    #[tokio::test]
    async fn bulk_errors_are_reported() {
        let (url, _) =
            fake(Arc::new(|_, _, _| (200, r#"{"errors": true, "items": [{"index": {"status": 400, "error": {"type": "mapper_parsing_exception"}}}]}"#.into())))
                .await;
        let store = OpenSearchStore::new(&url, opts());
        let e = store.upsert(vec![VectorRecord::new("a", vec![1.0], "t", json!({}))]).await.unwrap_err();
        assert!(e.to_string().contains("refused a record") && e.to_string().contains("mapper_parsing_exception"));
    }

    #[tokio::test]
    async fn live() {
        for key in ["AH_LIVE_OPENSEARCH_URL", "AH_LIVE_ELASTICSEARCH_URL"] {
            let Ok(url) = std::env::var(key) else { continue };
            let store = super::super::vector_store(&url).unwrap();
            let report = store.check(8).await;
            assert_eq!(report["ok"], true, "{report:#}");
        }
    }
}
