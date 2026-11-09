//! Qdrant — self-hosted or Qdrant Cloud.
//!
//! This file also holds what the HTTP stores beside it share: the transport
//! settings every one of them takes (`timeout`, `retries`, `retry`, extra
//! headers), and JSON written the way Python's `json.dumps` writes it, so a
//! value stored as text reads the same from either language.

use async_trait::async_trait;
use reqwest::Method;
use serde_json::{Map, Value, json};

use super::UrlParts;
use crate::errors::{Error, Result};
use crate::knowledge::base::{Condition, HttpClient, VectorHit, VectorRecord, VectorStore, as_uuid};
use crate::llm_providers::resilience::RetryPolicy;

// ---- shared by the HTTP stores in this folder --------------------------------

/// How a store speaks HTTP: the timeout, the retries, any extra headers.
#[derive(Debug, Clone)]
pub(super) struct Settings {
    pub timeout: f64,
    pub retries: u32,
    pub retry: Option<RetryPolicy>,
    pub headers: Vec<(String, String)>,
}

impl Default for Settings {
    fn default() -> Self {
        Self { timeout: 30.0, retries: 3, retry: None, headers: vec![] }
    }
}

impl Settings {
    /// A client: these settings, then the store's own headers (which win), and
    /// the secrets that must never appear in an error.
    pub fn client(&self, name: &str, own: &[(&str, String)], secrets: &[&str]) -> HttpClient {
        let mut http = HttpClient::new(name, self.timeout, self.retries);
        if let Some(policy) = &self.retry {
            http.retry = policy.clone();
        }
        for (k, v) in &self.headers {
            http = http.header(k, v.clone());
        }
        for (k, v) in own {
            http = http.header(k, v.clone());
        }
        for s in secrets {
            http = http.secret(*s);
        }
        http
    }

    /// `timeout` and `retries` from a URL's query string.
    pub fn from_url(p: &mut UrlParts, store: &str) -> Result<Self> {
        let mut out = Self::default();
        if let Some(t) = p.take("timeout") {
            out.timeout = t.trim().parse().map_err(|_| Error::config(format!("{store}: timeout={t} is not a number of seconds")))?;
        }
        if let Some(r) = p.take("retries") {
            out.retries = r.trim().parse().map_err(|_| Error::config(format!("{store}: retries={r} is not a whole number")))?;
        }
        Ok(out)
    }
}

/// Anything left in a URL's query string is an option this store does not take.
pub(super) fn no_more(p: &UrlParts, store: &str) -> Result<()> {
    match p.given.keys().next() {
        Some(key) => Err(Error::config(format!("{store} takes no option '{key}' (in the URL's query string)"))),
        None => Ok(()),
    }
}

/// The given value, or the environment's when it is empty — Python's `x or os.environ.get(..)`.
pub(super) fn or_env(given: &str, key: &str) -> String {
    if given.is_empty() { std::env::var(key).unwrap_or_default() } else { given.to_string() }
}

/// What Python's `str()` makes of a JSON value: an id that came back as a number, say.
pub(super) fn text_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        other => py_dumps(other),
    }
}

/// A vector as JSON numbers, each written as the shortest decimal that is the
/// same `f32` — `0.6`, not `0.6000000238418579`.
pub(super) fn floats(vector: &[f32]) -> Value {
    Value::Array(vector.iter().map(|x| x.to_string().parse::<f64>().ok().and_then(serde_json::Number::from_f64).map_or(Value::Null, Value::Number)).collect())
}

/// `float(x or 0.0)`.
pub(super) fn number(value: Option<&Value>) -> f64 {
    value.and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))).unwrap_or(0.0)
}

/// A JSON object, or an empty one.
pub(super) fn object(value: Option<&Value>) -> Map<String, Value> {
    value.and_then(Value::as_object).cloned().unwrap_or_default()
}

struct PyFormatter;

impl serde_json::ser::Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { writer.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { writer.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

/// JSON as Python's `json.dumps(value, ensure_ascii=False)` writes it: `", "` and `": "`.
pub(super) fn py_dumps<T: serde::Serialize + ?Sized>(value: &T) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyFormatter);
    if value.serialize(&mut ser).is_err() {
        return String::new();
    }
    String::from_utf8(out).unwrap_or_default()
}

// ---- Qdrant ---------------------------------------------------------------------

/// ```ignore
/// QdrantStore::new("http://localhost:6333").collection("handbook");
/// QdrantStore::new("https://xyz.cloud.qdrant.io").api_key("…").collection("handbook");
/// ```
///
/// Qdrant takes UUIDs or integers as ids; yours is kept in the payload and is
/// what comes back.
#[derive(Debug, Clone)]
pub struct QdrantStore {
    pub url: String,
    pub collection: String,
    pub api_key: String,
    settings: Settings,
    http: HttpClient,
}

impl Default for QdrantStore {
    fn default() -> Self {
        Self::new("http://localhost:6333")
    }
}

impl QdrantStore {
    /// The key comes from `$QDRANT_API_KEY` unless one is given.
    pub fn new(url: &str) -> Self {
        let settings = Settings::default();
        let mut store = Self {
            url: url.trim_end_matches('/').to_string(),
            collection: "knowledge".into(),
            api_key: or_env("", "QDRANT_API_KEY"),
            http: settings.client("qdrant", &[], &[]),
            settings,
        };
        store.rebuild();
        store
    }

    pub fn collection(mut self, collection: &str) -> Self {
        self.collection = collection.into();
        self
    }

    pub fn api_key(mut self, key: &str) -> Self {
        self.api_key = or_env(key, "QDRANT_API_KEY");
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

    /// `qdrant://host:6333/collection?api_key=…`, or the defaults for a bare `qdrant`.
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        if bare {
            return Ok(Self::default());
        }
        let settings = Settings::from_url(&mut p, "qdrant")?;
        let named = p.take("collection");
        let key = p.take("api_key").unwrap_or_default();
        no_more(&p, "qdrant")?;
        let mut store = Self::new(&p.web()).api_key(&key);
        store.settings = settings;
        store.rebuild();
        if !p.path.is_empty() {
            store.collection = p.path.clone();
        } else if let Some(c) = named {
            store.collection = c;
        }
        Ok(store)
    }

    fn rebuild(&mut self) {
        let own: Vec<(&str, String)> = if self.api_key.is_empty() { vec![] } else { vec![("api-key", self.api_key.clone())] };
        self.http = self.settings.client("qdrant", &own, &[&self.api_key]);
    }

    fn base(&self) -> String {
        format!("{}/collections/{}", self.url, self.collection)
    }

    /// The filter in Qdrant's terms: `must` and `must_not`, on the `meta` payload.
    pub fn filter(wanted: &[Condition]) -> Option<Value> {
        let mut must = Vec::new();
        let mut must_not = Vec::new();
        for c in wanted {
            let key = format!("meta.{}", c.field);
            match c.op.as_str() {
                "$eq" => must.push(json!({"key": key, "match": {"value": c.value}})),
                "$ne" => must_not.push(json!({"key": key, "match": {"value": c.value}})),
                "$in" => must.push(json!({"key": key, "match": {"any": c.value}})),
                op => {
                    let range = &op[1..]; // $gt → gt, $gte → gte, …
                    must.push(json!({"key": key, "range": {range: c.value}}));
                }
            }
        }
        if must.is_empty() && must_not.is_empty() {
            return None;
        }
        let mut out = Map::new();
        if !must.is_empty() {
            out.insert("must".into(), Value::Array(must));
        }
        if !must_not.is_empty() {
            out.insert("must_not".into(), Value::Array(must_not));
        }
        Some(Value::Object(out))
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    fn name(&self) -> &str {
        "qdrant"
    }

    fn batch_size(&self) -> usize {
        256
    }

    async fn ensure(&self, dimension: usize) -> Result<()> {
        let base = self.base();
        if self.http.json(Method::GET, &base, None, &[404]).await?.status == 404 {
            // 409: another replica created it between the look and the leap.
            self.http.json(Method::PUT, &base, Some(&json!({"vectors": {"size": dimension, "distance": "Cosine"}})), &[409]).await?;
        }
        Ok(())
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        let points: Vec<Value> = records
            .iter()
            .map(|r| json!({"id": as_uuid(&r.id), "vector": floats(&r.vector), "payload": {"ref": r.id, "text": r.text, "meta": r.metadata}}))
            .collect();
        self.http.json(Method::PUT, &format!("{}/points?wait=true", self.base()), Some(&json!({"points": points})), &[]).await?;
        Ok(())
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let mut body = json!({"query": floats(vector), "limit": k, "with_payload": true});
        if let Some(found) = Self::filter(wanted) {
            body["filter"] = found;
        }
        let mut answer = self.http.json(Method::POST, &format!("{}/points/query", self.base()), Some(&body), &[404]).await?;
        if answer.status == 404 {
            // Before 1.10 the same question was asked at /points/search.
            let map = body.as_object_mut().expect("an object");
            let query = map.remove("query").unwrap_or_default();
            map.insert("vector".into(), query);
            answer = self.http.json(Method::POST, &format!("{}/points/search", self.base()), Some(&body), &[]).await?;
        }
        let result = answer.body.get("result").cloned().unwrap_or(Value::Null);
        let points = match &result {
            Value::Object(m) => m.get("points").and_then(Value::as_array).cloned().unwrap_or_default(),
            Value::Array(a) => a.clone(),
            _ => vec![],
        };
        Ok(points
            .iter()
            .map(|p| {
                let payload = object(p.get("payload"));
                let id = match payload.get("ref").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                    Some(r) => r.to_string(),
                    None => text_of(p.get("id").unwrap_or(&Value::Null)),
                };
                VectorHit {
                    id,
                    score: number(p.get("score")),
                    text: payload.get("text").and_then(Value::as_str).unwrap_or_default().to_string(),
                    metadata: object(payload.get("meta")),
                }
            })
            .collect())
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        let mut found = Self::filter(wanted).unwrap_or_else(|| json!({}));
        if let Some(ids) = ids {
            let has: Vec<String> = ids.iter().map(|i| as_uuid(i)).collect();
            let map = found.as_object_mut().expect("an object");
            let must = map.entry("must").or_insert_with(|| json!([]));
            if let Some(list) = must.as_array_mut() {
                list.push(json!({"has_id": has}));
            }
        }
        self.http.json(Method::POST, &format!("{}/points/delete?wait=true", self.base()), Some(&json!({"filter": found})), &[]).await?;
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let answer = self.http.json(Method::POST, &format!("{}/points/count", self.base()), Some(&json!({"exact": true})), &[]).await?;
        Ok(number(answer.body.get("result").and_then(|r| r.get("count"))) as usize)
    }

    async fn clear(&self) -> Result<()> {
        self.http.json(Method::POST, &format!("{}/points/delete?wait=true", self.base()), Some(&json!({"filter": {}})), &[]).await?;
        Ok(())
    }
}

/// A stand-in database for the tests: records every request, answers as told.
#[cfg(test)]
pub(super) mod fake {
    use std::sync::{Arc, Mutex};

    use axum::Router;
    use axum::body::Bytes;
    use axum::extract::State;
    use axum::http::{HeaderMap, Method, StatusCode, Uri};
    use axum::response::{IntoResponse, Response};
    use serde_json::Value;

    #[derive(Debug, Clone)]
    pub struct Seen {
        pub method: String,
        pub path: String,
        pub query: String,
        pub headers: HeaderMap,
        pub raw: String,
        /// The body as JSON, or `null`.
        pub body: Value,
    }

    impl Seen {
        pub fn header(&self, name: &str) -> String {
            self.headers.get(name).and_then(|v| v.to_str().ok()).unwrap_or_default().to_string()
        }
    }

    type Answer = Arc<dyn Fn(&Seen) -> (u16, Value) + Send + Sync>;

    #[derive(Clone)]
    struct Shared {
        seen: Arc<Mutex<Vec<Seen>>>,
        answer: Answer,
    }

    pub struct Fake {
        pub url: String,
        seen: Arc<Mutex<Vec<Seen>>>,
    }

    impl Fake {
        pub fn seen(&self) -> Vec<Seen> {
            self.seen.lock().unwrap().clone()
        }
    }

    /// Serve on 127.0.0.1; `answer` gives each request its status and JSON
    /// body (`null` for an empty body).
    pub async fn serve(answer: impl Fn(&Seen) -> (u16, Value) + Send + Sync + 'static) -> Fake {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let shared = Shared { seen: seen.clone(), answer: Arc::new(answer) };
        let app = Router::new().fallback(handle).with_state(shared);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Fake { url: format!("http://{addr}"), seen }
    }

    async fn handle(State(shared): State<Shared>, method: Method, uri: Uri, headers: HeaderMap, body: Bytes) -> Response {
        let raw = String::from_utf8_lossy(&body).into_owned();
        let seen = Seen {
            method: method.to_string(),
            path: uri.path().to_string(),
            query: uri.query().unwrap_or_default().to_string(),
            headers,
            body: serde_json::from_str(&raw).unwrap_or(Value::Null),
            raw,
        };
        let (status, value) = (shared.answer)(&seen);
        shared.seen.lock().unwrap().push(seen);
        let text = if value.is_null() { String::new() } else { value.to_string() };
        (StatusCode::from_u16(status).unwrap(), [("content-type", "application/json")], text).into_response()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::knowledge::base::conditions;

    fn record() -> VectorRecord {
        VectorRecord::new("doc#0", vec![0.6, 0.8], "Refunds take five days.", json!({"source": "faq.md", "year": 2026}))
    }

    fn wanted() -> Value {
        json!({"source": "faq.md", "year": {"$gte": 2025}, "lang": {"$in": ["en", "fr"]}})
    }

    #[test]
    fn filters_become_must_and_must_not() {
        let c = conditions(Some(&json!({"a": 1, "b": {"$ne": true}, "c": {"$lt": 3}, "d": {"$in": ["x"]}}))).unwrap();
        let mut f = QdrantStore::filter(&c).unwrap();
        let mut must = f["must"].as_array().unwrap().clone();
        must.sort_by_key(|v| v.to_string());
        assert_eq!(
            must,
            vec![
                json!({"key": "meta.a", "match": {"value": 1}}),
                json!({"key": "meta.c", "range": {"lt": 3}}),
                json!({"key": "meta.d", "match": {"any": ["x"]}}),
            ]
        );
        assert_eq!(f["must_not"].take(), json!([{"key": "meta.b", "match": {"value": true}}]));
        assert!(QdrantStore::filter(&[]).is_none());
    }

    #[test]
    fn python_json() {
        assert_eq!(py_dumps(&json!({"a": [1, "é"], "b": 1.5})), r#"{"a": [1, "é"], "b": 1.5}"#);
        assert_eq!(text_of(&json!(7)), "7");
        assert_eq!(floats(&[0.6, 1.0, -0.1]).to_string(), "[0.6,1.0,-0.1]");
    }

    #[test]
    fn urls() {
        let q = QdrantStore::from_url(UrlParts::parse("qdrant+https://xyz.cloud.qdrant.io:6333/handbook?api_key=k").unwrap(), false).unwrap();
        assert_eq!((q.url.as_str(), q.collection.as_str(), q.api_key.as_str()), ("https://xyz.cloud.qdrant.io:6333", "handbook", "k"));
        let q = QdrantStore::from_url(UrlParts { scheme: "qdrant".into(), ..Default::default() }, true).unwrap();
        assert_eq!((q.url.as_str(), q.collection.as_str()), ("http://localhost:6333", "knowledge"));
        assert!(QdrantStore::from_url(UrlParts::parse("qdrant://h/c?colour=red").unwrap(), false).is_err());
        let q = QdrantStore::from_url(UrlParts::parse("qdrant://h/c?timeout=5&retries=1").unwrap(), false).unwrap();
        assert_eq!((q.settings.timeout, q.settings.retries), (5.0, 1));
    }

    #[tokio::test]
    async fn requests_are_what_qdrant_expects() {
        let fake = fake::serve(|s| match (s.method.as_str(), s.path.as_str()) {
            ("GET", "/collections/kb") => (404, json!({})),
            (_, p) if p.ends_with("/points/query") => {
                (200, json!({"result": {"points": [{"id": "u", "score": 0.91, "payload": {"ref": "doc#0", "text": "T", "meta": {"year": 2026}}}]}}))
            }
            (_, p) if p.ends_with("/points/count") => (200, json!({"result": {"count": 7}})),
            _ => (200, json!({"result": true})),
        })
        .await;
        let store = QdrantStore::new(&fake.url).collection("kb").api_key("secret-key").retries(0);
        store.ensure(2).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen[1].method, "PUT");
        assert_eq!(seen[1].body, json!({"vectors": {"size": 2, "distance": "Cosine"}}));
        assert_eq!(seen[0].header("api-key"), "secret-key");

        store.upsert(vec![record()]).await.unwrap();
        let seen = fake.seen();
        let point = &seen[2].body["points"][0];
        assert_eq!(point["payload"], json!({"ref": "doc#0", "text": "Refunds take five days.", "meta": {"source": "faq.md", "year": 2026}}));
        assert_eq!(point["id"], json!(as_uuid("doc#0")));
        assert_eq!(seen[2].query, "wait=true");

        let hits = store.query(&[0.6, 0.8], 3, Some(&wanted())).await.unwrap();
        let seen = fake.seen();
        let mut must = seen[3].body["filter"]["must"].as_array().unwrap().clone();
        must.sort_by_key(|v| v.to_string());
        assert_eq!(
            must,
            vec![
                json!({"key": "meta.lang", "match": {"any": ["en", "fr"]}}),
                json!({"key": "meta.source", "match": {"value": "faq.md"}}),
                json!({"key": "meta.year", "range": {"gte": 2025}}),
            ]
        );
        assert_eq!(seen[3].body["limit"], json!(3));
        assert_eq!(hits, vec![VectorHit { id: "doc#0".into(), score: 0.91, text: "T".into(), metadata: object(Some(&json!({"year": 2026}))) }]);

        store.delete(Some(vec!["doc#0".into()]), Some(&json!({"draft": {"$ne": true}}))).await.unwrap();
        let seen = fake.seen();
        let body = &seen[4].body["filter"];
        assert_eq!(body["must_not"], json!([{"key": "meta.draft", "match": {"value": true}}]));
        assert_eq!(body["must"][0]["has_id"], json!([point["id"]]));
        assert_eq!(seen[4].query, "wait=true");

        assert_eq!(store.count().await.unwrap(), 7);
        assert_eq!(fake.seen()[5].body, json!({"exact": true}));
        store.clear().await.unwrap();
        assert_eq!(fake.seen()[6].body, json!({"filter": {}}));
    }

    #[tokio::test]
    async fn an_old_qdrant_is_searched_instead() {
        let fake = fake::serve(|s| {
            if s.path.ends_with("/points/query") { (404, json!({})) } else { (200, json!({"result": [{"id": 5, "score": 0.5, "payload": {"text": "x"}}]})) }
        })
        .await;
        let store = QdrantStore::new(&fake.url).retries(0);
        let hits = store.query(&[1.0, 0.0], 2, None).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen[1].path, "/collections/knowledge/points/search");
        assert!(seen[1].body.get("query").is_none() && seen[1].body["vector"] == json!([1.0, 0.0]));
        assert_eq!((hits[0].id.as_str(), hits[0].score), ("5", 0.5));
    }

    #[tokio::test]
    async fn retried_when_it_may_help_and_keys_stay_out_of_errors() {
        let answers = Mutex::new(vec![(200, json!({"result": {"count": 7}})), (429, json!({})), (503, json!({}))]);
        let fake = fake::serve(move |_| answers.lock().unwrap().pop().unwrap()).await;
        let fast = RetryPolicy { max_retries: 3, initial_delay: 0.01, max_delay: 0.02, max_retry_after: 20.0, ..RetryPolicy::default() };
        let store = QdrantStore::new(&fake.url).api_key("k").retry(fast);
        assert_eq!(store.count().await.unwrap(), 7);
        assert_eq!(fake.seen().len(), 3);

        let fake = fake::serve(|_| (403, json!({"status": {"error": "bad key qd-secret-key"}}))).await;
        let store = QdrantStore::new(&fake.url).api_key("qd-secret-key").retries(0);
        let e = store.count().await.unwrap_err();
        assert!(matches!(e, Error::VectorStore { status: Some(403), .. }));
        assert!(!e.to_string().contains("qd-secret-key"));
    }

    #[tokio::test]
    async fn live() {
        // e.g. http://localhost:6333; the key, if any, from $QDRANT_API_KEY.
        let Ok(url) = std::env::var("AH_LIVE_QDRANT_URL") else { return };
        let store = QdrantStore::new(&url).collection("ah_check");
        let report = store.check(8).await;
        assert_eq!(report["ok"], json!(true), "{report}");
    }
}
