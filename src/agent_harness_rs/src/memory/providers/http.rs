//! Memory behind your own HTTP API — no extra driver.
//!
//! Use it when memory has to live in a service you already run, or behind an
//! internal API that enforces its own access rules. The contract is small:
//!
//! ```text
//! POST   {base}/records            {record}
//! GET    {base}/records            ?scope=&kind=&limit=&user_id=&...
//! DELETE {base}/records            ?scope=&user_id=&...
//! GET    {base}/docs/{key}         -> {"text": "..."}
//! PUT    {base}/docs/{key}         {"text": "...", "name": ..., "namespace": ...}
//! ```
//!
//! The same contract the Python `HTTPMemory` speaks, so one service serves both.

use async_trait::async_trait;
use base64::Engine as _;
use reqwest::Method;
use serde_json::{Value, json};

use crate::errors::{Error, Result};
use crate::knowledge::base::{HttpClient, Reply};
use crate::memory::base::{MemoryRecord, MemoryStore, RecordQuery, doc_key};
use crate::memory::providers::objects::{encode, split_url};
use crate::memory::trace::Trace;

/// A REST backend. Bring your own service; this speaks to it.
#[derive(Debug)]
pub struct HttpMemory {
    pub base_url: String,
    http: HttpClient,
}

impl HttpMemory {
    pub fn new(base_url: &str) -> Self {
        Self { base_url: base_url.trim_end_matches('/').to_string(), http: HttpClient::new("http-memory", 30.0, 2) }
    }

    /// `https://api.example.com/memory`. A user and password in the URL are
    /// sent as basic auth (and left out of the base URL), as httpx does.
    pub fn from_url(url: &str) -> Result<Self> {
        let parts = split_url(url);
        if parts.host.is_empty() {
            return Err(Error::config(format!("HttpMemory needs a URL with a host (got '{url}')")));
        }
        let Some(user) = parts.user else { return Ok(Self::new(url)) };
        let (scheme, rest) = url.split_once("://").unwrap_or(("https", url));
        let rest = rest.split_once('@').map_or(rest, |(_, r)| r);
        let token = base64::engine::general_purpose::STANDARD.encode(format!("{user}:{}", parts.password.unwrap_or_default()));
        Ok(Self::new(&format!("{scheme}://{rest}")).header("authorization", &format!("Basic {token}")))
    }

    /// Sent with every request: a key, mostly. Never shown in an error.
    pub fn header(mut self, name: &str, value: &str) -> Self {
        self.http = self.http.header(name, value).secret(value);
        self
    }

    pub fn timeout(mut self, seconds: f64) -> Self {
        let (headers, secrets, retry) = (self.http.headers.clone(), self.http.secrets.clone(), self.http.retry.clone());
        self.http = HttpClient::new("http-memory", seconds, 0);
        (self.http.headers, self.http.secrets, self.http.retry) = (headers, secrets, retry);
        self
    }

    pub fn retries(mut self, retries: u32) -> Self {
        self.http.retry.max_retries = retries;
        self
    }

    /// One call. A 404 is `None`; anything else at or above 300 is an error
    /// that says what the service said.
    async fn request(&self, method: Method, path: &str, body: Option<&Value>) -> Result<Option<Value>> {
        let reply: Reply = self.http.json(method, &format!("{}{path}", self.base_url), body, &[404]).await?;
        if reply.status == 404 {
            return Ok(None);
        }
        Ok(Some(reply.body).filter(|b| b.as_object().is_none_or(|o| !o.is_empty())))
    }

    fn params(scope: Option<&str>, kind: Option<&str>, limit: Option<usize>, trace: Option<&Trace>) -> String {
        let mut p: Vec<(String, String)> = Vec::new();
        if let Some(s) = scope.filter(|s| !s.is_empty()) {
            p.push(("scope".into(), s.into()));
        }
        if let Some(k) = kind.filter(|k| !k.is_empty()) {
            p.push(("kind".into(), k.into()));
        }
        if let Some(n) = limit.filter(|n| *n > 0) {
            p.push(("limit".into(), n.to_string()));
        }
        if let Some(t) = trace {
            p.extend(t.filters().into_iter().map(|(f, v)| (f.to_string(), v)));
        }
        if p.is_empty() {
            return String::new();
        }
        format!("?{}", p.iter().map(|(k, v)| format!("{k}={}", encode(v, false))).collect::<Vec<_>>().join("&"))
    }
}

#[async_trait]
impl MemoryStore for HttpMemory {
    async fn append(&self, record: MemoryRecord) -> Result<MemoryRecord> {
        self.request(Method::POST, "/records", Some(&serde_json::to_value(&record).unwrap_or_default())).await?;
        Ok(record)
    }

    async fn all(&self, query: RecordQuery) -> Result<Vec<MemoryRecord>> {
        let path = format!("/records{}", Self::params(query.scope.as_deref(), query.kind.as_deref(), query.limit, query.trace.as_ref()));
        let Some(rows) = self.request(Method::GET, &path, None).await? else { return Ok(vec![]) };
        let rows = if rows.is_object() { rows["records"].clone() } else { rows };
        let mut out: Vec<MemoryRecord> = rows.as_array().into_iter().flatten().filter_map(|r| serde_json::from_value(r.clone()).ok()).collect();
        out.sort_by(|a, b| a.ts.total_cmp(&b.ts));
        Ok(out)
    }

    async fn clear(&self, scope: Option<&str>, trace: Option<&Trace>) -> Result<()> {
        self.request(Method::DELETE, &format!("/records{}", Self::params(scope, None, None, trace)), None).await?;
        Ok(())
    }

    async fn read_doc(&self, name: &str, trace: Option<&Trace>) -> Result<String> {
        let found = self.request(Method::GET, &format!("/docs/{}", encode(&doc_key(name, trace), false)), None).await?;
        Ok(match found {
            None => String::new(),
            Some(Value::Object(o)) => o.get("text").and_then(Value::as_str).unwrap_or("").to_string(),
            Some(Value::String(s)) => s,
            Some(other) => other.to_string(),
        })
    }

    async fn write_doc(&self, name: &str, text: &str, trace: Option<&Trace>) -> Result<()> {
        let namespace = trace.map_or_else(|| "_shared".to_string(), Trace::slug);
        let body = json!({"name": name, "text": text, "namespace": namespace});
        self.request(Method::PUT, &format!("/docs/{}", encode(&doc_key(name, trace), false)), Some(&body)).await?;
        Ok(())
    }

    fn describe(&self) -> String {
        format!("memory API at {}", self.base_url)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use axum::body::Bytes;
    use axum::http::{HeaderMap, Method as AxMethod, StatusCode, Uri};
    use axum::response::{IntoResponse, Response};

    use super::*;
    use crate::memory::providers::objects::decode;
    use crate::memory::providers::objects::testing::{query_of, serve};

    #[derive(Default)]
    struct State {
        records: Vec<Value>,
        docs: BTreeMap<String, String>,
        seen: Vec<String>,
        auth: Vec<String>,
    }

    async fn service(state: Arc<Mutex<State>>, method: AxMethod, uri: Uri, headers: HeaderMap, body: Bytes) -> Response {
        let mut s = state.lock().unwrap();
        s.seen.push(format!("{method} {}", uri.path()));
        s.auth.push(headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("").to_string());
        if uri.path() == "/boom/records" {
            return (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response();
        }
        if uri.path().ends_with("/records") {
            return match method {
                AxMethod::POST => {
                    s.records.push(serde_json::from_slice(&body).unwrap());
                    (StatusCode::CREATED, axum::Json(json!({}))).into_response()
                }
                AxMethod::DELETE => {
                    s.records.clear();
                    StatusCode::NO_CONTENT.into_response()
                }
                _ => {
                    let wanted = query_of(&uri);
                    let rows: Vec<Value> = s
                        .records
                        .iter()
                        .filter(|r| wanted.iter().filter(|(k, _)| k != "limit").all(|(k, v)| r[k].is_null() || r[k].as_str() == Some(v.as_str())))
                        .cloned()
                        .collect();
                    axum::Json(rows).into_response()
                }
            };
        }
        let key = decode(uri.path().rsplit('/').next().unwrap_or(""));
        if method == AxMethod::PUT {
            let text = serde_json::from_slice::<Value>(&body).unwrap()["text"].as_str().unwrap().to_string();
            s.docs.insert(key, text);
            return StatusCode::NO_CONTENT.into_response();
        }
        match s.docs.get(&key) {
            Some(t) => axum::Json(json!({"text": t})).into_response(),
            None => StatusCode::NOT_FOUND.into_response(),
        }
    }

    #[tokio::test]
    async fn the_http_backend_speaks_the_documented_contract() {
        let state = Arc::new(Mutex::new(State::default()));
        let shared = state.clone();
        let base = serve(axum::Router::new().fallback(move |m: AxMethod, u: Uri, h: HeaderMap, b: Bytes| service(shared.clone(), m, u, h, b))).await;
        let store = HttpMemory::from_url(&format!("{}/memory", base.replace("http://", "http://me:pw@"))).unwrap();
        let alice = Trace::user("alice");
        let mut r = MemoryRecord::new("user", "fact", "alice's fact");
        alice.stamp(&mut r);
        store.append(r).await.unwrap();
        assert_eq!(state.lock().unwrap().records[0]["user_id"], "alice");

        let rows = store.all(RecordQuery::scope("user").traced(Some(&alice))).await.unwrap();
        assert_eq!(rows.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(), ["alice's fact"]);
        assert!(state.lock().unwrap().seen.contains(&"GET /memory/records".to_string()));
        assert!(store.all(RecordQuery::default().traced(Some(&Trace::user("bob")))).await.unwrap().is_empty());

        store.write_doc("user.md", "- alice", Some(&alice)).await.unwrap();
        assert!(state.lock().unwrap().docs.contains_key("u-alice/user.md"));
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "- alice");
        assert_eq!(store.read_doc("missing.md", Some(&alice)).await.unwrap(), "");

        store.clear(None, Some(&alice)).await.unwrap();
        assert!(store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap().is_empty());
        let basic = format!("Basic {}", base64::engine::general_purpose::STANDARD.encode("me:pw"));
        assert!(state.lock().unwrap().auth.iter().all(|a| *a == basic));

        let broken = HttpMemory::from_url(&format!("{base}/boom")).unwrap().retries(0);
        let err = broken.append(MemoryRecord::default()).await.unwrap_err();
        assert!(err.to_string().contains("500"), "{err}");
    }
}
