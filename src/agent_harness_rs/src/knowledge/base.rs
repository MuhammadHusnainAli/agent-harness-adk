//! One interface over every vector database.
//!
//! A vector store keeps records — an id, a vector, the text it came from, some
//! metadata — and finds the ones nearest a query vector. Every store does
//! those four things the same way:
//!
//! ```ignore
//! store.ensure(1536).await?;
//! store.upsert(vec![VectorRecord::new("a", vector, "the text", json!({"source": "faq.md"}))]).await?;
//! let hits = store.query(&vector, 5, Some(&json!({"source": "faq.md"}))).await?;
//! store.delete(None, Some(&json!({"source": "faq.md"}))).await?;
//! ```
//!
//! **Scores** are cosine similarity, higher is nearer, whatever the database
//! reports natively — so a threshold means the same thing on every backend.
//!
//! **Filters** are one small language, translated for each database:
//! `{"source": "faq.md"}` (equals), `{"lang": {"$in": ["en", "fr"]}}` (any of),
//! `{"year": {"$gte": 2024}, "draft": {"$ne": true}}` (ranges, not-equal;
//! keys are ANDed), and `$and` at the top level.
//!
//! **Ids** are yours: any string. A database that only takes UUIDs or integers
//! is given one derived from yours, and yours is what comes back.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use reqwest::Method;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::errors::{Error, Result};
use crate::llm_providers::resilience::{RetryPolicy, retry_after};

/// The operators a filter may use on a field.
pub const OPS: [&str; 7] = ["$eq", "$ne", "$in", "$gt", "$gte", "$lt", "$lte"];
/// What `KnowledgeBase` and `SemanticMemory` filter on. A store that has to be
/// told which fields are filterable is told these without being asked.
pub const ALWAYS_FILTERABLE: [&str; 6] = ["doc", "namespace", "scope", "user_id", "tenant_id", "session_id"];
/// The namespace ids are derived in, for stores that will not take a string.
const NAMESPACE: uuid::Uuid = uuid::Uuid::from_u128(0x6f1d6c1e_1a4e_4c0b_9a51_7c0f3a2b9d10);

/// A vector database refused something, or could not be reached.
pub fn store_error(store: &str, message: impl Into<String>) -> Error {
    Error::VectorStore { message: message.into(), store: store.into(), status: None, retryable: false }
}

/// One thing to remember: where it sits, what it says, what it is about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub text: String,
    /// Flat: strings, numbers, booleans, and lists of strings.
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl VectorRecord {
    /// `metadata` is a JSON object; anything else is taken as none.
    pub fn new(id: impl Into<String>, vector: Vec<f32>, text: impl Into<String>, metadata: Value) -> Self {
        let metadata = match metadata {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        Self { id: id.into(), vector, text: text.into(), metadata }
    }
}

/// One record a query found.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorHit {
    pub id: String,
    /// Cosine similarity: 1.0 is the same direction, 0 is unrelated.
    pub score: f64,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

/// The same direction, length one — so a dot product is a cosine.
pub fn normalise(vector: &[f32]) -> Vec<f32> {
    let length = vector.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if length == 0.0 { vector.to_vec() } else { vector.iter().map(|x| (*x as f64 / length) as f32).collect() }
}

/// One condition of a filter: `field op value`.
#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    pub field: String,
    pub op: String,
    pub value: Value,
}

/// A filter as a list of conditions, all of which must hold.
pub fn conditions(filter: Option<&Value>) -> Result<Vec<Condition>> {
    let mut out = Vec::new();
    let Some(filter) = filter else { return Ok(out) };
    let Some(map) = filter.as_object() else {
        if filter.is_null() {
            return Ok(out);
        }
        return Err(store_error("", "a filter is an object of field → condition"));
    };
    for (key, value) in map {
        if key == "$and" {
            for part in value.as_array().into_iter().flatten() {
                out.extend(conditions(Some(part))?);
            }
            continue;
        }
        if key.starts_with('$') {
            return Err(store_error("", format!("filters take $and at the top level, and {} on a field — not '{key}'", OPS.join(", "))));
        }
        match value {
            Value::Object(ops) => {
                for (op, operand) in ops {
                    if !OPS.contains(&op.as_str()) {
                        return Err(store_error("", format!("unknown filter operator '{op}' on '{key}'; there are: {}", OPS.join(", "))));
                    }
                    if op == "$in" && !operand.is_array() {
                        return Err(store_error("", format!("$in on '{key}' takes a list")));
                    }
                    out.push(Condition { field: key.clone(), op: op.clone(), value: operand.clone() });
                }
            }
            other => out.push(Condition { field: key.clone(), op: "$eq".into(), value: other.clone() }),
        }
    }
    Ok(out)
}

fn same(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) if !a.is_boolean() && !b.is_boolean() => x == y,
        _ => a == b,
    }
}

fn order(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::String(x), Value::String(y)) => Some(x.cmp(y)),
        (Value::Bool(_), _) | (_, Value::Bool(_)) => None,
        _ => a.as_f64()?.partial_cmp(&b.as_f64()?),
    }
}

/// Does this metadata satisfy every condition? For stores that filter here.
pub fn holds(metadata: &Map<String, Value>, wanted: &[Condition]) -> bool {
    for c in wanted {
        let found = metadata.get(&c.field).unwrap_or(&Value::Null);
        let ok = match c.op.as_str() {
            "$eq" => match found {
                Value::Array(items) => items.iter().any(|i| same(i, &c.value)),
                _ => same(found, &c.value),
            },
            "$ne" => match found {
                Value::Array(items) => !items.iter().any(|i| same(i, &c.value)),
                _ => !same(found, &c.value),
            },
            "$in" => {
                let options = c.value.as_array().cloned().unwrap_or_default();
                match found {
                    Value::Array(items) => items.iter().any(|i| options.iter().any(|o| same(i, o))),
                    _ => options.iter().any(|o| same(found, o)),
                }
            }
            op => {
                !found.is_null()
                    && order(found, &c.value).is_some_and(|o| match op {
                        "$gt" => o.is_gt(),
                        "$gte" => o.is_ge(),
                        "$lt" => o.is_lt(),
                        _ => o.is_le(),
                    })
            }
        };
        if !ok {
            return false;
        }
    }
    true
}

/// A UUID for a store that insists on one: the id itself if it is one already,
/// otherwise one derived from it — the same every time, and the same as Python's.
pub fn as_uuid(record_id: &str) -> String {
    match uuid::Uuid::parse_str(record_id) {
        Ok(u) => u.hyphenated().to_string(),
        Err(_) => uuid::Uuid::new_v5(&NAMESPACE, record_id.as_bytes()).hyphenated().to_string(),
    }
}

/// A value as it is written inside a filter expression.
pub fn literal(value: &Value) -> String {
    match value {
        Value::Bool(b) => if *b { "true" } else { "false" }.into(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap_or_default(),
        other => serde_json::to_string(&other.to_string()).unwrap_or_default(),
    }
}

/// A vector database. Five methods make one; the rest come with the trait.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// What this store is called in errors and listings.
    fn name(&self) -> &str;
    /// The most records one upsert request carries.
    fn batch_size(&self) -> usize {
        100
    }
    /// The most results one query may ask for.
    fn max_k(&self) -> usize {
        1000
    }
    /// The metadata fields this store can filter on, when it must be told.
    fn filterable(&self) -> Option<Vec<String>> {
        None
    }

    /// Create the collection if it is not there. Safe to call every time.
    async fn ensure(&self, dimension: usize) -> Result<()>;
    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()>;
    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>>;
    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()>;
    /// How many records it holds.
    async fn count(&self) -> Result<usize>;

    /// Remove every record. Stores that can do it cheaply override this.
    async fn clear(&self) -> Result<()> {
        Err(store_error(self.name(), format!("{} cannot be emptied from here — drop the collection in the database", self.name())))
    }

    /// Let go of connections.
    async fn close(&self) {}

    // ---- what callers use ---------------------------------------------------
    /// Add records, replacing any that have the same id. Returns how many.
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<usize> {
        for row in &records {
            if row.id.is_empty() {
                return Err(store_error(self.name(), "every record needs an id that is a string"));
            }
            if row.vector.is_empty() || row.vector.iter().any(|x| !x.is_finite()) {
                return Err(store_error(
                    self.name(),
                    format!("record '{}' has no usable vector: it is empty, or holds something that is not a finite number", row.id),
                ));
            }
        }
        let n = records.len();
        let size = self.batch_size().max(1);
        let mut rows = records;
        while !rows.is_empty() {
            let rest = rows.split_off(size.min(rows.len()));
            self.upsert_batch(rows).await?;
            rows = rest;
        }
        Ok(n)
    }

    /// The `k` records nearest `vector`, nearest first.
    async fn query(&self, vector: &[f32], k: usize, filter: Option<&Value>) -> Result<Vec<VectorHit>> {
        if vector.is_empty() {
            return Err(store_error(self.name(), "a query needs a vector"));
        }
        let wanted = conditions(filter).map_err(|e| renamed(e, self.name()))?;
        let mut hits = self.query_raw(vector, k.clamp(1, self.max_k()), &wanted).await?;
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(k.max(1));
        Ok(hits)
    }

    /// Remove records: these ids, or everything a filter matches.
    async fn delete(&self, ids: Option<Vec<String>>, filter: Option<&Value>) -> Result<()> {
        let wanted = conditions(filter).map_err(|e| renamed(e, self.name()))?;
        let listed: Option<Vec<String>> = ids.map(|ids| ids.into_iter().filter(|i| !i.is_empty()).collect());
        match &listed {
            None if wanted.is_empty() => Err(store_error(self.name(), "delete takes ids or a filter — to empty the store, use clear()")),
            Some(l) if l.is_empty() && wanted.is_empty() => Ok(()),
            _ => self.delete_raw(listed.as_deref(), &wanted).await,
        }
    }

    /// Prove the store works: create, upsert, find, filter, delete.
    ///
    /// Writes three records under ids beginning `__check__` and removes them.
    /// Never fails: each step's outcome is in the answer.
    async fn check(&self, dimension: usize) -> Value {
        check_store(self, dimension).await
    }
}

fn renamed(e: Error, store: &str) -> Error {
    match e {
        Error::VectorStore { message, status, retryable, .. } => Error::VectorStore { message, store: store.into(), status, retryable },
        other => other,
    }
}

async fn check_store<S: VectorStore + ?Sized>(store: &S, dimension: usize) -> Value {
    let dimension = dimension.max(3);
    let prefix = format!("__check__{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
    let rows: Vec<VectorRecord> = ["a", "b", "b"]
        .iter()
        .enumerate()
        .map(|(n, k)| {
            let v: Vec<f32> = (0..dimension).map(|i| if i == n { 1.0 } else { 0.05 }).collect();
            VectorRecord::new(format!("{prefix}-{n}"), normalise(&v), format!("check {n}"), json!({"doc": prefix, "n": n, "scope": k}))
        })
        .collect();
    let numbers = store.filterable().is_none_or(|f| f.iter().any(|x| x == "n"));
    let mut steps: Vec<Value> = Vec::new();
    let doc = json!({"doc": prefix});
    let ids = |hits: &[VectorHit]| -> Vec<String> {
        let mut v: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
        v.sort();
        v
    };
    let pair = vec![rows[1].id.clone(), rows[2].id.clone()];

    macro_rules! step {
        ($name:expr, $body:expr) => {{
            let started = Instant::now();
            let outcome: Result<String> = async { $body }.await;
            let ok = outcome.is_ok();
            let detail = match outcome {
                Ok(d) => d,
                Err(e) => format!("{}: {e}", e.kind_name()).chars().take(300).collect(),
            };
            steps.push(json!({"step": $name, "ok": ok, "detail": detail, "seconds": started.elapsed().as_secs_f64()}));
            ok
        }};
    }

    let mut ok = step!("ensure", store.ensure(dimension).await.map(|_| String::new()));
    ok = ok && step!("upsert", store.upsert(rows.clone()).await.map(|_| String::new()));
    ok = ok
        && step!("query", {
            let mut last = vec![];
            for _ in 0..40 {
                last = store.query(&rows[0].vector, 3, Some(&doc)).await?;
                if last.first().is_some_and(|h| h.id == rows[0].id) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            let Some(top) = last.first().filter(|h| h.id == rows[0].id) else {
                return Err(store_error(store.name(), format!("{} was not the nearest record: {:?}", rows[0].id, ids(&last))));
            };
            if !(0.98..=1.001).contains(&top.score) {
                return Err(store_error(store.name(), format!("an identical vector scored {:.4}, not 1.0 — the score is not cosine similarity", top.score)));
            }
            if top.text != "check 0" || top.metadata.get("n").and_then(Value::as_f64) != Some(0.0) {
                return Err(store_error(store.name(), format!("text or metadata did not come back: {top:?}")));
            }
            Ok(format!("score {:.4}", top.score))
        });
    ok = ok
        && step!("filter", {
            let hits = store.query(&rows[0].vector, 3, Some(&json!({"doc": prefix, "scope": "b"}))).await?;
            if ids(&hits) != pair {
                return Err(store_error(store.name(), format!("the filter returned {:?}", ids(&hits))));
            }
            let either = store.query(&rows[0].vector, 3, Some(&json!({"doc": prefix, "scope": {"$in": ["b", "c"]}}))).await?;
            if ids(&either) != pair {
                return Err(store_error(store.name(), format!("$in returned {:?}", ids(&either))));
            }
            if !numbers {
                return Ok("equals, $in".into());
            }
            let ranged = store.query(&rows[0].vector, 3, Some(&json!({"doc": prefix, "n": {"$gte": 1}}))).await?;
            if ids(&ranged) != pair {
                return Err(store_error(store.name(), format!("$gte returned {:?}", ids(&ranged))));
            }
            Ok("equals, $in, $gte".into())
        });
    ok = ok
        && step!("replace", {
            store
                .upsert(vec![VectorRecord::new(rows[1].id.clone(), rows[1].vector.clone(), "check 1 again", json!({"doc": prefix, "n": 1, "scope": "c"}))])
                .await?;
            let mut hits = vec![];
            for _ in 0..40 {
                hits = store.query(&rows[1].vector, 3, Some(&doc)).await?;
                if hits.len() == 3 && hits[0].id == rows[1].id && hits[0].text == "check 1 again" {
                    return Ok("same id, new content, still three records".into());
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            Err(store_error(store.name(), format!("upserting an existing id left {:?}", ids(&hits))))
        });
    ok = ok
        && step!("delete by id", {
            store.delete(Some(vec![rows[0].id.clone()]), None).await?;
            let mut left = vec![];
            for _ in 0..40 {
                left = store.query(&rows[0].vector, 3, Some(&doc)).await?;
                if !left.iter().any(|h| h.id == rows[0].id) {
                    return Ok(format!("{} left", left.len()));
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            Err(store_error(store.name(), format!("still there after deleting by id: {:?}", ids(&left))))
        });
    if ok {
        ok = step!("delete by filter", {
            store.delete(None, Some(&doc)).await?;
            let mut left = vec![];
            for _ in 0..40 {
                left = store.query(&rows[0].vector, 3, Some(&doc)).await?;
                if left.is_empty() {
                    return Ok("none left".into());
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            Err(store_error(store.name(), format!("still there after deleting by filter: {:?}", ids(&left))))
        });
    } else {
        // Leave nothing behind either way.
        let _ = store.delete(Some(rows.iter().map(|r| r.id.clone()).collect()), None).await;
    }
    json!({"store": store.name(), "ok": ok, "steps": steps})
}

/// Signs a request just before it is sent: `(method, url, body, headers)`.
pub type Signer = Arc<dyn Fn(&str, &str, &[u8], &mut HeaderMap) -> Result<()> + Send + Sync>;

/// What a request came back with: the status and the JSON body (`{"text": ..}`
/// when the body is not JSON, `{}` when it is empty).
#[derive(Debug, Clone)]
pub struct Reply {
    pub status: u16,
    pub body: Value,
}

/// A vector database spoken to over HTTP: the client, the retries, the errors.
///
/// A call that may succeed in a moment — 429, 5xx, a timeout, a dropped
/// connection — is tried again with back-off. Anything else is an
/// `Error::VectorStore` that says what the database said, with keys removed.
#[derive(Clone)]
pub struct HttpClient {
    pub name: String,
    pub client: reqwest::Client,
    pub retry: RetryPolicy,
    /// Sent with every request: the key, mostly.
    pub headers: Vec<(String, String)>,
    /// Never shown in an error.
    pub secrets: Vec<String>,
    pub signer: Option<Signer>,
}

impl std::fmt::Debug for HttpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<HttpClient {}>", self.name)
    }
}

impl HttpClient {
    pub fn new(name: &str, timeout: f64, retries: u32) -> Self {
        crate::llm_providers::base::install_tls();
        let client = reqwest::Client::builder().timeout(Duration::from_secs_f64(timeout.max(0.1))).pool_max_idle_per_host(16).build().unwrap_or_default();
        Self {
            name: name.into(),
            client,
            retry: RetryPolicy { max_retries: retries, initial_delay: 0.4, max_delay: 8.0, max_retry_after: 20.0, ..RetryPolicy::default() },
            headers: vec![],
            secrets: vec![],
            signer: None,
        }
    }

    pub fn header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    pub fn secret(mut self, secret: impl Into<String>) -> Self {
        let s = secret.into();
        if !s.is_empty() {
            self.secrets.push(s);
        }
        self
    }

    pub fn redact(&self, text: &str) -> String {
        let mut out = text.to_string();
        for s in &self.secrets {
            if s.len() >= 4 {
                out = out.replace(s.as_str(), "***");
            }
        }
        out
    }

    /// A JSON request. `ok` lists statuses besides 2xx that are an answer
    /// rather than a failure — a 404 when asking whether something exists.
    pub async fn json(&self, method: Method, url: &str, body: Option<&Value>, ok: &[u16]) -> Result<Reply> {
        let bytes = body.map(|b| serde_json::to_vec(b).unwrap_or_default());
        self.send(method, url, bytes, body.is_some().then_some("application/json"), &[], ok).await
    }

    /// Any request: raw bytes, extra headers.
    pub async fn send(
        &self,
        method: Method,
        url: &str,
        body: Option<Vec<u8>>,
        content_type: Option<&str>,
        extra: &[(&str, String)],
        ok: &[u16],
    ) -> Result<Reply> {
        let body = body.unwrap_or_default();
        let started = Instant::now();
        let mut attempt = 0u32;
        loop {
            let mut headers = HeaderMap::new();
            for (k, v) in self.headers.iter().map(|(k, v)| (k.as_str(), v.clone())).chain(extra.iter().map(|(k, v)| (*k, v.clone()))) {
                if let (Ok(k), Ok(v)) = (HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(&v)) {
                    headers.insert(k, v);
                }
            }
            if let Some(ct) = content_type
                && !headers.contains_key("content-type")
            {
                headers.insert("content-type", HeaderValue::from_str(ct).unwrap_or(HeaderValue::from_static("application/json")));
            }
            if let Some(sign) = &self.signer {
                sign(method.as_str(), url, &body, &mut headers)?;
            }
            let mut request = self.client.request(method.clone(), url).headers(headers);
            if !body.is_empty() {
                request = request.body(body.clone());
            }
            let (problem, wait) = match request.send().await {
                Ok(response) => {
                    let status = response.status().as_u16();
                    let response_headers = response.headers().clone();
                    let text = response.text().await.unwrap_or_default();
                    if (200..300).contains(&status) || ok.contains(&status) {
                        let body = if text.trim().is_empty() { json!({}) } else { serde_json::from_str(&text).unwrap_or_else(|_| json!({"text": text})) };
                        return Ok(Reply { status, body });
                    }
                    let squeezed: String = text.chars().take(800).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
                    let detail: String = self.redact(&squeezed).chars().take(400).collect();
                    let again = matches!(status, 408 | 425 | 429) || status >= 500;
                    let path = url.split('?').next().unwrap_or(url);
                    let problem = Error::VectorStore {
                        message: format!("{}: {} {} → {status}: {detail}", self.name, method, self.redact(path)),
                        store: self.name.clone(),
                        status: Some(status),
                        retryable: again,
                    };
                    (problem, retry_after(Some(&response_headers), None, status == 429))
                }
                Err(e) => {
                    // Only the kind: the message of a transport error can carry the URL.
                    let kind = if e.is_timeout() {
                        "TimeoutError"
                    } else if e.is_connect() {
                        "ConnectError"
                    } else {
                        "TransportError"
                    };
                    (
                        Error::VectorStore {
                            message: format!("{} could not be reached ({kind})", self.name),
                            store: self.name.clone(),
                            status: None,
                            retryable: true,
                        },
                        None,
                    )
                }
            };
            let retryable = matches!(problem, Error::VectorStore { retryable: true, .. });
            if !retryable || attempt >= self.retry.max_retries {
                return Err(problem);
            }
            let delay = wait.unwrap_or_else(|| self.retry.backoff(attempt));
            if delay > self.retry.max_retry_after || self.retry.max_elapsed.is_some_and(|m| started.elapsed().as_secs_f64() + delay > m) {
                return Err(problem);
            }
            tokio::time::sleep(Duration::from_secs_f64(delay)).await;
            attempt += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_parse_and_hold() {
        let c = conditions(Some(&json!({"a": 1, "b": {"$in": ["x", "y"]}, "$and": [{"c": {"$gte": 2}}]}))).unwrap();
        assert_eq!(c.len(), 3);
        let meta = json!({"a": 1.0, "b": ["y", "z"], "c": 3}).as_object().unwrap().clone();
        assert!(holds(&meta, &c));
        let meta = json!({"a": 1, "b": "q", "c": 3}).as_object().unwrap().clone();
        assert!(!holds(&meta, &c));
        assert!(conditions(Some(&json!({"a": {"$like": 1}}))).is_err());
        assert!(conditions(Some(&json!({"$or": []}))).is_err());
        assert!(conditions(Some(&json!({"a": {"$in": 1}}))).is_err());
        let ne = conditions(Some(&json!({"d": {"$ne": true}}))).unwrap();
        assert!(holds(&Map::new(), &ne), "a missing field is not equal to anything");
        let gt = conditions(Some(&json!({"d": {"$gt": 1}}))).unwrap();
        assert!(!holds(&Map::new(), &gt));
    }

    #[test]
    fn ids_and_literals_match_python() {
        // uuid.uuid5(UUID("6f1d6c1e-1a4e-4c0b-9a51-7c0f3a2b9d10"), "doc#0")
        assert_eq!(as_uuid("doc#0"), "7303403b-8700-5186-8d0e-6f26146cb95a");
        let u = "6f1d6c1e-1a4e-4c0b-9a51-7c0f3a2b9d10";
        assert_eq!(as_uuid(u), u);
        assert_eq!(literal(&json!(true)), "true");
        assert_eq!(literal(&json!("a\"b")), "\"a\\\"b\"");
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!((cosine(&normalise(&[3.0, 4.0]), &[0.6, 0.8]) - 1.0).abs() < 1e-6);
    }
}
