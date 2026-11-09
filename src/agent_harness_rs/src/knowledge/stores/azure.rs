//! Azure AI Search.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use reqwest::Method;
use serde_json::{Map, Value, json};

use super::UrlParts;
use crate::errors::{Error, Result};
use crate::knowledge::base::{ALWAYS_FILTERABLE, Condition, HttpClient, Reply, VectorHit, VectorRecord, VectorStore, store_error};

const NAME: &str = "azure-ai-search";
/// The REST API version every request names.
pub const API_VERSION: &str = "2024-07-01";

const OPS: [(&str, &str); 6] = [("$eq", "eq"), ("$ne", "ne"), ("$gt", "gt"), ("$gte", "ge"), ("$lt", "lt"), ("$lte", "le")];
const TYPES: [(&str, &str); 10] = [
    ("string", "Edm.String"),
    ("str", "Edm.String"),
    ("int", "Edm.Int64"),
    ("integer", "Edm.Int64"),
    ("number", "Edm.Double"),
    ("float", "Edm.Double"),
    ("bool", "Edm.Boolean"),
    ("boolean", "Edm.Boolean"),
    ("strings", "Collection(Edm.String)"),
    ("list", "Collection(Edm.String)"),
];

/// Produces a Microsoft Entra ID access token, fresh enough to use.
pub type TokenFn = Arc<dyn Fn() -> BoxFuture<'static, Result<String>> + Send + Sync>;

/// How requests sign in: a key, a fixed Entra ID token, or a function that
/// returns one.
#[derive(Clone, Default)]
pub enum AzureAuth {
    /// `api-key: …`; the key from the options, else `AZURE_SEARCH_API_KEY`.
    #[default]
    Key,
    Token(String),
    TokenFn(TokenFn),
}

/// The options of an [`AzureAISearchStore`] — the same as the Python constructor's.
#[derive(Clone)]
pub struct AzureOptions {
    /// `knowledge` by default.
    pub index: String,
    /// Falls back to `AZURE_SEARCH_API_KEY`.
    pub api_key: Option<String>,
    /// Entra ID instead of a key.
    pub auth: AzureAuth,
    /// Metadata fields to filter on, and their kind: `string`, `int`,
    /// `number`, `bool` or `strings` (a list).
    pub filterable: IndexMap<String, String>,
    pub timeout: f64,
    pub retries: u32,
    pub headers: Vec<(String, String)>,
}

impl Default for AzureOptions {
    fn default() -> Self {
        Self { index: "knowledge".into(), api_key: None, auth: AzureAuth::Key, filterable: IndexMap::new(), timeout: 30.0, retries: 3, headers: vec![] }
    }
}

/// Azure AI Search.
///
/// ```no_run
/// # use agent_harness::knowledge::stores::azure::{AzureAISearchStore, AzureOptions};
/// let store = AzureAISearchStore::new("https://my-service.search.windows.net", AzureOptions {
///     index: "handbook".into(),
///     api_key: Some("…".into()),
///     filterable: [("source".to_string(), "string".to_string()), ("year".into(), "int".into())].into_iter().collect(),
///     ..Default::default()
/// }).unwrap();
/// ```
///
/// An Azure index has a fixed schema, so the metadata that is filtered on is
/// declared: `filterable` maps each field to `string`, `int`, `number`, `bool`
/// or `strings` (a list). All metadata is kept and returned either way.
///
/// `auth: AzureAuth::Token(..)` or `AzureAuth::TokenFn(..)` signs in with
/// Microsoft Entra ID instead of a key.
pub struct AzureAISearchStore {
    endpoint: String,
    index: String,
    api_key: String,
    auth: AzureAuth,
    filterable: IndexMap<String, String>,
    http: HttpClient,
}

impl std::fmt::Debug for AzureAISearchStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<AzureAISearchStore {} {}>", self.endpoint, self.index)
    }
}

/// A document key: letters, digits, dashes, underscores and equals signs.
pub fn key(record_id: &str) -> String {
    base64::engine::general_purpose::URL_SAFE.encode(record_id.as_bytes())
}

/// Python's `str()` of a JSON value.
fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

/// A value as OData writes it.
fn lit(v: &Value) -> String {
    match v {
        Value::Bool(b) => if *b { "true" } else { "false" }.into(),
        Value::Number(n) => n.to_string(),
        other => format!("'{}'", py_str(other).replace('\'', "''")),
    }
}

impl AzureAISearchStore {
    pub fn new(endpoint: &str, o: AzureOptions) -> Result<Self> {
        let mut filterable: IndexMap<String, String> = ALWAYS_FILTERABLE.iter().map(|k| (k.to_string(), "Edm.String".to_string())).collect();
        for (name, kind) in o.filterable {
            let Some((_, edm)) = TYPES.iter().find(|(t, _)| *t == kind.to_ascii_lowercase()) else {
                let mut kinds: Vec<&str> = TYPES.iter().map(|(t, _)| *t).collect();
                kinds.sort();
                return Err(store_error(NAME, format!("'{name}' is '{kind}'; a filterable field is one of {}", kinds.join(", "))));
            };
            filterable.insert(name, edm.to_string());
        }
        let api_key = o.api_key.filter(|k| !k.is_empty()).or_else(|| std::env::var("AZURE_SEARCH_API_KEY").ok()).unwrap_or_default();
        let mut http = HttpClient::new(NAME, o.timeout, o.retries).secret(api_key.clone());
        if let AzureAuth::Token(t) = &o.auth {
            http = http.secret(t.clone());
        }
        for (k, v) in o.headers {
            http = http.header(&k, v);
        }
        Ok(Self { endpoint: endpoint.trim_end_matches('/').to_string(), index: o.index, api_key, auth: o.auth, filterable, http })
    }

    /// As `vector_store("azure://my-service.search.windows.net/handbook?api_key=…")`
    /// builds it. `filterable=source:string,year:int` declares fields (an
    /// addition: Python cannot take a mapping in a URL).
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        if bare {
            return Err(Error::config("AzureAISearchStore needs an endpoint — azure://<service>.search.windows.net/<index>?api_key=…"));
        }
        let mut o = AzureOptions::default();
        if !p.path.is_empty() {
            o.index = p.path.clone();
        }
        for (k, value) in std::mem::take(&mut p.given) {
            match k.as_str() {
                "index" => o.index = value,
                "api_key" => o.api_key = Some(value),
                "token" => o.auth = AzureAuth::Token(value),
                "filterable" => {
                    for pair in value.split(',').filter(|s| !s.is_empty()) {
                        let (field, kind) = pair.split_once(':').unwrap_or((pair, "string"));
                        o.filterable.insert(field.trim().into(), kind.trim().into());
                    }
                }
                "timeout" => o.timeout = value.parse().map_err(|_| Error::config(format!("timeout is a number of seconds, not '{value}'")))?,
                "retries" => o.retries = value.parse().map_err(|_| Error::config(format!("retries is a whole number, not '{value}'")))?,
                other => return Err(Error::config(format!("AzureAISearchStore takes no option '{other}'"))),
            }
        }
        Self::new(&format!("https://{}", p.netloc), o)
    }

    /// The filterable fields and their Edm types.
    pub fn filterable_fields(&self) -> &IndexMap<String, String> {
        &self.filterable
    }

    fn url(&self, path: &str) -> String {
        format!("{}/indexes/{}{path}?api-version={API_VERSION}", self.endpoint, self.index)
    }

    async fn call(&self, method: Method, url: &str, body: Option<&Value>, ok: &[u16]) -> Result<Reply> {
        let auth = match &self.auth {
            AzureAuth::Key => ("api-key", self.api_key.clone()),
            AzureAuth::Token(t) => ("authorization", format!("Bearer {t}")),
            AzureAuth::TokenFn(f) => ("authorization", format!("Bearer {}", f().await?)),
        };
        let bytes = body.map(|b| serde_json::to_vec(b).unwrap_or_default());
        self.http.send(method, url, bytes, body.is_some().then_some("application/json"), &[auth], ok).await
    }

    /// A filter as OData — and, given ids, limited to their keys.
    pub fn odata(&self, wanted: &[Condition], ids: Option<&[String]>) -> Result<String> {
        let mut parts = vec![];
        if let Some(ids) = ids {
            parts.push(format!("search.in(id, '{}', ',')", ids.iter().map(|i| key(i)).collect::<Vec<_>>().join(",")));
        }
        for c in wanted {
            let Some(kind) = self.filterable.get(&c.field) else {
                let now: Vec<&str> = self.filterable.keys().map(String::as_str).collect();
                return Err(store_error(
                    NAME,
                    format!(
                        "'{0}' is not filterable in this Azure index — declare it: filterable={{'{0}': 'string'}}. Filterable now: {1}",
                        c.field,
                        if now.is_empty() { "nothing".to_string() } else { now.join(", ") }
                    ),
                ));
            };
            let field = format!("m_{}", c.field);
            if kind.starts_with("Collection") {
                let values = if c.op == "$in" { c.value.as_array().cloned().unwrap_or_default() } else { vec![c.value.clone()] };
                let test = values.iter().map(|v| format!("t eq {}", lit(v))).collect::<Vec<_>>().join(" or ");
                let clause = format!("{field}/any(t: {test})");
                parts.push(if c.op == "$ne" { format!("not {clause}") } else { clause });
            } else if c.op == "$in" {
                let any = c.value.as_array().into_iter().flatten().map(|v| format!("{field} eq {}", lit(v))).collect::<Vec<_>>().join(" or ");
                parts.push(format!("({any})"));
            } else {
                let op = OPS.iter().find(|(o, _)| *o == c.op).map_or("eq", |(_, a)| *a);
                parts.push(format!("{field} {op} {}", lit(&c.value)));
            }
        }
        Ok(parts.join(" and "))
    }

    async fn index_actions(&self, actions: Vec<Value>) -> Result<()> {
        let answer = self.call(Method::POST, &self.url("/docs/index"), Some(&json!({"value": actions})), &[]).await?;
        let failed: Vec<&Value> = answer
            .body
            .get("value")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|row| !row["status"].as_bool().unwrap_or(false) && row["statusCode"].as_u64() != Some(404))
            .collect();
        if let Some(first) = failed.first() {
            let shown: String = first.to_string().chars().take(300).collect();
            return Err(store_error(NAME, format!("azure refused {} records: {shown}", failed.len())));
        }
        Ok(())
    }

    fn row(&self, r: &VectorRecord) -> Value {
        let mut row = Map::new();
        row.insert("@search.action".into(), json!("mergeOrUpload"));
        row.insert("id".into(), json!(key(&r.id)));
        row.insert("ref".into(), json!(r.id));
        row.insert("text".into(), json!(r.text));
        row.insert("vector".into(), json!(r.vector));
        row.insert("meta".into(), json!(serde_json::to_string(&r.metadata).unwrap_or_else(|_| "{}".into())));
        for (name, kind) in &self.filterable {
            let value = r.metadata.get(name).cloned().unwrap_or(Value::Null);
            let value = if kind.starts_with("Collection") {
                let items = match value {
                    Value::Array(items) => items,
                    Value::Null => vec![],
                    other => vec![other],
                };
                Value::Array(items.iter().map(|v| Value::String(py_str(v))).collect())
            } else {
                value
            };
            row.insert(format!("m_{name}"), value);
        }
        Value::Object(row)
    }
}

#[async_trait]
impl VectorStore for AzureAISearchStore {
    fn name(&self) -> &str {
        NAME
    }

    fn batch_size(&self) -> usize {
        500
    }

    fn filterable(&self) -> Option<Vec<String>> {
        Some(self.filterable.keys().cloned().collect())
    }

    async fn ensure(&self, dimension: usize) -> Result<()> {
        if self.call(Method::GET, &self.url(""), None, &[404]).await?.status != 404 {
            return Ok(());
        }
        let mut fields = vec![
            json!({"name": "id", "type": "Edm.String", "key": true, "filterable": true}),
            json!({"name": "ref", "type": "Edm.String", "filterable": true}),
            json!({"name": "text", "type": "Edm.String", "searchable": true}),
            json!({"name": "meta", "type": "Edm.String"}),
            json!({"name": "vector", "type": "Collection(Edm.Single)", "searchable": true, "dimensions": dimension, "vectorSearchProfile": "default"}),
        ];
        fields.extend(self.filterable.iter().map(|(name, kind)| json!({"name": format!("m_{name}"), "type": kind, "filterable": true})));
        let body = json!({
            "name": self.index, "fields": fields,
            "vectorSearch": {
                "algorithms": [{"name": "hnsw", "kind": "hnsw", "hnswParameters": {"metric": "cosine"}}],
                "profiles": [{"name": "default", "algorithm": "hnsw"}]}});
        self.call(Method::PUT, &self.url(""), Some(&body), &[]).await?;
        Ok(())
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        let actions = records.iter().map(|r| self.row(r)).collect();
        self.index_actions(actions).await
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let mut body = json!({
            "select": "ref,text,meta", "top": k,
            "vectorQueries": [{"kind": "vector", "vector": vector, "fields": "vector", "k": k}]});
        let found = self.odata(wanted, None)?;
        if !found.is_empty() {
            body["filter"] = json!(found);
        }
        let answer = self.call(Method::POST, &self.url("/docs/search"), Some(&body), &[]).await?;
        let mut out = vec![];
        for row in answer.body.get("value").and_then(Value::as_array).into_iter().flatten() {
            let metadata = row["meta"].as_str().and_then(|m| serde_json::from_str::<Map<String, Value>>(m).ok()).unwrap_or_default();
            let score = row["@search.score"].as_f64().unwrap_or(0.0);
            out.push(VectorHit {
                id: row["ref"].as_str().unwrap_or_default().to_string(),
                // Azure scores cosine as 1 / (2 - cosine).
                score: if score != 0.0 { 2.0 - 1.0 / score } else { 0.0 },
                text: row["text"].as_str().unwrap_or_default().to_string(),
                metadata,
            });
        }
        Ok(out)
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        if let Some(ids) = ids
            && wanted.is_empty()
        {
            for part in ids.chunks(500) {
                self.index_actions(part.iter().map(|i| json!({"@search.action": "delete", "id": key(i)})).collect()).await?;
            }
            return Ok(());
        }
        let found = self.odata(wanted, ids)?;
        // There is no delete-by-filter: find what matches, then remove it by key.
        for _ in 0..1000 {
            let answer =
                self.call(Method::POST, &self.url("/docs/search"), Some(&json!({"select": "id", "top": 500, "filter": found, "search": "*"})), &[]).await?;
            let keys: Vec<String> =
                answer.body.get("value").and_then(Value::as_array).into_iter().flatten().filter_map(|r| r["id"].as_str().map(str::to_string)).collect();
            if keys.is_empty() {
                return Ok(());
            }
            let n = keys.len();
            self.index_actions(keys.into_iter().map(|k| json!({"@search.action": "delete", "id": k})).collect()).await?;
            if n < 500 {
                return Ok(());
            }
        }
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let answer = self.call(Method::GET, &self.url("/docs/$count"), None, &[]).await?;
        // A bare number — which may come back as text, behind a byte-order mark.
        Ok(match &answer.body {
            Value::Number(n) => n.as_u64().unwrap_or(0) as usize,
            other => other["text"].as_str().map(|t| t.trim_start_matches('\u{feff}').trim()).and_then(|t| t.parse().ok()).unwrap_or(0),
        })
    }

    async fn clear(&self) -> Result<()> {
        self.call(Method::DELETE, &self.url(""), None, &[404]).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::knowledge::base::conditions;

    type Seen = Arc<Mutex<Vec<(String, String, reqwest::header::HeaderMap, String)>>>;
    type Answer = Arc<dyn Fn(&str, &str, &str) -> (u16, String) + Send + Sync>;

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

    fn store(endpoint: &str) -> AzureAISearchStore {
        let filterable = [("source".to_string(), "string".to_string()), ("year".into(), "int".into()), ("tags".into(), "strings".into())].into_iter().collect();
        AzureAISearchStore::new(
            endpoint,
            AzureOptions { index: "handbook".into(), api_key: Some("azure-key".into()), filterable, retries: 0, ..Default::default() },
        )
        .unwrap()
    }

    #[test]
    fn keys_and_odata_match_python() {
        // base64.urlsafe_b64encode(b"doc#0?").decode()
        assert_eq!(key("doc#0?"), "ZG9jIzA_");
        assert_eq!(key("a"), "YQ==");
        let s = store("https://x");
        let c = |field: &str, op: &str, value: Value| Condition { field: field.into(), op: op.into(), value };
        let wanted = vec![
            c("source", "$eq", json!("it's")),
            c("year", "$gte", json!(2024)),
            c("year", "$ne", json!(2025)),
            c("tags", "$in", json!(["a", "b"])),
            c("doc", "$in", json!(["d1", "d2"])),
        ];
        assert_eq!(
            s.odata(&wanted, Some(&["a".to_string()])).unwrap(),
            "search.in(id, 'YQ==', ',') and m_source eq 'it''s' and m_year ge 2024 and m_year ne 2025 and m_tags/any(t: t eq 'a' or t eq 'b') and (m_doc eq 'd1' or m_doc eq 'd2')"
        );
        let ne = vec![c("tags", "$ne", json!("x")), c("source", "$eq", json!(true))];
        assert_eq!(s.odata(&ne, None).unwrap(), "not m_tags/any(t: t eq 'x') and m_source eq true");
        let e = s.odata(&conditions(Some(&json!({"nope": 1}))).unwrap(), None).unwrap_err();
        assert!(e.to_string().contains("'nope' is not filterable"));
        assert!(
            AzureAISearchStore::new(
                "https://x",
                AzureOptions { filterable: [("a".to_string(), "date".to_string())].into_iter().collect(), ..Default::default() }
            )
            .is_err()
        );
        assert_eq!(s.filterable().unwrap().len(), ALWAYS_FILTERABLE.len() + 3);
    }

    #[test]
    fn urls_are_read_like_python() {
        let s = AzureAISearchStore::from_url(UrlParts::parse("azure://svc.search.windows.net/hand?api_key=k1234&filterable=year:int").unwrap(), false).unwrap();
        assert_eq!((s.endpoint.as_str(), s.index.as_str(), s.api_key.as_str()), ("https://svc.search.windows.net", "hand", "k1234"));
        assert_eq!(s.filterable["year"], "Edm.Int64");
        assert!(AzureAISearchStore::from_url(UrlParts { scheme: "azure".into(), ..Default::default() }, true).is_err());
    }

    #[tokio::test]
    async fn requests_have_the_python_shape() {
        let (url, seen) = fake(Arc::new(|method, path, body| match (method, path.split('?').next().unwrap()) {
            ("GET", "/indexes/handbook") => (404, "{}".into()),
            ("POST", "/indexes/handbook/docs/index") => (200, r#"{"value": [{"key": "YQ==", "status": true, "statusCode": 200}]}"#.into()),
            ("POST", "/indexes/handbook/docs/search") if body.contains("\"select\":\"id\"") => (200, r#"{"value": [{"id": "YQ=="}]}"#.into()),
            ("POST", "/indexes/handbook/docs/search") => {
                (200, r#"{"value": [{"ref": "a", "text": "hello", "meta": "{\"year\": 2024, \"tags\": [\"x\"]}", "@search.score": 1.0}]}"#.into())
            }
            ("GET", "/indexes/handbook/docs/$count") => (200, "\u{feff}12".into()),
            _ => (204, String::new()),
        }))
        .await;
        let s = store(&url);
        s.ensure(3).await.unwrap();
        s.upsert(vec![VectorRecord::new("a", vec![1.0, 0.0, 0.0], "hello", json!({"year": 2024, "tags": "x", "flag": true}))]).await.unwrap();
        let hits = s.query(&[1.0, 0.0, 0.0], 4, Some(&json!({"year": {"$gte": 2024}}))).await.unwrap();
        assert_eq!((hits[0].id.as_str(), hits[0].score, hits[0].metadata["year"].as_i64()), ("a", 1.0, Some(2024)));
        s.delete(None, Some(&json!({"year": 2024}))).await.unwrap();
        assert_eq!(s.count().await.unwrap(), 12);
        s.clear().await.unwrap();

        let seen = seen.lock().unwrap();
        assert!(seen.iter().all(|r| r.1.ends_with("api-version=2024-07-01") && r.2["api-key"] == "azure-key"));
        let index: Value = serde_json::from_str(&seen[1].3).unwrap();
        assert_eq!(seen[1].0, "PUT");
        assert_eq!(
            index["fields"][4],
            json!({"name": "vector", "type": "Collection(Edm.Single)", "searchable": true, "dimensions": 3, "vectorSearchProfile": "default"})
        );
        assert_eq!(index["fields"][5], json!({"name": "m_doc", "type": "Edm.String", "filterable": true}));
        assert_eq!(index["vectorSearch"]["algorithms"][0]["hnswParameters"], json!({"metric": "cosine"}));
        let up: Value = serde_json::from_str(&seen[2].3).unwrap();
        let row = &up["value"][0];
        assert_eq!(row["@search.action"], "mergeOrUpload");
        assert_eq!((row["id"].as_str(), row["ref"].as_str()), (Some("YQ=="), Some("a")));
        assert_eq!(row["m_tags"], json!(["x"]));
        assert_eq!(row["m_year"], 2024);
        assert_eq!(row["m_source"], Value::Null);
        let meta: Value = serde_json::from_str(row["meta"].as_str().unwrap()).unwrap();
        assert_eq!(meta, json!({"year": 2024, "tags": "x", "flag": true}));
        let q: Value = serde_json::from_str(&seen[3].3).unwrap();
        assert_eq!(
            q,
            json!({"select": "ref,text,meta", "top": 4, "vectorQueries": [{"kind": "vector", "vector": [1.0, 0.0, 0.0], "fields": "vector", "k": 4}], "filter": "m_year ge 2024"})
        );
        let find: Value = serde_json::from_str(&seen[4].3).unwrap();
        assert_eq!(find, json!({"select": "id", "top": 500, "filter": "m_year eq 2024", "search": "*"}));
        let gone: Value = serde_json::from_str(&seen[5].3).unwrap();
        assert_eq!(gone, json!({"value": [{"@search.action": "delete", "id": "YQ=="}]}));
        assert_eq!(seen.last().unwrap().0, "DELETE");
    }

    #[tokio::test]
    async fn tokens_and_refusals() {
        let (url, seen) =
            fake(Arc::new(|_, _, _| (200, r#"{"value": [{"key": "x", "status": false, "statusCode": 400, "errorMessage": "bad"}]}"#.into()))).await;
        let f: TokenFn = Arc::new(|| Box::pin(async { Ok("entra-token".to_string()) }));
        let s = AzureAISearchStore::new(&url, AzureOptions { auth: AzureAuth::TokenFn(f), retries: 0, ..Default::default() }).unwrap();
        let e = s.upsert(vec![VectorRecord::new("x", vec![1.0], "", json!({}))]).await.unwrap_err();
        assert!(e.to_string().starts_with("azure refused 1 records"), "{e}");
        assert_eq!(seen.lock().unwrap()[0].2["authorization"], "Bearer entra-token");
    }

    #[tokio::test]
    async fn live() {
        let Ok(url) = std::env::var("AH_LIVE_AZURE_SEARCH_URL") else { return };
        let store = super::super::vector_store(&url).unwrap();
        let report = store.check(8).await;
        assert_eq!(report["ok"], true, "{report:#}");
    }
}
