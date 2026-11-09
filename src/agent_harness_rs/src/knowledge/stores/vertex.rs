//! Vertex AI Vector Search.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use reqwest::Method;
use serde_json::{Map, Value, json};

use super::UrlParts;
use crate::errors::{Error, Result};
use crate::knowledge::base::{Condition, HttpClient, Reply, VectorHit, VectorRecord, VectorStore, holds, normalise, store_error};

const NAME: &str = "vertex-vector-search";

const NUMERIC: [(&str, &str); 6] =
    [("$eq", "EQUAL"), ("$ne", "NOT_EQUAL"), ("$gt", "GREATER"), ("$gte", "GREATER_EQUAL"), ("$lt", "LESS"), ("$lte", "LESS_EQUAL")];

/// What a record says, by id: `(text, metadata)`.
pub type Held = HashMap<String, (String, Map<String, Value>)>;

/// Where the text and metadata of each record live.
///
/// Vertex AI Vector Search keeps vectors and the values they are filtered on,
/// and nothing else — so what a record *says* has to live somewhere.
/// [`SqlitePayloads`] is the default place. For several replicas, implement
/// this over a database they share.
#[async_trait]
pub trait Payloads: Send + Sync {
    async fn put(&self, rows: Vec<(String, String, Map<String, Value>)>) -> Result<()>;
    async fn get(&self, ids: &[String]) -> Result<Held>;
    async fn delete(&self, ids: &[String]) -> Result<()>;
    async fn all(&self) -> Result<Held>;
}

/// Payloads in this process, gone with it. For tests.
type HeldPayloads = IndexMap<String, (String, Map<String, Value>)>;
#[derive(Debug, Default)]
pub struct MemoryPayloads(Mutex<HeldPayloads>);

#[async_trait]
impl Payloads for MemoryPayloads {
    async fn put(&self, rows: Vec<(String, String, Map<String, Value>)>) -> Result<()> {
        let mut held = self.0.lock().unwrap_or_else(|p| p.into_inner());
        for (id, text, meta) in rows {
            held.insert(id, (text, meta));
        }
        Ok(())
    }

    async fn get(&self, ids: &[String]) -> Result<Held> {
        let held = self.0.lock().unwrap_or_else(|p| p.into_inner());
        Ok(ids.iter().filter_map(|i| held.get(i).map(|v| (i.clone(), v.clone()))).collect())
    }

    async fn delete(&self, ids: &[String]) -> Result<()> {
        let mut held = self.0.lock().unwrap_or_else(|p| p.into_inner());
        for i in ids {
            held.shift_remove(i);
        }
        Ok(())
    }

    async fn all(&self) -> Result<Held> {
        Ok(self.0.lock().unwrap_or_else(|p| p.into_inner()).iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }
}

#[cfg(feature = "sqlite")]
pub use sqlite::SqlitePayloads;

#[cfg(feature = "sqlite")]
mod sqlite {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use rusqlite::Connection;
    use serde_json::{Map, Value};

    use super::{Held, NAME, Payloads};
    use crate::errors::Result;
    use crate::knowledge::base::store_error;

    fn sql_error(e: impl std::fmt::Display) -> crate::errors::Error {
        store_error(NAME, format!("payloads: {e}"))
    }

    /// The text and metadata of each record, in a SQLite file — the same
    /// `payloads (id, text, metadata)` table the Python `SQLitePayloads` writes.
    #[derive(Debug, Clone)]
    pub struct SqlitePayloads {
        path: String,
        db: Arc<Mutex<Option<Connection>>>,
    }

    impl SqlitePayloads {
        /// `vertex-payloads.db` in Python's default.
        pub fn new(path: impl Into<String>) -> Self {
            Self { path: path.into(), db: Arc::default() }
        }

        async fn with<T: Send + 'static>(&self, f: impl FnOnce(&Connection) -> Result<T> + Send + 'static) -> Result<T> {
            let (db, path) = (self.db.clone(), self.path.clone());
            tokio::task::spawn_blocking(move || {
                let mut slot = db.lock().unwrap_or_else(|p| p.into_inner());
                if slot.is_none() {
                    let db = Connection::open(&path).map_err(sql_error)?;
                    let _ = db.busy_timeout(std::time::Duration::from_secs(30));
                    db.execute("CREATE TABLE IF NOT EXISTS payloads (id TEXT PRIMARY KEY, text TEXT, metadata TEXT)", []).map_err(sql_error)?;
                    *slot = Some(db);
                }
                f(slot.as_ref().expect("opened above"))
            })
            .await
            .map_err(sql_error)?
        }
    }

    fn read(rows: &mut rusqlite::Rows<'_>, out: &mut Held) -> Result<()> {
        while let Some(r) = rows.next().map_err(sql_error)? {
            let id: String = r.get(0).map_err(sql_error)?;
            let text: Option<String> = r.get(1).map_err(sql_error)?;
            let meta: Option<String> = r.get(2).map_err(sql_error)?;
            let meta: Map<String, Value> = serde_json::from_str(meta.as_deref().unwrap_or("{}")).unwrap_or_default();
            out.insert(id, (text.unwrap_or_default(), meta));
        }
        Ok(())
    }

    #[async_trait]
    impl Payloads for SqlitePayloads {
        async fn put(&self, rows: Vec<(String, String, Map<String, Value>)>) -> Result<()> {
            self.with(move |db| {
                let tx = db.unchecked_transaction().map_err(sql_error)?;
                {
                    let mut stmt = tx
                        .prepare("INSERT INTO payloads VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET text=excluded.text, metadata=excluded.metadata")
                        .map_err(sql_error)?;
                    for (id, text, meta) in &rows {
                        let meta = serde_json::to_string(meta).unwrap_or_else(|_| "{}".into());
                        stmt.execute(rusqlite::params![id, text, meta]).map_err(sql_error)?;
                    }
                }
                tx.commit().map_err(sql_error)
            })
            .await
        }

        async fn get(&self, ids: &[String]) -> Result<Held> {
            let ids = ids.to_vec();
            self.with(move |db| {
                let mut out = Held::new();
                for part in ids.chunks(500) {
                    let marks = vec!["?"; part.len()].join(",");
                    let mut stmt = db.prepare(&format!("SELECT id, text, metadata FROM payloads WHERE id IN ({marks})")).map_err(sql_error)?;
                    let mut rows = stmt.query(rusqlite::params_from_iter(part.iter())).map_err(sql_error)?;
                    read(&mut rows, &mut out)?;
                }
                Ok(out)
            })
            .await
        }

        async fn delete(&self, ids: &[String]) -> Result<()> {
            let ids = ids.to_vec();
            self.with(move |db| {
                let tx = db.unchecked_transaction().map_err(sql_error)?;
                {
                    let mut stmt = tx.prepare("DELETE FROM payloads WHERE id = ?").map_err(sql_error)?;
                    for i in &ids {
                        stmt.execute([i]).map_err(sql_error)?;
                    }
                }
                tx.commit().map_err(sql_error)
            })
            .await
        }

        async fn all(&self) -> Result<Held> {
            self.with(|db| {
                let mut out = Held::new();
                let mut stmt = db.prepare("SELECT id, text, metadata FROM payloads").map_err(sql_error)?;
                let mut rows = stmt.query([]).map_err(sql_error)?;
                read(&mut rows, &mut out)?;
                Ok(out)
            })
            .await
        }
    }
}

/// Produces a Google access token, fresh enough to use.
pub type TokenFn = Arc<dyn Fn() -> BoxFuture<'static, Result<String>> + Send + Sync>;

/// A Google access token, from whichever source is available: the function
/// given, the token given, `GOOGLE_ACCESS_TOKEN`, then `gcloud auth
/// print-access-token` (cached until shortly before it expires).
///
/// Python also tries google-auth's Application Default Credentials (service
/// account files, the metadata server) before gcloud. Without that library
/// here, pass a `TokenFn` that does it, or a token.
struct GoogleAuth {
    token: Option<String>,
    auth: Option<TokenFn>,
    cached: tokio::sync::Mutex<(String, f64)>,
}

impl GoogleAuth {
    async fn token(&self) -> Result<String> {
        if let Some(f) = &self.auth {
            return f().await;
        }
        if let Some(t) = self.token.clone().filter(|t| !t.is_empty()).or_else(|| std::env::var("GOOGLE_ACCESS_TOKEN").ok().filter(|t| !t.is_empty())) {
            return Ok(t);
        }
        let mut cached = self.cached.lock().await;
        if !cached.0.is_empty() && crate::types::now() < cached.1 - 60.0 {
            return Ok(cached.0.clone());
        }
        // No library: borrow the token the CLI already has.
        let mut command = if cfg!(windows) {
            let mut c = tokio::process::Command::new("cmd");
            c.args(["/C", "gcloud"]);
            c
        } else {
            tokio::process::Command::new("gcloud")
        };
        command.args(["auth", "print-access-token"]).kill_on_drop(true);
        let missing = || {
            store_error(
                NAME,
                "Vertex needs a Google access token — set GOOGLE_ACCESS_TOKEN, run `gcloud auth application-default login`, or pass a token or a token function",
            )
        };
        let output = match tokio::time::timeout(std::time::Duration::from_secs(30), command.output()).await {
            Ok(Ok(o)) => o,
            _ => return Err(missing()),
        };
        if !output.status.success() {
            let err: String = String::from_utf8_lossy(&output.stderr).trim().chars().take(300).collect();
            return Err(store_error(NAME, format!("gcloud could not produce a token: {err}")));
        }
        let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if token.is_empty() {
            return Err(missing());
        }
        *cached = (token.clone(), crate::types::now() + 3000.0);
        Ok(token)
    }
}

/// The options of a [`VertexVectorStore`] — the same as the Python constructor's.
#[derive(Clone)]
pub struct VertexOptions {
    pub project: String,
    pub region: String,
    /// The index id.
    pub index: String,
    /// The index endpoint id.
    pub endpoint: String,
    pub deployed_index: String,
    /// The endpoint's public domain: `1234.us-central1-56789.vdb.vertexai.goog`.
    /// A leading `https://` is dropped; one written `http://…` is kept as is
    /// (for a local stand-in).
    pub endpoint_domain: String,
    /// Where text and metadata go. A SQLite file `vertex-payloads.db` when
    /// `None` (in memory when the crate is built without `sqlite`).
    pub payloads: Option<Arc<dyn Payloads>>,
    /// A ready-made Google access token. Else `GOOGLE_ACCESS_TOKEN`, then gcloud.
    pub token: Option<String>,
    /// A function that returns a token; wins over `token`.
    pub auth: Option<TokenFn>,
    /// Overrides `https://{region}-aiplatform.googleapis.com`, for a local stand-in.
    pub api_url: Option<String>,
    pub timeout: f64,
    pub retries: u32,
    pub headers: Vec<(String, String)>,
}

impl Default for VertexOptions {
    fn default() -> Self {
        Self {
            project: String::new(),
            region: String::new(),
            index: String::new(),
            endpoint: String::new(),
            deployed_index: String::new(),
            endpoint_domain: String::new(),
            payloads: None,
            token: None,
            auth: None,
            api_url: None,
            timeout: 30.0,
            retries: 3,
            headers: vec![],
        }
    }
}

/// Vertex AI Vector Search.
///
/// ```no_run
/// # use agent_harness::knowledge::stores::vertex::{VertexVectorStore, VertexOptions};
/// let store = VertexVectorStore::new(VertexOptions {
///     project: "my-project".into(), region: "us-central1".into(),
///     index: "1234567890".into(), endpoint: "9876543210".into(),
///     deployed_index: "handbook_v1".into(),
///     endpoint_domain: "1234.us-central1-56789.vdb.vertexai.goog".into(),
///     ..Default::default()
/// }).unwrap();
/// ```
///
/// The index is one you created for *streaming updates* and deployed to an
/// endpoint — that takes the better part of an hour and is done in the Cloud
/// console or with `gcloud`, not here. Create it with the dot-product distance:
/// vectors are sent at unit length, so a dot product is a cosine.
///
/// Vector Search stores no text. Each record's text and metadata go to
/// `payloads` (a SQLite file by default); what is filtered on is also sent to
/// Vertex as restricts.
pub struct VertexVectorStore {
    project: String,
    region: String,
    endpoint: String,
    deployed_index: String,
    index_url: String,
    endpoint_base: String,
    payloads: Arc<dyn Payloads>,
    auth: GoogleAuth,
    http: HttpClient,
}

impl std::fmt::Debug for VertexVectorStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<VertexVectorStore {}>", self.index_url)
    }
}

fn is_int(n: &serde_json::Number) -> bool {
    n.is_i64() || n.is_u64()
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

fn restricts_of(tokens: Vec<Value>, numbers: Vec<Value>) -> Map<String, Value> {
    let mut out = Map::new();
    if !tokens.is_empty() {
        out.insert("restricts".into(), Value::Array(tokens));
    }
    if !numbers.is_empty() {
        out.insert("numericRestricts".into(), Value::Array(numbers));
    }
    out
}

/// A record's metadata as the restricts Vertex filters on.
pub fn restricts(metadata: &Map<String, Value>) -> Map<String, Value> {
    let (mut tokens, mut numbers) = (vec![], vec![]);
    for (key, value) in metadata {
        match value {
            Value::Bool(b) => tokens.push(json!({"namespace": key, "allowList": [b.to_string()]})),
            Value::Number(n) if is_int(n) => numbers.push(json!({"namespace": key, "valueInt": n.to_string()})),
            Value::Number(n) => numbers.push(json!({"namespace": key, "valueDouble": n})),
            Value::String(s) => tokens.push(json!({"namespace": key, "allowList": [s]})),
            Value::Array(items) if !items.is_empty() => tokens.push(json!({"namespace": key, "allowList": items.iter().map(py_str).collect::<Vec<_>>()})),
            _ => {}
        }
    }
    restricts_of(tokens, numbers)
}

/// A filter as the restricts of a query.
pub fn wanted_restricts(wanted: &[Condition]) -> Result<Map<String, Value>> {
    let (mut tokens, mut numbers) = (vec![], vec![]);
    let written = |v: &Value| match v {
        Value::Bool(b) => b.to_string(),
        other => py_str(other),
    };
    for c in wanted {
        let name = &c.field;
        if c.op == "$in" {
            tokens.push(json!({"namespace": name, "allowList": c.value.as_array().into_iter().flatten().map(written).collect::<Vec<_>>()}));
        } else if let Value::Number(n) = &c.value {
            let op = NUMERIC.iter().find(|(o, _)| *o == c.op).map_or("EQUAL", |(_, v)| *v);
            numbers.push(if is_int(n) {
                json!({"namespace": name, "op": op, "valueInt": n.to_string()})
            } else {
                json!({"namespace": name, "op": op, "valueDouble": n})
            });
        } else if c.op == "$eq" || c.op == "$ne" {
            let list = if c.op == "$eq" { "allowList" } else { "denyList" };
            tokens.push(json!({"namespace": name, list: [written(&c.value)]}));
        } else {
            return Err(store_error(NAME, format!("a range on '{name}' needs a number, not {}", c.value)));
        }
    }
    Ok(restricts_of(tokens, numbers))
}

fn as_count(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

impl VertexVectorStore {
    pub fn new(o: VertexOptions) -> Result<Self> {
        for (field, value) in [
            ("project", &o.project),
            ("region", &o.region),
            ("index", &o.index),
            ("endpoint", &o.endpoint),
            ("deployed_index", &o.deployed_index),
            ("endpoint_domain", &o.endpoint_domain),
        ] {
            if value.is_empty() {
                return Err(Error::config(format!("VertexVectorStore needs {field}")));
            }
        }
        let domain = o.endpoint_domain.trim_end_matches('/');
        let endpoint_base =
            if domain.starts_with("http://") { domain.to_string() } else { format!("https://{}", domain.strip_prefix("https://").unwrap_or(domain)) };
        let api = o.api_url.map(|u| u.trim_end_matches('/').to_string()).unwrap_or_else(|| format!("https://{}-aiplatform.googleapis.com", o.region));
        let index_url = format!("{api}/v1/projects/{}/locations/{}/indexes/{}", o.project, o.region, o.index);
        let payloads: Arc<dyn Payloads> = match o.payloads {
            Some(p) => p,
            #[cfg(feature = "sqlite")]
            None => Arc::new(SqlitePayloads::new("vertex-payloads.db")),
            #[cfg(not(feature = "sqlite"))]
            None => Arc::new(MemoryPayloads::default()),
        };
        let mut http = HttpClient::new(NAME, o.timeout, o.retries).secret(o.token.clone().unwrap_or_default());
        for (k, v) in o.headers {
            http = http.header(&k, v);
        }
        Ok(Self {
            project: o.project,
            region: o.region,
            endpoint: o.endpoint,
            deployed_index: o.deployed_index,
            index_url,
            endpoint_base,
            payloads,
            auth: GoogleAuth { token: o.token, auth: o.auth, cached: tokio::sync::Mutex::new((String::new(), 0.0)) },
            http,
        })
    }

    /// As `vector_store("vertex://my-project/us-central1/1234567890?endpoint=…&deployed_index=…&endpoint_domain=…")`
    /// builds it. `payloads=<file>` puts the payloads in that SQLite file (an addition).
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        if bare {
            return Err(Error::config(
                "VertexVectorStore needs project, region, index, endpoint, deployed_index and endpoint_domain — vertex://<project>/<region>/<index>?endpoint=…&deployed_index=…&endpoint_domain=…",
            ));
        }
        let (region, index) = p.path.split_once('/').unwrap_or((&p.path, ""));
        let mut o = VertexOptions { project: p.netloc.clone(), region: region.into(), index: index.into(), ..Default::default() };
        for (k, value) in std::mem::take(&mut p.given) {
            match k.as_str() {
                "endpoint" => o.endpoint = value,
                "deployed_index" => o.deployed_index = value,
                "endpoint_domain" => o.endpoint_domain = value,
                "token" => o.token = Some(value),
                "payloads" => {
                    #[cfg(feature = "sqlite")]
                    {
                        o.payloads = Some(Arc::new(SqlitePayloads::new(value)));
                    }
                    #[cfg(not(feature = "sqlite"))]
                    {
                        let _ = value;
                        return Err(Error::config("payloads in a file need the `sqlite` feature"));
                    }
                }
                "timeout" => o.timeout = value.parse().map_err(|_| Error::config(format!("timeout is a number of seconds, not '{value}'")))?,
                "retries" => o.retries = value.parse().map_err(|_| Error::config(format!("retries is a whole number, not '{value}'")))?,
                other => return Err(Error::config(format!("VertexVectorStore takes no option '{other}'"))),
            }
        }
        Self::new(o)
    }

    async fn call(&self, method: Method, url: &str, body: Option<&Value>) -> Result<Reply> {
        let bearer = format!("Bearer {}", self.auth.token().await?);
        let bytes = body.map(|b| serde_json::to_vec(b).unwrap_or_default());
        self.http.send(method, url, bytes, body.is_some().then_some("application/json"), &[("authorization", bearer)], &[]).await
    }
}

#[async_trait]
impl VectorStore for VertexVectorStore {
    fn name(&self) -> &str {
        NAME
    }

    fn batch_size(&self) -> usize {
        100
    }

    async fn ensure(&self, dimension: usize) -> Result<()> {
        let found = self.call(Method::GET, &self.index_url, None).await?;
        let held = as_count(&found.body["metadata"]["config"]["dimensions"]).unwrap_or(0);
        if held != 0 && held as usize != dimension {
            return Err(store_error(NAME, format!("the Vertex index holds {held}-dimension vectors; {dimension} was asked for")));
        }
        Ok(())
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        // The text first: a vector with nothing to say is worse than text with
        // no vector yet.
        self.payloads.put(records.iter().map(|r| (r.id.clone(), r.text.clone(), r.metadata.clone())).collect()).await?;
        let datapoints: Vec<Value> = records
            .iter()
            .map(|r| {
                let mut point = Map::new();
                point.insert("datapointId".into(), json!(r.id));
                point.insert("featureVector".into(), json!(normalise(&r.vector)));
                point.extend(restricts(&r.metadata));
                Value::Object(point)
            })
            .collect();
        self.call(Method::POST, &format!("{}:upsertDatapoints", self.index_url), Some(&json!({"datapoints": datapoints}))).await?;
        Ok(())
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let mut datapoint = Map::new();
        datapoint.insert("featureVector".into(), json!(normalise(vector)));
        datapoint.extend(wanted_restricts(wanted)?);
        let url = format!("{}/v1/projects/{}/locations/{}/indexEndpoints/{}:findNeighbors", self.endpoint_base, self.project, self.region, self.endpoint);
        let body = json!({"deployedIndexId": self.deployed_index, "queries": [{"neighborCount": k, "datapoint": datapoint}]});
        let answer = self.call(Method::POST, &url, Some(&body)).await?;
        let nearest: Vec<Value> = answer.body["nearestNeighbors"][0]["neighbors"].as_array().cloned().unwrap_or_default();
        let found: Vec<(String, f64)> = nearest
            .iter()
            .filter_map(|n| n["datapoint"]["datapointId"].as_str().filter(|s| !s.is_empty()).map(|id| (id.to_string(), n["distance"].as_f64().unwrap_or(0.0))))
            .collect();
        let ids: Vec<String> = found.iter().map(|(id, _)| id.clone()).collect();
        let mut held = self.payloads.get(&ids).await?;
        Ok(found
            .into_iter()
            .map(|(id, score)| {
                let (text, metadata) = held.remove(&id).unwrap_or_default();
                VectorHit { id, score, text, metadata }
            })
            .collect())
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        let ids: Vec<String> = if wanted.is_empty() {
            ids.map(<[String]>::to_vec).unwrap_or_default()
        } else {
            // Vertex removes by id; the payloads know which ids a filter means.
            let mut keys: Vec<String> = self
                .payloads
                .all()
                .await?
                .into_iter()
                .filter(|(key, (_, meta))| ids.is_none_or(|ids| ids.contains(key)) && holds(meta, wanted))
                .map(|(key, _)| key)
                .collect();
            keys.sort();
            keys
        };
        for part in ids.chunks(1000) {
            self.call(Method::POST, &format!("{}:removeDatapoints", self.index_url), Some(&json!({"datapointIds": part}))).await?;
            self.payloads.delete(part).await?;
        }
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let found = self.call(Method::GET, &self.index_url, None).await?;
        Ok(as_count(&found.body["indexStats"]["vectorsCount"]).unwrap_or(0) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn c(field: &str, op: &str, value: Value) -> Condition {
        Condition { field: field.into(), op: op.into(), value }
    }

    #[test]
    fn restricts_match_python() {
        let meta = json!({"lang": "en", "draft": true, "year": 2024, "score": 0.5, "tags": ["a", true], "empty": [], "none": null});
        let r = Value::Object(restricts(meta.as_object().unwrap()));
        let tokens = r["restricts"].as_array().unwrap();
        assert!(tokens.contains(&json!({"namespace": "lang", "allowList": ["en"]})));
        assert!(tokens.contains(&json!({"namespace": "draft", "allowList": ["true"]})));
        assert!(tokens.contains(&json!({"namespace": "tags", "allowList": ["a", "True"]})));
        assert_eq!(tokens.len(), 3);
        let numbers = r["numericRestricts"].as_array().unwrap();
        assert!(numbers.contains(&json!({"namespace": "year", "valueInt": "2024"})));
        assert!(numbers.contains(&json!({"namespace": "score", "valueDouble": 0.5})));

        let w = wanted_restricts(&[
            c("lang", "$in", json!(["en", true])),
            c("year", "$gte", json!(2024)),
            c("score", "$lt", json!(0.5)),
            c("draft", "$ne", json!(true)),
            c("doc", "$eq", json!("d")),
        ])
        .unwrap();
        assert_eq!(
            Value::Object(w),
            json!({
                "restricts": [{"namespace": "lang", "allowList": ["en", "true"]}, {"namespace": "draft", "denyList": ["true"]}, {"namespace": "doc", "allowList": ["d"]}],
                "numericRestricts": [{"namespace": "year", "op": "GREATER_EQUAL", "valueInt": "2024"}, {"namespace": "score", "op": "LESS", "valueDouble": 0.5}]})
        );
        assert!(wanted_restricts(&[c("lang", "$gt", json!("a"))]).is_err());
        assert!(wanted_restricts(&[]).unwrap().is_empty());
    }

    #[test]
    fn urls_are_read_like_python() {
        let s = VertexVectorStore::from_url(
            UrlParts::parse("vertex://proj/europe-west1/123?endpoint=456&deployed_index=d1&endpoint_domain=https://x.vdb.vertexai.goog/").unwrap(),
            false,
        )
        .unwrap();
        assert_eq!(s.index_url, "https://europe-west1-aiplatform.googleapis.com/v1/projects/proj/locations/europe-west1/indexes/123");
        assert_eq!(s.endpoint_base, "https://x.vdb.vertexai.goog");
        assert!(VertexVectorStore::from_url(UrlParts::parse("vertex://proj/r/i").unwrap(), false).is_err());
        assert!(VertexVectorStore::from_url(UrlParts { scheme: "vertex".into(), ..Default::default() }, true).is_err());
    }

    #[tokio::test]
    async fn requests_have_the_python_shape() {
        let (url, seen) = fake(Arc::new(|method, path, _| match (method, path) {
            ("GET", _) => (200, r#"{"metadata": {"config": {"dimensions": 2}}, "indexStats": {"vectorsCount": "5"}}"#.into()),
            (_, p) if p.ends_with(":findNeighbors") => (
                200,
                r#"{"nearestNeighbors": [{"id": "0", "neighbors": [{"datapoint": {"datapointId": "a"}, "distance": 0.9}, {"datapoint": {"datapointId": "zz"}, "distance": 0.1}]}]}"#.into(),
            ),
            _ => (200, "{}".into()),
        }))
        .await;
        let payloads = Arc::new(MemoryPayloads::default());
        let store = VertexVectorStore::new(VertexOptions {
            project: "proj".into(),
            region: "us-central1".into(),
            index: "123".into(),
            endpoint: "456".into(),
            deployed_index: "d1".into(),
            endpoint_domain: url.clone(),
            api_url: Some(url.clone()),
            payloads: Some(payloads.clone()),
            token: Some("google-token".into()),
            retries: 0,
            ..Default::default()
        })
        .unwrap();
        store.ensure(2).await.unwrap();
        assert!(store.ensure(3).await.unwrap_err().to_string().contains("holds 2-dimension"));
        store
            .upsert(vec![
                VectorRecord::new("a", vec![3.0, 4.0], "hello", json!({"lang": "en"})),
                VectorRecord::new("b", vec![1.0, 0.0], "b", json!({"lang": "fr"})),
            ])
            .await
            .unwrap();
        let hits = store.query(&[3.0, 4.0], 2, Some(&json!({"lang": "en"}))).await.unwrap();
        assert_eq!((hits[0].id.as_str(), hits[0].text.as_str(), hits[0].score), ("a", "hello", 0.9));
        assert_eq!((hits[1].id.as_str(), hits[1].text.as_str()), ("zz", ""));
        store.delete(None, Some(&json!({"lang": "fr"}))).await.unwrap();
        assert_eq!(payloads.all().await.unwrap().len(), 1);
        assert_eq!(store.count().await.unwrap(), 5);

        let seen = seen.lock().unwrap();
        assert!(seen.iter().all(|r| r.2["authorization"] == "Bearer google-token"));
        let index = "/v1/projects/proj/locations/us-central1/indexes/123";
        assert_eq!(seen[2].1, format!("{index}:upsertDatapoints"));
        let up: Value = serde_json::from_str(&seen[2].3).unwrap();
        assert_eq!(up["datapoints"][0]["datapointId"], "a");
        assert_eq!(up["datapoints"][0]["restricts"], json!([{"namespace": "lang", "allowList": ["en"]}]));
        let v = up["datapoints"][0]["featureVector"].as_array().unwrap();
        assert!((v[0].as_f64().unwrap() - 0.6).abs() < 1e-6);
        assert_eq!(seen[3].1, "/v1/projects/proj/locations/us-central1/indexEndpoints/456:findNeighbors");
        let q: Value = serde_json::from_str(&seen[3].3).unwrap();
        assert_eq!(q["deployedIndexId"], "d1");
        assert_eq!(q["queries"][0]["neighborCount"], 2);
        assert_eq!(q["queries"][0]["datapoint"]["restricts"], json!([{"namespace": "lang", "allowList": ["en"]}]));
        assert_eq!(seen[4].1, format!("{index}:removeDatapoints"));
        assert_eq!(serde_json::from_str::<Value>(&seen[4].3).unwrap(), json!({"datapointIds": ["b"]}));
    }

    #[tokio::test]
    async fn token_functions_are_used() {
        let f: TokenFn = Arc::new(|| Box::pin(async { Ok("from-fn".to_string()) }));
        let auth = GoogleAuth { token: Some("ignored".into()), auth: Some(f), cached: tokio::sync::Mutex::new((String::new(), 0.0)) };
        assert_eq!(auth.token().await.unwrap(), "from-fn");
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_payloads_round_trip() {
        let path = std::env::temp_dir().join(format!("vertex-payloads-{}.db", uuid::Uuid::new_v4().simple()));
        let p = SqlitePayloads::new(path.to_string_lossy().to_string());
        p.put(vec![("a".into(), "one".into(), json!({"n": 1}).as_object().unwrap().clone()), ("b".into(), "two".into(), Map::new())]).await.unwrap();
        p.put(vec![("a".into(), "uno".into(), Map::new())]).await.unwrap();
        let got = p.get(&["a".into(), "c".into()]).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got["a"].0, "uno");
        p.delete(&["a".into()]).await.unwrap();
        assert_eq!(p.all().await.unwrap().keys().collect::<Vec<_>>(), ["b"]);
        drop(p);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn live() {
        let Ok(url) = std::env::var("AH_LIVE_VERTEX_URL") else { return };
        let store = super::super::vector_store(&url).unwrap();
        let report = store.check(8).await;
        assert_eq!(report["ok"], true, "{report:#}");
    }
}
