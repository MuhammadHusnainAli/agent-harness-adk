//! Cloudflare Vectorize.

use async_trait::async_trait;
use indexmap::IndexMap;
use reqwest::Method;
use serde_json::{Map, Value, json};

use super::UrlParts;
use super::qdrant::{Settings, floats, number, object, or_env, py_dumps, text_of};
use crate::errors::Result;
use crate::knowledge::base::{ALWAYS_FILTERABLE, Condition, HttpClient, Reply, VectorHit, VectorRecord, VectorStore, store_error};
use crate::llm_providers::resilience::RetryPolicy;

/// ```ignore
/// VectorizeStore::new("account-id")?.index("handbook").api_token("…")
///     .with_filterable([("source", "string"), ("year", "number")]);
/// ```
///
/// Vectorize filters only on metadata it has an index for: `filterable` names
/// them (`string`, `number` or `boolean`) and they are created with the index.
/// It applies writes a few seconds after accepting them, returns at most 20
/// results with their metadata, and cannot delete by filter — so removing a
/// document means removing its ids, which `KnowledgeBase` does.
#[derive(Debug, Clone)]
pub struct VectorizeStore {
    pub account_id: String,
    pub index: String,
    pub api_token: String,
    pub namespace: String,
    /// Field → `string`, `number` or `boolean`; `doc` and `namespace` always.
    pub fields: IndexMap<String, String>,
    pub url: String,
    settings: Settings,
    http: HttpClient,
}

impl VectorizeStore {
    /// An empty `account_id` is taken from `$CLOUDFLARE_ACCOUNT_ID`; the token
    /// comes from `$CLOUDFLARE_API_TOKEN` unless one is given.
    pub fn new(account_id: &str) -> Result<Self> {
        let account_id = or_env(account_id, "CLOUDFLARE_ACCOUNT_ID");
        if account_id.is_empty() {
            return Err(store_error("cloudflare-vectorize", "Vectorize needs account_id= or $CLOUDFLARE_ACCOUNT_ID"));
        }
        let settings = Settings::default();
        let mut store = Self {
            account_id,
            index: "knowledge".into(),
            api_token: or_env("", "CLOUDFLARE_API_TOKEN"),
            namespace: String::new(),
            fields: ALWAYS_FILTERABLE[..2].iter().map(|f| (f.to_string(), "string".to_string())).collect(),
            url: "https://api.cloudflare.com/client/v4".into(),
            http: settings.client("cloudflare-vectorize", &[], &[]),
            settings,
        };
        store.rebuild();
        Ok(store)
    }

    pub fn index(mut self, index: &str) -> Self {
        self.index = index.into();
        self
    }

    pub fn api_token(mut self, token: &str) -> Self {
        self.api_token = or_env(token, "CLOUDFLARE_API_TOKEN");
        self.rebuild();
        self
    }

    pub fn namespace(mut self, namespace: &str) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// The metadata fields to index, and their kinds: `string`, `number` or `boolean`.
    pub fn with_filterable<K: Into<String>, V: AsRef<str>>(mut self, fields: impl IntoIterator<Item = (K, V)>) -> Self {
        for (k, v) in fields {
            self.fields.insert(k.into(), v.as_ref().to_lowercase());
        }
        self
    }

    /// The API's root (`https://api.cloudflare.com/client/v4` by default).
    pub fn url(mut self, url: &str) -> Self {
        self.url = url.trim_end_matches('/').to_string();
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

    /// `vectorize://account-id/index?api_token=…&namespace=…&filterable=source:string,year:number`,
    /// or the environment's account for a bare `vectorize`.
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        if bare {
            return Self::new("");
        }
        let settings = Settings::from_url(&mut p, "cloudflare-vectorize")?;
        let named = p.take("index");
        let token = p.take("api_token").unwrap_or_default();
        let namespace = p.take("namespace").unwrap_or_default();
        let url = p.take("url");
        let fields: Vec<(String, String)> = match p.take("filterable") {
            None => vec![],
            Some(list) => list
                .split(',')
                .filter(|f| !f.trim().is_empty())
                .map(|f| {
                    let (name, kind) = f.split_once(':').unwrap_or((f, "string"));
                    (name.trim().to_string(), kind.trim().to_string())
                })
                .collect(),
        };
        let account = p.take("account_id");
        super::qdrant::no_more(&p, "cloudflare-vectorize")?;
        let account = if p.netloc.is_empty() { account.unwrap_or_default() } else { p.netloc.clone() };
        let mut store = Self::new(&account)?.api_token(&token).namespace(&namespace).with_filterable(fields);
        store.settings = settings;
        store.rebuild();
        if !p.path.is_empty() {
            store.index = p.path.clone();
        } else if let Some(i) = named {
            store.index = i;
        }
        if let Some(u) = url {
            store = store.url(&u);
        }
        Ok(store)
    }

    fn rebuild(&mut self) {
        let own = [("authorization", format!("Bearer {}", self.api_token))];
        self.http = self.settings.client("cloudflare-vectorize", &own, &[&self.api_token]);
    }

    fn root(&self) -> String {
        format!("{}/accounts/{}/vectorize/v2/indexes", self.url, self.account_id)
    }

    /// Cloudflare's envelope checked: `success: false` is a failure whatever the status.
    fn checked(&self, reply: Reply) -> Result<Reply> {
        if (200..300).contains(&reply.status) && reply.body.get("success") == Some(&Value::Bool(false)) {
            let errors: String = py_dumps(reply.body.get("errors").unwrap_or(&Value::Null)).chars().take(300).collect();
            return Err(store_error("cloudflare-vectorize", format!("vectorize: {errors}")));
        }
        Ok(reply)
    }

    async fn call(&self, method: Method, path: &str, body: Option<&Value>, ok: &[u16]) -> Result<Reply> {
        let reply = self.http.json(method, &format!("{}{path}", self.root()), body, ok).await?;
        self.checked(reply)
    }

    /// The filter in Vectorize's terms: `{field: {op: value}}`, on declared fields only.
    pub fn filter(&self, wanted: &[Condition]) -> Result<Map<String, Value>> {
        let mut out = Map::new();
        for c in wanted {
            if !self.fields.contains_key(&c.field) {
                let now = if self.fields.is_empty() { "nothing".to_string() } else { self.fields.keys().cloned().collect::<Vec<_>>().join(", ") };
                return Err(store_error(
                    "cloudflare-vectorize",
                    format!("'{0}' has no metadata index in Vectorize — declare it: filterable={{'{0}': 'string'}}. Filterable now: {now}", c.field),
                ));
            }
            let entry = out.entry(c.field.clone()).or_insert_with(|| json!({}));
            entry[c.op.as_str()] = c.value.clone();
        }
        Ok(out)
    }
}

#[async_trait]
impl VectorStore for VectorizeStore {
    fn name(&self) -> &str {
        "cloudflare-vectorize"
    }

    fn batch_size(&self) -> usize {
        500
    }

    /// With metadata returned, Vectorize answers with at most this many.
    fn max_k(&self) -> usize {
        20
    }

    fn filterable(&self) -> Option<Vec<String>> {
        Some(self.fields.keys().cloned().collect())
    }

    async fn ensure(&self, dimension: usize) -> Result<()> {
        if !matches!(self.call(Method::GET, &format!("/{}", self.index), None, &[404, 410]).await?.status, 404 | 410) {
            return Ok(());
        }
        self.call(Method::POST, "", Some(&json!({"name": self.index, "config": {"dimensions": dimension, "metric": "cosine"}})), &[409]).await?;
        for (name, kind) in &self.fields {
            self.call(Method::POST, &format!("/{}/metadata_index/create", self.index), Some(&json!({"propertyName": name, "indexType": kind})), &[409]).await?;
        }
        Ok(())
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        let mut lines = String::new();
        for r in &records {
            let mut metadata = r.metadata.clone();
            metadata.insert("_text".into(), json!(r.text));
            let mut row = json!({"id": r.id, "values": floats(&r.vector), "metadata": metadata});
            if !self.namespace.is_empty() {
                row["namespace"] = json!(self.namespace);
            }
            lines.push_str(&py_dumps(&row));
            lines.push('\n');
        }
        let reply = self
            .http
            .send(Method::POST, &format!("{}/{}/upsert", self.root(), self.index), Some(lines.into_bytes()), Some("application/x-ndjson"), &[], &[])
            .await?;
        self.checked(reply)?;
        Ok(())
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let mut body = json!({"vector": floats(vector), "topK": k, "returnMetadata": "all", "returnValues": false});
        let found = self.filter(wanted)?;
        if !found.is_empty() {
            body["filter"] = Value::Object(found);
        }
        if !self.namespace.is_empty() {
            body["namespace"] = json!(self.namespace);
        }
        let answer = self.call(Method::POST, &format!("/{}/query", self.index), Some(&body), &[]).await?.body;
        Ok(answer
            .get("result")
            .and_then(|r| r.get("matches"))
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
        let Some(ids) = ids.filter(|_| wanted.is_empty()) else {
            return Err(store_error("cloudflare-vectorize", "Vectorize deletes by id only — it has no delete-by-filter"));
        };
        for chunk in ids.chunks(1000) {
            self.call(Method::POST, &format!("/{}/delete_by_ids", self.index), Some(&json!({"ids": chunk})), &[]).await?;
        }
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let answer = self.call(Method::GET, &format!("/{}/info", self.index), None, &[]).await?.body;
        let result = object(answer.get("result"));
        let n = number(result.get("vectorCount"));
        Ok(if n != 0.0 { n } else { number(result.get("vectorsCount")) } as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::super::qdrant::fake;
    use super::*;
    use crate::knowledge::base::conditions;

    #[test]
    fn filters_and_urls() {
        let store = VectorizeStore::new("acct").unwrap().with_filterable([("source", "String")]);
        assert_eq!(store.filterable().unwrap(), vec!["doc", "namespace", "source"]);
        assert_eq!(store.fields["source"], "string");
        let c = conditions(Some(&json!({"source": {"$ne": "x", "$in": ["a"]}}))).unwrap();
        assert_eq!(Value::Object(store.filter(&c).unwrap()), json!({"source": {"$ne": "x", "$in": ["a"]}}));
        let e = store.filter(&conditions(Some(&json!({"colour": "red"}))).unwrap()).unwrap_err();
        assert!(
            e.to_string()
                .contains("'colour' has no metadata index in Vectorize — declare it: filterable={'colour': 'string'}. Filterable now: doc, namespace, source")
        );
        let v = VectorizeStore::from_url(UrlParts::parse("vectorize://acct123/docs?api_token=t0ken&filterable=year:number").unwrap(), false).unwrap();
        assert_eq!((v.account_id.as_str(), v.index.as_str(), v.api_token.as_str()), ("acct123", "docs", "t0ken"));
        assert_eq!(v.fields.get("year").map(String::as_str), Some("number"));
        assert_eq!(v.root(), "https://api.cloudflare.com/client/v4/accounts/acct123/vectorize/v2/indexes");
    }

    #[tokio::test]
    async fn requests_are_what_vectorize_expects() {
        let fake = fake::serve(|s| match (s.method.as_str(), s.path.as_str()) {
            ("GET", p) if p.ends_with("/indexes/kb") => (404, json!({})),
            (_, p) if p.ends_with("/query") => {
                (200, json!({"success": true, "result": {"matches": [{"id": "doc#0", "score": 0.9, "metadata": {"year": 2026, "_text": "T"}}]}}))
            }
            (_, p) if p.ends_with("/info") => (200, json!({"success": true, "result": {"vectorsCount": 6}})),
            (_, p) if p.ends_with("/delete_by_ids") => (200, json!({"success": false, "errors": [{"code": 1, "message": "nope"}]})),
            _ => (200, json!({"success": true, "result": {}})),
        })
        .await;
        let store = VectorizeStore::new("acct")
            .unwrap()
            .url(&fake.url)
            .index("kb")
            .api_token("cf-secret")
            .with_filterable([("source", "string"), ("year", "number"), ("lang", "string")])
            .retries(0);
        store.ensure(2).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen[0].path, "/accounts/acct/vectorize/v2/indexes/kb");
        assert_eq!(seen[1].body, json!({"name": "kb", "config": {"dimensions": 2, "metric": "cosine"}}));
        let created: Vec<Value> = seen.iter().filter(|s| s.path.ends_with("metadata_index/create")).map(|s| s.body["propertyName"].clone()).collect();
        assert_eq!(created, vec![json!("doc"), json!("namespace"), json!("source"), json!("year"), json!("lang")]);
        assert_eq!(seen[0].header("authorization"), "Bearer cf-secret");

        let record = VectorRecord::new("doc#0", vec![0.6, 0.8], "Refunds take five days.", json!({"source": "faq.md"}));
        store.upsert(vec![record]).await.unwrap();
        let upsert = fake.seen().last().unwrap().clone();
        assert_eq!(upsert.header("content-type"), "application/x-ndjson");
        assert!(upsert.raw.ends_with('\n'));
        let line: Value = serde_json::from_str(upsert.raw.lines().next().unwrap()).unwrap();
        assert_eq!(line, json!({"id": "doc#0", "values": [0.6, 0.8], "metadata": {"source": "faq.md", "_text": "Refunds take five days."}}));

        let wanted = json!({"source": "faq.md", "year": {"$gte": 2025}, "lang": {"$in": ["en", "fr"]}});
        let hits = store.query(&[1.0, 0.0], 500, Some(&wanted)).await.unwrap();
        let body = fake.seen().last().unwrap().body.clone();
        assert_eq!(body["topK"], json!(20), "its ceiling with metadata");
        assert_eq!(body["filter"], json!({"source": {"$eq": "faq.md"}, "year": {"$gte": 2025}, "lang": {"$in": ["en", "fr"]}}));
        assert_eq!(body["returnMetadata"], json!("all"));
        assert_eq!(hits, vec![VectorHit { id: "doc#0".into(), score: 0.9, text: "T".into(), metadata: object(Some(&json!({"year": 2026}))) }]);

        let e = store.delete(None, Some(&json!({"source": "x"}))).await.unwrap_err();
        assert!(e.to_string().contains("deletes by id only"));
        let e = store.delete(Some(vec!["a".into()]), None).await.unwrap_err();
        assert!(e.to_string().contains("nope"), "success: false is a failure: {e}");
        assert_eq!(fake.seen().last().unwrap().body, json!({"ids": ["a"]}));
        assert_eq!(store.count().await.unwrap(), 6);
        assert!(store.clear().await.is_err(), "Vectorize cannot be emptied from here");
    }

    #[tokio::test]
    async fn live() {
        // The account and token from $CLOUDFLARE_ACCOUNT_ID and $CLOUDFLARE_API_TOKEN.
        let Ok(index) = std::env::var("AH_LIVE_VECTORIZE_INDEX") else { return };
        let report = VectorizeStore::new("").unwrap().index(&index).with_filterable([("n", "number")]).check(8).await;
        assert_eq!(report["ok"], json!(true), "{report}");
    }
}
