//! Milvus — self-hosted, or Zilliz Cloud.

use async_trait::async_trait;
use reqwest::Method;
use serde_json::{Map, Value, json};

use super::UrlParts;
use super::qdrant::{Settings, floats, no_more, number, object, or_env, text_of};
use crate::errors::Result;
use crate::knowledge::base::{Condition, HttpClient, VectorHit, VectorRecord, VectorStore, literal, store_error};
use crate::llm_providers::resilience::RetryPolicy;

/// `$eq` → `==`, and so on.
fn operator(op: &str) -> &'static str {
    match op {
        "$eq" => "==",
        "$ne" => "!=",
        "$gt" => ">",
        "$gte" => ">=",
        "$lt" => "<",
        _ => "<=",
    }
}

/// ```ignore
/// MilvusStore::new("http://localhost:19530").collection("handbook");
/// MilvusStore::new("https://in03-x.api.gcp-us-west1.zillizcloud.com").token("…").collection("handbook");
/// ```
///
/// Spoken to over Milvus's REST API (v2), so no client library is needed.
#[derive(Debug, Clone)]
pub struct MilvusStore {
    pub url: String,
    pub collection: String,
    /// Sent as `dbName` when not empty.
    pub database: String,
    pub token: String,
    settings: Settings,
    http: HttpClient,
}

impl Default for MilvusStore {
    fn default() -> Self {
        Self::new("http://localhost:19530")
    }
}

impl MilvusStore {
    /// The token comes from `$MILVUS_TOKEN` unless one is given.
    pub fn new(url: &str) -> Self {
        let settings = Settings::default();
        let mut store = Self {
            url: url.trim_end_matches('/').to_string(),
            collection: "knowledge".into(),
            database: String::new(),
            token: or_env("", "MILVUS_TOKEN"),
            http: settings.client("milvus", &[], &[]),
            settings,
        };
        store.rebuild();
        store
    }

    pub fn collection(mut self, collection: &str) -> Self {
        self.collection = collection.into();
        self
    }

    pub fn database(mut self, database: &str) -> Self {
        self.database = database.into();
        self
    }

    /// `user:password`, or a Zilliz Cloud API key.
    pub fn token(mut self, token: &str) -> Self {
        self.token = or_env(token, "MILVUS_TOKEN");
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

    /// `milvus://host:19530/collection?token=…&database=…` (or `zilliz+https://…`),
    /// or the defaults for a bare `milvus`.
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        if bare {
            return Ok(Self::default());
        }
        let settings = Settings::from_url(&mut p, "milvus")?;
        let named = p.take("collection");
        let token = p.take("token").unwrap_or_default();
        let database = p.take("database").unwrap_or_default();
        no_more(&p, "milvus")?;
        let mut store = Self::new(&p.web()).token(&token).database(&database);
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
        let own: Vec<(&str, String)> = if self.token.is_empty() { vec![] } else { vec![("authorization", format!("Bearer {}", self.token))] };
        self.http = self.settings.client("milvus", &own, &[&self.token]);
    }

    /// One call: the collection and database added, and Milvus's in-body code checked.
    async fn call(&self, path: &str, body: Value) -> Result<Value> {
        let mut payload = Map::new();
        payload.insert("collectionName".into(), json!(self.collection));
        payload.extend(object(Some(&body)));
        if !self.database.is_empty() {
            payload.insert("dbName".into(), json!(self.database));
        }
        let answer = self.http.json(Method::POST, &format!("{}/v2/vectordb/{path}", self.url), Some(&Value::Object(payload)), &[]).await?.body;
        // Milvus answers 200 to everything and says how it went in the body.
        let code = answer.get("code").cloned().unwrap_or(Value::Null);
        if !(code.is_null() || code.as_f64() == Some(0.0) || code.as_f64() == Some(200.0)) {
            let message = text_of(answer.get("message").unwrap_or(&Value::Null));
            let message: String = self.http.redact(&message).chars().take(300).collect();
            return Err(store_error("milvus", format!("milvus: {path} → {}: {message}", text_of(&code))));
        }
        Ok(answer.get("data").cloned().unwrap_or(Value::Null))
    }

    /// The filter as a Milvus boolean expression, over the `meta` JSON field.
    pub fn expression(wanted: &[Condition], ids: Option<&[String]>) -> String {
        let mut parts = Vec::new();
        if let Some(ids) = ids {
            parts.push(format!("id in [{}]", ids.iter().map(|i| literal(&json!(i))).collect::<Vec<_>>().join(", ")));
        }
        for c in wanted {
            let field = format!("meta[\"{}\"]", c.field);
            if c.op == "$in" {
                let options: Vec<String> = c.value.as_array().into_iter().flatten().map(literal).collect();
                parts.push(format!("{field} in [{}]", options.join(", ")));
            } else {
                parts.push(format!("{field} {} {}", operator(&c.op), literal(&c.value)));
            }
        }
        parts.join(" and ")
    }
}

#[async_trait]
impl VectorStore for MilvusStore {
    fn name(&self) -> &str {
        "milvus"
    }

    fn batch_size(&self) -> usize {
        500
    }

    async fn ensure(&self, dimension: usize) -> Result<()> {
        if self.call("collections/has", json!({})).await?.get("has").and_then(Value::as_bool).unwrap_or(false) {
            return Ok(());
        }
        let body = json!({
            "schema": {"autoId": false, "enableDynamicField": true, "fields": [
                {"fieldName": "id", "dataType": "VarChar", "isPrimary": true, "elementTypeParams": {"max_length": "512"}},
                {"fieldName": "vector", "dataType": "FloatVector", "elementTypeParams": {"dim": dimension.to_string()}},
                {"fieldName": "text", "dataType": "VarChar", "elementTypeParams": {"max_length": "65535"}},
                {"fieldName": "meta", "dataType": "JSON"},
            ]},
            "indexParams": [{"fieldName": "vector", "indexName": "vector", "metricType": "COSINE", "indexType": "AUTOINDEX"}],
        });
        self.call("collections/create", body).await?;
        Ok(())
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        let data: Vec<Value> = records
            .iter()
            .map(|r| json!({"id": r.id, "vector": floats(&r.vector), "text": r.text.chars().take(65000).collect::<String>(), "meta": r.metadata}))
            .collect();
        self.call("entities/upsert", json!({"data": data})).await?;
        Ok(())
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let mut body = json!({
            "data": [floats(vector)], "annsField": "vector", "limit": k, "outputFields": ["id", "text", "meta"],
            // Read what was just written, not a moment ago.
            "consistencyLevel": "Strong",
        });
        let filter = Self::expression(wanted, None);
        if !filter.is_empty() {
            body["filter"] = json!(filter);
        }
        let rows = self.call("entities/search", body).await?;
        Ok(rows
            .as_array()
            .into_iter()
            .flatten()
            .map(|row| {
                let metadata = match row.get("meta") {
                    Some(Value::String(s)) => serde_json::from_str::<Value>(s).ok().and_then(|v| v.as_object().cloned()).unwrap_or_default(),
                    other => object(other),
                };
                VectorHit {
                    id: row.get("id").map(text_of).unwrap_or_default(),
                    score: number(row.get("distance")),
                    text: row.get("text").and_then(Value::as_str).unwrap_or_default().to_string(),
                    metadata,
                }
            })
            .collect())
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        self.call("entities/delete", json!({"filter": Self::expression(wanted, ids)})).await?;
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let rows = self.call("entities/query", json!({"filter": "", "outputFields": ["count(*)"], "consistencyLevel": "Strong"})).await?;
        Ok(number(rows.as_array().and_then(|r| r.first()).and_then(|r| r.get("count(*)"))) as usize)
    }

    async fn clear(&self) -> Result<()> {
        self.call("collections/drop", json!({})).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::qdrant::fake;
    use super::*;
    use crate::knowledge::base::conditions;

    #[test]
    fn expressions() {
        let c = conditions(Some(&json!({"$and": [{"source": "faq.md"}, {"year": {"$gte": 2025}}, {"lang": {"$in": ["en", "fr"]}}]}))).unwrap();
        assert_eq!(MilvusStore::expression(&c, None), r#"meta["source"] == "faq.md" and meta["year"] >= 2025 and meta["lang"] in ["en", "fr"]"#);
        let ne = conditions(Some(&json!({"draft": {"$ne": true}}))).unwrap();
        assert_eq!(MilvusStore::expression(&ne, Some(&["a".into(), "b".into()])), r#"id in ["a", "b"] and meta["draft"] != true"#);
        assert_eq!(MilvusStore::expression(&[], None), "");
    }

    #[test]
    fn urls() {
        let m = MilvusStore::from_url(UrlParts::parse("zilliz+https://in03.zillizcloud.com/handbook?token=t0ken&database=db").unwrap(), false).unwrap();
        assert_eq!((m.url.as_str(), m.collection.as_str(), m.token.as_str(), m.database.as_str()), ("https://in03.zillizcloud.com", "handbook", "t0ken", "db"));
        let m = MilvusStore::from_url(UrlParts::default(), true).unwrap();
        assert_eq!((m.url.as_str(), m.collection.as_str()), ("http://localhost:19530", "knowledge"));
    }

    #[tokio::test]
    async fn requests_are_what_milvus_expects() {
        let fake = fake::serve(|s| match s.path.as_str() {
            "/v2/vectordb/collections/has" => (200, json!({"code": 0, "data": {"has": false}})),
            "/v2/vectordb/entities/search" => (200, json!({"code": 0, "data": [{"id": "doc#0", "distance": 0.9, "text": "T", "meta": {"year": 2026}}]})),
            "/v2/vectordb/entities/delete" => (200, json!({"code": 1100, "message": "bad filter root:Milvus"})),
            "/v2/vectordb/entities/query" => (200, json!({"code": 0, "data": [{"count(*)": 12}]})),
            _ => (200, json!({"code": 0, "data": {}})),
        })
        .await;
        let store = MilvusStore::new(&fake.url).collection("kb").token("root:Milvus").database("db1").retries(0);
        store.ensure(2).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen[0].body, json!({"collectionName": "kb", "dbName": "db1"}));
        assert_eq!(seen[1].path, "/v2/vectordb/collections/create");
        assert_eq!(seen[1].body["indexParams"][0]["metricType"], json!("COSINE"));
        assert_eq!(seen[1].body["schema"]["fields"][1]["elementTypeParams"], json!({"dim": "2"}));
        assert_eq!(seen[1].header("authorization"), "Bearer root:Milvus");

        store.upsert(vec![VectorRecord::new("doc#0", vec![1.0, 0.0], "T", json!({"year": 2026}))]).await.unwrap();
        assert_eq!(fake.seen()[2].body["data"], json!([{"id": "doc#0", "vector": [1.0, 0.0], "text": "T", "meta": {"year": 2026}}]));

        let hits = store.query(&[1.0, 0.0], 5, Some(&json!({"year": {"$gte": 2025}}))).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen[3].body["filter"], json!(r#"meta["year"] >= 2025"#));
        assert_eq!(seen[3].body["consistencyLevel"], json!("Strong"));
        assert_eq!(hits, vec![VectorHit { id: "doc#0".into(), score: 0.9, text: "T".into(), metadata: object(Some(&json!({"year": 2026}))) }]);

        // Milvus says 200 and reports the failure in the body; it is still a failure.
        let e = store.delete(Some(vec!["a".into()]), None).await.unwrap_err();
        assert!(e.to_string().contains("1100") && !e.to_string().contains("root:Milvus"), "{e}");
        assert_eq!(store.count().await.unwrap(), 12);
        store.clear().await.unwrap();
        assert_eq!(fake.seen().last().unwrap().path, "/v2/vectordb/collections/drop");
    }

    #[tokio::test]
    async fn live() {
        // e.g. http://localhost:19530; the token, if any, from $MILVUS_TOKEN.
        let Ok(url) = std::env::var("AH_LIVE_MILVUS_URL") else { return };
        let report = MilvusStore::new(&url).collection("ah_check").check(8).await;
        assert_eq!(report["ok"], json!(true), "{report}");
    }
}
