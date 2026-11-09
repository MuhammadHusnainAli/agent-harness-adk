//! Upstash Vector.

use async_trait::async_trait;
use reqwest::Method;
use serde_json::{Value, json};

use super::UrlParts;
use super::qdrant::{Settings, floats, no_more, number, object, or_env, text_of};
use crate::errors::Result;
use crate::knowledge::base::{Condition, HttpClient, VectorHit, VectorRecord, VectorStore, holds, literal};
use crate::llm_providers::resilience::RetryPolicy;

/// `$eq` → `=`, and so on.
fn operator(op: &str) -> &'static str {
    match op {
        "$eq" => "=",
        "$ne" => "!=",
        "$gt" => ">",
        "$gte" => ">=",
        "$lt" => "<",
        _ => "<=",
    }
}

/// ```ignore
/// UpstashStore::new("https://x-y-z-vector.upstash.io").token("…").namespace("handbook");
/// ```
///
/// The index is created in the Upstash console, with the dimension of your
/// embedding model and the cosine metric.
#[derive(Debug, Clone)]
pub struct UpstashStore {
    pub url: String,
    pub token: String,
    pub namespace: String,
    settings: Settings,
    http: HttpClient,
}

impl Default for UpstashStore {
    /// The address and token from `$UPSTASH_VECTOR_REST_URL` and `$UPSTASH_VECTOR_REST_TOKEN`.
    fn default() -> Self {
        Self::new("")
    }
}

impl UpstashStore {
    /// An empty `url` is taken from `$UPSTASH_VECTOR_REST_URL`; the token comes
    /// from `$UPSTASH_VECTOR_REST_TOKEN` unless one is given.
    pub fn new(url: &str) -> Self {
        let settings = Settings::default();
        let mut store = Self {
            url: or_env(url, "UPSTASH_VECTOR_REST_URL").trim_end_matches('/').to_string(),
            token: or_env("", "UPSTASH_VECTOR_REST_TOKEN"),
            namespace: String::new(),
            http: settings.client("upstash", &[], &[]),
            settings,
        };
        store.rebuild();
        store
    }

    pub fn token(mut self, token: &str) -> Self {
        self.token = or_env(token, "UPSTASH_VECTOR_REST_TOKEN");
        self.rebuild();
        self
    }

    pub fn namespace(mut self, namespace: &str) -> Self {
        self.namespace = namespace.into();
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

    /// `upstash://x-y-z-vector.upstash.io?token=…&namespace=…` (always HTTPS),
    /// or the environment's for a bare `upstash`.
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        if bare {
            return Ok(Self::default());
        }
        let settings = Settings::from_url(&mut p, "upstash")?;
        let token = p.take("token").unwrap_or_default();
        let namespace = p.take("namespace").unwrap_or_default();
        no_more(&p, "upstash")?;
        let mut store = Self::new(&format!("https://{}", p.netloc)).token(&token).namespace(&namespace);
        store.settings = settings;
        store.rebuild();
        Ok(store)
    }

    fn rebuild(&mut self) {
        let own = [("authorization", format!("Bearer {}", self.token))];
        self.http = self.settings.client("upstash", &own, &[&self.token]);
    }

    fn path(&self, action: &str) -> String {
        if self.namespace.is_empty() { format!("{}/{action}", self.url) } else { format!("{}/{action}/{}", self.url, self.namespace) }
    }

    /// The filter as Upstash's SQL-like expression.
    pub fn expression(wanted: &[Condition]) -> String {
        let mut parts = Vec::new();
        for c in wanted {
            if c.op == "$in" {
                let options: Vec<String> = c.value.as_array().into_iter().flatten().map(|v| literal(v).replace('"', "'")).collect();
                parts.push(format!("{} IN ({})", c.field, options.join(", ")));
            } else {
                let written = match &c.value {
                    Value::String(s) => format!("'{}'", s.replace('\'', "\\'")),
                    other => literal(other),
                };
                parts.push(format!("{} {} {written}", c.field, operator(&c.op)));
            }
        }
        parts.join(" AND ")
    }
}

#[async_trait]
impl VectorStore for UpstashStore {
    fn name(&self) -> &str {
        "upstash"
    }

    fn batch_size(&self) -> usize {
        500
    }

    async fn ensure(&self, _dimension: usize) -> Result<()> {
        // There, and the key works.
        self.http.json(Method::GET, &format!("{}/info", self.url), None, &[]).await?;
        Ok(())
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        let rows: Vec<Value> = records.iter().map(|r| json!({"id": r.id, "vector": floats(&r.vector), "metadata": r.metadata, "data": r.text})).collect();
        self.http.json(Method::POST, &self.path("upsert"), Some(&Value::Array(rows)), &[]).await?;
        Ok(())
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let mut body = json!({"vector": floats(vector), "topK": k, "includeMetadata": true, "includeData": true});
        let filter = Self::expression(wanted);
        if !filter.is_empty() {
            body["filter"] = json!(filter);
        }
        let answer = self.http.json(Method::POST, &self.path("query"), Some(&body), &[]).await?.body;
        // Upstash scores cosine as (1 + cosine) / 2.
        Ok(answer
            .get("result")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|row| VectorHit {
                id: text_of(row.get("id").unwrap_or(&Value::Null)),
                score: 2.0 * number(row.get("score")) - 1.0,
                text: row.get("data").and_then(Value::as_str).unwrap_or_default().to_string(),
                metadata: object(row.get("metadata")),
            })
            .collect())
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        let mut filter = Self::expression(wanted);
        let mut listed: Vec<String> = ids.map(<[String]>::to_vec).unwrap_or_default();
        if ids.is_some() && !filter.is_empty() {
            // Ids and a filter together: fetch the ids, keep those that match.
            let fetched = self.http.json(Method::POST, &self.path("fetch"), Some(&json!({"ids": listed, "includeMetadata": true})), &[]).await?.body;
            listed = fetched
                .get("result")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|row| !row.is_null() && holds(&object(row.get("metadata")), wanted))
                .map(|row| text_of(row.get("id").unwrap_or(&Value::Null)))
                .collect();
            filter.clear();
        }
        let body = if filter.is_empty() { json!({"ids": listed}) } else { json!({"filter": filter}) };
        self.http.json(Method::POST, &self.path("delete"), Some(&body), &[]).await?;
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let info = self.http.json(Method::GET, &format!("{}/info", self.url), None, &[]).await?.body;
        let info = object(info.get("result"));
        if !self.namespace.is_empty() {
            let space = info.get("namespaces").and_then(|n| n.get(&self.namespace));
            return Ok(number(space.and_then(|s| s.get("vectorCount"))) as usize);
        }
        Ok(number(info.get("vectorCount")) as usize)
    }

    async fn clear(&self) -> Result<()> {
        self.http.json(Method::POST, &self.path("reset"), None, &[]).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::qdrant::fake;
    use super::*;
    use crate::knowledge::base::conditions;

    #[test]
    fn expressions_and_urls() {
        let c = conditions(Some(&json!({"$and": [{"source": "faq.md"}, {"year": {"$gte": 2025}}, {"lang": {"$in": ["en", "fr"]}}]}))).unwrap();
        assert_eq!(UpstashStore::expression(&c), "source = 'faq.md' AND year >= 2025 AND lang IN ('en', 'fr')");
        let q = conditions(Some(&json!({"who": {"$ne": "o'neil"}, "ok": true}))).unwrap();
        let e = UpstashStore::expression(&q);
        assert!(e.contains(r"who != 'o\'neil'") && e.contains("ok = true"), "{e}");
        let u = UpstashStore::from_url(UrlParts::parse("upstash://u.upstash.io?token=tok1&namespace=acme").unwrap(), false).unwrap();
        assert_eq!((u.url.as_str(), u.token.as_str(), u.namespace.as_str()), ("https://u.upstash.io", "tok1", "acme"));
        assert_eq!(u.path("query"), "https://u.upstash.io/query/acme");
    }

    #[tokio::test]
    async fn requests_are_what_upstash_expects() {
        let fake = fake::serve(|s| match s.path.as_str() {
            "/query/acme" => (200, json!({"result": [{"id": "doc#0", "score": 0.95, "data": "T", "metadata": {"year": 2026}}]})),
            "/fetch/acme" => (200, json!({"result": [{"id": "a", "metadata": {"year": 2026}}, null, {"id": "b", "metadata": {"year": 1999}}]})),
            "/info" => (200, json!({"result": {"vectorCount": 9, "namespaces": {"acme": {"vectorCount": 4}}}})),
            _ => (200, json!({"result": "Success"})),
        })
        .await;
        let store = UpstashStore::new(&fake.url).token("up-secret").namespace("acme").retries(0);
        store.ensure(2).await.unwrap();
        let record = VectorRecord::new("doc#0", vec![0.6, 0.8], "Refunds take five days.", json!({"source": "faq.md", "year": 2026}));
        store.upsert(vec![record]).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen[0].path, "/info");
        assert_eq!(seen[1].header("authorization"), "Bearer up-secret");
        assert_eq!(seen[1].path, "/upsert/acme");
        assert_eq!(
            seen[1].body,
            json!([{"id": "doc#0", "vector": [0.6, 0.8], "data": "Refunds take five days.", "metadata": {"source": "faq.md", "year": 2026}}])
        );

        let hits = store.query(&[1.0, 0.0], 3, Some(&json!({"source": "faq.md"}))).await.unwrap();
        assert_eq!(fake.seen()[2].body["filter"], json!("source = 'faq.md'"));
        assert!((hits[0].score - 0.9).abs() < 1e-9 && hits[0].text == "T");

        store.delete(Some(vec!["a".into(), "b".into()]), Some(&json!({"year": {"$gte": 2025}}))).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen[3].body, json!({"ids": ["a", "b"], "includeMetadata": true}));
        assert_eq!((seen[4].path.as_str(), &seen[4].body), ("/delete/acme", &json!({"ids": ["a"]})));
        store.delete(None, Some(&json!({"year": 2026}))).await.unwrap();
        assert_eq!(fake.seen()[5].body, json!({"filter": "year = 2026"}));

        assert_eq!(store.count().await.unwrap(), 4);
        store.clear().await.unwrap();
        let last = fake.seen().last().unwrap().clone();
        assert_eq!((last.method.as_str(), last.path.as_str(), last.raw.as_str()), ("POST", "/reset/acme", ""));
    }

    #[tokio::test]
    async fn live() {
        // The address and token from $UPSTASH_VECTOR_REST_URL and $UPSTASH_VECTOR_REST_TOKEN.
        if std::env::var("AH_LIVE_UPSTASH").is_err() {
            return;
        }
        let report = UpstashStore::default().namespace("ah_check").check(8).await;
        assert_eq!(report["ok"], json!(true), "{report}");
    }
}
