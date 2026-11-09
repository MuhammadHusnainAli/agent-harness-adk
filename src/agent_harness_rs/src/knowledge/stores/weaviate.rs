//! Weaviate — self-hosted or Weaviate Cloud.

use async_trait::async_trait;
use reqwest::Method;
use serde_json::{Map, Value, json};

use super::UrlParts;
use super::qdrant::{Settings, floats, no_more, number, object, or_env, py_dumps, text_of};
use crate::errors::Result;
use crate::knowledge::base::{Condition, HttpClient, VectorHit, VectorRecord, VectorStore, as_uuid, store_error};
use crate::llm_providers::resilience::RetryPolicy;

/// Every character outside `[0-9A-Za-z_]` becomes `_`.
fn cleaned(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect()
}

/// `$eq` → `Equal`, and so on.
fn operator(op: &str) -> &'static str {
    match op {
        "$eq" => "Equal",
        "$ne" => "NotEqual",
        "$gt" => "GreaterThan",
        "$gte" => "GreaterThanEqual",
        "$lt" => "LessThan",
        _ => "LessThanEqual",
    }
}

fn is_number(v: &Value) -> bool {
    v.is_number()
}

/// ```ignore
/// WeaviateStore::new("http://localhost:8080").collection("Handbook");
/// WeaviateStore::new("https://xyz.weaviate.cloud").api_key("…").collection("Handbook");
/// ```
///
/// Metadata is kept twice: whole, as it was given, and field by field (as
/// `m_<name>`) so it can be filtered on.
#[derive(Debug, Clone)]
pub struct WeaviateStore {
    pub url: String,
    /// The class name: cleaned, and starting with a capital.
    pub collection: String,
    pub api_key: String,
    settings: Settings,
    http: HttpClient,
}

impl Default for WeaviateStore {
    fn default() -> Self {
        Self::new("http://localhost:8080")
    }
}

impl WeaviateStore {
    /// The key comes from `$WEAVIATE_API_KEY` unless one is given.
    pub fn new(url: &str) -> Self {
        let settings = Settings::default();
        let mut store = Self {
            url: url.trim_end_matches('/').to_string(),
            collection: "Knowledge".into(),
            api_key: or_env("", "WEAVIATE_API_KEY"),
            http: settings.client("weaviate", &[], &[]),
            settings,
        };
        store.rebuild();
        store
    }

    /// Weaviate's class names start with a capital, and hold only `[0-9A-Za-z_]`.
    pub fn collection(mut self, collection: &str) -> Self {
        let mut name = cleaned(collection);
        if name.is_empty() {
            name = "Knowledge".into();
        }
        let mut chars = name.chars();
        let head = chars.next().map(|c| c.to_ascii_uppercase()).unwrap_or('K');
        self.collection = std::iter::once(head).chain(chars).collect();
        self
    }

    pub fn api_key(mut self, key: &str) -> Self {
        self.api_key = or_env(key, "WEAVIATE_API_KEY");
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

    /// `weaviate://host:8080/Collection?api_key=…`, or the defaults for a bare `weaviate`.
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        if bare {
            return Ok(Self::default());
        }
        let settings = Settings::from_url(&mut p, "weaviate")?;
        let named = p.take("collection");
        let key = p.take("api_key").unwrap_or_default();
        no_more(&p, "weaviate")?;
        let mut store = Self::new(&p.web()).api_key(&key);
        store.settings = settings;
        store.rebuild();
        if !p.path.is_empty() {
            store = store.collection(&p.path.clone());
        } else if let Some(c) = named {
            store = store.collection(&c);
        }
        Ok(store)
    }

    fn rebuild(&mut self) {
        let own: Vec<(&str, String)> = if self.api_key.is_empty() { vec![] } else { vec![("authorization", format!("Bearer {}", self.api_key))] };
        self.http = self.settings.client("weaviate", &own, &[&self.api_key]);
    }

    /// The property a metadata field is kept in.
    pub fn field(name: &str) -> String {
        format!("m_{}", cleaned(name))
    }

    /// The filter as GraphQL, which is how Weaviate is asked.
    pub fn graphql_filter(wanted: &[Condition]) -> Result<String> {
        fn value(v: &Value) -> String {
            match v {
                Value::Bool(b) => format!("valueBoolean: {}", if *b { "true" } else { "false" }),
                // Weaviate's own schema calls every number it is given a `number`.
                Value::Number(n) => format!("valueNumber: {n}"),
                other => format!("valueText: {}", py_dumps(&text_of(other))),
            }
        }
        let mut parts = Vec::new();
        for c in wanted {
            let path = format!("path: [\"{}\"]", Self::field(&c.field));
            if c.op == "$in" {
                let options = c.value.as_array().cloned().unwrap_or_default();
                if options.is_empty() {
                    return Err(store_error("weaviate", "$in needs at least one value"));
                }
                let numbers = options.iter().all(is_number);
                let kind = if numbers { "valueNumber" } else { "valueText" };
                let items: Vec<String> = options.iter().map(|v| if numbers { py_dumps(v) } else { py_dumps(&text_of(v)) }).collect();
                parts.push(format!("{{{path}, operator: ContainsAny, {kind}: [{}]}}", items.join(", ")));
            } else {
                parts.push(format!("{{{path}, operator: {}, {}}}", operator(&c.op), value(&c.value)));
            }
        }
        Ok(match parts.len() {
            0 => String::new(),
            1 => parts.remove(0),
            _ => format!("{{operator: And, operands: [{}]}}", parts.join(", ")),
        })
    }

    /// The same filter as JSON, which is how a batch delete takes it.
    pub fn rest_filter(wanted: &[Condition]) -> Value {
        let one = |c: &Condition| -> Value {
            let path = json!([Self::field(&c.field)]);
            if c.op == "$in" {
                let options = c.value.as_array().cloned().unwrap_or_default();
                let numbers = options.iter().all(is_number);
                // In JSON, a list of values goes under the …Array spelling.
                if numbers {
                    json!({"path": path, "operator": "ContainsAny", "valueNumberArray": options})
                } else {
                    json!({"path": path, "operator": "ContainsAny", "valueTextArray": options.iter().map(text_of).collect::<Vec<_>>()})
                }
            } else {
                let (kind, value) = match &c.value {
                    Value::Bool(_) => ("valueBoolean", c.value.clone()),
                    Value::Number(_) => ("valueNumber", c.value.clone()),
                    other => ("valueText", Value::String(text_of(other))),
                };
                json!({"path": path, "operator": operator(&c.op), kind: value})
            }
        };
        let mut parts: Vec<Value> = wanted.iter().map(one).collect();
        if parts.len() == 1 { parts.remove(0) } else { json!({"operator": "And", "operands": parts}) }
    }

    async fn graphql(&self, query: &str) -> Result<Value> {
        let answer = self.http.json(Method::POST, &format!("{}/v1/graphql", self.url), Some(&json!({"query": query})), &[]).await?.body;
        if let Some(errors) = answer.get("errors").filter(|e| !e.is_null() && *e != &json!([]) && *e != &json!({})) {
            let text: String = py_dumps(errors).chars().take(400).collect();
            return Err(store_error("weaviate", format!("weaviate: {text}")));
        }
        Ok(answer.get("data").cloned().unwrap_or_else(|| json!({})))
    }
}

/// A property Weaviate can hold as it is: a string, number or boolean, or a list of strings.
fn plain(v: &Value) -> bool {
    match v {
        Value::String(_) | Value::Number(_) | Value::Bool(_) => true,
        Value::Array(items) => items.iter().all(Value::is_string),
        _ => false,
    }
}

#[async_trait]
impl VectorStore for WeaviateStore {
    fn name(&self) -> &str {
        "weaviate"
    }

    fn batch_size(&self) -> usize {
        200
    }

    async fn ensure(&self, _dimension: usize) -> Result<()> {
        if self.http.json(Method::GET, &format!("{}/v1/schema/{}", self.url, self.collection), None, &[404]).await?.status != 404 {
            return Ok(());
        }
        let body = json!({
            "class": self.collection, "vectorizer": "none",
            "vectorIndexConfig": {"distance": "cosine"},
            "properties": [
                {"name": "ref", "dataType": ["text"]},
                {"name": "text", "dataType": ["text"]},
                {"name": "meta", "dataType": ["text"], "indexFilterable": false, "indexSearchable": false},
            ],
        });
        self.http.json(Method::POST, &format!("{}/v1/schema", self.url), Some(&body), &[422]).await?;
        Ok(())
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        let objects: Vec<Value> = records
            .iter()
            .map(|r| {
                let mut properties: Map<String, Value> = r.metadata.iter().filter(|(_, v)| plain(v)).map(|(k, v)| (Self::field(k), v.clone())).collect();
                properties.insert("ref".into(), json!(r.id));
                properties.insert("text".into(), json!(r.text));
                properties.insert("meta".into(), json!(py_dumps(&r.metadata)));
                json!({"class": self.collection, "id": as_uuid(&r.id), "vector": floats(&r.vector), "properties": properties})
            })
            .collect();
        let answer = self.http.json(Method::POST, &format!("{}/v1/batch/objects", self.url), Some(&json!({"objects": objects})), &[]).await?.body;
        for item in answer.as_array().into_iter().flatten() {
            let error = item.get("result").and_then(|r| r.get("errors")).and_then(|e| e.get("error"));
            if let Some(error) = error.filter(|e| !e.is_null() && *e != &json!([])) {
                let text: String = py_dumps(error).chars().take(300).collect();
                return Err(store_error("weaviate", format!("weaviate refused a record: {text}")));
            }
        }
        Ok(())
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let filter = Self::graphql_filter(wanted)?;
        let filter = if filter.is_empty() { String::new() } else { format!(", where: {filter}") };
        let query = format!(
            "{{ Get {{ {}(nearVector: {{vector: {}}}, limit: {k}{filter}) {{ ref text meta _additional {{ id distance }} }} }} }}",
            self.collection,
            py_dumps(vector)
        );
        let data = self.graphql(&query).await?;
        let rows = data.get("Get").and_then(|g| g.get(&self.collection)).and_then(Value::as_array).cloned().unwrap_or_default();
        Ok(rows
            .iter()
            .map(|row| {
                let extra = object(row.get("_additional"));
                let meta = row
                    .get("meta")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(|s| serde_json::from_str::<Value>(s).ok().and_then(|v| v.as_object().cloned()).unwrap_or_default())
                    .unwrap_or_default();
                let id = match row.get("ref").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                    Some(r) => r.to_string(),
                    None => extra.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
                };
                VectorHit {
                    id,
                    score: 1.0 - number(extra.get("distance")),
                    text: row.get("text").and_then(Value::as_str).unwrap_or_default().to_string(),
                    metadata: meta,
                }
            })
            .collect())
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        let by_id = |ids: &[String]| json!({"path": ["id"], "operator": "ContainsAny", "valueTextArray": ids.iter().map(|i| as_uuid(i)).collect::<Vec<_>>()});
        let filter = match ids {
            Some(ids) if wanted.is_empty() => by_id(ids),
            Some(ids) => json!({"operator": "And", "operands": [Self::rest_filter(wanted), by_id(ids)]}),
            None => Self::rest_filter(wanted),
        };
        let body = json!({"match": {"class": self.collection, "where": filter}});
        // A batch delete removes at most 10,000 at a time; go round until none match.
        for _ in 0..1000 {
            let answer = self.http.json(Method::DELETE, &format!("{}/v1/batch/objects", self.url), Some(&body), &[]).await?.body;
            let results = object(answer.get("results"));
            let matches = number(results.get("matches"));
            if matches == 0.0 || matches <= number(results.get("successful")) + number(results.get("failed")) {
                return Ok(());
            }
        }
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let data = self.graphql(&format!("{{ Aggregate {{ {} {{ meta {{ count }} }} }} }}", self.collection)).await?;
        let rows = data.get("Aggregate").and_then(|a| a.get(&self.collection)).and_then(Value::as_array).cloned().unwrap_or_default();
        Ok(number(rows.first().and_then(|r| r.get("meta")).and_then(|m| m.get("count"))) as usize)
    }

    async fn clear(&self) -> Result<()> {
        self.http.json(Method::DELETE, &format!("{}/v1/schema/{}", self.url, self.collection), None, &[404]).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::qdrant::fake;
    use super::*;
    use crate::knowledge::base::conditions;

    fn wanted() -> Value {
        json!({"source": "faq.md", "year": {"$gte": 2025}, "lang": {"$in": ["en", "fr"]}})
    }

    #[test]
    fn names_and_filters() {
        assert_eq!(WeaviateStore::new("http://w").collection("hand book").collection, "Hand_book");
        assert_eq!(WeaviateStore::new("http://w").collection("").collection, "Knowledge");
        assert_eq!(WeaviateStore::field("a-b"), "m_a_b");
        let c = conditions(Some(&wanted())).unwrap();
        let q = WeaviateStore::graphql_filter(&c).unwrap();
        assert!(q.contains(r#"path: ["m_source"], operator: Equal, valueText: "faq.md""#), "{q}");
        assert!(q.contains(r#"path: ["m_year"], operator: GreaterThanEqual, valueNumber: 2025"#), "{q}");
        assert!(q.contains(r#"operator: ContainsAny, valueText: ["en", "fr"]"#) && q.starts_with("{operator: And, operands: ["), "{q}");
        let one = conditions(Some(&json!({"draft": {"$ne": true}}))).unwrap();
        assert_eq!(WeaviateStore::graphql_filter(&one).unwrap(), r#"{path: ["m_draft"], operator: NotEqual, valueBoolean: true}"#);
        assert!(WeaviateStore::graphql_filter(&conditions(Some(&json!({"a": {"$in": []}}))).unwrap()).is_err());
        let rest = WeaviateStore::rest_filter(&conditions(Some(&json!({"n": {"$in": [1, 2]}}))).unwrap());
        assert_eq!(rest, json!({"path": ["m_n"], "operator": "ContainsAny", "valueNumberArray": [1, 2]}));
        let rest = WeaviateStore::rest_filter(&c);
        assert_eq!(rest["operator"], json!("And"));
        assert!(rest["operands"].as_array().unwrap().contains(&json!({"path": ["m_lang"], "operator": "ContainsAny", "valueTextArray": ["en", "fr"]})));
        assert!(rest["operands"].as_array().unwrap().contains(&json!({"path": ["m_year"], "operator": "GreaterThanEqual", "valueNumber": 2025})));
    }

    #[test]
    fn urls() {
        let w = WeaviateStore::from_url(UrlParts::parse("weaviate://localhost:8080/handbook").unwrap(), false).unwrap();
        assert_eq!((w.url.as_str(), w.collection.as_str()), ("http://localhost:8080", "Handbook"));
        let w = WeaviateStore::from_url(UrlParts::default(), true).unwrap();
        assert_eq!(w.collection, "Knowledge");
    }

    #[tokio::test]
    async fn requests_are_what_weaviate_expects() {
        let fake = fake::serve(|s| match (s.method.as_str(), s.path.as_str()) {
            ("GET", _) => (404, json!({})),
            ("POST", "/v1/batch/objects") => (200, json!([{"result": {}}])),
            ("POST", "/v1/graphql") if s.body["query"].as_str().unwrap().contains("Aggregate") => {
                (200, json!({"data": {"Aggregate": {"Handbook": [{"meta": {"count": 3}}]}}}))
            }
            ("POST", "/v1/graphql") => (
                200,
                json!({"data": {"Get": {"Handbook": [{"ref": "doc#0", "text": "T", "meta": "{\"year\": 2026}", "_additional": {"id": "u", "distance": 0.25}}]}}}),
            ),
            ("DELETE", "/v1/batch/objects") => (200, json!({"results": {"matches": 2, "successful": 2, "failed": 0}})),
            _ => (200, json!({})),
        })
        .await;
        let store = WeaviateStore::new(&fake.url).collection("handbook").api_key("wv-key").retries(0);
        store.ensure(2).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen[1].path, "/v1/schema");
        assert_eq!(seen[1].body["class"], json!("Handbook"));
        assert_eq!(seen[1].body["vectorIndexConfig"], json!({"distance": "cosine"}));
        assert_eq!(seen[0].header("authorization"), "Bearer wv-key");

        let record = VectorRecord::new("doc#0", vec![0.6, 0.8], "Refunds take five days.", json!({"source": "faq.md", "year": 2026, "deep": {"x": 1}}));
        store.upsert(vec![record]).await.unwrap();
        let seen = fake.seen();
        let item = &seen[2].body["objects"][0];
        assert_eq!(item["class"], json!("Handbook"));
        assert_eq!(item["id"], json!(as_uuid("doc#0")));
        assert_eq!(item["properties"]["m_source"], json!("faq.md"));
        assert!(item["properties"].get("m_deep").is_none());
        let meta: Value = serde_json::from_str(item["properties"]["meta"].as_str().unwrap()).unwrap();
        assert_eq!(meta, json!({"source": "faq.md", "year": 2026, "deep": {"x": 1}}));

        let hits = store.query(&[1.0, 0.0], 4, Some(&wanted())).await.unwrap();
        let query = fake.seen()[3].body["query"].as_str().unwrap().to_string();
        assert!(query.starts_with("{ Get { Handbook(nearVector: {vector: [1.0, 0.0]}, limit: 4, where: {operator: And"), "{query}");
        assert!(query.ends_with("{ ref text meta _additional { id distance } } } }"), "{query}");
        assert_eq!(hits, vec![VectorHit { id: "doc#0".into(), score: 0.75, text: "T".into(), metadata: object(Some(&json!({"year": 2026}))) }]);

        store.delete(Some(vec!["doc#0".into()]), None).await.unwrap();
        let seen = fake.seen();
        assert_eq!(seen[4].method, "DELETE");
        assert_eq!(
            seen[4].body,
            json!({"match": {"class": "Handbook", "where": {"path": ["id"], "operator": "ContainsAny", "valueTextArray": [as_uuid("doc#0")]}}})
        );
        assert_eq!(seen.len(), 5, "everything matched was removed, so once is enough");

        assert_eq!(store.count().await.unwrap(), 3);
        store.clear().await.unwrap();
        let seen = fake.seen();
        assert_eq!((seen[6].method.as_str(), seen[6].path.as_str()), ("DELETE", "/v1/schema/Handbook"));
    }

    #[tokio::test]
    async fn graphql_errors_are_errors() {
        let fake = fake::serve(|_| (200, json!({"errors": [{"message": "no class"}]}))).await;
        let e = WeaviateStore::new(&fake.url).retries(0).count().await.unwrap_err();
        assert!(e.to_string().contains("no class"), "{e}");
    }

    #[tokio::test]
    async fn live() {
        // e.g. http://localhost:8080; the key, if any, from $WEAVIATE_API_KEY.
        let Ok(url) = std::env::var("AH_LIVE_WEAVIATE_URL") else { return };
        let report = WeaviateStore::new(&url).collection("AhCheck").check(8).await;
        assert_eq!(report["ok"], json!(true), "{report}");
    }
}
