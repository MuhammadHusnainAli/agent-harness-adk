//! Amazon S3 Vectors, and retrieval from Amazon Bedrock Knowledge Bases.
//!
//! Amazon's other vector stores are databases this library already speaks to:
//! OpenSearch Service and OpenSearch Serverless (`OpenSearchStore`, signed with
//! SigV4), Aurora and RDS for PostgreSQL (`PgVectorStore`), and MemoryDB
//! (`RedisVectorStore`). SageMaker hosts models; it is not a vector store.
//!
//! Credentials come from the options, then the environment (`AWS_ACCESS_KEY_ID`,
//! `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`). Python also asks botocore —
//! instance roles, SSO, `~/.aws` profiles — when it is installed; here those
//! have to be exported into the environment first.

use async_trait::async_trait;
use reqwest::Method;
use serde_json::{Map, Value, json};

use super::UrlParts;
use super::opensearch::aws_signer;
use crate::errors::{Error, Result};
use crate::knowledge::base::{Condition, HttpClient, Reply, VectorHit, VectorRecord, VectorStore, conditions, holds, store_error};
use crate::knowledge::kb::Retriever;
use crate::llm_providers::sigv4;

const NAME: &str = "s3-vectors";

/// Region and credentials for a signed AWS service.
#[derive(Clone)]
pub struct AwsOptions {
    /// Else `AWS_REGION`, `AWS_DEFAULT_REGION`, then `us-east-1`.
    pub region: Option<String>,
    pub access_key: Option<String>,
    pub secret_key: Option<String>,
    pub session_token: Option<String>,
    pub timeout: f64,
    pub retries: u32,
    pub headers: Vec<(String, String)>,
}

impl Default for AwsOptions {
    fn default() -> Self {
        Self { region: None, access_key: None, secret_key: None, session_token: None, timeout: 30.0, retries: 3, headers: vec![] }
    }
}

fn signed_client(name: &'static str, service: &'static str, o: AwsOptions) -> (String, HttpClient) {
    let region = o.region.filter(|r| !r.is_empty()).unwrap_or_else(sigv4::default_region);
    let mut http = HttpClient::new(name, o.timeout, o.retries)
        .secret(o.secret_key.clone().unwrap_or_default())
        .secret(std::env::var("AWS_SECRET_ACCESS_KEY").unwrap_or_default());
    for (k, v) in o.headers {
        http = http.header(&k, v);
    }
    let keys = [o.access_key, o.secret_key, o.session_token].map(|k| k.filter(|v| !v.is_empty()));
    http.signer = Some(aws_signer(name, name, region.clone(), service, keys));
    (region, http)
}

/// Amazon S3 Vectors.
///
/// ```no_run
/// # use agent_harness::knowledge::stores::aws::{S3VectorsStore, AwsOptions};
/// let store = S3VectorsStore::new("my-vector-bucket", "handbook", "", AwsOptions { region: Some("eu-west-1".into()), ..Default::default() });
/// ```
///
/// The vector bucket is created in AWS; the index is created here if it is not
/// there. S3 Vectors answers with at most 30 results, and deletes by key — a
/// delete by filter lists the index to find the keys.
pub struct S3VectorsStore {
    bucket: String,
    index: String,
    url: String,
    region: String,
    http: HttpClient,
}

impl std::fmt::Debug for S3VectorsStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<S3VectorsStore {}/{} {}>", self.bucket, self.index, self.region)
    }
}

/// A filter as S3 Vectors writes it: the same operators, `$and` for several.
pub fn s3_filter(wanted: &[Condition]) -> Option<Value> {
    let mut parts: Vec<Value> = wanted.iter().map(|c| json!({c.field.clone(): {c.op.clone(): c.value}})).collect();
    match parts.len() {
        0 => None,
        1 => parts.pop(),
        _ => Some(json!({"$and": parts})),
    }
}

impl S3VectorsStore {
    /// `index` is `knowledge` in Python's default; `url` overrides
    /// `https://s3vectors.{region}.api.aws` when not empty.
    pub fn new(bucket: &str, index: &str, url: &str, options: AwsOptions) -> Self {
        let (region, http) = signed_client(NAME, "s3vectors", options);
        let url = if url.is_empty() { format!("https://s3vectors.{region}.api.aws") } else { url.trim_end_matches('/').to_string() };
        Self { bucket: bucket.into(), index: index.into(), url, region, http }
    }

    /// As `vector_store("s3-vectors://my-bucket/handbook?region=eu-west-1")` builds it.
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        if bare || p.netloc.is_empty() {
            return Err(Error::config("S3VectorsStore needs a vector bucket — s3-vectors://<bucket>/<index>?region=…"));
        }
        let mut o = AwsOptions::default();
        let mut index = if p.path.is_empty() { "knowledge".to_string() } else { p.path.clone() };
        let mut url = String::new();
        for (k, value) in std::mem::take(&mut p.given) {
            match k.as_str() {
                "index" => index = value,
                "url" => url = value,
                "region" => o.region = Some(value),
                "access_key" => o.access_key = Some(value),
                "secret_key" => o.secret_key = Some(value),
                "session_token" => o.session_token = Some(value),
                "timeout" => o.timeout = value.parse().map_err(|_| Error::config(format!("timeout is a number of seconds, not '{value}'")))?,
                "retries" => o.retries = value.parse().map_err(|_| Error::config(format!("retries is a whole number, not '{value}'")))?,
                other => return Err(Error::config(format!("S3VectorsStore takes no option '{other}'"))),
            }
        }
        Ok(Self::new(&p.netloc, &index, &url, o))
    }

    pub fn region(&self) -> &str {
        &self.region
    }

    async fn call(&self, operation: &str, ok: &[u16], body: Value) -> Result<Reply> {
        let mut full = json!({"vectorBucketName": self.bucket, "indexName": self.index});
        if let (Some(m), Value::Object(extra)) = (full.as_object_mut(), body) {
            m.extend(extra);
        }
        self.http.json(Method::POST, &format!("{}/{operation}", self.url), Some(&full), ok).await
    }

    async fn listed(&self, metadata: bool) -> Result<Vec<Value>> {
        let mut rows = vec![];
        let mut token: Option<String> = None;
        for _ in 0..100_000 {
            let mut body = json!({"maxResults": 500, "returnMetadata": metadata});
            if let Some(t) = &token {
                body["nextToken"] = json!(t);
            }
            let page = self.call("ListVectors", &[], body).await?.body;
            rows.extend(page["vectors"].as_array().cloned().unwrap_or_default());
            token = page["nextToken"].as_str().filter(|t| !t.is_empty()).map(str::to_string);
            if token.is_none() {
                break;
            }
        }
        Ok(rows)
    }
}

#[async_trait]
impl VectorStore for S3VectorsStore {
    fn name(&self) -> &str {
        NAME
    }

    fn batch_size(&self) -> usize {
        500
    }

    fn max_k(&self) -> usize {
        30
    }

    async fn ensure(&self, dimension: usize) -> Result<()> {
        let found = self.call("GetIndex", &[404], json!({})).await?;
        if found.status == 404 {
            let body = json!({"dataType": "float32", "dimension": dimension, "distanceMetric": "cosine",
                              "metadataConfiguration": {"nonFilterableMetadataKeys": ["_text"]}});
            self.call("CreateIndex", &[409], body).await?;
            return Ok(());
        }
        let held = &found.body["index"]["dimension"];
        let held = held.as_u64().or_else(|| held.as_str().and_then(|s| s.parse().ok())).unwrap_or(0);
        if held != 0 && held as usize != dimension {
            return Err(store_error(NAME, format!("the S3 Vectors index holds {held}-dimension vectors; {dimension} was asked for")));
        }
        Ok(())
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        let vectors: Vec<Value> = records
            .into_iter()
            .map(|r| {
                let mut meta: Map<String, Value> = r.metadata.into_iter().filter(|(_, v)| !v.is_null()).collect();
                meta.insert("_text".into(), json!(r.text));
                json!({"key": r.id, "data": {"float32": r.vector}, "metadata": meta})
            })
            .collect();
        self.call("PutVectors", &[], json!({"vectors": vectors})).await?;
        Ok(())
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let mut body = json!({"queryVector": {"float32": vector}, "topK": k, "returnMetadata": true, "returnDistance": true});
        if let Some(f) = s3_filter(wanted) {
            body["filter"] = f;
        }
        let answer = self.call("QueryVectors", &[], body).await?;
        let mut out = vec![];
        for row in answer.body["vectors"].as_array().into_iter().flatten() {
            let mut metadata = row["metadata"].as_object().cloned().unwrap_or_default();
            let text = match metadata.remove("_text") {
                Some(Value::String(s)) => s,
                _ => String::new(),
            };
            let Some(key) = row["key"].as_str() else {
                return Err(store_error(NAME, format!("S3 Vectors answered a vector with no key: {row}")));
            };
            out.push(VectorHit { id: key.into(), score: 1.0 - row["distance"].as_f64().unwrap_or(0.0), text, metadata });
        }
        Ok(out)
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        let keys: Vec<String> = if wanted.is_empty() {
            ids.map(<[String]>::to_vec).unwrap_or_default()
        } else {
            self.listed(true)
                .await?
                .into_iter()
                .filter_map(|row| {
                    let key = row["key"].as_str()?.to_string();
                    let meta = row["metadata"].as_object().cloned().unwrap_or_default();
                    (ids.is_none_or(|ids| ids.contains(&key)) && holds(&meta, wanted)).then_some(key)
                })
                .collect()
        };
        for part in keys.chunks(500) {
            self.call("DeleteVectors", &[], json!({"keys": part})).await?;
        }
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.listed(false).await?.len())
    }
}

/// Retrieval from a knowledge base that Amazon Bedrock manages.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use agent_harness::knowledge::stores::aws::{BedrockKnowledgeBase, AwsOptions};
/// # use agent_harness::knowledge::KnowledgeBase;
/// let kb = BedrockKnowledgeBase::new("ABCDEFGHIJ", "", AwsOptions { region: Some("us-east-1".into()), ..Default::default() });
/// let knowledge = KnowledgeBase::builder().retriever(Arc::new(kb));
/// ```
///
/// Bedrock does the chunking, the embedding and the storage; this asks it a
/// question. It is a retriever, not a store: there is nothing to upsert.
pub struct BedrockKnowledgeBase {
    knowledge_base_id: String,
    url: String,
    http: HttpClient,
}

impl std::fmt::Debug for BedrockKnowledgeBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<BedrockKnowledgeBase {}>", self.knowledge_base_id)
    }
}

/// A filter as Bedrock writes it: `equals`, `in`, `greaterThan`…, `andAll` for several.
pub fn bedrock_filter(wanted: &[Condition]) -> Option<Value> {
    let names = [
        ("$eq", "equals"),
        ("$ne", "notEquals"),
        ("$in", "in"),
        ("$gt", "greaterThan"),
        ("$gte", "greaterThanOrEquals"),
        ("$lt", "lessThan"),
        ("$lte", "lessThanOrEquals"),
    ];
    let mut parts: Vec<Value> = wanted
        .iter()
        .map(|c| {
            let name = names.iter().find(|(o, _)| *o == c.op).map_or("equals", |(_, n)| *n);
            json!({name: {"key": c.field, "value": c.value}})
        })
        .collect();
    match parts.len() {
        0 => None,
        1 => parts.pop(),
        _ => Some(json!({"andAll": parts})),
    }
}

impl BedrockKnowledgeBase {
    /// `url` overrides `https://bedrock-agent-runtime.{region}.amazonaws.com` when not empty.
    pub fn new(knowledge_base_id: &str, url: &str, options: AwsOptions) -> Self {
        let (region, http) = signed_client("bedrock-knowledge-base", "bedrock", options);
        let url = if url.is_empty() { format!("https://bedrock-agent-runtime.{region}.amazonaws.com") } else { url.trim_end_matches('/').to_string() };
        Self { knowledge_base_id: knowledge_base_id.into(), url, http }
    }
}

#[async_trait]
impl Retriever for BedrockKnowledgeBase {
    async fn retrieve(&self, query: &str, k: usize, filter: Option<&Value>) -> Result<Vec<VectorHit>> {
        let mut search = json!({"numberOfResults": k.clamp(1, 100)});
        if let Some(f) = bedrock_filter(&conditions(filter)?) {
            search["filter"] = f;
        }
        let body = json!({"retrievalQuery": {"text": query}, "retrievalConfiguration": {"vectorSearchConfiguration": search}});
        let answer = self.http.json(Method::POST, &format!("{}/knowledgebases/{}/retrieve", self.url, self.knowledge_base_id), Some(&body), &[]).await?;
        let mut out = vec![];
        for (n, row) in answer.body["retrievalResults"].as_array().into_iter().flatten().enumerate() {
            let place = row["location"].as_object().into_iter().flat_map(|m| m.values()).find_map(|v| {
                let v = v.as_object()?;
                v.get("uri").or_else(|| v.get("url")).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string)
            });
            let mut metadata = row["metadata"].as_object().cloned().unwrap_or_default();
            if let Some(place) = place {
                metadata.insert("source".into(), json!(place));
            }
            let id = match metadata.get("x-amz-bedrock-kb-chunk-id") {
                Some(Value::String(s)) if !s.is_empty() => s.clone(),
                Some(v) if !v.is_null() && v != &json!(false) && v != &json!(0) && v != &json!("") => v.to_string(),
                _ => n.to_string(),
            };
            out.push(VectorHit {
                id,
                score: row["score"].as_f64().unwrap_or(0.0),
                text: row["content"]["text"].as_str().unwrap_or_default().to_string(),
                metadata,
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

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

    fn opts() -> AwsOptions {
        AwsOptions {
            region: Some("eu-west-1".into()),
            access_key: Some("AKIDEXAMPLE".into()),
            secret_key: Some("not-a-real-secret".into()),
            retries: 0,
            ..Default::default()
        }
    }

    fn c(field: &str, op: &str, value: Value) -> Condition {
        Condition { field: field.into(), op: op.into(), value }
    }

    #[test]
    fn filters_translate() {
        assert_eq!(s3_filter(&[]), None);
        assert_eq!(s3_filter(&[c("a", "$eq", json!(1))]), Some(json!({"a": {"$eq": 1}})));
        assert_eq!(s3_filter(&[c("a", "$eq", json!(1)), c("b", "$in", json!(["x"]))]), Some(json!({"$and": [{"a": {"$eq": 1}}, {"b": {"$in": ["x"]}}]})));
        assert_eq!(
            bedrock_filter(&[c("a", "$gte", json!(2)), c("b", "$ne", json!("x"))]),
            Some(json!({"andAll": [{"greaterThanOrEquals": {"key": "a", "value": 2}}, {"notEquals": {"key": "b", "value": "x"}}]}))
        );
    }

    #[test]
    fn urls_are_read_like_python() {
        let s = S3VectorsStore::from_url(UrlParts::parse("s3-vectors://bucket/hand?region=ap-south-1").unwrap(), false).unwrap();
        assert_eq!((s.bucket.as_str(), s.index.as_str(), s.url.as_str()), ("bucket", "hand", "https://s3vectors.ap-south-1.api.aws"));
        assert!(S3VectorsStore::from_url(UrlParts { scheme: "s3".into(), ..Default::default() }, true).is_err());
        assert!(S3VectorsStore::from_url(UrlParts::parse("s3://b/i?bogus=1").unwrap(), false).is_err());
    }

    #[tokio::test]
    async fn requests_have_the_python_shape() {
        let (url, seen) = fake(Arc::new(|_, path, body| match path {
            "/GetIndex" => (404, r#"{"message": "not found"}"#.into()),
            "/QueryVectors" => (200, r#"{"vectors": [{"key": "a", "distance": 0.25, "metadata": {"n": 1, "_text": "hello"}}]}"#.into()),
            "/ListVectors" if !body.contains("nextToken") => {
                (200, r#"{"vectors": [{"key": "a", "metadata": {"n": 1}}, {"key": "b", "metadata": {"n": 2}}], "nextToken": "t2"}"#.into())
            }
            "/ListVectors" => (200, r#"{"vectors": [{"key": "c", "metadata": {"n": 1}}]}"#.into()),
            _ => (200, "{}".into()),
        }))
        .await;
        let s = S3VectorsStore::new("bucket", "hand", &url, opts());
        s.ensure(3).await.unwrap();
        s.upsert(vec![VectorRecord::new("a", vec![1.0, 0.5, 0.0], "hello", json!({"n": 1, "gone": null}))]).await.unwrap();
        let hits = s.query(&[1.0, 0.5, 0.0], 99, Some(&json!({"n": 1}))).await.unwrap();
        assert_eq!((hits[0].id.as_str(), hits[0].score, hits[0].text.as_str()), ("a", 0.75, "hello"));
        assert!(!hits[0].metadata.contains_key("_text"));
        s.delete(None, Some(&json!({"n": 1}))).await.unwrap();
        assert_eq!(s.count().await.unwrap(), 3);

        let seen = seen.lock().unwrap();
        let body = |i: usize| serde_json::from_str::<Value>(&seen[i].3).unwrap();
        let paths: Vec<&str> = seen.iter().map(|r| r.1.as_str()).collect();
        assert_eq!(
            paths,
            ["/GetIndex", "/CreateIndex", "/PutVectors", "/QueryVectors", "/ListVectors", "/ListVectors", "/DeleteVectors", "/ListVectors", "/ListVectors"]
        );
        assert_eq!(body(0), json!({"vectorBucketName": "bucket", "indexName": "hand"}));
        assert_eq!(
            body(1),
            json!({"vectorBucketName": "bucket", "indexName": "hand", "dataType": "float32", "dimension": 3, "distanceMetric": "cosine",
                   "metadataConfiguration": {"nonFilterableMetadataKeys": ["_text"]}})
        );
        assert_eq!(body(2)["vectors"], json!([{"key": "a", "data": {"float32": [1.0, 0.5, 0.0]}, "metadata": {"n": 1, "_text": "hello"}}]));
        assert_eq!(
            body(3),
            json!({"vectorBucketName": "bucket", "indexName": "hand", "queryVector": {"float32": [1.0, 0.5, 0.0]}, "topK": 30, "returnMetadata": true, "returnDistance": true, "filter": {"n": {"$eq": 1}}})
        );
        assert_eq!(body(4)["maxResults"], 500);
        assert_eq!(body(5)["nextToken"], "t2");
        assert_eq!(body(6)["keys"], json!(["a", "c"]));
        assert_eq!(body(7)["returnMetadata"], false);
        let auth = seen[0].2["authorization"].to_str().unwrap();
        assert!(auth.contains("/eu-west-1/s3vectors/aws4_request"), "{auth}");
        assert!(seen[0].2.contains_key("x-amz-content-sha256") && seen[0].2.contains_key("x-amz-date"));
    }

    #[tokio::test]
    async fn bedrock_retrieves() {
        let (url, seen) = fake(Arc::new(|_, _, _| {
            (
                200,
                r#"{"retrievalResults": [
                    {"content": {"text": "refunds take 5 days"}, "score": 0.8, "location": {"s3Location": {"uri": "s3://b/faq.md"}},
                     "metadata": {"x-amz-bedrock-kb-chunk-id": "chunk-1"}},
                    {"content": {"text": "second"}, "score": 0.5}]}"#
                    .into(),
            )
        }))
        .await;
        let kb = BedrockKnowledgeBase::new("KB1", &url, opts());
        let hits = kb.retrieve("refunds?", 500, Some(&json!({"lang": "en"}))).await.unwrap();
        assert_eq!((hits[0].id.as_str(), hits[0].metadata["source"].as_str()), ("chunk-1", Some("s3://b/faq.md")));
        assert_eq!((hits[1].id.as_str(), hits[1].text.as_str()), ("1", "second"));
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].1, "/knowledgebases/KB1/retrieve");
        assert_eq!(
            serde_json::from_str::<Value>(&seen[0].3).unwrap(),
            json!({"retrievalQuery": {"text": "refunds?"}, "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 100, "filter": {"equals": {"key": "lang", "value": "en"}}}}})
        );
        assert!(seen[0].2["authorization"].to_str().unwrap().contains("/eu-west-1/bedrock/aws4_request"));
    }

    #[tokio::test]
    async fn live() {
        let Ok(url) = std::env::var("AH_LIVE_S3_VECTORS_URL") else { return };
        let store = super::super::vector_store(&url).unwrap();
        let report = store.check(8).await;
        assert_eq!(report["ok"], true, "{report:#}");
    }
}
