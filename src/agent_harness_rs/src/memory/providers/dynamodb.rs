//! DynamoDB memory. Serverless, and it scales without anyone watching it.
//!
//! Spoken to over the DynamoDB JSON API with SigV4 — no SDK. The items are
//! the Python `DynamoDBMemory`'s, so one table serves both.
//!
//! ```no_run
//! # use agent_harness::memory::providers::dynamodb::DynamoDbMemory;
//! let store = DynamoDbMemory::from_url("dynamodb://agent-memory?region=eu-west-1");
//! let local = DynamoDbMemory::from_url("dynamodb://agents?endpoint=http://localhost:8000&create=true");
//! ```
//!
//! **Options** on the URL, under the Python keyword names: `region`
//! (`region_name`), `endpoint` (`endpoint_url`), `aws_access_key_id`,
//! `aws_secret_access_key`, `aws_session_token`, `create`. Credentials and
//! region resolve as for [`S3Memory`](super::s3::S3Memory): explicit, the
//! environment, the AWS profile files; the endpoint also from
//! `AWS_ENDPOINT_URL_DYNAMODB` / `AWS_ENDPOINT_URL`. No instance-metadata,
//! ECS, SSO or assume-role provider.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Method;
use serde_json::{Map, Value, json};

use crate::errors::{Error, Result};
use crate::knowledge::base::HttpClient;
use crate::llm_providers::sigv4::{AwsCredentials, signer};
use crate::memory::base::{MemoryRecord, MemoryStore, RecordQuery, doc_key, last};
use crate::memory::providers::objects::{env, option, slug_of, split_url, truthy};
use crate::memory::providers::s3::{aws_credentials, aws_region};
use crate::memory::trace::Trace;
use crate::runtime::session::SessionStore;
use crate::sessions::dynamodb::DynamoDbSessionStore;
use crate::types::now;

/// The scopes `all()` reads when none is named, as Python's.
pub const SCOPES: [&str; 4] = ["user", "session", "orchestrator", "job"];

/// What DynamoDB said no with: its exception name (`ConditionalCheckFailedException`)
/// and message.
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    pub kind: String,
    pub message: String,
}

/// One table, partitioned by namespace and sorted by time.
///
/// Partition key `pk` is `{namespace}#{scope}`, sort key `sk` is `{ts}#{id}`
/// — so reading one user's memory is a single query, newest first, and
/// `limit` is served by DynamoDB. Documents sit under `pk = _docs`.
///
/// Cheap to clone: clones share the client and the "table is ready" flag.
#[derive(Debug, Clone)]
pub struct DynamoDbMemory {
    pub table_name: String,
    /// Create the table (on-demand billing) on first use if it is missing.
    pub create: bool,
    pub endpoint: String,
    http: HttpClient,
    ready: Arc<AtomicBool>,
}

impl DynamoDbMemory {
    /// A table, with the region, endpoint and credentials found in the
    /// environment and the AWS profile files.
    pub fn new(table: &str) -> Self {
        let region = aws_region(None);
        let endpoint =
            env("AWS_ENDPOINT_URL_DYNAMODB").or_else(|| env("AWS_ENDPOINT_URL")).unwrap_or_else(|| format!("https://dynamodb.{region}.amazonaws.com"));
        let mut me = Self {
            table_name: table.into(),
            create: false,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            http: HttpClient::new("dynamodb", 30.0, 3),
            ready: Arc::default(),
        };
        me.sign(aws_credentials(None, None, None), &region);
        me
    }

    fn sign(&mut self, credentials: AwsCredentials, region: &str) {
        self.http.secrets = [credentials.secret_key.clone()].into_iter().chain(credentials.token.clone()).filter(|s| !s.is_empty()).collect();
        self.http.signer = Some(signer(credentials, region.to_string(), "dynamodb", false));
    }

    /// Sign as these credentials, for this region (the endpoint stays).
    pub fn credentials(mut self, credentials: AwsCredentials, region: &str) -> Self {
        self.sign(credentials, region);
        self
    }

    pub fn endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = endpoint.trim_end_matches('/').to_string();
        self
    }

    pub fn create(mut self, create: bool) -> Self {
        self.create = create;
        self
    }

    pub fn retries(mut self, retries: u32) -> Self {
        self.http.retry.max_retries = retries;
        self
    }

    /// `dynamodb://table?region=…`. The table defaults to `agent_memory`.
    pub fn from_url(url: &str) -> Result<Self> {
        let parts = split_url(url);
        let o = &parts.query;
        let table = [parts.host.as_str(), parts.path.trim_start_matches('/')].into_iter().find(|t| !t.is_empty()).unwrap_or("agent_memory").to_string();
        let region = option(o, "region").or_else(|| option(o, "region_name"));
        let mut me = Self::new(&table);
        if let Some(e) = option(o, "endpoint").or_else(|| option(o, "endpoint_url")) {
            me = me.endpoint(e);
        } else if let Some(r) = region
            && env("AWS_ENDPOINT_URL_DYNAMODB").is_none()
            && env("AWS_ENDPOINT_URL").is_none()
        {
            me = me.endpoint(&format!("https://dynamodb.{r}.amazonaws.com"));
        }
        if region.is_some() || option(o, "aws_access_key_id").is_some() {
            let creds = aws_credentials(option(o, "aws_access_key_id"), option(o, "aws_secret_access_key"), option(o, "aws_session_token"));
            me = me.credentials(creds, &aws_region(region));
        }
        me.create = option(o, "create").is_some_and(truthy);
        Ok(me)
    }

    /// One API call. `Ok(Err(refusal))` is DynamoDB saying no (a 400 with an
    /// exception name); throttling is retried here before it gets that far.
    pub async fn call(&self, operation: &str, body: &Value) -> Result<std::result::Result<Value, Refusal>> {
        let bytes = serde_json::to_vec(body).unwrap_or_default();
        let target = [("x-amz-target", format!("DynamoDB_20120810.{operation}"))];
        let mut attempt = 0u32;
        loop {
            let reply =
                self.http.send(Method::POST, &format!("{}/", self.endpoint), Some(bytes.clone()), Some("application/x-amz-json-1.0"), &target, &[400]).await?;
            if reply.status != 400 {
                return Ok(Ok(reply.body));
            }
            let kind = reply.body["__type"].as_str().unwrap_or("").rsplit('#').next().unwrap_or("").to_string();
            let message = reply.body["message"].as_str().or_else(|| reply.body["Message"].as_str()).unwrap_or("").to_string();
            let throttled = matches!(kind.as_str(), "ProvisionedThroughputExceededException" | "ThrottlingException" | "RequestLimitExceeded");
            if throttled && attempt < self.http.retry.max_retries {
                tokio::time::sleep(Duration::from_secs_f64(self.http.retry.backoff(attempt))).await;
                attempt += 1;
                continue;
            }
            return Ok(Err(Refusal { kind, message }));
        }
    }

    /// [`call`](Self::call), with a refusal as an error.
    pub async fn request(&self, operation: &str, body: Value) -> Result<Value> {
        self.call(operation, &body).await?.map_err(|r| Error::VectorStore {
            message: format!("dynamodb: {operation} on {} refused: {}: {}", self.table_name, r.kind, r.message),
            store: "dynamodb".into(),
            status: Some(400),
            retryable: false,
        })
    }

    /// Create the table first, when asked to and not done yet.
    pub async fn ensure(&self) -> Result<()> {
        if !self.create || self.ready.load(Ordering::Acquire) {
            return Ok(());
        }
        let made = self
            .call(
                "CreateTable",
                &json!({
                    "TableName": self.table_name,
                    "KeySchema": [{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}],
                    "AttributeDefinitions": [{"AttributeName": "pk", "AttributeType": "S"}, {"AttributeName": "sk", "AttributeType": "S"}],
                    "BillingMode": "PAY_PER_REQUEST"
                }),
            )
            .await?;
        if let Err(r) = made
            && r.kind != "ResourceInUseException"
        {
            return Err(Error::VectorStore {
                message: format!("dynamodb: could not create {}: {}: {}", self.table_name, r.kind, r.message),
                store: "dynamodb".into(),
                status: Some(400),
                retryable: false,
            });
        }
        // A new table takes a moment to become ACTIVE; Python does not wait, and its first write can fail.
        for _ in 0..60 {
            let described = self.call("DescribeTable", &json!({"TableName": self.table_name})).await?;
            if described.as_ref().is_ok_and(|d| d["Table"]["TableStatus"].as_str().is_none_or(|s| s == "ACTIVE")) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        self.ready.store(true, Ordering::Release);
        Ok(())
    }

    /// The namespace partition: the trace's, else the record's own, else `_shared`.
    pub fn namespace(trace: Option<&Trace>, record: Option<&MemoryRecord>) -> String {
        match (trace, record) {
            (Some(t), _) => t.slug(),
            (None, Some(r)) => slug_of(r),
            (None, None) => "_shared".into(),
        }
    }

    fn sort_key(record: &MemoryRecord) -> String {
        format!("{:.6}#{}", record.ts, record.id)
    }

    async fn query_scope(&self, partition: String, limit: Option<usize>) -> Result<Vec<Value>> {
        let mut items = Vec::new();
        let mut start: Option<Value> = None;
        loop {
            let mut body = json!({
                "TableName": self.table_name,
                "KeyConditionExpression": "pk = :pk",
                "ExpressionAttributeValues": {":pk": {"S": partition}},
                "ScanIndexForward": false
            });
            if let Some(n) = limit {
                body["Limit"] = json!(n - items.len().min(n));
            }
            if let Some(s) = start.take() {
                body["ExclusiveStartKey"] = s;
            }
            let page = self.request("Query", body).await?;
            items.extend(page["Items"].as_array().cloned().unwrap_or_default());
            match page.get("LastEvaluatedKey").filter(|k| k.is_object()) {
                Some(k) if limit.is_none_or(|n| items.len() < n) => start = Some(k.clone()),
                _ => return Ok(items),
            }
        }
    }
}

#[async_trait]
impl MemoryStore for DynamoDbMemory {
    async fn append(&self, record: MemoryRecord) -> Result<MemoryRecord> {
        self.ensure().await?;
        let item = json!({
            "pk": {"S": format!("{}#{}", Self::namespace(None, Some(&record)), record.scope)},
            "sk": {"S": Self::sort_key(&record)},
            "body": {"S": serde_json::to_string(&record).unwrap_or_default()},
            "kind": {"S": record.kind.clone()},
        });
        self.request("PutItem", json!({"TableName": self.table_name, "Item": item})).await?;
        Ok(record)
    }

    async fn all(&self, query: RecordQuery) -> Result<Vec<MemoryRecord>> {
        self.ensure().await?;
        let namespace = Self::namespace(query.trace.as_ref(), None);
        let scopes: Vec<String> = match &query.scope {
            Some(s) if !s.is_empty() => vec![s.clone()],
            _ => SCOPES.iter().map(|s| s.to_string()).collect(),
        };
        let limit = query.limit.filter(|n| *n > 0);
        let per_scope = if query.kind.is_none() { limit } else { None };
        let pages = futures::future::join_all(scopes.iter().map(|s| self.query_scope(format!("{namespace}#{s}"), per_scope))).await;
        let mut rows = Vec::new();
        for items in pages {
            for item in items? {
                if let Some(r) = item["body"]["S"].as_str().and_then(|b| serde_json::from_str::<MemoryRecord>(b).ok()) {
                    rows.push(r);
                }
            }
        }
        rows.retain(|r| query.kind.as_ref().is_none_or(|k| &r.kind == k) && query.trace.as_ref().is_none_or(|t| t.matches(r)));
        rows.sort_by(|a, b| a.ts.total_cmp(&b.ts));
        Ok(last(rows, limit))
    }

    /// The records of this namespace (only `_shared` without a trace, as in
    /// Python), and — unlike Python, which leaves them — the trace's documents.
    async fn clear(&self, scope: Option<&str>, trace: Option<&Trace>) -> Result<()> {
        let rows = self.all(RecordQuery { scope: scope.map(str::to_string), trace: trace.cloned(), ..Default::default() }).await?;
        let namespace = Self::namespace(trace, None);
        for r in rows {
            let key = json!({"pk": {"S": format!("{namespace}#{}", r.scope)}, "sk": {"S": Self::sort_key(&r)}});
            self.request("DeleteItem", json!({"TableName": self.table_name, "Key": key})).await?;
        }
        if let Some(t) = trace.filter(|t| !t.is_empty()) {
            // Query again until nothing is left: each pass deletes what it found.
            loop {
                let page = self
                    .request(
                        "Query",
                        json!({"TableName": self.table_name, "KeyConditionExpression": "pk = :pk AND begins_with(sk, :p)",
                               "ExpressionAttributeValues": {":pk": {"S": "_docs"}, ":p": {"S": format!("{}/", t.slug())}}}),
                    )
                    .await?;
                let found = page["Items"].as_array().cloned().unwrap_or_default();
                if found.is_empty() {
                    break;
                }
                for item in found {
                    self.request("DeleteItem", json!({"TableName": self.table_name, "Key": {"pk": item["pk"], "sk": item["sk"]}})).await?;
                }
            }
        }
        Ok(())
    }

    async fn read_doc(&self, name: &str, trace: Option<&Trace>) -> Result<String> {
        self.ensure().await?;
        let found = self.request("GetItem", json!({"TableName": self.table_name, "Key": {"pk": {"S": "_docs"}, "sk": {"S": doc_key(name, trace)}}})).await?;
        Ok(found["Item"]["body"]["S"].as_str().unwrap_or("").to_string())
    }

    async fn write_doc(&self, name: &str, text: &str, trace: Option<&Trace>) -> Result<()> {
        self.ensure().await?;
        let item = json!({"pk": {"S": "_docs"}, "sk": {"S": doc_key(name, trace)}, "body": {"S": text}, "updated": {"N": now().to_string()}});
        self.request("PutItem", json!({"TableName": self.table_name, "Item": item})).await?;
        Ok(())
    }

    fn describe(&self) -> String {
        format!("dynamodb://{}", self.table_name)
    }

    /// Sessions in the same table (see [`DynamoDbSessionStore`]).
    fn sessions(&self, _options: &Map<String, Value>) -> Option<Result<Arc<dyn SessionStore>>> {
        Some(Ok(Arc::new(DynamoDbSessionStore::new(self.clone()))))
    }
}

#[cfg(test)]
pub(crate) mod fake {
    //! Enough of DynamoDB to keep the stores honest: conditional puts with
    //! ALL_OLD, the 400 KB item limit, begins_with, newest-first queries with
    //! a Limit, and tiny pages so paging is exercised.

    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use axum::Json;
    use axum::http::{HeaderMap, StatusCode};
    use axum::response::{IntoResponse, Response};

    use super::*;
    use crate::memory::providers::objects::testing::serve;

    pub type Items = Arc<Mutex<BTreeMap<(String, String), Value>>>;

    fn refuse(kind: &str) -> Response {
        (StatusCode::BAD_REQUEST, Json(json!({"__type": format!("com.amazonaws.dynamodb.v20120810#{kind}"), "message": kind}))).into_response()
    }

    fn key_of(v: &Value) -> (String, String) {
        (v["pk"]["S"].as_str().unwrap_or("").to_string(), v["sk"]["S"].as_str().unwrap_or("").to_string())
    }

    fn size(item: &Value) -> usize {
        item.as_object()
            .map_or(0, |o| o.iter().map(|(k, v)| k.len() + v.as_object().and_then(|m| m.values().next()).map_or(0, |x| x.as_str().map_or(8, str::len))).sum())
    }

    async fn handle(items: Items, headers: HeaderMap, raw: axum::body::Bytes) -> Response {
        let body: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
        let auth = headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("");
        if !auth.starts_with("AWS4-HMAC-SHA256 Credential=test/") || !auth.contains("x-amz-target") {
            return (StatusCode::FORBIDDEN, Json(json!({"__type": "UnrecognizedClientException"}))).into_response();
        }
        let op = headers.get("x-amz-target").and_then(|v| v.to_str().ok()).unwrap_or("").trim_start_matches("DynamoDB_20120810.").to_string();
        let mut all = items.lock().unwrap();
        match op.as_str() {
            "CreateTable" => Json(json!({})).into_response(),
            "DescribeTable" => Json(json!({"Table": {"TableStatus": "ACTIVE"}})).into_response(),
            "PutItem" => {
                let item = body["Item"].clone();
                assert!(size(&item) <= 400_000, "an item may not exceed 400 KB");
                let key = key_of(&item);
                let held = all.get(&key).cloned();
                let values = &body["ExpressionAttributeValues"];
                match body["ConditionExpression"].as_str() {
                    Some("attribute_not_exists(pk)") if held.is_some() => return refuse("ConditionalCheckFailedException"),
                    Some("#v = :expect") if held.as_ref().is_none_or(|h| h["version"] != values[":expect"]) => {
                        return refuse("ConditionalCheckFailedException");
                    }
                    _ => {}
                }
                all.insert(key, item);
                match held {
                    Some(h) if body["ReturnValues"] == "ALL_OLD" => Json(json!({"Attributes": h})).into_response(),
                    _ => Json(json!({})).into_response(),
                }
            }
            "GetItem" => match all.get(&key_of(&body["Key"])) {
                Some(item) => Json(json!({"Item": item})).into_response(),
                None => Json(json!({})).into_response(),
            },
            "DeleteItem" => {
                all.remove(&key_of(&body["Key"]));
                Json(json!({})).into_response()
            }
            "Query" => {
                let values = &body["ExpressionAttributeValues"];
                let pk = values[":pk"]["S"].as_str().unwrap_or("").to_string();
                let begins = values[":p"]["S"].as_str().unwrap_or("").to_string();
                let mut rows: Vec<&Value> = all.iter().filter(|((p, s), _)| *p == pk && s.starts_with(&begins)).map(|(_, v)| v).collect();
                if body["ScanIndexForward"] == false {
                    rows.reverse();
                }
                let start = body["ExclusiveStartKey"].get("n").and_then(Value::as_u64).unwrap_or(0) as usize;
                let take = body["Limit"].as_u64().map_or(2, |l| (l as usize).min(2));
                let page: Vec<Value> = rows.iter().skip(start).take(take).map(|v| (*v).clone()).collect();
                let mut out = json!({"Items": page});
                if start + take < rows.len() {
                    out["LastEvaluatedKey"] = json!({"n": start + take});
                }
                Json(out).into_response()
            }
            _ => refuse("UnknownOperationException"),
        }
    }

    /// A table on a fake server, and the items it holds.
    pub async fn table() -> (DynamoDbMemory, Items) {
        let items: Items = Arc::default();
        let shared = items.clone();
        let router = axum::Router::new().fallback(move |h: HeaderMap, b: axum::body::Bytes| handle(shared.clone(), h, b));
        let base = serve(router).await;
        let store = DynamoDbMemory::new("agents").endpoint(&base).credentials(AwsCredentials::new("test", "secret", None), "us-east-1").create(true).retries(0);
        (store, items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitions_by_namespace_and_sorts_by_time() {
        let mut bob = MemoryRecord::new("user", "fact", "x");
        Trace::user("bob").stamp(&mut bob);
        assert_eq!(DynamoDbMemory::namespace(Some(&Trace::user("alice")), None), "u-alice");
        assert_eq!(DynamoDbMemory::namespace(None, Some(&bob)), "u-bob");
        assert_eq!(DynamoDbMemory::namespace(None, None), "_shared");
        bob.ts = 1000.5;
        assert!(DynamoDbMemory::sort_key(&bob).starts_with("1000.500000#mem_"));
        let store = DynamoDbMemory::from_url("dynamodb://chats?region=eu-west-1&create=1").unwrap();
        assert_eq!((store.table_name.as_str(), store.create), ("chats", true));
        assert_eq!(DynamoDbMemory::from_url("dynamodb://").unwrap().table_name, "agent_memory");
    }

    #[tokio::test]
    async fn memory_in_a_table_is_scoped_by_trace() {
        let (store, items) = fake::table().await;
        let (alice, bob) = (Trace::user("alice"), Trace::user("bob"));
        for (t, text, ts) in [(&alice, "alice's fact about refunds", 1000.0), (&alice, "alice's second", 1001.0), (&bob, "bob's fact", 1002.0)] {
            let mut r = MemoryRecord::new("user", "fact", text);
            r.ts = ts;
            t.stamp(&mut r);
            store.append(r).await.unwrap();
        }
        assert!(items.lock().unwrap().keys().any(|(pk, sk)| pk == "u-alice#user" && sk.starts_with("1000.000000#")));
        let texts = |rows: Vec<MemoryRecord>| rows.into_iter().map(|r| r.text).collect::<Vec<_>>();
        assert_eq!(texts(store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap()), ["alice's fact about refunds", "alice's second"]);
        assert_eq!(texts(store.all(RecordQuery::default().traced(Some(&alice)).limit(1)).await.unwrap()), ["alice's second"]);
        assert_eq!(texts(store.all(RecordQuery::scope("user").traced(Some(&bob))).await.unwrap()), ["bob's fact"]);
        assert_eq!(store.search("refunds", None, 5, Some(&alice)).await.unwrap()[0].text, "alice's fact about refunds");

        store.write_doc("user.md", "- alice", Some(&alice)).await.unwrap();
        store.write_doc("user.md", "- bob", Some(&bob)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "- alice");
        assert_eq!(store.read_doc("missing.md", Some(&alice)).await.unwrap(), "");
        store.clear(None, Some(&alice)).await.unwrap();
        assert!(store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap().is_empty());
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "");
        assert_eq!(store.read_doc("user.md", Some(&bob)).await.unwrap(), "- bob");
        assert_eq!(store.all(RecordQuery::default().traced(Some(&bob))).await.unwrap().len(), 1);
    }

    /// `AH_LIVE_DYNAMODB_TABLE` (and `AH_LIVE_DYNAMODB_ENDPOINT` for DynamoDB
    /// Local) to run against a real table, created if missing.
    #[tokio::test]
    async fn a_real_table_holds_the_contract() {
        let Some(table) = env("AH_LIVE_DYNAMODB_TABLE") else { return };
        let mut url = format!("dynamodb://{table}?create=true");
        if let Some(e) = env("AH_LIVE_DYNAMODB_ENDPOINT") {
            url.push_str(&format!("&endpoint={e}"));
        }
        let store = DynamoDbMemory::from_url(&url).unwrap();
        let trace = Trace::user(crate::types::new_id("live"));
        let mut r = MemoryRecord::new("user", "fact", "live");
        trace.stamp(&mut r);
        store.append(r).await.unwrap();
        assert_eq!(store.all(RecordQuery::default().traced(Some(&trace))).await.unwrap().len(), 1);
        store.clear(None, Some(&trace)).await.unwrap();
        crate::memory::providers::objects::testing::exercise_sessions(store.sessions(&Map::new()).unwrap().unwrap().as_ref(), true).await;
    }
}
