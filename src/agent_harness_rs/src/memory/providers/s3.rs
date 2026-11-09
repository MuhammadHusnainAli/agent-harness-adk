//! Amazon S3 memory (and anything S3-compatible: MinIO, R2, Spaces).
//!
//! Spoken to over the S3 REST API with SigV4 — no SDK. The keys are the
//! Python `S3Memory`'s (see [`objects`](super::objects)), so one bucket serves
//! both.
//!
//! ```no_run
//! # use agent_harness::memory::providers::s3::S3Memory;
//! let store = S3Memory::from_url("s3://my-bucket/agent-memory?region=eu-west-1");
//! let minio = S3Memory::from_url("s3://my-bucket?endpoint=http://localhost:9000");
//! ```
//!
//! **Options** ride on the URL, under the names the Python keyword arguments
//! use: `region` (`region_name`), `endpoint` (`endpoint_url`),
//! `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`,
//! `addressing_style` (`path` or `virtual`), `prefix`, `concurrency`,
//! `conditional`.
//!
//! **Credentials**: the explicit ones, then `AWS_ACCESS_KEY_ID` /
//! `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`, then the profile
//! (`AWS_PROFILE`, else `default`) in `~/.aws/credentials`. The region: the
//! explicit one, `AWS_REGION` / `AWS_DEFAULT_REGION`, the profile's in
//! `~/.aws/config`, else `us-east-1`. The endpoint: the explicit one,
//! `AWS_ENDPOINT_URL_S3`, `AWS_ENDPOINT_URL`, else AWS. Unlike botocore there
//! is no instance-metadata, ECS, SSO or assume-role provider, and a bucket in
//! another region is not followed by redirect — name the region.
//!
//! A custom endpoint is addressed path-style (`{endpoint}/{bucket}/{key}`),
//! which is what MinIO wants; AWS is addressed virtual-hosted unless the
//! bucket name has a dot in it.

use async_trait::async_trait;
use base64::Engine as _;
use reqwest::Method;

use crate::errors::{Error, Result};
use crate::knowledge::base::HttpClient;
use crate::llm_providers::sigv4::{AwsCredentials, default_region, signer};
use crate::memory::providers::objects::{
    Fetched, ObjectBackend, ObjectStoreMemory, PutIf, encode, env, option, send_raw, split_url, xml_escape, xml_value, xml_values,
};

/// Memory in an S3 bucket: [`ObjectStoreMemory`] over an [`S3Bucket`].
pub type S3Memory = ObjectStoreMemory<S3Bucket>;

/// One S3 bucket, and how to reach and sign for it.
#[derive(Debug, Clone)]
pub struct S3Bucket {
    pub bucket: String,
    pub region: String,
    /// A custom endpoint (MinIO, R2, LocalStack); `None` for AWS.
    pub endpoint: Option<String>,
    /// Address as `{endpoint}/{bucket}` rather than `{bucket}.{endpoint}`.
    pub path_style: bool,
    credentials: AwsCredentials,
    http: HttpClient,
}

impl S3Bucket {
    /// A bucket with the region, endpoint and credentials found in the
    /// environment and the AWS profile files.
    pub fn new(bucket: &str) -> Self {
        let endpoint = env("AWS_ENDPOINT_URL_S3").or_else(|| env("AWS_ENDPOINT_URL"));
        let mut me = Self {
            bucket: bucket.into(),
            region: aws_region(None),
            path_style: endpoint.is_some() || bucket.contains('.'),
            endpoint: endpoint.map(|e| e.trim_end_matches('/').to_string()),
            credentials: aws_credentials(None, None, None),
            http: HttpClient::new("s3", 60.0, 3),
        };
        me.sign();
        me
    }

    fn sign(&mut self) {
        self.http.secrets = [self.credentials.secret_key.clone()].into_iter().chain(self.credentials.token.clone()).filter(|s| !s.is_empty()).collect();
        self.http.signer = Some(signer(self.credentials.clone(), self.region.clone(), "s3", true));
    }

    pub fn region(mut self, region: &str) -> Self {
        self.region = region.into();
        self.sign();
        self
    }

    /// A custom endpoint, e.g. `http://localhost:9000` for MinIO. Turns on
    /// path-style addressing.
    pub fn endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = Some(endpoint.trim_end_matches('/').to_string());
        self.path_style = true;
        self
    }

    pub fn path_style(mut self, path_style: bool) -> Self {
        self.path_style = path_style;
        self
    }

    pub fn credentials(mut self, credentials: AwsCredentials) -> Self {
        self.credentials = credentials;
        self.sign();
        self
    }

    /// Requests that may succeed in a moment are tried this many more times.
    pub fn retries(mut self, retries: u32) -> Self {
        self.http.retry.max_retries = retries;
        self
    }

    /// The URL of the bucket itself, for listing and batch deletes.
    fn bucket_url(&self) -> String {
        let virtual_host = |scheme: &str, host: &str| format!("{scheme}://{}.{host}/", self.bucket);
        match (&self.endpoint, self.path_style) {
            (Some(e), true) => format!("{e}/{}", encode(&self.bucket, false)),
            (Some(e), false) => {
                let (scheme, host) = e.split_once("://").unwrap_or(("https", e));
                virtual_host(scheme, host)
            }
            (None, true) => format!("https://s3.{}.amazonaws.com/{}", self.region, encode(&self.bucket, false)),
            (None, false) => virtual_host("https", &format!("s3.{}.amazonaws.com", self.region)),
        }
    }

    fn object_url(&self, key: &str) -> String {
        let base = self.bucket_url();
        format!("{}/{}", base.trim_end_matches('/'), encode(key, true))
    }
}

#[async_trait]
impl ObjectBackend for S3Bucket {
    async fn put(&self, key: &str, body: Vec<u8>, content_type: &str, condition: &PutIf) -> Result<bool> {
        let mut headers = vec![("content-type", content_type.to_string())];
        match condition {
            PutIf::Always => {}
            PutIf::Absent => headers.push(("if-none-match", "*".into())),
            PutIf::Matches(tag) => headers.push(("if-match", tag.clone())),
        }
        let ok: &[u16] = if *condition == PutIf::Always { &[] } else { &[409, 412] };
        let reply = send_raw(&self.http, Method::PUT, &self.object_url(key), body, &headers, ok).await?;
        // 412: the condition failed. 409: another conditional write won the race.
        Ok(!matches!(reply.status, 409 | 412))
    }

    async fn get(&self, key: &str) -> Result<Option<Fetched>> {
        let reply = send_raw(&self.http, Method::GET, &self.object_url(key), vec![], &[], &[404]).await?;
        if reply.status == 404 {
            return Ok(None);
        }
        let tag = reply.header("etag");
        Ok(Some(Fetched { body: reply.body, tag }))
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let mut keys = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut query = format!("list-type=2&max-keys=1000&prefix={}", encode(prefix, false));
            if let Some(t) = &token {
                query.push_str(&format!("&continuation-token={}", encode(t, false)));
            }
            let reply = send_raw(&self.http, Method::GET, &format!("{}?{query}", self.bucket_url()), vec![], &[], &[]).await?;
            let xml = reply.text();
            keys.extend(xml_values(&xml, "Key"));
            token = xml_value(&xml, "NextContinuationToken");
            if xml_value(&xml, "IsTruncated").as_deref() != Some("true") || token.is_none() {
                return Ok(keys);
            }
        }
    }

    async fn delete(&self, keys: &[String]) -> Result<()> {
        for chunk in keys.chunks(1000) {
            // The API's batch ceiling.
            let objects: String = chunk.iter().map(|k| format!("<Object><Key>{}</Key></Object>", xml_escape(k))).collect();
            let body = format!("<Delete><Quiet>true</Quiet>{objects}</Delete>").into_bytes();
            let md5 = base64::engine::general_purpose::STANDARD.encode(md5(&body));
            let headers = [("content-type", "application/xml".to_string()), ("content-md5", md5)];
            let reply = send_raw(&self.http, Method::POST, &format!("{}?delete", self.bucket_url()), body, &headers, &[]).await?;
            let xml = reply.text();
            let failed: Vec<String> = xml_values(&xml, "Error").into_iter().filter(|e| xml_value(e, "Code").as_deref() != Some("NoSuchKey")).collect();
            if let Some(first) = failed.first() {
                return Err(Error::VectorStore {
                    message: format!("s3: {} of {} objects were not deleted: {}", failed.len(), chunk.len(), xml_value(first, "Message").unwrap_or_default()),
                    store: "s3".into(),
                    status: None,
                    retryable: false,
                });
            }
        }
        Ok(())
    }

    fn describe(&self) -> String {
        format!("s3://{}", self.bucket)
    }
}

impl S3Memory {
    /// `s3://bucket/prefix?region=…` (or `minio://…`). The prefix defaults to
    /// `agent-memory`.
    pub fn from_url(url: &str) -> Result<Self> {
        let parts = split_url(url);
        if parts.host.is_empty() {
            return Err(Error::config(format!("S3Memory needs a bucket: s3://bucket/prefix (got '{url}')")));
        }
        let o = &parts.query;
        let mut bucket = S3Bucket::new(&parts.host);
        if let Some(e) = option(o, "endpoint").or_else(|| option(o, "endpoint_url")) {
            bucket = bucket.endpoint(e);
        }
        if let Some(r) = option(o, "region").or_else(|| option(o, "region_name")) {
            bucket = bucket.region(r);
        }
        match option(o, "addressing_style") {
            Some("path") => bucket = bucket.path_style(true),
            Some("virtual") => bucket = bucket.path_style(false),
            _ => {}
        }
        if option(o, "aws_access_key_id").is_some() {
            bucket = bucket.credentials(aws_credentials(option(o, "aws_access_key_id"), option(o, "aws_secret_access_key"), option(o, "aws_session_token")));
        }
        let prefix = parts.path.trim_matches('/');
        let prefix = if prefix.is_empty() { option(o, "prefix").unwrap_or("agent-memory") } else { prefix };
        Ok(S3Memory::new(bucket, prefix).configured(o))
    }
}

// ---- the AWS profile files ---------------------------------------------------------

fn home() -> Option<std::path::PathBuf> {
    env("HOME").or_else(|| env("USERPROFILE")).map(std::path::PathBuf::from)
}

/// The keys of one `[section]` of an INI file.
fn ini_section(path: Option<std::path::PathBuf>, section: &str) -> Vec<(String, String)> {
    let Some(text) = path.and_then(|p| std::fs::read_to_string(p).ok()) else { return vec![] };
    let mut inside = false;
    let mut out = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') && line.ends_with(']') {
            inside = line[1..line.len() - 1].trim() == section;
        } else if inside && let Some((k, v)) = line.split_once('=') {
            out.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    out
}

fn profile() -> String {
    env("AWS_PROFILE").unwrap_or_else(|| "default".into())
}

/// Explicit keys, then the environment, then the profile in
/// `~/.aws/credentials` (or `AWS_SHARED_CREDENTIALS_FILE`).
pub(crate) fn aws_credentials(access_key: Option<&str>, secret_key: Option<&str>, token: Option<&str>) -> AwsCredentials {
    let found = AwsCredentials::resolve(access_key, secret_key, token);
    if found.is_set() {
        return found;
    }
    let file = env("AWS_SHARED_CREDENTIALS_FILE").map(Into::into).or_else(|| home().map(|h| h.join(".aws").join("credentials")));
    let keys = ini_section(file, &profile());
    let get = |k: &str| option(&keys, k).map(str::to_string);
    match (get("aws_access_key_id"), get("aws_secret_access_key")) {
        (Some(a), Some(s)) => AwsCredentials::new(a, s, get("aws_session_token")),
        _ => found,
    }
}

/// Explicit, then `AWS_REGION` / `AWS_DEFAULT_REGION`, then the profile's
/// region in `~/.aws/config` (or `AWS_CONFIG_FILE`), else `us-east-1`.
pub(crate) fn aws_region(explicit: Option<&str>) -> String {
    if let Some(r) = explicit.filter(|r| !r.is_empty()) {
        return r.into();
    }
    if env("AWS_REGION").is_some() || env("AWS_DEFAULT_REGION").is_some() {
        return default_region();
    }
    let name = profile();
    let section = if name == "default" { name } else { format!("profile {name}") };
    let file = env("AWS_CONFIG_FILE").map(Into::into).or_else(|| home().map(|h| h.join(".aws").join("config")));
    option(&ini_section(file, &section), "region").map_or_else(default_region, str::to_string)
}

// ---- MD5, which DeleteObjects insists on -------------------------------------------

/// MD5 of `data` (RFC 1321). Only for the `Content-MD5` header S3 requires on
/// a batch delete — never for anything that needs to be secure.
pub fn md5(data: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4,
        11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64).map(|i| ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32).collect();
    let mut state: [u32; 4] = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_le_bytes());
    for block in message.chunks(64) {
        let m: Vec<u32> = block.chunks(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let [mut a, mut b, mut c, mut d] = state;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let rotated = a.wrapping_add(f).wrapping_add(k[i]).wrapping_add(m[g]).rotate_left(S[i]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(rotated);
        }
        state = [state[0].wrapping_add(a), state[1].wrapping_add(b), state[2].wrapping_add(c), state[3].wrapping_add(d)];
    }
    let mut out = [0u8; 16];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use axum::body::Bytes;
    use axum::http::{HeaderMap, Method as AxMethod, StatusCode, Uri};
    use axum::response::{IntoResponse, Response};
    use serde_json::Map;

    use super::*;
    use crate::memory::base::MemoryStore;
    use crate::memory::providers::objects::testing::{exercise_memory, exercise_sessions, query_of, serve};
    use crate::memory::providers::objects::{decode, option};

    type Objects = Arc<Mutex<(BTreeMap<String, (Vec<u8>, String)>, u64)>>;

    /// Enough of S3 (path-style) to drive the store for real: signed requests,
    /// paged listings, ETags with If-Match / If-None-Match, batch deletes that
    /// check their Content-MD5.
    async fn fake_s3(state: Objects, method: AxMethod, uri: Uri, headers: HeaderMap, body: Bytes) -> Response {
        let auth = headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("");
        if !auth.starts_with("AWS4-HMAC-SHA256 Credential=test/") || !headers.contains_key("x-amz-content-sha256") {
            return (StatusCode::FORBIDDEN, "<Error><Code>AccessDenied</Code></Error>").into_response();
        }
        let path = decode(uri.path());
        let Some(rest) = path.strip_prefix("/bucket") else { return StatusCode::NOT_FOUND.into_response() };
        let key = rest.trim_start_matches('/').to_string();
        let query = query_of(&uri);
        let mut s = state.lock().unwrap();
        match (method, key.is_empty()) {
            (AxMethod::GET, true) => {
                let prefix = option(&query, "prefix").unwrap_or("").to_string();
                let start: usize = option(&query, "continuation-token").and_then(|t| t.strip_prefix("p+/=")).and_then(|t| t.parse().ok()).unwrap_or(0);
                let keys: Vec<&String> = s.0.keys().filter(|k| k.starts_with(&prefix)).collect();
                let page: String = keys.iter().skip(start).take(2).map(|k| format!("<Contents><Key>{}</Key></Contents>", xml_escape(k))).collect();
                let more = start + 2 < keys.len();
                let token = if more { format!("<NextContinuationToken>p+/={}</NextContinuationToken>", start + 2) } else { String::new() };
                format!("<ListBucketResult><IsTruncated>{more}</IsTruncated>{page}{token}</ListBucketResult>").into_response()
            }
            (AxMethod::POST, true) if query.iter().any(|(k, _)| k == "delete") => {
                let md5_header = headers.get("content-md5").and_then(|v| v.to_str().ok()).unwrap_or("");
                if md5_header != base64::engine::general_purpose::STANDARD.encode(md5(&body)) {
                    return (StatusCode::BAD_REQUEST, "<Error><Code>InvalidDigest</Code></Error>").into_response();
                }
                for k in xml_values(&String::from_utf8_lossy(&body), "Key") {
                    s.0.remove(&k);
                }
                "<DeleteResult></DeleteResult>".into_response()
            }
            (AxMethod::PUT, false) => {
                let held = s.0.get(&key).map(|(_, t)| t.clone());
                let header = |n: &str| headers.get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
                if header("if-none-match").as_deref() == Some("*") && held.is_some() {
                    return StatusCode::PRECONDITION_FAILED.into_response();
                }
                if let Some(want) = header("if-match")
                    && held.as_deref() != Some(want.as_str())
                {
                    return StatusCode::PRECONDITION_FAILED.into_response();
                }
                s.1 += 1;
                let tag = format!("\"etag-{}\"", s.1);
                s.0.insert(key, (body.to_vec(), tag.clone()));
                ([("etag", tag)], "").into_response()
            }
            (AxMethod::GET, false) => match s.0.get(&key) {
                Some((b, t)) => ([("etag", t.clone())], b.clone()).into_response(),
                None => (StatusCode::NOT_FOUND, "<Error><Code>NoSuchKey</Code></Error>").into_response(),
            },
            _ => StatusCode::METHOD_NOT_ALLOWED.into_response(),
        }
    }

    async fn fake() -> (String, Objects) {
        let state: Objects = Arc::default();
        let shared = state.clone();
        let router = axum::Router::new().fallback(move |m: AxMethod, u: Uri, h: HeaderMap, b: Bytes| fake_s3(shared.clone(), m, u, h, b));
        (serve(router).await, state)
    }

    #[test]
    fn md5_matches_the_rfc_vectors() {
        let hex = |d: [u8; 16]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(md5(b"")), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(md5(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(hex(md5(&[b'a'; 1000])), "cabe45dcc9ae5b66ba86600cca6b8ba8");
    }

    #[test]
    fn urls_address_the_bucket_as_s3_expects() {
        let b = S3Bucket::new("my-bucket").region("eu-west-1");
        let b = S3Bucket { endpoint: None, path_style: false, ..b };
        assert_eq!(b.object_url("mem/a b.json"), "https://my-bucket.s3.eu-west-1.amazonaws.com/mem/a%20b.json");
        let dotted = S3Bucket { bucket: "my.bucket".into(), path_style: true, ..b.clone() };
        assert_eq!(dotted.object_url("k"), "https://s3.eu-west-1.amazonaws.com/my.bucket/k");
        let minio = b.endpoint("http://localhost:9000/");
        assert_eq!(minio.object_url("k"), "http://localhost:9000/my-bucket/k");
        let store = S3Memory::from_url("minio://bkt?endpoint=http://127.0.0.1:9000&region_name=us-west-2&concurrency=4").unwrap();
        assert_eq!((store.prefix.as_str(), store.concurrency, store.objects().region.as_str()), ("agent-memory", 4, "us-west-2"));
        assert!(S3Memory::from_url("s3://").is_err());
    }

    fn store_at(endpoint: &str, prefix: &str) -> S3Memory {
        let bucket = S3Bucket::new("bucket").endpoint(endpoint).region("us-east-1").credentials(AwsCredentials::new("test", "secret", None)).retries(0);
        S3Memory::new(bucket, prefix)
    }

    #[tokio::test]
    async fn memory_over_the_s3_api_is_scoped_by_trace() {
        let (endpoint, state) = fake().await;
        let store = store_at(&endpoint, "mem");
        let mut r = crate::memory::MemoryRecord::new("user", "fact", "alice's fact");
        crate::memory::Trace::user("alice").stamp(&mut r);
        store.append(r).await.unwrap();
        assert!(state.lock().unwrap().0.keys().next().unwrap().starts_with("mem/records/u-alice/user/"));
        exercise_memory(&store).await;
        assert!(state.lock().unwrap().0.is_empty());
    }

    #[tokio::test]
    async fn sessions_over_the_s3_api_hold_the_contract() {
        let (endpoint, state) = fake().await;
        let store = store_at(&endpoint, "chats");
        let sessions = store.sessions(&Map::new()).unwrap().unwrap();
        exercise_sessions(sessions.as_ref(), true).await;
        assert!(state.lock().unwrap().0.is_empty());
    }

    #[tokio::test]
    async fn a_request_without_credentials_is_refused_before_it_is_sent() {
        let (endpoint, _) = fake().await;
        let bucket = S3Bucket::new("bucket").endpoint(&endpoint).credentials(AwsCredentials::default()).retries(0);
        let err = S3Memory::new(bucket, "mem").read_doc("x", None).await.unwrap_err();
        assert!(err.to_string().contains("no AWS credentials"), "{err}");
    }

    /// `AH_LIVE_S3_BUCKET` (and `AH_LIVE_S3_ENDPOINT` for MinIO) to run
    /// against a real bucket with the ambient credentials.
    #[tokio::test]
    async fn a_real_bucket_holds_the_contract() {
        let Some(bucket) = env("AH_LIVE_S3_BUCKET") else { return };
        let mut url = format!("s3://{bucket}/ah-rust-test-{}", crate::types::new_id(""));
        if let Some(e) = env("AH_LIVE_S3_ENDPOINT") {
            url.push_str(&format!("?endpoint={e}"));
        }
        let store = S3Memory::from_url(&url).unwrap();
        exercise_memory(&store).await;
        exercise_sessions(store.sessions(&Map::new()).unwrap().unwrap().as_ref(), true).await;
    }
}
