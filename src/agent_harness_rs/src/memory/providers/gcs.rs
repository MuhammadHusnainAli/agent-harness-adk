//! Google Cloud Storage memory.
//!
//! Spoken to over the GCS JSON API with a bearer token — no SDK. The keys are
//! the Python `GCSMemory`'s (see [`objects`](super::objects)), so one bucket
//! serves both.
//!
//! ```no_run
//! # use agent_harness::memory::providers::gcs::GcsMemory;
//! let store = GcsMemory::from_url("gs://my-bucket/agent-memory");
//! ```
//!
//! **Options** on the URL: `prefix`, `endpoint`, `token`, `concurrency`,
//! `conditional`. (`project` is accepted and ignored: object calls do not need
//! it.)
//!
//! **Credentials**, in order: an explicit `token`; `GOOGLE_OAUTH_ACCESS_TOKEN`;
//! `STORAGE_EMULATOR_HOST` (an emulator, called without credentials, as the
//! Python client does); the JSON key file in `GOOGLE_APPLICATION_CREDENTIALS`
//! or gcloud's application-default file — a `service_account` key (signed
//! RS256 here) or an `authorized_user` refresh token; then the metadata server
//! (`GCE_METADATA_HOST`, else `metadata.google.internal`). Tokens are cached
//! until a minute before they expire. Not covered, unlike google-auth:
//! external-account (workload identity federation) and impersonation files.
//!
//! Python deletes in batches of 100 through the batch endpoint; that is a
//! multipart protocol, so here each object is deleted with its own request,
//! `concurrency` at a time.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use futures::StreamExt;
use reqwest::Method;
use serde_json::Value;

use crate::errors::{Error, Result};
use crate::knowledge::base::HttpClient;
use crate::memory::providers::objects::{Fetched, ObjectBackend, ObjectStoreMemory, PutIf, encode, env, option, send_raw, split_url};
use crate::types::now;

/// Memory in a GCS bucket: [`ObjectStoreMemory`] over a [`GcsBucket`].
pub type GcsMemory = ObjectStoreMemory<GcsBucket>;

const SCOPE: &str = "https://www.googleapis.com/auth/devstorage.read_write";

/// Where a bearer token comes from.
#[derive(Clone)]
pub enum GcsAuth {
    /// No credentials: an emulator.
    Anonymous,
    /// A token you already have.
    Token(String),
    /// A service-account key: a JWT signed with its private key, swapped for a token.
    ServiceAccount { email: String, key_der: Vec<u8>, token_uri: String },
    /// gcloud's user login: a refresh token swapped for an access token.
    AuthorizedUser { client_id: String, client_secret: String, refresh_token: String },
    /// The metadata server of the machine this runs on.
    Metadata(String),
}

impl std::fmt::Debug for GcsAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Anonymous => "Anonymous",
            Self::Token(_) => "Token",
            Self::ServiceAccount { .. } => "ServiceAccount",
            Self::AuthorizedUser { .. } => "AuthorizedUser",
            Self::Metadata(_) => "Metadata",
        })
    }
}

impl GcsAuth {
    /// The standard chain (see the module docs).
    pub fn discover() -> Result<Self> {
        if let Some(t) = env("GOOGLE_OAUTH_ACCESS_TOKEN") {
            return Ok(Self::Token(t));
        }
        if env("STORAGE_EMULATOR_HOST").is_some() {
            return Ok(Self::Anonymous);
        }
        let well_known = if cfg!(windows) {
            env("APPDATA").map(|a| std::path::PathBuf::from(a).join("gcloud").join("application_default_credentials.json"))
        } else {
            env("HOME").map(|h| std::path::PathBuf::from(h).join(".config").join("gcloud").join("application_default_credentials.json"))
        };
        let file = env("GOOGLE_APPLICATION_CREDENTIALS").map(std::path::PathBuf::from).or(well_known.filter(|p| p.exists()));
        if let Some(path) = file {
            let text = std::fs::read_to_string(&path).map_err(|e| Error::config(format!("GCS credentials {}: {e}", path.display())))?;
            return Self::from_json(&text);
        }
        let host = env("GCE_METADATA_HOST").unwrap_or_else(|| "metadata.google.internal".into());
        Ok(Self::Metadata(format!("http://{host}/computeMetadata/v1/instance/service-accounts/default/token")))
    }

    /// A credentials JSON file's contents.
    pub fn from_json(text: &str) -> Result<Self> {
        let blob: Value = serde_json::from_str(text).map_err(|e| Error::config(format!("GCS credentials are not JSON: {e}")))?;
        let s = |k: &str| blob.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        match s("type").as_str() {
            "service_account" => {
                let pem = s("private_key");
                let body: String = pem.lines().filter(|l| !l.starts_with("-----")).collect();
                let key_der = base64::engine::general_purpose::STANDARD
                    .decode(body.trim())
                    .map_err(|e| Error::config(format!("GCS service-account private_key is not PEM: {e}")))?;
                let token_uri = Some(s("token_uri")).filter(|u| !u.is_empty()).unwrap_or_else(|| "https://oauth2.googleapis.com/token".into());
                Ok(Self::ServiceAccount { email: s("client_email"), key_der, token_uri })
            }
            "authorized_user" => Ok(Self::AuthorizedUser { client_id: s("client_id"), client_secret: s("client_secret"), refresh_token: s("refresh_token") }),
            other => Err(Error::config(format!("GCS credentials of type '{other}' are not supported; use a service-account key, gcloud's login or a token"))),
        }
    }
}

/// An RS256 JWT asking for `scope`, signed with a PKCS#8 key.
pub fn service_account_jwt(email: &str, key_der: &[u8], token_uri: &str, at: f64) -> Result<String> {
    let b64 = |b: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b);
    let iat = at as i64;
    let header = b64(br#"{"alg":"RS256","typ":"JWT"}"#);
    let claims = b64(serde_json::json!({"iss": email, "scope": SCOPE, "aud": token_uri, "iat": iat, "exp": iat + 3600}).to_string().as_bytes());
    let input = format!("{header}.{claims}");
    let key = ring::signature::RsaKeyPair::from_pkcs8(key_der).map_err(|e| Error::config(format!("GCS service-account key was refused: {e}")))?;
    let mut signature = vec![0u8; key.public().modulus_len()];
    key.sign(&ring::signature::RSA_PKCS1_SHA256, &ring::rand::SystemRandom::new(), input.as_bytes(), &mut signature)
        .map_err(|_| Error::config("GCS service-account key could not sign"))?;
    Ok(format!("{input}.{}", b64(&signature)))
}

/// One GCS bucket, and how to reach and authorise against it.
#[derive(Debug, Clone)]
pub struct GcsBucket {
    pub bucket: String,
    /// `https://storage.googleapis.com`, or an emulator.
    pub endpoint: String,
    auth: GcsAuth,
    cached: Arc<tokio::sync::Mutex<Option<(String, f64)>>>,
    http: HttpClient,
    /// Deletes in flight at once.
    pub concurrency: usize,
}

impl GcsBucket {
    /// A bucket with the credentials the standard chain finds.
    pub fn new(bucket: &str) -> Result<Self> {
        let endpoint = env("STORAGE_EMULATOR_HOST")
            .map_or_else(|| "https://storage.googleapis.com".to_string(), |h| if h.contains("://") { h } else { format!("http://{h}") });
        Ok(Self::with_auth(bucket, GcsAuth::discover()?).endpoint(&endpoint))
    }

    pub fn with_auth(bucket: &str, auth: GcsAuth) -> Self {
        let mut http = HttpClient::new("gcs", 60.0, 3);
        if let GcsAuth::Token(t) = &auth {
            http = http.secret(t.clone());
        }
        Self { bucket: bucket.into(), endpoint: "https://storage.googleapis.com".into(), auth, cached: Arc::default(), http, concurrency: 16 }
    }

    pub fn endpoint(mut self, endpoint: &str) -> Self {
        self.endpoint = endpoint.trim_end_matches('/').to_string();
        self
    }

    pub fn retries(mut self, retries: u32) -> Self {
        self.http.retry.max_retries = retries;
        self
    }

    async fn token(&self) -> Result<Option<String>> {
        match &self.auth {
            GcsAuth::Anonymous => return Ok(None),
            GcsAuth::Token(t) => return Ok(Some(t.clone())),
            _ => {}
        }
        let mut cached = self.cached.lock().await;
        if let Some((token, until)) = cached.as_ref()
            && *until > now()
        {
            return Ok(Some(token.clone()));
        }
        let form = |pairs: &[(&str, &str)]| pairs.iter().map(|(k, v)| format!("{k}={}", encode(v, false))).collect::<Vec<_>>().join("&").into_bytes();
        #[allow(clippy::type_complexity)]
        let (url, body, extra): (String, Option<Vec<u8>>, Vec<(&str, String)>) = match &self.auth {
            GcsAuth::Anonymous | GcsAuth::Token(_) => return Ok(None),
            GcsAuth::ServiceAccount { email, key_der, token_uri } => {
                let jwt = service_account_jwt(email, key_der, token_uri, now())?;
                (token_uri.clone(), Some(form(&[("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"), ("assertion", &jwt)])), vec![])
            }
            GcsAuth::AuthorizedUser { client_id, client_secret, refresh_token } => (
                "https://oauth2.googleapis.com/token".into(),
                Some(form(&[("grant_type", "refresh_token"), ("client_id", client_id), ("client_secret", client_secret), ("refresh_token", refresh_token)])),
                vec![],
            ),
            GcsAuth::Metadata(url) => (url.clone(), None, vec![("metadata-flavor", "Google".to_string())]),
        };
        let auth_http = HttpClient::new("gcs-auth", 30.0, 2);
        let reply = match body {
            Some(b) => auth_http.send(Method::POST, &url, Some(b), Some("application/x-www-form-urlencoded"), &extra, &[]).await,
            None => auth_http.send(Method::GET, &url, None, None, &extra, &[]).await,
        }
        .map_err(|e| Error::config(format!("GCS: no access token ({e}); set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_OAUTH_ACCESS_TOKEN")))?;
        let token = reply.body["access_token"].as_str().unwrap_or("").to_string();
        if token.is_empty() {
            return Err(Error::config("GCS: the token endpoint returned no access_token"));
        }
        let lifetime = reply.body["expires_in"].as_f64().unwrap_or(3600.0);
        *cached = Some((token.clone(), now() + lifetime - 60.0));
        Ok(Some(token))
    }

    async fn headers(&self, extra: &[(&'static str, String)]) -> Result<Vec<(&'static str, String)>> {
        let mut h: Vec<(&'static str, String)> = extra.to_vec();
        if let Some(t) = self.token().await? {
            h.push(("authorization", format!("Bearer {t}")));
        }
        Ok(h)
    }

    fn object_url(&self, key: &str) -> String {
        format!("{}/storage/v1/b/{}/o/{}", self.endpoint, encode(&self.bucket, false), encode(key, false))
    }
}

#[async_trait]
impl ObjectBackend for GcsBucket {
    async fn put(&self, key: &str, body: Vec<u8>, content_type: &str, condition: &PutIf) -> Result<bool> {
        let mut url = format!("{}/upload/storage/v1/b/{}/o?uploadType=media&name={}", self.endpoint, encode(&self.bucket, false), encode(key, false));
        match condition {
            PutIf::Always => {}
            PutIf::Absent => url.push_str("&ifGenerationMatch=0"),
            PutIf::Matches(generation) => url.push_str(&format!("&ifGenerationMatch={}", encode(generation, false))),
        }
        let headers = self.headers(&[("content-type", content_type.to_string())]).await?;
        let ok: &[u16] = if *condition == PutIf::Always { &[] } else { &[412] };
        Ok(send_raw(&self.http, Method::POST, &url, body, &headers, ok).await?.status != 412)
    }

    async fn get(&self, key: &str) -> Result<Option<Fetched>> {
        let headers = self.headers(&[]).await?;
        let reply = send_raw(&self.http, Method::GET, &format!("{}?alt=media", self.object_url(key)), vec![], &headers, &[404]).await?;
        if reply.status == 404 {
            return Ok(None);
        }
        let tag = reply.header("x-goog-generation");
        Ok(Some(Fetched { body: reply.body, tag }))
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let mut keys = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut url = format!(
                "{}/storage/v1/b/{}/o?prefix={}&fields={}",
                self.endpoint,
                encode(&self.bucket, false),
                encode(prefix, false),
                encode("items(name),nextPageToken", false)
            );
            if let Some(t) = &token {
                url.push_str(&format!("&pageToken={}", encode(t, false)));
            }
            let headers = self.headers(&[]).await?;
            let reply = send_raw(&self.http, Method::GET, &url, vec![], &headers, &[]).await?;
            let page: Value = serde_json::from_slice(&reply.body).unwrap_or(Value::Null);
            keys.extend(page["items"].as_array().into_iter().flatten().filter_map(|i| i["name"].as_str().map(str::to_string)));
            token = page["nextPageToken"].as_str().map(str::to_string);
            if token.is_none() {
                return Ok(keys);
            }
        }
    }

    async fn delete(&self, keys: &[String]) -> Result<()> {
        let headers = self.headers(&[]).await?;
        let results: Vec<Result<()>> = futures::stream::iter(keys.iter().map(|k| self.object_url(k)).collect::<Vec<String>>())
            .map(|url: String| {
                let headers = &headers;
                async move { send_raw(&self.http, Method::DELETE, &url, vec![], headers, &[404]).await.map(|_| ()) }
            })
            .buffer_unordered(self.concurrency.max(1))
            .collect()
            .await;
        results.into_iter().collect()
    }

    fn describe(&self) -> String {
        format!("gs://{}", self.bucket)
    }
}

impl GcsMemory {
    /// `gs://bucket/prefix` (or `gcs://…`). The prefix defaults to `agent-memory`.
    pub fn from_url(url: &str) -> Result<Self> {
        let parts = split_url(url);
        if parts.host.is_empty() {
            return Err(Error::config(format!("GcsMemory needs a bucket: gs://bucket/prefix (got '{url}')")));
        }
        let o = &parts.query;
        let mut bucket = match option(o, "token") {
            Some(t) => GcsBucket::with_auth(&parts.host, GcsAuth::Token(t.into())),
            None => GcsBucket::new(&parts.host)?,
        };
        if let Some(e) = option(o, "endpoint") {
            bucket = bucket.endpoint(e);
        }
        if let Some(n) = option(o, "concurrency").and_then(|n| n.parse().ok()) {
            bucket.concurrency = n;
        }
        let prefix = parts.path.trim_matches('/');
        let prefix = if prefix.is_empty() { option(o, "prefix").unwrap_or("agent-memory") } else { prefix };
        Ok(GcsMemory::new(bucket, prefix).configured(o))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use axum::body::Bytes;
    use axum::http::{HeaderMap, Method as AxMethod, StatusCode, Uri};
    use axum::response::{IntoResponse, Response};
    use serde_json::{Map, json};

    use super::*;
    use crate::memory::base::MemoryStore;
    use crate::memory::providers::objects::decode;
    use crate::memory::providers::objects::testing::{exercise_memory, exercise_sessions, query_of, serve};

    type Objects = Arc<Mutex<(BTreeMap<String, (Vec<u8>, u64)>, u64)>>;

    /// Enough of the GCS JSON API: media uploads with ifGenerationMatch,
    /// alt=media reads with x-goog-generation, paged listings, deletes.
    async fn fake_gcs(state: Objects, method: AxMethod, uri: Uri, headers: HeaderMap, body: Bytes) -> Response {
        if headers.get("authorization").and_then(|v| v.to_str().ok()) != Some("Bearer tok") {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        let q = query_of(&uri);
        let path = uri.path().to_string();
        let mut s = state.lock().unwrap();
        if path == "/upload/storage/v1/b/bucket/o" && method == AxMethod::POST {
            let name = option(&q, "name").unwrap_or("").to_string();
            let held = s.0.get(&name).map(|(_, g)| *g).unwrap_or(0);
            if let Some(want) = option(&q, "ifGenerationMatch").and_then(|g| g.parse::<u64>().ok())
                && want != held
            {
                return StatusCode::PRECONDITION_FAILED.into_response();
            }
            s.1 += 1;
            let generation = s.1;
            s.0.insert(name, (body.to_vec(), generation));
            return axum::Json(json!({"generation": generation.to_string()})).into_response();
        }
        if path == "/storage/v1/b/bucket/o" {
            let prefix = option(&q, "prefix").unwrap_or("").to_string();
            let start: usize = option(&q, "pageToken").and_then(|t| t.parse().ok()).unwrap_or(0);
            let keys: Vec<&String> = s.0.keys().filter(|k| k.starts_with(&prefix)).collect();
            let items: Vec<Value> = keys.iter().skip(start).take(3).map(|k| json!({"name": k})).collect();
            let mut page = json!({"items": items});
            if start + 3 < keys.len() {
                page["nextPageToken"] = json!((start + 3).to_string());
            }
            return axum::Json(page).into_response();
        }
        let Some(raw) = path.strip_prefix("/storage/v1/b/bucket/o/") else { return StatusCode::NOT_FOUND.into_response() };
        let name = decode(raw);
        match method {
            AxMethod::GET => match s.0.get(&name) {
                Some((b, g)) => ([("x-goog-generation", g.to_string())], b.clone()).into_response(),
                None => StatusCode::NOT_FOUND.into_response(),
            },
            AxMethod::DELETE => match s.0.remove(&name) {
                Some(_) => StatusCode::NO_CONTENT.into_response(),
                None => StatusCode::NOT_FOUND.into_response(),
            },
            _ => StatusCode::METHOD_NOT_ALLOWED.into_response(),
        }
    }

    async fn store(prefix: &str) -> (GcsMemory, Objects) {
        let state: Objects = Arc::default();
        let shared = state.clone();
        let router = axum::Router::new().fallback(move |m: AxMethod, u: Uri, h: HeaderMap, b: Bytes| fake_gcs(shared.clone(), m, u, h, b));
        let base = serve(router).await;
        let url = format!("gs://bucket/{prefix}?token=tok&endpoint={}", encode(&base, false));
        let mut store = GcsMemory::from_url(&url).unwrap();
        store = store.with_concurrency(4);
        (store, state)
    }

    #[tokio::test]
    async fn memory_over_the_gcs_api_is_scoped_by_trace() {
        let (store, state) = store("mem").await;
        exercise_memory(&store).await;
        assert!(state.lock().unwrap().0.is_empty());
    }

    #[tokio::test]
    async fn sessions_over_the_gcs_api_hold_the_contract() {
        let (store, state) = store("chats").await;
        exercise_sessions(store.sessions(&Map::new()).unwrap().unwrap().as_ref(), true).await;
        assert!(state.lock().unwrap().0.is_empty());
    }

    #[test]
    fn credential_files_are_read() {
        let user = GcsAuth::from_json(r#"{"type":"authorized_user","client_id":"c","client_secret":"s","refresh_token":"r"}"#).unwrap();
        assert!(matches!(user, GcsAuth::AuthorizedUser { .. }));
        assert!(GcsAuth::from_json(r#"{"type":"external_account"}"#).is_err());
        assert!(service_account_jwt("a@b", b"not a key", "https://t", 0.0).is_err());
    }

    /// `AH_LIVE_GCS_BUCKET` to run against a real bucket with the ambient credentials.
    #[tokio::test]
    async fn a_real_bucket_holds_the_contract() {
        let Some(bucket) = env("AH_LIVE_GCS_BUCKET") else { return };
        let store = GcsMemory::from_url(&format!("gs://{bucket}/ah-rust-test-{}", crate::types::new_id(""))).unwrap();
        exercise_memory(&store).await;
        exercise_sessions(store.sessions(&Map::new()).unwrap().unwrap().as_ref(), true).await;
    }
}
