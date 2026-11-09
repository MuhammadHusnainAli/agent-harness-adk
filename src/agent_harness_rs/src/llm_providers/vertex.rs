//! Claude and Gemini on Google Cloud Vertex AI.
//!
//! Vertex serves both families, each in its own publisher path, and both in
//! the same wire format they use directly — so the encoding is shared with
//! [`AnthropicProvider`] and [`GeminiProvider`], and only the URL and the auth
//! change.
//!
//! ```no_run
//! use agent_harness::llm_providers::vertex::{VertexGeminiProvider, VertexOptions, VertexProvider};
//!
//! let options = VertexOptions { project: Some("my-project".into()), region: Some("europe-west1".into()), ..Default::default() };
//! let claude = VertexProvider::new(options.clone()).unwrap();
//! let gemini = VertexGeminiProvider::new(options).unwrap();
//! ```
//!
//! Auth is a Google access token ([`GoogleAuth`]): one you pass in, else
//! `GOOGLE_ACCESS_TOKEN` or `GOOGLE_OAUTH_ACCESS_TOKEN`, else what google-auth's
//! application default credentials would find — the key file in
//! `GOOGLE_APPLICATION_CREDENTIALS` or gcloud's application-default login (a
//! `service_account` key, signed RS256 here, or an `authorized_user` refresh
//! token), then the GCE metadata server — and last `gcloud auth
//! print-access-token`. A 401 drops the cached token and fetches a new one,
//! once. Completion and streaming both go through the shared retrying
//! transport. Not covered, unlike google-auth: workload identity federation
//! (`external_account`) and impersonation files.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use base64::Engine as _;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use reqwest::Method;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Value, json};

use super::anthropic::{self, AnthropicProvider, AnthropicStream};
use super::base::{Auth, CompletionRequest, HttpTransport, Provider, ProviderField, ProviderSpec, TransportConfig, default_client, sse_events};
use super::bedrock::relabel;
use super::gemini::{self, GeminiProvider};
use super::parameters::ParameterPlan;
use super::sigv4::uri_encode;
use crate::errors::{Error, ProviderError, ProviderErrorKind as K, Result};
use crate::types::{ModelResponse, StreamEvent, StreamEventType, now};

pub const VERTEX_VERSION: &str = "vertex-2023-10-16";
pub const SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const EMBEDDING_MODEL: &str = "gemini-embedding-001";

/// Produces a Google access token — the stand-in for a google-auth credentials object.
pub type TokenFn = Arc<dyn Fn() -> BoxFuture<'static, Result<String>> + Send + Sync>;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn auth_error(message: impl Into<String>) -> Error {
    ProviderError::new(K::Authentication, message).provider("vertex").into()
}

/// Where a Google access token comes from.
#[derive(Clone)]
pub enum GoogleCredentials {
    /// A ready-made token. Never refreshed.
    Token(String),
    /// A JSON key file (service account or gcloud login), read when a token is needed.
    KeyFile(std::path::PathBuf),
    /// A service-account key: a JWT signed with its private key, swapped for a token.
    ServiceAccount { email: String, key_der: Vec<u8>, token_uri: String },
    /// gcloud's application-default login: a refresh token swapped for an access token.
    AuthorizedUser { client_id: String, client_secret: String, refresh_token: String, token_uri: String },
    /// The metadata server of the machine this runs on (the token URL).
    Metadata(String),
    /// `gcloud auth print-access-token`.
    Gcloud,
    /// Your own source.
    Function(TokenFn),
}

impl std::fmt::Debug for GoogleCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Token(_) => f.write_str("Token"),
            Self::KeyFile(p) => write!(f, "KeyFile({})", p.display()),
            Self::ServiceAccount { email, .. } => write!(f, "ServiceAccount({email})"),
            Self::AuthorizedUser { .. } => f.write_str("AuthorizedUser"),
            Self::Metadata(u) => write!(f, "Metadata({u})"),
            Self::Gcloud => f.write_str("Gcloud"),
            Self::Function(_) => f.write_str("Function"),
        }
    }
}

impl GoogleCredentials {
    /// A credentials JSON file's contents: `service_account` or `authorized_user`.
    pub fn from_json(text: &str) -> Result<Self> {
        let blob: Value = serde_json::from_str(text).map_err(|e| auth_error(format!("Google credentials are not JSON: {e}")))?;
        Self::from_value(&blob)
    }

    pub fn from_value(blob: &Value) -> Result<Self> {
        let s = |k: &str| blob.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let token_uri = Some(s("token_uri")).filter(|u| !u.is_empty()).unwrap_or_else(|| "https://oauth2.googleapis.com/token".into());
        match s("type").as_str() {
            "service_account" => {
                let body: String = s("private_key").lines().filter(|l| !l.starts_with("-----")).collect();
                let key_der = base64::engine::general_purpose::STANDARD
                    .decode(body.trim())
                    .map_err(|e| auth_error(format!("the service-account private_key is not PEM: {e}")))?;
                Ok(Self::ServiceAccount { email: s("client_email"), key_der, token_uri })
            }
            "authorized_user" => {
                Ok(Self::AuthorizedUser { client_id: s("client_id"), client_secret: s("client_secret"), refresh_token: s("refresh_token"), token_uri })
            }
            other => Err(auth_error(format!("Google credentials of type '{other}' are not supported; use a service-account key, gcloud's login or a token"))),
        }
    }

    /// The metadata server: `GCE_METADATA_HOST`, else `metadata.google.internal`.
    pub fn metadata() -> Self {
        let host = env("GCE_METADATA_HOST").unwrap_or_else(|| "metadata.google.internal".into());
        Self::Metadata(format!("http://{host}/computeMetadata/v1/instance/service-accounts/default/token"))
    }

    /// What application default credentials would try, then gcloud.
    pub fn default_chain() -> Vec<Self> {
        if let Some(path) = env("GOOGLE_APPLICATION_CREDENTIALS") {
            return vec![Self::KeyFile(path.into())];
        }
        let well_known = if cfg!(windows) {
            env("APPDATA").map(|a| std::path::PathBuf::from(a).join("gcloud").join("application_default_credentials.json"))
        } else {
            env("HOME").map(|h| std::path::PathBuf::from(h).join(".config").join("gcloud").join("application_default_credentials.json"))
        };
        let mut chain: Vec<Self> = well_known.filter(|p| p.exists()).map(Self::KeyFile).into_iter().collect();
        chain.push(Self::metadata());
        chain.push(Self::Gcloud);
        chain
    }
}

/// An RS256 JWT asking for the cloud-platform scope, signed with a PKCS#8 key.
pub fn service_account_jwt(email: &str, key_der: &[u8], token_uri: &str, at: f64) -> Result<String> {
    let b64 = |b: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b);
    let iat = at as i64;
    let header = b64(br#"{"alg":"RS256","typ":"JWT"}"#);
    let claims = b64(json!({"iss": email, "scope": SCOPE, "aud": token_uri, "iat": iat, "exp": iat + 3600}).to_string().as_bytes());
    let input = format!("{header}.{claims}");
    let key = ring::signature::RsaKeyPair::from_pkcs8(key_der).map_err(|e| auth_error(format!("the service-account key was refused: {e}")))?;
    let mut signature = vec![0u8; key.public().modulus_len()];
    key.sign(&ring::signature::RSA_PKCS1_SHA256, &ring::rand::SystemRandom::new(), input.as_bytes(), &mut signature)
        .map_err(|_| auth_error("the service-account key could not sign"))?;
    Ok(format!("{input}.{}", b64(&signature)))
}

/// POST a form (or GET, without one) to a token endpoint; the token and its lifetime.
pub(crate) async fn token_request(
    client: &reqwest::Client,
    url: &str,
    form: Option<&[(&str, &str)]>,
    headers: &[(&str, &str)],
) -> std::result::Result<(String, f64), String> {
    let mut builder = match form {
        Some(pairs) => {
            let body = pairs.iter().map(|(k, v)| format!("{k}={}", uri_encode(v, false))).collect::<Vec<_>>().join("&");
            client.post(url).header("content-type", "application/x-www-form-urlencoded").body(body)
        }
        None => client.get(url),
    };
    for (k, v) in headers {
        builder = builder.header(*k, *v);
    }
    let resp = builder.send().await.map_err(|e| e.to_string())?;
    let status = resp.status().as_u16();
    let text = resp.text().await.unwrap_or_default();
    if status >= 300 {
        return Err(format!("{status}: {}", text.chars().take(300).collect::<String>()));
    }
    let body: Value = serde_json::from_str(&text).map_err(|_| format!("not JSON: {}", text.chars().take(200).collect::<String>()))?;
    let token = body["access_token"].as_str().unwrap_or("").to_string();
    if token.is_empty() {
        return Err("no access_token in the reply".into());
    }
    let number = |v: &Value| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()));
    let expires = number(&body["expires_on"]).or_else(|| number(&body["expires_in"]).map(|s| now() + s)).unwrap_or_else(|| now() + 3600.0);
    Ok((token, expires))
}

/// A Google access token, from whichever source is available.
///
/// Tokens are cached until a minute before they expire, so a long run does not
/// re-authenticate on every call.
pub struct GoogleAuth {
    fixed: Option<String>,
    chain: Vec<GoogleCredentials>,
    cache: Mutex<(String, f64)>,
    fetching: tokio::sync::Mutex<()>,
    client: reqwest::Client,
}

impl std::fmt::Debug for GoogleAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleAuth").field("fixed", &self.fixed.is_some()).field("chain", &self.chain).finish()
    }
}

impl GoogleAuth {
    /// `token`, else `GOOGLE_ACCESS_TOKEN` / `GOOGLE_OAUTH_ACCESS_TOKEN`, else the default chain.
    pub fn new(token: Option<String>) -> Self {
        match token.filter(|t| !t.is_empty()).or_else(|| env("GOOGLE_ACCESS_TOKEN")).or_else(|| env("GOOGLE_OAUTH_ACCESS_TOKEN")) {
            Some(t) => Self::with_chain(Some(t), vec![]),
            None => Self::with_chain(None, GoogleCredentials::default_chain()),
        }
    }

    /// One source, used alone.
    pub fn from_credentials(credentials: GoogleCredentials) -> Self {
        match credentials {
            GoogleCredentials::Token(t) => Self::with_chain(Some(t), vec![]),
            other => Self::with_chain(None, vec![other]),
        }
    }

    /// Sources tried in order until one gives a token.
    pub fn with_chain(fixed: Option<String>, chain: Vec<GoogleCredentials>) -> Self {
        Self { fixed, chain, cache: Mutex::new((String::new(), 0.0)), fetching: tokio::sync::Mutex::new(()), client: default_client(30.0, 3.0) }
    }

    /// False for a fixed token: fetching again would give the same one.
    pub fn refreshable(&self) -> bool {
        self.fixed.is_none()
    }

    /// Forget the cached token, so the next call fetches a fresh one.
    pub fn invalidate(&self) {
        *self.cache.lock().unwrap_or_else(|p| p.into_inner()) = (String::new(), 0.0);
    }

    fn cached(&self) -> Option<String> {
        let c = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        (!c.0.is_empty() && now() < c.1 - 60.0).then(|| c.0.clone())
    }

    pub async fn token(&self) -> Result<String> {
        if let Some(t) = &self.fixed {
            return Ok(t.clone());
        }
        if let Some(t) = self.cached() {
            return Ok(t);
        }
        let _one_at_a_time = self.fetching.lock().await;
        if let Some(t) = self.cached() {
            return Ok(t);
        }
        let mut failures = Vec::new();
        for source in &self.chain {
            match self.fetch(source).await {
                Ok((token, expires)) => {
                    *self.cache.lock().unwrap_or_else(|p| p.into_inner()) = (token.clone(), expires);
                    return Ok(token);
                }
                Err(e) => failures.push(format!("{source:?}: {e}")),
            }
        }
        Err(auth_error(format!(
            "Vertex needs a Google access token — set GOOGLE_APPLICATION_CREDENTIALS, run `gcloud auth application-default login`, \
             or pass access_token (tried: {})",
            if failures.is_empty() { "nothing".to_string() } else { failures.join("; ") }
        )))
    }

    async fn fetch(&self, source: &GoogleCredentials) -> std::result::Result<(String, f64), String> {
        match source {
            GoogleCredentials::Token(t) => Ok((t.clone(), f64::MAX)),
            GoogleCredentials::KeyFile(path) => {
                let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
                let parsed = GoogleCredentials::from_json(&text).map_err(|e| e.to_string())?;
                Box::pin(self.fetch(&parsed)).await
            }
            GoogleCredentials::ServiceAccount { email, key_der, token_uri } => {
                let jwt = service_account_jwt(email, key_der, token_uri, now()).map_err(|e| e.to_string())?;
                token_request(&self.client, token_uri, Some(&[("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"), ("assertion", &jwt)]), &[]).await
            }
            GoogleCredentials::AuthorizedUser { client_id, client_secret, refresh_token, token_uri } => {
                let form = [
                    ("grant_type", "refresh_token"),
                    ("client_id", client_id.as_str()),
                    ("client_secret", client_secret.as_str()),
                    ("refresh_token", refresh_token.as_str()),
                ];
                token_request(&self.client, token_uri, Some(&form), &[]).await
            }
            GoogleCredentials::Metadata(url) => token_request(&self.client, url, None, &[("metadata-flavor", "Google")]).await,
            GoogleCredentials::Function(f) => f().await.map(|t| (t, now() + 3000.0)).map_err(|e| e.to_string()),
            GoogleCredentials::Gcloud => gcloud_token().await,
        }
    }
}

/// Borrow the token the gcloud CLI already has.
async fn gcloud_token() -> std::result::Result<(String, f64), String> {
    let mut command = if cfg!(windows) {
        let mut c = tokio::process::Command::new("cmd");
        c.args(["/C", "gcloud"]);
        c
    } else {
        tokio::process::Command::new("gcloud")
    };
    command.args(["auth", "print-access-token"]).kill_on_drop(true);
    let output = match tokio::time::timeout(std::time::Duration::from_secs(30), command.output()).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => return Err(format!("gcloud is not available: {e}")),
        Err(_) => return Err("gcloud timed out".into()),
    };
    if !output.status.success() {
        return Err(format!("gcloud could not produce a token: {}", String::from_utf8_lossy(&output.stderr).trim().chars().take(300).collect::<String>()));
    }
    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.is_empty() {
        return Err("gcloud printed no token".into());
    }
    Ok((token, now() + 3000.0))
}

struct BearerAuth(Arc<GoogleAuth>);

#[async_trait]
impl Auth for BearerAuth {
    async fn headers(&self, _: &Method, _: &str, _: &[u8]) -> Result<HeaderMap> {
        let token = self.0.token().await?;
        let mut h = HeaderMap::new();
        h.insert("authorization", HeaderValue::from_str(&format!("Bearer {token}")).map_err(|e| Error::config(e.to_string()))?);
        Ok(h)
    }

    async fn refresh(&self) -> bool {
        if !self.0.refreshable() {
            return false;
        }
        self.0.invalidate();
        true
    }
}

const FIELDS: [ProviderField; 4] = [
    ProviderField {
        required: true,
        env: &["GOOGLE_CLOUD_PROJECT"],
        example: Some("my-project"),
        ..ProviderField::new("project", "str", "The Google Cloud project id.")
    },
    ProviderField {
        env: &["GOOGLE_CLOUD_REGION", "CLOUD_ML_REGION"],
        example: Some("europe-west1"),
        ..ProviderField::new("region", "str", "The Vertex region, or 'global' (default us-central1).")
    },
    ProviderField {
        one_of: Some("google"),
        env: &["GOOGLE_ACCESS_TOKEN", "GOOGLE_OAUTH_ACCESS_TOKEN"],
        ..ProviderField::new(
            "access_token",
            "secret",
            "A ready-made OAuth token. Otherwise application default credentials (key files, the metadata server) or gcloud.",
        )
    },
    ProviderField {
        one_of: Some("google"),
        ..ProviderField::new("credentials", "object", "GoogleCredentials (as a setting: a key file's JSON, or its path).")
    },
];

pub const SPEC: ProviderSpec = ProviderSpec {
    name: "vertex",
    display_name: "Google Vertex AI (Claude)",
    description: "Claude served from your Google Cloud project on Vertex AI.",
    docs_url: "https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude",
    auth_type: "google-oauth",
    env_key: "",
    default_model: "claude-opus-5",
    base_url: "https://aiplatform.googleapis.com",
    capabilities: &["streaming", "tools", "vision", "thinking", "json_schema", "prompt_caching"],
    parameters: anthropic::SPEC.parameters,
    modalities: anthropic::SPEC.modalities,
    fields: &FIELDS,
};

pub const GEMINI_SPEC: ProviderSpec = ProviderSpec {
    name: "vertex-gemini",
    display_name: "Google Vertex AI (Gemini)",
    description: "Gemini served from your Google Cloud project on Vertex AI.",
    docs_url: "https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference",
    auth_type: "google-oauth",
    env_key: "",
    default_model: "gemini-2.5-pro",
    base_url: "https://aiplatform.googleapis.com",
    capabilities: &["streaming", "tools", "vision", "thinking", "json_schema", "embeddings"],
    parameters: gemini::SPEC.parameters,
    modalities: gemini::SPEC.modalities,
    fields: &FIELDS,
};

/// Where a Vertex project lives and how to authenticate to it.
#[derive(Debug, Clone, Default)]
pub struct VertexOptions {
    /// Else `GOOGLE_CLOUD_PROJECT`. Required.
    pub project: Option<String>,
    /// Else `GOOGLE_CLOUD_REGION`, `CLOUD_ML_REGION`, then `us-central1`.
    pub region: Option<String>,
    pub access_token: Option<String>,
    /// Used instead of the default chain.
    pub credentials: Option<GoogleCredentials>,
}

/// The URL shape and auth both Vertex providers share.
struct Setup {
    project: String,
    region: String,
    endpoint: String,
    auth: Arc<GoogleAuth>,
}

fn setup(options: VertexOptions, config: &TransportConfig) -> Result<Setup> {
    let project =
        options.project.filter(|p| !p.is_empty()).or_else(|| env("GOOGLE_CLOUD_PROJECT")).ok_or_else(|| {
            Error::from(ProviderError::new(K::Other, "Vertex needs a project — pass project=... or set GOOGLE_CLOUD_PROJECT").provider("vertex"))
        })?;
    let region = options
        .region
        .filter(|r| !r.is_empty())
        .or_else(|| env("GOOGLE_CLOUD_REGION"))
        .or_else(|| env("CLOUD_ML_REGION"))
        .unwrap_or_else(|| "us-central1".into());
    let auth = match (options.access_token.filter(|t| !t.is_empty()), options.credentials) {
        (Some(t), _) => GoogleAuth::from_credentials(GoogleCredentials::Token(t)),
        (None, Some(c)) => GoogleAuth::from_credentials(c),
        (None, None) => GoogleAuth::new(None),
    };
    let endpoint = vertex_endpoint(&project, &region, Some(&config.base_url));
    Ok(Setup { project, region, endpoint, auth: Arc::new(auth) })
}

/// `https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}`
/// (no region prefix for `global`). `host`, when given, replaces the scheme and
/// host — a Private Service Connect endpoint, a proxy, a test server.
pub fn vertex_endpoint(project: &str, region: &str, host: Option<&str>) -> String {
    let host = host
        .filter(|h| !h.is_empty())
        .map(|h| h.trim_end_matches('/').to_string())
        .unwrap_or_else(|| if region == "global" { "https://aiplatform.googleapis.com".into() } else { format!("https://{region}-aiplatform.googleapis.com") });
    format!("{host}/v1/projects/{project}/locations/{region}")
}

fn tail(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

/// Claude on Vertex AI.
pub struct VertexProvider {
    inner: AnthropicProvider,
    pub project: String,
    pub region: String,
    pub auth: Arc<GoogleAuth>,
    endpoint: String,
}

impl VertexProvider {
    pub fn new(options: VertexOptions) -> Result<Self> {
        Self::with_config(options, TransportConfig::default())
    }

    /// `config.base_url`, if set, replaces the scheme and host of the endpoint.
    pub fn with_config(options: VertexOptions, mut config: TransportConfig) -> Result<Self> {
        let s = setup(options, &config)?;
        config.base_url = s.endpoint.clone();
        let transport = HttpTransport::new(SPEC.name, config, Arc::new(BearerAuth(s.auth.clone())));
        Ok(Self { inner: AnthropicProvider { transport }, project: s.project, region: s.region, auth: s.auth, endpoint: s.endpoint })
    }

    /// `https://…/v1/projects/{project}/locations/{region}`.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// The model is in the URL, and the API version moves into the body.
    pub fn build_payload(&self, req: &CompletionRequest, stream: bool) -> Value {
        let mut payload = self.inner.build_payload(req, stream);
        if let Some(p) = payload.as_object_mut() {
            p.shift_remove("model");
            p.insert("anthropic_version".into(), json!(VERTEX_VERSION));
            if !stream {
                p.shift_remove("stream");
            }
        }
        payload
    }

    fn model_path(model: &str, stream: bool) -> String {
        format!("/publishers/anthropic/models/{}:{}", tail(model), if stream { "streamRawPredict" } else { "rawPredict" })
    }

    fn stream_error(&self, error: &Value) -> Error {
        relabel(self.inner.stream_error(error), SPEC.name)
    }
}

#[async_trait]
impl Provider for VertexProvider {
    fn spec(&self) -> &ProviderSpec {
        &SPEC
    }

    fn transport(&self) -> Option<&HttpTransport> {
        Some(&self.inner.transport)
    }

    fn plan(&self, req: &CompletionRequest, plan: &mut ParameterPlan) {
        Provider::plan(&self.inner, req, plan);
    }

    fn payload(&self, req: &CompletionRequest) -> Option<Value> {
        Some(self.build_payload(req, false))
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<ModelResponse> {
        let started = Instant::now();
        let raw = self.inner.transport.post(&Self::model_path(&req.model, false), &self.build_payload(req, false), &[], req.timeout).await?;
        if raw["type"] == "error" {
            return Err(self.stream_error(&raw["error"]));
        }
        let mut resp = self.inner.decode(raw, req, started.elapsed().as_secs_f64() * 1000.0)?;
        resp.provider = SPEC.name.into();
        Ok(resp)
    }

    fn stream<'a>(&'a self, req: &'a CompletionRequest) -> BoxStream<'a, Result<StreamEvent>> {
        self.inner.transport.retrying(move || {
            Box::pin(async_stream::try_stream! {
                let payload = self.build_payload(req, true);
                let path = Self::model_path(&req.model, true);
                let err = |e: &Value| self.stream_error(e);
                let mut state = AnthropicStream::new(self, &err, &req.model);
                let mut events = sse_events(self.inner.transport.stream_lines(&path, &payload, &[], req.timeout));
                while let Some(item) = events.next().await {
                    let (_, data) = item?;
                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }
                    let parsed: Value = serde_json::from_str(&data).map_err(|e| Error::config(format!("bad stream event: {e}")))?;
                    for e in state.feed(&parsed)? {
                        yield e;
                    }
                }
                for e in state.finish() {
                    yield e;
                }
            })
        })
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        Err(Error::config("Vertex lists partner models in Model Garden"))
    }
}

/// Gemini on Vertex AI.
pub struct VertexGeminiProvider {
    inner: GeminiProvider,
    pub project: String,
    pub region: String,
    pub auth: Arc<GoogleAuth>,
    endpoint: String,
}

impl VertexGeminiProvider {
    pub fn new(options: VertexOptions) -> Result<Self> {
        Self::with_config(options, TransportConfig::default())
    }

    /// `config.base_url`, if set, replaces the scheme and host of the endpoint.
    pub fn with_config(options: VertexOptions, mut config: TransportConfig) -> Result<Self> {
        let s = setup(options, &config)?;
        config.base_url = format!("{}/publishers/google", s.endpoint);
        let transport = HttpTransport::new(GEMINI_SPEC.name, config, Arc::new(BearerAuth(s.auth.clone())));
        Ok(Self { inner: GeminiProvider { transport }, project: s.project, region: s.region, auth: s.auth, endpoint: s.endpoint })
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Vertex names a model by its last path segment.
    fn local(req: &CompletionRequest) -> Option<CompletionRequest> {
        req.model.contains('/').then(|| CompletionRequest { model: tail(&req.model).to_string(), ..req.clone() })
    }
}

/// Stamp this provider's name on a response the shared Gemini code built.
fn restamp(mut event: StreamEvent) -> StreamEvent {
    if event.kind == StreamEventType::StepEnd
        && let Some(r) = event.data.get_mut("response")
    {
        r["provider"] = json!(GEMINI_SPEC.name);
    }
    event
}

#[async_trait]
impl Provider for VertexGeminiProvider {
    fn spec(&self) -> &ProviderSpec {
        &GEMINI_SPEC
    }

    fn transport(&self) -> Option<&HttpTransport> {
        Some(&self.inner.transport)
    }

    fn plan(&self, req: &CompletionRequest, plan: &mut ParameterPlan) {
        Provider::plan(&self.inner, req, plan);
    }

    fn payload(&self, req: &CompletionRequest) -> Option<Value> {
        Some(self.inner.build_payload(req))
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<ModelResponse> {
        let local = Self::local(req);
        let mut resp = self.inner.complete(local.as_ref().unwrap_or(req)).await.map_err(|e| relabel(e, GEMINI_SPEC.name))?;
        resp.provider = GEMINI_SPEC.name.into();
        Ok(resp)
    }

    fn stream<'a>(&'a self, req: &'a CompletionRequest) -> BoxStream<'a, Result<StreamEvent>> {
        Box::pin(async_stream::try_stream! {
            let local = Self::local(req);
            let mut inner = self.inner.stream(local.as_ref().unwrap_or(req));
            while let Some(event) = inner.next().await {
                yield restamp(event.map_err(|e| relabel(e, GEMINI_SPEC.name))?);
            }
        })
    }

    async fn embed(&self, texts: &[String], model: Option<&str>) -> Result<Vec<Vec<f32>>> {
        let name = tail(model.unwrap_or(EMBEDDING_MODEL));
        let instances: Vec<Value> = texts.iter().map(|t| json!({"content": t})).collect();
        let raw = self.inner.transport.post(&format!("/models/{name}:predict"), &json!({"instances": instances}), &[], None).await?;
        Ok(raw["predictions"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|p| p["embeddings"]["values"].as_array().map(|v| v.iter().filter_map(Value::as_f64).map(|x| x as f32).collect()).unwrap_or_default())
            .collect())
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        Err(Error::config("Vertex lists models in Model Garden"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_providers::testing::{Fake, cfg};
    use crate::types::Message;

    fn options(token: &str) -> VertexOptions {
        VertexOptions { project: Some("my-project".into()), region: Some("europe-west1".into()), access_token: Some(token.into()), credentials: None }
    }

    #[tokio::test]
    async fn builds_the_publisher_url_and_moves_the_version_into_the_body() {
        let fake = Fake::json(json!({
            "content": [{"type": "text", "text": "hello from vertex"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 9, "output_tokens": 3}
        }))
        .await;
        let p = VertexProvider::with_config(options("a-token"), cfg(&fake.url)).unwrap();
        let resp = p.complete(&CompletionRequest::new("claude-opus-5", vec![Message::user("hello")])).await.unwrap();
        let seen = fake.last();
        assert_eq!(seen.path, "/v1/projects/my-project/locations/europe-west1/publishers/anthropic/models/claude-opus-5:rawPredict");
        assert_eq!(seen.headers["authorization"], "Bearer a-token");
        assert_eq!(seen.body["anthropic_version"], VERTEX_VERSION);
        assert!(seen.body.get("model").is_none());
        assert!(seen.body.get("stream").is_none());
        assert_eq!(resp.text(), "hello from vertex");
        assert_eq!(resp.provider, "vertex");
    }

    #[tokio::test]
    async fn claude_streams_over_sse() {
        let fake = Fake::sse(&[
            json!({"type": "message_start", "message": {"model": "claude-opus-5", "usage": {"input_tokens": 4}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        ])
        .await;
        let p = VertexProvider::with_config(options("t"), cfg(&fake.url)).unwrap();
        let req = CompletionRequest::new("publishers/anthropic/models/claude-opus-5", vec![Message::user("x")]);
        let events: Vec<StreamEvent> = p.stream(&req).map(|e| e.unwrap()).collect().await;
        assert_eq!(events[0].text, "hi");
        let resp = super::super::base::response_from_event(events.last().unwrap()).unwrap();
        assert_eq!(resp.provider, "vertex");
        let seen = fake.last();
        assert!(seen.path.ends_with("/publishers/anthropic/models/claude-opus-5:streamRawPredict"), "{}", seen.path);
        assert_eq!(seen.body["stream"], true);
    }

    #[tokio::test]
    async fn gemini_uses_the_google_publisher() {
        let fake = Fake::json(json!({
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "gemini on vertex"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}
        }))
        .await;
        let p = VertexGeminiProvider::with_config(
            VertexOptions { project: Some("p".into()), region: Some("us-central1".into()), access_token: Some("t".into()), credentials: None },
            cfg(&fake.url),
        )
        .unwrap();
        let mut req = CompletionRequest::new("gemini-2.5-pro", vec![Message::user("hi")]);
        req.system = Some("Be brief.".into());
        let resp = p.complete(&req).await.unwrap();
        let seen = fake.last();
        assert_eq!(seen.path, "/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-pro:generateContent");
        assert_eq!(seen.body["systemInstruction"]["parts"][0]["text"], "Be brief.");
        assert_eq!(resp.text(), "gemini on vertex");
        assert_eq!(resp.provider, "vertex-gemini");
    }

    #[tokio::test]
    async fn gemini_streams_and_embeds() {
        let fake = Fake::sse(&[json!({"candidates": [{"content": {"parts": [{"text": "a"}]}, "finishReason": "STOP"}]})]).await;
        let p = VertexGeminiProvider::with_config(options("t"), cfg(&fake.url)).unwrap();
        let req = CompletionRequest::new("publishers/google/models/gemini-2.5-flash", vec![Message::user("x")]);
        let events: Vec<StreamEvent> = p.stream(&req).map(|e| e.unwrap()).collect().await;
        assert_eq!(events[0].text, "a");
        assert_eq!(events.last().unwrap().data["response"]["provider"], "vertex-gemini");
        assert!(fake.last().path.ends_with("/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"), "{}", fake.last().path);

        let fake = Fake::json(json!({"predictions": [{"embeddings": {"values": [0.5, 1.0]}}, {"embeddings": {"values": [2.0]}}]})).await;
        let p = VertexGeminiProvider::with_config(options("t"), cfg(&fake.url)).unwrap();
        let vectors = p.embed(&["a".into(), "b".into()], None).await.unwrap();
        assert_eq!(vectors, vec![vec![0.5, 1.0], vec![2.0]]);
        let seen = fake.last();
        assert!(seen.path.ends_with("/publishers/google/models/gemini-embedding-001:predict"));
        assert_eq!(seen.body, json!({"instances": [{"content": "a"}, {"content": "b"}]}));
    }

    #[test]
    fn the_global_region_uses_the_unprefixed_host() {
        let p =
            VertexProvider::new(VertexOptions { project: Some("p".into()), region: Some("global".into()), access_token: Some("t".into()), credentials: None })
                .unwrap();
        assert!(p.endpoint().starts_with("https://aiplatform.googleapis.com/v1/projects/p/locations/global"));
        assert_eq!(vertex_endpoint("p", "us-east5", None), "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5");
    }

    #[test]
    fn without_a_project_it_says_so() {
        if env("GOOGLE_CLOUD_PROJECT").is_some() {
            return; // cannot unset it safely while other tests run
        }
        let err = VertexProvider::new(VertexOptions { access_token: Some("t".into()), ..Default::default() }).err().unwrap();
        assert!(err.to_string().contains("needs a project"), "{err}");
    }

    #[tokio::test]
    async fn the_metadata_server_token_is_cached_and_refreshed_after_a_401() {
        // The metadata server and Vertex are the same fake: first a token, then
        // a 401, a token again, and an answer.
        let token = json!({"access_token": "meta-token", "expires_in": 3600}).to_string().into_bytes();
        let answer = json!({"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}}).to_string().into_bytes();
        let ct = "application/json".to_string();
        let fake = Fake::start(vec![
            (200, ct.clone(), token.clone()),
            (401, ct.clone(), br#"{"error":{"message":"expired"}}"#.to_vec()),
            (200, ct.clone(), token),
            (200, ct.clone(), answer.clone()),
            (200, ct, answer),
        ])
        .await;
        let auth = GoogleCredentials::Metadata(format!("{}/token", fake.url));
        let opts = VertexOptions { project: Some("p".into()), region: Some("r".into()), access_token: None, credentials: Some(auth) };
        let p = VertexProvider::with_config(opts, cfg(&fake.url)).unwrap();
        let req = CompletionRequest::new("claude-opus-5", vec![Message::user("x")]);
        assert_eq!(p.complete(&req).await.unwrap().text(), "ok");
        p.complete(&req).await.unwrap(); // the cached token: no third token fetch
        let seen = fake.all();
        assert_eq!(seen.len(), 5);
        assert_eq!(seen[0].path, "/token");
        assert_eq!(seen[0].headers["metadata-flavor"], "Google");
        assert_eq!(seen[1].headers["authorization"], "Bearer meta-token");
        assert_eq!(seen[2].path, "/token");
        assert!(seen[4].path.ends_with(":rawPredict"));
    }

    #[tokio::test]
    async fn a_refresh_token_is_swapped_for_an_access_token() {
        let fake = Fake::json(json!({"access_token": "user-token", "expires_in": 3599})).await;
        let creds = GoogleCredentials::from_json(
            &json!({"type": "authorized_user", "client_id": "cid", "client_secret": "cs", "refresh_token": "r/t", "token_uri": format!("{}/oauth", fake.url)})
                .to_string(),
        )
        .unwrap();
        let auth = GoogleAuth::from_credentials(creds);
        assert_eq!(auth.token().await.unwrap(), "user-token");
        assert_eq!(auth.token().await.unwrap(), "user-token");
        let seen = fake.all();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].raw, "grant_type=refresh_token&client_id=cid&client_secret=cs&refresh_token=r%2Ft");
        assert!(GoogleCredentials::from_json(r#"{"type": "external_account"}"#).is_err());
    }

    #[tokio::test]
    async fn a_token_function_is_used_and_errors_name_every_source() {
        let f: TokenFn = Arc::new(|| Box::pin(async { Ok("from-fn".to_string()) }));
        assert_eq!(GoogleAuth::from_credentials(GoogleCredentials::Function(f)).token().await.unwrap(), "from-fn");
        let auth = GoogleAuth::with_chain(None, vec![GoogleCredentials::KeyFile("/definitely/not/here.json".into())]);
        let err = auth.token().await.unwrap_err();
        assert!(err.to_string().contains("not/here.json"), "{err}");
        assert_eq!(err.as_provider().unwrap().kind, K::Authentication);
    }

    /// `AH_LIVE_VERTEX=1` with `GOOGLE_CLOUD_PROJECT` and Google credentials.
    #[tokio::test]
    async fn live_vertex() {
        if std::env::var("AH_LIVE_VERTEX").as_deref() != Ok("1") {
            return;
        }
        let p = VertexGeminiProvider::new(VertexOptions::default()).unwrap();
        let mut req = CompletionRequest::new(std::env::var("AH_VERTEX_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".into()), vec![Message::user("Say hi.")]);
        req.max_tokens = 50;
        assert!(!p.complete(&req).await.unwrap().text().is_empty());
    }
}
