//! Azure OpenAI and Azure AI Foundry.
//!
//! Azure keeps the OpenAI wire format and changes the envelope: the model is a
//! *deployment* in the URL, the API version is a query parameter, and the key
//! goes in an `api-key` header rather than a bearer token.
//!
//! ```no_run
//! use agent_harness::llm_providers::azure::{AzureOpenAIProvider, AzureOptions};
//!
//! let azure = AzureOpenAIProvider::new(AzureOptions {
//!     endpoint: Some("https://my-resource.openai.azure.com".into()),
//!     deployment: Some("gpt-4.1".into()),
//!     ..Default::default()
//! })
//! .unwrap();
//! let foundry = AzureOpenAIProvider::foundry(AzureOptions {
//!     endpoint: Some("https://my-project.services.ai.azure.com".into()),
//!     deployment: Some("claude-opus-5".into()),
//!     ..Default::default()
//! })
//! .unwrap();
//! ```
//!
//! Both take a `credential` for Entra ID instead of a key ([`AzureCredential`]):
//! a token function (what `azure-identity`'s `get_token` is in Python), a
//! ready-made token, or [`AzureCredential::Default`] — a client secret or
//! federated token from `AZURE_TENANT_ID` / `AZURE_CLIENT_ID` /
//! `AZURE_CLIENT_SECRET` / `AZURE_FEDERATED_TOKEN_FILE`, else managed identity,
//! else `az account get-access-token`. Completion, streaming and embeddings all
//! go through the same retrying transport.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use reqwest::Method;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Map, Value, json};

use super::base::{Auth, CompletionRequest, HttpTransport, Provider, ProviderField, ProviderSpec, TransportConfig, default_client, key_from_env, sse_events};
use super::openai::{OPENAI, OpenAIFlavor, OpenAIProvider};
use super::parameters::ParameterPlan;
use super::sigv4::uri_encode;
use super::vertex::token_request;
use crate::errors::{Error, ProviderError, ProviderErrorKind as K, Result};
use crate::memory::providers::azure_blob::AzureAuth;
use crate::types::{MediaBlock, MediaKind, ModelResponse, StreamEvent, StreamEventType, now};

pub const SCOPE: &str = "https://cognitiveservices.azure.com/.default";
const RESOURCE: &str = "https://cognitiveservices.azure.com";

/// An Entra ID token and when it expires (seconds since the epoch).
pub type AzureTokenFn = Arc<dyn Fn() -> BoxFuture<'static, Result<(String, f64)>> + Send + Sync>;

/// Entra ID, in place of a key.
#[derive(Clone)]
pub enum AzureCredential {
    /// A bearer token you already have. Never refreshed.
    Token(String),
    /// Called for a token whenever the cached one is within a minute of expiring.
    Function(AzureTokenFn),
    /// From the environment: client secret or federated token, else managed
    /// identity, else the Azure CLI.
    Default,
}

impl std::fmt::Debug for AzureCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Token(_) => "Token",
            Self::Function(_) => "Function",
            Self::Default => "Default",
        })
    }
}

const AZURE_FIELDS: [ProviderField; 5] = [
    ProviderField {
        required: true,
        env: &["AZURE_OPENAI_ENDPOINT"],
        example: Some("https://my-resource.openai.azure.com"),
        ..ProviderField::new("endpoint", "url", "The resource endpoint from the Azure portal.")
    },
    ProviderField {
        env: &["AZURE_OPENAI_DEPLOYMENT"],
        example: Some("gpt-4.1-prod"),
        ..ProviderField::new("deployment", "str", "The deployment name. Defaults to the model id.")
    },
    ProviderField::new("api_version", "str", "The Azure OpenAI REST API version (default 2024-10-21)."),
    ProviderField {
        one_of: Some("auth"),
        env: &["AZURE_OPENAI_API_KEY"],
        ..ProviderField::new("api_key", "secret", "The resource key (Keys and Endpoint blade).")
    },
    ProviderField {
        one_of: Some("auth"),
        ..ProviderField::new("credential", "object", "AzureCredential, for Entra ID instead of a key (as a setting: \"default\" or {\"token\": ...}).")
    },
];

const FOUNDRY_FIELDS: [ProviderField; 6] = [
    ProviderField {
        required: true,
        env: &["AZURE_AI_ENDPOINT"],
        example: Some("https://my-project.services.ai.azure.com"),
        ..ProviderField::new("endpoint", "url", "The Foundry project endpoint.")
    },
    ProviderField { example: Some("claude-opus-5"), ..ProviderField::new("deployment", "str", "The model deployment. Defaults to the model id.") },
    ProviderField::new("api_version", "str", "The Foundry inference API version (default 2024-05-01-preview)."),
    ProviderField::new("path_template", "str", "Override the route if Foundry moves it again."),
    ProviderField { one_of: Some("auth"), env: &["AZURE_AI_API_KEY"], ..ProviderField::new("api_key", "secret", "The project key.") },
    ProviderField { one_of: Some("auth"), ..ProviderField::new("credential", "object", "AzureCredential, for Entra ID.") },
];

pub const SPEC: ProviderSpec = ProviderSpec {
    name: "azure",
    display_name: "Azure OpenAI",
    description: "OpenAI models deployed in your own Azure OpenAI resource.",
    docs_url: "https://learn.microsoft.com/azure/ai-services/openai/reference",
    auth_type: "api_key or Entra ID",
    env_key: "AZURE_OPENAI_API_KEY",
    default_model: "",
    base_url: "",
    capabilities: &["streaming", "tools", "vision", "thinking", "json_schema", "embeddings"],
    parameters: OPENAI.spec.parameters,
    modalities: OPENAI.spec.modalities,
    fields: &AZURE_FIELDS,
};

pub const FOUNDRY_SPEC: ProviderSpec = ProviderSpec {
    name: "azure-foundry",
    display_name: "Azure AI Foundry",
    description: "Many model families (Claude, Llama, Mistral, GPT, ...) served from one Azure AI Foundry endpoint.",
    docs_url: "https://learn.microsoft.com/azure/ai-foundry/",
    env_key: "AZURE_AI_API_KEY",
    fields: &FOUNDRY_FIELDS,
    ..SPEC
};

/// `{deployment}` and `{api_version}` are filled in per request.
pub const AZURE_PATH: &str = "/openai/deployments/{deployment}/chat/completions?api-version={api_version}";
pub const FOUNDRY_PATH: &str = "/models/chat/completions?api-version={api_version}";

/// Where the resource is and how to authenticate to it.
#[derive(Debug, Clone, Default)]
pub struct AzureOptions {
    /// Else `AZURE_OPENAI_ENDPOINT` (Foundry: `AZURE_AI_ENDPOINT`), else `TransportConfig::base_url`.
    pub endpoint: Option<String>,
    /// Else `AZURE_OPENAI_DEPLOYMENT`, else the request's model id.
    pub deployment: Option<String>,
    /// `2024-10-21` (Foundry: `2024-05-01-preview`) when not given.
    pub api_version: Option<String>,
    /// Else `AZURE_OPENAI_API_KEY` (Foundry: `AZURE_AI_API_KEY`).
    pub api_key: Option<String>,
    /// Entra ID instead of the key.
    pub credential: Option<AzureCredential>,
    /// Override the chat route; see [`AZURE_PATH`].
    pub path_template: Option<String>,
}

struct AzureKeyAuth {
    provider: &'static str,
    env_key: &'static str,
    key: String,
    credential: Option<AzureCredential>,
    cache: Mutex<(String, f64)>,
    fetching: tokio::sync::Mutex<()>,
    client: reqwest::Client,
}

impl AzureKeyAuth {
    fn cached(&self) -> Option<String> {
        let c = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        (!c.0.is_empty() && now() < c.1 - 60.0).then(|| c.0.clone())
    }

    /// An Entra ID token, cached until shortly before it expires.
    async fn bearer(&self, credential: &AzureCredential) -> Result<String> {
        if let AzureCredential::Token(t) = credential {
            return Ok(t.clone());
        }
        if let Some(t) = self.cached() {
            return Ok(t);
        }
        let _one_at_a_time = self.fetching.lock().await;
        if let Some(t) = self.cached() {
            return Ok(t);
        }
        let (token, expires) = match credential {
            AzureCredential::Token(t) => (t.clone(), f64::MAX),
            AzureCredential::Function(f) => f().await?,
            AzureCredential::Default => self.default_token().await?,
        };
        *self.cache.lock().unwrap_or_else(|p| p.into_inner()) = (token.clone(), expires);
        Ok(token)
    }

    async fn default_token(&self) -> Result<(String, f64)> {
        let fail = |e: String| -> Error {
            ProviderError::new(K::Authentication, format!("{}: no Entra ID token ({e})", self.provider)).provider(self.provider).into()
        };
        match AzureAuth::from_environment() {
            AzureAuth::ClientCredentials { authority, tenant, client_id, secret, assertion_file } => {
                let mut form: Vec<(&str, String)> = vec![("grant_type", "client_credentials".into()), ("client_id", client_id), ("scope", SCOPE.into())];
                if let Some(s) = secret {
                    form.push(("client_secret", s));
                }
                if let Some(file) = assertion_file {
                    let assertion = std::fs::read_to_string(&file).map_err(|e| fail(format!("{file}: {e}")))?;
                    form.push(("client_assertion_type", "urn:ietf:params:oauth:client-assertion-type:jwt-bearer".into()));
                    form.push(("client_assertion", assertion.trim().to_string()));
                }
                let pairs: Vec<(&str, &str)> = form.iter().map(|(k, v)| (*k, v.as_str())).collect();
                let url = format!("{}/{tenant}/oauth2/v2.0/token", authority.trim_end_matches('/'));
                token_request(&self.client, &url, Some(&pairs), &[]).await.map_err(fail)
            }
            AzureAuth::ManagedIdentity { client_id } => {
                let id = client_id.map(|c| format!("&client_id={}", uri_encode(&c, false))).unwrap_or_default();
                let resource = uri_encode(RESOURCE, false);
                let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
                let managed = match (env("IDENTITY_ENDPOINT"), env("IDENTITY_HEADER")) {
                    (Some(e), Some(h)) => {
                        token_request(&self.client, &format!("{e}?api-version=2019-08-01&resource={resource}{id}"), None, &[("x-identity-header", &h)]).await
                    }
                    _ => {
                        let url = format!("http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource={resource}{id}");
                        token_request(&self.client, &url, None, &[("metadata", "true")]).await
                    }
                };
                match managed {
                    Ok(t) => Ok(t),
                    Err(m) => az_cli_token().await.map_err(|c| fail(format!("managed identity: {m}; Azure CLI: {c}"))),
                }
            }
            _ => Err(fail("unsupported credential".into())),
        }
    }
}

/// `az account get-access-token`, as `DefaultAzureCredential` falls back to.
async fn az_cli_token() -> std::result::Result<(String, f64), String> {
    let mut command = if cfg!(windows) {
        let mut c = tokio::process::Command::new("cmd");
        c.args(["/C", "az"]);
        c
    } else {
        tokio::process::Command::new("az")
    };
    command.args(["account", "get-access-token", "--resource", RESOURCE, "--output", "json"]).kill_on_drop(true);
    let output = match tokio::time::timeout(std::time::Duration::from_secs(30), command.output()).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => return Err(e.to_string()),
        Err(_) => return Err("timed out".into()),
    };
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().chars().take(300).collect());
    }
    let v: Value = serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())?;
    let token = v["accessToken"].as_str().unwrap_or("").to_string();
    if token.is_empty() {
        return Err("no accessToken".into());
    }
    Ok((token, v["expires_on"].as_f64().unwrap_or_else(|| now() + 3000.0)))
}

#[async_trait]
impl Auth for AzureKeyAuth {
    async fn headers(&self, _: &Method, _: &str, _: &[u8]) -> Result<HeaderMap> {
        let mut h = HeaderMap::new();
        let v = |s: &str| HeaderValue::from_str(s).map_err(|e| Error::config(e.to_string()));
        match &self.credential {
            Some(c) => {
                h.insert("authorization", v(&format!("Bearer {}", self.bearer(c).await?))?);
            }
            None if self.key.is_empty() => {
                return Err(ProviderError::new(
                    K::Authentication,
                    format!("No API key for {}. Set ${}, pass api_key=..., or give credential= for Entra ID", self.provider, self.env_key),
                )
                .provider(self.provider)
                .into());
            }
            None => {
                h.insert("api-key", v(&self.key)?);
            }
        }
        Ok(h)
    }

    async fn refresh(&self) -> bool {
        match &self.credential {
            Some(AzureCredential::Token(_)) | None => false,
            Some(_) => {
                *self.cache.lock().unwrap_or_else(|p| p.into_inner()) = (String::new(), 0.0);
                true
            }
        }
    }
}

/// Azure AI Foundry is the same adapter on a different route.
pub type AzureFoundryProvider = AzureOpenAIProvider;

/// An Azure OpenAI deployment, or (built with [`AzureOpenAIProvider::foundry`])
/// an Azure AI Foundry endpoint.
///
/// Foundry serves many model families on one endpoint and routes by the model
/// in the body; its route is overridable because it has moved before and will
/// again.
pub struct AzureOpenAIProvider {
    inner: OpenAIProvider,
    pub endpoint: String,
    /// Empty: the request's model id is the deployment.
    pub deployment: String,
    pub api_version: String,
    pub path_template: String,
    pub foundry: bool,
}

impl AzureOpenAIProvider {
    pub fn new(options: AzureOptions) -> Result<Self> {
        Self::with_config(options, TransportConfig::default())
    }

    pub fn with_config(options: AzureOptions, config: TransportConfig) -> Result<Self> {
        Self::build(options, config, false)
    }

    pub fn foundry(options: AzureOptions) -> Result<Self> {
        Self::foundry_with_config(options, TransportConfig::default())
    }

    pub fn foundry_with_config(options: AzureOptions, config: TransportConfig) -> Result<Self> {
        Self::build(options, config, true)
    }

    fn build(options: AzureOptions, mut config: TransportConfig, foundry: bool) -> Result<Self> {
        let spec = if foundry { FOUNDRY_SPEC } else { SPEC };
        let endpoint_env = if foundry { "AZURE_AI_ENDPOINT" } else { "AZURE_OPENAI_ENDPOINT" };
        let endpoint = options
            .endpoint
            .filter(|e| !e.is_empty())
            .or_else(|| std::env::var(endpoint_env).ok().filter(|e| !e.is_empty()))
            .or_else(|| Some(config.base_url.clone()).filter(|e| !e.is_empty()))
            .map(|e| e.trim_end_matches('/').to_string())
            .ok_or_else(|| {
                Error::from(
                    ProviderError::new(K::Other, format!("{} needs an endpoint — pass endpoint=... or set {endpoint_env}", spec.display_name))
                        .provider(spec.name),
                )
            })?;
        config.base_url = endpoint.clone();
        let flavor = OpenAIFlavor { spec, key_required: false, ..OPENAI };
        let spec = &flavor.spec;
        let auth = AzureKeyAuth {
            provider: spec.name,
            env_key: spec.env_key,
            key: key_from_env(options.api_key, spec.env_key),
            credential: options.credential,
            cache: Mutex::new((String::new(), 0.0)),
            fetching: tokio::sync::Mutex::new(()),
            client: default_client(30.0, 3.0),
        };
        let inner = OpenAIProvider { transport: HttpTransport::new(flavor.spec.name, config, Arc::new(auth)), flavor };
        let deployment = options
            .deployment
            .filter(|d| !d.is_empty())
            .or_else(|| std::env::var("AZURE_OPENAI_DEPLOYMENT").ok().filter(|d| !d.is_empty()))
            .unwrap_or_default();
        let (version, path) = if foundry { ("2024-05-01-preview", FOUNDRY_PATH) } else { ("2024-10-21", AZURE_PATH) };
        Ok(Self {
            inner,
            endpoint,
            deployment,
            api_version: options.api_version.filter(|v| !v.is_empty()).unwrap_or_else(|| version.into()),
            path_template: options.path_template.unwrap_or_else(|| path.into()),
            foundry,
        })
    }

    fn deployment_for(&self, model: &str) -> Result<String> {
        let d = if self.deployment.is_empty() { model } else { &self.deployment };
        if d.is_empty() {
            return Err(ProviderError::new(K::Other, "Azure needs a deployment name — pass deployment=... or use the deployment as the model id")
                .provider(self.name())
                .into());
        }
        Ok(d.to_string())
    }

    /// The chat route for this request, with the deployment and version filled in.
    pub fn chat_path(&self, req: &CompletionRequest) -> Result<String> {
        let deployment = if self.foundry { self.deployment_for(&req.model).unwrap_or_default() } else { self.deployment_for(&req.model)? };
        Ok(self.path_template.replace("{deployment}", &deployment).replace("{api_version}", &self.api_version))
    }

    /// The OpenAI body. Azure OpenAI drops the model (the deployment is in the
    /// URL); Foundry routes by it, so it is the deployment, or the model id.
    pub fn build_payload(&self, req: &CompletionRequest, stream: bool) -> Value {
        let mut payload = self.inner.build_payload(req, stream);
        if let Some(p) = payload.as_object_mut() {
            if self.foundry {
                let model = if self.deployment.is_empty() { req.model.clone() } else { self.deployment.clone() };
                p.insert("model".into(), json!(model));
            } else {
                p.shift_remove("model");
            }
        }
        payload
    }

    fn embeddings_path(&self, model: &str) -> String {
        if self.foundry {
            format!("/models/embeddings?api-version={}", self.api_version)
        } else {
            format!("/openai/deployments/{model}/embeddings?api-version={}", self.api_version)
        }
    }
}

#[async_trait]
impl Provider for AzureOpenAIProvider {
    fn spec(&self) -> &ProviderSpec {
        &self.inner.flavor.spec
    }

    fn transport(&self) -> Option<&HttpTransport> {
        Some(&self.inner.transport)
    }

    fn accepts(&self, kind: MediaKind, block: &MediaBlock, model: &str) -> bool {
        self.inner.accepts(kind, block, model)
    }

    fn plan(&self, req: &CompletionRequest, plan: &mut ParameterPlan) {
        Provider::plan(&self.inner, req, plan);
    }

    fn payload(&self, req: &CompletionRequest) -> Option<Value> {
        Some(self.build_payload(req, false))
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<ModelResponse> {
        let started = Instant::now();
        let raw = self.inner.transport.post(&self.chat_path(req)?, &self.build_payload(req, false), &[], req.timeout).await?;
        self.inner.decode(raw, req, started.elapsed().as_secs_f64() * 1000.0)
    }

    /// The deltas are gathered into the shape of a whole response, which the
    /// OpenAI decoder then reads — so a streamed answer maps exactly as a
    /// complete one does.
    fn stream<'a>(&'a self, req: &'a CompletionRequest) -> BoxStream<'a, Result<StreamEvent>> {
        self.inner.transport.retrying(move || {
            Box::pin(async_stream::try_stream! {
                let path = self.chat_path(req)?;
                let payload = self.build_payload(req, true);
                let mut text = String::new();
                let mut thinking = String::new();
                let mut calls: BTreeMap<u64, (String, String, String)> = BTreeMap::new();
                let mut usage = Value::Null;
                let mut finish = Value::Null;
                let mut model = req.model.clone();
                let mut events = sse_events(self.inner.transport.stream_lines(&path, &payload, &[], req.timeout));
                while let Some(item) = events.next().await {
                    let (_, chunk) = item?;
                    if chunk.is_empty() {
                        continue;
                    }
                    if chunk.trim() == "[DONE]" {
                        break;
                    }
                    let data: Value = serde_json::from_str(&chunk).map_err(|e| Error::config(format!("bad stream chunk: {e}")))?;
                    if !data["error"].is_null() {
                        Err(self.inner.body_error(&data["error"]))?;
                    }
                    if let Some(m) = data["model"].as_str().filter(|m| !m.is_empty()) {
                        model = m.into();
                    }
                    if data["usage"].is_object() {
                        usage = data["usage"].clone();
                    }
                    for choice in data["choices"].as_array().into_iter().flatten() {
                        let delta = &choice["delta"];
                        if let Some(r) = delta["reasoning_content"].as_str().or(delta["reasoning"].as_str()).filter(|r| !r.is_empty()) {
                            thinking.push_str(r);
                            yield StreamEvent { text: r.into(), ..StreamEvent::new(StreamEventType::Thinking) };
                        }
                        if let Some(c) = delta["content"].as_str().filter(|c| !c.is_empty()) {
                            text.push_str(c);
                            yield StreamEvent::text(c);
                        }
                        for call in delta["tool_calls"].as_array().into_iter().flatten() {
                            let idx = call["index"].as_u64().unwrap_or(calls.len() as u64);
                            let slot = calls.entry(idx).or_default();
                            if let Some(id) = call["id"].as_str().filter(|s| !s.is_empty()) {
                                slot.0 = id.into();
                            }
                            if let Some(n) = call["function"]["name"].as_str().filter(|s| !s.is_empty()) {
                                slot.1 = n.into();
                            }
                            if let Some(a) = call["function"]["arguments"].as_str() {
                                slot.2.push_str(a);
                            }
                        }
                        if choice["finish_reason"].is_string() {
                            finish = choice["finish_reason"].clone();
                        }
                    }
                }
                let mut message = Map::new();
                if !text.is_empty() {
                    message.insert("content".into(), json!(text));
                }
                if !thinking.is_empty() {
                    message.insert("reasoning_content".into(), json!(thinking));
                }
                let tool_calls: Vec<Value> = calls.into_values().map(|(id, name, args)| json!({"id": id, "function": {"name": name, "arguments": args}})).collect();
                if !tool_calls.is_empty() {
                    message.insert("tool_calls".into(), json!(tool_calls));
                }
                let whole = json!({"model": model, "choices": [{"finish_reason": finish, "message": message}], "usage": usage});
                let mut resp = self.inner.decode(whole, req, 0.0)?;
                resp.raw = Some(json!({}));
                for u in resp.tool_uses() {
                    yield StreamEvent::new(StreamEventType::ToolCall).with_data("id", json!(u.id)).with_data("name", json!(u.name)).with_data("input", Value::Object(u.input.clone()));
                }
                yield StreamEvent::new(StreamEventType::StepEnd).with_data("response", serde_json::to_value(&resp).unwrap_or_default());
            })
        })
    }

    async fn embed(&self, texts: &[String], model: Option<&str>) -> Result<Vec<Vec<f32>>> {
        let name = model.unwrap_or(self.inner.flavor.embedding_model);
        let body = if self.foundry { json!({"model": name, "input": texts}) } else { json!({"input": texts}) };
        let raw = self.inner.transport.post(&self.embeddings_path(name), &body, &[], None).await?;
        let mut rows: Vec<&Value> = raw["data"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
        rows.sort_by_key(|r| r["index"].as_u64().unwrap_or(0));
        Ok(rows
            .into_iter()
            .map(|r| r["embedding"].as_array().map(|e| e.iter().filter_map(Value::as_f64).map(|x| x as f32).collect()).unwrap_or_default())
            .collect())
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        Err(Error::config("Azure deployments are listed in the portal"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_providers::testing::{Fake, cfg};
    use crate::types::{Message, StopReason};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn reply(text: &str) -> Value {
        json!({"model": "gpt-4.1", "choices": [{"finish_reason": "stop", "message": {"content": text}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    }

    fn opts(url: &str) -> AzureOptions {
        AzureOptions { endpoint: Some(url.into()), api_key: Some("secret".into()), ..Default::default() }
    }

    #[tokio::test]
    async fn uses_a_deployment_and_an_api_key() {
        let fake = Fake::json(reply("hello from azure")).await;
        let p = AzureOpenAIProvider::with_config(
            AzureOptions { deployment: Some("gpt-4.1-prod".into()), api_version: Some("2024-10-21".into()), ..opts(&fake.url) },
            cfg(""),
        )
        .unwrap();
        let resp = p.complete(&CompletionRequest::new("gpt-4.1", vec![Message::user("hello")])).await.unwrap();
        let seen = fake.last();
        assert_eq!(seen.path, "/openai/deployments/gpt-4.1-prod/chat/completions?api-version=2024-10-21");
        assert_eq!(seen.headers["api-key"], "secret");
        assert!(!seen.headers.contains_key("authorization"));
        assert!(seen.body.get("model").is_none());
        assert_eq!(resp.text(), "hello from azure");
        assert_eq!(resp.provider, "azure");
        assert_eq!(resp.usage.input_tokens, 7);
    }

    #[tokio::test]
    async fn the_model_id_is_used_when_no_deployment_is_given() {
        let fake = Fake::json(reply("x")).await;
        let p = AzureOpenAIProvider::with_config(opts(&fake.url), cfg("")).unwrap();
        if !p.deployment.is_empty() {
            return; // AZURE_OPENAI_DEPLOYMENT is set in this environment
        }
        p.complete(&CompletionRequest::new("my-deployment", vec![Message::user("x")])).await.unwrap();
        assert!(fake.last().path.contains("/deployments/my-deployment/"));
    }

    #[tokio::test]
    async fn entra_id_is_used_instead_of_a_key_and_cached() {
        let fake = Fake::json(reply("x")).await;
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let f: AzureTokenFn = Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(("entra-token".to_string(), 9_999_999_999.0)) })
        });
        let p = AzureOpenAIProvider::with_config(
            AzureOptions {
                endpoint: Some(fake.url.clone()),
                deployment: Some("d".into()),
                credential: Some(AzureCredential::Function(f)),
                ..Default::default()
            },
            cfg(""),
        )
        .unwrap();
        for text in ["x", "y"] {
            p.complete(&CompletionRequest::new("d", vec![Message::user(text)])).await.unwrap();
        }
        assert_eq!(fake.last().headers["authorization"], "Bearer entra-token");
        assert!(!fake.last().headers.contains_key("api-key"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn foundry_routes_differently_and_keeps_the_model_in_the_body() {
        let fake = Fake::json(reply("from foundry")).await;
        let p = AzureOpenAIProvider::foundry_with_config(AzureOptions { deployment: Some("claude-opus-5".into()), ..opts(&fake.url) }, cfg("")).unwrap();
        let resp = p.complete(&CompletionRequest::new("claude-opus-5", vec![Message::user("hi")])).await.unwrap();
        let seen = fake.last();
        assert_eq!(seen.path, "/models/chat/completions?api-version=2024-05-01-preview");
        assert_eq!(seen.body["model"], "claude-opus-5");
        assert_eq!(resp.text(), "from foundry");
        assert_eq!(resp.provider, "azure-foundry");

        let p = AzureOpenAIProvider::foundry_with_config(
            AzureOptions { deployment: Some("d".into()), path_template: Some("/openai/v1/chat/completions".into()), ..opts(&fake.url) },
            cfg(""),
        )
        .unwrap();
        p.complete(&CompletionRequest::new("d", vec![Message::user("x")])).await.unwrap();
        assert_eq!(fake.last().path, "/openai/v1/chat/completions");
    }

    #[test]
    fn without_an_endpoint_it_says_so() {
        if std::env::var("AZURE_OPENAI_ENDPOINT").is_ok() {
            return;
        }
        let err = AzureOpenAIProvider::new(AzureOptions { api_key: Some("k".into()), ..Default::default() }).err().unwrap();
        assert!(err.to_string().contains("needs an endpoint"), "{err}");
    }

    #[tokio::test]
    async fn without_a_key_it_says_where_to_put_one() {
        if std::env::var("AZURE_OPENAI_API_KEY").is_ok() {
            return;
        }
        let p = AzureOpenAIProvider::with_config(AzureOptions { endpoint: Some("http://127.0.0.1:9".into()), ..Default::default() }, cfg("")).unwrap();
        let err = p.complete(&CompletionRequest::new("d", vec![Message::user("x")])).await.unwrap_err();
        assert!(err.to_string().contains("AZURE_OPENAI_API_KEY"), "{err}");
    }

    #[tokio::test]
    async fn streams_tool_calls_through_the_decoder() {
        let fake = Fake::sse(&[
            json!({"choices": [{"delta": {"content": "Let me "}}]}),
            json!({"choices": [{"delta": {"content": "check", "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "look", "arguments": "{\"q\":"}}]}}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}),
        ])
        .await;
        let p = AzureOpenAIProvider::with_config(AzureOptions { deployment: Some("dep".into()), ..opts(&fake.url) }, cfg("")).unwrap();
        let req = CompletionRequest::new("gpt-4.1", vec![Message::user("x")]);
        let events: Vec<StreamEvent> = p.stream(&req).map(|e| e.unwrap()).collect().await;
        let text: String = events.iter().filter(|e| e.kind == StreamEventType::Text).map(|e| e.text.as_str()).collect();
        assert_eq!(text, "Let me check");
        let call = events.iter().find(|e| e.kind == StreamEventType::ToolCall).unwrap();
        assert_eq!(call.data["input"], json!({"q": 1}));
        let resp = super::super::base::response_from_event(events.last().unwrap()).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.usage.output_tokens, 2);
        assert_eq!(resp.provider, "azure");
        let seen = fake.last();
        assert_eq!(seen.path, "/openai/deployments/dep/chat/completions?api-version=2024-10-21");
        assert_eq!(seen.body["stream"], true);
    }

    #[tokio::test]
    async fn embeddings_by_deployment_and_by_model() {
        let fake = Fake::json(json!({"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]})).await;
        let p = AzureOpenAIProvider::with_config(opts(&fake.url), cfg("")).unwrap();
        assert_eq!(p.embed(&["a".into(), "b".into()], Some("emb")).await.unwrap(), vec![vec![1.0], vec![2.0]]);
        assert_eq!(fake.last().path, "/openai/deployments/emb/embeddings?api-version=2024-10-21");
        assert_eq!(fake.last().body, json!({"input": ["a", "b"]}));
        let f = AzureOpenAIProvider::foundry_with_config(opts(&fake.url), cfg("")).unwrap();
        f.embed(&["a".into()], Some("emb")).await.unwrap();
        assert_eq!(fake.last().path, "/models/embeddings?api-version=2024-05-01-preview");
        assert_eq!(fake.last().body, json!({"model": "emb", "input": ["a"]}));
    }

    /// `AH_LIVE_AZURE=1` with `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_DEPLOYMENT`.
    #[tokio::test]
    async fn live_azure() {
        if std::env::var("AH_LIVE_AZURE").as_deref() != Ok("1") {
            return;
        }
        let p = AzureOpenAIProvider::new(AzureOptions::default()).unwrap();
        let mut req = CompletionRequest::new("", vec![Message::user("Say hi.")]);
        req.max_tokens = 20;
        assert!(!p.complete(&req).await.unwrap().text().is_empty());
    }
}
