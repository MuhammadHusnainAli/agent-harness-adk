//! The provider contract every model backend implements.
//!
//! One shape in, one shape out. The agent loop never learns which vendor it is
//! talking to. Adapters speak raw HTTP through one [`HttpTransport`], so every
//! backend travels the same retry, timeout, circuit-breaker and cost path.

use std::collections::HashSet;
use std::sync::{Arc, LazyLock, Mutex, Once, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
use indexmap::IndexMap;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Method, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;

use super::parameters::{Effort, GENERATION_PARAMETERS, ParameterPlan, SAMPLING_PARAMETERS};
use super::resilience::{CircuitBreaker, ProviderStats, RetryEvent, RetryPolicy, classify, request_id};
use crate::errors::{Error, ProviderError, ProviderErrorKind as K, Result};
use crate::tools::ToolSchema;
use crate::types::{MediaBlock, MediaKind, Message, ModelResponse, StopReason, StreamEvent, StreamEventType, Usage, round8};

// ---------------------------------------------------------------------------
// models and pricing
// ---------------------------------------------------------------------------

/// How capable — and costly — a model is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Fast,
    #[default]
    Balanced,
    Deep,
}

/// What the harness needs to know about a model to route and bill it.
/// Costs are USD per million tokens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub provider: String,
    pub context_window: u64,
    pub max_output: u64,
    pub input_cost: f64,
    pub output_cost: f64,
    pub cache_read_cost: Option<f64>,
    pub cache_write_cost: Option<f64>,
    pub supports_tools: bool,
    pub supports_thinking: bool,
    pub tier: Tier,
}

impl ModelInfo {
    pub fn new(id: &str, provider: &str) -> Self {
        Self {
            id: id.into(),
            provider: provider.into(),
            context_window: 200_000,
            max_output: 8192,
            input_cost: 0.0,
            output_cost: 0.0,
            cache_read_cost: None,
            cache_write_cost: None,
            supports_tools: true,
            supports_thinking: false,
            tier: Tier::Balanced,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn m(id: &str, provider: &str, ctx: u64, out: u64, i: f64, o: f64, cr: Option<f64>, cw: Option<f64>, thinking: bool, tier: Tier) -> ModelInfo {
    ModelInfo {
        context_window: ctx,
        max_output: out,
        input_cost: i,
        output_cost: o,
        cache_read_cost: cr,
        cache_write_cost: cw,
        supports_thinking: thinking,
        tier,
        ..ModelInfo::new(id, provider)
    }
}

/// Prices are USD per million tokens. Override with [`register_model`].
pub static MODELS: LazyLock<RwLock<IndexMap<String, ModelInfo>>> = LazyLock::new(|| {
    use Tier::*;
    let list = [
        m("claude-opus-5", "anthropic", 1_000_000, 128_000, 5.0, 25.0, Some(0.5), Some(6.25), true, Deep),
        m("claude-fable-5-1", "anthropic", 1_000_000, 128_000, 10.0, 50.0, Some(0.25), Some(12.5), true, Deep),
        m("claude-opus-4-8", "anthropic", 1_000_000, 128_000, 5.0, 25.0, Some(0.5), Some(6.25), true, Deep),
        m("claude-sonnet-5", "anthropic", 1_000_000, 128_000, 2.0, 10.0, Some(0.2), Some(2.5), true, Balanced),
        m("claude-haiku-4-5", "anthropic", 200_000, 64_000, 1.0, 5.0, Some(0.1), Some(1.25), true, Fast),
        m("gpt-4.1", "openai", 1_000_000, 32_768, 2.0, 8.0, Some(0.5), None, false, Balanced),
        m("gpt-4.1-mini", "openai", 1_000_000, 32_768, 0.4, 1.6, Some(0.1), None, false, Fast),
        m("gpt-4o", "openai", 128_000, 16_384, 2.5, 10.0, Some(1.25), None, false, Balanced),
        m("gpt-4o-mini", "openai", 128_000, 16_384, 0.15, 0.6, Some(0.075), None, false, Fast),
        m("o3", "openai", 200_000, 100_000, 2.0, 8.0, None, None, true, Deep),
        m("gemini-2.5-pro", "gemini", 1_048_576, 65_536, 1.25, 10.0, None, None, true, Deep),
        m("gemini-2.5-flash", "gemini", 1_048_576, 65_536, 0.3, 2.5, None, None, true, Balanced),
        m("gemini-2.0-flash", "gemini", 1_048_576, 8192, 0.1, 0.4, None, None, false, Fast),
    ];
    RwLock::new(list.into_iter().map(|i| (i.id.clone(), i)).collect())
});

/// Teach the harness about a model it does not ship with.
pub fn register_model(info: ModelInfo) -> ModelInfo {
    MODELS.write().unwrap_or_else(|p| p.into_inner()).insert(info.id.clone(), info.clone());
    info
}

/// Strip platform prefixes so pricing and routing still recognise a model:
/// `anthropic.claude-opus-5`, `us.anthropic.claude-opus-5`, `claude-opus-5@20260401`.
pub fn normalise_model(model: &str) -> String {
    let mut name = model.split('@').next().unwrap_or(model).to_string();
    for prefix in ["anthropic.", "us.anthropic.", "eu.anthropic.", "apac.anthropic.", "google.", "publishers/anthropic/models/", "publishers/google/models/"] {
        if let Some(rest) = name.strip_prefix(prefix) {
            name = rest.to_string();
        }
    }
    for region in ["us.", "eu.", "apac."] {
        if name.starts_with(region) {
            name = name.split_once('.').map(|x| x.1).unwrap_or("").to_string();
            break;
        }
    }
    name
}

/// Exact match first, then longest known prefix — so dated snapshots still bill.
pub fn model_info(model: &str) -> Option<ModelInfo> {
    let models = MODELS.read().unwrap_or_else(|p| p.into_inner());
    let normal = normalise_model(model);
    for c in [model, normal.as_str()] {
        if let Some(info) = models.get(c) {
            return Some(info.clone());
        }
    }
    for c in [model, normal.as_str()] {
        let best = models.iter().filter(|(k, _)| c.starts_with(k.as_str())).max_by_key(|(k, _)| k.len());
        if let Some((_, info)) = best {
            return Some(info.clone());
        }
    }
    None
}

/// Infer the vendor from the model id, so callers rarely have to say it.
pub fn provider_for_model(model: &str) -> String {
    if let Some(info) = model_info(model) {
        return info.provider;
    }
    let low = model.to_lowercase();
    let starts = |ps: &[&str]| ps.iter().any(|p| low.starts_with(p));
    if starts(&["claude"]) {
        "anthropic"
    } else if starts(&["gpt", "o1", "o3", "o4", "chatgpt"]) {
        "openai"
    } else if starts(&["gemini", "models/gemini"]) {
        "gemini"
    } else if starts(&["deepseek"]) {
        "deepseek"
    } else if starts(&["grok"]) {
        "xai"
    } else if starts(&["mistral", "codestral", "magistral", "ministral", "devstral", "pixtral"]) {
        "mistral"
    } else {
        "anthropic"
    }
    .into()
}

/// Dollars for one call. Unknown models cost nothing — never guess a price up.
pub fn estimate_cost(model: &str, usage: &Usage) -> f64 {
    let Some(info) = model_info(model) else { return 0.0 };
    let cost = usage.input_tokens as f64 * info.input_cost
        + usage.output_tokens as f64 * info.output_cost
        + usage.cache_read_tokens as f64 * info.cache_read_cost.unwrap_or(info.input_cost)
        + usage.cache_write_tokens as f64 * info.cache_write_cost.unwrap_or(info.input_cost);
    round8(cost / 1_000_000.0)
}

// ---------------------------------------------------------------------------
// the request
// ---------------------------------------------------------------------------

/// Everything one model call needs, in a form every adapter accepts.
///
/// A parameter a provider does not have is dropped rather than guessed at.
/// `extra` goes into the payload untouched — the escape hatch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub tools: Vec<ToolSchema>,
    /// `"auto"`, `"any"`, `"none"`, or `{"name": ...}`.
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub effort: Option<Effort>,
    #[serde(default)]
    pub thinking: Option<bool>,
    /// An explicit thinking-token ceiling; -1 lets Gemini decide.
    #[serde(default)]
    pub thinking_budget: Option<i64>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub response_schema: Option<Value>,
    #[serde(default)]
    pub response_mime_type: Option<String>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Prompt caching, where supported.
    #[serde(default)]
    pub cache: Option<bool>,
    /// Anthropic fast mode.
    #[serde(default)]
    pub speed: Option<String>,
    /// End-user id, for abuse tracing.
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default)]
    pub safety_settings: Vec<Value>,
    /// Seconds, per attempt.
    #[serde(default)]
    pub timeout: Option<f64>,
    /// Merged into the payload verbatim.
    #[serde(default)]
    pub extra: Map<String, Value>,
}

fn default_max_tokens() -> u32 {
    8192
}

impl CompletionRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            system: None,
            tools: vec![],
            tool_choice: None,
            max_tokens: 8192,
            effort: None,
            thinking: None,
            thinking_budget: None,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            seed: None,
            stop: vec![],
            response_schema: None,
            response_mime_type: None,
            parallel_tool_calls: None,
            cache: None,
            speed: None,
            user: None,
            metadata: Map::new(),
            safety_settings: vec![],
            timeout: None,
            extra: Map::new(),
        }
    }

    /// The generation parameters, by name, as a [`ParameterPlan`] reads them.
    pub fn generation_values(&self) -> IndexMap<String, Value> {
        let v = serde_json::to_value(self).unwrap_or_default();
        GENERATION_PARAMETERS.iter().map(|n| (n.to_string(), v.get(*n).cloned().unwrap_or(Value::Null))).collect()
    }
}

/// One thing a provider needs (or can take) to connect.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderField {
    /// The constructor setting.
    pub name: &'static str,
    pub description: &'static str,
    /// str · secret · url · int · float · bool · object
    pub r#type: &'static str,
    pub required: bool,
    /// Satisfied if *any* field sharing this group is set.
    pub one_of: Option<&'static str>,
    /// Environment variables read, in order.
    pub env: &'static [&'static str],
    pub example: Option<&'static str>,
}

impl ProviderField {
    pub const fn new(name: &'static str, r#type: &'static str, description: &'static str) -> Self {
        Self { name, description, r#type, required: false, one_of: None, env: &[], example: None }
    }

    pub fn secret(&self) -> bool {
        self.r#type == "secret"
    }
}

/// Settings every HTTP-backed provider takes, on top of its own fields.
pub const COMMON_FIELDS: [ProviderField; 9] = [
    ProviderField::new("timeout", "float", "Seconds to wait for a response (per attempt)."),
    ProviderField::new("connect_timeout", "float", "Seconds to wait to open the connection."),
    ProviderField::new("max_retries", "int", "Retries on 429, 5xx, timeouts and dropped connections."),
    ProviderField::new("retry", "object", "A RetryPolicy: backoff, jitter, wall-clock ceiling, and the longest Retry-After to honour."),
    ProviderField::new("max_concurrency", "int", "Cap on in-flight requests from this provider instance."),
    ProviderField::new("circuit_breaker", "object", "A CircuitBreaker, or none."),
    ProviderField::new("on_retry", "object", "Called with a RetryEvent before every retry."),
    ProviderField::new("headers", "object", "Extra HTTP headers sent with every request."),
    ProviderField::new("client", "object", "Your own reqwest::Client (proxies, mTLS)."),
];

/// What `list_llm_providers()` shows about a backend.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderSpec {
    pub name: &'static str,
    pub display_name: &'static str,
    pub description: &'static str,
    pub docs_url: &'static str,
    /// api_key · bearer · sigv4 · google-oauth · none
    pub auth_type: &'static str,
    pub env_key: &'static str,
    pub default_model: &'static str,
    pub base_url: &'static str,
    /// streaming · tools · vision · thinking · json_schema · embeddings · list_models
    pub capabilities: &'static [&'static str],
    /// The sampling parameters this backend accepts. Anything else is dropped.
    pub parameters: &'static [&'static str],
    /// What a model here can be handed besides text.
    pub modalities: &'static [MediaKind],
    pub fields: &'static [ProviderField],
}

// ---------------------------------------------------------------------------
// the contract
// ---------------------------------------------------------------------------

/// A model backend. Implementations send through an [`HttpTransport`].
#[async_trait]
pub trait Provider: Send + Sync {
    /// What this backend is, what it needs, what it accepts.
    fn spec(&self) -> &ProviderSpec;

    fn name(&self) -> &str {
        self.spec().name
    }

    /// One round trip: messages in, one assistant message out.
    async fn complete(&self, req: &CompletionRequest) -> Result<ModelResponse>;

    /// Token stream, ending with a `step_end` event that carries the response.
    /// Backends without native streaming get one built from `complete`.
    fn stream<'a>(&'a self, req: &'a CompletionRequest) -> BoxStream<'a, Result<StreamEvent>> {
        Box::pin(async_stream::try_stream! {
            let resp = self.complete(req).await?;
            for event in events_from(&resp) {
                yield event;
            }
        })
    }

    async fn embed(&self, _texts: &[String], _model: Option<&str>) -> Result<Vec<Vec<f32>>> {
        Err(Error::config(format!("{} has no embedding endpoint wired up", self.name())))
    }

    /// Model ids this account can use, straight from the provider.
    async fn list_models(&self) -> Result<Vec<String>> {
        Err(Error::config(format!("{} cannot list its models", self.name())))
    }

    /// The transport, if this backend speaks HTTP — for stats and health.
    fn transport(&self) -> Option<&HttpTransport> {
        None
    }

    /// Model kinds this backend accepts besides text.
    fn modalities(&self) -> &[MediaKind] {
        self.spec().modalities
    }

    /// Can `model` be sent this attachment as it is? A document means a PDF.
    fn accepts(&self, kind: MediaKind, block: &MediaBlock, _model: &str) -> bool {
        if kind == MediaKind::Document && block.media_type != "application/pdf" {
            return false;
        }
        self.modalities().contains(&kind)
    }

    /// Model-specific parameter rules. Adapters override this.
    fn plan(&self, _req: &CompletionRequest, _plan: &mut ParameterPlan) {}

    /// Decide what this provider sends for the request's generation parameters.
    fn plan_parameters(&self, req: &CompletionRequest) -> ParameterPlan {
        let mut plan = ParameterPlan::new(self.name(), &req.model, req.generation_values());
        let label = if self.spec().display_name.is_empty() { self.name() } else { self.spec().display_name };
        for name in SAMPLING_PARAMETERS {
            if plan.contains(name) && !self.spec().parameters.contains(&name) {
                plan.drop(name, format!("{label} has no {name}"));
            }
        }
        self.plan(req, &mut plan);
        report(&plan);
        plan
    }

    /// The exact JSON body this request would put on the wire, if known.
    fn payload(&self, _req: &CompletionRequest) -> Option<Value> {
        None
    }

    /// What would be sent for this request — without sending it.
    fn explain(&self, req: &CompletionRequest) -> Value {
        let mut out = self.plan_parameters(req).as_json();
        if let Some(p) = self.payload(req) {
            out["payload"] = p;
        }
        out
    }

    /// Counters plus the circuit's state — what to put on a dashboard.
    fn health(&self) -> Value {
        let mut out = json!({"provider": self.name()});
        match self.transport() {
            Some(t) => {
                if let (Value::Object(o), Value::Object(s)) = (&mut out, t.stats().snapshot()) {
                    o.extend(s);
                }
                out["circuit"] = json!(t.circuit.as_ref().map(|c| c.state().as_str()).unwrap_or("disabled"));
            }
            None => out["circuit"] = json!("disabled"),
        }
        out
    }

    /// Is this provider reachable, and does it accept our credentials?
    /// Never fails — the answer is in the result.
    async fn ping(&self, model: Option<&str>) -> Value {
        let started = std::time::Instant::now();
        let ms = || (started.elapsed().as_secs_f64() * 10_000.0).round() / 10.0;
        let outcome: Result<Value> = async {
            match self.list_models().await {
                Ok(models) => Ok(json!({"models": models.len()})),
                Err(Error::Configuration(_)) => {
                    let target = model.unwrap_or(self.spec().default_model);
                    if target.is_empty() {
                        return Err(ProviderError::new(K::Other, format!("{}: pass a model to ping it", self.name())).into());
                    }
                    let mut req = CompletionRequest::new(target, vec![Message::user("ping")]);
                    req.max_tokens = 1;
                    let resp = self.complete(&req).await?;
                    Ok(json!({"model": resp.model}))
                }
                Err(e) => Err(e),
            }
        }
        .await;
        match outcome {
            Ok(Value::Object(detail)) => {
                let mut out = json!({"provider": self.name(), "ok": true, "latency_ms": ms()});
                out.as_object_mut().expect("object").extend(detail);
                out
            }
            Ok(_) => json!({"provider": self.name(), "ok": true, "latency_ms": ms()}),
            Err(e) => json!({"provider": self.name(), "ok": false, "latency_ms": ms(), "error": format!("{}: {e}", e.kind_name())}),
        }
    }

    /// Build a response, stamping the cost from the model table.
    fn finish(&self, message: Message, stop_reason: StopReason, mut usage: Usage, model: &str, raw: Value, latency_ms: f64) -> ModelResponse {
        usage.calls = usage.calls.max(1);
        usage.cost_usd = estimate_cost(model, &usage);
        ModelResponse { message, stop_reason, usage, model: model.into(), provider: self.name().into(), latency_ms, raw: Some(raw) }
    }
}

/// (provider, model, parameter, "dropped" | "adjusted")
type ReportKey = (String, String, String, &'static str);

static REPORTED: LazyLock<Mutex<HashSet<ReportKey>>> = LazyLock::new(Default::default);

/// Log each drop or adjustment once per provider, model and parameter.
fn report(plan: &ParameterPlan) {
    let mut seen = REPORTED.lock().unwrap_or_else(|p| p.into_inner());
    for (kind, notes) in [("dropped", &plan.dropped), ("adjusted", &plan.adjusted)] {
        for (name, why) in notes {
            let key = (plan.provider.clone(), plan.model.clone(), name.clone(), kind);
            if seen.contains(&key) {
                continue;
            }
            if seen.len() > 10_000 {
                seen.clear();
            }
            seen.insert(key);
            tracing::info!("{}/{}: {kind} {name} — {why}", plan.provider, plan.model);
        }
    }
}

/// The events a non-streaming response stands for.
pub fn events_from(resp: &ModelResponse) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    let text = resp.text();
    if !text.is_empty() {
        events.push(StreamEvent::text(text));
    }
    for u in resp.tool_uses() {
        events.push(
            StreamEvent::new(StreamEventType::ToolCall)
                .with_data("name", json!(u.name))
                .with_data("input", Value::Object(u.input.clone()))
                .with_data("id", json!(u.id)),
        );
    }
    events.push(StreamEvent::new(StreamEventType::StepEnd).with_data("response", serde_json::to_value(resp).unwrap_or_default()));
    events
}

/// The [`ModelResponse`] a `step_end` event carries.
pub fn response_from_event(event: &StreamEvent) -> Option<ModelResponse> {
    serde_json::from_value(event.data.get("response")?.clone()).ok()
}

/// An attachment's bytes as base64, or `None` if its file has gone.
pub fn media_data(block: &MediaBlock) -> Option<String> {
    block.load().ok()
}

/// What a model is told in place of an attachment it cannot be given.
pub fn media_note(kind: MediaKind, block: &MediaBlock) -> String {
    let label = if !block.name.is_empty() { block.name.clone() } else { block.url.clone().unwrap_or_else(|| kind.as_str().into()) };
    if block.inline() && media_data(block).is_none() {
        return format!("[attachment '{label}' is no longer available]");
    }
    let k = kind.as_str();
    format!("[{k} attachment '{label}' was not sent: this model cannot take {k} input]")
}

// ---------------------------------------------------------------------------
// transport
// ---------------------------------------------------------------------------

/// How a backend authenticates. Called on every attempt, so a signature or
/// token is always fresh.
#[async_trait]
pub trait Auth: Send + Sync {
    /// Headers to add to one attempt.
    async fn headers(&self, method: &Method, url: &str, body: &[u8]) -> Result<HeaderMap>;

    /// Drop a cached token after a 401. True if a retry could now succeed.
    async fn refresh(&self) -> bool {
        false
    }
}

/// Fixed headers — an API key in a header, say.
#[derive(Debug, Clone, Default)]
pub struct StaticAuth(pub HeaderMap);

impl StaticAuth {
    pub fn header(name: &str, value: &str) -> Self {
        let mut h = HeaderMap::new();
        if let (Ok(n), Ok(v)) = (HeaderName::from_bytes(name.as_bytes()), HeaderValue::from_str(value)) {
            h.insert(n, v);
        }
        Self(h)
    }

    pub fn bearer(token: &str) -> Self {
        if token.is_empty() { Self::default() } else { Self::header("authorization", &format!("Bearer {token}")) }
    }
}

#[async_trait]
impl Auth for StaticAuth {
    async fn headers(&self, _: &Method, _: &str, _: &[u8]) -> Result<HeaderMap> {
        Ok(self.0.clone())
    }
}

/// Called before every retry.
pub type OnRetry = Arc<dyn Fn(&RetryEvent) + Send + Sync>;

/// Connection settings every HTTP provider takes.
#[derive(Clone)]
pub struct TransportConfig {
    pub base_url: String,
    /// Seconds to wait for a response (per read).
    pub timeout: f64,
    pub connect_timeout: f64,
    pub retry: RetryPolicy,
    pub max_concurrency: Option<usize>,
    pub circuit_breaker: Option<CircuitBreaker>,
    pub on_retry: Option<OnRetry>,
    pub headers: HeaderMap,
    pub client: Option<reqwest::Client>,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            timeout: 600.0,
            connect_timeout: 10.0,
            retry: RetryPolicy::default(),
            max_concurrency: None,
            circuit_breaker: Some(CircuitBreaker::default()),
            on_retry: None,
            headers: HeaderMap::new(),
            client: None,
        }
    }
}

static TLS: Once = Once::new();

/// Install `ring` as the process's rustls crypto provider, if none is installed.
///
/// The crate's own HTTP clients do this themselves. Call it before building a
/// `reqwest::Client` of your own in a program that uses this crate, so that
/// client has a TLS backend too. Safe to call any number of times.
pub fn install_tls() {
    TLS.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

pub(crate) fn default_client(timeout: f64, connect_timeout: f64) -> reqwest::Client {
    install_tls();
    reqwest::Client::builder()
        .read_timeout(Duration::from_secs_f64(timeout))
        .connect_timeout(Duration::from_secs_f64(connect_timeout))
        .pool_max_idle_per_host(32)
        .build()
        .expect("reqwest client")
}

/// A streamed HTTP body, holding its concurrency slot until dropped.
pub struct StreamedResponse {
    pub response: Response,
    _permit: Option<OwnedSemaphorePermit>,
}

/// The one place a request goes out. Retries, classifies, and keeps score.
///
/// - retries with exponential backoff and jitter on 408/409/425/429/5xx/529,
///   timeouts and dropped connections, honouring every Retry-After variant;
/// - typed errors that carry the vendor's request id;
/// - a shared cool-down, so calls running alongside a throttled one wait too;
/// - a circuit breaker, an optional concurrency cap, a token refresh on 401.
pub struct HttpTransport {
    pub provider: String,
    pub base_url: String,
    pub retry: RetryPolicy,
    pub circuit: Option<CircuitBreaker>,
    pub on_retry: Option<OnRetry>,
    pub extra_headers: HeaderMap,
    pub timeout: f64,
    pub connect_timeout: f64,
    client: reqwest::Client,
    auth: Arc<dyn Auth>,
    slots: Option<Arc<Semaphore>>,
    stats: Mutex<ProviderStats>,
    cooldown_until: Mutex<Option<Instant>>,
}

impl std::fmt::Debug for HttpTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpTransport").field("provider", &self.provider).field("base_url", &self.base_url).finish_non_exhaustive()
    }
}

enum Sent {
    Json(Value),
    Stream(StreamedResponse),
}

impl HttpTransport {
    pub fn new(provider: &str, config: TransportConfig, auth: Arc<dyn Auth>) -> Self {
        let client = config.client.clone().unwrap_or_else(|| default_client(config.timeout, config.connect_timeout));
        Self {
            provider: provider.into(),
            base_url: config.base_url.trim_end_matches('/').to_string(),
            retry: config.retry,
            circuit: config.circuit_breaker,
            on_retry: config.on_retry,
            extra_headers: config.headers,
            timeout: config.timeout,
            connect_timeout: config.connect_timeout,
            client,
            auth,
            slots: config.max_concurrency.filter(|n| *n > 0).map(|n| Arc::new(Semaphore::new(n))),
            stats: Mutex::new(ProviderStats::default()),
            cooldown_until: Mutex::new(None),
        }
    }

    pub fn stats(&self) -> ProviderStats {
        self.stats.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    fn with_stats<R>(&self, f: impl FnOnce(&mut ProviderStats) -> R) -> R {
        f(&mut self.stats.lock().unwrap_or_else(|p| p.into_inner()))
    }

    pub fn url(&self, path: &str) -> String {
        if path.starts_with("http://") || path.starts_with("https://") { path.into() } else { format!("{}{path}", self.base_url) }
    }

    /// POST JSON, get JSON — with every retry and safety rail applied.
    pub async fn post(&self, path: &str, payload: &Value, params: &[(String, String)], timeout: Option<f64>) -> Result<Value> {
        match self.request(Method::POST, &self.url(path), Some(payload), params, timeout, false).await? {
            Sent::Json(v) => Ok(v),
            Sent::Stream(_) => unreachable!("not streaming"),
        }
    }

    pub async fn get(&self, path: &str, params: &[(String, String)], timeout: Option<f64>) -> Result<Value> {
        match self.request(Method::GET, &self.url(path), None, params, timeout, false).await? {
            Sent::Json(v) => Ok(v),
            Sent::Stream(_) => unreachable!("not streaming"),
        }
    }

    /// POST and hand back the open response. Opening is retried like any call.
    pub async fn open_stream(&self, path: &str, payload: &Value, params: &[(String, String)], timeout: Option<f64>) -> Result<StreamedResponse> {
        match self.request(Method::POST, &self.url(path), Some(payload), params, timeout, true).await? {
            Sent::Stream(s) => Ok(s),
            Sent::Json(_) => unreachable!("streaming"),
        }
    }

    /// POST and yield the response line by line (SSE). A failure once bytes
    /// are flowing is returned as an error in the stream.
    pub fn stream_lines<'a>(
        &'a self,
        path: &'a str,
        payload: &'a Value,
        params: &'a [(String, String)],
        timeout: Option<f64>,
    ) -> BoxStream<'a, Result<String>> {
        let name = self.provider.clone();
        Box::pin(async_stream::try_stream! {
            let opened = self.open_stream(path, payload, params, timeout).await?;
            let StreamedResponse { response, _permit } = opened;
            let mut bytes = response.bytes_stream();
            let mut buf: Vec<u8> = Vec::new();
            while let Some(chunk) = bytes.next().await {
                let chunk: Bytes = chunk.map_err(|e| stream_error(&name, &e))?;
                buf.extend_from_slice(&chunk);
                while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                    let line: Vec<u8> = buf.drain(..=pos).collect();
                    let text = String::from_utf8_lossy(&line[..line.len() - 1]);
                    yield text.trim_end_matches('\r').to_string();
                }
            }
            if !buf.is_empty() {
                yield String::from_utf8_lossy(&buf).trim_end_matches('\r').to_string();
            }
            drop(_permit);
        })
    }

    async fn acquire(&self) -> Option<OwnedSemaphorePermit> {
        match &self.slots {
            Some(s) => s.clone().acquire_owned().await.ok(),
            None => None,
        }
    }

    async fn wait_for_cooldown(&self, resume_at: Option<Instant>) {
        let until = *self.cooldown_until.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(until) = until {
            let from = resume_at.map_or_else(Instant::now, |r| r.max(Instant::now()));
            if until > from {
                tokio::time::sleep(until - from).await;
            }
        }
    }

    fn settle(&self, ok: bool, healthy: Option<bool>) {
        self.with_stats(|s| if ok { s.succeeded += 1 } else { s.failed += 1 });
        if let Some(c) = &self.circuit {
            if healthy.unwrap_or(ok) {
                c.record_success();
            } else {
                c.record_failure();
            }
        }
    }

    /// Seconds before the next attempt, or `None` to give up now.
    pub fn next_delay(&self, error: &ProviderError, attempt: u32, elapsed: f64) -> Option<f64> {
        let policy = &self.retry;
        if !error.retryable || attempt > policy.max_retries {
            return None;
        }
        if error.kind == K::Timeout && !policy.retry_on_timeout {
            return None;
        }
        if error.kind == K::Connection && !policy.retry_on_connection_error {
            return None;
        }
        let delay = match error.retry_after {
            Some(after) if after > policy.max_retry_after => return None,
            Some(after) => after,
            None => policy.backoff(attempt - 1),
        };
        if policy.max_elapsed.is_some_and(|max| elapsed + delay > max) {
            return None;
        }
        Some(delay)
    }

    /// Count the retry, log it, tell the observer.
    pub fn before_retry(&self, event: &RetryEvent) {
        self.with_stats(|s| {
            s.retries += 1;
            s.total_backoff_s += event.delay;
        });
        tracing::warn!("{}: {} — retry {}/{} in {:.2}s", event.provider, event.reason(), event.attempt, self.retry.max_retries, event.delay);
        if let Some(cb) = &self.on_retry {
            // A broken observer must not break the call.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cb(event)));
        }
    }

    async fn request(
        &self,
        method: Method,
        url: &str,
        payload: Option<&Value>,
        params: &[(String, String)],
        timeout: Option<f64>,
        stream: bool,
    ) -> Result<Sent> {
        self.with_stats(|s| s.calls += 1);
        if let Some(c) = &self.circuit
            && !c.allow()
        {
            self.with_stats(|s| {
                s.circuit_rejections += 1;
                s.failed += 1;
            });
            let wait = c.retry_in();
            return Err(ProviderError::new(
                K::Unavailable,
                format!("{} circuit is open after {} consecutive failures — not calling it for another {wait:.0}s", self.provider, c.failures()),
            )
            .provider(&self.provider)
            .retry_after(Some(wait))
            .into());
        }
        // Frees the half-open probe slot if this future is dropped mid-call.
        struct ProbeGuard<'a>(Option<&'a CircuitBreaker>);
        impl Drop for ProbeGuard<'_> {
            fn drop(&mut self) {
                if let Some(c) = self.0 {
                    c.release_probe();
                }
            }
        }
        let mut guard = ProbeGuard(self.circuit.as_ref());

        let body: Vec<u8> = payload.map(|p| serde_json::to_vec(p).unwrap_or_default()).unwrap_or_default();
        let mut base = HeaderMap::new();
        base.insert("accept", HeaderValue::from_static(if stream { "text/event-stream" } else { "application/json" }));
        base.insert("user-agent", HeaderValue::from_str(&format!("agent-harness/{}", crate::VERSION)).expect("ascii"));
        if payload.is_some() {
            base.insert("content-type", HeaderValue::from_static("application/json"));
        }
        base.extend(self.extra_headers.clone());

        let started = Instant::now();
        let mut waited = 0.0;
        let mut resume_at: Option<Instant> = None;
        let mut attempt = 0u32;
        let mut refreshed = false;
        loop {
            self.wait_for_cooldown(resume_at).await;
            let permit = self.acquire().await;
            let mut headers = base.clone();
            headers.extend(self.auth.headers(&method, url, &body).await?);
            self.with_stats(|s| s.requests += 1);
            let mut builder = self.client.request(method.clone(), url).headers(headers);
            if !params.is_empty() {
                builder = builder.query(params);
            }
            if !body.is_empty() {
                builder = builder.body(body.clone());
            }
            if let (Some(t), false) = (timeout, stream) {
                builder = builder.timeout(Duration::from_secs_f64(t));
            }
            let error: ProviderError = match builder.send().await {
                Err(e) if e.is_timeout() => {
                    self.with_stats(|s| s.timeouts += 1);
                    ProviderError::new(K::Timeout, format!("{} timed out after {:.1}s ({e})", self.provider, started.elapsed().as_secs_f64()))
                        .provider(&self.provider)
                }
                Err(e) => {
                    let fatal = e.is_builder();
                    ProviderError::new(K::Connection, format!("{} request failed: {e}", self.provider)).provider(&self.provider).retryable(!fatal)
                }
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let rid = request_id(Some(resp.headers()));
                    self.with_stats(|s| {
                        *s.by_status.entry(status).or_insert(0) += 1;
                        if let Some(r) = &rid {
                            s.last_request_id = r.clone();
                        }
                    });
                    if status < 300 {
                        if stream {
                            self.settle(true, None);
                            guard.0 = None;
                            return Ok(Sent::Stream(StreamedResponse { response: resp, _permit: permit }));
                        }
                        let text = resp.text().await.unwrap_or_default();
                        match serde_json::from_str::<Value>(&text) {
                            Ok(data) => {
                                self.settle(true, None);
                                guard.0 = None;
                                return Ok(Sent::Json(data));
                            }
                            Err(_) => {
                                let mut e = ProviderError::new(
                                    K::Unavailable,
                                    format!("{} sent a {status} that is not JSON: {:?}", self.provider, text.chars().take(200).collect::<String>()),
                                )
                                .provider(&self.provider)
                                .status(status);
                                e.request_id = rid;
                                e
                            }
                        }
                    } else {
                        let headers = resp.headers().clone();
                        let raw = resp.text().await.unwrap_or_default();
                        classify(&self.provider, status, Some(&headers), Some(&raw), Some(&self.retry))
                    }
                }
            };
            drop(permit);
            let mut error = error;
            if error.kind.is_rate_limit() {
                self.with_stats(|s| s.rate_limited += 1);
            }
            if error.kind == K::Authentication && !refreshed && self.auth.refresh().await {
                refreshed = true; // a stale token, not a retry
                continue;
            }
            attempt += 1;
            error.attempts = attempt;
            let elapsed = started.elapsed().as_secs_f64().max(waited);
            let Some(delay) = self.next_delay(&error, attempt, elapsed) else {
                self.with_stats(|s| s.last_error = format!("{}: {error}", error.kind.name()));
                error.is_final = true;
                // A 4xx still proves the provider is up; only the request was bad.
                self.settle(false, Some(!error.retryable && error.status.is_some()));
                guard.0 = None;
                return Err(error.into());
            };
            if error.kind.is_rate_limit() && error.retry_after.is_some() {
                let until = Instant::now() + Duration::from_secs_f64(delay);
                let mut cd = self.cooldown_until.lock().unwrap_or_else(|p| p.into_inner());
                *cd = Some(cd.map_or(until, |c| c.max(until)));
            }
            self.before_retry(&RetryEvent { provider: self.provider.clone(), attempt, delay, error, url: url.into() });
            resume_at = Some(Instant::now() + Duration::from_secs_f64(delay));
            waited += delay;
            tokio::time::sleep(Duration::from_secs_f64(delay)).await;
        }
    }

    /// Retry a native stream while it is still safe to — until the first event
    /// has been handed on. After that a failure is returned, because starting
    /// over would repeat text the caller has already shown.
    pub fn retrying<'a, F>(&'a self, make: F) -> BoxStream<'a, Result<StreamEvent>>
    where
        F: Fn() -> BoxStream<'a, Result<StreamEvent>> + Send + 'a,
    {
        Box::pin(async_stream::try_stream! {
            let started = Instant::now();
            let mut waited = 0.0;
            let mut attempt = 0u32;
            loop {
                let mut emitted = false;
                let mut inner = make();
                let mut failure: Option<ProviderError> = None;
                while let Some(item) = inner.next().await {
                    match item {
                        Ok(event) => {
                            emitted = true;
                            yield event;
                        }
                        Err(Error::Provider(p)) if !emitted && !p.is_final => {
                            failure = Some(*p);
                            break;
                        }
                        Err(e) => Err(e)?,
                    }
                }
                let Some(mut err) = failure else { break };
                attempt += 1;
                err.attempts = attempt;
                let elapsed = started.elapsed().as_secs_f64().max(waited);
                let Some(delay) = self.next_delay(&err, attempt, elapsed) else { Err(Error::from(err))?; unreachable!() };
                self.before_retry(&RetryEvent { provider: self.provider.clone(), attempt, delay, error: err, url: String::new() });
                waited += delay;
                tokio::time::sleep(Duration::from_secs_f64(delay)).await;
            }
        })
    }
}

fn stream_error(provider: &str, e: &reqwest::Error) -> Error {
    let kind = if e.is_timeout() { K::Timeout } else { K::Connection };
    let what = if e.is_timeout() { "stalled" } else { "dropped" };
    ProviderError::new(kind, format!("{provider} stream {what}: {e}")).provider(provider).into()
}

/// Server-sent events as `(event, data)`, joining multi-line `data:` fields.
pub fn sse_events<'a>(mut lines: BoxStream<'a, Result<String>>) -> BoxStream<'a, Result<(String, String)>> {
    Box::pin(async_stream::try_stream! {
        let mut event = String::new();
        let mut data: Vec<String> = Vec::new();
        while let Some(line) = lines.next().await {
            let line = line?;
            if line.is_empty() {
                if !data.is_empty() {
                    yield (std::mem::take(&mut event), data.join("\n"));
                    data.clear();
                }
                event.clear();
                continue;
            }
            if line.starts_with(':') {
                continue;
            }
            let (name, value) = line.split_once(':').unwrap_or((line.as_str(), ""));
            let value = value.strip_prefix(' ').unwrap_or(value);
            match name {
                "event" => event = value.to_string(),
                "data" => data.push(value.to_string()),
                _ => {}
            }
        }
        if !data.is_empty() {
            yield (event, data.join("\n"));
        }
    })
}

/// An API key from the argument, else the environment.
pub fn key_from_env(api_key: Option<String>, env_key: &str) -> String {
    api_key.filter(|k| !k.is_empty()).or_else(|| if env_key.is_empty() { None } else { std::env::var(env_key).ok() }).unwrap_or_default()
}

/// The "no API key" error, with a hint where to put it.
pub fn missing_key(provider: &str, env_key: &str) -> Error {
    let hint = if env_key.is_empty() { "Pass".to_string() } else { format!("Set ${env_key} or pass") };
    ProviderError::new(K::Authentication, format!("No API key for {provider}. {hint} api_key=... (see describe_llm_provider('{provider}'))"))
        .provider(provider)
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[test]
    fn model_lookup_and_cost() {
        assert_eq!(model_info("claude-opus-5").unwrap().provider, "anthropic");
        assert_eq!(model_info("claude-opus-5-20260401").unwrap().id, "claude-opus-5");
        assert_eq!(model_info("us.anthropic.claude-opus-5").unwrap().id, "claude-opus-5");
        assert_eq!(normalise_model("claude-opus-5@20260401"), "claude-opus-5");
        assert!(model_info("totally-unknown").is_none());
        let usage = Usage { input_tokens: 1_000_000, output_tokens: 100_000, ..Default::default() };
        assert_eq!(estimate_cost("claude-opus-5", &usage), 7.5);
        assert_eq!(estimate_cost("mystery", &usage), 0.0);
        register_model(ModelInfo { input_cost: 1.0, ..ModelInfo::new("my-model", "openai") });
        assert_eq!(provider_for_model("my-model"), "openai");
        assert_eq!(provider_for_model("grok-4"), "xai");
        assert_eq!(provider_for_model("o4-mini"), "openai");
        assert_eq!(provider_for_model("whatever"), "anthropic");
    }

    #[test]
    fn request_generation_values() {
        let mut req = CompletionRequest::new("m", vec![]);
        req.temperature = Some(0.5);
        req.effort = Some(Effort::High);
        let v = req.generation_values();
        assert_eq!(v["temperature"], json!(0.5));
        assert_eq!(v["effort"], json!("high"));
        assert_eq!(v["max_tokens"], json!(8192));
        let plan = ParameterPlan::new("p", "m", v);
        assert!(!plan.contains("seed"));
    }

    #[tokio::test]
    async fn sse_parsing() {
        let lines: Vec<Result<String>> = [": comment", "event: message_start", "data: {\"a\":1}", "", "data: line1", "data: line2", "", "data: tail"]
            .into_iter()
            .map(|s| Ok(s.to_string()))
            .collect();
        let events: Vec<(String, String)> = sse_events(Box::pin(stream::iter(lines))).map(|e| e.unwrap()).collect().await;
        assert_eq!(events, vec![("message_start".into(), "{\"a\":1}".into()), (String::new(), "line1\nline2".into()), (String::new(), "tail".into()),]);
    }

    #[test]
    fn media_notes() {
        let gone = MediaBlock::from_path("/definitely/not/here.png", None, MediaKind::Image);
        assert_eq!(media_note(MediaKind::Image, &gone), "[attachment 'here.png' is no longer available]");
        let url = MediaBlock::from_url("https://x/a.mp3", None, MediaKind::Audio);
        assert_eq!(media_note(MediaKind::Audio, &url), "[audio attachment 'a.mp3' was not sent: this model cannot take audio input]");
    }

    #[test]
    fn next_delay_rules() {
        let t = HttpTransport::new(
            "p",
            TransportConfig { retry: RetryPolicy { jitter: 0.0, max_retries: 2, ..Default::default() }, ..Default::default() },
            Arc::new(StaticAuth::default()),
        );
        let rl = ProviderError::new(K::RateLimit, "x");
        assert_eq!(t.next_delay(&rl, 1, 0.0), Some(0.5));
        assert_eq!(t.next_delay(&rl, 3, 0.0), None);
        assert_eq!(t.next_delay(&rl.clone().retry_after(Some(5.0)), 1, 0.0), Some(5.0));
        assert_eq!(t.next_delay(&rl.clone().retry_after(Some(500.0)), 1, 0.0), None);
        assert_eq!(t.next_delay(&ProviderError::new(K::InvalidRequest, "x"), 1, 0.0), None);
    }
}
