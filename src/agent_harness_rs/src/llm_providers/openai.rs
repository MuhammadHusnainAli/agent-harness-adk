//! OpenAI Chat Completions adapter.
//!
//! Also the engine for everything that speaks the same wire format — the
//! presets in [`super::compatible`] (Groq, Together, OpenRouter, DeepSeek,
//! Mistral, xAI, Ollama, vLLM, ...). Python subclasses this adapter; in Rust a
//! preset is an [`OpenAIFlavor`]: a table of the class attributes it changes.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::Method;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Map, Value, json};

use super::base::{
    Auth, CompletionRequest, HttpTransport, Provider, ProviderField, ProviderSpec, TransportConfig, key_from_env, media_data, media_note, missing_key,
    sse_events,
};
use super::parameters::{Effort, ParameterPlan, SAMPLING_PARAMETERS, nearest_effort};
use super::resilience::classify;
use crate::errors::{Error, ProviderError, ProviderErrorKind, Result};
use crate::types::{
    ContentBlock, MediaBlock, MediaKind, Message, ModelResponse, Role, StopReason, StreamEvent, StreamEventType, TextBlock, ThinkingBlock, ToolUseBlock, Usage,
};

fn stop_map(s: &str) -> StopReason {
    match s {
        "length" => StopReason::MaxTokens,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        "content_filter" => StopReason::Error,
        _ => StopReason::EndTurn,
    }
}

/// How `effort` reaches the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffortStyle {
    /// `reasoning_effort: "high"` — OpenAI's own spelling.
    ReasoningEffort,
    /// `reasoning: {effort | max_tokens | enabled}` — OpenRouter's.
    OpenRouter,
    /// Never sent.
    Never,
}

/// Everything a preset changes about the OpenAI adapter.
#[derive(Debug, Clone)]
pub struct OpenAIFlavor {
    pub spec: ProviderSpec,
    /// False for local servers (Ollama, vLLM) that run without a key.
    pub key_required: bool,
    /// Ask for token usage at the end of a stream. Some servers reject it.
    pub stream_usage: bool,
    /// Models that take `max_completion_tokens` and reject sampling.
    pub reasoning_prefixes: &'static [&'static str],
    pub effort_style: EffortStyle,
    /// Model-family prefix → the reasoning levels it takes.
    pub effort_families: &'static [(&'static str, &'static [Effort])],
    /// Payload keys this server does not accept.
    pub drop_params: &'static [&'static str],
    /// Payload keys this server spells differently: `("seed", "random_seed")`.
    pub rename_params: &'static [(&'static str, &'static str)],
    pub embedding_model: &'static str,
    /// An environment variable that overrides the base URL.
    pub base_url_env: &'static str,
    /// Models that additionally reject penalties and stop sequences (xAI's Grok).
    pub strict_reasoning_models: &'static [&'static str],
}

use Effort::{High, Low, Medium, Minimal};

const OPENAI_FIELDS: [ProviderField; 4] = [
    ProviderField {
        required: true,
        env: &["OPENAI_API_KEY"],
        example: Some("sk-..."),
        ..ProviderField::new("api_key", "secret", "An OpenAI API key from platform.openai.com.")
    },
    ProviderField::new("base_url", "url", "Any OpenAI-compatible endpoint."),
    ProviderField { env: &["OPENAI_ORG_ID"], ..ProviderField::new("organization", "str", "Bill a specific organisation.") },
    ProviderField { env: &["OPENAI_PROJECT_ID"], ..ProviderField::new("project", "str", "Bill a specific project.") },
];

pub const OPENAI: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "openai",
        display_name: "OpenAI",
        description: "GPT and o-series models through the Chat Completions API.",
        docs_url: "https://platform.openai.com/docs/api-reference/chat",
        auth_type: "api_key",
        env_key: "OPENAI_API_KEY",
        default_model: "gpt-4.1",
        base_url: "https://api.openai.com/v1",
        capabilities: &["streaming", "tools", "vision", "thinking", "json_schema", "embeddings", "list_models"],
        parameters: &["temperature", "top_p", "frequency_penalty", "presence_penalty", "seed"],
        modalities: &[MediaKind::Image, MediaKind::Document],
        fields: &OPENAI_FIELDS,
    },
    key_required: true,
    stream_usage: true,
    reasoning_prefixes: &["o1", "o3", "o4", "gpt-5"],
    effort_style: EffortStyle::ReasoningEffort,
    effort_families: &[("gpt-5", &[Minimal, Low, Medium, High]), ("o1", &[Low, Medium, High]), ("o3", &[Low, Medium, High]), ("o4", &[Low, Medium, High])],
    drop_params: &[],
    rename_params: &[],
    embedding_model: "text-embedding-3-small",
    base_url_env: "",
    strict_reasoning_models: &[],
};

struct OpenAIAuth {
    provider: &'static str,
    env_key: &'static str,
    key: String,
    key_required: bool,
    organization: String,
    project: String,
}

#[async_trait]
impl Auth for OpenAIAuth {
    async fn headers(&self, _: &Method, _: &str, _: &[u8]) -> Result<HeaderMap> {
        let mut h = HeaderMap::new();
        if self.key.is_empty() {
            if self.key_required {
                return Err(missing_key(self.provider, self.env_key));
            }
            return Ok(h);
        }
        let v = |s: &str| HeaderValue::from_str(s).map_err(|e| Error::config(e.to_string()));
        h.insert("authorization", v(&format!("Bearer {}", self.key))?);
        if !self.organization.is_empty() {
            h.insert("openai-organization", v(&self.organization)?);
        }
        if !self.project.is_empty() {
            h.insert("openai-project", v(&self.project)?);
        }
        Ok(h)
    }
}

/// GPT, o-series, and every OpenAI-compatible server.
pub struct OpenAIProvider {
    pub flavor: OpenAIFlavor,
    pub transport: HttpTransport,
}

const AUDIO_FORMATS: [(&str, &str); 5] = [("audio/wav", "wav"), ("audio/x-wav", "wav"), ("audio/wave", "wav"), ("audio/mpeg", "mp3"), ("audio/mp3", "mp3")];

fn audio_format(media_type: &str) -> Option<&'static str> {
    AUDIO_FORMATS.iter().find(|(m, _)| *m == media_type).map(|(_, f)| *f)
}

impl OpenAIProvider {
    /// OpenAI itself. Reads `OPENAI_API_KEY`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`.
    pub fn new(api_key: Option<String>) -> Self {
        let org = std::env::var("OPENAI_ORG_ID").unwrap_or_default();
        let project = std::env::var("OPENAI_PROJECT_ID").unwrap_or_default();
        Self::build(OPENAI, api_key, TransportConfig::default(), org, project)
    }

    /// Any flavor; `config.base_url` (if set) beats the flavor's env var and default.
    pub fn with_flavor(flavor: OpenAIFlavor, api_key: Option<String>, config: TransportConfig) -> Result<Self> {
        let mut config = config;
        if config.base_url.is_empty() {
            let from_env = if flavor.base_url_env.is_empty() { None } else { std::env::var(flavor.base_url_env).ok().filter(|s| !s.is_empty()) };
            config.base_url = from_env.unwrap_or_else(|| flavor.spec.base_url.to_string());
        }
        if config.base_url.is_empty() {
            let env = if flavor.base_url_env.is_empty() { String::new() } else { format!(" or set {}", flavor.base_url_env) };
            return Err(ProviderError::new(ProviderErrorKind::Other, format!("{} needs base_url=...{env}", flavor.spec.display_name))
                .provider(flavor.spec.name)
                .into());
        }
        Ok(Self::build(flavor, api_key, config, String::new(), String::new()))
    }

    fn build(flavor: OpenAIFlavor, api_key: Option<String>, mut config: TransportConfig, organization: String, project: String) -> Self {
        if config.base_url.is_empty() {
            config.base_url = flavor.spec.base_url.into();
        }
        let auth = OpenAIAuth {
            provider: flavor.spec.name,
            env_key: flavor.spec.env_key,
            key: key_from_env(api_key, flavor.spec.env_key),
            key_required: flavor.key_required,
            organization,
            project,
        };
        let transport = HttpTransport::new(flavor.spec.name, config, Arc::new(auth));
        Self { flavor, transport }
    }

    fn is_reasoning(&self, model: &str) -> bool {
        self.flavor.reasoning_prefixes.iter().any(|p| model.starts_with(p))
    }

    /// The reasoning levels this model takes, or `None` if it takes none.
    /// Matched against the id and its last path segment.
    pub fn effort_levels(&self, model: &str) -> Option<&'static [Effort]> {
        let tail = model.rsplit('/').next().unwrap_or(model);
        self.flavor
            .effort_families
            .iter()
            .filter(|(prefix, _)| model.starts_with(prefix) || tail.starts_with(prefix))
            .max_by_key(|(prefix, _)| prefix.len())
            .map(|(_, levels)| *levels)
    }

    // ---- encoding -----------------------------------------------------
    fn encode_media(&self, kind: MediaKind, block: &MediaBlock, model: &str) -> Value {
        let note = || json!({"type": "text", "text": media_note(kind, block)});
        if !self.accepts(kind, block, model) {
            return note();
        }
        if kind == MediaKind::Image
            && let (Some(url), false) = (&block.url, block.inline())
        {
            return json!({"type": "image_url", "image_url": {"url": url}});
        }
        let Some(data) = block.inline().then(|| media_data(block)).flatten() else { return note() };
        match kind {
            MediaKind::Image => json!({"type": "image_url", "image_url": {"url": format!("data:{};base64,{data}", block.media_type)}}),
            MediaKind::Audio => match audio_format(&block.media_type) {
                Some(fmt) => json!({"type": "input_audio", "input_audio": {"data": data, "format": fmt}}),
                None => note(),
            },
            MediaKind::Document if block.media_type == "application/pdf" => {
                let name = if block.name.is_empty() { "document.pdf" } else { &block.name };
                json!({"type": "file", "file": {"filename": name, "file_data": format!("data:application/pdf;base64,{data}")}})
            }
            _ => note(),
        }
    }

    pub fn encode_messages(&self, req: &CompletionRequest) -> Vec<Value> {
        let mut out = Vec::new();
        if let Some(system) = &req.system {
            out.push(json!({"role": "system", "content": system}));
        }
        for msg in &req.messages {
            if msg.role == Role::System {
                out.push(json!({"role": "system", "content": msg.text()}));
                continue;
            }
            // Tool results become their own `tool` messages, one per result.
            for b in &msg.content {
                if let ContentBlock::ToolResult(r) = b {
                    let content = if r.content.is_empty() && r.is_error { "error".to_string() } else { r.content.clone() };
                    out.push(json!({"role": "tool", "tool_call_id": r.tool_use_id, "content": content}));
                }
            }
            let mut parts: Vec<Value> = Vec::new();
            let mut text_chunks: Vec<String> = Vec::new();
            let mut tool_calls: Vec<Value> = Vec::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text(t) => {
                        text_chunks.push(t.text.clone());
                        parts.push(json!({"type": "text", "text": t.text}));
                    }
                    // Provider-specific; never replayed across vendors.
                    ContentBlock::Thinking(_) | ContentBlock::ToolResult(_) => {}
                    ContentBlock::ToolUse(u) => tool_calls.push(json!({
                        "id": u.id, "type": "function",
                        "function": {"name": u.name, "arguments": Value::Object(u.input.clone()).to_string()},
                    })),
                    other => {
                        let (kind, m) = other.as_media().expect("media");
                        let part = self.encode_media(kind, m, &req.model);
                        if part["type"] == "text" {
                            text_chunks.push(part["text"].as_str().unwrap_or("").to_string());
                        }
                        parts.push(part);
                    }
                }
            }
            if msg.role == Role::Assistant {
                if text_chunks.is_empty() && tool_calls.is_empty() {
                    continue;
                }
                let content = if text_chunks.is_empty() { Value::Null } else { json!(text_chunks.join("\n")) };
                let mut entry = json!({"role": "assistant", "content": content});
                if !tool_calls.is_empty() {
                    entry["tool_calls"] = json!(tool_calls);
                }
                out.push(entry);
            } else {
                if parts.is_empty() {
                    continue;
                }
                let has_media = parts.iter().any(|p| p["type"] != "text");
                let content = if has_media { json!(parts) } else { json!(text_chunks.join("\n")) };
                out.push(json!({"role": "user", "content": content}));
            }
        }
        out
    }

    fn reasoning_payload(&self, plan: &ParameterPlan, payload: &mut Map<String, Value>) {
        let effort = plan.get("effort").and_then(Value::as_str);
        match self.flavor.effort_style {
            EffortStyle::OpenRouter => {
                let mut reasoning = Map::new();
                match effort {
                    Some("none") => {
                        reasoning.insert("enabled".into(), json!(false));
                    }
                    Some(e) => {
                        reasoning.insert("effort".into(), json!(e));
                    }
                    None => {}
                }
                if let (Some(b), false) = (plan.get("thinking_budget"), effort == Some("none")) {
                    reasoning.insert("max_tokens".into(), b.clone());
                }
                if !reasoning.is_empty() {
                    payload.insert("reasoning".into(), Value::Object(reasoning));
                }
            }
            EffortStyle::ReasoningEffort => {
                if let Some(e) = effort {
                    payload.insert("reasoning_effort".into(), json!(e));
                }
            }
            EffortStyle::Never => {}
        }
    }

    pub fn build_payload(&self, req: &CompletionRequest, stream: bool) -> Value {
        let mut p = Map::new();
        p.insert("model".into(), json!(req.model));
        p.insert("messages".into(), json!(self.encode_messages(req)));
        let plan = self.plan_parameters(req);
        let tokens_key = if self.is_reasoning(&req.model) { "max_completion_tokens" } else { "max_tokens" };
        p.insert(tokens_key.into(), plan.get("max_tokens").cloned().unwrap_or(json!(req.max_tokens)));
        if !req.tools.is_empty() {
            let tools: Vec<Value> = req
                .tools
                .iter()
                .map(|t| json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}))
                .collect();
            p.insert("tools".into(), json!(tools));
            let choice = match &req.tool_choice {
                Some(Value::String(s)) if s == "any" => Some(json!("required")),
                Some(Value::String(s)) if s == "auto" || s == "none" => Some(json!(s)),
                Some(Value::Object(o)) => o.get("name").and_then(Value::as_str).map(|n| json!({"type": "function", "function": {"name": n}})),
                _ => None,
            };
            if let Some(c) = choice {
                p.insert("tool_choice".into(), c);
            }
        }
        for name in SAMPLING_PARAMETERS {
            if let Some(v) = plan.get(name) {
                p.insert(name.into(), v.clone());
            }
        }
        self.reasoning_payload(&plan, &mut p);
        if let (Some(par), false) = (req.parallel_tool_calls, req.tools.is_empty()) {
            p.insert("parallel_tool_calls".into(), json!(par));
        }
        if let Some(u) = &req.user {
            p.insert("user".into(), json!(u));
        }
        if let Some(stop) = plan.get("stop") {
            p.insert("stop".into(), stop.clone());
        }
        if let Some(schema) = &req.response_schema {
            p.insert("response_format".into(), json!({"type": "json_schema", "json_schema": {"name": "response", "schema": schema, "strict": false}}));
        }
        if stream {
            p.insert("stream".into(), json!(true));
            if self.flavor.stream_usage {
                p.insert("stream_options".into(), json!({"include_usage": true}));
            }
        }
        for key in self.flavor.drop_params {
            p.remove(*key);
        }
        for (old, new) in self.flavor.rename_params {
            if let Some(v) = p.remove(*old) {
                p.insert((*new).into(), v);
            }
        }
        p.extend(req.extra.clone());
        Value::Object(p)
    }

    // ---- decoding -----------------------------------------------------
    pub fn usage(raw: &Value) -> Usage {
        let u = &raw["usage"];
        let cached = u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
        Usage {
            input_tokens: u["prompt_tokens"].as_u64().unwrap_or(0).saturating_sub(cached),
            output_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
            cache_read_tokens: cached,
            calls: 1,
            ..Default::default()
        }
    }

    pub fn decode(&self, raw: Value, req: &CompletionRequest, latency_ms: f64) -> Result<ModelResponse> {
        if !raw["error"].is_null() {
            return Err(self.body_error(&raw["error"]));
        }
        let Some(choice) = raw["choices"].as_array().and_then(|c| c.first()) else {
            let mut e = ProviderError::new(ProviderErrorKind::Other, format!("{} returned no choices", self.name())).provider(self.name());
            e.body = Some(raw.to_string().chars().take(2000).collect());
            return Err(e.into());
        };
        let msg = &choice["message"];
        let mut blocks = Vec::new();
        // DeepSeek, vLLM and OpenRouter return the reasoning alongside the answer.
        if let Some(r) = msg["reasoning_content"].as_str().or(msg["reasoning"].as_str()).filter(|r| !r.is_empty()) {
            blocks.push(ContentBlock::Thinking(ThinkingBlock { thinking: r.into(), signature: None, redacted: None }));
        }
        let content = msg["content"].as_str().filter(|c| !c.is_empty());
        let refusal = msg["refusal"].as_str().filter(|c| !c.is_empty());
        if let Some(c) = content {
            blocks.push(ContentBlock::text(c));
        } else if let Some(r) = refusal {
            blocks.push(ContentBlock::text(format!("[refused] {r}")));
        }
        for call in msg["tool_calls"].as_array().into_iter().flatten() {
            let f = &call["function"];
            blocks.push(ContentBlock::ToolUse(tool_use(call["id"].as_str(), f["name"].as_str().unwrap_or(""), &f["arguments"])));
        }
        let mut stop = stop_map(choice["finish_reason"].as_str().unwrap_or("stop"));
        if refusal.is_some() && content.is_none() {
            stop = StopReason::Error;
        }
        if stop == StopReason::EndTurn && blocks.iter().any(|b| matches!(b, ContentBlock::ToolUse(_))) {
            stop = StopReason::ToolUse; // some compatible servers say "stop" after a tool call
        }
        let model = raw["model"].as_str().filter(|m| !m.is_empty()).unwrap_or(&req.model).to_string();
        let usage = Self::usage(&raw);
        Ok(self.finish(Message::assistant_blocks(blocks), stop, usage, &model, raw, latency_ms))
    }

    /// An error delivered inside a 200 — mid-stream, or by a lenient proxy.
    pub fn body_error(&self, error: &Value) -> Error {
        let err = if error.is_object() { error.clone() } else { json!({"message": error.to_string()}) };
        let code = err["code"].as_str().or(err["type"].as_str()).unwrap_or("").to_string();
        let status = err["status"].as_u64().map(|s| s as u16).unwrap_or_else(|| {
            if code.contains("rate_limit") {
                429
            } else if code == "invalid_request_error" || code == "context_length_exceeded" {
                400
            } else {
                500
            }
        });
        classify(self.name(), status, None, Some(&json!({"error": err}).to_string()), Some(&self.transport.retry)).into()
    }
}

/// Some compatible servers omit the call id; mint one so the result can match.
fn tool_use(id: Option<&str>, name: &str, arguments: &Value) -> ToolUseBlock {
    let input = parse_args(arguments);
    match id.filter(|i| !i.is_empty()) {
        Some(id) => ToolUseBlock { id: id.into(), name: name.into(), input, thought_signature: None },
        None => ToolUseBlock::new(name, input),
    }
}

/// Tool arguments arrive as a JSON string — or, from some servers, already parsed.
fn parse_args(arguments: &Value) -> Map<String, Value> {
    match arguments {
        Value::Object(m) => m.clone(),
        Value::String(s) => serde_json::from_str::<Value>(if s.is_empty() { "{}" } else { s }).ok().and_then(|v| v.as_object().cloned()).unwrap_or_default(),
        _ => Map::new(),
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn spec(&self) -> &ProviderSpec {
        &self.flavor.spec
    }

    fn transport(&self) -> Option<&HttpTransport> {
        Some(&self.transport)
    }

    fn accepts(&self, kind: MediaKind, block: &MediaBlock, model: &str) -> bool {
        let modalities = self.modalities();
        match kind {
            // Only the audio models listen, and only to WAV or MP3 they are sent.
            MediaKind::Audio => {
                (modalities.contains(&MediaKind::Audio) || model.contains("audio")) && block.inline() && audio_format(&block.media_type).is_some()
            }
            _ if !modalities.contains(&kind) => false,
            MediaKind::Document => block.media_type == "application/pdf" && block.inline(),
            _ => true,
        }
    }

    fn plan(&self, req: &CompletionRequest, plan: &mut ParameterPlan) {
        let model = req.model.as_str();
        if self.is_reasoning(model) {
            for name in ["temperature", "top_p", "frequency_penalty", "presence_penalty", "seed", "top_k", "min_p", "repetition_penalty"] {
                plan.drop(name, format!("{model} is a reasoning model and rejects sampling settings"));
            }
            plan.drop("stop", format!("{model} is a reasoning model and rejects stop sequences"));
        }
        let (effort, thinking) = (plan.effort(), plan.bool("thinking"));
        if self.flavor.effort_style == EffortStyle::OpenRouter {
            const LEVELS: [Effort; 4] = [Minimal, Low, Medium, High];
            if thinking == Some(false) && effort.is_none() {
                plan.set("effort", json!("none"));
            } else if let Some(e) = effort.filter(|e| *e != Effort::None) {
                let mapped = nearest_effort(e, &LEVELS);
                if mapped != e {
                    plan.adjust("effort", json!(mapped.as_str()), "OpenRouter's levels are minimal, low, medium, high");
                }
            }
            if plan.contains("thinking_budget") && plan.effort().is_some_and(|e| e != Effort::None) {
                plan.drop("effort", "thinking_budget was given — OpenRouter takes one or the other");
            }
            plan.values.shift_remove("thinking");
        } else {
            let levels = if self.flavor.effort_style == EffortStyle::Never { None } else { self.effort_levels(model) };
            match levels {
                None => {
                    plan.drop("effort", format!("{model} has no reasoning effort setting"));
                    if thinking == Some(true) {
                        plan.drop("thinking", format!("{model} has no thinking switch"));
                    }
                    plan.values.shift_remove("thinking");
                    plan.drop("thinking_budget", format!("{model} has no thinking budget"));
                }
                Some(levels) => {
                    if effort.is_none() && thinking == Some(false) {
                        plan.set("effort", json!(levels[0].as_str()));
                        plan.adjusted.insert("effort".into(), format!("thinking=False → '{}', the least reasoning {model} allows", levels[0]));
                    } else if let Some(e) = effort {
                        let mapped = nearest_effort(e, levels);
                        if mapped != e {
                            let names: Vec<&str> = levels.iter().map(|l| l.as_str()).collect();
                            plan.adjust("effort", json!(mapped.as_str()), &format!("{model} takes {}", names.join(", ")));
                        }
                    }
                    plan.values.shift_remove("thinking"); // expressed through the effort
                    plan.drop("thinking_budget", format!("{model} takes an effort level, not a token budget"));
                }
            }
        }
        if self.flavor.strict_reasoning_models.iter().any(|p| model.starts_with(p)) {
            for name in ["frequency_penalty", "presence_penalty", "stop"] {
                plan.drop(name, format!("{model} is a reasoning model and rejects {name}"));
            }
        }
    }

    fn payload(&self, req: &CompletionRequest) -> Option<Value> {
        Some(self.build_payload(req, false))
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<ModelResponse> {
        let started = Instant::now();
        let raw = self.transport.post("/chat/completions", &self.build_payload(req, false), &[], req.timeout).await?;
        self.decode(raw, req, started.elapsed().as_secs_f64() * 1000.0)
    }

    fn stream<'a>(&'a self, req: &'a CompletionRequest) -> BoxStream<'a, Result<StreamEvent>> {
        self.transport.retrying(move || {
            Box::pin(async_stream::try_stream! {
                let payload = self.build_payload(req, true);
                let mut text = String::new();
                let mut thinking = String::new();
                let mut calls: BTreeMap<u64, (String, String, String)> = BTreeMap::new();
                let mut usage = Usage { calls: 1, ..Default::default() };
                let mut stop = StopReason::EndTurn;
                let mut model = req.model.clone();
                let mut events = sse_events(self.transport.stream_lines("/chat/completions", &payload, &[], req.timeout));
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
                        Err(self.body_error(&data["error"]))?;
                    }
                    if let Some(m) = data["model"].as_str().filter(|m| !m.is_empty()) {
                        model = m.into();
                    }
                    if data["usage"].is_object() {
                        usage = Self::usage(&data);
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
                        if let Some(f) = choice["finish_reason"].as_str() {
                            stop = stop_map(f);
                        }
                    }
                }
                let mut blocks = Vec::new();
                if !thinking.is_empty() {
                    blocks.push(ContentBlock::Thinking(ThinkingBlock { thinking, signature: None, redacted: None }));
                }
                if !text.is_empty() {
                    blocks.push(ContentBlock::Text(TextBlock::new(text)));
                }
                let had_calls = !calls.is_empty();
                for (_, (id, name, args)) in calls {
                    let block = tool_use(Some(&id), &name, &Value::String(args));
                    yield StreamEvent::new(StreamEventType::ToolCall).with_data("id", json!(block.id)).with_data("name", json!(block.name)).with_data("input", Value::Object(block.input.clone()));
                    blocks.push(ContentBlock::ToolUse(block));
                }
                if had_calls && stop == StopReason::EndTurn {
                    stop = StopReason::ToolUse;
                }
                let resp = self.finish(Message::assistant_blocks(blocks), stop, usage, &model, json!({}), 0.0);
                yield StreamEvent::new(StreamEventType::StepEnd).with_data("response", serde_json::to_value(resp).unwrap_or_default());
            })
        })
    }

    async fn embed(&self, texts: &[String], model: Option<&str>) -> Result<Vec<Vec<f32>>> {
        let name = model.unwrap_or(self.flavor.embedding_model);
        if name.is_empty() {
            return Err(Error::config(format!("{} has no default embedding model — pass one", self.name())));
        }
        let raw = self.transport.post("/embeddings", &json!({"model": name, "input": texts}), &[], None).await?;
        let mut rows: Vec<&Value> = raw["data"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
        rows.sort_by_key(|r| r["index"].as_u64().unwrap_or(0));
        Ok(rows
            .into_iter()
            .map(|r| r["embedding"].as_array().map(|e| e.iter().filter_map(Value::as_f64).map(|x| x as f32).collect()).unwrap_or_default())
            .collect())
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let raw = self.transport.get("/models", &[], None).await?;
        let mut ids: Vec<String> = raw["data"].as_array().into_iter().flatten().filter_map(|m| m["id"].as_str().map(str::to_string)).collect();
        ids.sort();
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolSchema;
    use crate::types::{ToolResultBlock, attach_bytes};

    fn p() -> OpenAIProvider {
        OpenAIProvider::new(Some("k".into()))
    }

    #[test]
    fn encodes_tool_round_trip() {
        let call = ToolUseBlock { id: "c1".into(), name: "add".into(), input: json!({"a": 1}).as_object().unwrap().clone(), thought_signature: None };
        let mut req = CompletionRequest::new(
            "gpt-4.1",
            vec![
                Message::user("add"),
                Message::assistant_blocks(vec![ContentBlock::text("ok"), ContentBlock::ToolUse(call)]),
                Message::tool_results(vec![ToolResultBlock { tool_use_id: "c1".into(), content: "2".into(), is_error: false }]),
            ],
        );
        req.system = Some("sys".into());
        let msgs = p().encode_messages(&req);
        assert_eq!(msgs[0], json!({"role": "system", "content": "sys"}));
        assert_eq!(msgs[1], json!({"role": "user", "content": "add"}));
        assert_eq!(msgs[2]["tool_calls"][0]["function"]["arguments"], "{\"a\":1}");
        assert_eq!(msgs[3], json!({"role": "tool", "tool_call_id": "c1", "content": "2"}));
        assert_eq!(msgs.len(), 4);
    }

    #[test]
    fn media_parts() {
        let req = CompletionRequest::new("gpt-4.1", vec![Message::user_with("see", vec![attach_bytes(b"x", "image/png"), attach_bytes(b"y", "audio/wav")])]);
        let msgs = p().encode_messages(&req);
        let parts = msgs[0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "image_url");
        assert!(parts[0]["image_url"]["url"].as_str().unwrap().starts_with("data:image/png;base64,"));
        assert_eq!(parts[1]["type"], "text"); // gpt-4.1 does not listen
        let audio_req = CompletionRequest::new("gpt-4o-audio-preview", vec![Message::user_with("", vec![attach_bytes(b"y", "audio/wav")])]);
        assert_eq!(p().encode_messages(&audio_req)[0]["content"][0]["input_audio"]["format"], "wav");
    }

    #[test]
    fn reasoning_models() {
        let mut req = CompletionRequest::new("o3-mini", vec![Message::user("x")]);
        req.temperature = Some(0.2);
        req.stop = vec!["END".into()];
        req.effort = Some(Effort::Max);
        req.tools = vec![ToolSchema { name: "t".into(), description: String::new(), parameters: json!({}) }];
        req.tool_choice = Some(json!("any"));
        let body = p().build_payload(&req, true);
        assert_eq!(body["max_completion_tokens"], 8192);
        assert!(body.get("temperature").is_none());
        assert!(body.get("stop").is_none());
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["stream_options"]["include_usage"], true);

        let mut off = CompletionRequest::new("gpt-5", vec![]);
        off.thinking = Some(false);
        let plan = p().plan_parameters(&off);
        assert_eq!(plan.get("effort"), Some(&json!("minimal")));

        let mut plain = CompletionRequest::new("gpt-4.1", vec![]);
        plain.effort = Some(Effort::High);
        assert!(p().build_payload(&plain, false).get("reasoning_effort").is_none());
    }

    #[test]
    fn decodes_tool_calls_and_reasoning() {
        let raw = json!({"model": "gpt-4.1", "choices": [{"finish_reason": "stop", "message": {
            "content": "hi", "reasoning_content": "think",
            "tool_calls": [{"function": {"name": "add", "arguments": "{\"a\":2}"}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 40}}});
        let r = p().decode(raw, &CompletionRequest::new("gpt-4.1", vec![]), 0.0).unwrap();
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert!(r.tool_uses()[0].id.starts_with("call_"));
        assert_eq!(r.tool_uses()[0].input["a"], 2);
        assert_eq!(r.usage.input_tokens, 60);
        assert_eq!(r.usage.cache_read_tokens, 40);
        assert!(matches!(r.message.content[0], ContentBlock::Thinking(_)));
        let refusal = json!({"choices": [{"message": {"refusal": "no"}}]});
        let r = p().decode(refusal, &CompletionRequest::new("gpt-4.1", vec![]), 0.0).unwrap();
        assert_eq!(r.stop_reason, StopReason::Error);
        assert_eq!(r.text(), "[refused] no");
        let err = p().decode(json!({"error": {"code": "rate_limit_exceeded", "message": "slow"}}), &CompletionRequest::new("m", vec![]), 0.0).unwrap_err();
        assert_eq!(err.as_provider().unwrap().kind, ProviderErrorKind::RateLimit);
    }
}
