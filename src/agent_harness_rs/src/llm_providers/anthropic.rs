//! Anthropic Messages API adapter.
//!
//! [`AnthropicStream`] folds stream events into [`StreamEvent`]s and a final
//! response; it is transport-agnostic so Bedrock and Vertex can reuse it.

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
    normalise_model, sse_events,
};
use super::parameters::{Effort, ParameterPlan, nearest_effort};
use super::resilience::classify;
use crate::errors::{Error, Result};
use crate::types::{
    ContentBlock, MediaKind, Message, ModelResponse, Role, StopReason, StreamEvent, StreamEventType, TextBlock, ThinkingBlock, ToolUseBlock, Usage,
};

/// These take `thinking: {"type": "adaptive"}`; older ones still want budget_tokens.
const ADAPTIVE_THINKING: [&str; 6] = ["claude-opus-5", "claude-opus-4", "claude-sonnet-5", "claude-fable", "claude-mythos", "claude-sonnet-4-6"];

/// Claude's effort levels. `none` is handled by switching thinking off.
const EFFORTS: [Effort; 5] = [Effort::Low, Effort::Medium, Effort::High, Effort::Xhigh, Effort::Max];

fn stop_map(s: &str) -> StopReason {
    match s {
        "tool_use" | "pause_turn" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        "refusal" => StopReason::Error,
        _ => StopReason::EndTurn,
    }
}

/// What an SSE `error` event's type means, as the HTTP status it would have had.
fn stream_error_status(kind: &str) -> u16 {
    match kind {
        "overloaded_error" => 529,
        "rate_limit_error" => 429,
        "timeout_error" => 504,
        "authentication_error" => 401,
        "permission_error" => 403,
        "not_found_error" => 404,
        "request_too_large" => 413,
        "invalid_request_error" => 400,
        _ => 500,
    }
}

/// Models that size their own thinking — on any platform's spelling of the id.
pub fn adaptive(model: &str) -> bool {
    let m = normalise_model(model);
    ADAPTIVE_THINKING.iter().any(|p| m.starts_with(p))
}

const FIELDS: [ProviderField; 2] = [
    ProviderField {
        required: true,
        env: &["ANTHROPIC_API_KEY"],
        example: Some("sk-ant-..."),
        ..ProviderField::new("api_key", "secret", "An Anthropic API key from console.anthropic.com.")
    },
    ProviderField::new("base_url", "url", "Override for a proxy or gateway."),
];

pub const SPEC: ProviderSpec = ProviderSpec {
    name: "anthropic",
    display_name: "Anthropic",
    description: "Claude models through the Anthropic Messages API.",
    docs_url: "https://docs.anthropic.com/en/api/messages",
    auth_type: "api_key",
    env_key: "ANTHROPIC_API_KEY",
    default_model: "claude-opus-5",
    base_url: "https://api.anthropic.com",
    capabilities: &["streaming", "tools", "vision", "thinking", "json_schema", "prompt_caching", "list_models"],
    parameters: &["temperature", "top_p", "top_k"],
    modalities: &[MediaKind::Image, MediaKind::Document],
    fields: &FIELDS,
};

const API_VERSION: &str = "2023-06-01";

struct KeyAuth {
    key: String,
}

#[async_trait]
impl Auth for KeyAuth {
    async fn headers(&self, _: &Method, _: &str, _: &[u8]) -> Result<HeaderMap> {
        if self.key.is_empty() {
            return Err(missing_key(SPEC.name, SPEC.env_key));
        }
        let mut h = HeaderMap::new();
        h.insert("x-api-key", HeaderValue::from_str(&self.key).map_err(|e| Error::config(e.to_string()))?);
        h.insert("anthropic-version", HeaderValue::from_static(API_VERSION));
        Ok(h)
    }
}

/// Claude through the Anthropic Messages API.
pub struct AnthropicProvider {
    pub transport: HttpTransport,
}

impl AnthropicProvider {
    /// Reads `ANTHROPIC_API_KEY` when `api_key` is `None`.
    pub fn new(api_key: Option<String>) -> Self {
        Self::with_config(api_key, TransportConfig::default())
    }

    pub fn with_config(api_key: Option<String>, mut config: TransportConfig) -> Self {
        if config.base_url.is_empty() {
            config.base_url = SPEC.base_url.into();
        }
        let key = key_from_env(api_key, SPEC.env_key);
        Self { transport: HttpTransport::new(SPEC.name, config, Arc::new(KeyAuth { key })) }
    }

    // ---- encoding -----------------------------------------------------
    pub fn encode_block(block: &ContentBlock) -> Option<Value> {
        Some(match block {
            ContentBlock::Text(t) => json!({"type": "text", "text": t.text}),
            ContentBlock::Thinking(t) => {
                if let Some(r) = t.redacted.as_ref().filter(|r| !r.is_empty()) {
                    json!({"type": "redacted_thinking", "data": r})
                } else {
                    // Unsigned thinking (another vendor's) is rejected.
                    let sig = t.signature.as_ref().filter(|s| !s.is_empty())?;
                    json!({"type": "thinking", "thinking": t.thinking, "signature": sig})
                }
            }
            ContentBlock::ToolUse(u) => json!({"type": "tool_use", "id": u.id, "name": u.name, "input": u.input}),
            ContentBlock::ToolResult(r) => json!({"type": "tool_result", "tool_use_id": r.tool_use_id, "content": r.content, "is_error": r.is_error}),
            other => {
                let (kind, m) = other.as_media().expect("media");
                let sendable = kind == MediaKind::Image || (kind == MediaKind::Document && m.media_type == "application/pdf");
                if sendable {
                    let k = kind.as_str();
                    if let (Some(url), false) = (&m.url, m.inline()) {
                        return Some(json!({"type": k, "source": {"type": "url", "url": url}}));
                    }
                    if let Some(data) = media_data(m) {
                        return Some(json!({"type": k, "source": {"type": "base64", "media_type": m.media_type, "data": data}}));
                    }
                }
                json!({"type": "text", "text": media_note(kind, m)})
            }
        })
    }

    /// Anthropic keeps the system prompt out of the message list.
    pub fn encode_messages(messages: &[Message]) -> (Vec<Value>, Vec<String>) {
        let mut out: Vec<Value> = Vec::new();
        let mut systems = Vec::new();
        for msg in messages {
            if msg.role == Role::System {
                systems.push(msg.text());
                continue;
            }
            let role = if msg.role == Role::Assistant { "assistant" } else { "user" };
            let blocks: Vec<Value> = msg.content.iter().filter_map(Self::encode_block).collect();
            if blocks.is_empty() {
                continue;
            }
            // Merging same-role turns keeps the cache prefix tidy.
            if let Some(last) = out.last_mut()
                && last["role"] == role
            {
                last["content"].as_array_mut().expect("array").extend(blocks);
                continue;
            }
            out.push(json!({"role": role, "content": blocks}));
        }
        (out, systems)
    }

    pub fn build_payload(&self, req: &CompletionRequest, stream: bool) -> Value {
        let (messages, inline_systems) = Self::encode_messages(&req.messages);
        let system: Vec<String> = req.system.iter().cloned().chain(inline_systems).filter(|s| !s.is_empty()).collect();
        let mut p = Map::new();
        p.insert("model".into(), json!(req.model));
        p.insert("max_tokens".into(), json!(req.max_tokens));
        p.insert("messages".into(), json!(messages));
        if !system.is_empty() {
            p.insert("system".into(), json!(system.join("\n\n")));
        }
        if !req.tools.is_empty() {
            let tools: Vec<Value> = req.tools.iter().map(|t| json!({"name": t.name, "description": t.description, "input_schema": t.parameters})).collect();
            p.insert("tools".into(), json!(tools));
            let choice = match &req.tool_choice {
                Some(Value::String(s)) if s == "any" => Some(json!({"type": "any"})),
                Some(Value::String(s)) if s == "none" => Some(json!({"type": "none"})),
                Some(Value::String(s)) if s == "auto" => Some(json!({"type": "auto"})),
                Some(Value::Object(o)) => o.get("name").and_then(Value::as_str).map(|n| json!({"type": "tool", "name": n})),
                _ => None,
            };
            if let Some(c) = choice {
                p.insert("tool_choice".into(), c);
            }
        }
        let plan = self.plan_parameters(req);
        p.insert("max_tokens".into(), plan.get("max_tokens").cloned().unwrap_or(json!(req.max_tokens)));
        if let Some(stop) = plan.get("stop") {
            p.insert("stop_sequences".into(), stop.clone());
        }
        if plan.bool("thinking") == Some(true) {
            let thinking = if adaptive(&req.model) {
                json!({"type": "adaptive", "display": "summarized"})
            } else {
                json!({"type": "enabled", "budget_tokens": plan.get("thinking_budget")})
            };
            p.insert("thinking".into(), thinking);
        }
        for key in ["temperature", "top_p", "top_k"] {
            if let Some(v) = plan.get(key) {
                p.insert(key.into(), v.clone());
            }
        }
        if let Some(e) = plan.get("effort") {
            p.insert("output_config".into(), json!({"effort": e}));
        }
        if let Some(schema) = &req.response_schema {
            let oc = p.entry("output_config").or_insert_with(|| json!({}));
            oc["format"] = json!({"type": "json_schema", "schema": schema});
        }
        if req.parallel_tool_calls == Some(false) && !req.tools.is_empty() {
            let tc = p.entry("tool_choice").or_insert_with(|| json!({"type": "auto"}));
            tc["disable_parallel_tool_use"] = json!(true);
        }
        if req.cache == Some(true) {
            p.insert("cache_control".into(), json!({"type": "ephemeral"}));
        }
        if let Some(speed) = &req.speed {
            p.insert("speed".into(), json!(speed));
        }
        if req.user.is_some() || !req.metadata.is_empty() {
            let mut meta = req.metadata.clone();
            if let Some(u) = &req.user {
                meta.insert("user_id".into(), json!(u));
            }
            p.insert("metadata".into(), Value::Object(meta));
        }
        if stream {
            p.insert("stream".into(), json!(true));
        }
        p.extend(req.extra.clone());
        Value::Object(p)
    }

    // ---- decoding -----------------------------------------------------
    pub fn decode_blocks(raw: &[Value]) -> Vec<ContentBlock> {
        let s = |b: &Value, k: &str| b.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        raw.iter()
            .filter_map(|b| match b.get("type").and_then(Value::as_str)? {
                "text" => Some(ContentBlock::Text(TextBlock::new(s(b, "text")))),
                "thinking" => Some(ContentBlock::Thinking(ThinkingBlock {
                    thinking: s(b, "thinking"),
                    signature: Some(s(b, "signature")).filter(|x| !x.is_empty()),
                    redacted: None,
                })),
                // Encrypted, but it must go back verbatim or the next turn fails.
                "redacted_thinking" => Some(ContentBlock::Thinking(ThinkingBlock { thinking: String::new(), signature: None, redacted: Some(s(b, "data")) })),
                "tool_use" => Some(ContentBlock::ToolUse(ToolUseBlock {
                    id: s(b, "id"),
                    name: s(b, "name"),
                    input: b.get("input").and_then(Value::as_object).cloned().unwrap_or_default(),
                    thought_signature: None,
                })),
                _ => None,
            })
            .collect()
    }

    pub fn usage(raw: &Value) -> Usage {
        let u = &raw["usage"];
        let n = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
        Usage {
            input_tokens: n("input_tokens"),
            output_tokens: n("output_tokens"),
            cache_read_tokens: n("cache_read_input_tokens"),
            cache_write_tokens: n("cache_creation_input_tokens"),
            calls: 1,
            ..Default::default()
        }
    }

    pub fn decode(&self, raw: Value, req: &CompletionRequest, latency_ms: f64) -> Result<ModelResponse> {
        if raw["type"] == "error" {
            return Err(self.stream_error(&raw["error"]));
        }
        let mut blocks = Self::decode_blocks(raw["content"].as_array().map(Vec::as_slice).unwrap_or(&[]));
        let stop = stop_map(raw["stop_reason"].as_str().unwrap_or("end_turn"));
        if stop == StopReason::Error {
            let detail = raw["stop_details"]["explanation"].as_str().unwrap_or("request refused");
            blocks.push(ContentBlock::text(format!("[refused] {detail}")));
        }
        let model = raw["model"].as_str().filter(|m| !m.is_empty()).unwrap_or(&req.model).to_string();
        let usage = Self::usage(&raw);
        Ok(self.finish(Message::assistant_blocks(blocks), stop, usage, &model, raw, latency_ms))
    }

    pub fn stream_error(&self, error: &Value) -> Error {
        let kind = error["type"].as_str().unwrap_or("api_error");
        classify(SPEC.name, stream_error_status(kind), None, Some(&json!({"error": error}).to_string()), Some(&self.transport.retry)).into()
    }
}

/// Folds Anthropic stream events into StreamEvents and, at the end, a response.
pub struct AnthropicStream<'a> {
    provider: &'a dyn Provider,
    error: &'a (dyn Fn(&Value) -> Error + Send + Sync),
    pub model: String,
    blocks: BTreeMap<u64, Map<String, Value>>,
    usage: Usage,
    stop_reason: StopReason,
    stop_detail: String,
}

impl<'a> AnthropicStream<'a> {
    pub fn new(provider: &'a dyn Provider, error: &'a (dyn Fn(&Value) -> Error + Send + Sync), model: &str) -> Self {
        Self {
            provider,
            error,
            model: model.into(),
            blocks: BTreeMap::new(),
            usage: Usage { calls: 1, ..Default::default() },
            stop_reason: StopReason::EndTurn,
            stop_detail: String::new(),
        }
    }

    fn append(slot: &mut Map<String, Value>, key: &str, more: &str) {
        let current = slot.get(key).and_then(Value::as_str).unwrap_or("").to_string();
        slot.insert(key.into(), json!(current + more));
    }

    pub fn feed(&mut self, data: &Value) -> Result<Vec<StreamEvent>> {
        let mut out = Vec::new();
        match data["type"].as_str().unwrap_or("") {
            "message_start" => {
                let msg = &data["message"];
                if let Some(m) = msg["model"].as_str().filter(|m| !m.is_empty()) {
                    self.model = m.into();
                }
                self.usage = AnthropicProvider::usage(msg);
            }
            "content_block_start" => {
                let mut slot = data["content_block"].as_object().cloned().unwrap_or_default();
                slot.insert("_json".into(), json!(""));
                let idx = data["index"].as_u64().unwrap_or(self.blocks.len() as u64);
                self.blocks.insert(idx, slot);
            }
            "content_block_delta" => {
                let delta = &data["delta"];
                let idx = data["index"].as_u64().unwrap_or(0);
                let slot = self.blocks.entry(idx).or_insert_with(|| {
                    let mut m = Map::new();
                    m.insert("type".into(), json!("text"));
                    m.insert("text".into(), json!(""));
                    m
                });
                let text = |k: &str| delta[k].as_str().unwrap_or("").to_string();
                match delta["type"].as_str().unwrap_or("") {
                    "text_delta" => {
                        Self::append(slot, "text", &text("text"));
                        out.push(StreamEvent::text(text("text")));
                    }
                    "thinking_delta" => {
                        Self::append(slot, "thinking", &text("thinking"));
                        out.push(StreamEvent { text: text("thinking"), ..StreamEvent::new(StreamEventType::Thinking) });
                    }
                    // Without it the thinking block cannot be sent back next turn.
                    "signature_delta" => Self::append(slot, "signature", &text("signature")),
                    "input_json_delta" => Self::append(slot, "_json", &text("partial_json")),
                    _ => {}
                }
            }
            "message_delta" => {
                let delta = &data["delta"];
                if let Some(s) = delta["stop_reason"].as_str() {
                    self.stop_reason = stop_map(s);
                }
                if let Some(e) = delta["stop_details"]["explanation"].as_str() {
                    self.stop_detail = e.into();
                }
                // These counts are cumulative, so they replace rather than add.
                let u = &data["usage"];
                if let Some(o) = u.get("output_tokens") {
                    self.usage.output_tokens = o.as_u64().unwrap_or(0);
                }
                for (k, f) in [
                    ("input_tokens", &mut self.usage.input_tokens),
                    ("cache_read_input_tokens", &mut self.usage.cache_read_tokens),
                    ("cache_creation_input_tokens", &mut self.usage.cache_write_tokens),
                ] {
                    if let Some(v) = u.get(k).and_then(Value::as_u64) {
                        *f = v;
                    }
                }
            }
            "error" => return Err((self.error)(&data["error"])),
            _ => {}
        }
        Ok(out)
    }

    pub fn finish(self) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        let mut final_blocks = Vec::new();
        for (_, mut slot) in self.blocks {
            if let Some(Value::String(j)) = slot.remove("_json")
                && !j.is_empty()
            {
                slot.insert("input".into(), serde_json::from_str(&j).unwrap_or_else(|_| json!({})));
            }
            for block in AnthropicProvider::decode_blocks(&[Value::Object(slot)]) {
                if let ContentBlock::ToolUse(u) = &block {
                    out.push(
                        StreamEvent::new(StreamEventType::ToolCall)
                            .with_data("id", json!(u.id))
                            .with_data("name", json!(u.name))
                            .with_data("input", Value::Object(u.input.clone())),
                    );
                }
                final_blocks.push(block);
            }
        }
        if self.stop_reason == StopReason::Error {
            let detail = if self.stop_detail.is_empty() { "request refused" } else { &self.stop_detail };
            final_blocks.push(ContentBlock::text(format!("[refused] {detail}")));
        }
        let resp = self.provider.finish(Message::assistant_blocks(final_blocks), self.stop_reason, self.usage, &self.model, json!({}), 0.0);
        out.push(StreamEvent::new(StreamEventType::StepEnd).with_data("response", serde_json::to_value(resp).unwrap_or_default()));
        out
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn spec(&self) -> &ProviderSpec {
        &SPEC
    }

    fn transport(&self) -> Option<&HttpTransport> {
        Some(&self.transport)
    }

    /// Claude's rules, which the API enforces with a 400 if broken.
    fn plan(&self, req: &CompletionRequest, plan: &mut ParameterPlan) {
        let model = normalise_model(&req.model);
        let is_adaptive = adaptive(&req.model);

        match plan.effort() {
            Some(Effort::None) => {
                plan.drop("effort", "'none' means no thinking, and Claude's lowest effort is 'low'");
                if plan.bool("thinking") == Some(true) {
                    plan.drop("thinking", "effort='none' switches thinking off");
                }
            }
            Some(e) if !EFFORTS.contains(&e) => {
                let names: Vec<&str> = EFFORTS.iter().map(|e| e.as_str()).collect();
                plan.adjust("effort", json!(nearest_effort(e, &EFFORTS).as_str()), &format!("Claude's effort levels are {}", names.join(", ")));
            }
            _ => {}
        }

        // A budget on its own is a request to think.
        if plan.contains("thinking_budget") && plan.get("thinking").is_none() && !plan.dropped.contains_key("effort") {
            plan.set("thinking", json!(true));
        }
        let forced = matches!(&req.tool_choice, Some(Value::String(s)) if s == "any") || matches!(req.tool_choice, Some(Value::Object(_)));
        if plan.bool("thinking") == Some(true) && forced {
            plan.drop("thinking", "Claude cannot think when tool_choice forces a tool");
        }
        if plan.bool("thinking") == Some(false) {
            plan.values.shift_remove("thinking"); // off is the default: nothing to send
        }
        let thinking = plan.bool("thinking") == Some(true);

        if thinking && is_adaptive {
            plan.drop("thinking_budget", format!("{model} sizes its own thinking — use effort"));
        } else if thinking {
            let max_tokens = plan.i64("max_tokens").unwrap_or(req.max_tokens as i64);
            match plan.i64("thinking_budget") {
                None => plan.set("thinking_budget", json!(1024.max((max_tokens - 1).min(max_tokens / 2)))),
                Some(b) if b < 1024 => plan.adjust("thinking_budget", json!(1024), "Claude's minimum thinking budget is 1024"),
                _ => {}
            }
            let budget = plan.i64("thinking_budget").unwrap_or(1024);
            if budget >= max_tokens {
                plan.adjust("max_tokens", json!(budget + 1024), "max_tokens must exceed the thinking budget");
            }
        } else {
            plan.drop("thinking_budget", "thinking is off");
        }

        if is_adaptive {
            for name in ["temperature", "top_p", "top_k"] {
                plan.drop(name, format!("{model} rejects sampling settings"));
            }
        } else if thinking {
            plan.drop("temperature", "not allowed with extended thinking");
            plan.drop("top_k", "not allowed with extended thinking");
            if plan.f64("top_p").is_some_and(|p| p < 0.95) {
                plan.adjust("top_p", json!(0.95), "with extended thinking top_p must be 0.95–1");
            }
        } else {
            if plan.f64("temperature").is_some_and(|t| t > 1.0) {
                plan.adjust("temperature", json!(1.0), "Claude's temperature range is 0–1");
            }
            if plan.contains("temperature") && plan.contains("top_p") {
                plan.drop("top_p", "Claude takes temperature or top_p, not both — temperature was kept");
            }
        }
    }

    fn payload(&self, req: &CompletionRequest) -> Option<Value> {
        Some(self.build_payload(req, false))
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<ModelResponse> {
        let started = Instant::now();
        let raw = self.transport.post("/v1/messages", &self.build_payload(req, false), &[], req.timeout).await?;
        self.decode(raw, req, started.elapsed().as_secs_f64() * 1000.0)
    }

    fn stream<'a>(&'a self, req: &'a CompletionRequest) -> BoxStream<'a, Result<StreamEvent>> {
        self.transport.retrying(move || {
            Box::pin(async_stream::try_stream! {
                let payload = self.build_payload(req, true);
                let err = |e: &Value| self.stream_error(e);
                let mut state = AnthropicStream::new(self, &err, &req.model);
                let mut events = sse_events(self.transport.stream_lines("/v1/messages", &payload, &[], req.timeout));
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
        let mut models = Vec::new();
        let mut params = vec![("limit".to_string(), "1000".to_string())];
        loop {
            let raw = self.transport.get("/v1/models", &params, None).await?;
            models.extend(raw["data"].as_array().into_iter().flatten().filter_map(|m| m["id"].as_str().map(str::to_string)));
            match (raw["has_more"].as_bool(), raw["last_id"].as_str()) {
                (Some(true), Some(last)) => params = vec![("limit".into(), "1000".into()), ("after_id".into(), last.into())],
                _ => return Ok(models),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::attach_bytes;

    fn provider() -> AnthropicProvider {
        AnthropicProvider::new(Some("k".into()))
    }

    #[test]
    fn payload_shape() {
        let p = provider();
        let mut req = CompletionRequest::new(
            "claude-haiku-4-5",
            vec![Message::system("be brief"), Message::user_with("what is this", vec![attach_bytes(b"png", "image/png")])],
        );
        req.system = Some("you help".into());
        req.tools = vec![crate::tools::ToolSchema { name: "t".into(), description: "d".into(), parameters: json!({"type": "object"}) }];
        req.tool_choice = Some(json!({"name": "t"}));
        req.temperature = Some(1.5);
        req.top_p = Some(0.5);
        req.seed = Some(3);
        let body = p.build_payload(&req, true);
        assert_eq!(body["system"], "you help\n\nbe brief");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["content"][0]["source"]["type"], "base64");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(body["tool_choice"], json!({"type": "tool", "name": "t"}));
        assert_eq!(body["temperature"], 1.0);
        assert!(body.get("top_p").is_none());
        assert!(body.get("seed").is_none());
        assert_eq!(body["stream"], true);
        let plan = p.plan_parameters(&req);
        assert_eq!(plan.dropped["seed"], "Anthropic has no seed");
    }

    #[test]
    fn thinking_rules() {
        let p = provider();
        let mut req = CompletionRequest::new("claude-haiku-4-5", vec![Message::user("x")]);
        req.thinking_budget = Some(10);
        req.max_tokens = 1000;
        req.top_p = Some(0.5);
        let plan = p.plan_parameters(&req);
        assert_eq!(plan.get("thinking"), Some(&json!(true)));
        assert_eq!(plan.get("thinking_budget"), Some(&json!(1024)));
        assert_eq!(plan.get("max_tokens"), Some(&json!(2048)));
        assert_eq!(plan.get("top_p"), Some(&json!(0.95)));
        let body = p.build_payload(&req, false);
        assert_eq!(body["thinking"], json!({"type": "enabled", "budget_tokens": 1024}));

        let mut adaptive_req = CompletionRequest::new("claude-opus-5", vec![Message::user("x")]);
        adaptive_req.thinking = Some(true);
        adaptive_req.effort = Some(Effort::Minimal);
        adaptive_req.temperature = Some(0.3);
        let body = p.build_payload(&adaptive_req, false);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "low");
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn unsigned_thinking_is_not_sent() {
        let blocks = [
            ContentBlock::Thinking(ThinkingBlock { thinking: "hm".into(), signature: None, redacted: None }),
            ContentBlock::Thinking(ThinkingBlock { thinking: String::new(), signature: None, redacted: Some("enc".into()) }),
        ];
        let encoded: Vec<Value> = blocks.iter().filter_map(AnthropicProvider::encode_block).collect();
        assert_eq!(encoded, vec![json!({"type": "redacted_thinking", "data": "enc"})]);
    }

    #[test]
    fn decodes_response() {
        let p = provider();
        let raw = json!({"model": "claude-opus-5", "stop_reason": "tool_use", "content": [
            {"type": "thinking", "thinking": "plan", "signature": "sig"},
            {"type": "text", "text": "calling"},
            {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 1}}
        ], "usage": {"input_tokens": 1000, "output_tokens": 100}});
        let req = CompletionRequest::new("claude-opus-5", vec![]);
        let r = p.decode(raw, &req, 5.0).unwrap();
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(r.text(), "calling");
        assert_eq!(r.tool_uses()[0].id, "toolu_1");
        assert_eq!(r.usage.cost_usd, 0.0075);
        let err = p.decode(json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}), &req, 0.0).unwrap_err();
        assert_eq!(err.as_provider().unwrap().status, Some(529));
    }

    #[test]
    fn stream_assembly() {
        let p = provider();
        let err = |e: &Value| p.stream_error(e);
        let mut s = AnthropicStream::new(&p, &err, "claude-opus-5");
        let feed = [
            json!({"type": "message_start", "message": {"model": "claude-opus-5", "usage": {"input_tokens": 10}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "add"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "2}"}}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}}),
        ];
        let mut texts = Vec::new();
        for f in &feed {
            texts.extend(s.feed(f).unwrap().into_iter().map(|e| e.text));
        }
        assert_eq!(texts, vec!["Hi"]);
        let events = s.finish();
        assert_eq!(events[0].kind, StreamEventType::ToolCall);
        assert_eq!(events[0].data["input"], json!({"a": 2}));
        let resp = super::super::base::response_from_event(events.last().unwrap()).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 7);
    }
}
