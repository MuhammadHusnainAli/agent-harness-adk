//! Google Gemini (Generative Language API) adapter.
//!
//! Decoding is public so Gemini on Vertex AI, which serves the same format
//! behind a different URL and auth, can share it.

use std::collections::HashMap;
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
use crate::errors::{Error, ProviderError, ProviderErrorKind, Result};
use crate::types::{
    ContentBlock, MediaKind, Message, ModelResponse, Role, StopReason, StreamEvent, StreamEventType, TextBlock, ThinkingBlock, ToolUseBlock, Usage,
};

fn stop_map(s: &str) -> StopReason {
    match s {
        "MAX_TOKENS" => StopReason::MaxTokens,
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" | "MALFORMED_FUNCTION_CALL" => StopReason::Error,
        _ => StopReason::EndTurn,
    }
}

/// google.rpc status names, as the HTTP status they stand for.
fn rpc_status(s: &str) -> u16 {
    match s {
        "RESOURCE_EXHAUSTED" => 429,
        "UNAVAILABLE" => 503,
        "DEADLINE_EXCEEDED" => 504,
        "INVALID_ARGUMENT" => 400,
        "NOT_FOUND" => 404,
        "PERMISSION_DENIED" => 403,
        "UNAUTHENTICATED" => 401,
        _ => 500,
    }
}

/// Gemini takes an OpenAPI subset and 400s on stray JSON Schema keywords.
const SCHEMA_DROP: [&str; 14] = [
    "$schema",
    "additionalProperties",
    "definitions",
    "$defs",
    "title",
    "default",
    "examples",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "const",
    "allOf",
    "oneOf",
    "not",
    "patternProperties",
];

/// Effort as a Gemini 2.5 thinking budget, in tokens.
fn effort_budget(e: Effort) -> i64 {
    match e {
        Effort::None => 0,
        Effort::Minimal => 512,
        Effort::Low => 1024,
        Effort::Medium => 8192,
        Effort::High => 16384,
        Effort::Xhigh => 24576,
        Effort::Max => 32768,
    }
}

/// (lowest, highest, can switch off) for a Gemini 2.5 model's thinking budget.
fn budget_range(model: &str) -> (i64, i64, bool) {
    if model.contains("pro") {
        (128, 32768, false)
    } else if model.contains("lite") {
        (512, 24576, true)
    } else {
        (1, 24576, true)
    }
}

/// A JSON Schema with the keywords Gemini rejects taken out.
pub fn clean_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => {
            let mut out: Map<String, Value> =
                map.iter().filter(|(k, _)| !SCHEMA_DROP.contains(&k.as_str())).map(|(k, v)| (k.clone(), clean_schema(v))).collect();
            if out.get("type") == Some(&json!("object")) && !out.contains_key("properties") {
                out.insert("properties".into(), json!({}));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(clean_schema).collect()),
        other => other.clone(),
    }
}

const FIELDS: [ProviderField; 2] = [
    ProviderField {
        required: true,
        env: &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        example: Some("AIza..."),
        ..ProviderField::new("api_key", "secret", "A Gemini API key from aistudio.google.com.")
    },
    ProviderField::new("base_url", "url", "Override for a proxy or gateway."),
];

pub const SPEC: ProviderSpec = ProviderSpec {
    name: "gemini",
    display_name: "Google Gemini",
    description: "Gemini models through the Generative Language API (AI Studio).",
    docs_url: "https://ai.google.dev/api/generate-content",
    auth_type: "api_key",
    env_key: "GEMINI_API_KEY",
    default_model: "gemini-2.5-pro",
    base_url: "https://generativelanguage.googleapis.com/v1beta",
    capabilities: &["streaming", "tools", "vision", "thinking", "json_schema", "embeddings", "list_models"],
    parameters: &["temperature", "top_p", "top_k", "seed", "frequency_penalty", "presence_penalty"],
    modalities: &[MediaKind::Image, MediaKind::Audio, MediaKind::Video, MediaKind::Document],
    fields: &FIELDS,
};

const EMBEDDING_MODEL: &str = "gemini-embedding-001";

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
        h.insert("x-goog-api-key", HeaderValue::from_str(&self.key).map_err(|e| Error::config(e.to_string()))?);
        Ok(h)
    }
}

/// Gemini through AI Studio.
pub struct GeminiProvider {
    pub transport: HttpTransport,
}

impl GeminiProvider {
    /// Reads `GEMINI_API_KEY`, then `GOOGLE_API_KEY`, when `api_key` is `None`.
    pub fn new(api_key: Option<String>) -> Self {
        Self::with_config(api_key, TransportConfig::default())
    }

    pub fn with_config(api_key: Option<String>, mut config: TransportConfig) -> Self {
        if config.base_url.is_empty() {
            config.base_url = SPEC.base_url.into();
        }
        let mut key = key_from_env(api_key, SPEC.env_key);
        if key.is_empty() {
            key = std::env::var("GOOGLE_API_KEY").unwrap_or_default();
        }
        Self { transport: HttpTransport::new(SPEC.name, config, Arc::new(KeyAuth { key })) }
    }

    // ---- encoding -----------------------------------------------------
    /// The `contents` list and any system messages found inline.
    pub fn encode_contents(&self, req: &CompletionRequest) -> (Vec<Value>, Vec<String>) {
        // Gemini's functionResponse is keyed by name, not by call id.
        let names: HashMap<&str, &str> = req.messages.iter().flat_map(|m| m.tool_uses()).map(|u| (u.id.as_str(), u.name.as_str())).collect();
        let mut contents: Vec<Value> = Vec::new();
        let mut systems = Vec::new();
        for msg in &req.messages {
            if msg.role == Role::System {
                systems.push(msg.text());
                continue;
            }
            let role = if msg.role == Role::Assistant { "model" } else { "user" };
            let mut parts: Vec<Value> = Vec::new();
            for block in &msg.content {
                let (part, signature) = match block {
                    ContentBlock::Text(t) if !t.text.is_empty() => (Some(json!({"text": t.text})), t.thought_signature.as_ref()),
                    ContentBlock::Text(_) | ContentBlock::Thinking(_) => (None, None),
                    ContentBlock::ToolUse(u) => (Some(json!({"functionCall": {"name": u.name, "args": u.input}})), u.thought_signature.as_ref()),
                    ContentBlock::ToolResult(r) => {
                        let name = names.get(r.tool_use_id.as_str()).copied().unwrap_or("tool");
                        let error = if r.is_error { json!(true) } else { Value::Null };
                        (Some(json!({"functionResponse": {"name": name, "response": {"output": r.content, "error": error}}})), None)
                    }
                    other => {
                        // Every kind goes the same way: the bytes inline, or a URI
                        // Gemini can fetch (a Files API upload, a YouTube link).
                        let (kind, m) = other.as_media().expect("media");
                        let ok = self.accepts(kind, m, &req.model);
                        let data = if m.inline() { media_data(m) } else { None };
                        let part = match (data, &m.url) {
                            (Some(d), _) if ok => json!({"inlineData": {"mimeType": m.media_type, "data": d}}),
                            (None, Some(url)) if ok && !m.inline() => json!({"fileData": {"mimeType": m.media_type, "fileUri": url}}),
                            _ => json!({"text": media_note(kind, m)}),
                        };
                        (Some(part), None)
                    }
                };
                if let Some(mut part) = part {
                    // Thinking models need their signature back on the next turn.
                    if let Some(sig) = signature {
                        part["thoughtSignature"] = json!(sig);
                    }
                    parts.push(part);
                }
            }
            if parts.is_empty() {
                continue;
            }
            if let Some(last) = contents.last_mut()
                && last["role"] == role
            {
                last["parts"].as_array_mut().expect("array").extend(parts);
                continue;
            }
            contents.push(json!({"role": role, "parts": parts}));
        }
        (contents, systems)
    }

    pub fn build_payload(&self, req: &CompletionRequest) -> Value {
        let (contents, inline_systems) = self.encode_contents(req);
        let system: Vec<String> = req.system.iter().cloned().chain(inline_systems).filter(|s| !s.is_empty()).collect();
        let plan = self.plan_parameters(req);
        let mut generation = Map::new();
        generation.insert("maxOutputTokens".into(), plan.get("max_tokens").cloned().unwrap_or(json!(req.max_tokens)));
        for (field, key) in [
            ("temperature", "temperature"),
            ("top_p", "topP"),
            ("top_k", "topK"),
            ("seed", "seed"),
            ("frequency_penalty", "frequencyPenalty"),
            ("presence_penalty", "presencePenalty"),
        ] {
            if let Some(v) = plan.get(field) {
                generation.insert(key.into(), v.clone());
            }
        }
        if let Some(stop) = plan.get("stop") {
            generation.insert("stopSequences".into(), stop.clone());
        }
        if let Some(mt) = &req.response_mime_type {
            generation.insert("responseMimeType".into(), json!(mt));
        }
        if let Some(schema) = &req.response_schema {
            generation.insert("responseMimeType".into(), json!("application/json"));
            generation.insert("responseSchema".into(), clean_schema(schema));
        }
        let thinking = plan.bool("thinking") == Some(true);
        if plan.contains("thinking_budget") || plan.contains("thinking_level") || thinking {
            let mut config = Map::new();
            config.insert("includeThoughts".into(), json!(thinking));
            if let Some(level) = plan.get("thinking_level") {
                config.insert("thinkingLevel".into(), level.clone());
            } else if let Some(budget) = plan.get("thinking_budget") {
                config.insert("thinkingBudget".into(), budget.clone());
            }
            generation.insert("thinkingConfig".into(), Value::Object(config));
        }
        let mut p = Map::new();
        p.insert("contents".into(), json!(contents));
        p.insert("generationConfig".into(), Value::Object(generation));
        if !req.safety_settings.is_empty() {
            p.insert("safetySettings".into(), json!(req.safety_settings));
        }
        if !system.is_empty() {
            p.insert("systemInstruction".into(), json!({"parts": [{"text": system.join("\n\n")}]}));
        }
        if !req.tools.is_empty() {
            let decls: Vec<Value> =
                req.tools.iter().map(|t| json!({"name": t.name, "description": t.description, "parameters": clean_schema(&t.parameters)})).collect();
            p.insert("tools".into(), json!([{"functionDeclarations": decls}]));
            let cfg = match &req.tool_choice {
                Some(Value::Object(o)) if o.get("name").and_then(Value::as_str).is_some() => json!({"mode": "ANY", "allowedFunctionNames": [o["name"]]}),
                Some(Value::String(s)) if s == "any" => json!({"mode": "ANY"}),
                Some(Value::String(s)) if s == "none" => json!({"mode": "NONE"}),
                _ => json!({"mode": "AUTO"}),
            };
            p.insert("toolConfig".into(), json!({"functionCallingConfig": cfg}));
        }
        p.extend(req.extra.clone());
        Value::Object(p)
    }

    // ---- decoding -----------------------------------------------------
    pub fn usage(raw: &Value) -> Usage {
        let u = &raw["usageMetadata"];
        let cached = u["cachedContentTokenCount"].as_u64().unwrap_or(0);
        Usage {
            input_tokens: u["promptTokenCount"].as_u64().unwrap_or(0).saturating_sub(cached),
            output_tokens: u["candidatesTokenCount"].as_u64().unwrap_or(0),
            cache_read_tokens: cached,
            calls: 1,
            ..Default::default()
        }
    }

    pub fn decode_parts(parts: &[Value]) -> Vec<ContentBlock> {
        let mut blocks: Vec<ContentBlock> = Vec::new();
        for part in parts {
            let sig = part["thoughtSignature"].as_str().filter(|s| !s.is_empty()).map(str::to_string);
            if let Some(call) = part.get("functionCall") {
                let mut u = ToolUseBlock::new(call["name"].as_str().unwrap_or(""), call["args"].as_object().cloned().unwrap_or_default());
                if let Some(id) = call["id"].as_str().filter(|s| !s.is_empty()) {
                    u.id = id.into();
                }
                u.thought_signature = sig;
                blocks.push(ContentBlock::ToolUse(u));
            } else if let Some(text) = part["text"].as_str().filter(|t| !t.is_empty()) {
                if part["thought"].as_bool() == Some(true) {
                    blocks.push(ContentBlock::Thinking(ThinkingBlock { thinking: text.into(), signature: None, redacted: None }));
                } else {
                    blocks.push(ContentBlock::Text(TextBlock { text: text.into(), thought_signature: sig }));
                }
            } else if let Some(sig) = sig {
                // A bare signature part belongs to the block before it.
                match blocks.last_mut() {
                    Some(ContentBlock::Text(t)) => t.thought_signature = Some(sig),
                    Some(ContentBlock::ToolUse(u)) => u.thought_signature = Some(sig),
                    _ => {}
                }
            }
        }
        blocks
    }

    pub fn decode(&self, raw: Value, req: &CompletionRequest, latency_ms: f64) -> Result<ModelResponse> {
        if !raw["error"].is_null() {
            return Err(self.body_error(&raw["error"]));
        }
        let Some(cand) = raw["candidates"].as_array().and_then(|c| c.first()) else {
            let reason = raw["promptFeedback"]["blockReason"].as_str().unwrap_or("no candidates");
            let mut e = ProviderError::new(ProviderErrorKind::Other, format!("{} returned nothing: {reason}", SPEC.name)).provider(SPEC.name);
            e.body = Some(raw.to_string().chars().take(2000).collect());
            return Err(e.into());
        };
        let blocks = Self::decode_parts(cand["content"]["parts"].as_array().map(Vec::as_slice).unwrap_or(&[]));
        let mut stop = stop_map(cand["finishReason"].as_str().unwrap_or("STOP"));
        if blocks.iter().any(|b| matches!(b, ContentBlock::ToolUse(_))) {
            stop = StopReason::ToolUse;
        }
        let model = raw["modelVersion"].as_str().filter(|m| !m.is_empty()).unwrap_or(&req.model).to_string();
        let usage = Self::usage(&raw);
        Ok(self.finish(Message::assistant_blocks(blocks), stop, usage, &model, raw, latency_ms))
    }

    pub fn body_error(&self, error: &Value) -> Error {
        let err = if error.is_object() { error.clone() } else { json!({"message": error.to_string()}) };
        let status = err["code"].as_u64().map(|c| c as u16).unwrap_or_else(|| rpc_status(err["status"].as_str().unwrap_or("")));
        classify(SPEC.name, status, None, Some(&json!({"error": err}).to_string()), Some(&self.transport.retry)).into()
    }

    fn model_path(model: &str, stream: bool) -> String {
        let name = model.strip_prefix("models/").unwrap_or(model);
        format!("/models/{name}:{}", if stream { "streamGenerateContent" } else { "generateContent" })
    }
}

/// Merge streamed text and thinking fragments back into whole blocks.
fn merge_fragments(blocks: Vec<ContentBlock>) -> Vec<ContentBlock> {
    let mut merged: Vec<ContentBlock> = Vec::new();
    for block in blocks {
        match (merged.last_mut(), block) {
            (Some(ContentBlock::Text(prev)), ContentBlock::Text(next)) if prev.thought_signature.is_none() => {
                prev.text.push_str(&next.text);
                prev.thought_signature = next.thought_signature;
            }
            (Some(ContentBlock::Thinking(prev)), ContentBlock::Thinking(next)) => prev.thinking.push_str(&next.thinking),
            (_, block) => merged.push(block),
        }
    }
    merged
}

#[async_trait]
impl Provider for GeminiProvider {
    fn spec(&self) -> &ProviderSpec {
        &SPEC
    }

    fn transport(&self) -> Option<&HttpTransport> {
        Some(&self.transport)
    }

    /// Gemini 2.5 thinks in token budgets, Gemini 3 in levels, earlier models
    /// not at all — `effort` and `thinking` become whichever this model takes.
    fn plan(&self, req: &CompletionRequest, plan: &mut ParameterPlan) {
        let normal = normalise_model(&req.model);
        let model = normal.rsplit('/').next().unwrap_or(&normal).to_string();
        let effort = plan.effort();
        let mut thinking = plan.bool("thinking");
        let budget = plan.i64("thinking_budget");
        if effort == Some(Effort::None) && thinking == Some(true) {
            plan.drop("thinking", "effort='none' switches thinking off");
            thinking = None;
        }

        if model.starts_with("gemini-3") {
            let levels: &[Effort] =
                if model.contains("pro") { &[Effort::Low, Effort::High] } else { &[Effort::Minimal, Effort::Low, Effort::Medium, Effort::High] };
            if budget.is_some() {
                plan.drop("effort", "thinking_budget was given; Gemini takes one or the other");
            } else if effort.is_some() || thinking == Some(false) {
                let wanted = effort.unwrap_or(Effort::None);
                let level = nearest_effort(wanted, levels);
                plan.values.shift_remove("effort");
                plan.set("thinking_level", json!(level.as_str()));
                let names: Vec<&str> = levels.iter().map(|l| l.as_str()).collect();
                let why = if wanted == Effort::None {
                    format!("{model} cannot switch thinking off; its lowest level is '{level}'")
                } else {
                    format!("{model} takes {}", names.join(", "))
                };
                let suffix = if level != wanted { format!(": {why}") } else { String::new() };
                plan.adjusted.insert("effort".into(), format!("'{wanted}' → thinkingLevel '{level}'{suffix}"));
            }
            return;
        }

        if !model.starts_with("gemini-2.5") {
            for name in ["effort", "thinking_budget"] {
                plan.drop(name, format!("{model} does not think"));
            }
            if thinking == Some(true) {
                plan.drop("thinking", format!("{model} does not think"));
            }
            return;
        }

        let (lowest, highest, can_disable) = budget_range(&model);
        let mut budget = budget;
        if budget.is_none() {
            if let Some(e) = effort {
                let b = effort_budget(e);
                plan.values.shift_remove("effort");
                plan.adjusted.insert("effort".into(), format!("'{e}' → thinkingBudget {b}"));
                budget = Some(b);
            } else if thinking == Some(false) {
                budget = Some(0);
            }
            if let Some(b) = budget {
                plan.set("thinking_budget", json!(b));
            }
        } else if effort.is_some() {
            plan.drop("effort", "thinking_budget was given; Gemini takes one or the other");
        }
        let Some(b) = budget.filter(|b| *b != -1) else { return }; // -1: let the model decide
        if b == 0 && !can_disable {
            plan.adjust("thinking_budget", json!(lowest), &format!("{model} cannot switch thinking off; {lowest} is its minimum"));
        } else if b != 0 && !(lowest..=highest).contains(&b) {
            plan.adjust("thinking_budget", json!(b.clamp(lowest, highest)), &format!("{model} takes a thinking budget of {lowest}–{highest}"));
        }
    }

    fn payload(&self, req: &CompletionRequest) -> Option<Value> {
        Some(self.build_payload(req))
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<ModelResponse> {
        let started = Instant::now();
        let raw = self.transport.post(&Self::model_path(&req.model, false), &self.build_payload(req), &[], req.timeout).await?;
        self.decode(raw, req, started.elapsed().as_secs_f64() * 1000.0)
    }

    fn stream<'a>(&'a self, req: &'a CompletionRequest) -> BoxStream<'a, Result<StreamEvent>> {
        self.transport.retrying(move || {
            Box::pin(async_stream::try_stream! {
                let payload = self.build_payload(req);
                let path = Self::model_path(&req.model, true);
                let params = vec![("alt".to_string(), "sse".to_string())];
                let mut blocks: Vec<ContentBlock> = Vec::new();
                let mut usage = Usage { calls: 1, ..Default::default() };
                let mut stop = StopReason::EndTurn;
                let mut model = req.model.clone();
                let mut events = sse_events(self.transport.stream_lines(&path, &payload, &params, req.timeout));
                while let Some(item) = events.next().await {
                    let (_, chunk) = item?;
                    if chunk.is_empty() {
                        continue;
                    }
                    let mut data: Value = serde_json::from_str(&chunk).map_err(|e| Error::config(format!("bad stream chunk: {e}")))?;
                    if let Value::Array(items) = data {
                        // A JSON array rather than SSE framing.
                        data = items.into_iter().next().unwrap_or_default();
                    }
                    if !data["error"].is_null() {
                        Err(self.body_error(&data["error"]))?;
                    }
                    if let Some(m) = data["modelVersion"].as_str().filter(|m| !m.is_empty()) {
                        model = m.into();
                    }
                    if data["usageMetadata"].is_object() {
                        usage = Self::usage(&data); // cumulative totals
                    }
                    for cand in data["candidates"].as_array().into_iter().flatten() {
                        for block in Self::decode_parts(cand["content"]["parts"].as_array().map(Vec::as_slice).unwrap_or(&[])) {
                            match &block {
                                ContentBlock::Text(t) => yield StreamEvent::text(t.text.clone()),
                                ContentBlock::Thinking(t) => yield StreamEvent { text: t.thinking.clone(), ..StreamEvent::new(StreamEventType::Thinking) },
                                ContentBlock::ToolUse(u) => yield StreamEvent::new(StreamEventType::ToolCall).with_data("id", json!(u.id)).with_data("name", json!(u.name)).with_data("input", Value::Object(u.input.clone())),
                                _ => {}
                            }
                            blocks.push(block);
                        }
                        if let Some(f) = cand["finishReason"].as_str() {
                            stop = stop_map(f);
                        }
                    }
                }
                if blocks.iter().any(|b| matches!(b, ContentBlock::ToolUse(_))) {
                    stop = StopReason::ToolUse;
                }
                let resp = self.finish(Message::assistant_blocks(merge_fragments(blocks)), stop, usage, &model, json!({}), 0.0);
                yield StreamEvent::new(StreamEventType::StepEnd).with_data("response", serde_json::to_value(resp).unwrap_or_default());
            })
        })
    }

    async fn embed(&self, texts: &[String], model: Option<&str>) -> Result<Vec<Vec<f32>>> {
        let name = model.unwrap_or(EMBEDDING_MODEL);
        let name = name.strip_prefix("models/").unwrap_or(name);
        let requests: Vec<Value> = texts.iter().map(|t| json!({"model": format!("models/{name}"), "content": {"parts": [{"text": t}]}})).collect();
        let raw = self.transport.post(&format!("/models/{name}:batchEmbedContents"), &json!({"requests": requests}), &[], None).await?;
        Ok(raw["embeddings"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|e| e["values"].as_array().map(|v| v.iter().filter_map(Value::as_f64).map(|x| x as f32).collect()).unwrap_or_default())
            .collect())
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();
        let mut params = vec![("pageSize".to_string(), "1000".to_string())];
        loop {
            let raw = self.transport.get("/models", &params, None).await?;
            models.extend(
                raw["models"].as_array().into_iter().flatten().filter_map(|m| m["name"].as_str()).map(|n| n.strip_prefix("models/").unwrap_or(n).to_string()),
            );
            match raw["nextPageToken"].as_str().filter(|t| !t.is_empty()) {
                Some(token) => params = vec![("pageSize".into(), "1000".into()), ("pageToken".into(), token.into())],
                None => return Ok(models),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolSchema;
    use crate::types::ToolResultBlock;

    fn p() -> GeminiProvider {
        GeminiProvider::new(Some("k".into()))
    }

    #[test]
    fn schema_cleaning() {
        let s = json!({"type": "object", "title": "X", "additionalProperties": false, "properties": {"a": {"type": "object", "default": 1}}});
        assert_eq!(clean_schema(&s), json!({"type": "object", "properties": {"a": {"type": "object", "properties": {}}}}));
    }

    #[test]
    fn encodes_function_calls_by_name() {
        let mut call = ToolUseBlock::new("add", json!({"a": 1}).as_object().unwrap().clone());
        call.thought_signature = Some("sig".into());
        let id = call.id.clone();
        let mut req = CompletionRequest::new(
            "gemini-2.5-flash",
            vec![
                Message::system("sys"),
                Message::user("add"),
                Message::assistant_blocks(vec![ContentBlock::ToolUse(call)]),
                Message::tool_results(vec![ToolResultBlock { tool_use_id: id, content: "2".into(), is_error: false }]),
            ],
        );
        req.tools = vec![ToolSchema { name: "add".into(), description: String::new(), parameters: json!({"type": "object"}) }];
        req.tool_choice = Some(json!({"name": "add"}));
        let body = p().build_payload(&req);
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "sys");
        assert_eq!(body["contents"][1]["parts"][0]["thoughtSignature"], "sig");
        assert_eq!(body["contents"][2]["parts"][0]["functionResponse"]["name"], "add");
        assert_eq!(body["toolConfig"]["functionCallingConfig"], json!({"mode": "ANY", "allowedFunctionNames": ["add"]}));
        assert_eq!(body["tools"][0]["functionDeclarations"][0]["parameters"], json!({"type": "object", "properties": {}}));
    }

    #[test]
    fn thinking_plans() {
        let mut r = CompletionRequest::new("gemini-2.5-pro", vec![]);
        r.effort = Some(Effort::None);
        let plan = p().plan_parameters(&r);
        assert_eq!(plan.get("thinking_budget"), Some(&json!(128)));
        let mut r3 = CompletionRequest::new("gemini-3-pro-preview", vec![]);
        r3.effort = Some(Effort::Medium);
        let body = p().build_payload(&r3);
        assert_eq!(body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "low");
        let mut old = CompletionRequest::new("gemini-2.0-flash", vec![]);
        old.effort = Some(Effort::High);
        assert_eq!(p().plan_parameters(&old).dropped["effort"], "gemini-2.0-flash does not think");
    }

    #[test]
    fn decodes_and_merges() {
        let raw = json!({"modelVersion": "gemini-2.5-flash", "candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": "hmm", "thought": true}, {"text": "Hi"}, {"thoughtSignature": "s1"}, {"functionCall": {"name": "add", "args": {"a": 1}}}
        ]}}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3}});
        let r = p().decode(raw, &CompletionRequest::new("x", vec![]), 0.0).unwrap();
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(r.text(), "Hi");
        match &r.message.content[1] {
            ContentBlock::Text(t) => assert_eq!(t.thought_signature.as_deref(), Some("s1")),
            other => panic!("{other:?}"),
        }
        let merged = merge_fragments(vec![ContentBlock::text("a"), ContentBlock::text("b")]);
        assert_eq!(merged, vec![ContentBlock::text("ab")]);
        let err = p().decode(json!({"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}}), &CompletionRequest::new("x", vec![]), 0.0).unwrap_err();
        assert_eq!(err.as_provider().unwrap().status, Some(429));
    }
}
