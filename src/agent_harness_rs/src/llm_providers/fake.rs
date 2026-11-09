//! A scripted provider. No network, no key — the way you test an agent.
//!
//! ```
//! use agent_harness::llm_providers::fake::{FakeProvider, Script, tool_call};
//! use serde_json::json;
//!
//! let provider = FakeProvider::new(vec![
//!     tool_call("add", json!({"a": 1, "b": 2})),
//!     Script::text("3"),
//! ]);
//! ```

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use blake2::Blake2b;
use blake2::digest::Digest;
use blake2::digest::consts::U8;
use futures::stream::BoxStream;
use serde_json::{Value, json};

use super::base::{CompletionRequest, Provider, ProviderSpec, events_from};
use crate::errors::{Error, Result};
use crate::types::{ContentBlock, MediaKind, Message, ModelResponse, StopReason, StreamEvent, StreamEventType, ToolUseBlock, Usage};

/// One scripted reply.
#[derive(Clone)]
pub enum Script {
    Text(String),
    Message(Message),
    Response(ModelResponse),
    ToolUse(ToolUseBlock),
    Blocks(Vec<ContentBlock>),
    Error(Error),
    /// Computed from the request when it arrives.
    Func(Arc<dyn Fn(&CompletionRequest) -> Script + Send + Sync>),
}

impl Script {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    pub fn func(f: impl Fn(&CompletionRequest) -> Script + Send + Sync + 'static) -> Self {
        Self::Func(Arc::new(f))
    }
}

impl From<&str> for Script {
    fn from(s: &str) -> Self {
        Self::Text(s.into())
    }
}

impl From<String> for Script {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

/// Sugar for scripting a tool call into a FakeProvider timeline.
pub fn tool_call(name: &str, args: Value) -> Script {
    Script::ToolUse(ToolUseBlock::new(name, args.as_object().cloned().unwrap_or_default()))
}

pub const SPEC: ProviderSpec = ProviderSpec {
    name: "fake",
    display_name: "Fake (scripted)",
    description: "A scripted provider for tests: no network, no key, no cost.",
    docs_url: "",
    auth_type: "none",
    env_key: "",
    default_model: "fake-1",
    base_url: "http://fake.invalid",
    capabilities: &["streaming", "tools", "embeddings"],
    parameters: &super::parameters::SAMPLING_PARAMETERS,
    modalities: &[MediaKind::Image, MediaKind::Audio, MediaKind::Video, MediaKind::Document],
    fields: &[],
};

struct State {
    responses: Vec<Script>,
    cursor: usize,
    requests: Vec<CompletionRequest>,
}

/// Replies from a script, in order; `default` once it runs out (or loops).
pub struct FakeProvider {
    state: Mutex<State>,
    pub default: String,
    pub looping: bool,
    /// Stream a reply a word at a time, `stream_delay` apart.
    pub stream_words: bool,
    pub stream_delay: Duration,
}

impl FakeProvider {
    pub fn new(responses: Vec<Script>) -> Self {
        Self {
            state: Mutex::new(State { responses, cursor: 0, requests: vec![] }),
            default: "done".into(),
            looping: false,
            stream_words: false,
            stream_delay: Duration::ZERO,
        }
    }

    pub fn with_default(mut self, text: &str) -> Self {
        self.default = text.into();
        self
    }

    pub fn looping(mut self) -> Self {
        self.looping = true;
        self
    }

    pub fn streaming_words(mut self, delay: Duration) -> Self {
        self.stream_words = true;
        self.stream_delay = delay;
        self
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn queue(&self, more: Vec<Script>) {
        self.lock().responses.extend(more);
    }

    /// Every request this provider has received, oldest first.
    pub fn requests(&self) -> Vec<CompletionRequest> {
        self.lock().requests.clone()
    }

    /// How many scripted replies have been used.
    pub fn cursor(&self) -> usize {
        self.lock().cursor
    }

    fn next(&self, req: &CompletionRequest) -> Script {
        let item = {
            let mut st = self.lock();
            st.requests.push(req.clone());
            if st.cursor >= st.responses.len() {
                if self.looping && !st.responses.is_empty() {
                    st.cursor = 0;
                } else {
                    return Script::Text(self.default.clone());
                }
            }
            let item = st.responses[st.cursor].clone();
            st.cursor += 1;
            item
        };
        match item {
            Script::Func(f) => f(req),
            other => other,
        }
    }
}

#[async_trait]
impl Provider for FakeProvider {
    fn spec(&self) -> &ProviderSpec {
        &SPEC
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<ModelResponse> {
        let message = match self.next(req) {
            Script::Response(r) => return Ok(r),
            Script::Error(e) => return Err(e),
            Script::Message(m) => m,
            Script::ToolUse(t) => Message::assistant_blocks(vec![ContentBlock::ToolUse(t)]),
            Script::Blocks(b) => Message::assistant_blocks(b),
            Script::Text(t) => Message::assistant(t),
            Script::Func(_) => return Err(Error::config("FakeProvider: a scripted function returned another function")),
        };
        let stop = if message.tool_uses().is_empty() { StopReason::EndTurn } else { StopReason::ToolUse };
        let words: usize = message
            .content
            .iter()
            .map(|b| match b {
                ContentBlock::Text(t) => t.text.split_whitespace().count(),
                _ => 0,
            })
            .sum();
        let usage = Usage {
            input_tokens: req.messages.iter().map(|m| m.text().chars().count() as u64).sum::<u64>() / 4,
            output_tokens: words.max(1) as u64,
            calls: 1,
            ..Default::default()
        };
        let model = if req.model.is_empty() { SPEC.default_model } else { &req.model };
        Ok(self.finish(message, stop, usage, model, json!({}), 0.0))
    }

    fn stream<'a>(&'a self, req: &'a CompletionRequest) -> BoxStream<'a, Result<StreamEvent>> {
        Box::pin(async_stream::try_stream! {
            let resp = self.complete(req).await?;
            if !self.stream_words {
                for e in events_from(&resp) {
                    yield e;
                }
                return;
            }
            let text = resp.text();
            let words: Vec<&str> = if text.is_empty() { vec![] } else { text.split(' ').collect() };
            for (i, w) in words.iter().enumerate() {
                if !self.stream_delay.is_zero() {
                    tokio::time::sleep(self.stream_delay).await;
                }
                let sep = if i < words.len() - 1 { " " } else { "" };
                yield StreamEvent::text(format!("{w}{sep}"));
            }
            for e in events_from(&resp).into_iter().filter(|e| e.kind != StreamEventType::Text) {
                yield e;
            }
        })
    }

    async fn embed(&self, texts: &[String], _model: Option<&str>) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| hash_embedding(t, 256)).collect())
    }
}

/// A deterministic bag-of-words embedding with no model behind it.
///
/// Free, offline and stable; identical to the Python `hash_embedding`, so
/// vectors stored by either language compare.
pub fn hash_embedding(text: &str, dims: usize) -> Vec<f32> {
    let mut vec = vec![0f64; dims];
    let cleaned: String = text.chars().map(|c| if c.is_alphanumeric() { c.to_lowercase().next().unwrap_or(c) } else { ' ' }).collect();
    for token in cleaned.split_whitespace() {
        let digest = Blake2b::<U8>::digest(token.as_bytes());
        let idx = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize % dims;
        let sign = if digest[4] % 2 == 1 { 1.0 } else { -1.0 };
        vec[idx] += sign;
    }
    let norm = vec.iter().map(|v| v * v).sum::<f64>().sqrt();
    vec.into_iter().map(|v| if norm > 0.0 { (v / norm) as f32 } else { v as f32 }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn req() -> CompletionRequest {
        CompletionRequest::new("", vec![Message::user("hello there")])
    }

    #[tokio::test]
    async fn scripted_replies_in_order() {
        let p = FakeProvider::new(vec!["one".into(), tool_call("add", json!({"a": 1}))]);
        let r1 = p.complete(&req()).await.unwrap();
        assert_eq!(r1.text(), "one");
        assert_eq!(r1.model, "fake-1");
        assert_eq!(r1.provider, "fake");
        let r2 = p.complete(&req()).await.unwrap();
        assert_eq!(r2.stop_reason, StopReason::ToolUse);
        assert_eq!(r2.tool_uses()[0].input["a"], 1);
        assert_eq!(p.complete(&req()).await.unwrap().text(), "done");
        assert_eq!(p.requests().len(), 3);
    }

    #[tokio::test]
    async fn functions_errors_and_looping() {
        let p = FakeProvider::new(vec![Script::func(|r| Script::Text(format!("saw {}", r.messages.len()))), Script::Error(Error::config("boom"))]).looping();
        assert_eq!(p.complete(&req()).await.unwrap().text(), "saw 1");
        assert!(p.complete(&req()).await.is_err());
        assert_eq!(p.complete(&req()).await.unwrap().text(), "saw 1");
    }

    #[tokio::test]
    async fn streams_words() {
        let p = FakeProvider::new(vec!["a b c".into()]).streaming_words(Duration::ZERO);
        let r = req();
        let events: Vec<StreamEvent> = p.stream(&r).map(|e| e.unwrap()).collect().await;
        let texts: Vec<&str> = events.iter().filter(|e| e.kind == StreamEventType::Text).map(|e| e.text.as_str()).collect();
        assert_eq!(texts, vec!["a ", "b ", "c"]);
        assert_eq!(events.last().unwrap().kind, StreamEventType::StepEnd);
    }

    #[test]
    fn hash_embedding_matches_python() {
        // Reference values computed with the Python implementation.
        let v = hash_embedding("Hello, world agent harness", 256);
        let nonzero: Vec<(usize, f32)> = v.iter().copied().enumerate().filter(|(_, x)| *x != 0.0).collect();
        assert_eq!(nonzero, vec![(127, 0.5), (146, -0.5), (168, 0.5), (193, -0.5)]);
        assert_eq!(v.len(), 256);
        let norm: f32 = v.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-5);
        assert_eq!(hash_embedding("", 8), vec![0.0; 8]);
        assert_eq!(hash_embedding("hello world", 256), hash_embedding("HELLO   world!", 256));
    }
}
