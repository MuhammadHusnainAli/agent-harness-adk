//! LLM providers: one contract, every major backend, resolved by name or model id.
//!
//! ```no_run
//! use agent_harness::llm_providers::{check_llm_provider, describe_llm_provider, list_llm_providers};
//! use serde_json::json;
//!
//! for p in list_llm_providers() {
//!     println!("{} configured={}", p.name, p.configured);
//! }
//! println!("{}", describe_llm_provider("bedrock").unwrap().render()); // every field, and where it's read
//! println!("{}", check_llm_provider("azure", &json!({})).unwrap().render()); // what's missing, offline
//! ```
//!
//! Every backend sends through the same [`base::HttpTransport`], so every one
//! gets the same retries, backoff, Retry-After handling, typed errors,
//! circuit breaker and stats.

pub mod anthropic;
pub mod azure;
pub mod base;
pub mod bedrock;
pub mod catalog;
pub mod compatible;
pub mod fake;
pub mod gemini;
pub mod openai;
pub mod parameters;
pub mod resilience;
pub mod sigv4;
pub mod vertex;

pub use crate::tools::ToolSchema;
pub use anthropic::AnthropicProvider;
pub use azure::{AzureCredential, AzureFoundryProvider, AzureOpenAIProvider, AzureOptions};
pub use base::{
    Auth, COMMON_FIELDS, CompletionRequest, HttpTransport, MODELS, ModelInfo, Provider, ProviderField, ProviderSpec, StaticAuth, Tier, TransportConfig,
    estimate_cost, model_info, normalise_model, provider_for_model, register_model,
};
pub use bedrock::{BedrockOptions, BedrockProvider};
pub use catalog::{
    ListOptions, ProviderCheck, ProviderInfo, check_llm_provider, close_all, describe_llm_provider, get_provider, get_provider_with, list_llm_providers,
    list_llm_providers_with, ping_llm_provider, register_provider, register_provider_with, resolve_provider,
};
pub use fake::{FakeProvider, Script, hash_embedding, tool_call};
pub use gemini::GeminiProvider;
pub use openai::{OpenAIFlavor, OpenAIProvider};
pub use parameters::{Effort, GENERATION_PARAMETERS, ParameterPlan, SAMPLING_PARAMETERS, nearest_effort, validate_parameters};
pub use resilience::{CircuitBreaker, ProviderStats, RetryEvent, RetryPolicy};
pub use sigv4::AwsCredentials;
pub use vertex::{GoogleAuth, GoogleCredentials, VertexGeminiProvider, VertexOptions, VertexProvider};

/// A local stand-in for a cloud API: records each request, answers from a script.
#[cfg(test)]
pub(crate) mod testing {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use serde_json::Value;

    use super::base::TransportConfig;
    use super::resilience::RetryPolicy;

    /// One request as the server saw it.
    #[derive(Debug, Clone)]
    pub struct Seen {
        /// The path and query.
        pub path: String,
        /// Lower-case names.
        pub headers: HashMap<String, String>,
        /// The body as JSON (`Null` if it is not).
        pub body: Value,
        pub raw: String,
    }

    type Reply = (u16, String, Vec<u8>);

    pub struct Fake {
        pub url: String,
        seen: Arc<Mutex<Vec<Seen>>>,
    }

    impl Fake {
        /// Answers each request with the next reply; the last one repeats.
        pub async fn start(replies: Vec<Reply>) -> Self {
            let seen: Arc<Mutex<Vec<Seen>>> = Arc::default();
            let record = seen.clone();
            let replies = Arc::new(replies);
            let app = axum::Router::new().fallback(move |req: axum::extract::Request| {
                let record = record.clone();
                let replies = replies.clone();
                async move {
                    let (parts, body) = req.into_parts();
                    let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap_or_default();
                    let raw = String::from_utf8_lossy(&bytes).to_string();
                    let n = {
                        let mut seen = record.lock().unwrap();
                        seen.push(Seen {
                            path: parts.uri.path_and_query().map(|p| p.to_string()).unwrap_or_default(),
                            headers: parts.headers.iter().map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string())).collect(),
                            body: serde_json::from_str(&raw).unwrap_or(Value::Null),
                            raw,
                        });
                        seen.len() - 1
                    };
                    let (status, ct, body) = replies[n.min(replies.len() - 1)].clone();
                    axum::response::Response::builder().status(status).header("content-type", ct).body(axum::body::Body::from(body)).unwrap()
                }
            });
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            tokio::spawn(async move {
                let _ = axum::serve(listener, app).await;
            });
            Self { url, seen }
        }

        pub async fn json(body: Value) -> Self {
            Self::start(vec![(200, "application/json".into(), body.to_string().into_bytes())]).await
        }

        pub async fn raw(body: Vec<u8>, content_type: &str) -> Self {
            Self::start(vec![(200, content_type.into(), body)]).await
        }

        /// Server-sent events, one `data:` line per value.
        pub async fn sse(events: &[Value]) -> Self {
            let body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
            Self::raw(body.into_bytes(), "text/event-stream").await
        }

        pub fn all(&self) -> Vec<Seen> {
            self.seen.lock().unwrap().clone()
        }

        pub fn last(&self) -> Seen {
            self.all().pop().expect("no request reached the fake")
        }
    }

    /// No retries, no circuit breaker, `base_url` as given.
    pub fn cfg(base_url: &str) -> TransportConfig {
        TransportConfig { base_url: base_url.into(), retry: RetryPolicy { max_retries: 0, ..Default::default() }, circuit_breaker: None, ..Default::default() }
    }
}
