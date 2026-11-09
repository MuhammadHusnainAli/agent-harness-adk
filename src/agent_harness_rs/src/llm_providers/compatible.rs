//! Every vendor that speaks the OpenAI Chat Completions format, preconfigured.
//!
//! Each preset is an [`OpenAIFlavor`]: where the API lives, which environment
//! variable holds the key, and any parameter the server spells differently.
//! The encoding, streaming, retries and typed errors are all [`OpenAIProvider`]'s.
//!
//! ```no_run
//! use agent_harness::llm_providers::compatible::preset_provider;
//! use agent_harness::llm_providers::TransportConfig;
//!
//! let groq = preset_provider("groq", None, TransportConfig::default()).unwrap();
//! ```
//!
//! Local servers (Ollama, LM Studio, vLLM) run without a key; the rest need one.

use reqwest::header::HeaderValue;

use super::base::{ProviderField, ProviderSpec, TransportConfig};
use super::openai::{EffortStyle, OpenAIFlavor, OpenAIProvider};
use super::parameters::{Effort, SAMPLING_PARAMETERS};
use crate::errors::{Error, Result};
use crate::types::MediaKind;

const CAPS: &[&str] = &["streaming", "tools", "json_schema", "list_models"];
const OPENAI_LIKE: &[&str] = &["temperature", "top_p", "frequency_penalty", "presence_penalty", "seed"];
/// Open-weight reasoning models that take `reasoning_effort`, wherever they're hosted.
const OPEN_REASONING: &[(&str, &[Effort])] = &[("gpt-oss", &[Effort::Low, Effort::Medium, Effort::High])];

const fn key(env: &'static [&'static str], example: Option<&'static str>, required: bool, description: &'static str) -> ProviderField {
    ProviderField { required, env, example, ..ProviderField::new("api_key", "secret", description) }
}

const fn url(env: &'static [&'static str], required: bool) -> ProviderField {
    ProviderField { required, env, ..ProviderField::new("base_url", "url", "Where the OpenAI-compatible API lives.") }
}

/// Any server that speaks the OpenAI format — a gateway, a proxy, your own.
/// Needs a base URL, and a key only if the server asks for one.
pub const OPENAI_COMPATIBLE: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "openai-compatible",
        display_name: "OpenAI-compatible",
        description: "Any endpoint that speaks the OpenAI Chat Completions format: gateways, proxies, self-hosted servers.",
        docs_url: "https://platform.openai.com/docs/api-reference/chat",
        auth_type: "api_key",
        env_key: "OPENAI_COMPATIBLE_API_KEY",
        default_model: "",
        base_url: "",
        capabilities: CAPS,
        parameters: &SAMPLING_PARAMETERS,
        modalities: &[MediaKind::Image],
        fields: &[
            url(&["OPENAI_COMPATIBLE_BASE_URL"], true),
            key(&["OPENAI_COMPATIBLE_API_KEY"], None, false, "Sent as a bearer token, if the server wants one."),
        ],
    },
    key_required: false,
    stream_usage: true,
    reasoning_prefixes: &["o1", "o3", "o4", "gpt-5"],
    effort_style: EffortStyle::ReasoningEffort,
    effort_families: OPEN_REASONING,
    drop_params: &[],
    rename_params: &[],
    embedding_model: "",
    base_url_env: "OPENAI_COMPATIBLE_BASE_URL",
    strict_reasoning_models: &[],
};

const C: OpenAIFlavor = OPENAI_COMPATIBLE;

pub const OPENROUTER: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "openrouter",
        display_name: "OpenRouter",
        description: "Hundreds of models from every major lab behind one key.",
        docs_url: "https://openrouter.ai/docs",
        env_key: "OPENROUTER_API_KEY",
        default_model: "anthropic/claude-sonnet-5",
        base_url: "https://openrouter.ai/api/v1",
        capabilities: &["streaming", "tools", "json_schema", "list_models", "vision", "thinking"],
        fields: &[
            key(&["OPENROUTER_API_KEY"], Some("sk-or-..."), true, "The API key ($OPENROUTER_API_KEY)."),
            ProviderField::new("app_url", "str", "Your site, sent as HTTP-Referer for OpenRouter's rankings."),
            ProviderField::new("app_name", "str", "Your app's name, sent as X-Title."),
        ],
        ..C.spec
    },
    key_required: true,
    base_url_env: "",
    effort_style: EffortStyle::OpenRouter,
    reasoning_prefixes: &[],
    ..C
};

pub const GROQ: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "groq",
        display_name: "Groq",
        description: "Open-weight models on Groq's LPUs — very low latency.",
        docs_url: "https://console.groq.com/docs/api-reference",
        env_key: "GROQ_API_KEY",
        default_model: "llama-3.3-70b-versatile",
        base_url: "https://api.groq.com/openai/v1",
        parameters: OPENAI_LIKE,
        fields: &[key(&["GROQ_API_KEY"], Some("gsk_..."), true, "The API key ($GROQ_API_KEY).")],
        ..C.spec
    },
    key_required: true,
    base_url_env: "",
    ..C
};

pub const TOGETHER: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "together",
        display_name: "Together AI",
        description: "Open-weight models hosted by Together AI.",
        docs_url: "https://docs.together.ai/reference/chat-completions-1",
        env_key: "TOGETHER_API_KEY",
        default_model: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        base_url: "https://api.together.xyz/v1",
        capabilities: &["streaming", "tools", "json_schema", "list_models", "embeddings"],
        fields: &[key(&["TOGETHER_API_KEY"], None, true, "The API key ($TOGETHER_API_KEY).")],
        ..C.spec
    },
    key_required: true,
    base_url_env: "",
    ..C
};

pub const DEEPSEEK: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "deepseek",
        display_name: "DeepSeek",
        description: "DeepSeek's chat and reasoner models.",
        docs_url: "https://api-docs.deepseek.com/",
        env_key: "DEEPSEEK_API_KEY",
        default_model: "deepseek-chat",
        base_url: "https://api.deepseek.com/v1",
        parameters: &["temperature", "top_p", "frequency_penalty", "presence_penalty"],
        capabilities: &["streaming", "tools", "json_schema", "list_models", "thinking"],
        fields: &[key(&["DEEPSEEK_API_KEY"], Some("sk-..."), true, "The API key ($DEEPSEEK_API_KEY).")],
        ..C.spec
    },
    key_required: true,
    base_url_env: "",
    // deepseek-reasoner always reasons; there is no level to set.
    effort_families: &[],
    ..C
};

pub const MISTRAL: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "mistral",
        display_name: "Mistral AI",
        description: "Mistral's hosted models (La Plateforme).",
        docs_url: "https://docs.mistral.ai/api/",
        env_key: "MISTRAL_API_KEY",
        default_model: "mistral-large-latest",
        base_url: "https://api.mistral.ai/v1",
        capabilities: &["streaming", "tools", "json_schema", "list_models", "vision", "embeddings"],
        parameters: OPENAI_LIKE,
        fields: &[key(&["MISTRAL_API_KEY"], None, true, "The API key ($MISTRAL_API_KEY).")],
        ..C.spec
    },
    key_required: true,
    base_url_env: "",
    embedding_model: "mistral-embed",
    effort_families: &[],
    // Mistral names the seed differently and rejects fields it does not know.
    rename_params: &[("seed", "random_seed")],
    drop_params: &["user", "stream_options"],
    stream_usage: false,
    ..C
};

pub const XAI: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "xai",
        display_name: "xAI",
        description: "Grok models from xAI.",
        docs_url: "https://docs.x.ai/docs/api-reference",
        env_key: "XAI_API_KEY",
        default_model: "grok-4",
        base_url: "https://api.x.ai/v1",
        capabilities: &["streaming", "tools", "json_schema", "list_models", "vision", "thinking"],
        parameters: OPENAI_LIKE,
        fields: &[key(&["XAI_API_KEY"], Some("xai-..."), true, "The API key ($XAI_API_KEY).")],
        ..C.spec
    },
    key_required: true,
    base_url_env: "",
    effort_families: &[("grok-3-mini", &[Effort::Low, Effort::High])],
    // Grok's reasoning models reject penalties and stop sequences.
    strict_reasoning_models: &["grok-4", "grok-3-mini", "grok-code"],
    ..C
};

pub const FIREWORKS: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "fireworks",
        display_name: "Fireworks AI",
        description: "Open-weight models hosted by Fireworks AI.",
        docs_url: "https://docs.fireworks.ai/api-reference",
        env_key: "FIREWORKS_API_KEY",
        default_model: "accounts/fireworks/models/llama-v3p3-70b-instruct",
        base_url: "https://api.fireworks.ai/inference/v1",
        fields: &[key(&["FIREWORKS_API_KEY"], Some("fw_..."), true, "The API key ($FIREWORKS_API_KEY).")],
        ..C.spec
    },
    key_required: true,
    base_url_env: "",
    ..C
};

pub const CEREBRAS: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "cerebras",
        display_name: "Cerebras",
        description: "Open-weight models on Cerebras wafer-scale hardware.",
        docs_url: "https://inference-docs.cerebras.ai/api-reference",
        env_key: "CEREBRAS_API_KEY",
        default_model: "llama-3.3-70b",
        base_url: "https://api.cerebras.ai/v1",
        parameters: &["temperature", "top_p", "seed"],
        fields: &[key(&["CEREBRAS_API_KEY"], Some("csk-..."), true, "The API key ($CEREBRAS_API_KEY).")],
        ..C.spec
    },
    key_required: true,
    base_url_env: "",
    ..C
};

pub const OLLAMA: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "ollama",
        display_name: "Ollama",
        description: "Models running locally under Ollama. No key needed.",
        docs_url: "https://github.com/ollama/ollama/blob/main/docs/openai.md",
        env_key: "OLLAMA_API_KEY",
        default_model: "llama3.2",
        base_url: "http://localhost:11434/v1",
        capabilities: &["streaming", "tools", "json_schema", "list_models", "vision", "embeddings"],
        parameters: OPENAI_LIKE,
        fields: &[url(&["OLLAMA_BASE_URL"], false), key(&["OLLAMA_API_KEY"], None, false, "Only for an Ollama behind an authenticating proxy.")],
        ..C.spec
    },
    base_url_env: "OLLAMA_BASE_URL",
    embedding_model: "nomic-embed-text",
    ..C
};

pub const LMSTUDIO: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "lmstudio",
        display_name: "LM Studio",
        description: "Models served by LM Studio's local server. No key needed.",
        docs_url: "https://lmstudio.ai/docs/app/api/endpoints/openai",
        env_key: "LMSTUDIO_API_KEY",
        base_url: "http://localhost:1234/v1",
        capabilities: &["streaming", "tools", "json_schema", "list_models", "embeddings"],
        parameters: &["temperature", "top_p", "frequency_penalty", "presence_penalty", "seed", "top_k"],
        fields: &[url(&["LMSTUDIO_BASE_URL"], false)],
        ..C.spec
    },
    base_url_env: "LMSTUDIO_BASE_URL",
    ..C
};

pub const VLLM: OpenAIFlavor = OpenAIFlavor {
    spec: ProviderSpec {
        name: "vllm",
        display_name: "vLLM",
        description: "A self-hosted vLLM server's OpenAI-compatible API.",
        docs_url: "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html",
        env_key: "VLLM_API_KEY",
        base_url: "",
        capabilities: &["streaming", "tools", "json_schema", "list_models", "thinking", "embeddings"],
        fields: &[url(&["VLLM_BASE_URL"], true), key(&["VLLM_API_KEY"], None, false, "Only if the server was started with --api-key.")],
        ..C.spec
    },
    base_url_env: "VLLM_BASE_URL",
    ..C
};

/// Every OpenAI-compatible preset, by name.
pub const COMPATIBLE_PROVIDERS: [OpenAIFlavor; 12] =
    [OPENAI_COMPATIBLE, OPENROUTER, GROQ, TOGETHER, DEEPSEEK, MISTRAL, XAI, FIREWORKS, CEREBRAS, OLLAMA, LMSTUDIO, VLLM];

/// The preset called `name`, if there is one.
pub fn preset(name: &str) -> Option<OpenAIFlavor> {
    COMPATIBLE_PROVIDERS.iter().find(|f| f.spec.name == name).cloned()
}

/// A provider for a preset. `config.base_url`, when set, overrides where it lives.
///
/// Ollama also honours `OLLAMA_HOST` (`host:port` or a URL).
pub fn preset_provider(name: &str, api_key: Option<String>, mut config: TransportConfig) -> Result<OpenAIProvider> {
    let flavor = preset(name).ok_or_else(|| Error::config(format!("no OpenAI-compatible preset named '{name}'")))?;
    if name == "ollama"
        && config.base_url.is_empty()
        && std::env::var("OLLAMA_BASE_URL").map_or(true, |v| v.is_empty())
        && let Ok(host) = std::env::var("OLLAMA_HOST").map(|h| h.trim().to_string())
        && !host.is_empty()
    {
        let host = if host.starts_with("http") { host } else { format!("http://{host}") };
        config.base_url = format!("{}/v1", host.trim_end_matches('/'));
    }
    OpenAIProvider::with_flavor(flavor, api_key, config)
}

/// OpenRouter, with the optional ranking headers it reads.
pub fn openrouter(api_key: Option<String>, app_url: Option<&str>, app_name: Option<&str>, mut config: TransportConfig) -> Result<OpenAIProvider> {
    if let Some(u) = app_url.and_then(|u| HeaderValue::from_str(u).ok()) {
        config.headers.entry("HTTP-Referer").or_insert(u);
    }
    if let Some(n) = app_name.and_then(|n| HeaderValue::from_str(n).ok()) {
        config.headers.entry("X-Title").or_insert(n);
    }
    OpenAIProvider::with_flavor(OPENROUTER, api_key, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_providers::base::{CompletionRequest, Provider};
    use crate::types::Message;
    use serde_json::json;

    fn cfg(url: &str) -> TransportConfig {
        TransportConfig { base_url: url.into(), ..Default::default() }
    }

    #[test]
    fn presets_exist_and_need_urls() {
        assert_eq!(COMPATIBLE_PROVIDERS.len(), 12);
        assert!(preset("groq").is_some());
        assert!(preset("nope").is_none());
        let err = OpenAIProvider::with_flavor(VLLM, None, TransportConfig::default());
        if std::env::var("VLLM_BASE_URL").is_err() {
            assert!(err.is_err());
        }
        let p = preset_provider("vllm", None, cfg("http://gpu:8000/v1/")).unwrap();
        assert_eq!(p.transport.base_url, "http://gpu:8000/v1");
    }

    #[test]
    fn mistral_renames_and_drops() {
        let p = preset_provider("mistral", Some("k".into()), TransportConfig::default()).unwrap();
        let mut req = CompletionRequest::new("mistral-large-latest", vec![Message::user("x")]);
        req.seed = Some(7);
        req.user = Some("u".into());
        let body = p.build_payload(&req, true);
        assert_eq!(body["random_seed"], 7);
        assert!(body.get("seed").is_none());
        assert!(body.get("user").is_none());
        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn openrouter_reasoning_and_headers() {
        let p = openrouter(Some("k".into()), Some("https://me.dev"), Some("me"), TransportConfig::default()).unwrap();
        assert_eq!(p.transport.extra_headers["x-title"], "me");
        let mut req = CompletionRequest::new("anthropic/claude-sonnet-5", vec![]);
        req.effort = Some(Effort::Max);
        assert_eq!(p.build_payload(&req, false)["reasoning"], json!({"effort": "high"}));
        let mut off = CompletionRequest::new("x", vec![]);
        off.thinking = Some(false);
        assert_eq!(p.build_payload(&off, false)["reasoning"], json!({"enabled": false}));
    }

    #[test]
    fn xai_strict_reasoning_and_gpt_oss() {
        let p = preset_provider("xai", Some("k".into()), TransportConfig::default()).unwrap();
        let mut req = CompletionRequest::new("grok-4", vec![]);
        req.presence_penalty = Some(0.5);
        assert!(p.plan_parameters(&req).dropped.contains_key("presence_penalty"));
        let groq = preset_provider("groq", Some("k".into()), TransportConfig::default()).unwrap();
        let mut oss = CompletionRequest::new("openai/gpt-oss-120b", vec![]);
        oss.effort = Some(Effort::Xhigh);
        assert_eq!(groq.build_payload(&oss, false)["reasoning_effort"], "high");
        assert_eq!(groq.plan_parameters(&CompletionRequest { top_k: Some(3), ..oss }).dropped["top_k"], "Groq has no top_k");
    }
}
