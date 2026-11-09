//! Claude on Amazon Bedrock.
//!
//! Bedrock speaks the Anthropic Messages format, so the encoding and decoding
//! are shared with [`AnthropicProvider`] — what differs is the endpoint, the
//! fact that the model lives in the URL rather than the body, and SigV4
//! signing instead of an API key.
//!
//! ```no_run
//! use agent_harness::llm_providers::bedrock::{BedrockOptions, BedrockProvider};
//!
//! let bedrock = BedrockProvider::new(BedrockOptions { region: Some("eu-west-1".into()), ..Default::default() });
//! ```
//!
//! Credentials come from the usual places: explicit options, the standard
//! `AWS_*` environment variables, then the static keys of a profile in
//! `~/.aws/credentials` (`profile`, else `AWS_PROFILE`, else `default`). A
//! Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`) works too, and skips signing
//! altogether. Python also falls back on botocore when it is installed, which
//! covers SSO, `credential_process` and instance roles; those are not tried
//! here — pass the keys they produce.
//!
//! Every attempt is signed afresh, so a retry after a long back-off never goes
//! out with a stale timestamp, and credentials from the chain are resolved
//! again once if AWS says they have expired. Streaming decodes Bedrock's binary
//! event stream — CRC-checked — into the same events the direct API sends.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use base64::Engine as _;
use futures::StreamExt;
use futures::stream::BoxStream;
use reqwest::Method;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Value, json};

use super::anthropic::{AnthropicProvider, AnthropicStream};
use super::base::{CompletionRequest, HttpTransport, Provider, ProviderField, ProviderSpec, TransportConfig};
use super::parameters::ParameterPlan;
use super::resilience::classify;
use super::sigv4::{AwsCredentials, SignRequest, sign, uri_encode};
use crate::errors::{Error, ProviderError, ProviderErrorKind as K, Result};
use crate::types::{MediaKind, ModelResponse, StreamEvent};

/// Bedrock takes the API version in the body instead of a header.
pub const BEDROCK_VERSION: &str = "bedrock-2023-05-31";

/// Bedrock's stream exceptions, as the HTTP status they would have had.
fn exception_status(name: &str) -> u16 {
    match name {
        "throttlingException" => 429,
        "serviceUnavailableException" | "modelNotReadyException" => 503,
        "internalServerException" | "modelStreamErrorException" => 500,
        "modelTimeoutException" => 504,
        "validationException" => 400,
        "accessDeniedException" => 403,
        "resourceNotFoundException" => 404,
        _ => 500,
    }
}

const FIELDS: [ProviderField; 8] = [
    ProviderField {
        env: &["AWS_REGION", "AWS_DEFAULT_REGION"],
        example: Some("eu-west-1"),
        ..ProviderField::new("region", "str", "The AWS region the model is enabled in (default us-east-1).")
    },
    ProviderField { one_of: Some("aws"), env: &["AWS_ACCESS_KEY_ID"], ..ProviderField::new("access_key", "secret", "An IAM access key id (with secret_key).") },
    ProviderField { env: &["AWS_SECRET_ACCESS_KEY"], ..ProviderField::new("secret_key", "secret", "The secret for access_key.") },
    ProviderField { env: &["AWS_SESSION_TOKEN"], ..ProviderField::new("session_token", "secret", "For temporary credentials (STS, SSO).") },
    ProviderField {
        one_of: Some("aws"),
        env: &["AWS_PROFILE"],
        ..ProviderField::new("profile", "str", "A named profile from ~/.aws/credentials (its static keys).")
    },
    ProviderField {
        one_of: Some("aws"),
        env: &["AWS_BEARER_TOKEN_BEDROCK"],
        ..ProviderField::new("api_key", "secret", "A Bedrock API key, used instead of SigV4.")
    },
    ProviderField {
        one_of: Some("aws"),
        ..ProviderField::new("credentials", "object", "AwsCredentials (as a setting: {access_key, secret_key, session_token}).")
    },
    ProviderField::new("base_url", "url", "A VPC endpoint or other override."),
];

pub const SPEC: ProviderSpec = ProviderSpec {
    name: "bedrock",
    display_name: "Amazon Bedrock",
    description: "Claude on AWS Bedrock, SigV4-signed or with a Bedrock API key.",
    docs_url: "https://docs.aws.amazon.com/bedrock/latest/userguide/",
    auth_type: "sigv4 or Bedrock API key",
    env_key: "",
    default_model: "anthropic.claude-opus-5",
    base_url: "",
    capabilities: &["streaming", "tools", "vision", "thinking", "json_schema", "prompt_caching"],
    parameters: &["temperature", "top_p", "top_k"],
    modalities: &[MediaKind::Image, MediaKind::Document],
    fields: &FIELDS,
};

/// How to reach Bedrock. Everything is optional; see the module docs for the
/// order credentials are looked for in.
#[derive(Debug, Clone, Default)]
pub struct BedrockOptions {
    /// Else `AWS_REGION`, `AWS_DEFAULT_REGION`, then `us-east-1`.
    pub region: Option<String>,
    pub access_key: Option<String>,
    pub secret_key: Option<String>,
    pub session_token: Option<String>,
    /// A profile in `~/.aws/credentials`. Else `AWS_PROFILE`, then `default`.
    pub profile: Option<String>,
    /// Fixed credentials: used as they are, never re-resolved.
    pub credentials: Option<AwsCredentials>,
    /// A Bedrock API key. Only read from `AWS_BEARER_TOKEN_BEDROCK` when no
    /// SigV4 keys were given.
    pub api_key: Option<String>,
    /// A VPC endpoint or other override. `TransportConfig::base_url` wins.
    pub base_url: Option<String>,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// The static keys of `profile` in the shared credentials file
/// (`AWS_SHARED_CREDENTIALS_FILE`, else `~/.aws/credentials`).
pub fn profile_credentials(profile: &str) -> Option<AwsCredentials> {
    let path = env("AWS_SHARED_CREDENTIALS_FILE")
        .map(std::path::PathBuf::from)
        .or_else(|| env("HOME").or_else(|| env("USERPROFILE")).map(|h| std::path::Path::new(&h).join(".aws").join("credentials")))?;
    parse_profile(&std::fs::read_to_string(path).ok()?, profile)
}

/// One profile's keys out of a credentials file's text.
pub fn parse_profile(text: &str, profile: &str) -> Option<AwsCredentials> {
    let mut current = String::new();
    let mut values: HashMap<String, String> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            current = name.trim().trim_start_matches("profile ").trim().to_string();
            continue;
        }
        if current == profile
            && let Some((k, v)) = line.split_once('=')
        {
            values.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let creds = AwsCredentials::new(
        values.get("aws_access_key_id").cloned().unwrap_or_default(),
        values.get("aws_secret_access_key").cloned().unwrap_or_default(),
        values.get("aws_session_token").cloned().filter(|t| !t.is_empty()),
    );
    creds.is_set().then_some(creds)
}

/// Explicit values, then the environment, then a profile's static keys.
pub fn resolve_credentials(access_key: Option<&str>, secret_key: Option<&str>, token: Option<&str>, profile: Option<&str>) -> AwsCredentials {
    let creds = AwsCredentials::resolve(access_key, secret_key, token);
    if creds.is_set() {
        return creds;
    }
    let name = profile.map(str::to_string).or_else(|| env("AWS_PROFILE")).unwrap_or_else(|| "default".into());
    profile_credentials(&name).unwrap_or(creds)
}

/// AWS wants each path segment of a non-S3 request encoded twice in the
/// canonical request. [`sign`] decodes the path and encodes it once, so it is
/// handed the path with every `%` escaped one more time.
fn signing_url(url: &str) -> String {
    let Some(start) = url.find("://").map(|i| i + 3) else { return url.into() };
    let Some(slash) = url[start..].find('/').map(|i| start + i) else { return url.into() };
    let (head, rest) = url.split_at(slash);
    let (path, query) = rest.split_once('?').map_or((rest, None), |(p, q)| (p, Some(q)));
    let mut out = format!("{head}{}", path.replace('%', "%25"));
    if let Some(q) = query {
        out.push('?');
        out.push_str(q);
    }
    out
}

struct BedrockAuth {
    region: String,
    bearer: String,
    fixed: Option<AwsCredentials>,
    access_key: Option<String>,
    secret_key: Option<String>,
    session_token: Option<String>,
    profile: Option<String>,
    cached: Mutex<Option<AwsCredentials>>,
}

impl BedrockAuth {
    fn credentials(&self) -> AwsCredentials {
        if let Some(c) = &self.fixed {
            return c.clone();
        }
        let mut cached = self.cached.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(c) = cached.as_ref() {
            return c.clone();
        }
        let c = resolve_credentials(self.access_key.as_deref(), self.secret_key.as_deref(), self.session_token.as_deref(), self.profile.as_deref());
        if c.is_set() {
            *cached = Some(c.clone());
        }
        c
    }
}

#[async_trait]
impl super::base::Auth for BedrockAuth {
    async fn headers(&self, method: &Method, url: &str, body: &[u8]) -> Result<HeaderMap> {
        let streaming = url.split('?').next().unwrap_or(url).ends_with("/invoke-with-response-stream");
        let accept = if streaming { "application/vnd.amazon.eventstream" } else { "application/json" };
        let mut h = HeaderMap::new();
        h.insert("accept", HeaderValue::from_static(accept));
        if streaming {
            h.insert("x-amzn-bedrock-accept", HeaderValue::from_static("application/json"));
        }
        if !self.bearer.is_empty() {
            h.insert("authorization", HeaderValue::from_str(&format!("Bearer {}", self.bearer)).map_err(|e| Error::config(e.to_string()))?);
            return Ok(h);
        }
        let creds = self.credentials();
        if !creds.is_set() {
            return Err(ProviderError::new(
                K::Authentication,
                "Bedrock needs AWS credentials — set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, pass them in BedrockOptions, \
                 set AWS_BEARER_TOKEN_BEDROCK, or put keys for AWS_PROFILE in ~/.aws/credentials",
            )
            .provider(SPEC.name)
            .into());
        }
        let mut signed: Vec<(&str, &str)> = vec![("accept", accept)];
        if streaming {
            signed.push(("x-amzn-bedrock-accept", "application/json"));
        }
        let url = signing_url(url);
        let req = SignRequest {
            method: method.as_str(),
            url: &url,
            region: &self.region,
            service: "bedrock",
            body,
            headers: &signed,
            now: None,
            content_sha_header: true,
        };
        for (k, v) in sign(&req, &creds) {
            if k == "host" {
                continue; // reqwest sends the same one
            }
            if let (Ok(k), Ok(v)) = (HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(&v)) {
                h.insert(k, v);
            }
        }
        Ok(h)
    }

    /// Chain credentials (a rotated profile) expire; resolve them again once.
    async fn refresh(&self) -> bool {
        if self.fixed.is_some() || !self.bearer.is_empty() || self.access_key.is_some() || self.secret_key.is_some() {
            return false;
        }
        *self.cached.lock().unwrap_or_else(|p| p.into_inner()) = None;
        true
    }
}

/// Re-label an error the shared Anthropic code made, so it names this provider.
pub(crate) fn relabel(err: Error, name: &str) -> Error {
    match err {
        Error::Provider(mut p) => {
            if let Some(rest) = p.message.strip_prefix(&format!("{} ", p.provider)) {
                p.message = format!("{name} {rest}");
            }
            p.provider = name.into();
            Error::Provider(p)
        }
        other => other,
    }
}

/// Anthropic models served by Bedrock, signed with SigV4.
pub struct BedrockProvider {
    inner: AnthropicProvider,
    /// The region requests are signed for.
    pub region: String,
}

impl BedrockProvider {
    pub fn new(options: BedrockOptions) -> Self {
        Self::with_config(options, TransportConfig::default())
    }

    pub fn with_config(options: BedrockOptions, mut config: TransportConfig) -> Self {
        let region = options.region.clone().filter(|r| !r.is_empty()).unwrap_or_else(super::sigv4::default_region);
        if config.base_url.is_empty() {
            config.base_url = options.base_url.clone().filter(|u| !u.is_empty()).unwrap_or_else(|| format!("https://bedrock-runtime.{region}.amazonaws.com"));
        }
        let explicit = options.access_key.is_some() || options.secret_key.is_some() || options.credentials.is_some();
        let bearer =
            options.api_key.clone().filter(|k| !k.is_empty()).or_else(|| if explicit { None } else { env("AWS_BEARER_TOKEN_BEDROCK") }).unwrap_or_default();
        let auth = BedrockAuth {
            region: region.clone(),
            bearer,
            fixed: options.credentials,
            access_key: options.access_key,
            secret_key: options.secret_key,
            session_token: options.session_token,
            profile: options.profile,
            cached: Mutex::new(None),
        };
        Self { inner: AnthropicProvider { transport: HttpTransport::new(SPEC.name, config, Arc::new(auth)) }, region }
    }

    /// The Anthropic body, minus the model (it is in the URL) and `stream`,
    /// plus the API version (there is no header to put it in).
    pub fn build_payload(&self, req: &CompletionRequest, stream: bool) -> Value {
        let mut payload = self.inner.build_payload(req, stream);
        if let Some(p) = payload.as_object_mut() {
            p.shift_remove("model");
            p.shift_remove("stream");
            p.insert("anthropic_version".into(), json!(BEDROCK_VERSION));
        }
        payload
    }

    /// `/model/{id}/invoke`. ARNs and versioned ids carry `:` and `/`, which are escaped.
    pub fn model_path(model: &str, stream: bool) -> String {
        let suffix = if stream { "invoke-with-response-stream" } else { "invoke" };
        format!("/model/{}/{suffix}", uri_encode(model, false))
    }

    fn stream_error(&self, error: &Value) -> Error {
        relabel(self.inner.stream_error(error), SPEC.name)
    }

    fn decode(&self, raw: Value, req: &CompletionRequest, latency_ms: f64) -> Result<ModelResponse> {
        if raw["type"] == "error" {
            return Err(self.stream_error(&raw["error"]));
        }
        let mut resp = self.inner.decode(raw, req, latency_ms)?;
        resp.provider = SPEC.name.into();
        Ok(resp)
    }

    /// One event-stream message as an Anthropic stream event, if it is one.
    fn frame_event(&self, frame: &EventFrame) -> Result<Option<Value>> {
        let kind = frame.headers.get(":message-type").map_or("event", String::as_str);
        if kind == "exception" || kind == "error" {
            let name = frame.headers.get(":exception-type").or_else(|| frame.headers.get(":error-code")).map_or("", String::as_str);
            let body = String::from_utf8_lossy(&frame.payload);
            return Err(classify(SPEC.name, exception_status(name), None, Some(&body), Some(&self.inner.transport.retry)).into());
        }
        if frame.headers.get(":event-type").map(String::as_str) != Some("chunk") {
            return Ok(None);
        }
        let envelope: Value = if frame.payload.is_empty() {
            json!({})
        } else {
            serde_json::from_slice(&frame.payload).map_err(|e| Error::config(format!("bad bedrock chunk: {e}")))?
        };
        let Some(b64) = envelope["bytes"].as_str() else { return Ok(None) };
        let bytes = base64::engine::general_purpose::STANDARD.decode(b64).map_err(|e| Error::config(format!("bad bedrock chunk: {e}")))?;
        Ok(Some(serde_json::from_slice(&bytes).map_err(|e| Error::config(format!("bad bedrock chunk: {e}")))?))
    }
}

fn dropped(e: &reqwest::Error) -> Error {
    let (kind, what) = if e.is_timeout() { (K::Timeout, "stalled") } else { (K::Connection, "dropped") };
    ProviderError::new(kind, format!("bedrock stream {what}: {e}")).provider(SPEC.name).into()
}

#[async_trait]
impl Provider for BedrockProvider {
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
        self.decode(raw, req, started.elapsed().as_secs_f64() * 1000.0)
    }

    fn stream<'a>(&'a self, req: &'a CompletionRequest) -> BoxStream<'a, Result<StreamEvent>> {
        self.inner.transport.retrying(move || {
            Box::pin(async_stream::try_stream! {
                let payload = self.build_payload(req, true);
                let path = Self::model_path(&req.model, true);
                let err = |e: &Value| self.stream_error(e);
                let mut state = AnthropicStream::new(self, &err, &req.model);
                let opened = self.inner.transport.open_stream(&path, &payload, &[], req.timeout).await?;
                let mut bytes = opened.response.bytes_stream();
                let mut decoder = EventStreamDecoder::default();
                while let Some(chunk) = bytes.next().await {
                    let chunk = chunk.map_err(|e| dropped(&e))?;
                    for frame in decoder.push(&chunk)? {
                        if let Some(data) = self.frame_event(&frame)? {
                            for e in state.feed(&data)? {
                                yield e;
                            }
                        }
                    }
                }
                for e in state.finish() {
                    yield e;
                }
            })
        })
    }

    async fn embed(&self, _texts: &[String], _model: Option<&str>) -> Result<Vec<Vec<f32>>> {
        Err(Error::config("embeddings on Bedrock use a different model family — call it directly"))
    }
}

// ---------------------------------------------------------------------------
// the event stream
// ---------------------------------------------------------------------------

/// One `application/vnd.amazon.eventstream` message: its string headers and payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventFrame {
    pub headers: HashMap<String, String>,
    pub payload: Vec<u8>,
}

/// AWS's event-stream framing → [`EventFrame`]s, fed a chunk at a time.
///
/// Each message: total length, headers length, a CRC of those eight bytes, the
/// headers, the payload, then a CRC of everything before it. Both CRCs are
/// checked, so a corrupted frame is an error rather than garbled JSON.
#[derive(Debug, Default)]
pub struct EventStreamDecoder {
    buffer: Vec<u8>,
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = flate2::Crc::new();
    crc.update(data);
    crc.sum()
}

fn corrupt(what: &str) -> Error {
    ProviderError::new(K::Other, format!("bedrock event stream is corrupt ({what})")).provider(SPEC.name).retryable(true).into()
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

impl EventStreamDecoder {
    /// Add bytes; get back every message they complete.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<EventFrame>> {
        self.buffer.extend_from_slice(chunk);
        let mut out = Vec::new();
        while self.buffer.len() >= 12 {
            let total = be32(&self.buffer[0..4]) as usize;
            let header_len = be32(&self.buffer[4..8]) as usize;
            if crc32(&self.buffer[..8]) != be32(&self.buffer[8..12]) {
                return Err(corrupt("prelude CRC"));
            }
            if total < 16 || 12 + header_len > total - 4 {
                return Err(corrupt("length"));
            }
            if self.buffer.len() < total {
                break;
            }
            let message: Vec<u8> = self.buffer.drain(..total).collect();
            if crc32(&message[..total - 4]) != be32(&message[total - 4..]) {
                return Err(corrupt("message CRC"));
            }
            out.push(EventFrame { headers: event_headers(&message[12..12 + header_len]), payload: message[12 + header_len..total - 4].to_vec() });
        }
        Ok(out)
    }
}

/// Decode event-stream headers. Only strings matter here; the rest are skipped.
fn event_headers(raw: &[u8]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut i = 0;
    while i < raw.len() {
        let name_len = raw[i] as usize;
        let Some(name) = raw.get(i + 1..i + 1 + name_len) else { break };
        let name = String::from_utf8_lossy(name).to_string();
        i += 1 + name_len;
        let Some(&kind) = raw.get(i) else { break };
        i += 1;
        let fixed = match kind {
            0 | 1 => 0,
            2 => 1,
            3 => 2,
            4 => 4,
            5 | 8 => 8,
            9 => 16,
            6 | 7 => {
                let Some(len) = raw.get(i..i + 2).map(|b| u16::from_be_bytes([b[0], b[1]]) as usize) else { break };
                let Some(value) = raw.get(i + 2..i + 2 + len) else { break };
                if kind == 7 {
                    out.insert(name, String::from_utf8_lossy(value).to_string());
                }
                i += 2 + len;
                continue;
            }
            _ => break,
        };
        i += fixed;
    }
    out
}

/// Build one event-stream frame of string headers. The inverse of
/// [`EventStreamDecoder`], for tests and fakes.
pub fn encode_event(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
    let mut raw = Vec::new();
    for (k, v) in headers {
        raw.push(k.len() as u8);
        raw.extend_from_slice(k.as_bytes());
        raw.push(7);
        raw.extend_from_slice(&(v.len() as u16).to_be_bytes());
        raw.extend_from_slice(v.as_bytes());
    }
    let total = (12 + raw.len() + payload.len() + 4) as u32;
    let mut message = Vec::new();
    message.extend_from_slice(&total.to_be_bytes());
    message.extend_from_slice(&(raw.len() as u32).to_be_bytes());
    let prelude_crc = crc32(&message);
    message.extend_from_slice(&prelude_crc.to_be_bytes());
    message.extend_from_slice(&raw);
    message.extend_from_slice(payload);
    let crc = crc32(&message);
    message.extend_from_slice(&crc.to_be_bytes());
    message
}

/// A Bedrock stream chunk carrying one Anthropic event, framed as Bedrock sends it.
pub fn encode_chunk(event: &Value) -> Vec<u8> {
    let b64 = base64::engine::general_purpose::STANDARD.encode(event.to_string());
    encode_event(&[(":message-type", "event"), (":event-type", "chunk"), (":content-type", "application/json")], json!({"bytes": b64}).to_string().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_providers::testing::{Fake, cfg};
    use crate::tools::ToolSchema;
    use crate::types::{Message, StopReason, StreamEventType};

    fn creds() -> AwsCredentials {
        AwsCredentials::new("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", None)
    }

    fn provider(url: &str) -> BedrockProvider {
        BedrockProvider::with_config(BedrockOptions { region: Some("eu-west-1".into()), credentials: Some(creds()), ..Default::default() }, cfg(url))
    }

    #[tokio::test]
    async fn signs_and_puts_the_model_in_the_url() {
        let fake = Fake::json(json!({
            "content": [{"type": "text", "text": "hello from bedrock"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 12, "output_tokens": 4}
        }))
        .await;
        let p = provider(&fake.url);
        let mut req = CompletionRequest::new("anthropic.claude-opus-5", vec![Message::user("hello")]);
        req.system = Some("Be brief.".into());
        req.max_tokens = 512;
        let resp = p.complete(&req).await.unwrap();
        let seen = fake.last();
        assert_eq!(seen.path, "/model/anthropic.claude-opus-5/invoke");
        let auth = &seen.headers["authorization"];
        assert!(auth.starts_with("AWS4-HMAC-SHA256 "), "{auth}");
        assert!(auth.contains("/eu-west-1/bedrock/aws4_request"));
        assert!(auth.contains("SignedHeaders=accept;host;x-amz-content-sha256;x-amz-date"), "{auth}");
        assert!(seen.headers.contains_key("x-amz-date"));
        assert_eq!(seen.headers["accept"], "application/json");
        assert_eq!(seen.body["anthropic_version"], BEDROCK_VERSION);
        assert!(seen.body.get("model").is_none());
        assert_eq!(seen.body["system"], "Be brief.");
        assert_eq!(seen.body["max_tokens"], 512);
        assert_eq!(resp.text(), "hello from bedrock");
        assert_eq!(resp.provider, "bedrock");
        assert_eq!(resp.usage.input_tokens, 12);
        assert!(resp.usage.cost_usd > 0.0);
    }

    #[test]
    fn the_signature_covers_a_double_encoded_path() {
        // What AWS computes for `/model/anthropic.claude-v2%3A1/invoke`: the
        // canonical path is `/model/anthropic.claude-v2%253A1/invoke`.
        let path = BedrockProvider::model_path("anthropic.claude-v2:1", false);
        assert_eq!(path, "/model/anthropic.claude-v2%3A1/invoke");
        let url = format!("https://bedrock-runtime.us-east-1.amazonaws.com{path}");
        assert_eq!(signing_url(&url), "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2%253A1/invoke");
        let at = |u: &str| {
            let req = SignRequest {
                method: "POST",
                url: u,
                region: "us-east-1",
                service: "bedrock",
                body: b"{}",
                headers: &[],
                now: Some(1.0),
                content_sha_header: true,
            };
            sign(&req, &creds()).into_iter().find(|(k, _)| k == "authorization").unwrap().1
        };
        // Once decoded and re-encoded by `sign`, the two spellings differ.
        assert_ne!(at(&url), at(&signing_url(&url)));
    }

    #[tokio::test]
    async fn an_api_key_skips_signing() {
        let fake = Fake::json(json!({"content": [], "stop_reason": "end_turn", "usage": {}})).await;
        let p = BedrockProvider::with_config(BedrockOptions { api_key: Some("bedrock-key".into()), ..Default::default() }, cfg(&fake.url));
        p.complete(&CompletionRequest::new("anthropic.claude-opus-5", vec![Message::user("x")])).await.unwrap();
        let seen = fake.last();
        assert_eq!(seen.headers["authorization"], "Bearer bedrock-key");
        assert!(!seen.headers.contains_key("x-amz-date"));
    }

    #[tokio::test]
    async fn without_credentials_it_says_what_to_do() {
        let p = BedrockProvider::with_config(BedrockOptions { credentials: Some(AwsCredentials::default()), ..Default::default() }, cfg("http://127.0.0.1:9"));
        let err = p.complete(&CompletionRequest::new("anthropic.claude-opus-5", vec![Message::user("x")])).await.unwrap_err();
        assert!(err.to_string().contains("AWS_ACCESS_KEY_ID"), "{err}");
        assert_eq!(err.as_provider().unwrap().kind, K::Authentication);
    }

    #[tokio::test]
    async fn tools_survive_the_round_trip() {
        let fake = Fake::json(json!({"content": [], "stop_reason": "end_turn", "usage": {}})).await;
        let mut req = CompletionRequest::new("anthropic.claude-opus-5", vec![Message::user("x")]);
        req.tools = vec![ToolSchema { name: "lookup".into(), description: "Look up".into(), parameters: json!({"type": "object", "properties": {}}) }];
        provider(&fake.url).complete(&req).await.unwrap();
        assert_eq!(fake.last().body["tools"][0]["name"], "lookup");
    }

    #[tokio::test]
    async fn streams_the_event_stream() {
        let events = [
            json!({"type": "message_start", "message": {"model": "claude-opus-5", "usage": {"input_tokens": 10}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}),
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}),
        ];
        let mut body: Vec<u8> = events.iter().flat_map(encode_chunk).collect();
        body.extend(encode_event(&[(":message-type", "event"), (":event-type", "metrics")], b"{}"));
        let fake = Fake::raw(body, "application/vnd.amazon.eventstream").await;
        let p = provider(&fake.url);
        let req = CompletionRequest::new("anthropic.claude-opus-5", vec![Message::user("hi")]);
        let out: Vec<StreamEvent> = p.stream(&req).map(|e| e.unwrap()).collect().await;
        let text: String = out.iter().filter(|e| e.kind == StreamEventType::Text).map(|e| e.text.as_str()).collect();
        assert_eq!(text, "Hello");
        let resp = super::super::base::response_from_event(out.last().unwrap()).unwrap();
        assert_eq!(resp.provider, "bedrock");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert_eq!(resp.usage.output_tokens, 2);
        let seen = fake.last();
        assert_eq!(seen.path, "/model/anthropic.claude-opus-5/invoke-with-response-stream");
        assert_eq!(seen.headers["accept"], "application/vnd.amazon.eventstream");
        assert_eq!(seen.headers["x-amzn-bedrock-accept"], "application/json");
        assert!(seen.body.get("stream").is_none());
    }

    #[tokio::test]
    async fn a_stream_exception_is_classified() {
        let body = encode_event(&[(":message-type", "exception"), (":exception-type", "validationException")], br#"{"message":"bad input"}"#);
        let fake = Fake::raw(body, "application/vnd.amazon.eventstream").await;
        let req = CompletionRequest::new("anthropic.claude-opus-5", vec![Message::user("hi")]);
        let p = provider(&fake.url);
        let results: Vec<Result<StreamEvent>> = p.stream(&req).collect().await;
        let err = results.into_iter().find_map(|r| r.err()).unwrap();
        let pe = err.as_provider().unwrap();
        assert_eq!(pe.status, Some(400));
        assert_eq!(pe.provider, "bedrock");
        assert!(pe.message.contains("bad input"));
    }

    #[test]
    fn frames_split_anywhere_and_corruption_is_caught() {
        let frame = encode_event(&[(":event-type", "chunk")], b"payload");
        let mut d = EventStreamDecoder::default();
        let mut got = Vec::new();
        for b in frame.iter().chain(frame.iter()) {
            got.extend(d.push(&[*b]).unwrap());
        }
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].headers[":event-type"], "chunk");
        assert_eq!(got[0].payload, b"payload");
        let mut bad = frame.clone();
        let n = bad.len();
        bad[n - 6] ^= 1;
        let err = EventStreamDecoder::default().push(&bad).unwrap_err();
        assert!(err.to_string().contains("message CRC"), "{err}");
        assert!(err.as_provider().unwrap().retryable);
        let mut bad = frame;
        bad[1] ^= 1;
        assert!(EventStreamDecoder::default().push(&bad).unwrap_err().to_string().contains("prelude CRC"));
    }

    #[test]
    fn profiles_are_read_from_the_credentials_file() {
        let text = "[default]\naws_access_key_id = A\naws_secret_access_key = B\n\n[profile work]\naws_access_key_id=C\naws_secret_access_key=D\naws_session_token=T\n";
        assert_eq!(parse_profile(text, "default"), Some(AwsCredentials::new("A", "B", None)));
        assert_eq!(parse_profile(text, "work"), Some(AwsCredentials::new("C", "D", Some("T".into()))));
        assert_eq!(parse_profile(text, "none"), None);
    }

    /// `AH_LIVE_BEDROCK=1` with real AWS credentials; `AH_BEDROCK_MODEL` picks the model.
    #[tokio::test]
    async fn live_bedrock() {
        if std::env::var("AH_LIVE_BEDROCK").as_deref() != Ok("1") {
            return;
        }
        let model = std::env::var("AH_BEDROCK_MODEL").unwrap_or_else(|_| "us.anthropic.claude-haiku-4-5-20251001-v1:0".into());
        let p = BedrockProvider::new(BedrockOptions::default());
        let mut req = CompletionRequest::new(model, vec![Message::user("Say hi in one word.")]);
        req.max_tokens = 20;
        assert!(!p.complete(&req).await.unwrap().text().is_empty());
        let events: Vec<StreamEvent> = p.stream(&req).map(|e| e.unwrap()).collect().await;
        assert_eq!(events.last().unwrap().kind, StreamEventType::StepEnd);
    }
}
