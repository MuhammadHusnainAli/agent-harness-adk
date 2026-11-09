//! What happens when a model call fails: retries, backoff, classification, circuits.
//!
//! 1. **What went wrong.** [`classify`] turns a status, headers and body into a
//!    typed [`ProviderError`], using each vendor's own error codes.
//! 2. **How long to wait.** [`retry_after`] reads every hint a provider gives:
//!    `Retry-After` (seconds or HTTP date), `retry-after-ms`, OpenAI's
//!    `x-ratelimit-reset-*`, Gemini's `retryDelay` in the body.
//! 3. **Whether to try again.** [`RetryPolicy`].
//! 4. **Whether to call at all.** [`CircuitBreaker`].

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use regex::Regex;
use reqwest::header::HeaderMap;
use serde_json::{Value, json};

use crate::errors::{ProviderError, ProviderErrorKind};

/// 408 request timeout, 409 lock contention, 425 too early, 429 rate limited.
pub const RETRY_STATUSES: [u16; 4] = [408, 409, 425, 429];
/// The 5xx statuses that mean "this will never work".
pub const NEVER_RETRY_5XX: [u16; 2] = [501, 505];

/// A uniformly random fraction in `[0, 1)`.
pub(crate) fn random_fraction() -> f64 {
    let (a, _) = uuid::Uuid::new_v4().as_u64_pair();
    (a >> 11) as f64 / (1u64 << 53) as f64
}

/// How hard to try before giving up on a call.
///
/// Waits grow as `initial_delay * multiplier ^ attempt`, capped at
/// `max_delay`, with `jitter` of that shaved off at random.
#[derive(Debug, Clone, PartialEq)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_delay: f64,
    pub max_delay: f64,
    pub multiplier: f64,
    /// Fraction of each wait that is randomised: 0 is deterministic, 1 is full jitter.
    pub jitter: f64,
    /// The longest a server may ask us to wait. Longer and the call fails at
    /// once, carrying `retry_after`, so a fallback model can take over.
    pub max_retry_after: f64,
    /// A ceiling on the whole call, retries and waits included.
    pub max_elapsed: Option<f64>,
    pub retry_on_timeout: bool,
    pub retry_on_connection_error: bool,
    pub retry_statuses: Vec<u16>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: 0.5,
            max_delay: 30.0,
            multiplier: 2.0,
            jitter: 0.25,
            max_retry_after: 60.0,
            max_elapsed: None,
            retry_on_timeout: true,
            retry_on_connection_error: true,
            retry_statuses: RETRY_STATUSES.to_vec(),
        }
    }
}

impl RetryPolicy {
    /// Fail fast — e.g. in tests.
    pub fn none() -> Self {
        Self { max_retries: 0, ..Self::default() }
    }

    pub fn should_retry_status(&self, status: u16) -> bool {
        self.retry_statuses.contains(&status) || ((500..600).contains(&status) && !NEVER_RETRY_5XX.contains(&status))
    }

    /// The wait before retry number `attempt + 1` when the server gave no hint.
    pub fn backoff(&self, attempt: u32) -> f64 {
        let base = self.max_delay.min(self.initial_delay * self.multiplier.powi(attempt as i32));
        let spread = base * self.jitter.clamp(0.0, 1.0);
        (base - random_fraction() * spread).max(0.0)
    }
}

/// Handed to `on_retry` before each wait, so a retry is never silent.
#[derive(Debug, Clone)]
pub struct RetryEvent {
    pub provider: String,
    /// The attempt that just failed, from 1.
    pub attempt: u32,
    /// Seconds until the next one.
    pub delay: f64,
    pub error: ProviderError,
    pub url: String,
}

impl RetryEvent {
    pub fn reason(&self) -> String {
        format!("{}: {}", self.error.kind.name(), self.error)
    }
}

/// Running counters for one provider instance.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProviderStats {
    /// HTTP attempts, retries included.
    pub requests: u64,
    /// Logical calls (complete / stream / embed).
    pub calls: u64,
    pub succeeded: u64,
    pub failed: u64,
    pub retries: u64,
    pub rate_limited: u64,
    pub timeouts: u64,
    pub circuit_rejections: u64,
    pub total_backoff_s: f64,
    pub last_error: String,
    pub last_request_id: String,
    pub by_status: BTreeMap<u16, u64>,
}

impl ProviderStats {
    pub fn snapshot(&self) -> Value {
        json!({
            "requests": self.requests, "calls": self.calls,
            "succeeded": self.succeeded, "failed": self.failed,
            "retries": self.retries, "rate_limited": self.rate_limited,
            "timeouts": self.timeouts, "circuit_rejections": self.circuit_rejections,
            "total_backoff_s": (self.total_backoff_s * 1000.0).round() / 1000.0,
            "last_error": self.last_error, "last_request_id": self.last_request_id,
            "by_status": self.by_status.iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
        })
    }
}

/// The state of a [`CircuitBreaker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl CircuitState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Closed => "closed",
            Self::Open => "open",
            Self::HalfOpen => "half_open",
        }
    }
}

#[derive(Debug)]
struct CircuitInner {
    failures: u32,
    opened_at: Option<Instant>,
    state: CircuitState,
    probing: bool,
}

/// Stop calling a provider that keeps failing, then probe it back to health.
///
/// `closed` → calls flow; `open` (after `failure_threshold` consecutive
/// failures) → calls fail at once for `reset_timeout`; `half_open` → one probe
/// is let through. Only failures that say the *provider* is unwell count.
/// Cheap to clone: clones share state.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub failure_threshold: u32,
    pub reset_timeout: Duration,
    inner: Arc<Mutex<CircuitInner>>,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(5, Duration::from_secs(30))
    }
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failure_threshold: failure_threshold.max(1),
            reset_timeout,
            inner: Arc::new(Mutex::new(CircuitInner { failures: 0, opened_at: None, state: CircuitState::Closed, probing: false })),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CircuitInner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn refresh(&self, inner: &mut CircuitInner) {
        if inner.state == CircuitState::Open && inner.opened_at.is_some_and(|t| t.elapsed() >= self.reset_timeout) {
            inner.state = CircuitState::HalfOpen;
            inner.probing = false;
        }
    }

    pub fn state(&self) -> CircuitState {
        let mut inner = self.lock();
        self.refresh(&mut inner);
        inner.state
    }

    pub fn failures(&self) -> u32 {
        self.lock().failures
    }

    /// May a call go out now? Reserves the probe slot when half open.
    pub fn allow(&self) -> bool {
        let mut inner = self.lock();
        self.refresh(&mut inner);
        match inner.state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen if !inner.probing => {
                inner.probing = true;
                true
            }
            _ => false,
        }
    }

    /// Seconds until a probe is let through.
    pub fn retry_in(&self) -> f64 {
        let inner = self.lock();
        match inner.opened_at {
            Some(t) => (self.reset_timeout.as_secs_f64() - t.elapsed().as_secs_f64()).max(0.0),
            None => 0.0,
        }
    }

    pub fn record_success(&self) {
        let mut inner = self.lock();
        inner.failures = 0;
        inner.state = CircuitState::Closed;
        inner.probing = false;
    }

    pub fn record_failure(&self) {
        let mut inner = self.lock();
        inner.failures += 1;
        inner.probing = false;
        if inner.state == CircuitState::HalfOpen || inner.failures >= self.failure_threshold {
            inner.state = CircuitState::Open;
            inner.opened_at = Some(Instant::now());
        }
    }

    /// A probe was abandoned (cancelled mid-call): free the slot.
    pub(crate) fn release_probe(&self) {
        self.lock().probing = false;
    }

    pub fn reset(&self) {
        self.record_success();
    }
}

// --- reading what the provider told us ---------------------------------------

/// `"1.5s"`, `"250ms"`, `"6m0s"`, `"1h2m"`, or a bare number of seconds.
pub fn parse_duration(text: &str) -> Option<f64> {
    let text = text.trim().to_lowercase();
    if text.is_empty() {
        return None;
    }
    if let Ok(v) = text.parse::<f64>() {
        return Some(v.max(0.0));
    }
    let re = Regex::new(r"(\d+(?:\.\d+)?)(ms|h|m|s)").expect("valid regex");
    let compact = text.replace(' ', "");
    let mut joined = String::new();
    let mut total = 0.0;
    for cap in re.captures_iter(&text) {
        let (n, u) = (&cap[1], &cap[2]);
        joined.push_str(n);
        joined.push_str(u);
        let scale = match u {
            "ms" => 0.001,
            "s" => 1.0,
            "m" => 60.0,
            _ => 3600.0,
        };
        total += n.parse::<f64>().ok()? * scale;
    }
    if joined.is_empty() || joined != compact {
        return None;
    }
    Some(total)
}

fn header<'a>(headers: Option<&'a HeaderMap>, name: &str) -> Option<&'a str> {
    headers?.get(name)?.to_str().ok().filter(|s| !s.is_empty())
}

/// Seconds the provider asked us to wait, from whichever hint it gave.
/// OpenAI's bucket-reset headers ride on every response, so they are only
/// read when the call was actually rate limited.
pub fn retry_after(headers: Option<&HeaderMap>, body: Option<&str>, rate_limited: bool) -> Option<f64> {
    if let Some(ms) = header(headers, "retry-after-ms")
        && let Ok(v) = ms.parse::<f64>()
    {
        return Some((v / 1000.0).max(0.0));
    }
    if let Some(value) = header(headers, "retry-after") {
        if let Ok(v) = value.parse::<f64>() {
            return Some(v.max(0.0));
        }
        if let Ok(when) = httpdate::parse_http_date(value) {
            let secs = when.duration_since(SystemTime::now()).map(|d| d.as_secs_f64()).unwrap_or(0.0);
            return Some(secs);
        }
    }
    if rate_limited {
        let found: Vec<f64> =
            ["x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"].iter().filter_map(|h| parse_duration(header(headers, h).unwrap_or(""))).collect();
        if !found.is_empty() {
            return found.into_iter().reduce(f64::max);
        }
    }
    if let Some(body) = body {
        let re = Regex::new(r#""retryDelay"\s*:\s*"([^"]+)""#).expect("valid regex");
        if let Some(cap) = re.captures(body) {
            return parse_duration(&cap[1]);
        }
    }
    None
}

/// The vendor's request id, from whichever header carries it.
pub fn request_id(headers: Option<&HeaderMap>) -> Option<String> {
    ["request-id", "x-request-id", "x-amzn-requestid", "x-amz-request-id", "x-goog-request-id", "apim-request-id", "x-ms-request-id", "cf-ray"]
        .iter()
        .find_map(|n| header(headers, n).map(str::to_string))
}

fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn scalar(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// (message, code) out of whatever error envelope the vendor uses.
fn error_fields(body: Option<&str>) -> (String, String) {
    let Some(body) = body.filter(|b| !b.is_empty()) else { return (String::new(), String::new()) };
    let Ok(mut data) = serde_json::from_str::<Value>(body) else { return (clip(body.trim(), 500), String::new()) };
    if let Value::Array(items) = &data
        && let Some(first) = items.first()
    {
        data = first.clone();
    }
    let Value::Object(obj) = &data else { return (clip(&data.to_string(), 500), String::new()) };
    let err = obj.get("error").unwrap_or(&data);
    match err {
        Value::String(s) => (s.clone(), scalar(obj.get("code").or_else(|| obj.get("type")))),
        Value::Object(e) => {
            let message = [e.get("message"), obj.get("message"), obj.get("Message")].into_iter().map(scalar).find(|s| !s.is_empty()).unwrap_or_default();
            let mut code = [e.get("code"), e.get("type"), e.get("status"), obj.get("__type")]
                .into_iter()
                .map(scalar)
                .find(|s| !s.is_empty() && s != "0")
                .unwrap_or_default();
            if let Some(Value::Array(details)) = e.get("details") {
                for d in details {
                    if let Some(reason) = d.get("reason").and_then(Value::as_str) {
                        code = if code.is_empty() { reason.to_string() } else { format!("{code}:{reason}") };
                    }
                }
            }
            (message, code)
        }
        other => (clip(&other.to_string(), 500), String::new()),
    }
}

const CONTEXT_MARKERS: [&str; 8] = [
    "context_length_exceeded",
    "context length",
    "context window",
    "prompt is too long",
    "input is too long",
    "too many tokens",
    "maximum number of tokens",
    "request_too_large",
];
const QUOTA_MARKERS: [&str; 5] = ["insufficient_quota", "billing", "credit balance", "exceeded your current quota", "quota exceeded"];
const AUTH_MARKERS: [&str; 6] = ["api_key_invalid", "invalid api key", "invalid x-api-key", "unauthenticated", "permission_denied", "invalid_api_key"];

/// The right [`ProviderError`] kind for an HTTP failure.
pub fn classify(provider: &str, status: u16, headers: Option<&HeaderMap>, body: Option<&str>, policy: Option<&RetryPolicy>) -> ProviderError {
    let (message, code) = error_fields(body);
    let lowered = format!("{message} {code}").to_lowercase();
    let rid = request_id(headers);
    let wait = retry_after(headers, body, status == 429);
    let detail = if !message.is_empty() {
        message.clone()
    } else {
        let b = clip(body.unwrap_or("").trim(), 300);
        if b.is_empty() { "no body".into() } else { b }
    };
    let mut text = format!("{provider} returned {status}: {detail}");
    if let Some(r) = &rid {
        text.push_str(&format!(" (request id {r})"));
    }
    let has = |markers: &[&str]| markers.iter().any(|m| lowered.contains(m));
    let default_policy = RetryPolicy::default();
    let policy = policy.unwrap_or(&default_policy);

    use ProviderErrorKind as K;
    let kind = if status == 429 || lowered.contains("throttl") || lowered.contains("resource_exhausted") {
        if has(&QUOTA_MARKERS) { K::QuotaExceeded } else { K::RateLimit }
    } else if has(&CONTEXT_MARKERS) || status == 413 {
        K::ContextWindowExceeded
    } else if status == 401 || status == 403 || has(&AUTH_MARKERS) {
        K::Authentication
    } else if status == 404 || lowered.contains("model_not_found") || lowered.contains("not_found_error") {
        K::ModelNotFound
    } else if status == 402 {
        K::QuotaExceeded
    } else if status == 408 || status == 504 {
        K::Timeout
    } else if policy.should_retry_status(status) || lowered.contains("overloaded") {
        K::Unavailable
    } else {
        K::InvalidRequest
    };

    let mut err = ProviderError::new(kind, text).provider(provider).status(status).retry_after(wait);
    err.body = Some(clip(body.unwrap_or(""), 2000));
    err.request_id = rid;
    err.code = if code.is_empty() { None } else { Some(code) };
    // OpenAI and Anthropic say outright whether a retry is worthwhile. Believe them.
    match header(headers, "x-should-retry").map(str::to_lowercase).as_deref() {
        Some("true") => err.retryable = true,
        Some("false") => err.retryable = false,
        _ => {}
    }
    err
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn durations() {
        assert_eq!(parse_duration("1.5s"), Some(1.5));
        assert_eq!(parse_duration("250ms"), Some(0.25));
        assert_eq!(parse_duration("6m0s"), Some(360.0));
        assert_eq!(parse_duration("1h2m"), Some(3720.0));
        assert_eq!(parse_duration("7"), Some(7.0));
        assert_eq!(parse_duration("soon"), None);
        assert_eq!(parse_duration(""), None);
    }

    #[test]
    fn retry_after_sources() {
        assert_eq!(retry_after(Some(&headers(&[("retry-after-ms", "1500")])), None, true), Some(1.5));
        assert_eq!(retry_after(Some(&headers(&[("retry-after", "3")])), None, true), Some(3.0));
        let h = headers(&[("x-ratelimit-reset-requests", "2s"), ("x-ratelimit-reset-tokens", "6m0s")]);
        assert_eq!(retry_after(Some(&h), None, true), Some(360.0));
        assert_eq!(retry_after(Some(&h), None, false), None);
        let body = r#"{"error":{"details":[{"retryDelay":"12s"}]}}"#;
        assert_eq!(retry_after(None, Some(body), true), Some(12.0));
        let past = headers(&[("retry-after", "Wed, 21 Oct 2015 07:28:00 GMT")]);
        assert_eq!(retry_after(Some(&past), None, true), Some(0.0));
    }

    #[test]
    fn classifies_by_status_and_body() {
        use ProviderErrorKind as K;
        let c = |status, body: &str| classify("p", status, None, Some(body), None);
        assert_eq!(c(429, "{}").kind, K::RateLimit);
        assert_eq!(c(429, r#"{"error":{"type":"insufficient_quota","message":"x"}}"#).kind, K::QuotaExceeded);
        assert_eq!(c(400, r#"{"error":{"message":"prompt is too long"}}"#).kind, K::ContextWindowExceeded);
        assert_eq!(c(401, "").kind, K::Authentication);
        assert_eq!(c(404, "").kind, K::ModelNotFound);
        assert_eq!(c(402, "").kind, K::QuotaExceeded);
        assert_eq!(c(504, "").kind, K::Timeout);
        assert_eq!(c(503, "").kind, K::Unavailable);
        assert_eq!(c(529, "").kind, K::Unavailable);
        assert_eq!(c(501, "").kind, K::InvalidRequest);
        assert_eq!(c(400, r#"{"error":{"message":"bad"}}"#).kind, K::InvalidRequest);
        assert!(!c(400, "").retryable);
        assert!(c(503, "").retryable);
    }

    #[test]
    fn classify_message_and_headers() {
        let h = headers(&[("request-id", "req_1"), ("x-should-retry", "false"), ("retry-after", "2")]);
        let err = classify("anthropic", 529, Some(&h), Some(r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#), None);
        assert_eq!(err.message, "anthropic returned 529: Overloaded (request id req_1)");
        assert_eq!(err.code.as_deref(), Some("overloaded_error"));
        assert!(!err.retryable);
        assert_eq!(err.request_id.as_deref(), Some("req_1"));
        assert_eq!(err.retry_after, Some(2.0));
        let google = classify(
            "gemini",
            429,
            None,
            Some(r#"{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"m","details":[{"reason":"RATE_LIMIT_EXCEEDED"}]}}"#),
            None,
        );
        assert_eq!(google.code.as_deref(), Some("429:RATE_LIMIT_EXCEEDED"));
        assert_eq!(classify("p", 500, None, None, None).message, "p returned 500: no body");
    }

    #[test]
    fn backoff_grows_and_caps() {
        let p = RetryPolicy { jitter: 0.0, ..Default::default() };
        assert_eq!(p.backoff(0), 0.5);
        assert_eq!(p.backoff(2), 2.0);
        assert_eq!(p.backoff(20), 30.0);
        let j = RetryPolicy::default().backoff(1);
        assert!((0.75..=1.0).contains(&j));
        assert!(p.should_retry_status(409));
        assert!(!p.should_retry_status(505));
    }

    #[test]
    fn circuit_opens_and_probes() {
        let c = CircuitBreaker::new(2, Duration::from_millis(0));
        c.record_failure();
        assert_eq!(c.state(), CircuitState::Closed);
        c.record_failure();
        // reset_timeout of zero: open becomes half_open immediately.
        assert_eq!(c.state(), CircuitState::HalfOpen);
        assert!(c.allow());
        assert!(!c.allow());
        c.record_success();
        assert_eq!(c.state(), CircuitState::Closed);

        let slow = CircuitBreaker::new(1, Duration::from_secs(60));
        slow.record_failure();
        assert_eq!(slow.state(), CircuitState::Open);
        assert!(!slow.allow());
        assert!(slow.retry_in() > 59.0);
    }
}
