//! Web search, as a tool: a query in, a ranked list of pages out.
//!
//! ```no_run
//! use agent_harness::toolkits::{WebSearch, web_search};
//!
//! # async fn demo() -> agent_harness::errors::Result<()> {
//! let tool = web_search(); // whichever engine has a key in the environment
//! let search = WebSearch::builder().engine("brave").engine("duckduckgo").allowed_domains(["*.gov"]).build()?;
//! let hits = search.search("heat pump subsidy 2026", None, Some("year"), None).await?;
//! let tool = search.as_tool(None, None);
//! # Ok(()) }
//! ```
//!
//! [`web_search`] uses whichever search engine has a key in the environment —
//! Tavily, Brave, Exa, Serper, Google, or a SearXNG instance of your own — and
//! DuckDuckGo, which needs no key, when none does. Name one, or several to
//! fall back through, when the choice should be yours.
//!
//! Every engine is spoken to over plain HTTP, and every one is answered the
//! same way: title, URL, snippet, and the date when the engine knows it.
//!
//! A search is something that fails — a key runs out, an engine throttles, a
//! page layout changes — so the failure is handled here rather than handed to
//! the model. A call that may succeed later is retried with back-off,
//! honouring `Retry-After`; an engine that cannot answer is passed over for the
//! next one; an engine that keeps failing is left alone for a minute; the
//! whole search has a deadline. Results are de-duplicated, stripped of markup
//! and tracking parameters, held to the domain policy, and kept for a few
//! minutes so the same question is not asked twice. When nothing works the
//! model is told which engine failed and why — never with a key in the message.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::{BoxFuture, Shared};
use indexmap::IndexMap;
use regex::Regex;
use reqwest::header::HeaderMap;
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::errors::{Error, Result};
use crate::governance::{fnmatch, py_str, py_str_repr};
use crate::llm_providers::resilience::{CircuitBreaker, CircuitState, RetryPolicy, retry_after};
use crate::tools::{Tool, ToolContext};

/// The recency values a search understands.
pub const RECENCY: [&str; 4] = ["day", "week", "month", "year"];
/// The safe-search levels.
pub const SAFE: [&str; 3] = ["off", "moderate", "strict"];
/// Set to an engine name, or several separated by commas, to choose the
/// engines of a search that was not given any.
pub const ENGINE_ENV: &str = "AGENT_HARNESS_SEARCH";
/// Every engine that ships, in the order they are preferred when the choice is
/// left to the environment.
pub const ENGINES: [&str; 7] = ["tavily", "brave", "exa", "serper", "google", "searxng", "duckduckgo"];

const MAX_QUERY_CHARS: usize = 400;
const MAX_QUERY_WORDS: usize = 50;
/// More `site:` operators than this and engines ignore the query.
const MAX_SITES: usize = 8;
const TITLE_CHARS: usize = 200;

static MARKUP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").expect("valid"));
/// Emphasis an engine puts around the matched words; it leaves no gap behind.
static INLINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)</?(?:b|strong|em|i|mark|span|u)\b[^>]*>").expect("valid"));
static SPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").expect("valid"));
static CONTROL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]").expect("valid"));
static ISO_DATE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\d{4}-\d{2}-\d{2}").expect("valid"));
static GLOB: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[*?\[\]]").expect("valid"));
/// Query parameters that say where a click came from, and nothing about the page.
static TRACKING: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(utm_\w+|fbclid|gclid|gclsrc|dclid|msclkid|yclid|mc_cid|mc_eid|igshid|_hsenc|_hsmi|oly_anon_id|oly_enc_id|srsltid|ref_src|ref_url)$")
        .expect("valid")
});
static QUOTA: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)quota|credit|billing|plan limit|usage limit").expect("valid"));

// ----------------------------------------------------------------------
// the environment
// ----------------------------------------------------------------------

#[cfg(test)]
thread_local! {
    /// Tests see this instead of the process environment, so no test reads a
    /// key from the machine it runs on.
    static TEST_ENV: std::cell::RefCell<HashMap<String, String>> = std::cell::RefCell::new(HashMap::new());
}

/// An environment variable, when it is set to something.
fn env(name: &str) -> Option<String> {
    #[cfg(test)]
    {
        TEST_ENV.with(|e| e.borrow().get(name).cloned()).filter(|v| !v.is_empty())
    }
    #[cfg(not(test))]
    {
        std::env::var(name).ok().filter(|v| !v.is_empty())
    }
}

// ----------------------------------------------------------------------
// what goes in, what comes out
// ----------------------------------------------------------------------

/// One search, as every engine is asked it.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchQuery {
    pub text: String,
    pub limit: usize,
    /// "day", "week", "month" or "year"; empty for any time.
    pub recency: String,
    /// Only these domains. Plain names or globs.
    pub domains: Vec<String>,
    /// Never these.
    pub exclude: Vec<String>,
    pub region: String,
    pub language: String,
    pub safe: String,
    pub snippet_chars: usize,
}

impl SearchQuery {
    /// A query with the defaults: 5 results, any time, moderate safe search.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            limit: 5,
            recency: String::new(),
            domains: Vec::new(),
            exclude: Vec::new(),
            region: String::new(),
            language: String::new(),
            safe: "moderate".into(),
            snippet_chars: 500,
        }
    }

    /// The wanted domains an engine can be told about — globs it cannot.
    pub fn include_domains(&self) -> Vec<String> {
        self.domains.iter().filter(|d| !GLOB.is_match(d)).cloned().collect()
    }

    pub fn exclude_domains(&self) -> Vec<String> {
        self.exclude.iter().filter(|d| !GLOB.is_match(d)).cloned().collect()
    }

    /// The text with `site:` operators, for an engine with no domain filter.
    pub fn with_sites(&self) -> String {
        let mut text = self.text.clone();
        let wanted: Vec<String> = self.include_domains().into_iter().take(MAX_SITES).collect();
        // Only when every wanted domain fits: half a filter loses results.
        if !wanted.is_empty() && wanted.len() == self.domains.len() {
            let sites = wanted.iter().map(|d| format!("site:{d}")).collect::<Vec<_>>().join(" OR ");
            text = if wanted.len() == 1 { format!("{text} {sites}") } else { format!("{text} ({sites})") };
        }
        for domain in self.exclude_domains().iter().take(MAX_SITES) {
            text = format!("{text} -site:{domain}");
        }
        text
    }

    fn days(&self) -> i64 {
        match self.recency.as_str() {
            "day" => 1,
            "week" => 7,
            "month" => 31,
            _ => 366,
        }
    }
}

/// One page a search found.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    /// When the page was published, as the engine reported it. Often empty.
    pub published: String,
    /// The engine that found it.
    pub engine: String,
    pub score: Option<f64>,
}

impl SearchResult {
    pub fn new(title: impl Into<String>, url: impl Into<String>) -> Self {
        Self { title: title.into(), url: url.into(), snippet: String::new(), published: String::new(), engine: String::new(), score: None }
    }

    /// What the model reads: the fields that have something in them.
    pub fn to_dict(&self) -> Value {
        let mut out = Map::new();
        out.insert("title".into(), json!(self.title));
        out.insert("url".into(), json!(self.url));
        if !self.snippet.is_empty() {
            out.insert("snippet".into(), json!(self.snippet));
        }
        if !self.published.is_empty() {
            out.insert("published".into(), json!(self.published));
        }
        Value::Object(out)
    }

    /// As an engine hands a result over: a mapping with every field.
    pub fn to_raw(&self) -> Value {
        json!({"title": self.title, "url": self.url, "snippet": self.snippet, "published": self.published, "score": self.score})
    }
}

/// One engine could not answer. Returned by engines, read by [`WebSearch`].
#[derive(Debug, Clone, PartialEq)]
pub struct SearchFailure {
    pub message: String,
    /// The same request may succeed in a moment (429, 5xx, timeout).
    pub retryable: bool,
    /// Seconds the engine asked us to wait, when it said.
    pub retry_after: Option<f64>,
    /// Counts towards leaving the engine alone for a while; a query the
    /// engine refused says nothing about its health.
    pub unhealthy: bool,
    pub status: Option<u16>,
}

impl SearchFailure {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), retryable: false, retry_after: None, unhealthy: true, status: None }
    }

    pub fn retryable(mut self) -> Self {
        self.retryable = true;
        self
    }

    fn healthy(mut self) -> Self {
        self.unhealthy = false;
        self
    }

    fn status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }
}

impl std::fmt::Display for SearchFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl From<Error> for SearchFailure {
    /// Any other error from an engine: its kind and message, as Python's `f"{kind}: {exc}"`.
    fn from(e: Error) -> Self {
        SearchFailure::new(format!("{}: {e}", e.kind_name()))
    }
}

// ----------------------------------------------------------------------
// tidying what an engine returned
// ----------------------------------------------------------------------

fn named_entity(name: &str) -> Option<&'static str> {
    Some(match name {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        "nbsp" => "\u{a0}",
        "hellip" => "\u{2026}",
        "mdash" => "\u{2014}",
        "ndash" => "\u{2013}",
        "lsquo" => "\u{2018}",
        "rsquo" => "\u{2019}",
        "sbquo" => "\u{201a}",
        "ldquo" => "\u{201c}",
        "rdquo" => "\u{201d}",
        "bdquo" => "\u{201e}",
        "laquo" => "\u{ab}",
        "raquo" => "\u{bb}",
        "lsaquo" => "\u{2039}",
        "rsaquo" => "\u{203a}",
        "middot" => "\u{b7}",
        "bull" => "\u{2022}",
        "copy" => "\u{a9}",
        "reg" => "\u{ae}",
        "trade" => "\u{2122}",
        "deg" => "\u{b0}",
        "plusmn" => "\u{b1}",
        "times" => "\u{d7}",
        "divide" => "\u{f7}",
        "euro" => "\u{20ac}",
        "pound" => "\u{a3}",
        "yen" => "\u{a5}",
        "cent" => "\u{a2}",
        "sect" => "\u{a7}",
        "para" => "\u{b6}",
        "shy" => "\u{ad}",
        "iexcl" => "\u{a1}",
        "iquest" => "\u{bf}",
        "frac12" => "\u{bd}",
        "frac14" => "\u{bc}",
        "frac34" => "\u{be}",
        "micro" => "\u{b5}",
        "eacute" => "\u{e9}",
        "Eacute" => "\u{c9}",
        "egrave" => "\u{e8}",
        "agrave" => "\u{e0}",
        "aacute" => "\u{e1}",
        "acirc" => "\u{e2}",
        "auml" => "\u{e4}",
        "ouml" => "\u{f6}",
        "uuml" => "\u{fc}",
        "Auml" => "\u{c4}",
        "Ouml" => "\u{d6}",
        "Uuml" => "\u{dc}",
        "szlig" => "\u{df}",
        "ccedil" => "\u{e7}",
        "ntilde" => "\u{f1}",
        "oacute" => "\u{f3}",
        "iacute" => "\u{ed}",
        "uacute" => "\u{fa}",
        "ecirc" => "\u{ea}",
        "ocirc" => "\u{f4}",
        "aring" => "\u{e5}",
        "oslash" => "\u{f8}",
        "aelig" => "\u{e6}",
        "rarr" => "\u{2192}",
        "larr" => "\u{2190}",
        "zwj" => "\u{200d}",
        "zwnj" => "\u{200c}",
        "thinsp" => "\u{2009}",
        "ensp" => "\u{2002}",
        "emsp" => "\u{2003}",
        _ => return None,
    })
}

/// `html.unescape`: numeric references and the named entities pages use.
fn unescape(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        let tail = &rest[at + 1..];
        let end = tail.find(|c: char| !(c.is_ascii_alphanumeric() || c == '#')).unwrap_or(tail.len());
        let name = &tail[..end];
        let semi = tail[end..].starts_with(';');
        let decoded = if let Some(num) = name.strip_prefix('#') {
            let code = match num.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok(),
                None => num.parse::<u32>().ok(),
            };
            code.map(|c| if c == 0 || (0xd800..0xe000).contains(&c) { '\u{fffd}' } else { char::from_u32(c).unwrap_or('\u{fffd}') }.to_string())
        } else if semi || matches!(name, "amp" | "lt" | "gt" | "quot" | "nbsp" | "copy" | "reg") {
            named_entity(name).map(str::to_string)
        } else {
            None
        };
        match decoded {
            Some(s) if !name.is_empty() => {
                out.push_str(&s);
                rest = &tail[end + usize::from(semi)..];
            }
            _ => {
                out.push('&');
                rest = tail;
            }
        }
    }
    out.push_str(rest);
    out
}

fn truthy(v: Option<&Value>) -> Option<&Value> {
    v.filter(|v| match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.0),
    })
}

/// Markup, entities and stray whitespace out; cut at a word when too long.
fn text(value: Option<&Value>, limit: usize) -> String {
    let raw = match value {
        None | Some(Value::Null) => return String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items.iter().filter(|v| truthy(Some(v)).is_some()).map(py_str).collect::<Vec<_>>().join(" \u{2026} "),
        Some(other) => py_str(other),
    };
    let stripped = MARKUP.replace_all(&INLINE.replace_all(&raw, ""), " ").into_owned();
    let unescaped = unescape(&stripped);
    let text = SPACE.replace_all(&CONTROL.replace_all(&unescaped, " "), " ").trim().to_string();
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= limit {
        return text;
    }
    let head: String = chars[..limit].iter().collect();
    let cut = if chars[limit / 2..limit].contains(&' ') { head.rsplit_once(' ').map_or(head.clone(), |(a, _)| a.to_string()) } else { head };
    format!("{}\u{2026}", cut.trim_end_matches([' ', ',', ';', ':', '.']))
}

fn published(value: Option<&Value>) -> String {
    let Some(Value::String(s)) = value else { return String::new() };
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    match ISO_DATE.find(s) {
        Some(m) => m.as_str().to_string(),
        None => text(Some(&json!(s)), 40),
    }
}

/// `urllib.parse.urlsplit`, near enough: (scheme, netloc, path, query, fragment).
#[derive(Debug, Clone, Default)]
struct Split {
    scheme: String,
    netloc: String,
    path: String,
    query: String,
    fragment: String,
}

fn urlsplit(url: &str) -> Split {
    let mut split = Split::default();
    let mut rest = url;
    if let Some(colon) = rest.find(':') {
        let scheme = &rest[..colon];
        if !scheme.is_empty() && scheme.starts_with(|c: char| c.is_ascii_alphabetic()) && scheme.chars().all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c))
        {
            split.scheme = scheme.to_lowercase();
            rest = &rest[colon + 1..];
        }
    }
    if let Some(after) = rest.strip_prefix("//") {
        let end = after.find(['/', '?', '#']).unwrap_or(after.len());
        split.netloc = after[..end].to_string();
        rest = &after[end..];
    }
    if let Some((before, fragment)) = rest.split_once('#') {
        split.fragment = fragment.to_string();
        rest = before;
    }
    if let Some((before, query)) = rest.split_once('?') {
        split.query = query.to_string();
        rest = before;
    }
    split.path = rest.to_string();
    split
}

fn urlunsplit(s: &Split) -> String {
    let mut out = String::new();
    if !s.scheme.is_empty() {
        out.push_str(&s.scheme);
        out.push(':');
    }
    if !s.netloc.is_empty() {
        out.push_str("//");
        out.push_str(&s.netloc);
        if !s.path.is_empty() && !s.path.starts_with('/') {
            out.push('/');
        }
    }
    out.push_str(&s.path);
    if !s.query.is_empty() {
        out.push('?');
        out.push_str(&s.query);
    }
    if !s.fragment.is_empty() {
        out.push('#');
        out.push_str(&s.fragment);
    }
    out
}

/// `urlsplit(url).hostname`: lowercased, without user, port or brackets.
fn hostname(netloc: &str) -> String {
    let host = netloc.rsplit_once('@').map_or(netloc, |(_, h)| h);
    let host = if let Some(inner) = host.strip_prefix('[') { inner.split(']').next().unwrap_or("") } else { host.split(':').next().unwrap_or("") };
    host.to_lowercase()
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let hex = |b: u8| (b as char).to_digit(16);
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && let (Some(hi), Some(lo)) = (bytes.get(i + 1).and_then(|b| hex(*b)), bytes.get(i + 2).and_then(|b| hex(*b)))
        {
            out.push((hi * 16 + lo) as u8);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `urllib.parse.quote_plus`.
fn quote_plus(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => out.push(b as char),
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// `urllib.parse.parse_qsl(query, keep_blank_values=True)`.
fn parse_qsl(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (percent_decode(&k.replace('+', " ")), percent_decode(&v.replace('+', " ")))
        })
        .collect()
}

fn urlencode(pairs: &[(String, String)]) -> String {
    pairs.iter().map(|(k, v)| format!("{}={}", quote_plus(k), quote_plus(v))).collect::<Vec<_>>().join("&")
}

fn host_of(url: &str) -> String {
    hostname(&urlsplit(url).netloc).trim_end_matches('.').to_string()
}

/// `https://Docs.Example.com/page` → `docs.example.com`. Globs are kept.
pub(crate) fn domain(value: &str) -> String {
    let mut value = value.trim().to_lowercase();
    if value.is_empty() {
        return value;
    }
    if let Some((_, after)) = value.split_once("://") {
        value = after.to_string();
    }
    value = value.split('/').next().unwrap_or("").split('?').next().unwrap_or("").trim().to_string();
    if let Some(rest) = value.strip_prefix("site:") {
        value = rest.to_string();
    }
    if !value.starts_with('[')
        && let Some((head, port)) = value.rsplit_once(':')
        && !port.is_empty()
        && port.chars().all(|c| c.is_ascii_digit())
    {
        value = head.to_string();
    }
    value.trim_start_matches('.').trim_end_matches('.').to_string()
}

/// A plain domain covers its subdomains; a glob is matched as written.
pub(crate) fn matches(host: &str, pattern: &str) -> bool {
    if host.is_empty() || pattern.is_empty() {
        return false;
    }
    if GLOB.is_match(pattern) {
        return fnmatch(host, pattern);
    }
    let pattern = pattern.strip_prefix("www.").unwrap_or(pattern);
    host == pattern || host.ends_with(&format!(".{pattern}"))
}

/// The URL without tracking parameters or a text-fragment anchor.
fn canonical(url: &str) -> String {
    let mut parts = urlsplit(url);
    let query: Vec<(String, String)> = parse_qsl(&parts.query).into_iter().filter(|(k, _)| !TRACKING.is_match(k)).collect();
    parts.query = urlencode(&query);
    if parts.fragment.starts_with(":~:") {
        parts.fragment.clear();
    }
    urlunsplit(&parts)
}

/// Two URLs with the same identity are the same page.
fn identity(url: &str) -> String {
    let parts = urlsplit(url);
    let host = hostname(&parts.netloc);
    let host = host.strip_prefix("www.").unwrap_or(&host);
    let mut query = parse_qsl(&parts.query);
    query.sort();
    format!("{host}{}?{}", parts.path.trim_end_matches('/'), urlencode(&query))
}

fn redact(text: &str, secrets: &[String]) -> String {
    let mut text = text.to_string();
    for secret in secrets {
        if secret.chars().count() >= 4 {
            text = text.replace(secret.as_str(), "***");
        }
    }
    text
}

/// `data[path[0]][path[1]]…` as a list of mappings, whatever was there.
fn items<'a>(data: &'a Value, path: &[&str]) -> Vec<&'a Map<String, Value>> {
    let mut node = Some(data);
    for key in path {
        node = node.and_then(|n| n.as_object()).and_then(|o| o.get(*key));
    }
    node.and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_object).collect()).unwrap_or_default()
}

/// Python's `f"{x:g}"` for the numbers a message shows.
pub(crate) fn fmt_g(x: f64) -> String {
    let rounded = (x * 1e6).round() / 1e6;
    format!("{rounded}")
}

// ----------------------------------------------------------------------
// engines
// ----------------------------------------------------------------------

/// The arguments of one HTTP request to an engine.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EngineRequest {
    pub method: String,
    pub url: String,
    /// Query-string parameters.
    pub params: Vec<(String, String)>,
    /// A JSON body.
    pub json: Option<Value>,
    /// A form body (`application/x-www-form-urlencoded`).
    pub form: Option<Vec<(String, String)>>,
    pub headers: Vec<(String, String)>,
}

/// The key, address and headers an engine was built with. Unset fields fall
/// back to the environment and the engine's own address.
#[derive(Debug, Clone, Default)]
pub struct EngineSettings {
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub headers: Vec<(String, String)>,
}

fn pairs(items: &[(&str, String)]) -> Vec<(String, String)> {
    items.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

/// Map a transport error to a failure. The message of a transport error can
/// carry the URL, and for one engine the URL carries the key — so only the
/// kind is reported.
fn classify(e: &reqwest::Error) -> SearchFailure {
    if e.is_timeout() {
        return SearchFailure::new("timed out").retryable();
    }
    let kind = if e.is_connect() {
        "ConnectError"
    } else if e.is_body() {
        "ReadError"
    } else if e.is_decode() {
        "DecodingError"
    } else if e.is_request() {
        "RequestError"
    } else {
        "TransportError"
    };
    SearchFailure::new(format!("could not be reached ({kind})")).retryable()
}

/// One search service, spoken to over HTTP.
///
/// Implement it for an engine that is not here: say how the request is built
/// ([`SearchEngine::request`]) and how the answer is read
/// ([`SearchEngine::parse`]), and [`WebSearch`] brings the retries, the
/// fallback, the domain policy and the cache. Override
/// [`SearchEngine::search`] for an engine that is not HTTP at all.
///
/// Results are JSON mappings with `title`, `url` (or `link`), `snippet` (or
/// `content` / `description`), `published` and `score` — whichever the engine has.
#[async_trait]
pub trait SearchEngine: Send + Sync {
    fn name(&self) -> String;

    /// The service's own address.
    fn url(&self) -> String {
        String::new()
    }

    /// Environment variables read for the key, the first that is set winning.
    fn env(&self) -> Vec<String> {
        Vec::new()
    }

    fn needs_key(&self) -> bool {
        true
    }

    /// The most results one request may ask for.
    fn max_results(&self) -> usize {
        20
    }

    /// The key, address and headers it was built with.
    fn settings(&self) -> Option<&EngineSettings> {
        None
    }

    /// The key: the one it was given, else the environment's — read when the
    /// engine is used, so a tool built early sees a key set afterwards.
    fn api_key(&self) -> String {
        if let Some(key) = self.settings().and_then(|s| s.api_key.clone()).filter(|k| !k.is_empty()) {
            return key;
        }
        self.env().iter().find_map(|v| env(v)).unwrap_or_default()
    }

    fn base_url(&self) -> String {
        self.settings().and_then(|s| s.base_url.clone()).unwrap_or_else(|| self.url()).trim_end_matches('/').to_string()
    }

    /// Empty when the engine can be called; otherwise what is missing.
    fn ready(&self) -> String {
        if self.needs_key() && self.api_key().is_empty() {
            let env = self.env();
            return format!("no API key \u{2014} set {}", if env.is_empty() { "api_key=".to_string() } else { env.join(" or ") });
        }
        if self.base_url().is_empty() {
            return "no URL to call".into();
        }
        String::new()
    }

    fn secrets(&self) -> Vec<String> {
        vec![self.api_key()]
    }

    /// The environment variables that set this engine up.
    fn needs(&self) -> Vec<String> {
        let env = self.env();
        if self.needs_key() && !env.is_empty() { vec![env.join(" or ")] } else { Vec::new() }
    }

    /// The HTTP request for a query.
    fn request(&self, _query: &SearchQuery) -> std::result::Result<EngineRequest, SearchFailure> {
        Err(SearchFailure::new("this engine builds no request"))
    }

    /// The engine's JSON, as a list of result mappings.
    fn parse(&self, _data: &Value, _query: &SearchQuery) -> std::result::Result<Vec<Value>, SearchFailure> {
        Ok(Vec::new())
    }

    /// A failure for an answer that is not a result.
    fn check(&self, status: u16, headers: &HeaderMap, body: &str) -> std::result::Result<(), SearchFailure> {
        check_status(self.needs_key(), status, headers, body)
    }

    /// The results in a successful answer.
    fn read(&self, _status: u16, _headers: &HeaderMap, body: &str, query: &SearchQuery) -> std::result::Result<Vec<Value>, SearchFailure> {
        let data: Value = serde_json::from_str(body).map_err(|_| SearchFailure::new("answered with something that is not JSON"))?;
        self.parse(&data, query)
    }

    async fn search(&self, query: &SearchQuery, client: &reqwest::Client) -> std::result::Result<Vec<Value>, SearchFailure> {
        let request = self.request(query)?;
        let method = reqwest::Method::from_bytes(request.method.to_uppercase().as_bytes()).map_err(|e| SearchFailure::new(e.to_string()))?;
        let mut builder = client.request(method, &request.url);
        let mut headers: IndexMap<String, String> = IndexMap::new();
        for (k, v) in self.settings().map(|s| s.headers.clone()).unwrap_or_default().into_iter().chain(request.headers) {
            headers.insert(k.to_lowercase(), v);
        }
        for (k, v) in &headers {
            builder = builder.header(k, v);
        }
        if !request.params.is_empty() {
            builder = builder.query(&request.params);
        }
        if let Some(body) = &request.json {
            builder = builder.json(body);
        }
        if let Some(form) = &request.form {
            builder = builder.header(reqwest::header::CONTENT_TYPE, "application/x-www-form-urlencoded").body(urlencode(form));
        }
        let response = builder.send().await.map_err(|e| classify(&e))?;
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let body = response.text().await.map_err(|e| classify(&e))?;
        self.check(status, &headers, &body)?;
        self.read(status, &headers, &body, query)
    }
}

/// The checks every HTTP engine makes of an answer's status.
fn check_status(needs_key: bool, status: u16, headers: &HeaderMap, body: &str) -> std::result::Result<(), SearchFailure> {
    if (200..300).contains(&status) {
        return Ok(());
    }
    let head: String = body.chars().take(300).collect();
    let detail = text(Some(&json!(head)), 200);
    if status == 401 || status == 403 {
        let message = if needs_key { "the API key was refused".to_string() } else { format!("the request was refused ({status})") };
        return Err(SearchFailure::new(message).status(status));
    }
    if status == 402 || (status == 429 && QUOTA.is_match(&detail)) {
        return Err(SearchFailure::new("out of quota").status(status));
    }
    if status == 429 {
        let mut f = SearchFailure::new("rate limited").retryable().status(status);
        f.retry_after = retry_after(Some(headers), None, true);
        return Err(f);
    }
    if status == 408 || status >= 500 {
        let mut f = SearchFailure::new(format!("the engine answered {status}")).retryable().status(status);
        f.retry_after = retry_after(Some(headers), None, false);
        return Err(f);
    }
    if (300..400).contains(&status) {
        return Err(SearchFailure::new(format!("redirected ({status}) \u{2014} check the engine's URL")).status(status));
    }
    let message = if detail.is_empty() { format!("refused the query ({status})") } else { format!("refused the query ({status}): {detail}") };
    Err(SearchFailure::new(message).healthy().status(status))
}

macro_rules! engine_struct {
    ($(#[$doc:meta])* $ty:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Default)]
        pub struct $ty {
            pub settings: EngineSettings,
        }

        impl $ty {
            /// With a key of its own, or `None` to read the environment.
            pub fn new(api_key: Option<&str>) -> Self {
                Self { settings: EngineSettings { api_key: api_key.map(str::to_string), ..Default::default() } }
            }

            /// Call another address — a proxy, or a fake in tests.
            pub fn with_base_url(mut self, url: &str) -> Self {
                self.settings.base_url = Some(url.to_string());
                self
            }

            /// Send this header with every request.
            pub fn with_header(mut self, name: &str, value: &str) -> Self {
                self.settings.headers.push((name.to_string(), value.to_string()));
                self
            }
        }
    };
}

fn strings(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

fn get(map: &Map<String, Value>, key: &str) -> Value {
    map.get(key).cloned().unwrap_or(Value::Null)
}

engine_struct!(
    /// Tavily: `TAVILY_API_KEY`.
    Tavily
);

#[async_trait]
impl SearchEngine for Tavily {
    fn name(&self) -> String {
        "tavily".into()
    }
    fn url(&self) -> String {
        "https://api.tavily.com".into()
    }
    fn env(&self) -> Vec<String> {
        strings(&["TAVILY_API_KEY"])
    }
    fn settings(&self) -> Option<&EngineSettings> {
        Some(&self.settings)
    }
    fn request(&self, query: &SearchQuery) -> std::result::Result<EngineRequest, SearchFailure> {
        let mut body = json!({"query": query.text, "max_results": query.limit, "search_depth": "basic"});
        if !query.recency.is_empty() {
            body["time_range"] = json!(query.recency);
        }
        if !query.include_domains().is_empty() {
            body["include_domains"] = json!(query.include_domains());
        }
        if !query.exclude_domains().is_empty() {
            body["exclude_domains"] = json!(query.exclude_domains());
        }
        Ok(EngineRequest {
            method: "POST".into(),
            url: format!("{}/search", self.base_url()),
            json: Some(body),
            headers: pairs(&[("Authorization", format!("Bearer {}", self.api_key()))]),
            ..Default::default()
        })
    }
    fn parse(&self, data: &Value, _query: &SearchQuery) -> std::result::Result<Vec<Value>, SearchFailure> {
        Ok(items(data, &["results"])
            .into_iter()
            .map(|r| json!({"title": get(r, "title"), "url": get(r, "url"), "snippet": get(r, "content"), "published": get(r, "published_date"), "score": get(r, "score")}))
            .collect())
    }
}

engine_struct!(
    /// Brave Search: `BRAVE_API_KEY` or `BRAVE_SEARCH_API_KEY`.
    Brave
);

#[async_trait]
impl SearchEngine for Brave {
    fn name(&self) -> String {
        "brave".into()
    }
    fn url(&self) -> String {
        "https://api.search.brave.com".into()
    }
    fn env(&self) -> Vec<String> {
        strings(&["BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY"])
    }
    fn settings(&self) -> Option<&EngineSettings> {
        Some(&self.settings)
    }
    fn request(&self, query: &SearchQuery) -> std::result::Result<EngineRequest, SearchFailure> {
        let mut params =
            pairs(&[("q", query.with_sites()), ("count", query.limit.to_string()), ("safesearch", query.safe.clone()), ("text_decorations", "false".into())]);
        if !query.recency.is_empty() {
            params.push(("freshness".into(), format!("p{}", &query.recency[..1])));
        }
        if !query.region.is_empty() {
            params.push(("country".into(), query.region.to_uppercase()));
        }
        if !query.language.is_empty() {
            params.push(("search_lang".into(), query.language.clone()));
        }
        Ok(EngineRequest {
            method: "GET".into(),
            url: format!("{}/res/v1/web/search", self.base_url()),
            params,
            headers: pairs(&[("X-Subscription-Token", self.api_key()), ("Accept", "application/json".into())]),
            ..Default::default()
        })
    }
    fn parse(&self, data: &Value, _query: &SearchQuery) -> std::result::Result<Vec<Value>, SearchFailure> {
        Ok(items(data, &["web", "results"])
            .into_iter()
            .map(|r| {
                let snippet = truthy(r.get("description")).or(r.get("extra_snippets")).cloned().unwrap_or(Value::Null);
                let date = truthy(r.get("page_age")).or(r.get("age")).cloned().unwrap_or(Value::Null);
                json!({"title": get(r, "title"), "url": get(r, "url"), "snippet": snippet, "published": date})
            })
            .collect())
    }
}

engine_struct!(
    /// Exa: `EXA_API_KEY`.
    Exa
);

#[async_trait]
impl SearchEngine for Exa {
    fn name(&self) -> String {
        "exa".into()
    }
    fn url(&self) -> String {
        "https://api.exa.ai".into()
    }
    fn env(&self) -> Vec<String> {
        strings(&["EXA_API_KEY"])
    }
    fn max_results(&self) -> usize {
        25
    }
    fn settings(&self) -> Option<&EngineSettings> {
        Some(&self.settings)
    }
    fn request(&self, query: &SearchQuery) -> std::result::Result<EngineRequest, SearchFailure> {
        let mut body =
            json!({"query": query.text, "numResults": query.limit, "type": "auto", "contents": {"text": {"maxCharacters": query.snippet_chars.max(200)}}});
        if !query.recency.is_empty() {
            let since = jiff::Timestamp::now() - jiff::SignedDuration::from_hours(24 * query.days());
            body["startPublishedDate"] = json!(since.strftime("%Y-%m-%dT%H:%M:%S.000Z").to_string());
        }
        if !query.include_domains().is_empty() {
            body["includeDomains"] = json!(query.include_domains());
        }
        if !query.exclude_domains().is_empty() {
            body["excludeDomains"] = json!(query.exclude_domains());
        }
        Ok(EngineRequest {
            method: "POST".into(),
            url: format!("{}/search", self.base_url()),
            json: Some(body),
            headers: pairs(&[("x-api-key", self.api_key())]),
            ..Default::default()
        })
    }
    fn parse(&self, data: &Value, _query: &SearchQuery) -> std::result::Result<Vec<Value>, SearchFailure> {
        Ok(items(data, &["results"])
            .into_iter()
            .map(|r| {
                let snippet = truthy(r.get("summary")).or(truthy(r.get("highlights"))).or(r.get("text")).cloned().unwrap_or(Value::Null);
                json!({"title": get(r, "title"), "url": get(r, "url"), "snippet": snippet, "published": get(r, "publishedDate"), "score": get(r, "score")})
            })
            .collect())
    }
}

engine_struct!(
    /// Google results, through serper.dev: `SERPER_API_KEY`.
    Serper
);

#[async_trait]
impl SearchEngine for Serper {
    fn name(&self) -> String {
        "serper".into()
    }
    fn url(&self) -> String {
        "https://google.serper.dev".into()
    }
    fn env(&self) -> Vec<String> {
        strings(&["SERPER_API_KEY"])
    }
    fn settings(&self) -> Option<&EngineSettings> {
        Some(&self.settings)
    }
    fn request(&self, query: &SearchQuery) -> std::result::Result<EngineRequest, SearchFailure> {
        let mut body = json!({"q": query.with_sites(), "num": query.limit});
        if !query.recency.is_empty() {
            body["tbs"] = json!(format!("qdr:{}", &query.recency[..1]));
        }
        if !query.region.is_empty() {
            body["gl"] = json!(query.region.to_lowercase());
        }
        if !query.language.is_empty() {
            body["hl"] = json!(query.language);
        }
        Ok(EngineRequest {
            method: "POST".into(),
            url: format!("{}/search", self.base_url()),
            json: Some(body),
            headers: pairs(&[("X-API-KEY", self.api_key())]),
            ..Default::default()
        })
    }
    fn parse(&self, data: &Value, _query: &SearchQuery) -> std::result::Result<Vec<Value>, SearchFailure> {
        Ok(items(data, &["organic"])
            .into_iter()
            .map(|r| json!({"title": get(r, "title"), "url": get(r, "link"), "snippet": get(r, "snippet"), "published": get(r, "date")}))
            .collect())
    }
}

/// Google's Programmable Search Engine: a key (`GOOGLE_SEARCH_API_KEY` or
/// `GOOGLE_API_KEY`) and the id of an engine (`GOOGLE_CSE_ID` or
/// `GOOGLE_SEARCH_ENGINE_ID`).
#[derive(Debug, Clone, Default)]
pub struct Google {
    pub settings: EngineSettings,
    pub cx: Option<String>,
}

impl Google {
    pub fn new(api_key: Option<&str>, cx: Option<&str>) -> Self {
        Self { settings: EngineSettings { api_key: api_key.map(str::to_string), ..Default::default() }, cx: cx.map(str::to_string) }
    }

    pub fn with_base_url(mut self, url: &str) -> Self {
        self.settings.base_url = Some(url.to_string());
        self
    }

    /// The search engine id: the one it was given, else the environment's.
    pub fn cx(&self) -> String {
        self.cx.clone().filter(|c| !c.is_empty()).or_else(|| ["GOOGLE_CSE_ID", "GOOGLE_SEARCH_ENGINE_ID"].iter().find_map(|v| env(v))).unwrap_or_default()
    }
}

#[async_trait]
impl SearchEngine for Google {
    fn name(&self) -> String {
        "google".into()
    }
    fn url(&self) -> String {
        "https://www.googleapis.com".into()
    }
    fn env(&self) -> Vec<String> {
        strings(&["GOOGLE_SEARCH_API_KEY", "GOOGLE_API_KEY"])
    }
    fn max_results(&self) -> usize {
        10
    }
    fn settings(&self) -> Option<&EngineSettings> {
        Some(&self.settings)
    }
    fn ready(&self) -> String {
        if self.cx().is_empty() {
            return "no search engine id \u{2014} set GOOGLE_CSE_ID".into();
        }
        if self.api_key().is_empty() {
            return "no API key \u{2014} set GOOGLE_SEARCH_API_KEY or GOOGLE_API_KEY".into();
        }
        String::new()
    }
    fn needs(&self) -> Vec<String> {
        strings(&["GOOGLE_SEARCH_API_KEY", "GOOGLE_CSE_ID"])
    }
    fn request(&self, query: &SearchQuery) -> std::result::Result<EngineRequest, SearchFailure> {
        let safe = if query.safe == "off" { "off" } else { "active" };
        let mut params =
            pairs(&[("key", self.api_key()), ("cx", self.cx()), ("q", query.with_sites()), ("num", query.limit.to_string()), ("safe", safe.into())]);
        if !query.recency.is_empty() {
            params.push(("dateRestrict".into(), format!("{}1", &query.recency[..1])));
        }
        if !query.region.is_empty() {
            params.push(("gl".into(), query.region.to_lowercase()));
        }
        if !query.language.is_empty() {
            params.push(("lr".into(), format!("lang_{}", query.language)));
        }
        Ok(EngineRequest { method: "GET".into(), url: format!("{}/customsearch/v1", self.base_url()), params, ..Default::default() })
    }
    fn parse(&self, data: &Value, _query: &SearchQuery) -> std::result::Result<Vec<Value>, SearchFailure> {
        Ok(items(data, &["items"])
            .into_iter()
            .map(|r| {
                let item = Value::Object(r.clone());
                let tags = items(&item, &["pagemap", "metatags"]);
                let date = tags.first().map_or(json!(""), |t| get(t, "article:published_time"));
                json!({"title": get(r, "title"), "url": get(r, "link"), "snippet": get(r, "snippet"), "published": date})
            })
            .collect())
    }
}

engine_struct!(
    /// A SearXNG instance of your own (`SEARXNG_URL`, optional
    /// `SEARXNG_API_KEY`), with the JSON format switched on.
    SearXNG
);

#[async_trait]
impl SearchEngine for SearXNG {
    fn name(&self) -> String {
        "searxng".into()
    }
    fn env(&self) -> Vec<String> {
        strings(&["SEARXNG_API_KEY"])
    }
    fn needs_key(&self) -> bool {
        false
    }
    fn settings(&self) -> Option<&EngineSettings> {
        Some(&self.settings)
    }
    fn base_url(&self) -> String {
        self.settings.base_url.clone().or_else(|| env("SEARXNG_URL")).unwrap_or_default().trim_end_matches('/').to_string()
    }
    fn ready(&self) -> String {
        if self.base_url().is_empty() { "no instance to call \u{2014} set SEARXNG_URL".into() } else { String::new() }
    }
    fn needs(&self) -> Vec<String> {
        strings(&["SEARXNG_URL"])
    }
    fn request(&self, query: &SearchQuery) -> std::result::Result<EngineRequest, SearchFailure> {
        let safe = SAFE.iter().position(|s| *s == query.safe).unwrap_or(1);
        let mut params = pairs(&[("q", query.with_sites()), ("format", "json".into()), ("safesearch", safe.to_string())]);
        if !query.recency.is_empty() {
            params.push(("time_range".into(), query.recency.clone()));
        }
        if !query.language.is_empty() {
            params.push(("language".into(), query.language.clone()));
        }
        let key = self.api_key();
        let headers = if key.is_empty() { Vec::new() } else { pairs(&[("Authorization", format!("Bearer {key}"))]) };
        Ok(EngineRequest { method: "GET".into(), url: format!("{}/search", self.base_url()), params, headers, ..Default::default() })
    }
    fn check(&self, status: u16, headers: &HeaderMap, body: &str) -> std::result::Result<(), SearchFailure> {
        if status == 403 {
            return Err(SearchFailure::new("the instance refused the request (403) \u{2014} the JSON format must be enabled in its settings").status(403));
        }
        check_status(false, status, headers, body)
    }
    fn parse(&self, data: &Value, _query: &SearchQuery) -> std::result::Result<Vec<Value>, SearchFailure> {
        Ok(items(data, &["results"])
            .into_iter()
            .map(|r| json!({"title": get(r, "title"), "url": get(r, "url"), "snippet": get(r, "content"), "published": get(r, "publishedDate"), "score": get(r, "score")}))
            .collect())
    }
}

static DDG_ANCHOR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<a\b([^>]*)>(.*?)</a>").expect("valid"));
static DDG_TITLE_CLASS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"\bclass="[^"]*\bresult__a\b"#).expect("valid"));
static DDG_SNIPPET_CLASS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"\bclass="[^"]*\bresult__snippet\b"#).expect("valid"));
static DDG_SNIPPETS: LazyLock<[Regex; 3]> =
    LazyLock::new(|| ["a", "div", "td"].map(|t| Regex::new(&format!(r"(?s)<{t}\b([^>]*)>(.*?)</{t}>")).expect("valid")));
static DDG_DATE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<span>(?:\s|&nbsp;)*(\d{4}-\d{2}-\d{2})T[\d:.]+\s*</span>").expect("valid"));
static HREF: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"\bhref="([^"]*)""#).expect("valid"));
const DDG_BLOCKED: [&str; 3] = ["anomaly-modal", "challenge-form", "bots use DuckDuckGo too"];

engine_struct!(
    /// DuckDuckGo's HTML results page. No key — and no promise of availability.
    ///
    /// This is a page meant for browsers, read here because it is the one
    /// search that works with nothing set up. It throttles heavy use and its
    /// layout can change; both are reported as failures, so a fallback engine
    /// takes over. For anything that matters, give the search a key.
    DuckDuckGo
);

impl DuckDuckGo {
    /// The page behind DuckDuckGo's redirect link; nothing for an advert.
    fn target(href: &str) -> String {
        let mut href = unescape(href);
        if href.starts_with("//") {
            href = format!("https:{href}");
        }
        let parts = urlsplit(&href);
        if hostname(&parts.netloc).ends_with("duckduckgo.com") {
            if parts.path.starts_with("/y.js") {
                return String::new();
            }
            let target = parse_qsl(&parts.query).into_iter().rev().find(|(k, _)| k == "uddg").map(|(_, v)| v).unwrap_or_default();
            let head: String = target.chars().take(12).collect();
            return if head.to_uppercase().contains("%3A") { percent_decode(&target) } else { target };
        }
        href
    }
}

#[async_trait]
impl SearchEngine for DuckDuckGo {
    fn name(&self) -> String {
        "duckduckgo".into()
    }
    fn url(&self) -> String {
        "https://html.duckduckgo.com".into()
    }
    fn needs_key(&self) -> bool {
        false
    }
    fn max_results(&self) -> usize {
        30
    }
    fn settings(&self) -> Option<&EngineSettings> {
        Some(&self.settings)
    }
    fn request(&self, query: &SearchQuery) -> std::result::Result<EngineRequest, SearchFailure> {
        let kp = match query.safe.as_str() {
            "strict" => "1",
            "off" => "-2",
            _ => "-1",
        };
        let mut form = pairs(&[("q", query.with_sites()), ("kp", kp.into())]);
        if !query.recency.is_empty() {
            form.push(("df".into(), query.recency[..1].to_string()));
        }
        if !query.region.is_empty() {
            let lang = if query.language.is_empty() { "en" } else { &query.language };
            form.push(("kl".into(), format!("{}-{lang}", query.region.to_lowercase())));
        }
        Ok(EngineRequest { method: "POST".into(), url: format!("{}/html/", self.base_url()), form: Some(form), ..Default::default() })
    }
    fn read(&self, status: u16, _headers: &HeaderMap, page: &str, _query: &SearchQuery) -> std::result::Result<Vec<Value>, SearchFailure> {
        if status == 202 || DDG_BLOCKED.iter().any(|m| page.contains(m)) {
            return Err(SearchFailure::new("DuckDuckGo is throttling this address \u{2014} give the search an API key, or try again later"));
        }
        let titles: Vec<regex::Captures> = DDG_ANCHOR.captures_iter(page).filter(|c| DDG_TITLE_CLASS.is_match(&c[1])).collect();
        let mut out = Vec::new();
        for (index, m) in titles.iter().enumerate() {
            let Some(href) = HREF.captures(&m[1]) else { continue };
            let whole = m.get(0).expect("match");
            let end = titles.get(index + 1).map_or(page.len(), |n| n.get(0).expect("match").start());
            let block = &page[whole.end()..end];
            let snippet = DDG_SNIPPETS
                .iter()
                .filter_map(|re| re.captures_iter(block).find(|c| DDG_SNIPPET_CLASS.is_match(&c[1])))
                .min_by_key(|c| c.get(0).map_or(usize::MAX, |g| g.start()))
                .map(|c| c[2].to_string())
                .unwrap_or_default();
            let date = DDG_DATE.captures(block).map(|c| c[1].to_string()).unwrap_or_default();
            out.push(json!({"title": &m[2], "url": Self::target(&href[1]), "snippet": snippet, "published": date}));
        }
        if out.is_empty() && !page.contains("no-results") && !page.contains("No results") {
            return Err(SearchFailure::new("the results page could not be read \u{2014} its layout may have changed"));
        }
        Ok(out)
    }
}

type SearchFn = dyn Fn(SearchQuery) -> BoxFuture<'static, std::result::Result<Vec<Value>, SearchFailure>> + Send + Sync;

/// A function of your own as an engine: `(query) -> results`.
pub struct FunctionEngine {
    name: String,
    f: Arc<SearchFn>,
}

impl FunctionEngine {
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(SearchQuery) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = std::result::Result<Vec<Value>, SearchFailure>> + Send + 'static,
    {
        let name = name.into();
        Self { name: if name.is_empty() { "custom".into() } else { name }, f: Arc::new(move |q| f(q).boxed()) }
    }
}

#[async_trait]
impl SearchEngine for FunctionEngine {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn needs_key(&self) -> bool {
        false
    }
    fn ready(&self) -> String {
        String::new()
    }
    async fn search(&self, query: &SearchQuery, _client: &reqwest::Client) -> std::result::Result<Vec<Value>, SearchFailure> {
        (self.f)(query.clone()).await
    }
}

/// An engine by name (`ddg`, `searx`, `google_cse` and `serper.dev` are
/// understood too), with a key of its own or `None` for the environment's.
pub fn engine_named(name: &str, api_key: Option<&str>) -> Result<Arc<dyn SearchEngine>> {
    let lowered = name.trim().to_lowercase();
    let canonical = match lowered.as_str() {
        "ddg" | "duck" => "duckduckgo",
        "searx" => "searxng",
        "google_cse" | "google-cse" => "google",
        "serper.dev" => "serper",
        other => other,
    };
    let key = api_key.filter(|k| !k.is_empty());
    Ok(match canonical {
        "tavily" => Arc::new(Tavily::new(key)),
        "brave" => Arc::new(Brave::new(key)),
        "exa" => Arc::new(Exa::new(key)),
        "serper" => Arc::new(Serper::new(key)),
        "google" => Arc::new(Google::new(key, None)),
        "searxng" => Arc::new(SearXNG::new(key)),
        "duckduckgo" => Arc::new(DuckDuckGo::new(key)),
        _ => return Err(Error::config(format!("no search engine named {}; there are: {}", py_str_repr(name), ENGINES.join(", ")))),
    })
}

/// Every engine that ships, and whether this environment can call it:
/// `[{"name", "ready", "missing", "needs"}]`.
pub fn search_engines() -> Vec<Value> {
    ENGINES
        .iter()
        .map(|name| {
            let engine = engine_named(name, None).expect("a shipped engine");
            let missing = engine.ready();
            json!({"name": name, "ready": missing.is_empty(), "missing": missing, "needs": engine.needs()})
        })
        .collect()
}

// ----------------------------------------------------------------------
// the search
// ----------------------------------------------------------------------

/// How one engine has fared: searches answered and failed, and the last error.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct EngineStats {
    pub answered: u64,
    pub failed: u64,
    pub last_error: String,
}

type CacheKey = (String, usize, String, Vec<String>);
type Pending = Shared<BoxFuture<'static, std::result::Result<Vec<SearchResult>, String>>>;

/// An engine as named in a [`WebSearchBuilder`].
#[derive(Clone)]
enum EngineSpec {
    Name(String),
    Engine(Arc<dyn SearchEngine>),
}

/// Settings for a [`WebSearch`]. Every field has a working default.
#[derive(Clone)]
pub struct WebSearchBuilder {
    engines: Vec<EngineSpec>,
    api_key: Option<String>,
    limit: usize,
    max_limit: usize,
    allowed: Vec<String>,
    blocked: Vec<String>,
    region: String,
    language: String,
    safe_search: String,
    snippet_chars: usize,
    timeout: f64,
    deadline: f64,
    retries: u32,
    retry: Option<RetryPolicy>,
    cache_ttl: f64,
    cache_size: usize,
    min_interval: f64,
    fallback_on_empty: bool,
    failure_threshold: u32,
    cooldown: f64,
    client: Option<reqwest::Client>,
    name: String,
}

impl Default for WebSearchBuilder {
    fn default() -> Self {
        Self {
            engines: Vec::new(),
            api_key: None,
            limit: 5,
            max_limit: 10,
            allowed: Vec::new(),
            blocked: Vec::new(),
            region: String::new(),
            language: String::new(),
            safe_search: "moderate".into(),
            snippet_chars: 500,
            timeout: 15.0,
            deadline: 45.0,
            retries: 2,
            retry: None,
            cache_ttl: 300.0,
            cache_size: 256,
            min_interval: 0.0,
            fallback_on_empty: true,
            failure_threshold: 3,
            cooldown: 60.0,
            client: None,
            name: "web_search".into(),
        }
    }
}

impl WebSearchBuilder {
    /// Add an engine by name; several fall back in the order added. None at
    /// all: `$AGENT_HARNESS_SEARCH`, else every engine with a key set, then DuckDuckGo.
    pub fn engine(mut self, name: &str) -> Self {
        self.engines.push(EngineSpec::Name(name.to_string()));
        self
    }

    /// Add an engine you built (a key of its own, another address, your own type).
    pub fn engine_impl(mut self, engine: impl SearchEngine + 'static) -> Self {
        self.engines.push(EngineSpec::Engine(Arc::new(engine)));
        self
    }

    pub fn engine_arc(mut self, engine: Arc<dyn SearchEngine>) -> Self {
        self.engines.push(EngineSpec::Engine(engine));
        self
    }

    /// A function of your own as an engine.
    pub fn function<F, Fut>(self, name: &str, f: F) -> Self
    where
        F: Fn(SearchQuery) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = std::result::Result<Vec<Value>, SearchFailure>> + Send + 'static,
    {
        self.engine_impl(FunctionEngine::new(name, f))
    }

    /// The key for the one named engine.
    pub fn api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    /// Results when the model does not say.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = n;
        self
    }

    /// The most results the model may ask for.
    pub fn max_limit(mut self, n: usize) -> Self {
        self.max_limit = n;
        self
    }

    /// Only these domains. A plain name covers its subdomains; globs work.
    /// The model can narrow a search to some domains, never widen it past these.
    pub fn allowed_domains<I: IntoIterator<Item = S>, S: AsRef<str>>(mut self, domains: I) -> Self {
        self.allowed = domains.into_iter().map(|d| d.as_ref().to_string()).collect();
        self
    }

    pub fn blocked_domains<I: IntoIterator<Item = S>, S: AsRef<str>>(mut self, domains: I) -> Self {
        self.blocked = domains.into_iter().map(|d| d.as_ref().to_string()).collect();
        self
    }

    pub fn region(mut self, region: &str) -> Self {
        self.region = region.trim().to_string();
        self
    }

    pub fn language(mut self, language: &str) -> Self {
        self.language = language.trim().to_string();
        self
    }

    /// "off", "moderate" or "strict".
    pub fn safe_search(mut self, level: &str) -> Self {
        self.safe_search = level.to_string();
        self
    }

    pub fn snippet_chars(mut self, n: usize) -> Self {
        self.snippet_chars = n;
        self
    }

    /// Seconds for one request.
    pub fn timeout(mut self, seconds: f64) -> Self {
        self.timeout = seconds;
        self
    }

    /// Seconds for the whole search, retries and fallbacks included.
    pub fn deadline(mut self, seconds: f64) -> Self {
        self.deadline = seconds;
        self
    }

    /// Tries after the first, per engine.
    pub fn retries(mut self, n: u32) -> Self {
        self.retries = n;
        self
    }

    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.retry = Some(policy);
        self
    }

    /// Seconds an answer is kept; 0 turns the cache off.
    pub fn cache_ttl(mut self, seconds: f64) -> Self {
        self.cache_ttl = seconds;
        self
    }

    pub fn cache_size(mut self, n: usize) -> Self {
        self.cache_size = n;
        self
    }

    /// Seconds between two requests to one engine.
    pub fn min_interval(mut self, seconds: f64) -> Self {
        self.min_interval = seconds;
        self
    }

    /// Ask the next engine when one finds nothing.
    pub fn fallback_on_empty(mut self, yes: bool) -> Self {
        self.fallback_on_empty = yes;
        self
    }

    /// Failures in a row before an engine is left alone for `cooldown` seconds.
    pub fn failure_threshold(mut self, n: u32) -> Self {
        self.failure_threshold = n;
        self
    }

    pub fn cooldown(mut self, seconds: f64) -> Self {
        self.cooldown = seconds;
        self
    }

    /// Use this HTTP client instead of one per search.
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// The tool name, and the name errors carry.
    pub fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Check the settings and build the search. Names are checked now, so a
    /// typo fails where it was written; keys are read when a search is made.
    pub fn build(self) -> Result<WebSearch> {
        if !SAFE.contains(&self.safe_search.as_str()) {
            return Err(Error::config(format!("safe_search is one of {} \u{2014} got {}", SAFE.join(", "), py_str_repr(&self.safe_search))));
        }
        if self.api_key.is_some() && self.engines.len() != 1 {
            return Err(Error::config("api_key= goes with one named engine; for several, pass each as an engine of its own \u{2014} Brave::new(Some(key))"));
        }
        let engines = if self.engines.is_empty() {
            None
        } else {
            let mut out = Vec::new();
            for spec in &self.engines {
                out.push(match spec {
                    EngineSpec::Name(n) => engine_named(n, self.api_key.as_deref())?,
                    EngineSpec::Engine(e) => e.clone(),
                });
            }
            Some(out)
        };
        let max_limit = self.max_limit.max(1);
        let retry = self.retry.clone().unwrap_or(RetryPolicy {
            max_retries: self.retries,
            initial_delay: 0.5,
            max_delay: 8.0,
            max_retry_after: 10.0,
            ..RetryPolicy::default()
        });
        Ok(WebSearch {
            inner: Arc::new(Inner {
                engines: Mutex::new(engines),
                name: Mutex::new(self.name.clone()),
                max_limit,
                limit: self.limit.clamp(1, max_limit),
                allowed: self.allowed.iter().map(|d| domain(d)).filter(|d| !d.is_empty()).collect(),
                blocked: self.blocked.iter().map(|d| domain(d)).filter(|d| !d.is_empty()).collect(),
                region: self.region,
                language: self.language,
                safe_search: self.safe_search,
                snippet_chars: self.snippet_chars.max(80),
                timeout: self.timeout,
                deadline: self.deadline,
                retry,
                cache_ttl: self.cache_ttl,
                cache_size: self.cache_size.max(1),
                min_interval: self.min_interval,
                fallback_on_empty: self.fallback_on_empty,
                failure_threshold: self.failure_threshold,
                cooldown: self.cooldown,
                client: self.client,
                breakers: Mutex::new(HashMap::new()),
                cache: Mutex::new(IndexMap::new()),
                inflight: Mutex::new(HashMap::new()),
                next_call: Mutex::new(HashMap::new()),
                last_engine: Mutex::new(String::new()),
                stats: Mutex::new(IndexMap::new()),
            }),
        })
    }
}

struct Inner {
    engines: Mutex<Option<Vec<Arc<dyn SearchEngine>>>>,
    name: Mutex<String>,
    limit: usize,
    max_limit: usize,
    allowed: Vec<String>,
    blocked: Vec<String>,
    region: String,
    language: String,
    safe_search: String,
    snippet_chars: usize,
    timeout: f64,
    deadline: f64,
    retry: RetryPolicy,
    cache_ttl: f64,
    cache_size: usize,
    min_interval: f64,
    fallback_on_empty: bool,
    failure_threshold: u32,
    cooldown: f64,
    client: Option<reqwest::Client>,
    breakers: Mutex<HashMap<String, CircuitBreaker>>,
    cache: Mutex<IndexMap<CacheKey, (Instant, Vec<SearchResult>)>>,
    inflight: Mutex<HashMap<CacheKey, Pending>>,
    next_call: Mutex<HashMap<String, Instant>>,
    last_engine: Mutex<String>,
    stats: Mutex<IndexMap<String, EngineStats>>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// A web search with retries, fallback engines, a domain policy and a cache.
///
/// Cheap to clone: clones share the cache, the engines' health and the stats.
#[derive(Clone)]
pub struct WebSearch {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for WebSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let engines = lock(&self.inner.engines).as_ref().map(|e| e.iter().map(|e| e.name()).collect::<Vec<_>>().join(", "));
        write!(f, "<WebSearch {}>", engines.unwrap_or_else(|| "auto".into()))
    }
}

/// On a dropped search: an abandoned half-open probe counts as a failure, so
/// it does not hold the probe slot for ever.
struct ProbeGuard<'a> {
    breaker: &'a CircuitBreaker,
    armed: bool,
}

impl Drop for ProbeGuard<'_> {
    fn drop(&mut self) {
        if self.armed && self.breaker.state() == CircuitState::HalfOpen {
            self.breaker.record_failure();
        }
    }
}

impl Inner {
    fn name(&self) -> String {
        lock(&self.name).clone()
    }

    fn resolve(&self) -> Result<Vec<Arc<dyn SearchEngine>>> {
        let mut engines = lock(&self.engines);
        if engines.is_none() {
            let named: Vec<String> = env(ENGINE_ENV).unwrap_or_default().split(',').map(|n| n.trim().to_string()).filter(|n| !n.is_empty()).collect();
            *engines = Some(if named.is_empty() {
                let mut found: Vec<Arc<dyn SearchEngine>> =
                    ENGINES.iter().filter(|n| **n != "duckduckgo").filter_map(|n| engine_named(n, None).ok()).filter(|e| e.ready().is_empty()).collect();
                found.push(Arc::new(DuckDuckGo::default()));
                found
            } else {
                named.iter().map(|n| engine_named(n, None)).collect::<Result<_>>()?
            });
        }
        Ok(engines.clone().unwrap_or_default())
    }

    fn breaker(&self, name: &str) -> CircuitBreaker {
        lock(&self.breakers)
            .entry(name.to_string())
            .or_insert_with(|| CircuitBreaker::new(self.failure_threshold, Duration::from_secs_f64(self.cooldown.max(0.0))))
            .clone()
    }

    fn count(&self, name: &str, answered: bool, error: &str) {
        let mut stats = lock(&self.stats);
        let entry = stats.entry(name.to_string()).or_default();
        if answered {
            entry.answered += 1;
        } else {
            entry.failed += 1;
        }
        if !error.is_empty() {
            entry.last_error = error.to_string();
        }
    }

    fn query(&self, text: &str, limit: Option<usize>, recency: Option<&str>, domains: Option<Vec<String>>) -> Result<SearchQuery> {
        let name = self.name();
        let mut text = SPACE.replace_all(&CONTROL.replace_all(text, " "), " ").trim().to_string();
        if text.is_empty() {
            return Err(Error::tool("the query is empty \u{2014} say what to search for", &name));
        }
        if text.chars().count() > MAX_QUERY_CHARS {
            let head: String = text.chars().take(MAX_QUERY_CHARS).collect();
            text = head.rsplit_once(' ').map_or(head.clone(), |(a, _)| a.to_string());
        }
        text = text.split(' ').take(MAX_QUERY_WORDS).collect::<Vec<_>>().join(" ");
        let count = limit.filter(|n| *n > 0).unwrap_or(self.limit).clamp(1, self.max_limit);

        let asked = recency.unwrap_or("").trim().to_lowercase();
        let when = match asked.as_str() {
            "d" | "today" | "24h" => "day",
            "w" => "week",
            "m" => "month",
            "y" => "year",
            "any" | "all" | "none" => "",
            other => other,
        }
        .to_string();
        if !when.is_empty() && !RECENCY.contains(&when.as_str()) {
            return Err(Error::tool(
                format!("recency is one of {}, or empty for any time \u{2014} got {}", RECENCY.join(", "), py_str_repr(recency.unwrap_or(""))),
                &name,
            ));
        }

        let mut wanted: Vec<String> = Vec::new();
        for d in domains.unwrap_or_default().iter().map(|d| domain(d)).filter(|d| !d.is_empty()) {
            if !wanted.contains(&d) {
                wanted.push(d);
            }
        }
        for d in &wanted {
            if self.blocked.iter().any(|b| matches(d, b)) {
                return Err(Error::tool(format!("{d} is on the blocked list"), &name));
            }
            if !self.allowed.is_empty() && !GLOB.is_match(d) && !self.allowed.iter().any(|a| matches(d, a)) {
                return Err(Error::tool(format!("{d} is not on the allowed list \u{2014} this search covers {}", self.allowed.join(", ")), &name));
            }
        }
        Ok(SearchQuery {
            text,
            limit: count,
            recency: when,
            domains: if wanted.is_empty() { self.allowed.clone() } else { wanted },
            exclude: self.blocked.clone(),
            region: self.region.clone(),
            language: self.language.clone(),
            safe: self.safe_search.clone(),
            snippet_chars: self.snippet_chars,
        })
    }

    fn permitted(&self, url: &str, query: &SearchQuery) -> bool {
        let host = host_of(url);
        if host.is_empty() || self.blocked.iter().any(|b| matches(&host, b)) {
            return false;
        }
        if !self.allowed.is_empty() && !self.allowed.iter().any(|a| matches(&host, a)) {
            return false;
        }
        query.domains.is_empty() || query.domains.iter().any(|d| matches(&host, d))
    }

    /// An engine's results, cleaned, filtered and without repeats.
    fn tidy(&self, raw: &[Value], query: &SearchQuery, engine: &str) -> Vec<SearchResult> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for item in raw {
            let Some(item) = item.as_object() else { continue };
            let url = truthy(item.get("url")).or(truthy(item.get("link"))).and_then(Value::as_str).unwrap_or("");
            let lowered = url.trim().to_lowercase();
            if !(lowered.starts_with("http://") || lowered.starts_with("https://")) {
                continue;
            }
            let url = canonical(url.trim());
            let id = identity(&url);
            if seen.contains(&id) || !self.permitted(&url, query) {
                continue;
            }
            seen.insert(id);
            let title = text(item.get("title"), TITLE_CHARS);
            let snippet = truthy(item.get("snippet")).or(truthy(item.get("content"))).or(item.get("description"));
            out.push(SearchResult {
                title: if title.is_empty() { host_of(&url) } else { title },
                snippet: text(snippet, query.snippet_chars),
                published: published(item.get("published")),
                engine: engine.to_string(),
                score: item.get("score").and_then(Value::as_f64),
                url,
            });
            if out.len() >= query.limit {
                break;
            }
        }
        out
    }

    async fn pace(&self, name: &str) {
        if self.min_interval <= 0.0 {
            return;
        }
        let now = Instant::now();
        let slot = {
            let mut next = lock(&self.next_call);
            let slot = next.get(name).copied().filter(|t| *t > now).unwrap_or(now);
            next.insert(name.to_string(), slot + Duration::from_secs_f64(self.min_interval));
            slot
        };
        if slot > now {
            tokio::time::sleep(slot - now).await;
        }
    }

    /// One engine, tried until it answers, cannot, or the time is up.
    async fn ask(
        &self,
        engine: &Arc<dyn SearchEngine>,
        query: &SearchQuery,
        client: &reqwest::Client,
        started: Instant,
    ) -> std::result::Result<Vec<Value>, SearchFailure> {
        let missing = engine.ready();
        if !missing.is_empty() {
            return Err(SearchFailure::new(missing));
        }
        // Ask for a few more than wanted when some will be filtered away.
        let want = query.limit + if query.domains.is_empty() && query.exclude.is_empty() { 2 } else { 5 };
        let asked = SearchQuery { limit: want.min(engine.max_results()).max(1), ..query.clone() };
        let mut attempt = 0;
        loop {
            self.pace(&engine.name()).await;
            let left = self.deadline - started.elapsed().as_secs_f64();
            if left <= 0.0 {
                return Err(SearchFailure::new("the search ran out of time").healthy());
            }
            let allowed = self.timeout.min(left);
            let failure = match tokio::time::timeout(Duration::from_secs_f64(allowed), engine.search(&asked, client)).await {
                Ok(Ok(found)) => return Ok(found),
                Ok(Err(f)) => f,
                Err(_) => SearchFailure::new(format!("no answer within {}s", fmt_g(allowed))).retryable(),
            };
            if !failure.retryable || attempt >= self.retry.max_retries {
                return Err(failure);
            }
            let wait = match failure.retry_after {
                None => self.retry.backoff(attempt),
                Some(w) if w > self.retry.max_retry_after => return Err(failure),
                Some(w) => w,
            };
            if started.elapsed().as_secs_f64() + wait >= self.deadline {
                return Err(failure);
            }
            tokio::time::sleep(Duration::from_secs_f64(wait.max(0.0))).await;
            attempt += 1;
        }
    }

    async fn run(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        let engines = self.resolve()?;
        let started = Instant::now();
        let mut problems = Vec::new();
        let mut answered = false;
        let client = match &self.client {
            Some(c) => c.clone(),
            None => {
                crate::llm_providers::base::install_tls();
                // Redirects are not followed — a key in a header must not
                // travel to wherever an engine points.
                reqwest::Client::builder()
                    .redirect(reqwest::redirect::Policy::none())
                    .user_agent(format!("agent-harness/{}", crate::VERSION))
                    .connect_timeout(Duration::from_secs_f64(self.timeout.max(0.001)))
                    .build()
                    .map_err(|e| Error::tool(e.to_string(), self.name()))?
            }
        };
        for engine in &engines {
            let name = engine.name();
            let breaker = self.breaker(&name);
            if !breaker.allow() {
                problems.push(format!("{name}: left alone after repeated failures, for another {:.0}s", breaker.retry_in()));
                continue;
            }
            let mut guard = ProbeGuard { breaker: &breaker, armed: true };
            let outcome = self.ask(engine, &query, &client, started).await;
            guard.armed = false;
            match outcome {
                Err(failure) => {
                    if failure.unhealthy {
                        breaker.record_failure();
                    } else {
                        breaker.record_success();
                    }
                    let message = redact(&failure.message, &engine.secrets());
                    self.count(&name, false, &message);
                    problems.push(format!("{name}: {message}"));
                }
                Ok(raw) => {
                    breaker.record_success();
                    self.count(&name, true, "");
                    *lock(&self.last_engine) = name.clone();
                    answered = true;
                    let results = self.tidy(&raw, &query, &name);
                    if !results.is_empty() || !self.fallback_on_empty {
                        return Ok(results);
                    }
                }
            }
        }
        if answered {
            return Ok(Vec::new());
        }
        let why = if problems.is_empty() { "no engine to ask".to_string() } else { problems.join("; ") };
        Err(Error::tool(format!("web search failed \u{2014} {why}"), self.name()))
    }
}

/// Removes this caller's in-flight entry when it finishes or is dropped, so
/// nothing is left behind for the next question.
struct InflightGuard<'a> {
    inner: &'a Inner,
    key: CacheKey,
    pending: Pending,
}

impl Drop for InflightGuard<'_> {
    fn drop(&mut self) {
        let mut inflight = lock(&self.inner.inflight);
        if inflight.get(&self.key).is_some_and(|p| p.ptr_eq(&self.pending)) {
            inflight.remove(&self.key);
        }
    }
}

impl WebSearch {
    pub fn builder() -> WebSearchBuilder {
        WebSearchBuilder::default()
    }

    /// The engines a search goes through, in order.
    pub fn engines(&self) -> Result<Vec<String>> {
        Ok(self.inner.resolve()?.iter().map(|e| e.name()).collect())
    }

    /// The engine that answered the last search.
    pub fn last_engine(&self) -> String {
        lock(&self.inner.last_engine).clone()
    }

    /// Per engine: how many searches it answered and how many it failed.
    pub fn stats(&self) -> IndexMap<String, EngineStats> {
        lock(&self.inner.stats).clone()
    }

    pub fn clear_cache(&self) {
        lock(&self.inner.cache).clear();
    }

    /// Search, and return what was found. Fails with a tool error when no engine answers.
    ///
    /// `limit` defaults to the search's own; `recency` is "day", "week",
    /// "month" or "year"; `domains` narrows the search to some sites.
    pub async fn search(&self, query: &str, limit: Option<usize>, recency: Option<&str>, domains: Option<Vec<String>>) -> Result<Vec<SearchResult>> {
        let inner = &self.inner;
        let asked = inner.query(query, limit, recency, domains)?;
        let key: CacheKey = (asked.text.to_lowercase(), asked.limit, asked.recency.clone(), asked.domains.clone());
        if inner.cache_ttl > 0.0 {
            let mut cache = lock(&inner.cache);
            if let Some((expires, kept)) = cache.get(&key).cloned() {
                if expires > Instant::now() {
                    let value = cache.shift_remove(&key).expect("present");
                    cache.insert(key.clone(), value);
                    return Ok(kept);
                }
                cache.shift_remove(&key);
            }
        }
        // The model often asks the same thing twice in one step: ask once.
        let pending = {
            let mut inflight = lock(&inner.inflight);
            match inflight.get(&key) {
                Some(p) => p.clone(),
                None => {
                    let shared = self.inner.clone();
                    let p: Pending = async move { shared.run(asked).await.map_err(|e| e.to_string()) }.boxed().shared();
                    inflight.insert(key.clone(), p.clone());
                    p
                }
            }
        };
        let guard = InflightGuard { inner, key: key.clone(), pending: pending.clone() };
        let outcome = pending.await;
        let first = {
            let mut inflight = lock(&inner.inflight);
            let mine = inflight.get(&key).is_some_and(|p| p.ptr_eq(&guard.pending));
            if mine {
                inflight.remove(&key);
            }
            mine
        };
        drop(guard);
        match outcome {
            Ok(results) => {
                if first && inner.cache_ttl > 0.0 {
                    let mut cache = lock(&inner.cache);
                    cache.insert(key, (Instant::now() + Duration::from_secs_f64(inner.cache_ttl), results.clone()));
                    while cache.len() > inner.cache_size {
                        cache.shift_remove_index(0);
                    }
                }
                Ok(results)
            }
            Err(message) => Err(Error::tool(message, inner.name())),
        }
    }

    /// The search as a tool an agent can call. `name` renames the search too.
    pub fn as_tool(&self, name: Option<&str>, description: Option<&str>) -> Tool {
        if let Some(name) = name {
            *lock(&self.inner.name) = name.to_string();
        }
        let tool_name = self.inner.name();
        let (default_limit, most) = (self.inner.limit, self.inner.max_limit);
        let limit_doc = if most != default_limit { format!("how many results to return, at most {most}.") } else { "how many results to return.".to_string() };
        let parameters = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what to look for, as you would type it into a search engine."},
                "limit": {"type": "integer", "default": default_limit, "description": limit_doc},
                "recency": {"type": "string", "default": "", "description": "only pages from the last \"day\", \"week\", \"month\" or \"year\"; leave empty for any time."},
                "domains": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}], "default": null, "description": "only search these sites, e.g. [\"docs.python.org\"]."},
            },
            "required": ["query"],
        });
        let summary = "Search the web. Returns the best-matching pages: title, URL and a short snippet.\n\nA snippet is an extract, not the page: fetch the URL to read it in\nfull. Search again with different words when the results miss.";
        let search = self.clone();
        Tool::raw(tool_name, description.unwrap_or(summary), parameters, move |args: Map<String, Value>, _ctx: ToolContext| {
            let search = search.clone();
            async move {
                let name = search.inner.name();
                let Some(query) = args.get("query").and_then(Value::as_str).map(str::to_string) else {
                    return Err(Error::tool("the query must be text", &name));
                };
                let limit = match args.get("limit") {
                    Some(Value::Number(n)) => n.as_f64().map(|f| f.max(0.0) as usize),
                    Some(Value::String(s)) => s.trim().parse::<usize>().ok(),
                    _ => None,
                };
                let recency = match args.get("recency") {
                    Some(Value::String(s)) => s.clone(),
                    _ => String::new(),
                };
                let domains: Option<Vec<String>> = match args.get("domains") {
                    Some(Value::Array(a)) => Some(a.iter().map(py_str).collect()),
                    Some(Value::String(s)) => Some(s.split([',', ' ', '\t', '\n']).filter(|d| !d.is_empty()).map(str::to_string).collect()),
                    _ => None,
                };
                let results = search.search(&query, limit, Some(&recency), domains.clone()).await?;
                if results.is_empty() {
                    let scope = match &domains {
                        Some(d) if !d.is_empty() => format!(" on {}", d.join(", ")),
                        _ => String::new(),
                    };
                    let tail = if recency.is_empty() { "." } else { ", or a wider time range." };
                    return Ok(json!(format!("No results for {}{scope}. Try fewer or different words{tail}", py_str_repr(&query))));
                }
                Ok(Value::Array(results.iter().map(SearchResult::to_dict).collect()))
            }
        })
        .tags(["builtin", "web", "search"])
    }
}

/// Build a search tool with your own engines and domain policy.
pub fn make_search_tool(search: WebSearchBuilder, description: Option<&str>) -> Result<Tool> {
    Ok(search.build()?.as_tool(None, description))
}

static DEFAULT_SEARCH: LazyLock<WebSearch> = LazyLock::new(|| WebSearch::builder().build().expect("the default search builds"));

/// The `web_search` tool, set up from the environment the first time it is
/// called. Every tool this returns shares one cache.
pub fn web_search() -> Tool {
    DEFAULT_SEARCH.as_tool(None, None)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::Router;
    use axum::extract::{RawQuery, State};
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    use super::*;

    fn fast() -> RetryPolicy {
        RetryPolicy { max_retries: 2, initial_delay: 0.0, max_delay: 0.0, jitter: 0.0, max_retry_after: 1.0, ..RetryPolicy::default() }
    }

    fn pages() -> Value {
        json!([
            {"title": "Heat <strong>pumps</strong> &amp; grants", "url": "https://energy.gov/pumps?utm_source=x&id=7",
             "description": "A <b>grant</b> of up to   5,000.", "page_age": "2026-03-01T09:00:00"},
            {"title": "Subsidy guide", "url": "https://example.org/guide", "description": "How to apply."},
        ])
    }

    type Answer = Arc<dyn Fn(usize) -> (u16, Value) + Send + Sync>;

    #[derive(Clone)]
    struct Fake {
        seen: Arc<Mutex<Vec<String>>>,
        answer: Answer,
    }

    async fn handle(State(f): State<Fake>, RawQuery(q): RawQuery) -> impl IntoResponse {
        let n = {
            let mut seen = f.seen.lock().unwrap();
            seen.push(q.unwrap_or_default());
            seen.len()
        };
        let (status, body) = (f.answer)(n);
        (StatusCode::from_u16(status).unwrap(), axum::Json(body))
    }

    /// A Brave engine answered by `answer(call number)`, and the query strings it was sent.
    async fn brave(answer: Answer) -> (Brave, Arc<Mutex<Vec<String>>>) {
        let seen: Arc<Mutex<Vec<String>>> = Arc::default();
        let app = Router::new().fallback(handle).with_state(Fake { seen: seen.clone(), answer });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (Brave::new(Some("sk-secret-key")).with_base_url(&format!("http://{addr}")), seen)
    }

    fn param(query: &str, key: &str) -> String {
        query
            .split('&')
            .filter_map(|p| p.split_once('='))
            .find(|(k, _)| *k == key)
            .map(|(_, v)| crate::knowledge::stores::decode(&v.replace('+', " ")))
            .unwrap_or_default()
    }

    fn engine(name: &str, rows: Value) -> FunctionEngine {
        FunctionEngine::new(name, move |_q| {
            let rows = rows.clone();
            async move { Ok(rows.as_array().cloned().unwrap_or_default()) }
        })
    }

    #[tokio::test]
    async fn a_throttled_or_failing_call_is_retried() {
        let (b, seen) = brave(Arc::new(|n| match n {
            1 => (429, json!({})),
            2 => (503, json!({})),
            _ => (200, json!({"web": {"results": pages()}})),
        }))
        .await;
        let search = WebSearch::builder().engine_impl(b).retry(fast()).build().unwrap();
        assert_eq!(search.search("q", None, None, None).await.unwrap().len(), 2);
        assert_eq!(seen.lock().unwrap().len(), 3);
        assert_eq!(search.stats()["brave"], EngineStats { answered: 1, failed: 0, last_error: String::new() });
    }

    #[tokio::test]
    async fn every_failure_is_said_without_the_key_and_the_next_engine_answers() {
        let (b, _) = brave(Arc::new(|_| (401, json!({"error": "bad key sk-secret-key"})))).await;
        let backup = engine("backup", json!([{"title": "Found", "url": "https://a.example/"}]));
        let search = WebSearch::builder().engine_impl(b).engine_impl(backup).retry(fast()).build().unwrap();
        let hits = search.search("q", None, None, None).await.unwrap();
        assert_eq!(hits[0].engine, "backup");
        assert_eq!(search.stats()["brave"].last_error, "the API key was refused");

        let (b, seen) = brave(Arc::new(|_| (400, json!({"error": "bad query for token sk-secret-key"})))).await;
        let failing = FunctionEngine::new("broken", |_q| async { Err::<Vec<Value>, _>(SearchFailure::new("offline")) });
        let search = WebSearch::builder().engine_impl(b).engine_impl(failing).retry(fast()).build().unwrap();
        let message = search.search("q", None, None, None).await.unwrap_err().to_string();
        assert!(message.contains("brave: refused the query (400)"), "{message}");
        assert!(message.contains("broken: offline"), "{message}");
        assert!(!message.contains("secret-key"), "{message}");
        assert_eq!(seen.lock().unwrap().len(), 1, "a refused query is not worth asking again");
    }

    #[tokio::test]
    async fn a_missing_key_says_which_variable_to_set() {
        if std::env::var("BRAVE_API_KEY").is_ok() || std::env::var("BRAVE_SEARCH_API_KEY").is_ok() {
            return;
        }
        let search = WebSearch::builder().engine("brave").build().unwrap();
        let message = search.search("q", None, None, None).await.unwrap_err().to_string();
        assert!(message.contains("brave: no API key — set BRAVE_API_KEY"), "{message}");
    }

    #[tokio::test]
    async fn an_engine_that_keeps_failing_is_left_alone_for_a_while() {
        let (b, seen) = brave(Arc::new(|_| (500, json!({})))).await;
        let search = WebSearch::builder().engine_impl(b).retry(RetryPolicy::none()).failure_threshold(2).cache_ttl(0.0).build().unwrap();
        for _ in 0..2 {
            assert!(search.search("q", None, None, None).await.unwrap_err().to_string().contains("answered 500"));
        }
        let third = search.search("q", None, None, None).await.unwrap_err().to_string();
        assert!(third.contains("left alone after repeated failures"), "{third}");
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn an_engine_that_finds_nothing_passes_the_question_on() {
        let asked = Arc::new(Mutex::new(Vec::new()));
        let (a1, a2) = (asked.clone(), asked.clone());
        let empty = FunctionEngine::new("empty", move |_q| {
            a1.lock().unwrap().push("empty");
            async { Ok(vec![]) }
        });
        let full = FunctionEngine::new("full", move |_q| {
            a2.lock().unwrap().push("full");
            async { Ok(vec![json!({"title": "Found", "url": "https://a.example/"})]) }
        });
        let empty = Arc::new(empty) as Arc<dyn SearchEngine>;
        let full = Arc::new(full) as Arc<dyn SearchEngine>;
        let search = WebSearch::builder().engine_arc(empty.clone()).engine_arc(full.clone()).build().unwrap();
        assert_eq!(search.search("q", None, None, None).await.unwrap().len(), 1);
        assert_eq!(*asked.lock().unwrap(), vec!["empty", "full"]);
        asked.lock().unwrap().clear();
        let strict = WebSearch::builder().engine_arc(empty).engine_arc(full).fallback_on_empty(false).build().unwrap();
        assert!(strict.search("q", None, None, None).await.unwrap().is_empty());
        assert_eq!(*asked.lock().unwrap(), vec!["empty"]);
    }

    #[tokio::test]
    async fn results_are_cleaned_deduplicated_and_cut_to_the_limit() {
        let rows = json!([
            {"title": "", "url": "https://www.a.example/page/?utm_campaign=x&b=2&a=1#:~:text=hi"},
            {"title": "Same page", "url": "http://a.example/page?a=1&b=2"},
            {"title": "Not a page", "url": "javascript:alert(1)"},
            {"title": "No url"}, "junk", null,
            {"title": "Long", "url": "https://b.example/", "snippet": "word ".repeat(400)},
            {"title": "Third", "url": "https://c.example/"},
            {"title": "Fourth", "url": "https://d.example/"},
        ]);
        let search = WebSearch::builder().engine_impl(engine("rows", rows)).snippet_chars(120).build().unwrap();
        let hits = search.search("q", Some(3), None, None).await.unwrap();
        let urls: Vec<&str> = hits.iter().map(|h| h.url.as_str()).collect();
        assert_eq!(urls, ["https://www.a.example/page/?b=2&a=1", "https://b.example/", "https://c.example/"]);
        assert_eq!(hits[0].title, "www.a.example");
        assert!(hits[1].snippet.chars().count() <= 121 && hits[1].snippet.ends_with('…'));
    }

    #[tokio::test]
    async fn the_domain_policy_holds_whatever_the_engine_returns() {
        let rows = json!([
            {"title": "gov", "url": "https://www.energy.gov/a"}, {"title": "sub", "url": "https://data.energy.gov/b"},
            {"title": "lookalike", "url": "https://notenergy.gov.evil.example/c"},
            {"title": "blocked", "url": "https://ads.energy.gov/d"}, {"title": "other", "url": "https://example.org/e"},
        ]);
        let search = WebSearch::builder()
            .engine_impl(engine("rows", rows))
            .allowed_domains(["energy.gov", "*.europa.eu"])
            .blocked_domains(["ads.energy.gov"])
            .build()
            .unwrap();
        let titles = |hits: Vec<SearchResult>| hits.into_iter().map(|h| h.title).collect::<Vec<_>>();
        assert_eq!(titles(search.search("q", None, None, None).await.unwrap()), ["gov", "sub"]);
        assert_eq!(titles(search.search("q", None, None, Some(vec!["data.energy.gov".into()])).await.unwrap()), ["sub"]);
        assert!(search.search("q", None, None, Some(vec!["example.org".into()])).await.unwrap_err().to_string().contains("not on the allowed list"));
        assert!(search.search("q", None, None, Some(vec!["ads.energy.gov".into()])).await.unwrap_err().to_string().contains("blocked list"));
    }

    #[tokio::test]
    async fn the_query_is_checked_and_domains_become_site_operators() {
        let (b, seen) = brave(Arc::new(|_| (200, json!({"web": {"results": pages()}})))).await;
        let search = WebSearch::builder().engine_impl(b).retry(fast()).max_limit(4).blocked_domains(["pinterest.com"]).build().unwrap();
        assert!(search.search("  \n ", None, None, None).await.unwrap_err().to_string().contains("query is empty"));
        assert!(search.search("q", None, Some("fortnight"), None).await.unwrap_err().to_string().contains("recency is one of"));
        assert!(seen.lock().unwrap().is_empty(), "nothing is sent for a bad question");
        search.search("grants", None, None, Some(vec!["energy.gov".into(), "example.org".into()])).await.unwrap();
        assert_eq!(param(&seen.lock().unwrap()[0], "q"), "grants (site:energy.gov OR site:example.org) -site:pinterest.com");
        search.search("q", Some(99), None, None).await.unwrap();
        // Clamped to 4, not refused — and asked for 5 more, since a blocked domain filters some out.
        assert_eq!(param(seen.lock().unwrap().last().unwrap(), "count"), "9");
        search.search(&"word ".repeat(300), None, Some("W"), None).await.unwrap();
        let last = seen.lock().unwrap().last().unwrap().clone();
        // Cut to 50 words, and the blocked domain still said after them.
        let q = param(&last, "q");
        assert_eq!(q.split_whitespace().count(), 51);
        assert!(q.ends_with(" -site:pinterest.com"), "{q}");
        assert_eq!(param(&last, "freshness"), "pw");
    }

    #[tokio::test]
    async fn the_same_question_is_asked_once_even_in_flight() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let slow = FunctionEngine::new("slow", move |q| {
            counted.fetch_add(1, Ordering::SeqCst);
            async move {
                tokio::time::sleep(Duration::from_millis(20)).await;
                if q.text.contains("fail") {
                    return Err(SearchFailure::new("offline"));
                }
                Ok(vec![json!({"title": "Found", "url": "https://a.example/"})])
            }
        });
        let search = WebSearch::builder().engine_impl(slow).build().unwrap();
        let (a, b) = tokio::join!(search.search("Heat  pumps", None, None, None), search.search("heat pumps", None, None, None));
        assert_eq!(a.unwrap(), b.unwrap());
        assert_eq!(calls.load(Ordering::SeqCst), 1, "asked once, answered twice");
        search.search("heat pumps", None, None, None).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1, "then answered from the cache");
        let (x, y) = tokio::join!(search.search("fail", None, None, None), search.search("fail", None, None, None));
        assert!(x.is_err() && y.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 2, "a failure in flight is shared too");
        search.clear_cache();
        search.search("heat pumps", None, None, None).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn the_tool_has_a_small_schema_and_a_failure_is_something_the_model_reads() {
        let search =
            WebSearch::builder().engine_impl(engine("rows", json!([{"title": "Found", "url": "https://a.example/", "snippet": "yes"}]))).build().unwrap();
        let tool = search.as_tool(None, None);
        assert_eq!(tool.name, "web_search");
        assert!(tool.parameters["properties"]["query"].is_object());
        let out = tool.invoke(json!({"query": "q"}).as_object().cloned().unwrap(), ToolContext::default()).await.unwrap();
        assert!(out.to_string().contains("https://a.example/"), "{out}");
        assert!(!search_engines().is_empty());
    }
}
