//! Web fetch, with a domain policy and a size cap. Native tool, not a connector.
//!
//! The fetch tool refuses private and loopback addresses — by name, by the
//! address a name resolves to, and again at every redirect, so a public page
//! cannot bounce the harness onto `169.254.169.254` or `localhost`.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use futures::StreamExt;
use regex::Regex;
use reqwest::header::{CONTENT_TYPE, LOCATION, USER_AGENT};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value};

use super::net::is_private_host;
use crate::errors::{Error, Result};
use crate::governance::fnmatch;
use crate::tools::{Permission, Tool, ToolContext};

static TAGS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>").expect("valid"));
static MARKUP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").expect("valid"));
static BLANKS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").expect("valid"));

/// Host prefixes Python's fetch tool refuses before anything is resolved.
const PRIVATE: [&str; 7] = ["localhost", "127.", "0.0.0.0", "10.", "192.168.", "169.254.", "[::1]"];
/// httpx's default redirect limit.
const MAX_REDIRECTS: usize = 20;
/// The most of one body that is read, whatever `max_chars` says.
const MAX_BODY_BYTES: usize = 20_000_000;

fn to_text(body: &str, content_type: &str) -> String {
    if !content_type.contains("html") {
        return body.to_string();
    }
    let stripped = TAGS.replace_all(body, " ");
    let stripped = MARKUP.replace_all(&stripped, " ");
    BLANKS.replace_all(&stripped.replace("&nbsp;", " "), "\n\n").trim().to_string()
}

/// The first `max` characters, and a note when there were more.
fn cap(text: String, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((cut, _)) => format!("{}\n... [truncated at {max} characters]", &text[..cut]),
        None => text,
    }
}

/// How a fetch tool behaves. Every field has a working default.
#[derive(Debug, Clone)]
pub struct FetchOptions {
    /// Only these hosts (shell-style globs). Empty for any.
    pub allowed_domains: Vec<String>,
    /// Never these hosts (shell-style globs).
    pub blocked_domains: Vec<String>,
    pub max_chars: usize,
    /// Seconds to connect, and between reads.
    pub timeout: f64,
    /// Let it fetch localhost and private addresses.
    pub allow_private: bool,
    pub name: String,
    /// Hosts let through the private-address check as written (tests).
    pub(crate) trusted_hosts: Vec<String>,
}

impl Default for FetchOptions {
    fn default() -> Self {
        Self {
            allowed_domains: Vec::new(),
            blocked_domains: Vec::new(),
            max_chars: 40_000,
            timeout: 30.0,
            allow_private: false,
            name: "web_fetch".into(),
            trusted_hosts: Vec::new(),
        }
    }
}

fn yes() -> bool {
    true
}

#[derive(Deserialize, JsonSchema)]
struct FetchArgs {
    /// the absolute http(s) URL to fetch.
    url: String,
    /// strip HTML markup and return readable text.
    #[serde(default = "yes")]
    as_text: bool,
}

struct Fetcher {
    options: FetchOptions,
    client: reqwest::Client,
    resolved: Mutex<HashMap<String, bool>>,
}

impl Fetcher {
    /// Is this URL one the policy lets the tool open? Checked at every hop.
    async fn check(&self, url: &str) -> Result<reqwest::Url> {
        let name = &self.options.name;
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(Error::tool("only http and https URLs can be fetched", name));
        }
        let parsed = reqwest::Url::parse(url).map_err(|e| Error::tool(format!("{name} failed: {e}"), name))?;
        let host = parsed.host_str().unwrap_or("").to_lowercase();
        // Python's spelling of the host: everything up to the first `/`, then `:`.
        let crude = url.split_once("//").map_or("", |x| x.1).split('/').next().unwrap_or("").split(':').next().unwrap_or("").to_lowercase();
        if !self.options.allow_private
            && !self.options.trusted_hosts.contains(&host)
            && (PRIVATE.iter().any(|p| crude.starts_with(p) || host.starts_with(p)) || is_private_host(&host, Some(&self.resolved)).await)
        {
            return Err(Error::tool("refusing to fetch a private or loopback address", name));
        }
        if self.options.blocked_domains.iter().any(|p| fnmatch(&host, p)) {
            return Err(Error::tool(format!("{host} is on the blocked list"), name));
        }
        if !self.options.allowed_domains.is_empty() && !self.options.allowed_domains.iter().any(|p| fnmatch(&host, p)) {
            return Err(Error::tool(format!("{host} is not on the allowed list"), name));
        }
        Ok(parsed)
    }

    async fn fetch(&self, url: &str, as_text: bool) -> Result<String> {
        let name = &self.options.name;
        let failed = |e: reqwest::Error| Error::tool(format!("{name} failed: {e}"), name);
        let mut current = url.to_string();
        let mut hops = 0;
        let response = loop {
            let target = self.check(&current).await?;
            let response = self.client.get(target.clone()).header(USER_AGENT, "agent-harness/0.1").send().await.map_err(failed)?;
            let status = response.status().as_u16();
            if matches!(status, 301 | 302 | 303 | 307 | 308)
                && let Some(location) = response.headers().get(LOCATION).and_then(|v| v.to_str().ok())
            {
                hops += 1;
                if hops > MAX_REDIRECTS {
                    return Err(Error::tool(format!("{name} failed: Exceeded maximum allowed redirects."), name));
                }
                current = target.join(location).map_err(|e| Error::tool(format!("{name} failed: {e}"), name))?.to_string();
                continue;
            }
            break response;
        };
        let status = response.status().as_u16();
        if status >= 400 {
            return Err(Error::tool(format!("{url} returned {status}"), name));
        }
        let content_type = response.headers().get(CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            body.extend_from_slice(&chunk.map_err(failed)?);
            if body.len() >= MAX_BODY_BYTES {
                body.truncate(MAX_BODY_BYTES);
                break;
            }
        }
        let body = String::from_utf8_lossy(&body).into_owned();
        let text = if as_text { to_text(&body, &content_type) } else { body };
        Ok(cap(text, self.options.max_chars))
    }
}

/// Build a fetch tool with your own domain policy.
pub fn make_fetch_tool(options: FetchOptions) -> Tool {
    crate::llm_providers::base::install_tls();
    let timeout = Duration::from_secs_f64(options.timeout.max(0.001));
    let client =
        reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).connect_timeout(timeout).read_timeout(timeout).build().expect("reqwest client");
    let name = options.name.clone();
    let fetcher = Arc::new(Fetcher { options, client, resolved: Mutex::new(HashMap::new()) });
    Tool::new(name, move |args: FetchArgs, _ctx: ToolContext| {
        let fetcher = fetcher.clone();
        async move { fetcher.fetch(&args.url, args.as_text).await }
    })
    .describe("Fetch a URL and return its content.")
    .tags(["builtin", "web"])
    .cacheable(true)
}

/// The `web_fetch` tool with the default policy: any public address.
pub fn http_fetch() -> Tool {
    make_fetch_tool(FetchOptions::default())
}

// ---------------------------------------------------------------------------
// http_request
// ---------------------------------------------------------------------------

fn get() -> String {
    "GET".into()
}

#[derive(Deserialize, JsonSchema)]
struct HttpArgs {
    /// path appended to the configured base URL, or a full URL.
    path: String,
    /// GET, POST, PUT, PATCH or DELETE.
    #[serde(default = "get")]
    method: String,
    /// JSON body for write methods.
    #[serde(default)]
    body: Option<Map<String, Value>>,
    /// query-string parameters.
    #[serde(default)]
    query: Option<Map<String, Value>>,
}

/// How httpx writes a query value: `true`, `""` for null, numbers as Python prints them.
fn query_value(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::String(s) => s.clone(),
        other => crate::governance::py_str(other),
    }
}

/// A general HTTP tool for an internal API the agent is allowed to call.
///
/// Every call asks for approval first (`Permission::Ask`). Redirects are not
/// followed, and no address policy is applied: point it at your own API.
pub fn make_http_tool(base_url: &str, headers: &[(&str, &str)], name: &str, timeout: f64) -> Tool {
    crate::llm_providers::base::install_tls();
    let timeout = Duration::from_secs_f64(timeout.max(0.001));
    let client =
        reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).connect_timeout(timeout).read_timeout(timeout).build().expect("reqwest client");
    let base = base_url.trim_end_matches('/').to_string();
    let headers: Vec<(String, String)> = headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    let tool_name = name.to_string();
    Tool::new(name, move |args: HttpArgs, _ctx: ToolContext| {
        let (client, base, headers, name) = (client.clone(), base.clone(), headers.clone(), tool_name.clone());
        async move {
            let url = if args.path.starts_with("http") { args.path.clone() } else { format!("{base}/{}", args.path.trim_start_matches('/')) };
            let method = reqwest::Method::from_bytes(args.method.to_uppercase().as_bytes()).map_err(|e| Error::tool(format!("{name} failed: {e}"), &name))?;
            let mut request = client.request(method, &url);
            for (k, v) in &headers {
                request = request.header(k, v);
            }
            if let Some(query) = &args.query {
                let mut pairs = Vec::new();
                for (k, v) in query {
                    match v {
                        Value::Array(items) => pairs.extend(items.iter().map(|i| (k.clone(), query_value(i)))),
                        other => pairs.push((k.clone(), query_value(other))),
                    }
                }
                request = request.query(&pairs);
            }
            if let Some(body) = &args.body {
                request = request.json(body);
            }
            let response = request.send().await.map_err(|e| Error::tool(format!("{name} failed: {e}"), &name))?;
            let status = response.status().as_u16();
            let text = response.text().await.map_err(|e| Error::tool(format!("{name} failed: {e}"), &name))?;
            if status >= 400 {
                let head: String = text.chars().take(500).collect();
                return Err(Error::tool(format!("{} {url} \u{2192} {status}: {head}", args.method), &name));
            }
            Ok(serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text)))
        }
    })
    .describe("Call an HTTP API.")
    .tags(["builtin", "web", "api"])
    .permission(Permission::Ask)
}

#[cfg(test)]
mod tests {
    use axum::Router;
    use axum::http::{HeaderMap, StatusCode, header};
    use axum::response::IntoResponse;
    use axum::routing::{any, get};
    use serde_json::json;

    use super::*;

    async fn serve(app: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().cloned().unwrap()
    }

    async fn page() -> impl IntoResponse {
        ([(header::CONTENT_TYPE, "text/html")], "<html><body><h1>Title</h1><script>bad()</script><p>Body&nbsp;text</p></body></html>")
    }

    #[tokio::test]
    async fn web_fetch_strips_markup_and_honours_the_domain_policy() {
        let base = serve(Router::new().route("/page", get(page))).await;
        let fetch = make_fetch_tool(FetchOptions { allowed_domains: vec!["127.0.0.1".into()], allow_private: true, ..Default::default() });
        let text = fetch.invoke(args(json!({"url": format!("{base}/page")})), ToolContext::default()).await.unwrap();
        let text = text.as_str().unwrap();
        assert!(text.contains("Title") && text.contains("Body text"), "{text}");
        assert!(!text.contains("bad()") && !text.contains("<p>"));
        let raw = fetch.invoke(args(json!({"url": format!("{base}/page"), "as_text": false})), ToolContext::default()).await.unwrap();
        assert!(raw.as_str().unwrap().contains("<script>"));

        let e = fetch.invoke(args(json!({"url": "https://elsewhere.com/page"})), ToolContext::default()).await.unwrap_err();
        assert_eq!(e.to_string(), "elsewhere.com is not on the allowed list");

        let strict = http_fetch();
        assert_eq!(strict.name, "web_fetch");
        assert!(strict.cacheable && strict.description == "Fetch a URL and return its content.");
        for url in [
            "http://127.0.0.1/admin",
            "http://localhost:8080/",
            "http://[::1]/",
            "http://10.0.0.5/",
            "http://169.254.169.254/latest",
            "http://user@127.0.0.1/",
            "http://0x7f000001/",
        ] {
            let e = strict.invoke(args(json!({"url": url})), ToolContext::default()).await.unwrap_err();
            assert_eq!(e.to_string(), "refusing to fetch a private or loopback address", "{url}");
        }
        let e = strict.invoke(args(json!({"url": "file:///etc/passwd"})), ToolContext::default()).await.unwrap_err();
        assert_eq!(e.to_string(), "only http and https URLs can be fetched");
    }

    #[tokio::test]
    async fn a_redirect_to_a_private_address_is_refused_at_the_hop() {
        let base = serve(
            Router::new()
                .route("/hop", get(|| async { (StatusCode::FOUND, [(header::LOCATION, "/page")]) }))
                .route("/page", get(page))
                .route(
                    "/escape",
                    get(|headers: HeaderMap| async move {
                        let port = headers.get("host").and_then(|h| h.to_str().ok()).and_then(|h| h.rsplit(':').next()).unwrap_or("80").to_string();
                        (StatusCode::FOUND, [(header::LOCATION, format!("http://localhost:{port}/page"))])
                    }),
                )
                .route("/missing", get(|| async { StatusCode::NOT_FOUND })),
        )
        .await;
        // 127.0.0.1 stands in for a public host; `localhost` is still private.
        let fetch = make_fetch_tool(FetchOptions { trusted_hosts: vec!["127.0.0.1".into()], max_chars: 5, ..Default::default() });
        let text = fetch.invoke(args(json!({"url": format!("{base}/hop")})), ToolContext::default()).await.unwrap();
        assert_eq!(text, "Title\n... [truncated at 5 characters]");
        let e = fetch.invoke(args(json!({"url": format!("{base}/escape")})), ToolContext::default()).await.unwrap_err();
        assert_eq!(e.to_string(), "refusing to fetch a private or loopback address");
        let e = fetch.invoke(args(json!({"url": format!("{base}/missing")})), ToolContext::default()).await.unwrap_err();
        assert_eq!(e.to_string(), format!("{base}/missing returned 404"));
        let blocked = make_fetch_tool(FetchOptions { blocked_domains: vec!["127.*".into()], allow_private: true, ..Default::default() });
        let e = blocked.invoke(args(json!({"url": format!("{base}/page")})), ToolContext::default()).await.unwrap_err();
        assert_eq!(e.to_string(), "127.0.0.1 is on the blocked list");
    }

    #[tokio::test]
    async fn the_http_tool_calls_the_api_and_reports_failures() {
        let base = serve(Router::new().route(
            "/items",
            any(|method: axum::http::Method, uri: axum::http::Uri, headers: HeaderMap, body: String| async move {
                if uri.query() == Some("fail=true") {
                    return (StatusCode::BAD_REQUEST, "nope".to_string());
                }
                let auth = headers.get("x-key").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
                (StatusCode::OK, json!({"method": method.as_str(), "query": uri.query(), "auth": auth, "body": body}).to_string())
            }),
        ))
        .await;
        let tool = make_http_tool(&format!("{base}/"), &[("x-key", "k1")], "http_request", 30.0);
        assert_eq!(tool.permission, Permission::Ask);
        let out = tool
            .invoke(args(json!({"path": "/items", "method": "post", "body": {"a": 1}, "query": {"n": 2, "tag": ["x", "y"]}})), ToolContext::default())
            .await
            .unwrap();
        assert_eq!(out, json!({"method": "POST", "query": "n=2&tag=x&tag=y", "auth": "k1", "body": "{\"a\":1}"}));
        let e = tool.invoke(args(json!({"path": "items", "query": {"fail": true}})), ToolContext::default()).await.unwrap_err();
        assert_eq!(e.to_string(), format!("GET {base}/items \u{2192} 400: nope"));
    }
}
