//! An OpenAPI document, as tools: one for every operation the API offers.
//!
//! ```no_run
//! # async fn demo() -> agent_harness::Result<()> {
//! use agent_harness::toolkits::openapi::{OpenAPIOptions, openapi_tools};
//!
//! let api = openapi_tools("openapi.json", OpenAPIOptions { token: Some("tok".into()), ..Default::default() })?;
//! api.names(); // ["getOrder", "listOrders", "refundOrder"]
//! api.call("getOrder", serde_json::json!({"orderId": "4182"})).await?; // call one yourself
//! let tools = api.into_tools(); // hand them to an agent
//! # Ok(()) }
//! ```
//!
//! The document can be a file (JSON or YAML), a URL, the text itself, or a
//! value already loaded. OpenAPI 3.0 and 3.1 are read as they are; Swagger 2.0
//! is read too.
//!
//! Each operation becomes a tool whose arguments are the operation's own: its
//! path, query and header parameters, and the fields of its request body. The
//! model sees names, types and descriptions taken from the document; what it
//! never sees is the credential — that is added to the request here.

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use futures::FutureExt;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use regex::Regex;
use serde_json::{Map, Value, json};

use crate::errors::{Error, Result};
use crate::pyjson::{Dumps, dumps_with};
use crate::tools::{Permission, Tool, ToolContext, fnmatch};

const METHODS: [&str; 7] = ["get", "put", "post", "delete", "patch", "head", "options"];
const READS: [&str; 3] = ["get", "head", "options"];
/// Headers the document may list as parameters but which are not the model's to set.
const OWN_HEADERS: [&str; 3] = ["accept", "content-type", "authorization"];
/// What is kept of a schema. Everything else is noise to a model, or something
/// one provider or another refuses.
const KEEP: [&str; 13] =
    ["type", "description", "enum", "const", "format", "default", "minimum", "maximum", "minLength", "maxLength", "pattern", "minItems", "maxItems"];
const RETRY_STATUS: [u16; 4] = [429, 502, 503, 504];
const MAX_DEPTH: usize = 8;
const ACCEPT: &str = "application/json, text/plain;q=0.9, */*;q=0.5";

static NAME: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^A-Za-z0-9_-]+").expect("valid regex"));
static WORDS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^A-Za-z0-9]+").expect("valid regex"));
static TEMPLATE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\{([^}]+)\}").expect("valid regex"));

// ----------------------------------------------------------------------
// small helpers shared with the image toolkit
// ----------------------------------------------------------------------

/// A string as Python's `repr` writes it: `'it'`, or `"it's"`.
pub(crate) fn py_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Runs of whitespace as one space, ends trimmed — Python's `" ".join(s.split())`.
pub(crate) fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The first `n` characters.
pub(crate) fn head(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

/// Percent-encoding as Python's `quote(safe="")` (`plus` false) or `quote_plus` (`plus` true).
fn quote(text: &str, plus: bool) -> String {
    let mut out = String::with_capacity(text.len());
    for b in text.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => out.push(b as char),
            b' ' if plus => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Python's `urlencode` of pairs.
fn urlencode(pairs: &[(String, String)]) -> String {
    pairs.iter().map(|(k, v)| format!("{}={}", quote(k, true), quote(v, true))).collect::<Vec<_>>().join("&")
}

fn compact(value: &Value) -> String {
    dumps_with(value, Dumps { sort_keys: false, compact: true, ensure_ascii: true })
}

/// A value as it is written in a URL or a header.
fn plain(value: &Value, separator: &str) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(b) => (if *b { "true" } else { "false" }).into(),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Array(items) => items.iter().map(|v| plain(v, separator)).collect::<Vec<_>>().join(separator),
        Value::Object(_) => compact(value),
    }
}

/// The name Python gives a value's type, for messages.
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn text_of(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// A YAML value as JSON, with every mapping key written as a string
/// (`200:` in a `responses` block is a number to YAML).
fn yaml_to_json(value: serde_yaml::Value) -> Value {
    use serde_yaml::Value as Y;
    match value {
        Y::Null => Value::Null,
        Y::Bool(b) => Value::Bool(b),
        Y::Number(n) => {
            if let Some(i) = n.as_i64() {
                json!(i)
            } else if let Some(u) = n.as_u64() {
                json!(u)
            } else {
                n.as_f64().and_then(serde_json::Number::from_f64).map(Value::Number).unwrap_or(Value::Null)
            }
        }
        Y::String(s) => Value::String(s),
        Y::Sequence(items) => Value::Array(items.into_iter().map(yaml_to_json).collect()),
        Y::Mapping(map) => {
            let mut out = Map::new();
            for (k, v) in map {
                let key = match k {
                    Y::String(s) => s,
                    Y::Bool(b) => (if b { "True" } else { "False" }).into(),
                    Y::Null => "None".into(),
                    other => match yaml_to_json(other) {
                        Value::String(s) => s,
                        v => v.to_string(),
                    },
                };
                out.insert(key, yaml_to_json(v));
            }
            Value::Object(out)
        }
        Y::Tagged(tagged) => yaml_to_json(tagged.value),
    }
}

/// Python's `urljoin`, for the cases a `servers` entry needs.
fn urljoin(base: &str, url: &str) -> String {
    match reqwest::Url::parse(base).and_then(|b| b.join(url)) {
        Ok(joined) => joined.to_string(),
        Err(_) => url.to_string(),
    }
}

pub(crate) fn expand_user(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix('~') {
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
        if let Some(home) = home
            && (rest.is_empty() || rest.starts_with('/') || rest.starts_with('\\'))
        {
            return PathBuf::from(home).join(rest.trim_start_matches(['/', '\\']));
        }
    }
    PathBuf::from(path)
}

// ----------------------------------------------------------------------
// reading the document
// ----------------------------------------------------------------------

/// Where an OpenAPI document comes from: a file, a URL, its text, or a value
/// already loaded.
#[derive(Debug, Clone)]
pub enum OpenAPISpec {
    /// The document itself.
    Document(Value),
    /// A file path, an `http(s)://` URL, or the document's text.
    Source(String),
}

impl From<Value> for OpenAPISpec {
    fn from(value: Value) -> Self {
        match value {
            Value::String(s) => Self::Source(s),
            other => Self::Document(other),
        }
    }
}

impl From<&str> for OpenAPISpec {
    fn from(s: &str) -> Self {
        Self::Source(s.into())
    }
}

impl From<String> for OpenAPISpec {
    fn from(s: String) -> Self {
        Self::Source(s)
    }
}

impl From<&Path> for OpenAPISpec {
    fn from(p: &Path) -> Self {
        Self::Source(p.to_string_lossy().into_owned())
    }
}

impl From<PathBuf> for OpenAPISpec {
    fn from(p: PathBuf) -> Self {
        Self::Source(p.to_string_lossy().into_owned())
    }
}

impl From<&PathBuf> for OpenAPISpec {
    fn from(p: &PathBuf) -> Self {
        Self::Source(p.to_string_lossy().into_owned())
    }
}

fn parse(text: &str, place: &str) -> Result<Value> {
    let data = match serde_json::from_str::<Value>(text) {
        Ok(v) => v,
        Err(_) => match serde_yaml::from_str::<serde_yaml::Value>(text) {
            Ok(v) => yaml_to_json(v),
            Err(e) => return Err(Error::config(format!("{place} is neither JSON nor YAML: {e}"))),
        },
    };
    if !data.is_object() {
        return Err(Error::config(format!("{place} is not an OpenAPI document")));
    }
    Ok(data)
}

/// What a source amounts to before anything is fetched.
enum Plan {
    Ready(Value, String),
    Fetch(String),
}

fn plan(spec: OpenAPISpec) -> Result<Plan> {
    let source = match spec {
        OpenAPISpec::Document(Value::Object(map)) => return Ok(Plan::Ready(Value::Object(map), String::new())),
        OpenAPISpec::Document(other) => {
            return Err(Error::config(format!("an OpenAPI document is a file, a URL, its text, or a mapping — got {}", type_name(&other))));
        }
        OpenAPISpec::Source(s) => s,
    };
    let text = source.trim();
    if text.is_empty() {
        return Err(Error::config("an OpenAPI document is a file, a URL, its text, or a mapping — got str"));
    }
    if text.starts_with("http://") || text.starts_with("https://") {
        return Ok(Plan::Fetch(text.to_string()));
    }
    if text.starts_with('{') || text.contains('\n') {
        return Ok(Plan::Ready(parse(text, "the document")?, String::new()));
    }
    let file = expand_user(text);
    if !file.is_file() {
        return Err(Error::config(format!("no OpenAPI document at {}", file.display())));
    }
    let body = std::fs::read_to_string(&file).map_err(|e| Error::config(format!("{} could not be read: {e}", file.display())))?;
    Ok(Plan::Ready(parse(&body, &file.display().to_string())?, String::new()))
}

async fn fetch(url: &str, client: Option<reqwest::Client>) -> Result<Value> {
    let client = match client {
        Some(c) => c,
        None => {
            crate::llm_providers::base::install_tls();
            reqwest::Client::builder().timeout(Duration::from_secs(30)).build().map_err(|e| Error::config(format!("{url} could not be fetched: {e}")))?
        }
    };
    let response =
        client.get(url).header("accept", "application/json, */*").send().await.map_err(|e| Error::config(format!("{url} could not be fetched: {e}")))?;
    let status = response.status().as_u16();
    if status >= 400 {
        return Err(Error::config(format!("{url} returned HTTP {status}")));
    }
    let text = response.text().await.map_err(|e| Error::config(format!("{url} could not be fetched: {e}")))?;
    parse(&text, url)
}

/// Fetch from synchronous code, inside a runtime or not: on a thread of its own.
fn fetch_blocking(url: &str, client: Option<reqwest::Client>) -> Result<Value> {
    let url = url.to_string();
    std::thread::spawn(move || {
        let runtime =
            tokio::runtime::Builder::new_current_thread().enable_all().build().map_err(|e| Error::config(format!("{url} could not be fetched: {e}")))?;
        runtime.block_on(fetch(&url, client))
    })
    .join()
    .unwrap_or_else(|_| Err(Error::config("fetching the OpenAPI document failed")))
}

/// An OpenAPI document, with its references followed.
struct Document {
    data: Value,
    swagger: bool,
}

/// An operation that cannot be a tool. It is skipped, and the reason kept.
enum Problem {
    Unsupported(String),
    Fatal(Error),
}

impl From<Error> for Problem {
    fn from(e: Error) -> Self {
        Self::Fatal(e)
    }
}

impl Document {
    fn new(data: Value) -> Result<Self> {
        let has_paths = data.get("paths").is_some_and(Value::is_object);
        let marked = data.get("openapi").is_some_and(truthy) || data.get("swagger").is_some_and(truthy);
        if !has_paths || !marked {
            return Err(Error::config("this is not an OpenAPI document: it needs `openapi` (or `swagger`) and `paths`"));
        }
        let swagger = data.get("swagger").is_some_and(truthy) && !data.get("openapi").is_some_and(truthy);
        Ok(Self { data, swagger })
    }

    /// Follow `$ref` until there is something to read. A reference that comes
    /// back round to itself ends as an empty object.
    fn resolve(&self, node: &Value, seen: &[String]) -> Result<(Value, Vec<String>)> {
        let mut node = node.clone();
        let mut seen = seen.to_vec();
        while let Some(reference) = node.as_object().and_then(|m| m.get("$ref")) {
            let reference = text_of(Some(reference));
            if seen.contains(&reference) || seen.len() > 40 {
                return Ok((json!({"type": "object"}), seen));
            }
            let Some(pointer) = reference.strip_prefix("#/") else {
                return Ok((json!({"description": format!("(defined elsewhere: {reference})")}), seen));
            };
            let mut target = &self.data;
            for part in pointer.split('/') {
                let key = part.replace("~1", "/").replace("~0", "~");
                match target.as_object().and_then(|m| m.get(&key)) {
                    Some(next) => target = next,
                    None => return Err(Error::config(format!("the document refers to {reference}, which is not in it"))),
                }
            }
            node = target.clone();
            seen.push(reference);
        }
        Ok((node, seen))
    }

    /// A schema as a model should be shown it: references followed, `allOf`
    /// merged, and only the keywords that describe a value kept.
    fn schema(&self, node: &Value, seen: &[String], depth: usize, request: bool) -> Result<Map<String, Value>> {
        let (node, seen) = self.resolve(node, seen)?;
        let Value::Object(mut node) = node else { return Ok(Map::new()) };
        if depth > MAX_DEPTH {
            return Ok(Map::new());
        }
        if node.contains_key("allOf") {
            let merged = self.merged(&node, &seen)?;
            node.remove("allOf");
            for (k, v) in merged {
                if truthy(&v) {
                    node.insert(k, v);
                }
            }
        }
        let mut out = Map::new();
        for key in KEEP {
            if let Some(v) = node.get(key) {
                out.insert(key.into(), v.clone());
            }
        }
        if let Some(Value::String(d)) = out.get("description") {
            let d = head(&squash(d), 500);
            out.insert("description".into(), Value::String(d));
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(Value::Array(options)) = node.get(key) {
                let mut kept = vec![];
                for o in options {
                    let s = self.schema(o, &seen, depth + 1, request)?;
                    if !s.is_empty() {
                        kept.push(Value::Object(s));
                    }
                }
                if kept.is_empty() {
                    kept.push(json!({}));
                }
                out.insert("anyOf".into(), Value::Array(kept));
            }
        }
        if let Some(Value::Object(properties)) = node.get("properties") {
            out.entry("type").or_insert_with(|| json!("object"));
            let mut kept = Map::new();
            for (name, prop) in properties {
                let (resolved, _) = self.resolve(prop, &seen)?;
                if request && resolved.get("readOnly").is_some_and(truthy) {
                    continue; // the server sets it; the caller cannot
                }
                kept.insert(name.clone(), Value::Object(self.schema(prop, &seen, depth + 1, request)?));
            }
            let mut required: Vec<Value> = vec![];
            if let Some(Value::Array(names)) = node.get("required") {
                for r in names {
                    if r.as_str().is_some_and(|r| kept.contains_key(r)) && !required.contains(r) {
                        required.push(r.clone());
                    }
                }
            }
            out.insert("properties".into(), Value::Object(kept));
            if !required.is_empty() {
                out.insert("required".into(), Value::Array(required));
            }
        }
        if let Some(items) = node.get("items") {
            out.entry("type").or_insert_with(|| json!("array"));
            out.insert("items".into(), Value::Object(self.schema(items, &seen, depth + 1, request)?));
        }
        if let Some(extra @ Value::Object(_)) = node.get("additionalProperties") {
            out.entry("type").or_insert_with(|| json!("object"));
            out.insert("additionalProperties".into(), Value::Object(self.schema(extra, &seen, depth + 1, request)?));
        }
        Ok(out)
    }

    /// `allOf` inside `allOf`: flattened one level at a time.
    fn merged(&self, node: &Map<String, Value>, seen: &[String]) -> Result<Map<String, Value>> {
        let mut properties = Map::new();
        let mut required: Vec<Value> = vec![];
        let mut out = Map::new();
        if let Some(Value::Array(parts)) = node.get("allOf") {
            for part in parts {
                let (piece, _) = self.resolve(part, seen)?;
                let Value::Object(mut piece) = piece else { continue };
                if piece.contains_key("allOf") {
                    piece = self.merged(&piece, seen)?;
                }
                if let Some(Value::Object(p)) = piece.get("properties") {
                    properties.extend(p.clone());
                }
                if let Some(Value::Array(r)) = piece.get("required") {
                    required.extend(r.iter().cloned());
                }
                for key in ["type", "description", "enum", "format", "items"] {
                    if let Some(v) = piece.get(key) {
                        out.entry(key).or_insert_with(|| v.clone());
                    }
                }
            }
        }
        if let Some(Value::Object(p)) = node.get("properties") {
            properties.extend(p.clone());
        }
        if let Some(Value::Array(r)) = node.get("required") {
            required.extend(r.iter().cloned());
        }
        let mut result = Map::new();
        result.insert("properties".into(), Value::Object(properties));
        result.insert("required".into(), Value::Array(required));
        result.extend(out);
        Ok(result)
    }

    // ---- where the API is -----------------------------------------------
    fn server(&self, origin: &str) -> String {
        let beside = || if origin.is_empty() { String::new() } else { origin.rsplit_once('/').map(|(a, _)| a.to_string()).unwrap_or_default() };
        let data = &self.data;
        if self.swagger {
            let host = text_of(data.get("host"));
            if host.is_empty() {
                return beside();
            }
            let scheme = data.get("schemes").and_then(|s| s.get(0)).map(|s| text_of(Some(s))).unwrap_or_else(|| "https".into());
            return format!("{scheme}://{host}{}", text_of(data.get("basePath")));
        }
        let Some(first) = data.get("servers").and_then(|s| s.get(0)).and_then(Value::as_object) else {
            return beside();
        };
        let mut url = text_of(first.get("url"));
        if let Some(Value::Object(vars)) = first.get("variables") {
            for (name, variable) in vars {
                url = url.replace(&format!("{{{name}}}"), &text_of(variable.get("default")));
            }
        }
        // "/v1" means "on whatever host the document came from".
        if !origin.is_empty() && !url.starts_with("http") { urljoin(origin, &url) } else { url }
    }

    fn security_schemes(&self) -> Result<IndexMap<String, Value>> {
        let mut out = IndexMap::new();
        if self.swagger {
            if let Some(Value::Object(found)) = self.data.get("securityDefinitions") {
                for (name, scheme) in found {
                    let scheme = if scheme.get("type").and_then(Value::as_str) == Some("basic") {
                        json!({"type": "http", "scheme": "basic"})
                    } else if scheme.is_object() {
                        scheme.clone()
                    } else {
                        json!({})
                    };
                    out.insert(name.clone(), scheme);
                }
            }
            return Ok(out);
        }
        if let Some(Value::Object(found)) = self.data.get("components").and_then(|c| c.get("securitySchemes")) {
            for (name, scheme) in found {
                out.insert(name.clone(), self.resolve(scheme, &[])?.0);
            }
        }
        Ok(out)
    }
}

// ----------------------------------------------------------------------
// one operation
// ----------------------------------------------------------------------

/// One argument of a tool, and where in the request it goes.
#[derive(Debug, Clone)]
struct Field {
    name: String,
    wire: String,
    place: String,
    required: bool,
    explode: bool,
    style: String,
    /// Sent as JSON text, as `content` asks.
    encoded: bool,
}

impl Field {
    fn new(name: String, wire: &str, place: &str, required: bool) -> Self {
        Self { name, wire: wire.into(), place: place.into(), required, explode: true, style: String::new(), encoded: false }
    }
}

/// An operation of the API, read once, called many times.
#[derive(Debug, Clone, Default)]
struct Operation {
    method: String,
    path: String,
    operation_id: String,
    summary: String,
    tags: Vec<String>,
    fields: Vec<Field>,
    properties: Map<String, Value>,
    /// json · form · multipart · text
    body_kind: String,
    media_type: String,
    /// The body is one argument, not its fields.
    whole_body: bool,
    /// Form fields the document wants as `a[b]=c`.
    deep: HashSet<String>,
}

impl Operation {
    fn add(&mut self, field: Field, schema: Map<String, Value>) {
        self.properties.insert(field.name.clone(), Value::Object(schema));
        self.fields.push(field);
    }

    fn parameters(&self) -> Value {
        let required: Vec<&str> = self.fields.iter().filter(|f| f.required).map(|f| f.name.as_str()).collect();
        let mut out = json!({"type": "object", "properties": self.properties});
        if !required.is_empty() {
            out["required"] = json!(required);
        }
        out
    }

    fn label(&self) -> String {
        format!("{} {}", self.method.to_uppercase(), self.path)
    }
}

/// A parameter's name as an argument a model can name: `filter[status]` and
/// `X-Request-Id` are both fine on the wire and both awkward in a tool call.
fn argument(name: &str) -> String {
    let cleaned = NAME.replace_all(name, "_");
    let cleaned = cleaned.trim_matches('_');
    if cleaned.is_empty() { "value".into() } else { cleaned.into() }
}

fn is_file(schema: &Value) -> bool {
    matches!(schema.get("format").and_then(Value::as_str), Some("binary" | "base64")) || schema.get("type").and_then(Value::as_str) == Some("file")
}

fn keep_of(param: &Map<String, Value>) -> Value {
    let mut out = Map::new();
    for (k, v) in param {
        if KEEP.contains(&k.as_str()) || k == "items" {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

/// One query parameter, serialised the way the document says.
fn query_pairs(field: &Field, value: &Value) -> Vec<(String, String)> {
    if field.encoded {
        return vec![(field.wire.clone(), compact(value))];
    }
    match value {
        Value::Array(items) => {
            if field.explode {
                return items.iter().map(|v| (field.wire.clone(), plain(v, ","))).collect();
            }
            let separator = match field.style.as_str() {
                "spaceDelimited" => " ",
                "pipeDelimited" => "|",
                _ => ",",
            };
            vec![(field.wire.clone(), items.iter().map(|v| plain(v, ",")).collect::<Vec<_>>().join(separator))]
        }
        Value::Object(map) => {
            if field.style == "deepObject" {
                return map.iter().map(|(k, v)| (format!("{}[{k}]", field.wire), plain(v, ","))).collect();
            }
            if field.explode {
                return map.iter().map(|(k, v)| (k.clone(), plain(v, ","))).collect();
            }
            vec![(field.wire.clone(), map.iter().map(|(k, v)| format!("{k},{}", plain(v, ","))).collect::<Vec<_>>().join(","))]
        }
        other => vec![(field.wire.clone(), plain(other, ","))],
    }
}

/// A form body. A list is the field repeated; an object — or anything the
/// document marks `deepObject` — is written `a[b]=c`, which is what APIs that
/// take nested forms expect.
fn form_pairs(fields: &Map<String, Value>, deep: &HashSet<String>) -> Vec<(String, String)> {
    fn nested(pairs: &mut Vec<(String, String)>, key: String, value: &Value) {
        match value {
            Value::Object(map) => {
                for (inner, item) in map {
                    nested(pairs, format!("{key}[{inner}]"), item);
                }
            }
            Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    nested(pairs, format!("{key}[{index}]"), item);
                }
            }
            Value::Null => {}
            other => pairs.push((key, plain(other, ","))),
        }
    }
    let mut pairs = vec![];
    for (key, value) in fields {
        if deep.contains(key) || value.is_object() {
            nested(&mut pairs, key.clone(), value);
        } else if let Value::Array(items) = value {
            pairs.extend(items.iter().map(|item| (key.clone(), plain(item, ","))));
        } else {
            pairs.push((key.clone(), plain(value, ",")));
        }
    }
    pairs
}

/// A multipart body of plain fields: `(boundary, bytes)`.
/// One form field: its name, its text, and — for a file — (file name, media type, bytes).
pub(crate) type FormField = (String, String, Option<(String, String, Vec<u8>)>);

pub(crate) fn multipart(fields: &[FormField]) -> (String, Vec<u8>) {
    let boundary = format!("----agent-harness-{}", uuid::Uuid::new_v4().simple());
    let mut body = Vec::new();
    for (name, text, file) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        match file {
            Some((filename, media_type, data)) => {
                body.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {media_type}\r\n\r\n").as_bytes(),
                );
                body.extend_from_slice(data);
            }
            None => {
                body.extend_from_slice(format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes());
                body.extend_from_slice(text.as_bytes());
            }
        }
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (boundary, body)
}

// ----------------------------------------------------------------------
// credentials
// ----------------------------------------------------------------------

type Fetcher = Arc<dyn Fn() -> BoxFuture<'static, Result<String>> + Send + Sync>;

/// A credential, or the function that fetches a fresh one for each request —
/// so a token that expires can be renewed.
#[derive(Clone)]
pub enum Secret {
    Value(String),
    Fetch(Fetcher),
}

impl Secret {
    /// A secret fetched afresh for every request.
    pub fn from_fn<F, Fut>(f: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<String>> + Send + 'static,
    {
        Self::Fetch(Arc::new(move || f().boxed()))
    }

    async fn get(&self) -> Result<String> {
        match self {
            Self::Value(v) => Ok(v.clone()),
            Self::Fetch(f) => f().await,
        }
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(***)")
    }
}

impl From<&str> for Secret {
    fn from(s: &str) -> Self {
        Self::Value(s.into())
    }
}

impl From<String> for Secret {
    fn from(s: String) -> Self {
        Self::Value(s)
    }
}

/// A value sent with every request: a prefix (`"Bearer "`) and its secret.
#[derive(Debug, Clone)]
struct Sent {
    prefix: String,
    secret: Secret,
}

impl Sent {
    fn plain(secret: Secret) -> Self {
        Self { prefix: String::new(), secret }
    }

    async fn value(&self) -> Result<String> {
        Ok(format!("{}{}", self.prefix, self.secret.get().await?))
    }
}

// ----------------------------------------------------------------------
// the toolkit
// ----------------------------------------------------------------------

/// How an [`OpenAPIToolkit`] is made. Everything has a default.
#[derive(Debug, Clone)]
pub struct OpenAPIOptions {
    /// Where the API is. Left out, the document's own `servers`.
    pub base_url: Option<String>,
    /// A bearer token, sent as `Authorization: Bearer …`. [`Secret::from_fn`]
    /// is called for each request, so a token that expires can be renewed.
    pub token: Option<Secret>,
    /// The key for the document's `apiKey` security scheme, sent in the
    /// header, query or cookie that scheme names.
    pub api_key: Option<String>,
    /// A `(user, password)` pair.
    pub basic: Option<(String, String)>,
    /// Scheme name → secret, for a document with several schemes.
    pub credentials: IndexMap<String, String>,
    /// Sent with every request. A parameter fixed here is not offered to the model.
    pub headers: IndexMap<String, Secret>,
    /// Query parameters sent with every request; not offered to the model.
    pub params: IndexMap<String, Secret>,
    /// Which operations become tools — globs matched against the tool's name,
    /// its `operationId`, or `"GET /orders/{id}"`. `None` for all.
    pub include: Option<Vec<String>>,
    pub exclude: Vec<String>,
    /// Only operations with one of these tags.
    pub tags: Option<Vec<String>>,
    /// Only operations with these methods.
    pub methods: Option<Vec<String>>,
    /// Include operations the document marks deprecated.
    pub deprecated: bool,
    /// The permission of a tool that changes something (anything but GET,
    /// HEAD, OPTIONS): allow, ask (needs an approver), or deny.
    pub writes: Permission,
    /// Put in front of every tool name — to keep two APIs apart.
    pub prefix: String,
    /// What to call this API; the tag its tools carry. Left out, the document's title.
    pub name: Option<String>,
    /// Seconds per request.
    pub timeout: f64,
    /// Tries after the first. Only reads are retried.
    pub retries: u32,
    /// How much of a response the model is shown.
    pub max_response_chars: usize,
    /// Let the harness cache what GETs return.
    pub cache_reads: bool,
    /// An HTTP client of your own. It should not follow redirects: a
    /// credential must not go to wherever a response points.
    pub client: Option<reqwest::Client>,
}

impl Default for OpenAPIOptions {
    fn default() -> Self {
        Self {
            base_url: None,
            token: None,
            api_key: None,
            basic: None,
            credentials: IndexMap::new(),
            headers: IndexMap::new(),
            params: IndexMap::new(),
            include: None,
            exclude: vec![],
            tags: None,
            methods: None,
            deprecated: false,
            writes: Permission::Allow,
            prefix: String::new(),
            name: None,
            timeout: 30.0,
            retries: 2,
            max_response_chars: 40_000,
            cache_reads: false,
            client: None,
        }
    }
}

const OPTION_NAMES: [&str; 19] = [
    "base_url",
    "token",
    "api_key",
    "basic",
    "credentials",
    "headers",
    "params",
    "include",
    "exclude",
    "tags",
    "methods",
    "deprecated",
    "writes",
    "prefix",
    "name",
    "timeout",
    "retries",
    "max_response_chars",
    "cache_reads",
];

fn strings(value: &Value, what: &str) -> Result<Vec<String>> {
    match value {
        Value::Null => Ok(vec![]),
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(items) => Ok(items.iter().map(|v| plain(v, ",")).collect()),
        other => Err(Error::config(format!("{what} is a list of names — got {}", type_name(other)))),
    }
}

fn permission(value: &str) -> Result<Permission> {
    match value {
        "allow" => Ok(Permission::Allow),
        "ask" => Ok(Permission::Ask),
        "deny" => Ok(Permission::Deny),
        other => Err(Error::config(format!("writes must be allow, ask or deny — got {}", py_repr(other)))),
    }
}

impl OpenAPIOptions {
    /// Options as a file writes them (`token`, `include`, `writes: ask`, …).
    /// A key that is not an option is refused, as Python refuses an unexpected
    /// keyword argument.
    pub fn from_map(map: &Map<String, Value>) -> Result<Self> {
        let mut o = Self::default();
        for (key, value) in map {
            if !OPTION_NAMES.contains(&key.as_str()) {
                return Err(Error::config(format!("OpenAPIToolkit got an unexpected keyword argument {}", py_repr(key))));
            }
            let text = || match value {
                Value::Null => None,
                other => Some(plain(other, ",")),
            };
            let pairs = |what: &str| -> Result<IndexMap<String, String>> {
                match value {
                    Value::Null => Ok(IndexMap::new()),
                    Value::Object(m) => Ok(m.iter().map(|(k, v)| (k.clone(), plain(v, ","))).collect()),
                    other => Err(Error::config(format!("{what} is a mapping — got {}", type_name(other)))),
                }
            };
            match key.as_str() {
                "base_url" => o.base_url = text(),
                "token" => o.token = text().map(Secret::Value),
                "api_key" => o.api_key = text(),
                "basic" => {
                    let parts = strings(value, "basic")?;
                    if parts.len() != 2 {
                        return Err(Error::config("basic is a (user, password) pair"));
                    }
                    o.basic = Some((parts[0].clone(), parts[1].clone()));
                }
                "credentials" => o.credentials = pairs("credentials")?,
                "headers" => o.headers = pairs("headers")?.into_iter().map(|(k, v)| (k, Secret::Value(v))).collect(),
                "params" => o.params = pairs("params")?.into_iter().map(|(k, v)| (k, Secret::Value(v))).collect(),
                "include" => o.include = if value.is_null() { None } else { Some(strings(value, "include")?) },
                "exclude" => o.exclude = strings(value, "exclude")?,
                "tags" => o.tags = if value.is_null() { None } else { Some(strings(value, "tags")?) },
                "methods" => o.methods = if value.is_null() { None } else { Some(strings(value, "methods")?) },
                "deprecated" => o.deprecated = truthy(value),
                "writes" => o.writes = permission(&text().unwrap_or_default())?,
                "prefix" => o.prefix = text().unwrap_or_default(),
                "name" => o.name = text(),
                "timeout" => o.timeout = value.as_f64().ok_or_else(|| Error::config("timeout is a number of seconds"))?,
                "retries" => o.retries = value.as_u64().ok_or_else(|| Error::config("retries is a whole number"))? as u32,
                "max_response_chars" => o.max_response_chars = value.as_u64().ok_or_else(|| Error::config("max_response_chars is a whole number"))? as usize,
                "cache_reads" => o.cache_reads = truthy(value),
                _ => {}
            }
        }
        Ok(o)
    }
}

/// What the tools share: where the API is, and what goes with every request.
struct Caller {
    base_url: String,
    retries: u32,
    headers: IndexMap<String, Sent>,
    params: IndexMap<String, Sent>,
    cookies: IndexMap<String, Sent>,
    client: reqwest::Client,
}

/// The operations of an API, as tools.
///
/// Made with [`openapi_tools`], [`OpenAPIToolkit::new`] (synchronous; a URL
/// is fetched on a thread of its own) or [`OpenAPIToolkit::load`] (async).
pub struct OpenAPIToolkit {
    title: String,
    name: String,
    base_url: String,
    caller: Arc<Caller>,
    operations: IndexMap<String, Arc<Operation>>,
    tools: Vec<Tool>,
    /// Operations that could not be made into tools, and why.
    skipped: Vec<(String, String)>,
}

impl fmt::Debug for OpenAPIToolkit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<OpenAPIToolkit {} tools={}>", self.title, self.tools.len())
    }
}

impl OpenAPIToolkit {
    /// Read a document and make its operations into tools. A URL is fetched
    /// on a thread of its own, so this works inside a runtime or outside one.
    pub fn new(spec: impl Into<OpenAPISpec>, options: OpenAPIOptions) -> Result<Self> {
        let (data, origin) = match plan(spec.into())? {
            Plan::Ready(data, origin) => (data, origin),
            Plan::Fetch(url) => (fetch_blocking(&url, options.client.clone())?, url),
        };
        Self::build(data, &origin, options)
    }

    /// As [`OpenAPIToolkit::new`], fetching a URL on the caller's runtime.
    pub async fn load(spec: impl Into<OpenAPISpec>, options: OpenAPIOptions) -> Result<Self> {
        let (data, origin) = match plan(spec.into())? {
            Plan::Ready(data, origin) => (data, origin),
            Plan::Fetch(url) => (fetch(&url, options.client.clone()).await?, url),
        };
        Self::build(data, &origin, options)
    }

    fn build(data: Value, origin: &str, options: OpenAPIOptions) -> Result<Self> {
        let document = Document::new(data)?;
        let title = document.data.get("info").and_then(|i| i.get("title")).filter(|t| truthy(t)).map(|t| text_of(Some(t))).unwrap_or_else(|| "API".into());
        let name = argument(options.name.as_deref().filter(|n| !n.is_empty()).unwrap_or(&title)).to_lowercase();
        let base_url = match options.base_url.as_deref().filter(|b| !b.is_empty()) {
            Some(b) => b.to_string(),
            None => document.server(origin),
        }
        .trim_end_matches('/')
        .to_string();
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            return Err(Error::config(format!(
                "{title}: the document does not say where the API is (servers: {}) — pass base_url=\"https://…\"",
                if base_url.is_empty() { "none" } else { &base_url }
            )));
        }

        let mut headers: IndexMap<String, Sent> = options.headers.iter().map(|(k, v)| (k.clone(), Sent::plain(v.clone()))).collect();
        let mut params: IndexMap<String, Sent> = options.params.iter().map(|(k, v)| (k.clone(), Sent::plain(v.clone()))).collect();
        let mut cookies: IndexMap<String, Sent> = IndexMap::new();
        authorise(&document, &title, &options, &mut headers, &mut params, &mut cookies)?;

        let client = match options.client.clone() {
            Some(c) => c,
            None => {
                crate::llm_providers::base::install_tls();
                // Redirects are not followed: a credential must not go to
                // wherever a response points.
                reqwest::Client::builder()
                    .timeout(Duration::from_secs_f64(options.timeout.max(0.1)))
                    .redirect(reqwest::redirect::Policy::none())
                    .build()
                    .map_err(|e| Error::config(format!("{title}: no HTTP client: {e}")))?
            }
        };
        let caller = Arc::new(Caller { base_url: base_url.clone(), retries: options.retries, headers, params, cookies, client });
        let mut kit = Self { title, name, base_url, caller, operations: IndexMap::new(), tools: vec![], skipped: vec![] };

        let wanted_tags: Option<HashSet<&str>> = options.tags.as_ref().map(|t| t.iter().map(String::as_str).collect());
        let wanted_methods: HashSet<String> = match &options.methods {
            Some(m) => m.iter().map(|m| m.to_lowercase()).collect(),
            None => METHODS.iter().map(|m| m.to_string()).collect(),
        };
        let paths = document.data.get("paths").and_then(Value::as_object).cloned().unwrap_or_default();
        for (path, item) in &paths {
            let (item, _) = document.resolve(item, &[])?;
            let Value::Object(item) = item else { continue };
            for method in METHODS {
                let Some(Value::Object(raw)) = item.get(method) else { continue };
                if !wanted_methods.contains(method) {
                    continue;
                }
                let label = format!("{} {path}", method.to_uppercase());
                if raw.get("deprecated").is_some_and(truthy) && !options.deprecated {
                    kit.skipped.push((label, "deprecated".into()));
                    continue;
                }
                if let Some(wanted) = &wanted_tags {
                    let tags = raw.get("tags").and_then(Value::as_array).cloned().unwrap_or_default();
                    if !tags.iter().any(|t| t.as_str().is_some_and(|t| wanted.contains(t))) {
                        continue;
                    }
                }
                let operation_id = raw.get("operationId").filter(|v| truthy(v)).map(|v| text_of(Some(v))).unwrap_or_default();
                let tool_name = kit.tool_name(&options.prefix, &operation_id, method, path);
                let keys = [tool_name.as_str(), operation_id.as_str(), label.as_str()];
                if let Some(include) = &options.include
                    && !keys.iter().any(|k| include.iter().any(|p| fnmatch(k, p)))
                {
                    continue;
                }
                if keys.iter().any(|k| options.exclude.iter().any(|p| fnmatch(k, p))) {
                    continue;
                }
                match read_operation(&document, &kit.caller, method, path, &item, raw) {
                    Ok(operation) => {
                        let operation = Arc::new(operation);
                        let tool = kit.make_tool(&tool_name, operation.clone(), raw, &options);
                        kit.operations.insert(tool_name, operation);
                        kit.tools.push(tool);
                    }
                    Err(Problem::Unsupported(why)) => kit.skipped.push((label, why)),
                    Err(Problem::Fatal(e)) => return Err(e),
                }
            }
        }
        Ok(kit)
    }

    fn tool_name(&self, prefix: &str, operation_id: &str, method: &str, path: &str) -> String {
        let mut base = if operation_id.is_empty() { String::new() } else { NAME.replace_all(operation_id, "_").trim_matches('_').to_string() };
        if base.is_empty() {
            let mut words = vec![method.to_string()];
            words.extend(WORDS.split(path).filter(|w| !w.is_empty()).map(str::to_string));
            base = words.join("_");
        }
        let name = head(&format!("{prefix}{base}"), 64).trim_end_matches('_').to_string();
        let mut candidate = name.clone();
        let mut n = 2;
        while self.operations.contains_key(&candidate) {
            let suffix = format!("_{n}");
            candidate = format!("{}{suffix}", head(&name, 64 - suffix.len()));
            n += 1;
        }
        candidate
    }

    fn make_tool(&self, name: &str, operation: Arc<Operation>, raw: &Map<String, Value>, options: &OpenAPIOptions) -> Tool {
        let summary = operation.summary.trim().to_string();
        let detail = squash(&text_of(raw.get("description")));
        let said = if detail.is_empty() || detail == summary {
            summary.clone()
        } else if !summary.is_empty() {
            format!("{summary}. {detail}")
        } else {
            detail
        };
        let mut description = String::new();
        if !said.is_empty() {
            description = format!("{}. ", head(&said, 900).trim_end_matches('.'));
        }
        description.push_str(&format!("({} {})", operation.method.to_uppercase(), operation.path));
        let reads = READS.contains(&operation.method.as_str());
        let mut tags = vec!["openapi".to_string(), self.name.clone(), (if reads { "read" } else { "write" }).to_string()];
        tags.extend(operation.tags.iter().map(|t| argument(t).to_lowercase()));

        let caller = self.caller.clone();
        let tool_name = name.to_string();
        let op = operation.clone();
        Tool::raw(name, description, operation.parameters(), move |args: Map<String, Value>, _ctx: ToolContext| {
            let caller = caller.clone();
            let op = op.clone();
            let tool_name = tool_name.clone();
            async move { caller.call(&tool_name, &op, args).await }
        })
        .permission(if reads { Permission::Allow } else { options.writes })
        .cacheable(options.cache_reads && operation.method == "get")
        .max_output_chars(options.max_response_chars)
        .tags(tags)
    }

    // ---- the toolkit as a collection -------------------------------------

    /// The document's title.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// What this API is called; the tag its tools carry.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Where requests go.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The tool names, in the document's order.
    pub fn names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name.clone()).collect()
    }

    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// The tools, to hand to an agent.
    pub fn into_tools(self) -> Vec<Tool> {
        self.tools
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.operations.contains_key(name)
    }

    /// Operations that could not be made into tools, and why: `("POST /upload", "it needs a file upload (file)")`.
    pub fn skipped(&self) -> &[(String, String)] {
        &self.skipped
    }

    pub fn get(&self, name: &str) -> Result<&Tool> {
        self.tools.iter().find(|t| t.name == name).ok_or_else(|| self.no_such(name))
    }

    /// The request behind a tool: `{"method", "path", "operation_id", "api"}`
    /// (Python's `tool.operation`).
    pub fn operation(&self, name: &str) -> Option<Value> {
        let op = self.operations.get(name)?;
        Some(json!({"method": op.method.to_uppercase(), "path": op.path, "operation_id": op.operation_id, "api": self.title}))
    }

    fn no_such(&self, name: &str) -> Error {
        let names = self.names();
        Error::tool(format!("no operation {}; there are: {}", py_repr(name), if names.is_empty() { "none".into() } else { names.join(", ") }), name)
    }

    /// Call one operation yourself — the way to check an API is wired up
    /// before an agent is trusted with it. Fails with `Error::Tool`.
    pub async fn call(&self, name: &str, arguments: Value) -> Result<Value> {
        let Some(operation) = self.operations.get(name) else { return Err(self.no_such(name)) };
        let args = match arguments {
            Value::Object(m) => m,
            Value::Null => Map::new(),
            other => return Err(Error::tool(format!("{name} takes an object of arguments — got {}", type_name(&other)), name)),
        };
        self.caller.call(name, operation, args).await
    }

    /// Every tool, the request behind it, and what it takes.
    pub fn describe(&self) -> String {
        let mut lines = vec![format!("{} — {} — {} tools", self.title, self.base_url, self.tools.len())];
        for tool in &self.tools {
            let operation = &self.operations[&tool.name];
            let arguments: Vec<String> = operation.fields.iter().map(|f| format!("{}{}", f.name, if f.required { "" } else { "?" })).collect();
            let mark = match tool.permission {
                Permission::Allow => String::new(),
                Permission::Ask => " [ask]".into(),
                Permission::Deny => " [deny]".into(),
            };
            lines.push(format!("  {}({}){mark}", tool.name, arguments.join(", ")));
            let summary = if operation.summary.is_empty() { String::new() } else { format!(" — {}", operation.summary) };
            lines.push(format!("      {}{summary}", operation.label()));
        }
        for (label, reason) in &self.skipped {
            lines.push(format!("  (skipped {label}: {reason})"));
        }
        lines.join("\n")
    }
}

impl<'a> IntoIterator for &'a OpenAPIToolkit {
    type Item = &'a Tool;
    type IntoIter = std::slice::Iter<'a, Tool>;

    fn into_iter(self) -> Self::IntoIter {
        self.tools.iter()
    }
}

impl IntoIterator for OpenAPIToolkit {
    type Item = Tool;
    type IntoIter = std::vec::IntoIter<Tool>;

    fn into_iter(self) -> Self::IntoIter {
        self.tools.into_iter()
    }
}

fn authorise(
    document: &Document,
    title: &str,
    options: &OpenAPIOptions,
    headers: &mut IndexMap<String, Sent>,
    params: &mut IndexMap<String, Sent>,
    cookies: &mut IndexMap<String, Sent>,
) -> Result<()> {
    let schemes = document.security_schemes()?;
    if let Some(token) = &options.token {
        headers.insert("Authorization".into(), Sent { prefix: "Bearer ".into(), secret: token.clone() });
    }
    if let Some((user, password)) = &options.basic {
        headers.insert("Authorization".into(), Sent::plain(Secret::Value(format!("Basic {}", B64.encode(format!("{user}:{password}"))))));
    }
    let mut credentials: IndexMap<String, String> = IndexMap::new();
    if let Some(key) = &options.api_key {
        let Some(first) = schemes.iter().find(|(_, s)| s.get("type").and_then(Value::as_str) == Some("apiKey")).map(|(n, _)| n.clone()) else {
            return Err(Error::config(format!(
                "{title}: api_key= was given, but the document declares no apiKey security scheme to say where it goes — pass it as headers={{\"X-API-Key\": …}} or params={{\"api_key\": …}} instead"
            )));
        };
        credentials.insert(first, key.clone());
    }
    for (k, v) in &options.credentials {
        credentials.insert(k.clone(), v.clone());
    }
    for (scheme_name, secret) in credentials {
        let Some(scheme) = schemes.get(&scheme_name) else {
            let mut declared: Vec<&String> = schemes.keys().collect();
            declared.sort();
            let declared = if declared.is_empty() { "none".to_string() } else { declared.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ") };
            return Err(Error::config(format!("{title}: no security scheme called {}; the document declares: {declared}", py_repr(&scheme_name))));
        };
        let kind = scheme.get("type").and_then(Value::as_str).unwrap_or("");
        if kind == "apiKey" {
            let key = scheme.get("name").filter(|n| truthy(n)).map(|n| text_of(Some(n))).unwrap_or_else(|| scheme_name.clone());
            let target = match scheme.get("in").and_then(Value::as_str).unwrap_or("header") {
                "query" => &mut *params,
                "cookie" => &mut *cookies,
                _ => &mut *headers,
            };
            target.insert(key, Sent::plain(Secret::Value(secret)));
        } else if kind == "http" && text_of(scheme.get("scheme")).eq_ignore_ascii_case("basic") {
            headers.insert("Authorization".into(), Sent::plain(Secret::Value(format!("Basic {}", B64.encode(secret)))));
        } else {
            // http bearer, oauth2, openIdConnect: a token, sent the same way.
            headers.insert("Authorization".into(), Sent::plain(Secret::Value(format!("Bearer {secret}"))));
        }
    }
    Ok(())
}

fn read_operation(
    document: &Document,
    caller: &Caller,
    method: &str,
    path: &str,
    item: &Map<String, Value>,
    raw: &Map<String, Value>,
) -> Result<Operation, Problem> {
    let mut operation = Operation {
        method: method.into(),
        path: path.into(),
        operation_id: raw.get("operationId").filter(|v| truthy(v)).map(|v| text_of(Some(v))).unwrap_or_default(),
        summary: text_of(raw.get("summary")),
        tags: raw.get("tags").and_then(Value::as_array).map(|t| t.iter().map(|v| text_of(Some(v))).collect()).unwrap_or_default(),
        ..Default::default()
    };
    let fixed_headers: HashSet<String> = caller.headers.keys().map(|h| h.to_lowercase()).collect();
    let mut taken: HashSet<String> = HashSet::new();
    let mut claim = |wanted: &str, place: &str| -> String {
        let mut name = if taken.contains(wanted) { format!("{wanted}_{place}") } else { wanted.to_string() };
        let mut n = 2;
        while taken.contains(&name) {
            name = format!("{wanted}_{place}_{n}");
            n += 1;
        }
        taken.insert(name.clone());
        name
    };

    // The operation's own parameters replace the path's of the same name.
    let mut merged: IndexMap<(String, String), Map<String, Value>> = IndexMap::new();
    let listed =
        item.get("parameters").and_then(Value::as_array).into_iter().flatten().chain(raw.get("parameters").and_then(Value::as_array).into_iter().flatten());
    for entry in listed {
        let (param, _) = document.resolve(entry, &[])?;
        if let Value::Object(param) = param
            && param.get("name").is_some_and(truthy)
            && param.get("in").is_some_and(truthy)
        {
            merged.insert((text_of(param.get("name")), text_of(param.get("in"))), param);
        }
    }

    let mut body_schema: Option<Value> = None;
    let mut body_required = false;
    let mut form_fields: Vec<Map<String, Value>> = vec![];
    for ((wire, place), param) in &merged {
        let place = place.as_str();
        if place == "body" {
            // Swagger 2
            body_schema = Some(param.get("schema").cloned().filter(|s| !s.is_null()).unwrap_or_else(|| json!({})));
            operation.body_kind = "json".into();
            operation.media_type = "application/json".into();
            body_required = param.get("required").is_some_and(truthy);
            continue;
        }
        if place == "formData" {
            form_fields.push(param.clone());
            continue;
        }
        if !matches!(place, "path" | "query" | "header" | "cookie") {
            continue;
        }
        let lower = wire.to_lowercase();
        if place == "header" && (OWN_HEADERS.contains(&lower.as_str()) || fixed_headers.contains(&lower)) {
            continue;
        }
        if place == "query" && caller.params.contains_key(wire) {
            continue;
        }
        if place == "cookie" && caller.cookies.contains_key(wire) {
            continue;
        }
        let mut encoded = false;
        let mut schema = match param.get("content") {
            Some(Value::Object(content)) if !content.is_empty() => {
                encoded = true;
                let media = content.values().next().cloned().unwrap_or(Value::Null);
                document.schema(media.get("schema").unwrap_or(&Value::Null), &[], 0, true)?
            }
            _ => match param.get("schema") {
                Some(s) => document.schema(s, &[], 0, true)?,
                None => document.schema(&keep_of(param), &[], 0, true)?, // Swagger 2: the type is inline
            },
        };
        if param.get("description").is_some_and(truthy) && !schema.contains_key("description") {
            schema.insert("description".into(), Value::String(head(&squash(&text_of(param.get("description"))), 500)));
        }
        let style = match param.get("style").filter(|s| truthy(s)) {
            Some(s) => text_of(Some(s)),
            None => (if matches!(place, "query" | "cookie") { "form" } else { "simple" }).into(),
        };
        let mut explode = match param.get("explode") {
            Some(v) => truthy(v),
            None => style == "form",
        };
        match param.get("collectionFormat") {
            Some(Value::String(f)) if f == "multi" => explode = true,
            Some(_) => explode = false,
            None => {}
        }
        let name = claim(&argument(wire), place);
        let required = param.get("required").is_some_and(truthy) || place == "path";
        let field = Field { name, wire: wire.clone(), place: place.into(), required, explode, style, encoded };
        if schema.is_empty() {
            schema.insert("type".into(), json!("string"));
        }
        operation.add(field, schema);
    }

    for cap in TEMPLATE.captures_iter(path) {
        let name = &cap[1];
        if !operation.fields.iter().any(|f| f.place == "path" && f.wire == name) {
            // In the path and not declared: still has to be filled in.
            let field = Field::new(claim(&argument(name), "path"), name, "path", true);
            operation.add(field, json!({"type": "string"}).as_object().cloned().unwrap_or_default());
        }
    }

    if !form_fields.is_empty() {
        let consumes =
            raw.get("consumes").filter(|c| truthy(c)).or_else(|| document.data.get("consumes")).and_then(Value::as_array).cloned().unwrap_or_default();
        operation.body_kind = if consumes.iter().any(|c| c.as_str() == Some("multipart/form-data")) { "multipart" } else { "form" }.into();
        let mut properties = Map::new();
        for f in &form_fields {
            properties.insert(text_of(f.get("name")), keep_of(f));
        }
        let required: Vec<Value> = form_fields.iter().filter(|f| f.get("required").is_some_and(truthy)).map(|f| json!(text_of(f.get("name")))).collect();
        body_schema = Some(json!({"type": "object", "properties": properties, "required": required}));
    }
    let (request_body, _) = document.resolve(raw.get("requestBody").unwrap_or(&Value::Null), &[])?;
    if let Some(Value::Object(content)) = request_body.get("content") {
        body_required = request_body.get("required").is_some_and(truthy);
        let mut kind = "json";
        let mut chosen = content.keys().find(|m| m.to_lowercase().contains("json")).cloned();
        if chosen.is_none() {
            for (wanted, label) in [("application/x-www-form-urlencoded", "form"), ("multipart/form-data", "multipart"), ("text/", "text")] {
                chosen = content.keys().find(|m| m.to_lowercase().starts_with(wanted)).cloned();
                if chosen.is_some() {
                    kind = label;
                    break;
                }
            }
        }
        let Some(chosen) = chosen else {
            let listed: Vec<&str> = content.keys().map(String::as_str).collect();
            let listed = if listed.is_empty() { "undeclared".into() } else { listed.join(", ") };
            return Err(Problem::Unsupported(format!("its request body is {listed}, which cannot be sent from a tool call")));
        };
        operation.body_kind = kind.into();
        operation.media_type = chosen.clone();
        let media = content.get(&chosen).cloned().unwrap_or(Value::Null);
        body_schema = Some(media.get("schema").cloned().filter(|s| !s.is_null()).unwrap_or_else(|| json!({})));
        operation.deep = media
            .get("encoding")
            .and_then(Value::as_object)
            .map(|enc| enc.iter().filter(|(_, how)| how.get("style").and_then(Value::as_str) == Some("deepObject")).map(|(n, _)| n.clone()).collect())
            .unwrap_or_default();
    }

    if let Some(raw_schema) = body_schema {
        read_body(document, &mut operation, &raw_schema, body_required, &mut claim)?;
    }
    Ok(operation)
}

/// The request body, as arguments. An object's fields become arguments of
/// their own, which is what a model fills in best; anything else is one
/// argument called `body`.
fn read_body(
    document: &Document,
    operation: &mut Operation,
    raw_schema: &Value,
    required: bool,
    claim: &mut dyn FnMut(&str, &str) -> String,
) -> Result<(), Problem> {
    let (resolved, seen) = document.resolve(raw_schema, &[])?;
    let mut schema = document.schema(raw_schema, &[], 0, true)?;
    let needed: Vec<String> =
        schema.get("required").and_then(Value::as_array).map(|r| r.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let mut binary = vec![];
    if let Some(Value::Object(props)) = resolved.get("properties") {
        for (name, prop) in props {
            if is_file(&document.resolve(prop, &seen)?.0) {
                binary.push(name.clone());
            }
        }
    }
    for name in &binary {
        if needed.contains(name) {
            return Err(Problem::Unsupported(format!("it needs a file upload ({name})")));
        }
        if let Some(Value::Object(props)) = schema.get_mut("properties") {
            props.shift_remove(name);
        }
    }
    let properties = schema.get("properties").and_then(Value::as_object).cloned().unwrap_or_default();
    let flat = schema.get("type").and_then(Value::as_str) == Some("object")
        && !properties.is_empty()
        && !schema.contains_key("additionalProperties")
        && !schema.contains_key("anyOf");
    if operation.body_kind == "text" || !flat {
        if matches!(operation.body_kind.as_str(), "form" | "multipart") {
            // A form with no fields declared: there is nothing to fill in, and
            // the operation is called without a body.
            operation.body_kind = String::new();
            return Ok(());
        }
        operation.whole_body = true;
        if operation.body_kind == "text" {
            let mut text = Map::new();
            text.insert("type".into(), json!("string"));
            if let Some(d) = schema.get("description") {
                text.insert("description".into(), d.clone());
            }
            schema = text;
        }
        schema.entry("description").or_insert_with(|| json!("The request body."));
        let field = Field::new(claim("body", "body"), "body", "body", required);
        operation.add(field, schema);
        return Ok(());
    }
    for (name, prop) in properties {
        let field = Field::new(claim(&argument(&name), "body"), &name, "body", needed.contains(&name));
        operation.add(field, prop.as_object().cloned().unwrap_or_default());
    }
    Ok(())
}

impl Caller {
    /// Build the request the document describes, send it, read the answer.
    async fn call(&self, name: &str, operation: &Operation, arguments: Map<String, Value>) -> Result<Value> {
        let known: IndexMap<&str, &Field> = operation.fields.iter().map(|f| (f.name.as_str(), f)).collect();
        let mut unknown: Vec<&str> = arguments.keys().map(String::as_str).filter(|k| !known.contains_key(k)).collect();
        if !unknown.is_empty() {
            unknown.sort();
            let takes: Vec<&str> = known.keys().copied().collect();
            return Err(Error::tool(
                format!("{name} does not take {}; it takes {}", unknown.join(", "), if takes.is_empty() { "no arguments".into() } else { takes.join(", ") }),
                name,
            ));
        }
        let given: Map<String, Value> = arguments.into_iter().filter(|(_, v)| !v.is_null()).collect();
        let missing: Vec<&str> = operation.fields.iter().filter(|f| f.required && !given.contains_key(&f.name)).map(|f| f.name.as_str()).collect();
        if !missing.is_empty() {
            return Err(Error::tool(format!("{name} needs {}", missing.join(", ")), name));
        }

        let mut path = operation.path.clone();
        let mut query: Vec<(String, String)> = vec![];
        let mut headers: IndexMap<String, String> = IndexMap::new();
        headers.insert("accept".into(), ACCEPT.into());
        let mut cookies: IndexMap<String, String> = IndexMap::new();
        let mut body = Map::new();
        let mut whole: Option<Value> = None;
        for (key, value) in &given {
            let field = known[key.as_str()];
            match field.place.as_str() {
                "path" => path = path.replace(&format!("{{{}}}", field.wire), &quote(&plain(value, ","), false)),
                "query" => query.extend(query_pairs(field, value)),
                "header" => {
                    headers.insert(field.wire.clone(), plain(value, ","));
                }
                "cookie" => {
                    cookies.insert(field.wire.clone(), plain(value, ","));
                }
                _ if operation.whole_body => whole = Some(value.clone()),
                _ => {
                    body.insert(field.wire.clone(), value.clone());
                }
            }
        }
        for (key, sent) in &self.params {
            query.push((key.clone(), sent.value().await?));
        }
        for (key, sent) in &self.headers {
            headers.insert(key.clone(), sent.value().await?);
        }
        for (key, sent) in &self.cookies {
            cookies.insert(key.clone(), sent.value().await?);
        }
        if !cookies.is_empty() {
            headers.insert("cookie".into(), cookies.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("; "));
        }

        let payload = if operation.whole_body {
            whole
        } else if body.is_empty() {
            None
        } else {
            Some(Value::Object(body))
        };
        let mut content: Option<Vec<u8>> = None;
        if !operation.body_kind.is_empty()
            && let Some(payload) = payload
        {
            match operation.body_kind.as_str() {
                "json" => {
                    let media = if operation.media_type.is_empty() { "application/json" } else { operation.media_type.as_str() };
                    headers.insert("content-type".into(), media.into());
                    content = Some(serde_json::to_vec(&payload).unwrap_or_default());
                }
                "form" => {
                    let fields = payload.as_object().cloned().unwrap_or_default();
                    content = Some(urlencode(&form_pairs(&fields, &operation.deep)).into_bytes());
                    headers.insert("content-type".into(), "application/x-www-form-urlencoded".into());
                }
                "multipart" => {
                    let fields: Vec<FormField> =
                        payload.as_object().map(|m| m.iter().map(|(k, v)| (k.clone(), plain(v, ","), None)).collect()).unwrap_or_default();
                    let (boundary, bytes) = multipart(&fields);
                    headers.insert("content-type".into(), format!("multipart/form-data; boundary={boundary}"));
                    content = Some(bytes);
                }
                _ => {
                    let text = match &payload {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    content = Some(text.into_bytes());
                    let media = if operation.media_type.is_empty() { "text/plain" } else { operation.media_type.as_str() };
                    headers.insert("content-type".into(), media.into());
                }
            }
        }

        let url = format!("{}{path}", self.base_url);
        let method = operation.method.to_uppercase();
        let verb = reqwest::Method::from_bytes(method.as_bytes()).unwrap_or(reqwest::Method::GET);
        let attempts = if READS.contains(&operation.method.as_str()) { self.retries + 1 } else { 1 };
        for attempt in 0..attempts {
            let last = attempt + 1 == attempts;
            let mut request = self.client.request(verb.clone(), &url);
            if !query.is_empty() {
                request = request.query(&query);
            }
            for (k, v) in &headers {
                request = request.header(k.as_str(), v.as_str());
            }
            if let Some(bytes) = &content {
                request = request.body(bytes.clone());
            }
            match request.send().await {
                Err(e) => {
                    if last {
                        let kind = if e.is_timeout() {
                            "TimeoutException"
                        } else if e.is_connect() {
                            "ConnectError"
                        } else {
                            "HTTPError"
                        };
                        return Err(Error::tool(format!("{method} {} could not be reached: {kind}: {e}", operation.path), name));
                    }
                }
                Ok(response) => {
                    if !RETRY_STATUS.contains(&response.status().as_u16()) || last {
                        return answer(name, operation, response).await;
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs_f64((0.3 * 2f64.powi(attempt as i32)).min(5.0))).await;
        }
        Err(Error::tool(format!("{method} {} did not answer", operation.path), name))
    }
}

async fn answer(name: &str, operation: &Operation, response: reqwest::Response) -> Result<Value> {
    let status = response.status().as_u16();
    let header = |h: &str| response.headers().get(h).and_then(|v| v.to_str().ok()).map(str::to_string);
    let kind = header("content-type").unwrap_or_default().to_lowercase();
    let location = header("location");
    let label = operation.label();
    if (300..400).contains(&status) {
        return Err(Error::tool(
            format!("{label} answered {status}, a redirect to {}, which is not followed", location.as_deref().unwrap_or("elsewhere")),
            name,
        ));
    }
    let bytes = response.bytes().await.map_err(|e| Error::tool(format!("{label} could not be read: {e}"), name))?;
    let text = || String::from_utf8_lossy(&bytes).into_owned();
    if status >= 400 {
        let detail = head(&squash(&text()), 600);
        return Err(Error::tool(format!("{label} returned {status}{}", if detail.is_empty() { String::new() } else { format!(": {detail}") }), name));
    }
    if status == 204 || bytes.is_empty() {
        return Ok(Value::String(format!("{label} succeeded ({status}, no content).")));
    }
    if kind.contains("json") {
        return Ok(serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(text())));
    }
    if kind.starts_with("text/") || ["xml", "yaml", "csv", "javascript"].iter().any(|w| kind.contains(w)) || kind.is_empty() {
        return Ok(Value::String(text()));
    }
    Ok(Value::String(format!("{label} returned {} bytes of {}, which is not text.", bytes.len(), kind.split(';').next().unwrap_or(""))))
}

/// An OpenAPI document as tools. See [`OpenAPIToolkit`] and [`OpenAPIOptions`].
pub fn openapi_tools(spec: impl Into<OpenAPISpec>, options: OpenAPIOptions) -> Result<OpenAPIToolkit> {
    OpenAPIToolkit::new(spec, options)
}

/// The toolkits a file declares under `openapi:` — what a blueprint or a
/// workflow file builds.
///
/// ```yaml
/// openapi:
///   shop:
///     spec: ./shop.openapi.json
///     token_env: SHOP_TOKEN          # the secret is named, never written
///     include: [getOrder, listOrders]
///     writes: ask
/// ```
///
/// An entry may be just the spec (`shop: shop.json`). A relative `spec` is
/// looked for next to the file that declares it (`base_dir`). `token_env` and
/// `api_key_env` name environment variables, read now. The entry's key is the
/// toolkit's `name` unless it sets one.
pub fn toolkits_from_config(config: &Map<String, Value>, base_dir: Option<&Path>) -> Result<Vec<OpenAPIToolkit>> {
    let mut out = vec![];
    for (name, entry) in config {
        let mut options: Map<String, Value> = match entry {
            Value::String(s) => {
                let mut m = Map::new();
                m.insert("spec".into(), Value::String(s.clone()));
                m
            }
            Value::Object(m) => m.clone(),
            Value::Null => Map::new(),
            other => return Err(Error::config(format!("openapi.{name} is a spec or a mapping — got {}", type_name(other)))),
        };
        let spec = options.shift_remove("spec").filter(truthy);
        let url = options.shift_remove("url").filter(truthy);
        let Some(mut spec) = spec.or(url) else {
            return Err(Error::config(format!("openapi.{name} needs a `spec`: a file or a URL")));
        };
        if let (Value::String(s), Some(base)) = (&spec, base_dir)
            && !s.contains('\n')
            && !["http://", "https://", "{", "/", "~"].iter().any(|p| s.starts_with(p))
            && !Path::new(s).is_absolute()
        {
            spec = Value::String(base.join(s).to_string_lossy().into_owned());
        }
        for key in ["token", "api_key"] {
            if let Some(variable) = options.shift_remove(&format!("{key}_env")).filter(truthy) {
                let variable = text_of(Some(&variable));
                match std::env::var(&variable).ok().filter(|v| !v.is_empty()) {
                    Some(v) => {
                        options.insert(key.into(), Value::String(v));
                    }
                    None => return Err(Error::config(format!("openapi.{name}: the environment variable {variable} is not set"))),
                }
            }
        }
        if !options.contains_key("name") {
            options.insert("name".into(), Value::String(name.clone()));
        }
        let parsed = OpenAPIOptions::from_map(&options).map_err(|e| Error::config(format!("openapi.{name}: {e}")))?;
        out.push(OpenAPIToolkit::new(spec, parsed)?);
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Mutex;

    use axum::body::Bytes;
    use axum::http::{HeaderMap, Method, StatusCode, Uri};
    use axum::response::{IntoResponse, Response};

    fn shop_doc() -> Value {
        json!({
            "openapi": "3.0.3",
            "info": {"title": "Shop API", "version": "1.2.0"},
            "servers": [{"url": "https://{region}.shop.test/v1", "variables": {"region": {"default": "eu"}}}],
            "paths": {
                "/orders": {
                    "get": {
                        "operationId": "listOrders", "summary": "List orders", "tags": ["orders"],
                        "parameters": [
                            {"name": "status", "in": "query", "schema": {"type": "array", "items": {"$ref": "#/components/schemas/Status"}}},
                            {"name": "customer.id", "in": "query", "schema": {"type": "string"}},
                            {"name": "fields", "in": "query", "explode": false, "schema": {"type": "array", "items": {"type": "string"}}},
                            {"name": "filter", "in": "query", "style": "deepObject", "schema": {"type": "object"}},
                            {"$ref": "#/components/parameters/Limit"},
                            {"name": "X-Request-Id", "in": "header", "schema": {"type": "string"}},
                            {"name": "Authorization", "in": "header", "schema": {"type": "string"}}
                        ],
                        "responses": {"200": {"description": "ok"}}
                    },
                    "post": {
                        "operationId": "createOrder", "summary": "Create an order",
                        "description": "Creates an order for a customer.", "tags": ["orders"],
                        "requestBody": {"required": true, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/NewOrder"}}}},
                        "responses": {"201": {"description": "created"}}
                    }
                },
                "/orders/{orderId}": {
                    "parameters": [{"name": "orderId", "in": "path", "required": true, "description": "The order number.", "schema": {"type": "string"}}],
                    "get": {"operationId": "getOrder", "summary": "Get one order", "tags": ["orders"], "responses": {"200": {"description": "ok"}}},
                    "delete": {"operationId": "cancel order!", "tags": ["orders"], "responses": {"204": {"description": "gone"}}},
                    "patch": {
                        "summary": "Change an order",
                        "requestBody": {"content": {"application/merge-patch+json": {"schema": {"type": "object", "properties": {"orderId": {"type": "string"}, "note": {"type": "string"}}}}}},
                        "responses": {"200": {"description": "ok"}}
                    }
                },
                "/orders/{orderId}/tags": {
                    "put": {"operationId": "setTags",
                            "requestBody": {"required": true, "content": {"application/json": {"schema": {"type": "array", "items": {"type": "string"}}}}},
                            "responses": {"200": {"description": "ok"}}}
                },
                "/login": {
                    "post": {"operationId": "login", "tags": ["auth"],
                             "requestBody": {"content": {"application/x-www-form-urlencoded": {
                                 "encoding": {"scopes": {"style": "deepObject"}},
                                 "schema": {"type": "object", "required": ["user"], "properties": {
                                     "user": {"type": "string"}, "remember": {"type": "boolean"}, "device": {"type": "object"},
                                     "roles": {"type": "array", "items": {"type": "string"}},
                                     "scopes": {"type": "array", "items": {"type": "string"}}}}}}},
                             "responses": {"200": {"description": "ok"}}}
                },
                "/notes": {
                    "post": {"operationId": "addNote",
                             "requestBody": {"required": true, "content": {"text/plain": {"schema": {"type": "string"}}}},
                             "responses": {"200": {"description": "ok"}}}
                },
                "/upload": {
                    "post": {"operationId": "upload",
                             "requestBody": {"content": {"multipart/form-data": {"schema": {"type": "object", "required": ["file"],
                                 "properties": {"file": {"type": "string", "format": "binary"}}}}}},
                             "responses": {"200": {"description": "ok"}}}
                },
                "/legacy": {"get": {"operationId": "oldThing", "deprecated": true, "responses": {"200": {"description": "ok"}}}},
                "/tree": {"get": {"operationId": "getTree", "parameters": [
                    {"name": "node", "in": "query", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Node"}}}}],
                    "responses": {"200": {"description": "ok"}}}}
            },
            "components": {
                "parameters": {"Limit": {"name": "limit", "in": "query", "description": "How many to return.",
                                         "schema": {"type": "integer", "default": 20, "maximum": 100}}},
                "schemas": {
                    "Status": {"type": "string", "enum": ["open", "shipped", "refunded"]},
                    "Base": {"type": "object", "required": ["customer"], "properties": {
                        "customer": {"type": "string", "description": "Customer id.", "example": "c_1", "x-internal": true}}},
                    "NewOrder": {"allOf": [
                        {"$ref": "#/components/schemas/Base"},
                        {"type": "object", "required": ["items"], "properties": {
                            "id": {"type": "string", "readOnly": true},
                            "items": {"type": "array", "items": {"type": "object", "properties": {
                                "sku": {"type": "string"}, "quantity": {"type": "integer", "minimum": 1}}}},
                            "status": {"$ref": "#/components/schemas/Status"},
                            "gift": {"type": "boolean", "nullable": true}}}]},
                    "Node": {"type": "object", "properties": {
                        "name": {"type": "string"},
                        "children": {"type": "array", "items": {"$ref": "#/components/schemas/Node"}}}}
                },
                "securitySchemes": {
                    "key": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
                    "session": {"type": "apiKey", "in": "query", "name": "session"},
                    "oauth": {"type": "oauth2", "flows": {}},
                    "login": {"type": "http", "scheme": "basic"}
                }
            }
        })
    }

    /// One request the fake server saw.
    #[derive(Debug, Clone)]
    pub(crate) struct Seen {
        pub(crate) method: String,
        pub(crate) path: String,
        pub(crate) query: Vec<(String, String)>,
        pub(crate) headers: HeaderMap,
        pub(crate) body: Vec<u8>,
    }

    impl Seen {
        pub(crate) fn header(&self, name: &str) -> String {
            self.headers.get(name).and_then(|v| v.to_str().ok()).unwrap_or("").to_string()
        }
        pub(crate) fn param(&self, name: &str) -> Option<String> {
            self.query.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
        }
    }

    fn decode(text: &str) -> String {
        let text = text.replace('+', " ");
        let bytes = text.as_bytes();
        let mut out = vec![];
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%'
                && i + 3 <= bytes.len()
                && let Some(b) = std::str::from_utf8(&bytes[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok())
            {
                out.push(b);
                i += 3;
                continue;
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    pub(crate) type Answer = Arc<dyn Fn(&Seen) -> (u16, Vec<(&'static str, String)>, Vec<u8>) + Send + Sync>;

    pub(crate) struct Server {
        pub(crate) url: String,
        pub(crate) seen: Arc<Mutex<Vec<Seen>>>,
    }

    impl Server {
        pub(crate) fn last(&self) -> Seen {
            self.seen.lock().unwrap().last().cloned().expect("a request")
        }
        pub(crate) fn count(&self) -> usize {
            self.seen.lock().unwrap().len()
        }
    }

    pub(crate) async fn serve(answer: Answer) -> Server {
        let seen: Arc<Mutex<Vec<Seen>>> = Arc::default();
        let record = seen.clone();
        let app = axum::Router::new().fallback(move |method: Method, uri: Uri, headers: HeaderMap, body: Bytes| {
            let record = record.clone();
            let answer = answer.clone();
            async move {
                let query = uri
                    .query()
                    .map(|q| {
                        q.split('&')
                            .filter(|p| !p.is_empty())
                            .map(|p| p.split_once('=').map(|(k, v)| (decode(k), decode(v))).unwrap_or((decode(p), String::new())))
                            .collect()
                    })
                    .unwrap_or_default();
                let request = Seen { method: method.to_string(), path: uri.path().to_string(), query, headers, body: body.to_vec() };
                record.lock().unwrap().push(request.clone());
                let (status, headers, body) = answer(&request);
                let mut response: Response = (StatusCode::from_u16(status).unwrap(), body).into_response();
                for (k, v) in headers {
                    response.headers_mut().insert(k, v.parse().unwrap());
                }
                response
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Server { url, seen }
    }

    fn ok_json() -> Answer {
        Arc::new(|_| (200, vec![("content-type", "application/json".into())], br#"{"ok":true}"#.to_vec()))
    }

    pub(crate) fn json_answer(body: Value) -> (u16, Vec<(&'static str, String)>, Vec<u8>) {
        (200, vec![("content-type", "application/json".into())], serde_json::to_vec(&body).unwrap())
    }

    async fn shop(answer: Answer, options: OpenAPIOptions) -> (OpenAPIToolkit, Server) {
        let server = serve(answer).await;
        let options = OpenAPIOptions { base_url: Some(format!("{}/v1", server.url)), ..options };
        (openapi_tools(shop_doc(), options).unwrap(), server)
    }

    fn plain_shop(options: OpenAPIOptions) -> OpenAPIToolkit {
        openapi_tools(shop_doc(), options).unwrap()
    }

    #[test]
    fn every_operation_is_a_tool_with_the_operations_own_arguments() {
        let api = plain_shop(OpenAPIOptions::default());
        assert_eq!(api.names(), ["listOrders", "createOrder", "getOrder", "cancel_order", "patch_orders_orderId", "setTags", "login", "addNote", "getTree"]);
        assert_eq!(api.base_url(), "https://eu.shop.test/v1");
        assert_eq!(api.title(), "Shop API");
        let skipped: HashSet<(String, String)> = api.skipped().iter().cloned().collect();
        let expected: HashSet<(String, String)> =
            [("POST /upload".to_string(), "it needs a file upload (file)".to_string()), ("GET /legacy".to_string(), "deprecated".to_string())]
                .into_iter()
                .collect();
        assert_eq!(skipped, expected);
        assert!(api.len() == 9 && api.contains("getOrder"));

        let listing = api.get("listOrders").unwrap();
        assert_eq!(listing.description, "List orders. (GET /orders)");
        let properties = listing.parameters["properties"].as_object().unwrap();
        // Named so a model can say them; the credential's header is not offered.
        assert_eq!(properties.keys().collect::<Vec<_>>(), ["status", "customer_id", "fields", "filter", "limit", "X-Request-Id"]);
        assert_eq!(properties["status"], json!({"type": "array", "items": {"type": "string", "enum": ["open", "shipped", "refunded"]}}));
        assert_eq!(properties["limit"], json!({"type": "integer", "default": 20, "maximum": 100, "description": "How many to return."}));
        assert!(listing.parameters.get("required").is_none());
        let tags: HashSet<&str> = listing.tags.iter().map(String::as_str).collect();
        assert_eq!(tags, ["openapi", "shop_api", "read", "orders"].into_iter().collect());
        assert_eq!(api.operation("listOrders").unwrap(), json!({"method": "GET", "path": "/orders", "operation_id": "listOrders", "api": "Shop API"}));

        assert_eq!(
            api.get("getOrder").unwrap().parameters,
            json!({"type": "object", "required": ["orderId"], "properties": {"orderId": {"type": "string", "description": "The order number."}}})
        );
    }

    #[test]
    fn a_request_body_becomes_arguments() {
        let api = plain_shop(OpenAPIOptions::default());
        let create = api.get("createOrder").unwrap();
        assert_eq!(create.description, "Create an order. Creates an order for a customer. (POST /orders)");
        let properties = create.parameters["properties"].as_object().unwrap();
        // `allOf` merged, the reference followed, and what the server sets left out.
        assert_eq!(properties.keys().collect::<Vec<_>>(), ["customer", "items", "status", "gift"]);
        assert_eq!(properties["customer"], json!({"type": "string", "description": "Customer id."}));
        assert_eq!(properties["items"]["items"]["properties"]["quantity"], json!({"type": "integer", "minimum": 1}));
        assert_eq!(create.parameters["required"], json!(["customer", "items"]));
        assert!(create.tags.contains("write"));

        // A body that is not an object is one argument.
        assert_eq!(
            api.get("setTags").unwrap().parameters,
            json!({"type": "object", "required": ["orderId", "body"], "properties": {
                "orderId": {"type": "string"},
                "body": {"type": "array", "items": {"type": "string"}, "description": "The request body."}}})
        );
        // A body field with a parameter's name is told apart.
        let patch = api.get("patch_orders_orderId").unwrap().parameters["properties"].as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(patch, ["orderId", "orderId_body", "note"]);
        // A schema that refers to itself ends, rather than going round for ever.
        let node = &api.get("getTree").unwrap().parameters["properties"]["node"];
        assert_eq!(node["properties"]["children"]["items"], json!({"type": "object"}));
    }

    #[tokio::test]
    async fn arguments_go_where_the_document_says() {
        let (api, server) = shop(ok_json(), OpenAPIOptions::default()).await;
        api.call(
            "listOrders",
            json!({"status": ["open", "shipped"], "customer_id": "c 1", "fields": ["id", "total"], "filter": {"min": 10, "paid": true}, "limit": 5, "X-Request-Id": "r-1"}),
        )
        .await
        .unwrap();
        let sent = server.last();
        assert_eq!((sent.method.as_str(), sent.path.as_str()), ("GET", "/v1/orders"));
        let expect: Vec<(String, String)> = [
            ("status", "open"),
            ("status", "shipped"),
            ("customer.id", "c 1"),
            ("fields", "id,total"),
            ("filter[min]", "10"),
            ("filter[paid]", "true"),
            ("limit", "5"),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
        assert_eq!(sent.query, expect);
        assert_eq!(sent.header("x-request-id"), "r-1");

        api.call("getOrder", json!({"orderId": "41/82 ?"})).await.unwrap();
        assert_eq!(server.last().path, "/v1/orders/41%2F82%20%3F"); // stays one segment

        api.call("getTree", json!({"node": {"name": "root", "children": []}})).await.unwrap();
        let node: Value = serde_json::from_str(&server.last().param("node").unwrap()).unwrap();
        assert_eq!(node, json!({"name": "root", "children": []}));
    }

    #[tokio::test]
    async fn bodies_are_sent_as_the_document_says() {
        let (api, server) = shop(ok_json(), OpenAPIOptions::default()).await;
        api.call("createOrder", json!({"customer": "c_1", "gift": null, "items": [{"sku": "tea", "quantity": 2}]})).await.unwrap();
        let sent = server.last();
        assert_eq!(sent.method, "POST");
        assert_eq!(sent.header("content-type"), "application/json");
        assert_eq!(serde_json::from_slice::<Value>(&sent.body).unwrap(), json!({"customer": "c_1", "items": [{"sku": "tea", "quantity": 2}]}));

        api.call("setTags", json!({"orderId": "7", "body": ["gift", "fragile"]})).await.unwrap();
        assert_eq!(server.last().method, "PUT");
        assert_eq!(serde_json::from_slice::<Value>(&server.last().body).unwrap(), json!(["gift", "fragile"]));

        api.call("patch_orders_orderId", json!({"orderId": "7", "orderId_body": "8", "note": "hi"})).await.unwrap();
        assert_eq!(server.last().header("content-type"), "application/merge-patch+json");
        assert_eq!(serde_json::from_slice::<Value>(&server.last().body).unwrap(), json!({"orderId": "8", "note": "hi"}));
        assert_eq!(server.last().path, "/v1/orders/7");

        api.call("login", json!({"user": "ada", "remember": true})).await.unwrap();
        assert_eq!(server.last().header("content-type"), "application/x-www-form-urlencoded");
        assert_eq!(server.last().body, b"user=ada&remember=true");
        api.call("login", json!({"user": "ada", "device": {"os": "linux", "tags": ["a"]}, "roles": ["admin", "dev"], "scopes": ["read"]})).await.unwrap();
        assert_eq!(
            String::from_utf8(server.last().body).unwrap(),
            "user=ada&device%5Bos%5D=linux&device%5Btags%5D%5B0%5D=a&roles=admin&roles=dev&scopes%5B0%5D=read"
        );

        api.call("addNote", json!({"body": "call back tomorrow"})).await.unwrap();
        assert_eq!(server.last().body, b"call back tomorrow");
        assert_eq!(server.last().header("content-type"), "text/plain");
    }

    #[tokio::test]
    async fn what_comes_back_is_what_the_model_reads() {
        let answer: Answer = Arc::new(|r: &Seen| match (r.method.as_str(), r.path.as_str()) {
            ("GET", "/v1/orders/1") => json_answer(json!({"id": "1", "status": "shipped"})),
            ("GET", "/v1/orders/2") => (200, vec![("content-type", "text/plain".into())], b"plain words".to_vec()),
            ("GET", "/v1/orders/3") => (200, vec![("content-type", "image/png".into())], b"\x89PNG".to_vec()),
            ("DELETE", "/v1/orders/1") => (204, vec![], vec![]),
            ("GET", "/v1/orders/404") => (404, vec![("content-type", "application/json".into())], br#"{"error":"no such order"}"#.to_vec()),
            ("GET", "/v1/orders/302") => (302, vec![("location", "https://evil.test".into())], vec![]),
            _ => json_answer(json!({"ok": true})),
        });
        let (api, _server) = shop(answer, OpenAPIOptions::default()).await;
        assert_eq!(api.call("getOrder", json!({"orderId": "1"})).await.unwrap(), json!({"id": "1", "status": "shipped"}));
        assert_eq!(api.call("getOrder", json!({"orderId": "2"})).await.unwrap(), json!("plain words"));
        assert_eq!(
            api.call("getOrder", json!({"orderId": "3"})).await.unwrap(),
            json!("GET /orders/{orderId} returned 4 bytes of image/png, which is not text.")
        );
        assert_eq!(api.call("cancel_order", json!({"orderId": "1"})).await.unwrap(), json!("DELETE /orders/{orderId} succeeded (204, no content)."));
        let err = api.call("getOrder", json!({"orderId": "404"})).await.unwrap_err().to_string();
        assert!(err.contains(r#"returned 404: {"error":"no such order"}"#), "{err}");
        let err = api.call("getOrder", json!({"orderId": "302"})).await.unwrap_err().to_string();
        assert!(err.contains("a redirect to https://evil.test, which is not"), "{err}");

        assert!(api.call("getOrder", json!({})).await.unwrap_err().to_string().contains("getOrder needs orderId"));
        assert!(
            api.call("getOrder", json!({"orderId": "1", "colour": "red"})).await.unwrap_err().to_string().contains("does not take colour; it takes orderId")
        );
        assert!(api.call("nope", json!({})).await.unwrap_err().to_string().contains("no operation 'nope'"));
    }

    #[tokio::test]
    async fn a_read_is_tried_again_and_a_write_is_not() {
        let calls = Arc::new(Mutex::new(0));
        let counter = calls.clone();
        let flaky: Answer = Arc::new(move |_| {
            let mut n = counter.lock().unwrap();
            *n += 1;
            if *n < 3 { (503, vec![], vec![]) } else { json_answer(json!({"ok": *n})) }
        });
        let (api, _server) = shop(flaky, OpenAPIOptions::default()).await;
        assert_eq!(api.call("getOrder", json!({"orderId": "1"})).await.unwrap(), json!({"ok": 3}));
        *calls.lock().unwrap() = 0;
        let err = api.call("createOrder", json!({"customer": "c", "items": []})).await.unwrap_err().to_string();
        assert!(err.contains("returned 503"), "{err}");
        assert_eq!(*calls.lock().unwrap(), 1);

        // Nothing listening there.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let dead_url = format!("http://{}/v1", listener.local_addr().unwrap());
        drop(listener);
        let dead = openapi_tools(shop_doc(), OpenAPIOptions { base_url: Some(dead_url), retries: 0, ..Default::default() }).unwrap();
        let err = dead.call("getOrder", json!({"orderId": "1"})).await.unwrap_err().to_string();
        assert!(err.contains("could not be reached: ConnectError"), "{err}");
    }

    #[tokio::test]
    async fn credentials_are_sent_and_never_offered_to_the_model() {
        let (api, server) = shop(ok_json(), OpenAPIOptions { token: Some("tok-1".into()), ..Default::default() }).await;
        api.call("getOrder", json!({"orderId": "1"})).await.unwrap();
        assert_eq!(server.last().header("authorization"), "Bearer tok-1");

        let (api, server) = shop(ok_json(), OpenAPIOptions { api_key: Some("k-9".into()), ..Default::default() }).await;
        api.call("getOrder", json!({"orderId": "1"})).await.unwrap();
        assert_eq!(server.last().header("x-api-key"), "k-9");

        let credentials = [("session".to_string(), "s-1".to_string()), ("login".to_string(), "ada:pw".to_string())].into_iter().collect();
        let (api, server) = shop(ok_json(), OpenAPIOptions { credentials, ..Default::default() }).await;
        api.call("listOrders", json!({})).await.unwrap();
        assert_eq!(server.last().param("session").as_deref(), Some("s-1"));
        assert_eq!(server.last().header("authorization"), "Basic YWRhOnB3");

        let options = OpenAPIOptions {
            basic: Some(("ada".into(), "pw".into())),
            headers: [("X-Request-Id".to_string(), Secret::from("fixed"))].into_iter().collect(),
            params: [("limit".to_string(), Secret::from("3"))].into_iter().collect(),
            ..Default::default()
        };
        let (api, server) = shop(ok_json(), options).await;
        api.call("listOrders", json!({})).await.unwrap();
        assert_eq!(server.last().header("authorization"), "Basic YWRhOnB3");
        assert_eq!(server.last().header("x-request-id"), "fixed");
        assert_eq!(server.last().param("limit").as_deref(), Some("3"));
        // What is fixed here is not the model's to set.
        let properties = &api.get("listOrders").unwrap().parameters["properties"];
        assert!(properties.get("limit").is_none() && properties.get("X-Request-Id").is_none());
        for tool in &api {
            assert!(!(tool.parameters.to_string() + &tool.description).contains("tok"));
        }

        let issued = Arc::new(Mutex::new(vec!["t2", "t1"]));
        let token = Secret::from_fn(move || {
            let next = issued.lock().unwrap().pop().unwrap_or("");
            async move { Ok(next.to_string()) }
        });
        let (api, server) = shop(ok_json(), OpenAPIOptions { token: Some(token), ..Default::default() }).await;
        api.call("getOrder", json!({"orderId": "1"})).await.unwrap();
        api.call("getOrder", json!({"orderId": "1"})).await.unwrap();
        let sent: Vec<String> = server.seen.lock().unwrap().iter().map(|r| r.header("authorization")).collect();
        assert_eq!(sent, ["Bearer t1", "Bearer t2"]);

        let credentials = [("nope".to_string(), "x".to_string())].into_iter().collect();
        let err = openapi_tools(shop_doc(), OpenAPIOptions { credentials, ..Default::default() }).unwrap_err().to_string();
        assert!(err.contains("no security scheme called 'nope'; the document declares: key, login, oauth, session"), "{err}");
        let err = openapi_tools(
            json!({"openapi": "3.1.0", "paths": {}, "servers": [{"url": "https://a.test"}]}),
            OpenAPIOptions { api_key: Some("k".into()), ..Default::default() },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("declares no apiKey security scheme"), "{err}");
    }

    #[test]
    fn operations_are_chosen_by_name_tag_and_method() {
        let names = |options: OpenAPIOptions| plain_shop(options).names();
        let globs = |g: &[&str]| g.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(names(OpenAPIOptions { include: Some(globs(&["get*", "POST /login"])), ..Default::default() }), ["getOrder", "login", "getTree"]);
        assert_eq!(
            names(OpenAPIOptions { exclude: globs(&["*Order*", "cancel*", "patch_*"]), ..Default::default() }),
            ["setTags", "login", "addNote", "getTree"]
        );
        assert_eq!(names(OpenAPIOptions { tags: Some(globs(&["auth"])), ..Default::default() }), ["login"]);
        assert_eq!(names(OpenAPIOptions { methods: Some(globs(&["GET"])), ..Default::default() }), ["listOrders", "getOrder", "getTree"]);
        assert!(names(OpenAPIOptions { deprecated: true, ..Default::default() }).contains(&"oldThing".to_string()));
        assert_eq!(
            names(OpenAPIOptions { prefix: "shop_".into(), methods: Some(globs(&["get"])), ..Default::default() }),
            ["shop_listOrders", "shop_getOrder", "shop_getTree"]
        );

        let careful = plain_shop(OpenAPIOptions { writes: Permission::Ask, ..Default::default() });
        assert_eq!(careful.get("createOrder").unwrap().permission, Permission::Ask);
        assert_eq!(careful.get("getOrder").unwrap().permission, Permission::Allow);
        let mut bad = Map::new();
        bad.insert("writes".into(), json!("maybe"));
        assert!(OpenAPIOptions::from_map(&bad).unwrap_err().to_string().contains("writes must be"));
        let described = careful.describe();
        assert!(described.contains("createOrder(customer, items, status?, gift?) [ask]"), "{described}");
        assert!(described.contains("(skipped POST /upload: it needs a file upload (file))"));
    }

    // The blocking fetch waits on its own thread: the local server needs another to answer on.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_document_is_a_file_text_a_url_or_a_value() {
        let dir = std::env::temp_dir().join(format!("ah-openapi-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let as_json = dir.join("shop.json");
        std::fs::write(&as_json, serde_json::to_string(&shop_doc()).unwrap()).unwrap();
        let yaml = serde_yaml::to_string(&shop_doc()).unwrap();
        let as_yaml = dir.join("shop.yaml");
        std::fs::write(&as_yaml, &yaml).unwrap();
        for source in [
            OpenAPISpec::from(&as_json),
            OpenAPISpec::from(as_yaml.to_string_lossy().as_ref()),
            serde_json::to_string(&shop_doc()).unwrap().into(),
            yaml.clone().into(),
        ] {
            assert_eq!(openapi_tools(source, OpenAPIOptions::default()).unwrap().len(), 9);
        }
        // Unquoted numeric keys, as people write them.
        let handwritten = "openapi: 3.0.0\nservers: [{url: 'https://a.test'}]\npaths:\n  /x:\n    get:\n      operationId: getX\n      responses:\n        200: {description: ok}\n";
        assert_eq!(openapi_tools(handwritten, OpenAPIOptions::default()).unwrap().names(), ["getX"]);

        let mut doc = shop_doc();
        doc["servers"] = json!([{"url": "/api"}]);
        let body = serde_json::to_vec(&doc).unwrap();
        let host: Answer = Arc::new(move |r: &Seen| {
            if r.path == "/docs/openapi.json" { (200, vec![("content-type", "application/json".into())], body.clone()) } else { (404, vec![], vec![]) }
        });
        let server = serve(host).await;
        let fetched = openapi_tools(format!("{}/docs/openapi.json", server.url), OpenAPIOptions::default()).unwrap();
        assert_eq!(fetched.base_url(), format!("{}/api", server.url)); // relative to the document
        let loaded = OpenAPIToolkit::load(format!("{}/docs/openapi.json", server.url), OpenAPIOptions::default()).await.unwrap();
        assert_eq!(loaded.len(), 9);

        let cases: Vec<(OpenAPISpec, &str)> = vec![
            (dir.join("missing.json").into(), "no OpenAPI document at"),
            ("{not json".into(), "neither JSON nor YAML"),
            (json!({"info": {}}).into(), "this is not an OpenAPI document"),
            (json!({"openapi": "3.0.0", "paths": {}}).into(), "does not say where the API is"),
            (json!(42).into(), "is a file, a URL, its text, or a mapping"),
            (format!("{}/nope", server.url).into(), "returned HTTP 404"),
        ];
        for (bad, message) in cases {
            let err = openapi_tools(bad, OpenAPIOptions::default()).unwrap_err().to_string();
            assert!(err.contains(message), "{err} / {message}");
        }
        let broken = json!({"openapi": "3.0.0", "servers": [{"url": "https://a.test"}], "paths": {
            "/x": {"get": {"parameters": [{"$ref": "#/components/parameters/Gone"}]}}}});
        assert!(openapi_tools(broken, OpenAPIOptions::default()).unwrap_err().to_string().contains("which is not in it"));
        let based =
            openapi_tools(json!({"openapi": "3.0.0", "paths": {}}), OpenAPIOptions { base_url: Some("https://a.test/".into()), ..Default::default() }).unwrap();
        assert_eq!(based.base_url(), "https://a.test");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn swagger_2_is_read_too() {
        let document = json!({
            "swagger": "2.0", "info": {"title": "Pets"}, "host": "pets.test",
            "basePath": "/v2", "schemes": ["https"], "consumes": ["application/json"],
            "securityDefinitions": {"key": {"type": "apiKey", "in": "query", "name": "api_key"}, "login": {"type": "basic"}},
            "paths": {
                "/pets": {
                    "get": {"operationId": "findPets", "parameters": [
                        {"name": "tags", "in": "query", "type": "array", "items": {"type": "string"}, "collectionFormat": "csv"},
                        {"name": "api_key", "in": "query", "type": "string"}]},
                    "post": {"operationId": "addPet", "parameters": [
                        {"name": "pet", "in": "body", "required": true, "schema": {"$ref": "#/definitions/Pet"}}]}
                },
                "/pets/{id}": {"post": {"operationId": "updatePet", "consumes": ["application/x-www-form-urlencoded"], "parameters": [
                    {"name": "id", "in": "path", "required": true, "type": "integer"},
                    {"name": "name", "in": "formData", "type": "string", "required": true},
                    {"name": "photo", "in": "formData", "type": "file"}]}}
            },
            "definitions": {"Pet": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}}
        });
        let described = openapi_tools(document.clone(), OpenAPIOptions { api_key: Some("k-1".into()), ..Default::default() }).unwrap();
        assert_eq!(described.base_url(), "https://pets.test/v2");

        let server = serve(ok_json()).await;
        let api = openapi_tools(document, OpenAPIOptions { api_key: Some("k-1".into()), base_url: Some(format!("{}/v2", server.url)), ..Default::default() })
            .unwrap();
        assert_eq!(api.get("findPets").unwrap().parameters["properties"], json!({"tags": {"type": "array", "items": {"type": "string"}}}));
        assert_eq!(api.get("addPet").unwrap().parameters["required"], json!(["name"]));
        let update = api.get("updatePet").unwrap().parameters["properties"].as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(update, ["id", "name"]);

        api.call("findPets", json!({"tags": ["cat", "small"]})).await.unwrap();
        assert_eq!(server.last().query, vec![("tags".to_string(), "cat,small".to_string()), ("api_key".to_string(), "k-1".to_string())]);
        api.call("addPet", json!({"name": "Tom", "age": 3})).await.unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&server.last().body).unwrap(), json!({"name": "Tom", "age": 3}));
        api.call("updatePet", json!({"id": 7, "name": "Tim"})).await.unwrap();
        assert_eq!(server.last().path, "/v2/pets/7");
        assert_eq!(server.last().body, b"name=Tim");
    }

    #[tokio::test]
    async fn the_tools_call_the_api() {
        let answer: Answer = Arc::new(|r: &Seen| match (r.method.as_str(), r.path.as_str()) {
            ("GET", "/v1/orders/4182") => json_answer(json!({"id": "4182", "status": "shipped"})),
            ("POST", "/v1/orders") => (201, vec![("content-type", "application/json".into())], br#"{"id":"4183"}"#.to_vec()),
            _ => (404, vec![], vec![]),
        });
        let (api, server) = shop(answer, OpenAPIOptions { token: Some("tok-1".into()), ..Default::default() }).await;
        let tools = api.into_tools();
        let get = tools.iter().find(|t| t.name == "getOrder").unwrap();
        let create = tools.iter().find(|t| t.name == "createOrder").unwrap();
        let args = |v: Value| v.as_object().cloned().unwrap();
        let found = get.run("c1", args(json!({"orderId": "4182"})), ToolContext::default()).await;
        assert_eq!(serde_json::from_str::<Value>(&found.content).unwrap(), json!({"id": "4182", "status": "shipped"}));
        let made = create.run("c2", args(json!({"customer": "c_1", "items": [{"sku": "tea", "quantity": 1}]})), ToolContext::default()).await;
        assert_eq!(serde_json::from_str::<Value>(&made.content).unwrap(), json!({"id": "4183"}));
        let missing = get.run("c3", Map::new(), ToolContext::default()).await;
        assert!(missing.is_error);
        assert_eq!(missing.content, "Error: getOrder needs orderId"); // the model is told
        assert_eq!(server.count(), 2);
        assert_eq!(server.last().header("authorization"), "Bearer tok-1");
    }

    #[tokio::test]
    async fn a_file_can_declare_an_api() {
        let dir = std::env::temp_dir().join(format!("ah-openapi-cfg-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("shop.json"), serde_json::to_string(&shop_doc()).unwrap()).unwrap();
        let variable = format!("AH_TEST_SHOP_TOKEN_{}", uuid::Uuid::new_v4().simple());
        // SAFETY: the variable name is unique to this test.
        unsafe { std::env::set_var(&variable, "tok-7") };
        let config = json!({"shop": {"spec": "shop.json", "token_env": variable, "include": ["getOrder", "listOrders"]}});
        let kits = toolkits_from_config(config.as_object().unwrap(), Some(&dir)).unwrap();
        assert_eq!(kits[0].name(), "shop");
        assert_eq!(kits[0].names(), ["listOrders", "getOrder"]);
        assert_eq!(kits[0].caller.headers["Authorization"].value().await.unwrap(), "Bearer tok-7");

        let short = json!({"shop": "shop.json"});
        assert_eq!(toolkits_from_config(short.as_object().unwrap(), Some(&dir)).unwrap()[0].len(), 9);

        // SAFETY: as above.
        unsafe { std::env::remove_var(&variable) };
        let err = toolkits_from_config(config.as_object().unwrap(), Some(&dir)).unwrap_err().to_string();
        assert!(err.contains(&format!("{variable} is not set")), "{err}");
        let err = toolkits_from_config(json!({"shop": {}}).as_object().unwrap(), None).unwrap_err().to_string();
        assert!(err.contains("openapi.shop needs a `spec`"), "{err}");
        let err = toolkits_from_config(json!({"shop": {"spec": shop_doc(), "colour": "red"}}).as_object().unwrap(), None).unwrap_err().to_string();
        assert!(err.contains("unexpected keyword argument 'colour'"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
