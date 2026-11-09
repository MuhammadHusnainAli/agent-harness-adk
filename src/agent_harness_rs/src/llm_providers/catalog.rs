//! The provider registry, and what it tells you before you connect.
//!
//! ```no_run
//! use agent_harness::llm_providers::catalog::{check_llm_provider, describe_llm_provider, get_provider, get_provider_with, list_llm_providers, resolve_provider};
//! use serde_json::json;
//!
//! for spec in list_llm_providers() {
//!     println!("{} {:?} {}", spec.name, spec.required_fields().iter().map(|f| f.name).collect::<Vec<_>>(), spec.configured);
//! }
//! println!("{}", describe_llm_provider("azure").unwrap().render());          // every field, and where it's read
//! println!("{}", check_llm_provider("bedrock", &json!({"region": "eu-west-1"})).unwrap().render()); // what's missing, offline
//!
//! let claude = get_provider("anthropic").unwrap();
//! let by_model = resolve_provider(None, Some("gpt-4.1")).unwrap(); // → openai
//! let bedrock = get_provider_with("bedrock", &json!({"region": "eu-west-1"})).unwrap();
//! ```
//!
//! Every backend declares its connection fields — which are required, which
//! are secret, which environment variables they fall back to — so a newcomer
//! can see exactly what a provider needs without reading its source.
//!
//! Settings (Python's keyword arguments) are a JSON object. Settings that are
//! Rust objects in the constructors take a JSON form here: Bedrock
//! `credentials` is `{access_key, secret_key, session_token}`, Vertex
//! `credentials` a key file's JSON or its path, Azure `credential` is
//! `"default"` (the environment / managed identity / Azure CLI chain) or
//! `{"token": ...}`. `retry`, `on_retry` and `client` can only be given in code,
//! through a [`TransportConfig`].

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use indexmap::IndexMap;
use reqwest::header::{HeaderName, HeaderValue};
use serde::Serialize;
use serde_json::{Map, Value, json};

use super::anthropic::AnthropicProvider;
use super::azure::{self, AzureCredential, AzureOpenAIProvider, AzureOptions};
use super::base::{COMMON_FIELDS, MODELS, Provider, ProviderField, ProviderSpec, TransportConfig, provider_for_model};
use super::bedrock::{self, BedrockOptions, BedrockProvider};
use super::compatible::{COMPATIBLE_PROVIDERS, openrouter, preset_provider};
use super::fake::FakeProvider;
use super::gemini::GeminiProvider;
use super::openai::{OPENAI, OpenAIProvider};
use super::parameters::SAMPLING_PARAMETERS;
use super::sigv4::AwsCredentials;
use super::vertex::{self, GoogleCredentials, VertexGeminiProvider, VertexOptions, VertexProvider};
use crate::errors::{Error, Result};

/// Builds a provider with default settings, reading keys from the environment.
pub type Factory = Arc<dyn Fn() -> Result<Arc<dyn Provider>> + Send + Sync>;

/// Builds a provider from settings (a JSON object; empty for the defaults).
pub type Builder = Arc<dyn Fn(&Map<String, Value>) -> Result<Arc<dyn Provider>> + Send + Sync>;

#[derive(Clone)]
struct Entry {
    /// The name this backend is listed under; an alias points at it.
    canonical: String,
    /// `None` for a provider registered without one: built to find out.
    spec: Option<ProviderSpec>,
    /// A key is optional (local servers).
    keyless: bool,
    build: Builder,
}

fn arc<P: Provider + 'static>(p: P) -> Arc<dyn Provider> {
    Arc::new(p)
}

fn builder(f: impl Fn(&Map<String, Value>) -> Result<Arc<dyn Provider>> + Send + Sync + 'static) -> Builder {
    Arc::new(f)
}

// ---------------------------------------------------------------------------
// settings
// ---------------------------------------------------------------------------

fn text(settings: &Map<String, Value>, key: &str) -> Option<String> {
    match settings.get(key)? {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn number(settings: &Map<String, Value>, key: &str) -> Result<Option<f64>> {
    match settings.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).map(Some).ok_or_else(|| Error::config(format!("{key} must be a number"))),
    }
}

/// The common settings, as a [`TransportConfig`].
fn transport_config(settings: &Map<String, Value>) -> Result<TransportConfig> {
    let mut config = TransportConfig::default();
    for key in ["retry", "on_retry", "client"] {
        if settings.get(key).is_some_and(|v| !v.is_null()) {
            return Err(Error::config(format!("'{key}' cannot be given as a setting — build the provider in code with a TransportConfig")));
        }
    }
    if let Some(t) = number(settings, "timeout")? {
        config.timeout = t;
    }
    if let Some(t) = number(settings, "connect_timeout")? {
        config.connect_timeout = t;
    }
    if let Some(n) = number(settings, "max_retries")? {
        config.retry.max_retries = n as u32;
    }
    if let Some(n) = number(settings, "max_concurrency")? {
        config.max_concurrency = Some(n as usize);
    }
    match settings.get("circuit_breaker") {
        None | Some(Value::Bool(true)) => {}
        Some(Value::Null) | Some(Value::Bool(false)) => config.circuit_breaker = None,
        Some(_) => return Err(Error::config("circuit_breaker takes true or false here — pass a CircuitBreaker in code")),
    }
    if let Some(Value::Object(headers)) = settings.get("headers") {
        for (k, v) in headers {
            let v = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
            let (Ok(k), Ok(v)) = (HeaderName::from_bytes(k.as_bytes()), HeaderValue::from_str(&v)) else {
                return Err(Error::config(format!("header '{k}' is not a valid HTTP header")));
            };
            config.headers.insert(k, v);
        }
    }
    if let Some(url) = text(settings, "base_url") {
        config.base_url = url;
    }
    Ok(config)
}

fn bedrock_options(s: &Map<String, Value>) -> Result<BedrockOptions> {
    let credentials = match s.get("credentials") {
        None | Some(Value::Null) => None,
        Some(Value::Object(c)) => {
            Some(AwsCredentials::new(text(c, "access_key").unwrap_or_default(), text(c, "secret_key").unwrap_or_default(), text(c, "session_token")))
        }
        Some(_) => return Err(Error::config("bedrock: credentials is {access_key, secret_key, session_token}")),
    };
    Ok(BedrockOptions {
        region: text(s, "region"),
        access_key: text(s, "access_key"),
        secret_key: text(s, "secret_key"),
        session_token: text(s, "session_token"),
        profile: text(s, "profile"),
        credentials,
        api_key: text(s, "api_key"),
        base_url: text(s, "base_url"),
    })
}

fn vertex_options(s: &Map<String, Value>) -> Result<VertexOptions> {
    let credentials = match s.get("credentials") {
        None | Some(Value::Null) => None,
        Some(Value::String(path)) => Some(GoogleCredentials::KeyFile(path.into())),
        Some(v @ Value::Object(_)) => Some(GoogleCredentials::from_value(v)?),
        Some(_) => return Err(Error::config("vertex: credentials is a key file's JSON, or its path")),
    };
    Ok(VertexOptions { project: text(s, "project"), region: text(s, "region"), access_token: text(s, "access_token"), credentials })
}

fn azure_options(s: &Map<String, Value>) -> Result<AzureOptions> {
    let credential = match s.get("credential") {
        None | Some(Value::Null) => None,
        Some(Value::String(d)) if d == "default" => Some(AzureCredential::Default),
        Some(Value::Object(o)) if o.get("token").and_then(Value::as_str).is_some() => Some(AzureCredential::Token(text(o, "token").unwrap_or_default())),
        Some(_) => return Err(Error::config("azure: credential is \"default\" or {\"token\": ...} — or pass an AzureCredential in code")),
    };
    Ok(AzureOptions {
        endpoint: text(s, "endpoint"),
        deployment: text(s, "deployment"),
        api_version: text(s, "api_version"),
        api_key: text(s, "api_key"),
        credential,
        path_template: text(s, "path_template"),
    })
}

// ---------------------------------------------------------------------------
// the registry
// ---------------------------------------------------------------------------

/// Every backend this crate can build, by name (lower case). Aliases point at
/// the same entry: `claude` → anthropic, `google` → gemini, `aws` → bedrock,
/// `gcp` → vertex, `azure-openai` → azure, `foundry` → azure-foundry, `grok` → xai.
static PROVIDERS: LazyLock<RwLock<BTreeMap<String, Entry>>> = LazyLock::new(|| {
    let mut m: BTreeMap<String, Entry> = BTreeMap::new();
    let mut add = |name: &str, spec: ProviderSpec, keyless: bool, build: Builder| {
        m.insert(name.into(), Entry { canonical: name.into(), spec: Some(spec), keyless, build });
    };
    add("anthropic", super::anthropic::SPEC, false, builder(|s| Ok(arc(AnthropicProvider::with_config(text(s, "api_key"), transport_config(s)?)))));
    add(
        "openai",
        OPENAI.spec,
        false,
        builder(|s| {
            let mut config = transport_config(s)?;
            for (field, env, header) in [("organization", "OPENAI_ORG_ID", "openai-organization"), ("project", "OPENAI_PROJECT_ID", "openai-project")] {
                if let Some(v) = text(s, field).or_else(|| std::env::var(env).ok().filter(|v| !v.is_empty())).and_then(|v| HeaderValue::from_str(&v).ok()) {
                    config.headers.insert(header, v);
                }
            }
            Ok(arc(OpenAIProvider::with_flavor(OPENAI, text(s, "api_key"), config)?))
        }),
    );
    add("gemini", super::gemini::SPEC, false, builder(|s| Ok(arc(GeminiProvider::with_config(text(s, "api_key"), transport_config(s)?)))));
    add("fake", super::fake::SPEC, true, builder(|_| Ok(arc(FakeProvider::new(vec![])))));
    add("bedrock", bedrock::SPEC, false, builder(|s| Ok(arc(BedrockProvider::with_config(bedrock_options(s)?, transport_config(s)?)))));
    add("vertex", vertex::SPEC, false, builder(|s| Ok(arc(VertexProvider::with_config(vertex_options(s)?, transport_config(s)?)?))));
    add("vertex-gemini", vertex::GEMINI_SPEC, false, builder(|s| Ok(arc(VertexGeminiProvider::with_config(vertex_options(s)?, transport_config(s)?)?))));
    add("azure", azure::SPEC, false, builder(|s| Ok(arc(AzureOpenAIProvider::with_config(azure_options(s)?, transport_config(s)?)?))));
    add("azure-foundry", azure::FOUNDRY_SPEC, false, builder(|s| Ok(arc(AzureOpenAIProvider::foundry_with_config(azure_options(s)?, transport_config(s)?)?))));
    for flavor in COMPATIBLE_PROVIDERS {
        let name = flavor.spec.name;
        let build = if name == "openrouter" {
            builder(|s| Ok(arc(openrouter(text(s, "api_key"), text(s, "app_url").as_deref(), text(s, "app_name").as_deref(), transport_config(s)?)?)))
        } else {
            builder(move |s| Ok(arc(preset_provider(name, text(s, "api_key"), transport_config(s)?)?)))
        };
        add(name, flavor.spec, !flavor.key_required, build);
    }
    for (alias, target) in [
        ("claude", "anthropic"),
        ("google", "gemini"),
        ("aws", "bedrock"),
        ("gcp", "vertex"),
        ("azure-openai", "azure"),
        ("foundry", "azure-foundry"),
        ("grok", "xai"),
    ] {
        let entry = m[target].clone();
        m.insert(alias.into(), entry);
    }
    RwLock::new(m)
});

static CACHE: LazyLock<Mutex<HashMap<String, Arc<dyn Provider>>>> = LazyLock::new(Default::default);

/// Add your own backend: `register_provider("mine", Arc::new(|| Ok(Arc::new(Mine::new()))))`.
///
/// It is listed with the spec its provider reports (it is built to find out);
/// use [`register_provider_with`] to declare one, and to take settings.
pub fn register_provider(name: &str, build: Factory) {
    let key = name.trim().to_lowercase();
    let label = key.clone();
    let build: Builder = Arc::new(move |settings: &Map<String, Value>| {
        if !settings.is_empty() {
            let keys: Vec<&str> = settings.keys().map(String::as_str).collect();
            return Err(Error::config(format!("{label}: takes no settings (got {})", keys.join(", "))));
        }
        build()
    });
    insert(key, None, build);
}

/// Add your own backend with its spec — fields, description, capabilities —
/// so it shows up in [`list_llm_providers`] like any built-in, and a builder
/// that reads settings.
pub fn register_provider_with(name: &str, spec: ProviderSpec, build: Builder) {
    insert(name.trim().to_lowercase(), Some(spec), build);
}

fn insert(key: String, spec: Option<ProviderSpec>, build: Builder) {
    PROVIDERS.write().unwrap_or_else(|p| p.into_inner()).insert(key.clone(), Entry { canonical: key.clone(), spec, keyless: false, build });
    CACHE.lock().unwrap_or_else(|p| p.into_inner()).remove(&key);
}

/// Every registered name, aliases included, sorted.
pub fn provider_names() -> Vec<String> {
    PROVIDERS.read().unwrap_or_else(|p| p.into_inner()).keys().cloned().collect()
}

fn lookup(name: &str) -> Result<(String, Entry)> {
    let key = name.trim().to_lowercase();
    let providers = PROVIDERS.read().unwrap_or_else(|p| p.into_inner());
    match providers.get(&key) {
        Some(e) => Ok((key, e.clone())),
        None => {
            let known: Vec<&str> = providers.keys().map(String::as_str).collect();
            Err(Error::config(format!("Unknown provider '{name}'. Known: {}. See list_llm_providers().", known.join(", "))))
        }
    }
}

fn object(settings: &Value) -> Result<Map<String, Value>> {
    match settings {
        Value::Null => Ok(Map::new()),
        Value::Object(m) => Ok(m.clone()),
        _ => Err(Error::config("provider settings are a JSON object")),
    }
}

fn accepted(spec: &ProviderSpec) -> Vec<&'static str> {
    spec.fields.iter().chain(COMMON_FIELDS.iter()).map(|f| f.name).collect()
}

fn build(key: &str, entry: &Entry, settings: &Map<String, Value>) -> Result<Arc<dyn Provider>> {
    if let Some(spec) = &entry.spec {
        let ok = accepted(spec);
        let unknown: Vec<&str> = settings.keys().map(String::as_str).filter(|k| !ok.contains(k)).collect();
        if !unknown.is_empty() {
            return Err(Error::config(format!(
                "{key}: unknown setting(s) {}. It accepts: {}. See describe_llm_provider('{key}').",
                unknown.join(", "),
                ok.join(", ")
            )));
        }
    }
    (entry.build)(settings)
}

/// Build (or reuse) a provider by vendor name. Cached: one per name.
pub fn get_provider(name: &str) -> Result<Arc<dyn Provider>> {
    let (key, entry) = lookup(name)?;
    if let Some(p) = CACHE.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Ok(p.clone());
    }
    let built = build(&key, &entry, &Map::new())?;
    CACHE.lock().unwrap_or_else(|p| p.into_inner()).entry(key).or_insert(built.clone());
    Ok(built)
}

/// A fresh, uncached provider built from settings — Python's
/// `get_provider(name, **kwargs)`. A setting the provider does not have is
/// an error that lists the ones it does.
pub fn get_provider_with(name: &str, settings: &Value) -> Result<Arc<dyn Provider>> {
    let (key, entry) = lookup(name)?;
    build(&key, &entry, &object(settings)?)
}

/// A fresh, uncached provider by vendor name.
pub fn new_provider(name: &str) -> Result<Arc<dyn Provider>> {
    get_provider_with(name, &Value::Null)
}

/// Accept a vendor name, or nothing at all and work it out from the model.
pub fn resolve_provider(provider: Option<&str>, model: Option<&str>) -> Result<Arc<dyn Provider>> {
    match (provider, model) {
        (Some(name), _) => get_provider(name),
        (None, Some(m)) if !m.is_empty() => get_provider(&provider_for_model(m)),
        _ => get_provider("anthropic"),
    }
}

/// Forget every cached provider (their connections close when dropped).
pub fn close_all() {
    CACHE.lock().unwrap_or_else(|p| p.into_inner()).clear();
}

// ---------------------------------------------------------------------------
// describing what a provider needs
// ---------------------------------------------------------------------------

/// Whether a provider has what it needs to connect — decided offline.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderCheck {
    pub provider: String,
    pub ok: bool,
    /// What is still needed, each with where it could come from.
    pub missing: Vec<String>,
    /// field → where it was found: "argument" or "$ENV_VAR". Values are never shown.
    pub found: IndexMap<String, String>,
    pub unknown: Vec<String>,
    pub notes: Vec<String>,
}

impl ProviderCheck {
    pub fn render(&self) -> String {
        let mut lines = vec![format!("{}: {}", self.provider, if self.ok { "ready" } else { "not ready" })];
        lines.extend(self.missing.iter().map(|m| format!("  missing  {m}")));
        lines.extend(self.found.iter().map(|(k, v)| format!("  found    {k} ← {v}")));
        lines.extend(self.unknown.iter().map(|u| format!("  unknown  {u}")));
        lines.extend(self.notes.iter().map(|n| format!("  note     {n}")));
        lines.join("\n")
    }
}

/// Everything there is to know about a backend before connecting to it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderInfo {
    pub name: String,
    pub aliases: Vec<String>,
    pub display_name: String,
    pub description: String,
    pub docs_url: String,
    pub auth_type: String,
    pub default_model: String,
    pub base_url: String,
    pub capabilities: Vec<String>,
    /// The sampling parameters it accepts; the rest are dropped with a note.
    /// `effort`, `thinking`, `thinking_budget`, `max_tokens` and `stop` are
    /// understood by every provider and mapped to what each model takes.
    pub parameters: Vec<String>,
    pub modalities: Vec<String>,
    pub fields: Vec<ProviderField>,
    pub common_fields: Vec<ProviderField>,
    /// Models in the price table served by this provider.
    pub models: Vec<String>,
    /// True when everything required is already in the environment.
    pub configured: bool,
    pub missing: Vec<String>,
}

impl ProviderInfo {
    pub fn required_fields(&self) -> Vec<&ProviderField> {
        self.fields.iter().filter(|f| f.required || f.one_of.is_some()).collect()
    }

    pub fn optional_fields(&self) -> Vec<&ProviderField> {
        self.fields.iter().filter(|f| !(f.required || f.one_of.is_some())).collect()
    }

    /// A line of code that builds this provider, with placeholders. Keys come
    /// from the environment, not code, so secrets are left out.
    pub fn example(&self) -> String {
        let args: Vec<String> = self
            .fields
            .iter()
            .filter(|f| !(f.secret() || f.r#type == "object" || f.one_of.is_some()) && (f.required || f.example.is_some()))
            .map(|f| format!("\"{}\": \"{}\"", f.name, f.example.unwrap_or("...")))
            .collect();
        if args.is_empty() {
            format!("get_provider(\"{}\")", self.name)
        } else {
            format!("get_provider_with(\"{}\", &json!({{{}}}))", self.name, args.join(", "))
        }
    }

    pub fn render(&self) -> String {
        let aliases = if self.aliases.is_empty() { String::new() } else { format!(" (aliases: {})", self.aliases.join(", ")) };
        let mut lines = vec![format!("{} — provider=\"{}\"{aliases}", self.display_name, self.name)];
        if !self.description.is_empty() {
            lines.push(format!("  {}", self.description));
        }
        let missing = if self.missing.is_empty() { String::new() } else { format!(" — missing {}", self.missing.join(", ")) };
        lines.push(format!("  status   {}{missing}", if self.configured { "ready" } else { "needs setup" }));
        lines.push(format!("  auth     {}", self.auth_type));
        if !self.default_model.is_empty() {
            lines.push(format!("  default  {}", self.default_model));
        }
        if !self.capabilities.is_empty() {
            lines.push(format!("  can      {}", self.capabilities.join(", ")));
        }
        if !self.parameters.is_empty() {
            lines.push(format!("  params   {}, plus effort, thinking, max_tokens, stop", self.parameters.join(", ")));
        }
        if !self.fields.is_empty() {
            lines.push("  fields".into());
            for f in &self.fields {
                let need = if f.required {
                    "required".to_string()
                } else if let Some(g) = f.one_of {
                    format!("one of [{g}]")
                } else {
                    "optional".to_string()
                };
                let env = if f.env.is_empty() { String::new() } else { format!(" ← ${}", f.env.join(" / $")) };
                lines.push(format!("    {:<16} {need:<18} {:<7}{env}", f.name, f.r#type).trim_end().to_string());
                if !f.description.is_empty() {
                    lines.push(format!("    {:<16} {}", "", f.description));
                }
            }
        }
        lines.push(format!("  also     {}", self.common_fields.iter().map(|f| f.name).collect::<Vec<_>>().join(", ")));
        lines.push(format!("  example  {}", self.example()));
        if !self.docs_url.is_empty() {
            lines.push(format!("  docs     {}", self.docs_url));
        }
        lines.join("\n")
    }

    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or_default()
    }
}

fn is_set(v: Option<&Value>) -> bool {
    !matches!(v, None | Some(Value::Null) | Some(Value::Bool(false))) && v != Some(&json!(""))
}

fn env_set(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.is_empty())
}

/// Is `program` on `PATH`?
fn on_path(program: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else { return false };
    let names: Vec<String> = if cfg!(windows) { ["", ".exe", ".cmd", ".bat"].iter().map(|e| format!("{program}{e}")).collect() } else { vec![program.into()] };
    std::env::split_paths(&path).any(|dir| names.iter().any(|n| dir.join(n).is_file()))
}

/// A credential source that needs no setting: a profile, a key file, a CLI login.
fn ambient_auth(spec: &ProviderSpec) -> Option<String> {
    if spec.name == bedrock::SPEC.name {
        let profile = std::env::var("AWS_PROFILE").ok().filter(|p| !p.is_empty()).unwrap_or_else(|| "default".into());
        return bedrock::profile_credentials(&profile)
            .map(|_| format!("no explicit AWS credentials — the '{profile}' profile in ~/.aws/credentials will be used"));
    }
    if spec.auth_type == "google-oauth" {
        let chain = GoogleCredentials::default_chain();
        if matches!(chain.first(), Some(GoogleCredentials::KeyFile(_))) {
            return Some("no access_token — application default credentials will be used".into());
        }
        if env_set("GCE_METADATA_HOST") {
            return Some("no access_token — the metadata server will be used".into());
        }
        if on_path("gcloud") {
            return Some("no access_token — `gcloud auth print-access-token` will be used".into());
        }
    }
    None
}

fn check(key: &str, spec: &ProviderSpec, settings: &Map<String, Value>) -> ProviderCheck {
    let mut found: IndexMap<String, String> = IndexMap::new();
    for f in spec.fields {
        if is_set(settings.get(f.name)) {
            found.insert(f.name.into(), "argument".into());
        } else if let Some(env) = f.env.iter().find(|e| env_set(e)) {
            found.insert(f.name.into(), format!("${env}"));
        }
    }
    let mut missing = Vec::new();
    let mut notes = Vec::new();
    for f in spec.fields.iter().filter(|f| f.required && !found.contains_key(f.name)) {
        let where_ = if f.env.is_empty() { String::new() } else { format!(" (or set ${})", f.env.join(" / $")) };
        missing.push(format!("{}{where_}", f.name));
    }
    let mut groups: IndexMap<&str, Vec<&ProviderField>> = IndexMap::new();
    for f in spec.fields {
        if let Some(g) = f.one_of {
            groups.entry(g).or_default().push(f);
        }
    }
    for members in groups.values() {
        if members.iter().any(|m| found.contains_key(m.name)) {
            continue;
        }
        if let Some(note) = ambient_auth(spec) {
            notes.push(note);
            continue;
        }
        let options: Vec<String> = members.iter().map(|m| format!("{}{}", m.name, m.env.first().map(|e| format!(" (${e})")).unwrap_or_default())).collect();
        missing.push(format!("one of: {}", options.join(" | ")));
    }
    // A half-given AWS key pair is a mistake worth naming.
    if found.contains_key("access_key") && !found.contains_key("secret_key") {
        missing.push("secret_key (or set $AWS_SECRET_ACCESS_KEY)".into());
    }
    let ok_names = accepted(spec);
    let mut unknown: Vec<String> = settings.keys().filter(|k| !ok_names.contains(&k.as_str())).cloned().collect();
    unknown.sort();
    if !unknown.is_empty() {
        notes.push(format!("not a {key} setting: {} — see describe_llm_provider('{key}')", unknown.join(", ")));
    }
    ProviderCheck { provider: key.into(), ok: missing.is_empty() && unknown.is_empty(), missing, found, unknown, notes }
}

/// A registered provider's spec: declared, or read off a built instance.
fn spec_of(entry: &Entry) -> Option<ProviderSpec> {
    entry.spec.clone().or_else(|| (entry.build)(&Map::new()).ok().map(|p| p.spec().clone()))
}

fn info(key: &str, entry: &Entry) -> ProviderInfo {
    let aliases: Vec<String> =
        PROVIDERS.read().unwrap_or_else(|p| p.into_inner()).iter().filter(|(k, e)| e.canonical == key && k.as_str() != key).map(|(k, _)| k.clone()).collect();
    let Some(spec) = spec_of(entry) else {
        return ProviderInfo {
            name: key.into(),
            aliases,
            display_name: key.into(),
            description: String::new(),
            docs_url: String::new(),
            auth_type: String::new(),
            default_model: String::new(),
            base_url: String::new(),
            capabilities: vec![],
            parameters: vec![],
            modalities: vec![],
            fields: vec![],
            common_fields: COMMON_FIELDS.to_vec(),
            models: vec![],
            configured: false,
            missing: vec!["(could not be built to describe it)".into()],
        };
    };
    let checked = check(key, &spec, &Map::new());
    let mut capabilities: Vec<String> = spec.capabilities.iter().map(|c| c.to_string()).collect();
    capabilities.sort();
    let mut models: Vec<String> = MODELS.read().unwrap_or_else(|p| p.into_inner()).values().filter(|m| m.provider == spec.name).map(|m| m.id.clone()).collect();
    models.sort();
    ProviderInfo {
        name: key.into(),
        aliases,
        display_name: if spec.display_name.is_empty() { key.into() } else { spec.display_name.into() },
        description: spec.description.into(),
        docs_url: spec.docs_url.into(),
        auth_type: if entry.keyless { "none (a key is optional)".into() } else { spec.auth_type.into() },
        default_model: spec.default_model.into(),
        base_url: spec.base_url.into(),
        capabilities,
        parameters: SAMPLING_PARAMETERS.iter().filter(|p| spec.parameters.contains(p)).map(|p| p.to_string()).collect(),
        modalities: spec.modalities.iter().map(|m| m.as_str().to_string()).collect(),
        fields: spec.fields.to_vec(),
        common_fields: COMMON_FIELDS.to_vec(),
        models,
        configured: checked.ok,
        missing: checked.missing,
    }
}

/// Filters for [`list_llm_providers_with`].
#[derive(Debug, Clone, Default)]
pub struct ListOptions {
    /// Just the ones whose required settings are already present.
    pub configured_only: bool,
    /// Just the ones that can, e.g. `"embeddings"` or `"streaming"`.
    pub capability: Option<String>,
    /// Include the scripted `fake` provider.
    pub include_testing: bool,
}

/// Every provider this harness can talk to, with the fields each one needs —
/// one entry per backend, aliases folded in, the `fake` provider left out.
pub fn list_llm_providers() -> Vec<ProviderInfo> {
    list_llm_providers_with(&ListOptions::default())
}

pub fn list_llm_providers_with(options: &ListOptions) -> Vec<ProviderInfo> {
    let entries: Vec<(String, Entry)> =
        PROVIDERS.read().unwrap_or_else(|p| p.into_inner()).iter().filter(|(k, e)| e.canonical == **k).map(|(k, e)| (k.clone(), e.clone())).collect();
    entries
        .into_iter()
        .filter(|(k, _)| options.include_testing || k != "fake")
        .map(|(k, e)| info(&k, &e))
        .filter(|i| !options.configured_only || i.configured)
        .filter(|i| options.capability.as_ref().is_none_or(|c| i.capabilities.contains(c)))
        .collect()
}

/// One provider's full spec: fields, env vars, capabilities, an example.
/// An alias describes the provider it points at.
pub fn describe_llm_provider(name: &str) -> Result<ProviderInfo> {
    let (_, entry) = lookup(name)?;
    let canonical = entry.canonical.clone();
    let entry = lookup(&canonical).map(|(_, e)| e).unwrap_or(entry);
    Ok(info(&canonical, &entry))
}

/// Would `get_provider_with(name, settings)` have what it needs? Checked
/// offline. Settings and environment variables both count; secret values are
/// never echoed back, only where they were found.
pub fn check_llm_provider(name: &str, settings: &Value) -> Result<ProviderCheck> {
    let (key, entry) = lookup(name)?;
    let settings = object(settings)?;
    Ok(match spec_of(&entry) {
        Some(spec) => check(&key, &spec, &settings),
        None => ProviderCheck { provider: key, ok: true, missing: vec![], found: IndexMap::new(), unknown: vec![], notes: vec![] },
    })
}

/// Connect for real and report whether the credentials work. Never fails —
/// the answer is in the result.
pub async fn ping_llm_provider(name: &str, model: Option<&str>, settings: &Value) -> Value {
    let checked = match check_llm_provider(name, settings) {
        Ok(c) => c,
        Err(e) => return json!({"provider": name, "ok": false, "error": format!("{}: {e}", e.kind_name())}),
    };
    if !checked.ok {
        let reasons: Vec<String> = checked.missing.iter().chain(checked.unknown.iter()).cloned().collect();
        return json!({"provider": name, "ok": false, "error": format!("not configured: {}", reasons.join("; "))});
    }
    match get_provider_with(name, settings) {
        Ok(p) => p.ping(model).await,
        Err(e) => json!({"provider": name, "ok": false, "error": format!("{}: {e}", e.kind_name())}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_free(names: &[&str]) -> bool {
        names.iter().all(|n| !env_set(n))
    }

    #[test]
    fn names_aliases_and_models() {
        assert_eq!(get_provider("fake").unwrap().name(), "fake");
        assert_eq!(get_provider("Claude").unwrap().name(), "anthropic");
        assert_eq!(get_provider("grok").unwrap().name(), "xai");
        assert_eq!(resolve_provider(None, Some("gemini-2.5-pro")).unwrap().name(), "gemini");
        assert_eq!(resolve_provider(None, None).unwrap().name(), "anthropic");
        let err = get_provider("nope").err().unwrap().to_string();
        assert!(err.starts_with("Unknown provider 'nope'. Known: anthropic, aws, azure, azure-foundry, azure-openai, bedrock, cerebras"), "{err}");
        assert!(err.contains("list_llm_providers"));
        let a = get_provider("openai").unwrap();
        let b = get_provider("openai").unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn the_platforms_are_in_the_registry_and_built_by_name() {
        let names = provider_names();
        for name in ["bedrock", "aws", "vertex", "gcp", "vertex-gemini", "azure", "azure-openai", "azure-foundry", "foundry"] {
            assert!(names.contains(&name.to_string()), "{name}");
        }
        let p = get_provider_with("bedrock", &json!({"region": "us-west-2", "credentials": {"access_key": "AK", "secret_key": "SK"}})).unwrap();
        assert_eq!(p.name(), "bedrock");
        assert_eq!(p.transport().unwrap().base_url, "https://bedrock-runtime.us-west-2.amazonaws.com");
        let v = get_provider_with("gcp", &json!({"project": "p", "region": "global", "access_token": "t"})).unwrap();
        assert_eq!(v.transport().unwrap().base_url, "https://aiplatform.googleapis.com/v1/projects/p/locations/global");
        let a = get_provider_with("foundry", &json!({"endpoint": "https://x.services.ai.azure.com/", "credential": "default"})).unwrap();
        assert_eq!(a.name(), "azure-foundry");
        assert_eq!(a.transport().unwrap().base_url, "https://x.services.ai.azure.com");
        let g = get_provider_with("gemini", &json!({"api_key": "k", "timeout": 5, "max_retries": 1, "circuit_breaker": false})).unwrap();
        assert_eq!(g.transport().unwrap().retry.max_retries, 1);
        assert!(g.transport().unwrap().circuit.is_none());
    }

    #[test]
    fn every_registered_provider_is_listed_once_with_its_fields() {
        let specs: HashMap<String, ProviderInfo> = list_llm_providers().into_iter().map(|s| (s.name.clone(), s)).collect();
        for name in [
            "anthropic",
            "openai",
            "gemini",
            "bedrock",
            "vertex",
            "vertex-gemini",
            "azure",
            "azure-foundry",
            "openai-compatible",
            "openrouter",
            "groq",
            "together",
            "deepseek",
            "mistral",
            "xai",
            "fireworks",
            "cerebras",
            "ollama",
            "lmstudio",
            "vllm",
        ] {
            assert!(specs.contains_key(name), "{name}");
        }
        assert!(!specs.contains_key("fake") && !specs.contains_key("claude"));
        assert!(specs["anthropic"].aliases.contains(&"claude".to_string()));
        assert_eq!(specs["bedrock"].aliases, vec!["aws"]);
        for spec in specs.values() {
            assert!(!spec.display_name.is_empty() && !spec.description.is_empty() && !spec.docs_url.is_empty(), "{}", spec.name);
            assert!(!spec.capabilities.is_empty(), "{}", spec.name);
            assert!(!spec.common_fields.is_empty());
        }
        assert!(list_llm_providers_with(&ListOptions { include_testing: true, ..Default::default() }).iter().any(|s| s.name == "fake"));
        let embedders: Vec<String> =
            list_llm_providers_with(&ListOptions { capability: Some("embeddings".into()), ..Default::default() }).into_iter().map(|s| s.name).collect();
        assert!(embedders.contains(&"openai".into()) && embedders.contains(&"vertex-gemini".into()) && !embedders.contains(&"anthropic".into()));
    }

    #[test]
    fn required_fields_say_where_they_can_come_from() {
        let spec = describe_llm_provider("claude").unwrap();
        assert_eq!(spec.name, "anthropic");
        let key = spec.required_fields()[0];
        assert!(key.name == "api_key" && key.secret() && key.env == ["ANTHROPIC_API_KEY"]);
        if env_free(&["ANTHROPIC_API_KEY"]) {
            assert!(!spec.configured);
            assert_eq!(spec.missing, vec!["api_key (or set $ANTHROPIC_API_KEY)"]);
        }
        assert!(spec.models.contains(&"claude-opus-5".to_string()));
        assert_eq!(describe_llm_provider("AWS").unwrap().name, "bedrock");
        assert!(describe_llm_provider("vertex").unwrap().to_json()["fields"].as_array().unwrap().iter().any(|f| f["name"] == "project"));
    }

    #[test]
    fn checks_report_sources_and_either_or_auth() {
        let check = check_llm_provider("openai", &json!({"api_key": "sk-secret-value"})).unwrap();
        assert!(check.ok);
        assert_eq!(check.found["api_key"], "argument");
        assert!(!serde_json::to_string(&check).unwrap().contains("sk-secret-value"));
        assert!(!check.render().contains("sk-secret-value"));

        if env_free(&["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]) {
            let missing = check_llm_provider("azure", &json!({})).unwrap();
            assert!(!missing.ok);
            assert!(missing.missing.contains(&"endpoint (or set $AZURE_OPENAI_ENDPOINT)".to_string()), "{:?}", missing.missing);
            assert!(missing.missing.iter().any(|m| m.contains("api_key") && m.contains("credential")));
        }
        let ok = check_llm_provider("azure", &json!({"endpoint": "https://r.openai.azure.com", "credential": "default"})).unwrap();
        assert!(ok.ok && ok.found["credential"] == "argument");

        if env_free(&["AWS_SECRET_ACCESS_KEY"]) {
            let half = check_llm_provider("bedrock", &json!({"access_key": "AKID"})).unwrap();
            assert!(!half.ok && half.missing.iter().any(|m| m.contains("secret_key")));
        }
    }

    #[test]
    fn an_unknown_setting_is_caught_before_connecting() {
        let check = check_llm_provider("anthropic", &json!({"api_key": "k", "endpont": "typo"})).unwrap();
        assert!(!check.ok);
        assert_eq!(check.unknown, vec!["endpont"]);
        let err = get_provider_with("anthropic", &json!({"api_key": "k", "endpont": "typo"})).err().unwrap().to_string();
        assert!(err.contains("It accepts: api_key, base_url"), "{err}");
    }

    #[test]
    fn the_example_builds_the_provider_without_inlining_secrets() {
        let example = describe_llm_provider("azure").unwrap().example();
        assert!(example.starts_with("get_provider_with(\"azure\", &json!({\"endpoint\": "), "{example}");
        assert!(!example.contains("api_key"));
        assert!(describe_llm_provider("bedrock").unwrap().example().contains("\"region\": \"eu-west-1\""));
        let rendered = describe_llm_provider("bedrock").unwrap().render();
        assert!(rendered.starts_with("Amazon Bedrock — provider=\"bedrock\" (aliases: aws)"), "{rendered}");
        assert!(rendered.contains("one of [aws]"));
    }

    #[test]
    fn registered_providers_appear_with_their_own_fields() {
        register_provider("mine", Arc::new(|| Ok(Arc::new(FakeProvider::new(vec![]).with_default("mine")) as Arc<dyn Provider>)));
        assert_eq!(get_provider("mine").unwrap().name(), "fake");
        assert!(get_provider_with("mine", &json!({"x": 1})).is_err());
        const FIELDS: [ProviderField; 1] = [ProviderField { required: true, env: &["CUSTOM_TOKEN_UNSET_123"], ..ProviderField::new("token", "secret", "") }];
        let spec = ProviderSpec { name: "custom-llm", display_name: "Custom", description: "In-house gateway.", fields: &FIELDS, ..super::super::fake::SPEC };
        register_provider_with("custom-llm", spec, Arc::new(|_| Ok(Arc::new(FakeProvider::new(vec![])) as Arc<dyn Provider>)));
        let info = describe_llm_provider("custom-llm").unwrap();
        assert_eq!(info.fields[0].name, "token");
        assert!(!info.configured);
    }

    #[tokio::test]
    async fn ping_without_configuration_says_what_is_missing() {
        if !env_free(&["ANTHROPIC_API_KEY"]) {
            return;
        }
        let result = ping_llm_provider("anthropic", None, &Value::Null).await;
        assert_eq!(result["ok"], false);
        assert!(result["error"].as_str().unwrap().contains("ANTHROPIC_API_KEY"));
    }
}
