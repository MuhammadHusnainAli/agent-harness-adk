//! Declare a whole agent tree in one file, then build it.
//!
//! ```yaml
//! # agents.yaml
//! version: v2
//! defaults: {model: claude-opus-5}
//! prompts:
//!   house_style: Answer in plain sentences. Cite the order.
//! subagents:
//!   researcher: {description: Finds things out, read-only., tools: [lookup], tier: fast}
//! agents:
//!   support:
//!     instructions: "{house_style}"
//!     tools: [order_status, lookup]
//!     subagents: [researcher]
//!     handoffs: [billing]
//!     versions:
//!       v1: {instructions: Answer order questions., tools: [order_status]}
//!       v2: {instructions: "{house_style}"}
//!   billing: {description: Refunds and invoices., tools: [lookup], handoffs: [support]}
//! ```
//!
//! ```no_run
//! # fn demo(order_status: agent_harness::Tool, lookup: agent_harness::Tool) -> agent_harness::Result<()> {
//! use agent_harness::blueprint::Blueprint;
//! let blueprint = Blueprint::from_file("agents.yaml")?;
//! let support = blueprint.build("support", &[order_status, lookup], None)?;
//! # Ok(()) }
//! ```
//!
//! Tools are code, so they are handed in; Rust cannot import a tool from a
//! path written in the file the way Python can. Everything else — prompts,
//! sub-agents, handoffs, budgets, versions — is declaration, and belongs in the
//! file where it can be reviewed and diffed.
//!
//! An `a2a:` entry is an agent served elsewhere: [`Blueprint::remote`] reaches
//! it (Rust's `Agent` is always local, so it is not built in place as Python's
//! `RemoteAgent` is). The `openapi:` section's operations join the tools every
//! agent in the file can name.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::agent::{Agent, AgentBuilder, OutputContract};
use crate::errors::{Error, Result};
use crate::handoff::{Handoff, History};
use crate::harness::Harness;
use crate::llm_providers::parameters::Effort;
use crate::mcp::McpAgentServer;
use crate::prompts::Prompt;
use crate::runtime::Budget;
use crate::subagents::SubAgentSpec;
use crate::tools::{Tool, ToolRegistry};
use crate::versioning::AgentVersion;

/// One agent, as declared in the file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentEntry {
    pub instructions: String,
    pub description: String,
    pub model: Option<String>,
    pub mode: Option<Value>,
    pub depth: Option<String>,
    pub tier: Option<String>,
    pub effort: Option<Effort>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub thinking: Option<bool>,
    pub thinking_budget: Option<i64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub min_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub max_steps: Option<u32>,
    /// Tool names, from the tools handed to `build`. `None`: all of them.
    pub tools: Option<Vec<String>>,
    pub skills: Option<Value>,
    pub subagents: Vec<String>,
    /// A name, or `{agent: billing, history: text, sticky: false, description: ...}`.
    pub handoffs: Vec<Value>,
    pub memory: Option<bool>,
    pub trace: Option<Value>,
    pub budget: Option<Budget>,
    pub guardrails: Option<Value>,
    pub identity: Option<Value>,
    pub runtime_agents: Option<bool>,
    pub max_runtime_agents: Option<u32>,
    pub compact_at: Option<f64>,
    pub workspace: Option<Value>,
    pub allow_shell: Option<bool>,
    pub output_schema: Option<Value>,
    pub version: Option<String>,
    pub versions: IndexMap<String, Value>,
    pub a2a: Option<Value>,
    /// The environment variable holding this agent's own MCP key.
    pub mcp_key_env: Option<String>,
}

/// A file's worth of declarations, and the agents it can build.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Blueprint {
    pub version: String,
    /// Applied where neither the agent nor the caller said otherwise.
    pub defaults: Map<String, Value>,
    pub prompts: IndexMap<String, String>,
    pub guardrails: IndexMap<String, Value>,
    pub subagents: IndexMap<String, SubAgentSpec>,
    pub agents: IndexMap<String, AgentEntry>,
    pub memory: Option<Value>,
    pub sessions: Option<Value>,
    pub workflows: IndexMap<String, Value>,
    pub openapi: Map<String, Value>,
    #[serde(skip)]
    pub path: Option<String>,
}

fn known(names: impl Iterator<Item = String>) -> String {
    let mut v: Vec<String> = names.collect();
    v.sort();
    if v.is_empty() { "none".into() } else { v.join(", ") }
}

impl Blueprint {
    // ---- loading -----------------------------------------------------------
    /// Load a `.yaml`, `.yml` or `.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = path.as_ref();
        if !file.is_file() {
            return Err(Error::config(format!("no blueprint at {}", file.display())));
        }
        let fmt = file.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
        let mut b = Self::from_text(&std::fs::read_to_string(file)?, &fmt)?;
        b.path = Some(file.to_string_lossy().into_owned());
        Ok(b)
    }

    pub fn from_text(text: &str, fmt: &str) -> Result<Self> {
        let data: Value = if fmt == "json" {
            serde_json::from_str(text).map_err(|e| Error::config(format!("bad blueprint JSON: {e}")))?
        } else {
            serde_yaml::from_str(text).map_err(|e| Error::config(format!("bad blueprint YAML: {e}")))?
        };
        Self::from_value(data)
    }

    pub fn from_value(mut data: Value) -> Result<Self> {
        let Value::Object(top) = &mut data else { return Err(Error::config("a blueprint must be a mapping at the top level")) };
        // In a file the sub-agent's name is the mapping key.
        if let Some(Value::Object(subs)) = top.get_mut("subagents") {
            for (key, spec) in subs.iter_mut() {
                if let Value::Object(s) = spec {
                    s.entry("name").or_insert_with(|| Value::String(key.clone()));
                }
            }
        }
        let mut b: Blueprint = serde_json::from_value(data).map_err(|e| Error::config(format!("bad blueprint: {e}")))?;
        if b.version.is_empty() {
            b.version = "v1".into();
        }
        Ok(b)
    }

    /// Write it back out, in whichever format the extension asks for.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<PathBuf> {
        let file = path.as_ref();
        let text = if file.extension().is_some_and(|e| e == "json") {
            serde_json::to_string_pretty(self).map_err(|e| Error::config(e.to_string()))?
        } else {
            serde_yaml::to_string(self).map_err(|e| Error::config(e.to_string()))?
        };
        std::fs::write(file, text)?;
        Ok(file.to_path_buf())
    }

    pub fn names(&self) -> Vec<String> {
        let mut n: Vec<String> = self.agents.keys().cloned().collect();
        n.sort();
        n
    }

    // ---- resolving ----------------------------------------------------------
    /// Substitute `{prompt_name}` references from the prompts section;
    /// other braces are left alone.
    pub fn render(&self, text: &str) -> String {
        if text.is_empty() || !text.contains('{') || self.prompts.is_empty() {
            return text.to_string();
        }
        let vars: Map<String, Value> = self.prompts.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect();
        Prompt::new("_inline", text).render(&Value::Object(vars)).unwrap_or_else(|_| text.to_string())
    }

    fn tools(&self, wanted: &Option<Vec<String>>, registry: &ToolRegistry) -> Result<Vec<Tool>> {
        let Some(wanted) = wanted else { return Ok(registry.iter().cloned().collect()) };
        let missing: Vec<&String> = wanted.iter().filter(|n| !registry.contains(n)).collect();
        if !missing.is_empty() {
            let names: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
            return Err(Error::config(format!(
                "the blueprint asks for tools that were not supplied: {}. Pass them to build(...). Available: {}",
                names.join(", "),
                known(registry.names().into_iter())
            )));
        }
        Ok(wanted.iter().filter_map(|n| registry.get(n).ok().cloned()).collect())
    }

    fn subagent(&self, name: &str) -> Result<SubAgentSpec> {
        let Some(spec) = self.subagents.get(name) else {
            return Err(Error::config(format!("sub-agent '{name}' is not declared; known: {}", known(self.subagents.keys().cloned()))));
        };
        let mut spec = spec.clone();
        spec.instructions = self.render(&spec.instructions);
        Ok(spec)
    }

    fn version(&self, spec: &Value) -> Result<AgentVersion> {
        let mut map = spec.as_object().cloned().unwrap_or_default();
        let subs = map.remove("subagents");
        let mut v: AgentVersion = serde_json::from_value(Value::Object(map)).map_err(|e| Error::config(format!("bad version: {e}")))?;
        if let Some(i) = &v.instructions {
            v.instructions = Some(self.render(i));
        }
        if let Some(Value::Array(names)) = subs {
            v.subagent_specs = Some(names.iter().filter_map(Value::as_str).map(|n| self.subagent(n)).collect::<Result<_>>()?);
        }
        Ok(v)
    }

    /// An entry with the file's defaults filled in where it said nothing.
    fn with_defaults(&self, entry: &AgentEntry) -> Result<AgentEntry> {
        if self.defaults.is_empty() {
            return Ok(entry.clone());
        }
        let mut v = serde_json::to_value(entry).map_err(|e| Error::config(e.to_string()))?;
        if let Value::Object(m) = &mut v {
            for (k, d) in &self.defaults {
                let unset = match m.get(k) {
                    None | Some(Value::Null) => true,
                    Some(Value::String(s)) => s.is_empty(),
                    _ => false,
                };
                if unset {
                    m.insert(k.clone(), d.clone());
                }
            }
        }
        serde_json::from_value(v).map_err(|e| Error::config(format!("bad defaults: {e}")))
    }

    fn builder(&self, name: &str, entry: &AgentEntry, registry: &ToolRegistry, harness: &Harness) -> Result<AgentBuilder> {
        let e = self.with_defaults(entry)?;
        let mut b = Agent::builder(name)
            .instructions(self.render(&e.instructions))
            .description(e.description.clone())
            .tools(self.tools(&e.tools, registry)?)
            .harness(harness.clone());
        if let Some(on) = e.memory {
            b = b.memory(on);
        }
        if let Some(w) = &e.workspace {
            b = b.workspace_spec(w.clone());
        }
        // A mapping of its own, or the name of a set declared under `guardrails:`.
        match &e.guardrails {
            None | Some(Value::Null) => {}
            Some(Value::String(set)) => {
                let Some(held) = self.guardrails.get(set) else {
                    return Err(Error::config(format!("guardrails '{set}' are not defined; known: {}", known(self.guardrails.keys().cloned()))));
                };
                b = b.guardrails(crate::guardrails::AgentGuardrails::from_value(held).map_err(|err| Error::config(format!("{name}: guardrails: {err}")))?);
            }
            Some(rails) => {
                b = b.guardrails(crate::guardrails::AgentGuardrails::from_value(rails).map_err(|err| Error::config(format!("{name}: guardrails: {err}")))?)
            }
        }
        match &e.mode {
            None if e.depth.is_some() => return Err(Error::config(format!("{name}: depth needs a mode — mode: chat, research or cowork"))),
            None => {}
            Some(Value::String(m)) => b = b.mode_named(m, e.depth.as_deref()),
            Some(Value::Object(m)) => {
                let mode = m.get("name").and_then(Value::as_str).ok_or_else(|| Error::config(format!("{name}: a mode given as a mapping needs a `name`")))?;
                let depth = m.get("depth").and_then(Value::as_str).or(e.depth.as_deref());
                b = b.mode_named(mode, depth);
            }
            Some(other) => return Err(Error::config(format!("{name}: mode must be a name or a mapping — got {other}"))),
        }
        for sub in &e.subagents {
            b = b.subagent_spec(self.subagent(sub)?);
        }
        if let Some(m) = &e.model {
            b = b.model(m.clone());
        }
        if let Some(t) = &e.tier {
            b = b.tier(t.clone());
        }
        if let Some(x) = e.effort {
            b = b.effort(x);
        }
        if let Some(x) = e.temperature {
            b = b.temperature(x);
        }
        if let Some(x) = e.max_tokens {
            b = b.max_tokens(x);
        }
        if let Some(x) = e.thinking {
            b = b.thinking(x);
        }
        if let Some(x) = e.thinking_budget {
            b = b.thinking_budget(x);
        }
        if let Some(x) = e.top_p {
            b = b.top_p(x);
        }
        if let Some(x) = e.top_k {
            b = b.top_k(x);
        }
        if let Some(x) = e.min_p {
            b = b.min_p(x);
        }
        if let Some(x) = e.frequency_penalty {
            b = b.frequency_penalty(x);
        }
        if let Some(x) = e.presence_penalty {
            b = b.presence_penalty(x);
        }
        if let Some(x) = e.repetition_penalty {
            b = b.repetition_penalty(x);
        }
        if let Some(x) = e.seed {
            b = b.seed(x);
        }
        if let Some(x) = e.max_steps {
            b = b.max_steps(x);
        }
        if let Some(x) = e.runtime_agents {
            b = b.runtime_agents(x);
        }
        if let Some(x) = e.max_runtime_agents {
            b = b.max_runtime_agents(x);
        }
        if let Some(x) = e.compact_at {
            b = b.compact_at(x);
        }
        if let Some(budget) = &e.budget {
            b = b.budget(budget.clone());
        }
        if let Some(schema) = &e.output_schema {
            b = b.output_contract(OutputContract::schema(schema.clone()));
        }
        if !e.versions.is_empty() {
            for (key, spec) in &e.versions {
                b = b.add_version(key.clone(), self.version(spec)?);
            }
            b = b.version(e.version.clone().unwrap_or_else(|| self.version.clone()));
        }
        Ok(b)
    }

    // ---- building ------------------------------------------------------------
    /// Build one declared agent — and, with it, every agent it can hand the
    /// conversation to, on the same harness.
    pub fn build(&self, name: &str, tools: &[Tool], harness: Option<Harness>) -> Result<Agent> {
        let registry = ToolRegistry::new(tools.iter().cloned());
        let mut built = HashMap::new();
        self.build_into(name, &mut built, &registry, &harness.unwrap_or_default())
    }

    /// Every declared agent, sharing one harness.
    /// The memory store the file declares, if it declares one: a URL, or a
    /// mapping with a `url`.
    pub fn memory_store(&self) -> Result<Option<Arc<dyn crate::memory::MemoryStore>>> {
        match &self.memory {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(url)) => crate::memory::memory_provider(url).map(Some),
            Some(Value::Object(m)) => match m.get("url").and_then(Value::as_str) {
                Some(url) => crate::memory::memory_provider(url).map(Some),
                None => Err(Error::config("a memory section needs a `url`")),
            },
            Some(other) => Err(Error::config(format!("memory is a URL or a mapping with a `url` — got {other}"))),
        }
    }

    /// The session store the file declares, if it declares one. A mapping's
    /// other keys (`table`, `ttl`, `collection`, ...) go to the store.
    pub fn session_store(&self) -> Result<Option<Arc<dyn crate::runtime::SessionStore>>> {
        match &self.sessions {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(url)) => crate::sessions::session_provider(url).map(Some),
            Some(Value::Object(m)) => {
                let mut options = m.clone();
                let Some(Value::String(url)) = options.remove("url") else {
                    return Err(Error::config("a sessions section needs a `url`"));
                };
                crate::sessions::session_provider_with(&url, &options).map(Some)
            }
            Some(other) => Err(Error::config(format!("sessions is a URL or a mapping with a `url` — got {other}"))),
        }
    }

    /// The tools you passed, and the ones the file's `openapi:` section provides.
    fn with_apis(&self, registry: &ToolRegistry) -> Result<ToolRegistry> {
        if self.openapi.is_empty() {
            return Ok(registry.clone());
        }
        let mut all = registry.clone();
        for kit in self.api_toolkits()? {
            all.extend(kit.into_tools());
        }
        Ok(all)
    }

    /// The toolkits the `openapi:` section declares. A relative spec path is
    /// read beside the blueprint file.
    pub fn api_toolkits(&self) -> Result<Vec<crate::toolkits::openapi::OpenAPIToolkit>> {
        let base = self.path.as_ref().and_then(|p| std::path::Path::new(p).parent().map(std::path::Path::to_path_buf));
        crate::toolkits::openapi::toolkits_from_config(&self.openapi, base.as_deref())
    }

    /// An agent the file declares by where it is served (`a2a:`), not by how it
    /// is built. A secret is named in the file (`token_env`), never written there.
    pub fn remote(&self, name: &str, harness: Option<Harness>) -> Result<crate::a2a::RemoteAgent> {
        let Some(entry) = self.agents.get(name) else {
            return Err(Error::config(format!("no agent '{name}' in this blueprint; declared: {}", known(self.agents.keys().cloned()))));
        };
        let mut options = match &entry.a2a {
            Some(Value::String(url)) => {
                let mut m = Map::new();
                m.insert("url".into(), Value::String(url.clone()));
                m
            }
            Some(Value::Object(m)) => m.clone(),
            _ => return Err(Error::config(format!("{name}: is not an `a2a` agent"))),
        };
        let Some(Value::String(url)) = options.remove("url") else {
            return Err(Error::config(format!("{name}: `a2a` needs a `url`")));
        };
        let mut b = crate::a2a::RemoteAgent::builder(&url).name(name).description(&entry.description);
        if let Some(var) = options.remove("token_env").and_then(|v| v.as_str().map(str::to_string)) {
            let token = std::env::var(&var)
                .ok()
                .filter(|t| !t.is_empty())
                .ok_or_else(|| Error::config(format!("{name}: the environment variable {var} is not set")))?;
            b = b.token(&token);
        }
        if let Some(t) = options.remove("token").and_then(|v| v.as_str().map(str::to_string)) {
            b = b.token(&t);
        }
        if let Some(t) = options.remove("timeout").and_then(|v| v.as_f64()) {
            b = b.timeout(t);
        }
        if let Some(n) = options.remove("retries").and_then(|v| v.as_u64()) {
            b = b.retries(n as u32);
        }
        if let Some(Value::Object(headers)) = options.remove("headers") {
            for (k, v) in headers {
                b = b.header(&k, v.as_str().unwrap_or(&v.to_string()));
            }
        }
        if !options.is_empty() {
            let mut keys: Vec<&String> = options.keys().collect();
            keys.sort();
            return Err(Error::config(format!("{name}: unknown `a2a` options: {}", keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(", "))));
        }
        if let Some(h) = harness {
            b = b.harness(h);
        }
        if let Some(id) = &entry.identity {
            b = b.identity(id.clone());
        }
        b.build().map_err(|e| Error::config(format!("{name}: {e}")))
    }

    pub fn build_all(&self, tools: &[Tool], harness: Option<Harness>) -> Result<IndexMap<String, Agent>> {
        let registry = ToolRegistry::new(tools.iter().cloned());
        let shared = harness.unwrap_or_default();
        let mut built = HashMap::new();
        // Remote (`a2a`) entries are reached with `remote()`, not built.
        self.agents.iter().filter(|(_, e)| e.a2a.is_none()).map(|(n, _)| Ok((n.clone(), self.build_into(n, &mut built, &registry, &shared)?))).collect()
    }

    pub(crate) fn build_into(&self, name: &str, built: &mut HashMap<String, Agent>, registry: &ToolRegistry, harness: &Harness) -> Result<Agent> {
        if let Some(a) = built.get(name) {
            return Ok(a.clone());
        }
        let Some(entry) = self.agents.get(name) else {
            return Err(Error::config(format!("no agent '{name}' in this blueprint; declared: {}", known(self.agents.keys().cloned()))));
        };
        if entry.a2a.is_some() {
            return Err(Error::config(format!(
                "{name} is served elsewhere (`a2a`), so it is not built here — reach it with `blueprint.remote(\"{name}\", ..)` and hand it to an agent as a tool (`as_tool`) or through `RemoteAgent::delegate_tool`"
            )));
        }
        let registry = &self.with_apis(registry)?;
        let agent = self.builder(name, entry, registry, harness)?.build()?;
        built.insert(name.to_string(), agent.clone());
        // Introduced after it is built, so two agents may hand off to each other.
        for item in &entry.handoffs {
            let (target, options) = match item {
                Value::String(s) => (s.clone(), Map::new()),
                Value::Object(m) => (m.get("agent").and_then(Value::as_str).unwrap_or("").to_string(), m.clone()),
                other => return Err(Error::config(format!("{name}: a handoff needs an `agent` — got {other}"))),
            };
            if target.is_empty() {
                return Err(Error::config(format!("{name}: a handoff needs an `agent` — got {item}")));
            }
            if !self.agents.contains_key(&target) {
                return Err(Error::config(format!(
                    "{name} hands off to '{target}', which is not an agent in this blueprint; declared: {}",
                    known(self.agents.keys().cloned())
                )));
            }
            let to = self.build_into(&target, built, registry, harness)?;
            let mut handoff = Handoff::new(to);
            if let Some(d) = options.get("description").and_then(Value::as_str) {
                handoff = handoff.describe(d);
            }
            if let Some(s) = options.get("sticky").and_then(Value::as_bool) {
                handoff = handoff.sticky(s);
            }
            match options.get("history").and_then(Value::as_str) {
                None | Some("full") => {}
                Some("text") => handoff = handoff.history(History::Text),
                Some("fresh") => handoff = handoff.history(History::Fresh),
                Some(other) => return Err(Error::config(format!("handoff history must be one of full, text, fresh — got '{other}'"))),
            }
            agent.add_handoff(handoff)?;
        }
        Ok(agent)
    }

    /// Serve this file's agents over MCP, each behind the key its
    /// `mcp_key_env` names. `names` chooses which; left out, every agent that
    /// names a key (or all of them when none do).
    pub fn mcp_server(&self, names: Option<&[String]>, tools: &[Tool], harness: Option<Harness>) -> Result<McpAgentServer> {
        let keyed: Vec<String> = self.agents.iter().filter(|(_, e)| e.mcp_key_env.is_some()).map(|(n, _)| n.clone()).collect();
        let chosen: Vec<String> = match names {
            Some(n) => n.to_vec(),
            None if keyed.is_empty() => self.agents.keys().cloned().collect(),
            None => keyed,
        };
        let registry = ToolRegistry::new(tools.iter().cloned());
        let shared = harness.unwrap_or_default();
        let mut built = HashMap::new();
        let mut agents = Vec::new();
        let mut keys = Vec::new();
        for name in &chosen {
            let Some(entry) = self.agents.get(name) else {
                return Err(Error::config(format!("no agent '{name}' in this blueprint; declared: {}", known(self.agents.keys().cloned()))));
            };
            if let Some(var) = &entry.mcp_key_env {
                let secret = std::env::var(var)
                    .ok()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| Error::config(format!("{name}: the environment variable {var} is not set")))?;
                keys.push((name.clone(), secret));
            }
            agents.push(self.build_into(name, &mut built, &registry, &shared)?);
        }
        let mut server = McpAgentServer::builder(agents);
        for (agent, key) in keys {
            server = server.agent_key(&agent, key);
        }
        server.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::NoArgs;

    const FILE: &str = r#"
version: v2
defaults: {model: claude-opus-5, max_steps: 7}
prompts:
  house_style: Answer in plain sentences. Cite the order.
subagents:
  researcher:
    description: Finds things out, read-only.
    instructions: "Cite every claim. {house_style}"
    tools: [lookup]
    tier: fast
agents:
  support:
    instructions: "{house_style} Literal {braces} stay."
    tools: [order_status, lookup]
    subagents: [researcher]
    handoffs: [billing]
    versions:
      v1: {instructions: Answer order questions., tools: [order_status]}
      v2: {instructions: "{house_style}"}
  billing:
    description: Refunds and invoices.
    model: claude-haiku-4-5
    tools: [lookup]
    handoffs: [{agent: support, history: text, sticky: false}]
"#;

    fn tools() -> Vec<Tool> {
        vec![Tool::new("order_status", |_: NoArgs, _| async { Ok("shipped") }), Tool::new("lookup", |_: NoArgs, _| async { Ok("found") })]
    }

    fn harness() -> Harness {
        Harness::testing(None)
    }

    #[test]
    fn builds_the_tree() {
        let b = Blueprint::from_text(FILE, "yaml").unwrap();
        assert_eq!(b.names(), vec!["billing", "support"]);
        let support = b.build("support", &tools(), Some(harness())).unwrap();
        assert_eq!(support.version(), "v2");
        assert_eq!(support.instructions(), "Answer in plain sentences. Cite the order.");
        assert_eq!(support.model(), "claude-opus-5");
        assert_eq!(support.subagents()["researcher"].instructions(), "Cite every claim. Answer in plain sentences. Cite the order.");
        assert_eq!(support.subagents()["researcher"].tools().names(), vec!["lookup"]);
        let billing = support.handoffs()["billing"].agent.clone();
        assert_eq!(billing.model(), "claude-haiku-4-5");
        // The two hand off to each other.
        assert!(!billing.handoffs()["support"].sticky);
        assert_eq!(support.use_version("v1").unwrap().tools().names(), vec!["delegate", "handoff", "order_status", "recall", "remember"]);
        assert!(Arc::ptr_eq(&support.memory().unwrap().store, &support.use_version("v1").unwrap().memory().unwrap().store), "versions share memory");
        assert_eq!(b.render("{house_style} {other}"), "Answer in plain sentences. Cite the order. {other}");
    }

    #[test]
    fn explains_what_is_missing() {
        let b = Blueprint::from_text(FILE, "yaml").unwrap();
        let err = b.build("support", &[], Some(harness())).unwrap_err().to_string();
        assert!(err.starts_with("the blueprint asks for tools that were not supplied: order_status, lookup"), "{err}");
        assert!(b.build("nobody", &tools(), None).unwrap_err().to_string().starts_with("no agent 'nobody'"));
        let moded = Blueprint::from_text("agents: {a: {mode: chat, depth: fast}, b: {workspace: true}}", "yaml").unwrap();
        assert_eq!(moded.build("a", &[], Some(harness())).unwrap().max_steps_hint(), 6);
        let worker = moded.build("b", &[], Some(harness())).unwrap();
        assert!(worker.workspace().is_some() && worker.tools().contains("fs_write"));
        assert!(!worker.tools().contains("shell"), "on this machine a shell is asked for, never assumed");
        assert!(Blueprint::from_text("[1, 2]", "yaml").is_err());
    }

    #[test]
    fn round_trips_through_a_file() {
        let b = Blueprint::from_text(FILE, "yaml").unwrap();
        let dir = std::env::temp_dir().join(format!("ah-bp-{}", crate::types::new_id("")));
        std::fs::create_dir_all(&dir).unwrap();
        let file = b.save(dir.join("agents.json")).unwrap();
        let back = Blueprint::from_file(&file).unwrap();
        assert_eq!(back.subagents["researcher"].tier.as_deref(), Some("fast"));
        assert_eq!(back.agents["support"].versions.len(), 2);
        std::fs::remove_dir_all(dir).ok();
        let all = b.build_all(&tools(), Some(harness())).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn apis_and_remote_agents() {
        let spec = r#"{"openapi": "3.0.0", "info": {"title": "Shop", "version": "1"}, "servers": [{"url": "https://shop.test"}], "paths": {"/orders/{id}": {"get": {"operationId": "getOrder", "parameters": [{"name": "id", "in": "path", "required": true, "schema": {"type": "string"}}], "responses": {"200": {"description": "ok"}}}}}}"#;
        let text = format!(
            "openapi:
  shop:
    spec: '{spec}'
agents:
  clerk:
    instructions: Look orders up.
    tools: [getOrder]
  billing:
    description: Bills.
    a2a: {{url: 'https://billing.test/a2a', timeout: 5}}
"
        );
        let b = Blueprint::from_text(&text, "yaml").unwrap();
        let clerk = b.build("clerk", &[], Some(harness())).unwrap();
        assert!(clerk.tools().contains("getOrder"), "the file's API is a tool its agents can name");
        assert!(b.build("billing", &[], Some(harness())).unwrap_err().to_string().contains("is served elsewhere"));
        let remote = b.remote("billing", Some(harness())).unwrap();
        assert_eq!((remote.name(), remote.url()), ("billing", "https://billing.test/a2a"));
        assert_eq!(remote.as_tool(None, None).name, "billing");
        assert_eq!(b.build_all(&[], Some(harness())).unwrap().keys().collect::<Vec<_>>(), vec!["clerk"], "remote entries are reached, not built");
        assert!(b.remote("clerk", None).unwrap_err().to_string().contains("is not an `a2a` agent"));
    }
}
