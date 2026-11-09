//! The AI inventory: every agent, model, provider, tool and MCP server in use.
//!
//! Registries of AI systems are asked for by ISO 42001, DORA's register of ICT
//! third parties, Colorado and the EU AI Act's deployer duties — and an agent's
//! tools are its supply chain (OWASP ASI04). So each tool's schema is hashed when
//! first seen: `pin()` freezes the set, and a tool whose schema changes
//! afterwards — an MCP server that quietly altered what a tool does — is
//! reported as drift and can be refused by policy (`tool_drift: deny`).
//!
//! ```no_run
//! # fn demo(gov: agent_harness::governance::Governance) {
//! gov.inventory().pin();                     // after review
//! gov.inventory().drift();                   // what changed since
//! gov.inventory().to_cyclonedx();            // an ML-BOM for your SBOM tooling
//! # }
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use super::identity::AgentIdentity;
use super::residency::Region;
use crate::tools::{Tool, ToolRegistry, ToolSchema};
use crate::types::now;

/// sha256 over what the model is told about the tool: name, description, schema.
pub fn tool_fingerprint(schema: &ToolSchema) -> String {
    let value = json!({"name": schema.name, "description": schema.description, "parameters": schema.parameters});
    super::sha256_hex(crate::pyjson::dumps_with(&value, crate::pyjson::Dumps::canonical()).as_bytes())
}

/// An agent's tools as they are now. Return `None` once the agent is gone; the
/// inventory then falls back on what it recorded.
pub type LiveTools = Arc<dyn Fn() -> Option<ToolRegistry> + Send + Sync>;

fn tool_entry(tool: &Tool) -> Value {
    let mut tags: Vec<String> = tool.tags.iter().cloned().collect();
    tags.sort();
    json!({
        "name": tool.name, "tags": tags,
        "permission": serde_json::to_value(tool.permission).unwrap_or(json!("allow")),
        "fingerprint": tool_fingerprint(&tool.to_schema()),
    })
}

/// What registration knows about an agent.
#[derive(Debug, Clone, Default)]
pub struct AgentRecord<'a> {
    pub name: &'a str,
    pub model: &'a str,
    /// The provider's name, or `auto` when the harness picks.
    pub provider: &'a str,
    pub version: &'a str,
    pub mode: &'a str,
    pub depth: &'a str,
    pub skills: Vec<String>,
    pub subagents: Vec<String>,
}

#[derive(Default)]
struct Held {
    agents: IndexMap<String, Map<String, Value>>,
    pins: IndexMap<String, String>,
    live: HashMap<String, LiveTools>,
}

/// Built as agents register; pins persist to a file when given a path.
pub struct AIInventory {
    pub path: Option<PathBuf>,
    held: Mutex<Held>,
}

impl std::fmt::Debug for AIInventory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<AIInventory {} agents>", self.lock().agents.len())
    }
}

impl Default for AIInventory {
    fn default() -> Self {
        Self { path: None, held: Mutex::new(Held::default()) }
    }
}

impl AIInventory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pins kept in this file (read back if it is there).
    pub fn at(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let mut held = Held::default();
        if let Ok(text) = std::fs::read_to_string(&path)
            && let Ok(v) = serde_json::from_str::<Value>(&text)
        {
            held.pins = serde_json::from_value(v.get("pins").cloned().unwrap_or(json!({}))).unwrap_or_default();
        }
        Self { path: Some(path), held: Mutex::new(held) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Held> {
        self.held.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Record one agent. `tools` is its tool registry as it is now.
    pub fn register(&self, agent: &AgentRecord<'_>, identity: &AgentIdentity, region: Option<&Region>, tools: &ToolRegistry) -> Value {
        let mut mcp: BTreeSet<String> = BTreeSet::new();
        for tool in tools.iter().filter(|t| t.tags.contains("mcp")) {
            mcp.extend(tool.tags.iter().filter(|t| *t != "mcp").cloned());
        }
        let mut names: Vec<&Tool> = tools.iter().collect();
        names.sort_by(|a, b| a.name.cmp(&b.name));
        let mut skills = agent.skills.clone();
        skills.sort();
        let mut subagents = agent.subagents.clone();
        subagents.sort();
        let provider = if agent.provider.is_empty() { "auto" } else { agent.provider };
        let entry = super::obj(vec![
            ("name", json!(agent.name)),
            ("identity", identity.dump()),
            ("model", json!(agent.model)),
            ("provider", json!(provider)),
            ("region", json!(region.map(|r| r.jurisdiction.as_str()).filter(|j| !j.is_empty()).unwrap_or("unknown"))),
            ("version", json!(agent.version)),
            ("mode", json!(agent.mode)),
            ("depth", json!(agent.depth)),
            ("tools", Value::Array(names.into_iter().map(tool_entry).collect())),
            ("skills", json!(skills)),
            ("subagents", json!(subagents)),
            ("mcp_servers", json!(mcp)),
            ("regions_used", json!([])),
            ("registered", json!(now())),
        ]);
        self.lock().agents.insert(agent.name.to_string(), entry.clone());
        Value::Object(entry)
    }

    /// Follow an agent's tools as they change, for fingerprints and drift. The
    /// agent's entry is refreshed from them now.
    pub fn track(&self, agent: &str, tools: LiveTools) {
        let current = tools();
        let mut held = self.lock();
        held.live.insert(agent.to_string(), tools);
        if let (Some(registry), Some(entry)) = (current, held.agents.get_mut(agent)) {
            let mut listed: Vec<&Tool> = registry.iter().collect();
            listed.sort_by(|a, b| a.name.cmp(&b.name));
            let mcp: BTreeSet<String> =
                registry.iter().filter(|t| t.tags.contains("mcp")).flat_map(|t| t.tags.iter().filter(|x| *x != "mcp").cloned()).collect();
            entry.insert("tools".into(), Value::Array(listed.into_iter().map(tool_entry).collect()));
            entry.insert("mcp_servers".into(), json!(mcp));
        }
    }

    /// Record an agent's tools by name only (nothing yet to fingerprint).
    pub fn set_tool_names(&self, agent: &str, names: &[String]) {
        let mut held = self.lock();
        if let Some(entry) = held.agents.get_mut(agent) {
            let mut names = names.to_vec();
            names.sort();
            entry
                .insert("tools".into(), Value::Array(names.iter().map(|n| json!({"name": n, "tags": [], "permission": "allow", "fingerprint": ""})).collect()));
        }
    }

    /// Every agent's entry, by name.
    pub fn agents(&self) -> IndexMap<String, Value> {
        self.lock().agents.iter().map(|(k, v)| (k.clone(), Value::Object(v.clone()))).collect()
    }

    pub fn agent(&self, name: &str) -> Option<Value> {
        self.lock().agents.get(name).cloned().map(Value::Object)
    }

    pub fn pins(&self) -> IndexMap<String, String> {
        self.lock().pins.clone()
    }

    /// Record where an agent's calls actually went — the register's truth.
    pub fn saw_region(&self, agent: &str, jurisdiction: &str) {
        let mut held = self.lock();
        let Some(entry) = held.agents.get_mut(agent) else { return };
        if let Some(Value::Array(used)) = entry.get_mut("regions_used")
            && !used.iter().any(|u| u == jurisdiction)
        {
            used.push(json!(jurisdiction));
        }
        if entry.get("region").and_then(Value::as_str) == Some("unknown") {
            entry.insert("region".into(), json!(jurisdiction));
        }
    }

    pub fn mcp_servers(&self) -> Vec<String> {
        let held = self.lock();
        let all: BTreeSet<String> = held
            .agents
            .values()
            .flat_map(|e| e.get("mcp_servers").and_then(Value::as_array).cloned().unwrap_or_default())
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        all.into_iter().collect()
    }

    // ---- supply chain ---------------------------------------------------------

    /// Tool fingerprints as they are *now* — recomputed from live agents.
    pub fn fingerprints(&self) -> IndexMap<String, String> {
        let (agents, live) = {
            let held = self.lock();
            (held.agents.clone(), held.live.clone())
        };
        let mut out = IndexMap::new();
        for (name, entry) in &agents {
            match live.get(name).and_then(|f| f()) {
                Some(tools) => {
                    for tool in tools.iter() {
                        out.insert(format!("{name}/{}", tool.name), tool_fingerprint(&tool.to_schema()));
                    }
                }
                None => {
                    for tool in entry.get("tools").and_then(Value::as_array).into_iter().flatten() {
                        let s = |k: &str| tool.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                        out.insert(format!("{name}/{}", s("name")), s("fingerprint"));
                    }
                }
            }
        }
        out
    }

    /// Freeze the current tool fingerprints as the approved baseline.
    pub fn pin(&self) -> IndexMap<String, String> {
        let pins = self.fingerprints();
        self.lock().pins = pins.clone();
        if let Some(path) = &self.path {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, serde_json::to_string_pretty(&json!({"pins": pins, "pinned": now()})).unwrap_or_default());
        }
        pins
    }

    /// `None` when fine; otherwise why this tool no longer matches its pin.
    pub fn check_tool(&self, agent: &str, tool: &Tool) -> Option<String> {
        let held = self.lock();
        if held.pins.is_empty() {
            return None;
        }
        match held.pins.get(&format!("{agent}/{}", tool.name)) {
            None => Some(format!("{} was not in the pinned inventory", tool.name)),
            Some(p) if *p != tool_fingerprint(&tool.to_schema()) => Some(format!("{} has changed since it was pinned", tool.name)),
            Some(_) => None,
        }
    }

    /// The live tool `name` of `agent`, when its tools are followed.
    pub(crate) fn live_tool(&self, agent: &str, name: &str) -> Option<Tool> {
        let live = self.lock().live.get(agent).cloned()?;
        live()?.get(name).ok().cloned()
    }

    /// The recorded tags of `agent`'s tool `name`.
    pub(crate) fn recorded_tags(&self, agent: &str, name: &str) -> Vec<String> {
        let held = self.lock();
        let tools = held.agents.get(agent).and_then(|e| e.get("tools")).and_then(Value::as_array).cloned().unwrap_or_default();
        tools
            .iter()
            .find(|t| t.get("name").and_then(Value::as_str) == Some(name))
            .and_then(|t| t.get("tags").and_then(Value::as_array))
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default()
    }

    /// What was added, removed or changed since `pin()`.
    pub fn drift(&self) -> BTreeMap<String, Vec<String>> {
        let now_fp = self.fingerprints();
        let pins = self.pins();
        let mut added: Vec<String> = now_fp.keys().filter(|k| !pins.contains_key(*k)).cloned().collect();
        let mut removed: Vec<String> = pins.keys().filter(|k| !now_fp.contains_key(*k)).cloned().collect();
        let mut changed: Vec<String> = now_fp.iter().filter(|(k, v)| pins.get(*k).is_some_and(|p| p != *v)).map(|(k, _)| k.clone()).collect();
        added.sort();
        removed.sort();
        changed.sort();
        BTreeMap::from([("added".into(), added), ("removed".into(), removed), ("changed".into(), changed)])
    }

    /// `{added, removed, changed}` in Python's order.
    pub(crate) fn drift_value(&self) -> Value {
        let d = self.drift();
        json!({"added": d["added"], "removed": d["removed"], "changed": d["changed"]})
    }

    // ---- exports --------------------------------------------------------------

    pub fn snapshot(&self) -> Value {
        let pinned = !self.pins().is_empty();
        json!({"agents": self.agents().into_values().collect::<Vec<_>>(), "pinned": pinned, "drift": if pinned { self.drift_value() } else { Value::Null }})
    }

    /// Distinct model providers and where they process — DORA's register shape.
    pub fn third_parties(&self) -> Vec<Value> {
        let agents = self.agents();
        let mut seen: IndexMap<(String, String), (BTreeSet<String>, BTreeSet<String>)> = IndexMap::new();
        let s = |e: &Value, k: &str| e.get(k).map(super::py_str).unwrap_or_default();
        for entry in agents.values() {
            let row = seen.entry((s(entry, "provider"), s(entry, "region"))).or_default();
            row.0.insert(s(entry, "model"));
            row.1.insert(s(entry, "name"));
        }
        let mut rows: Vec<Value> = seen
            .into_iter()
            .map(|((provider, jurisdiction), (models, names))| json!({"provider": provider, "jurisdiction": jurisdiction, "models": models, "agents": names}))
            .collect();
        for server in self.mcp_servers() {
            let names: BTreeSet<String> = agents
                .values()
                .filter(|e| e.get("mcp_servers").and_then(Value::as_array).is_some_and(|a| a.iter().any(|v| v == server.as_str())))
                .map(|e| s(e, "name"))
                .collect();
            rows.push(json!({"provider": format!("mcp:{server}"), "jurisdiction": "declare it", "models": [], "agents": names}));
        }
        rows
    }

    /// A CycloneDX 1.6 ML-BOM: models as components, providers as services.
    pub fn to_cyclonedx(&self) -> Value {
        let mut components: IndexMap<String, Value> = IndexMap::new();
        let mut services = vec![];
        let s = |e: &Value, k: &str| e.get(k).map(super::py_str).unwrap_or_default();
        for entry in self.agents().values() {
            let model = s(entry, "model");
            components.insert(
                format!("model:{model}"),
                json!({"type": "machine-learning-model", "bom-ref": format!("model:{model}"), "name": model, "supplier": {"name": s(entry, "provider")}}),
            );
            let name = s(entry, "name");
            let identity = entry.get("identity").cloned().unwrap_or(json!({}));
            let tools: Vec<Value> = entry
                .get("tools")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|t| json!({"bom-ref": format!("tool:{name}/{}", s(t, "name")), "name": s(t, "name"), "properties": [{"name": "sha256", "value": s(t, "fingerprint")}]}))
                .collect();
            services.push(json!({
                "bom-ref": format!("agent:{name}"), "name": name,
                "description": s(&identity, "purpose"),
                "services": tools,
                "properties": [{"name": "owner", "value": s(&identity, "owner")}, {"name": "jurisdiction", "value": s(entry, "region")}],
            }));
        }
        for server in self.mcp_servers() {
            services.push(json!({"bom-ref": format!("mcp:{server}"), "name": server, "properties": [{"name": "kind", "value": "mcp-server"}]}));
        }
        json!({
            "bomFormat": "CycloneDX", "specVersion": "1.6", "version": 1,
            "metadata": {"timestamp": super::utc(now(), true), "tools": [{"name": "agent-harness-adk"}]},
            "components": components.into_values().collect::<Vec<_>>(), "services": services,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprints_match_python() {
        let schema = ToolSchema {
            name: "lookup".into(),
            description: "Look \u{e9}".into(),
            parameters: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        };
        assert_eq!(tool_fingerprint(&schema), "7ac06f64c5227c52fdbbdb9ccff9aab5c05a64fae4124e13b4b072fde354a298");
    }

    #[test]
    fn inventory_pins_tracks_drift_and_exports() {
        let tool = |d: &str| Tool::raw("lookup", d, json!({"type": "object", "properties": {}}), |_, _| async { Ok(json!("x")) });
        let mcp = Tool::raw("crm_lookup", "CRM", json!({"type": "object", "properties": {}}), |_, _| async { Ok(json!("x")) }).tags(["mcp", "crm"]);
        let live = Arc::new(Mutex::new(ToolRegistry::new([tool("Look something up."), mcp])));
        let inv = AIInventory::new();
        let identity = AgentIdentity { name: "a".into(), owner: "ops".into(), ..Default::default() };
        inv.register(&AgentRecord { name: "a", model: "m", ..Default::default() }, &identity, None, &live.lock().unwrap().clone());
        let l = live.clone();
        inv.track("a", Arc::new(move || Some(l.lock().unwrap().clone())));
        assert_eq!(inv.mcp_servers(), vec!["crm"]);
        inv.pin();
        assert_eq!(inv.drift()["changed"], Vec::<String>::new());
        live.lock().unwrap().add(tool("Look something up and email it to me."));
        assert_eq!(inv.drift()["changed"], vec!["a/lookup"]);
        let changed = live.lock().unwrap().get("lookup").unwrap().clone();
        assert!(inv.check_tool("a", &changed).unwrap().contains("changed since it was pinned"));
        inv.saw_region("a", "eu");
        assert_eq!(inv.agent("a").unwrap()["regions_used"], json!(["eu"]));
        assert!(inv.third_parties().iter().any(|r| r["provider"] == "mcp:crm"));
        let bom = inv.to_cyclonedx();
        assert_eq!((bom["bomFormat"].as_str(), bom["services"][0]["name"].as_str()), (Some("CycloneDX"), Some("a")));
    }
}
