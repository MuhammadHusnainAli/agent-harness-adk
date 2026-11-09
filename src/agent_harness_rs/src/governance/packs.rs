//! Jurisdiction and framework packs: the law, as data.
//!
//! A pack says three things about one law, regulation or standard:
//!
//! * **requirements** — its articles, each mapped to catalog controls (or marked
//!   `manual` where only people can meet it: appointing a DPO, registering in
//!   the EU database, signing a DPIA);
//! * **policy** — the rules, residency limits, retention and transparency
//!   settings it implies, merged into the governance policy;
//! * **incidents** — who must be told of what, and how fast.
//!
//! The built-in packs are the same YAML the Python package ships
//! (`governance/frameworks/packs/*.yaml`); your own go anywhere and are loaded
//! by path. Each carries the date its content was checked (`as_of`) and its
//! sources. Laws change; review a pack before relying on it.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::{LazyLock, Mutex};

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use super::frameworks::PACKS;
use super::policy::Policy;
use super::py_str;
use crate::errors::{Error, Result};

/// One article or clause, and the controls that meet it.
#[derive(Debug, Clone, PartialEq)]
pub struct Requirement {
    pub r#ref: String,
    pub title: String,
    pub controls: Vec<String>,
    pub manual: bool,
    pub note: String,
}

/// incident kind → `[{notify, within (hours), basis, min_severity}]`.
pub type Clocks = IndexMap<String, Vec<Map<String, Value>>>;

/// One law, regulation, guidance or standard.
#[derive(Debug, Clone, PartialEq)]
pub struct Pack {
    pub id: String,
    pub name: String,
    pub jurisdiction: String,
    /// `law`, `regulation`, `guidance`, `standard` or `framework`.
    pub kind: String,
    pub binding: bool,
    pub status: String,
    pub as_of: String,
    pub summary: String,
    pub sources: Vec<String>,
    pub requirements: Vec<Requirement>,
    pub policy: Policy,
    pub incidents: Clocks,
}

const KINDS: [&str; 5] = ["law", "regulation", "guidance", "standard", "framework"];
const PACK_FIELDS: [&str; 12] =
    ["id", "name", "jurisdiction", "kind", "binding", "status", "as_of", "summary", "sources", "requirements", "policy", "incidents"];

fn text(m: &Map<String, Value>, key: &str, default: &str) -> std::result::Result<String, String> {
    match m.get(key) {
        None => Ok(default.to_string()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(_) => Err(format!("`{key}` must be text")),
    }
}

fn texts(m: &Map<String, Value>, key: &str) -> std::result::Result<Vec<String>, String> {
    match m.get(key) {
        None => Ok(vec![]),
        Some(Value::Array(a)) => a.iter().map(|v| v.as_str().map(str::to_string).ok_or_else(|| format!("`{key}` must be a list of text"))).collect(),
        Some(_) => Err(format!("`{key}` must be a list")),
    }
}

impl Requirement {
    fn from_value(value: &Value) -> std::result::Result<Self, String> {
        let m = value.as_object().ok_or("a requirement must be a mapping")?;
        if let Some(k) = m.keys().find(|k| !["ref", "title", "controls", "manual", "note"].contains(&k.as_str())) {
            return Err(format!("unknown field `{k}` in a requirement"));
        }
        let required = |k: &str| match m.get(k) {
            Some(Value::String(s)) => Ok(s.clone()),
            Some(v) if !v.is_null() => Ok(py_str(v)),
            _ => Err(format!("a requirement needs `{k}`")),
        };
        Ok(Self {
            r#ref: required("ref")?,
            title: required("title")?,
            controls: texts(m, "controls")?,
            manual: m.get("manual").and_then(Value::as_bool).unwrap_or(false),
            note: text(m, "note", "")?,
        })
    }
}

impl Pack {
    /// A pack from its JSON (what its YAML reads as). Its rules are stamped with
    /// the pack id, so every decision can say which law asked for it.
    pub fn from_value(value: &Value) -> Result<Self> {
        let m = value.as_object().ok_or_else(|| Error::config("expected a mapping"))?;
        Self::parse(m).map_err(Error::config)
    }

    fn parse(m: &Map<String, Value>) -> std::result::Result<Self, String> {
        if let Some(k) = m.keys().find(|k| !PACK_FIELDS.contains(&k.as_str())) {
            return Err(format!("unknown field `{k}`"));
        }
        let id = match m.get("id") {
            Some(Value::String(s)) => s.clone(),
            _ => return Err("`id` is required".into()),
        };
        let name = match m.get("name") {
            Some(Value::String(s)) => s.clone(),
            _ => return Err("`name` is required".into()),
        };
        let kind = text(m, "kind", "law")?;
        if !KINDS.contains(&kind.as_str()) {
            return Err(format!("`kind` must be one of {}", KINDS.join(", ")));
        }
        let requirements = match m.get("requirements") {
            None | Some(Value::Null) => vec![],
            Some(Value::Array(a)) => a.iter().map(Requirement::from_value).collect::<std::result::Result<Vec<_>, _>>()?,
            Some(_) => return Err("`requirements` must be a list".into()),
        };
        let mut policy = match m.get("policy") {
            None | Some(Value::Null) => Policy::default(),
            Some(v) => Policy::from_value(v).map_err(|e| e.to_string())?,
        };
        let mut incidents = Clocks::new();
        if let Some(v) = m.get("incidents").filter(|v| !v.is_null()) {
            let kinds = v.as_object().ok_or("`incidents` must be a mapping")?;
            for (kind, clocks) in kinds {
                let clocks = clocks.as_array().ok_or("each incident kind lists its clocks")?;
                let clocks =
                    clocks.iter().map(|c| c.as_object().cloned().ok_or("an incident clock must be a mapping")).collect::<std::result::Result<Vec<_>, _>>()?;
                incidents.insert(kind.clone(), clocks);
            }
        }
        // Stamp the rules so every decision can say which law asked for it.
        policy.packs = vec![id.clone()];
        for rule in &mut policy.rules {
            if rule.source.is_empty() {
                rule.source = id.clone();
            }
            if !rule.id.starts_with(&format!("{id}/")) {
                rule.id = format!("{id}/{}", rule.id);
            }
        }
        Ok(Self {
            name,
            jurisdiction: text(m, "jurisdiction", "global")?,
            kind,
            binding: m.get("binding").and_then(Value::as_bool).unwrap_or(true),
            status: text(m, "status", "")?,
            as_of: m.get("as_of").filter(|v| !v.is_null()).map(py_str).unwrap_or_default(),
            summary: text(m, "summary", "")?,
            sources: texts(m, "sources")?,
            requirements,
            policy,
            incidents,
            id,
        })
    }

    /// Every control any of its requirements names.
    pub fn controls(&self) -> BTreeSet<String> {
        self.requirements.iter().flat_map(|r| r.controls.iter().cloned()).collect()
    }

    /// The pack as a JSON summary: id, name, jurisdiction, status, and so on.
    pub fn header(&self) -> Value {
        json!({
            "id": self.id, "name": self.name, "jurisdiction": self.jurisdiction, "kind": self.kind,
            "binding": self.binding, "status": self.status, "as_of": self.as_of, "sources": self.sources,
        })
    }
}

fn parse(text: &str, place: &str) -> Result<Pack> {
    let data: Value = serde_yaml::from_str(text).map_err(|e| Error::config(format!("pack {place} is invalid: {e}")))?;
    let Some(m) = data.as_object() else {
        return Err(Error::config(format!("pack {place}: expected a mapping")));
    };
    Pack::parse(m).map_err(|e| Error::config(format!("pack {place} is invalid: {e}")))
}

static BUILTIN: LazyLock<Mutex<HashMap<String, Pack>>> = LazyLock::new(Default::default);

/// A built-in pack by id (`"gdpr"`), or a YAML file by path.
pub fn load_pack(reference: &str) -> Result<Pack> {
    if reference.ends_with(".yaml") || reference.ends_with(".yml") || reference.contains('/') || reference.contains('\\') {
        let path = Path::new(reference);
        if !path.exists() {
            return Err(Error::config(format!("pack file not found: {reference}")));
        }
        let text = std::fs::read_to_string(path).map_err(|e| Error::config(format!("pack file {reference}: {e}")))?;
        return parse(&text, reference);
    }
    if let Some(p) = BUILTIN.lock().unwrap_or_else(|p| p.into_inner()).get(reference) {
        return Ok(p.clone());
    }
    let Some((_, text)) = PACKS.iter().find(|(id, _)| *id == reference) else {
        return Err(Error::config(format!("no governance pack named {}; available: {}", super::py_str_repr(reference), list_packs().join(", "))));
    };
    let pack = parse(text, reference)?;
    BUILTIN.lock().unwrap_or_else(|p| p.into_inner()).insert(reference.to_string(), pack.clone());
    Ok(pack)
}

/// The ids of every built-in pack.
pub fn list_packs() -> Vec<String> {
    let mut ids: Vec<String> = PACKS.iter().map(|(id, _)| id.to_string()).collect();
    ids.sort();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::governance::evidence::CONTROLS;

    #[test]
    fn every_pack_loads_with_sources_and_known_controls() {
        let packs = list_packs();
        assert!(packs.len() >= 25, "{}", packs.len());
        let known: BTreeSet<String> = CONTROLS.iter().map(|c| c.id.to_string()).collect();
        for id in &packs {
            let pack = load_pack(id).unwrap();
            assert!(!pack.sources.is_empty() && !pack.as_of.is_empty() && !pack.requirements.is_empty(), "{id}");
            assert!(pack.controls().is_subset(&known), "{id}");
            assert!(pack.policy.rules.iter().all(|r| r.id.starts_with(&format!("{id}/")) && r.source == *id), "{id}");
        }
        let gdpr = load_pack("gdpr").unwrap();
        assert_eq!(gdpr.as_of, "2026-09-24");
        assert_eq!(gdpr.policy.rules[0].id, "gdpr/art22-human-review");
        assert_eq!(gdpr.incidents["personal_data_breach"].len(), 2);
        assert!(load_pack("atlantis").unwrap_err().to_string().contains("no governance pack named 'atlantis'"));
    }
}
