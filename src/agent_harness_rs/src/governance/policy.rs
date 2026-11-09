//! Policy as code: declarative rules, compiled once, decided in microseconds.
//!
//! ```yaml
//! rules:
//!   - id: refunds-need-a-human
//!     on: tool
//!     match: {tags: [payments]}
//!     when: "args.amount > 100"
//!     effect: require_approval
//!     approvers: 2
//!     timeout: 15m
//! ```
//!
//! Every rule that matches an action contributes; the strictest effect wins
//! (`deny` > `require_approval` > `reroute` > `redact` > `log` > `allow`) and the
//! obligations of all of them are kept — two rules asking for different data
//! classes to be redacted both get their way. That is the "deny-overrides"
//! combination auditors expect, and it means adding a rule can only ever make a
//! policy stricter, never quietly loosen it.
//!
//! A rule whose condition cannot be evaluated takes `defaults.on_error` (`deny`
//! unless you say otherwise). A policy engine that fails open is not a control.
//! (Conditions here never raise — a missing field is `null` and an impossible
//! comparison is `false` — so in practice that default guards against nothing
//! but is kept, and merged, as Python keeps it.)
//!
//! The policy's JSON — and so its sha256, which every decision records — is the
//! one Python writes, so a policy hash means the same thing in both languages.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use indexmap::IndexMap;
use regex::Regex;
use serde_json::{Map, Value, json};

use super::expr::{Ctx, Expr, compile_expr, truthy};
use super::{fnmatch, py_str};
use crate::errors::{Error, Result};
use crate::pyjson::{Dumps, dumps_with};

/// The actions a rule can look at.
pub const ACTIONS: [&str; 7] = ["run", "tool", "tool_result", "egress", "delegate", "memory", "output"];
/// The effects, loosest first.
pub const EFFECTS: [&str; 6] = ["allow", "log", "redact", "reroute", "require_approval", "deny"];

/// How strict an effect is: `allow` 0 … `deny` 5.
pub fn strictness(effect: &str) -> usize {
    EFFECTS.iter().position(|e| *e == effect).unwrap_or(0)
}

static DURATION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)?\s*$").expect("valid"));

/// `90`, `"90s"`, `"15m"`, `"72h"`, `"30d"` → seconds.
pub fn parse_duration(value: Option<&Value>) -> Result<Option<f64>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => Ok(n.as_f64()),
        Some(Value::String(s)) => {
            let caps = DURATION.captures(s).ok_or_else(|| Error::config(format!("not a duration: {} (try '15m', '72h', '30d')", super::py_str_repr(s))))?;
            let n: f64 = caps[1].parse().unwrap_or(0.0);
            let unit = match caps.get(2).map(|m| m.as_str()) {
                Some("ms") => 0.001,
                Some("m") => 60.0,
                Some("h") => 3600.0,
                Some("d") => 86400.0,
                _ => 1.0,
            };
            Ok(Some(n * unit))
        }
        Some(other) => Err(Error::config(format!("not a duration: {other}"))),
    }
}

// ---- reading loose JSON the way pydantic does ---------------------------------

fn bad(what: &str, why: impl std::fmt::Display) -> Error {
    Error::config(format!("invalid {what}: {why}"))
}

fn as_map<'a>(value: &'a Value, what: &str) -> Result<&'a Map<String, Value>> {
    value.as_object().ok_or_else(|| bad(what, "expected a mapping"))
}

fn forbid_extra(map: &Map<String, Value>, allowed: &[&str], what: &str) -> Result<()> {
    match map.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => Err(bad(what, format!("unknown field `{k}`"))),
        None => Ok(()),
    }
}

/// `None` → `[]`, a string → `[it]`, a list → each as `str()`.
fn listify(value: Option<&Value>) -> Vec<String> {
    match value {
        None | Some(Value::Null) => vec![],
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().map(py_str).collect(),
        Some(other) => vec![py_str(other)],
    }
}

fn strings(value: Option<&Value>, what: &str, default: &[&str]) -> Result<Vec<String>> {
    match value {
        None => Ok(default.iter().map(|s| s.to_string()).collect()),
        Some(Value::Array(a)) => a.iter().map(|v| v.as_str().map(str::to_string).ok_or_else(|| bad(what, "expected a list of strings"))).collect(),
        Some(_) => Err(bad(what, "expected a list of strings")),
    }
}

fn string(map: &Map<String, Value>, key: &str, default: &str, what: &str) -> Result<String> {
    match map.get(key) {
        None => Ok(default.to_string()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(_) => Err(bad(what, format!("`{key}` must be a string"))),
    }
}

fn opt_string(map: &Map<String, Value>, key: &str, what: &str) -> Result<Option<String>> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(bad(what, format!("`{key}` must be a string"))),
    }
}

fn boolean(map: &Map<String, Value>, key: &str, default: bool, what: &str) -> Result<bool> {
    match map.get(key) {
        None => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(_) => Err(bad(what, format!("`{key}` must be true or false"))),
    }
}

fn opt_int(map: &Map<String, Value>, key: &str, what: &str) -> Result<Option<i64>> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) if n.is_i64() => Ok(n.as_i64()),
        Some(Value::Number(n)) if n.as_f64().is_some_and(|f| f.fract() == 0.0) => Ok(n.as_f64().map(|f| f as i64)),
        Some(_) => Err(bad(what, format!("`{key}` must be a whole number"))),
    }
}

fn one_of(value: String, allowed: &[&str], key: &str, what: &str) -> Result<String> {
    if allowed.contains(&value.as_str()) { Ok(value) } else { Err(bad(what, format!("`{key}` must be one of {}, not {value:?}", allowed.join(", ")))) }
}

fn transfers_of(value: Option<&Value>, what: &str) -> Result<IndexMap<String, Vec<String>>> {
    match value {
        None | Some(Value::Null) => Ok(IndexMap::new()),
        Some(Value::Object(m)) => m.iter().map(|(k, v)| Ok((k.clone(), strings(Some(v), what, &[])?))).collect(),
        Some(_) => Err(bad(what, "expected a mapping of lists")),
    }
}

fn dedup(items: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    items.into_iter().filter(|i| seen.insert(i.clone())).collect()
}

// ---- the models --------------------------------------------------------------

const MATCH_FIELDS: [&str; 10] = ["tool", "tags", "agent", "provider", "model", "region", "data", "risk", "purpose", "jurisdiction"];

/// Cheap pre-filters, checked before any condition. Globs; any-of per field.
///
/// An empty field matches everything. `data` matches when any of the listed
/// data classes is present in what is being decided about.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Match {
    pub tool: Vec<String>,
    pub tags: Vec<String>,
    pub agent: Vec<String>,
    pub provider: Vec<String>,
    pub model: Vec<String>,
    pub region: Vec<String>,
    pub data: Vec<String>,
    pub risk: Vec<String>,
    pub purpose: Vec<String>,
    pub jurisdiction: Vec<String>,
}

fn field<'a>(ctx: &'a Map<String, Value>, key: &str) -> Option<&'a Map<String, Value>> {
    ctx.get(key).and_then(Value::as_object)
}

fn one(patterns: &[String], value: Option<&Value>) -> bool {
    if patterns.is_empty() {
        return true;
    }
    match value {
        None | Some(Value::Null) => false,
        Some(v) => {
            let v = py_str(v);
            patterns.iter().any(|p| fnmatch(&v, p))
        }
    }
}

impl Match {
    pub fn from_value(value: &Value) -> Result<Self> {
        let m = as_map(value, "match")?;
        forbid_extra(m, &MATCH_FIELDS, "match")?;
        let f = |k: &str| listify(m.get(k));
        Ok(Self {
            tool: f("tool"),
            tags: f("tags"),
            agent: f("agent"),
            provider: f("provider"),
            model: f("model"),
            region: f("region"),
            data: f("data"),
            risk: f("risk"),
            purpose: f("purpose"),
            jurisdiction: f("jurisdiction"),
        })
    }

    pub fn dump(&self) -> Value {
        json!({
            "tool": self.tool, "tags": self.tags, "agent": self.agent, "provider": self.provider,
            "model": self.model, "region": self.region, "data": self.data, "risk": self.risk,
            "purpose": self.purpose, "jurisdiction": self.jurisdiction,
        })
    }

    pub fn empty(&self) -> bool {
        [&self.tool, &self.tags, &self.agent, &self.provider, &self.model, &self.region, &self.data, &self.risk, &self.purpose, &self.jurisdiction]
            .iter()
            .all(|f| f.is_empty())
    }

    pub fn matches(&self, ctx: &Ctx) -> bool {
        let empty = Map::new();
        let tool = field(ctx, "tool").unwrap_or(&empty);
        if !one(&self.tool, tool.get("name")) {
            return false;
        }
        if !self.tags.is_empty() {
            let have: BTreeSet<String> = tool.get("tags").and_then(Value::as_array).map(|a| a.iter().map(py_str).collect()).unwrap_or_default();
            if !self.tags.iter().any(|t| have.contains(t)) {
                return false;
            }
        }
        let agent = field(ctx, "agent").unwrap_or(&empty);
        if !one(&self.agent, agent.get("name")) || !one(&self.risk, agent.get("risk")) {
            return false;
        }
        if !one(&self.purpose, ctx.get("purpose")) {
            return false;
        }
        let provider = field(ctx, "provider").unwrap_or(&empty);
        if !one(&self.provider, provider.get("name")) || !one(&self.model, provider.get("model")) || !one(&self.region, provider.get("region")) {
            return false;
        }
        if !one(&self.jurisdiction, field(ctx, "principal").and_then(|p| p.get("jurisdiction"))) {
            return false;
        }
        if !self.data.is_empty() {
            let present: BTreeSet<String> =
                field(ctx, "data").and_then(|d| d.get("classes")).and_then(Value::as_array).map(|a| a.iter().map(py_str).collect()).unwrap_or_default();
            if !self.data.iter().any(|d| present.contains(d)) {
                return false;
            }
        }
        true
    }
}

const RULE_FIELDS: [&str; 12] = ["id", "on", "match", "when", "effect", "reason", "approvers", "timeout", "redact", "redact_mode", "controls", "source"];

/// One rule. `on` says which actions it looks at.
#[derive(Debug, Clone)]
pub struct PolicyRule {
    pub id: String,
    pub on: Vec<String>,
    pub r#match: Match,
    pub when: Option<String>,
    pub effect: String,
    pub reason: String,
    /// require_approval: distinct humans needed, and how long to wait for them.
    pub approvers: i64,
    pub timeout: Option<f64>,
    /// redact: which data classes to take out, and how (`mask` / `pseudonymize`).
    pub redact: Vec<String>,
    pub redact_mode: String,
    /// Which catalog controls this rule implements, and which pack it came from.
    pub controls: Vec<String>,
    pub source: String,
    test: Option<Arc<Expr>>,
}

impl PartialEq for PolicyRule {
    fn eq(&self, other: &Self) -> bool {
        self.dump() == other.dump()
    }
}

impl PolicyRule {
    pub fn from_value(value: &Value) -> Result<Self> {
        let m = as_map(value, "policy rule")?;
        // YAML 1.1 reads a bare `on:` key as the boolean true; JSON cannot, and
        // serde_yaml reads YAML 1.2, so `on` arrives as written.
        forbid_extra(m, &RULE_FIELDS, "policy rule")?;
        let what = "policy rule";
        let id = match m.get("id") {
            Some(Value::String(s)) => s.clone(),
            _ => return Err(bad(what, "`id` is required")),
        };
        let what_id = format!("policy rule {id}");
        let on = match m.get("on") {
            None => vec!["tool".to_string()],
            v => listify(v),
        };
        for action in &on {
            one_of(action.clone(), &ACTIONS, "on", &what_id)?;
        }
        let r#match = match m.get("match") {
            None | Some(Value::Null) => Match::default(),
            Some(v) => Match::from_value(v)?,
        };
        let when = opt_string(m, "when", &what_id)?;
        let approvers = opt_int(m, "approvers", &what_id)?.unwrap_or(1);
        if approvers < 1 {
            return Err(bad(&what_id, "`approvers` must be at least 1"));
        }
        let test = match when.as_deref() {
            Some(w) if !w.is_empty() => Some(compile_expr(w)?),
            _ => None,
        };
        Ok(Self {
            on,
            r#match,
            effect: one_of(string(m, "effect", "deny", &what_id)?, &EFFECTS, "effect", &what_id)?,
            reason: string(m, "reason", "", &what_id)?,
            approvers,
            timeout: parse_duration(m.get("timeout"))?,
            redact: strings(m.get("redact"), &what_id, &[])?,
            redact_mode: one_of(string(m, "redact_mode", "mask", &what_id)?, &["mask", "pseudonymize"], "redact_mode", &what_id)?,
            controls: strings(m.get("controls"), &what_id, &[])?,
            source: string(m, "source", "", &what_id)?,
            when,
            id,
            test,
        })
    }

    /// Python's `model_dump(mode="json")`.
    pub fn dump(&self) -> Value {
        json!({
            "id": self.id, "on": self.on, "match": self.r#match.dump(), "when": self.when,
            "effect": self.effect, "reason": self.reason, "approvers": self.approvers,
            "timeout": self.timeout, "redact": self.redact, "redact_mode": self.redact_mode,
            "controls": self.controls, "source": self.source,
        })
    }

    /// True when this rule applies.
    pub fn evaluate(&self, ctx: &Ctx) -> bool {
        self.r#match.matches(ctx) && self.test.as_ref().is_none_or(|t| truthy(&t.eval(ctx)))
    }
}

/// What happens when no rule matched, and when a rule cannot be evaluated.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyDefaults {
    pub effect: String,
    pub on_error: String,
}

impl Default for PolicyDefaults {
    fn default() -> Self {
        Self { effect: "allow".into(), on_error: "deny".into() }
    }
}

impl PolicyDefaults {
    pub fn from_value(value: &Value) -> Result<Self> {
        let m = as_map(value, "defaults")?;
        forbid_extra(m, &["effect", "on_error"], "defaults")?;
        Ok(Self {
            effect: one_of(string(m, "effect", "allow", "defaults")?, &EFFECTS, "effect", "defaults")?,
            on_error: one_of(string(m, "on_error", "deny", "defaults")?, &EFFECTS, "on_error", "defaults")?,
        })
    }

    pub fn dump(&self) -> Value {
        json!({"effect": self.effect, "on_error": self.on_error})
    }
}

/// Where data about people from each jurisdiction may be sent.
///
/// `transfers` maps a data subject's jurisdiction to the destinations their
/// personal data may reach; `"*"` means anywhere. The subject's jurisdiction
/// comes from the run's principal, or `home` when none is set. A jurisdiction
/// no pack speaks for is unrestricted.
///
/// The short form also works for a single-jurisdiction deployment:
///
/// ```yaml
/// residency: {home: eu, allow_transfer_to: [uk, ch], local_providers_are: on_prem}
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ResidencyConfig {
    pub home: Option<String>,
    pub transfers: IndexMap<String, Vec<String>>,
    /// Which data triggers the restriction. Non-personal data goes anywhere.
    pub applies_to: Vec<String>,
    /// A self-hosted model never leaves the building.
    pub on_prem_always_allowed: bool,
    /// A destination whose region cannot be resolved: `deny` or `allow`.
    pub unknown: String,
}

impl Default for ResidencyConfig {
    fn default() -> Self {
        Self { home: None, transfers: IndexMap::new(), applies_to: vec!["personal".into()], on_prem_always_allowed: true, unknown: "deny".into() }
    }
}

impl ResidencyConfig {
    pub fn from_value(value: &Value) -> Result<Self> {
        let what = "residency";
        let mut m = as_map(value, what)?.clone();
        if let Some(allowed) = m.remove("allow_transfer_to") {
            let allowed = listify(Some(&allowed));
            let home = m.get("home").and_then(Value::as_str).filter(|h| !h.is_empty()).map(str::to_string);
            let Some(home) = home else {
                return Err(bad(what, "`allow_transfer_to` needs `home` — whose data it limits"));
            };
            let mut transfers = m.get("transfers").and_then(Value::as_object).cloned().unwrap_or_default();
            transfers.insert(home, json!(allowed));
            m.insert("transfers".into(), Value::Object(transfers));
        }
        if let Some(value) = m.remove("local_providers_are") {
            let value = value.as_str().unwrap_or("");
            if !matches!(value, "on_prem" | "restricted") {
                return Err(bad(what, "`local_providers_are` is `on_prem` or `restricted`"));
            }
            m.insert("on_prem_always_allowed".into(), json!(value == "on_prem"));
        }
        forbid_extra(&m, &["home", "transfers", "applies_to", "on_prem_always_allowed", "unknown"], what)?;
        Ok(Self {
            home: opt_string(&m, "home", what)?,
            transfers: transfers_of(m.get("transfers"), what)?,
            applies_to: strings(m.get("applies_to"), what, &["personal"])?,
            on_prem_always_allowed: boolean(&m, "on_prem_always_allowed", true, what)?,
            unknown: one_of(string(&m, "unknown", "deny", what)?, &["deny", "allow"], "unknown", what)?,
        })
    }

    pub fn dump(&self) -> Value {
        json!({
            "home": self.home, "transfers": self.transfers, "applies_to": self.applies_to,
            "on_prem_always_allowed": self.on_prem_always_allowed, "unknown": self.unknown,
        })
    }

    /// Combine two packs' views. Where both restrict a jurisdiction, both must agree.
    pub fn merge(&self, other: Option<&ResidencyConfig>) -> ResidencyConfig {
        let Some(other) = other else { return self.clone() };
        let mut transfers = self.transfers.clone();
        for (origin, allowed) in &other.transfers {
            match transfers.get_mut(origin) {
                None => {
                    transfers.insert(origin.clone(), allowed.clone());
                }
                Some(mine) if mine.iter().any(|d| d == "*") => *mine = allowed.clone(),
                Some(mine) if !allowed.iter().any(|d| d == "*") => mine.retain(|d| allowed.contains(d)),
                Some(_) => {}
            }
        }
        let applies: BTreeSet<String> = self.applies_to.iter().chain(&other.applies_to).cloned().collect();
        ResidencyConfig {
            home: self.home.clone().or_else(|| other.home.clone()),
            transfers,
            applies_to: applies.into_iter().collect(),
            on_prem_always_allowed: self.on_prem_always_allowed && other.on_prem_always_allowed,
            unknown: if self.unknown == "deny" || other.unknown == "deny" { "deny".into() } else { "allow".into() },
        }
    }

    /// Where `origin`'s personal data may go. `None` means unrestricted.
    pub fn destinations(&self, origin: Option<&str>) -> Option<Vec<String>> {
        let allowed = self.transfers.get(origin?)?;
        if allowed.iter().any(|d| d == "*") {
            return None;
        }
        let mut out: BTreeSet<String> = allowed.iter().cloned().collect();
        out.insert(origin?.to_string());
        Some(out.into_iter().collect())
    }
}

/// How people are told they are dealing with AI, and how output is marked.
#[derive(Debug, Clone, PartialEq)]
pub struct TransparencyConfig {
    /// Attach a disclosure to every top-level result.
    pub disclose: bool,
    /// Also put a visible label into the output text itself...
    pub label_output: bool,
    /// ...for people in these jurisdictions (`"*"`: everyone). A pack that
    /// requires visible labels in China does not label answers to Germans.
    pub label_for: Vec<String>,
    /// Attach a machine-readable provenance manifest.
    pub provenance: bool,
    pub languages: Vec<String>,
}

impl Default for TransparencyConfig {
    fn default() -> Self {
        Self { disclose: false, label_output: false, label_for: vec!["*".into()], provenance: false, languages: vec!["en".into()] }
    }
}

impl TransparencyConfig {
    pub fn from_value(value: &Value) -> Result<Self> {
        let what = "transparency";
        let m = as_map(value, what)?;
        forbid_extra(m, &["disclose", "label_output", "label_for", "provenance", "languages"], what)?;
        Ok(Self {
            disclose: boolean(m, "disclose", false, what)?,
            label_output: boolean(m, "label_output", false, what)?,
            label_for: strings(m.get("label_for"), what, &["*"])?,
            provenance: boolean(m, "provenance", false, what)?,
            languages: strings(m.get("languages"), what, &["en"])?,
        })
    }

    pub fn dump(&self) -> Value {
        json!({
            "disclose": self.disclose, "label_output": self.label_output, "label_for": self.label_for,
            "provenance": self.provenance, "languages": self.languages,
        })
    }

    pub fn merge(&self, other: Option<&TransparencyConfig>) -> TransparencyConfig {
        let Some(other) = other else { return self.clone() };
        let label_for: BTreeSet<String> = [self, other].iter().filter(|c| c.label_output).flat_map(|c| c.label_for.iter().cloned()).collect();
        let label_for: Vec<String> = if label_for.is_empty() || label_for.contains("*") { vec!["*".into()] } else { label_for.into_iter().collect() };
        TransparencyConfig {
            disclose: self.disclose || other.disclose,
            label_output: self.label_output || other.label_output,
            label_for,
            provenance: self.provenance || other.provenance,
            languages: dedup(self.languages.iter().chain(&other.languages).cloned()),
        }
    }
}

const POLICY_FIELDS: [&str; 14] = [
    "name",
    "version",
    "description",
    "defaults",
    "rules",
    "purposes",
    "data_classes",
    "residency",
    "transparency",
    "log_retention_days",
    "data_retention_days",
    "max_delegation_depth",
    "tool_drift",
    "packs",
];

/// A complete, versioned policy. Packs are merged into one of these.
#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    pub name: String,
    pub version: String,
    pub description: String,
    pub defaults: PolicyDefaults,
    pub rules: Vec<PolicyRule>,
    /// purpose → the data classes a run with that purpose may handle.
    pub purposes: IndexMap<String, Vec<String>>,
    /// Your own data-class names (found by `DataClassifier` extras or declared
    /// by `data:<class>` tool tags). Anything else a policy names is a typo, and
    /// a typo in a deny rule would fail open, so it is refused.
    pub data_classes: Vec<String>,
    pub residency: Option<ResidencyConfig>,
    pub transparency: TransparencyConfig,
    /// Minimum days decision records must be kept (the longest any pack asks for).
    pub log_retention_days: Option<i64>,
    /// Maximum days conversational data may be kept (the shortest any pack allows).
    pub data_retention_days: Option<i64>,
    /// Highest delegation depth any agent may reach.
    pub max_delegation_depth: Option<i64>,
    /// A tool whose schema changed since it was pinned: `deny`, `log`, or `allow`.
    pub tool_drift: String,
    /// Where the rules came from, for the evidence report.
    pub packs: Vec<String>,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            name: "policy".into(),
            version: "1".into(),
            description: String::new(),
            defaults: PolicyDefaults::default(),
            rules: vec![],
            purposes: IndexMap::new(),
            data_classes: vec![],
            residency: None,
            transparency: TransparencyConfig::default(),
            log_retention_days: None,
            data_retention_days: None,
            max_delegation_depth: None,
            tool_drift: "log".into(),
            packs: vec![],
        }
    }
}

impl Policy {
    /// A policy from a JSON mapping (what a YAML file reads as).
    pub fn from_value(value: &Value) -> Result<Self> {
        let what = "policy";
        let mut m = as_map(value, what)?.clone();
        // `data: {purposes: ..., classes: ...}` reads naturally in a file.
        if let Some(Value::Object(section)) = m.get("data").cloned() {
            m.remove("data");
            let mut section = section;
            if let Some(purposes) = section.remove("purposes") {
                let mut merged = purposes.as_object().cloned().unwrap_or_default();
                if let Some(Value::Object(own)) = m.get("purposes") {
                    merged.extend(own.clone());
                }
                m.insert("purposes".into(), Value::Object(merged));
            }
            if let Some(classes) = section.remove("classes") {
                let mut all = m.get("data_classes").and_then(Value::as_array).cloned().unwrap_or_default();
                all.extend(classes.as_array().cloned().unwrap_or_default());
                m.insert("data_classes".into(), Value::Array(all));
            }
            if !section.is_empty() {
                return Err(bad(what, format!("unknown keys under `data`: {}", section.keys().cloned().collect::<Vec<_>>().join(", "))));
            }
        }
        forbid_extra(&m, &POLICY_FIELDS, what)?;
        let version = match m.get("version") {
            None => "1".to_string(),
            Some(v) => py_str(v), // `version: 3` in YAML is an int
        };
        let rules = match m.get("rules") {
            None | Some(Value::Null) => vec![],
            Some(Value::Array(a)) => a.iter().map(PolicyRule::from_value).collect::<Result<Vec<_>>>()?,
            Some(_) => return Err(bad(what, "`rules` must be a list")),
        };
        Ok(Self {
            name: string(&m, "name", "policy", what)?,
            version,
            description: string(&m, "description", "", what)?,
            defaults: m.get("defaults").map(PolicyDefaults::from_value).transpose()?.unwrap_or_default(),
            rules,
            purposes: transfers_of(m.get("purposes"), "policy purposes")?,
            data_classes: strings(m.get("data_classes"), what, &[])?,
            residency: match m.get("residency") {
                None | Some(Value::Null) => None,
                Some(v) => Some(ResidencyConfig::from_value(v)?),
            },
            transparency: m.get("transparency").map(TransparencyConfig::from_value).transpose()?.unwrap_or_default(),
            log_retention_days: opt_int(&m, "log_retention_days", what)?,
            data_retention_days: opt_int(&m, "data_retention_days", what)?,
            max_delegation_depth: opt_int(&m, "max_delegation_depth", what)?,
            tool_drift: one_of(string(&m, "tool_drift", "log", what)?, &["deny", "log", "allow"], "tool_drift", what)?,
            packs: strings(m.get("packs"), what, &[])?,
        })
    }

    /// A path to YAML/JSON, or YAML text.
    pub fn load(source: &str) -> Result<Self> {
        let mut text = source.to_string();
        let lower = source.to_lowercase();
        if !source.contains('\n') && [".yaml", ".yml", ".json"].iter().any(|s| lower.ends_with(s)) {
            let path = Path::new(source);
            if !path.exists() {
                return Err(Error::config(format!("policy file not found: {source}")));
            }
            text = std::fs::read_to_string(path).map_err(|e| Error::config(format!("policy file {source}: {e}")))?;
        }
        let data: Value = serde_yaml::from_str(&text).map_err(|e| Error::config(format!("policy is not valid YAML: {e}")))?;
        match data {
            Value::Null => Ok(Self::default()),
            Value::Object(_) => Self::from_value(&data),
            _ => Err(Error::config("a policy must be a mapping at the top level")),
        }
    }

    /// Every data-class name this policy refers to.
    pub fn class_names(&self) -> BTreeSet<String> {
        let mut names: BTreeSet<String> = self.purposes.values().flatten().cloned().collect();
        for rule in &self.rules {
            names.extend(rule.r#match.data.iter().cloned());
            names.extend(rule.redact.iter().cloned());
        }
        if let Some(r) = &self.residency {
            names.extend(r.applies_to.iter().cloned());
        }
        names
    }

    /// Python's `model_dump(mode="json")`.
    pub fn dump(&self) -> Value {
        json!({
            "name": self.name, "version": self.version, "description": self.description,
            "defaults": self.defaults.dump(),
            "rules": self.rules.iter().map(PolicyRule::dump).collect::<Vec<_>>(),
            "purposes": self.purposes, "data_classes": self.data_classes,
            "residency": self.residency.as_ref().map(ResidencyConfig::dump),
            "transparency": self.transparency.dump(),
            "log_retention_days": self.log_retention_days, "data_retention_days": self.data_retention_days,
            "max_delegation_depth": self.max_delegation_depth, "tool_drift": self.tool_drift,
            "packs": self.packs,
        })
    }

    /// sha256 over the canonical JSON — every decision records it.
    pub fn hash(&self) -> String {
        super::sha256_hex(dumps_with(&self.dump(), Dumps::canonical()).as_bytes())
    }

    /// This policy plus others. Every combination only tightens.
    pub fn merge(&self, others: &[&Policy]) -> Policy {
        let mut merged = self.clone();
        for other in others {
            merged.rules.extend(other.rules.iter().cloned());
            for (purpose, classes) in &other.purposes {
                match merged.purposes.get_mut(purpose) {
                    Some(mine) => mine.retain(|c| classes.contains(c)),
                    None => {
                        merged.purposes.insert(purpose.clone(), classes.clone());
                    }
                }
            }
            merged.residency = match &merged.residency {
                None => other.residency.clone(),
                Some(r) => Some(r.merge(other.residency.as_ref())),
            };
            merged.transparency = merged.transparency.merge(Some(&other.transparency));
            merged.log_retention_days = pick(merged.log_retention_days, other.log_retention_days, true);
            merged.data_retention_days = pick(merged.data_retention_days, other.data_retention_days, false);
            merged.max_delegation_depth = pick(merged.max_delegation_depth, other.max_delegation_depth, false);
            let order = ["allow", "log", "deny"];
            let rank = |d: &str| order.iter().position(|o| *o == d).unwrap_or(0);
            if rank(&other.tool_drift) > rank(&merged.tool_drift) {
                merged.tool_drift = other.tool_drift.clone();
            }
            if strictness(&other.defaults.effect) > strictness(&merged.defaults.effect) {
                merged.defaults.effect = other.defaults.effect.clone();
            }
            if strictness(&other.defaults.on_error) > strictness(&merged.defaults.on_error) {
                merged.defaults.on_error = other.defaults.on_error.clone();
            }
            merged.packs = dedup(merged.packs.iter().chain(&other.packs).cloned());
            merged.data_classes = dedup(merged.data_classes.iter().chain(&other.data_classes).cloned());
        }
        merged
    }
}

fn pick(a: Option<i64>, b: Option<i64>, longest: bool) -> Option<i64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if longest { a.max(b) } else { a.min(b) }),
        (a, b) => a.or(b),
    }
}

/// What the engine decided, why, and under which policy version.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub action: String,
    pub target: String,
    pub effect: String,
    pub rules: Vec<String>,
    pub reasons: Vec<String>,
    pub controls: Vec<String>,
    /// The policy hash, shortened.
    pub policy: String,
    pub approvers: i64,
    pub timeout: Option<f64>,
    pub redact: Vec<String>,
    pub redact_mode: String,
    /// False in monitor mode.
    pub enforced: bool,
    pub duration_us: f64,
}

impl Decision {
    pub fn new(action: &str, target: &str) -> Self {
        Self {
            action: action.into(),
            target: target.into(),
            effect: "allow".into(),
            rules: vec![],
            reasons: vec![],
            controls: vec![],
            policy: String::new(),
            approvers: 1,
            timeout: None,
            redact: vec![],
            redact_mode: "mask".into(),
            enforced: true,
            duration_us: 0.0,
        }
    }

    pub fn allowed(&self) -> bool {
        matches!(self.effect.as_str(), "allow" | "log" | "redact")
    }

    pub fn reason(&self) -> String {
        if self.reasons.is_empty() { self.effect.clone() } else { self.reasons.join("; ") }
    }

    /// Fold a built-in check (residency, delegation, drift) into the decision.
    pub fn tighten(&mut self, effect: &str, reason: &str, rule: &str, controls: &[&str]) -> &mut Self {
        if strictness(effect) > strictness(&self.effect) {
            self.effect = effect.to_string();
        }
        if !rule.is_empty() {
            self.rules.push(rule.to_string());
        }
        if !reason.is_empty() {
            self.reasons.push(reason.to_string());
        }
        self.controls = dedup(self.controls.iter().cloned().chain(controls.iter().map(|c| c.to_string())));
        self
    }
}

/// A compiled policy: rules indexed by action, evaluated in order.
#[derive(Debug, Clone)]
pub struct PolicyEngine {
    pub policy: Policy,
    pub hash: String,
    by_action: HashMap<String, Vec<usize>>,
}

impl PolicyEngine {
    pub fn new(policy: Policy) -> Self {
        let hash = policy.hash();
        let mut by_action: HashMap<String, Vec<usize>> = ACTIONS.iter().map(|a| (a.to_string(), vec![])).collect();
        for (index, rule) in policy.rules.iter().enumerate() {
            for action in &rule.on {
                by_action.entry(action.clone()).or_default().push(index);
            }
        }
        Self { policy, hash, by_action }
    }

    pub fn rules_for(&self, action: &str) -> Vec<&PolicyRule> {
        self.by_action.get(action).map(|ix| ix.iter().map(|&i| &self.policy.rules[i]).collect()).unwrap_or_default()
    }

    pub fn decide(&self, action: &str, ctx: &Ctx, target: &str) -> Decision {
        let started = Instant::now();
        // Starts at the loosest effect so a matching `allow` rule can carve an
        // exception out of a deny-by-default policy; the default applies only
        // when nothing matched.
        let mut decision = Decision::new(action, target);
        decision.policy = self.hash[..16].to_string();
        let mut matched = false;
        for rule in self.rules_for(action) {
            if !rule.evaluate(ctx) {
                continue;
            }
            // A `log` rule observes; it does not stand in for the default.
            matched = matched || rule.effect != "log";
            let reason = if rule.reason.is_empty() { format!("rule {}", rule.id) } else { rule.reason.clone() };
            let controls: Vec<&str> = rule.controls.iter().map(String::as_str).collect();
            decision.tighten(&rule.effect, &reason, &rule.id, &controls);
            if rule.effect == "require_approval" {
                decision.approvers = decision.approvers.max(rule.approvers);
                if let Some(t) = rule.timeout {
                    decision.timeout = Some(decision.timeout.map_or(t, |d| d.min(t)));
                }
            }
            if rule.effect == "redact" || !rule.redact.is_empty() {
                decision.redact = dedup(decision.redact.iter().chain(&rule.redact).cloned());
                if rule.redact_mode == "pseudonymize" {
                    decision.redact_mode = "pseudonymize".into();
                }
            }
        }
        if !matched {
            let default = &self.policy.defaults.effect;
            if strictness(default) > strictness(&decision.effect) {
                decision.effect = default.clone();
            }
            if default != "allow" {
                decision.reasons.push(format!("policy default is {}", decision.effect));
            }
        }
        decision.duration_us = started.elapsed().as_secs_f64() * 1e6;
        decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine(policy: Value) -> PolicyEngine {
        PolicyEngine::new(Policy::from_value(&policy).unwrap())
    }

    fn ctx(v: Value) -> Ctx {
        v.as_object().cloned().unwrap()
    }

    #[test]
    fn strictest_matching_rule_wins_and_obligations_combine() {
        let e = engine(json!({"rules": [
            {"id": "a", "on": "tool", "effect": "redact", "redact": ["contact"]},
            {"id": "b", "on": "tool", "effect": "redact", "redact": ["financial"]},
            {"id": "c", "on": "tool", "match": {"tool": "pay*"}, "effect": "require_approval", "approvers": 2, "timeout": "15m"},
        ]}));
        let d = e.decide("tool", &ctx(json!({"tool": {"name": "pay"}})), "pay");
        assert_eq!(d.effect, "require_approval");
        assert_eq!((d.approvers, d.timeout), (2, Some(900.0)));
        assert_eq!(d.redact, vec!["contact", "financial"]);
        assert_eq!(e.decide("tool", &ctx(json!({"tool": {"name": "lookup"}})), "").effect, "redact");
    }

    #[test]
    fn an_allow_rule_carves_an_exception_from_deny_by_default() {
        let e = engine(json!({"defaults": {"effect": "deny"}, "rules": [
            {"id": "ok", "on": "tool", "match": {"tool": "lookup"}, "effect": "allow"},
            {"id": "watch", "on": "tool", "effect": "log"},
        ]}));
        let allowed = e.decide("tool", &ctx(json!({"tool": {"name": "lookup"}})), "");
        assert!(allowed.allowed() && allowed.effect == "log");
        assert_eq!(e.decide("tool", &ctx(json!({"tool": {"name": "pay"}})), "").effect, "deny");
        // int('abc') is null, null > 1 is false: no match, allowed.
        let e = engine(json!({"rules": [{"id": "x", "on": "tool", "when": "int(args.n) > 1", "effect": "allow"}]}));
        assert_eq!(e.decide("tool", &ctx(json!({"args": {"n": "abc"}})), "").effect, "allow");
    }

    #[test]
    fn yaml_on_key_and_policy_hash() {
        let policy = Policy::load("rules:\n  - id: r\n    on: egress\n    effect: deny\n").unwrap();
        assert_eq!(policy.rules[0].on, vec!["egress"]);
        let other = Policy::load("rules:\n  - id: r\n    on: egress\n    effect: log\n").unwrap();
        assert!(policy.hash() != other.hash() && policy.hash().len() == 64);
        assert_eq!(Policy::from_value(&policy.dump()).unwrap().hash(), policy.hash());
    }

    #[test]
    fn hashes_match_python() {
        // sha256(json.dumps(Policy().model_dump(mode="json"), sort_keys=True, separators=(",", ":")))
        assert_eq!(Policy::default().hash(), "9ee2bdfb04034e23944c2633b307f8939d56d56d576ae08cbf7146378f02a49d");
        let policy = Policy::from_value(&json!({
            "rules": [{"id": "c", "on": "tool", "match": {"tool": "pay*"}, "when": "args.amount > 100",
                       "effect": "require_approval", "approvers": 2, "timeout": "15m"}],
            "residency": {"home": "eu", "allow_transfer_to": ["uk", "ch"]},
            "purposes": {"support": ["contact"]}, "log_retention_days": 183,
        }))
        .unwrap();
        assert_eq!(policy.hash(), "031f3279e83bd1c83975a036e48b7a7aa867b72a6630bd27791faed2e0222abd");
    }

    #[test]
    fn merging_packs_only_tightens() {
        let a = Policy::from_value(&json!({"residency": {"transfers": {"eu": ["eu", "uk", "us"]}}, "log_retention_days": 90, "data_retention_days": 365, "purposes": {"support": ["contact", "financial"]}})).unwrap();
        let b = Policy::from_value(&json!({"residency": {"transfers": {"eu": ["eu", "uk"], "sa": ["sa"]}}, "log_retention_days": 183, "data_retention_days": 30, "purposes": {"support": ["contact"]}})).unwrap();
        let merged = a.merge(&[&b]);
        assert_eq!(merged.residency.as_ref().unwrap().dump()["transfers"], json!({"eu": ["eu", "uk"], "sa": ["sa"]}));
        assert_eq!((merged.log_retention_days, merged.data_retention_days), (Some(183), Some(30)));
        assert_eq!(merged.purposes["support"], vec!["contact"]);
        let strict = Policy::from_value(&json!({"defaults": {"on_error": "log"}}))
            .unwrap()
            .merge(&[&Policy::from_value(&json!({"defaults": {"on_error": "deny"}})).unwrap()]);
        assert_eq!(strict.defaults.on_error, "deny");
    }

    #[test]
    fn the_plans_own_policy_syntax_works() {
        let policy = Policy::load(
            "version: 3\ndefaults: {effect: allow, on_error: deny}\ndata:\n  purposes:\n    customer_support: [contact, order, financial]\n  classes: [order]\nresidency:\n  home: eu\n  allow_transfer_to: [uk, ch]\n  local_providers_are: on_prem\nrules:\n  - {id: no-health-data-out, on: egress, when: \"data.special_category\", effect: redact}\n  - {id: max-delegation, on: delegate, when: \"delegation.depth > 2\", effect: deny}\n",
        )
        .unwrap();
        assert_eq!(policy.version, "3");
        assert_eq!(policy.purposes["customer_support"], vec!["contact", "order", "financial"]);
        let residency = policy.residency.as_ref().unwrap();
        assert_eq!(residency.dump()["transfers"], json!({"eu": ["uk", "ch"]}));
        assert_eq!(residency.destinations(Some("eu")), Some(vec!["ch".to_string(), "eu".into(), "uk".into()]));
        assert!(Policy::from_value(&json!({"rules": [{"id": "x", "bogus": 1}]})).is_err());
        assert!(Policy::from_value(&json!({"rules": [{"id": "x", "when": "__import__('os')"}]})).is_err());
        assert!(parse_duration(Some(&json!("soon"))).is_err());
        assert_eq!(parse_duration(Some(&json!("72h"))).unwrap(), Some(259_200.0));
    }
}
