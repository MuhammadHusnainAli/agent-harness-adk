//! Who an agent is, who answers for it, and what it may do.
//!
//! ```
//! use agent_harness::governance::AgentIdentity;
//!
//! let identity = AgentIdentity {
//!     owner: "ops@acme.com".into(),
//!     purpose: "customer_support".into(),
//!     tools: Some(vec!["order_*".into(), "issue_refund".into()]),
//!     data: Some(vec!["contact".into(), "financial".into()]),
//!     ..Default::default()
//! };
//! assert!(identity.accountable() && identity.allows_tool("order_status", &[]));
//! ```
//!
//! Delegation never widens authority. When an agent hands work to a sub-agent,
//! the sub-agent acts under the *intersection* of its own identity and every
//! identity above it — a research agent spawned by a support agent cannot reach a
//! tool the support agent could not. That is the least-privilege rule the
//! Singapore agentic framework and OWASP ASI03 (identity & privilege abuse) ask
//! for, enforced rather than recommended.

use std::collections::BTreeSet;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::errors::{Error, Result};
use crate::memory::Trace;

/// `minimal`, `limited`, `high` or `prohibited`.
pub type RiskTier = String;

/// The tiers, loosest first.
pub const RISK_TIERS: [&str; 4] = ["minimal", "limited", "high", "prohibited"];
/// `assistive`, `supervised`, `autonomous`.
pub const AUTONOMY: [&str; 3] = ["assistive", "supervised", "autonomous"];

/// Where a tier sits: 0 for `minimal` up to 3 for `prohibited`.
pub fn risk_rank(tier: &str) -> usize {
    RISK_TIERS.iter().position(|t| *t == tier).unwrap_or(0)
}

/// The stricter of two tiers (the first when they tie, as Python's `max`).
pub(crate) fn stricter<'a>(a: &'a str, b: &'a str) -> &'a str {
    if risk_rank(b) > risk_rank(a) { b } else { a }
}

fn supervised() -> String {
    "supervised".into()
}

fn yes() -> bool {
    true
}

/// The accountable description of one agent.
///
/// Left-out fields are unrestricted, not forbidden — but the evidence report
/// counts an agent with no `owner` or `purpose` as a gap, because every
/// framework asks who is accountable for it and what it is for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentIdentity {
    #[serde(default)]
    pub name: String,
    /// The accountable human or team — an email, a group, an employee id.
    #[serde(default)]
    pub owner: String,
    #[serde(default)]
    pub purpose: String,
    #[serde(default)]
    pub description: String,
    /// `None`: classify from `domains`.
    #[serde(default)]
    pub risk: Option<RiskTier>,
    /// Areas it decides or advises on: "employment", "credit", "health", ...
    /// These drive risk classification (see `governance::risk`).
    #[serde(default)]
    pub domains: Vec<String>,
    /// `assistive`, `supervised` or `autonomous`.
    #[serde(default = "supervised")]
    pub autonomy: String,
    /// Talks to members of the public (drives disclosure duties).
    #[serde(default = "yes")]
    pub user_facing: bool,
    /// Allowed tool name globs or tags (`tag:payments`). `None`: any tool.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Data classes it may handle. `None`: any.
    #[serde(default)]
    pub data: Option<Vec<String>>,
    /// Jurisdictions its model calls may go to. `None`: whatever residency allows.
    #[serde(default)]
    pub regions: Option<Vec<String>>,
    #[serde(default = "yes")]
    pub may_delegate: bool,
    #[serde(default)]
    pub max_delegation_depth: Option<i64>,
    #[serde(default)]
    pub labels: IndexMap<String, String>,
    /// Anything else, kept as given (Python's `extra="allow"`).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Default for AgentIdentity {
    fn default() -> Self {
        Self {
            name: String::new(),
            owner: String::new(),
            purpose: String::new(),
            description: String::new(),
            risk: None,
            domains: vec![],
            autonomy: supervised(),
            user_facing: true,
            tools: None,
            data: None,
            regions: None,
            may_delegate: true,
            max_delegation_depth: None,
            labels: IndexMap::new(),
            extra: Map::new(),
        }
    }
}

impl AgentIdentity {
    /// An identity from what an agent was given — a JSON mapping or nothing —
    /// named after the agent when it names nobody.
    pub fn of(value: Option<&Value>, name: &str) -> Result<Self> {
        let mut identity = match value {
            None | Some(Value::Null) => Self::default(),
            Some(v) => Self::from_value(v)?,
        };
        if identity.name.is_empty() {
            identity.name = name.to_string();
        }
        Ok(identity)
    }

    /// Read one from JSON, checking the tier and autonomy are ones we know.
    pub fn from_value(value: &Value) -> Result<Self> {
        let identity: Self = serde_json::from_value(value.clone()).map_err(|e| Error::config(format!("invalid agent identity: {e}")))?;
        if let Some(r) = &identity.risk
            && !RISK_TIERS.contains(&r.as_str())
        {
            return Err(Error::config(format!("invalid agent identity: risk must be one of {}, not {r:?}", RISK_TIERS.join(", "))));
        }
        if !AUTONOMY.contains(&identity.autonomy.as_str()) {
            return Err(Error::config(format!("invalid agent identity: autonomy must be one of {}, not {:?}", AUTONOMY.join(", "), identity.autonomy)));
        }
        Ok(identity)
    }

    /// The JSON Python's `model_dump()` gives.
    pub fn dump(&self) -> Value {
        serde_json::to_value(self).unwrap_or_default()
    }

    /// Someone answers for it, and it says what it is for.
    pub fn accountable(&self) -> bool {
        !self.owner.is_empty() && !self.purpose.is_empty()
    }

    pub fn allows_tool(&self, tool: &str, tags: &[String]) -> bool {
        let Some(patterns) = &self.tools else { return true };
        patterns.iter().any(|p| match p.strip_prefix("tag:") {
            Some(tag) => tags.iter().any(|t| t == tag),
            None => super::fnmatch(tool, p),
        })
    }
}

/// The person a run acts for — read from the agent's [`Trace`].
///
/// `jurisdiction` decides which residency rules apply to their data;
/// `consents` (from `tags={"consent": "special_category,marketing"}`) let
/// policies ask `'special_category' in principal.consents`. Sub-agents act
/// for the same principal as the run that started them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Principal {
    #[serde(default)]
    pub user: String,
    #[serde(default)]
    pub tenant: String,
    #[serde(default)]
    pub session: String,
    #[serde(default)]
    pub jurisdiction: Option<String>,
    #[serde(default)]
    pub consents: Vec<String>,
    #[serde(default)]
    pub purpose: String,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Principal {
    /// The data-subject key erasure and access work on: `tenant/user`.
    pub fn subject(&self) -> String {
        if self.user.is_empty() { String::new() } else { format!("{}/{}", self.tenant, self.user) }
    }

    pub fn from_trace(trace: Option<&Trace>, home: Option<&str>) -> Self {
        let Some(trace) = trace else {
            return Self { jurisdiction: home.map(str::to_string), ..Default::default() };
        };
        let tag = |k: &str| trace.tags.get(k).cloned().unwrap_or_default();
        let jurisdiction = Some(tag("jurisdiction")).filter(|j| !j.is_empty()).or_else(|| home.map(str::to_string));
        Self {
            user: trace.user_id.clone().unwrap_or_default(),
            tenant: trace.tenant_id.clone().unwrap_or_default(),
            session: trace.session_id.clone().unwrap_or_default(),
            jurisdiction,
            consents: tag("consent").split(',').map(str::trim).filter(|c| !c.is_empty()).map(str::to_string).collect(),
            purpose: tag("purpose"),
            extra: Map::new(),
        }
    }

    /// The mapping policies see as `principal`.
    pub fn context(&self) -> Map<String, Value> {
        let mut out = match serde_json::to_value(self) {
            Ok(Value::Object(m)) => m,
            _ => Map::new(),
        };
        out.insert("subject".into(), Value::String(self.subject()));
        out
    }
}

/// The identities a run is acting under, root first.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DelegationChain {
    pub identities: Vec<AgentIdentity>,
}

impl DelegationChain {
    pub fn new(identities: Vec<AgentIdentity>) -> Self {
        Self { identities }
    }

    /// 0 for a top-level run, 1 for its sub-agent, and so on.
    pub fn depth(&self) -> usize {
        self.identities.len().saturating_sub(1)
    }

    pub fn head(&self) -> Option<&AgentIdentity> {
        self.identities.last()
    }

    pub fn root(&self) -> Option<&AgentIdentity> {
        self.identities.first()
    }

    pub fn names(&self) -> Vec<String> {
        self.identities.iter().map(|i| i.name.clone()).collect()
    }

    pub fn child(&self, identity: AgentIdentity) -> Self {
        let mut identities = self.identities.clone();
        identities.push(identity);
        Self { identities }
    }

    /// May every level use this tool? `(false, why)` names the level that may not.
    pub fn allows_tool(&self, tool: &str, tags: &[String]) -> (bool, String) {
        let last = self.identities.len().saturating_sub(1);
        for (index, identity) in self.identities.iter().enumerate() {
            if !identity.allows_tool(tool, tags) {
                let via = if index == last { "" } else { " (inherited)" };
                let who = if identity.name.is_empty() { "agent" } else { identity.name.as_str() };
                return (false, format!("{who} may not use {tool}{via}"));
            }
        }
        (true, String::new())
    }

    fn intersect(&self, pick: impl Fn(&AgentIdentity) -> Option<&Vec<String>>) -> Option<BTreeSet<String>> {
        let mut allowed: Option<BTreeSet<String>> = None;
        for identity in &self.identities {
            if let Some(mine) = pick(identity) {
                let mine: BTreeSet<String> = mine.iter().cloned().collect();
                allowed = Some(match allowed {
                    None => mine,
                    Some(a) => a.intersection(&mine).cloned().collect(),
                });
            }
        }
        allowed
    }

    /// Intersection of every level's data allowance. `None`: unrestricted.
    pub fn allowed_data(&self) -> Option<BTreeSet<String>> {
        self.intersect(|i| i.data.as_ref())
    }

    pub fn allowed_regions(&self) -> Option<BTreeSet<String>> {
        self.intersect(|i| i.regions.as_ref())
    }

    /// May the head hand work to one more level?
    pub fn may_delegate(&self, max_depth: Option<i64>) -> (bool, String) {
        if let Some(head) = self.head()
            && !head.may_delegate
        {
            return (false, format!("{} is not allowed to delegate", head.name));
        }
        let mut limits: Vec<i64> = self.identities.iter().filter_map(|i| i.max_delegation_depth).collect();
        limits.extend(max_depth);
        if let Some(limit) = limits.iter().min()
            && self.depth() as i64 + 1 > *limit
        {
            return (false, format!("delegation depth {} exceeds the limit of {limit}", self.depth() + 1));
        }
        (true, String::new())
    }

    /// The highest risk anywhere in the chain.
    pub fn risk(&self) -> String {
        let mut tier: Option<&str> = None;
        for t in self.identities.iter().filter_map(|i| i.risk.as_deref()).filter(|t| !t.is_empty()) {
            tier = Some(match tier {
                None => t,
                Some(cur) => stricter(cur, t),
            });
        }
        tier.unwrap_or("minimal").to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn delegation_never_widens_authority() {
        let parent = AgentIdentity {
            name: "support".into(),
            tools: Some(vec!["lookup".into(), "delegate".into(), "tag:payments".into()]),
            data: Some(vec!["contact".into(), "financial".into()]),
            max_delegation_depth: Some(1),
            ..Default::default()
        };
        let child = AgentIdentity { name: "helper".into(), data: Some(vec!["contact".into(), "health".into()]), ..Default::default() };
        let chain = DelegationChain::new(vec![parent]).child(child);
        assert!(chain.allows_tool("lookup", &[]).0);
        assert!(chain.allows_tool("pay", &["payments".into()]).0);
        let (ok, why) = chain.allows_tool("shell", &[]);
        assert!(!ok && why.contains("inherited"), "{why}");
        assert_eq!(chain.allowed_data(), Some(BTreeSet::from(["contact".to_string()])));
        assert!(!chain.may_delegate(None).0);
    }

    #[test]
    fn identities_and_principals_read_as_python_writes_them() {
        let identity = AgentIdentity::of(Some(&json!({"owner": "ops", "team": "x"})), "bot").unwrap();
        assert_eq!(identity.name, "bot");
        let dumped = identity.dump();
        let keys: Vec<&String> = dumped.as_object().unwrap().keys().collect();
        assert_eq!(keys.first().unwrap().as_str(), "name");
        assert_eq!(keys.last().unwrap().as_str(), "team");
        assert!(AgentIdentity::of(Some(&json!({"risk": "huge"})), "x").is_err());

        let trace: Trace = serde_json::from_value(json!({"tenant_id": "acme", "user_id": "al", "tags": {"consent": "a, b", "jurisdiction": "sa"}})).unwrap();
        let principal = Principal::from_trace(Some(&trace), Some("eu"));
        assert_eq!(principal.subject(), "acme/al");
        assert_eq!(principal.consents, vec!["a", "b"]);
        assert_eq!(principal.context()["jurisdiction"], "sa");
        assert_eq!(Principal::from_trace(None, Some("eu")).jurisdiction.as_deref(), Some("eu"));
    }
}
