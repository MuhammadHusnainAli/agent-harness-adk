//! Trace: whose memory this is.
//!
//! A trace carries the identity a memory belongs to — a user, a session, a
//! tenant — and every backend uses it two ways: to *stamp* records as they are
//! written, and to *scope* reads so one user never sees another's.
//!
//! ```
//! use agent_harness::memory::Trace;
//! let trace = Trace::user("alice").session("s-42");
//! assert_eq!(trace.namespace(), "u=alice");
//! ```
//!
//! `scope` decides what documents like `user.md` are namespaced by. The
//! default, `user`, is what you almost always want: preferences follow the
//! person across sessions. `session` gives each conversation its own; `tenant`
//! shares one file across a whole organisation; `global` is a single shared file.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::memory::base::MemoryRecord;

/// How widely a trace's documents are shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    Global,
    Tenant,
    #[default]
    User,
    Session,
}

impl Scope {
    fn depth(self) -> usize {
        match self {
            Self::Global => 0,
            Self::Tenant => 1,
            Self::User => 2,
            Self::Session => 3,
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "global" => Some(Self::Global),
            "tenant" => Some(Self::Tenant),
            "user" => Some(Self::User),
            "session" => Some(Self::Session),
            _ => None,
        }
    }
}

static UNSAFE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^A-Za-z0-9._-]+").expect("valid"));

/// Who a memory belongs to, and how widely it is shared.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub tenant_id: Option<String>,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub scope: Scope,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
    /// Anything else you want carried along.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

fn set(v: &Option<String>) -> Option<&str> {
    v.as_deref().filter(|s| !s.is_empty())
}

impl Trace {
    pub fn new() -> Self {
        Self::default()
    }

    /// A trace for one user — what a bare user id means everywhere.
    pub fn user(user_id: impl Into<String>) -> Self {
        Self { user_id: Some(user_id.into()), ..Self::default() }
    }

    pub fn session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    pub fn with_agent(mut self, agent: impl Into<String>) -> Self {
        self.agent = Some(agent.into());
        self
    }

    pub fn scoped(mut self, scope: Scope) -> Self {
        self.scope = scope;
        self
    }

    /// The identity axes in the order they nest, narrower to the right.
    fn all_axes(&self) -> [(&'static str, &'static str, Option<&str>); 3] {
        [("tenant_id", "t", set(&self.tenant_id)), ("user_id", "u", set(&self.user_id)), ("session_id", "s", set(&self.session_id))]
    }

    /// True when nothing identifies this trace — a single shared memory.
    pub fn is_empty(&self) -> bool {
        self.all_axes().iter().all(|(_, _, v)| v.is_none())
    }

    /// The set identity axes, narrowed to `scope` (or an explicit depth).
    pub fn axes(&self, depth: Option<usize>) -> Vec<(&'static str, String)> {
        let limit = depth.unwrap_or(self.scope.depth());
        self.all_axes().into_iter().enumerate().filter(|(i, _)| *i < limit).filter_map(|(_, (_, short, v))| v.map(|v| (short, v.to_string()))).collect()
    }

    /// The stable key documents are stored under, e.g. `t=acme/u=alice`.
    pub fn namespace(&self) -> String {
        self.axes(None).iter().map(|(s, v)| format!("{s}={v}")).collect::<Vec<_>>().join("/")
    }

    /// `namespace`, safe for a filename, an object key or a table value.
    pub fn slug(&self) -> String {
        let axes = self.axes(None);
        if axes.is_empty() {
            return "_shared".into();
        }
        axes.iter().map(|(s, v)| format!("{s}-{}", UNSAFE.replace_all(v, "_"))).collect::<Vec<_>>().join("__")
    }

    /// Put this identity on a record before it is stored.
    pub fn stamp(&self, record: &mut MemoryRecord) {
        if record.tenant_id.is_none() {
            record.tenant_id = set(&self.tenant_id).map(str::to_string);
        }
        if record.user_id.is_none() {
            record.user_id = set(&self.user_id).map(str::to_string);
        }
        if record.session_id.is_none() {
            record.session_id = set(&self.session_id).map(str::to_string);
        }
        if let Some(agent) = set(&self.agent)
            && record.source.is_empty()
        {
            record.source = agent.to_string();
        }
    }

    /// The equality filters a query must apply. Empty means unscoped.
    pub fn filters(&self) -> Vec<(&'static str, String)> {
        self.all_axes().into_iter().filter_map(|(field, _, v)| v.map(|v| (field, v.to_string()))).collect()
    }

    /// Does this record belong to this trace? Used by in-memory backends.
    ///
    /// A record with no value on an axis is shared: it was written before the
    /// trace existed, or written deliberately without one, and hiding it would
    /// silently lose memory rather than isolate it.
    pub fn matches(&self, record: &MemoryRecord) -> bool {
        self.filters().iter().all(|(field, value)| match record.axis(field) {
            Some(found) => found == value,
            None => true,
        })
    }

    /// A narrower trace — the same user, a new session, say.
    pub fn child(&self, f: impl FnOnce(&mut Trace)) -> Trace {
        let mut t = self.clone();
        f(&mut t);
        t
    }

    /// The same identity, namespaced more or less widely.
    pub fn at(&self, scope: Scope) -> Trace {
        Trace { scope, ..self.clone() }
    }
}

impl std::fmt::Display for Trace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ns = self.namespace();
        f.write_str(if ns.is_empty() { "(shared)" } else { &ns })
    }
}

impl From<&str> for Trace {
    fn from(user_id: &str) -> Self {
        Trace::user(user_id)
    }
}

impl From<String> for Trace {
    fn from(user_id: String) -> Self {
        Trace::user(user_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespaces_and_matching() {
        let t = Trace::user("al ice").tenant("acme").session("s1");
        assert_eq!(t.namespace(), "t=acme/u=al ice");
        assert_eq!(t.slug(), "t-acme__u-al_ice");
        assert_eq!(t.at(Scope::Session).slug(), "t-acme__u-al_ice__s-s1");
        assert_eq!(t.at(Scope::Global).slug(), "_shared");
        assert_eq!(Trace::new().to_string(), "(shared)");
        let mut r = MemoryRecord::new("user", "fact", "x");
        assert!(t.matches(&r), "unowned records are shared");
        t.stamp(&mut r);
        assert_eq!(r.user_id.as_deref(), Some("al ice"));
        assert!(!Trace::user("bob").matches(&r));
        let json = serde_json::to_value(Trace { extra: [("team".into(), "x".into())].into_iter().collect(), ..Trace::user("a") }).unwrap();
        assert_eq!(json["team"], "x");
        assert_eq!(json["scope"], "user");
    }
}
