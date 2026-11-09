//! Permission and policy gate: allow, ask, or deny — per action.
//!
//! ```
//! use agent_harness::runtime::permissions::PolicyGate;
//! use agent_harness::Permission;
//!
//! let gate = PolicyGate::new(Permission::Allow).deny(["shell", "fs_write"]).ask(["http_*"]);
//! assert_eq!(gate.decide("shell", &Default::default()).0, Permission::Deny);
//! ```
//!
//! `ask` needs an approver. Without one, asking is denied — a harness should
//! never silently escalate itself.

use std::fmt;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use serde_json::{Map, Value};

use crate::errors::{Error, Result};
use crate::tools::{Permission, fnmatch};

pub type Args = Map<String, Value>;
/// Decides an `ask`: `(tool, args, reason)` → approved?
pub type Approver = Arc<dyn Fn(&str, &Args, &str) -> BoxFuture<'static, bool> + Send + Sync>;
/// Records every decision: `(tool, "allow" | "deny", args)`.
pub type Audit = Arc<dyn Fn(&str, &str, &Args) + Send + Sync>;
type When = Arc<dyn Fn(&Args) -> bool + Send + Sync>;

/// First matching rule wins. `pattern` globs the tool name.
#[derive(Clone)]
pub struct Rule {
    pub pattern: String,
    pub decision: Permission,
    pub reason: String,
    pub when: Option<When>,
}

impl fmt::Debug for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rule").field("pattern", &self.pattern).field("decision", &self.decision).field("reason", &self.reason).finish()
    }
}

impl Rule {
    pub fn new(pattern: impl Into<String>, decision: Permission, reason: impl Into<String>) -> Self {
        Self { pattern: pattern.into(), decision, reason: reason.into(), when: None }
    }

    /// Only match when `when(args)` is true.
    pub fn when(mut self, f: impl Fn(&Args) -> bool + Send + Sync + 'static) -> Self {
        self.when = Some(Arc::new(f));
        self
    }

    pub fn matches(&self, tool: &str, args: &Args) -> bool {
        fnmatch(tool, &self.pattern) && self.when.as_ref().is_none_or(|w| w(args))
    }
}

/// An approver that says yes to everything.
pub fn always_approve() -> Approver {
    Arc::new(|_, _, _| async { true }.boxed())
}

/// A blocking y/n prompt on stdin. Fine for a CLI, never for a server.
pub fn console_approver() -> Approver {
    Arc::new(|tool, args, reason| {
        let prompt = format!("\nAllow {tool}({}) ? {reason}\n[y/N] ", Value::Object(args.clone()));
        async move {
            tokio::task::spawn_blocking(move || {
                use std::io::Write;
                print!("{prompt}");
                let _ = std::io::stdout().flush();
                let mut answer = String::new();
                let _ = std::io::stdin().read_line(&mut answer);
                matches!(answer.trim().to_lowercase().as_str(), "y" | "yes")
            })
            .await
            .unwrap_or(false)
        }
        .boxed()
    })
}

/// Decides whether an action may run, before it runs.
#[derive(Clone)]
pub struct PolicyGate {
    pub default: Permission,
    /// Deny rules come first, then ask, then allow, then custom rules.
    pub rules: Vec<Rule>,
    pub approver: Option<Approver>,
    pub audit: Option<Audit>,
    deny: Vec<Rule>,
    ask: Vec<Rule>,
    allow: Vec<Rule>,
    custom: Vec<Rule>,
}

impl fmt::Debug for PolicyGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PolicyGate").field("default", &self.default).field("rules", &self.rules).finish_non_exhaustive()
    }
}

impl Default for PolicyGate {
    fn default() -> Self {
        Self::new(Permission::Allow)
    }
}

impl PolicyGate {
    pub fn new(default: Permission) -> Self {
        Self { default, rules: vec![], approver: None, audit: None, deny: vec![], ask: vec![], allow: vec![], custom: vec![] }
    }

    fn rebuild(mut self) -> Self {
        // Deny wins over ask wins over allow when patterns overlap.
        self.rules = [self.deny.clone(), self.ask.clone(), self.allow.clone(), self.custom.clone()].concat();
        self
    }

    pub fn deny<I: IntoIterator<Item = S>, S: Into<String>>(mut self, patterns: I) -> Self {
        self.deny.extend(patterns.into_iter().map(|p| Rule::new(p, Permission::Deny, "denied by policy")));
        self.rebuild()
    }

    pub fn ask<I: IntoIterator<Item = S>, S: Into<String>>(mut self, patterns: I) -> Self {
        self.ask.extend(patterns.into_iter().map(|p| Rule::new(p, Permission::Ask, "needs approval")));
        self.rebuild()
    }

    pub fn allow<I: IntoIterator<Item = S>, S: Into<String>>(mut self, patterns: I) -> Self {
        self.allow.extend(patterns.into_iter().map(|p| Rule::new(p, Permission::Allow, "")));
        self.rebuild()
    }

    pub fn rule(mut self, rule: Rule) -> Self {
        self.custom.push(rule);
        self.rebuild()
    }

    pub fn approver(mut self, approver: Approver) -> Self {
        self.approver = Some(approver);
        self
    }

    pub fn audit(mut self, audit: impl Fn(&str, &str, &Args) + Send + Sync + 'static) -> Self {
        self.audit = Some(Arc::new(audit));
        self
    }

    pub fn decide(&self, tool: &str, args: &Args) -> (Permission, String) {
        self.rules.iter().find(|r| r.matches(tool, args)).map(|r| (r.decision, r.reason.clone())).unwrap_or((self.default, String::new()))
    }

    /// What this call comes to, the tool's own declaration included. A tool's
    /// declared permission can only tighten the policy, never loosen it.
    pub fn needs(&self, tool: &str, args: &Args, tool_permission: Option<Permission>) -> (Permission, String) {
        let (decision, reason) = self.decide(tool, args);
        match tool_permission {
            Some(Permission::Deny) => (Permission::Deny, "the tool declares itself deny-by-default".into()),
            Some(Permission::Ask) if decision == Permission::Allow => (Permission::Ask, "the tool asks for confirmation".into()),
            _ => (decision, reason),
        }
    }

    fn record(&self, tool: &str, decision: &str, args: &Args) {
        if let Some(a) = &self.audit {
            a(tool, decision, args);
        }
    }

    /// `Err(PermissionDenied)` unless the action is allowed.
    ///
    /// `approved_by` names a person who already said yes to exactly this call.
    /// It stands in for the approver; it does not turn a deny into an allow.
    pub async fn check(&self, tool: &str, args: &Args, tool_permission: Option<Permission>, approved_by: Option<&str>) -> Result<()> {
        let (decision, reason) = self.needs(tool, args, tool_permission);
        let denied = |message: String| Error::PermissionDenied { message, tool: tool.into(), reason: reason.clone() };
        match decision {
            Permission::Allow => {
                self.record(tool, "allow", args);
                Ok(())
            }
            Permission::Ask if approved_by.is_some_and(|a| !a.is_empty()) => {
                self.record(tool, "allow", args);
                Ok(())
            }
            Permission::Deny => {
                self.record(tool, "deny", args);
                Err(denied(format!("{tool} is not permitted: {}", if reason.is_empty() { "policy" } else { &reason })))
            }
            Permission::Ask => {
                let Some(approver) = &self.approver else {
                    self.record(tool, "deny", args);
                    return Err(denied(format!("{tool} needs approval but no approver is configured")));
                };
                let verdict = approver(tool, args, &reason).await;
                self.record(tool, if verdict { "allow" } else { "deny" }, args);
                if verdict { Ok(()) } else { Err(denied(format!("{tool} was declined by the approver"))) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    fn args(v: Value) -> Args {
        v.as_object().cloned().unwrap_or_default()
    }

    #[tokio::test]
    async fn precedence_and_tightening() {
        let gate = PolicyGate::new(Permission::Allow).allow(["shell"]).deny(["shell"]).ask(["http_*"]);
        assert_eq!(gate.decide("shell", &Args::new()).0, Permission::Deny);
        assert_eq!(gate.decide("http_get", &Args::new()).0, Permission::Ask);
        assert_eq!(gate.needs("read", &Args::new(), Some(Permission::Ask)).0, Permission::Ask);
        assert_eq!(gate.needs("read", &Args::new(), Some(Permission::Deny)).0, Permission::Deny);
        let err = gate.check("shell", &Args::new(), None, None).await.unwrap_err();
        assert_eq!(err.to_string(), "shell is not permitted: denied by policy");
        let err = gate.check("http_get", &Args::new(), None, None).await.unwrap_err();
        assert_eq!(err.to_string(), "http_get needs approval but no approver is configured");
        gate.check("http_get", &Args::new(), None, Some("ana")).await.unwrap();
    }

    #[tokio::test]
    async fn approver_and_audit() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let sink = log.clone();
        let no: Approver = Arc::new(|_, _, _| async { false }.boxed());
        let gate = PolicyGate::new(Permission::Ask).approver(no).audit(move |t, d, _| sink.lock().unwrap().push(format!("{t}:{d}")));
        assert!(gate.check("x", &Args::new(), None, None).await.is_err());
        let gate = gate.approver(always_approve());
        gate.check("y", &Args::new(), None, None).await.unwrap();
        assert_eq!(*log.lock().unwrap(), vec!["x:deny", "y:allow"]);
    }

    #[test]
    fn conditional_rules() {
        let gate = PolicyGate::new(Permission::Allow)
            .rule(Rule::new("shell", Permission::Deny, "rm").when(|a| a.get("cmd").and_then(Value::as_str).is_some_and(|c| c.contains("rm -rf"))));
        assert_eq!(gate.decide("shell", &args(json!({"cmd": "ls"}))).0, Permission::Allow);
        assert_eq!(gate.decide("shell", &args(json!({"cmd": "rm -rf /"}))).0, Permission::Deny);
    }
}
