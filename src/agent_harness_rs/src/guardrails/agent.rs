//! Per-agent guardrails: what this agent must do, and when it may say it is done.
//!
//! The content engine ([`Guardrails`]) polices *text* — secrets, injection,
//! size. This polices *behaviour*: which tools an agent may touch, and what has
//! to be true before its answer is accepted.
//!
//! ```
//! use agent_harness::guardrails::{AgentGuardrails, GuardrailOptions, OnViolation};
//!
//! let rails = AgentGuardrails::from_options(vec![], GuardrailOptions {
//!     require_tools: vec!["order_status".into()],   // look it up, never guess
//!     forbid_tools: vec!["issue_refund".into()],    // not this agent's job
//!     must_include: vec!["order".into()],
//!     no_placeholders: true,
//!     on_violation: OnViolation::Retry,             // tell it what is missing, let it fix it
//!     ..Default::default()
//! }).unwrap();
//! assert_eq!(rails.tool_allowed("issue_refund").0, false);
//! ```
//!
//! A failed check is not a crash. The agent is told, in words, what is missing
//! and gets another turn — which is usually all it needs.

use std::any::Any;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use futures::FutureExt;
use indexmap::IndexMap;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use super::checks::{
    Check, CompletionContext, MaxCost, MaxSteps, MinLength, MustInclude, MustMatch, MustNotInclude, NoPlaceholders, RequireCitation, RequireJSON,
    RequireStructured, RequireTools, Violation,
};
use super::engine::{Guardrails, GuardrailsConfig};
use super::standard::{Grounded, NoInjection, NoPII, NoRepetition, NoSecrets, NotToxic};
use crate::errors::{Error, Result};
use crate::knowledge::kb::py_repr;

type CheckRef = Arc<dyn Check>;
type MakeCheck = fn() -> CheckRef;

/// What happens when an answer fails its checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnViolation {
    /// Tell the agent what is missing and let it answer again (up to `max_retries`).
    #[default]
    Retry,
    /// Fail the run with `GuardrailTripped`.
    Fail,
    /// Record it and deliver the answer anyway.
    Warn,
}

impl OnViolation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Retry => "retry",
            Self::Fail => "fail",
            Self::Warn => "warn",
        }
    }
}

impl FromStr for OnViolation {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "retry" => Ok(Self::Retry),
            "fail" => Ok(Self::Fail),
            "warn" => Ok(Self::Warn),
            other => Err(Error::config(format!("on_violation must be retry, fail or warn — got {}", py_repr(other)))),
        }
    }
}

impl fmt::Display for OnViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A list of names; a single string counts as a list of one.
fn one_or_many<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Many {
        One(String),
        List(Vec<String>),
        Nothing(()),
    }
    Ok(match Many::deserialize(d)? {
        Many::One(s) => vec![s],
        Many::List(v) => v,
        Many::Nothing(()) => vec![],
    })
}

fn maybe_many<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<Vec<String>>, D::Error> {
    let v = Option::<Value>::deserialize(d)?;
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(v) => one_or_many(v).map(Some).map_err(serde::de::Error::custom),
    }
}

/// `true` is 0.6, a number is the overlap, `false`/`0` is off.
fn grounded<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<f64>, D::Error> {
    Ok(match Value::deserialize(d)? {
        Value::Bool(true) => Some(0.6),
        Value::Number(n) => n.as_f64().filter(|x| *x != 0.0),
        Value::Null | Value::Bool(false) => None,
        other => return Err(serde::de::Error::custom(format!("grounded must be true, false or a number — got {other}"))),
    })
}

/// The keyword form of Python's `AgentGuardrails(...)`, and the dict a
/// blueprint or `SubAgentSpec` carries. Every field off by default.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GuardrailOptions {
    #[serde(deserialize_with = "one_or_many")]
    pub require_tools: Vec<String>,
    /// Enforced before the call; not also a completion check.
    #[serde(deserialize_with = "one_or_many")]
    pub forbid_tools: Vec<String>,
    /// When set, only these tools may be called.
    #[serde(deserialize_with = "maybe_many")]
    pub allow_tools: Option<Vec<String>>,
    #[serde(deserialize_with = "one_or_many")]
    pub must_include: Vec<String>,
    #[serde(deserialize_with = "one_or_many")]
    pub must_not_include: Vec<String>,
    pub must_match: Option<String>,
    pub min_output_chars: Option<usize>,
    pub max_steps: Option<u32>,
    pub max_cost_usd: Option<f64>,
    pub require_citation: bool,
    pub require_json: bool,
    pub require_structured: bool,
    pub no_placeholders: bool,
    pub no_pii: bool,
    pub no_secrets: bool,
    pub no_injection: bool,
    pub not_toxic: bool,
    pub no_repetition: bool,
    /// The minimum overlap with the tool results (`true` in a dict means 0.6).
    #[serde(deserialize_with = "grounded")]
    pub grounded: Option<f64>,
    /// This agent's own content rules, in place of the harness's. In a dict,
    /// the settings of [`GuardrailsConfig`].
    pub content: Option<GuardrailsConfig>,
    pub on_violation: OnViolation,
    pub max_retries: u32,
}

impl Default for GuardrailOptions {
    fn default() -> Self {
        Self {
            require_tools: vec![],
            forbid_tools: vec![],
            allow_tools: None,
            must_include: vec![],
            must_not_include: vec![],
            must_match: None,
            min_output_chars: None,
            max_steps: None,
            max_cost_usd: None,
            require_citation: false,
            require_json: false,
            require_structured: false,
            no_placeholders: false,
            no_pii: false,
            no_secrets: false,
            no_injection: false,
            not_toxic: false,
            no_repetition: false,
            grounded: None,
            content: None,
            on_violation: OnViolation::Retry,
            max_retries: 2,
        }
    }
}

const KEYWORDS: &[&str] = &[
    "require_tools",
    "forbid_tools",
    "allow_tools",
    "must_include",
    "must_not_include",
    "must_match",
    "min_output_chars",
    "max_steps",
    "max_cost_usd",
    "require_citation",
    "require_json",
    "require_structured",
    "no_placeholders",
    "no_pii",
    "no_secrets",
    "no_injection",
    "not_toxic",
    "no_repetition",
    "grounded",
    "content",
    "on_violation",
    "max_retries",
];

/// What a completion check found, and what the agent should do about it.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckResult {
    pub violations: Vec<Violation>,
    pub on_violation: OnViolation,
}

impl CheckResult {
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }

    /// Each violation as one line — what goes in `RunResult::violations`.
    pub fn lines(&self) -> Vec<String> {
        self.violations.iter().map(Violation::line).collect()
    }

    /// The message that sends the agent back round.
    pub fn feedback(&self) -> String {
        AgentGuardrails::feedback(&self.violations)
    }
}

/// How often each check objected, for dashboards and logs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GuardrailReport {
    pub checks: usize,
    pub violations: IndexMap<String, usize>,
    pub on_violation: OnViolation,
}

/// One object holding everything this agent is and is not allowed to do.
///
/// Cheap to clone; clones share the record of violations.
#[derive(Clone)]
pub struct AgentGuardrails {
    pub checks: Vec<Arc<dyn Check>>,
    /// Never callable. Enforced before the call, so the tool never runs.
    pub forbidden: Vec<String>,
    /// When set, the only tools that may be called.
    pub allowed: Option<Vec<String>>,
    /// This agent's own content rules; `None` falls back to the harness's.
    pub content: Option<Guardrails>,
    pub on_violation: OnViolation,
    pub max_retries: u32,
    violations: Arc<Mutex<Vec<Violation>>>,
}

impl Default for AgentGuardrails {
    /// No checks, no tool limits, retry twice.
    fn default() -> Self {
        Self { checks: vec![], forbidden: vec![], allowed: None, content: None, on_violation: OnViolation::Retry, max_retries: 2, violations: Arc::default() }
    }
}

impl AgentGuardrails {
    /// Your checks first, then the ones the options switch on, in Python's order.
    pub fn from_options(checks: Vec<Arc<dyn Check>>, options: GuardrailOptions) -> Result<Self> {
        let mut built = checks;
        let mut add = |c: Arc<dyn Check>| built.push(c);
        if !options.require_tools.is_empty() {
            add(Arc::new(RequireTools::new(options.require_tools)));
        }
        // `forbid_tools` is enforced before the call, so it deliberately does not
        // also become a completion check — the tool never ran, and penalising the
        // answer for a call that was already blocked just confuses the agent.
        // `ForbidTools` is still available as an explicit after-the-fact check.
        if !options.must_include.is_empty() {
            add(Arc::new(MustInclude::new(options.must_include)));
        }
        if !options.must_not_include.is_empty() {
            add(Arc::new(MustNotInclude::new(options.must_not_include)));
        }
        if let Some(pattern) = options.must_match.filter(|p| !p.is_empty()) {
            add(Arc::new(MustMatch::new(pattern)?));
        }
        if let Some(chars) = options.min_output_chars.filter(|n| *n > 0) {
            add(Arc::new(MinLength::new(chars)));
        }
        if let Some(steps) = options.max_steps.filter(|n| *n > 0) {
            add(Arc::new(MaxSteps::new(steps)));
        }
        if let Some(usd) = options.max_cost_usd.filter(|x| *x != 0.0) {
            add(Arc::new(MaxCost::new(usd)));
        }
        let flags: [(bool, MakeCheck); 9] = [
            (options.require_citation, || Arc::new(RequireCitation::default())),
            (options.require_json, || Arc::new(RequireJSON::default())),
            (options.require_structured, || Arc::new(RequireStructured::default())),
            (options.no_placeholders, || Arc::new(NoPlaceholders::default())),
            (options.no_pii, || Arc::new(NoPII::default())),
            (options.no_secrets, || Arc::new(NoSecrets::default())),
            (options.no_injection, || Arc::new(NoInjection::default())),
            (options.not_toxic, || Arc::new(NotToxic::default())),
            (options.no_repetition, || Arc::new(NoRepetition::default())),
        ];
        for (on, make) in flags {
            if on {
                add(make());
            }
        }
        if let Some(overlap) = options.grounded {
            add(Arc::new(Grounded::new(overlap)));
        }
        Ok(Self {
            checks: built,
            forbidden: options.forbid_tools,
            allowed: options.allow_tools,
            content: options.content.map(|c| Guardrails::new(vec![], c)),
            on_violation: options.on_violation,
            max_retries: options.max_retries,
            violations: Arc::default(),
        })
    }

    /// Build from the dict form Python accepts (`AgentGuardrails(**config)`), as
    /// a blueprint's YAML or a `SubAgentSpec` carries it. `null` or `{}` gives
    /// guardrails with nothing in them (Python's `from_dict` returns `None`;
    /// check [`is_empty`](Self::is_empty) and the tool lists if that matters).
    pub fn from_value(config: &Value) -> Result<Self> {
        let map = match config {
            Value::Null => return Ok(Self::default()),
            Value::Object(map) => map,
            other => return Err(Error::config(format!("guardrails must be a mapping of settings — got {other}"))),
        };
        if let Some(key) = map.keys().find(|k| !KEYWORDS.contains(&k.as_str())) {
            return Err(Error::config(format!("AgentGuardrails.__init__() got an unexpected keyword argument {}", py_repr(key))));
        }
        if let Some(Value::String(s)) = map.get("on_violation") {
            OnViolation::from_str(s)?;
        }
        if let Some(n) = map.get("max_retries").and_then(Value::as_i64)
            && n < 0
        {
            return Err(Error::config("max_retries cannot be negative"));
        }
        let options: GuardrailOptions = serde_json::from_value(config.clone()).map_err(|e| Error::config(format!("guardrails: {e}")))?;
        Self::from_options(vec![], options)
    }

    /// Add a check (any [`Check`], an `Arc` of one included).
    pub fn with(mut self, check: impl Check + 'static) -> Self {
        self.checks.push(Arc::new(check));
        self
    }

    // ---- before a tool runs ------------------------------------------------

    /// Checked before the call, so a forbidden tool never actually runs.
    pub fn tool_allowed(&self, name: &str) -> (bool, String) {
        if self.forbidden.iter().any(|t| t == name) {
            return (false, format!("{name} is not permitted for this agent"));
        }
        if let Some(allowed) = &self.allowed
            && !allowed.iter().any(|t| t == name)
        {
            let permitted = if allowed.is_empty() { "none".to_string() } else { allowed.join(", ") };
            return (false, format!("{name} is not on this agent's allowlist (permitted: {permitted})"));
        }
        (true, String::new())
    }

    // ---- before the answer is accepted --------------------------------------

    pub fn has_checks(&self) -> bool {
        !self.checks.is_empty()
    }

    /// True when a check must be awaited — an LLM judge, typically.
    pub fn has_async_checks(&self) -> bool {
        self.checks.iter().any(|c| c.needs_await())
    }

    /// Run the synchronous checks. Refuses if a check must be awaited.
    pub fn check(&self, ctx: &CompletionContext) -> Result<Vec<Violation>> {
        if self.has_async_checks() {
            return Err(Error::config(
                "these guardrails include a check that must be awaited (an LLM guard, most likely) — call `await rails.check_async(ctx)`",
            ));
        }
        let found = self.checks.iter().filter_map(|c| safely(c.as_ref(), ctx)).collect();
        Ok(self.collect(found))
    }

    /// Run every check, awaiting the ones that need it.
    ///
    /// Sync checks run first, in order. If any of them already objects, the
    /// async ones are skipped — there is no reason to pay a model to confirm
    /// what a regex just proved.
    pub async fn check_async(&self, ctx: &CompletionContext) -> Vec<Violation> {
        let (awaited, sync): (Vec<&CheckRef>, Vec<&CheckRef>) = self.checks.iter().partition(|c| c.needs_await());
        let found: Vec<Violation> = sync.iter().filter_map(|c| safely(c.as_ref(), ctx)).collect();
        if !found.is_empty() || awaited.is_empty() {
            return self.collect(found);
        }
        let results = futures::future::join_all(awaited.iter().map(|c| safely_async(c.as_ref(), ctx))).await;
        self.collect(results.into_iter().flatten().collect())
    }

    /// [`check_async`](Self::check_async), with what to do about it.
    pub async fn evaluate(&self, ctx: &CompletionContext) -> CheckResult {
        CheckResult { violations: self.check_async(ctx).await, on_violation: self.on_violation }
    }

    fn collect(&self, found: Vec<Violation>) -> Vec<Violation> {
        self.violations.lock().expect("violation record").extend(found.iter().cloned());
        found
    }

    /// The message the agent is given so it can put the answer right.
    pub fn feedback(violations: &[Violation]) -> String {
        let lead = if violations.len() > 1 { "That answer does not meet this task's requirements yet:" } else { "That answer does not meet a requirement:" };
        let body = violations.iter().map(|v| format!("- {}", v.line())).collect::<Vec<_>>().join("\n");
        format!("{lead}\n{body}\n\nPut it right and answer again.")
    }

    // ---- plumbing -----------------------------------------------------------

    /// How many completion checks there are.
    pub fn len(&self) -> usize {
        self.checks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.checks.is_empty()
    }

    /// Every violation recorded so far.
    pub fn violations(&self) -> Vec<Violation> {
        self.violations.lock().expect("violation record").clone()
    }

    pub fn report(&self) -> GuardrailReport {
        let mut counts = IndexMap::new();
        for v in self.violations.lock().expect("violation record").iter() {
            *counts.entry(v.check.clone()).or_insert(0) += 1;
        }
        GuardrailReport { checks: self.checks.len(), violations: counts, on_violation: self.on_violation }
    }
}

impl fmt::Debug for AgentGuardrails {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.checks.iter().map(|c| c.name()).collect();
        f.debug_struct("AgentGuardrails")
            .field("checks", &names)
            .field("forbidden", &self.forbidden)
            .field("allowed", &self.allowed)
            .field("content", &self.content)
            .field("on_violation", &self.on_violation)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

fn panic_text(payload: Box<dyn Any + Send>) -> String {
    payload.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| payload.downcast_ref::<String>().cloned()).unwrap_or_else(|| "it panicked".into())
}

fn broken(check: &dyn Check, why: String) -> Violation {
    Violation::new(check.name(), format!("the check itself failed: {why}"), "")
}

/// A broken check must not break the run.
fn safely(check: &dyn Check, ctx: &CompletionContext) -> Option<Violation> {
    match std::panic::catch_unwind(AssertUnwindSafe(|| check.check(ctx))) {
        Ok(Ok(found)) => found,
        Ok(Err(e)) => Some(broken(check, e.to_string())),
        Err(payload) => Some(broken(check, panic_text(payload))),
    }
}

async fn safely_async(check: &dyn Check, ctx: &CompletionContext) -> Option<Violation> {
    match AssertUnwindSafe(check.check_async(ctx)).catch_unwind().await {
        Ok(Ok(found)) => found,
        Ok(Err(e)) => Some(broken(check, e.to_string())),
        Err(payload) => Some(broken(check, panic_text(payload))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guardrails::{Custom, LLMGuard, Rule};
    use crate::llm_providers::{FakeProvider, Script};
    use serde_json::json;

    fn ctx(output: &str) -> CompletionContext {
        CompletionContext { agent: "a".into(), output: output.into(), ..Default::default() }
    }

    fn opts(o: GuardrailOptions) -> AgentGuardrails {
        AgentGuardrails::from_options(vec![], o).unwrap()
    }

    #[test]
    fn a_check_that_itself_explodes_is_reported_not_raised() {
        let rails = AgentGuardrails::default()
            .with(Custom::fallible(|_| -> Result<bool> { Err(Error::config("the check is buggy")) }).named("broken"))
            .with(Custom::new(|_| -> bool { panic!("it fell over") }).named("panics"));
        let found = rails.check(&ctx("anything")).unwrap();
        assert_eq!(found[0].line(), "the check itself failed: the check is buggy");
        assert_eq!(found[1].line(), "the check itself failed: it fell over");
    }

    #[test]
    fn keyword_form_builds_the_same_checks() {
        let rails = opts(GuardrailOptions {
            require_tools: vec!["order_status".into()],
            forbid_tools: vec!["issue_refund".into()],
            must_include: vec!["order".into()],
            no_placeholders: true,
            max_cost_usd: Some(0.25),
            min_output_chars: Some(10),
            ..Default::default()
        });
        // forbid_tools is enforced before the call, so it is not also a completion
        // check — five checks, not six.
        assert_eq!(rails.len(), 5);
        assert_eq!(rails.tool_allowed("issue_refund"), (false, "issue_refund is not permitted for this agent".to_string()));
        assert!(!rails.checks.iter().any(|c| c.name() == "forbid_tools"));
        let names: Vec<&str> = rails.checks.iter().map(|c| c.name()).collect();
        assert_eq!(names, ["require_tools", "must_include", "min_length", "max_cost", "no_placeholders"]);
    }

    #[test]
    fn an_allowlist_refuses_everything_else() {
        let rails = opts(GuardrailOptions { allow_tools: Some(vec!["order_status".into()]), ..Default::default() });
        assert!(rails.tool_allowed("order_status").0);
        assert_eq!(rails.tool_allowed("issue_refund"), (false, "issue_refund is not on this agent's allowlist (permitted: order_status)".to_string()));
        let none = opts(GuardrailOptions { allow_tools: Some(vec![]), ..Default::default() });
        assert_eq!(none.tool_allowed("x").1, "x is not on this agent's allowlist (permitted: none)");
    }

    #[test]
    fn bad_configuration_is_refused() {
        let err = AgentGuardrails::from_value(&json!({"on_violation": "explode"})).unwrap_err();
        assert_eq!(err.to_string(), "on_violation must be retry, fail or warn — got 'explode'");
        let err = AgentGuardrails::from_value(&json!({"max_retries": -1})).unwrap_err();
        assert_eq!(err.to_string(), "max_retries cannot be negative");
        let err = AgentGuardrails::from_value(&json!({"no_such_thing": true})).unwrap_err();
        assert_eq!(err.to_string(), "AgentGuardrails.__init__() got an unexpected keyword argument 'no_such_thing'");
        assert!(AgentGuardrails::from_value(&json!({"must_match": "(?<=x)y"})).is_err());
    }

    #[test]
    fn the_dict_form_builds_guardrails() {
        let rails = AgentGuardrails::from_value(&json!({
            "require_citation": true, "max_retries": 1, "on_violation": "warn",
            "forbid_tools": "issue_refund", "grounded": true, "max_steps": 0,
            "content": {"secrets": false, "injection": false, "blocklist": ["acme"]},
        }))
        .unwrap();
        assert_eq!(rails.len(), 2);
        assert_eq!(rails.max_retries, 1);
        assert_eq!(rails.on_violation, OnViolation::Warn);
        assert_eq!(rails.forbidden, ["issue_refund"]);
        let content = rails.content.as_ref().unwrap();
        assert_eq!(content.rules().len(), 1);
        assert!(content.check("we use ACME", "output", "").is_err());
        assert!(AgentGuardrails::from_value(&Value::Null).unwrap().is_empty());
    }

    #[test]
    fn the_keyword_flags_build_the_detector_checks() {
        let rails = opts(GuardrailOptions {
            no_pii: true,
            no_secrets: true,
            no_injection: true,
            not_toxic: true,
            no_repetition: true,
            grounded: Some(0.7),
            ..Default::default()
        });
        let names: Vec<&str> = rails.checks.iter().map(|c| c.name()).collect();
        assert_eq!(names, ["no_pii", "no_secrets", "no_injection", "not_toxic", "no_repetition", "grounded"]);
    }

    #[test]
    fn the_feedback_reads_like_something_an_agent_can_act_on() {
        let rails = opts(GuardrailOptions { require_tools: vec!["order_status".into()], must_include: vec!["order".into()], ..Default::default() });
        let found = rails.check(&ctx("dunno")).unwrap();
        assert_eq!(
            AgentGuardrails::feedback(&found),
            "That answer does not meet this task's requirements yet:\n\
             - you answered without calling order_status — call order_status and answer from what it returns\n\
             - your answer does not mention 'order' — say it explicitly\n\nPut it right and answer again."
        );
        assert!(AgentGuardrails::feedback(&found[..1]).starts_with("That answer does not meet a requirement:\n- "));
    }

    #[test]
    fn the_report_counts_what_tripped() {
        let rails = opts(GuardrailOptions { must_include: vec!["order".into()], require_citation: true, ..Default::default() });
        rails.check(&ctx("nothing here")).unwrap();
        let report = rails.report();
        assert_eq!(report.checks, 2);
        assert_eq!(report.violations, IndexMap::from([("must_include".to_string(), 1), ("require_citation".to_string(), 1)]));
        assert_eq!(report.on_violation, OnViolation::Retry);
    }

    fn judge(answer: &str) -> Arc<LLMGuard> {
        let p = Arc::new(FakeProvider::new(vec![Script::from(answer)]).looping());
        Arc::new(LLMGuard::with_provider(p, "claude-sonnet-5", "safety"))
    }

    #[tokio::test]
    async fn the_cheap_checks_run_first_and_short_circuit_the_expensive_one() {
        let guard = judge(r#"{"pass": true, "severity": "low", "reason": "ok"}"#);
        let rails = opts(GuardrailOptions { no_pii: true, ..Default::default() }).with(guard.clone());
        let found = rails.check_async(&ctx("write to ada@example.com")).await;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].check, "no_pii");
        assert_eq!(guard.calls(), 0, "the judge was never asked");
    }

    #[tokio::test]
    async fn the_judge_runs_when_the_cheap_checks_are_happy() {
        let guard = judge(r#"{"pass": false, "severity": "high", "reason": "off policy"}"#);
        let rails = opts(GuardrailOptions { no_pii: true, ..Default::default() }).with(guard.clone());
        let result = rails.evaluate(&ctx("nothing personal in here at all")).await;
        assert!(!result.passed());
        assert_eq!(result.violations[0].check, guard.name);
        assert_eq!(result.lines(), ["off policy — fix what the reason describes and answer again"]);
        assert_eq!(guard.calls(), 1);
    }

    #[test]
    fn calling_the_sync_path_with_an_async_check_says_what_to_do() {
        let rails = AgentGuardrails::default().with(judge(r#"{"pass": true}"#));
        assert!(rails.has_async_checks());
        assert!(rails.check(&ctx("x")).unwrap_err().to_string().contains("check_async"));
    }

    #[test]
    fn content_rules_travel_with_the_agent() {
        let strict = Guardrails::only(vec![Rule::new("no_ids", r"\b\d{4}\b", super::super::Action::Redact).unwrap().replacement("[id]")]);
        let rails = AgentGuardrails { content: Some(strict), ..Default::default() };
        let copy = rails.clone();
        assert_eq!(copy.content.unwrap().check("order 4182 shipped", "output", "").unwrap(), "order [id] shipped");
        fn send_sync<T: Send + Sync + Clone>() {}
        send_sync::<AgentGuardrails>();
        send_sync::<Guardrails>();
    }
}
