//! Completion checks: what an agent must have done before it may call itself done.
//!
//! A check looks at the finished run — the answer, the tools it called, what it
//! cost — and either passes or says what is missing. The agent is told, in
//! words, and gets another turn to put it right.
//!
//! ```
//! use agent_harness::guardrails::{AgentGuardrails, MaxCost, MustInclude, NoPlaceholders, RequireTools};
//!
//! let rails = AgentGuardrails::default()
//!     .with(RequireTools::new(["order_status"]))   // look it up, do not guess
//!     .with(MustInclude::new(["order"]))
//!     .with(NoPlaceholders::default())              // no "TODO", no "[insert name]"
//!     .with(MaxCost::new(0.25));
//! assert_eq!(rails.len(), 4);
//! ```

use std::fmt;
use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::{Error, Result};
use crate::knowledge::kb::py_repr;
use crate::types::RunResult;

/// Everything a check is allowed to look at.
#[derive(Debug, Clone, Default)]
pub struct CompletionContext {
    pub agent: String,
    pub output: String,
    pub steps: u32,
    pub cost_usd: f64,
    /// The tools this run called, its sub-agents' included.
    pub tools_called: Vec<String>,
    /// The names of the artifacts it produced.
    pub artifacts: Vec<String>,
    /// The structured output, `Null` when there is none.
    pub data: Value,
    /// The run itself: its messages carry the tool results.
    pub result: Option<RunResult>,
}

impl CompletionContext {
    /// The context Python's agent builds for its completion check: tools called
    /// by the run and by its direct sub-agents, artifact names, steps, cost, data.
    pub fn from_run(agent: impl Into<String>, output: impl Into<String>, result: &RunResult) -> Self {
        let mut called: Vec<String> = result.tool_calls.iter().map(|c| c.name.clone()).collect();
        called.extend(result.children.iter().flat_map(|child| child.tool_calls.iter().map(|c| c.name.clone())));
        Self {
            agent: agent.into(),
            output: output.into(),
            steps: result.steps,
            cost_usd: result.cost_usd(),
            tools_called: called,
            artifacts: result.artifacts.iter().map(|a| a.name.clone()).collect(),
            data: result.data.clone(),
            result: Some(result.clone()),
        }
    }

    /// The output, lower-cased.
    pub fn lowered(&self) -> String {
        self.output.to_lowercase()
    }
}

/// One unmet requirement, and what the agent should do about it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Violation {
    /// The name of the check that objected.
    pub check: String,
    pub detail: String,
    #[serde(default)]
    pub fix: String,
}

impl Violation {
    pub fn new(check: impl Into<String>, detail: impl Into<String>, fix: impl Into<String>) -> Self {
        Self { check: check.into(), detail: detail.into(), fix: fix.into() }
    }

    /// `"detail — fix"`, or just the detail.
    pub fn line(&self) -> String {
        if self.fix.is_empty() { self.detail.clone() } else { format!("{} — {}", self.detail, self.fix) }
    }
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.line())
    }
}

/// Anything that can inspect a finished run and object to it.
///
/// `Ok(None)` passes, `Ok(Some(v))` objects; an `Err` (or a panic) is reported
/// as "the check itself failed" rather than breaking the run. A check that has
/// to wait on something — an LLM judge — returns true from `needs_await` and
/// does its work in `check_async`.
#[async_trait]
pub trait Check: Send + Sync {
    fn name(&self) -> &str;

    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>>;

    /// Does running this check mean waiting (a model call, typically)?
    fn needs_await(&self) -> bool {
        false
    }

    async fn check_async(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        self.check(ctx)
    }
}

#[async_trait]
impl<T: Check + ?Sized> Check for Arc<T> {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        (**self).check(ctx)
    }
    fn needs_await(&self) -> bool {
        (**self).needs_await()
    }
    async fn check_async(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        (**self).check_async(ctx).await
    }
}

fn strings<I, S>(items: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    items.into_iter().map(Into::into).collect()
}

/// `name` field, `named()` builder and the `Check::name` accessor for a check.
macro_rules! named {
    ($t:ty) => {
        impl $t {
            /// The name violations are reported under.
            pub fn named(mut self, name: impl Into<String>) -> Self {
                self.name = name.into();
                self
            }
        }
    };
}

// --- what it must have done ---------------------------------------------------

/// The agent must have called these before answering. Stops it guessing.
#[derive(Debug, Clone)]
pub struct RequireTools {
    pub tools: Vec<String>,
    pub name: String,
}
named!(RequireTools);

impl RequireTools {
    pub fn new<I: IntoIterator<Item = S>, S: Into<String>>(tools: I) -> Self {
        Self { tools: strings(tools), name: "require_tools".into() }
    }
}

impl Check for RequireTools {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        let missing: Vec<&str> = self.tools.iter().filter(|t| !ctx.tools_called.contains(t)).map(String::as_str).collect();
        if missing.is_empty() {
            return Ok(None);
        }
        Ok(Some(Violation::new(
            &self.name,
            format!("you answered without calling {}", missing.join(", ")),
            format!("call {} and answer from what it returns", missing[0]),
        )))
    }
}

/// These may never be called. Enforced before the call, and checked after.
#[derive(Debug, Clone)]
pub struct ForbidTools {
    pub tools: Vec<String>,
    pub name: String,
}
named!(ForbidTools);

impl ForbidTools {
    pub fn new<I: IntoIterator<Item = S>, S: Into<String>>(tools: I) -> Self {
        Self { tools: strings(tools), name: "forbid_tools".into() }
    }
}

impl Check for ForbidTools {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        let used: Vec<&str> = self.tools.iter().filter(|t| ctx.tools_called.contains(t)).map(String::as_str).collect();
        if used.is_empty() {
            return Ok(None);
        }
        Ok(Some(Violation::new(&self.name, format!("you called {}, which is not permitted for this task", used.join(", ")), "")))
    }
}

// --- what the answer must look like -------------------------------------------

fn quoted(items: &[&String]) -> String {
    items.iter().map(|s| py_repr(s)).collect::<Vec<_>>().join(", ")
}

/// The answer has to mention these. One missing phrase is enough to fail.
#[derive(Debug, Clone)]
pub struct MustInclude {
    pub phrases: Vec<String>,
    pub case_sensitive: bool,
    pub name: String,
}
named!(MustInclude);

impl MustInclude {
    pub fn new<I: IntoIterator<Item = S>, S: Into<String>>(phrases: I) -> Self {
        Self { phrases: strings(phrases), case_sensitive: false, name: "must_include".into() }
    }

    pub fn case_sensitive(mut self, on: bool) -> Self {
        self.case_sensitive = on;
        self
    }
}

impl Check for MustInclude {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        let haystack = if self.case_sensitive { ctx.output.clone() } else { ctx.lowered() };
        let missing: Vec<&String> =
            self.phrases.iter().filter(|p| if self.case_sensitive { !haystack.contains(p.as_str()) } else { !haystack.contains(&p.to_lowercase()) }).collect();
        if missing.is_empty() {
            return Ok(None);
        }
        Ok(Some(Violation::new(&self.name, format!("your answer does not mention {}", quoted(&missing)), "say it explicitly")))
    }
}

/// The answer must avoid these — a phrase blocklist for the final reply.
#[derive(Debug, Clone)]
pub struct MustNotInclude {
    pub phrases: Vec<String>,
    pub case_sensitive: bool,
    pub name: String,
}
named!(MustNotInclude);

impl MustNotInclude {
    pub fn new<I: IntoIterator<Item = S>, S: Into<String>>(phrases: I) -> Self {
        Self { phrases: strings(phrases), case_sensitive: false, name: "must_not_include".into() }
    }

    pub fn case_sensitive(mut self, on: bool) -> Self {
        self.case_sensitive = on;
        self
    }
}

impl Check for MustNotInclude {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        let haystack = if self.case_sensitive { ctx.output.clone() } else { ctx.lowered() };
        let found: Vec<&String> =
            self.phrases.iter().filter(|p| if self.case_sensitive { haystack.contains(p.as_str()) } else { haystack.contains(&p.to_lowercase()) }).collect();
        if found.is_empty() {
            return Ok(None);
        }
        Ok(Some(Violation::new(&self.name, format!("your answer contains {}", quoted(&found)), "rewrite it without that")))
    }
}

/// The answer has to match a pattern — an order id, a verdict, a format.
///
/// Matched case-insensitive, `.` matching newlines. Rust's `regex` has no
/// look-around, so a pattern that uses it is refused when the check is made.
#[derive(Debug, Clone)]
pub struct MustMatch {
    pub pattern: String,
    pub description: String,
    pub name: String,
    regex: Regex,
}
named!(MustMatch);

impl MustMatch {
    pub fn new(pattern: impl Into<String>) -> Result<Self> {
        let pattern = pattern.into();
        let regex = Regex::new(&format!("(?is){pattern}")).map_err(|e| Error::config(format!("must_match pattern {pattern:?} does not compile: {e}")))?;
        Ok(Self { pattern, description: String::new(), name: "must_match".into(), regex })
    }

    /// What the violation says instead of quoting the pattern.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

impl Check for MustMatch {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        if self.regex.is_match(&ctx.output) {
            return Ok(None);
        }
        let detail = if self.description.is_empty() { format!("your answer does not match /{}/", self.pattern) } else { self.description.clone() };
        Ok(Some(Violation::new(&self.name, detail, "answer in the form that was asked for")))
    }
}

/// Guards against a one-word answer to a question that needed working through.
#[derive(Debug, Clone)]
pub struct MinLength {
    pub chars: usize,
    pub name: String,
}
named!(MinLength);

impl MinLength {
    pub fn new(chars: usize) -> Self {
        Self { chars, name: "min_length".into() }
    }
}

impl Default for MinLength {
    fn default() -> Self {
        Self::new(40)
    }
}

impl Check for MinLength {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        let length = ctx.output.trim().chars().count();
        if length >= self.chars {
            return Ok(None);
        }
        Ok(Some(Violation::new(
            &self.name,
            format!("your answer is {length} characters; this task needs at least {}", self.chars),
            "answer the question properly",
        )))
    }
}

/// Every claim needs a source. Checks the answer carries at least one.
#[derive(Debug, Clone)]
pub struct RequireCitation {
    pub markers: Vec<String>,
    pub name: String,
}
named!(RequireCitation);

impl Default for RequireCitation {
    fn default() -> Self {
        Self { markers: strings(["http://", "https://", "source:", "according to", "[1]", "§"]), name: "require_citation".into() }
    }
}

impl Check for RequireCitation {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        let lowered = ctx.lowered();
        if self.markers.iter().any(|m| lowered.contains(&m.to_lowercase())) {
            return Ok(None);
        }
        Ok(Some(Violation::new(&self.name, "your answer cites nothing", "give the source for each claim, or say you could not find one")))
    }
}

static FENCED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)```(?:json)?\s*(.*?)```").expect("valid"));

/// The JSON in `text`: the body of a ```json fence, else the whole text.
pub(crate) fn parse_json_answer(text: &str) -> Option<Value> {
    let fenced = FENCED.captures(text).map(|c| c[1].to_string());
    fenced.into_iter().chain([text.to_string()]).find_map(|candidate| serde_json::from_str(&candidate).ok())
}

/// The answer must be parseable JSON — the cheap version of a contract.
#[derive(Debug, Clone)]
pub struct RequireJSON {
    pub name: String,
}
named!(RequireJSON);

impl Default for RequireJSON {
    fn default() -> Self {
        Self { name: "require_json".into() }
    }
}

impl Check for RequireJSON {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        if parse_json_answer(ctx.output.trim()).is_some() {
            return Ok(None);
        }
        Ok(Some(Violation::new(&self.name, "your answer is not valid JSON", "reply with only the JSON object, no prose")))
    }
}

/// Used with an output contract: the contract must actually have been satisfied.
#[derive(Debug, Clone)]
pub struct RequireStructured {
    pub name: String,
}
named!(RequireStructured);

impl Default for RequireStructured {
    fn default() -> Self {
        Self { name: "require_structured".into() }
    }
}

impl Check for RequireStructured {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        if !ctx.data.is_null() {
            return Ok(None);
        }
        Ok(Some(Violation::new(
            &self.name,
            "you did not return the structured output this task requires",
            "reply with only the JSON object the schema describes",
        )))
    }
}

/// Catches work handed back half-finished: TODO, [insert x], lorem ipsum.
#[derive(Debug, Clone)]
pub struct NoPlaceholders {
    pub patterns: Vec<String>,
    pub name: String,
    compiled: Vec<Regex>,
}
named!(NoPlaceholders);

impl Default for NoPlaceholders {
    fn default() -> Self {
        Self::with_patterns([r"\bTODO\b", r"\bFIXME\b", r"\bTBD\b", r"lorem ipsum", r"\[insert[^\]]*\]", r"\[your[^\]]*\]", r"<placeholder>", r"xxx+"])
            .expect("built-in placeholder patterns compile")
    }
}

impl NoPlaceholders {
    /// Your own placeholder patterns, matched case-insensitive.
    pub fn with_patterns<I: IntoIterator<Item = S>, S: Into<String>>(patterns: I) -> Result<Self> {
        let patterns = strings(patterns);
        let compiled = patterns
            .iter()
            .map(|p| Regex::new(&format!("(?i){p}")).map_err(|e| Error::config(format!("placeholder pattern {p:?} does not compile: {e}"))))
            .collect::<Result<_>>()?;
        Ok(Self { patterns, name: "no_placeholders".into(), compiled })
    }
}

impl Check for NoPlaceholders {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        if !self.compiled.iter().any(|r| r.is_match(&ctx.output)) {
            return Ok(None);
        }
        Ok(Some(Violation::new(&self.name, "your answer still contains placeholder text", "fill it in, or say plainly what you could not find")))
    }
}

// --- budgets, as a completion condition ----------------------------------------

/// A soft step ceiling: the run is not killed, the answer is just rejected.
#[derive(Debug, Clone)]
pub struct MaxSteps {
    pub steps: u32,
    pub name: String,
}
named!(MaxSteps);

impl MaxSteps {
    pub fn new(steps: u32) -> Self {
        Self { steps, name: "max_steps".into() }
    }
}

impl Check for MaxSteps {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        if ctx.steps <= self.steps {
            return Ok(None);
        }
        Ok(Some(Violation::new(&self.name, format!("this took {} steps, the limit is {}", ctx.steps, self.steps), "")))
    }
}

/// A soft spend ceiling, checked at the end rather than enforced mid-run.
#[derive(Debug, Clone)]
pub struct MaxCost {
    pub usd: f64,
    pub name: String,
}
named!(MaxCost);

impl MaxCost {
    pub fn new(usd: f64) -> Self {
        Self { usd, name: "max_cost".into() }
    }
}

impl Check for MaxCost {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        if ctx.cost_usd <= self.usd {
            return Ok(None);
        }
        Ok(Some(Violation::new(&self.name, format!("this cost ${:.4}, the limit is ${:.4}", ctx.cost_usd, self.usd), "")))
    }
}

// --- your own -------------------------------------------------------------------

/// What a [`Custom`] check's function returns: `true`, `false`, or
/// `(false, "what is wrong")`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    pub passed: bool,
    pub detail: String,
}

impl From<bool> for Verdict {
    fn from(passed: bool) -> Self {
        Self { passed, detail: String::new() }
    }
}

impl From<(bool, String)> for Verdict {
    fn from((passed, detail): (bool, String)) -> Self {
        Self { passed, detail }
    }
}

impl From<(bool, &str)> for Verdict {
    fn from((passed, detail): (bool, &str)) -> Self {
        Self { passed, detail: detail.into() }
    }
}

type CustomFn = Arc<dyn Fn(&CompletionContext) -> Result<Verdict> + Send + Sync>;

/// Your own rule. Return `true`, or `false`, or `(false, "what is wrong")`.
#[derive(Clone)]
pub struct Custom {
    pub name: String,
    pub fix: String,
    f: CustomFn,
}
named!(Custom);

impl Custom {
    pub fn new<V: Into<Verdict>>(f: impl Fn(&CompletionContext) -> V + Send + Sync + 'static) -> Self {
        Self { name: "custom".into(), fix: String::new(), f: Arc::new(move |ctx| Ok(f(ctx).into())) }
    }

    /// A rule that can itself fail; the error is reported as "the check itself failed".
    pub fn fallible<V: Into<Verdict>>(f: impl Fn(&CompletionContext) -> Result<V> + Send + Sync + 'static) -> Self {
        Self { name: "custom".into(), fix: String::new(), f: Arc::new(move |ctx| f(ctx).map(Into::into)) }
    }

    /// What the agent is told to do about it.
    pub fn fix(mut self, fix: impl Into<String>) -> Self {
        self.fix = fix.into();
        self
    }
}

impl fmt::Debug for Custom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Custom").field("name", &self.name).field("fix", &self.fix).finish_non_exhaustive()
    }
}

impl Check for Custom {
    fn name(&self) -> &str {
        &self.name
    }
    fn check(&self, ctx: &CompletionContext) -> Result<Option<Violation>> {
        let verdict = (self.f)(ctx)?;
        if verdict.passed {
            return Ok(None);
        }
        let detail = if verdict.detail.is_empty() { format!("the {} check did not pass", self.name) } else { verdict.detail };
        Ok(Some(Violation::new(&self.name, detail, &self.fix)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx(output: &str) -> CompletionContext {
        CompletionContext { agent: "a".into(), output: output.into(), ..Default::default() }
    }

    fn called(tools: &[&str]) -> CompletionContext {
        CompletionContext { tools_called: tools.iter().map(|t| t.to_string()).collect(), ..ctx("") }
    }

    #[test]
    fn require_tools_objects_when_the_agent_guessed() {
        let check = RequireTools::new(["order_status", "lookup"]);
        assert!(check.check(&called(&["order_status", "lookup"])).unwrap().is_none());
        let v = check.check(&called(&["lookup"])).unwrap().unwrap();
        assert_eq!(v.detail, "you answered without calling order_status");
        assert_eq!(v.line(), "you answered without calling order_status — call order_status and answer from what it returns");
    }

    #[test]
    fn forbid_tools_objects_when_one_was_used() {
        let check = ForbidTools::new(["issue_refund"]);
        assert!(check.check(&called(&["order_status"])).unwrap().is_none());
        let v = check.check(&called(&["issue_refund"])).unwrap().unwrap();
        assert_eq!(v.line(), "you called issue_refund, which is not permitted for this task");
    }

    #[test]
    fn phrase_checks_are_case_insensitive_unless_asked() {
        assert!(MustInclude::new(["Order"]).check(&ctx("your order shipped")).unwrap().is_none());
        let v = MustInclude::new(["Order", "it's"]).case_sensitive(true).check(&ctx("your order shipped")).unwrap().unwrap();
        assert_eq!(v.detail, r#"your answer does not mention 'Order', "it's""#);
        let v = MustNotInclude::new(["guarantee"]).check(&ctx("we guarantee it")).unwrap().unwrap();
        assert_eq!(v.detail, "your answer contains 'guarantee'");
    }

    #[test]
    fn pattern_length_and_placeholder_checks() {
        assert!(MustMatch::new(r"\b\d{4}\b").unwrap().check(&ctx("order 4182")).unwrap().is_none());
        let v = MustMatch::new(r"\b\d{4}\b").unwrap().description("needs an order number").check(&ctx("no number")).unwrap().unwrap();
        assert_eq!(v.detail, "needs an order number");
        assert_eq!(MustMatch::new("x").unwrap().check(&ctx("y")).unwrap().unwrap().detail, "your answer does not match /x/");
        assert!(MustMatch::new("(?<=a)b").is_err(), "look-around is refused up front");
        assert_eq!(MinLength::new(20).check(&ctx(" short ")).unwrap().unwrap().detail, "your answer is 5 characters; this task needs at least 20");
        assert!(NoPlaceholders::default().check(&ctx("Dear [insert name], TODO")).unwrap().is_some());
        assert!(NoPlaceholders::default().check(&ctx("A complete answer.")).unwrap().is_none());
        assert!(NoPlaceholders::default().check(&ctx("todo later")).unwrap().is_some(), "case-insensitive, as Python");
    }

    #[test]
    fn citation_json_and_structured_checks() {
        assert!(RequireCitation::default().check(&ctx("see https://x.test/doc")).unwrap().is_none());
        assert!(RequireCitation::default().check(&ctx("trust me")).unwrap().is_some());
        assert!(RequireJSON::default().check(&ctx(r#"{"a": 1}"#)).unwrap().is_none());
        assert!(RequireJSON::default().check(&ctx("```json\n{\"a\": 1}\n```")).unwrap().is_none());
        assert!(RequireJSON::default().check(&ctx("not json")).unwrap().is_some());
        assert!(RequireStructured::default().check(&CompletionContext { data: json!({"a": 1}), ..ctx("") }).unwrap().is_none());
        assert!(RequireStructured::default().check(&ctx("")).unwrap().is_some());
    }

    #[test]
    fn budget_checks_are_soft_and_checked_at_the_end() {
        let steps = |n| CompletionContext { steps: n, ..ctx("") };
        assert!(MaxSteps::new(3).check(&steps(3)).unwrap().is_none());
        assert_eq!(MaxSteps::new(3).check(&steps(4)).unwrap().unwrap().line(), "this took 4 steps, the limit is 3");
        let v = MaxCost::new(0.5).check(&CompletionContext { cost_usd: 0.75, ..ctx("") }).unwrap().unwrap();
        assert_eq!(v.detail, "this cost $0.7500, the limit is $0.5000");
    }

    #[test]
    fn a_custom_check_can_explain_itself() {
        let check = Custom::new(|_| (false, "the total is missing")).named("has_total");
        let v = check.check(&ctx("x")).unwrap().unwrap();
        assert_eq!((v.check.as_str(), v.detail.as_str()), ("has_total", "the total is missing"));
        assert!(Custom::new(|_| true).check(&ctx("")).unwrap().is_none());
        assert_eq!(Custom::new(|_| false).check(&ctx("")).unwrap().unwrap().detail, "the custom check did not pass");
    }

    #[test]
    fn context_from_a_run_sees_sub_agent_tools() {
        use crate::types::ToolCall;
        let call = |name: &str| ToolCall { id: "c".into(), name: name.into(), args: Default::default(), agent: String::new(), step: 1 };
        let child = RunResult { tool_calls: vec![call("order_status")], ..Default::default() };
        let run = RunResult { tool_calls: vec![call("delegate")], children: vec![child], steps: 3, ..Default::default() };
        let ctx = CompletionContext::from_run("manager", "done", &run);
        assert_eq!(ctx.tools_called, ["delegate", "order_status"]);
        assert_eq!(ctx.steps, 3);
    }
}
