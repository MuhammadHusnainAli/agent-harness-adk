//! Quality evaluation: golden tasks, regression scoring, proof a change is an improvement.
//!
//! ```no_run
//! # async fn demo(agent: agent_harness::Agent) -> agent_harness::Result<()> {
//! use agent_harness::evaluation::{EvalReport, Evaluator, Expect, GoldenTask};
//!
//! let tasks = vec![
//!     GoldenTask::new("refund-window", "Can I refund a 40-day-old order?")
//!         .expect(Expect { contains: vec!["30-day".into(), "no".into()], not_contains: vec!["yes, ".into()], ..Default::default() }),
//!     GoldenTask::new("uses-lookup", "Where is order 4182?")
//!         .expect(Expect { tool_called: Some("order_status".into()), ..Default::default() }),
//! ];
//!
//! let report = Evaluator::new(tasks.clone()).run(&agent, "").await;
//! report.save("baseline.json")?;
//!
//! // after a change:
//! let after = Evaluator::new(tasks).run(&agent, "").await;
//! println!("{}", after.compare(&EvalReport::load("baseline.json")?).render());
//! # Ok(()) }
//! ```
//!
//! Graders are ordinary predicates over the run result, so a check can look at
//! the text, the tools that were called, the structured output, the cost or the
//! number of steps. An LLM judge is available when the answer is genuinely
//! open-ended, but reach for it last: it costs money and it is the least
//! reliable grader you have.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use futures::FutureExt;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::{Agent, RunOptions};
use crate::errors::{Error, Result};
use crate::guardrails::{pct, round_to};
use crate::knowledge::kb::py_repr;
use crate::types::{RunResult, now};

/// Python's `repr()` of a JSON value: `{'id': 'T-1', 'priority': 2}`.
fn py_value(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => match n.as_f64() {
            Some(f) if !n.is_i64() && !n.is_u64() && f.fract() == 0.0 && f.abs() < 1e16 => format!("{f:.1}"),
            _ => n.to_string(),
        },
        Value::String(s) => py_repr(s),
        Value::Array(items) => format!("[{}]", items.iter().map(py_value).collect::<Vec<_>>().join(", ")),
        Value::Object(map) => format!("{{{}}}", map.iter().map(|(k, v)| format!("{}: {}", py_repr(k), py_value(v))).collect::<Vec<_>>().join(", ")),
    }
}

fn yes() -> bool {
    true
}

/// What a good answer looks like. Every field set must hold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Expect {
    pub contains: Vec<String>,
    pub not_contains: Vec<String>,
    pub regex: Option<String>,
    pub equals: Option<String>,
    pub json_equals: Option<Value>,
    pub tool_called: Option<String>,
    pub tools_called: Vec<String>,
    pub tool_not_called: Option<String>,
    pub max_steps: Option<u32>,
    pub max_cost_usd: Option<f64>,
    #[serde(default = "yes")]
    pub no_error: bool,
    pub case_sensitive: bool,
}

impl Default for Expect {
    fn default() -> Self {
        Self {
            contains: vec![],
            not_contains: vec![],
            regex: None,
            equals: None,
            json_equals: None,
            tool_called: None,
            tools_called: vec![],
            tool_not_called: None,
            max_steps: None,
            max_cost_usd: None,
            no_error: true,
            case_sensitive: false,
        }
    }
}

impl Expect {
    /// `(passed, why it failed)`.
    pub fn check(&self, result: &RunResult) -> (bool, String) {
        let norm = |value: &str| if self.case_sensitive { value.to_string() } else { value.to_lowercase() };
        let text = norm(&result.output);

        if self.no_error
            && let Some(error) = &result.error
        {
            return (false, format!("the run errored: {error}"));
        }
        for needle in &self.contains {
            if !text.contains(&norm(needle)) {
                return (false, format!("missing {}", py_repr(needle)));
            }
        }
        for needle in &self.not_contains {
            if text.contains(&norm(needle)) {
                return (false, format!("should not contain {}", py_repr(needle)));
            }
        }
        if let Some(pattern) = self.regex.as_deref().filter(|p| !p.is_empty()) {
            let flags = if self.case_sensitive { "" } else { "(?i)" };
            match Regex::new(&format!("{flags}{pattern}")) {
                Ok(re) if re.is_match(&result.output) => {}
                Ok(_) => return (false, format!("does not match /{pattern}/")),
                Err(e) => return (false, format!("the pattern /{pattern}/ does not compile: {e}")),
            }
        }
        if let Some(equals) = &self.equals
            && norm(equals) != text.trim()
        {
            return (false, format!("expected exactly {}", py_repr(equals)));
        }
        if let Some(expected) = &self.json_equals
            && &result.data != expected
        {
            return (false, format!("structured output was {}", py_value(&result.data)));
        }

        let mut called: Vec<&str> = result.tool_calls.iter().map(|c| c.name.as_str()).collect();
        called.extend(result.children.iter().flat_map(|child| child.tool_calls.iter().map(|c| c.name.as_str())));
        if let Some(tool) = self.tool_called.as_deref().filter(|t| !t.is_empty())
            && !called.contains(&tool)
        {
            let list = if called.is_empty() { "none".to_string() } else { format!("[{}]", called.iter().map(|c| py_repr(c)).collect::<Vec<_>>().join(", ")) };
            return (false, format!("never called {} (called: {list})", py_repr(tool)));
        }
        for name in &self.tools_called {
            if !called.contains(&name.as_str()) {
                return (false, format!("never called {}", py_repr(name)));
            }
        }
        if let Some(tool) = self.tool_not_called.as_deref().filter(|t| !t.is_empty())
            && called.contains(&tool)
        {
            return (false, format!("should not have called {}", py_repr(tool)));
        }

        if let Some(max) = self.max_steps
            && result.steps > max
        {
            return (false, format!("took {} steps, budget was {max}", result.steps));
        }
        if let Some(max) = self.max_cost_usd
            && result.cost_usd() > max
        {
            return (false, format!("cost ${:.4}, budget was ${max:.4}", result.cost_usd()));
        }
        (true, String::new())
    }
}

/// What a grader says: `true`, `false`, or `(passed, "why")`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradeVerdict {
    pub passed: bool,
    /// `None` for a bare bool: a failure then reads "the custom grader said no".
    pub detail: Option<String>,
}

impl From<bool> for GradeVerdict {
    fn from(passed: bool) -> Self {
        Self { passed, detail: None }
    }
}

impl From<(bool, String)> for GradeVerdict {
    fn from((passed, detail): (bool, String)) -> Self {
        Self { passed, detail: Some(detail) }
    }
}

impl From<(bool, &str)> for GradeVerdict {
    fn from((passed, detail): (bool, &str)) -> Self {
        Self { passed, detail: Some(detail.into()) }
    }
}

type GradeFn = dyn Fn(&RunResult, &GoldenTask) -> BoxFuture<'static, GradeVerdict> + Send + Sync;

/// A custom grader, run after the expectations hold.
#[derive(Clone)]
pub struct Grader(Arc<GradeFn>);

impl Grader {
    /// A plain predicate.
    pub fn new<V: Into<GradeVerdict>>(f: impl Fn(&RunResult, &GoldenTask) -> V + Send + Sync + 'static) -> Self {
        Self(Arc::new(move |result, task| {
            let verdict = f(result, task).into();
            async move { verdict }.boxed()
        }))
    }

    /// A predicate that has to wait on something (a model, typically). It gets
    /// its own copies of the result and the task.
    pub fn from_async<V, F, Fut>(f: F) -> Self
    where
        V: Into<GradeVerdict> + 'static,
        F: Fn(RunResult, GoldenTask) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = V> + Send + 'static,
    {
        Self(Arc::new(move |result, task| f(result.clone(), task.clone()).map(Into::into).boxed()))
    }

    pub async fn grade(&self, result: &RunResult, task: &GoldenTask) -> GradeVerdict {
        (self.0)(result, task).await
    }
}

impl fmt::Debug for Grader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Grader(..)")
    }
}

/// One case in the suite: an input, and what a good answer looks like.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenTask {
    pub id: String,
    pub input: String,
    #[serde(default)]
    pub expect: Expect,
    #[serde(default = "one")]
    pub weight: f64,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub notes: String,
    #[serde(skip)]
    pub grader: Option<Grader>,
}

fn one() -> f64 {
    1.0
}

impl GoldenTask {
    pub fn new(id: impl Into<String>, input: impl Into<String>) -> Self {
        Self { id: id.into(), input: input.into(), expect: Expect::default(), weight: 1.0, tags: vec![], notes: String::new(), grader: None }
    }

    pub fn expect(mut self, expect: Expect) -> Self {
        self.expect = expect;
        self
    }

    pub fn weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    pub fn tags<I: IntoIterator<Item = S>, S: Into<String>>(mut self, tags: I) -> Self {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    pub fn notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = notes.into();
        self
    }

    pub fn grader(mut self, grader: Grader) -> Self {
        self.grader = Some(grader);
        self
    }
}

/// How one task went.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskOutcome {
    pub id: String,
    pub passed: bool,
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub detail: String,
    #[serde(default)]
    pub output: String,
    #[serde(default)]
    pub steps: u32,
    #[serde(default)]
    pub cost_usd: f64,
    #[serde(default)]
    pub duration_s: f64,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub error: Option<String>,
}

/// The suite's result. Save it and you have a baseline to score against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EvalReport {
    pub agent: String,
    pub model: String,
    pub label: String,
    pub ts: f64,
    pub outcomes: Vec<TaskOutcome>,
    pub duration_s: f64,
}

impl Default for EvalReport {
    fn default() -> Self {
        Self { agent: String::new(), model: String::new(), label: String::new(), ts: now(), outcomes: vec![], duration_s: 0.0 }
    }
}

impl EvalReport {
    pub fn total(&self) -> usize {
        self.outcomes.len()
    }

    pub fn passed(&self) -> Vec<&TaskOutcome> {
        self.outcomes.iter().filter(|o| o.passed).collect()
    }

    pub fn failed(&self) -> Vec<&TaskOutcome> {
        self.outcomes.iter().filter(|o| !o.passed).collect()
    }

    /// Weighted pass rate, 0.0 to 1.0.
    pub fn score(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.0;
        }
        round_to(self.outcomes.iter().map(|o| o.score).sum::<f64>() / self.outcomes.len() as f64, 4)
    }

    pub fn cost_usd(&self) -> f64 {
        round_to(self.outcomes.iter().map(|o| o.cost_usd).sum(), 6)
    }

    /// The score of each tag's tasks.
    pub fn by_tag(&self) -> IndexMap<String, f64> {
        let mut buckets: IndexMap<String, Vec<f64>> = IndexMap::new();
        for outcome in &self.outcomes {
            for tag in &outcome.tags {
                buckets.entry(tag.clone()).or_default().push(outcome.score);
            }
        }
        buckets.into_iter().map(|(tag, v)| (tag, round_to(v.iter().sum::<f64>() / v.len() as f64, 4))).collect()
    }

    pub fn render(&self) -> String {
        let mut lines = vec![format!(
            "{}: {}/{} passed · score {} · ${:.4} · {:.1}s",
            if self.label.is_empty() { &self.agent } else { &self.label },
            self.passed().len(),
            self.total(),
            pct(self.score(), 2),
            self.cost_usd(),
            self.duration_s
        )];
        for outcome in &self.outcomes {
            let mark = if outcome.passed { "PASS" } else { "FAIL" };
            let detail = if outcome.detail.is_empty() { String::new() } else { format!(" — {}", outcome.detail) };
            lines.push(format!("  [{mark}] {}{detail}", outcome.id));
        }
        lines.join("\n")
    }

    /// Write the report as JSON (creating the directory), and return the path.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<PathBuf> {
        let file = path.as_ref().to_path_buf();
        if let Some(parent) = file.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(self).map_err(|e| Error::Io(e.to_string()))?;
        std::fs::write(&file, text)?;
        Ok(file)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())?;
        serde_json::from_str(&text).map_err(|e| Error::config(format!("{} is not an eval report: {e}", path.as_ref().display())))
    }

    /// Score this run against a saved baseline.
    pub fn compare(&self, baseline: &EvalReport) -> Comparison {
        let before: HashMap<&str, &TaskOutcome> = baseline.outcomes.iter().map(|o| (o.id.as_str(), o)).collect();
        let (mut fixed, mut regressed, mut still_failing) = (vec![], vec![], vec![]);
        for outcome in &self.outcomes {
            let Some(was) = before.get(outcome.id.as_str()) else { continue };
            if outcome.passed && !was.passed {
                fixed.push(outcome.id.clone());
            } else if !outcome.passed && was.passed {
                regressed.push(outcome.id.clone());
            } else if !outcome.passed {
                still_failing.push(outcome.id.clone());
            }
        }
        let now_ids: HashSet<&str> = self.outcomes.iter().map(|o| o.id.as_str()).collect();
        let mut seen = HashSet::new();
        Comparison {
            baseline_label: if baseline.label.is_empty() { "baseline".into() } else { baseline.label.clone() },
            candidate_label: if self.label.is_empty() { "candidate".into() } else { self.label.clone() },
            baseline_score: baseline.score(),
            candidate_score: self.score(),
            fixed,
            regressed,
            still_failing,
            new_tasks: self.outcomes.iter().filter(|o| !before.contains_key(o.id.as_str())).map(|o| o.id.clone()).collect(),
            missing_tasks: baseline.outcomes.iter().filter(|o| !now_ids.contains(o.id.as_str()) && seen.insert(o.id.as_str())).map(|o| o.id.clone()).collect(),
            baseline_cost: baseline.cost_usd(),
            candidate_cost: self.cost_usd(),
        }
    }
}

/// Did the change help? The only question an eval exists to answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Comparison {
    pub baseline_label: String,
    pub candidate_label: String,
    pub baseline_score: f64,
    pub candidate_score: f64,
    pub fixed: Vec<String>,
    pub regressed: Vec<String>,
    pub still_failing: Vec<String>,
    pub new_tasks: Vec<String>,
    pub missing_tasks: Vec<String>,
    pub baseline_cost: f64,
    pub candidate_cost: f64,
}

impl Default for Comparison {
    fn default() -> Self {
        Self {
            baseline_label: "baseline".into(),
            candidate_label: "candidate".into(),
            baseline_score: 0.0,
            candidate_score: 0.0,
            fixed: vec![],
            regressed: vec![],
            still_failing: vec![],
            new_tasks: vec![],
            missing_tasks: vec![],
            baseline_cost: 0.0,
            candidate_cost: 0.0,
        }
    }
}

impl Comparison {
    pub fn delta(&self) -> f64 {
        round_to(self.candidate_score - self.baseline_score, 4)
    }

    /// Better overall, and nothing that used to work is broken.
    pub fn improved(&self) -> bool {
        self.delta() > 0.0 && self.regressed.is_empty()
    }

    /// `"regressed"`, `"improved"`, `"worse"` or `"unchanged"`.
    pub fn verdict(&self) -> &'static str {
        let delta = self.delta();
        if !self.regressed.is_empty() {
            "regressed"
        } else if delta > 0.0 {
            "improved"
        } else if delta < 0.0 {
            "worse"
        } else {
            "unchanged"
        }
    }

    pub fn render(&self) -> String {
        let mut lines = vec![
            format!("{}: {} → {} ({:+.2}%)", self.verdict().to_uppercase(), pct(self.baseline_score, 2), pct(self.candidate_score, 2), self.delta() * 100.0),
            format!("cost: ${:.4} → ${:.4}", self.baseline_cost, self.candidate_cost),
        ];
        for (label, ids) in [
            ("  fixed:        ", &self.fixed),
            ("  REGRESSED:    ", &self.regressed),
            ("  still failing: ", &self.still_failing),
            ("  new:          ", &self.new_tasks),
            ("  missing:      ", &self.missing_tasks),
        ] {
            if !ids.is_empty() {
                lines.push(format!("{label}{}", ids.join(", ")));
            }
        }
        lines.join("\n")
    }
}

/// What an [`Evaluator`] runs tasks against: an [`Agent`], or your own wrapper.
#[async_trait]
pub trait EvalSubject: Send + Sync {
    fn name(&self) -> String;
    fn model(&self) -> String;
    /// Run one task. `clean` means with no conversation history.
    async fn run_task(&self, input: &str, clean: bool) -> Result<RunResult>;
}

#[async_trait]
impl EvalSubject for Agent {
    fn name(&self) -> String {
        Agent::name(self).to_string()
    }
    fn model(&self) -> String {
        Agent::model(self).to_string()
    }
    async fn run_task(&self, input: &str, clean: bool) -> Result<RunResult> {
        self.run_with(input, RunOptions { messages: clean.then(Vec::new), ..Default::default() }).await
    }
}

/// Runs a suite of golden tasks against an agent.
#[derive(Debug, Clone)]
pub struct Evaluator {
    pub tasks: Vec<GoldenTask>,
    pub concurrency: usize,
    /// Each task starts with no conversation history.
    pub clean_run: bool,
}

impl Evaluator {
    /// Four tasks at a time, each on a clean history.
    pub fn new(tasks: Vec<GoldenTask>) -> Self {
        Self { tasks, concurrency: 4, clean_run: true }
    }

    pub fn concurrency(mut self, n: usize) -> Self {
        self.concurrency = n;
        self
    }

    pub fn clean_run(mut self, on: bool) -> Self {
        self.clean_run = on;
        self
    }

    // ---- loading -------------------------------------------------------

    /// Load tasks from JSON (a list, or `{"tasks": [...]}`) or JSONL.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = path.as_ref();
        let text = std::fs::read_to_string(file)?;
        let bad = |e: serde_json::Error| Error::config(format!("{}: {e}", file.display()));
        let tasks: Vec<GoldenTask> = if file.extension().is_some_and(|e| e == "jsonl") {
            text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).map_err(bad)).collect::<Result<_>>()?
        } else {
            let mut blob: Value = serde_json::from_str(&text).map_err(bad)?;
            if let Value::Object(map) = &mut blob {
                blob = map.remove("tasks").ok_or_else(|| Error::config(format!("{}: no \"tasks\" list", file.display())))?;
            }
            serde_json::from_value(blob).map_err(bad)?
        };
        Ok(Self::new(tasks))
    }

    pub fn add(&mut self, task: GoldenTask) -> &GoldenTask {
        self.tasks.push(task);
        self.tasks.last().expect("just pushed")
    }

    /// The tasks carrying any of these tags.
    pub fn select<I: IntoIterator<Item = S>, S: Into<String>>(&self, tags: I) -> Evaluator {
        let wanted: HashSet<String> = tags.into_iter().map(Into::into).collect();
        Evaluator { tasks: self.tasks.iter().filter(|t| t.tags.iter().any(|g| wanted.contains(g))).cloned().collect(), ..self.clone() }
    }

    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    // ---- running -------------------------------------------------------

    /// Run every task. Failures are recorded, never raised.
    pub async fn run(&self, agent: &dyn EvalSubject, label: &str) -> EvalReport {
        let started = Instant::now();
        let semaphore = tokio::sync::Semaphore::new(self.concurrency.max(1));

        let mut outcomes = futures::future::join_all(self.tasks.iter().map(|task| self.run_one(agent, &semaphore, task))).await;

        let by_id: HashMap<&str, f64> = self.tasks.iter().map(|t| (t.id.as_str(), t.weight)).collect();
        let total: f64 = by_id.values().sum();
        let total_weight = if total == 0.0 { 1.0 } else { total };
        // Normalise so weights actually matter in the headline score.
        let n = outcomes.len() as f64;
        for outcome in outcomes.iter_mut().filter(|o| o.passed) {
            outcome.score = round_to(by_id.get(outcome.id.as_str()).copied().unwrap_or(1.0) * n / total_weight, 4);
        }

        EvalReport {
            agent: agent.name(),
            model: agent.model(),
            label: label.to_string(),
            ts: now(),
            outcomes,
            duration_s: round_to(started.elapsed().as_secs_f64(), 3),
        }
    }

    async fn run_one(&self, agent: &dyn EvalSubject, semaphore: &tokio::sync::Semaphore, task: &GoldenTask) -> TaskOutcome {
        let _permit = semaphore.acquire().await.expect("semaphore open");
        let task_started = Instant::now();
        let elapsed = || round_to(task_started.elapsed().as_secs_f64(), 3);
        let result = match agent.run_task(&task.input, self.clean_run).await {
            Ok(result) => result,
            // A crash is a failed task, not a failed suite.
            Err(e) => {
                let why = format!("{}: {e}", e.kind_name());
                return TaskOutcome {
                    id: task.id.clone(),
                    passed: false,
                    score: 0.0,
                    detail: why.clone(),
                    output: String::new(),
                    steps: 0,
                    cost_usd: 0.0,
                    duration_s: elapsed(),
                    tags: task.tags.clone(),
                    error: Some(why),
                };
            }
        };

        let (mut passed, mut detail) = task.expect.check(&result);
        if passed && let Some(grader) = &task.grader {
            let verdict = grader.grade(&result, task).await;
            passed = verdict.passed;
            detail = verdict.detail.unwrap_or_else(|| if passed { String::new() } else { "the custom grader said no".into() });
        }
        TaskOutcome {
            id: task.id.clone(),
            passed,
            score: if passed { task.weight } else { 0.0 },
            detail,
            output: result.output.chars().take(2000).collect(),
            steps: result.steps,
            cost_usd: result.cost_usd(),
            duration_s: elapsed(),
            tags: task.tags.clone(),
            error: result.error.clone(),
        }
    }
}

/// A grader that asks a model. Use it last — it costs money and it can be wrong.
///
/// `judge_agent` scores the answer; `rubric` is what a passing answer must do.
pub fn llm_judge(judge_agent: Agent, rubric: impl Into<String>) -> Grader {
    let rubric: Arc<str> = rubric.into().into();
    Grader::from_async(move |result: RunResult, task: GoldenTask| {
        let (judge, rubric) = (judge_agent.clone(), rubric.clone());
        async move {
            let prompt = format!(
                "You are grading another agent's answer. Be strict and brief.\n\nTASK\n{}\n\nRUBRIC\n{rubric}\n\nANSWER\n{}\n\n\
                 Reply with PASS or FAIL on the first line, then one line of reason.",
                task.input, result.output
            );
            let verdict = match judge.run_with(prompt, RunOptions { messages: Some(vec![]), ..Default::default() }).await {
                Ok(verdict) => verdict,
                Err(e) => return (false, format!("{}: {e}", e.kind_name())),
            };
            let text = verdict.output.trim();
            let passed = text.to_uppercase().starts_with("PASS");
            let reason = text.lines().skip(1).collect::<Vec<_>>().join("\n").trim().to_string();
            let detail = if passed {
                String::new()
            } else if reason.is_empty() {
                "the judge said FAIL".into()
            } else {
                reason
            };
            (passed, detail)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_providers::{FakeProvider, Script, tool_call};
    use crate::tools::{Tool, ToolContext};
    use crate::types::{ToolCall, Usage};
    use schemars::JsonSchema;
    use serde_json::json;

    /// Look up an order.
    #[derive(Deserialize, JsonSchema)]
    struct OrderStatus {
        /// The order number.
        #[allow(dead_code)]
        order_id: String,
    }

    fn order_status() -> Tool {
        Tool::new("order_status", |_: OrderStatus, _: ToolContext| async move { Ok("shipped Thursday") })
    }

    fn agent_for(script: Vec<Script>, looping: bool) -> Agent {
        let mut p = FakeProvider::new(script);
        if looping {
            p = p.looping();
        }
        Agent::builder("support").provider(Arc::new(p)).tool(order_status()).memory(false).build().unwrap()
    }

    fn texts(items: &[&str]) -> Vec<Script> {
        items.iter().map(|s| Script::from(*s)).collect()
    }

    fn result(output: &str) -> RunResult {
        RunResult { output: output.into(), steps: 1, ..Default::default() }
    }

    fn with_tools(output: &str, tools: &[&str]) -> RunResult {
        let calls = tools.iter().map(|t| ToolCall { id: "c".into(), name: t.to_string(), args: Default::default(), agent: String::new(), step: 1 }).collect();
        RunResult { tool_calls: calls, ..result(output) }
    }

    fn contains(items: &[&str]) -> Expect {
        Expect { contains: items.iter().map(|s| s.to_string()).collect(), ..Default::default() }
    }

    // --- the expectations themselves

    #[test]
    fn contains_and_not_contains() {
        let expect = Expect { not_contains: vec!["yes, of course".into()], ..contains(&["30-day"]) };
        assert!(expect.check(&result("Our 30-day window has passed.")).0);
        assert_eq!(expect.check(&result("Yes, of course — refunded.")), (false, "missing '30-day'".to_string()));
    }

    #[test]
    fn matching_is_case_insensitive_unless_you_ask() {
        assert!(contains(&["SHIPPED"]).check(&result("it shipped")).0);
        assert!(!Expect { case_sensitive: true, ..contains(&["SHIPPED"]) }.check(&result("it shipped")).0);
    }

    #[test]
    fn regex_and_exact_equality() {
        let re = Expect { regex: Some(r"\b\d{4}\b".into()), ..Default::default() };
        assert!(re.check(&result("order 4182 shipped")).0);
        assert_eq!(re.check(&result("no number here")).1, r"does not match /\b\d{4}\b/");
        let eq = Expect { equals: Some("yes".into()), ..Default::default() };
        assert!(eq.check(&result("Yes")).0);
        assert_eq!(eq.check(&result("yes indeed")).1, "expected exactly 'yes'");
    }

    #[test]
    fn tool_expectations_look_at_the_run_not_the_text() {
        let expect = Expect { tool_called: Some("order_status".into()), ..Default::default() };
        assert!(expect.check(&with_tools("shipped", &["order_status"])).0);
        assert_eq!(expect.check(&result("shipped")).1, "never called 'order_status' (called: none)");
        assert_eq!(expect.check(&with_tools("x", &["a", "b"])).1, "never called 'order_status' (called: ['a', 'b'])");
        assert!(Expect { tool_not_called: Some("wire_money".into()), ..Default::default() }.check(&with_tools("done", &["order_status"])).0);
    }

    #[test]
    fn tool_expectations_see_sub_agent_calls_too() {
        let mut parent = result("done");
        parent.children.push(with_tools("child", &["order_status"]));
        assert!(Expect { tool_called: Some("order_status".into()), ..Default::default() }.check(&parent).0);
    }

    #[test]
    fn budgets_are_part_of_quality() {
        let costly = |steps, cost| RunResult { steps, usage: Usage { cost_usd: cost, ..Default::default() }, ..result("fine") };
        assert_eq!(Expect { max_steps: Some(2), ..Default::default() }.check(&costly(5, 0.0)).1, "took 5 steps, budget was 2");
        assert_eq!(Expect { max_cost_usd: Some(0.01), ..Default::default() }.check(&costly(1, 0.5)).1, "cost $0.5000, budget was $0.0100");
        assert!(Expect { max_steps: Some(5), max_cost_usd: Some(1.0), ..Default::default() }.check(&costly(3, 0.2)).0);
    }

    #[test]
    fn an_errored_run_fails_by_default() {
        let errored = RunResult { error: Some("BudgetExceeded: no money".into()), ..result("") };
        assert_eq!(Expect::default().check(&errored).1, "the run errored: BudgetExceeded: no money");
        assert!(Expect { no_error: false, ..Default::default() }.check(&errored).0);
    }

    #[test]
    fn structured_output_can_be_compared() {
        let good = RunResult { data: json!({"id": "T-1", "priority": 2}), ..result("{}") };
        assert!(Expect { json_equals: Some(json!({"id": "T-1", "priority": 2})), ..Default::default() }.check(&good).0);
        let (passed, why) = Expect { json_equals: Some(json!({"id": "T-2", "priority": 2})), ..Default::default() }.check(&good);
        assert!(!passed);
        assert_eq!(why, "structured output was {'id': 'T-1', 'priority': 2}");
    }

    // --- running a suite

    #[tokio::test]
    async fn a_suite_runs_and_scores() {
        let agent = agent_for(
            vec![tool_call("order_status", json!({"order_id": "4182"})), "Order 4182 shipped Thursday.".into(), "I cannot help with that.".into()],
            false,
        );
        let report = Evaluator::new(vec![
            GoldenTask::new("lookup", "Where is order 4182?")
                .expect(Expect { tool_called: Some("order_status".into()), ..contains(&["Thursday"]) })
                .tags(["orders"]),
            GoldenTask::new("refusal", "What is the meaning of life?").expect(contains(&["cannot"])).tags(["scope"]),
        ])
        .concurrency(1)
        .run(&agent, "v1")
        .await;

        assert_eq!(report.total(), 2);
        assert_eq!(report.passed().len(), 2, "{}", report.render());
        assert_eq!(report.score(), 1.0);
        assert_eq!(report.by_tag(), IndexMap::from([("orders".to_string(), 1.0), ("scope".to_string(), 1.0)]));
        assert!(report.render().starts_with("v1: 2/2 passed · score 100.00% · $"), "{}", report.render());
        assert_eq!(report.agent, "support");
    }

    #[tokio::test]
    async fn a_failing_task_records_why_without_failing_the_suite() {
        let agent = agent_for(texts(&["the wrong answer"]), true);
        let report =
            Evaluator::new(vec![GoldenTask::new("a", "q").expect(contains(&["the right answer"])), GoldenTask::new("b", "q").expect(contains(&["wrong"]))])
                .concurrency(1)
                .run(&agent, "")
                .await;
        let failed = report.failed();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].id, "a");
        assert_eq!(failed[0].detail, "missing 'the right answer'");
        assert_eq!(report.score(), 0.5);
        assert!(report.render().contains("  [FAIL] a — missing 'the right answer'"));
    }

    struct Exploding;

    #[async_trait]
    impl EvalSubject for Exploding {
        fn name(&self) -> String {
            "boom".into()
        }
        fn model(&self) -> String {
            String::new()
        }
        async fn run_task(&self, _: &str, _: bool) -> Result<RunResult> {
            Err(Error::Workflow("the agent exploded".into()))
        }
    }

    #[tokio::test]
    async fn a_crashing_agent_is_a_failed_task_not_a_crashed_suite() {
        let report = Evaluator::new(vec![GoldenTask::new("t", "go")]).run(&Exploding, "").await;
        assert_eq!(report.failed()[0].error.as_deref(), Some("WorkflowError: the agent exploded"));
        assert_eq!(report.score(), 0.0);
    }

    #[tokio::test]
    async fn weights_move_the_headline_score() {
        let agent = agent_for(texts(&["right", "wrong"]), false);
        let report = Evaluator::new(vec![
            GoldenTask::new("important", "q1").expect(contains(&["right"])).weight(9.0),
            GoldenTask::new("minor", "q2").expect(contains(&["right"])).weight(1.0),
        ])
        .concurrency(1)
        .run(&agent, "")
        .await;
        assert_eq!(report.passed().iter().map(|o| o.id.as_str()).collect::<Vec<_>>(), ["important"]);
        assert_eq!(report.score(), 0.9);
    }

    #[tokio::test]
    async fn a_custom_grader_runs_after_the_expectations() {
        let grader = Grader::new(|run: &RunResult, _: &GoldenTask| (run.output.contains("42"), "the number is missing"));
        let agent = agent_for(texts(&["the answer is 42", "no number here"]), false);
        let report = Evaluator::new(vec![
            GoldenTask::new("has-number", "q1").grader(grader.clone()),
            GoldenTask::new("no-number", "q2").grader(grader),
            GoldenTask::new("bare", "q3").grader(Grader::new(|_: &RunResult, _: &GoldenTask| false)),
        ])
        .concurrency(1)
        .run(&agent, "")
        .await;
        assert_eq!(report.passed().iter().map(|o| o.id.as_str()).collect::<Vec<_>>(), ["has-number"]);
        assert_eq!(report.failed()[0].detail, "the number is missing");
        assert_eq!(report.failed()[1].detail, "the custom grader said no");
    }

    #[tokio::test]
    async fn an_llm_judge_can_grade_open_ended_answers() {
        let judge_provider = Arc::new(FakeProvider::new(texts(&["PASS\nit answers the question"])));
        let judge = Agent::builder("judge").provider(judge_provider.clone()).memory(false).build().unwrap();

        let task = || GoldenTask::new("open", "explain it").grader(llm_judge(judge.clone(), "must actually explain the thing"));
        let report = Evaluator::new(vec![task()]).run(&agent_for(texts(&["a thoughtful open-ended answer"]), false), "").await;
        assert_eq!(report.score(), 1.0);
        let asked = judge_provider.requests()[0].messages.last().unwrap().text();
        assert!(asked.contains("RUBRIC\nmust actually explain the thing\n\nANSWER\na thoughtful open-ended answer"), "{asked}");

        judge_provider.queue(texts(&["FAIL\nit dodged the question"]));
        let report = Evaluator::new(vec![task()]).run(&agent_for(texts(&["waffle"]), false), "").await;
        assert_eq!(report.score(), 0.0);
        assert_eq!(report.failed()[0].detail, "it dodged the question");
    }

    #[test]
    fn tasks_can_be_selected_by_tag() {
        let suite = Evaluator::new(vec![GoldenTask::new("a", "q").tags(["fast"]), GoldenTask::new("b", "q").tags(["slow"])]);
        assert_eq!(suite.select(["fast"]).tasks.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(), ["a"]);
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ah-eval-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_suite_loads_from_a_file() {
        let dir = scratch("load");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("golden.json");
        std::fs::write(&path, json!({"tasks": [{"id": "one", "input": "q", "expect": {"contains": ["x"]}, "tags": ["t"]}]}).to_string()).unwrap();
        let suite = Evaluator::from_file(&path).unwrap();
        assert_eq!(suite.len(), 1);
        assert_eq!(suite.tasks[0].expect.contains, ["x"]);
        assert!(suite.tasks[0].expect.no_error);

        let lines = dir.join("golden.jsonl");
        std::fs::write(&lines, "{\"id\": \"a\", \"input\": \"q\"}\n\n{\"id\": \"b\", \"input\": \"q\", \"weight\": 2}\n").unwrap();
        let suite = Evaluator::from_file(&lines).unwrap();
        assert_eq!((suite.len(), suite.tasks[1].weight), (2, 2.0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- regression scoring

    #[tokio::test]
    async fn comparing_against_a_baseline_names_what_broke() {
        let tasks = vec![GoldenTask::new("a", "q1").expect(contains(&["good"])), GoldenTask::new("b", "q2").expect(contains(&["good"]))];
        let baseline = Evaluator::new(tasks.clone()).concurrency(1).run(&agent_for(texts(&["good", "bad"]), false), "before").await;
        let dir = scratch("compare");
        let saved = baseline.save(dir.join("nested").join("baseline.json")).unwrap();

        // A change fixes b but breaks a.
        let candidate = Evaluator::new(tasks).concurrency(1).run(&agent_for(texts(&["bad", "good"]), false), "after").await;
        let comparison = candidate.compare(&EvalReport::load(&saved).unwrap());
        assert_eq!(comparison.fixed, ["b"]);
        assert_eq!(comparison.regressed, ["a"]);
        assert_eq!(comparison.verdict(), "regressed");
        assert!(!comparison.improved(), "a regression is never an improvement");
        assert_eq!(comparison.render(), "REGRESSED: 50.00% → 50.00% (+0.00%)\ncost: $0.0000 → $0.0000\n  fixed:        b\n  REGRESSED:    a");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_clean_improvement_is_reported_as_one() {
        let tasks = vec![GoldenTask::new("a", "q").expect(contains(&["good"]))];
        let baseline = Evaluator::new(tasks.clone()).run(&agent_for(texts(&["bad"]), false), "before").await;
        let candidate = Evaluator::new(tasks).run(&agent_for(texts(&["good"]), false), "after").await;
        let comparison = candidate.compare(&baseline);
        assert_eq!(comparison.verdict(), "improved");
        assert!(comparison.improved());
        assert_eq!(comparison.fixed, ["a"]);
        assert!(comparison.regressed.is_empty());
        assert!(comparison.delta() > 0.0);
        assert!(comparison.render().starts_with("IMPROVED: 0.00% → 100.00% (+100.00%)"));
    }

    #[tokio::test]
    async fn added_and_removed_tasks_are_called_out() {
        let old = Evaluator::new(vec![GoldenTask::new("a", "q")]).run(&agent_for(texts(&["x"]), false), "").await;
        let new = Evaluator::new(vec![GoldenTask::new("b", "q")]).run(&agent_for(texts(&["x"]), false), "").await;
        let comparison = new.compare(&old);
        assert_eq!((comparison.new_tasks, comparison.missing_tasks), (vec!["b".to_string()], vec!["a".to_string()]));
    }

    #[test]
    fn an_unchanged_score_is_not_an_improvement() {
        let same = EvalReport::default();
        assert_eq!(same.compare(&same).verdict(), "unchanged");
    }
}
