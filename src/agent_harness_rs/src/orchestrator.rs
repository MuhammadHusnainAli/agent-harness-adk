//! The accountable manager: plan → staff → run → consolidate → review.
//!
//! One orchestrator owns the job. It turns a request into a costed plan with
//! its acceptance tests written first, decides for every task whether to reuse
//! a sub-agent from the bench or have the factory write a new one, runs what
//! can run in parallel, checks each hand-back against what its task was to be
//! done by, plans a task again when it comes back short, consolidates, and only
//! then accepts the work through independent critics. The job is saved as it
//! goes, so one cut short is picked up by [`Orchestrator::resume`].
//!
//! ```no_run
//! # async fn demo() -> agent_harness::Result<()> {
//! use agent_harness::orchestrator::Orchestrator;
//! let boss = Orchestrator::builder("lead").max_concurrency(4).critics(2).build()?;
//! let result = boss.run("Compare the three vendors' SLAs and recommend one.").await?;
//! println!("{} — ${:.4}", result.output, result.cost_usd());
//! # Ok(()) }
//! ```

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::future::join_all;
use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::sync::Semaphore;

use crate::agent::{Agent, OutputContract, RunOptions, build_agent};
use crate::errors::{Error, Result};
use crate::harness::Harness;
use crate::llm_providers::base::{Provider, model_info};
use crate::prompts::Prompt;
use crate::runtime::{Budget, BudgetGuard, Session};
use crate::subagents::{Bench, SubAgentFactory, SubAgentSpec};
use crate::tools::Tool;
use crate::types::{Artifact, RunResult, StopReason, Usage, new_id, now};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    #[default]
    Pending,
    Running,
    Done,
    Failed,
    Blocked,
    Skipped,
    Replaced,
}

/// One unit of work a single sub-agent can finish alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Task {
    pub id: String,
    pub statement: String,
    pub depends_on: Vec<String>,
    pub done_when: String,
    pub agent: String,
    pub reused: bool,
    pub status: TaskStatus,
    pub output: String,
    /// Kept when a task fails or runs out of time.
    pub partial: String,
    pub error: Option<String>,
    pub cost_usd: f64,
    pub attempts: u32,
    /// Checked against `done_when`: true met, false could not be judged, none nothing to check.
    pub checked: Option<bool>,
    /// What the check found missing, the last time it found something.
    pub missing: String,
    /// The task this one was planned in place of.
    pub replaces: String,
}

impl Default for Task {
    fn default() -> Self {
        Self {
            id: new_id("task"),
            statement: String::new(),
            depends_on: vec![],
            done_when: String::new(),
            agent: String::new(),
            reused: true,
            status: TaskStatus::Pending,
            output: String::new(),
            partial: String::new(),
            error: None,
            cost_usd: 0.0,
            attempts: 0,
            checked: None,
            missing: String::new(),
            replaces: String::new(),
        }
    }
}

impl Task {
    pub fn new(id: &str, statement: &str) -> Self {
        Self { id: id.into(), statement: statement.into(), ..Default::default() }
    }

    /// What consolidation can work with — the result, or what got done.
    pub fn usable_output(&self) -> &str {
        if self.output.is_empty() { &self.partial } else { &self.output }
    }
}

/// A costed plan: the goal, how we will know it is done, and the task graph.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Plan {
    pub goal: String,
    pub definition_of_done: Vec<String>,
    pub constraints: Vec<String>,
    pub tasks: Vec<Task>,
    pub estimate_usd: f64,
    /// The most tasks that will run at one time.
    pub parallelism: usize,
    pub notes: String,
}

impl Plan {
    pub fn by_id(&self, id: &str) -> Option<&Task> {
        self.tasks.iter().find(|t| t.id == id)
    }

    fn by_id_mut(&mut self, id: &str) -> Option<&mut Task> {
        self.tasks.iter_mut().find(|t| t.id == id)
    }

    /// Tasks grouped into rounds that can each run in parallel.
    pub fn waves(&self) -> Vec<Vec<&Task>> {
        let mut done: Vec<&str> = Vec::new();
        let mut remaining: Vec<&Task> = self.tasks.iter().collect();
        let mut out = Vec::new();
        while !remaining.is_empty() {
            let ready: Vec<&Task> = remaining.iter().copied().filter(|t| t.depends_on.iter().all(|d| done.contains(&d.as_str()))).collect();
            if ready.is_empty() {
                out.push(remaining); // a cycle or a dangling dependency — run the rest serially
                break;
            }
            done.extend(ready.iter().map(|t| t.id.as_str()));
            remaining.retain(|t| !done.contains(&t.id.as_str()));
            out.push(ready);
        }
        out
    }

    pub fn render(&self) -> String {
        let mut lines = vec![format!("Goal: {}", self.goal)];
        if !self.definition_of_done.is_empty() {
            let dod: Vec<String> = self.definition_of_done.iter().map(|d| format!("  - {d}")).collect();
            lines.push(format!("Definition of done:\n{}", dod.join("\n")));
        }
        for t in &self.tasks {
            let deps = if t.depends_on.is_empty() { String::new() } else { format!(" (after {})", t.depends_on.join(", ")) };
            lines.push(format!("  [{}] {}{deps}", t.id, t.statement));
        }
        if self.estimate_usd > 0.0 {
            lines.push(format!("Estimated cost: ${:.3}", self.estimate_usd));
        }
        lines.join("\n")
    }
}

/// The critics' verdict against the definition of done.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Review {
    pub accepted: bool,
    pub score: f64,
    pub problems: Vec<String>,
    pub gaps: Vec<String>,
    pub summary: String,
    /// Did any critic actually return a verdict? False is unreviewed, not accepted.
    pub reviewed: bool,
    /// Each critic's own verdict.
    pub verdicts: Vec<Value>,
}

// --- what the planner, checker and critics must return ---------------------------
#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct TaskDraft {
    id: String,
    statement: String,
    #[serde(default)]
    depends_on: Vec<String>,
    #[serde(default)]
    done_when: String,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct PlanDraft {
    goal: String,
    #[serde(default)]
    definition_of_done: Vec<String>,
    #[serde(default)]
    constraints: Vec<String>,
    #[serde(default)]
    tasks: Vec<TaskDraft>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct CheckDraft {
    met: bool,
    #[serde(default)]
    missing: String,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct ReplanDraft {
    #[serde(default)]
    tasks: Vec<TaskDraft>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct ReviewDraft {
    accepted: bool,
    #[serde(default)]
    score: f64,
    #[serde(default)]
    problems: Vec<String>,
    #[serde(default)]
    gaps: Vec<String>,
    #[serde(default)]
    summary: String,
}

fn parse<T: serde::de::DeserializeOwned>(data: &Value) -> Option<T> {
    serde_json::from_value(data.clone()).ok()
}

pub const PLANNER_PROMPT: &str = "Turn this request into a costed plan.

REQUEST
{request}

SUB-AGENTS ON THE BENCH
{bench}

Rules:
- Write the definition of done FIRST: the acceptance tests, written before any
  work starts. Each one must be checkable by someone who was not involved.
- Break the goal into tasks one worker could finish alone. Give every task a
  short id (t1, t2, ...) and list what it depends on. Independent tasks must
  have empty dependencies so they can run at the same time.
- As few tasks as the goal allows. Do not invent work the request did not ask for.
- Note any constraint the request imposes (deadline, format, tone, sources).";

pub const CHECK_PROMPT: &str = "Check this hand-back against what the task was to be done by. You did not do the work.

TASK
{task}

DONE WHEN
{done_when}

HAND-BACK
{output}

Say whether the hand-back meets \"done when\" — judged on what is written, not on
what is promised. If it does not, say exactly what is missing, in one or two
sentences the worker could act on.";

pub const REPLAN_PROMPT: &str = "A task in this plan came back short. Plan what to do instead.

GOAL
{goal}

THE TASK THAT CAME BACK SHORT
{task}

DONE WHEN
{done_when}

WHAT WENT WRONG
{error}

WHAT IT DID PRODUCE
{partial}

SUB-AGENTS ON THE BENCH
{bench}

Write the task or tasks that will get this done by a different route: split it
into smaller steps, narrow it, or come at it another way. Do not repeat the task
as it was. Give each a short id and say what it depends on among the others you
write. If there is no better route, return no tasks.";

pub const CONSOLIDATE_PROMPT: &str = "Consolidate these sub-agent results into the answer the request asked for.

REQUEST
{request}

DEFINITION OF DONE
{dod}

RESULTS
{results}

Merge what agrees, de-duplicate what repeats, and rank what matters most first.
Attribute anything contested to the sub-agent that said it. Do not add findings
nobody reported. Write the deliverable itself — not a description of it.";

pub const REVIEW_PROMPT: &str = "Review this deliverable against the definition of done. You did not do the work.

REQUEST
{request}

DEFINITION OF DONE
{dod}

DELIVERABLE
{deliverable}

{lens}Check every acceptance test. Accept only if all of them hold. If you reject,
list the specific gaps — each one must be actionable as a task on its own.";

const MANAGER_INSTRUCTIONS: &str = "You own this job: the goal, the plan, the budget and the answer.
- Decide reuse-or-create for every task; never hand a sub-agent more access than its task needs.
- Delegate work that can run in parallel; never wait on one sub-agent when two could run.
- Re-plan when a result comes back short.
- You are the only component that speaks to the requester.";

fn fill(template: &str, vars: Value) -> String {
    Prompt::new("_", template).render(&vars).unwrap_or_else(|_| template.to_string())
}

fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn dod(plan: &Plan) -> String {
    if plan.definition_of_done.is_empty() { "(none)".into() } else { plan.definition_of_done.iter().map(|d| format!("- {d}")).collect::<Vec<_>>().join("\n") }
}

/// Builds an [`Orchestrator`].
pub struct OrchestratorBuilder {
    name: String,
    harness: Option<Harness>,
    bench: Option<Bench>,
    specs: Vec<SubAgentSpec>,
    agents: Vec<Agent>,
    tools: Vec<Tool>,
    model: Option<String>,
    provider: Option<Arc<dyn Provider>>,
    tier: String,
    budget: Option<Budget>,
    max_concurrency: usize,
    max_rework: u32,
    review: bool,
    critics: Vec<String>,
    accept_majority: bool,
    use_factory: bool,
    task_retries: u32,
    task_timeout: Option<f64>,
    check_tasks: bool,
    max_replans: u32,
    on_over_estimate: String,
    tokens_per_task: u64,
    persist: bool,
    instructions: String,
}

impl OrchestratorBuilder {
    pub fn harness(mut self, h: Harness) -> Self {
        self.harness = Some(h);
        self
    }

    pub fn bench(mut self, b: Bench) -> Self {
        self.bench = Some(b);
        self
    }

    /// A blueprint to add to the bench.
    pub fn subagent_spec(mut self, s: SubAgentSpec) -> Self {
        self.specs.push(s);
        self
    }

    /// A ready-built agent to staff tasks with, by name.
    pub fn subagent(mut self, a: Agent) -> Self {
        self.agents.push(a);
        self
    }

    pub fn tools<I: IntoIterator<Item = Tool>>(mut self, tools: I) -> Self {
        self.tools.extend(tools);
        self
    }

    pub fn model(mut self, m: &str) -> Self {
        self.model = Some(m.into());
        self
    }

    pub fn provider(mut self, p: Arc<dyn Provider>) -> Self {
        self.provider = Some(p);
        self
    }

    pub fn tier(mut self, t: &str) -> Self {
        self.tier = t.into();
        self
    }

    pub fn budget(mut self, b: Budget) -> Self {
        self.budget = Some(b);
        self
    }

    pub fn max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self
    }

    /// How many times a rejected deliverable is reworked.
    pub fn max_rework(mut self, n: u32) -> Self {
        self.max_rework = n;
        self
    }

    pub fn review(mut self, on: bool) -> Self {
        self.review = on;
        self
    }

    /// A number of independent critics.
    pub fn critics(mut self, n: usize) -> Self {
        self.critics = vec![String::new(); n.max(1)];
        self
    }

    /// What each critic is to look for: `["accuracy of every figure", "tone"]`.
    pub fn critic_lenses<I: IntoIterator<Item = S>, S: Into<String>>(mut self, lenses: I) -> Self {
        self.critics = lenses.into_iter().map(|l| l.into().trim().to_string()).collect();
        if self.critics.is_empty() {
            self.critics = vec![String::new()];
        }
        self
    }

    /// Accept when most critics accept, rather than all.
    pub fn accept_majority(mut self, on: bool) -> Self {
        self.accept_majority = on;
        self
    }

    pub fn use_factory(mut self, on: bool) -> Self {
        self.use_factory = on;
        self
    }

    pub fn task_retries(mut self, n: u32) -> Self {
        self.task_retries = n;
        self
    }

    /// Seconds one task may run.
    pub fn task_timeout(mut self, s: f64) -> Self {
        self.task_timeout = Some(s);
        self
    }

    /// Check each hand-back against its task's `done_when`.
    pub fn check_tasks(mut self, on: bool) -> Self {
        self.check_tasks = on;
        self
    }

    pub fn max_replans(mut self, n: u32) -> Self {
        self.max_replans = n;
        self
    }

    /// `warn`, `stop` or `ignore` when the work would cost more than the budget has left.
    pub fn on_over_estimate(mut self, what: &str) -> Self {
        self.on_over_estimate = what.into();
        self
    }

    pub fn tokens_per_task(mut self, n: u64) -> Self {
        self.tokens_per_task = n.max(1);
        self
    }

    /// Keep the job in the session store as it goes, so it can be resumed.
    pub fn persist(mut self, on: bool) -> Self {
        self.persist = on;
        self
    }

    pub fn instructions(mut self, text: &str) -> Self {
        self.instructions = text.into();
        self
    }

    pub fn build(self) -> Result<Orchestrator> {
        if !["warn", "stop", "ignore"].contains(&self.on_over_estimate.as_str()) {
            return Err(Error::config("on_over_estimate is 'warn', 'stop' or 'ignore'"));
        }
        let mut harness = self.harness.unwrap_or_default();
        if let Some(b) = &self.budget {
            harness = harness.with_budget(b.clone());
        }
        let mut bench = self.bench.unwrap_or_else(Bench::standard);
        for s in &self.specs {
            bench.register(s.clone());
        }
        let name = self.name.clone();
        let agent = |n: &str, instructions: &str, tier: &str, steps: u32| {
            let mut b = Agent::builder(n).instructions(instructions).tier(tier).harness(harness.clone()).max_steps(steps).memory(false).persist_session(false);
            if let Some(m) = &self.model {
                b = b.model(m.clone());
            }
            if let Some(p) = &self.provider {
                b = b.provider(p.clone());
            }
            b
        };
        let instructions = if self.instructions.is_empty() { MANAGER_INSTRUCTIONS.to_string() } else { self.instructions.clone() };
        let manager = agent(&name, &instructions, &self.tier, 4)
            .description("Owns the goal, the plan, the budget and the answer for one job.")
            .tools(self.tools.clone())
            .memory(true)
            .persist_session(true)
            .build()?;
        let critic = "You are an independent critic. You did not do the work and you have no stake in accepting it.";
        let checker = "You check a piece of work against what it was to be done by. You did not do the work and you have no stake in passing it.";
        let planner = agent(&format!("{name}.planner"), "You plan work. You do not do the work.", &self.tier, 3)
            .output_contract(OutputContract::of::<PlanDraft>())
            .build()?;
        let reviewer = agent(&format!("{name}.reviewer"), critic, &self.tier, 3).output_contract(OutputContract::of::<ReviewDraft>()).build()?;
        let checker = agent(&format!("{name}.checker"), checker, "balanced", 2).output_contract(OutputContract::of::<CheckDraft>()).build()?;
        let replanner = agent(&format!("{name}.replanner"), "You plan work. You do not do the work.", &self.tier, 3)
            .output_contract(OutputContract::of::<ReplanDraft>())
            .build()?;
        let factory = if self.use_factory {
            let builder = agent(&format!("{name}.factory"), "You write specifications for sub-agents.", "balanced", 2).build()?;
            Some(SubAgentFactory { builder, bench: Arc::new(Mutex::new(bench.clone())) })
        } else {
            None
        };
        Ok(Orchestrator {
            harness,
            bench: Arc::new(Mutex::new(bench)),
            manager,
            planner,
            reviewer,
            checker,
            replanner,
            factory,
            live: Mutex::new(self.agents.into_iter().map(|a| (a.name().to_string(), a)).collect()),
            max_concurrency: self.max_concurrency,
            max_rework: self.max_rework,
            review: self.review,
            critics: self.critics,
            accept_majority: self.accept_majority,
            task_retries: self.task_retries,
            task_timeout: self.task_timeout,
            check_tasks: self.check_tasks,
            max_replans: self.max_replans,
            on_over_estimate: self.on_over_estimate,
            tokens_per_task: self.tokens_per_task,
            persist: self.persist,
            seen_tasks: Mutex::new((0, 0.0)),
        })
    }
}

/// What one job carries while it runs.
struct Job {
    id: String,
    request: String,
    guard: Arc<BudgetGuard>,
    results: Mutex<IndexMap<String, RunResult>>,
    replans: Mutex<u32>,
    warnings: Mutex<Vec<String>>,
    record: Mutex<Map<String, Value>>,
    version: Mutex<u64>,
    prior: Usage,
    prior_by_agent: Map<String, Value>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Owns the goal, the plan, the budget and the answer.
pub struct Orchestrator {
    pub harness: Harness,
    pub bench: Arc<Mutex<Bench>>,
    pub manager: Agent,
    planner: Agent,
    reviewer: Agent,
    checker: Agent,
    replanner: Agent,
    factory: Option<SubAgentFactory>,
    live: Mutex<HashMap<String, Agent>>,
    max_concurrency: usize,
    max_rework: u32,
    review: bool,
    critics: Vec<String>,
    accept_majority: bool,
    task_retries: u32,
    task_timeout: Option<f64>,
    check_tasks: bool,
    max_replans: u32,
    on_over_estimate: String,
    tokens_per_task: u64,
    persist: bool,
    /// What tasks have actually cost here: (count, total USD).
    seen_tasks: Mutex<(u32, f64)>,
}

impl std::fmt::Debug for Orchestrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Orchestrator {}>", self.manager.name())
    }
}

impl Orchestrator {
    pub fn builder(name: &str) -> OrchestratorBuilder {
        OrchestratorBuilder {
            name: name.into(),
            harness: None,
            bench: None,
            specs: vec![],
            agents: vec![],
            tools: vec![],
            model: None,
            provider: None,
            tier: "deep".into(),
            budget: None,
            max_concurrency: 4,
            max_rework: 1,
            review: true,
            critics: vec![String::new()],
            accept_majority: false,
            use_factory: true,
            task_retries: 1,
            task_timeout: None,
            check_tasks: true,
            max_replans: 2,
            on_over_estimate: "warn".into(),
            tokens_per_task: 6000,
            persist: true,
            instructions: String::new(),
        }
    }

    fn charged(job: Option<&Job>) -> RunOptions {
        RunOptions { messages: Some(vec![]), guard: job.map(|j| j.guard.clone()), ..Default::default() }
    }

    // ---- 1: plan and scope ------------------------------------------------
    /// A request as a costed plan, acceptance tests written first.
    pub async fn plan(&self, request: &str) -> Result<Plan> {
        self.plan_in(request, None).await
    }

    async fn plan_in(&self, request: &str, job: Option<&Job>) -> Result<Plan> {
        let catalogue = lock(&self.bench).catalogue();
        let result = self.planner.run_with(fill(PLANNER_PROMPT, json!({"request": request, "bench": catalogue})), Self::charged(job)).await?;
        let mut plan = match parse::<PlanDraft>(&result.data).filter(|d| !d.tasks.is_empty()) {
            None => Plan {
                goal: request.into(),
                definition_of_done: vec!["The request is answered.".into()],
                tasks: vec![Task::new("t1", request)],
                ..Default::default()
            },
            Some(d) => Plan {
                goal: if d.goal.is_empty() { request.into() } else { d.goal },
                definition_of_done: d.definition_of_done,
                constraints: d.constraints,
                tasks: d.tasks.into_iter().map(|t| Task { depends_on: t.depends_on, done_when: t.done_when, ..Task::new(&t.id, &t.statement) }).collect(),
                ..Default::default()
            },
        };
        plan.estimate_usd = self.estimate(&mut plan, None);
        if let Some(m) = self.manager.memory() {
            let data = json!({"tasks": plan.tasks.len(), "estimate_usd": plan.estimate_usd});
            m.orchestrator.plan(&plan.goal, data.as_object().cloned().unwrap_or_default()).await?;
        }
        Ok(plan)
    }

    /// What one task is expected to cost: what tasks here have been costing,
    /// once a few have finished; a guess from the model's prices until then.
    fn per_task(&self, tokens_per_task: Option<u64>) -> f64 {
        let (count, spent) = *lock(&self.seen_tasks);
        if tokens_per_task.is_none() && count >= 3 {
            return spent / count as f64;
        }
        let Some(info) = model_info(self.manager.model()) else { return 0.0 };
        let tokens = tokens_per_task.unwrap_or(self.tokens_per_task) as f64;
        (tokens * 0.75 * info.input_cost + tokens * 0.25 * info.output_cost) / 1e6
    }

    /// A cost estimate; sets `plan.parallelism`.
    pub fn estimate(&self, plan: &mut Plan, tokens_per_task: Option<u64>) -> f64 {
        let per = self.per_task(tokens_per_task);
        let widest = plan.waves().iter().map(Vec::len).max().unwrap_or(1);
        plan.parallelism = widest.clamp(1, self.max_concurrency);
        let checks = if self.check_tasks { plan.tasks.iter().filter(|t| !t.done_when.is_empty()).count() } else { 0 };
        let overhead = per * (2.0 + 0.2 * checks as f64);
        ((per * plan.tasks.len().max(1) as f64 + overhead) * 1e4).round() / 1e4
    }

    /// Why the work still to do will not fit in the budget; empty when it will.
    fn affordable(&self, plan: &Plan) -> String {
        let Some(left) = self.harness.guard().remaining_usd() else { return String::new() };
        if self.on_over_estimate == "ignore" {
            return String::new();
        }
        let waiting = plan.tasks.iter().filter(|t| t.status == TaskStatus::Pending).count();
        if waiting == 0 {
            return String::new();
        }
        let finished: Vec<&Task> =
            plan.tasks.iter().filter(|t| matches!(t.status, TaskStatus::Done | TaskStatus::Failed | TaskStatus::Replaced) && t.cost_usd > 0.0).collect();
        let (needed, basis) = if finished.is_empty() {
            (plan.estimate_usd, "the estimate".to_string())
        } else {
            let each = finished.iter().map(|t| t.cost_usd).sum::<f64>() / finished.len() as f64;
            (each * waiting as f64, format!("its tasks are costing ${each:.4} each"))
        };
        if needed <= left {
            return String::new();
        }
        let s = if waiting == 1 { "" } else { "s" };
        format!("the {waiting} task{s} still to run would cost about ${needed:.4} going by {basis}, and the budget has ${left:.4} left")
    }

    // ---- 2: the staffing decision -------------------------------------------
    /// Reuse a pre-defined sub-agent, or have the factory write a new one.
    pub async fn staff(&self, task: &mut Task) -> Result<Agent> {
        self.staff_in(task, None).await
    }

    async fn staff_in(&self, task: &mut Task, job: Option<&Job>) -> Result<Agent> {
        if let Some(a) = lock(&self.live).get(&task.agent).filter(|_| !task.agent.is_empty()) {
            return Ok(a.clone());
        }
        let found = lock(&self.bench).find(&task.statement, 0.3).cloned();
        let reused = found.is_some();
        let spec = match (found, &self.factory) {
            (Some(s), _) => s,
            (None, None) => SubAgentSpec::new(&format!("worker_{}", task.id), &clip(&task.statement, 80), "Finish the task you were given."),
            (None, Some(f)) => f.create(&task.statement, &self.manager.tools().names(), false, job.map(|j| j.guard.clone())).await,
        };
        task.agent = spec.name.clone();
        task.reused = reused;
        let worker = {
            let mut live = lock(&self.live);
            if !live.contains_key(&spec.name) {
                live.insert(spec.name.clone(), build_agent(&spec, &self.manager)?);
            }
            live[&spec.name].clone()
        };
        if let Some(m) = self.manager.memory() {
            m.orchestrator.staffing(&task.statement.chars().take(80).collect::<String>(), &spec.name, reused).await?;
        }
        Ok(worker)
    }

    // ---- 3: work assignment ----------------------------------------------------
    async fn run_task(&self, mut task: Task, plan: &Plan, request: &str, job: &Job) -> Task {
        let worker = match self.staff_in(&mut task, Some(job)).await {
            Ok(w) => w,
            Err(e) => {
                task.status = TaskStatus::Failed;
                task.error = Some(format!("{}: {e}", e.kind_name()));
                return task;
            }
        };
        let context = Self::context_for(&task, plan);
        for attempt in 1..=self.task_retries + 1 {
            // A second attempt is told what the first was missing.
            let brief = Self::brief(&task, request, &context);
            task.attempts = attempt;
            task.status = TaskStatus::Running;
            // Its own ceiling if its spec set one; its spend counts to the job.
            let options =
                RunOptions { messages: Some(vec![]), guard: Some(job.guard.child(worker.budget().cloned().unwrap_or_default())), ..Default::default() };
            let run = worker.run_with(brief, options);
            let outcome = match self.task_timeout {
                Some(s) => tokio::time::timeout(Duration::from_secs_f64(s), run).await.ok(),
                None => Some(run.await),
            };
            let result = match outcome {
                None => {
                    task.error = Some(format!("missed its {}s deadline", self.task_timeout.unwrap_or_default()));
                    continue;
                }
                Some(Err(e)) => {
                    task.error = Some(format!("{}: {e}", e.kind_name()));
                    continue;
                }
                Some(Ok(r)) => r,
            };
            task.cost_usd = ((task.cost_usd + result.cost_usd()) * 1e6).round() / 1e6;
            if let Some(m) = self.manager.memory() {
                let _ = m.orchestrator.spend(worker.name(), result.cost_usd(), result.usage.total_tokens()).await;
            }
            // Partial delivery is kept: something consolidation can use.
            if !result.output.is_empty() {
                task.partial = result.output.clone();
            }
            if result.error.is_none() {
                let missing = self.short_of(&mut task, &result.output, job).await;
                if missing.is_empty() {
                    task.output = result.output.clone();
                    task.status = TaskStatus::Done;
                    task.error = None;
                    lock(&job.results).insert(task.id.clone(), result);
                    return task;
                }
                // It answered, but not with what the task was to be done by.
                task.error = Some(format!("short of done-when: {missing}"));
                task.missing = missing;
                lock(&job.results).insert(task.id.clone(), result);
            } else {
                task.error = result.error.clone();
                lock(&job.results).entry(task.id.clone()).or_insert(result);
            }
        }
        task.status = TaskStatus::Failed;
        task
    }

    /// What a hand-back is missing against `done_when`; empty when it meets
    /// it. A check that cannot be made does not hold the work back.
    async fn short_of(&self, task: &mut Task, output: &str, job: &Job) -> String {
        if !self.check_tasks || task.done_when.trim().is_empty() || output.trim().is_empty() {
            task.checked = None;
            return String::new();
        }
        let prompt = fill(CHECK_PROMPT, json!({"task": task.statement, "done_when": task.done_when, "output": clip(output, 30_000)}));
        let verdict = self.checker.run_with(prompt, Self::charged(Some(job))).await;
        let Some(draft) = verdict.ok().and_then(|v| parse::<CheckDraft>(&v.data)) else {
            task.checked = Some(false);
            return String::new();
        };
        task.checked = Some(true);
        if draft.met {
            task.missing.clear();
            return String::new();
        }
        let m = draft.missing.trim();
        if m.is_empty() { "it does not meet the criterion".into() } else { m.into() }
    }

    /// Plan another route for a task that came back short. The new tasks take
    /// its place in the graph.
    async fn replan(&self, task_id: &str, plan: &mut Plan, job: &Job) -> Result<Vec<String>> {
        let Some(task) = plan.by_id(task_id).cloned() else { return Ok(vec![]) };
        {
            let mut n = lock(&job.replans);
            if *n >= self.max_replans || !task.replaces.is_empty() {
                return Ok(vec![]); // the budget for second routes is spent
            }
            *n += 1;
        }
        let prompt = fill(
            REPLAN_PROMPT,
            json!({
                "goal": plan.goal, "task": task.statement,
                "done_when": if task.done_when.is_empty() { "(not stated)" } else { &task.done_when },
                "error": task.error.clone().unwrap_or_else(|| "it did not finish".into()),
                "partial": if task.partial.is_empty() { "(nothing)".to_string() } else { clip(&task.partial, 8_000) },
                "bench": lock(&self.bench).catalogue(),
            }),
        );
        let answer = self.replanner.run_with(prompt, Self::charged(Some(job))).await?;
        let Some(draft) = parse::<ReplanDraft>(&answer.data).filter(|d| !d.tasks.is_empty()) else { return Ok(vec![]) };
        let mut taken: Vec<String> = plan.tasks.iter().map(|t| t.id.clone()).collect();
        let mut renamed: IndexMap<String, String> = IndexMap::new();
        for item in &draft.tasks {
            let mut fresh = format!("{}.{}", task.id, renamed.len() + 1);
            while taken.contains(&fresh) {
                fresh.push('x');
            }
            taken.push(fresh.clone());
            renamed.insert(item.id.clone(), fresh);
        }
        let added: Vec<Task> = draft
            .tasks
            .iter()
            .map(|item| Task {
                id: renamed[&item.id].clone(),
                statement: item.statement.clone(),
                replaces: task.id.clone(),
                done_when: if item.done_when.is_empty() { task.done_when.clone() } else { item.done_when.clone() },
                // Among themselves as planned; and all after what the failed task waited for.
                depends_on: task.depends_on.iter().cloned().chain(item.depends_on.iter().filter_map(|d| renamed.get(d).cloned())).collect(),
                ..Default::default()
            })
            .collect();
        let ends: Vec<String> = added.iter().filter(|t| !added.iter().any(|o| o.depends_on.contains(&t.id))).map(|t| t.id.clone()).collect();
        for other in plan.tasks.iter_mut() {
            if other.depends_on.contains(&task.id) {
                other.depends_on.retain(|d| *d != task.id);
                other.depends_on.extend(ends.iter().cloned());
            }
        }
        if let Some(t) = plan.by_id_mut(&task.id) {
            t.status = TaskStatus::Replaced;
        }
        let ids: Vec<String> = added.iter().map(|t| t.id.clone()).collect();
        plan.tasks.extend(added);
        Ok(ids)
    }

    /// Only what this task depends on — never the whole job's history.
    fn context_for(task: &Task, plan: &Plan) -> String {
        let mut parts = Vec::new();
        if let Some(earlier) = plan.by_id(&task.replaces).filter(|e| !task.replaces.is_empty() && !e.partial.is_empty()) {
            parts.push(format!(
                "### An earlier attempt at this ({}) came back short — {}. What it produced:\n{}",
                earlier.statement,
                earlier.error.clone().unwrap_or_else(|| "unfinished".into()),
                earlier.partial
            ));
        }
        for dep in task.depends_on.iter().filter_map(|d| plan.by_id(d)) {
            if !dep.usable_output().is_empty() {
                let note = if dep.output.is_empty() { " — partial, this task did not finish" } else { "" };
                parts.push(format!("### Result of {} ({}){note}\n{}", dep.id, dep.statement, dep.usable_output()));
            }
        }
        parts.join("\n\n")
    }

    fn brief(task: &Task, request: &str, context: &str) -> String {
        let mut parts = vec![format!("## Your task\n{}", task.statement), format!("## Why it matters\nIt is part of: {request}")];
        if !task.done_when.is_empty() {
            parts.push(format!("## Done when\n{}", task.done_when));
        }
        if !task.missing.is_empty() {
            parts.push(format!("## What the last attempt was missing\n{}\nPut that right in this one.", task.missing));
        }
        if !context.is_empty() {
            parts.push(format!("## What earlier tasks produced\n{context}"));
        }
        parts.join("\n\n")
    }

    /// The pending tasks whose dependencies are all in; with a cycle, those
    /// that wait only on each other, rather than hang.
    fn ready(plan: &Plan) -> Vec<String> {
        let status = |id: &str| plan.by_id(id).map(|t| t.status);
        let waiting: Vec<&Task> = plan.tasks.iter().filter(|t| t.status == TaskStatus::Pending).collect();
        let ready: Vec<String> =
            waiting.iter().filter(|t| t.depends_on.iter().all(|d| status(d).is_none_or(|s| s == TaskStatus::Done))).map(|t| t.id.clone()).collect();
        if !ready.is_empty() {
            return ready;
        }
        waiting
            .iter()
            .filter(|t| t.depends_on.iter().all(|d| status(d).is_none_or(|s| matches!(s, TaskStatus::Done | TaskStatus::Pending))))
            .map(|t| t.id.clone())
            .collect()
    }

    async fn execute(&self, plan: &mut Plan, request: &str, job: &Job) -> Result<()> {
        let gate = Semaphore::new(self.max_concurrency);
        for _ in 0..plan.tasks.len() * 4 + 50 {
            let runnable = Self::ready(plan);
            if runnable.is_empty() {
                break;
            }
            let short = self.affordable(plan);
            if !short.is_empty() && !lock(&job.warnings).contains(&short) {
                lock(&job.warnings).push(short.clone());
                if self.on_over_estimate == "stop" {
                    // Better to hand back what is finished than to start work that
                    // will be cut off half-way through.
                    for t in plan.tasks.iter_mut().filter(|t| t.status == TaskStatus::Pending) {
                        t.status = TaskStatus::Skipped;
                        t.error = Some(format!("not started: {short}"));
                    }
                    break;
                }
            }
            let snapshot = plan.clone();
            let wave = runnable.iter().filter_map(|id| snapshot.by_id(id).cloned()).map(|t| {
                let gate = &gate;
                let snapshot = &snapshot;
                async move {
                    let _permit = gate.acquire().await.ok();
                    self.run_task(t, snapshot, request, job).await
                }
            });
            for done in join_all(wave).await {
                if let Some(slot) = plan.by_id_mut(&done.id) {
                    *slot = done;
                }
            }
            for id in &runnable {
                if plan.by_id(id).is_some_and(|t| t.status == TaskStatus::Failed)
                    && let Err(e) = self.replan(id, plan, job).await
                {
                    // A second route that cannot be planned leaves the first failure standing.
                    tracing::warn!("{id} could not be re-planned: {e}");
                }
            }
            self.checkpoint(Some(plan), "executing", job, Map::new()).await?;
            // Down the whole chain: what waits on a failure is blocked.
            loop {
                let failed: Vec<String> = plan
                    .tasks
                    .iter()
                    .filter(|t| matches!(t.status, TaskStatus::Failed | TaskStatus::Blocked | TaskStatus::Skipped))
                    .map(|t| t.id.clone())
                    .collect();
                let mut changed = false;
                for t in plan.tasks.iter_mut() {
                    if t.status == TaskStatus::Pending && t.depends_on.iter().any(|d| failed.contains(d)) {
                        t.status = TaskStatus::Blocked;
                        t.error = Some("a task it depends on failed".into());
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        Ok(())
    }

    // ---- 4: consolidate, review, accept or rework ------------------------------
    async fn consolidate(&self, plan: &Plan, request: &str, job: &Job) -> Result<String> {
        // What was planned again is spoken for by the tasks that replaced it.
        let usable: Vec<&Task> = plan.tasks.iter().filter(|t| !t.usable_output().is_empty() && t.status != TaskStatus::Replaced).collect();
        if usable.is_empty() {
            return Ok(String::new());
        }
        // A single finished task is the answer; a single partial one is not.
        if usable.len() == 1 && plan.tasks.len() == 1 && usable[0].status == TaskStatus::Done {
            return Ok(usable[0].usable_output().to_string());
        }
        let body: Vec<String> = usable
            .iter()
            .map(|t| {
                let how = if t.reused { "reused" } else { "purpose-built" };
                let partial = if t.status == TaskStatus::Done { "" } else { " [PARTIAL — this task did not finish]" };
                format!("### {} — {} ({how}){partial}\nTask: {}\n{}", t.id, t.agent, t.statement, t.usable_output())
            })
            .collect();
        let prompt = fill(CONSOLIDATE_PROMPT, json!({"request": request, "results": body.join("\n\n"), "dod": dod(plan)}));
        Ok(self.manager.run_with(prompt, Self::charged(Some(job))).await?.output)
    }

    async fn critic(&self, number: usize, lens: &str, plan: &Plan, deliverable: &str, request: &str, job: Option<&Job>) -> Option<Value> {
        let lens_line = if lens.is_empty() { String::new() } else { format!("You are reviewing it for one thing in particular: {lens}.\n") };
        let prompt = fill(REVIEW_PROMPT, json!({"request": request, "deliverable": clip(deliverable, 40_000), "dod": dod(plan), "lens": lens_line}));
        for _ in 0..2 {
            // Asked again once if it says nothing usable.
            if let Some(d) = self.reviewer.run_with(prompt.clone(), Self::charged(job)).await.ok().and_then(|r| parse::<ReviewDraft>(&r.data)) {
                return Some(
                    json!({"critic": number, "lens": lens, "accepted": d.accepted, "score": d.score, "problems": d.problems, "gaps": d.gaps, "summary": d.summary}),
                );
            }
        }
        None
    }

    /// The critics' verdict. Each reviews alone; accepted when all (or most)
    /// accept. If none gives a verdict the deliverable is *unreviewed*.
    pub async fn review(&self, plan: &Plan, deliverable: &str, request: &str) -> Review {
        self.review_in(plan, deliverable, request, None).await
    }

    async fn review_in(&self, plan: &Plan, deliverable: &str, request: &str, job: Option<&Job>) -> Review {
        if !self.review || deliverable.is_empty() {
            return Review { accepted: true, score: 1.0, summary: "review disabled".into(), reviewed: false, ..Default::default() };
        }
        let answers = join_all(self.critics.iter().enumerate().map(|(n, lens)| self.critic(n + 1, lens, plan, deliverable, request, job))).await;
        let verdicts: Vec<Value> = answers.into_iter().flatten().collect();
        if verdicts.is_empty() {
            return Review {
                accepted: false,
                score: 0.0,
                reviewed: false,
                summary: "no critic returned a usable verdict; the deliverable is unreviewed".into(),
                ..Default::default()
            };
        }
        let yes = verdicts.iter().filter(|v| v["accepted"] == true).count();
        let accepted = if self.accept_majority { yes * 2 > verdicts.len() } else { yes == verdicts.len() };
        let merged = |key: &str| {
            let mut seen: IndexMap<String, String> = IndexMap::new();
            for v in &verdicts {
                for item in v[key].as_array().into_iter().flatten().filter_map(Value::as_str) {
                    seen.entry(item.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")).or_insert_with(|| item.to_string());
                }
            }
            seen.into_values().collect::<Vec<_>>()
        };
        let score = verdicts.iter().filter_map(|v| v["score"].as_f64()).sum::<f64>() / verdicts.len() as f64;
        Review {
            accepted,
            score: (score * 1000.0).round() / 1000.0,
            problems: if accepted { vec![] } else { merged("problems") },
            gaps: if accepted { vec![] } else { merged("gaps") },
            summary: verdicts.iter().filter_map(|v| v["summary"].as_str()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" | "),
            reviewed: true,
            verdicts: verdicts
                .iter()
                .map(|v| json!({"critic": v["critic"], "lens": v["lens"], "accepted": v["accepted"], "score": v["score"], "summary": v["summary"]}))
                .collect(),
        }
    }

    // ---- a job that outlives the process ------------------------------------------
    fn key(job_id: &str) -> String {
        format!("orchestrator_{job_id}")
    }

    fn by_agent(job: &Job) -> Map<String, Value> {
        let mut merged = job.prior_by_agent.clone();
        let current: BTreeMap<String, Usage> = job.guard.by_agent();
        for (name, used) in current {
            let held = merged.entry(name).or_insert_with(|| json!({"cost_usd": 0.0, "tokens": 0, "calls": 0}));
            let cost = held["cost_usd"].as_f64().unwrap_or(0.0) + used.cost_usd;
            held["cost_usd"] = json!((cost * 1e6).round() / 1e6);
            held["tokens"] = json!(held["tokens"].as_u64().unwrap_or(0) + used.total_tokens());
            held["calls"] = json!(held["calls"].as_u64().unwrap_or(0) + used.calls);
        }
        let sorted: BTreeMap<String, Value> = merged.into_iter().collect();
        sorted.into_iter().collect()
    }

    /// Write the job down as it stands, after every wave and every stage.
    async fn checkpoint(&self, plan: Option<&Plan>, stage: &str, job: &Job, more: Map<String, Value>) -> Result<()> {
        if !self.persist {
            return Ok(());
        }
        let spent = job.prior.clone() + job.guard.usage();
        let results: Map<String, Value> =
            lock(&job.results).iter().map(|(k, r)| (k.clone(), json!({"agent": r.agent, "output": r.output, "error": r.error, "usage": r.usage}))).collect();
        let record = {
            let mut rec = lock(&job.record);
            rec.insert("stage".into(), json!(stage));
            rec.insert("updated".into(), json!(now()));
            rec.insert("replans".into(), json!(*lock(&job.replans)));
            if let Some(p) = plan {
                rec.insert("plan".into(), serde_json::to_value(p).unwrap_or_default());
            }
            rec.insert("usage".into(), serde_json::to_value(&spent).unwrap_or_default());
            rec.insert("by_agent".into(), Value::Object(Self::by_agent(job)));
            rec.insert("results".into(), Value::Object(results));
            rec.extend(more);
            rec.clone()
        };
        let version = *lock(&job.version);
        let mut session = Session {
            id: Self::key(&job.id),
            agent: "#jobs".into(),
            title: clip(&job.request, 80),
            version,
            created: record.get("created").and_then(Value::as_f64).unwrap_or_else(now),
            ..Default::default()
        };
        session.metadata.insert("job".into(), Value::Object(record));
        match self.harness.sessions.save(&mut session).await {
            Ok(()) => {
                *lock(&job.version) = session.version;
                Ok(())
            }
            // Someone else has written to this job: two managers would do the work twice.
            Err(Error::SessionConflict { .. }) => Err(Error::config(format!("job {} is being run by another process; this one has stopped", job.id))),
            Err(e) => {
                let warning = format!("the job could not be saved ({}); it will not be resumable from here", e.kind_name());
                let mut w = lock(&job.warnings);
                if !w.contains(&warning) {
                    w.push(warning);
                }
                Ok(())
            }
        }
    }

    /// Where a job stands: its stage, each task's status, what it has cost.
    pub async fn job(&self, job_id: &str) -> Result<Map<String, Value>> {
        let session = self.harness.sessions.load(&Self::key(job_id)).await.map_err(|_| Error::config(format!("no job '{job_id}'")))?;
        let Some(Value::Object(mut record)) = session.metadata.get("job").cloned() else { return Err(Error::config(format!("no job '{job_id}'"))) };
        record.insert("_version".into(), json!(session.version));
        Ok(record)
    }

    /// Carry on a job that was cut short. Finished tasks are not run again; a
    /// task that was running is started afresh.
    pub async fn resume(&self, job_id: &str) -> Result<RunResult> {
        let record = self.job(job_id).await?;
        let request = record.get("request").and_then(Value::as_str).unwrap_or("").to_string();
        let Some(mut plan) = record.get("plan").and_then(|p| serde_json::from_value::<Plan>(p.clone()).ok()) else {
            return self.run_job(&request, None, Some(job_id.into()), Some(record)).await;
        };
        for t in plan.tasks.iter_mut().filter(|t| t.status == TaskStatus::Running) {
            t.status = TaskStatus::Pending;
        }
        if matches!(record.get("status").and_then(Value::as_str), Some("done" | "failed")) {
            let review: Review = record.get("review").and_then(|r| serde_json::from_value(r.clone()).ok()).unwrap_or_default();
            return Ok(RunResult {
                agent: self.manager.name().into(),
                run_id: job_id.into(),
                output: record.get("deliverable").and_then(Value::as_str).unwrap_or("").into(),
                error: record.get("error").and_then(Value::as_str).map(str::to_string),
                stop_reason: record.get("stop_reason").and_then(|s| serde_json::from_value(s.clone()).ok()).unwrap_or_default(),
                data: json!({"plan": plan, "review": review}),
                spend: record.get("spend").and_then(Value::as_object).cloned().unwrap_or_default(),
                ..Default::default()
            });
        }
        self.run_job(&request, Some(plan), Some(job_id.into()), Some(record)).await
    }

    /// Plan, staff, run, consolidate and review — one costed job.
    /// `result.run_id` is the job's id: `resume(run_id)` picks it back up.
    pub async fn run(&self, request: &str) -> Result<RunResult> {
        self.run_job(request, None, None, None).await
    }

    /// Run a plan you already have (or edited).
    pub async fn run_plan(&self, request: &str, plan: Plan) -> Result<RunResult> {
        self.run_job(request, Some(plan), None, None).await
    }

    async fn run_job(&self, request: &str, plan: Option<Plan>, job_id: Option<String>, prior: Option<Map<String, Value>>) -> Result<RunResult> {
        let run_id = job_id.unwrap_or_else(|| new_id("job"));
        let prior = prior.unwrap_or_default();
        let results: IndexMap<String, RunResult> = prior
            .get("results")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .map(|(k, h)| {
                        (
                            k.clone(),
                            RunResult {
                                agent: h["agent"].as_str().unwrap_or("").into(),
                                output: h["output"].as_str().unwrap_or("").into(),
                                error: h["error"].as_str().map(str::to_string),
                                run_id: k.clone(),
                                usage: serde_json::from_value(h["usage"].clone()).unwrap_or_default(),
                                ..Default::default()
                            },
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut record = Map::new();
        record.insert("id".into(), json!(run_id));
        record.insert("request".into(), json!(request));
        record.insert("status".into(), json!("running"));
        record.insert("created".into(), prior.get("created").cloned().unwrap_or_else(|| json!(now())));
        let start_round = prior.get("round").and_then(Value::as_u64).unwrap_or(0) as u32;
        record.insert("round".into(), json!(start_round));
        // One guard for the whole job: the plan, the specs, the work, the
        // consolidation and the review were all paid for.
        let job = Job {
            id: run_id.clone(),
            request: request.into(),
            guard: self.harness.guard().child(Budget::default()),
            results: Mutex::new(results),
            replans: Mutex::new(prior.get("replans").and_then(Value::as_u64).unwrap_or(0) as u32),
            warnings: Mutex::default(),
            record: Mutex::new(record),
            version: Mutex::new(prior.get("_version").and_then(Value::as_u64).unwrap_or(0)),
            prior: prior.get("usage").and_then(|u| serde_json::from_value(u.clone()).ok()).unwrap_or_default(),
            prior_by_agent: prior.get("by_agent").and_then(Value::as_object).cloned().unwrap_or_default(),
        };
        let mut result = RunResult { agent: self.manager.name().into(), run_id: run_id.clone(), ..Default::default() };
        let mut deliverable = String::new();
        let mut review = Review::default();
        let mut plan = plan;
        let resumed = !prior.is_empty();

        let outcome: Result<()> = async {
            // Claimed before any work: a job someone else runs is found out here.
            self.checkpoint(plan.as_ref(), if resumed { "resumed" } else { "planning" }, &job, Map::new()).await?;
            if plan.is_none() {
                plan = Some(self.plan_in(request, Some(&job)).await?);
            }
            let plan = plan.as_mut().expect("planned");
            self.checkpoint(Some(plan), "planned", &job, Map::new()).await?;
            // Cut short between consolidating and reviewing: what was consolidated is still good.
            let mut held = if prior.get("stage").and_then(Value::as_str) == Some("consolidated") {
                prior.get("deliverable").and_then(Value::as_str).unwrap_or("").to_string()
            } else {
                String::new()
            };
            for round in start_round..=self.max_rework {
                if !held.is_empty() {
                    deliverable = std::mem::take(&mut held);
                } else {
                    self.execute(plan, request, &job).await?;
                    deliverable = self.consolidate(plan, request, &job).await?;
                    let mut more = Map::new();
                    more.insert("deliverable".into(), json!(deliverable));
                    more.insert("round".into(), json!(round));
                    self.checkpoint(Some(plan), "consolidated", &job, more).await?;
                }
                review = self.review_in(plan, &deliverable, request, Some(&job)).await;
                let mut more = Map::new();
                more.insert("review".into(), serde_json::to_value(&review).unwrap_or_default());
                more.insert("deliverable".into(), json!(deliverable));
                more.insert("round".into(), json!(round));
                self.checkpoint(Some(plan), "reviewed", &job, more).await?;
                if review.accepted || round == self.max_rework || !review.reviewed {
                    break;
                }
                if plan.tasks.iter().any(|t| t.error.as_deref().is_some_and(|e| e.starts_with("not started:"))) {
                    break; // no money for the work; none for rework
                }
                // Rejected: re-plan only the gaps, then run those.
                let gaps = if review.gaps.is_empty() { review.problems.clone() } else { review.gaps.clone() };
                for gap in gaps {
                    plan.tasks.push(Task { done_when: "the gap the reviewer named is closed".into(), ..Task::new(&clip(&new_id("gap"), 8), &gap) });
                }
                let mut more = Map::new();
                more.insert("round".into(), json!(round + 1));
                self.checkpoint(Some(plan), "rework", &job, more).await?;
            }
            result.output = deliverable.clone();
            result.data = json!({"plan": plan, "review": review});
            result.stop_reason = if review.accepted { StopReason::EndTurn } else { StopReason::Stopped };
            if self.review && !deliverable.is_empty() && !review.reviewed {
                lock(&job.warnings).push("the deliverable was not reviewed: no critic returned a usable verdict".into());
            }
            if self.on_over_estimate == "stop" && plan.tasks.iter().any(|t| t.error.as_deref().is_some_and(|e| e.starts_with("not started:"))) {
                result.stop_reason = StopReason::Budget;
                result.budget_exceeded = Some("estimate".into());
            }
            Ok(())
        }
        .await;
        if let Err(e) = &outcome {
            // The manager reports; it does not crash the caller.
            result.error = Some(format!("{}: {e}", e.kind_name()));
            result.stop_reason = StopReason::Error;
        }

        let children: Vec<RunResult> = lock(&job.results).values().cloned().collect();
        for child in &children {
            result.artifacts.extend(child.artifacts.clone());
        }
        result.steps = children.len() as u32;
        if let Some(p) = &plan {
            // What tasks cost here, for the next plan's estimate.
            let paid: Vec<f64> = p.tasks.iter().filter(|t| t.status == TaskStatus::Done && t.cost_usd > 0.0).map(|t| t.cost_usd).collect();
            let mut seen = lock(&self.seen_tasks);
            *seen = (seen.0 + paid.len() as u32, seen.1 + paid.iter().sum::<f64>());
        }
        // `usage` is the manager's own: everything the job paid for that was not
        // the sub-agents' work. `cost_usd()` adds the children's.
        let total = job.prior.clone() + job.guard.usage();
        let work = children.iter().fold(Usage::default(), |acc, r| acc + r.usage.clone());
        result.usage = Usage {
            input_tokens: total.input_tokens.saturating_sub(work.input_tokens),
            output_tokens: total.output_tokens.saturating_sub(work.output_tokens),
            cache_read_tokens: total.cache_read_tokens.saturating_sub(work.cache_read_tokens),
            cache_write_tokens: total.cache_write_tokens.saturating_sub(work.cache_write_tokens),
            cost_usd: (total.cost_usd - work.cost_usd).max(0.0),
            calls: total.calls.saturating_sub(work.calls),
        };
        let r6 = |x: f64| (x * 1e6).round() / 1e6;
        let mut spend = Map::new();
        spend.insert("total_usd".into(), json!(r6(total.cost_usd)));
        spend.insert("work_usd".into(), json!(r6(work.cost_usd)));
        spend.insert("overhead_usd".into(), json!(r6(result.usage.cost_usd)));
        spend.insert("tokens".into(), json!(total.total_tokens()));
        spend.insert("calls".into(), json!(total.calls));
        spend.insert("by_agent".into(), Value::Object(Self::by_agent(&job)));
        result.spend = spend;
        result.children = children;
        if !result.error.as_deref().unwrap_or("").contains("is being run by another process") {
            let mut more = Map::new();
            more.insert("status".into(), json!(if result.error.is_some() { "failed" } else { "done" }));
            more.insert("deliverable".into(), json!(deliverable));
            more.insert("review".into(), serde_json::to_value(&review).unwrap_or_default());
            more.insert("stop_reason".into(), serde_json::to_value(result.stop_reason).unwrap_or_default());
            more.insert("error".into(), json!(result.error));
            more.insert("spend".into(), Value::Object(result.spend.clone()));
            if let Err(e) = self.checkpoint(plan.as_ref(), "finished", &job, more).await {
                lock(&job.warnings).push(e.to_string());
            }
        }
        result.warnings.extend(lock(&job.warnings).iter().cloned());
        result.artifacts.push(Artifact {
            media_type: "application/json".into(),
            produced_by: self.manager.name().into(),
            ..Artifact::new("plan.json", plan.as_ref().map(|p| serde_json::to_string_pretty(p).unwrap_or_default()).unwrap_or_else(|| "{}".into()))
        });
        if !deliverable.is_empty() {
            result.artifacts.push(Artifact {
                media_type: "text/markdown".into(),
                produced_by: self.manager.name().into(),
                ..Artifact::new("deliverable.md", deliverable)
            });
        }
        Ok(result)
    }

    /// One glance at the run: spend and throughput.
    pub async fn report(&self) -> Value {
        self.harness.report().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waves_group_independent_tasks() {
        let plan = Plan {
            tasks: vec![
                Task::new("t1", "a"),
                Task::new("t2", "b"),
                Task { depends_on: vec!["t1".into(), "t2".into()], ..Task::new("t3", "c") },
                Task { depends_on: vec!["ghost".into()], ..Task::new("t4", "d") },
            ],
            ..Default::default()
        };
        let waves: Vec<Vec<&str>> = plan.waves().iter().map(|w| w.iter().map(|t| t.id.as_str()).collect()).collect();
        assert_eq!(waves, vec![vec!["t1", "t2"], vec!["t3"], vec!["t4"]]);
        assert!(plan.render().contains("[t3] c (after t1, t2)"));
    }

    #[test]
    fn readiness_and_cycles() {
        let mut plan = Plan {
            tasks: vec![Task { depends_on: vec!["t2".into()], ..Task::new("t1", "a") }, Task { depends_on: vec!["t1".into()], ..Task::new("t2", "b") }],
            ..Default::default()
        };
        assert_eq!(Orchestrator::ready(&plan), vec!["t1", "t2"], "a cycle runs together rather than hang");
        plan.tasks[0].status = TaskStatus::Done;
        assert_eq!(Orchestrator::ready(&plan), vec!["t2"]);
    }
}
