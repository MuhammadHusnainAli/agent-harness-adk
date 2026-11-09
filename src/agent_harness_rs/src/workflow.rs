//! Declared workflows: the steps are written down, and run as written.
//!
//! ```yaml
//! name: refunds
//! inputs: {order: {required: true}, message: {required: true}}
//! steps:
//!   - id: charges
//!     tool: find_charges
//!     args: {order: "{{ inputs.order }}"}
//!   - id: classify
//!     agent: classifier
//!     input: "Charges: {{ steps.charges.output }}\nCustomer: {{ inputs.message }}"
//!     parse: json
//!     save: verdict
//!   - if: "state.verdict.duplicate"
//!     then:
//!       - tool: issue_refund
//!         args: {order: "{{ inputs.order }}", amount: "{{ state.verdict.amount }}"}
//!     else:
//!       - agent: support
//!         input: "Explain why order {{ inputs.order }} is not refunded."
//! output: "{{ previous }}"
//! ```
//!
//! ```no_run
//! # async fn demo(classifier: agent_harness::Agent, support: agent_harness::Agent,
//! #               find_charges: agent_harness::Tool, issue_refund: agent_harness::Tool) -> agent_harness::Result<()> {
//! use agent_harness::workflow::Workflow;
//! use serde_json::json;
//! let workflow = Workflow::from_file("refunds.yaml")?
//!     .agents([classifier, support])
//!     .tools([find_charges, issue_refund])
//!     .build()?;
//! let result = workflow.run(json!({"order": "4182", "message": "I was charged twice."})).await?;
//! # Ok(()) }
//! ```
//!
//! A step is one of: `agent`, `tool`, `set`, `steps` (a sequence), `parallel`,
//! `foreach`, `loop`, `if`, `switch`, `graph` (nodes with `needs`), `wait`,
//! `fail`, `return`. Any step may carry `when`, `retry`, `timeout`,
//! `on_error: continue` and `save`. Values are templates: `{{ ... }}` holds an
//! [`crate::expr`] expression over `inputs`, `state`, `steps.<id>` and
//! `previous`. A value that is one expression keeps its type; inside other
//! text it is written out. Expressions are parsed, never executed as code.
//!
//! Tools are passed in: Rust cannot import one from a path in the file.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, FuturesUnordered, StreamExt};
use indexmap::IndexMap;
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value, json};
use tokio::sync::Semaphore;

use crate::agent::{Agent, RunOptions, SessionRef, extract_json};
use crate::blueprint::{AgentEntry, Blueprint};
use crate::errors::{Error, Result};
use crate::expr::{CONSTANTS, Ctx, compile, text, truthy};
use crate::harness::Harness;
use crate::runtime::{Budget, BudgetGuard, Session};
use crate::subagents::SubAgentSpec;
use crate::tools::{Tool, ToolRegistry};
use crate::types::{Artifact, Usage, attach, new_id, round8};

/// What a step can be. Exactly one of these keys says which.
pub const KINDS: [&str; 12] = ["agent", "tool", "set", "parallel", "foreach", "loop", "if", "switch", "graph", "wait", "fail", "return"];

static ID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").expect("valid regex"));
static TEMPLATE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)\{\{(.*?)\}\}").expect("valid regex"));

fn werr(message: impl Into<String>) -> Error {
    Error::Workflow(message.into())
}

// ----------------------------------------------------------------------
// templates
// ----------------------------------------------------------------------
fn whole_template(s: &str) -> Option<String> {
    let t = s.trim();
    let caps = TEMPLATE.captures(t)?;
    let m = caps.get(0)?;
    (m.start() == 0 && m.end() == t.len()).then(|| caps[1].trim().to_string())
}

/// A condition may be written with or without the braces.
fn bare(source: &str) -> String {
    whole_template(source).unwrap_or_else(|| source.trim().to_string())
}

/// Fill the templates in a value — a string, or anything holding strings.
/// `"{{ state.total }}"` is the total itself; `"Total: {{ state.total }}"` is text.
pub fn render(value: &Value, ctx: &Ctx) -> Result<Value> {
    Ok(match value {
        Value::String(s) if s.contains("{{") => {
            if let Some(expr) = whole_template(s) {
                compile(&expr)?.eval(ctx)
            } else {
                let mut out = String::new();
                let mut last = 0;
                for caps in TEMPLATE.captures_iter(s) {
                    let m = caps.get(0).expect("whole match");
                    out.push_str(&s[last..m.start()]);
                    out.push_str(&text(&compile(caps[1].trim())?.eval(ctx)));
                    last = m.end();
                }
                out.push_str(&s[last..]);
                Value::String(out)
            }
        }
        Value::Object(m) => Value::Object(m.iter().map(|(k, v)| Ok((k.clone(), render(v, ctx)?))).collect::<Result<_>>()?),
        Value::Array(a) => Value::Array(a.iter().map(|v| render(v, ctx)).collect::<Result<_>>()?),
        other => other.clone(),
    })
}

fn expressions(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::String(s) => out.extend(TEMPLATE.captures_iter(s).map(|c| c[1].trim().to_string())),
        Value::Object(m) => m.values().for_each(|v| expressions(v, out)),
        Value::Array(a) => a.iter().for_each(|v| expressions(v, out)),
        _ => {}
    }
}

fn condition(source: &str, ctx: &Ctx) -> Result<bool> {
    Ok(truthy(&compile(&bare(source))?.eval(ctx)))
}

// ----------------------------------------------------------------------
// the declaration
// ----------------------------------------------------------------------
/// Try a step again when it fails: `retry: 2`, or the mapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Retry {
    #[serde(default = "one")]
    pub max: u32,
    #[serde(default)]
    pub delay: f64,
    #[serde(default = "two")]
    pub backoff: f64,
}

fn one() -> u32 {
    1
}

fn two() -> f64 {
    2.0
}

/// How long a loop goes on: `loop: 5`, or the mapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoopSpec {
    #[serde(default = "ten")]
    pub max: u32,
    /// Checked after each pass.
    #[serde(default)]
    pub until: Option<String>,
    /// Checked before each pass.
    #[serde(default, rename = "while")]
    pub while_: Option<String>,
}

fn ten() -> u32 {
    10
}

impl Default for LoopSpec {
    fn default() -> Self {
        Self { max: 10, until: None, while_: None }
    }
}

/// One branch of a `switch`. Without `when`, it is the default.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    #[serde(default)]
    pub when: Option<String>,
    #[serde(default)]
    pub steps: Vec<Step>,
}

fn retry_de<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<Retry>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum R {
        Count(u32),
        Full(Retry),
    }
    Ok(match Option::<R>::deserialize(d)? {
        None => None,
        Some(R::Count(n)) => Some(Retry { max: n, delay: 0.0, backoff: 2.0 }),
        Some(R::Full(r)) => Some(r),
    })
}

fn loop_de<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<LoopSpec>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum L {
        Flag(bool),
        Count(u32),
        Full(LoopSpec),
    }
    Ok(match Option::<L>::deserialize(d)? {
        None | Some(L::Flag(false)) => None,
        Some(L::Flag(true)) => Some(LoopSpec::default()),
        Some(L::Count(n)) => Some(LoopSpec { max: n, ..Default::default() }),
        Some(L::Full(l)) => Some(l),
    })
}

/// `return: null` is still a return: keep that it was written.
fn present<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<Value>, D::Error> {
    Value::deserialize(d).map(Some)
}

fn item() -> String {
    "item".into()
}

fn all() -> String {
    "all".into()
}

fn fail_mode() -> String {
    "fail".into()
}

/// One step, as declared. Which kind it is, is the key it carries.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub description: String,
    // ---- what it is: exactly one -------------------------------------
    /// A name, or an agent declared right here (as in a blueprint).
    #[serde(default)]
    pub agent: Option<Value>,
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(default, rename = "set")]
    pub assign: Option<Map<String, Value>>,
    #[serde(default)]
    pub parallel: Option<Vec<Step>>,
    #[serde(default)]
    pub foreach: Option<Value>,
    #[serde(default, rename = "loop", deserialize_with = "loop_de")]
    pub loop_: Option<LoopSpec>,
    #[serde(default, rename = "if")]
    pub if_: Option<String>,
    #[serde(default)]
    pub switch: Option<Vec<Case>>,
    #[serde(default)]
    pub graph: Option<Vec<Step>>,
    #[serde(default)]
    pub wait: Option<Value>,
    #[serde(default)]
    pub fail: Option<String>,
    #[serde(default, rename = "return", deserialize_with = "present")]
    pub return_: Option<Value>,
    /// A body for `foreach` and `loop`; on its own, a sequence run as one step.
    #[serde(default)]
    pub steps: Option<Vec<Step>>,
    // ---- what goes with a kind ---------------------------------------
    #[serde(default)]
    pub input: Option<Value>,
    #[serde(default)]
    pub attachments: Option<Value>,
    #[serde(default)]
    pub version: Option<String>,
    /// Carry this agent's conversation through the whole run.
    #[serde(default)]
    pub thread: bool,
    /// `json`: the agent's answer must be (and is parsed as) JSON.
    #[serde(default)]
    pub parse: Option<String>,
    #[serde(default)]
    pub args: Map<String, Value>,
    #[serde(default = "item", rename = "as")]
    pub as_: String,
    #[serde(default)]
    pub concurrency: Option<usize>,
    #[serde(default)]
    pub then: Option<Vec<Step>>,
    #[serde(default, rename = "else")]
    pub else_: Option<Vec<Step>>,
    #[serde(default)]
    pub needs: Vec<String>,
    /// `all` (default) or `any`: what a graph node waits for.
    #[serde(default = "all")]
    pub join: String,
    // ---- what any step may carry -------------------------------------
    #[serde(default)]
    pub when: Option<String>,
    #[serde(default)]
    pub save: Option<String>,
    #[serde(default, deserialize_with = "retry_de")]
    pub retry: Option<Retry>,
    #[serde(default)]
    pub timeout: Option<f64>,
    /// `fail` (default) or `continue`.
    #[serde(default = "fail_mode")]
    pub on_error: String,
}

impl Step {
    fn container(id: &str, steps: Option<Vec<Step>>, graph: Option<Vec<Step>>) -> Step {
        let mut s: Step = serde_json::from_value(json!({"id": id})).expect("an empty step parses");
        s.steps = steps;
        s.graph = graph;
        s
    }

    fn kinds(&self) -> Vec<&'static str> {
        let present = [
            ("agent", self.agent.is_some()),
            ("tool", self.tool.is_some()),
            ("set", self.assign.is_some()),
            ("parallel", self.parallel.is_some()),
            ("foreach", self.foreach.is_some()),
            ("loop", self.loop_.is_some()),
            ("if", self.if_.is_some()),
            ("switch", self.switch.is_some()),
            ("graph", self.graph.is_some()),
            ("wait", self.wait.is_some()),
            ("fail", self.fail.is_some()),
            ("return", self.return_.is_some()),
        ];
        let mut found: Vec<&'static str> = present.iter().filter(|(_, p)| *p).map(|(k, _)| *k).collect();
        if found.is_empty() && self.steps.is_some() {
            found.push("steps");
        }
        if found.is_empty() {
            found.push("");
        }
        found
    }

    /// Which kind of step it is.
    pub fn kind(&self) -> &'static str {
        self.kinds()[0]
    }

    /// Its bodies, each with a word for where it hangs.
    fn children(&self) -> Vec<(String, &Vec<Step>)> {
        let mut out = Vec::new();
        for (label, body) in [("parallel", &self.parallel), ("graph", &self.graph), ("steps", &self.steps), ("then", &self.then), ("else", &self.else_)] {
            if let Some(b) = body.as_ref().filter(|b| !b.is_empty()) {
                out.push((label.to_string(), b));
            }
        }
        for (i, case) in self.switch.iter().flatten().enumerate() {
            out.push((if case.when.is_some() { format!("case {}", i + 1) } else { "default".into() }, &case.steps));
        }
        out
    }

    fn children_mut(&mut self) -> Vec<&mut Vec<Step>> {
        let mut out: Vec<&mut Vec<Step>> = Vec::new();
        for body in [&mut self.parallel, &mut self.graph, &mut self.steps, &mut self.then, &mut self.else_].into_iter().flatten() {
            out.push(body);
        }
        for case in self.switch.iter_mut().flatten() {
            out.push(&mut case.steps);
        }
        out
    }
}

/// A whole workflow, as a file declares it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowSpec {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "version_one")]
    pub version: Value,
    /// A name to its default, or to `{default, required, type, description}`.
    #[serde(default)]
    pub inputs: IndexMap<String, Value>,
    /// What the shared state starts as.
    #[serde(default)]
    pub state: Map<String, Value>,
    #[serde(default)]
    pub steps: Option<Vec<Step>>,
    /// Instead of `steps`: nodes that say what they need, run as a graph.
    #[serde(default)]
    pub graph: Option<Vec<Step>>,
    /// What the run hands back. Left out, the last step's output.
    #[serde(default)]
    pub output: Option<Value>,
    #[serde(default)]
    pub budget: Option<Budget>,
    #[serde(default)]
    pub timeout: Option<f64>,
    /// How many steps one run may execute — a loop that never ends, ends here.
    #[serde(default = "thousand")]
    pub max_steps: u32,
    /// How many branches run at once, where a step does not say.
    #[serde(default = "eight")]
    pub concurrency: usize,
    // ---- agents declared with it, exactly as in a blueprint ----------
    #[serde(default)]
    pub agents: IndexMap<String, Value>,
    #[serde(default)]
    pub subagents: IndexMap<String, Value>,
    #[serde(default)]
    pub prompts: IndexMap<String, String>,
    #[serde(default)]
    pub defaults: Map<String, Value>,
    #[serde(default)]
    pub guardrails: IndexMap<String, Value>,
    #[serde(default)]
    pub openapi: Map<String, Value>,
}

fn default_name() -> String {
    "workflow".into()
}

fn version_one() -> Value {
    json!("1")
}

fn thousand() -> u32 {
    1000
}

fn eight() -> usize {
    8
}

impl WorkflowSpec {
    pub fn input_specs(&self) -> IndexMap<String, Map<String, Value>> {
        const KEYS: [&str; 4] = ["default", "required", "description", "type"];
        self.inputs
            .iter()
            .map(|(name, v)| {
                let spec = match v {
                    Value::Object(m) if m.keys().all(|k| KEYS.contains(&k.as_str())) => m.clone(),
                    other => {
                        let mut m = Map::new();
                        m.insert("default".into(), other.clone());
                        m
                    }
                };
                (name.clone(), spec)
            })
            .collect()
    }
}

// ----------------------------------------------------------------------
// what a run produces
// ----------------------------------------------------------------------
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepStatus {
    #[default]
    Done,
    Skipped,
    Failed,
}

/// What one step came to. `steps.<id>` in a template is one of these.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StepResult {
    pub id: String,
    pub kind: String,
    pub status: StepStatus,
    pub output: Value,
    /// The structured form: an agent's output contract, parsed JSON, or what a tool returned.
    pub data: Value,
    pub error: Option<String>,
    pub attempts: u32,
    pub duration_ms: f64,
    pub cost_usd: f64,
    /// The agent that answered, for an agent step.
    pub agent: String,
    pub iterations: u32,
    pub exhausted: bool,
    /// The branch an `if` or a `switch` took.
    pub branch: String,
}

impl StepResult {
    pub fn ok(&self) -> bool {
        self.status != StepStatus::Failed
    }
}

/// One beat of a running workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowEvent {
    /// workflow_start · step_start · step_end · step_skipped · step_failed · step_retry · workflow_end
    #[serde(rename = "type")]
    pub event: String,
    pub step: String,
    pub kind: String,
    pub text: String,
    pub data: Map<String, Value>,
}

impl WorkflowEvent {
    fn new(event: &str, step: &str, kind: &str, text: &str) -> Self {
        Self { event: event.into(), step: step.into(), kind: kind.into(), text: text.into(), data: Map::new() }
    }

    fn with(mut self, key: &str, value: Value) -> Self {
        self.data.insert(key.into(), value);
        self
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkflowStatus {
    #[default]
    Done,
    Failed,
    Stopped,
}

/// The one object a caller gets back from `workflow.run()`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkflowResult {
    pub workflow: String,
    pub run_id: String,
    pub status: WorkflowStatus,
    pub output: Value,
    pub inputs: Map<String, Value>,
    pub state: Map<String, Value>,
    pub steps: IndexMap<String, StepResult>,
    pub error: Option<String>,
    /// The step the failure started in.
    pub failed_step: String,
    pub usage: Usage,
    pub cost_usd: f64,
    pub steps_run: u32,
    pub duration_ms: f64,
    pub artifacts: Vec<Artifact>,
}

impl WorkflowResult {
    pub fn ok(&self) -> bool {
        self.status == WorkflowStatus::Done
    }
}

// ----------------------------------------------------------------------
// running
// ----------------------------------------------------------------------
/// How a step's work can end other than by finishing.
enum Flow {
    /// A `return` step: the workflow is finished, with this.
    Return(Value),
    /// A step failed and was not told to carry on.
    Failed(String),
    Error(Error),
}

impl From<Error> for Flow {
    fn from(e: Error) -> Self {
        Flow::Error(e)
    }
}

type FlowResult<T> = std::result::Result<T, Flow>;

/// What the steps came to, as a template sees `steps`: a branch with its own
/// view (one pass of a `foreach`) writes there, and reads through to its parent.
#[derive(Default)]
struct StepsView {
    own: Mutex<Map<String, Value>>,
    parent: Option<Arc<StepsView>>,
}

impl StepsView {
    fn set(&self, id: &str, result: &StepResult) {
        lock(&self.own).insert(id.into(), serde_json::to_value(result).unwrap_or_default());
    }

    fn flatten(&self) -> Map<String, Value> {
        let mut out = self.parent.as_ref().map(|p| p.flatten()).unwrap_or_default();
        out.extend(lock(&self.own).clone());
        out
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// What a template can see from where a step stands.
#[derive(Clone)]
struct Scope {
    inputs: Arc<Map<String, Value>>,
    state: Arc<Mutex<Map<String, Value>>>,
    steps: Arc<StepsView>,
    names: Map<String, Value>,
}

impl Scope {
    fn context(&self) -> Ctx {
        let mut c = Map::new();
        c.insert("previous".into(), Value::Null);
        c.extend(self.names.clone());
        c.insert("inputs".into(), Value::Object((*self.inputs).clone()));
        c.insert("state".into(), Value::Object(lock(&self.state).clone()));
        c.insert("steps".into(), Value::Object(self.steps.flatten()));
        c
    }

    /// The same state, its own `previous`; with `own_steps`, what its steps
    /// come to is not seen outside it.
    fn child(&self, own_steps: bool, names: &[(&str, Value)]) -> Scope {
        let steps = if own_steps { Arc::new(StepsView { own: Mutex::default(), parent: Some(self.steps.clone()) }) } else { self.steps.clone() };
        let mut n = self.names.clone();
        for (k, v) in names {
            n.insert((*k).into(), v.clone());
        }
        Scope { inputs: self.inputs.clone(), state: self.state.clone(), steps, names: n }
    }
}

type Emit = Arc<dyn Fn(WorkflowEvent) + Send + Sync>;

struct Run {
    id: String,
    guard: Option<Arc<BudgetGuard>>,
    emit: Option<Emit>,
    results: Mutex<IndexMap<String, StepResult>>,
    usage: Mutex<Usage>,
    cost: Mutex<f64>,
    count: AtomicU32,
    failed: Mutex<String>,
    threads: tokio::sync::Mutex<HashMap<String, Session>>,
    artifacts: Mutex<Vec<Artifact>>,
}

impl Run {
    fn emit(&self, e: WorkflowEvent) {
        if let Some(f) = &self.emit {
            f(e);
        }
    }

    fn mark_failed(&self, step: &str) {
        let mut f = lock(&self.failed);
        if f.is_empty() {
            *f = step.into();
        }
    }
}

struct Inner {
    spec: WorkflowSpec,
    root: Step,
    ids: IndexMap<String, Step>,
    harness: Harness,
    tools: ToolRegistry,
    agents: HashMap<String, Agent>,
    runner: OnceLock<Agent>,
}

/// A declared workflow, with the agents and tools its steps name. Cheap to clone.
#[derive(Clone)]
pub struct Workflow {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Workflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Workflow {} steps={}>", self.inner.spec.name, self.inner.ids.len())
    }
}

/// Builds a [`Workflow`]: the declaration, plus the agents and tools it names.
pub struct WorkflowBuilder {
    spec: WorkflowSpec,
    agents: Vec<Agent>,
    tools: Vec<Tool>,
    harness: Option<Harness>,
    blueprint: Option<Blueprint>,
}

impl WorkflowBuilder {
    /// The agents its `agent` steps name. Agents the file declares itself are built for you.
    pub fn agents<I: IntoIterator<Item = Agent>>(mut self, agents: I) -> Self {
        self.agents.extend(agents);
        self
    }

    /// The tools its `tool` steps call, and that its declared agents may be given.
    pub fn tools<I: IntoIterator<Item = Tool>>(mut self, tools: I) -> Self {
        self.tools.extend(tools);
        self
    }

    /// The harness it runs on. Left out, the one its agents share.
    pub fn harness(mut self, harness: Harness) -> Self {
        self.harness = Some(harness);
        self
    }

    /// A blueprint to find agents in that were not handed over.
    pub fn blueprint(mut self, blueprint: Blueprint) -> Self {
        self.blueprint = Some(blueprint);
        self
    }

    pub fn build(self) -> Result<Workflow> {
        let spec = self.spec;
        let name = spec.name.clone();
        if !spec.openapi.is_empty() {
            return Err(Error::config("the `openapi` section is not in the Rust port yet (M5; see track.md)"));
        }
        let mut root = match (&spec.steps, &spec.graph) {
            (Some(s), None) => Step::container("_root", Some(s.clone()), None),
            (None, Some(g)) => Step::container("_root", None, Some(g.clone())),
            _ => return Err(Error::config(format!("workflow '{name}' needs `steps` or `graph` — one of them"))),
        };
        let mut counter = 0u32;
        let mut seen = HashSet::new();
        number(&mut root, &mut seen, &mut counter)?;
        let mut ids = IndexMap::new();
        collect(&root, &mut ids);
        check(&root, &ids, &spec)?;

        let given: HashMap<String, Agent> = self.agents.into_iter().map(|a| (a.name().to_string(), a)).collect();
        let harness = self.harness.or_else(|| given.values().next().map(|a| a.harness().clone())).unwrap_or_default();
        let tools = ToolRegistry::new(self.tools);
        for step in ids.values() {
            if let Some(t) = &step.tool
                && !tools.contains(t)
            {
                return Err(Error::config(format!(
                    "workflow '{name}', step '{}': no tool named '{t}'. Pass it to `.tools(...)`. Known: {}",
                    step.id,
                    known(tools.names().into_iter())
                )));
            }
        }
        let agents = resolve_agents(&spec, &ids, given, self.blueprint, &tools, &harness)?;
        Ok(Workflow { inner: Arc::new(Inner { spec, root, ids, harness, tools, agents, runner: OnceLock::new() }) })
    }
}

fn known(names: impl Iterator<Item = String>) -> String {
    let mut v: Vec<String> = names.collect();
    v.sort();
    v.dedup();
    if v.is_empty() { "none".into() } else { v.join(", ") }
}

/// Every agent a step names: handed in, or built from a declaration.
fn resolve_agents(
    spec: &WorkflowSpec,
    ids: &IndexMap<String, Step>,
    given: HashMap<String, Agent>,
    blueprint: Option<Blueprint>,
    tools: &ToolRegistry,
    harness: &Harness,
) -> Result<HashMap<String, Agent>> {
    let mut wanted: IndexMap<String, String> = IndexMap::new(); // agent name → the step that names it
    let mut inline: IndexMap<String, Value> = IndexMap::new();
    for step in ids.values() {
        match &step.agent {
            Some(Value::Object(m)) => {
                inline.insert(step.id.clone(), Value::Object(m.clone()));
                wanted.insert(step.id.clone(), step.id.clone());
            }
            Some(Value::String(n)) => {
                wanted.entry(n.clone()).or_insert_with(|| step.id.clone());
            }
            Some(other) => return Err(Error::config(format!("step '{}': `agent` is a name or a mapping — got {other}", step.id))),
            None => {}
        }
    }
    if wanted.keys().all(|n| given.contains_key(n)) {
        return Ok(wanted.keys().map(|n| (n.clone(), given[n].clone())).collect());
    }
    let parse_entry =
        |v: &Value| serde_json::from_value::<AgentEntry>(v.clone()).map_err(|e| Error::config(format!("the workflow is not valid — agents: {e}")));
    let mut declared = blueprint.unwrap_or_default();
    for (k, v) in spec.agents.iter().chain(inline.iter()) {
        declared.agents.insert(k.clone(), parse_entry(v)?);
    }
    for (k, v) in &spec.subagents {
        let mut v = v.clone();
        if let Value::Object(m) = &mut v {
            m.entry("name").or_insert_with(|| json!(k));
        }
        let s: SubAgentSpec = serde_json::from_value(v).map_err(|e| Error::config(format!("the workflow is not valid — subagents: {e}")))?;
        declared.subagents.insert(k.clone(), s);
    }
    declared.prompts.extend(spec.prompts.clone());
    declared.guardrails.extend(spec.guardrails.clone());
    declared.defaults.extend(spec.defaults.clone());
    let mut built = HashMap::new();
    let mut out = HashMap::new();
    for (name, step) in &wanted {
        if let Some(a) = given.get(name) {
            out.insert(name.clone(), a.clone());
        } else if declared.agents.contains_key(name) {
            out.insert(name.clone(), declared.build_into(name, &mut built, tools, harness)?);
        } else {
            return Err(Error::config(format!(
                "workflow '{}', step '{step}': no agent named '{name}'. Pass it to `.agents(...)`, or declare it under `agents:`. Known: {}",
                spec.name,
                known(given.keys().cloned().chain(declared.agents.keys().cloned()))
            )));
        }
    }
    Ok(out)
}

impl Workflow {
    pub fn builder(spec: WorkflowSpec) -> WorkflowBuilder {
        WorkflowBuilder { spec, agents: vec![], tools: vec![], harness: None, blueprint: None }
    }

    /// From the mapping a file holds.
    pub fn from_value(data: Value) -> Result<WorkflowBuilder> {
        if !data.is_object() {
            return Err(Error::config("a workflow must be a mapping at the top level"));
        }
        let spec: WorkflowSpec = serde_json::from_value(data).map_err(|e| Error::config(format!("the workflow is not valid — {e}")))?;
        Ok(Self::builder(spec))
    }

    pub fn from_text(text: &str, fmt: &str) -> Result<WorkflowBuilder> {
        let data: Value = if fmt == "json" {
            serde_json::from_str(text).map_err(|e| Error::config(format!("the workflow could not be read: {e}")))?
        } else {
            serde_yaml::from_str(text).map_err(|e| Error::config(format!("the workflow could not be read: {e}")))?
        };
        Self::from_value(data)
    }

    /// Load a `.yaml`, `.yml` or `.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<WorkflowBuilder> {
        let file = path.as_ref();
        if !file.is_file() {
            return Err(Error::config(format!("no workflow at {}", file.display())));
        }
        let fmt = file.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
        Self::from_text(&std::fs::read_to_string(file)?, &fmt)
    }

    pub fn name(&self) -> &str {
        &self.inner.spec.name
    }

    pub fn spec(&self) -> &WorkflowSpec {
        &self.inner.spec
    }

    /// Every step, by id.
    pub fn steps(&self) -> &IndexMap<String, Step> {
        &self.inner.ids
    }

    pub fn harness(&self) -> &Harness {
        &self.inner.harness
    }

    /// The agent a `tool` step is called through, so the call passes the same
    /// hooks and permission gate as any other.
    fn runner(&self) -> Result<&Agent> {
        if let Some(a) = self.inner.runner.get() {
            return Ok(a);
        }
        let spec = &self.inner.spec;
        let description = if spec.description.is_empty() { format!("the {} workflow", spec.name) } else { spec.description.clone() };
        let mut b = Agent::builder(&spec.name)
            .description(description)
            .tools(self.inner.tools.iter().cloned())
            .harness(self.inner.harness.clone())
            .memory(false)
            .persist_session(false);
        if let Some(p) = &self.inner.harness.provider {
            b = b.provider(p.clone());
        }
        let _ = self.inner.runner.set(b.build()?);
        Ok(self.inner.runner.get().expect("just set"))
    }

    // ---- describing --------------------------------------------------------
    /// The workflow as an outline: what runs, in what shape.
    pub fn describe(&self) -> String {
        let spec = &self.inner.spec;
        let mut lines = vec![if spec.description.is_empty() { spec.name.clone() } else { format!("{} — {}", spec.name, spec.description) }];
        let specs = spec.input_specs();
        if !specs.is_empty() {
            let names: Vec<String> = specs.iter().map(|(n, s)| format!("{n}{}", if s.get("required").is_some_and(truthy) { "" } else { "?" })).collect();
            lines.push(format!("inputs: {}", names.join(", ")));
        }
        fn walk(steps: &[Step], depth: usize, lines: &mut Vec<String>) {
            for step in steps {
                let kind = step.kind();
                let detail = match kind {
                    "agent" => Some(match &step.agent {
                        Some(Value::String(n)) => n.clone(),
                        _ => step.id.clone(),
                    }),
                    "tool" => step.tool.clone(),
                    "foreach" => Some(format!("as {}", step.as_)),
                    "if" => step.if_.clone(),
                    "wait" => step.wait.as_ref().map(text),
                    _ => None,
                };
                let extras: Vec<String> = [
                    (!step.needs.is_empty()).then(|| format!("needs {}", step.needs.join(", "))),
                    step.when.as_ref().map(|w| format!("when {w}")),
                    step.loop_.as_ref().and_then(|l| l.until.as_ref()).map(|u| format!("until {u}")),
                    step.save.as_ref().map(|s| format!("→ state.{s}")),
                ]
                .into_iter()
                .flatten()
                .collect();
                let mut line = format!("{}- {}: {kind}", "  ".repeat(depth), step.id);
                if let Some(d) = detail {
                    line.push_str(&format!(" {d}"));
                }
                if !extras.is_empty() {
                    line.push_str(&format!("  ({})", extras.join(" · ")));
                }
                lines.push(line);
                let bodies = step.children();
                let several = bodies.len() > 1;
                for (label, body) in bodies {
                    if several || ["then", "else", "default"].contains(&label.as_str()) {
                        lines.push(format!("{}{label}:", "  ".repeat(depth + 1)));
                        walk(body, depth + 2, lines);
                    } else {
                        walk(body, depth + 1, lines);
                    }
                }
            }
        }
        let root = &self.inner.root;
        walk(root.graph.as_deref().or(root.steps.as_deref()).unwrap_or(&[]), 1, &mut lines);
        lines.join("\n")
    }

    // ---- inputs ------------------------------------------------------------
    fn inputs(&self, given: Value) -> Result<Map<String, Value>> {
        let spec = &self.inner.spec;
        let name = &spec.name;
        let specs = spec.input_specs();
        let given: Map<String, Value> = match given {
            Value::Null => Map::new(),
            Value::Object(m) => m,
            // One thing handed over: it is the input, whatever it is called.
            other => {
                let names: Vec<String> = if specs.is_empty() { vec!["input".into()] } else { specs.keys().cloned().collect() };
                if names.len() != 1 {
                    return Err(Error::config(format!("workflow '{name}' takes {} — pass a mapping", names.join(", "))));
                }
                let mut m = Map::new();
                m.insert(names[0].clone(), other);
                m
            }
        };
        if specs.is_empty() {
            return Ok(given);
        }
        let mut unknown: Vec<&String> = given.keys().filter(|k| !specs.contains_key(*k)).collect();
        unknown.sort();
        if !unknown.is_empty() {
            let u: Vec<&str> = unknown.iter().map(|s| s.as_str()).collect();
            let takes: Vec<&str> = specs.keys().map(String::as_str).collect();
            return Err(Error::config(format!("workflow '{name}' does not take {}; it takes {}", u.join(", "), takes.join(", "))));
        }
        let mut out = Map::new();
        for (key, s) in &specs {
            let value = match given.get(key) {
                Some(v) => v.clone(),
                None if s.get("required").is_some_and(truthy) => return Err(Error::config(format!("workflow '{name}' needs the input '{key}'"))),
                None => s.get("default").cloned().unwrap_or(Value::Null),
            };
            if let Some(t) = s.get("type").and_then(Value::as_str)
                && !value.is_null()
            {
                let ok = match t {
                    "string" => value.is_string(),
                    "number" => value.is_number(),
                    "integer" => value.is_i64() || value.is_u64(),
                    "boolean" => value.is_boolean(),
                    "array" => value.is_array(),
                    "object" => value.is_object(),
                    _ => true,
                };
                if !ok {
                    let got = match &value {
                        Value::String(_) => "str",
                        Value::Number(n) if n.is_f64() => "float",
                        Value::Number(_) => "int",
                        Value::Bool(_) => "bool",
                        Value::Array(_) => "list",
                        _ => "dict",
                    };
                    return Err(Error::config(format!("workflow '{name}': input '{key}' must be {t} — got {got}")));
                }
            }
            out.insert(key.clone(), value);
        }
        Ok(out)
    }

    // ---- running -----------------------------------------------------------
    /// Run it to the end. A step that fails ends the run with `result.error`
    /// set — it is not an `Err`. What is an `Err` is a mistake in the call: an
    /// input that is missing, or one the workflow does not take.
    pub async fn run(&self, inputs: Value) -> Result<WorkflowResult> {
        self.run_with(inputs, Map::new(), None, None).await
    }

    /// Run with a starting state, an event callback, and a run id.
    pub async fn run_with(&self, inputs: Value, state: Map<String, Value>, on_event: Option<Emit>, run_id: Option<String>) -> Result<WorkflowResult> {
        let harness = &self.inner.harness;
        let spec = &self.inner.spec;
        let values = self.inputs(inputs)?;
        let mut result =
            WorkflowResult { workflow: spec.name.clone(), run_id: run_id.unwrap_or_else(|| new_id("wf")), inputs: values.clone(), ..Default::default() };
        let mut shared = spec.state.clone();
        shared.extend(state);
        let guard = spec.budget.as_ref().map(|b| harness.guard().child(b.clone()));
        let run = Arc::new(Run {
            id: result.run_id.clone(),
            guard,
            emit: on_event,
            results: Mutex::default(),
            usage: Mutex::default(),
            cost: Mutex::new(0.0),
            count: AtomicU32::new(0),
            failed: Mutex::default(),
            threads: tokio::sync::Mutex::default(),
            artifacts: Mutex::default(),
        });
        let scope = Scope { inputs: Arc::new(values.clone()), state: Arc::new(Mutex::new(shared)), steps: Arc::new(StepsView::default()), names: Map::new() };
        let began = Instant::now();
        run.emit(WorkflowEvent::new("workflow_start", "", "", &spec.name).with("run_id", json!(run.id)).with("inputs", Value::Object(values.clone())));

        let outcome: FlowResult<Value> = async {
            let data: Map<String, Value> =
                [("workflow".to_string(), json!(spec.name)), ("inputs".to_string(), Value::Object(values.clone()))].into_iter().collect();
            let start = harness.hooks.emit("workflow_start", &spec.name, &run.id, 0, data).await;
            if start.blocked {
                return Err(Flow::Error(werr(format!("{} may not run: {}", spec.name, start.reason))));
            }
            let last = within(self.body(&self.inner.root, scope.clone(), run.clone()), spec.timeout, &spec.name).await?;
            let mut ctx = scope.context();
            ctx.insert("previous".into(), last.clone());
            Ok(match &spec.output {
                Some(o) => render(o, &ctx)?,
                None => last,
            })
        }
        .await;
        match outcome {
            Ok(v) | Err(Flow::Return(v)) => result.output = v,
            Err(Flow::Error(Error::StopRequested(m))) => {
                result.status = WorkflowStatus::Stopped;
                result.error = Some(m);
            }
            Err(Flow::Failed(m)) => {
                result.status = WorkflowStatus::Failed;
                result.error = Some(m);
                result.failed_step = lock(&run.failed).clone();
            }
            Err(Flow::Error(e)) => {
                result.status = WorkflowStatus::Failed;
                result.error = Some(e.to_string());
                result.failed_step = lock(&run.failed).clone();
            }
        }
        result.state = lock(&scope.state).clone();
        result.steps = lock(&run.results).clone();
        result.usage = lock(&run.usage).clone();
        result.cost_usd = round8(*lock(&run.cost));
        result.steps_run = run.count.load(Ordering::SeqCst);
        result.artifacts = lock(&run.artifacts).clone();
        result.duration_ms = began.elapsed().as_secs_f64() * 1000.0;
        let data: Map<String, Value> =
            [("workflow".to_string(), json!(spec.name)), ("result".to_string(), serde_json::to_value(&result).unwrap_or_default())].into_iter().collect();
        harness.hooks.emit("workflow_end", &spec.name, &run.id, 0, data).await;
        let status = serde_json::to_value(result.status).unwrap_or_default();
        run.emit(WorkflowEvent::new("workflow_end", "", "", status.as_str().unwrap_or("")).with("result", serde_json::to_value(&result).unwrap_or_default()));
        Ok(result)
    }

    /// The run as it happens: a `step_start` and `step_end` for every step,
    /// ending with `workflow_end`, whose `data["result"]` is the result. Drop
    /// the stream and the run is cancelled.
    pub fn stream(&self, inputs: Value) -> BoxStream<'static, Result<WorkflowEvent>> {
        let (tx, mut rx) = futures::channel::mpsc::unbounded::<WorkflowEvent>();
        let workflow = self.clone();
        let emit: Emit = Arc::new(move |e| {
            let _ = tx.unbounded_send(e);
        });
        Box::pin(async_stream::try_stream! {
            let mut work = Box::pin(workflow.run_with(inputs, Map::new(), Some(emit), None));
            let finished = loop {
                tokio::select! {
                    Some(e) = rx.next() => yield e,
                    done = &mut work => break done,
                }
            };
            while let Ok(e) = rx.try_recv() {
                yield e;
            }
            finished?;
        })
    }

    /// Expose the workflow as a tool an agent can call.
    pub fn as_tool(&self, name: Option<&str>, description: Option<&str>) -> Tool {
        let spec = &self.inner.spec;
        let mut specs = spec.input_specs();
        if specs.is_empty() {
            let mut m = Map::new();
            m.insert("required".into(), json!(true));
            specs.insert("input".into(), m);
        }
        let mut props = Map::new();
        let mut required = Vec::new();
        for (k, s) in &specs {
            let mut p = Map::new();
            if let Some(t) = s.get("type").and_then(Value::as_str).filter(|t| ["string", "number", "integer", "boolean", "array", "object"].contains(t)) {
                p.insert("type".into(), json!(t));
            }
            p.insert("description".into(), json!(s.get("description").and_then(Value::as_str).unwrap_or(k)));
            props.insert(k.clone(), Value::Object(p));
            if s.get("required").is_some_and(truthy) {
                required.push(k.clone());
            }
        }
        let description = description
            .map(str::to_string)
            .unwrap_or_else(|| if spec.description.is_empty() { format!("Run the {} workflow.", spec.name) } else { spec.description.clone() });
        let workflow = self.clone();
        Tool::raw(name.unwrap_or(&spec.name), description, json!({"type": "object", "properties": props, "required": required}), move |args, _ctx| {
            let workflow = workflow.clone();
            async move {
                let result = workflow.run(Value::Object(args)).await?;
                Ok(json!(if result.ok() {
                    text(&result.output)
                } else {
                    let status = serde_json::to_value(result.status).unwrap_or_default();
                    format!("[{}] {}", status.as_str().unwrap_or(""), result.error.unwrap_or_default())
                }))
            }
        })
        .tags(["workflow"])
    }

    // ---- the engine ----------------------------------------------------------
    fn body<'a>(&'a self, step: &'a Step, scope: Scope, run: Arc<Run>) -> BoxFuture<'a, FlowResult<Value>> {
        async move {
            if step.graph.is_some() {
                return self.graph(step, scope, run).await;
            }
            self.sequence(step.steps.as_deref().unwrap_or(&[]), scope, run).await
        }
        .boxed()
    }

    /// One after another. Each sees the one before as `previous`.
    fn sequence<'a>(&'a self, steps: &'a [Step], mut scope: Scope, run: Arc<Run>) -> BoxFuture<'a, FlowResult<Value>> {
        async move {
            let mut last = scope.names.get("previous").cloned().unwrap_or(Value::Null);
            for step in steps {
                let outcome = self.step(step, scope.clone(), run.clone()).await?;
                if outcome.status == StepStatus::Done {
                    last = outcome.output.clone();
                    scope.names.insert("previous".into(), last.clone());
                }
            }
            Ok(last)
        }
        .boxed()
    }

    /// Run branches at once, capped. The first failure stops the rest.
    async fn fan<T: Send>(&self, jobs: Vec<BoxFuture<'_, FlowResult<T>>>, limit: Option<usize>) -> FlowResult<Vec<T>> {
        let gate = Semaphore::new(limit.unwrap_or(self.inner.spec.concurrency).max(1));
        let gated = jobs.into_iter().map(|job| {
            let gate = &gate;
            async move {
                let _permit = gate.acquire().await.map_err(|_| Flow::Error(werr("cancelled")))?;
                job.await
            }
        });
        futures::future::try_join_all(gated).await
    }

    /// One step, with everything any step may carry around it.
    fn step<'a>(&'a self, step: &'a Step, scope: Scope, run: Arc<Run>) -> BoxFuture<'a, FlowResult<StepResult>> {
        async move {
            let harness = &self.inner.harness;
            let name = &self.inner.spec.name;
            let kind = step.kind();
            let mut outcome = StepResult { id: step.id.clone(), kind: kind.into(), attempts: 1, ..Default::default() };
            let keep = |o: &StepResult| {
                lock(&run.results).insert(step.id.clone(), o.clone());
                scope.steps.set(&step.id, o);
            };
            if let Some(w) = &step.when
                && !condition(w, &scope.context())?
            {
                outcome.status = StepStatus::Skipped;
                run.emit(WorkflowEvent::new("step_skipped", &step.id, kind, w));
                keep(&outcome);
                return Ok(outcome);
            }
            let count = run.count.fetch_add(1, Ordering::SeqCst) + 1;
            if count > self.inner.spec.max_steps {
                run.mark_failed(&step.id);
                return Err(Flow::Error(werr(format!(
                    "{name} ran {} steps without finishing — a loop that does not end? Raise `max_steps` if it is meant to",
                    self.inner.spec.max_steps
                ))));
            }
            run.emit(WorkflowEvent::new("step_start", &step.id, kind, &step.description));
            let began = Instant::now();
            let retry = step.retry.clone().unwrap_or(Retry { max: 0, delay: 0.0, backoff: 2.0 });
            let mut problem = String::new();
            let mut nested = false;
            for attempt in 1..=retry.max + 1 {
                outcome.attempts = attempt;
                let data: Map<String, Value> = [("workflow", json!(name)), ("step_id", json!(step.id)), ("kind", json!(kind)), ("attempt", json!(attempt))]
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect();
                let asked = harness.hooks.emit("workflow_step", name, &run.id, 0, data).await;
                let attempt_result: FlowResult<()> = if asked.blocked {
                    Err(Flow::Error(werr(format!("not permitted: {}", asked.reason))))
                } else {
                    within(self.execute(step, scope.clone(), run.clone(), &mut outcome), step.timeout, &format!("step '{}'", step.id)).await
                };
                match attempt_result {
                    Ok(()) => {
                        problem.clear();
                        break;
                    }
                    Err(Flow::Return(v)) => return Err(Flow::Return(v)),
                    Err(Flow::Error(e @ Error::StopRequested(_))) => return Err(Flow::Error(e)),
                    Err(failure) => {
                        run.mark_failed(&step.id);
                        // A step inside this one failed: its message says it all.
                        (problem, nested) = match failure {
                            Flow::Failed(m) => (m, true),
                            Flow::Error(e) => (e.to_string(), false),
                            Flow::Return(_) => unreachable!("handled above"),
                        };
                        if attempt <= retry.max {
                            run.emit(WorkflowEvent::new("step_retry", &step.id, kind, &problem).with("attempt", json!(attempt)));
                            tokio::time::sleep(Duration::from_secs_f64((retry.delay * retry.backoff.powi(attempt as i32 - 1)).max(0.0))).await;
                            continue;
                        }
                        break;
                    }
                }
            }
            outcome.duration_ms = began.elapsed().as_secs_f64() * 1000.0;
            if !problem.is_empty() {
                outcome.status = StepStatus::Failed;
                outcome.error = Some(problem.clone());
                keep(&outcome);
                let continues = step.on_error == "continue";
                run.emit(WorkflowEvent::new("step_failed", &step.id, kind, &problem).with("continues", json!(continues)));
                if continues {
                    lock(&run.failed).clear();
                    return Ok(outcome);
                }
                return Err(Flow::Failed(if nested { problem } else { format!("step '{}' failed: {problem}", step.id) }));
            }
            if outcome.attempts > 1 {
                lock(&run.failed).clear(); // it got there in the end
            }
            if let Some(key) = &step.save {
                let kept = if outcome.data.is_null() { outcome.output.clone() } else { outcome.data.clone() };
                lock(&scope.state).insert(key.clone(), kept);
            }
            let said: String = text(&outcome.output).chars().take(400).collect();
            run.emit(
                WorkflowEvent::new("step_end", &step.id, kind, &said)
                    .with("duration_ms", json!((outcome.duration_ms * 10.0).round() / 10.0))
                    .with("cost_usd", json!(outcome.cost_usd))
                    .with("attempts", json!(outcome.attempts)),
            );
            keep(&outcome);
            Ok(outcome)
        }
        .boxed()
    }

    /// Do what the step is, and write what it came to on `outcome`.
    async fn execute(&self, step: &Step, scope: Scope, run: Arc<Run>, outcome: &mut StepResult) -> FlowResult<()> {
        let ctx = scope.context();
        match step.kind() {
            "agent" => self.agent(step, &ctx, &run, outcome).await?,
            "tool" => {
                let args = match render(&Value::Object(step.args.clone()), &ctx)? {
                    Value::Object(m) => m,
                    _ => Map::new(),
                };
                let called = self.runner()?.call_tool(step.tool.as_deref().unwrap_or(""), args).await;
                if called.is_error {
                    return Err(Flow::Error(werr(called.content)));
                }
                outcome.output = called.value.clone().filter(|v| !v.is_null()).unwrap_or(Value::String(called.content.clone()));
                if !outcome.output.is_string() {
                    outcome.data = outcome.output.clone();
                }
            }
            "set" => {
                let values = render(&Value::Object(step.assign.clone().unwrap_or_default()), &ctx)?;
                if let Value::Object(m) = &values {
                    lock(&scope.state).extend(m.clone());
                }
                outcome.output = values;
            }
            "wait" => {
                let seconds = render(step.wait.as_ref().unwrap_or(&Value::Null), &ctx)?;
                let secs = match &seconds {
                    Value::Number(n) => n.as_f64(),
                    Value::String(s) => s.trim().parse::<f64>().ok(),
                    _ => None,
                }
                .ok_or_else(|| werr(format!("cannot wait {seconds} seconds")))?;
                tokio::time::sleep(Duration::from_secs_f64(secs.max(0.0))).await;
                outcome.output = scope.names.get("previous").cloned().unwrap_or(Value::Null);
            }
            "fail" => {
                let said = text(&render(&json!(step.fail.clone().unwrap_or_default()), &ctx)?);
                return Err(Flow::Error(werr(if said.is_empty() { "failed".to_string() } else { said })));
            }
            "return" => return Err(Flow::Return(render(step.return_.as_ref().unwrap_or(&Value::Null), &ctx)?)),
            "steps" => outcome.output = self.sequence(step.steps.as_deref().unwrap_or(&[]), scope.child(false, &[]), run).await?,
            "parallel" => {
                let branches = step.parallel.as_deref().unwrap_or(&[]);
                let jobs: Vec<BoxFuture<'_, FlowResult<StepResult>>> = branches.iter().map(|b| self.step(b, scope.child(false, &[]), run.clone())).collect();
                let results = self.fan(jobs, step.concurrency).await?;
                outcome.output = Value::Object(results.into_iter().filter(|r| r.status == StepStatus::Done).map(|r| (r.id, r.output)).collect());
            }
            "foreach" => {
                let items = match render(step.foreach.as_ref().unwrap_or(&Value::Null), &ctx)? {
                    Value::Object(m) => m.into_iter().map(|(k, v)| json!({"key": k, "value": v})).collect(),
                    Value::Number(n) if n.is_i64() => (0..n.as_i64().unwrap_or(0).max(0)).map(|i| json!(i)).collect(),
                    Value::Null => vec![],
                    Value::Array(a) => a,
                    other => {
                        let got = if other.is_string() { "str" } else { "a value that is not a list" };
                        return Err(Flow::Error(werr(format!("foreach needs a list — got {got}"))));
                    }
                };
                let body = step.steps.as_deref().unwrap_or(&[]);
                outcome.iterations = items.len() as u32;
                let jobs: Vec<BoxFuture<'_, FlowResult<Value>>> = items
                    .into_iter()
                    .enumerate()
                    .map(|(i, item)| {
                        let inner = scope.child(true, &[(step.as_.as_str(), item), ("index", json!(i))]);
                        self.sequence(body, inner, run.clone())
                    })
                    .collect();
                outcome.output = Value::Array(self.fan(jobs, Some(step.concurrency.unwrap_or(1))).await?);
            }
            "loop" => {
                let spec = step.loop_.clone().unwrap_or_default();
                let mut inner = scope.child(false, &[]);
                let mut last = Value::Null;
                outcome.exhausted = true;
                for index in 0..spec.max {
                    inner.names.insert("loop".into(), json!({"index": index, "iteration": index + 1, "max": spec.max}));
                    if let Some(w) = &spec.while_
                        && !condition(w, &inner.context())?
                    {
                        outcome.exhausted = false;
                        break;
                    }
                    last = self.sequence(step.steps.as_deref().unwrap_or(&[]), inner.clone(), run.clone()).await?;
                    inner.names.insert("previous".into(), last.clone());
                    outcome.iterations = index + 1;
                    if let Some(u) = &spec.until
                        && condition(u, &inner.context())?
                    {
                        outcome.exhausted = false;
                        break;
                    }
                }
                if spec.until.is_none() && spec.while_.is_none() {
                    outcome.exhausted = false; // a counted loop simply finished
                }
                outcome.output = last;
            }
            "if" => {
                let taken = condition(step.if_.as_deref().unwrap_or(""), &ctx)?;
                outcome.branch = if taken { "then" } else { "else" }.into();
                let body = if taken { &step.then } else { &step.else_ };
                outcome.output = match body {
                    Some(b) if !b.is_empty() => self.sequence(b, scope.child(false, &[]), run).await?,
                    _ => Value::Null,
                };
            }
            "switch" => {
                outcome.output = Value::Null;
                for (i, case) in step.switch.iter().flatten().enumerate() {
                    let hit = match &case.when {
                        None => true,
                        Some(w) => condition(w, &ctx)?,
                    };
                    if hit {
                        outcome.branch = if case.when.is_some() { format!("case {}", i + 1) } else { "default".into() };
                        outcome.output = self.sequence(&case.steps, scope.child(false, &[]), run).await?;
                        break;
                    }
                }
            }
            "graph" => outcome.output = self.graph(step, scope, run).await?,
            other => return Err(Flow::Error(werr(format!("step '{}' is '{other}', which this engine cannot run", step.id)))),
        }
        Ok(())
    }

    async fn agent(&self, step: &Step, ctx: &Ctx, run: &Arc<Run>, outcome: &mut StepResult) -> FlowResult<()> {
        let name = match &step.agent {
            Some(Value::String(n)) => n.clone(),
            _ => step.id.clone(),
        };
        let mut agent = self.inner.agents.get(&name).cloned().ok_or_else(|| werr(format!("no agent named '{name}'")))?;
        if let Some(v) = &step.version {
            agent = agent.use_version(v)?;
        }
        let asked = if let Some(input) = &step.input {
            render(input, ctx)?
        } else if ctx.get("previous").is_some_and(|p| !p.is_null()) {
            ctx["previous"].clone() // the step before feeds this one
        } else {
            match ctx.get("inputs") {
                Some(Value::Object(m)) if m.len() == 1 => m.values().next().cloned().unwrap_or(Value::Null),
                Some(other) => other.clone(),
                None => Value::Null,
            }
        };
        let mut options = RunOptions::default();
        if step.thread {
            // One conversation for this agent through the whole run.
            let held = run.threads.lock().await.entry(name.clone()).or_insert_with(|| Session::new(agent.name())).clone();
            options.session = Some(SessionRef::Session(Box::new(held)));
        } else {
            options.messages = Some(vec![]); // each step is its own task
        }
        if let Some(files) = &step.attachments {
            let files = render(files, ctx)?;
            let list = match files {
                Value::Array(a) => a,
                other => vec![other],
            };
            options.attachments = list.iter().filter_map(Value::as_str).map(|p| attach(p, None)).collect();
        }
        if let Some(g) = &run.guard {
            options.guard = Some(g.child(agent.budget().cloned().unwrap_or_default()));
        }
        let result = agent.run_with(text(&asked), options).await?;
        if step.thread {
            let latest = match agent.harness().sessions.load(&result.session_id).await {
                Ok(s) => s,
                Err(_) => Session { id: result.session_id.clone(), messages: result.messages.clone(), ..Session::new(agent.name()) },
            };
            run.threads.lock().await.insert(name.clone(), latest);
        }
        {
            let mut usage = lock(&run.usage);
            *usage += &result.usage;
            for child in &result.children {
                *usage += &child.usage;
            }
        }
        *lock(&run.cost) += result.cost_usd();
        {
            let mut arts = lock(&run.artifacts);
            for a in &result.artifacts {
                if !arts.contains(a) {
                    arts.push(a.clone());
                }
            }
        }
        outcome.cost_usd = round8(outcome.cost_usd + result.cost_usd());
        outcome.agent = result.agent.clone();
        let who = if result.agent.is_empty() { name.clone() } else { result.agent.clone() };
        if let Some(e) = &result.error {
            return Err(Flow::Error(werr(format!("{who}: {e}"))));
        }
        if let Some(b) = &result.budget_exceeded {
            return Err(Flow::Error(werr(format!("{who} stopped at its budget ({b})"))));
        }
        outcome.output = Value::String(result.output.clone());
        if !result.data.is_null() {
            outcome.data = result.data.clone();
        } else if step.parse.as_deref() == Some("json") {
            outcome.data = extract_json(&result.output).ok_or_else(|| {
                let head: String = result.output.chars().take(200).collect();
                werr(format!("{name} was to answer in JSON and did not: {head:?}"))
            })?;
        }
        Ok(())
    }

    /// Nodes that say what they need. Each starts the moment its needs are
    /// met, so everything that can run at once does.
    async fn graph(&self, step: &Step, scope: Scope, run: Arc<Run>) -> FlowResult<Value> {
        let nodes: IndexMap<String, &Step> = step.graph.iter().flatten().map(|n| (n.id.clone(), n)).collect();
        let mut waiting: IndexMap<String, &Step> = nodes.clone();
        let mut settled: IndexMap<String, StepResult> = IndexMap::new();
        let gate = Semaphore::new(step.concurrency.unwrap_or(self.inner.spec.concurrency).max(1));
        let mut running = FuturesUnordered::new();
        loop {
            let ready: Vec<String> = waiting.iter().filter(|(_, n)| n.needs.iter().all(|need| settled.contains_key(need))).map(|(k, _)| k.clone()).collect();
            for name in ready {
                let node = waiting.shift_remove(&name).expect("ready node is waiting");
                let met: Vec<StepResult> = node.needs.iter().map(|n| settled[n].clone()).collect();
                let arrived: Vec<&StepResult> = met.iter().filter(|r| r.status == StepStatus::Done).collect();
                let enough = if node.join == "any" { !arrived.is_empty() } else { arrived.len() == met.len() };
                if !met.is_empty() && !enough {
                    // What it was waiting for never came: it does not run either.
                    let skipped = StepResult { id: node.id.clone(), kind: node.kind().into(), status: StepStatus::Skipped, attempts: 1, ..Default::default() };
                    lock(&run.results).insert(node.id.clone(), skipped.clone());
                    scope.steps.set(&node.id, &skipped);
                    let missing: Vec<&str> = met.iter().filter(|r| r.status != StepStatus::Done).map(|r| r.id.as_str()).collect();
                    run.emit(WorkflowEvent::new("step_skipped", &node.id, node.kind(), &format!("needs {}", missing.join(", "))));
                    settled.insert(name, skipped);
                    continue;
                }
                let previous = if node.needs.len() == 1 && !arrived.is_empty() {
                    arrived[0].output.clone()
                } else if !arrived.is_empty() {
                    Value::Object(arrived.iter().map(|r| (r.id.clone(), r.output.clone())).collect())
                } else {
                    scope.names.get("previous").cloned().unwrap_or(Value::Null)
                };
                let child = scope.child(false, &[("previous", previous)]);
                let run = run.clone();
                let gate = &gate;
                running.push(async move {
                    let _permit = gate.acquire().await.map_err(|_| Flow::Error(werr("cancelled")))?;
                    let r = self.step(node, child, run).await?;
                    Ok::<_, Flow>((name, r))
                });
            }
            if running.is_empty() {
                if waiting.is_empty() {
                    break;
                }
                // Only nodes waiting on skipped ones remain: settle them on the next pass.
                continue;
            }
            // The first failure drops `running`, which cancels the rest.
            let (name, r) = running.next().await.expect("not empty")?;
            settled.insert(name, r);
        }
        Ok(Value::Object(settled.into_iter().filter(|(_, r)| r.status == StepStatus::Done).map(|(k, r)| (k, r.output)).collect()))
    }
}

async fn within<T>(work: impl std::future::Future<Output = FlowResult<T>>, seconds: Option<f64>, what: &str) -> FlowResult<T> {
    match seconds {
        None => work.await,
        Some(s) => {
            let shown = if s.fract() == 0.0 { format!("{}", s as i64) } else { format!("{s}") };
            tokio::time::timeout(Duration::from_secs_f64(s), work).await.unwrap_or_else(|_| Err(Flow::Error(werr(format!("{what} timed out after {shown}s")))))
        }
    }
}

// ----------------------------------------------------------------------
// checking a declaration before anything runs
// ----------------------------------------------------------------------
/// Give every step a name, and refuse two with the same one.
fn number(step: &mut Step, seen: &mut HashSet<String>, counter: &mut u32) -> Result<()> {
    for body in step.children_mut() {
        for child in body.iter_mut() {
            if child.id.is_empty() {
                *counter += 1;
                let kind = child.kind();
                child.id = format!("{}_{counter}", if kind.is_empty() { "step" } else { kind });
            } else if !ID.is_match(&child.id) {
                return Err(Error::config(format!(
                    "step id '{}' must be a plain name — letters, digits and underscores — so templates can say steps.name",
                    child.id
                )));
            }
            if !seen.insert(child.id.clone()) {
                return Err(Error::config(format!("two steps are called '{}'", child.id)));
            }
            number(child, seen, counter)?;
        }
    }
    Ok(())
}

fn collect(step: &Step, ids: &mut IndexMap<String, Step>) {
    for (_, body) in step.children() {
        for child in body {
            ids.insert(child.id.clone(), child.clone());
            collect(child, ids);
        }
    }
}

/// Checks one expression (source, names in scope, where it is).
type ExprCheck<'a> = dyn Fn(&str, &HashSet<String>, &str) -> Result<()> + 'a;
/// Checks every template inside some values.
type TemplateCheck<'a> = dyn Fn(&[Option<Value>], &HashSet<String>, &str) -> Result<()> + 'a;

/// Everything that can be known to be wrong without running it.
fn check(root: &Step, ids: &IndexMap<String, Step>, spec: &WorkflowSpec) -> Result<()> {
    let inputs: HashSet<String> = spec.input_specs().keys().cloned().collect();
    let base: HashSet<String> = ["inputs", "state", "steps", "previous"].into_iter().map(String::from).collect();
    let expression = |source: &str, names: &HashSet<String>, where_: &str| -> Result<()> {
        let source = bare(source);
        let e = compile(&source)?;
        let (used, fields) = e.references();
        for n in used {
            if !names.contains(&n) && !CONSTANTS.contains(&n.as_str()) {
                let mut k: Vec<&String> = names.iter().collect();
                k.sort();
                let k: Vec<&str> = k.iter().map(|s| s.as_str()).collect();
                return Err(Error::config(format!("step '{where_}': '{source}' uses '{n}', which is not known here. Known: {}", k.join(", "))));
            }
        }
        for (target, key) in fields {
            if target == "steps" && !ids.contains_key(&key) {
                return Err(Error::config(format!(
                    "step '{where_}': '{source}' refers to steps.{key}, and no step is called that. Steps: {}",
                    known(ids.keys().cloned())
                )));
            }
            if target == "inputs" && !inputs.is_empty() && !inputs.contains(&key) {
                return Err(Error::config(format!(
                    "step '{where_}': '{source}' refers to inputs.{key}, which the workflow does not declare. Inputs: {}",
                    known(inputs.iter().cloned())
                )));
            }
        }
        Ok(())
    };
    let templates = |values: &[Option<Value>], names: &HashSet<String>, where_: &str| -> Result<()> {
        let mut found = Vec::new();
        for v in values.iter().flatten() {
            expressions(v, &mut found);
        }
        found.iter().try_for_each(|s| expression(s, names, where_))
    };
    fn walk(step: &Step, names: &HashSet<String>, node: bool, expression: &ExprCheck, templates: &TemplateCheck) -> Result<()> {
        let kinds = step.kinds();
        let id = &step.id;
        if kinds == [""] {
            return Err(Error::config(format!("step '{id}' does nothing — give it one of: {}, or `steps`", KINDS.join(", "))));
        }
        if kinds.len() > 1 {
            return Err(Error::config(format!("step '{id}' is {} at once — a step is one thing; nest the other inside it", kinds.join(" and "))));
        }
        let kind = kinds[0];
        if !step.needs.is_empty() && !node {
            return Err(Error::config(format!("step '{id}' has `needs`, which only a node of a `graph` can have")));
        }
        if matches!(kind, "foreach" | "loop") && step.steps.as_ref().is_none_or(Vec::is_empty) {
            return Err(Error::config(format!("step '{id}': a {kind} needs `steps` to run")));
        }
        if !matches!(kind, "foreach" | "loop" | "steps") && step.steps.is_some() {
            return Err(Error::config(format!("step '{id}': `steps` belongs to a foreach, a loop, or a sequence — not to {kind}")));
        }
        if kind != "if" && (step.then.is_some() || step.else_.is_some()) {
            return Err(Error::config(format!("step '{id}': `then` and `else` belong to an `if`")));
        }
        if kind == "if" && step.then.as_ref().is_none_or(Vec::is_empty) && step.else_.as_ref().is_none_or(Vec::is_empty) {
            return Err(Error::config(format!("step '{id}': an `if` needs a `then`")));
        }
        if kind == "switch" {
            let cases = step.switch.as_deref().unwrap_or(&[]);
            if cases.len() > 1 && cases[..cases.len() - 1].iter().any(|c| c.when.is_none()) {
                return Err(Error::config(format!("step '{id}': a case with no `when` is the default, and must come last")));
            }
        }
        if step.save.as_ref().is_some_and(|s| !ID.is_match(s)) {
            return Err(Error::config(format!("step '{id}': `save` names a key in the state — a plain name")));
        }
        if kind == "foreach" && (!ID.is_match(&step.as_) || ["inputs", "state", "steps", "previous"].contains(&step.as_.as_str())) {
            return Err(Error::config(format!("step '{id}': `as: {}` is not a name an item can have", step.as_)));
        }
        if !matches!(step.on_error.as_str(), "fail" | "continue") {
            return Err(Error::config(format!("step '{id}': on_error must be fail or continue — got '{}'", step.on_error)));
        }
        if !matches!(step.join.as_str(), "all" | "any") {
            return Err(Error::config(format!("step '{id}': join must be all or any — got '{}'", step.join)));
        }
        if step.parse.as_ref().is_some_and(|p| p != "json") {
            return Err(Error::config(format!("step '{id}': parse may only be json")));
        }
        if step.retry.as_ref().is_some_and(|r| r.max > 20 || r.delay < 0.0 || r.backoff < 1.0) {
            return Err(Error::config(format!("step '{id}': retry max is 0–20, delay at least 0, backoff at least 1")));
        }
        if step.timeout.is_some_and(|t| t <= 0.0) || step.concurrency == Some(0) {
            return Err(Error::config(format!("step '{id}': timeout and concurrency must be greater than 0")));
        }
        if step.loop_.as_ref().is_some_and(|l| l.max < 1 || l.max > 10_000) {
            return Err(Error::config(format!("step '{id}': loop max is 1–10000")));
        }
        for source in [&step.when, &step.if_].into_iter().flatten() {
            expression(source, names, id)?;
        }
        let assign = step.assign.clone().map(Value::Object);
        let fail = step.fail.clone().map(Value::String);
        templates(
            &[
                step.input.clone(),
                Some(Value::Object(step.args.clone())),
                assign,
                step.wait.clone(),
                fail,
                step.return_.clone(),
                step.attachments.clone(),
                step.foreach.clone(),
            ],
            names,
            id,
        )?;
        for case in step.switch.iter().flatten() {
            if let Some(w) = &case.when {
                expression(w, names, id)?;
            }
        }
        let mut inner = names.clone();
        if kind == "foreach" {
            inner.insert(step.as_.clone());
            inner.insert("index".into());
        }
        if kind == "loop" {
            inner.insert("loop".into());
            if let Some(l) = &step.loop_ {
                for s in [&l.until, &l.while_].into_iter().flatten() {
                    expression(s, &inner, id)?;
                }
            }
        }
        if kind == "graph" {
            acyclic(step)?;
        }
        for (_, body) in step.children() {
            for child in body {
                walk(child, &inner, kind == "graph", expression, templates)?;
            }
        }
        Ok(())
    }
    if spec.max_steps < 1 || spec.concurrency < 1 || spec.timeout.is_some_and(|t| t <= 0.0) {
        return Err(Error::config("the workflow is not valid — max_steps and concurrency must be at least 1, and timeout greater than 0"));
    }
    for (_, body) in root.children() {
        for child in body {
            walk(child, &base, root.graph.is_some(), &expression, &templates)?;
        }
    }
    if root.graph.is_some() {
        acyclic(root)?;
    }
    templates(std::slice::from_ref(&spec.output), &base, "output")
}

/// A graph's `needs` must name its own nodes and never come back round.
fn acyclic(step: &Step) -> Result<()> {
    let nodes: IndexMap<&str, &Step> = step.graph.iter().flatten().map(|n| (n.id.as_str(), n)).collect();
    for node in nodes.values() {
        for need in &node.needs {
            if !nodes.contains_key(need.as_str()) {
                return Err(Error::config(format!(
                    "step '{}' needs '{need}', which is not a node of the same graph. Nodes: {}",
                    node.id,
                    known(nodes.keys().map(|s| s.to_string()))
                )));
            }
        }
    }
    let mut done: HashSet<&str> = HashSet::new();
    while done.len() < nodes.len() {
        let ready: Vec<&str> =
            nodes.iter().filter(|(n, node)| !done.contains(*n) && node.needs.iter().all(|need| done.contains(need.as_str()))).map(|(n, _)| *n).collect();
        if ready.is_empty() {
            let stuck = known(nodes.keys().filter(|n| !done.contains(*n)).map(|s| s.to_string()));
            return Err(Error::config(format!(
                "these nodes wait on each other and can never start: {stuck}. A graph runs forwards — use a `loop` to go round again"
            )));
        }
        done.extend(ready);
    }
    Ok(())
}

impl Blueprint {
    /// Build one declared workflow, and the agents its steps name.
    pub fn workflow(&self, name: &str, tools: &[Tool], harness: Option<Harness>, agents: Vec<Agent>) -> Result<Workflow> {
        let Some(spec) = self.workflows.get(name) else {
            return Err(Error::config(format!("no workflow '{name}' in this blueprint; declared: {}", known(self.workflows.keys().cloned()))));
        };
        let mut data = spec.clone();
        if let Value::Object(m) = &mut data {
            m.insert("name".into(), json!(name));
        }
        let mut b = Workflow::from_value(data)?.agents(agents).tools(tools.iter().cloned()).blueprint(self.clone());
        if let Some(h) = harness {
            b = b.harness(h);
        }
        b.build()
    }
}
