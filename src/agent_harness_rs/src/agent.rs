//! The agent, and the loop that drives it: think → act → observe → repeat.
//!
//! ```no_run
//! use agent_harness::{Agent, Tool, ToolContext};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! /// Look up the status of a customer order.
//! #[derive(Deserialize, JsonSchema)]
//! struct OrderStatus {
//!     /// the order number, digits only.
//!     order_id: String,
//! }
//!
//! # async fn demo() -> agent_harness::Result<()> {
//! let order_status = Tool::new("order_status", |a: OrderStatus, _: ToolContext| async move {
//!     Ok(format!("order {} shipped", a.order_id))
//! });
//! let agent = Agent::builder("support")
//!     .instructions("Answer customer questions about orders. Look the order up before answering.")
//!     .tool(order_status)
//!     .build()?;
//! let result = agent.run("Where is order 4182?").await?;
//! println!("{} {} {}", result.output, result.cost_usd(), result.steps);
//! # Ok(()) }
//! ```
//!
//! Ported from Python: the model ↔ tool loop, streaming, parallel tool calls
//! under the policy gate and hooks, budgets (stop or raise), output contracts
//! with retries, sessions with conflict merge or fork, compaction,
//! attachments, model egress checks and fallback, sub-agents (`delegate`),
//! run-time specialists (`spawn_agent`), handoffs, versions and `resume`.
//! Memory, modes, guardrails, approvals and tracing arrive with later
//! milestones.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

use futures::stream::{BoxStream, StreamExt};
use indexmap::IndexMap;
use regex::Regex;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::attachments::prepare_attachments;
use crate::context::{ContextAssembler, ContextCompactor, Summarizer, close_open_tool_calls};
use crate::errors::{Error, ProviderError, ProviderErrorKind, Result};
use crate::guardrails::{AgentGuardrails, CompletionContext, Guardrails, OnViolation};
use crate::handoff::{Handoff, Relay, active_holder, not_an_agent};
use crate::harness::Harness;
use crate::llm_providers::base::{CompletionRequest, Provider, model_info, response_from_event};
use crate::llm_providers::catalog::{get_provider, resolve_provider};
use crate::llm_providers::parameters::{Effort, validate_parameters};
use crate::memory::{MemoryManager, MemoryOptions, Trace};
use crate::modes::{Mode, Notebook, resolve_mode};
use crate::prompts::Prompt;
use crate::runtime::workspace::{Snapshot, Workspace, workspace_tools};
use crate::runtime::{Approval, ApprovalCall, Budget, BudgetGuard, Checkpoint, HookEngine, OnExceed, PolicyGate, Session, Span};
use crate::skills::SkillRegistry;
use crate::subagents::{SubAgentFactory, SubAgentSpec};
use crate::tools::{Permission, RunScope, Tool, ToolContext, ToolRegistry};
use crate::types::{
    ContentBlock, HandoffRecord, Message, ModelResponse, RunResult, StopReason, StreamEvent, StreamEventType, ToolCall, ToolOutcome, ToolUseBlock, new_id,
};
use crate::versioning::AgentVersion;

/// How many images returned by tools a conversation keeps.
pub const TOOL_IMAGES_KEPT: usize = 3;
/// The most run-time agents one agent may spin up in a run.
pub const MAX_RUNTIME_AGENTS: u32 = 100;
/// Tags that mark a tool as orchestration-only: never inherited by a sub-agent.
const NOT_INHERITED: [&str; 3] = ["delegation", "memory", crate::modes::LEAD];

const CONTRACT: &str = "Your final message must be a single JSON object matching this schema, with no prose and no code fence around it:\n";

fn identity(name: &str, description: &str) -> String {
    let purpose = if description.is_empty() { String::new() } else { format!(" {description}") };
    format!(
        "You are {name}, an AI agent.{purpose}\nWork to the point. Say what you did and what you found; do not narrate what you are about to do. If you cannot complete something, say so plainly and explain what is missing."
    )
}

/// Checks a parsed answer; `Err` says what is wrong with it.
type Validator = Arc<dyn Fn(&Value) -> std::result::Result<(), String> + Send + Sync>;

/// A declared output type: the schema the model is told, and the check its
/// answer must pass.
#[derive(Clone)]
pub struct OutputContract {
    pub schema: Value,
    validate: Validator,
}

impl OutputContract {
    /// The contract for `T`: its JSON Schema, and serde as the check.
    pub fn of<T: DeserializeOwned + JsonSchema + 'static>() -> Self {
        let mut schema = serde_json::to_value(schemars::schema_for!(T)).unwrap_or_default();
        if let Value::Object(m) = &mut schema {
            m.remove("$schema");
        }
        Self { schema, validate: Arc::new(|v| serde_json::from_value::<T>(v.clone()).map(|_| ()).map_err(|e| e.to_string())) }
    }

    /// A contract from a JSON Schema: the answer must be an object with the
    /// `required` fields, each of the declared primitive type.
    pub fn schema(schema: Value) -> Self {
        let rules = schema.clone();
        Self {
            schema,
            validate: Arc::new(move |v| {
                let Value::Object(obj) = v else { return Err("expected a JSON object".into()) };
                let mut problems = Vec::new();
                for r in rules.get("required").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
                    if !obj.contains_key(r) {
                        problems.push(format!("{r}: Field required"));
                    }
                }
                for (k, p) in rules.get("properties").and_then(Value::as_object).into_iter().flatten() {
                    let (Some(want), Some(got)) = (p.get("type").and_then(Value::as_str), obj.get(k)) else { continue };
                    let ok = match want {
                        "string" => got.is_string(),
                        "integer" => got.is_i64() || got.is_u64(),
                        "number" => got.is_number(),
                        "boolean" => got.is_boolean(),
                        "array" => got.is_array(),
                        "object" => got.is_object(),
                        _ => true,
                    } || got.is_null();
                    if !ok {
                        problems.push(format!("{k}: expected {want}"));
                    }
                }
                if problems.is_empty() { Ok(()) } else { Err(problems.join("; ")) }
            }),
        }
    }
}

/// Where a run's conversation lives.
#[derive(Debug, Clone)]
pub enum SessionRef {
    Id(String),
    Session(Box<Session>),
}

impl From<&str> for SessionRef {
    fn from(id: &str) -> Self {
        Self::Id(id.into())
    }
}

impl From<String> for SessionRef {
    fn from(id: String) -> Self {
        Self::Id(id)
    }
}

impl From<Session> for SessionRef {
    fn from(s: Session) -> Self {
        Self::Session(Box::new(s))
    }
}

/// Everything a single run can be told, beyond the task.
#[derive(Clone, Default)]
pub struct RunOptions {
    /// Continue this conversation.
    pub session: Option<SessionRef>,
    /// Run on this history instead — a one-off with a session of its own.
    pub messages: Option<Vec<Message>>,
    /// Files, images, recordings that come with the task.
    pub attachments: Vec<ContentBlock>,
    pub max_steps: Option<u32>,
    pub model: Option<String>,
    pub run_id: Option<String>,
    /// Spend against this guard instead of the agent's own.
    pub guard: Option<Arc<BudgetGuard>>,
    /// Work in this sandbox (an id a run reported as `sandbox_id`) rather than
    /// the one the conversation remembers.
    pub sandbox_id: Option<String>,
}

impl RunOptions {
    pub fn session(session: impl Into<SessionRef>) -> Self {
        Self { session: Some(session.into()), ..Default::default() }
    }
}

/// One accountable agent: a model, a prompt, its tools, its sub-agents.
/// Cheap to clone; clones are the same agent.
#[derive(Clone)]
pub struct Agent {
    inner: Arc<AgentInner>,
}

struct AgentInner {
    name: String,
    description: String,
    instructions: String,
    model: String,
    provider: Arc<dyn Provider>,
    explicit_provider: bool,
    /// Every generation parameter, ready to be copied into a request.
    template: CompletionRequest,
    tools: RwLock<ToolRegistry>,
    skills: Option<SkillRegistry>,
    harness: Harness,
    hooks: HookEngine,
    policy: PolicyGate,
    budget: Option<Budget>,
    max_steps: u32,
    output: Option<OutputContract>,
    parallel_tools: bool,
    contract_retries: u32,
    persist_session: bool,
    tool_images_kept: usize,
    mode: Option<Mode>,
    assembler: ContextAssembler,
    compactor: ContextCompactor,
    fallback_providers: Mutex<HashMap<String, Arc<dyn Provider>>>,
    // ---- composition ---------------------------------------------------
    subagents: RwLock<IndexMap<String, Agent>>,
    handoffs: RwLock<IndexMap<String, Handoff>>,
    max_handoffs: u32,
    runtime_agents: bool,
    max_runtime_agents: u32,
    runtime_agent_tools: Option<Vec<String>>,
    spawned: AtomicU32,
    total_spawned: AtomicU32,
    factory: OnceLock<SubAgentFactory>,
    // ---- versions and conversations -------------------------------------
    version: String,
    versions: IndexMap<String, AgentVersion>,
    base: Option<Box<AgentBuilder>>,
    siblings: Arc<Mutex<HashMap<String, Agent>>>,
    /// The conversation `resume()` follows.
    thread: Mutex<Option<Session>>,
    following: AtomicBool,
    // ---- memory -----------------------------------------------------------
    memory: Option<MemoryManager>,
    trace: Option<Trace>,
    guardrails: Option<AgentGuardrails>,
    workspace: Option<Arc<dyn Workspace>>,
    /// The record a run with no session keeps adding to: the memory session's.
    log: Mutex<Option<Session>>,
}

/// What an agent remembers with.
#[derive(Clone, Default)]
enum MemoryChoice {
    /// Its own manager over the harness's memory store (the default).
    #[default]
    On,
    Off,
    /// This manager — narrowed to the agent's trace, if it has one.
    Manager(MemoryManager),
    /// This manager exactly: a version of the same agent shares its memory.
    Shared(MemoryManager),
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Agent {} model={} tools={}>", self.inner.name, self.inner.model, self.tools().len())
    }
}

/// Builds an [`Agent`]. Every setting has a working default.
#[derive(Clone)]
pub struct AgentBuilder {
    name: String,
    description: String,
    instructions: String,
    model: Option<String>,
    tier: Option<String>,
    provider: Option<Arc<dyn Provider>>,
    provider_name: Option<String>,
    params: CompletionRequest,
    tools: Vec<Tool>,
    skills: Option<SkillRegistry>,
    harness: Option<Harness>,
    hooks: Option<HookEngine>,
    policy: Option<PolicyGate>,
    budget: Option<Budget>,
    max_steps: Option<u32>,
    output: Option<OutputContract>,
    parallel_tools: bool,
    contract_retries: u32,
    persist_session: bool,
    mode: Option<Mode>,
    mode_spec: Option<(String, Option<String>)>,
    max_context_tokens: Option<usize>,
    compact_at: Option<f64>,
    compact_keep_last: usize,
    compact_target: f64,
    compactor: Option<ContextCompactor>,
    subagents: Vec<Agent>,
    subagent_specs: Vec<SubAgentSpec>,
    handoffs: Vec<Handoff>,
    max_handoffs: u32,
    runtime_agents: Option<bool>,
    max_runtime_agents: Option<u32>,
    runtime_agent_tools: Option<Vec<String>>,
    version: String,
    versions: IndexMap<String, AgentVersion>,
    memory: MemoryChoice,
    trace: Option<Trace>,
    guardrails: Option<AgentGuardrails>,
    identity: Option<Value>,
    workspace: Option<WorkspaceChoice>,
    allow_shell: Option<bool>,
}

/// Where an agent works.
#[derive(Clone)]
enum WorkspaceChoice {
    /// A folder from the harness's broker (or whatever the broker hands out).
    Folder,
    /// A workspace built elsewhere: a folder, a container, a sandbox.
    Given(Arc<dyn Workspace>),
    /// A sandbox from its name, URL or mapping.
    Sandbox(Value),
}

/// Words that mean yes and no for a workspace given as text.
const YES: [&str; 8] = ["true", "yes", "on", "1", "enable", "enabled", "local", "isolated"];
const NO: [&str; 8] = ["false", "no", "off", "0", "disable", "disabled", "none", ""];

impl AgentBuilder {
    fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            instructions: String::new(),
            model: None,
            tier: None,
            provider: None,
            provider_name: None,
            params: CompletionRequest::new("", vec![]),
            tools: vec![],
            skills: None,
            harness: None,
            hooks: None,
            policy: None,
            budget: None,
            max_steps: None,
            output: None,
            parallel_tools: true,
            contract_retries: 2,
            persist_session: true,
            mode: None,
            mode_spec: None,
            max_context_tokens: None,
            compact_at: None,
            compact_keep_last: 8,
            compact_target: 0.6,
            compactor: None,
            subagents: vec![],
            subagent_specs: vec![],
            handoffs: vec![],
            max_handoffs: 5,
            runtime_agents: None,
            max_runtime_agents: None,
            runtime_agent_tools: None,
            version: "v1".into(),
            versions: IndexMap::new(),
            memory: MemoryChoice::On,
            trace: None,
            guardrails: None,
            identity: None,
            workspace: None,
            allow_shell: None,
        }
    }

    /// Give the agent somewhere to work: its file tools (and `shell`, when
    /// allowed) act inside it, and cowork hands back the files it wrote.
    pub fn workspace(mut self, workspace: Arc<dyn Workspace>) -> Self {
        self.workspace = Some(WorkspaceChoice::Given(workspace));
        self
    }

    /// A workspace said as data, the way Python and blueprints say it: `true`
    /// is a folder from the harness's broker; a name, a URL or a mapping is a
    /// sandbox — `"docker"`, `"e2b://template"`, `{"sandbox": "daytona", ...}`.
    pub fn workspace_spec(mut self, spec: Value) -> Self {
        self.workspace = match spec {
            Value::Bool(true) => Some(WorkspaceChoice::Folder),
            Value::Bool(false) | Value::Null => None,
            Value::String(s) if YES.contains(&s.trim().to_lowercase().as_str()) => Some(WorkspaceChoice::Folder),
            Value::String(s) if NO.contains(&s.trim().to_lowercase().as_str()) => None,
            other => Some(WorkspaceChoice::Sandbox(other)),
        };
        self
    }

    /// May the agent run shell commands in its workspace? On this machine a
    /// shell is asked for, never assumed; in a sandbox it is on by default.
    pub fn allow_shell(mut self, on: bool) -> Self {
        self.allow_shell = Some(on);
        self
    }

    /// Who this agent is, who answers for it, and what it may do — an
    /// `AgentIdentity` as JSON. Governance fills in a default, and reports the
    /// gap, when it is left out.
    pub fn identity(mut self, identity: Value) -> Self {
        self.identity = Some(identity);
        self
    }

    /// What this agent may touch, and what has to be true of its answer before
    /// it is accepted. Its `content` rules, if any, replace the harness's.
    pub fn guardrails(mut self, guardrails: AgentGuardrails) -> Self {
        self.guardrails = Some(guardrails);
        self
    }

    /// Memory on (the default) or off. On, the agent loads `user.md` into its
    /// prompt, recalls precedent for each task, and can `remember` and `recall`.
    pub fn memory(mut self, on: bool) -> Self {
        self.memory = if on { MemoryChoice::On } else { MemoryChoice::Off };
        self
    }

    /// Remember with this manager — one store and index shared by many agents.
    pub fn memory_manager(mut self, memory: MemoryManager) -> Self {
        self.memory = MemoryChoice::Manager(memory);
        self
    }

    /// Whose memory this is: a [`Trace`], or a bare user id.
    pub fn trace(mut self, trace: impl Into<Trace>) -> Self {
        self.trace = Some(trace.into());
        self
    }

    pub fn instructions(mut self, text: impl Into<String>) -> Self {
        self.instructions = text.into();
        self
    }

    /// Instructions from a prompt template, rendered with its own variables.
    pub fn prompt(mut self, prompt: &Prompt) -> Self {
        self.instructions = prompt.to_string();
        self
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.description = text.into();
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// `fast`, `balanced` or `deep`: let the router pick the model.
    pub fn tier(mut self, tier: impl Into<String>) -> Self {
        self.tier = Some(tier.into());
        self
    }

    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// A provider by name: `"openai"`, `"groq"`, `"ollama"`...
    pub fn provider_name(mut self, name: impl Into<String>) -> Self {
        self.provider_name = Some(name.into());
        self
    }

    pub fn effort(mut self, effort: Effort) -> Self {
        self.params.effort = Some(effort);
        self
    }

    pub fn temperature(mut self, t: f64) -> Self {
        self.params.temperature = Some(t);
        self
    }

    pub fn max_tokens(mut self, n: u32) -> Self {
        self.params.max_tokens = n;
        self
    }

    pub fn thinking(mut self, on: bool) -> Self {
        self.params.thinking = Some(on);
        self
    }

    pub fn thinking_budget(mut self, tokens: i64) -> Self {
        self.params.thinking_budget = Some(tokens);
        self
    }

    pub fn top_p(mut self, p: f64) -> Self {
        self.params.top_p = Some(p);
        self
    }

    pub fn top_k(mut self, k: u32) -> Self {
        self.params.top_k = Some(k);
        self
    }

    pub fn min_p(mut self, p: f64) -> Self {
        self.params.min_p = Some(p);
        self
    }

    pub fn frequency_penalty(mut self, p: f64) -> Self {
        self.params.frequency_penalty = Some(p);
        self
    }

    pub fn presence_penalty(mut self, p: f64) -> Self {
        self.params.presence_penalty = Some(p);
        self
    }

    pub fn repetition_penalty(mut self, p: f64) -> Self {
        self.params.repetition_penalty = Some(p);
        self
    }

    pub fn seed(mut self, seed: i64) -> Self {
        self.params.seed = Some(seed);
        self
    }

    pub fn cache(mut self, on: bool) -> Self {
        self.params.cache = Some(on);
        self
    }

    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.params.user = Some(user.into());
        self
    }

    pub fn stop<I: IntoIterator<Item = S>, S: Into<String>>(mut self, stop: I) -> Self {
        self.params.stop = stop.into_iter().map(Into::into).collect();
        self
    }

    pub fn tool_choice(mut self, choice: Value) -> Self {
        self.params.tool_choice = Some(choice);
        self
    }

    /// Anything else, merged into the provider payload verbatim.
    pub fn model_option(mut self, key: &str, value: Value) -> Self {
        self.params.extra.insert(key.into(), value);
        self
    }

    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tools<I: IntoIterator<Item = Tool>>(mut self, tools: I) -> Self {
        self.tools.extend(tools);
        self
    }

    pub fn skills(mut self, skills: SkillRegistry) -> Self {
        self.skills = Some(skills);
        self
    }

    pub fn harness(mut self, harness: Harness) -> Self {
        self.harness = Some(harness);
        self
    }

    /// This agent's own hooks, run before the harness-wide ones.
    pub fn hooks(mut self, hooks: HookEngine) -> Self {
        self.hooks = Some(hooks);
        self
    }

    pub fn policy(mut self, policy: PolicyGate) -> Self {
        self.policy = Some(policy);
        self
    }

    pub fn budget(mut self, budget: Budget) -> Self {
        self.budget = Some(budget);
        self
    }

    pub fn max_steps(mut self, n: u32) -> Self {
        self.max_steps = Some(n);
        self
    }

    /// A way of working: `modes::chat`, `modes::research`, `modes::cowork`.
    /// It fills in only what was left unset.
    pub fn mode(mut self, mode: Mode) -> Self {
        self.mode = Some(mode);
        self.mode_spec = None;
        self
    }

    /// A mode by name (`chat`, `research`, `cowork`, `voice`) at a depth.
    pub fn mode_named(mut self, name: &str, depth: Option<&str>) -> Self {
        self.mode = None;
        self.mode_spec = Some((name.into(), depth.map(str::to_string)));
        self
    }

    /// The final answer must be a JSON object that deserialises into `T`;
    /// it lands in `result.data`.
    pub fn output_type<T: DeserializeOwned + JsonSchema + 'static>(mut self) -> Self {
        self.output = Some(OutputContract::of::<T>());
        self
    }

    pub fn output_contract(mut self, contract: OutputContract) -> Self {
        self.output = Some(contract);
        self
    }

    pub fn parallel_tools(mut self, on: bool) -> Self {
        self.parallel_tools = on;
        self
    }

    pub fn contract_retries(mut self, n: u32) -> Self {
        self.contract_retries = n;
        self
    }

    pub fn persist_session(mut self, on: bool) -> Self {
        self.persist_session = on;
        self
    }

    pub fn max_context_tokens(mut self, n: usize) -> Self {
        self.max_context_tokens = Some(n);
        self
    }

    /// Compact at this many tokens — or, at or below 1.0, this fraction of
    /// the model's context window.
    pub fn compact_at(mut self, at: f64) -> Self {
        self.compact_at = Some(at);
        self
    }

    pub fn compact_keep_last(mut self, n: usize) -> Self {
        self.compact_keep_last = n;
        self
    }

    pub fn compact_target(mut self, ratio: f64) -> Self {
        self.compact_target = ratio;
        self
    }

    pub fn compactor(mut self, compactor: ContextCompactor) -> Self {
        self.compactor = Some(compactor);
        self
    }

    /// A sub-agent this one may `delegate` a task to.
    pub fn subagent(mut self, agent: Agent) -> Self {
        self.subagents.push(agent);
        self
    }

    /// A sub-agent built from a blueprint, sharing this agent's harness.
    pub fn subagent_spec(mut self, spec: SubAgentSpec) -> Self {
        self.subagent_specs.push(spec);
        self
    }

    /// Somewhere this agent may hand the conversation: an `Agent`, or a
    /// `Handoff` saying how.
    pub fn handoff(mut self, to: impl Into<Handoff>) -> Self {
        self.handoffs.push(to.into());
        self
    }

    /// How many times a conversation may change hands in one turn.
    pub fn max_handoffs(mut self, n: u32) -> Self {
        self.max_handoffs = n;
        self
    }

    /// Let this agent write and run new specialists during a run.
    pub fn runtime_agents(mut self, on: bool) -> Self {
        self.runtime_agents = Some(on);
        self
    }

    /// How many run-time agents one run may spin up: 0 to 100.
    pub fn max_runtime_agents(mut self, n: u32) -> Self {
        self.max_runtime_agents = Some(n);
        self
    }

    /// The tools a run-time agent may be given (default: all but orchestration tools).
    pub fn runtime_agent_tools<I: IntoIterator<Item = S>, S: Into<String>>(mut self, tools: I) -> Self {
        self.runtime_agent_tools = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    /// Which version is active.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    pub fn add_version(mut self, name: impl Into<String>, version: AgentVersion) -> Self {
        self.versions.insert(name.into(), version);
        self
    }

    /// Apply a version's overrides to this builder.
    fn apply(&mut self, v: &AgentVersion) {
        if let Some(i) = &v.instructions {
            self.instructions = i.clone();
        }
        if let Some(d) = v.description.as_ref().filter(|d| !d.is_empty()) {
            self.description = d.clone();
        }
        if let Some(m) = v.model.as_ref().filter(|m| !m.is_empty()) {
            self.model = Some(m.clone());
        }
        if let Some(t) = v.tier.as_ref().filter(|t| !t.is_empty()) {
            self.tier = Some(t.clone());
        }
        let p = &mut self.params;
        p.effort = v.effort.or(p.effort);
        p.temperature = v.temperature.or(p.temperature);
        if let Some(n) = v.max_tokens.filter(|n| *n > 0) {
            p.max_tokens = n;
        }
        p.thinking = v.thinking.or(p.thinking);
        p.thinking_budget = v.thinking_budget.or(p.thinking_budget);
        p.top_p = v.top_p.or(p.top_p);
        p.top_k = v.top_k.or(p.top_k);
        p.min_p = v.min_p.or(p.min_p);
        p.frequency_penalty = v.frequency_penalty.or(p.frequency_penalty);
        p.presence_penalty = v.presence_penalty.or(p.presence_penalty);
        p.repetition_penalty = v.repetition_penalty.or(p.repetition_penalty);
        p.seed = v.seed.or(p.seed);
        if let Some(n) = v.max_steps.filter(|n| *n > 0) {
            self.max_steps = Some(n);
        }
        if let Some(m) = &v.mode {
            // A version that changes the mode takes its depth with it.
            self.mode = None;
            self.mode_spec = Some((m.clone(), v.depth.clone()));
        } else if let Some(d) = &v.depth {
            let current = self.mode.as_ref().map(|m| m.name.clone()).or_else(|| self.mode_spec.as_ref().map(|s| s.0.clone()));
            if let Some(name) = current {
                self.mode = None;
                self.mode_spec = Some((name, Some(d.clone())));
            }
        }
        if v.compact_at.is_some() {
            self.compact_at = v.compact_at;
        }
        if let Some(n) = v.contract_retries {
            self.contract_retries = n;
        }
        if v.runtime_agents.is_some() {
            self.runtime_agents = v.runtime_agents;
        }
        if v.max_runtime_agents.is_some() {
            self.max_runtime_agents = v.max_runtime_agents;
        }
        if v.budget.is_some() {
            self.budget = v.budget.clone();
        }
        if let Some(wanted) = &v.tools {
            self.tools.retain(|t| wanted.contains(&t.name));
        }
        if v.subagents.is_some() || v.subagent_specs.is_some() {
            self.subagents = v.subagents.clone().unwrap_or_default();
            self.subagent_specs = v.subagent_specs.clone().unwrap_or_default();
        }
        if let Some(h) = &v.handoffs {
            self.handoffs = h.iter().cloned().map(Handoff::from).collect();
        }
    }

    pub fn build(self) -> Result<Agent> {
        self.build_with(None)
    }

    fn build_with(mut self, siblings: Option<Arc<Mutex<HashMap<String, Agent>>>>) -> Result<Agent> {
        // --- versions: what a version leaves unset falls through ---------
        let base = if self.versions.is_empty() {
            None
        } else {
            let Some(active) = self.versions.get(&self.version).cloned() else {
                let mut known: Vec<&String> = self.versions.keys().collect();
                known.sort();
                let known: Vec<&str> = known.iter().map(|s| s.as_str()).collect();
                return Err(Error::config(format!("{}: version '{}' is not defined; known: {}", self.name, self.version, known.join(", "))));
            };
            let base = self.clone();
            self.apply(&active);
            Some(Box::new(base))
        };
        // --- mode: fills in only what was left unset ----------------------
        let mode = match (&self.mode, &self.mode_spec) {
            (Some(m), _) => Some(m.clone()),
            (None, Some((name, depth))) => Some(resolve_mode(name, depth.as_deref()).map_err(|e| Error::config(format!("{}: {e}", self.name)))?),
            (None, None) => None,
        };
        let max_steps = self.max_steps.unwrap_or_else(|| mode.as_ref().map_or(20, |m| m.max_steps));
        let helpers = mode.as_ref().map_or(0, |m| m.helpers);
        let runtime_agents = self.runtime_agents.unwrap_or(helpers > 0);
        let max_runtime_agents = self.max_runtime_agents.unwrap_or(if helpers > 0 { helpers } else { 5 });
        if self.tier.is_none() {
            self.tier = mode.as_ref().and_then(|m| m.tier.clone());
        }
        if max_runtime_agents > MAX_RUNTIME_AGENTS {
            return Err(Error::config(format!("max_runtime_agents must be between 0 and {MAX_RUNTIME_AGENTS} — got {max_runtime_agents}")));
        }
        if runtime_agents && max_runtime_agents == 0 {
            return Err(Error::config(format!(
                "{}: runtime_agents is enabled but max_runtime_agents is 0 — give it a budget between 1 and 100, or disable it",
                self.name
            )));
        }
        let p = &self.params;
        let opt = |x: Option<Value>| x.unwrap_or(Value::Null);
        // Checked now, so a bad value fails when the agent is built — not mid-run.
        validate_parameters(&[
            ("effort", opt(p.effort.map(|e| json!(e.as_str())))),
            ("temperature", opt(p.temperature.map(|v| json!(v)))),
            ("max_tokens", json!(p.max_tokens)),
            ("thinking", opt(p.thinking.map(|v| json!(v)))),
            ("thinking_budget", opt(p.thinking_budget.map(|v| json!(v)))),
            ("top_p", opt(p.top_p.map(|v| json!(v)))),
            ("top_k", opt(p.top_k.map(|v| json!(v)))),
            ("min_p", opt(p.min_p.map(|v| json!(v)))),
            ("frequency_penalty", opt(p.frequency_penalty.map(|v| json!(v)))),
            ("presence_penalty", opt(p.presence_penalty.map(|v| json!(v)))),
            ("repetition_penalty", opt(p.repetition_penalty.map(|v| json!(v)))),
            ("seed", opt(p.seed.map(|v| json!(v)))),
            ("stop", json!(p.stop)),
        ])?;
        let harness = self.harness.clone().unwrap_or_default();
        let task = format!("{} {}", self.name, self.description);
        let (model, routed_effort) = harness.router.pick(self.model.as_deref(), self.tier.as_deref(), &task);
        let explicit_provider = self.provider.is_some() || self.provider_name.is_some();
        let provider = match (&self.provider, &self.provider_name, &harness.provider) {
            (Some(p), _, _) => p.clone(),
            (None, Some(name), _) => get_provider(name)?,
            (None, None, Some(p)) => p.clone(),
            (None, None, None) => resolve_provider(None, Some(&model))?,
        };
        let mut template = self.params.clone();
        template.model = model.clone();
        template.effort = template.effort.or(routed_effort);
        template.response_schema = self.output.as_ref().map(|o| o.schema.clone());

        let mut tools = ToolRegistry::new(self.tools.clone());
        if let Some(skills) = &self.skills {
            tools.extend(skills.tools());
        }
        // --- memory -------------------------------------------------------
        let trace = self.trace.clone().map(|t| Trace { agent: Some(self.name.clone()), ..t });
        let memory = match &self.memory {
            MemoryChoice::Off => None,
            MemoryChoice::Shared(m) => Some(m.clone()),
            MemoryChoice::Manager(m) => Some(match &trace {
                Some(t) => m.for_trace(t.clone()),
                None => m.clone(),
            }),
            MemoryChoice::On => Some(MemoryManager::new(MemoryOptions {
                store: Some(harness.memory_store.clone()),
                summarize: Some(summarizer(provider.clone(), &harness, &model)),
                trace: trace.clone(),
                ..Default::default()
            })),
        };
        let trace = memory.as_ref().map(MemoryManager::trace).or(trace);
        if let Some(m) = &memory {
            tools.extend(m.tools());
        }
        // --- workspace ----------------------------------------------------
        let workspace: Option<Arc<dyn Workspace>> = match &self.workspace {
            None => None,
            Some(WorkspaceChoice::Given(w)) => Some(w.clone()),
            Some(WorkspaceChoice::Sandbox(spec)) => Some(Arc::new(crate::sandboxes::sandbox(spec)?)),
            Some(WorkspaceChoice::Folder) => {
                let w = harness.workspaces.acquire(&self.name, true)?;
                if self.allow_shell.is_none() && !w.isolated() {
                    w.set_allow_shell(false); // on this machine a shell is asked for, never assumed
                }
                Some(w)
            }
        };
        if let Some(w) = &workspace {
            // The harness closes it, so a sandbox is not left running.
            harness.workspaces.adopt(w.clone());
            if let Some(on) = self.allow_shell {
                w.set_allow_shell(on);
            }
            tools.extend(workspace_tools(w.clone()));
        }
        if let Some(m) = &mode {
            tools.extend(m.tools());
            // With somewhere to work, a mode can read documents and run Python there.
            if let Some(w) = &workspace {
                if m.documents {
                    tools.extend([crate::toolkits::documents::make_document_tool(Some(w.clone()), "parse_document")]);
                }
                if m.python {
                    tools.extend([crate::toolkits::compute::make_python_tool(w.clone())]);
                }
            }
        }
        let hooks = match &self.hooks {
            Some(h) => h.merge(Some(&harness.hooks)),
            None => harness.hooks.clone(),
        };
        let policy = self.policy.clone().unwrap_or_else(|| harness.policy.clone());
        let window = (model_info(&model).map_or(200_000, |i| i.context_window) as f64 * 0.66) as usize;
        let compact_at = match (self.compact_at, self.max_context_tokens) {
            (None, max) => max.unwrap_or(window),
            (Some(f), _) if f > 0.0 && f <= 1.0 => ((window as f64 / 0.66 * f) as usize).max(1),
            (Some(n), _) if n >= 1.0 => n as usize,
            (Some(bad), _) => {
                return Err(Error::config(format!(
                    "{}: compact_at must be a positive token count, or a fraction of the context window between 0 and 1 — got {bad}",
                    self.name
                )));
            }
        };
        let summarize = summarizer(provider.clone(), &harness, &model);
        let compactor = self.compactor.clone().unwrap_or(ContextCompactor {
            max_tokens: compact_at,
            keep_last: self.compact_keep_last,
            target_ratio: self.compact_target,
            summarize: Some(summarize),
            ..Default::default()
        });
        let description = if self.description.is_empty() { format!("{} agent", self.name) } else { self.description.clone() };
        let assembler = ContextAssembler {
            identity: identity(&self.name, &self.description),
            instructions: self.instructions.clone(),
            skills: self.skills.clone(),
            ..Default::default()
        };
        let agent = Agent {
            inner: Arc::new(AgentInner {
                name: self.name.clone(),
                description,
                instructions: self.instructions.clone(),
                model,
                provider,
                explicit_provider,
                template,
                tools: RwLock::new(tools),
                skills: self.skills.clone(),
                harness,
                hooks,
                policy,
                budget: self.budget.clone(),
                max_steps,
                output: self.output.clone(),
                parallel_tools: self.parallel_tools,
                contract_retries: self.contract_retries,
                persist_session: self.persist_session,
                tool_images_kept: TOOL_IMAGES_KEPT,
                mode,
                assembler,
                compactor,
                fallback_providers: Mutex::new(HashMap::new()),
                subagents: RwLock::new(IndexMap::new()),
                handoffs: RwLock::new(IndexMap::new()),
                max_handoffs: self.max_handoffs,
                runtime_agents,
                max_runtime_agents,
                runtime_agent_tools: self.runtime_agent_tools.clone(),
                spawned: AtomicU32::new(0),
                total_spawned: AtomicU32::new(0),
                factory: OnceLock::new(),
                version: self.version.clone(),
                versions: self.versions.clone(),
                base,
                siblings: siblings.unwrap_or_default(),
                thread: Mutex::new(None),
                following: AtomicBool::new(false),
                memory,
                trace,
                guardrails: self.guardrails.clone(),
                workspace: workspace.clone(),
                log: Mutex::new(None),
            }),
        };
        if let Some(gov) = agent.inner.harness.governance.clone() {
            let info = crate::governance::AgentInfo {
                name: agent.inner.name.clone(),
                description: agent.inner.description.clone(),
                model: agent.inner.model.clone(),
                tools: agent.tools().names(),
                identity: self.identity.clone(),
            };
            // Held weakly: the inventory reads the live tool list, and never keeps the agent alive.
            let weak = Arc::downgrade(&agent.inner);
            let live: crate::governance::inventory::LiveTools =
                Arc::new(move || weak.upgrade().map(|me| me.tools.read().unwrap_or_else(|p| p.into_inner()).clone()));
            gov.register_agent_with_tools(&info, live)?;
        }
        if let Some(m) = &agent.inner.memory {
            // Every durable memory write passes the `memory_write` hook first:
            // a handler may refuse it, or rewrite it — to strip personal data, say.
            let weak = Arc::downgrade(&agent.inner);
            m.set_on_write(Some(Arc::new(move |text: String, scope: String, kind: String| {
                let weak = weak.clone();
                Box::pin(async move {
                    let Some(me) = weak.upgrade() else { return Ok(text) };
                    let trace = me.trace.as_ref().map(|t| serde_json::to_value(t).unwrap_or_default()).unwrap_or(Value::Null);
                    let hook = me
                        .hooks
                        .emit(
                            "memory_write",
                            &me.name,
                            "",
                            0,
                            data(vec![("text", json!(text)), ("scope", json!(scope)), ("kind", json!(kind)), ("trace", trace)]),
                        )
                        .await;
                    if hook.blocked {
                        return Err(Error::PermissionDenied {
                            message: format!("memory write refused: {}", hook.reason),
                            tool: "remember".into(),
                            reason: hook.reason,
                        });
                    }
                    Ok(match hook.replacement {
                        Some(Value::String(s)) => s,
                        Some(other) => other.to_string(),
                        None => text,
                    })
                }) as futures::future::BoxFuture<'static, Result<String>>
            })));
        }
        for child in self.subagents {
            agent.add_subagent(child);
        }
        for spec in self.subagent_specs {
            agent.add_subagent_spec(spec)?;
        }
        for h in self.handoffs {
            agent.add_handoff(h)?;
        }
        if agent.inner.runtime_agents {
            agent.add_tool(agent.spawn_tool());
            if !agent.tools().contains("delegate") {
                agent.add_tool(agent.delegate_tool());
            }
        }
        Ok(agent)
    }
}

/// Turn a sub-agent blueprint into a live agent that shares `parent`'s harness.
///
/// Tools: the spec's allowlist, or the parent's own tools minus the
/// orchestration-only ones — delegation never cascades by default.
pub fn build_agent(spec: &SubAgentSpec, parent: &Agent) -> Result<Agent> {
    let tools: Vec<Tool> = match &spec.tools {
        None => parent.tools().iter().filter(|t| !t.tags.iter().any(|g| NOT_INHERITED.contains(&g.as_str()))).cloned().collect(),
        Some(globs) => parent.tools().select(Some(globs)).iter().cloned().collect(),
    };
    let held = &parent.inner.policy;
    let policy = if spec.allow.is_empty() && spec.ask.is_empty() && spec.deny.is_empty() {
        held.clone()
    } else {
        // The spec's rules first, then the parent's — as the Python harness does.
        let mut gate = PolicyGate::new(held.default).deny(spec.deny.clone()).ask(spec.ask.clone()).allow(spec.allow.clone());
        for rule in &held.rules {
            gate = gate.rule(rule.clone());
        }
        gate.approver = held.approver.clone();
        gate.audit = held.audit.clone();
        gate
    };
    let mut b = Agent::builder(&spec.name)
        .instructions(spec.instructions.clone())
        .description(spec.description.clone())
        .tools(tools)
        .harness(parent.inner.harness.clone())
        .policy(policy)
        .max_tokens(spec.max_tokens)
        .memory(spec.memory)
        .persist_session(false);
    // A provider handed to the parent explicitly is the backend for the whole tree.
    if parent.inner.explicit_provider {
        b = b.provider(parent.inner.provider.clone());
    }
    if let Some(m) = &spec.model {
        b = b.model(m.clone());
    }
    if let Some(t) = &spec.tier {
        b = b.tier(t.clone());
    }
    if let Some(e) = spec.effort {
        b = b.effort(e);
    }
    let p = &mut b.params;
    p.temperature = spec.temperature;
    p.thinking = spec.thinking;
    p.thinking_budget = spec.thinking_budget;
    p.top_p = spec.top_p;
    p.top_k = spec.top_k;
    p.min_p = spec.min_p;
    p.frequency_penalty = spec.frequency_penalty;
    p.presence_penalty = spec.presence_penalty;
    p.repetition_penalty = spec.repetition_penalty;
    p.seed = spec.seed;
    // A mode decides the step ceiling unless the spec said one itself.
    match &spec.mode {
        Some(m) => b = b.mode_named(m, spec.depth.as_deref()),
        None => b = b.max_steps(spec.max_steps),
    }
    if let Some(budget) = &spec.budget {
        b = b.budget(budget.clone());
    }
    if let Some(schema) = &spec.output_schema {
        b = b.output_contract(OutputContract::schema(schema.clone()));
    }
    // A folder of its own, or the shared one parallel sub-agents hand files
    // through. A folder on this machine has a shell only if the spec asked.
    let harness = &parent.inner.harness;
    let workspace = match spec.workspace {
        crate::subagents::spec::WorkspaceKind::Isolated => Some(harness.workspaces.acquire(&format!("{}-{}", spec.name, new_id("")), true)?),
        crate::subagents::spec::WorkspaceKind::Shared => Some(harness.workspaces.shared()?),
        crate::subagents::spec::WorkspaceKind::None => None,
    };
    if let Some(w) = workspace {
        b = b.workspace(w).allow_shell(spec.allow_shell);
    }
    if let Some(rails) = spec.guardrails.as_ref().filter(|g| !g.is_null()) {
        b = b.guardrails(AgentGuardrails::from_value(rails).map_err(|e| Error::config(format!("{}: guardrails: {e}", spec.name)))?);
    }
    if let (Some(skills), Some(names)) = (parent.skills(), &spec.skills) {
        b = b.skills(skills.select(Some(names)));
    }
    b.build()
}

/// A single cheap model call used for compaction.
fn summarizer(provider: Arc<dyn Provider>, harness: &Harness, agent_model: &str) -> Summarizer {
    let (fast, _) = harness.router.pick(None, Some("fast"), "");
    let model = if provider.name() == "fake" { agent_model.to_string() } else { fast };
    Arc::new(move |prompt: String| {
        let provider = provider.clone();
        let model = model.clone();
        Box::pin(async move {
            let mut req = CompletionRequest::new(model, vec![Message::user(prompt)]);
            req.max_tokens = 1500;
            provider.complete(&req).await.map(|r| r.text()).unwrap_or_default()
        })
    })
}

/// What the driver hands back: events as they happen, then the result.
enum Beat {
    Event(StreamEvent),
    Done(Box<RunResult>),
}

macro_rules! tri {
    ($label:lifetime, $e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => break $label Err(e),
        }
    };
}

fn data(pairs: Vec<(&str, Value)>) -> Map<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

fn ev(kind: StreamEventType, agent: &str, step: u32) -> StreamEvent {
    StreamEvent::new(kind).with_step(agent, step)
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// A model we could not reach is worth retrying elsewhere. A 400 is not.
fn is_reachability_problem(e: &Error) -> bool {
    let Error::Provider(p) = e else { return false };
    if p.retryable || p.kind == ProviderErrorKind::QuotaExceeded {
        return true;
    }
    match p.status {
        None => !(p.kind == ProviderErrorKind::Authentication || p.kind.is_invalid_request()),
        Some(s) => [408, 429, 500, 502, 503, 504, 529].contains(&s),
    }
}

/// Pull a JSON object out of a reply, fenced or not.
pub fn extract_json(text: &str) -> Option<Value> {
    if text.is_empty() {
        return None;
    }
    let mut candidates = vec![text.trim().to_string()];
    let fence = Regex::new(r"(?s)```(?:json)?\s*(.*?)```").expect("valid regex");
    if let Some(c) = fence.captures(text) {
        candidates.insert(0, c[1].trim().to_string());
    }
    if let (Some(start), Some(end)) = (text.find('{'), text.rfind('}'))
        && end > start
    {
        candidates.push(text[start..=end].to_string());
    }
    candidates.iter().find_map(|c| serde_json::from_str(c).ok())
}

/// Thousands separators, as Python's `f"{x:,.0f}"`.
fn grouped(x: f64) -> String {
    let digits = format!("{:.0}", x.abs());
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if x < 0.0 { format!("-{out}") } else { out }
}

/// The line appended to partial work when a ceiling is reached.
fn budget_note(kind: &str, limit: f64, spent: f64) -> String {
    let show = |x: f64| if x >= 1.0 { grouped(x) } else { format!("{x}") };
    let axis = if kind.is_empty() { "budget".to_string() } else { kind.replace('_', " ") };
    format!("\n\n[The budget for this agent is exceeded — {axis} {} of {}. The answer above is what it completed before stopping.]", show(spent), show(limit))
}

/// Replace all but the newest `keep` tool-returned images with a note.
fn drop_old_tool_media(messages: &mut [Message], keep: usize) {
    let mut left = keep;
    for message in messages.iter_mut().rev() {
        if !message.content.iter().any(|b| matches!(b, ContentBlock::ToolResult(_))) {
            continue;
        }
        for block in message.content.iter_mut().rev() {
            let Some((kind, m)) = block.as_media() else { continue };
            if left > 0 {
                left -= 1;
                continue;
            }
            let what = if m.name.is_empty() { kind.as_str().to_string() } else { m.name.clone() };
            *block = ContentBlock::text(format!("[an earlier {what} was here; it is no longer kept — take a new one if you need to look again]"));
        }
    }
}

fn hook_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// What travels into a run besides its options: the relay, when the run is
/// one leg of a conversation changing hands.
#[derive(Clone, Default)]
struct Internal {
    relay: Option<Arc<Mutex<Relay>>>,
    /// A helper's view of its lead's notebook: its own todos, the shared ledger.
    notebook: Option<Arc<Notebook>>,
    /// Picked back up from a stored approval.
    resume: Option<Approval>,
    /// The span this run nests under — a sub-agent's under its lead's.
    parent_span: Option<Span>,
    /// The run that started this one, for a sub-agent.
    parent_run_id: Option<String>,
}

/// Arguments in one spelling, so two sets can be told to be the same.
fn canonical(args: &Map<String, Value>) -> String {
    crate::pyjson::dumps_with(&Value::Object(args.clone()), crate::pyjson::Dumps { sort_keys: true, ensure_ascii: false, ..Default::default() })
}

fn detail(pairs: Vec<(&str, Value)>) -> Map<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

impl Agent {
    pub fn builder(name: &str) -> AgentBuilder {
        AgentBuilder::new(name)
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }

    pub fn description(&self) -> &str {
        &self.inner.description
    }

    pub fn instructions(&self) -> &str {
        &self.inner.instructions
    }

    pub fn model(&self) -> &str {
        &self.inner.model
    }

    pub fn provider(&self) -> &Arc<dyn Provider> {
        &self.inner.provider
    }

    /// A snapshot of this agent's tools.
    pub fn tools(&self) -> ToolRegistry {
        self.inner.tools.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    pub fn add_tool(&self, tool: Tool) {
        self.inner.tools.write().unwrap_or_else(|p| p.into_inner()).add(tool);
    }

    pub fn skills(&self) -> Option<&SkillRegistry> {
        self.inner.skills.as_ref()
    }

    pub fn harness(&self) -> &Harness {
        &self.inner.harness
    }

    pub fn budget(&self) -> Option<&Budget> {
        self.inner.budget.as_ref()
    }

    /// The parameters every request from this agent starts from.
    pub fn generation(&self) -> &CompletionRequest {
        &self.inner.template
    }

    /// The mode this agent works in, if any.
    pub fn mode(&self) -> Option<&Mode> {
        self.inner.mode.as_ref()
    }

    /// How hard its mode works: fast, balanced or deep. "" without one.
    pub fn depth(&self) -> &str {
        self.inner.mode.as_ref().map_or("", |m| m.depth.as_str())
    }

    /// The step ceiling a run gets when it names none.
    pub fn max_steps_hint(&self) -> u32 {
        self.inner.max_steps
    }

    /// Does one run pick up where the last one left off?
    pub fn conversational(&self) -> bool {
        self.inner.mode.as_ref().is_some_and(|m| m.conversational)
    }

    fn is(&self, other: &Agent) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    // ---- versions ------------------------------------------------------
    /// Which version of this agent is running.
    pub fn version(&self) -> &str {
        &self.inner.version
    }

    pub fn available_versions(&self) -> Vec<String> {
        let mut v: Vec<String> = self.inner.versions.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn version_spec(&self, version: Option<&str>) -> Option<&AgentVersion> {
        self.inner.versions.get(version.unwrap_or(&self.inner.version))
    }

    /// This agent, configured as `version`. Built once, then reused; the
    /// harness and provider are shared, so versions are comparable.
    pub fn use_version(&self, version: &str) -> Result<Agent> {
        if version == self.inner.version {
            return Ok(self.clone());
        }
        let Some(base) = &self.inner.base else {
            return Err(Error::config(format!("{}: version '{version}' is not defined; known: none", self.inner.name)));
        };
        if !self.inner.versions.contains_key(version) {
            return Err(Error::config(format!("{}: version '{version}' is not defined; known: {}", self.inner.name, self.available_versions().join(", "))));
        }
        if let Some(a) = lock(&self.inner.siblings).get(version) {
            return Ok(a.clone());
        }
        let mut base = (**base).clone().version(version);
        // The harness, the provider and the memory are infrastructure, shared by every version.
        if let Some(m) = &self.inner.memory {
            base.memory = MemoryChoice::Shared(m.clone());
        }
        // Every version of this agent works in the one workspace.
        if let Some(w) = &self.inner.workspace {
            base.workspace = Some(WorkspaceChoice::Given(w.clone()));
        }
        let sibling = base.build_with(Some(self.inner.siblings.clone()))?;
        // A handoff added after the agent was built is part of how every
        // version hands off — unless the version says where it hands off.
        if self.inner.versions[version].handoffs.is_none() {
            let held = sibling.handoffs();
            for (name, h) in self.handoffs() {
                if !held.contains_key(&name) {
                    sibling.add_handoff(h)?;
                }
            }
        }
        lock(&self.inner.siblings).insert(version.into(), sibling.clone());
        Ok(sibling)
    }

    // ---- conversations -------------------------------------------------
    /// Pick a conversation back up: every run from here continues it.
    pub async fn resume(&self, session_id: &str) -> Result<Session> {
        let loaded = self.inner.harness.sessions.load(session_id).await?;
        *lock(&self.inner.thread) = Some(loaded.clone());
        self.inner.following.store(true, Ordering::SeqCst);
        Ok(loaded)
    }

    /// Start a fresh conversation: the next run does not see the last one.
    ///
    /// Only the thread is dropped. What the agent remembered about the user
    /// stays.
    pub fn new_session(&self) {
        *lock(&self.inner.thread) = None;
        self.inner.following.store(false, Ordering::SeqCst);
        if let Some(m) = &self.inner.memory {
            m.new_session();
        }
    }

    /// The system prompt a run would start with, naming these tools.
    pub(crate) fn system_prompt(&self, tool_names: &[String]) -> String {
        let mut assembler = self.inner.assembler.clone();
        if let Some(m) = &self.inner.mode {
            assembler.mode = m.prompt(tool_names);
        }
        assembler.build(tool_names, "")
    }

    /// The text rules for this agent: its own if it has them, else the harness's.
    pub(crate) fn content_guardrails(&self) -> crate::guardrails::Guardrails {
        self.inner.content()
    }

    /// The agent's hooks, for the parts of the harness that act on its behalf.
    pub(crate) fn hook_engine(&self) -> &HookEngine {
        &self.inner.hooks
    }

    /// Save a session as a run would: merged if someone else saved first,
    /// forked if it cannot be. Problems become warnings on `result`.
    pub(crate) async fn save_session_record(&self, session: &mut Session, result: &mut RunResult, added: &[Message]) {
        self.inner.save_session(session, result, added, false).await;
    }

    /// Where this agent works, if it was given a workspace.
    pub fn workspace(&self) -> Option<&Arc<dyn Workspace>> {
        self.inner.workspace.as_ref()
    }

    /// Pick a conversation back up — and the sandbox it was working in. The
    /// session remembers which sandbox that was, so the id alone is enough;
    /// `sandbox_id` names a different one. If that sandbox is gone, the next
    /// run starts a new one and puts back the files the conversation wrote.
    pub async fn resume_with_sandbox(&self, session_id: &str, sandbox_id: Option<&str>) -> Result<Session> {
        let mut loaded = self.inner.harness.sessions.load(session_id).await?;
        if let Some(id) = sandbox_id {
            let Some(sb) = self.inner.workspace.as_ref().and_then(|w| w.as_sandbox()) else {
                return Err(Error::config(format!("{}: sandbox_id='{id}' was given, but this agent's workspace is not a sandbox", self.inner.name)));
            };
            loaded.metadata.insert("sandbox".into(), json!({"sandbox": sb.sandbox.name(), "id": id}));
        }
        *lock(&self.inner.thread) = Some(loaded.clone());
        self.inner.following.store(true, Ordering::SeqCst);
        Ok(loaded)
    }

    /// What this agent may touch and must do, if it was given guardrails.
    pub fn guardrails(&self) -> Option<&AgentGuardrails> {
        self.inner.guardrails.as_ref()
    }

    /// The agent's memory, unless it was built with `.memory(false)`.
    pub fn memory(&self) -> Option<&MemoryManager> {
        self.inner.memory.as_ref()
    }

    /// Whose memory this is, and whose sessions.
    pub fn trace(&self) -> Option<&Trace> {
        self.inner.trace.as_ref()
    }

    /// Session close → user memory update. Returns the new `user.md`.
    pub async fn close_session(&self, summary: Option<&str>) -> Result<String> {
        match &self.inner.memory {
            Some(m) => m.close_session(summary).await,
            None => Ok(String::new()),
        }
    }

    /// The conversation this agent is following, if any.
    pub fn thread(&self) -> Option<Session> {
        lock(&self.inner.thread).clone()
    }

    // ---- sub-agents ----------------------------------------------------
    /// Attach a sub-agent this one may `delegate` to. Returns it.
    pub fn add_subagent(&self, child: Agent) -> Agent {
        self.inner.subagents.write().unwrap_or_else(|p| p.into_inner()).insert(child.name().to_string(), child.clone());
        // Rebuilt, not just added: the tool's schema carries the roster.
        self.add_tool(self.delegate_tool());
        child
    }

    /// Attach a sub-agent built from a blueprint.
    pub fn add_subagent_spec(&self, spec: SubAgentSpec) -> Result<Agent> {
        let child = build_agent(&spec, self)?;
        Ok(self.add_subagent(child))
    }

    pub fn subagents(&self) -> IndexMap<String, Agent> {
        self.inner.subagents.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// How many more specialists this agent may spin up in this run.
    pub fn runtime_agents_remaining(&self) -> u32 {
        if !self.inner.runtime_agents {
            return 0;
        }
        self.inner.max_runtime_agents.saturating_sub(self.inner.spawned.load(Ordering::SeqCst))
    }

    /// How many specialists this agent has spun up, ever.
    pub fn total_spawned(&self) -> u32 {
        self.inner.total_spawned.load(Ordering::SeqCst)
    }

    /// The factory this agent writes new specialists with.
    pub fn factory(&self) -> Result<&SubAgentFactory> {
        if let Some(f) = self.inner.factory.get() {
            return Ok(f);
        }
        let (model, _) = self.inner.harness.router.pick(None, Some("balanced"), "");
        let mut b = Agent::builder(&format!("{}.factory", self.inner.name))
            .instructions("You write specifications for sub-agents.")
            .model(model)
            .harness(self.inner.harness.clone())
            .max_steps(2)
            .memory(false)
            .persist_session(false);
        if self.inner.explicit_provider {
            b = b.provider(self.inner.provider.clone());
        }
        let _ = self.inner.factory.set(SubAgentFactory::new(b.build()?));
        Ok(self.inner.factory.get().expect("just set"))
    }

    fn catalogue(&self) -> String {
        self.subagents().iter().map(|(n, a)| format!("{n}: {}", a.description())).collect::<Vec<_>>().join("; ")
    }

    fn delegate_tool(&self) -> Tool {
        let mut names: Vec<String> = self.subagents().keys().cloned().collect();
        names.sort();
        let schema = json!({
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "enum": names, "description": self.catalogue()},
                "task": {"type": "string", "description": "The complete task, stated on its own."},
                "context": {"type": "string", "description": "Facts the sub-agent needs and cannot look up itself."},
            },
            "required": ["agent_name", "task"],
        });
        let me: Weak<AgentInner> = Arc::downgrade(&self.inner);
        Tool::raw(
            "delegate",
            "Hand one self-contained task to a sub-agent and get its result back. Sub-agents start with no memory of this conversation, so put everything they need in `task` and `context`. Delegate several at once when the tasks do not depend on each other.",
            schema,
            move |args, ctx| {
                let me = me.clone();
                async move {
                    let agent = Agent { inner: me.upgrade().ok_or_else(|| Error::config("the agent is gone"))? };
                    let s = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                    Ok(json!(agent.delegate(&s("agent_name"), &s("task"), &s("context"), &ctx).await?))
                }
            },
        )
        .tags(["builtin", "delegation"])
    }

    /// Run an existing sub-agent on one task and hand back only its result.
    async fn delegate(&self, agent_name: &str, task: &str, context: &str, ctx: &ToolContext) -> Result<String> {
        let child = self.subagents().get(agent_name).cloned();
        let Some(child) = child else {
            let mut known: Vec<String> = self.subagents().keys().cloned().collect();
            known.sort();
            let known = if known.is_empty() { "none".into() } else { known.join(", ") };
            let left = self.runtime_agents_remaining();
            let suffix =
                if self.inner.runtime_agents && left > 0 { format!(" Use `spawn_agent` to build one for this task ({left} left).") } else { String::new() };
            return Ok(format!("No sub-agent named '{agent_name}'. Available: {known}.{suffix}"));
        };
        if let Some(m) = &self.inner.memory {
            m.orchestrator.staffing(&task.chars().take(80).collect::<String>(), child.name(), true).await?;
        }
        self.run_child(&child, task, context, ctx).await
    }

    /// Run one sub-agent under this agent's budget and hand back its result.
    async fn run_child(&self, child: &Agent, task: &str, context: &str, ctx: &ToolContext) -> Result<String> {
        let scope = ctx.run.clone();
        let guard = scope.as_ref().and_then(|s| s.guard.clone()).unwrap_or_else(|| self.inner.harness.guard());
        // Asked before any budget is spent on it: a refused hand-off costs nothing.
        let start = self
            .inner
            .hooks
            .emit("subagent_start", child.name(), &ctx.run_id, ctx.step, data(vec![("task", json!(task)), ("parent", json!(self.inner.name))]))
            .await;
        if start.blocked {
            return Ok(format!("[{} not permitted] {}", child.name(), start.reason));
        }
        guard.subagent()?;
        let mut brief = if context.is_empty() { task.to_string() } else { format!("{task}\n\n## Context you were given\n{context}") };
        // A helper keeps its own todo list but writes in the run's one source
        // ledger, so the number it cites is the number the lead can cite.
        let shared = scope.as_ref().and_then(|s| s.notebook.clone()).filter(|n| n.ledger).map(|n| Arc::new(n.child()));
        if shared.is_some() && child.tools().contains("record_source") {
            brief.push_str("\n\n## Sources\nRecord every source you rely on with `record_source` and cite it by the number that returns — [n] after the claim it supports.");
        }
        let options = RunOptions {
            messages: Some(vec![]), // every sub-agent starts clean
            guard: Some(guard.child(child.budget().cloned().unwrap_or_default())),
            ..Default::default()
        };
        let tracer = &self.inner.harness.tracer;
        let parent = ctx.span.as_ref().or_else(|| scope.as_ref().and_then(|s| s.span.as_ref()));
        let mut span = tracer.start(&format!("subagent:{}", child.name()), "subagent", parent);
        let internal = Internal { relay: None, notebook: shared, resume: None, parent_span: Some(span.clone()), parent_run_id: Some(ctx.run_id.clone()) };
        let result = child.run_internal(Message::user(brief), options, internal).await;
        match &result {
            Ok(r) => {
                span.set("cost_usd", r.cost_usd());
                if let Some(e) = &r.error {
                    span.fail(e.clone());
                }
            }
            Err(e) => span.fail(format!("{}: {e}", e.kind_name())),
        }
        tracer.finish(span);
        let result = result?;
        if let Some(m) = &self.inner.memory {
            m.orchestrator.spend(child.name(), result.cost_usd(), result.usage.total_tokens()).await?;
        }
        self.inner
            .hooks
            .emit("subagent_end", child.name(), &ctx.run_id, ctx.step, data(vec![("result", serde_json::to_value(&result).unwrap_or_default())]))
            .await;
        let handback = match &result.error {
            Some(e) => format!("[{} failed] {e}", child.name()),
            None if result.artifacts.is_empty() => result.output.clone(),
            None => {
                let names: Vec<&str> = result.artifacts.iter().map(|a| a.name.as_str()).collect();
                format!("{}\n\nArtefacts produced: {}", result.output, names.join(", "))
            }
        };
        if let Some(scope) = scope {
            lock(&scope.children).push(result);
        }
        Ok(handback)
    }

    fn spawn_tool(&self) -> Tool {
        let schema = json!({
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The complete task, stated on its own."},
                "purpose": {"type": "string", "description": "One line: what this specialist is accountable for."},
                "tools": {"type": "array", "items": {"type": "string"}, "description": "The smallest set of tool names it needs. Omit to let the factory choose."},
            },
            "required": ["task"],
        });
        let description = format!(
            "Build a specialist for one task that no existing sub-agent covers, and run it. You may spin up {} of these in this run; spend them on work that genuinely needs its own worker, and call this several times in one turn when the tasks are independent — they run at the same time. The specialist starts with no memory of this conversation, so put everything it needs in `task`.",
            self.inner.max_runtime_agents
        );
        let me: Weak<AgentInner> = Arc::downgrade(&self.inner);
        Tool::raw("spawn_agent", description, schema, move |args, ctx| {
            let me = me.clone();
            async move {
                let agent = Agent { inner: me.upgrade().ok_or_else(|| Error::config("the agent is gone"))? };
                let task = args.get("task").and_then(Value::as_str).unwrap_or("").to_string();
                let purpose = args.get("purpose").and_then(Value::as_str).unwrap_or("").to_string();
                let tools: Option<Vec<String>> =
                    args.get("tools").and_then(Value::as_array).map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect());
                Ok(json!(agent.spawn(&task, &purpose, tools, &ctx).await?))
            }
        })
        .tags(["builtin", "delegation", "runtime"])
    }

    /// Write a new specialist for `task`, run it, and hand back its result.
    async fn spawn(&self, task: &str, purpose: &str, tools: Option<Vec<String>>, ctx: &ToolContext) -> Result<String> {
        let inner = &self.inner;
        if !inner.runtime_agents {
            return Ok("run-time agents are disabled for this agent.".into());
        }
        let max = inner.max_runtime_agents;
        // Claimed before anything is awaited, so parallel calls cannot overrun.
        // (`fetch_update` is `try_update` from Rust 1.95; kept for the 1.88 MSRV.)
        #[allow(deprecated)]
        let claimed = inner.spawned.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| (n < max).then_some(n + 1));
        if claimed.is_err() {
            return Ok(format!("No run-time agents left: {max} of {max} already spun up in this run. Finish the work with the tools and sub-agents you have."));
        }
        inner.total_spawned.fetch_add(1, Ordering::SeqCst);
        let mut allowed: Vec<String> = match &inner.runtime_agent_tools {
            Some(t) => t.clone(),
            None => self.tools().iter().filter(|t| !t.tags.iter().any(|g| NOT_INHERITED.contains(&g.as_str()))).map(|t| t.name.clone()).collect(),
        };
        if let Some(wanted) = tools.filter(|w| !w.is_empty()) {
            let narrowed: Vec<String> = allowed.iter().filter(|n| wanted.contains(n)).cloned().collect();
            if !narrowed.is_empty() {
                allowed = narrowed;
            }
        }
        let guard = ctx.run.as_ref().and_then(|s| s.guard.clone());
        let mut spec = self.factory()?.create(task, &allowed, false, guard).await;
        if !purpose.is_empty() {
            spec.description = purpose.into();
        }
        if inner.mode.as_ref().is_some_and(|m| m.sources)
            && let Some(tools) = &mut spec.tools
            && !tools.iter().any(|t| t == "record_source")
        {
            tools.push("record_source".into());
        }
        tracing::info!("{} spun up {} for: {}", inner.name, spec.name, task.chars().take(120).collect::<String>());
        let child = build_agent(&spec, self)?;
        if let Some(m) = &inner.memory {
            m.orchestrator.staffing(&task.chars().take(80).collect::<String>(), &spec.name, false).await?;
        }
        self.add_subagent(child.clone()); // it can be re-used by name now
        self.run_child(&child, task, "", ctx).await
    }

    // ---- handoffs ------------------------------------------------------
    /// Let this agent hand the conversation to another. Two agents that hand
    /// to each other are built first and introduced afterwards.
    pub fn add_handoff(&self, to: impl Into<Handoff>) -> Result<Handoff> {
        let handoff = to.into();
        if handoff.agent.is(self) || handoff.name() == self.inner.name {
            return Err(not_an_agent(&self.inner.name));
        }
        if let Ok(held) = self.tools().get("handoff")
            && !held.tags.contains("handoff")
        {
            return Err(Error::config(format!("{} already has a tool called 'handoff'; rename it to give this agent handoffs", self.inner.name)));
        }
        self.inner.handoffs.write().unwrap_or_else(|p| p.into_inner()).insert(handoff.name().to_string(), handoff.clone());
        self.add_tool(self.handoff_tool());
        Ok(handoff)
    }

    pub fn handoffs(&self) -> IndexMap<String, Handoff> {
        self.inner.handoffs.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The agent called `name` that a conversation starting here can end up
    /// with — this one, or any it can reach by handing off.
    pub fn handoff_agent(&self, name: &str) -> Option<Agent> {
        let mut seen: Vec<Agent> = Vec::new();
        let mut queue: VecDeque<Agent> = VecDeque::from([self.clone()]);
        while let Some(agent) = queue.pop_front() {
            if seen.iter().any(|s| s.is(&agent)) {
                continue;
            }
            if agent.name() == name {
                return Some(agent);
            }
            queue.extend(agent.handoffs().into_values().map(|h| h.agent));
            seen.push(agent);
        }
        None
    }

    fn handoff_tool(&self) -> Tool {
        let handoffs = self.handoffs();
        let mut names: Vec<&String> = handoffs.keys().collect();
        names.sort();
        let catalogue: Vec<String> = handoffs.iter().map(|(n, h)| format!("{n}: {}", h.description)).collect();
        let schema = json!({
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "enum": names, "description": catalogue.join("; ")},
                "reason": {"type": "string", "description": "One line: why this agent, and what the user needs from it."},
            },
            "required": ["agent_name"],
        });
        let me: Weak<AgentInner> = Arc::downgrade(&self.inner);
        Tool::raw(
            "handoff",
            "Hand this conversation to another agent, who takes over and answers the user directly from here on. Use it when what the user needs is that agent's job rather than yours. This is not for getting a task done and reported back to you — once you hand off, you are out of the conversation, so do not answer the question yourself as well.",
            schema,
            move |args, ctx| {
                let me = me.clone();
                async move {
                    let agent = Agent { inner: me.upgrade().ok_or_else(|| Error::config("the agent is gone"))? };
                    let s = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                    Ok(json!(agent.hand_off(&s("agent_name"), &s("reason"), &ctx).await?))
                }
            },
        )
        // "delegation" keeps it off every sub-agent built from this agent's tools.
        .tags(["builtin", "delegation", "handoff"])
    }

    /// Ask for the conversation to go to another agent once this step ends.
    /// A refusal is an error the model reads, so it can answer itself.
    async fn hand_off(&self, agent_name: &str, reason: &str, ctx: &ToolContext) -> Result<String> {
        let refuse = |m: String| Error::tool(m, "handoff");
        let Some(handoff) = self.handoffs().get(agent_name).cloned() else {
            let mut known: Vec<String> = self.handoffs().keys().cloned().collect();
            known.sort();
            let known = if known.is_empty() { "none".into() } else { known.join(", ") };
            return Err(refuse(format!("no agent named '{agent_name}' to hand off to. Available: {known}")));
        };
        let Some(relay) = ctx.run.as_ref().and_then(|s| s.relay.clone()) else {
            return Err(refuse("there is no conversation to hand over here — answer it yourself".into()));
        };
        let target = handoff.agent.clone();
        {
            // Claimed before anything is awaited, so two handoffs asked for in
            // one turn cannot both be granted.
            let mut r = lock(&relay);
            if let Some(p) = &r.pending {
                return Err(refuse(format!("this conversation is already being handed to {}", p.handoff.name())));
            }
            if r.hops.len() as u32 >= r.limit {
                return Err(refuse(format!(
                    "this conversation has already changed hands {} times in this turn, which is the limit ({}). Answer the user yourself.",
                    r.hops.len(),
                    r.limit
                )));
            }
            r.claim(handoff.clone(), reason);
        }
        let record = HandoffRecord {
            source: self.inner.name.clone(),
            target: target.name().into(),
            reason: reason.into(),
            step: ctx.step,
            run_id: ctx.run_id.clone(),
            ts: crate::types::now(),
        };
        let asked = self
            .inner
            .hooks
            .emit(
                "handoff",
                target.name(),
                &ctx.run_id,
                ctx.step,
                data(vec![
                    ("source", json!(self.inner.name)),
                    ("parent", json!(self.inner.name)),
                    ("reason", json!(reason)),
                    ("record", serde_json::to_value(&record).unwrap_or_default()),
                ]),
            )
            .await;
        let allowed = if asked.blocked {
            Err(Error::PermissionDenied { message: asked.reason.clone(), tool: "handoff".into(), reason: asked.reason.clone() })
        } else {
            handoff.announce(&record)
        };
        if let Err(e) = allowed {
            lock(&relay).release();
            return Err(refuse(format!("the handoff to {} was not permitted: {e}", target.name())));
        }
        lock(&relay).confirm(record);
        let why = if reason.is_empty() { String::new() } else { format!(": {reason}") };
        // Read by the agent taking over — it is the last thing in its history.
        Ok(format!("[{} handed this conversation to {}{why}. {} has it now and answers the user directly.]", self.inner.name, target.name(), target.name()))
    }

    // ---- running -------------------------------------------------------
    async fn run_internal(&self, task: Message, options: RunOptions, internal: Internal) -> Result<RunResult> {
        let mut beats = self.clone().converse(task, options, internal, false).await?;
        while let Some(beat) = beats.next().await {
            if let Beat::Done(result) = beat {
                return Ok(*result);
            }
        }
        Err(Error::config("the agent loop produced no result"))
    }

    /// Run to completion and return the result.
    pub async fn run(&self, task: impl Into<Message>) -> Result<RunResult> {
        self.run_with(task, RunOptions::default()).await
    }

    pub async fn run_with(&self, task: impl Into<Message>, options: RunOptions) -> Result<RunResult> {
        let mut beats = self.clone().converse(task.into(), options, Internal::default(), false).await?;
        while let Some(beat) = beats.next().await {
            if let Beat::Done(result) = beat {
                return Ok(*result);
            }
        }
        Err(Error::config("the agent loop produced no result"))
    }

    /// Carry on a run that stopped to wait for a person.
    ///
    /// The run continues from the step it stopped in: the approved call is run
    /// with the arguments that were approved, a declined one is answered with
    /// the reviewer's reason, and tools that had already run are not run again.
    /// It may stop again, at a later call — `result.approval` is then the next.
    ///
    /// Fails with `Error::Approval` if there is no such approval, it has not
    /// been answered yet, or it was already resumed. An approval is resumed once.
    pub async fn resume_approval(&self, approval_id: &str) -> Result<RunResult> {
        let mut beats = self.resumed(approval_id, false).await?;
        while let Some(beat) = beats.next().await {
            if let Beat::Done(result) = beat {
                return Ok(*result);
            }
        }
        Err(Error::config("the agent loop produced no result"))
    }

    /// `resume_approval`, as a stream of events.
    pub async fn stream_approval(&self, approval_id: &str) -> Result<BoxStream<'static, StreamEvent>> {
        let beats = self.resumed(approval_id, true).await?;
        Ok(beats
            .map(|beat| match beat {
                Beat::Event(e) => e,
                Beat::Done(result) => StreamEvent { text: result.output.clone(), ..ev(StreamEventType::RunEnd, &result.agent, result.steps) }
                    .with_data("result", serde_json::to_value(&*result).unwrap_or_default()),
            })
            .boxed())
    }

    async fn resumed(&self, approval_id: &str, token_stream: bool) -> Result<BoxStream<'static, Beat>> {
        let Some(desk) = self.inner.harness.approvals.clone() else {
            return Err(Error::config(format!(
                "{}: this harness keeps no approvals — give the Harness that the run was paused on an approvals desk (`with_approvals`)",
                self.inner.name
            )));
        };
        let user = self.inner.trace.as_ref().and_then(|t| t.user_id.clone());
        let tenant = self.inner.trace.as_ref().and_then(|t| t.tenant_id.clone());
        let record = desk.claim(approval_id, &self.inner.name, user.as_deref(), tenant.as_deref(), Some(&self.inner.name)).await?;
        let session = if record.session_id.is_empty() { None } else { self.inner.harness.sessions.load(&record.session_id).await.ok() };
        let notebook = record.notebook.as_ref().map(|n| Arc::new(Notebook::restore(n)));
        let options = RunOptions {
            session: session.map(|s| SessionRef::Session(Box::new(s))),
            messages: Some(record.messages.clone()),
            run_id: Some(record.run_id.clone()),
            ..Default::default()
        };
        let id = record.id.clone();
        let task = Message::user(record.task.clone());
        let internal = Internal { notebook, resume: Some(record), ..Default::default() };
        let mut inner = match self.clone().drive(task, options, internal, token_stream).await {
            Ok(s) => s,
            Err(e) => {
                // The approved tool may have run. It is not tried again by itself.
                let _ = desk.finish(&id, detail(vec![("error", json!(format!("{}: {e}", e.kind_name()).chars().take(500).collect::<String>()))]), true).await;
                return Err(e);
            }
        };
        Ok(Box::pin(async_stream::stream! {
            let mut finished = false;
            while let Some(beat) = inner.next().await {
                if let Beat::Done(r) = &beat {
                    let next = r.approval.as_ref().and_then(|a| a["id"].as_str()).unwrap_or("").to_string();
                    let outcome = detail(vec![
                        ("stop_reason", json!(r.stop_reason.as_str())),
                        ("output", json!(r.output.chars().take(500).collect::<String>())),
                        ("error", json!(r.error)),
                        ("session_id", json!(r.session_id)),
                        ("next", json!(next)),
                    ]);
                    let _ = desk.finish(&id, outcome, r.error.is_some()).await;
                    finished = true;
                }
                yield beat;
            }
            if !finished {
                let _ = desk.finish(&id, detail(vec![("error", json!("no result"))]), true).await;
            }
        }))
    }

    /// Token-by-token events, ending with a `run_end` event whose
    /// `data["result"]` is the [`RunResult`] as JSON.
    pub async fn stream(&self, task: impl Into<Message>, options: RunOptions) -> Result<BoxStream<'static, StreamEvent>> {
        let beats = self.clone().converse(task.into(), options, Internal::default(), true).await?;
        Ok(beats
            .map(|beat| match beat {
                Beat::Event(e) => e,
                Beat::Done(result) => StreamEvent { text: result.output.clone(), ..ev(StreamEventType::RunEnd, &result.agent, result.steps) }
                    .with_data("result", serde_json::to_value(&*result).unwrap_or_default()),
            })
            .boxed())
    }

    /// Expose this agent as a tool another agent can call.
    pub fn as_tool(&self, name: Option<&str>, description: Option<&str>) -> Tool {
        let agent = self.clone();
        let schema = json!({"type": "object", "properties": {"task": {"type": "string", "description": "The task, stated in full."}}, "required": ["task"]});
        Tool::raw(name.unwrap_or(&self.inner.name), description.unwrap_or(&self.inner.description), schema, move |args, _ctx| {
            let agent = agent.clone();
            async move {
                let task = args.get("task").and_then(Value::as_str).unwrap_or("").to_string();
                let result = agent.run_with(task, RunOptions { messages: Some(vec![]), ..Default::default() }).await?;
                Ok(json!(match result.error {
                    Some(e) => format!("[failed] {e}"),
                    None => result.output,
                }))
            }
        })
        .tags(["agent"])
    }

    /// Run one of this agent's tools under all of its rails, outside a run.
    /// Never fails: a refusal or a failure is an outcome with `is_error` set.
    pub async fn call_tool(&self, name: &str, args: Map<String, Value>) -> ToolOutcome {
        let call = ToolUseBlock::new(name, args);
        let guard = self.inner.harness.guard();
        let ctx = ToolContext { agent: self.inner.name.clone(), ..Default::default() };
        match self.inner.run_tool(&self.tools(), &call, &ctx, &guard, "", 0).await {
            Ok(o) => o,
            Err(e) => ToolOutcome { call_id: call.id, name: name.into(), is_error: true, content: format!("Not run: {e}"), ..Default::default() },
        }
    }

    fn threaded(&self, opts: &RunOptions) -> bool {
        (self.conversational() || self.inner.following.load(Ordering::SeqCst)) && opts.messages.is_none()
    }

    async fn session_for(&self, session: &Option<SessionRef>, threaded: bool) -> Result<Session> {
        match session {
            Some(SessionRef::Session(s)) => Ok((**s).clone()),
            Some(SessionRef::Id(id)) => self.inner.harness.sessions.load(id).await,
            None if threaded => Ok(lock(&self.inner.thread).clone().unwrap_or_else(|| Session::new(&self.inner.name))),
            None => Ok(match &self.inner.memory {
                // A run with no session adds to the memory session's record.
                Some(m) => {
                    let id = m.session().id;
                    match lock(&self.inner.log).clone().filter(|l| l.id == id) {
                        Some(log) => log,
                        None => Session { id, ..Session::new(&self.inner.name) },
                    }
                }
                None => Session::new(&self.inner.name),
            }),
        }
    }

    /// One run, as the caller sees it — however many agents it takes.
    ///
    /// An agent with nowhere to hand off to is simply driven. Otherwise the
    /// conversation goes to whoever has it, and each time it is handed on the
    /// next agent is driven on the same history and session. The caller still
    /// sees one `run_start`, one `run_end`, and a `handoff` event between.
    async fn converse(self, task: Message, opts: RunOptions, base: Internal, token_stream: bool) -> Result<BoxStream<'static, Beat>> {
        if self.handoffs().is_empty() {
            return self.drive(task, opts, base, token_stream).await;
        }
        let threaded = self.threaded(&opts);
        let one_off = opts.messages.is_some() && opts.session.is_none();
        let relay = Arc::new(Mutex::new(Relay {
            entry: self.inner.name.clone(),
            limit: self.inner.max_handoffs,
            persist: self.inner.persist_session,
            keeper: self.inner.name.clone(),
            ..Default::default()
        }));
        let mut agent = self.clone();
        let mut call = opts.clone();
        if opts.messages.is_none() && (threaded || opts.session.is_some()) {
            // A conversation being continued: who was it left with?
            let held = self.session_for(&opts.session, threaded).await?;
            if opts.session.is_some() {
                call.session = Some(SessionRef::Session(Box::new(held.clone()))); // loaded once
            }
            let active = active_holder(&held).unwrap_or_else(|| self.inner.name.clone());
            let holder = if active == self.inner.name { Some(self.clone()) } else { self.handoff_agent(&active) };
            lock(&relay).session = Some(held.clone());
            match holder {
                // Gone from the configuration: the agent that was asked takes it back.
                None => tracing::info!("{active} had this conversation but cannot be given it; {} takes it back", self.inner.name),
                Some(h) if !h.is(&self) => {
                    lock(&relay).keeper = h.name().to_string();
                    call = RunOptions {
                        session: Some(SessionRef::Session(Box::new(held.clone()))),
                        messages: Some(close_open_tool_calls(held.messages.clone())),
                        attachments: opts.attachments.clone(),
                        run_id: opts.run_id.clone(),
                        guard: opts.guard.clone(),
                        ..Default::default()
                    };
                    agent = h;
                }
                Some(_) => {}
            }
        }
        let task_text = task.text();
        Ok(async_stream::stream! {
            let mut agent = agent;
            let mut call = call;
            let mut segments: Vec<RunResult> = Vec::new();
            let mut final_result: Option<RunResult>;
            loop {
                final_result = None;
                let message = if segments.is_empty() { task.clone() } else { Message::user(task_text.clone()) };
                let leg = agent.clone().drive(message, call.clone(), Internal { relay: Some(relay.clone()), notebook: base.notebook.clone(), resume: None, parent_span: base.parent_span.clone(), parent_run_id: base.parent_run_id.clone() }, token_stream).await;
                let mut leg = match leg {
                    Ok(l) => l,
                    Err(e) => {
                        let result = RunResult { agent: agent.name().into(), error: Some(format!("{}: {e}", e.kind_name())), stop_reason: StopReason::Error, ..Default::default() };
                        final_result = Some(result);
                        break;
                    }
                };
                while let Some(beat) = leg.next().await {
                    match beat {
                        Beat::Done(r) => final_result = Some(*r),
                        Beat::Event(e) if e.kind != StreamEventType::RunStart || segments.is_empty() => yield Beat::Event(e),
                        Beat::Event(_) => {}
                    }
                }
                let Some(done) = final_result.take() else { break };
                let pending = lock(&relay).pending.take();
                let Some(pending) = pending else {
                    final_result = Some(done);
                    break;
                };
                let target = pending.handoff.agent.clone();
                yield Beat::Event(StreamEvent { text: target.name().into(), ..ev(StreamEventType::Handoff, agent.name(), done.steps) }
                    .with_data("from", json!(agent.name()))
                    .with_data("to", json!(target.name()))
                    .with_data("reason", json!(pending.reason)));
                let (view, rewrote) = pending.handoff.view(&done.messages, &agent, &pending.reason);
                let session = {
                    let mut r = lock(&relay);
                    r.rewrote = rewrote;
                    r.takeover = true;
                    r.session.clone()
                };
                segments.push(done);
                call = RunOptions {
                    messages: Some(view),
                    session: if one_off { None } else { session.map(|s| SessionRef::Session(Box::new(s))) },
                    guard: opts.guard.clone(),
                    ..Default::default()
                };
                agent = target;
            }
            let mut result = final_result.unwrap_or_default();
            for earlier in segments.into_iter().rev() {
                result.usage = earlier.usage.clone() + result.usage.clone();
                result.steps += earlier.steps;
                result.tool_calls.splice(0..0, earlier.tool_calls);
                result.children.splice(0..0, earlier.children);
                let have: Vec<(String, Option<String>)> = result.artifacts.iter().map(|a| (a.name.clone(), a.path.clone())).collect();
                let older: Vec<_> = earlier.artifacts.into_iter().filter(|a| !have.contains(&(a.name.clone(), a.path.clone()))).collect();
                result.artifacts.splice(0..0, older);
                result.warnings.splice(0..0, earlier.warnings);
            }
            let (hops, keeper, session) = {
                let r = lock(&relay);
                (r.hops.clone(), r.keeper.clone(), r.session.clone())
            };
            result.handoffs = hops;
            result.active_agent = if keeper.is_empty() { result.agent.clone() } else { keeper };
            if threaded && let Some(s) = session {
                *lock(&self.inner.thread) = Some(s);
            }
            yield Beat::Done(Box::new(result));
        }
        .boxed())
    }

    /// Set up one agent's run (the parts that can fail before it starts), then
    /// hand back the stream that drives it.
    async fn drive(self, task: Message, opts: RunOptions, internal: Internal, token_stream: bool) -> Result<BoxStream<'static, Beat>> {
        let one_off = opts.messages.is_some() && opts.session.is_none();
        let threaded = self.threaded(&opts);
        let session_obj = if one_off { Session::new(&self.inner.name) } else { self.session_for(&opts.session, threaded).await? };
        Ok(self.drive_loop(task, opts, internal, session_obj, threaded, token_stream).boxed())
    }

    fn drive_loop(
        self,
        task: Message,
        opts: RunOptions,
        internal: Internal,
        mut session_obj: Session,
        threaded: bool,
        token_stream: bool,
    ) -> impl futures::Stream<Item = Beat> + Send + 'static {
        async_stream::stream! {
            let me = self.inner.clone();
            let harness = me.harness.clone();
            let relay = internal.relay.clone();
            let run_id = opts.run_id.clone().unwrap_or_else(|| new_id("run"));
            let guard = opts.guard.clone().unwrap_or_else(|| match &me.budget {
                Some(b) => harness.guard().child(b.clone()),
                None => harness.guard(),
            });
            let steps_allowed = opts.max_steps.unwrap_or(me.max_steps);
            let model = opts.model.clone().unwrap_or_else(|| me.model.clone());
            let one_off = opts.messages.is_some() && opts.session.is_none();
            let continuing = threaded || opts.session.is_some();
            let takeover = relay.as_ref().is_some_and(|r| lock(r).takeover);
            if let Some(r) = &relay {
                lock(r).attach(&session_obj);
            }

            let task_text = task.text();
            let attached: Vec<ContentBlock> = task.content.iter().filter(|b| b.as_media().is_some()).cloned().chain(opts.attachments.iter().cloned()).collect();
            let mut result = RunResult { agent: me.name.clone(), run_id: run_id.clone(), session_id: session_obj.id.clone(), ..Default::default() };

            let mut history: Vec<Message> = match &opts.messages {
                Some(m) => m.clone(),
                None if continuing => close_open_tool_calls(session_obj.messages.clone()),
                None => vec![],
            };
            let mut earlier: Vec<Message> = if continuing { vec![] } else { session_obj.messages.clone() };
            if let Some(r) = &relay {
                // Every agent in a chain adds to the one record the first found.
                let mut r = lock(r);
                if takeover { earlier = r.earlier.clone() } else { r.earlier = earlier.clone() }
            }
            let began_at = history.len();
            let mut rewritten = takeover && relay.as_ref().is_some_and(|r| lock(r).rewrote);

            let resuming = internal.resume.clone();
            if let Some(r) = &resuming {
                if session_obj.version == r.session_version {
                    // The session holds the paused run at its end; what came
                    // before that is kept as it is.
                    let held = session_obj.messages.clone();
                    earlier = held[..held.len().saturating_sub(history.len())].to_vec();
                } else if opts.session.is_some() {
                    // The conversation went on while this waited. The paused
                    // run is finished beside it, not written over it.
                    let moved = session_obj.id.clone();
                    session_obj = session_obj.fork(Some(0));
                    earlier.clear();
                    result.session_id = session_obj.id.clone();
                    result.warnings.push(format!("session {moved} moved on while this run waited for approval; the run is kept as {}", session_obj.id));
                }
                if me.budget.is_some() {
                    // What it spent before it stopped still counts against its ceiling.
                    let _ = guard.record(&r.usage, &me.name, "");
                }
            }
            // A run can stop and wait for a person only where it can be picked up
            // again: at the top of a conversation, not inside a sub-agent's task
            // or a chain of handoffs.
            let pausable = harness.approvals.is_some() && relay.is_none() && (!one_off || resuming.is_some());
            let mut paused: Option<Approval> = None;
            let memory = me.memory.clone();
            let content = me.content();
            // What a takeover continues was checked by the agent it came to.
            let mut tripped: Option<Error> = None;
            let task_text = if takeover || resuming.is_some() {
                task_text
            } else {
                match content.check(&task_text, "input", "task") {
                    Ok(t) => t,
                    Err(e) => {
                        tripped = Some(e);
                        task_text
                    }
                }
            };
            let mut task_message = Message::user(task_text.clone());
            let task_index = history.len();
            if !takeover && resuming.is_none() {
                history.push(task_message.clone());
                if let Some(m) = &memory {
                    m.with_session(|s| s.add_message(task_message.clone()));
                }
            }
            if opts.messages.is_none() || opts.guard.is_none() {
                me.spawned.store(0, Ordering::SeqCst); // a fresh run gets a fresh agent budget
            }

            let trace = me.trace.as_ref().map(|t| serde_json::to_value(t).unwrap_or_default()).unwrap_or(Value::Null);
            let parent = internal.parent_run_id.clone().map_or(Value::Null, Value::String);
            let started = me
                .hooks
                .emit("run_start", &me.name, &run_id, 0, data(vec![("task", json!(task_text)), ("model", json!(model)), ("trace", trace), ("parent_run_id", parent)]))
                .await;
            harness.control.enter(&me.name, &run_id);
            let mut run_span = harness.tracer.start(&format!("agent:{}", me.name), "run", internal.parent_span.as_ref());
            run_span.set("task", task_text.chars().take(120).collect::<String>());
            run_span.set("model", model.clone());
            result.trace_id = run_span.trace_id.clone();
            harness.audit.record(&me.name, "run_start", &task_text.chars().take(120).collect::<String>(), "ok", &run_id, detail(vec![("model", json!(model))]));
            harness.journal.assignment(&me.name, &task_text.chars().take(500).collect::<String>(), &run_id, &result.trace_id);
            yield Beat::Event(ev(StreamEventType::RunStart, &me.name, 0).with_data("task", json!(task_text)).with_data("run_id", json!(run_id)));

            let contract = me.output.as_ref().map(|o| format!("{CONTRACT}{}", serde_json::to_string_pretty(&o.schema).unwrap_or_default())).unwrap_or_default();
            let mut last_text = String::new();
            let mut final_text = String::new();
            let mut retries_left = me.contract_retries;
            let profile = me.mode.clone();
            if let Some(p) = &profile {
                result.mode = p.name.clone();
                result.depth = p.depth.clone();
            }
            let notebook: Option<Arc<Notebook>> = match (&profile, internal.notebook.clone()) {
                (_, Some(n)) => Some(n),
                (Some(p), None) if p.keeps_notebook() => {
                    // Taking the conversation over is taking the work over:
                    // the same todo list, the same numbered sources.
                    let carried = if takeover { relay.as_ref().and_then(|r| lock(r).notebook.clone()) } else { None };
                    Some(carried.filter(|c| c.ledger == p.sources).unwrap_or_else(|| Arc::new(Notebook::new(p.sources, p.sources && p.verify_sources))))
                }
                _ => None,
            };
            if let (Some(r), Some(n)) = (&relay, &notebook) {
                lock(r).notebook = Some(n.clone());
            }
            if let Some(n) = &notebook {
                n.saw(&[&task_text]);
            }
            let mut reported = notebook.as_ref().map_or(0, |n| n.revision());
            let mut mode_retries = profile.as_ref().map_or(0, |p| p.retries);
            let mut guard_retries = me.guardrails.as_ref().map_or(0, |g| g.max_retries);
            let mut files_before: Option<Snapshot> = None;
            let scope = Arc::new(RunScope {
                guard: Some(guard.clone()),
                children: Mutex::new(vec![]),
                relay: relay.clone(),
                notebook: notebook.clone(),
                span: Some(run_span.clone()),
                approval: resuming.clone(),
                pausable,
            });

            let outcome: Result<()> = 'run: {
                if let Some(e) = tripped.take() {
                    break 'run Err(e);
                }
                if started.blocked {
                    break 'run Err(Error::PermissionDenied { message: format!("{} may not run: {}", me.name, started.reason), tool: String::new(), reason: started.reason.clone() });
                }
                if let Some(p) = profile.as_ref().filter(|p| p.sources) {
                    // Its own notebook, memory and delegation are not ways of finding anything out.
                    let reads = !me.subagents.read().unwrap_or_else(|p| p.into_inner()).is_empty()
                        || me.tools.read().unwrap_or_else(|p| p.into_inner()).iter().any(|t| !t.tags.iter().any(|g| matches!(g.as_str(), "mode" | "memory" | "delegation")));
                    if !reads {
                        break 'run Err(Error::config(format!("{} is in {} mode but has nothing to read with — give it a search, fetch or lookup tool, a sub-agent, or an MCP server", me.name, p.name)));
                    }
                }
                if !attached.is_empty() && !takeover {
                    let mut ready = tri!('run, prepare_attachments(&attached, me.provider.as_ref(), &model, harness.speech.as_ref()).await);
                    // Text made from a file passes the same input checks as anything else coming in.
                    for block in ready.iter_mut() {
                        if let ContentBlock::Text(t) = block {
                            t.text = tri!('run, content.check(&t.text, "input", "attachment"));
                        }
                    }
                    task_message.content.splice(0..0, ready);
                    history[task_index] = task_message.clone();
                }
                if (resuming.is_some() && !one_off) || (resuming.is_none() && opts.messages.is_none()) {
                    // Back into the sandbox it was working in, if it had one.
                    let notice = tri!('run, me.rejoin(&session_obj, opts.sandbox_id.as_deref()).await);
                    if !notice.is_empty() && resuming.is_none() && !takeover {
                        // Said in the conversation itself, so it is still there
                        // the next time this history is read.
                        if let Some(ContentBlock::Text(t)) = task_message.content.iter_mut().rev().find(|b| matches!(b, ContentBlock::Text(_))) {
                            t.text = format!("{notice}\n\n{task_text}");
                        }
                        history[task_index] = task_message.clone();
                    }
                }
                if let (Some(p), Some(ws)) = (&profile, &me.workspace)
                    && p.collect_files
                {
                    // Also what starts a sandbox — so one that cannot start ends
                    // the run here, before a model call is paid for.
                    match ws.snapshot().await {
                        Ok(s) => files_before = Some(s),
                        Err(e) => break 'run Err(Error::config(format!("{}'s workspace is not usable: {e}", me.name))),
                    }
                }
                let first_step = resuming.as_ref().map_or(1, |r| r.step.max(1));
                let mut step = first_step - 1;
                let mut finished = false;
                while step < steps_allowed {
                    step += 1;
                    tri!('run, harness.control.check(&format!("{} step {step}", me.name)));
                    tri!('run, guard.step());
                    result.steps = step;
                    yield Beat::Event(ev(StreamEventType::StepStart, &me.name, step));
                    me.hooks.emit("step_start", &me.name, &run_id, step, Map::new()).await;

                    let tools = me.tools.read().unwrap_or_else(|p| p.into_inner()).clone();
                    // Finishing a step that stopped for approval: the model already
                    // asked for these tools, so it is not asked again.
                    let finishing = resuming.is_some() && step == first_step;
                    let (calls, response): (Vec<ToolUseBlock>, Option<ModelResponse>) = if finishing {
                        (history.last().map(|m| m.tool_uses().into_iter().cloned().collect()).unwrap_or_default(), None)
                    } else {
                        let mut pins = memory.as_ref().map(|m| m.session().facts).unwrap_or_default();
                        pins.extend(notebook.as_ref().map(|n| n.pins()).unwrap_or_default());
                        if let Some(compacted) = me.compactor.compact(&history, &pins).await {
                            rewritten = true;
                            history = compacted;
                        }
                        let mut assembler = me.assembler.clone();
                        if let Some(p) = &profile {
                            assembler.mode = p.prompt(&tools.names());
                        }
                        let blocks = match &memory {
                            Some(m) => tri!('run, m.prompt_blocks(&task_text).await),
                            None => vec![],
                        };
                        let mut system = assembler.build_with_memory(&tools.names(), &contract, &blocks);
                        let mut request = me.template.clone();
                        // A mode's last step is for handing over, not for one more tool
                        // call whose result nobody would ever read.
                        if let Some(p) = profile.as_ref().filter(|_| step == steps_allowed && steps_allowed > 1) {
                            system = format!("{system}\n\n{}", p.wrap_up);
                            if !tools.is_empty() {
                                request.tool_choice = Some(json!("none"));
                            }
                        }
                        request.model = model.clone();
                        request.messages = history.clone();
                        request.system = Some(system);
                        request.tools = tools.schemas();

                        let hook = me.hooks.emit("pre_model", &me.name, &run_id, step, data(vec![("request", serde_json::to_value(&request).unwrap_or_default())])).await;
                        if hook.blocked {
                            break 'run Err(Error::StopRequested(hook.reason));
                        }
                        if let Some(r) = hook.replacement.and_then(|v| serde_json::from_value::<CompletionRequest>(v).ok()) {
                            request = r;
                        }

                        let mut response: Option<ModelResponse> = None;
                        let mut calls = me.clone().model_events(request, step, token_stream, run_id.clone(), Some(run_span.clone()));
                        while let Some(item) = calls.next().await {
                            match tri!('run, item) {
                                ModelBeat::Event(e) => yield Beat::Event(e),
                                ModelBeat::Response(r) => response = Some(*r),
                            }
                        }
                        let Some(mut response) = response else {
                            break 'run Err(ProviderError::new(ProviderErrorKind::Other, "no response from the model").provider(me.provider.name()).into());
                        };
                        // Capture what it said *before* charging for it: recording
                        // the usage is what trips a budget.
                        let said = response.text();
                        if !said.is_empty() {
                            last_text = said;
                        }
                        result.usage += &response.usage;
                        tri!('run, guard.record(&response.usage, &me.name, &task_text.chars().take(60).collect::<String>()));

                        let post = me.hooks.emit("post_model", &me.name, &run_id, step, data(vec![("response", serde_json::to_value(&response).unwrap_or_default())])).await;
                        if let Some(r) = post.replacement.and_then(|v| serde_json::from_value::<ModelResponse>(v).ok()) {
                            response = r;
                        }
                        history.push(response.message.clone());
                        if let Some(m) = &memory {
                            m.with_session(|s| s.add_message(response.message.clone()));
                        }
                        if harness.checkpoints.should_save(step) {
                            harness.checkpoints.save(Checkpoint {
                                agent: me.name.clone(),
                                session_id: session_obj.id.clone(),
                                messages: history.clone(),
                                usage: result.usage.clone(),
                                ..Checkpoint::new(&run_id, step)
                            });
                        }

                        let calls: Vec<ToolUseBlock> = response.tool_uses().into_iter().cloned().collect();
                        (calls, Some(response))
                    };
                    if !calls.is_empty() {
                        let ctx = ToolContext { agent: me.name.clone(), run_id: run_id.clone(), session_id: session_obj.id.clone(), step, run: Some(scope.clone()), ..Default::default() };
                        let outcomes = me.execute_tools(&tools, &calls, &ctx, &guard).await;
                        let finished_children: Vec<RunResult> = std::mem::take(&mut *lock(&scope.children));
                        for child in finished_children {
                            result.artifacts.extend(child.artifacts.clone());
                            result.children.push(child);
                        }
                        let outcomes = tri!('run, outcomes);
                        if outcomes.iter().any(|o| o.pending.is_some()) {
                            // Someone has to say yes first, and nobody is here to ask.
                            // The run stops at this step and is written down;
                            // `resume_approval` finishes the step.
                            paused = Some(Approval {
                                run_id: run_id.clone(),
                                agent: me.name.clone(),
                                user_id: session_obj.user_id.clone(),
                                tenant_id: session_obj.tenant_id.clone(),
                                task: task_text.clone(),
                                model: model.clone(),
                                step,
                                calls: outcomes
                                    .iter()
                                    .filter_map(|o| {
                                        let p = o.pending.as_ref()?;
                                        Some(ApprovalCall {
                                            id: o.call_id.clone(),
                                            tool: o.name.clone(),
                                            args: p["args"].as_object().cloned().unwrap_or_default(),
                                            reason: p["reason"].as_str().unwrap_or("").to_string(),
                                            status: "pending".into(),
                                            by: String::new(),
                                            note: String::new(),
                                            decided: None,
                                        })
                                    })
                                    .collect(),
                                done: outcomes
                                    .iter()
                                    .filter(|o| o.pending.is_none())
                                    .map(|o| json!({"id": o.call_id, "name": o.name, "content": o.content, "is_error": o.is_error, "media": o.media}))
                                    .collect(),
                                messages: history.clone(),
                                notebook: notebook.as_ref().map(|n| n.snapshot()),
                                ..Approval::default()
                            });
                            result.stop_reason = StopReason::Approval;
                            yield Beat::Event(ev(StreamEventType::StepEnd, &me.name, step));
                            finished = true;
                            break;
                        }
                        // What the run has now read — the evidence a recorded source is checked against.
                        if let Some(n) = notebook.as_ref().filter(|n| n.tracking) {
                            for (call, outcome) in calls.iter().zip(&outcomes) {
                                let reads = tools.get(&call.name).is_ok_and(|t| !t.tags.contains("mode"));
                                if !outcome.is_error && reads {
                                    n.saw(&[&call.name, &Value::Object(call.input.clone()).to_string(), &outcome.content]);
                                }
                            }
                        }
                        for (call, outcome) in calls.iter().zip(&outcomes) {
                            result.tool_calls.push(ToolCall { id: call.id.clone(), name: call.name.clone(), args: call.input.clone(), agent: me.name.clone(), step });
                            let mut e = StreamEvent { text: outcome.content.chars().take(400).collect(), ..ev(StreamEventType::ToolResult, &me.name, step) }
                                .with_data("tool", json!(outcome.name))
                                .with_data("error", json!(outcome.is_error));
                            if !outcome.media.is_empty() {
                                e = e.with_data("media", serde_json::to_value(&outcome.media).unwrap_or_default());
                            }
                            yield Beat::Event(e);
                        }
                        let mut tool_message = Message::tool_results(outcomes.iter().map(ToolOutcome::as_block).collect());
                        let seen: Vec<ContentBlock> = outcomes.iter().flat_map(|o| o.media.iter().cloned()).collect();
                        let had_media = !seen.is_empty();
                        tool_message.content.extend(seen);
                        if let Some(m) = &memory {
                            m.with_session(|s| s.add_message(tool_message.clone()));
                        }
                        history.push(tool_message);
                        if had_media {
                            drop_old_tool_media(&mut history, me.tool_images_kept);
                        }
                        if let Some(n) = notebook.as_ref().filter(|n| n.revision() != reported) {
                            reported = n.revision();
                            let mut e = StreamEvent { text: n.todo_summary(), ..ev(StreamEventType::Progress, &me.name, step) };
                            if let Value::Object(d) = n.progress() {
                                e.data = d;
                            }
                            yield Beat::Event(e);
                        }
                        yield Beat::Event(ev(StreamEventType::StepEnd, &me.name, step));
                        if relay.as_ref().is_some_and(|r| lock(r).pending.is_some()) {
                            // The conversation is someone else's now: this agent
                            // gives no answer, so nothing that judges one applies.
                            result.stop_reason = StopReason::Handoff;
                            finished = true;
                            break;
                        }
                        continue;
                    }

                    // No tool calls — this is the answer.
                    let Some(response) = response else {
                        break 'run Err(Error::Approval("the step that stopped for approval has no tool calls to finish".into()));
                    };
                    final_text = response.text();
                    if let Some(output) = &me.output {
                        let checked = match extract_json(&final_text) {
                            None => Err("no JSON object found in the reply".to_string()),
                            Some(v) => (output.validate)(&v).map(|_| v),
                        };
                        match checked {
                            Ok(v) => result.data = v,
                            Err(problem) if retries_left > 0 => {
                                retries_left -= 1;
                                history.push(Message::user(format!("That did not satisfy the output contract ({problem}). Reply with only the JSON object.")));
                                yield Beat::Event(ev(StreamEventType::StepEnd, &me.name, step));
                                continue;
                            }
                            Err(problem) => break 'run Err(Error::OutputContract(problem)),
                        }
                    }
                    // What the mode requires: nothing left open, every citation
                    // leading to a recorded source. Sent back to be finished; if it
                    // still is not, delivered with `violations` saying what is missing.
                    let unmet = profile.as_ref().map(|p| p.unmet(&final_text, notebook.as_deref())).unwrap_or_default();
                    if !unmet.is_empty() && mode_retries > 0 && step < steps_allowed {
                        mode_retries -= 1;
                        history.push(Message::user(Mode::feedback(&unmet)));
                        yield Beat::Event(ev(StreamEventType::StepEnd, &me.name, step));
                        continue;
                    }
                    result.violations = unmet;
                    if let Some(rails) = me.guardrails.as_ref().filter(|g| g.has_checks()) {
                        let checked = rails.evaluate(&CompletionContext::from_run(me.name.clone(), final_text.clone(), &result)).await;
                        if !checked.passed() {
                            result.violations.extend(checked.lines());
                            let joined = result.violations.join("; ");
                            harness.journal.write("guardrail", &joined, &me.name, &run_id, &result.trace_id, Map::new());
                            let unmet: Vec<&str> = checked.violations.iter().map(|v| v.check.as_str()).collect();
                            harness.audit.record(&me.name, "guardrail", "completion", rails.on_violation.as_str(), &run_id, detail(vec![("unmet", json!(unmet))]));
                            if rails.on_violation == OnViolation::Retry && guard_retries > 0 {
                                guard_retries -= 1;
                                history.push(Message::user(AgentGuardrails::feedback(&checked.violations)));
                                yield Beat::Event(ev(StreamEventType::StepEnd, &me.name, step));
                                continue;
                            }
                            if rails.on_violation != OnViolation::Warn {
                                break 'run Err(Error::GuardrailTripped {
                                    message: format!("the answer did not meet this agent's guardrails: {joined}"),
                                    rule: checked.violations[0].check.clone(),
                                    place: "completion".into(),
                                });
                            }
                        }
                    }
                    result.stop_reason = response.stop_reason;
                    yield Beat::Event(ev(StreamEventType::StepEnd, &me.name, step));
                    finished = true;
                    break;
                }
                if !finished {
                    break 'run Err(Error::MaxStepsExceeded(format!("{} did not finish within {steps_allowed} steps", me.name)));
                }
                if result.stop_reason == StopReason::Approval {
                    let waiting = paused.as_ref().map(Approval::describe).unwrap_or_default();
                    let said = if last_text.is_empty() { format!("Waiting for approval: {waiting}.") } else { last_text.clone() };
                    result.output = tri!('run, content.check(&said, "output", &me.name));
                } else if result.stop_reason != StopReason::Handoff {
                    if let (Some(p), None) = (&profile, &me.output) {
                        final_text = p.close(&final_text, notebook.as_deref());
                    }
                    final_text = tri!('run, content.check(&final_text, "output", &me.name));
                    result.output = final_text.clone();
                }
                Ok(())
            };

            match outcome {
                Ok(()) => {}
                Err(Error::BudgetExceeded { message, kind, limit, spent }) => {
                    // A budget is a ceiling, not a failure: by default the run
                    // ends cleanly with what it produced and a note saying why.
                    if guard.budget.on_exceed == OnExceed::Stop {
                        let mut partial = [&final_text, &last_text, &result.output].into_iter().find(|t| !t.is_empty()).cloned().unwrap_or_default();
                        if let (Some(p), None) = (&profile, &me.output) {
                            partial = p.close(&partial, notebook.as_deref());
                        }
                        let partial = content.check(&partial, "output", &me.name).unwrap_or_default();
                        result.output = format!("{partial}{}", budget_note(&kind, limit, spent)).trim().to_string();
                        result.stop_reason = StopReason::Budget;
                        result.budget_exceeded = Some(if kind.is_empty() { "budget".into() } else { kind.clone() });
                        yield Beat::Event(ev(StreamEventType::StepEnd, &me.name, result.steps).with_data("budget_exceeded", json!(kind)));
                    } else {
                        let e = Error::BudgetExceeded { message, kind, limit, spent };
                        result.error = Some(format!("{}: {e}", e.kind_name()));
                        result.stop_reason = StopReason::Error;
                        if result.output.is_empty() {
                            result.output = if final_text.is_empty() { last_text.clone() } else { final_text.clone() };
                        }
                        me.hooks.emit("error", &me.name, &run_id, result.steps, data(vec![("error", json!(e.to_string()))])).await;
                        yield Beat::Event(StreamEvent { text: result.error.clone().unwrap_or_default(), ..ev(StreamEventType::Error, &me.name, result.steps) });
                    }
                }
                Err(e) => {
                    result.error = Some(format!("{}: {e}", e.kind_name()));
                    result.stop_reason = if matches!(e, Error::StopRequested(_)) { StopReason::Stopped } else { StopReason::Error };
                    if result.output.is_empty() {
                        result.output = if final_text.is_empty() { last_text.clone() } else { final_text.clone() };
                    }
                    me.hooks.emit("error", &me.name, &run_id, result.steps, data(vec![("error", json!(e.to_string()))])).await;
                    yield Beat::Event(StreamEvent { text: result.error.clone().unwrap_or_default(), ..ev(StreamEventType::Error, &me.name, result.steps) });
                }
            }

            let persist = match &relay {
                Some(r) => {
                    let mut r = lock(r);
                    if result.stop_reason != StopReason::Handoff {
                        r.cancel(); // asked for, but the run did not get there
                    }
                    r.stamp(&mut session_obj, &me.name, result.error.is_some() && result.steps == 0);
                    r.persist
                }
                None => me.persist_session,
            };
            if let Some(n) = &notebook {
                result.todos = n.todos();
                result.sources = n.sources();
            }
            let mut produced: Vec<crate::types::Artifact> = Vec::new();
            if let Some(p) = &profile {
                // A report is only worth keeping if the run finished one; files
                // were written whether it finished or not.
                let finished = result.error.is_none() && result.budget_exceeded.is_none();
                let have: Vec<(String, Option<String>)> = result.artifacts.iter().map(|a| (a.name.clone(), a.path.clone())).collect();
                produced = p.artifacts(if finished { &result.output } else { "" }, &me.name);
                if let (true, Some(ws), Some(before)) = (p.collect_files, &me.workspace, &files_before) {
                    match crate::modes::collect_files(ws.as_ref(), before, &me.name).await {
                        Ok(files) => produced.extend(files),
                        // A sandbox that died must not take the answer down with it.
                        Err(e) => {
                            harness.journal.write("error", &format!("could not collect the files this run wrote: {e}"), &me.name, &run_id, &result.trace_id, Map::new());
                        }
                    }
                }
                result.artifacts.extend(produced.iter().filter(|a| !have.contains(&(a.name.clone(), a.path.clone()))).cloned());
            }
            if opts.messages.is_none() {
                me.remember_sandbox(&mut session_obj, &mut result, &produced).await;
            }
            if let Some(m) = &memory {
                let mine = m.session().artifacts;
                result.artifacts.extend(mine.iter().cloned());
                if !mine.is_empty() {
                    // Several agents write to this session; each keeps the rest.
                    let names: Vec<&str> = mine.iter().map(|a| a.name.as_str()).collect();
                    let mut kept: Vec<_> = if relay.is_some() { session_obj.artifacts.iter().filter(|a| !names.contains(&a.name.as_str())).cloned().collect() } else { vec![] };
                    kept.extend(mine.iter().cloned());
                    session_obj.artifacts = kept;
                }
            }
            result.messages = history.clone();
            session_obj.messages = [earlier, history.clone()].concat();
            session_obj.usage += &result.usage;
            if let Some(r) = &resuming {
                // The whole run, not only the part after the pause.
                let mut total = r.usage.clone();
                total += &result.usage;
                result.usage = total;
            }
            if session_obj.title.is_empty() {
                session_obj.title = task_text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(80).collect();
            }
            if persist && !one_off {
                let added = history.get(began_at..).unwrap_or(&[]).to_vec();
                me.save_session(&mut session_obj, &mut result, &added, rewritten).await;
            }
            if let Some(r) = &relay {
                lock(r).session = Some(session_obj.clone());
            }
            if threaded {
                *lock(&me.thread) = Some(session_obj.clone());
            } else if opts.session.is_none() && !one_off {
                *lock(&me.log) = Some(session_obj.clone());
            }
            if let (Some(mut held), true) = (paused.take(), result.stop_reason == StopReason::Approval) {
                // Stored after the session, so the record knows which version of
                // the conversation it belongs to.
                let stored = persist && !one_off;
                held.session_id = if stored { session_obj.id.clone() } else { String::new() };
                held.session_version = if stored { session_obj.version } else { 0 };
                held.usage = result.usage.clone();
                let desk = harness.approvals.clone().expect("pausable only with a desk");
                match desk.open(held).await {
                    Ok(opened) => {
                        harness.journal.write("decision", &format!("paused for approval {}: {}", opened.id, opened.describe()), &me.name, &run_id, &result.trace_id, Map::new());
                        result.approval = Some(opened.summary());
                        yield Beat::Event(
                            StreamEvent { text: opened.describe(), ..ev(StreamEventType::ApprovalRequired, &me.name, result.steps) }.with_data("approval", opened.summary()),
                        );
                    }
                    Err(e) => {
                        result.error = Some(format!("the run could not be paused for approval: {e}"));
                        result.stop_reason = StopReason::Error;
                    }
                }
            }
            if !result.artifacts.is_empty() {
                harness.deliverables.extend(&result.artifacts, &run_id);
            }
            let said = if result.output.is_empty() { result.error.clone().unwrap_or_default() } else { result.output.clone() };
            harness.journal.write(
                "handback",
                &said.chars().take(500).collect::<String>(),
                &me.name,
                &run_id,
                &result.trace_id,
                detail(vec![("cost_usd", json!(result.cost_usd())), ("steps", json!(result.steps))]),
            );
            let ended = if result.error.is_some() { "error" } else { "ok" };
            harness.audit.record(&me.name, "run_end", result.stop_reason.as_str(), ended, &run_id, detail(vec![("cost_usd", json!(result.cost_usd()))]));
            run_span.set("cost_usd", result.cost_usd());
            run_span.set("steps", result.steps);
            if let Some(e) = &result.error {
                run_span.fail(e.clone());
            }
            harness.tracer.finish(run_span);
            harness.control.leave(&run_id);
            let ended = me.hooks.emit("run_end", &me.name, &run_id, result.steps, data(vec![("result", serde_json::to_value(&result).unwrap_or_default())])).await;
            // A handler may hand back the result changed — governance withholds,
            // redacts or labels the output, and adds its disclosure and provenance.
            if let Some(changed) = ended.replacement.and_then(|v| serde_json::from_value::<RunResult>(v).ok()) {
                result = changed;
            }
            yield Beat::Done(Box::new(result));
        }
    }
}

/// One model call's output: events, then the response.
enum ModelBeat {
    Event(StreamEvent),
    Response(Box<ModelResponse>),
}

impl AgentInner {
    /// Put this agent back in the sandbox its conversation was using. Returns
    /// what the model has to be told when that was not possible: it is in a
    /// new sandbox, and which of its files came back.
    async fn rejoin(&self, session: &Session, sandbox_id: Option<&str>) -> Result<String> {
        let Some(ws) = self.workspace.as_ref().and_then(|w| w.as_sandbox()) else {
            if let Some(id) = sandbox_id {
                return Err(Error::config(format!("{}: sandbox_id='{id}' was given, but this agent's workspace is not a sandbox", self.name)));
            }
            return Ok(String::new());
        };
        let held = session.metadata.get("sandbox").and_then(Value::as_object).cloned().unwrap_or_default();
        if held.is_empty() && sandbox_id.is_none() {
            return Ok(String::new()); // a conversation with no sandbox behind it yet
        }
        let sb = &ws.sandbox;
        // What was asked for now beats what the sandbox was built with, which
        // beats what the conversation remembers.
        let elsewhere = !held.is_empty() && held.get("sandbox").and_then(Value::as_str) != Some(sb.name());
        let remembered = if elsewhere { String::new() } else { held.get("id").and_then(Value::as_str).unwrap_or("").to_string() };
        let wanted = sandbox_id.map(str::to_string).or(sb.attach_id().await).unwrap_or(remembered);
        if sb.running().await {
            let id = sb.id().await;
            if !wanted.is_empty() && id != wanted {
                return Err(Error::config(format!(
                    "{} is already working in {} sandbox {id}, but this conversation's files are in {wanted} — build an agent for each conversation that has its own sandbox",
                    self.name,
                    sb.name()
                )));
            }
            return Ok(String::new());
        }
        if !wanted.is_empty() && sb.attach_id().await.as_deref() != Some(wanted.as_str()) {
            sb.attach(&wanted).await?;
        }
        ws.start().await.map_err(|e| Error::config(format!("{}'s sandbox could not be picked back up: {e}", self.name)))?;
        let (replaced, reason) = sb.replaced().await;
        let gone = if !replaced.is_empty() {
            replaced
        } else if elsewhere {
            held.get("id").and_then(Value::as_str).unwrap_or("").to_string()
        } else {
            String::new()
        };
        if gone.is_empty() {
            return Ok(String::new());
        }
        // A new sandbox: put back what the conversation is known to have written.
        let (mut restored, mut lost) = (Vec::new(), Vec::new());
        for (name, content) in session.metadata.get("sandbox_files").and_then(Value::as_object).cloned().unwrap_or_default() {
            match content.as_str() {
                None => lost.push(name),
                Some(text) => {
                    if !ws.exists(&name).await {
                        match ws.write(&name, text.as_bytes(), false).await {
                            Ok(_) => restored.push(name),
                            Err(_) => lost.push(name),
                        }
                    }
                }
            }
        }
        let id = sb.id().await;
        let detail: Map<String, Value> = [("was", json!(gone)), ("restored", json!(restored.len())), ("lost", json!(lost.len())), ("reason", json!(reason))]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        self.harness.audit.record(&self.name, "sandbox_replaced", &id, "ok", "", detail);
        self.harness.journal.write(
            "decision",
            &format!("sandbox {gone} was gone; continuing in {id} with {} files restored", restored.len()),
            &self.name,
            "",
            "",
            Map::new(),
        );
        let mut lines =
            vec![format!("[Workspace notice: the sandbox this conversation was working in ({gone}) is no longer available, so you are in a new one.")];
        if !restored.is_empty() {
            lines.push(format!("Put back from the conversation's record: {}.", restored.join(", ")));
        }
        if !lost.is_empty() {
            lines.push(format!("Could not be put back (not text, or too large): {}.", lost.join(", ")));
        }
        lines.push("Anything else that was installed, built or written there is gone — check before you rely on it.]".into());
        Ok(lines.join(" "))
    }

    /// Note on the session which sandbox it is in and what it wrote there, so
    /// the conversation can be picked up — in it, or without it.
    async fn remember_sandbox(&self, session: &mut Session, result: &mut RunResult, produced: &[crate::types::Artifact]) {
        let Some(ws) = self.workspace.as_ref().and_then(|w| w.as_sandbox()) else { return };
        if !ws.sandbox.running().await {
            return;
        }
        result.sandbox_id = ws.sandbox.id().await;
        session.metadata.insert("sandbox".into(), json!({"sandbox": ws.sandbox.name(), "id": result.sandbox_id, "workdir": ws.root().await}));
        let report = self.mode.as_ref().map(|m| m.report.clone()).unwrap_or_default();
        let mut files: Map<String, Value> = session.metadata.get("sandbox_files").and_then(Value::as_object).cloned().unwrap_or_default();
        for a in produced.iter().filter(|a| a.produced_by == self.name && a.name != report) {
            files.insert(a.name.clone(), if a.content.is_empty() { Value::Null } else { json!(a.content) });
        }
        if !files.is_empty() {
            if let Ok(present) = ws.snapshot().await {
                files.retain(|name, _| present.contains_key(name));
            }
            // The record is for getting a conversation going again, not a backup:
            // past the cap, the largest files are remembered by name only.
            let mut budget: usize = 2_000_000;
            let mut order: Vec<(String, usize)> = files.iter().map(|(n, v)| (n.clone(), v.as_str().map_or(0, |s| s.chars().count()))).collect();
            order.sort_by_key(|(_, size)| *size);
            for (name, size) in order {
                if size > budget {
                    files.insert(name, Value::Null);
                } else {
                    budget -= size;
                }
            }
        }
        session.metadata.insert("sandbox_files".into(), Value::Object(files));
    }

    /// The text rules for this agent: its own if it has them, else the harness's.
    fn content(&self) -> Guardrails {
        self.guardrails.as_ref().and_then(|g| g.content.clone()).unwrap_or_else(|| self.harness.guardrails.clone())
    }

    /// The provider for a fallback model, cached per agent.
    fn provider_for(&self, model: &str) -> Result<Arc<dyn Provider>> {
        if model == self.model || self.explicit_provider {
            return Ok(self.provider.clone()); // an explicit backend serves everything
        }
        let mut cache = lock(&self.fallback_providers);
        if let Some(p) = cache.get(model) {
            return Ok(p.clone());
        }
        let p = resolve_provider(None, Some(model))?;
        cache.insert(model.into(), p.clone());
        Ok(p)
    }

    /// One model call, walking the fallback chain. Streams tokens when asked.
    fn model_events(
        self: Arc<Self>,
        request: CompletionRequest,
        step: u32,
        token_stream: bool,
        run_id: String,
        parent: Option<Span>,
    ) -> BoxStream<'static, Result<ModelBeat>> {
        Box::pin(async_stream::try_stream! {
            let harness = self.harness.clone();
            let estimated = request.messages.iter().map(|m| m.text().chars().count() as u64).sum::<u64>() / 4;
            harness.rate().acquire(estimated).await;
            let chain = harness.router.chain(&request.model);
            let mut refused: Vec<String> = Vec::new();
            for (attempt, model) in chain.iter().enumerate() {
                let last = attempt + 1 == chain.len();
                let provider = self.provider_for(model)?;
                let mut req = request.clone();
                req.model = model.clone();
                // Fired per backend actually tried, so a handler sees where the
                // data is really going. A refusal moves on down the chain.
                let egress = self.hooks.emit("model_egress", &self.name, &run_id, step, data(vec![
                    ("request", serde_json::to_value(&req).unwrap_or_default()),
                    ("provider", json!(provider.name())),
                    ("model", json!(model)),
                ])).await;
                if egress.blocked {
                    refused.push(format!("{model}: {}", egress.reason));
                    if last {
                        Err(Error::PermissionDenied {
                            message: format!("no model in the chain may receive this request — {}", refused.join("; ")),
                            tool: model.clone(),
                            reason: egress.reason.clone(),
                        })?;
                    }
                    continue;
                }
                if let Some(r) = egress.replacement.and_then(|v| serde_json::from_value::<CompletionRequest>(v).ok()) {
                    req = r;
                }
                let started = std::time::Instant::now();
                let outcome = Self::one_call(&self.name, provider.as_ref(), &req, step, token_stream).await;
                let ms = started.elapsed().as_secs_f64() * 1000.0;
                let failed = outcome.as_ref().err().map(|e| format!("{}: {e}", e.kind_name()));
                harness.health.record(provider.name(), "model", ms, failed.is_none(), failed.as_deref().unwrap_or(""));
                let mut span = harness.tracer.start(&format!("model:{model}"), "model", parent.as_ref());
                span.set("step", step);
                match &outcome {
                    Ok((_, r)) => {
                        span.set("cost_usd", r.usage.cost_usd);
                        span.set("tokens", r.usage.total_tokens());
                    }
                    Err(_) => span.fail(failed.clone().unwrap_or_default()),
                }
                harness.tracer.finish(span);
                match outcome {
                    Ok((events, response)) => {
                        harness.rate().record(&response.usage).await;
                        for e in events {
                            yield ModelBeat::Event(e);
                        }
                        yield ModelBeat::Response(Box::new(response));
                        return;
                    }
                    Err(e) if !last && is_reachability_problem(&e) => {
                        tracing::warn!("{model} unreachable ({e}); trying {}", chain[attempt + 1]);
                    }
                    Err(e) => Err(e)?,
                }
            }
        })
    }

    /// One call to one provider: the events it produced, and its response.
    async fn one_call(
        agent: &str,
        provider: &dyn Provider,
        request: &CompletionRequest,
        step: u32,
        token_stream: bool,
    ) -> Result<(Vec<StreamEvent>, ModelResponse)> {
        if !token_stream {
            return Ok((vec![], provider.complete(request).await?));
        }
        let mut events = Vec::new();
        let mut response = None;
        let mut stream = provider.stream(request);
        while let Some(item) = stream.next().await {
            let e = item?;
            match e.kind {
                StreamEventType::Text | StreamEventType::Thinking => events.push(StreamEvent { text: e.text, ..ev(e.kind, agent, step) }),
                StreamEventType::ToolCall => {
                    let name = e.data.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                    events.push(StreamEvent { text: name, data: e.data, ..ev(StreamEventType::ToolCall, agent, step) });
                }
                StreamEventType::StepEnd => response = response_from_event(&e),
                _ => {}
            }
        }
        let response = response
            .ok_or_else(|| Error::from(ProviderError::new(ProviderErrorKind::Other, "the stream ended without a final response").provider(provider.name())))?;
        Ok((events, response))
    }

    /// Run every tool the model asked for, in parallel, under the rails.
    /// Fails only when a budget or a stop request ends the run.
    async fn execute_tools(&self, tools: &ToolRegistry, calls: &[ToolUseBlock], ctx: &ToolContext, guard: &Arc<BudgetGuard>) -> Result<Vec<ToolOutcome>> {
        let raw: Vec<Result<ToolOutcome>> = if calls.len() == 1 || !self.parallel_tools {
            let mut out = Vec::new();
            for c in calls {
                out.push(self.one_tool(tools, c, ctx, guard).await);
            }
            out
        } else {
            self.harness.scheduler.map(calls.iter(), |c| self.one_tool(tools, c, ctx, guard)).await
        };
        raw.into_iter().collect()
    }

    /// One call — unless it was settled before a pause: run then, or declined since.
    async fn one_tool(&self, tools: &ToolRegistry, call: &ToolUseBlock, ctx: &ToolContext, guard: &Arc<BudgetGuard>) -> Result<ToolOutcome> {
        if let Some(record) = ctx.run.as_ref().and_then(|s| s.approval.as_ref()) {
            if let Some(held) = record.done.iter().find(|d| d["id"].as_str() == Some(call.id.as_str())) {
                // It ran before the pause. It is not run again.
                return Ok(ToolOutcome {
                    call_id: call.id.clone(),
                    name: call.name.clone(),
                    content: held["content"].as_str().unwrap_or("").to_string(),
                    is_error: held["is_error"].as_bool().unwrap_or(false),
                    media: serde_json::from_value(held["media"].clone()).unwrap_or_default(),
                    ..Default::default()
                });
            }
            if let Some(verdict) = record.calls.iter().find(|c| c.id == call.id && c.status != "approved") {
                let why = if verdict.status == "denied" {
                    format!("declined by {}{}", verdict.by, if verdict.note.is_empty() { String::new() } else { format!(": {}", verdict.note) })
                } else {
                    "the request for approval expired without an answer".to_string()
                };
                let audit = detail(vec![("reason", json!(why)), ("approval", json!(record.id))]);
                self.harness.audit.record(&self.name, "tool_call", &call.name, "deny", &ctx.run_id, audit);
                return Ok(ToolOutcome {
                    call_id: call.id.clone(),
                    name: call.name.clone(),
                    is_error: true,
                    content: format!("Not permitted: {why}"),
                    ..Default::default()
                });
            }
        }
        let tracer = &self.harness.tracer;
        let mut span = tracer.start(&format!("tool:{}", call.name), "tool", ctx.run.as_ref().and_then(|s| s.span.as_ref()));
        span.set("step", ctx.step);
        let mut own = ctx.clone();
        own.span = Some(span.clone());
        let outcome = self.run_tool(tools, call, &own, guard, &ctx.run_id, ctx.step).await;
        match &outcome {
            Ok(o) => {
                span.set("error", o.is_error);
                span.set("cached", o.cached);
                if o.is_error {
                    span.status = "error".into();
                }
            }
            Err(e) => span.fail(format!("{}: {e}", e.kind_name())),
        }
        tracer.finish(span);
        outcome
    }

    async fn run_tool(
        &self,
        tools: &ToolRegistry,
        call: &ToolUseBlock,
        ctx: &ToolContext,
        guard: &BudgetGuard,
        run_id: &str,
        step: u32,
    ) -> Result<ToolOutcome> {
        let refuse = |content: String| ToolOutcome { call_id: call.id.clone(), name: call.name.clone(), content, is_error: true, ..Default::default() };
        let entry = match tools.get(&call.name) {
            Ok(t) => t,
            Err(e) => return Ok(refuse(e.to_string())),
        };
        if let Some(rails) = &self.guardrails {
            let (permitted, reason) = rails.tool_allowed(&call.name);
            if !permitted {
                self.harness.audit.record(&self.name, "tool_call", &call.name, "deny", run_id, detail(vec![("reason", json!(reason))]));
                return Ok(refuse(format!("Not permitted: {reason}")));
            }
        }
        let mut tags: Vec<&String> = entry.tags.iter().collect();
        tags.sort();
        let mut args = call.input.clone();
        let hook = self
            .hooks
            .emit(
                "pre_tool",
                &self.name,
                run_id,
                step,
                data(vec![("tool", json!(call.name)), ("args", Value::Object(args.clone())), ("tags", json!(tags)), ("permission", json!(entry.permission))]),
            )
            .await;
        if hook.blocked {
            return Ok(refuse(format!("Blocked: {}", hook.reason)));
        }
        if let Some(Value::Object(replaced)) = hook.replacement {
            args = replaced;
        }
        let harness = &self.harness;
        let scope = ctx.run.as_ref();
        let record = scope.and_then(|s| s.approval.as_ref());
        let grant = record.and_then(|r| r.calls.iter().find(|c| c.id == call.id && c.status == "approved"));
        if let (Some(g), Some(r)) = (grant, record)
            && canonical(&g.args) != canonical(&args)
        {
            // Approved for one thing, about to do another. It does neither.
            let audit = detail(vec![("reason", json!("arguments differ from those approved")), ("approval", json!(r.id))]);
            harness.audit.record(&self.name, "tool_call", &call.name, "deny", run_id, audit);
            return Ok(refuse("Not permitted: this call is not the one that was approved — its arguments have changed.".into()));
        }
        if grant.is_none() && scope.is_some_and(|s| s.pausable) && self.policy.approver.is_none() {
            let (decision, reason) = self.policy.needs(&call.name, &args, Some(entry.permission));
            if decision == Permission::Ask {
                harness.audit.record(
                    &self.name,
                    "tool_call",
                    &call.name,
                    "pending",
                    run_id,
                    detail(vec![("args", Value::Object(args.clone())), ("reason", json!(reason))]),
                );
                let reason = if reason.is_empty() { "needs approval".to_string() } else { reason };
                return Ok(ToolOutcome {
                    call_id: call.id.clone(),
                    name: call.name.clone(),
                    content: "Waiting for approval.".into(),
                    pending: Some(json!({"reason": reason, "args": args})),
                    ..Default::default()
                });
            }
        }
        if let Err(e) = self.policy.check(&call.name, &args, Some(entry.permission), grant.map(|g| g.by.as_str())).await {
            harness.audit.record(&self.name, "tool_call", &call.name, "deny", run_id, detail(vec![("reason", json!(e.to_string()))]));
            harness.journal.write("denied", &e.to_string(), &self.name, run_id, "", detail(vec![("tool", json!(call.name))]));
            return Ok(refuse(format!("Not permitted: {e}")));
        }
        let mut allowed = detail(vec![("args", Value::Object(args.clone()))]);
        if let (Some(g), Some(r)) = (grant, record) {
            allowed.insert("approved_by".into(), json!(g.by));
            allowed.insert("approval".into(), json!(r.id));
        }
        harness.audit.record(&self.name, "tool_call", &call.name, "allow", run_id, allowed);
        let key = if entry.cacheable { harness.cache.key(&[json!("tool"), json!(call.name), Value::Object(args.clone())]) } else { String::new() };
        if !key.is_empty()
            && let Some(hit) = harness.cache.get(&key)
        {
            let content = match hit {
                Value::String(s) => s,
                other => other.to_string(),
            };
            return Ok(ToolOutcome { call_id: call.id.clone(), name: call.name.clone(), content, cached: true, ..Default::default() });
        }
        harness.control.check(&format!("{} tool {}", self.name, call.name))?;
        guard.tool_call()?;
        let started = std::time::Instant::now();
        let mut outcome = entry.run(&call.id, args.clone(), ctx.clone()).await;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        harness.health.record(&call.name, "tool", ms, !outcome.is_error, if outcome.is_error { &outcome.content } else { "" });
        if !outcome.is_error {
            match self.content().check(&outcome.content, "input", &call.name) {
                Ok(text) => outcome.content = text,
                Err(e) => outcome = refuse(format!("Blocked: {e}")),
            }
        }
        let post = self
            .hooks
            .emit(
                "post_tool",
                &self.name,
                run_id,
                step,
                data(vec![("tool", json!(call.name)), ("args", Value::Object(args)), ("outcome", serde_json::to_value(&outcome).unwrap_or_default())]),
            )
            .await;
        if let Some(v) = post.replacement {
            outcome.content = hook_text(&v);
        }
        if !key.is_empty() && !outcome.is_error {
            self.harness.cache.set(&key, json!(outcome.content), None);
        }
        let line: String = outcome.content.chars().take(200).collect();
        self.harness.journal.write(
            "tool",
            &format!("{} → {line}", call.name),
            &self.name,
            run_id,
            "",
            detail(vec![("tool", json!(call.name)), ("error", json!(outcome.is_error))]),
        );
        Ok(outcome)
    }

    /// Save the conversation without losing anyone's turn — or the answer.
    /// A conflict merges this run's messages into what is there now; when they
    /// no longer line up (compaction, a narrowed handoff), the run is forked.
    async fn save_session(&self, session: &mut Session, result: &mut RunResult, added: &[Message], rewritten: bool) {
        let sessions = self.harness.sessions.clone();
        let outcome: Result<()> = async {
            match sessions.save(session).await {
                Ok(()) => return Ok(()),
                Err(Error::SessionConflict { .. }) => {}
                Err(e) => return Err(e),
            }
            if !rewritten {
                for _ in 0..3 {
                    let mut latest = sessions.load(&session.id).await?;
                    latest.messages = [close_open_tool_calls(latest.messages.clone()), added.to_vec()].concat();
                    latest.usage += &result.usage;
                    latest.metadata.extend(session.metadata.clone());
                    if latest.title.is_empty() {
                        latest.title = session.title.clone();
                    }
                    match sessions.save(&mut latest).await {
                        Ok(()) => {
                            *session = latest;
                            return Ok(());
                        }
                        Err(Error::SessionConflict { .. }) => continue,
                        Err(e) => return Err(e),
                    }
                }
            }
            let mut forked = Session {
                id: new_id("ses"),
                parent_id: Some(session.id.clone()),
                version: 0,
                title: if session.title.is_empty() { String::new() } else { format!("{} (fork)", session.title) },
                ..session.clone()
            };
            sessions.save(&mut forked).await?;
            result
                .warnings
                .push(format!("session {} was changed by another request while this run was working; this run is kept as {}", session.id, forked.id));
            result.session_id = forked.id.clone();
            *session = forked;
            Ok(())
        }
        .await;
        if let Err(e) = outcome {
            // The work is done and paid for. A store that is down loses the
            // record of it, not the answer.
            result.warnings.push(format!("the session could not be saved: {}: {e}", e.kind_name()));
        }
    }
}

impl From<&str> for Message {
    fn from(text: &str) -> Self {
        Message::user(text)
    }
}

impl From<String> for Message {
    fn from(text: String) -> Self {
        Message::user(text)
    }
}

impl From<&String> for Message {
    fn from(text: &String) -> Self {
        Message::user(text.clone())
    }
}
