//! Handoff: another agent takes over the conversation.
//!
//! ```no_run
//! use agent_harness::Agent;
//! # async fn demo() -> agent_harness::Result<()> {
//! let billing = Agent::builder("billing").instructions("Handle refunds and invoices.").build()?;
//! let triage = Agent::builder("triage").instructions("Work out what the customer needs.").handoff(billing).build()?;
//! let result = triage.run("I was charged twice.").await?;
//! result.agent;      // "billing" — who answered
//! result.handoffs;   // [triage → billing: "a billing question"]
//! # Ok(()) }
//! ```
//!
//! Delegation and a handoff are different things. A sub-agent is given a
//! task, starts clean, and reports back. A handoff gives the *conversation*
//! away: the other agent sees what was said, answers the user itself, and
//! keeps answering on the turns that follow.
//!
//! What the agent taking over sees is the handoff's [`History`]: everything,
//! only what was said, only the user's last message, or your own filter.
//!
//! Two agents that hand off to each other hold each other, so they are never
//! freed; agents normally live as long as the program.

use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::agent::Agent;
use crate::context::close_open_tool_calls;
use crate::errors::{Error, Result};
use crate::runtime::Session;
use crate::types::{ContentBlock, HandoffRecord, Message, Role};

/// How many handoffs a session remembers the story of.
const TRAIL: usize = 20;

/// Messages in, the messages the agent taking over should have.
pub type HistoryFilter = Arc<dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync>;

/// What the agent taking over gets to see.
#[derive(Clone, Default)]
pub enum History {
    /// Everything, tool calls included.
    #[default]
    Full,
    /// What was said, not what was looked up.
    Text,
    /// Only the user's last message.
    Fresh,
    Custom(HistoryFilter),
}

impl std::fmt::Debug for History {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Full => "full",
            Self::Text => "text",
            Self::Fresh => "fresh",
            Self::Custom(_) => "custom",
        })
    }
}

/// Called with the record once a handoff is allowed; `Err` refuses it.
pub type OnHandoff = Arc<dyn Fn(&HandoffRecord) -> Result<()> + Send + Sync>;

/// Somewhere an agent may hand the conversation, and how.
#[derive(Clone)]
pub struct Handoff {
    pub agent: Agent,
    /// When to hand off to it — what the model choosing reads.
    pub description: String,
    pub history: History,
    /// Keep the conversation on the turns that follow (true), or answer this
    /// turn only (false).
    pub sticky: bool,
    pub on_handoff: Option<OnHandoff>,
}

impl std::fmt::Debug for Handoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handoff").field("agent", &self.agent.name()).field("history", &self.history).field("sticky", &self.sticky).finish()
    }
}

impl From<Agent> for Handoff {
    fn from(agent: Agent) -> Self {
        Self::new(agent)
    }
}

impl Handoff {
    pub fn new(agent: Agent) -> Self {
        let description = agent.description().to_string();
        Self { agent, description, history: History::Full, sticky: true, on_handoff: None }
    }

    pub fn describe(mut self, description: &str) -> Self {
        self.description = description.into();
        self
    }

    pub fn history(mut self, history: History) -> Self {
        self.history = history;
        self
    }

    pub fn sticky(mut self, sticky: bool) -> Self {
        self.sticky = sticky;
        self
    }

    pub fn on_handoff(mut self, f: impl Fn(&HandoffRecord) -> Result<()> + Send + Sync + 'static) -> Self {
        self.on_handoff = Some(Arc::new(f));
        self
    }

    pub fn name(&self) -> &str {
        self.agent.name()
    }

    pub(crate) fn announce(&self, record: &HandoffRecord) -> Result<()> {
        match &self.on_handoff {
            Some(f) => f(record),
            None => Ok(()),
        }
    }

    /// The conversation as the agent taking over gets it, and whether it is
    /// no longer simply the messages handed in.
    pub fn view(&self, history: &[Message], source: &Agent, reason: &str) -> (Vec<Message>, bool) {
        let target = &self.agent;
        let mut how = self.history.clone();
        // A model given tool calls and no tools rejects the conversation.
        if matches!(how, History::Full) && target.tools().is_empty() {
            how = History::Text;
        }
        let mut seen = match &how {
            History::Custom(f) => close_open_tool_calls(f(history.to_vec())),
            History::Text => spoken(history),
            History::Fresh => latest_request(history),
            History::Full => history.to_vec(),
        };
        let mut rewrote = !matches!(how, History::Full);
        // A model's reasoning is signed for that model; another one refuses it.
        if target.model() != source.model() && seen.iter().any(|m| m.content.iter().any(|b| matches!(b, ContentBlock::Thinking(_)))) {
            seen = without(&seen, |b| matches!(b, ContentBlock::Thinking(_)));
            rewrote = true;
        }
        // With the tool call gone, nothing says it changed hands — and a model
        // cannot answer a conversation that ends on itself.
        let custom = matches!(how, History::Custom(_));
        if !matches!(how, History::Full) && (!custom || seen.last().is_none_or(|m| m.role != Role::User)) {
            let why = if reason.is_empty() { String::new() } else { format!(": {reason}") };
            seen.push(Message::user(format!(
                "[{} handed this conversation to {}{why}. You are {}: carry on from here and answer the user directly.]",
                source.name(),
                target.name(),
                target.name()
            )));
        }
        (seen, rewrote)
    }
}

/// The same conversation with some kinds of block taken out.
fn without(messages: &[Message], drop: impl Fn(&ContentBlock) -> bool) -> Vec<Message> {
    messages
        .iter()
        .filter_map(|m| {
            let kept: Vec<ContentBlock> = m.content.iter().filter(|b| !drop(b)).cloned().collect();
            if kept.len() == m.content.len() {
                Some(m.clone())
            } else if kept.is_empty() {
                None
            } else {
                Some(Message { content: kept, ..m.clone() })
            }
        })
        .collect()
}

/// What was said: no tool calls, results or reasoning. Neighbouring
/// assistant turns become one.
fn spoken(messages: &[Message]) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::new();
    for m in without(messages, |b| matches!(b, ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_) | ContentBlock::Thinking(_))) {
        match out.last_mut() {
            Some(last) if last.role == Role::Assistant && m.role == Role::Assistant => last.content.extend(m.content),
            _ => out.push(m),
        }
    }
    out
}

/// The last thing the user themselves said.
fn latest_request(messages: &[Message]) -> Vec<Message> {
    messages
        .iter()
        .rev()
        .find(|m| {
            m.role == Role::User
                && !m.content.iter().any(|b| matches!(b, ContentBlock::ToolResult(_)))
                && (!m.media().is_empty() || m.content.iter().any(|b| matches!(b, ContentBlock::Text(t) if !t.text.is_empty())))
        })
        .cloned()
        .into_iter()
        .collect()
}

/// A handoff the model asked for, waiting for the step to finish.
#[derive(Debug, Clone)]
pub(crate) struct Pending {
    pub handoff: Handoff,
    pub reason: String,
    /// Who kept the conversation before this was asked.
    pub keeper: String,
}

/// What travels with a conversation while it changes hands within one run.
#[derive(Debug, Clone, Default)]
pub(crate) struct Relay {
    pub entry: String,
    pub limit: u32,
    pub persist: bool,
    /// Who has the conversation on the next turn.
    pub keeper: String,
    pub hops: Vec<HandoffRecord>,
    pub pending: Option<Pending>,
    /// Handed the conversation mid-run: no new user message to add.
    pub takeover: bool,
    pub rewrote: bool,
    pub session: Option<Session>,
    pub earlier: Vec<Message>,
    pub trail: Option<Vec<Value>>,
    /// The todo list and source ledger of the agent that last had one.
    pub notebook: Option<Arc<crate::modes::Notebook>>,
}

impl Relay {
    pub fn attach(&mut self, session: &Session) {
        self.session = Some(session.clone());
        if self.trail.is_none() {
            let kept = session.metadata.get("handoff").and_then(|h| h.get("trail")).and_then(Value::as_array).cloned().unwrap_or_default();
            self.trail = Some(kept);
        }
    }

    pub fn claim(&mut self, handoff: Handoff, reason: &str) {
        self.pending = Some(Pending { handoff, reason: reason.into(), keeper: self.keeper.clone() });
    }

    pub fn confirm(&mut self, record: HandoffRecord) {
        if self.pending.as_ref().is_some_and(|p| p.handoff.sticky) {
            self.keeper = record.target.clone();
        }
        self.hops.push(record);
    }

    /// A handoff asked for and refused before it was recorded.
    pub fn release(&mut self) {
        self.pending = None;
    }

    /// The run that asked for a handoff failed before it could happen.
    pub fn cancel(&mut self) {
        let Some(p) = self.pending.take() else { return };
        self.keeper = p.keeper;
        if self.hops.last().is_some_and(|h| h.target == p.handoff.name()) {
            self.hops.pop();
        }
    }

    /// Write on the session who has the conversation now, and how it got
    /// there, so the next turn goes to them.
    pub fn stamp(&mut self, session: &mut Session, agent: &str, never_started: bool) {
        if never_started && self.keeper == agent && agent != self.entry {
            // A conversation it could not even begin on would strand every turn after.
            self.keeper = if self.takeover && !self.hops.is_empty() { self.hops[self.hops.len() - 1].source.clone() } else { self.entry.clone() };
        }
        if self.hops.is_empty() && !session.metadata.contains_key("handoff") {
            return;
        }
        let mut trail: Vec<Value> = self.trail.clone().unwrap_or_default();
        trail.extend(self.hops.iter().map(|h| serde_json::to_value(h).unwrap_or_default()));
        let start = trail.len().saturating_sub(TRAIL);
        let active = if self.keeper.is_empty() { agent.to_string() } else { self.keeper.clone() };
        let mut m = Map::new();
        m.insert("active".into(), json!(active));
        m.insert("entry".into(), json!(self.entry));
        m.insert("trail".into(), Value::Array(trail[start..].to_vec()));
        session.metadata.insert("handoff".into(), Value::Object(m));
    }
}

/// Who a session says has the conversation now, if anyone.
pub(crate) fn active_holder(session: &Session) -> Option<String> {
    session.metadata.get("handoff")?.get("active")?.as_str().map(str::to_string).filter(|s| !s.is_empty())
}

pub(crate) fn not_an_agent(name: &str) -> Error {
    Error::config(format!("{name} cannot hand the conversation to itself"))
}
