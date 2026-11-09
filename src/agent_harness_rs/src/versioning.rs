//! Agent versions: several configurations of one agent, switchable by name.
//!
//! ```no_run
//! use agent_harness::{Agent, versioning::AgentVersion};
//!
//! # async fn demo() -> agent_harness::Result<()> {
//! let agent = Agent::builder("support")
//!     .version("v2")
//!     .add_version("v1", AgentVersion { instructions: Some("Answer order questions.".into()), ..Default::default() })
//!     .add_version("v2", AgentVersion { instructions: Some("Answer order questions. Cite the order.".into()), ..Default::default() })
//!     .build()?;
//! agent.run("Where is 4182?").await?;                             // v2, the active version
//! agent.use_version("v1")?.run("Where is 4182?").await?;          // the old one, for comparison
//! # Ok(()) }
//! ```
//!
//! A version says what is *different*; everything it leaves unset falls
//! through to how the agent was built. The harness, provider and session
//! store are shared, so switching is cheap and versions are comparable.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::agent::Agent;
use crate::llm_providers::parameters::Effort;
use crate::runtime::Budget;
use crate::subagents::SubAgentSpec;

/// One version of an agent's configuration. Every field is optional.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentVersion {
    pub instructions: Option<String>,
    pub description: Option<String>,
    pub model: Option<String>,
    /// chat, research or cowork. A version that names a mode takes that
    /// mode's depth too; one that names only a depth keeps the mode.
    pub mode: Option<String>,
    pub depth: Option<String>,
    pub tier: Option<String>,
    pub effort: Option<Effort>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub max_steps: Option<u32>,
    pub thinking: Option<bool>,
    pub thinking_budget: Option<i64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub min_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<i64>,
    /// Tool names, selected from the tools the agent was given. `None` keeps
    /// them all; an empty list takes them all away.
    pub tools: Option<Vec<String>>,
    /// Replaces the sub-agents. Not serialised: agents are live objects.
    #[serde(skip)]
    pub subagents: Option<Vec<Agent>>,
    /// Replaces the sub-agents with ones built from blueprints.
    #[serde(skip)]
    pub subagent_specs: Option<Vec<SubAgentSpec>>,
    /// Replaces where this version may hand the conversation.
    #[serde(skip)]
    pub handoffs: Option<Vec<Agent>>,
    pub prompts: Map<String, Value>,
    /// The keyword form of the agent's guardrails (ported in M4).
    pub guardrails: Option<Value>,
    pub budget: Option<Budget>,
    pub runtime_agents: Option<bool>,
    pub max_runtime_agents: Option<u32>,
    pub compact_at: Option<f64>,
    pub contract_retries: Option<u32>,
    pub notes: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_python_dict_form() {
        let v: AgentVersion = serde_json::from_value(serde_json::json!({
            "instructions": "Answer order questions.", "tools": ["order_status"],
            "model": "claude-sonnet-5", "effort": "low", "guardrails": {"require_tools": ["x"]}
        }))
        .unwrap();
        assert_eq!(v.tools.as_deref(), Some(&["order_status".to_string()][..]));
        assert_eq!(v.effort, Some(Effort::Low));
        assert!(v.subagents.is_none());
    }
}
