//! # agent-harness-adk
//!
//! A fast, lightweight harness for building production AI agents in Rust:
//! agents, tools, skills, prompts, model providers, and the runtime rails
//! underneath them.
//!
//! This crate is the Rust port of the Python package of the same name
//! (`pip install agent-harness-adk`). The concepts, names and JSON shapes match,
//! so a session saved by one can be read by the other. See `track.md` for which
//! parts of the Python library have been ported so far.

pub mod a2a;
pub mod agent;
pub mod attachments;
pub mod blueprint;
pub mod browser;
#[cfg(feature = "cli")]
pub mod cli;
pub mod context;
pub mod errors;
pub mod evaluation;
pub mod expr;
pub mod governance;
pub mod guardrails;
pub mod handoff;
pub mod harness;
pub mod knowledge;
pub mod llm_providers;
pub mod mcp;
pub mod memory;
pub mod modes;
pub mod orchestrator;
pub mod prompts;
pub mod pyjson;
pub mod runtime;
pub mod sandboxes;
pub mod sessions;
pub mod skills;
pub mod spec;
pub mod subagents;
pub mod toolkits;
pub mod tools;
pub mod types;
pub mod versioning;
pub mod voice;
pub mod workflow;

pub use agent::{Agent, AgentBuilder, OutputContract, RunOptions, SessionRef};
pub use handoff::{Handoff, History};
pub use harness::Harness;
pub use knowledge::{KnowledgeBase, vector_store};
pub use memory::{MemoryManager, Trace, memory_provider};
pub use prompts::{Prompt, PromptLibrary};
pub use sessions::session_provider;
pub use skills::{Skill, SkillRegistry};
pub use subagents::SubAgentSpec;
pub use tools::{NoArgs, Permission, Tool, ToolContext, ToolRegistry, ToolSchema};
pub use versioning::AgentVersion;

pub use errors::{Error, ProviderError, ProviderErrorKind, Result};
pub use types::{
    Artifact, ContentBlock, MediaBlock, MediaKind, Message, ModelResponse, Role, RunResult, StopReason, StreamEvent, StreamEventType, TextBlock, ToolCall,
    ToolOutcome, ToolResultBlock, ToolUseBlock, Usage, attach, attach_bytes, new_id,
};

pub use llm_providers::base::install_tls;

/// The crate version, as Python's `agent_harness.__version__`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
