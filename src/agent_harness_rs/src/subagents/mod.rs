//! Sub-agents: the bench you reuse, and the factory that writes a new one.
//!
//! Two supplies, one decision. Before staffing a task, ask: is there a
//! pre-defined sub-agent that covers this? If yes it is reused as it is. If
//! no, the factory writes a brand-new specialist during the run, which exists
//! only for this job.

pub mod bench;
pub mod factory;
pub mod spec;

pub use bench::{BENCH_SPECS, Bench};
pub use factory::{FACTORY_PROMPT, SubAgentFactory};
pub use spec::{SubAgentSpec, WorkspaceKind};

/// Turn a blueprint into a live agent that shares its parent's harness.
pub use crate::agent::build_agent;
