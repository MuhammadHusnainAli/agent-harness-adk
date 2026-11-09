//! The runtime rails under every run: budgets, permissions, hooks, sessions,
//! and the records kept of it — audit trail, journal, traces, checkpoints,
//! deliverables — with the stop switch, the result cache, service health and
//! approvals that outlive the process.

pub mod approvals;
pub mod audit;
pub mod budget;
pub mod cache;
pub mod checkpoints;
pub mod control;
pub mod deliverables;
pub mod health;
pub mod hooks;
pub mod journal;
pub mod permissions;
pub mod replay;
pub mod router;
pub mod scheduler;
pub mod session;
pub mod tracing;
pub mod workspace;

pub use approvals::{Approval, ApprovalCall, ApprovalStore, Approvals, FileApprovalStore, MemoryApprovalStore, SessionApprovalStore};
pub use audit::{AuditEntry, AuditTrail};
pub use budget::{Budget, BudgetGuard, OnExceed, RateGuard, RateLimit};
pub use cache::ResultCache;
pub use checkpoints::{Checkpoint, Checkpointer};
pub use control::{StopController, StopState};
pub use deliverables::DeliverableStore;
pub use health::{ComponentHealth, ServiceHealth};
pub use hooks::{HookContext, HookEngine};
pub use journal::{JournalEntry, RunJournal};
pub use permissions::{PolicyGate, Rule, always_approve, console_approver};
pub use replay::{RecordingProvider, ReplayProvider, Replayer, request_key};
pub use router::{ModelRouter, RouteRule};
pub use scheduler::ConcurrencyScheduler;
pub use session::{DurableBackend, FileSessionStore, InMemorySessionStore, ListFilter, Session, SessionStore};
pub use tracing::{Span, Tracer};
pub use workspace::{DockerWorkspace, LocalWorkspace, Workspace, WorkspaceBroker, WorkspaceOptions, workspace_tools};
