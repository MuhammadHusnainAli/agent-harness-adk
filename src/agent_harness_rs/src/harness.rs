//! The harness core: the shared services every agent in a run depends on.
//!
//! One `Harness` is created per application (or per run) and handed to every
//! agent, so spend, permissions, hooks and sessions are consistent across the
//! whole tree instead of being re-invented per agent.
//!
//! Besides the provider, router, hooks, policy, sessions and budget it holds
//! the records kept of every run — the audit trail, the journal, the tracer,
//! checkpoints, deliverables — and the stop switch, the result cache, service
//! health and the approvals desk. Scheduler, workspaces and governance arrive
//! with their milestones (see `track.md`).

use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use crate::attachments::Transcriber;
use crate::errors::Result;
use crate::llm_providers::{FakeProvider, Provider};
use crate::runtime::approvals::{Approvals, FileApprovalStore, SessionApprovalStore};
use crate::runtime::router::ModelRouter;
use crate::runtime::{
    AuditTrail, Budget, BudgetGuard, Checkpointer, DeliverableStore, FileSessionStore, HookEngine, InMemorySessionStore, PolicyGate, RateGuard, RateLimit,
    Replayer, ResultCache, RunJournal, ServiceHealth, SessionStore, StopController, StopState, Tracer,
};

/// Shared runtime services. Everything has a working default.
#[derive(Clone)]
pub struct Harness {
    /// The provider agents use when they name none.
    pub provider: Option<Arc<dyn Provider>>,
    /// Picks the model (and effort) for an agent that names none, and the
    /// fallback chain when a model cannot be reached.
    pub router: ModelRouter,
    pub hooks: HookEngine,
    pub policy: PolicyGate,
    pub sessions: Arc<dyn SessionStore>,
    /// Where every agent's memory is kept, unless it is given its own.
    pub memory_store: Arc<dyn crate::memory::MemoryStore>,
    pub budget: Budget,
    /// Content guardrails every agent's input and output pass through, unless
    /// the agent has its own.
    pub guardrails: crate::guardrails::Guardrails,
    pub tracer: Tracer,
    pub journal: RunJournal,
    pub cache: ResultCache,
    /// Snapshots of a run at its steps; off unless `every` is set (`local()` sets 1).
    pub checkpoints: Checkpointer,
    pub deliverables: DeliverableStore,
    pub audit: AuditTrail,
    pub health: ServiceHealth,
    pub control: StopController,
    /// Where a run that needs a person's approval is kept until it is given.
    /// `None`: a tool that needs asking, with nobody to ask, is refused.
    pub approvals: Option<Approvals>,
    /// Caps how much runs at once: parallel tool calls and the sub-agents they start.
    pub scheduler: crate::runtime::ConcurrencyScheduler,
    /// Hands out workspaces — a folder each, or a sandbox each — and closes them.
    pub workspaces: crate::runtime::WorkspaceBroker,
    /// Policy, identity, data, residency, oversight and evidence — attached
    /// with [`Harness::with_governance`].
    pub governance: Option<crate::governance::Governance>,
    pub rate_limit: RateLimit,
    /// Turns a recording into text, for a model that cannot listen.
    pub speech: Option<Arc<dyn Transcriber>>,
    guard: Arc<Mutex<Option<Arc<BudgetGuard>>>>,
    rate: Arc<Mutex<Option<Arc<RateGuard>>>>,
}

impl Default for Harness {
    fn default() -> Self {
        Self {
            provider: None,
            router: ModelRouter::default(),
            hooks: HookEngine::new(),
            policy: PolicyGate::default(),
            sessions: Arc::new(InMemorySessionStore::new()),
            memory_store: Arc::new(crate::memory::InMemoryStore::new()),
            budget: Budget::default(),
            guardrails: crate::guardrails::Guardrails::default(),
            tracer: Tracer::new(),
            journal: RunJournal::new(),
            cache: ResultCache::new(),
            checkpoints: Checkpointer::new(0),
            deliverables: DeliverableStore::new(),
            audit: AuditTrail::new(),
            health: ServiceHealth::new(),
            control: StopController::new(),
            approvals: None,
            scheduler: crate::runtime::ConcurrencyScheduler::default(),
            workspaces: crate::runtime::WorkspaceBroker::default(),
            governance: None,
            rate_limit: RateLimit::default(),
            speech: None,
            guard: Arc::default(),
            rate: Arc::default(),
        }
    }
}

impl std::fmt::Debug for Harness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Harness").field("provider", &self.provider.as_ref().map(|p| p.name().to_string())).field("budget", &self.budget).finish_non_exhaustive()
    }
}

impl Harness {
    pub fn new() -> Self {
        Self::default()
    }

    /// No disk, no network: a scripted provider (or the one you pass).
    pub fn testing(provider: Option<Arc<dyn Provider>>) -> Self {
        Self { provider: Some(provider.unwrap_or_else(|| Arc::new(FakeProvider::new(vec![])))), ..Self::default() }
    }

    /// Everything persisted under one directory: `<root>/sessions` and
    /// `<root>/memory` — the usual production shape on one machine.
    pub fn local(root: impl AsRef<Path>) -> Result<Self> {
        let base = root.as_ref();
        let sessions = FileSessionStore::new(base.join("sessions"))?;
        let memory = crate::memory::FileStore::new(base.join("memory"))?;
        let audit = AuditTrail::at(base.join("audit.jsonl"));
        let mut approvals = Approvals::new(Arc::new(FileApprovalStore::new(base.join("approvals"))?));
        approvals.audit = Some(audit.clone());
        Ok(Self {
            sessions: Arc::new(sessions),
            memory_store: Arc::new(memory),
            tracer: Tracer::new().jsonl(base.join("traces.jsonl")),
            journal: RunJournal::at(base.join("journal.jsonl")),
            checkpoints: Checkpointer::at(base.join("checkpoints"), 1),
            cache: ResultCache::at(base.join("cache")),
            deliverables: DeliverableStore::at(base.join("deliverables")),
            workspaces: crate::runtime::WorkspaceBroker::new(Some(base.join("workspaces"))),
            audit,
            approvals: Some(approvals),
            ..Self::default()
        })
    }

    /// Memory and chat sessions in one database, on one connection pool.
    ///
    /// ```no_run
    /// # use agent_harness::Harness;
    /// let harness = Harness::on("postgresql://user:pass@host/agents");
    /// let harness = Harness::on("mongodb://host:27017/agents");
    /// ```
    ///
    /// The usual shape for a service with more than one replica: nothing that
    /// has to be shared lives on a local disk.
    pub fn on(url: &str) -> Result<Self> {
        let store = crate::memory::memory_provider(url)?;
        let sessions = crate::sessions::session_store_on(&store, &serde_json::Map::new())?;
        // A run paused for approval is kept beside the chats, in the same database.
        let h = Self { sessions: sessions.clone(), memory_store: store, ..Self::default() };
        Ok(h.with_approvals(Approvals::new(Arc::new(SessionApprovalStore::new(sessions)))))
    }

    /// Govern every agent on this harness: the governance's signer goes onto the
    /// audit trail, its hooks onto the harness, and each agent registers its
    /// identity when it is built.
    pub fn with_governance(mut self, governance: crate::governance::Governance) -> Result<Self> {
        governance.attach(&mut self)?;
        self.governance = Some(governance);
        Ok(self)
    }

    /// Keep runs that stop for a person here, and record each decision in this
    /// harness's audit trail.
    pub fn with_approvals(mut self, mut approvals: Approvals) -> Self {
        if approvals.audit.is_none() {
            approvals.audit = Some(self.audit.clone());
        }
        self.approvals = Some(approvals);
        self
    }

    /// Approvals kept beside this harness's sessions — any database it keeps chats in.
    pub fn with_session_approvals(self) -> Self {
        let store = SessionApprovalStore::new(self.sessions.clone());
        self.with_approvals(Approvals::new(Arc::new(store)))
    }

    pub fn with_guardrails(mut self, guardrails: crate::guardrails::Guardrails) -> Self {
        self.guardrails = guardrails;
        self
    }

    pub fn with_tracer(mut self, tracer: Tracer) -> Self {
        self.tracer = tracer;
        self
    }

    pub fn with_checkpoints(mut self, checkpoints: Checkpointer) -> Self {
        self.checkpoints = checkpoints;
        self
    }

    /// Stop every sandbox this harness started and remove what was temporary.
    pub async fn close(&self) {
        self.workspaces.close().await;
    }

    /// Re-run a recorded run, whole or from a step.
    pub fn replayer(&self) -> Replayer {
        Replayer::new(self.checkpoints.clone())
    }

    /// Stop every run on this harness at its next safe boundary. `mode` is
    /// `drain` (in-flight work hands back what it has) or `abort`.
    pub fn stop(&self, reason: &str, mode: &str, requested_by: &str) -> StopState {
        let state = self.control.stop(reason, mode, requested_by);
        let detail = serde_json::json!({"reason": reason}).as_object().cloned().unwrap_or_default();
        self.audit.record(if requested_by.is_empty() { "human" } else { requested_by }, "stop", "", &state.mode, "", detail);
        state
    }

    /// Keep memory here: a store, or `memory_provider(url)?`.
    pub fn with_memory_store(mut self, store: Arc<dyn crate::memory::MemoryStore>) -> Self {
        self.memory_store = store;
        self
    }

    pub fn with_provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn with_router(mut self, router: ModelRouter) -> Self {
        self.router = router;
        self
    }

    pub fn with_sessions(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.sessions = store;
        self
    }

    pub fn with_hooks(mut self, hooks: HookEngine) -> Self {
        self.hooks = hooks;
        self
    }

    pub fn with_policy(mut self, policy: PolicyGate) -> Self {
        self.policy = policy;
        self
    }

    pub fn with_budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        *self.guard.lock().unwrap_or_else(|p| p.into_inner()) = None;
        self
    }

    pub fn with_rate_limit(mut self, limit: RateLimit) -> Self {
        self.rate_limit = limit;
        *self.rate.lock().unwrap_or_else(|p| p.into_inner()) = None;
        self
    }

    pub fn with_speech(mut self, speech: Arc<dyn Transcriber>) -> Self {
        self.speech = Some(speech);
        self
    }

    /// The run-wide budget guard. Sub-agents get children of this.
    pub fn guard(&self) -> Arc<BudgetGuard> {
        self.guard.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(|| Arc::new(BudgetGuard::new(self.budget.clone()))).clone()
    }

    pub fn reset_budget(&self, budget: Option<Budget>) -> Arc<BudgetGuard> {
        let fresh = Arc::new(BudgetGuard::new(budget.unwrap_or_else(|| self.budget.clone())));
        *self.guard.lock().unwrap_or_else(|p| p.into_inner()) = Some(fresh.clone());
        fresh
    }

    /// Throughput limiter shared by every agent on this harness.
    pub fn rate(&self) -> Arc<RateGuard> {
        self.rate.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(|| Arc::new(RateGuard::new(self.rate_limit.clone()))).clone()
    }

    /// One glance at the run: spend and throughput.
    pub async fn report(&self) -> Value {
        json!({
            "budget": self.guard().report(), "rate": self.rate().stats().await,
            "health": self.health.snapshot(Some((self.scheduler.running(), self.scheduler.queued(), self.scheduler.max_concurrency()))),
            "scheduler": self.scheduler.stats(), "cache": self.cache.stats(),
            "control": self.control.report(), "audit_entries": self.audit.len(),
            "governance": self.governance.as_ref().map(|g| g.summary()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Usage;

    #[test]
    fn guard_is_shared_until_reset() {
        let h = Harness::testing(None).with_budget(Budget { max_steps: Some(3), ..Default::default() });
        let g = h.guard();
        g.record(&Usage { input_tokens: 5, ..Default::default() }, "a", "").unwrap();
        assert_eq!(h.clone().guard().usage().input_tokens, 5);
        assert_eq!(h.reset_budget(None).usage().input_tokens, 0);
        assert_eq!(h.guard().budget.max_steps, Some(3));
        assert_eq!(h.provider.as_ref().unwrap().name(), "fake");
    }
}
