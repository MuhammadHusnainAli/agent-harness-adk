//! [`Governance`]: one object that turns the controls on.
//!
//! ```no_run
//! # use agent_harness::Harness;
//! # use agent_harness::governance::{Governance, GovernanceOptions, Policy};
//! # fn main() -> agent_harness::Result<()> {
//! let gov = Governance::from_packs(&["eu-ai-act", "gdpr", "ksa-pdpl", "owasp-agentic"], GovernanceOptions {
//!     policy: Some(Policy::load("governance.yaml")?),
//!     home: Some("eu".into()),
//!     signing_key: Some(std::env::var("AUDIT_KEY").unwrap_or_default().into_bytes()),
//!     ..Default::default()
//! })?;
//! let mut harness = Harness::local(".harness")?;
//! gov.attach(&mut harness)?;
//! # Ok(()) }
//! ```
//!
//! From then on every run on that harness is governed. It works entirely through
//! the hook points the loop already has — `run_start`, `model_egress`,
//! `post_model`, `pre_tool`, `post_tool`, `subagent_start`, `handoff`,
//! `memory_write`, `run_end` — so it adds nothing to a harness that does not use
//! it, and it sees exactly what the loop does: the real provider and model of
//! each call, the final arguments of each tool call, each hand-off to a
//! sub-agent.
//!
//! Every decision is recorded in the audit trail with the policy hash that made
//! it. `Mode::Monitor` makes the same decisions and records them without
//! enforcing any — the way to roll governance out on a live system.
//!
//! What the hooks read (beyond what the loop always sends): `run_start` may
//! carry `trace` (the agent's [`Trace`], for the person the run acts for) and
//! `parent_run_id` (the run that started it); without `parent_run_id` a run
//! started right after a `subagent_start`/`handoff` for the same agent is taken
//! as that delegation's child. `model_egress` may give `provider` as an object
//! with `name`, `base_url` and `region`. `run_end` answers with a replacement:
//! the result, with its `output` governed and `disclosure`, `provenance` and
//! `governance` added.

use std::collections::{BTreeSet, HashMap};
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::{Arc, Mutex, Weak};

use futures::FutureExt;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use super::data::{Classification, DataClassifier, Pseudonymizer, expand_classes, is_personal, known_classes};
use super::evidence::ComplianceReport;
use super::identity::{AgentIdentity, DelegationChain, Principal};
use super::incidents::{Incident, IncidentDesk, IncidentRecord, Notifier};
use super::inventory::{AIInventory, AgentRecord, LiveTools};
use super::monitor::{RunState, RuntimeMonitor, SharedRun, lock, taint};
use super::oversight::{ApprovalNotify, ApprovalRecord, ApprovalRequest, Approver, OversightDesk};
use super::packs::{Pack, load_pack};
use super::policy::{Decision, Policy, PolicyEngine, ResidencyConfig};
use super::records::{HMACSigner, RetentionSweeper, SubjectVault};
use super::residency::{ProviderInfo, Region, RegionResolver};
use super::rights::SubjectRights;
use super::risk::{RiskAssessment, assess, impact_assessment};
use super::transparency::{ProvenanceManifest, disclosure, jurisdiction_language, label};
use super::{obj, py_str_repr};
use crate::errors::{Error, Result};
use crate::harness::Harness;
use crate::llm_providers::base::CompletionRequest;
use crate::memory::{MemoryStore, Trace};
use crate::runtime::audit::{AuditTrail, Scrub, Signer};
use crate::runtime::hooks::HookContext;
use crate::runtime::{DeliverableStore, RunJournal, SessionStore, Tracer};
use crate::tools::ToolRegistry;
use crate::types::{ContentBlock, ModelResponse, now};

/// `Enforce` decides and enforces; `Monitor` decides and records, enforcing nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Mode {
    #[default]
    Enforce,
    Monitor,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Enforce => "enforce",
            Self::Monitor => "monitor",
        }
    }

    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "enforce" => Ok(Self::Enforce),
            "monitor" => Ok(Self::Monitor),
            other => Err(Error::config(format!("mode must be 'enforce' or 'monitor', not {}", py_str_repr(other)))),
        }
    }
}

/// How tool arguments enter the audit trail: `Scrubbed` replaces the personal
/// values the classifier can find; `Tokens` replaces *every* string argument
/// with the person's token — for when arguments may hold names or free text no
/// pattern can recognise.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AuditArgs {
    #[default]
    Scrubbed,
    Tokens,
}

/// Everything [`Governance`] can be given. All of it has a working default.
pub struct GovernanceOptions {
    /// Your own policy; packs are merged on top of it.
    pub policy: Option<Policy>,
    /// Packs already loaded (in addition to any named to `from_packs`).
    pub packs: Vec<Pack>,
    pub mode: Mode,
    /// Whose data a run with no `jurisdiction` tag concerns.
    pub home: Option<String>,
    /// Region overrides for the resolver: provider name, host or `model:<glob>` → jurisdiction.
    pub regions: IndexMap<String, String>,
    pub signer: Option<Arc<dyn Signer>>,
    /// An HMAC key, when no `signer` is given.
    pub signing_key: Option<Vec<u8>>,
    pub approver: Option<Approver>,
    /// Seconds a request waits for people before it expires (a no).
    pub approval_timeout: f64,
    pub approval_notify: Option<ApprovalNotify>,
    /// Where the subject vault is kept. In memory without one.
    pub vault: Option<std::path::PathBuf>,
    /// Where inventory pins are kept.
    pub inventory: Option<std::path::PathBuf>,
    pub classifier: Option<DataClassifier>,
    pub monitor: Option<RuntimeMonitor>,
    /// Told about every incident.
    pub notify: Vec<Notifier>,
    /// The deploying organisation, named in provenance manifests.
    pub service_provider: String,
    pub pseudonym_key: Option<Vec<u8>>,
    pub audit_args: AuditArgs,
    /// Where the harness's audit trail is kept, for the evidence report (the
    /// trail itself does not say). `Harness::local(root)` keeps it at
    /// `root/audit.jsonl`.
    pub audit_path: Option<String>,
}

impl Default for GovernanceOptions {
    fn default() -> Self {
        Self {
            policy: None,
            packs: vec![],
            mode: Mode::Enforce,
            home: None,
            regions: IndexMap::new(),
            signer: None,
            signing_key: None,
            approver: None,
            approval_timeout: 900.0,
            approval_notify: None,
            vault: None,
            inventory: None,
            classifier: None,
            monitor: None,
            notify: vec![],
            service_provider: String::new(),
            pseudonym_key: None,
            audit_args: AuditArgs::Scrubbed,
            audit_path: None,
        }
    }
}

/// What `register_agent` reads from an agent.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AgentInfo {
    pub name: String,
    pub description: String,
    pub model: String,
    pub tools: Vec<String>,
    /// The agent's identity as JSON (`AgentIdentity` fields), if it declared one.
    pub identity: Option<Value>,
}

/// The harness services governance reads and erases from.
#[derive(Clone)]
pub(crate) struct Services {
    pub audit: AuditTrail,
    pub sessions: Arc<dyn SessionStore>,
    pub memory_store: Arc<dyn MemoryStore>,
    pub journal: RunJournal,
    pub tracer: Tracer,
    pub deliverables: DeliverableStore,
}

#[derive(Default)]
struct State {
    services: Option<Services>,
    identities: IndexMap<String, AgentIdentity>,
    assessments: IndexMap<String, RiskAssessment>,
    stats: IndexMap<(String, String), u64>,
    data_seen: IndexMap<String, u64>,
    states: IndexMap<String, SharedRun>,
    /// run id → the person it acts for (`_anonymous` for nobody).
    subjects: HashMap<String, String>,
    /// child agent → runs that have just delegated to it.
    pending_parents: HashMap<String, Vec<String>>,
    trail_key: Option<String>,
    #[cfg(test)]
    fail: bool,
}

struct Inner {
    mode: Mode,
    packs: Vec<Pack>,
    policy: Policy,
    engine: PolicyEngine,
    classifier: DataClassifier,
    signer: Option<Arc<dyn Signer>>,
    resolver: RegionResolver,
    pseudonymizer: Pseudonymizer,
    vault: SubjectVault,
    inventory: AIInventory,
    monitor: RuntimeMonitor,
    oversight: OversightDesk,
    incidents: IncidentDesk,
    retention: RetentionSweeper,
    service_provider: String,
    audit_args: AuditArgs,
    audit_path: Option<String>,
    state: Mutex<State>,
}

/// Policy, identity, data, residency, oversight, records — attached to a harness.
/// Cheap to clone; clones are the same governance layer.
#[derive(Clone)]
pub struct Governance {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Governance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Governance {} {} packs={:?}>", self.inner.policy.name, self.inner.mode.as_str(), self.pack_ids())
    }
}

type HookResult = std::result::Result<(), String>;

fn audit_detail(pairs: Vec<(&str, Value)>) -> Map<String, Value> {
    obj(pairs)
}

fn sorted(set: &BTreeSet<String>) -> Value {
    json!(set.iter().collect::<Vec<_>>())
}

fn data_context(classes: &BTreeSet<String>) -> Value {
    let mut m = obj(vec![
        ("classes", sorted(classes)),
        ("personal", json!(classes.contains("personal"))),
        ("special_category", json!(classes.contains("special_category"))),
    ]);
    for c in classes {
        m.insert(c.clone(), json!(true));
    }
    Value::Object(m)
}

fn withhold(value: &Value) -> Value {
    match value {
        Value::String(s) if !s.is_empty() => json!("[withheld]"),
        Value::Object(m) => Value::Object(m.iter().map(|(k, v)| (k.clone(), withhold(v))).collect()),
        Value::Array(a) => Value::Array(a.iter().map(withhold).collect()),
        other => other.clone(),
    }
}

fn map_strings(value: &Value, f: &dyn Fn(&str) -> String) -> Value {
    match value {
        Value::String(s) => Value::String(f(s)),
        Value::Object(m) => Value::Object(m.iter().map(|(k, v)| (k.clone(), map_strings(v, f))).collect()),
        Value::Array(a) => Value::Array(a.iter().map(|v| map_strings(v, f)).collect()),
        other => other.clone(),
    }
}

fn request_texts(request: &CompletionRequest) -> Vec<String> {
    let mut texts = vec![request.system.clone().unwrap_or_default()];
    for message in &request.messages {
        for block in &message.content {
            match block {
                ContentBlock::Text(t) => texts.push(t.text.clone()),
                ContentBlock::ToolResult(r) => texts.push(r.content.clone()),
                ContentBlock::ToolUse(u) => texts.push(crate::pyjson::dumps(&Value::Object(u.input.clone()))),
                _ => {}
            }
        }
    }
    texts
}

fn text_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

impl Governance {
    // ------------------------------------------------------------------
    // construction
    // ------------------------------------------------------------------

    /// Governance from your options: your policy plus `options.packs`.
    pub fn new(options: GovernanceOptions) -> Result<Self> {
        let GovernanceOptions {
            policy,
            packs,
            mode,
            home,
            regions,
            signer,
            signing_key,
            approver,
            approval_timeout,
            approval_notify,
            vault,
            inventory,
            classifier,
            monitor,
            notify,
            service_provider,
            pseudonym_key,
            audit_args,
            audit_path,
        } = options;
        let pack_policies: Vec<&Policy> = packs.iter().map(|p| &p.policy).collect();
        let mut merged = policy.unwrap_or_default().merge(&pack_policies);
        if let Some(home) = home.filter(|h| !h.is_empty()) {
            let mut residency = merged.residency.clone().unwrap_or_default();
            residency.home = Some(home);
            merged.residency = Some(residency);
        } else if let Some(r) = merged.residency.as_mut()
            && r.home.is_none()
        {
            // One jurisdiction restricting transfers is the obvious home. With
            // several, a principal without a jurisdiction would be unrestricted
            // — the evidence report flags that until `home` is set.
            let restricted: Vec<String> = r.transfers.iter().filter(|(_, d)| !d.iter().any(|x| x == "*")).map(|(o, _)| o.clone()).collect();
            if restricted.len() == 1 {
                r.home = restricted.into_iter().next();
            }
        }
        let classifier = classifier.unwrap_or_default();
        // A misspelt class in a deny rule would never match — fail open. Refuse it.
        let known = known_classes(merged.data_classes.iter().cloned().chain(classifier.extra_classes()));
        let unknown: Vec<String> = merged.class_names().into_iter().filter(|c| !known.contains(c)).collect();
        if !unknown.is_empty() {
            return Err(Error::config(format!(
                "unknown data classes in the policy: {} \u{2014} known: {}; declare your own under `data_classes:`",
                unknown.join(", "),
                known.into_iter().collect::<Vec<_>>().join(", ")
            )));
        }
        let signer: Option<Arc<dyn Signer>> = match (signer, signing_key) {
            (Some(s), _) => Some(s),
            (None, Some(k)) if !k.is_empty() => Some(Arc::new(HMACSigner::new(&k, None)?)),
            _ => None,
        };
        let vault = match vault {
            Some(p) => SubjectVault::at(p)?,
            None => SubjectVault::new(),
        };
        let inventory = inventory.map_or_else(AIInventory::new, AIInventory::at);
        let engine = PolicyEngine::new(merged.clone());
        let retention = RetentionSweeper::new(merged.data_retention_days);
        let clocks = packs.iter().map(|p| (p.id.clone(), p.incidents.clone())).collect();
        let inner = Arc::new_cyclic(|weak: &Weak<Inner>| {
            let w = weak.clone();
            let record_approval: ApprovalRecord = Arc::new(move |request: &ApprovalRequest, action: &str| {
                if let Some(inner) = w.upgrade() {
                    Governance { inner }.record_approval(request, action);
                }
            });
            let w = weak.clone();
            let record_incident: IncidentRecord = Arc::new(move |incident: &Incident| {
                if let Some(inner) = w.upgrade() {
                    Governance { inner }.record_incident(incident);
                }
            });
            let mut oversight = OversightDesk::new(approver);
            oversight.default_timeout = approval_timeout;
            oversight.record = Some(record_approval);
            oversight.notify = approval_notify;
            Inner {
                mode,
                packs,
                policy: merged,
                engine,
                classifier,
                signer,
                resolver: RegionResolver { overrides: regions },
                pseudonymizer: Pseudonymizer::new(pseudonym_key.as_deref()),
                vault,
                inventory,
                monitor: monitor.unwrap_or_default(),
                oversight,
                incidents: IncidentDesk::new(clocks, notify, Some(record_incident)),
                retention,
                service_provider,
                audit_args,
                audit_path,
                state: Mutex::new(State::default()),
            }
        });
        Ok(Self { inner })
    }

    /// Governance from built-in packs (by id) or pack files (by path), plus your options.
    pub fn from_packs(packs: &[&str], mut options: GovernanceOptions) -> Result<Self> {
        let mut loaded = packs.iter().map(|p| load_pack(p)).collect::<Result<Vec<_>>>()?;
        loaded.append(&mut options.packs);
        options.packs = loaded;
        Self::new(options)
    }

    /// A governance config file: `packs`, `policy`, `home`, `regions`, `mode`,
    /// `vault`, `inventory`, `service_provider`, `approval_timeout`,
    /// `signing_key_env`. What the options set wins over the file.
    ///
    /// The signing key is never read from the file — pass `signing_key` or set
    /// `signing_key_env` to the name of an environment variable holding it.
    pub fn from_file(path: impl AsRef<Path>, mut options: GovernanceOptions) -> Result<Self> {
        let path = path.as_ref();
        let where_ = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|e| Error::config(format!("{where_}: {e}")))?;
        let config: Value = serde_yaml::from_str(&text).map_err(|e| Error::config(format!("{where_}: {e}")))?;
        let config = match config {
            Value::Null => Map::new(),
            Value::Object(m) => m,
            _ => return Err(Error::config(format!("{where_}: a governance config must be a mapping"))),
        };
        let known = ["packs", "policy", "home", "regions", "mode", "vault", "inventory", "service_provider", "approval_timeout", "signing_key_env"];
        let mut unknown: Vec<&String> = config.keys().filter(|k| !known.contains(&k.as_str())).collect();
        if !unknown.is_empty() {
            unknown.sort();
            return Err(Error::config(format!("{where_}: unknown keys {}", unknown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))));
        }
        let s = |k: &str| config.get(k).and_then(Value::as_str).map(str::to_string);
        if let Some(env) = s("signing_key_env").filter(|e| !e.is_empty())
            && options.signing_key.is_none()
            && options.signer.is_none()
        {
            match std::env::var(&env) {
                Ok(v) if !v.is_empty() => options.signing_key = Some(v.into_bytes()),
                _ => return Err(Error::config(format!("{where_}: ${env} is not set"))),
            }
        }
        if options.policy.is_none()
            && let Some(p) = config.get("policy").filter(|p| !p.is_null())
        {
            options.policy = Some(match p {
                Value::String(source) => Policy::load(source)?,
                other => Policy::from_value(other)?,
            });
        }
        if options.home.is_none() {
            options.home = s("home");
        }
        if options.regions.is_empty()
            && let Some(Value::Object(r)) = config.get("regions")
        {
            options.regions = r.iter().map(|(k, v)| (k.clone(), super::py_str(v))).collect();
        }
        if options.mode == Mode::Enforce
            && let Some(m) = s("mode")
        {
            options.mode = Mode::parse(&m)?;
        }
        if options.vault.is_none() {
            options.vault = s("vault").map(Into::into);
        }
        if options.inventory.is_none() {
            options.inventory = s("inventory").map(Into::into);
        }
        if options.service_provider.is_empty() {
            options.service_provider = s("service_provider").unwrap_or_default();
        }
        if let Some(t) = config.get("approval_timeout").and_then(Value::as_f64)
            && options.approval_timeout == 900.0
        {
            options.approval_timeout = t;
        }
        let packs: Vec<String> = match config.get("packs") {
            Some(Value::Array(a)) => a.iter().map(super::py_str).collect(),
            Some(Value::String(one)) => vec![one.clone()],
            _ => vec![],
        };
        let refs: Vec<&str> = packs.iter().map(String::as_str).collect();
        Self::from_packs(&refs, options)
    }

    fn st(&self) -> std::sync::MutexGuard<'_, State> {
        self.inner.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    // ------------------------------------------------------------------
    // what it holds
    // ------------------------------------------------------------------

    pub fn mode(&self) -> &'static str {
        self.inner.mode.as_str()
    }

    pub fn policy(&self) -> &Policy {
        &self.inner.policy
    }

    pub fn engine(&self) -> &PolicyEngine {
        &self.inner.engine
    }

    pub fn packs(&self) -> &[Pack] {
        &self.inner.packs
    }

    pub fn pack_ids(&self) -> Vec<String> {
        self.inner.packs.iter().map(|p| p.id.clone()).collect()
    }

    pub fn signer(&self) -> Option<&Arc<dyn Signer>> {
        self.inner.signer.as_ref()
    }

    pub fn classifier(&self) -> &DataClassifier {
        &self.inner.classifier
    }

    pub fn resolver(&self) -> &RegionResolver {
        &self.inner.resolver
    }

    pub fn pseudonymizer(&self) -> &Pseudonymizer {
        &self.inner.pseudonymizer
    }

    pub fn vault(&self) -> &SubjectVault {
        &self.inner.vault
    }

    pub fn inventory(&self) -> &AIInventory {
        &self.inner.inventory
    }

    pub fn monitor(&self) -> &RuntimeMonitor {
        &self.inner.monitor
    }

    pub fn oversight(&self) -> &OversightDesk {
        &self.inner.oversight
    }

    pub fn incidents(&self) -> &IncidentDesk {
        &self.inner.incidents
    }

    pub fn retention(&self) -> &RetentionSweeper {
        &self.inner.retention
    }

    pub fn identities(&self) -> IndexMap<String, AgentIdentity> {
        self.st().identities.clone()
    }

    pub fn assessments(&self) -> IndexMap<String, RiskAssessment> {
        self.st().assessments.clone()
    }

    /// Decisions made: `(action, effect)` → how many.
    pub fn stats(&self) -> IndexMap<(String, String), u64> {
        self.st().stats.clone()
    }

    /// Data classes seen, by how many runs saw them.
    pub fn data_seen(&self) -> IndexMap<String, u64> {
        self.st().data_seen.clone()
    }

    /// The audit trail of the harness this is attached to.
    pub fn audit(&self) -> Option<AuditTrail> {
        self.st().services.as_ref().map(|s| s.audit.clone())
    }

    /// The key id the attached trail is signed with, when it is signed.
    pub fn trail_key_id(&self) -> Option<String> {
        self.st().trail_key.clone()
    }

    pub fn audit_path(&self) -> Option<String> {
        self.inner.audit_path.clone()
    }

    pub(crate) fn services(&self) -> Result<Services> {
        self.st().services.clone().ok_or_else(|| Error::config("attach governance to a harness first"))
    }

    /// Data-subject requests: `access(trace)` and `erase(trace)`.
    pub fn rights(&self) -> SubjectRights {
        SubjectRights::new(self.clone())
    }

    // ------------------------------------------------------------------
    // attaching
    // ------------------------------------------------------------------

    /// Wire into a harness: the signer onto its audit trail, personal data kept
    /// out of the trail, and the hooks that make every decision.
    pub fn attach(&self, harness: &mut Harness) -> Result<()> {
        if self.st().services.is_some() {
            return Err(Error::config("this Governance is already attached to a harness"));
        }
        let audit = harness.audit.clone();
        let entries = audit.entries();
        let mut trail_key = entries.iter().rev().find(|e| !e.signature.is_empty()).map(|e| e.key_id.clone());
        if let Some(signer) = &self.inner.signer {
            let unsigned = entries.iter().any(|e| e.signature.is_empty());
            // A trail with a signer of its own reports its unsigned entries.
            let has_signer = !entries.is_empty() && (entries.iter().all(|e| !e.signature.is_empty()) || audit.verify().1.contains("is not signed"));
            if unsigned && !has_signer {
                return Err(Error::config(
                    "the audit trail already holds unsigned entries; a signed trail must be signed from its first entry \u{2014} start a new audit file",
                ));
            }
            if !has_signer {
                audit.set_signer(Some(signer.clone()));
                trail_key = Some(signer.key_id());
            }
        }
        {
            let mut st = self.st();
            st.trail_key = trail_key;
            st.services = Some(Services {
                audit: audit.clone(),
                sessions: harness.sessions.clone(),
                memory_store: harness.memory_store.clone(),
                journal: harness.journal.clone(),
                tracer: harness.tracer.clone(),
                deliverables: harness.deliverables.clone(),
            });
        }
        // Personal data reaches the trail only as the current person's tokens —
        // including what the loop itself records (task text, tool arguments).
        let weak = Arc::downgrade(&self.inner);
        let scrub: Scrub = Arc::new(move |action: &str, target: &str, detail: Map<String, Value>| match weak.upgrade() {
            Some(inner) => Governance { inner }.scrub_entry(action, target, detail),
            None => (target.to_string(), detail),
        });
        audit.set_scrub(Some(scrub));
        for event in ["run_start", "run_end", "model_egress", "post_model", "pre_tool", "post_tool", "subagent_start", "handoff", "memory_write"] {
            let gov = self.clone();
            harness.hooks.on_async(event, move |hook: HookContext| {
                let gov = gov.clone();
                async move { gov.guarded(event, hook).await }
            });
        }
        let packs = self.pack_ids();
        audit.record(
            "governance",
            "governance.attach",
            "",
            "ok",
            "",
            audit_detail(vec![("policy", json!(self.inner.engine.hash[..16])), ("mode", json!(self.mode())), ("packs", json!(packs))]),
        );
        Ok(())
    }

    /// A governance failure is a refusal, never a crash and never a pass.
    ///
    /// In enforce mode the action is blocked and the error recorded; in monitor
    /// mode it is recorded and the action goes ahead.
    async fn guarded(&self, event: &'static str, mut hook: HookContext) -> HookContext {
        let original = hook.clone();
        let outcome = AssertUnwindSafe(self.handle(event, &mut hook)).catch_unwind().await;
        let error = match outcome {
            Ok(Ok(())) => return hook,
            Ok(Err(e)) => e,
            Err(panic) => {
                panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into())
            }
        };
        // Whatever the handler half-did is dropped: decide from the original.
        let mut hook = original;
        *self.st().stats.entry(("error".into(), event.into())).or_default() += 1;
        let enforce = self.inner.mode == Mode::Enforce;
        let actor = if hook.agent.is_empty() { "governance".to_string() } else { hook.agent.clone() };
        self.audit_record(
            &actor,
            "governance.error",
            event,
            if enforce { "deny" } else { "log" },
            &hook.run_id,
            audit_detail(vec![("error", json!(format!("GovernanceError: {error}")))]),
        );
        if enforce && event != "run_end" {
            hook.block("governance could not decide (GovernanceError); refused");
        }
        hook
    }

    async fn handle(&self, event: &str, hook: &mut HookContext) -> HookResult {
        #[cfg(test)]
        if self.st().fail {
            return Err("classifier exploded".into());
        }
        match event {
            "run_start" => self.on_run_start(hook).await,
            "run_end" => self.on_run_end(hook).await,
            "model_egress" => self.on_egress(hook).await,
            "post_model" => self.on_post_model(hook),
            "pre_tool" => self.on_pre_tool(hook).await,
            "post_tool" => self.on_post_tool(hook).await,
            "subagent_start" | "handoff" => self.on_subagent_start(hook),
            "memory_write" => self.on_memory_write(hook),
            _ => Ok(()),
        }
    }

    // ------------------------------------------------------------------
    // what the audit trail may keep
    // ------------------------------------------------------------------

    /// The person records are made for now. Python reads the current run from
    /// a context variable; here, when every live run acts for one person, it is
    /// that person, and otherwise nobody in particular (`_anonymous`) — records
    /// are still tokenised, but cannot be tied to one person's key.
    fn current_subject(&self) -> String {
        let st = self.st();
        let subjects: BTreeSet<&String> = st.subjects.values().collect();
        if subjects.len() == 1 { subjects.into_iter().next().cloned().unwrap_or_default() } else { "_anonymous".into() }
    }

    /// What the audit trail may keep of one entry.
    ///
    /// A task's text can name anyone, and no pattern finds every name, so it
    /// enters the immutable trail only as the person's token (the readable text
    /// stays in the run journal, which erasure rewrites). Everything else is
    /// kept, with the personal values the classifier finds tokenised.
    fn scrub_entry(&self, action: &str, target: &str, detail: Map<String, Value>) -> (String, Map<String, Value>) {
        let subject = self.current_subject();
        let attempt = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let target = if matches!(action, "run_start" | "run_end") && !target.is_empty() {
                self.inner.vault.token(&subject, target, "task")
            } else {
                text_of(&self.scrub(&json!(target), &subject))
            };
            let clean: Map<String, Value> = detail
                .iter()
                .map(|(k, v)| {
                    let v = if k == "args" && self.inner.audit_args == AuditArgs::Tokens {
                        map_strings(v, &|s| if s.is_empty() { String::new() } else { self.inner.vault.token(&subject, s, "arg") })
                    } else {
                        self.scrub(v, &subject)
                    };
                    (k.clone(), v)
                })
                .collect();
            (target, clean)
        }));
        attempt.unwrap_or_else(|_| {
            // Unable to tell what is personal: keep nothing rather than everything.
            if let Ok(mut st) = self.inner.state.try_lock() {
                *st.stats.entry(("error".into(), "audit_scrub".into())).or_default() += 1;
            }
            (text_of(&withhold(&json!(target))), withhold(&Value::Object(detail)).as_object().cloned().unwrap_or_default())
        })
    }

    /// Personal values → per-subject tokens, so erasure works on the record.
    fn scrub(&self, value: &Value, subject: &str) -> Value {
        match value {
            Value::String(s) => {
                let mut found: Vec<_> =
                    self.inner.classifier.scan(s).findings.into_iter().filter(|f| is_personal(&f.data_class) && f.kind != "keyword").collect();
                if found.is_empty() {
                    return value.clone();
                }
                found.sort_by_key(|f| f.span.0);
                let who = if subject.is_empty() { "_anonymous" } else { subject };
                let mut out = String::new();
                let mut cursor = 0;
                for f in found {
                    if f.span.0 < cursor {
                        continue;
                    }
                    out.push_str(&s[cursor..f.span.0]);
                    out.push_str(&self.inner.vault.token(who, f.value(s), &f.kind));
                    cursor = f.span.1;
                }
                out.push_str(&s[cursor..]);
                Value::String(out)
            }
            Value::Object(m) => Value::Object(m.iter().map(|(k, v)| (k.clone(), self.scrub(v, subject))).collect()),
            Value::Array(a) => Value::Array(a.iter().map(|v| self.scrub(v, subject)).collect()),
            other => other.clone(),
        }
    }

    // ------------------------------------------------------------------
    // registering agents
    // ------------------------------------------------------------------

    /// Register an agent built on the governed harness: classify its risk
    /// (refusing a prohibited use), record it in the inventory, and return the
    /// identity it will run under.
    pub fn register_agent(&self, info: &AgentInfo) -> Result<AgentIdentity> {
        self.register(info, None)
    }

    /// `register_agent`, and follow the agent's tools for fingerprints, drift
    /// and `data:` tags. `tools` answers the agent's tools as they are now
    /// (`None` once the agent is gone — hold the agent weakly).
    pub fn register_agent_with_tools(&self, info: &AgentInfo, tools: LiveTools) -> Result<AgentIdentity> {
        self.register(info, Some(tools))
    }

    fn register(&self, info: &AgentInfo, tools: Option<LiveTools>) -> Result<AgentIdentity> {
        let mut identity = AgentIdentity::of(info.identity.as_ref(), &info.name)?;
        let mut assessment = assess(&identity);
        if assessment.prohibited() {
            self.audit_record("governance", "governance.register", &info.name, "deny", "", audit_detail(vec![("reasons", json!(assessment.reasons))]));
            if self.inner.mode == Mode::Enforce {
                return Err(Error::config(format!("{}: prohibited use \u{2014} {}", info.name, assessment.reasons.join("; "))));
            }
        }
        if !self.inner.packs.is_empty() {
            // Report the classification under the frameworks actually in force.
            assessment = assessment.only(&self.inner.packs.iter().map(|p| p.id.clone()).collect());
        }
        if identity.risk.as_deref() != Some(assessment.tier.as_str()) {
            identity.risk = Some(assessment.tier.clone());
        }
        {
            let mut st = self.st();
            st.identities.insert(info.name.clone(), identity.clone());
            st.assessments.insert(info.name.clone(), assessment.clone());
        }
        let record = AgentRecord { name: &info.name, model: &info.model, ..Default::default() };
        let registry = tools.as_ref().and_then(|t| t()).unwrap_or_default();
        self.inner.inventory.register(&record, &identity, None::<&Region>, &registry);
        if registry.is_empty() && !info.tools.is_empty() {
            self.inner.inventory.set_tool_names(&info.name, &info.tools);
        }
        if let Some(t) = tools {
            self.inner.inventory.track(&info.name, t);
        }
        self.audit_record(
            "governance",
            "governance.register",
            &info.name,
            "ok",
            "",
            audit_detail(vec![
                ("risk", json!(assessment.tier)),
                ("owner", json!(identity.owner)),
                ("purpose", json!(identity.purpose)),
                ("frameworks", json!(assessment.frameworks)),
            ]),
        );
        Ok(identity)
    }

    /// Follow an agent's tools from now on (see `register_agent_with_tools`).
    pub fn track_tools(&self, agent: &str, tools: LiveTools) {
        self.inner.inventory.track(agent, tools);
    }

    // ------------------------------------------------------------------
    // deciding and recording
    // ------------------------------------------------------------------

    /// Ask the policy engine, marked enforced or not by the mode.
    pub fn decide(&self, action: &str, ctx: &Map<String, Value>, target: &str) -> Decision {
        let mut decision = self.inner.engine.decide(action, ctx, target);
        decision.enforced = self.inner.mode == Mode::Enforce;
        decision
    }

    fn audit_record(&self, actor: &str, action: &str, target: &str, decision: &str, run_id: &str, detail: Map<String, Value>) {
        let audit = self.st().services.as_ref().map(|s| s.audit.clone());
        if let Some(a) = audit {
            a.record(actor, action, target, decision, run_id, detail);
        }
    }

    fn record(&self, decision: &Decision, state: Option<&SharedRun>, classes: &BTreeSet<String>, extra: Vec<(&str, Value)>) {
        *self.st().stats.entry((decision.action.clone(), decision.effect.clone())).or_default() += 1;
        let (agent, run_id, subject) = match state {
            Some(s) => {
                let mut s = lock(s);
                if !decision.allowed() {
                    s.denials += 1;
                }
                (s.agent.clone(), s.run_id.clone(), s.subject())
            }
            None => ("governance".to_string(), String::new(), String::new()),
        };
        let mut detail = audit_detail(vec![
            ("policy", json!(decision.policy)),
            ("rules", json!(decision.rules)),
            ("reasons", json!(decision.reasons)),
            ("controls", json!(decision.controls)),
            ("enforced", json!(decision.enforced)),
            ("data", sorted(classes)),
            ("subject", json!(self.inner.vault.subject_ref(&subject, true))),
        ]);
        for (k, v) in extra {
            detail.insert(k.to_string(), self.scrub(&v, &subject));
        }
        self.audit_record(&agent, &format!("governance.{}", decision.action), &decision.target, &decision.effect, &run_id, detail);
    }

    fn record_approval(&self, request: &ApprovalRequest, action: &str) {
        self.audit_record(
            &request.agent,
            &format!("governance.{action}"),
            &request.target,
            &request.status,
            &request.run_id,
            audit_detail(vec![
                ("approval", json!(request.id)),
                ("approvers", json!(request.approvers)),
                ("rules", json!(request.rules)),
                ("votes", json!(request.votes.iter().map(|v| json!({"by": v.by, "approve": v.approve, "note": v.note})).collect::<Vec<_>>())),
            ]),
        );
    }

    fn record_incident(&self, incident: &Incident) {
        let actor = if incident.agent.is_empty() { "governance" } else { incident.agent.as_str() };
        self.audit_record(
            actor,
            "governance.incident",
            &incident.kind,
            &incident.severity,
            &incident.run_id,
            audit_detail(vec![
                ("incident", json!(incident.id)),
                ("title", json!(incident.title)),
                ("deadlines", json!(incident.deadlines.iter().map(|d| format!("{}: {} by {}", d.framework, d.notify, d.due_iso())).collect::<Vec<_>>())),
            ]),
        );
    }

    // ------------------------------------------------------------------
    // context
    // ------------------------------------------------------------------

    /// The run with this id, or else the latest live run of this agent.
    fn state_for(&self, run_id: &str, agent: &str) -> Option<SharedRun> {
        let st = self.st();
        if let Some(s) = st.states.get(run_id) {
            return Some(s.clone());
        }
        st.states.values().rev().find(|s| lock(s).agent == agent).cloned()
    }

    fn principal(&self, trace: Option<&Trace>) -> Map<String, Value> {
        let home = self.inner.policy.residency.as_ref().and_then(|r| r.home.clone());
        Principal::from_trace(trace, home.as_deref()).context()
    }

    fn context(&self, state: Option<&SharedRun>, extra: Vec<(&str, Value)>) -> Map<String, Value> {
        let mut ctx = match state {
            Some(s) => {
                let s = lock(s);
                let mut agent = s.chain.head().map(|i| i.dump()).and_then(|v| v.as_object().cloned()).unwrap_or_default();
                agent.insert("name".into(), json!(s.agent));
                obj(vec![
                    ("agent", Value::Object(agent)),
                    ("principal", Value::Object(s.principal.clone())),
                    ("purpose", json!(s.purpose)),
                    ("run", Value::Object(s.context())),
                    ("delegation", json!({"depth": s.chain.depth(), "chain": s.chain.names()})),
                    ("data", data_context(&s.classes)),
                ])
            }
            None => obj(vec![
                ("agent", json!({"name": ""})),
                ("principal", json!({})),
                ("purpose", json!("")),
                ("run", json!({})),
                ("delegation", json!({"depth": 0, "chain": []})),
                ("data", data_context(&BTreeSet::new())),
            ]),
        };
        for (k, v) in extra {
            ctx.insert(k.to_string(), v);
        }
        ctx
    }

    fn classes_of(state: Option<&SharedRun>) -> BTreeSet<String> {
        state.map(|s| lock(s).classes.clone()).unwrap_or_default()
    }

    fn classify(&self, state: Option<&SharedRun>, texts: &[String], declared: &[String]) -> Classification {
        let mut result = Classification { findings: vec![], declared: declared.iter().cloned().collect() };
        for text in texts.iter().filter(|t| !t.is_empty()) {
            result = result.union(self.inner.classifier.scan(text));
        }
        if let Some(s) = state {
            taint(s, &result.classes());
        }
        result
    }

    fn enforce(&self, decision: &Decision) -> bool {
        self.inner.mode == Mode::Enforce && !decision.allowed()
    }

    fn residency(&self) -> ResidencyConfig {
        self.inner.policy.residency.clone().unwrap_or_default()
    }

    // ------------------------------------------------------------------
    // the hooks
    // ------------------------------------------------------------------

    async fn on_run_start(&self, hook: &mut HookContext) -> HookResult {
        let agent = hook.agent.clone();
        let parent = {
            let mut st = self.st();
            let explicit = hook.data.get("parent_run_id").and_then(Value::as_str).and_then(|id| st.states.get(id).cloned());
            match explicit {
                Some(p) => Some(p),
                None => {
                    let waiting = st.pending_parents.get_mut(&agent).and_then(|q| if q.is_empty() { None } else { Some(q.remove(0)) });
                    // Left behind by a run that has since ended: not a parent.
                    waiting.and_then(|id| st.states.get(&id).cloned())
                }
            }
        };
        let identity = self.st().identities.get(&agent).cloned().unwrap_or_else(|| AgentIdentity { name: agent.clone(), ..Default::default() });
        let trace: Option<Trace> = hook.data.get("trace").filter(|t| !t.is_null()).and_then(|t| serde_json::from_value(t.clone()).ok());
        let (chain, principal, parent_purpose, parent_chain) = match &parent {
            Some(p) => {
                let p = lock(p);
                (p.chain.child(identity.clone()), p.principal.clone(), p.purpose.clone(), Some(p.chain.clone()))
            }
            None => (DelegationChain::new(vec![identity.clone()]), self.principal(trace.as_ref()), String::new(), None),
        };
        let purpose = [principal.get("purpose").and_then(Value::as_str).unwrap_or("").to_string(), identity.purpose.clone(), parent_purpose]
            .into_iter()
            .find(|p| !p.is_empty())
            .unwrap_or_default();
        let depth = chain.depth();
        let state: SharedRun = Arc::new(Mutex::new(RunState {
            run_id: hook.run_id.clone(),
            agent: agent.clone(),
            chain,
            principal,
            purpose: purpose.clone(),
            parent,
            ..Default::default()
        }));
        {
            let subject = lock(&state).subject();
            let mut st = self.st();
            st.states.insert(hook.run_id.clone(), state.clone());
            st.subjects.insert(hook.run_id.clone(), if subject.is_empty() { "_anonymous".into() } else { subject });
        }
        let task = hook.data.get("task").map(text_of).unwrap_or_default();
        let found = self.classify(Some(&state), &[task], &[]);

        let mut decision = self.decide("run", &self.context(Some(&state), vec![]), &agent);
        if identity.risk.as_deref() == Some("prohibited") {
            decision.tighten("deny", &format!("{agent} is a prohibited use"), "risk", &["C12"]);
        }
        if let Some(chain) = parent_chain {
            let (ok, why) = chain.may_delegate(self.inner.policy.max_delegation_depth);
            if !ok {
                decision.tighten("deny", &why, "delegation", &["C2"]);
            }
        }
        if !self.inner.policy.purposes.is_empty() && !self.inner.policy.purposes.contains_key(&purpose) {
            let shown = if purpose.is_empty() { "(none)" } else { purpose.as_str() };
            decision.tighten("deny", &format!("purpose {} is not declared in the policy", py_str_repr(shown)), "purpose", &["C7"]);
        }
        self.record(&decision, Some(&state), &found.classes(), vec![("depth", json!(depth)), ("purpose", json!(purpose))]);
        if self.enforce(&decision) {
            hook.block(&decision.reason());
        }
        Ok(())
    }

    async fn on_run_end(&self, hook: &mut HookContext) -> HookResult {
        let state = {
            let mut st = self.st();
            st.subjects.remove(&hook.run_id);
            st.states.shift_remove(&hook.run_id)
        };
        let (Some(state), Some(Value::Object(mut result))) = (state, hook.data.get("result").cloned()) else { return Ok(()) };
        let output = result.get("output").and_then(Value::as_str).unwrap_or("").to_string();
        let found = self.classify(Some(&state), std::slice::from_ref(&output), &[]);
        let agent = lock(&state).agent.clone();
        let mut decision = self.decide("output", &self.context(Some(&state), vec![]), &agent);
        let disallowed = self.disallowed(&state, &found.classes(), None);
        if !disallowed.is_empty() {
            decision.tighten("redact", "the output carries data this run may not hand out", "purpose", &["C5", "C7"]);
            decision.redact = disallowed.into_iter().collect();
        }
        let mut out = output.clone();
        if decision.effect == "deny" && decision.enforced {
            out = format!("[withheld by policy: {}]", decision.reason());
        } else if decision.effect == "redact" && decision.enforced && !output.is_empty() {
            out = self.redact_text(&output, &decision, Some(&state));
        }
        self.record(&decision, Some(&state), &found.classes(), vec![]);

        let transparency = &self.inner.policy.transparency;
        let s = lock(&state);
        if s.top_level() {
            if transparency.disclose {
                result.insert("disclosure".into(), json!(disclosure(&transparency.languages)));
            }
            let reader = s.principal.get("jurisdiction").and_then(Value::as_str).unwrap_or("").to_string();
            if transparency.label_output && !out.is_empty() && (transparency.label_for.iter().any(|j| j == "*") || transparency.label_for.contains(&reader)) {
                out = label(&out, &[jurisdiction_language(&reader)]);
            }
            if transparency.provenance && !out.is_empty() {
                let mut manifest = ProvenanceManifest {
                    agent: s.agent.clone(),
                    run_id: s.run_id.clone(),
                    models: s.models.iter().cloned().collect(),
                    providers: s.providers.iter().cloned().collect(),
                    service_provider: self.inner.service_provider.clone(),
                    policy: self.inner.engine.hash[..16].to_string(),
                    labels: obj(vec![("explicit", json!(transparency.label_output)), ("implicit", json!(true)), ("languages", json!(transparency.languages))]),
                    ..ProvenanceManifest::for_output(&out)
                };
                if let Some(signer) = &self.inner.signer {
                    manifest.sign(signer.as_ref());
                }
                result.insert("provenance".into(), serde_json::to_value(&manifest).unwrap_or_default());
            }
        }
        result.insert("output".into(), json!(out));
        result.insert(
            "governance".into(),
            json!({
                "policy": self.inner.engine.hash[..16], "mode": self.mode(),
                "data_classes": sorted(&s.classes), "regions": sorted(&s.regions),
                "denials": s.denials, "untrusted": s.untrusted, "delegation_depth": s.chain.depth(),
            }),
        );
        let classes = s.classes.clone();
        let subject = s.subject();
        let run_id = s.run_id.clone();
        drop(s);
        {
            let mut st = self.st();
            for c in classes {
                *st.data_seen.entry(c).or_default() += 1;
            }
        }
        if !subject.is_empty() {
            // Where this person's data now lives, so erasure can find all of it.
            let vault = &self.inner.vault;
            if let Some(id) = result.get("session_id").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                vault.note(&subject, "sessions", id);
            }
            vault.note(&subject, "runs", &run_id);
            if let Some(id) = result.get("trace_id").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                vault.note(&subject, "traces", id);
            }
        }
        hook.replace(Value::Object(result));
        Ok(())
    }

    async fn on_egress(&self, hook: &mut HookContext) -> HookResult {
        let state = self.state_for(&hook.run_id, &hook.agent);
        let raw = hook.replacement.clone().or_else(|| hook.data.get("request").cloned()).unwrap_or(Value::Null);
        let request: CompletionRequest = serde_json::from_value(raw).map_err(|e| format!("unreadable request: {e}"))?;
        let provider = ProviderInfo::from_value(hook.data.get("provider").unwrap_or(&Value::Null));
        let model = hook.data.get("model").and_then(Value::as_str).unwrap_or("").to_string();
        let region = self.inner.resolver.resolve(&provider, &model);
        let found = self.classify(state.as_ref(), &request_texts(&request), &[]);
        let classes = match &state {
            Some(s) => lock(s).classes.clone(),
            None => found.classes(),
        };
        let pname = provider.name.clone();
        let mut ctx = self
            .context(state.as_ref(), vec![("provider", json!({"name": pname, "model": model, "region": region.jurisdiction, "location": region.location}))]);
        ctx.insert("data".into(), data_context(&classes));
        let mut decision = self.decide("egress", &ctx, &format!("{pname}:{model}"));
        let origin = self.residency_check(&mut decision, state.as_ref(), &region, &classes);
        if let Some(s) = &state {
            let disallowed = self.disallowed(s, &found.classes(), Some(&classes));
            if !disallowed.is_empty() {
                decision.tighten("redact", "purpose and delegation limits: remove data this run may not use", "purpose", &["C5", "C7"]);
                let mut all: BTreeSet<String> = decision.redact.iter().cloned().collect();
                all.extend(disallowed);
                decision.redact = all.into_iter().collect();
            }
        }
        if decision.effect == "require_approval" {
            self.settle_approval(
                &mut decision,
                state.as_ref(),
                &hook.agent,
                obj(vec![("provider", json!(pname)), ("model", json!(model)), ("region", json!(region.jurisdiction))]),
            )
            .await;
        }
        self.record(
            &decision,
            state.as_ref(),
            &classes,
            vec![("region", json!(region.jurisdiction)), ("location", json!(region.location)), ("origin", json!(origin.clone().unwrap_or_default()))],
        );
        if self.enforce(&decision) {
            hook.block(&decision.reason());
            return Ok(());
        }
        if let Some(s) = &state {
            let mut s = lock(s);
            s.models.insert(model.clone());
            s.providers.insert(pname.clone());
            s.regions.insert(region.jurisdiction.clone());
        }
        self.inner.inventory.saw_region(&hook.agent, &region.jurisdiction);
        if let Some(origin) = origin
            && region.jurisdiction != origin
            && region.jurisdiction != "on_prem"
            && classes.iter().any(|c| expand_classes(&self.residency().applies_to).contains(c))
        {
            self.audit_record(
                &hook.agent,
                "governance.transfer",
                &region.jurisdiction,
                "allow",
                &hook.run_id,
                audit_detail(vec![
                    ("origin", json!(origin)),
                    ("destination", json!(region.jurisdiction)),
                    ("location", json!(region.location)),
                    ("provider", json!(pname)),
                    ("model", json!(model)),
                    ("data", sorted(&classes)),
                ]),
            );
        }
        if decision.effect == "redact" && decision.enforced {
            let sent = self.redact_request(&request, &decision, state.as_ref());
            hook.replace(serde_json::to_value(sent).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    fn residency_check(&self, decision: &mut Decision, state: Option<&SharedRun>, region: &Region, classes: &BTreeSet<String>) -> Option<String> {
        let residency = self.inner.policy.residency.as_ref();
        let (jurisdiction, chain_regions, agent) = match state {
            Some(s) => {
                let s = lock(s);
                (
                    s.principal.get("jurisdiction").and_then(Value::as_str).filter(|j| !j.is_empty()).map(str::to_string),
                    s.chain.allowed_regions(),
                    s.agent.clone(),
                )
            }
            None => (None, None, "agent".to_string()),
        };
        let origin = jurisdiction.or_else(|| residency.and_then(|r| r.home.clone()));
        if let Some(allowed) = chain_regions
            && !allowed.contains(&region.jurisdiction)
            && region.jurisdiction != "on_prem"
        {
            let names: Vec<&str> = allowed.iter().map(String::as_str).collect();
            decision.tighten("reroute", &format!("{agent} may only use models in {}", names.join(", ")), "identity.regions", &["C2", "C6"]);
        }
        let residency = residency?;
        if !classes.iter().any(|c| expand_classes(&residency.applies_to).contains(c)) {
            return origin;
        }
        let Some(allowed) = residency.destinations(origin.as_deref()) else { return origin };
        let dest = &region.jurisdiction;
        if dest == "on_prem" && residency.on_prem_always_allowed {
            return origin;
        }
        if !region.known() {
            if residency.unknown == "deny" {
                let what = if region.location.is_empty() { "this model" } else { region.location.as_str() };
                decision.tighten(
                    "reroute",
                    &format!("cannot tell where {what} processes data, and personal data is present \u{2014} declare its region"),
                    "residency",
                    &["C6"],
                );
            }
            return origin;
        }
        if !allowed.contains(dest) {
            decision.tighten(
                "reroute",
                &format!("personal data of a subject in {} may not be sent to {dest} (allowed: {})", origin.as_deref().unwrap_or("None"), allowed.join(", ")),
                "residency",
                &["C6"],
            );
        }
        origin
    }

    fn disallowed(&self, state: &SharedRun, found: &BTreeSet<String>, classes: Option<&BTreeSet<String>>) -> BTreeSet<String> {
        let present: BTreeSet<String> = classes.unwrap_or(found).iter().filter(|c| is_personal(c)).cloned().collect();
        let (purpose, chain) = {
            let s = lock(state);
            (s.purpose.clone(), s.chain.allowed_data())
        };
        let purposes = &self.inner.policy.purposes;
        let mut allowed: Option<BTreeSet<String>> = None;
        if !purposes.is_empty()
            && let Some(classes) = purposes.get(&purpose)
        {
            allowed = Some(expand_classes(classes));
        }
        if let Some(chain) = chain {
            let chain = expand_classes(&chain);
            allowed = Some(match allowed {
                None => chain,
                Some(a) => a.intersection(&chain).cloned().collect(),
            });
        }
        match allowed {
            None => BTreeSet::new(),
            Some(a) => present.difference(&a).cloned().collect(),
        }
    }

    fn redact_text(&self, text: &str, decision: &Decision, state: Option<&SharedRun>) -> String {
        let classes = if decision.redact.is_empty() { vec!["personal".to_string()] } else { decision.redact.clone() };
        let subject = state.map(|s| lock(s).subject()).unwrap_or_default();
        if decision.redact_mode == "pseudonymize" {
            if let Some(s) = state {
                lock(s).pseudonymized = true;
            }
            return self.inner.classifier.redact(text, &classes, Some(&self.inner.pseudonymizer), &subject);
        }
        self.inner.classifier.redact(text, &classes, None, "")
    }

    fn redact_request(&self, request: &CompletionRequest, decision: &Decision, state: Option<&SharedRun>) -> CompletionRequest {
        let mut sent = request.clone();
        if let Some(system) = sent.system.as_mut().filter(|s| !s.is_empty()) {
            *system = self.redact_text(system, decision, state);
        }
        for message in &mut sent.messages {
            for block in &mut message.content {
                match block {
                    ContentBlock::Text(t) => t.text = self.redact_text(&t.text, decision, state),
                    ContentBlock::ToolResult(r) => r.content = self.redact_text(&r.content, decision, state),
                    ContentBlock::ToolUse(u) => {
                        let mapped = map_strings(&Value::Object(u.input.clone()), &|s| self.redact_text(s, decision, state));
                        u.input = mapped.as_object().cloned().unwrap_or_default();
                    }
                    _ => {}
                }
            }
        }
        sent
    }

    fn on_post_model(&self, hook: &mut HookContext) -> HookResult {
        let Some(state) = self.state_for(&hook.run_id, &hook.agent) else { return Ok(()) };
        if !lock(&state).pseudonymized {
            return Ok(());
        }
        let raw = hook.replacement.clone().or_else(|| hook.data.get("response").cloned()).unwrap_or(Value::Null);
        let mut response: ModelResponse = serde_json::from_value(raw).map_err(|e| format!("unreadable response: {e}"))?;
        let p = &self.inner.pseudonymizer;
        let mut changed = false;
        for block in &mut response.message.content {
            match block {
                ContentBlock::Text(t) if t.text.contains('\u{27e8}') => {
                    t.text = p.restore(&t.text);
                    changed = true;
                }
                ContentBlock::ToolUse(u) => {
                    let restored = p.restore_value(&Value::Object(u.input.clone()));
                    if restored != Value::Object(u.input.clone()) {
                        u.input = restored.as_object().cloned().unwrap_or_default();
                        changed = true;
                    }
                }
                _ => {}
            }
        }
        if changed {
            hook.replace(serde_json::to_value(response).map_err(|e| e.to_string())?);
        }
        Ok(())
    }

    async fn on_pre_tool(&self, hook: &mut HookContext) -> HookResult {
        let state = self.state_for(&hook.run_id, &hook.agent);
        let name = hook.data.get("tool").and_then(Value::as_str).unwrap_or("").to_string();
        let args = match &hook.replacement {
            Some(v @ Value::Object(_)) => v.clone(),
            _ => hook.data.get("args").cloned().unwrap_or(json!({})),
        };
        let tags: Vec<String> = hook.data.get("tags").and_then(Value::as_array).map(|a| a.iter().map(super::py_str).collect()).unwrap_or_default();
        let found = self.classify(state.as_ref(), &[crate::pyjson::dumps(&args)], &[]);
        let mut ctx = self.context(
            state.as_ref(),
            vec![
                ("tool", json!({"name": name, "tags": tags, "permission": hook.data.get("permission").cloned().unwrap_or(Value::Null)})),
                ("args", args.clone()),
            ],
        );
        let mut classes = Self::classes_of(state.as_ref());
        classes.extend(found.classes());
        ctx.insert("data".into(), data_context(&classes));
        let mut decision = self.decide("tool", &ctx, &name);

        if let Some(s) = &state {
            let (ok, why) = lock(s).chain.allows_tool(&name, &tags);
            if !ok {
                decision.tighten("deny", &why, "identity.tools", &["C2"]);
            }
            if let Some(stop) = self.inner.monitor.on_tool(s, &name, &args) {
                decision.tighten("deny", &stop, "monitor", &["C14"]);
                let key = if stop.contains("breaker") { format!("breaker:{name}") } else { format!("loop:{name}") };
                self.incident_once(s, "runaway", &key, &stop, &hook.agent, &name).await;
            }
        }
        if let Some(drift) = self.drift(&hook.agent, &name)
            && self.inner.policy.tool_drift != "allow"
        {
            let effect = if self.inner.policy.tool_drift == "deny" { "deny" } else { "log" };
            decision.tighten(effect, &drift, "supply_chain", &["C13"]);
        }
        if decision.effect == "require_approval" {
            self.settle_approval(&mut decision, state.as_ref(), &hook.agent, args.as_object().cloned().unwrap_or_default()).await;
        }
        self.record(&decision, state.as_ref(), &found.classes(), vec![("args", args.clone()), ("tags", json!(tags))]);
        if self.enforce(&decision) {
            hook.block(&decision.reason());
        } else if decision.effect == "redact" && decision.enforced {
            hook.replace(map_strings(&args, &|s| self.redact_text(s, &decision, state.as_ref())));
        }
        Ok(())
    }

    fn drift(&self, agent: &str, tool: &str) -> Option<String> {
        let live = self.inner.inventory.live_tool(agent, tool)?;
        self.inner.inventory.check_tool(agent, &live)
    }

    async fn on_post_tool(&self, hook: &mut HookContext) -> HookResult {
        let state = self.state_for(&hook.run_id, &hook.agent);
        let Some(outcome) = hook.data.get("outcome").filter(|o| !o.is_null()).cloned() else { return Ok(()) };
        let name = hook.data.get("tool").and_then(Value::as_str).unwrap_or("").to_string();
        let is_error = outcome.get("is_error").and_then(Value::as_bool).unwrap_or(false);
        if let Some(s) = &state
            && self.inner.monitor.on_tool_outcome(s, &name, is_error)
        {
            let n = lock(s).failures.get(&name).copied().unwrap_or(0);
            self.incident_once(s, "runaway", &format!("breaker:{name}"), &format!("{name} failed {n} times in a row; breaker open"), &hook.agent, &name).await;
        }
        if is_error {
            return Ok(());
        }
        let content = match &hook.replacement {
            Some(v) => text_of(v),
            None => outcome.get("content").map(text_of).unwrap_or_default(),
        };
        let mut declared: Vec<String> = hook
            .data
            .get("tags")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).filter_map(|t| t.strip_prefix("data:")).map(str::to_string).collect())
            .unwrap_or_default();
        if declared.is_empty() {
            let tags: Vec<String> = match self.inner.inventory.live_tool(&hook.agent, &name) {
                Some(t) => t.tags.iter().cloned().collect(),
                None => self.inner.inventory.recorded_tags(&hook.agent, &name),
            };
            declared = tags.iter().filter_map(|t| t.strip_prefix("data:")).map(str::to_string).collect();
        }
        let found = self.classify(state.as_ref(), std::slice::from_ref(&content), &declared);
        let score = state.as_ref().map_or(0.0, |s| self.inner.monitor.on_tool_result(s, &content));
        let mut ctx = self.context(state.as_ref(), vec![("tool", json!({"name": name})), ("result", json!({"injection": score}))]);
        let mut classes = Self::classes_of(state.as_ref());
        classes.extend(found.classes());
        ctx.insert("data".into(), data_context(&classes));
        let mut decision = self.decide("tool_result", &ctx, &name);
        if let Some(s) = &state {
            let disallowed = self.disallowed(s, &found.classes(), None);
            if !disallowed.is_empty() {
                decision.tighten("redact", "tool returned data this run may not use", "purpose", &["C5", "C7"]);
                decision.redact = disallowed.into_iter().collect();
            }
        }
        if decision.effect != "allow" || !found.classes().is_empty() || score != 0.0 {
            self.record(&decision, state.as_ref(), &found.classes(), vec![("injection", json!((score * 100.0).round() / 100.0))]);
        }
        if self.enforce(&decision) {
            hook.replace(json!(format!("[withheld by policy: {}]", decision.reason())));
        } else if decision.effect == "redact" && decision.enforced {
            hook.replace(json!(self.redact_text(&content, &decision, state.as_ref())));
        }
        Ok(())
    }

    fn on_subagent_start(&self, hook: &mut HookContext) -> HookResult {
        let parent_agent = hook.data.get("parent").and_then(Value::as_str).unwrap_or("").to_string();
        let state = self.state_for(&hook.run_id, &parent_agent);
        let child = hook.agent.clone();
        let chain = state.as_ref().map(|s| lock(s).chain.clone()).unwrap_or_default();
        let ctx = self.context(
            state.as_ref(),
            vec![("child", json!({"name": child})), ("delegation", json!({"depth": chain.depth() + 1, "chain": chain.names(), "child": child}))],
        );
        let mut decision = self.decide("delegate", &ctx, &child);
        let (ok, why) = chain.may_delegate(self.inner.policy.max_delegation_depth);
        if !ok {
            decision.tighten("deny", &why, "delegation", &["C2"]);
        }
        if let Some(s) = &state
            && let Some(storm) = self.inner.monitor.on_delegate(s)
        {
            decision.tighten("deny", &storm, "monitor", &["C14"]);
        }
        match self.st().identities.get(&child) {
            None => {
                decision.tighten(
                    "deny",
                    &format!("{child} is not registered with governance \u{2014} it was built on a different harness, so its runs would go unchecked"),
                    "registration",
                    &["C1", "C2"],
                );
            }
            Some(i) if i.risk.as_deref() == Some("prohibited") => {
                decision.tighten("deny", &format!("{child} is a prohibited use"), "risk", &[]);
            }
            Some(_) => {}
        }
        self.record(&decision, state.as_ref(), &BTreeSet::new(), vec![("depth", json!(chain.depth() + 1))]);
        if self.enforce(&decision) {
            hook.block(&decision.reason());
        } else if let Some(s) = &state {
            let run_id = lock(s).run_id.clone();
            self.st().pending_parents.entry(child).or_default().push(run_id);
        }
        Ok(())
    }

    fn on_memory_write(&self, hook: &mut HookContext) -> HookResult {
        let state = self.state_for(&hook.run_id, &hook.agent);
        let text = match &hook.replacement {
            Some(v) => text_of(v),
            None => hook.data.get("text").map(text_of).unwrap_or_default(),
        };
        let found = self.classify(None, std::slice::from_ref(&text), &[]); // what is kept, not what the run saw
        let mut ctx = self.context(state.as_ref(), vec![("memory", json!({"scope": hook.data.get("scope"), "kind": hook.data.get("kind")}))]);
        if state.is_none() {
            // Written outside a run.
            let trace: Option<Trace> = hook.data.get("trace").filter(|t| !t.is_null()).and_then(|t| serde_json::from_value(t.clone()).ok());
            ctx.insert("principal".into(), Value::Object(self.principal(trace.as_ref())));
        }
        ctx.insert("data".into(), data_context(&found.classes()));
        let scope = hook.data.get("scope").map(super::py_str).unwrap_or_else(|| "None".into());
        let mut decision = self.decide("memory", &ctx, &scope);
        if found.classes().contains("credentials") {
            decision.tighten("redact", "credentials are never kept in memory", "builtin.credentials", &["C5"]);
            let mut all: BTreeSet<String> = decision.redact.iter().cloned().collect();
            all.insert("credentials".into());
            decision.redact = all.into_iter().collect();
        }
        self.record(&decision, state.as_ref(), &found.classes(), vec![]);
        if self.enforce(&decision) {
            hook.block(&decision.reason());
        } else if decision.effect == "redact" && decision.enforced {
            hook.replace(json!(self.redact_text(&text, &decision, None)));
        }
        Ok(())
    }

    /// Ask the people. A yes clears the gate (other obligations stay); anything
    /// else is a deny. In monitor mode nobody is asked and the decision is
    /// recorded as `require_approval` — what would have happened.
    async fn settle_approval(&self, decision: &mut Decision, state: Option<&SharedRun>, agent: &str, args: Map<String, Value>) {
        if self.inner.mode == Mode::Monitor {
            return;
        }
        let (requester, run_id) = state
            .map(|s| {
                let s = lock(s);
                (s.principal.get("user").and_then(Value::as_str).unwrap_or("").to_string(), s.run_id.clone())
            })
            .unwrap_or_default();
        let request = ApprovalRequest {
            reason: decision.reason(),
            args,
            rules: decision.rules.clone(),
            approvers: decision.approvers,
            timeout: decision.timeout,
            requester,
            run_id,
            ..ApprovalRequest::new(agent, &decision.action, &decision.target)
        };
        if self.inner.oversight.request(request).await.status == "approved" {
            decision.effect = if decision.redact.is_empty() { "log".into() } else { "redact".into() };
            decision.reasons.push("approved".into());
        } else {
            decision.tighten("deny", "approval was not given", "oversight", &["C8"]);
        }
    }

    /// One incident per cause per run, however often the cause repeats.
    async fn incident_once(&self, state: &SharedRun, kind: &str, key: &str, title: &str, agent: &str, tool: &str) {
        let run_id = {
            let mut s = lock(state);
            if !s.incidents.insert(key.to_string()) {
                return;
            }
            s.run_id.clone()
        };
        self.inner.incidents.open(kind, "medium", title, agent, &run_id, obj(vec![("tool", json!(tool))])).await;
    }

    // ------------------------------------------------------------------
    // the things people ask for
    // ------------------------------------------------------------------

    /// The evidence report for one pack (active or not).
    pub fn report(&self, pack: &str) -> ComplianceReport {
        ComplianceReport::build(self, Some(pack))
    }

    /// The evidence report for every active pack.
    pub fn report_all(&self) -> ComplianceReport {
        ComplianceReport::build(self, None)
    }

    /// A draft DPIA (`dpia`), FRIA (`fria`) or Korean impact assessment (`korea`).
    pub fn impact_assessment(&self, agent: &str, kind: &str) -> Result<String> {
        let identity = self.st().identities.get(agent).cloned().ok_or_else(|| Error::config(format!("{agent} is not registered with governance")))?;
        let seen: Vec<String> = self.data_seen().into_iter().filter(|(_, n)| *n > 0).map(|(c, _)| c).collect();
        Ok(impact_assessment(&identity, self.inner.inventory.agent(agent).as_ref(), Some(&self.inner.policy), kind, &seen))
    }

    /// Apply the data-retention limit now.
    pub async fn sweep(&self) -> Result<Map<String, Value>> {
        let services = self.services()?;
        self.inner.retention.sweep(&services.sessions, Some(&services.audit), None, Some(&self.inner.vault)).await
    }

    /// The state of things, for the harness report.
    pub fn summary(&self) -> Value {
        let mut decisions: IndexMap<String, IndexMap<String, u64>> = IndexMap::new();
        let st = self.st();
        for ((action, effect), n) in &st.stats {
            decisions.entry(action.clone()).or_default().insert(effect.clone(), *n);
        }
        let agents = st.identities.len();
        let seen = st.data_seen.clone();
        drop(st);
        let policy = &self.inner.policy;
        json!({
            "policy": format!("{}@{}", policy.name, policy.version),
            "policy_hash": self.inner.engine.hash[..16], "mode": self.mode(),
            "packs": self.pack_ids(), "agents": agents,
            "decisions": decisions,
            "approvals_pending": self.inner.oversight.pending().len(),
            "incidents_open": self.inner.incidents.open_incidents().len(),
            "data_seen": seen,
            "signed": self.inner.signer.is_some(),
            "checked_at": now(),
        })
    }

    #[cfg(test)]
    pub(crate) fn fail_for_test(&self, on: bool) {
        self.st().fail = on;
    }

    /// Register agents' tools by registry right away (no live tracking).
    pub fn register_agent_tools(&self, info: &AgentInfo, tools: ToolRegistry) -> Result<AgentIdentity> {
        let tools = Arc::new(tools);
        self.register(info, Some(Arc::new(move || Some((*tools).clone()))))
    }
}

#[cfg(test)]
mod tests;
