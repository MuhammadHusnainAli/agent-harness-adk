//! Approvals that outlive the process: a run paused for a person, kept in a store.
//!
//! ```no_run
//! # async fn demo(agent: agent_harness::Agent) -> agent_harness::Result<()> {
//! let result = agent.run("Refund order 4182.").await?;   // refund asks first
//! let id = result.approval.as_ref().unwrap()["id"].as_str().unwrap().to_string();
//! // ...ten hours later, in another process, on another machine:
//! let desk = agent.harness().approvals.clone().unwrap();
//! desk.approve(&id, "maria", "checked the order", None).await?;
//! let result = agent.resume_approval(&id).await?;       // carries on from that step
//! # Ok(()) }
//! ```
//!
//! When a tool needs approval and nobody is there to give it, the run does not
//! wait and does not fail. It stops at that step and is written down: the
//! conversation so far, the call that is waiting with its exact arguments, and
//! what the other tools of the same step returned. Nothing has to stay running.
//!
//! When the answer comes, the run is picked up from the record. The approved
//! call is executed with the arguments that were approved — the model is not
//! asked again, so it cannot ask for something else — and the tools that had
//! already run are not run a second time. A call that was declined is answered
//! with the reviewer's reason, and the agent carries on from there.
//!
//! A record is resumed once. The claim is taken before the tool runs, in a
//! write the store refuses to a second taker, so two workers that both see an
//! approval cannot both make the refund. Records are the Python harness's JSON.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::errors::{Error, Result};
use crate::runtime::audit::AuditTrail;
use crate::runtime::session::{ListFilter, Session, SessionStore};
use crate::types::{Message, Usage, new_id, now};

fn approval_error(message: impl Into<String>) -> Error {
    Error::Approval(message.into())
}

fn conflict(id: &str, message: String) -> Error {
    Error::SessionConflict { message, session_id: id.into() }
}

fn pending() -> String {
    "pending".into()
}

/// One tool call waiting for a person.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalCall {
    /// The id the model gave the call. What `approve(..., call)` names.
    pub id: String,
    pub tool: String,
    /// Exactly what will be run. An approval is for these arguments and no others.
    #[serde(default)]
    pub args: Map<String, Value>,
    /// Why it needs asking.
    #[serde(default)]
    pub reason: String,
    /// pending | approved | denied
    #[serde(default = "pending")]
    pub status: String,
    #[serde(default)]
    pub by: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub decided: Option<f64>,
}

impl ApprovalCall {
    pub fn describe(&self) -> String {
        let args: Vec<String> = self
            .args
            .iter()
            .map(|(k, v)| format!("{k}={}", crate::pyjson::dumps_with(v, crate::pyjson::Dumps { ensure_ascii: false, ..Default::default() })))
            .collect();
        format!("{}({})", self.tool, args.join(", "))
    }
}

fn apr() -> String {
    new_id("apr")
}

/// A run that stopped to ask, and everything needed to carry it on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Approval {
    #[serde(default = "apr")]
    pub id: String,
    #[serde(default)]
    pub run_id: String,
    #[serde(default)]
    pub agent: String,
    /// Whose run it is. Listing and resuming are held to it.
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub tenant_id: Option<String>,
    /// The conversation it belongs to, and the version it was at when it paused.
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub session_version: u64,
    #[serde(default)]
    pub task: String,
    #[serde(default)]
    pub model: String,
    /// The step it stopped in. Resuming finishes this step, then goes on.
    #[serde(default)]
    pub step: u32,
    /// pending | approved | denied | expired | resuming | resumed | failed
    #[serde(default = "pending")]
    pub status: String,
    #[serde(default)]
    pub calls: Vec<ApprovalCall>,
    /// What the other tools of that step returned. They are not run again.
    #[serde(default)]
    pub done: Vec<Value>,
    /// The conversation as the model had it, ending with its request for the tools.
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub usage: Usage,
    /// A mode's todo list and source ledger, when the run kept them.
    #[serde(default)]
    pub notebook: Option<Value>,
    #[serde(default = "now")]
    pub created: f64,
    #[serde(default = "now")]
    pub updated: f64,
    /// After this, an unanswered request counts as declined. `None`: never.
    #[serde(default)]
    pub expires: Option<f64>,
    /// When a worker took it to resume, and which.
    #[serde(default)]
    pub claimed: Option<f64>,
    #[serde(default)]
    pub claimed_by: String,
    /// How the resumed run ended.
    #[serde(default)]
    pub outcome: Map<String, Value>,
    #[serde(default)]
    pub version: u64,
}

impl Default for Approval {
    fn default() -> Self {
        serde_json::from_value(json!({})).expect("every field has a default")
    }
}

impl Approval {
    pub fn waiting(&self) -> Vec<&ApprovalCall> {
        self.calls.iter().filter(|c| c.status == "pending").collect()
    }

    pub fn expired(&self) -> bool {
        self.status == "pending" && self.expires.is_some_and(|e| now() >= e)
    }

    /// The same rule as a session: an axis nobody set is not a wall.
    pub fn owned_by(&self, user_id: Option<&str>, tenant_id: Option<&str>) -> bool {
        let ok = |held: &Option<String>, given: Option<&str>| match (held.as_deref().filter(|s| !s.is_empty()), given.filter(|s| !s.is_empty())) {
            (Some(h), Some(g)) => h == g,
            _ => true,
        };
        ok(&self.user_id, user_id) && ok(&self.tenant_id, tenant_id)
    }

    /// One line for a person deciding: who wants to do what.
    pub fn describe(&self) -> String {
        format!("{} wants to run {}", self.agent, self.calls.iter().map(ApprovalCall::describe).collect::<Vec<_>>().join("; "))
    }

    /// Everything but the conversation — what a list of approvals shows.
    pub fn summary(&self) -> Value {
        let task: String = self.task.chars().take(200).collect();
        json!({
            "id": self.id, "status": if self.expired() { "expired" } else { &self.status },
            "agent": self.agent, "run_id": self.run_id, "session_id": self.session_id,
            "user_id": self.user_id, "tenant_id": self.tenant_id, "task": task,
            "step": self.step, "created": self.created, "expires": self.expires,
            "calls": self.calls, "outcome": self.outcome,
        })
    }

    /// The request's status, from its calls'.
    fn settle(&mut self) {
        if self.status != "pending" || !self.waiting().is_empty() {
            return;
        }
        self.status = if self.calls.iter().any(|c| c.status == "approved") { "approved" } else { "denied" }.into();
    }
}

/// An edit to a stored record: `Ok(true)` to save it, `Ok(false)` to leave it
/// as it is, an error to refuse.
pub type Edit<'a> = &'a (dyn Fn(&mut Approval) -> Result<bool> + Send + Sync);

/// Where paused runs are kept. `save` must refuse a stale write — that refusal
/// is what makes a run resume once and a decision count once.
#[async_trait]
pub trait ApprovalStore: Send + Sync {
    async fn load(&self, approval_id: &str) -> Result<Option<Approval>>;
    async fn create(&self, approval: Approval) -> Result<Approval>;
    /// Store a changed record. `SessionConflict` if someone else saved it since it was loaded.
    async fn save(&self, approval: Approval) -> Result<Approval>;
    /// The newest first.
    async fn list(&self, limit: usize) -> Result<Vec<Approval>>;
    async fn delete(&self, _approval_id: &str) -> Result<()> {
        Ok(())
    }

    /// Load, edit, save — again if someone else saved in between.
    async fn change(&self, approval_id: &str, edit: Edit<'_>) -> Result<Approval> {
        for _ in 0..8 {
            let Some(mut record) = self.load(approval_id).await? else {
                return Err(approval_error(format!("no approval '{approval_id}'")));
            };
            if !edit(&mut record)? {
                return Ok(record);
            }
            record.updated = now();
            match self.save(record).await {
                Ok(saved) => return Ok(saved),
                Err(Error::SessionConflict { .. }) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(approval_error(format!("approval {approval_id} could not be saved: it kept changing")))
    }
}

/// In this process, gone with it. For tests, and for trying it out.
#[derive(Debug, Default)]
pub struct MemoryApprovalStore {
    held: Mutex<IndexMap<String, Approval>>,
}

impl MemoryApprovalStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ApprovalStore for MemoryApprovalStore {
    async fn load(&self, approval_id: &str) -> Result<Option<Approval>> {
        Ok(self.held.lock().unwrap_or_else(|p| p.into_inner()).get(approval_id).cloned())
    }

    async fn create(&self, mut approval: Approval) -> Result<Approval> {
        approval.version = 1;
        self.held.lock().unwrap_or_else(|p| p.into_inner()).insert(approval.id.clone(), approval.clone());
        Ok(approval)
    }

    async fn save(&self, mut approval: Approval) -> Result<Approval> {
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        if held.get(&approval.id).is_some_and(|h| h.version != approval.version) {
            return Err(conflict(&approval.id, format!("approval {} was saved by someone else", approval.id)));
        }
        approval.version += 1;
        held.insert(approval.id.clone(), approval.clone());
        Ok(approval)
    }

    async fn list(&self, limit: usize) -> Result<Vec<Approval>> {
        let mut rows: Vec<Approval> = self.held.lock().unwrap_or_else(|p| p.into_inner()).values().cloned().collect();
        rows.sort_by(|a, b| b.created.partial_cmp(&a.created).unwrap_or(std::cmp::Ordering::Equal));
        rows.truncate(limit);
        Ok(rows)
    }

    async fn delete(&self, approval_id: &str) -> Result<()> {
        self.held.lock().unwrap_or_else(|p| p.into_inner()).shift_remove(approval_id);
        Ok(())
    }
}

/// One JSON file per paused run, in a directory. What `Harness::local` uses.
///
/// Good for one machine. For several replicas, keep approvals where the
/// sessions are: [`SessionApprovalStore`].
#[derive(Debug, Clone)]
pub struct FileApprovalStore {
    pub root: PathBuf,
}

impl FileApprovalStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(|e| Error::Io(format!("{}: {e}", root.display())))?;
        Ok(Self { root })
    }

    fn path(&self, approval_id: &str) -> PathBuf {
        let safe: String = approval_id.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
        self.root.join(format!("{safe}.json"))
    }

    fn read(path: &std::path::Path) -> Option<Approval> {
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
    }

    /// Check-then-write as one step for every process sharing the directory:
    /// a lock file is the one thing a filesystem gives to exactly one of them.
    fn write(&self, approval: &Approval, expect: Option<u64>) -> bool {
        let path = self.path(&approval.id);
        let lock = path.with_extension("lock");
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&lock) {
                Ok(_) => break,
                Err(_) => {
                    let stale = std::fs::metadata(&lock)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .is_some_and(|age| age > Duration::from_secs(30));
                    if stale {
                        let _ = std::fs::remove_file(&lock); // its holder died mid-write
                        continue;
                    }
                    if Instant::now() > deadline {
                        return false;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
        let written = (|| {
            let held = Self::read(&path);
            match expect {
                None if held.is_some() => return false,
                Some(v) if held.as_ref().is_none_or(|h| h.version != v) => return false,
                _ => {}
            }
            let scratch = path.with_extension(format!("{}.tmp", new_id("")));
            let Ok(body) = serde_json::to_string_pretty(approval) else { return false };
            std::fs::write(&scratch, body).is_ok() && std::fs::rename(&scratch, &path).is_ok()
        })();
        let _ = std::fs::remove_file(&lock);
        written
    }
}

#[async_trait]
impl ApprovalStore for FileApprovalStore {
    async fn load(&self, approval_id: &str) -> Result<Option<Approval>> {
        let path = self.path(approval_id);
        Ok(tokio::task::spawn_blocking(move || Self::read(&path)).await.ok().flatten())
    }

    async fn create(&self, mut approval: Approval) -> Result<Approval> {
        approval.version = 1;
        let (me, rec) = (self.clone(), approval.clone());
        if !tokio::task::spawn_blocking(move || me.write(&rec, None)).await.unwrap_or(false) {
            return Err(conflict(&approval.id, format!("approval {} already exists", approval.id)));
        }
        Ok(approval)
    }

    async fn save(&self, mut approval: Approval) -> Result<Approval> {
        let expect = approval.version;
        approval.version += 1;
        let (me, rec) = (self.clone(), approval.clone());
        if !tokio::task::spawn_blocking(move || me.write(&rec, Some(expect))).await.unwrap_or(false) {
            return Err(conflict(&approval.id, format!("approval {} was saved by someone else", approval.id)));
        }
        Ok(approval)
    }

    async fn list(&self, limit: usize) -> Result<Vec<Approval>> {
        let root = self.root.clone();
        Ok(tokio::task::spawn_blocking(move || {
            let mut files: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&root)
                .map(|d| {
                    d.flatten()
                        .map(|e| e.path())
                        .filter(|p| p.extension().is_some_and(|x| x == "json"))
                        .filter_map(|p| Some((std::fs::metadata(&p).and_then(|m| m.modified()).ok()?, p)))
                        .collect()
                })
                .unwrap_or_default();
            files.sort_by_key(|f| std::cmp::Reverse(f.0));
            files.iter().take(limit * 2).filter_map(|(_, p)| Self::read(p)).take(limit).collect()
        })
        .await
        .unwrap_or_default())
    }

    async fn delete(&self, approval_id: &str) -> Result<()> {
        let _ = std::fs::remove_file(self.path(approval_id));
        Ok(())
    }
}

/// Approvals kept in a session store: any database the harness keeps chats in.
///
/// Each paused run is one row there, under an id and an agent name of its own,
/// so it is never listed among an agent's chats — and it is versioned the same
/// way a chat is, which is what a claim relies on.
pub struct SessionApprovalStore {
    pub sessions: Arc<dyn SessionStore>,
    pub prefix: String,
}

impl SessionApprovalStore {
    /// The agent name every approval row is filed under.
    pub const AGENT: &'static str = "#approvals";

    pub fn new(sessions: Arc<dyn SessionStore>) -> Self {
        Self { sessions, prefix: "approval_".into() }
    }

    fn session(&self, approval: &Approval) -> Session {
        let mut blob = serde_json::to_value(approval).unwrap_or_default();
        if let Value::Object(m) = &mut blob {
            m.remove("version");
        }
        let mut metadata = Map::new();
        metadata.insert("approval".into(), blob);
        Session {
            id: format!("{}{}", self.prefix, approval.id),
            agent: Self::AGENT.into(),
            title: approval.describe().chars().take(120).collect(),
            user_id: approval.user_id.clone(),
            tenant_id: approval.tenant_id.clone(),
            version: approval.version,
            created: approval.created,
            metadata,
            ..Session::new(Self::AGENT)
        }
    }

    fn record(session: &Session) -> Option<Approval> {
        let mut a: Approval = serde_json::from_value(session.metadata.get("approval")?.clone()).ok()?;
        a.version = session.version;
        Some(a)
    }
}

#[async_trait]
impl ApprovalStore for SessionApprovalStore {
    async fn load(&self, approval_id: &str) -> Result<Option<Approval>> {
        match self.sessions.load(&format!("{}{approval_id}", self.prefix)).await {
            Ok(s) => Ok(Self::record(&s)),
            Err(Error::Configuration(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    async fn create(&self, mut approval: Approval) -> Result<Approval> {
        approval.version = 0;
        let mut s = self.session(&approval);
        self.sessions.save(&mut s).await?;
        approval.version = s.version;
        Ok(approval)
    }

    async fn save(&self, mut approval: Approval) -> Result<Approval> {
        let mut s = self.session(&approval);
        self.sessions.save(&mut s).await?;
        approval.version = s.version;
        Ok(approval)
    }

    async fn list(&self, limit: usize) -> Result<Vec<Approval>> {
        let rows = self.sessions.list(&ListFilter { limit, agent: Some(Self::AGENT.into()), ..Default::default() }).await?;
        Ok(rows.iter().filter_map(Self::record).collect())
    }

    async fn delete(&self, approval_id: &str) -> Result<()> {
        self.sessions.delete(&format!("{}{approval_id}", self.prefix)).await
    }
}

/// Told about each new request — post it to Slack, a ticket queue, a web page.
pub type Notify = Arc<dyn Fn(Approval) -> BoxFuture<'static, Result<()>> + Send + Sync>;

/// Where paused runs wait for people, and where people answer. Cheap to clone.
///
/// - `expires` — seconds an unanswered request stays open (7 days by default);
///   after that it counts as declined. `None`: it waits for ever.
/// - `notify` — called with each new [`Approval`]. One that fails does not lose
///   the request: it is stored first.
/// - `self_approval` — may the person the agent is acting for approve their
///   own request? `false` enforces a second pair of eyes.
/// - `claim_timeout` — seconds after which a resume that never finished may be
///   released by hand with [`Approvals::release`].
#[derive(Clone)]
pub struct Approvals {
    pub store: Arc<dyn ApprovalStore>,
    pub expires: Option<f64>,
    pub notify: Option<Notify>,
    pub self_approval: bool,
    pub claim_timeout: f64,
    /// The harness's audit trail, set when this is given to a harness.
    pub audit: Option<AuditTrail>,
}

impl std::fmt::Debug for Approvals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Approvals expires={:?}>", self.expires)
    }
}

impl Default for Approvals {
    fn default() -> Self {
        Self::new(Arc::new(MemoryApprovalStore::new()))
    }
}

impl Approvals {
    pub fn new(store: Arc<dyn ApprovalStore>) -> Self {
        Self { store, expires: Some(7.0 * 24.0 * 3600.0), notify: None, self_approval: true, claim_timeout: 900.0, audit: None }
    }

    fn note(&self, actor: &str, action: &str, approval: &Approval, decision: &str, mut detail: Map<String, Value>) {
        if let Some(a) = &self.audit {
            detail.insert("agent".into(), json!(approval.agent));
            a.record(actor, action, &approval.id, decision, &approval.run_id, detail);
        }
    }

    // ---- the harness side ---------------------------------------------------
    /// Store a paused run and tell whoever answers these.
    pub async fn open(&self, mut approval: Approval) -> Result<Approval> {
        if approval.expires.is_none()
            && let Some(e) = self.expires
        {
            approval.expires = Some(approval.created + e);
        }
        let saved = self.store.create(approval).await?;
        let calls: Vec<String> = saved.calls.iter().map(ApprovalCall::describe).collect();
        self.note(&saved.agent, "approval_requested", &saved, "pending", json!({"calls": calls}).as_object().cloned().unwrap_or_default());
        if let Some(notify) = &self.notify
            && let Err(e) = notify(saved.clone()).await
        {
            let error: String = e.to_string().chars().take(200).collect();
            self.note("system", "approval_notify_failed", &saved, "error", json!({"error": error}).as_object().cloned().unwrap_or_default());
        }
        Ok(saved)
    }

    /// Take a decided request to resume it. Only one taker gets it.
    pub async fn claim(&self, approval_id: &str, by: &str, user_id: Option<&str>, tenant_id: Option<&str>, agent: Option<&str>) -> Result<Approval> {
        let take = |r: &mut Approval| -> Result<bool> {
            if !r.owned_by(user_id, tenant_id) {
                // In the words of one that does not exist: an id cannot be probed.
                return Err(approval_error(format!("no approval '{approval_id}'")));
            }
            if let Some(a) = agent
                && r.agent != a
            {
                return Err(approval_error(format!("approval {} is for the agent '{}', not '{a}'", r.id, r.agent)));
            }
            if r.expired() {
                r.status = "expired".into();
            }
            match r.status.as_str() {
                "pending" => {
                    let waiting: Vec<String> = r.waiting().iter().map(|c| c.describe()).collect();
                    return Err(approval_error(format!("approval {} is still waiting for an answer on: {}", r.id, waiting.join(", "))));
                }
                "resuming" => {
                    let who = if r.claimed_by.is_empty() { String::new() } else { format!(" by {}", r.claimed_by) };
                    return Err(approval_error(format!("approval {} is already being resumed{who} — if that worker is gone, release it first", r.id)));
                }
                "resumed" | "failed" => return Err(approval_error(format!("approval {} was already resumed ({})", r.id, r.status))),
                _ => {}
            }
            r.status = "resuming".into();
            r.claimed = Some(now());
            r.claimed_by = by.into();
            Ok(true)
        };
        let claimed = self.store.change(approval_id, &take).await?;
        self.note(if by.is_empty() { &claimed.agent } else { by }, "approval_resumed", &claimed, "ok", Map::new());
        Ok(claimed)
    }

    pub async fn finish(&self, approval_id: &str, outcome: Map<String, Value>, failed: bool) -> Result<Approval> {
        let close = move |r: &mut Approval| -> Result<bool> {
            r.status = if failed { "failed" } else { "resumed" }.into();
            r.outcome = outcome.clone();
            // The conversation is in the session now; the record keeps the
            // decision and how it ended.
            r.messages.clear();
            r.done.clear();
            Ok(true)
        };
        self.store.change(approval_id, &close).await
    }

    /// Give back a claim whose worker died before finishing, so the run can be
    /// resumed again. The approved tool may already have run — check first.
    pub async fn release(&self, approval_id: &str) -> Result<Approval> {
        let timeout = self.claim_timeout;
        let back = move |r: &mut Approval| -> Result<bool> {
            if r.status != "resuming" {
                return Err(approval_error(format!("approval {} is not being resumed ({})", r.id, r.status)));
            }
            let held = now() - r.claimed.unwrap_or(0.0);
            if held < timeout {
                return Err(approval_error(format!("approval {} was claimed {held:.0}s ago; it can be released after {timeout:.0}s", r.id)));
            }
            r.status = if r.calls.iter().any(|c| c.status == "approved") { "approved" } else { "denied" }.into();
            r.claimed = None;
            r.claimed_by.clear();
            Ok(true)
        };
        self.store.change(approval_id, &back).await
    }

    // ---- the human side -----------------------------------------------------
    pub async fn get(&self, approval_id: &str) -> Result<Approval> {
        self.store.load(approval_id).await?.ok_or_else(|| approval_error(format!("no approval '{approval_id}'")))
    }

    /// The requests still waiting for an answer, newest first.
    pub async fn pending(&self, user_id: Option<&str>, tenant_id: Option<&str>, agent: Option<&str>) -> Result<Vec<Approval>> {
        Ok(self.list(None, user_id, tenant_id, agent, 200).await?.into_iter().filter(|a| a.status == "pending" && !a.expired()).collect())
    }

    pub async fn list(&self, status: Option<&str>, user_id: Option<&str>, tenant_id: Option<&str>, agent: Option<&str>, limit: usize) -> Result<Vec<Approval>> {
        Ok(self
            .store
            .list(limit)
            .await?
            .into_iter()
            .filter(|a| {
                a.owned_by(user_id, tenant_id)
                    && agent.is_none_or(|g| a.agent == g)
                    && status.is_none_or(|s| s == if a.expired() { "expired" } else { a.status.as_str() })
            })
            .collect())
    }

    /// Say yes — to every waiting call, or to the one named.
    pub async fn approve(&self, approval_id: &str, by: &str, note: &str, call: Option<&str>) -> Result<Approval> {
        self.decide(approval_id, true, by, note, call).await
    }

    /// Say no. The agent is told who declined and why, and carries on.
    pub async fn deny(&self, approval_id: &str, by: &str, note: &str, call: Option<&str>) -> Result<Approval> {
        self.decide(approval_id, false, by, note, call).await
    }

    async fn decide(&self, approval_id: &str, yes: bool, by: &str, note: &str, call: Option<&str>) -> Result<Approval> {
        if by.trim().is_empty() {
            return Err(approval_error("an approval is given by someone — pass by"));
        }
        let self_approval = self.self_approval;
        let decide = |r: &mut Approval| -> Result<bool> {
            if r.expired() {
                return Err(approval_error(format!("approval {} expired without an answer", r.id)));
            }
            if r.status != "pending" {
                return Err(approval_error(format!("approval {} is already {}", r.id, r.status)));
            }
            if yes && !self_approval && r.user_id.as_deref().is_some_and(|u| !u.is_empty() && u == by) {
                return Err(approval_error("separation of duties: a request cannot be approved by the person it was made for"));
            }
            let mut any = false;
            for c in r.calls.iter_mut().filter(|c| c.status == "pending" && call.is_none_or(|id| id == c.id)) {
                c.status = if yes { "approved" } else { "denied" }.into();
                c.by = by.into();
                c.note = note.into();
                c.decided = Some(now());
                any = true;
            }
            if !any {
                return Err(approval_error(format!("approval {} has no waiting call {}", r.id, call.map_or("None".to_string(), |c| format!("'{c}'")))));
            }
            r.settle();
            Ok(true)
        };
        let decided = self.store.change(approval_id, &decide).await?;
        let note: String = note.chars().take(200).collect();
        self.note(
            by,
            "approval_decided",
            &decided,
            if yes { "approve" } else { "deny" },
            json!({"note": note, "call": call.unwrap_or("all")}).as_object().cloned().unwrap_or_default(),
        );
        Ok(decided)
    }

    /// Mark every unanswered request that is past its time. Returns them —
    /// resume each to let its agent wrap up.
    pub async fn expire(&self) -> Result<Vec<Approval>> {
        let lapse = |h: &mut Approval| -> Result<bool> {
            if !h.expired() {
                return Ok(false);
            }
            h.status = "expired".into();
            Ok(true)
        };
        let mut out = vec![];
        for r in self.store.list(1000).await? {
            if r.expired() {
                out.push(self.store.change(&r.id, &lapse).await?);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paused() -> Approval {
        Approval {
            agent: "support".into(),
            user_id: Some("u1".into()),
            calls: vec![ApprovalCall {
                id: "c1".into(),
                tool: "refund".into(),
                args: json!({"order": "4182"}).as_object().cloned().unwrap(),
                reason: "money".into(),
                status: "pending".into(),
                by: String::new(),
                note: String::new(),
                decided: None,
            }],
            ..Approval::default()
        }
    }

    async fn the_desk(store: Arc<dyn ApprovalStore>) {
        let desk = Approvals { self_approval: false, ..Approvals::new(store) };
        let a = desk.open(paused()).await.unwrap();
        assert_eq!(a.describe(), "support wants to run refund(order=\"4182\")");
        assert!(desk.claim(&a.id, "w", None, None, None).await.unwrap_err().to_string().contains("still waiting for an answer on: refund"));
        assert!(desk.approve(&a.id, "u1", "", None).await.unwrap_err().to_string().starts_with("separation of duties"));
        assert_eq!(desk.pending(Some("u1"), None, None).await.unwrap().len(), 1);
        assert!(desk.pending(Some("u2"), None, None).await.unwrap().is_empty());
        let decided = desk.approve(&a.id, "maria", "ok", None).await.unwrap();
        assert_eq!((decided.status.as_str(), decided.calls[0].by.as_str()), ("approved", "maria"));
        assert!(desk.claim(&a.id, "w", Some("u2"), None, None).await.unwrap_err().to_string().starts_with("no approval"));
        desk.claim(&a.id, "w1", Some("u1"), None, Some("support")).await.unwrap();
        assert!(desk.claim(&a.id, "w2", None, None, None).await.unwrap_err().to_string().contains("already being resumed by w1"));
        assert!(desk.release(&a.id).await.unwrap_err().to_string().contains("can be released after 900s"));
        let done = desk.finish(&a.id, json!({"stop_reason": "end_turn"}).as_object().cloned().unwrap(), false).await.unwrap();
        assert_eq!(done.status, "resumed");
        assert!(desk.claim(&a.id, "w", None, None, None).await.unwrap_err().to_string().contains("was already resumed (resumed)"));
        let lapsed = desk.open(Approval { expires: Some(now() - 1.0), ..paused() }).await.unwrap();
        assert_eq!(desk.expire().await.unwrap()[0].id, lapsed.id);
    }

    #[tokio::test]
    async fn every_store_keeps_the_desk_honest() {
        the_desk(Arc::new(MemoryApprovalStore::new())).await;
        let dir = std::env::temp_dir().join(format!("ah_apr_{}", uuid::Uuid::new_v4().simple()));
        the_desk(Arc::new(FileApprovalStore::new(&dir).unwrap())).await;
        let _ = std::fs::remove_dir_all(dir);
        the_desk(Arc::new(SessionApprovalStore::new(Arc::new(crate::runtime::InMemorySessionStore::new())))).await;
    }

    #[tokio::test]
    async fn a_stale_save_is_refused() {
        let store = MemoryApprovalStore::new();
        let a = store.create(paused()).await.unwrap();
        store.save(a.clone()).await.unwrap();
        assert!(matches!(store.save(a).await, Err(Error::SessionConflict { .. })));
    }
}
