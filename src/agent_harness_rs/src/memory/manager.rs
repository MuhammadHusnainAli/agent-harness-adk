//! The four memory scopes, and the manager that wires them to an agent.
//!
//! - **user** — preferences, standards, settled decisions: loaded in full,
//!   permanent, rewritten at the end of every session.
//! - **session** — the whole conversation plus the artefacts it produced: in
//!   full, this session only.
//! - **orchestrator** — plans, staffing decisions, spend, findings: a *digest*
//!   only, never the whole history.
//! - **sub-agent** — only the resources its task produced: nothing carried in,
//!   every sub-agent starts clean, and it dies with the task.

use std::sync::{Arc, Mutex, RwLock};

use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::context::Summarizer;
use crate::errors::Result;
use crate::memory::base::{InMemoryStore, MemoryRecord, MemoryStore};
use crate::memory::semantic::{Embedder, MemoryIndex, SemanticMemory};
use crate::memory::trace::Trace;
use crate::prompts::Prompt;
use crate::tools::Tool;
use crate::types::{Artifact, Message, new_id, now};

const DISTILL: &str = "You maintain a long-lived memory file about one user.

Rewrite the file below so the next session starts better informed than this one did.

Rules:
- Keep preferences, standards, and decisions that are already settled.
- Fold in anything new from this session's summary; drop what it contradicts.
- Keep it under {max_chars} characters. Bullets, no preamble, no commentary.
- Do not record one-off task details, secrets, or anything that will be stale tomorrow.

CURRENT FILE
{current}

THIS SESSION
{summary}

Return the complete new file and nothing else.";

/// The prompt that folds a session into `user.md`.
pub fn distill_prompt() -> Prompt {
    Prompt::new("memory.distill", DISTILL)
}

/// Called with every durable write before it is stored — `(text, scope, kind)`
/// — and returns the text to keep (possibly rewritten), or an error to refuse it.
pub type WriteHook = Arc<dyn Fn(String, String, String) -> BoxFuture<'static, Result<String>> + Send + Sync>;

fn read<T: Clone>(lock: &RwLock<T>) -> T {
    lock.read().unwrap_or_else(|p| p.into_inner()).clone()
}

fn chars(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

/// `user.md` — what is true across sessions. Loaded in full, every message.
#[derive(Clone)]
pub struct UserMemory {
    pub store: Arc<dyn MemoryStore>,
    pub doc: String,
    pub max_chars: usize,
    trace: Arc<RwLock<Trace>>,
}

impl UserMemory {
    pub fn new(store: Arc<dyn MemoryStore>, trace: Trace) -> Self {
        Self { store, doc: "user.md".into(), max_chars: 8000, trace: Arc::new(RwLock::new(trace)) }
    }

    fn trace(&self) -> Trace {
        read(&self.trace)
    }

    pub async fn load(&self) -> Result<String> {
        Ok(self.store.read_doc(&self.doc, Some(&self.trace())).await?.trim().to_string())
    }

    pub async fn write(&self, text: &str) -> Result<()> {
        self.store.write_doc(&self.doc, &chars(text.trim(), self.max_chars), Some(&self.trace())).await
    }

    /// Record a durable fact and append it to the document.
    pub async fn remember(&self, text: &str, kind: &str, tags: Vec<String>) -> Result<MemoryRecord> {
        let mut record = MemoryRecord::new("user", kind, text.trim()).tags(tags);
        self.trace().stamp(&mut record);
        let record = self.store.append(record).await?;
        let current = self.load().await?;
        let line = record.line();
        if !current.contains(&line) {
            self.write(&if current.is_empty() { line } else { format!("{current}\n{line}") }).await?;
        }
        Ok(record)
    }

    pub async fn block(&self) -> Result<String> {
        let text = self.load().await?;
        Ok(if text.is_empty() { String::new() } else { format!("## What you already know about this user\n{text}") })
    }

    /// Session close → user memory update. Falls back to appending with no model.
    pub async fn distill(&self, summary: &str, summarize: Option<&Summarizer>) -> Result<String> {
        let current = self.load().await?;
        if summary.trim().is_empty() {
            return Ok(current);
        }
        let Some(summarize) = summarize else {
            let today = chrono_date();
            self.write(&format!("{current}\n\n## Session {today}\n{summary}")).await?;
            return self.load().await;
        };
        let prompt = distill_prompt()
            .render(&json!({"current": if current.is_empty() { "(empty)" } else { &current }, "summary": summary, "max_chars": self.max_chars}))?;
        let rewritten = summarize(prompt).await;
        if !rewritten.trim().is_empty() {
            self.write(&rewritten).await?;
        }
        self.load().await
    }
}

/// Today as `YYYY-MM-DD`, in UTC.
fn chrono_date() -> String {
    let days = (now() / 86_400.0).floor() as i64;
    // Civil-from-days (Howard Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// The live conversation and everything it produced. Dies with the session.
#[derive(Debug, Clone)]
pub struct SessionMemory {
    pub id: String,
    pub messages: Vec<Message>,
    pub artifacts: Vec<Artifact>,
    pub started: f64,
    /// Facts that must survive compaction.
    pub facts: Vec<String>,
}

impl Default for SessionMemory {
    fn default() -> Self {
        Self::new(None)
    }
}

impl SessionMemory {
    pub fn new(session_id: Option<String>) -> Self {
        Self { id: session_id.unwrap_or_else(|| new_id("ses")), messages: vec![], artifacts: vec![], started: now(), facts: vec![] }
    }

    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn add_artifact(&mut self, artifact: Artifact) {
        self.artifacts.push(artifact);
    }

    /// A fact that must survive compaction.
    pub fn pin(&mut self, fact: &str) {
        if !fact.is_empty() && !self.facts.iter().any(|f| f == fact) {
            self.facts.push(fact.into());
        }
    }

    pub fn transcript(&self, limit: Option<usize>) -> String {
        let skip = limit.map_or(0, |n| self.messages.len().saturating_sub(n));
        self.messages
            .iter()
            .skip(skip)
            .filter_map(|m| {
                let t = m.text();
                (!t.is_empty()).then(|| format!("{}: {t}", m.role.as_str()))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn digest(&self, limit: usize) -> String {
        let mut parts = Vec::new();
        if !self.facts.is_empty() {
            parts.push(format!("Pinned facts:\n{}", self.facts.iter().map(|f| format!("- {f}")).collect::<Vec<_>>().join("\n")));
        }
        if !self.artifacts.is_empty() {
            let rows: Vec<String> =
                self.artifacts.iter().map(|a| format!("- {} (by {})", a.name, if a.produced_by.is_empty() { "agent" } else { &a.produced_by })).collect();
            parts.push(format!("Artefacts produced:\n{}", rows.join("\n")));
        }
        let tail = self.transcript(Some(limit));
        if !tail.is_empty() {
            parts.push(format!("Recent exchange:\n{tail}"));
        }
        parts.join("\n\n")
    }

    pub fn reset(&mut self) {
        self.messages.clear();
        self.artifacts.clear();
    }
}

/// Plans, staffing calls, spend, findings — read back as a digest, never raw.
#[derive(Clone)]
pub struct OrchestratorMemory {
    pub store: Arc<dyn MemoryStore>,
    pub job_id: String,
    trace: Arc<RwLock<Trace>>,
    local: Arc<Mutex<Vec<MemoryRecord>>>,
}

impl OrchestratorMemory {
    pub fn new(store: Arc<dyn MemoryStore>, trace: Trace) -> Self {
        Self { store, job_id: new_id("job"), trace: Arc::new(RwLock::new(trace)), local: Arc::default() }
    }

    pub async fn note(&self, kind: &str, text: &str, data: Map<String, Value>) -> Result<MemoryRecord> {
        let mut record = MemoryRecord::new("orchestrator", kind, text).data(data);
        record.source = self.job_id.clone();
        read(&self.trace).stamp(&mut record);
        self.local.lock().unwrap_or_else(|p| p.into_inner()).push(record.clone());
        self.store.append(record).await
    }

    pub async fn plan(&self, text: &str, data: Map<String, Value>) -> Result<MemoryRecord> {
        self.note("plan", text, data).await
    }

    pub async fn staffing(&self, task: &str, agent: &str, reused: bool) -> Result<MemoryRecord> {
        let verb = if reused { "reused" } else { "built" };
        let data = json!({"task": task, "agent": agent, "reused": reused}).as_object().cloned().unwrap_or_default();
        self.note("staffing", &format!("{task} → {verb} {agent}"), data).await
    }

    pub async fn finding(&self, text: &str, data: Map<String, Value>) -> Result<MemoryRecord> {
        self.note("finding", text, data).await
    }

    pub async fn spend(&self, agent: &str, usd: f64, tokens: u64) -> Result<MemoryRecord> {
        let data = json!({"agent": agent, "usd": usd, "tokens": tokens}).as_object().cloned().unwrap_or_default();
        self.note("spend", &format!("{agent}: ${usd:.4}"), data).await
    }

    /// A compact read-back. This is the only thing that reaches the model.
    pub fn digest(&self, limit: usize) -> String {
        let local = self.local.lock().unwrap_or_else(|p| p.into_inner());
        if local.is_empty() {
            return String::new();
        }
        let mut lines = Vec::new();
        for kind in ["plan", "staffing", "finding", "spend", "note"] {
            let rows: Vec<&MemoryRecord> = local.iter().filter(|r| r.kind == kind).collect();
            if rows.is_empty() {
                continue;
            }
            if kind == "spend" {
                let total: f64 = rows.iter().map(|r| r.data.get("usd").and_then(Value::as_f64).unwrap_or(0.0)).sum();
                lines.push(format!("spend so far: ${total:.4} across {} sub-agents", rows.len()));
                continue;
            }
            let head = &rows[rows.len().saturating_sub(limit)..];
            lines.push(format!("{kind}:\n{}", head.iter().map(|r| format!("  - {}", r.text)).collect::<Vec<_>>().join("\n")));
        }
        if lines.is_empty() { String::new() } else { format!("## Job so far\n{}", lines.join("\n")) }
    }

    pub fn total_spend(&self) -> f64 {
        let local = self.local.lock().unwrap_or_else(|p| p.into_inner());
        let total: f64 = local.iter().filter(|r| r.kind == "spend").map(|r| r.data.get("usd").and_then(Value::as_f64).unwrap_or(0.0)).sum();
        (total * 1e6).round() / 1e6
    }
}

/// Nothing carried in; only the resources the task produced carried out.
#[derive(Debug, Clone, Default)]
pub struct SubAgentMemory {
    pub task: String,
    pub agent: String,
    pub artifacts: Vec<Artifact>,
    pub notes: Vec<String>,
}

impl SubAgentMemory {
    pub fn new(task: &str, agent: &str) -> Self {
        Self { task: task.into(), agent: agent.into(), ..Self::default() }
    }

    pub fn produce(&mut self, mut artifact: Artifact) -> Artifact {
        if artifact.produced_by.is_empty() {
            artifact.produced_by = self.agent.clone();
        }
        self.artifacts.push(artifact.clone());
        artifact
    }

    pub fn note(&mut self, text: &str) {
        self.notes.push(text.into());
    }

    /// The hand-back: typed artefacts, not a transcript.
    pub fn resources(&self) -> Vec<Artifact> {
        self.artifacts.clone()
    }

    pub fn close(&mut self) -> Vec<Artifact> {
        self.notes.clear();
        std::mem::take(&mut self.artifacts)
    }
}

/// How a [`MemoryManager`] is put together. Every field has a working default.
#[derive(Default)]
pub struct MemoryOptions {
    /// The store underneath. In memory by default.
    pub store: Option<Arc<dyn MemoryStore>>,
    /// Wrap the store in [`SemanticMemory`] (on by default).
    pub semantic: Option<bool>,
    pub embedder: Option<Arc<dyn Embedder>>,
    pub index: Option<MemoryIndex>,
    pub user_doc: Option<String>,
    pub summarize: Option<Summarizer>,
    pub recall_limit: Option<usize>,
    pub trace: Option<Trace>,
    pub session: Option<SessionMemory>,
}

/// One object an agent holds that knows about all four scopes. Cheap to clone;
/// clones share everything.
#[derive(Clone)]
pub struct MemoryManager {
    pub store: Arc<dyn MemoryStore>,
    trace: Arc<RwLock<Trace>>,
    pub user: UserMemory,
    session: Arc<Mutex<SessionMemory>>,
    pub orchestrator: OrchestratorMemory,
    pub summarize: Option<Summarizer>,
    pub recall_limit: usize,
    on_write: Arc<RwLock<Option<WriteHook>>>,
}

impl std::fmt::Debug for MemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<MemoryManager {} trace={}>", self.store.describe(), self.trace())
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(MemoryOptions::default())
    }
}

impl MemoryManager {
    pub fn new(options: MemoryOptions) -> Self {
        let base = options.store.unwrap_or_else(|| Arc::new(InMemoryStore::new()));
        let store: Arc<dyn MemoryStore> = if options.semantic.unwrap_or(true) {
            let mut s = SemanticMemory::new(base);
            if let Some(e) = options.embedder {
                s = s.embedder(e);
            }
            if let Some(i) = options.index {
                s = s.index(i);
            }
            Arc::new(s)
        } else {
            base
        };
        Self::over(store, options.trace.unwrap_or_default(), options.session, options.summarize, options.recall_limit.unwrap_or(4), options.user_doc)
    }

    /// A manager over a store used exactly as it is (no semantic wrapper added).
    pub fn over(
        store: Arc<dyn MemoryStore>,
        mut trace: Trace,
        session: Option<SessionMemory>,
        summarize: Option<Summarizer>,
        recall_limit: usize,
        user_doc: Option<String>,
    ) -> Self {
        // A session id on the trace and the live session should be the same thing.
        let session = session.unwrap_or_else(|| SessionMemory::new(trace.session_id.clone()));
        if trace.session_id.is_none() {
            trace.session_id = Some(session.id.clone());
        }
        let shared = Arc::new(RwLock::new(trace));
        let mut user = UserMemory::new(store.clone(), Trace::default());
        user.trace = shared.clone();
        if let Some(doc) = user_doc {
            user.doc = doc;
        }
        let mut orchestrator = OrchestratorMemory::new(store.clone(), Trace::default());
        orchestrator.trace = shared.clone();
        Self { store, trace: shared, user, session: Arc::new(Mutex::new(session)), orchestrator, summarize, recall_limit, on_write: Arc::default() }
    }

    /// Whose memory this is.
    pub fn trace(&self) -> Trace {
        read(&self.trace)
    }

    /// Change whose memory this is — the session id after `new_session`, say.
    pub fn set_trace(&self, f: impl FnOnce(&mut Trace)) {
        f(&mut self.trace.write().unwrap_or_else(|p| p.into_inner()));
    }

    /// The live session.
    pub fn session(&self) -> SessionMemory {
        self.session.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Change the live session in place.
    pub fn with_session<T>(&self, f: impl FnOnce(&mut SessionMemory) -> T) -> T {
        f(&mut self.session.lock().unwrap_or_else(|p| p.into_inner()))
    }

    /// Start a fresh session. A trace that was following the old one follows
    /// the new one; a session id you set yourself is left alone.
    pub fn new_session(&self) {
        let fresh = SessionMemory::new(None);
        let fresh_id = fresh.id.clone();
        let old = std::mem::replace(&mut *self.session.lock().unwrap_or_else(|p| p.into_inner()), fresh);
        self.set_trace(|t| {
            if t.session_id.as_deref() == Some(old.id.as_str()) {
                t.session_id = Some(fresh_id);
            }
        });
    }

    /// Every durable write passes this first — the agent wires its
    /// `memory_write` hook here.
    pub fn set_on_write(&self, hook: Option<WriteHook>) {
        *self.on_write.write().unwrap_or_else(|p| p.into_inner()) = hook;
    }

    async fn vetted(&self, text: &str, scope: &str, kind: &str) -> Result<String> {
        let hook = read(&self.on_write);
        match hook {
            Some(h) => h(text.into(), scope.into(), kind.into()).await,
            None => Ok(text.into()),
        }
    }

    // ---- scopes -----------------------------------------------------------
    pub fn subagent(&self, task: &str, agent: &str) -> SubAgentMemory {
        SubAgentMemory::new(task, agent)
    }

    pub async fn remember(&self, text: &str, kind: &str, scope: &str, tags: Vec<String>) -> Result<MemoryRecord> {
        let text = self.vetted(text, scope, kind).await?;
        if scope == "user" {
            return self.user.remember(&text, kind, tags).await;
        }
        let mut record = MemoryRecord::new(scope, kind, &text).tags(tags);
        self.trace().stamp(&mut record);
        self.store.append(record).await
    }

    pub async fn recall(&self, query: &str, limit: Option<usize>, scope: Option<&str>) -> Result<Vec<MemoryRecord>> {
        self.store.search(query, scope, limit.filter(|n| *n > 0).unwrap_or(self.recall_limit), Some(&self.trace())).await
    }

    /// The same backend, scoped to somebody else. One store, many users.
    ///
    /// The store is reused as it is — including its vector index — so serving
    /// a request per user costs a small object, not a rebuilt index.
    pub fn for_trace(&self, trace: Trace) -> MemoryManager {
        let doc = Some(self.user.doc.clone());
        Self::over(self.store.clone(), trace, None, self.summarize.clone(), self.recall_limit, doc)
    }

    pub async fn close(&self) {
        self.store.close().await;
    }

    // ---- what reaches the model --------------------------------------------
    /// User memory in full, orchestrator as a digest, precedent if relevant.
    pub async fn prompt_blocks(&self, query: &str) -> Result<Vec<String>> {
        let mut blocks = Vec::new();
        let user = self.user.block().await?;
        if !user.is_empty() {
            blocks.push(user);
        }
        let digest = self.orchestrator.digest(20);
        if !digest.is_empty() {
            blocks.push(digest);
        }
        if !query.is_empty() {
            let recalled: Vec<String> = self.recall(query, None, None).await?.into_iter().filter(|h| h.scope != "user").map(|h| h.line()).collect();
            if !recalled.is_empty() {
                blocks.push(format!("## Relevant precedent\n{}", recalled.join("\n")));
            }
        }
        Ok(blocks)
    }

    /// Session close → user memory update. Returns the new `user.md`.
    pub async fn close_session(&self, summary: Option<&str>) -> Result<String> {
        let text = match summary {
            Some(s) => s.to_string(),
            None => self.session().digest(12),
        };
        let text = self.vetted(&text, "user", "distill").await?;
        self.user.distill(&text, self.summarize.as_ref()).await
    }

    // ---- tools the agent can call itself -----------------------------------
    /// `remember` and `recall`, tagged `builtin` and `memory`.
    pub fn tools(&self) -> Vec<Tool> {
        let m = self.clone();
        let remember = Tool::new("remember", move |a: RememberArgs, _| {
            let m = m.clone();
            async move {
                let record = m.remember(&a.fact, &a.kind, "user", vec![]).await?;
                Ok(format!("remembered: {}", record.text))
            }
        })
        .tags(["builtin", "memory"]);
        let m = self.clone();
        let recall = Tool::new("recall", move |a: RecallArgs, _| {
            let m = m.clone();
            async move {
                let hits = m.recall(&a.query, Some(a.limit), None).await?;
                let lines: Vec<String> = hits.iter().map(MemoryRecord::line).collect();
                Ok(if lines.is_empty() { "nothing on record".to_string() } else { lines.join("\n") })
            }
        })
        .tags(["builtin", "memory"]);
        vec![remember, recall]
    }
}

fn fact_kind() -> String {
    "fact".into()
}

fn five() -> usize {
    5
}

/// Save something worth knowing in future sessions.
#[derive(Deserialize, JsonSchema)]
struct RememberArgs {
    /// one durable sentence — a preference, standard or decision.
    fact: String,
    /// preference, standard, decision or fact.
    #[serde(default = "fact_kind")]
    kind: String,
}

/// Search memory for past decisions, findings and preferences.
#[derive(Deserialize, JsonSchema)]
struct RecallArgs {
    /// what you are trying to remember.
    query: String,
    /// how many hits to return.
    #[serde(default = "five")]
    limit: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolContext;

    #[tokio::test]
    async fn the_four_scopes() {
        let m = MemoryManager::new(MemoryOptions { trace: Some(Trace::user("alice")), ..Default::default() });
        let session_id = m.session().id;
        assert_eq!(m.trace().session_id.as_deref(), Some(session_id.as_str()));
        m.remember("Bills in EUR", "preference", "user", vec![]).await.unwrap();
        m.remember("Bills in EUR", "preference", "user", vec![]).await.unwrap();
        assert_eq!(m.user.load().await.unwrap(), "- [preference] Bills in EUR", "a line already there is not added twice");
        m.remember("Vendor B chosen for SLA", "decision", "project", vec![]).await.unwrap();
        m.orchestrator.plan("three tasks", Map::new()).await.unwrap();
        m.orchestrator.spend("researcher", 0.25, 100).await.unwrap();
        m.orchestrator.spend("writer", 0.5, 100).await.unwrap();
        assert_eq!(m.orchestrator.total_spend(), 0.75);

        let blocks = m.prompt_blocks("which vendor was chosen").await.unwrap();
        assert_eq!(blocks[0], "## What you already know about this user\n- [preference] Bills in EUR");
        assert_eq!(blocks[1], "## Job so far\nplan:\n  - three tasks\nspend so far: $0.7500 across 2 sub-agents");
        assert_eq!(blocks[2], "## Relevant precedent\n- [decision] Vendor B chosen for SLA");

        // Bob shares the store but not the memory.
        let bob = m.for_trace(Trace::user("bob"));
        assert_eq!(bob.user.load().await.unwrap(), "");
        assert!(bob.recall("vendor chosen", None, None).await.unwrap().is_empty());

        let tools = m.tools();
        let out = tools[1].invoke(json!({"query": "vendor"}).as_object().cloned().unwrap(), ToolContext::default()).await.unwrap();
        assert_eq!(out, json!("- [decision] Vendor B chosen for SLA"));
        assert_eq!(tools[0].parameters["properties"]["kind"]["default"], "fact");

        m.new_session();
        assert_ne!(m.session().id, session_id);
        assert_eq!(m.trace().session_id, Some(m.session().id));

        // Recall is held to the trace, session included: a new session starts without precedent.
        assert!(m.recall("vendor chosen", None, None).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn writes_pass_the_hook_and_sessions_distill() {
        let m = MemoryManager::default();
        m.set_on_write(Some(Arc::new(|text: String, _s, _k| Box::pin(async move { Ok(text.replace("4111", "****")) }))));
        let r = m.remember("card 4111", "fact", "user", vec![]).await.unwrap();
        assert_eq!(r.text, "card ****");
        m.with_session(|s| {
            s.add_message(Message::user("hi"));
            s.pin("budget is 10k");
        });
        let user_md = m.close_session(None).await.unwrap();
        assert!(user_md.starts_with("- [fact] card ****\n\n## Session 20"), "{user_md}");
        assert!(user_md.contains("Pinned facts:\n- budget is 10k") && user_md.contains("user: hi"));
        let summarize: Summarizer =
            Arc::new(|prompt: String| Box::pin(async move { if prompt.contains("CURRENT FILE") { "- distilled".to_string() } else { String::new() } }));
        let m2 = MemoryManager::new(MemoryOptions { summarize: Some(summarize), ..Default::default() });
        assert_eq!(m2.close_session(Some("did things")).await.unwrap(), "- distilled");
        let mut sub = m2.subagent("t", "w");
        sub.produce(Artifact::new("a.md", "x"));
        assert_eq!(sub.close()[0].produced_by, "w");
    }
}
