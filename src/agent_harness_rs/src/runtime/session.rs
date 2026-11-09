//! Session store: resume, fork or branch a run. A long job survives a restart.
//!
//! A [`Session`] is one conversation: its messages, what it cost, who it
//! belongs to. Every store that outlives the process holds a session under
//! three promises: it knows **whose** it is, it is **versioned** (a stale save
//! is refused with `SessionConflict`), and saving is **all or nothing**.
//!
//! The JSON is the same as the Python harness writes, so a
//! [`FileSessionStore`] directory can be shared by both.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::sync::Mutex;

use crate::errors::{Error, Result};
use crate::types::{Artifact, Message, Usage, new_id, now};

/// Everything needed to pick a conversation back up where it stopped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Session {
    pub id: String,
    pub agent: String,
    pub title: String,
    /// Whose conversation this is.
    pub user_id: Option<String>,
    pub tenant_id: Option<String>,
    /// How many times it has been saved. A store accepts a save only from the
    /// version it holds.
    pub version: u64,
    pub messages: Vec<Message>,
    pub artifacts: Vec<Artifact>,
    pub usage: Usage,
    pub created: f64,
    pub updated: f64,
    pub parent_id: Option<String>,
    pub forked_at: Option<usize>,
    pub metadata: Map<String, Value>,
}

impl Default for Session {
    fn default() -> Self {
        let t = now();
        Self {
            id: new_id("ses"),
            agent: String::new(),
            title: String::new(),
            user_id: None,
            tenant_id: None,
            version: 0,
            messages: vec![],
            artifacts: vec![],
            usage: Usage::default(),
            created: t,
            updated: t,
            parent_id: None,
            forked_at: None,
            metadata: Map::new(),
        }
    }
}

impl Session {
    pub fn new(agent: &str) -> Self {
        Self { agent: agent.into(), ..Default::default() }
    }

    pub fn add(&mut self, message: Message) {
        self.messages.push(message);
        self.updated = now();
    }

    /// Branch a run: same history up to `at`, new id, parent recorded.
    pub fn fork(&self, at: Option<usize>) -> Session {
        let cut = at.map_or(self.messages.len(), |a| a.min(self.messages.len()));
        Session {
            agent: self.agent.clone(),
            title: if self.title.is_empty() { String::new() } else { format!("{} (fork)", self.title) },
            user_id: self.user_id.clone(),
            tenant_id: self.tenant_id.clone(),
            messages: self.messages[..cut].to_vec(),
            artifacts: self.artifacts.clone(),
            parent_id: Some(self.id.clone()),
            forked_at: Some(cut),
            metadata: self.metadata.clone(),
            ..Default::default()
        }
    }

    /// May someone acting as this user, in this tenant, have it? An axis
    /// nobody set is not a wall — that is a single-user setup.
    pub fn owned_by(&self, user_id: Option<&str>, tenant_id: Option<&str>) -> bool {
        let ok = |mine: &Option<String>, theirs: Option<&str>| match (mine.as_deref().filter(|s| !s.is_empty()), theirs.filter(|s| !s.is_empty())) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        };
        ok(&self.user_id, user_id) && ok(&self.tenant_id, tenant_id)
    }

    /// One row for a list of chats — everything but the messages.
    pub fn summary(&self) -> Value {
        json!({"id": self.id, "agent": self.agent, "title": self.title, "user_id": self.user_id,
               "tenant_id": self.tenant_id, "messages": self.messages.len(), "created": self.created,
               "updated": self.updated, "cost_usd": self.usage.cost_usd, "version": self.version})
    }

    /// The messages a resumed run should start from.
    pub fn replay(&self) -> Vec<Message> {
        self.messages.clone()
    }
}

fn conflict(id: &str, version: u64) -> Error {
    Error::SessionConflict {
        message: format!("session {id} was saved by someone else since it was loaded (it is no longer at version {version})"),
        session_id: id.into(),
    }
}

fn missing(id: &str) -> Error {
    Error::config(format!("no session '{id}'"))
}

fn matches(s: &Session, user_id: Option<&str>, tenant_id: Option<&str>, agent: Option<&str>) -> bool {
    user_id.is_none_or(|u| s.user_id.as_deref() == Some(u)) && tenant_id.is_none_or(|t| s.tenant_id.as_deref() == Some(t)) && agent.is_none_or(|a| s.agent == a)
}

/// Narrow a listing to an owner or an agent.
#[derive(Debug, Clone, Default)]
pub struct ListFilter {
    pub limit: usize,
    pub user_id: Option<String>,
    pub tenant_id: Option<String>,
    pub agent: Option<String>,
}

impl ListFilter {
    pub fn limit(limit: usize) -> Self {
        Self { limit, ..Default::default() }
    }

    fn effective_limit(&self) -> usize {
        if self.limit == 0 { 50 } else { self.limit }
    }
}

/// Where conversations are kept. Four methods make a store.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Save, bumping `session.version`; `SessionConflict` if it is stale.
    async fn save(&self, session: &mut Session) -> Result<()>;
    async fn load(&self, session_id: &str) -> Result<Session>;
    /// The newest sessions first.
    async fn list(&self, filter: &ListFilter) -> Result<Vec<Session>>;
    async fn delete(&self, session_id: &str) -> Result<()>;

    async fn fork(&self, session_id: &str, at: Option<usize>) -> Result<Session> {
        let mut forked = self.load(session_id).await?.fork(at);
        self.save(&mut forked).await?;
        Ok(forked)
    }

    async fn resume(&self, session_id: &str) -> Result<Session> {
        self.load(session_id).await
    }

    /// The store's name, for reports.
    fn kind(&self) -> &'static str {
        "SessionStore"
    }

    /// Prove the store does what a chat needs, against the real backend: save,
    /// read back, list for the owner only, refuse a stale save, delete.
    async fn check(&self) -> Value {
        let mut steps: Vec<Value> = Vec::new();
        let mut probe = Session {
            agent: "_check".into(),
            title: "store check".into(),
            user_id: Some("_check_user".into()),
            tenant_id: Some("_check_tenant".into()),
            messages: vec![Message::user("hello"), Message::assistant("héllo ✓")],
            ..Default::default()
        };
        let id = probe.id.clone();
        let mut record = |name: &str, began: Instant, outcome: Result<String>| {
            let seconds = (began.elapsed().as_secs_f64() * 1000.0).round() / 1000.0;
            let ok = outcome.is_ok();
            let detail = match outcome {
                Ok(d) => d,
                Err(e) => format!("{}: {e}", e.kind_name()),
            };
            steps.push(json!({"step": name, "ok": ok, "detail": detail, "seconds": seconds}));
            ok
        };

        let began = Instant::now();
        let saved = self.save(&mut probe).await.map(|_| format!("version {}", probe.version));
        if record("save", began, saved) {
            let began = Instant::now();
            let loaded = async {
                let back = self.load(&id).await?;
                let texts: Vec<String> = back.messages.iter().map(Message::text).collect();
                if texts != ["hello", "héllo ✓"] {
                    return Err(Error::config("the messages did not come back as saved"));
                }
                if back.user_id.as_deref() != Some("_check_user") || back.tenant_id.as_deref() != Some("_check_tenant") {
                    return Err(Error::config("the owner did not come back as saved"));
                }
                Ok(format!("{} messages", back.messages.len()))
            }
            .await;
            record("load", began, loaded);

            let began = Instant::now();
            let listed = async {
                let owner = ListFilter { limit: 5, user_id: Some("_check_user".into()), tenant_id: Some("_check_tenant".into()), agent: None };
                if !self.list(&owner).await?.iter().any(|s| s.id == id) {
                    return Err(Error::config("the session is not listed for its owner"));
                }
                let other = ListFilter { user_id: Some("_somebody_else".into()), ..owner };
                if self.list(&other).await?.iter().any(|s| s.id == id) {
                    return Err(Error::config("the session is listed for someone else"));
                }
                Ok("listed for its owner only".to_string())
            }
            .await;
            record("list by owner", began, listed);

            let began = Instant::now();
            let refused = async {
                let mut stale = self.load(&id).await?;
                let mut fresh = self.load(&id).await?;
                fresh.title = "saved first".into();
                self.save(&mut fresh).await?;
                match self.save(&mut stale).await {
                    Err(Error::SessionConflict { .. }) => Ok("a stale save is refused".to_string()),
                    _ => Err(Error::config("a save from an old version was accepted — two requests on one chat would overwrite each other")),
                }
            }
            .await;
            record("refuse a stale save", began, refused);
        }
        let began = Instant::now();
        let deleted = async {
            self.delete(&id).await?;
            match self.load(&id).await {
                Err(Error::Configuration(_)) => Ok(String::new()),
                _ => Err(Error::config("the session is still there after delete")),
            }
        }
        .await;
        record("delete", began, deleted);
        let ok = steps.iter().all(|s| s["ok"] == true);
        json!({"store": self.kind(), "ok": ok, "steps": steps})
    }
}

/// The shape every store that outlives the process shares. A backend says how
/// to read one, write one *if it is still at the version expected*, list and
/// delete; versioning and the conflict are handled once, by the blanket
/// [`SessionStore`] impl.
#[async_trait]
pub trait DurableBackend: Send + Sync {
    async fn read(&self, session_id: &str) -> Result<Option<Session>>;
    /// Store `session` if what is held is at version `expect` (0: not held).
    /// `false` means somebody else got there first.
    async fn write(&self, session: &Session, expect: u64) -> Result<bool>;
    async fn list_sessions(&self, filter: &ListFilter) -> Result<Vec<Session>>;
    async fn remove(&self, session_id: &str) -> Result<()>;
    fn backend_name(&self) -> &'static str;
}

#[async_trait]
impl<T: DurableBackend> SessionStore for T {
    async fn save(&self, session: &mut Session) -> Result<()> {
        let (expect, was) = (session.version, session.updated);
        session.version = expect + 1;
        session.updated = now();
        match self.write(session, expect).await {
            Ok(true) => Ok(()),
            Ok(false) => {
                session.version = expect;
                session.updated = was;
                Err(conflict(&session.id, expect))
            }
            Err(e) => {
                session.version = expect;
                session.updated = was;
                Err(e)
            }
        }
    }

    async fn load(&self, session_id: &str) -> Result<Session> {
        self.read(session_id).await?.ok_or_else(|| missing(session_id))
    }

    async fn list(&self, filter: &ListFilter) -> Result<Vec<Session>> {
        let f = ListFilter { limit: filter.effective_limit(), ..filter.clone() };
        self.list_sessions(&f).await
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        self.remove(session_id).await
    }

    fn kind(&self) -> &'static str {
        self.backend_name()
    }
}

/// The default. Gone when the process exits.
#[derive(Debug, Default)]
pub struct InMemorySessionStore {
    sessions: Mutex<HashMap<String, Session>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn save(&self, session: &mut Session) -> Result<()> {
        let mut all = self.sessions.lock().await;
        if let Some(held) = all.get(&session.id)
            && held.version != session.version
        {
            return Err(conflict(&session.id, session.version));
        }
        session.version += 1;
        session.updated = now();
        all.insert(session.id.clone(), session.clone());
        Ok(())
    }

    async fn load(&self, session_id: &str) -> Result<Session> {
        self.sessions.lock().await.get(session_id).cloned().ok_or_else(|| missing(session_id))
    }

    async fn list(&self, filter: &ListFilter) -> Result<Vec<Session>> {
        let all = self.sessions.lock().await;
        let mut rows: Vec<Session> =
            all.values().filter(|s| matches(s, filter.user_id.as_deref(), filter.tenant_id.as_deref(), filter.agent.as_deref())).cloned().collect();
        rows.sort_by(|a, b| b.updated.total_cmp(&a.updated));
        rows.truncate(filter.effective_limit());
        Ok(rows)
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        self.sessions.lock().await.remove(session_id);
        Ok(())
    }

    fn kind(&self) -> &'static str {
        "InMemorySessionStore"
    }
}

/// One JSON file per session under `root`. For one machine.
///
/// A save replaces the file in one step, so a reader never finds half of one,
/// and it is refused if the file has moved on since the session was loaded.
#[derive(Debug)]
pub struct FileSessionStore {
    pub root: PathBuf,
    lock: Mutex<()>,
}

impl FileSessionStore {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root, lock: Mutex::new(()) })
    }

    fn path(&self, session_id: &str) -> PathBuf {
        let safe: String = session_id.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
        self.root.join(format!("{safe}.json"))
    }

    fn parse(text: &str) -> Option<Session> {
        serde_json::from_str(text).ok()
    }

    fn held_version(path: &Path) -> u64 {
        std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).and_then(|v| v["version"].as_u64()).unwrap_or(0)
    }
}

#[async_trait]
impl DurableBackend for FileSessionStore {
    async fn read(&self, session_id: &str) -> Result<Option<Session>> {
        let path = self.path(session_id);
        match tokio::fs::read_to_string(&path).await {
            Ok(text) => Ok(Self::parse(&text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    async fn write(&self, session: &Session, expect: u64) -> Result<bool> {
        let payload = serde_json::to_string_pretty(session).map_err(|e| Error::Io(e.to_string()))?;
        let path = self.path(&session.id);
        let _guard = self.lock.lock().await;
        tokio::task::spawn_blocking(move || -> Result<bool> {
            // A file written before sessions were versioned reads as 0.
            if Self::held_version(&path) != expect {
                return Ok(false);
            }
            let scratch = path.with_extension(format!("{}.tmp", new_id("")));
            std::fs::write(&scratch, payload)?;
            std::fs::rename(&scratch, &path)?;
            Ok(true)
        })
        .await
        .map_err(|e| Error::Io(e.to_string()))?
    }

    async fn list_sessions(&self, filter: &ListFilter) -> Result<Vec<Session>> {
        let root = self.root.clone();
        let f = filter.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<Session>> {
            let mut files: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&root)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "json"))
                .map(|p| (std::fs::metadata(&p).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH), p))
                .collect();
            files.sort_by_key(|f| std::cmp::Reverse(f.0));
            let mut out = Vec::new();
            for (_, file) in files {
                if let Some(s) = std::fs::read_to_string(&file).ok().and_then(|t| Self::parse(&t))
                    && matches(&s, f.user_id.as_deref(), f.tenant_id.as_deref(), f.agent.as_deref())
                {
                    out.push(s);
                    if out.len() >= f.limit {
                        break;
                    }
                }
            }
            Ok(out)
        })
        .await
        .map_err(|e| Error::Io(e.to_string()))?
    }

    async fn remove(&self, session_id: &str) -> Result<()> {
        match tokio::fs::remove_file(self.path(session_id)).await {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.into()),
            _ => Ok(()),
        }
    }

    fn backend_name(&self) -> &'static str {
        "FileSessionStore"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fork_and_ownership() {
        let mut s = Session { title: "chat".into(), user_id: Some("u1".into()), ..Session::new("a") };
        s.add(Message::user("1"));
        s.add(Message::assistant("2"));
        let f = s.fork(Some(1));
        assert_eq!(f.messages.len(), 1);
        assert_eq!(f.title, "chat (fork)");
        assert_eq!(f.parent_id.as_deref(), Some(s.id.as_str()));
        assert!(s.owned_by(Some("u1"), None));
        assert!(!s.owned_by(Some("u2"), None));
        assert!(s.owned_by(None, Some("t")));
        assert_eq!(s.summary()["messages"], 2);
    }

    #[test]
    fn reads_python_session_json() {
        let raw = r#"{"id": "ses_abc", "agent": "support", "version": 3, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}], "ts": 1.0}],
            "usage": {"input_tokens": 5, "cost_usd": 0.01}, "metadata": {"k": 1}}"#;
        let s: Session = serde_json::from_str(raw).unwrap();
        assert_eq!(s.version, 3);
        assert_eq!(s.messages[0].text(), "hi");
        assert_eq!(s.usage.input_tokens, 5);
    }

    #[tokio::test]
    async fn memory_store_passes_check() {
        let store = InMemorySessionStore::new();
        let report = store.check().await;
        assert_eq!(report["ok"], true, "{report}");
        assert_eq!(report["steps"].as_array().unwrap().len(), 5);
    }

    #[tokio::test]
    async fn file_store_passes_check_and_lists() {
        let dir = std::env::temp_dir().join(format!("ah-sessions-{}", new_id("")));
        let store = FileSessionStore::new(&dir).unwrap();
        let report = store.check().await;
        assert_eq!(report["ok"], true, "{report}");
        let mut s = Session::new("bot");
        store.save(&mut s).await.unwrap();
        assert_eq!(s.version, 1);
        let forked = store.fork(&s.id, None).await.unwrap();
        assert_eq!(forked.version, 1);
        let rows = store.list(&ListFilter { agent: Some("bot".into()), ..Default::default() }).await.unwrap();
        assert_eq!(rows.len(), 2);
        std::fs::remove_dir_all(dir).ok();
    }
}
