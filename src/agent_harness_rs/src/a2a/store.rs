//! Where A2A tasks are kept — so any replica can answer for any task.
//!
//! A task outlives the request that started it: a client sends a message to one
//! replica and asks another how it went. So a task's record lives in a store,
//! not in the process that is working on it.
//!
//! By default that store is the harness's own session store — the one chats
//! are kept in. Name a database there and tasks are in it too, versioned the
//! same way:
//!
//! ```no_run
//! # async fn demo(agent: agent_harness::Agent) -> agent_harness::Result<()> {
//! // agent built on Harness::on("postgresql://user:pass@host/agents")
//! let server = agent_harness::a2a::A2AServer::builder(agent).build()?;  // tasks shared by every replica
//! # Ok(()) }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::Mutex;

use crate::errors::{Error, Result};
use crate::runtime::{Session, SessionStore};
use crate::types::now;

/// One A2A task, and what the server needs to know about it besides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TaskRecord {
    pub id: String,
    /// Which of the server's agents it was sent to.
    pub agent: String,
    /// Whose it is. Another caller asking for it is told there is no such task.
    pub user_id: Option<String>,
    pub tenant_id: Option<String>,
    /// The task as the protocol sends it.
    pub task: Value,
    /// Where to send word when it is over.
    pub push: Vec<Value>,
    /// Set by a replica that was asked to cancel a task it is not running; the
    /// one that is running it stops at its next step.
    pub cancel_requested: bool,
    pub created: f64,
    /// When it last changed state — what a lost worker is detected by.
    pub updated: f64,
    #[serde(skip_serializing)]
    pub version: u64,
}

impl Default for TaskRecord {
    fn default() -> Self {
        let t = now();
        Self {
            id: String::new(),
            agent: String::new(),
            user_id: None,
            tenant_id: None,
            task: Value::Object(Map::new()),
            push: vec![],
            cancel_requested: false,
            created: t,
            updated: t,
            version: 0,
        }
    }
}

impl TaskRecord {
    /// The task's state: `submitted`, `working`, `completed`…, or `unknown`.
    pub fn state(&self) -> &str {
        self.task.get("status").and_then(|s| s.get("state")).and_then(Value::as_str).unwrap_or("unknown")
    }

    pub fn owned_by(&self, user_id: Option<&str>, tenant_id: Option<&str>) -> bool {
        self.user_id.as_deref() == user_id && self.tenant_id.as_deref() == tenant_id
    }
}

fn conflict(task_id: &str, message: String) -> Error {
    Error::SessionConflict { message, session_id: task_id.to_string() }
}

/// Three methods keep tasks. `save` must refuse a stale write.
#[async_trait]
pub trait TaskStore: Send + Sync {
    async fn load(&self, task_id: &str) -> Result<Option<TaskRecord>>;

    /// Store a new task. False if one with this id is already there.
    async fn create(&self, record: &mut TaskRecord) -> Result<bool>;

    /// Store a changed task. `SessionConflict` if it has been saved by someone
    /// else since it was loaded.
    async fn save(&self, record: &mut TaskRecord) -> Result<()>;

    async fn delete(&self, _task_id: &str) -> Result<()> {
        Ok(())
    }

    /// Load, edit, save — again if someone else saved in between. `edit`
    /// returns false to leave the task as it is.
    async fn change(&self, task_id: &str, edit: &mut (dyn for<'r> FnMut(&'r mut TaskRecord) -> bool + Send)) -> Result<Option<TaskRecord>> {
        for _ in 0..8 {
            let Some(mut record) = self.load(task_id).await? else { return Ok(None) };
            if !edit(&mut record) {
                return Ok(Some(record));
            }
            match self.save(&mut record).await {
                Ok(()) => return Ok(Some(record)),
                Err(Error::SessionConflict { .. }) => continue,
                Err(e) => return Err(e),
            }
        }
        Err(conflict(task_id, format!("task {task_id} could not be saved: it kept changing")))
    }
}

/// In this process, gone with it. For one replica, and for tests.
#[derive(Debug, Default)]
pub struct MemoryTaskStore {
    tasks: Mutex<HashMap<String, TaskRecord>>,
}

impl MemoryTaskStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl TaskStore for MemoryTaskStore {
    async fn load(&self, task_id: &str) -> Result<Option<TaskRecord>> {
        Ok(self.tasks.lock().await.get(task_id).cloned())
    }

    async fn create(&self, record: &mut TaskRecord) -> Result<bool> {
        let mut tasks = self.tasks.lock().await;
        if tasks.contains_key(&record.id) {
            return Ok(false);
        }
        record.version = 1;
        tasks.insert(record.id.clone(), record.clone());
        Ok(true)
    }

    async fn save(&self, record: &mut TaskRecord) -> Result<()> {
        let mut tasks = self.tasks.lock().await;
        if let Some(held) = tasks.get(&record.id)
            && held.version != record.version
        {
            return Err(conflict(&record.id, format!("task {} was saved by someone else", record.id)));
        }
        record.version += 1;
        tasks.insert(record.id.clone(), record.clone());
        Ok(())
    }

    async fn delete(&self, task_id: &str) -> Result<()> {
        self.tasks.lock().await.remove(task_id);
        Ok(())
    }
}

/// Tasks kept in a session store: any database the harness keeps chats in.
///
/// Each task is one session, under an id and an agent name of its own
/// (`a2a_task_<id>`, `<agent>#a2a`), so it is never listed among an agent's
/// chats. The record is the session's `metadata["a2a"]`, as Python keeps it.
#[derive(Clone)]
pub struct SessionTaskStore {
    pub sessions: Arc<dyn SessionStore>,
    pub prefix: String,
}

impl std::fmt::Debug for SessionTaskStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionTaskStore").field("store", &self.sessions.kind()).field("prefix", &self.prefix).finish()
    }
}

impl SessionTaskStore {
    pub fn new(sessions: Arc<dyn SessionStore>) -> Self {
        Self { sessions, prefix: "a2a_task_".into() }
    }

    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.into();
        self
    }

    fn session(&self, record: &TaskRecord) -> Session {
        let mut metadata = Map::new();
        metadata.insert("a2a".into(), serde_json::to_value(record).unwrap_or_default());
        Session {
            id: format!("{}{}", self.prefix, record.id),
            agent: format!("{}#a2a", record.agent),
            title: format!("A2A task {}", record.id),
            user_id: record.user_id.clone(),
            tenant_id: record.tenant_id.clone(),
            version: record.version,
            created: record.created,
            metadata,
            ..Session::default()
        }
    }
}

#[async_trait]
impl TaskStore for SessionTaskStore {
    async fn load(&self, task_id: &str) -> Result<Option<TaskRecord>> {
        let session = match self.sessions.load(&format!("{}{task_id}", self.prefix)).await {
            Ok(s) => s,
            Err(Error::Configuration(_)) => return Ok(None),
            Err(e) => return Err(e),
        };
        let Some(held) = session.metadata.get("a2a").filter(|v| v.is_object()) else { return Ok(None) };
        let mut record: TaskRecord = serde_json::from_value(held.clone()).map_err(|e| Error::config(format!("task {task_id} could not be read: {e}")))?;
        record.version = session.version;
        Ok(Some(record))
    }

    async fn create(&self, record: &mut TaskRecord) -> Result<bool> {
        record.version = 0;
        let mut session = self.session(record);
        match self.sessions.save(&mut session).await {
            Ok(()) => {
                record.version = session.version;
                Ok(true)
            }
            Err(Error::SessionConflict { .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }

    async fn save(&self, record: &mut TaskRecord) -> Result<()> {
        let mut session = self.session(record);
        self.sessions.save(&mut session).await?;
        record.version = session.version;
        Ok(())
    }

    async fn delete(&self, task_id: &str) -> Result<()> {
        self.sessions.delete(&format!("{}{task_id}", self.prefix)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{InMemorySessionStore, ListFilter};
    use serde_json::json;

    async fn exercise(store: &dyn TaskStore) {
        let mut record = TaskRecord { id: "t1".into(), agent: "desk".into(), task: json!({"status": {"state": "submitted"}}), ..Default::default() };
        assert!(store.create(&mut record).await.unwrap());
        assert!(!store.create(&mut record.clone()).await.unwrap());
        let mut a = store.load("t1").await.unwrap().unwrap();
        let mut b = store.load("t1").await.unwrap().unwrap();
        assert_eq!(a.state(), "submitted");
        a.task["status"]["state"] = json!("working");
        store.save(&mut a).await.unwrap();
        assert!(matches!(store.save(&mut b).await, Err(Error::SessionConflict { .. })));
        let changed = store
            .change("t1", &mut |r| {
                r.cancel_requested = true;
                true
            })
            .await
            .unwrap()
            .unwrap();
        assert!(changed.cancel_requested);
        let left = store.change("t1", &mut |_| false).await.unwrap().unwrap();
        assert_eq!(left.version, changed.version);
        assert!(store.change("ghost", &mut |_| true).await.unwrap().is_none());
        assert!(store.load("ghost").await.unwrap().is_none());
        store.delete("t1").await.unwrap();
        assert!(store.load("t1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn memory_store_refuses_stale_writes() {
        exercise(&MemoryTaskStore::new()).await;
    }

    #[tokio::test]
    async fn session_store_keeps_tasks_apart_from_chats() {
        let sessions: Arc<dyn SessionStore> = Arc::new(InMemorySessionStore::new());
        let store = SessionTaskStore::new(sessions.clone());
        exercise(&store).await;
        let mut r = TaskRecord { id: "t2".into(), agent: "desk".into(), user_id: Some("ada".into()), ..Default::default() };
        store.create(&mut r).await.unwrap();
        let held = sessions.load("a2a_task_t2").await.unwrap();
        assert_eq!((held.agent.as_str(), held.title.as_str()), ("desk#a2a", "A2A task t2"));
        assert_eq!(held.user_id.as_deref(), Some("ada"));
        let kept = held.metadata["a2a"].as_object().unwrap();
        assert!(!kept.contains_key("version"));
        assert_eq!(kept.keys().next().unwrap(), "id");
        assert!(sessions.list(&ListFilter { agent: Some("desk".into()), ..ListFilter::limit(10) }).await.unwrap().is_empty());
    }
}
