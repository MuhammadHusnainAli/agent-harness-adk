//! Sessions in a relational database: SQLite, PostgreSQL, MySQL / MariaDB.
//!
//! One row per conversation. Who it belongs to and when it last changed are
//! real, indexed columns, so "this user's chats, newest first" is one query the
//! database answers from an index; the conversation itself is one JSON
//! document beside them.
//!
//! A save is a single statement that only matches the version the session was
//! loaded at. Each save also leaves its own mark on the row, and the save is
//! confirmed by reading that mark back — which works the same on every driver,
//! whatever it reports about rows affected.
#![cfg(any(feature = "sqlite", feature = "postgres", feature = "mysql"))]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;

use crate::errors::{Error, Result};
use crate::memory::providers::sql::{SqlDriver, SqlRow, SqlValue, create_index, marks, valid_name};
use crate::runtime::session::{DurableBackend, ListFilter, Session};
use crate::types::new_id;

/// The columns beside the id, in the order every insert and update uses.
const COLUMNS: [&str; 9] = ["agent", "user_id", "tenant_id", "title", "created", "updated", "version", "rev", "payload"];

/// Sessions in a table, over any of the SQL memory backends.
///
/// ```no_run
/// # #[cfg(feature = "sqlite")] {
/// # use agent_harness::memory::providers::sqlite::SqliteMemory;
/// let memory = SqliteMemory::open(".harness/memory.db").unwrap();
/// let sessions = memory.session_store("agent_sessions").unwrap();
/// # }
/// ```
///
/// It borrows the backend's connection pool and dialect, so a harness whose
/// memory is already in Postgres keeps its chats there on the same connections.
#[derive(Debug)]
pub struct SqlSessionStore {
    db: Arc<dyn SqlDriver>,
    pub table: String,
    /// The driver epoch the table was made on (0: not yet).
    ensured: AtomicU64,
}

impl SqlSessionStore {
    /// Sessions in `table` over this driver. The name is checked: it goes
    /// into the SQL as written.
    pub fn new(db: Arc<dyn SqlDriver>, table: &str) -> Result<Self> {
        if !valid_name(table) {
            return Err(Error::config(format!("not a table name: '{table}'")));
        }
        Ok(Self { db, table: table.into(), ensured: AtomicU64::new(0) })
    }

    /// Release the connections — the memory backend's too, since they are shared.
    pub async fn close(&self) {
        self.db.close().await;
    }

    async fn ensure(&self) -> Result<()> {
        let epoch = self.db.epoch();
        if self.ensured.load(Ordering::Acquire) == epoch {
            return Ok(());
        }
        let (db, t) = (&*self.db, &self.table);
        let (key, txt, flt) = (db.key_type(), db.text_type(), db.float_type());
        db.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {t} (
            id {key} PRIMARY KEY,
            agent {key},
            user_id {key},
            tenant_id {key},
            title {txt},
            created {flt} NOT NULL,
            updated {flt} NOT NULL,
            version INTEGER NOT NULL,
            rev {key} NOT NULL,
            payload {txt} NOT NULL
        )"
            ),
            vec![],
        )
        .await?;
        // Listing is by owner, newest first.
        create_index(db, &format!("CREATE INDEX IF NOT EXISTS {t}_owner_idx ON {t} (tenant_id, user_id, updated)")).await?;
        self.ensured.store(epoch, Ordering::Release);
        Ok(())
    }

    fn values(session: &Session, rev: &str) -> Result<Vec<SqlValue>> {
        let payload = serde_json::to_string(session).map_err(|e| Error::Io(e.to_string()))?;
        Ok(vec![
            (&session.agent).into(),
            session.user_id.clone().into(),
            session.tenant_id.clone().into(),
            (&session.title).into(),
            session.created.into(),
            session.updated.into(),
            SqlValue::Int(session.version as i64),
            rev.into(),
            payload.into(),
        ])
    }

    /// The mark the last save left on the row, if there is a row.
    async fn mark(&self, session_id: &str) -> Result<Option<String>> {
        let rows = self.db.fetch(&self.db.q(&format!("SELECT rev FROM {} WHERE id = ?", self.table)), vec![session_id.into()]).await?;
        Ok(rows.first().and_then(|r| r.get("rev")).and_then(SqlValue::as_str).map(str::to_string))
    }

    /// The column rules over the version in the JSON.
    fn session(row: &SqlRow) -> Option<Session> {
        let payload = row.get("payload")?.as_str()?;
        let mut found: Session = serde_json::from_str(payload).ok()?;
        if let Some(v) = row.get("version").and_then(SqlValue::as_i64).filter(|v| *v != 0) {
            found.version = v as u64;
        }
        Some(found)
    }
}

#[async_trait]
impl DurableBackend for SqlSessionStore {
    async fn read(&self, session_id: &str) -> Result<Option<Session>> {
        self.ensure().await?;
        let rows = self.db.fetch(&self.db.q(&format!("SELECT payload, version FROM {} WHERE id = ?", self.table)), vec![session_id.into()]).await?;
        Ok(rows.first().and_then(Self::session))
    }

    async fn write(&self, session: &Session, expect: u64) -> Result<bool> {
        self.ensure().await?;
        let (db, t) = (&*self.db, &self.table);
        let rev = new_id("rev");
        let mut values = Self::values(session, &rev)?;
        if expect == 0 {
            let sql = format!("INSERT INTO {t} (id, {}) VALUES ({})", COLUMNS.join(", "), marks(COLUMNS.len() + 1));
            values.insert(0, (&session.id).into());
            if let Err(e) = db.execute(&db.q(&sql), values).await {
                // Already there — somebody created it first — or a real failure.
                if self.mark(&session.id).await?.is_none() {
                    return Err(e);
                }
                return Ok(false);
            }
        } else {
            let assignments: Vec<String> = COLUMNS.iter().map(|c| format!("{c} = ?")).collect();
            let sql = format!("UPDATE {t} SET {} WHERE id = ? AND version = ?", assignments.join(", "));
            values.push((&session.id).into());
            values.push(SqlValue::Int(expect as i64));
            db.execute(&db.q(&sql), values).await?;
        }
        Ok(self.mark(&session.id).await?.as_deref() == Some(rev.as_str()))
    }

    async fn list_sessions(&self, filter: &ListFilter) -> Result<Vec<Session>> {
        self.ensure().await?;
        let mut clauses: Vec<String> = Vec::new();
        let mut params: Vec<SqlValue> = Vec::new();
        for (column, value) in [("tenant_id", &filter.tenant_id), ("user_id", &filter.user_id), ("agent", &filter.agent)] {
            if let Some(v) = value {
                clauses.push(format!("{column} = ?"));
                params.push(v.into());
            }
        }
        let clause = if clauses.is_empty() { String::new() } else { format!(" WHERE {}", clauses.join(" AND ")) };
        params.push(SqlValue::Int(filter.limit as i64));
        let sql = format!("SELECT payload, version FROM {}{clause} ORDER BY updated DESC LIMIT ?", self.table);
        let rows = self.db.fetch(&self.db.q(&sql), params).await?;
        Ok(rows.iter().filter_map(Self::session).collect())
    }

    async fn remove(&self, session_id: &str) -> Result<()> {
        self.ensure().await?;
        self.db.execute(&self.db.q(&format!("DELETE FROM {} WHERE id = ?", self.table)), vec![session_id.into()]).await
    }

    fn backend_name(&self) -> &'static str {
        "SQLSessionStore"
    }
}
