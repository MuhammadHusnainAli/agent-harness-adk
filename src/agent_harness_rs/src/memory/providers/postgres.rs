//! PostgreSQL memory, over `sqlx`.
#![cfg(feature = "postgres")]

use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use sqlx::postgres::{PgArguments, PgConnectOptions, PgPool, PgPoolOptions, PgRow};
use sqlx::query::Query;
use sqlx::{AssertSqlSafe, Postgres, Row};
use tokio::sync::Mutex;

use crate::errors::Result;
use crate::memory::providers::sql::{ParamStyle, SqlDriver, SqlMemory, SqlRow, SqlValue, db_error};

/// A connection pool over one Postgres database.
///
/// ```no_run
/// # use agent_harness::memory::providers::postgres::PostgresMemory;
/// let store = PostgresMemory::from_url("postgresql://user:pass@host/db").unwrap();
/// ```
///
/// Nothing connects until the first call.
pub type PostgresMemory = SqlMemory<PgDriver>;

impl SqlMemory<PgDriver> {
    /// A pool over the database `url` names (`postgres://` or `postgresql://`).
    /// The URL is checked now; the pool opens on first use.
    pub fn from_url(url: &str) -> Result<Self> {
        Ok(Self::with_driver(PgDriver::from_url(url)?))
    }
}

/// The pool, opened on first use.
#[derive(Debug)]
pub struct PgDriver {
    options: PgConnectOptions,
    pub min_size: u32,
    pub max_size: u32,
    pool: Mutex<Option<PgPool>>,
    epoch: AtomicU64,
}

impl PgDriver {
    pub fn from_url(url: &str) -> Result<Self> {
        let options = PgConnectOptions::from_str(url).map_err(|e| db_error("postgres", e))?;
        Ok(Self { options, min_size: 1, max_size: 10, pool: Mutex::new(None), epoch: AtomicU64::new(1) })
    }

    async fn pool(&self) -> Result<PgPool> {
        let mut held = self.pool.lock().await;
        if let Some(pool) = held.as_ref() {
            return Ok(pool.clone());
        }
        let pool = PgPoolOptions::new()
            .min_connections(self.min_size)
            .max_connections(self.max_size)
            .connect_with(self.options.clone())
            .await
            .map_err(|e| db_error("postgres", e))?;
        *held = Some(pool.clone());
        Ok(pool)
    }
}

/// Bind every parameter in order.
pub(crate) fn bind_pg(sql: &str, params: Vec<SqlValue>) -> Query<'static, Postgres, PgArguments> {
    let mut query = sqlx::query(AssertSqlSafe(sql.to_string()));
    for p in params {
        query = match p {
            SqlValue::Null => query.bind(None::<String>),
            SqlValue::Int(i) => query.bind(i),
            SqlValue::Float(f) => query.bind(f),
            SqlValue::Text(s) => query.bind(s),
        };
    }
    query
}

/// A row as column → value, whatever the column types.
pub(crate) fn pg_row(row: &PgRow) -> SqlRow {
    use sqlx::Column;
    let mut out = SqlRow::with_capacity(row.len());
    for (i, column) in row.columns().iter().enumerate() {
        let value = if let Ok(v) = row.try_get::<Option<String>, _>(i) {
            v.into()
        } else if let Ok(v) = row.try_get::<Option<f64>, _>(i) {
            v.into()
        } else if let Ok(v) = row.try_get::<Option<i64>, _>(i) {
            v.into()
        } else if let Ok(v) = row.try_get::<Option<i32>, _>(i) {
            v.map(i64::from).into()
        } else if let Ok(v) = row.try_get::<Option<f32>, _>(i) {
            v.map(f64::from).into()
        } else if let Ok(v) = row.try_get::<Option<i16>, _>(i) {
            v.map(i64::from).into()
        } else if let Ok(v) = row.try_get::<Option<bool>, _>(i) {
            v.map(i64::from).into()
        } else {
            SqlValue::Null
        };
        out.insert(column.name().to_string(), value);
    }
    out
}

#[async_trait]
impl SqlDriver for PgDriver {
    fn name(&self) -> &'static str {
        "postgres"
    }

    fn paramstyle(&self) -> ParamStyle {
        ParamStyle::Numeric
    }

    fn key_type(&self) -> &'static str {
        "TEXT"
    }

    fn upsert_clause(&self) -> Option<&'static str> {
        Some("ON CONFLICT (doc_key) DO UPDATE SET text = EXCLUDED.text, updated = EXCLUDED.updated")
    }

    async fn execute(&self, sql: &str, params: Vec<SqlValue>) -> Result<()> {
        let pool = self.pool().await?;
        bind_pg(sql, params).execute(&pool).await.map_err(|e| db_error("postgres", e))?;
        Ok(())
    }

    async fn fetch(&self, sql: &str, params: Vec<SqlValue>) -> Result<Vec<SqlRow>> {
        let pool = self.pool().await?;
        let rows = bind_pg(sql, params).fetch_all(&pool).await.map_err(|e| db_error("postgres", e))?;
        Ok(rows.iter().map(pg_row).collect())
    }

    fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    async fn close(&self) {
        let pool = self.pool.lock().await.take();
        if let Some(pool) = pool {
            pool.close().await;
            self.epoch.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn describe(&self) -> String {
        format!("PostgreSQL at {}:{}", self.options.get_host(), self.options.get_port())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::base::{MemoryRecord, MemoryStore, RecordQuery};
    use crate::memory::trace::Trace;
    use crate::runtime::session::{Session, SessionStore};
    use crate::types::new_id;

    #[test]
    fn nothing_connects_until_used() {
        let store = PostgresMemory::from_url("postgresql://u:p@db.internal:5433/agents").unwrap();
        assert_eq!(store.describe(), "PostgreSQL at db.internal:5433");
        assert!(store.driver().q("a = ? AND b = ?").ends_with("a = $1 AND b = $2"));
        assert!(store.schema().join(" ").contains("DOUBLE PRECISION"));
        assert!(crate::sessions::session_provider("postgresql://u:p@h/db").is_ok());
    }

    /// Runs against a real database when `AH_LIVE_POSTGRES_URL` is set.
    #[tokio::test]
    async fn a_real_postgres_holds_the_contract() {
        let Ok(url) = std::env::var("AH_LIVE_POSTGRES_URL") else { return };
        let table = format!("ah_test_{}", new_id(""));
        let store = PostgresMemory::from_url(&url).unwrap().table(&table).unwrap();
        let alice = Trace::user("alice");
        let mut r = MemoryRecord::new("user", "fact", "Bill in EUR");
        alice.stamp(&mut r);
        store.append(r.clone()).await.unwrap();
        store.append(MemoryRecord::new("user", "fact", "shared")).await.unwrap();
        let mut bob = MemoryRecord::new("user", "fact", "bob's");
        Trace::user("bob").stamp(&mut bob);
        store.append(bob).await.unwrap();
        assert_eq!(store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap().len(), 2);
        assert_eq!(store.all(RecordQuery::default().limit(1)).await.unwrap().len(), 1);
        store.write_doc("user.md", "a", Some(&alice)).await.unwrap();
        store.write_doc("user.md", "b", Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "b");
        store.clear(None, Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "");
        let sessions = store.session_store(&format!("{table}_sessions")).unwrap();
        let report = sessions.check().await;
        assert_eq!(report["ok"], true, "{report}");
        let mut s = Session::new("bot");
        sessions.save(&mut s).await.unwrap();
        let base = sessions.load(&s.id).await.unwrap();
        let (mut a, mut b) = (base.clone(), base);
        sessions.save(&mut a).await.unwrap();
        assert!(sessions.save(&mut b).await.is_err());
        for t in [table.clone(), format!("{table}_docs"), format!("{table}_sessions")] {
            store.driver().execute(&format!("DROP TABLE IF EXISTS {t}"), vec![]).await.unwrap();
        }
        store.close().await;
    }
}
