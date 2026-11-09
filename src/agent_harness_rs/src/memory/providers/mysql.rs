//! MySQL and MariaDB memory, over `sqlx`.
#![cfg(feature = "mysql")]

use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use sqlx::mysql::{MySqlArguments, MySqlConnectOptions, MySqlPool, MySqlPoolOptions, MySqlRow};
use sqlx::query::Query;
use sqlx::{AssertSqlSafe, MySql, Row};
use tokio::sync::Mutex;

use crate::errors::{Error, Result};
use crate::memory::providers::sql::{ParamStyle, SqlDriver, SqlMemory, SqlRow, SqlValue, db_error};

/// A connection pool over one MySQL database.
///
/// ```no_run
/// # use agent_harness::memory::providers::mysql::MySqlMemory;
/// let store = MySqlMemory::from_url("mysql://user:pass@host:3306/agents").unwrap();
/// ```
///
/// Note the 191-character key columns: that is the largest index a utf8mb4
/// column can take on older MySQL, and it is plenty for an id. Nothing
/// connects until the first call.
pub type MySqlMemory = SqlMemory<MySqlDriver>;

impl SqlMemory<MySqlDriver> {
    /// A pool over the database `url` names (`mysql://` or `mariadb://`).
    /// Whatever the URL leaves out is `root@localhost:3306/agents`.
    pub fn from_url(url: &str) -> Result<Self> {
        Ok(Self::with_driver(MySqlDriver::from_url(url)?))
    }
}

/// Where to connect, as read from the URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectArgs {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: String,
    pub db: String,
}

impl Default for ConnectArgs {
    fn default() -> Self {
        Self { host: "localhost".into(), port: 3306, user: "root".into(), password: String::new(), db: "agents".into() }
    }
}

impl ConnectArgs {
    /// `mysql://user:pass@host:port/db`; every part may be left out.
    pub fn parse(url: &str) -> Result<Self> {
        let mut args = Self::default();
        let rest = url.split_once("://").map_or(url, |(_, r)| r);
        let rest = rest.split(['?', '#']).next().unwrap_or("");
        let (netloc, path) = rest.split_once('/').unwrap_or((rest, ""));
        let (userinfo, hostport) = match netloc.rsplit_once('@') {
            Some((u, h)) => (Some(u), h),
            None => (None, netloc),
        };
        if let Some(info) = userinfo {
            let (user, password) = info.split_once(':').unwrap_or((info, ""));
            if !user.is_empty() {
                args.user = decode(user);
            }
            if !password.is_empty() {
                args.password = decode(password);
            }
        }
        let (host, port) = if let Some(inner) = hostport.strip_prefix('[') {
            let (h, tail) = inner.split_once(']').unwrap_or((inner, ""));
            (h, tail.strip_prefix(':'))
        } else {
            match hostport.rsplit_once(':') {
                Some((h, p)) => (h, Some(p)),
                None => (hostport, None),
            }
        };
        if !host.is_empty() {
            args.host = host.to_string();
        }
        if let Some(p) = port.filter(|p| !p.is_empty()) {
            args.port = p.parse().map_err(|_| Error::config(format!("'{p}' is not a port")))?;
        }
        if !path.is_empty() {
            args.db = decode(path.trim_start_matches('/'));
        }
        Ok(args)
    }
}

fn decode(text: &str) -> String {
    crate::knowledge::stores::decode(text)
}

/// The pool, opened on first use.
#[derive(Debug)]
pub struct MySqlDriver {
    pub connect_args: ConnectArgs,
    pub pool_size: u32,
    pool: Mutex<Option<MySqlPool>>,
    epoch: AtomicU64,
}

impl MySqlDriver {
    pub fn from_url(url: &str) -> Result<Self> {
        Ok(Self { connect_args: ConnectArgs::parse(url)?, pool_size: 10, pool: Mutex::new(None), epoch: AtomicU64::new(1) })
    }

    async fn pool(&self) -> Result<MySqlPool> {
        let mut held = self.pool.lock().await;
        if let Some(pool) = held.as_ref() {
            return Ok(pool.clone());
        }
        let a = &self.connect_args;
        let options = MySqlConnectOptions::new().host(&a.host).port(a.port).username(&a.user).password(&a.password).database(&a.db);
        let pool = MySqlPoolOptions::new().min_connections(1).max_connections(self.pool_size).connect_with(options).await.map_err(|e| db_error("mysql", e))?;
        *held = Some(pool.clone());
        Ok(pool)
    }
}

fn bind(sql: &str, params: Vec<SqlValue>) -> Query<'static, MySql, MySqlArguments> {
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

fn row(row: &MySqlRow) -> SqlRow {
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
        } else if let Ok(v) = row.try_get::<Option<Vec<u8>>, _>(i) {
            v.map(|b| String::from_utf8_lossy(&b).into_owned()).into()
        } else {
            SqlValue::Null
        };
        out.insert(column.name().to_string(), value);
    }
    out
}

#[async_trait]
impl SqlDriver for MySqlDriver {
    fn name(&self) -> &'static str {
        "mysql"
    }

    /// `sqlx` takes `?` on MySQL (the Python driver, `aiomysql`, takes `%s`).
    fn paramstyle(&self) -> ParamStyle {
        ParamStyle::Qmark
    }

    fn text_type(&self) -> &'static str {
        "LONGTEXT"
    }

    fn float_type(&self) -> &'static str {
        "DOUBLE"
    }

    fn key_type(&self) -> &'static str {
        "VARCHAR(191)"
    }

    fn upsert_clause(&self) -> Option<&'static str> {
        Some("ON DUPLICATE KEY UPDATE text = VALUES(text), updated = VALUES(updated)")
    }

    async fn execute(&self, sql: &str, params: Vec<SqlValue>) -> Result<()> {
        let pool = self.pool().await?;
        bind(sql, params).execute(&pool).await.map_err(|e| db_error("mysql", e))?;
        Ok(())
    }

    async fn fetch(&self, sql: &str, params: Vec<SqlValue>) -> Result<Vec<SqlRow>> {
        let pool = self.pool().await?;
        let rows = bind(sql, params).fetch_all(&pool).await.map_err(|e| db_error("mysql", e))?;
        Ok(rows.iter().map(row).collect())
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
        format!("MySQL at {}:{}/{}", self.connect_args.host, self.connect_args.port, self.connect_args.db)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::base::{MemoryRecord, MemoryStore, RecordQuery};
    use crate::memory::trace::Trace;
    use crate::runtime::session::SessionStore;
    use crate::types::new_id;

    #[test]
    fn reads_its_connection_details_from_a_url() {
        let store = MySqlMemory::from_url("mysql://bob:s3cret@db.internal:3307/agents").unwrap();
        let want = ConnectArgs { host: "db.internal".into(), port: 3307, user: "bob".into(), password: "s3cret".into(), db: "agents".into() };
        assert_eq!(store.driver().connect_args, want);
        assert_eq!(ConnectArgs::parse("mariadb://h").unwrap(), ConnectArgs { host: "h".into(), ..ConnectArgs::default() });
        assert_eq!(ConnectArgs::parse("mysql://a%40b:p%3Aw@[::1]/d").unwrap().password, "p:w");
        assert_eq!(ConnectArgs::parse("mysql://a%40b:p%3Aw@[::1]/d").unwrap().user, "a@b");
        assert!(ConnectArgs::parse("mysql://h:nope/d").is_err());
        assert!(crate::sessions::session_provider("mysql://u:p@h/db").is_ok());
    }

    #[test]
    fn keys_stay_indexable() {
        let store = MySqlMemory::from_url("mysql://h/db").unwrap();
        assert!(store.driver().key_type().contains("191"));
        assert!(store.schema().iter().any(|s| s.contains("191")));
        assert!(store.driver().q("a = ?").ends_with('?'));
    }

    /// Runs against a real database when `AH_LIVE_MYSQL_URL` is set.
    #[tokio::test]
    async fn a_real_mysql_holds_the_contract() {
        let Ok(url) = std::env::var("AH_LIVE_MYSQL_URL") else { return };
        let table = format!("ah_test_{}", new_id(""));
        let store = MySqlMemory::from_url(&url).unwrap().table(&table).unwrap();
        let alice = Trace::user("alice");
        let mut r = MemoryRecord::new("user", "fact", "Bill in EUR ✓");
        alice.stamp(&mut r);
        store.append(r.clone()).await.unwrap();
        store.append(MemoryRecord::new("user", "fact", "shared")).await.unwrap();
        let mine = store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap();
        assert_eq!(mine.len(), 2);
        assert!(mine.iter().any(|m| m == &r));
        store.write_doc("user.md", "a", Some(&alice)).await.unwrap();
        store.write_doc("user.md", "b", Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "b");
        store.clear(None, Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "");
        let sessions = store.session_store(&format!("{table}_sessions")).unwrap();
        let report = sessions.check().await;
        assert_eq!(report["ok"], true, "{report}");
        for t in [table.clone(), format!("{table}_docs"), format!("{table}_sessions")] {
            store.driver().execute(&format!("DROP TABLE IF EXISTS {t}"), vec![]).await.unwrap();
        }
        store.close().await;
    }
}
