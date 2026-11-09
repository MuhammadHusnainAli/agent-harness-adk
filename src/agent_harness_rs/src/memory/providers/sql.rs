//! Shared SQL implementation. Every relational backend is a thin dialect on top.
//!
//! One table for records, one for documents, and an index that matches how the
//! harness actually reads: narrowed by trace first, then scope, then time.
//! Queries are built once with `?` placeholders and converted per dialect, so
//! there is a single place where the SQL can be wrong.
//!
//! The tables, the columns and the JSON in them are the Python harness's, so a
//! database written by one is read by the other.
#![cfg(any(feature = "sqlite", feature = "postgres", feature = "mysql"))]

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::errors::{Error, Result};
use crate::memory::base::{MemoryRecord, MemoryStore, RecordQuery, doc_key};
use crate::memory::trace::Trace;
use crate::runtime::session::SessionStore;
use crate::sessions::sql::SqlSessionStore;
use crate::types::now;

/// The record columns, in the order every insert and select uses.
pub const COLUMNS: [&str; 11] = ["id", "scope", "kind", "text", "data", "tags", "source", "user_id", "session_id", "tenant_id", "ts"];

/// One bound parameter, or one value read back.
#[derive(Debug, Clone, PartialEq)]
pub enum SqlValue {
    Null,
    Int(i64),
    Float(f64),
    Text(String),
}

impl SqlValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Int(i) => Some(*i as f64),
            Self::Text(s) => s.parse().ok(),
            Self::Null => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            Self::Float(f) => Some(*f as i64),
            Self::Text(s) => s.parse().ok(),
            Self::Null => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

impl From<&str> for SqlValue {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<String> for SqlValue {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&String> for SqlValue {
    fn from(s: &String) -> Self {
        Self::Text(s.clone())
    }
}

impl<T: Into<SqlValue>> From<Option<T>> for SqlValue {
    fn from(v: Option<T>) -> Self {
        v.map_or(Self::Null, Into::into)
    }
}

impl From<f64> for SqlValue {
    fn from(f: f64) -> Self {
        Self::Float(f)
    }
}

impl From<i64> for SqlValue {
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}

/// One row read back: column name → value.
pub type SqlRow = HashMap<String, SqlValue>;

/// How a driver writes a bound parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamStyle {
    /// `?` — SQLite, and MySQL through `sqlx`.
    Qmark,
    /// `%s`.
    Format,
    /// `$1`, `$2`, ... — PostgreSQL.
    Numeric,
}

/// Convert `?` placeholders to whatever this driver expects.
pub fn placeholders(sql: &str, style: ParamStyle) -> String {
    match style {
        ParamStyle::Qmark => sql.to_string(),
        ParamStyle::Format => sql.replace('?', "%s"),
        ParamStyle::Numeric => {
            let mut out = String::with_capacity(sql.len() + 8);
            let mut index = 0;
            for c in sql.chars() {
                if c == '?' {
                    index += 1;
                    out.push_str(&format!("${index}"));
                } else {
                    out.push(c);
                }
            }
            out
        }
    }
}

/// Is this usable as a table name? Letters, digits and underscores.
pub fn valid_name(name: &str) -> bool {
    let rest: String = name.chars().filter(|c| *c != '_').collect();
    !rest.is_empty() && rest.chars().all(char::is_alphanumeric)
}

/// A database refused something, or could not be reached.
pub fn db_error(store: &str, e: impl std::fmt::Display) -> Error {
    Error::VectorStore { message: format!("{store}: {e}"), store: store.into(), status: None, retryable: false }
}

/// What a dialect must provide: run a statement, run a query, and say how it
/// spells placeholders and column types. The memory store and the session
/// store both sit on one of these, so they share its connections.
#[async_trait]
pub trait SqlDriver: Send + Sync + std::fmt::Debug {
    /// `sqlite`, `postgres`, `mysql`.
    fn name(&self) -> &'static str;
    fn paramstyle(&self) -> ParamStyle;
    fn text_type(&self) -> &'static str {
        "TEXT"
    }
    fn float_type(&self) -> &'static str {
        "DOUBLE PRECISION"
    }
    /// Indexable on MySQL too.
    fn key_type(&self) -> &'static str {
        "VARCHAR(191)"
    }
    /// The tail of an insert that turns it into an upsert on `doc_key`, if
    /// the dialect has one. `None` uses the portable two-step.
    fn upsert_clause(&self) -> Option<&'static str> {
        None
    }

    async fn execute(&self, sql: &str, params: Vec<SqlValue>) -> Result<()>;
    async fn fetch(&self, sql: &str, params: Vec<SqlValue>) -> Result<Vec<SqlRow>>;

    /// Goes up by one every time the connection is closed, so whoever created
    /// tables on the old one knows to check again.
    fn epoch(&self) -> u64;
    /// Release the connection or pool. The next call opens a new one.
    async fn close(&self);
    fn describe(&self) -> String;

    /// Convert `?` placeholders for this driver.
    fn q(&self, sql: &str) -> String {
        placeholders(sql, self.paramstyle())
    }
}

/// Create an index. MySQL has no `IF NOT EXISTS` for an index, so the plain
/// form is tried next, and "already there" is fine.
pub async fn create_index(driver: &dyn SqlDriver, if_not_exists: &str) -> Result<()> {
    if driver.execute(if_not_exists, vec![]).await.is_ok() {
        return Ok(());
    }
    let plain = if_not_exists.replacen("IF NOT EXISTS ", "", 1);
    if let Err(e) = driver.execute(&plain, vec![]).await {
        // It already exists, or this database cannot index these columns
        // (MySQL will not index a LONGTEXT without a prefix): the table works
        // either way, only slower.
        tracing::debug!(error = %e, "index not created: {plain}");
    }
    Ok(())
}

/// A fixed `?` list: `?, ?, ?`.
pub fn marks(n: usize) -> String {
    vec!["?"; n].join(", ")
}

/// JSON exactly as Python's `json.dumps` writes it: `", "` and `": "` between
/// items, and everything outside ASCII escaped.
pub fn py_dumps(value: &Value) -> String {
    use serde::Serialize;
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyFormatter);
    if value.serialize(&mut ser).is_err() {
        return "null".into();
    }
    String::from_utf8(out).unwrap_or_else(|_| "null".into())
}

struct PyFormatter;

impl serde_json::ser::Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }

    fn write_string_fragment<W: ?Sized + std::io::Write>(&mut self, w: &mut W, fragment: &str) -> std::io::Result<()> {
        for c in fragment.chars() {
            if c.is_ascii() {
                w.write_all(&[c as u8])?;
            } else {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    write!(w, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }
}

fn loads(value: Option<&SqlValue>) -> Option<Value> {
    match value {
        Some(SqlValue::Text(s)) if !s.is_empty() => serde_json::from_str(s).ok(),
        _ => None,
    }
}

/// Records and documents in two tables, with the trace as indexed columns.
///
/// `SqliteMemory`, `PostgresMemory` and `MySqlMemory` are this, over their
/// own driver.
#[derive(Debug)]
pub struct SqlMemory<D: SqlDriver> {
    driver: Arc<D>,
    pub table: String,
    pub docs_table: String,
    /// The driver epoch the tables were made on (0: not yet).
    ensured: AtomicU64,
}

impl<D: SqlDriver + 'static> SqlMemory<D> {
    /// Over a driver, in `agent_memory` and `agent_memory_docs`.
    pub fn with_driver(driver: D) -> Self {
        Self { driver: Arc::new(driver), table: "agent_memory".into(), docs_table: "agent_memory_docs".into(), ensured: AtomicU64::new(0) }
    }

    /// Records in `table`, documents in `{table}_docs`.
    pub fn table(mut self, table: &str) -> Result<Self> {
        if !valid_name(table) {
            return Err(Error::config(format!("not a table name: '{table}'")));
        }
        self.table = table.into();
        self.docs_table = format!("{table}_docs");
        Ok(self)
    }

    /// Documents in a table of their own choosing.
    pub fn docs_table(mut self, docs_table: &str) -> Result<Self> {
        if !valid_name(docs_table) {
            return Err(Error::config(format!("not a table name: '{docs_table}'")));
        }
        self.docs_table = docs_table.into();
        Ok(self)
    }

    /// The connection this store holds — a session store can share it.
    pub fn driver(&self) -> &Arc<D> {
        &self.driver
    }

    /// A session store on the same connections, in `table` (`agent_sessions`
    /// by default).
    pub fn session_store(&self, table: &str) -> Result<SqlSessionStore> {
        SqlSessionStore::new(self.driver.clone(), table)
    }

    /// The statements that make the tables and the index.
    pub fn schema(&self) -> Vec<String> {
        let d = &*self.driver;
        let (t, docs, txt, flt, key) = (&self.table, &self.docs_table, d.text_type(), d.float_type(), d.key_type());
        vec![
            format!(
                "CREATE TABLE IF NOT EXISTS {t} (
                id {key} PRIMARY KEY,
                scope {txt} NOT NULL,
                kind {txt} NOT NULL,
                text {txt},
                data {txt},
                tags {txt},
                source {txt},
                user_id {key},
                session_id {key},
                tenant_id {key},
                ts {flt} NOT NULL
            )"
            ),
            // The harness reads narrowed by trace, then scope, then newest first.
            format!("CREATE INDEX IF NOT EXISTS {t}_trace_idx ON {t} (user_id, session_id, tenant_id, scope, ts)"),
            format!(
                "CREATE TABLE IF NOT EXISTS {docs} (
                doc_key {key} PRIMARY KEY,
                name {txt} NOT NULL,
                namespace {txt} NOT NULL,
                text {txt},
                updated {flt} NOT NULL
            )"
            ),
        ]
    }

    async fn ensure(&self) -> Result<()> {
        let epoch = self.driver.epoch();
        if self.ensured.load(Ordering::Acquire) == epoch {
            return Ok(());
        }
        for statement in self.schema() {
            if statement.starts_with("CREATE INDEX") {
                create_index(&*self.driver, &statement).await?;
            } else {
                self.driver.execute(&statement, vec![]).await?;
            }
        }
        self.ensured.store(epoch, Ordering::Release);
        Ok(())
    }

    /// Build the filter once — a record with no value on an axis is shared.
    pub fn where_clause(scope: Option<&str>, kind: Option<&str>, trace: Option<&Trace>) -> (String, Vec<SqlValue>) {
        let mut clauses: Vec<String> = Vec::new();
        let mut params: Vec<SqlValue> = Vec::new();
        if let Some(s) = scope {
            clauses.push("scope = ?".into());
            params.push(s.into());
        }
        if let Some(k) = kind {
            clauses.push("kind = ?".into());
            params.push(k.into());
        }
        if let Some(t) = trace {
            for (field, value) in t.filters() {
                clauses.push(format!("({field} = ? OR {field} IS NULL)"));
                params.push(value.into());
            }
        }
        let clause = if clauses.is_empty() { String::new() } else { format!(" WHERE {}", clauses.join(" AND ")) };
        (clause, params)
    }

    /// A record as the row it is stored in.
    pub fn row(record: &MemoryRecord) -> Vec<SqlValue> {
        vec![
            (&record.id).into(),
            (&record.scope).into(),
            (&record.kind).into(),
            (&record.text).into(),
            py_dumps(&Value::Object(record.data.clone())).into(),
            py_dumps(&Value::from(record.tags.clone())).into(),
            (&record.source).into(),
            record.user_id.clone().into(),
            record.session_id.clone().into(),
            record.tenant_id.clone().into(),
            record.ts.into(),
        ]
    }

    /// A stored row back as a record.
    pub fn record(row: &SqlRow) -> MemoryRecord {
        let text = |k: &str| row.get(k).and_then(SqlValue::as_str).unwrap_or_default().to_string();
        let opt = |k: &str| row.get(k).and_then(SqlValue::as_str).map(str::to_string);
        let data = match loads(row.get("data")) {
            Some(Value::Object(m)) => m,
            _ => Map::new(),
        };
        let tags = loads(row.get("tags")).and_then(|v| serde_json::from_value::<Vec<String>>(v).ok()).unwrap_or_default();
        MemoryRecord {
            id: text("id"),
            scope: text("scope"),
            kind: text("kind"),
            text: text("text"),
            data,
            tags,
            source: text("source"),
            user_id: opt("user_id"),
            session_id: opt("session_id"),
            tenant_id: opt("tenant_id"),
            ts: row.get("ts").and_then(SqlValue::as_f64).unwrap_or(0.0),
            embedding: None,
        }
    }

    async fn upsert_doc(&self, key: String, name: &str, namespace: String, text: &str, updated: f64) -> Result<()> {
        let d = &*self.driver;
        let docs = &self.docs_table;
        if let Some(clause) = d.upsert_clause() {
            let sql = d.q(&format!("INSERT INTO {docs} (doc_key, name, namespace, text, updated) VALUES (?, ?, ?, ?, ?) {clause}"));
            return d.execute(&sql, vec![key.into(), name.into(), namespace.into(), text.into(), updated.into()]).await;
        }
        // Dialects differ on upsert; this is the portable two-step.
        d.execute(&d.q(&format!("UPDATE {docs} SET text = ?, updated = ? WHERE doc_key = ?")), vec![text.into(), updated.into(), (&key).into()]).await?;
        let rows = d.fetch(&d.q(&format!("SELECT doc_key FROM {docs} WHERE doc_key = ?")), vec![(&key).into()]).await?;
        if rows.is_empty() {
            let sql = d.q(&format!("INSERT INTO {docs} (doc_key, name, namespace, text, updated) VALUES (?, ?, ?, ?, ?)"));
            d.execute(&sql, vec![key.into(), name.into(), namespace.into(), text.into(), updated.into()]).await?;
        }
        Ok(())
    }
}

#[async_trait]
impl<D: SqlDriver + 'static> MemoryStore for SqlMemory<D> {
    async fn append(&self, record: MemoryRecord) -> Result<MemoryRecord> {
        self.ensure().await?;
        let sql = format!("INSERT INTO {} ({}) VALUES ({})", self.table, COLUMNS.join(", "), marks(COLUMNS.len()));
        self.driver.execute(&self.driver.q(&sql), Self::row(&record)).await?;
        Ok(record)
    }

    async fn all(&self, query: RecordQuery) -> Result<Vec<MemoryRecord>> {
        self.ensure().await?;
        let (clause, mut params) = Self::where_clause(query.scope.as_deref(), query.kind.as_deref(), query.trace.as_ref());
        // Newest-first with LIMIT so the database does the work, then flip back
        // to chronological order, which is what every caller expects.
        let mut sql = format!("SELECT {} FROM {}{clause} ORDER BY ts DESC", COLUMNS.join(", "), self.table);
        if let Some(n) = query.limit.filter(|n| *n > 0) {
            sql.push_str(" LIMIT ?");
            params.push(SqlValue::Int(n as i64));
        }
        let rows = self.driver.fetch(&self.driver.q(&sql), params).await?;
        Ok(rows.iter().rev().map(Self::record).collect())
    }

    async fn clear(&self, scope: Option<&str>, trace: Option<&Trace>) -> Result<()> {
        self.ensure().await?;
        let d = &*self.driver;
        let (clause, params) = Self::where_clause(scope, None, trace);
        d.execute(&d.q(&format!("DELETE FROM {}{clause}", self.table)), params).await?;
        if let Some(t) = trace {
            d.execute(&d.q(&format!("DELETE FROM {} WHERE namespace = ?", self.docs_table)), vec![t.slug().into()]).await?;
        } else if scope.is_none() {
            d.execute(&format!("DELETE FROM {}", self.docs_table), vec![]).await?;
        }
        Ok(())
    }

    async fn read_doc(&self, name: &str, trace: Option<&Trace>) -> Result<String> {
        self.ensure().await?;
        let d = &*self.driver;
        let rows = d.fetch(&d.q(&format!("SELECT text FROM {} WHERE doc_key = ?", self.docs_table)), vec![doc_key(name, trace).into()]).await?;
        Ok(rows.first().and_then(|r| r.get("text")).and_then(SqlValue::as_str).unwrap_or_default().to_string())
    }

    async fn write_doc(&self, name: &str, text: &str, trace: Option<&Trace>) -> Result<()> {
        self.ensure().await?;
        let namespace = trace.map_or_else(|| "_shared".to_string(), Trace::slug);
        self.upsert_doc(doc_key(name, trace), name, namespace, text, now()).await
    }

    async fn close(&self) {
        self.driver.close().await;
    }

    fn describe(&self) -> String {
        self.driver.describe()
    }

    fn sessions(&self, options: &Map<String, Value>) -> Option<Result<Arc<dyn SessionStore>>> {
        Some(session_options(options).and_then(|table| Ok(Arc::new(self.session_store(&table)?) as Arc<dyn SessionStore>)))
    }
}

/// The session store's own options: only `table`.
fn session_options(options: &Map<String, Value>) -> Result<String> {
    let mut table = "agent_sessions".to_string();
    for (key, value) in options {
        match (key.as_str(), value) {
            ("table", Value::String(t)) => table = t.clone(),
            ("table", other) => return Err(Error::config(format!("SQLSessionStore: table must be a string, not {other}"))),
            (other, _) => return Err(Error::config(format!("SQLSessionStore: unexpected option '{other}'"))),
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn each_dialect_writes_its_own_placeholders() {
        let sql = "SELECT * FROM t WHERE a = ? AND b = ?";
        assert!(placeholders(sql, ParamStyle::Qmark).ends_with("a = ? AND b = ?"));
        assert!(placeholders(sql, ParamStyle::Format).ends_with("a = %s AND b = %s"));
        assert!(placeholders(sql, ParamStyle::Numeric).ends_with("a = $1 AND b = $2"));
    }

    #[test]
    fn json_is_written_as_python_writes_it() {
        assert_eq!(py_dumps(&json!({"a": [1, "é"], "b": {"c": null}})), r#"{"a": [1, "\u00e9"], "b": {"c": null}}"#);
        assert_eq!(py_dumps(&json!(["x", "😀"])), r#"["x", "\ud83d\ude00"]"#);
        assert_eq!(py_dumps(&json!({})), "{}");
        assert_eq!(py_dumps(&json!([])), "[]");
    }

    #[test]
    fn names_are_checked() {
        assert!(valid_name("agent_sessions") && valid_name("chats2"));
        assert!(!valid_name("chats; drop table x") && !valid_name("___") && !valid_name(""));
    }
}
