//! SQLite memory. No server, and the library is bundled into the build.
//!
//! The right default when you want memory that survives a restart but do not
//! want to run a database. One file, indexed, and it handles millions of
//! records before you need anything bigger.
#![cfg(feature = "sqlite")]

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use rusqlite::Connection;
use rusqlite::types::{Value as Lite, ValueRef};

use crate::errors::{Error, Result};
use crate::memory::providers::sql::{ParamStyle, SqlDriver, SqlMemory, SqlRow, SqlValue, db_error};

/// A single SQLite file. `:memory:` gives a throwaway in-process database.
///
/// ```no_run
/// # use agent_harness::memory::providers::sqlite::SqliteMemory;
/// let store = SqliteMemory::open(".harness/memory.db").unwrap();
/// let in_process = SqliteMemory::open(":memory:").unwrap();
/// ```
pub type SqliteMemory = SqlMemory<SqliteDriver>;

impl SqlMemory<SqliteDriver> {
    /// The file at `path`, made (with its folder) on first use.
    pub fn open(path: &str) -> Result<Self> {
        Ok(Self::with_driver(SqliteDriver::new(path)))
    }

    /// Where the file is.
    pub fn path(&self) -> &str {
        &self.driver().path
    }
}

/// One connection, one lock: a SQLite connection is not shared across
/// threads, and every call here hops to a blocking worker thread.
#[derive(Debug)]
pub struct SqliteDriver {
    pub path: String,
    /// How long to wait on a lock another process holds.
    pub timeout: Duration,
    conn: Arc<Mutex<Option<Connection>>>,
    epoch: AtomicU64,
}

impl SqliteDriver {
    pub fn new(path: &str) -> Self {
        Self { path: path.into(), timeout: Duration::from_secs(10), conn: Arc::new(Mutex::new(None)), epoch: AtomicU64::new(1) }
    }

    fn connect(path: &str, timeout: Duration) -> rusqlite::Result<Connection> {
        if path != ":memory:"
            && let Some(parent) = Path::new(path).parent().filter(|p| !p.as_os_str().is_empty())
        {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(path)?;
        conn.busy_timeout(timeout)?;
        // WAL lets readers and a writer work at once; NORMAL is the usual
        // durability/throughput trade for an append-heavy workload.
        if path != ":memory:" {
            conn.query_row("PRAGMA journal_mode=WAL", [], |_| Ok(()))?;
        }
        conn.execute_batch("PRAGMA synchronous=NORMAL")?;
        Ok(conn)
    }

    async fn run(&self, sql: &str, params: Vec<SqlValue>, fetch: bool) -> Result<Vec<SqlRow>> {
        let (conn, path, timeout, sql) = (self.conn.clone(), self.path.clone(), self.timeout, sql.to_string());
        tokio::task::spawn_blocking(move || -> Result<Vec<SqlRow>> {
            let mut held = conn.lock().unwrap_or_else(|p| p.into_inner());
            if held.is_none() {
                *held = Some(Self::connect(&path, timeout).map_err(|e| db_error("sqlite", e))?);
            }
            let db = held.as_ref().expect("connected above");
            let values: Vec<Lite> = params.into_iter().map(lite).collect();
            let fail = |e: rusqlite::Error| db_error("sqlite", e);
            if !fetch {
                db.execute(&sql, rusqlite::params_from_iter(values)).map_err(fail)?;
                return Ok(vec![]);
            }
            let mut stmt = db.prepare(&sql).map_err(fail)?;
            let names: Vec<String> = stmt.column_names().map(str::to_string).collect();
            let mut rows = stmt.query(rusqlite::params_from_iter(values)).map_err(fail)?;
            let mut out = Vec::new();
            while let Some(row) = rows.next().map_err(fail)? {
                let mut found = SqlRow::with_capacity(names.len());
                for (i, name) in names.iter().enumerate() {
                    found.insert(name.clone(), value(row.get_ref(i).map_err(fail)?));
                }
                out.push(found);
            }
            Ok(out)
        })
        .await
        .map_err(|e| Error::Io(e.to_string()))?
    }
}

fn lite(v: SqlValue) -> Lite {
    match v {
        SqlValue::Null => Lite::Null,
        SqlValue::Int(i) => Lite::Integer(i),
        SqlValue::Float(f) => Lite::Real(f),
        SqlValue::Text(s) => Lite::Text(s),
    }
}

fn value(v: ValueRef<'_>) -> SqlValue {
    match v {
        ValueRef::Null => SqlValue::Null,
        ValueRef::Integer(i) => SqlValue::Int(i),
        ValueRef::Real(f) => SqlValue::Float(f),
        ValueRef::Text(t) | ValueRef::Blob(t) => SqlValue::Text(String::from_utf8_lossy(t).into_owned()),
    }
}

#[async_trait]
impl SqlDriver for SqliteDriver {
    fn name(&self) -> &'static str {
        "sqlite"
    }

    fn paramstyle(&self) -> ParamStyle {
        ParamStyle::Qmark
    }

    fn float_type(&self) -> &'static str {
        "REAL"
    }

    fn key_type(&self) -> &'static str {
        "TEXT"
    }

    // SQLite has a real upsert, so skip the portable two-step.
    fn upsert_clause(&self) -> Option<&'static str> {
        Some("ON CONFLICT(doc_key) DO UPDATE SET text = excluded.text, updated = excluded.updated")
    }

    async fn execute(&self, sql: &str, params: Vec<SqlValue>) -> Result<()> {
        self.run(sql, params, false).await.map(|_| ())
    }

    async fn fetch(&self, sql: &str, params: Vec<SqlValue>) -> Result<Vec<SqlRow>> {
        self.run(sql, params, true).await
    }

    fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    async fn close(&self) {
        let conn = self.conn.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let mut held = conn.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(db) = held.take() {
                let _ = db.close();
            }
        })
        .await;
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    fn describe(&self) -> String {
        format!("SQLite at {}", self.path)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Value, json};

    use super::*;
    use crate::errors::Error;
    use crate::memory::base::{MemoryRecord, MemoryStore, RecordQuery};
    use crate::memory::trace::Trace;
    use crate::runtime::session::{ListFilter, Session, SessionStore};
    use crate::types::{Message, Usage, new_id};

    fn temp_db(name: &str) -> (std::path::PathBuf, String) {
        let dir = std::env::temp_dir().join(format!("ah_sql_{}", new_id("")));
        let file = dir.join(name).to_string_lossy().replace('\\', "/");
        (dir, file)
    }

    #[test]
    fn the_filter_treats_an_untagged_record_as_shared() {
        let (clause, params) = SqliteMemory::where_clause(Some("user"), None, Some(&Trace::user("alice")));
        assert!(clause.contains("scope = ?"));
        assert!(clause.contains("(user_id = ? OR user_id IS NULL)"));
        assert_eq!(params, vec![SqlValue::from("user"), SqlValue::from("alice")]);
        assert_eq!(SqliteMemory::where_clause(None, None, None).0, "");
    }

    #[test]
    fn the_schema_indexes_the_way_the_harness_reads() {
        let schema = SqliteMemory::open(":memory:").unwrap().schema().join(" ");
        assert!(schema.contains("user_id, session_id, tenant_id, scope, ts"));
        assert!(schema.contains("agent_memory_docs"));
        assert!(SqliteMemory::open(":memory:").unwrap().table("bad name").is_err());
    }

    #[tokio::test]
    async fn memory_is_scoped_by_trace() {
        let (dir, file) = temp_db("m.db");
        let store = SqliteMemory::open(&file).unwrap();
        let alice = Trace::user("alice");
        let mut a = MemoryRecord::new("user", "decision", "Bill in EUR always").tags(["billing"]);
        a.data = json!({"currency": "EUR", "π": [1, null]}).as_object().cloned().unwrap();
        alice.stamp(&mut a);
        store.append(a.clone()).await.unwrap();
        let mut b = MemoryRecord::new("user", "fact", "Bob likes tea");
        b.ts = a.ts + 1.0;
        Trace::user("bob").stamp(&mut b);
        store.append(b).await.unwrap();
        let mut shared = MemoryRecord::new("project", "note", "Shared euro note");
        shared.ts = a.ts + 2.0;
        store.append(shared).await.unwrap();

        let mine = store.all(RecordQuery::default().traced(Some(&alice))).await.unwrap();
        assert_eq!(mine.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(), ["Bill in EUR always", "Shared euro note"]);
        assert_eq!(mine[0], a, "a record comes back as it went in");
        let found = store.search("bill EUR always", None, 5, Some(&alice)).await.unwrap();
        assert_eq!(found[0].text, "Bill in EUR always");
        assert_eq!(store.all(RecordQuery::scope("project")).await.unwrap().len(), 1);
        assert_eq!(store.all(RecordQuery::default().kind("fact")).await.unwrap().len(), 1);
        let newest = store.all(RecordQuery::default().limit(2)).await.unwrap();
        assert_eq!(newest.iter().map(|r| r.text.as_str()).collect::<Vec<_>>(), ["Bob likes tea", "Shared euro note"], "the newest, oldest first");

        // Documents, per trace.
        store.write_doc("user.md", "- likes EUR", Some(&alice)).await.unwrap();
        store.write_doc("user.md", "- likes EUR, always", Some(&alice)).await.unwrap();
        store.write_doc("team.md", "shared", None).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "- likes EUR, always");
        assert_eq!(store.read_doc("user.md", Some(&Trace::user("bob"))).await.unwrap(), "");
        assert_eq!(store.read_doc("team.md", None).await.unwrap(), "shared");

        // Clearing a trace takes its records and documents, and nobody else's.
        store.clear(None, Some(&alice)).await.unwrap();
        assert_eq!(store.read_doc("user.md", Some(&alice)).await.unwrap(), "");
        assert_eq!(store.read_doc("team.md", None).await.unwrap(), "shared");
        let left = store.all(RecordQuery::default()).await.unwrap();
        assert!(left.iter().all(|r| r.user_id.as_deref() != Some("alice")));
        assert!(left.iter().any(|r| r.text == "Bob likes tea"));

        // A scope alone leaves documents be; nothing at all clears them too.
        store.clear(Some("project"), None).await.unwrap();
        assert_eq!(store.all(RecordQuery::default()).await.unwrap().len(), 1);
        assert_eq!(store.read_doc("team.md", None).await.unwrap(), "shared");
        store.clear(None, None).await.unwrap();
        assert!(store.all(RecordQuery::default()).await.unwrap().is_empty());
        assert_eq!(store.read_doc("team.md", None).await.unwrap(), "");

        // Closing is safe twice, and the store opens again on the next call.
        store.close().await;
        store.close().await;
        store.write_doc("again.md", "x", None).await.unwrap();
        assert_eq!(store.read_doc("again.md", None).await.unwrap(), "x");
        store.close().await;
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn rows_are_the_python_rows() {
        let (dir, file) = temp_db("py.db");
        let store = SqliteMemory::open(&file).unwrap();
        let mut r = MemoryRecord::new("user", "fact", "café").tags(["a", "b"]);
        r.data = json!({"k": "é"}).as_object().cloned().unwrap();
        store.append(r.clone()).await.unwrap();
        let rows = store.driver().fetch("SELECT data, tags, user_id FROM agent_memory", vec![]).await.unwrap();
        assert_eq!(rows[0]["data"], SqlValue::from(r#"{"k": "\u00e9"}"#));
        assert_eq!(rows[0]["tags"], SqlValue::from(r#"["a", "b"]"#));
        assert!(rows[0]["user_id"].is_null());
        // A row Python wrote reads back, bad JSON and all.
        store
            .driver()
            .execute(
                "INSERT INTO agent_memory (id, scope, kind, text, data, tags, source, user_id, session_id, tenant_id, ts) VALUES ('mem_py', 'user', 'note', NULL, 'not json', '', NULL, 'alice', NULL, NULL, 5)",
                vec![],
            )
            .await
            .unwrap();
        let back = store.all(RecordQuery::default().traced(Some(&Trace::user("alice")))).await.unwrap();
        let py = back.iter().find(|r| r.id == "mem_py").unwrap();
        assert_eq!((py.text.as_str(), py.data.len(), py.tags.len(), py.ts), ("", 0, 0, 5.0));
        store.close().await;
        std::fs::remove_dir_all(dir).ok();
    }

    fn conversation(title: &str) -> Session {
        Session {
            agent: "support".into(),
            title: title.into(),
            user_id: Some("alice".into()),
            tenant_id: Some("acme".into()),
            messages: vec![Message::user("Where is order 4182?"), Message::assistant("It shipped ✓")],
            usage: Usage { input_tokens: 120, output_tokens: 8, cost_usd: 0.0021, ..Default::default() },
            metadata: json!({"sandbox": {"sandbox": "docker", "id": "abc"}, "$weird.key": [1, {"a": null}]}).as_object().cloned().unwrap(),
            ..Default::default()
        }
    }

    fn table(name: &str) -> Map<String, Value> {
        json!({"table": name}).as_object().cloned().unwrap()
    }

    #[tokio::test]
    async fn sessions_hold_the_contract() {
        let (dir, file) = temp_db("chats.db");
        let memory = SqliteMemory::open(&file).unwrap();
        let store = memory.sessions(&table("chats")).unwrap().unwrap();
        assert_eq!(store.kind(), "SQLSessionStore");

        // It comes back as it went in, and counts its saves.
        let mut first = conversation("Order 4182");
        store.save(&mut first).await.unwrap();
        assert_eq!(first.version, 1);
        let mut back = store.load(&first.id).await.unwrap();
        let same = Session { created: first.created, updated: first.updated, ..back.clone() };
        assert_eq!(same, first);
        back.messages.push(Message::user("Thanks."));
        store.save(&mut back).await.unwrap();
        let held = store.load(&first.id).await.unwrap();
        assert_eq!((held.version, held.messages.len()), (2, 3));

        // A stale save is refused, and leaves both sides as they were.
        let mut stale = store.load(&first.id).await.unwrap();
        let mut fresh = store.load(&first.id).await.unwrap();
        fresh.title = "saved first".into();
        store.save(&mut fresh).await.unwrap();
        stale.title = "saved second".into();
        assert!(matches!(store.save(&mut stale).await, Err(Error::SessionConflict { .. })));
        assert_eq!(stale.version, 2);
        let held = store.load(&first.id).await.unwrap();
        assert_eq!((held.title.as_str(), held.version), ("saved first", 3));

        // Two that both think they are creating it: one does.
        let mut twin = Session { id: first.id.clone(), agent: "support".into(), ..Default::default() };
        assert!(matches!(store.save(&mut twin).await, Err(Error::SessionConflict { .. })));

        // Listing: newest first, and narrowed to whose it is.
        let mut second = Session { user_id: Some("bob".into()), ..conversation("Bob's") };
        let mut third = Session { tenant_id: Some("globex".into()), ..conversation("Another tenant") };
        let mut fourth = Session { agent: "billing".into(), ..conversation("Alice again") };
        for s in [&mut second, &mut third, &mut fourth] {
            tokio::time::sleep(Duration::from_millis(10)).await;
            store.save(s).await.unwrap();
        }
        let owner = |user: &str| ListFilter { user_id: Some(user.into()), tenant_id: Some("acme".into()), ..Default::default() };
        let titles = |rows: Vec<Session>| rows.into_iter().map(|s| s.title).collect::<Vec<_>>();
        assert_eq!(titles(store.list(&owner("alice")).await.unwrap()), ["Alice again", "saved first"]);
        assert_eq!(titles(store.list(&owner("bob")).await.unwrap()), ["Bob's"]);
        let acme: Vec<String> =
            store.list(&ListFilter { tenant_id: Some("acme".into()), limit: 50, ..Default::default() }).await.unwrap().into_iter().map(|s| s.id).collect();
        assert!(acme.contains(&first.id) && acme.contains(&second.id) && acme.contains(&fourth.id) && !acme.contains(&third.id));
        assert_eq!(titles(store.list(&ListFilter { agent: Some("billing".into()), ..owner("alice") }).await.unwrap()), ["Alice again"]);
        assert_eq!(titles(store.list(&ListFilter { limit: 1, ..owner("alice") }).await.unwrap()), ["Alice again"]);
        assert!(store.list(&owner("nobody")).await.unwrap().is_empty());

        // A long conversation, and one that shrinks.
        let mut long = conversation("long");
        long.messages.extend((0..700).map(|n| Message::user(format!("turn {n} {}", "ab".repeat(600)))));
        store.save(&mut long).await.unwrap();
        let mut long = store.load(&long.id).await.unwrap();
        assert_eq!(long.messages.len(), 702);
        long.messages.truncate(10);
        store.save(&mut long).await.unwrap();
        assert_eq!(store.load(&long.id).await.unwrap().messages.len(), 10);

        // A fork is its own session, owned by the same person.
        let fork = store.fork(&first.id, Some(1)).await.unwrap();
        assert_ne!(fork.id, first.id);
        assert_eq!((fork.parent_id.as_deref(), fork.user_id.as_deref(), fork.messages.len()), (Some(first.id.as_str()), Some("alice"), 1));

        // Of many racing from one version, exactly one wins.
        let base = store.load(&first.id).await.unwrap();
        let racers = (0..8).map(|_| {
            let (store, mut copy) = (store.clone(), base.clone());
            tokio::spawn(async move { store.save(&mut copy).await })
        });
        let outcomes: Vec<_> = futures::future::join_all(racers).await.into_iter().map(|r| r.unwrap()).collect();
        assert_eq!(outcomes.iter().filter(|o| o.is_ok()).count(), 1);
        assert!(outcomes.iter().all(|o| matches!(o, Ok(()) | Err(Error::SessionConflict { .. }))));

        // Delete, and deleting what is not there is not an error.
        store.delete(&first.id).await.unwrap();
        assert!(matches!(store.load(&first.id).await, Err(Error::Configuration(m)) if m.contains("no session")));
        assert!(!store.list(&owner("alice")).await.unwrap().iter().any(|s| s.id == first.id));
        store.delete(&first.id).await.unwrap();

        // It proves itself, and shares the memory's connection.
        let report = store.check().await;
        assert_eq!(report["ok"], true, "{report}");
        memory.write_doc("user.md", "x", None).await.unwrap();
        let rows = memory.driver().fetch("SELECT COUNT(*) AS n FROM chats", vec![]).await.unwrap();
        assert_eq!(rows[0]["n"], SqlValue::Int(5));
        memory.close().await;
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn sessions_read_python_rows_and_refuse_bad_tables() {
        let memory = SqliteMemory::open(":memory:").unwrap();
        assert!(memory.sessions(&table("chats; drop table x")).unwrap().is_err());
        assert!(memory.sessions(&json!({"ttl": 5}).as_object().cloned().unwrap()).unwrap().is_err());
        let store = memory.session_store("agent_sessions").unwrap();
        assert_eq!(store.check().await["ok"], true);
        // The version column rules over the one in the JSON.
        let payload = r#"{"id": "ses_py", "agent": "support", "version": 1, "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}"#;
        memory
            .driver()
            .execute(
                "INSERT INTO agent_sessions (id, agent, user_id, tenant_id, title, created, updated, version, rev, payload) VALUES ('ses_py', 'support', NULL, NULL, '', 1, 1, 4, 'rev_x', ?)",
                vec![payload.into()],
            )
            .await
            .unwrap();
        let mut held = store.load("ses_py").await.unwrap();
        assert_eq!((held.version, held.messages[0].text().as_str()), (4, "hi"));
        store.save(&mut held).await.unwrap();
        assert_eq!(store.load("ses_py").await.unwrap().version, 5);
        assert!(store.load("ses_never_was").await.is_err());
    }

    #[tokio::test]
    async fn urls_reach_sqlite() {
        let (dir, file) = temp_db("u.db");
        let url = format!("sqlite://{file}");
        let memory = crate::memory::providers::memory_provider(&url).unwrap();
        assert!(memory.describe().starts_with("SQLite at"));
        let store = crate::sessions::session_provider(&url).unwrap();
        assert_eq!(store.check().await["ok"], true);
        memory.close().await;
        std::fs::remove_dir_all(dir).ok();
    }
}
