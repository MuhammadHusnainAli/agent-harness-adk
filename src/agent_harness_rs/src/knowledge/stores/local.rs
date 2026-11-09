//! Vector stores that need nothing: one in memory, one in a SQLite file.
//!
//! Both compare the query with every record — exact, and fine into the tens of
//! thousands of records. Past that, use a database built for it.

use std::sync::Mutex;

use async_trait::async_trait;
use indexmap::IndexMap;

use crate::errors::Result;
use crate::knowledge::base::{Condition, VectorHit, VectorRecord, VectorStore, cosine, holds, store_error};

fn top(mut scored: Vec<VectorHit>, k: usize) -> Vec<VectorHit> {
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// In this process, gone with it. For tests, and for trying things out.
#[derive(Debug, Default)]
pub struct MemoryVectorStore {
    rows: Mutex<(IndexMap<String, VectorRecord>, usize)>,
}

impl MemoryVectorStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dimension(&self) -> usize {
        self.rows.lock().unwrap_or_else(|p| p.into_inner()).1
    }
}

#[async_trait]
impl VectorStore for MemoryVectorStore {
    fn name(&self) -> &str {
        "memory"
    }

    fn batch_size(&self) -> usize {
        10_000
    }

    async fn ensure(&self, dimension: usize) -> Result<()> {
        let mut held = self.rows.lock().unwrap_or_else(|p| p.into_inner());
        if held.1 != 0 && !held.0.is_empty() && dimension != held.1 {
            return Err(store_error(
                "memory",
                format!("this store holds {}-dimension vectors; {dimension} was asked for — a different embedding model needs a different store", held.1),
            ));
        }
        held.1 = dimension;
        Ok(())
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        let mut held = self.rows.lock().unwrap_or_else(|p| p.into_inner());
        if held.0.is_empty() {
            held.1 = 0; // empty: whatever comes next sets it
        }
        for record in records {
            if held.1 != 0 && record.vector.len() != held.1 {
                return Err(store_error("memory", format!("record '{}' has {} dimensions; this store holds {}", record.id, record.vector.len(), held.1)));
            }
            if held.1 == 0 {
                held.1 = record.vector.len();
            }
            held.0.insert(record.id.clone(), record);
        }
        Ok(())
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let held = self.rows.lock().unwrap_or_else(|p| p.into_inner());
        let scored = held
            .0
            .values()
            .filter(|r| holds(&r.metadata, wanted))
            .map(|r| VectorHit { id: r.id.clone(), score: cosine(vector, &r.vector), text: r.text.clone(), metadata: r.metadata.clone() })
            .collect();
        Ok(top(scored, k))
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        let mut held = self.rows.lock().unwrap_or_else(|p| p.into_inner());
        held.0.retain(|key, r| !(ids.is_none_or(|ids| ids.contains(key)) && holds(&r.metadata, wanted)));
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.rows.lock().unwrap_or_else(|p| p.into_inner()).0.len())
    }

    async fn clear(&self) -> Result<()> {
        self.rows.lock().unwrap_or_else(|p| p.into_inner()).0.clear();
        Ok(())
    }
}

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteVectorStore;

#[cfg(feature = "sqlite")]
mod sqlite {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use rusqlite::{Connection, params};
    use serde_json::{Map, Value};

    use super::top;
    use crate::errors::Result;
    use crate::knowledge::base::{Condition, VectorHit, VectorRecord, VectorStore, cosine, holds, store_error};

    type Row = (String, Vec<f32>, String, Map<String, Value>);

    struct State {
        db: Option<Connection>,
        cache: Option<Arc<Vec<Row>>>,
    }

    /// Vectors in one SQLite file. Survives a restart; needs no server. The
    /// file is the same one the Python `SQLiteVectorStore` writes.
    ///
    /// ```no_run
    /// # use agent_harness::knowledge::stores::SqliteVectorStore;
    /// let store = SqliteVectorStore::new("knowledge.db", "handbook").unwrap();
    /// ```
    pub struct SqliteVectorStore {
        path: String,
        table: String,
        state: Arc<Mutex<State>>,
    }

    impl std::fmt::Debug for SqliteVectorStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "<SqliteVectorStore {} {}>", self.path, self.table)
        }
    }

    fn sql_error(e: impl std::fmt::Display) -> crate::errors::Error {
        store_error("sqlite", format!("sqlite: {e}"))
    }

    impl SqliteVectorStore {
        pub fn new(path: &str, table: &str) -> Result<Self> {
            if table.is_empty() || !table.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(store_error("sqlite", format!("'{table}' is not a usable table name")));
            }
            Ok(Self { path: path.into(), table: table.into(), state: Arc::new(Mutex::new(State { db: None, cache: None })) })
        }

        /// Run `f` with the open connection, off the async runtime.
        async fn with<T: Send + 'static>(&self, f: impl FnOnce(&Connection, &mut Option<Arc<Vec<Row>>>, &str) -> Result<T> + Send + 'static) -> Result<T> {
            let state = self.state.clone();
            let (path, table) = (self.path.clone(), self.table.clone());
            tokio::task::spawn_blocking(move || {
                let mut s = state.lock().unwrap_or_else(|p| p.into_inner());
                if s.db.is_none() {
                    if path != ":memory:"
                        && let Some(parent) = std::path::Path::new(&path).parent()
                        && !parent.as_os_str().is_empty()
                    {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let db = Connection::open(&path).map_err(sql_error)?;
                    let _ = db.busy_timeout(std::time::Duration::from_secs(30));
                    let _: std::result::Result<String, _> = db.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0));
                    db.execute(
                        &format!("CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, dimension INTEGER NOT NULL, vector BLOB NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL)"),
                        [],
                    )
                    .map_err(sql_error)?;
                    s.db = Some(db);
                }
                let State { db, cache } = &mut *s;
                f(db.as_ref().expect("opened above"), cache, &table)
            })
            .await
            .map_err(sql_error)?
        }

        fn load(db: &Connection, cache: &mut Option<Arc<Vec<Row>>>, table: &str) -> Result<Arc<Vec<Row>>> {
            if let Some(c) = cache {
                return Ok(c.clone());
            }
            let mut stmt = db.prepare(&format!("SELECT id, vector, text, metadata FROM {table}")).map_err(sql_error)?;
            let rows = stmt
                .query_map([], |r| {
                    let blob: Vec<u8> = r.get(1)?;
                    let meta: String = r.get(3)?;
                    // array('f').tobytes(): native-endian float32, which is little-endian everywhere this runs.
                    let vector = blob.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
                    Ok((r.get::<_, String>(0)?, vector, r.get::<_, String>(2)?, serde_json::from_str(&meta).unwrap_or_default()))
                })
                .map_err(sql_error)?
                .collect::<std::result::Result<Vec<Row>, _>>()
                .map_err(sql_error)?;
            let rows = Arc::new(rows);
            *cache = Some(rows.clone());
            Ok(rows)
        }
    }

    #[async_trait]
    impl VectorStore for SqliteVectorStore {
        fn name(&self) -> &str {
            "sqlite"
        }

        fn batch_size(&self) -> usize {
            1000
        }

        async fn ensure(&self, dimension: usize) -> Result<()> {
            let path = self.path.clone();
            self.with(move |db, _, table| {
                let held: Option<i64> = db.query_row(&format!("SELECT dimension FROM {table} LIMIT 1"), [], |r| r.get(0)).ok();
                match held {
                    Some(h) if h as usize != dimension => Err(store_error(
                        "sqlite",
                        format!("{path} holds {h}-dimension vectors; {dimension} was asked for — a different embedding model needs a different table"),
                    )),
                    _ => Ok(()),
                }
            })
            .await
        }

        async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
            self.with(move |db, cache, table| {
                let sql = format!(
                    "INSERT INTO {table} (id, dimension, vector, text, metadata) VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(id) DO UPDATE SET dimension=excluded.dimension, vector=excluded.vector, text=excluded.text, metadata=excluded.metadata"
                );
                db.execute_batch("BEGIN").map_err(sql_error)?;
                for r in &records {
                    let blob: Vec<u8> = r.vector.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let meta = serde_json::to_string(&r.metadata).unwrap_or_else(|_| "{}".into());
                    if let Err(e) = db.execute(&sql, params![r.id, r.vector.len() as i64, blob, r.text, meta]) {
                        let _ = db.execute_batch("ROLLBACK");
                        return Err(sql_error(e));
                    }
                }
                db.execute_batch("COMMIT").map_err(sql_error)?;
                *cache = None;
                Ok(())
            })
            .await
        }

        async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
            let (vector, wanted) = (vector.to_vec(), wanted.to_vec());
            self.with(move |db, cache, table| {
                let rows = Self::load(db, cache, table)?;
                let scored = rows
                    .iter()
                    .filter(|(_, _, _, meta)| holds(meta, &wanted))
                    .map(|(id, held, text, meta)| VectorHit { id: id.clone(), score: cosine(&vector, held), text: text.clone(), metadata: meta.clone() })
                    .collect();
                Ok(top(scored, k))
            })
            .await
        }

        async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
            let (ids, wanted) = (ids.map(<[String]>::to_vec), wanted.to_vec());
            self.with(move |db, cache, table| {
                let rows = Self::load(db, cache, table)?;
                let doomed: Vec<&String> =
                    rows.iter().filter(|(key, _, _, meta)| ids.as_ref().is_none_or(|ids| ids.contains(key)) && holds(meta, &wanted)).map(|r| &r.0).collect();
                for key in doomed {
                    db.execute(&format!("DELETE FROM {table} WHERE id = ?1"), params![key]).map_err(sql_error)?;
                }
                *cache = None;
                Ok(())
            })
            .await
        }

        async fn count(&self) -> Result<usize> {
            self.with(|db, _, table| db.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get::<_, i64>(0)).map(|n| n as usize).map_err(sql_error))
                .await
        }

        async fn clear(&self) -> Result<()> {
            self.with(|db, cache, table| {
                db.execute(&format!("DELETE FROM {table}"), []).map_err(sql_error)?;
                *cache = None;
                Ok(())
            })
            .await
        }

        async fn close(&self) {
            let state = self.state.clone();
            let _ = tokio::task::spawn_blocking(move || state.lock().unwrap_or_else(|p| p.into_inner()).db = None).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn memory_store_passes_its_own_check() {
        let report = MemoryVectorStore::new().check(8).await;
        assert_eq!(report["ok"], true, "{report:#}");
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_store_passes_its_own_check() {
        let dir = std::env::temp_dir().join(format!("ah_vs_{}", uuid::Uuid::new_v4().simple()));
        let path = dir.join("k.db");
        let store = SqliteVectorStore::new(path.to_str().unwrap(), "vectors").unwrap();
        let report = store.check(8).await;
        assert_eq!(report["ok"], true, "{report:#}");
        assert!(store.ensure(8).await.is_ok());
        store.close().await;
        let _ = std::fs::remove_dir_all(dir);
    }
}
