//! PostgreSQL with pgvector — your own, or Supabase, Neon, RDS, Aurora,
//! Cloud SQL, AlloyDB, Azure Database for PostgreSQL.
#![cfg(feature = "postgres")]

use std::str::FromStr;

use async_trait::async_trait;
use serde_json::{Map, Value, json};
use sqlx::postgres::{PgArguments, PgConnectOptions, PgPool, PgPoolOptions};
use sqlx::query::Query;
use sqlx::{AssertSqlSafe, Postgres, Row};
use tokio::sync::Mutex;

use super::UrlParts;
use crate::errors::{Error, Result};
use crate::knowledge::base::{Condition, VectorHit, VectorRecord, VectorStore, store_error};

/// One bound parameter of a pgvector statement.
#[derive(Debug, Clone, PartialEq)]
enum Arg {
    Text(String),
    Float(f64),
    Int(i64),
    Texts(Vec<String>),
}

fn bind(sql: String, args: Vec<Arg>) -> Query<'static, Postgres, PgArguments> {
    let mut query = sqlx::query(AssertSqlSafe(sql));
    for a in args {
        query = match a {
            Arg::Text(s) => query.bind(s),
            Arg::Float(f) => query.bind(f),
            Arg::Int(i) => query.bind(i),
            Arg::Texts(v) => query.bind(v),
        };
    }
    query
}

fn op_sql(op: &str) -> &'static str {
    match op {
        "$eq" => "=",
        "$ne" => "<>",
        "$gt" => ">",
        "$gte" => ">=",
        "$lt" => "<",
        _ => "<=",
    }
}

/// A Postgres table with a `vector` column.
///
/// ```no_run
/// # use agent_harness::knowledge::vector_store;
/// let store = vector_store("pgvector://user:pass@host/db?table=handbook").unwrap();
/// ```
///
/// Needs the `vector` extension, which is created if the role may create it.
/// An HNSW index on cosine distance is built with the table. Nothing connects
/// until the first call.
#[derive(Debug)]
pub struct PgVectorStore {
    options: PgConnectOptions,
    /// `"schema"."table"`, quoted, as it goes into the SQL.
    pub table: String,
    bare: String,
    pub min_size: u32,
    pub max_size: u32,
    /// Build the HNSW and metadata indexes with the table.
    pub index: bool,
    pool: Mutex<Option<PgPool>>,
}

impl PgVectorStore {
    /// `dsn` is a `postgresql://` URL (`pgvector://` is taken too); the table
    /// is `schema.table`.
    pub fn new(dsn: &str, table: &str, schema: &str) -> Result<Self> {
        for part in [table, schema] {
            let rest: String = part.chars().filter(|c| *c != '_').collect();
            if rest.is_empty() || !rest.chars().all(char::is_alphanumeric) {
                return Err(store_error("pgvector", format!("'{part}' is not a usable name")));
            }
        }
        let dsn = match dsn.strip_prefix("pgvector://") {
            Some(rest) => format!("postgresql://{rest}"),
            None => dsn.to_string(),
        };
        let options = PgConnectOptions::from_str(&dsn).map_err(|e| store_error("pgvector", e.to_string()))?;
        Ok(Self { options, table: format!("\"{schema}\".\"{table}\""), bare: table.into(), min_size: 1, max_size: 10, index: true, pool: Mutex::new(None) })
    }

    /// From a URL: `pgvector://user:pass@host/db?table=handbook`. `table`
    /// (default `knowledge`), `schema` (default `public`) and `index` are the
    /// store's; anything else in the query goes to the connection.
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        if bare {
            return Err(Error::config("pgvector needs a database: pgvector://user:pass@host/db?table=handbook"));
        }
        let table = p.take("table").unwrap_or_else(|| "knowledge".into());
        let schema = p.take("schema").unwrap_or_else(|| "public".into());
        let index = p.take("index").is_none_or(|v| !matches!(v.as_str(), "0" | "false" | "no"));
        let query = p.rest_query();
        let dsn = format!("postgresql://{}{}{}", p.netloc, p.raw_path, if query.is_empty() { String::new() } else { format!("?{query}") });
        let mut store = Self::new(&dsn, &table, &schema)?;
        store.index = index;
        Ok(store)
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
            .map_err(|e| fail(&e))?;
        *held = Some(pool.clone());
        Ok(pool)
    }

    /// pgvector's text form: `[0.1,0.2,0.3]`.
    fn vector(vector: &[f32]) -> String {
        let parts: Vec<String> = vector.iter().map(|x| format!("{:?}", f64::from(*x))).collect();
        format!("[{}]", parts.join(","))
    }

    /// The `WHERE` body for these conditions (and ids), its arguments added to `args`.
    fn where_sql(wanted: &[Condition], args: &mut Vec<Arg>, ids: Option<&[String]>) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(ids) = ids {
            args.push(Arg::Texts(ids.to_vec()));
            parts.push(format!("id = ANY(${}::text[])", args.len()));
        }
        for c in wanted {
            let name = &c.field;
            match c.op.as_str() {
                "$in" => {
                    // Any of these values — or, for a list in the metadata, any shared.
                    let values = c.value.as_array().cloned().unwrap_or_default();
                    args.push(Arg::Texts(values.iter().map(|v| v.to_string()).collect()));
                    args.push(Arg::Text(name.clone()));
                    parts.push(format!("(metadata -> ${}) @> ANY(${}::jsonb[]) ", args.len(), args.len() - 1));
                }
                "$eq" | "$ne" => {
                    args.push(Arg::Text(json!({ name: c.value }).to_string()));
                    let contains = format!("metadata @> ${}::jsonb", args.len());
                    args.push(Arg::Text(json!({ name: [c.value] }).to_string()));
                    let held = format!("({contains} OR metadata @> ${}::jsonb)", args.len());
                    parts.push(if c.op == "$eq" { held } else { format!("NOT {held}") });
                }
                op => {
                    args.push(Arg::Text(name.clone()));
                    let field = format!("(metadata ->> ${})", args.len());
                    let cast = match &c.value {
                        Value::String(s) => {
                            args.push(Arg::Text(s.clone()));
                            ""
                        }
                        Value::Bool(b) => {
                            args.push(Arg::Float(if *b { 1.0 } else { 0.0 }));
                            "::float8"
                        }
                        other => {
                            args.push(Arg::Float(other.as_f64().unwrap_or(0.0)));
                            "::float8"
                        }
                    };
                    parts.push(format!("{field}{cast} {} ${}", op_sql(op), args.len()));
                }
            }
        }
        parts.join(" AND ")
    }
}

fn fail(e: &sqlx::Error) -> Error {
    store_error("pgvector", e.to_string())
}

#[async_trait]
impl VectorStore for PgVectorStore {
    fn name(&self) -> &str {
        "pgvector"
    }

    fn batch_size(&self) -> usize {
        500
    }

    async fn ensure(&self, dimension: usize) -> Result<()> {
        let pool = self.pool().await?;
        let mut db = pool.acquire().await.map_err(|e| fail(&e))?;
        if let Err(e) = sqlx::query("CREATE EXTENSION IF NOT EXISTS vector").execute(&mut *db).await {
            let held: Option<i32> =
                sqlx::query_scalar("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetch_optional(&mut *db).await.map_err(|e| fail(&e))?;
            if held.is_none() {
                let kind = match &e {
                    sqlx::Error::Database(d) => d.code().map_or_else(|| "DatabaseError".to_string(), |c| c.into_owned()),
                    other => format!("{other:?}").split(['(', ' ']).next().unwrap_or("Error").to_string(),
                };
                return Err(store_error(
                    "pgvector",
                    format!("the pgvector extension is not installed in this database, and this role may not create it ({kind})"),
                ));
            }
        }
        let t = &self.table;
        let create = format!(
            "CREATE TABLE IF NOT EXISTS {t} (id TEXT PRIMARY KEY, vector vector({dimension}) NOT NULL, text TEXT NOT NULL DEFAULT '', metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb)"
        );
        sqlx::query(AssertSqlSafe(create)).execute(&mut *db).await.map_err(|e| fail(&e))?;
        if self.index {
            let bare = &self.bare;
            for statement in [
                format!("CREATE INDEX IF NOT EXISTS \"{bare}_vector_hnsw\" ON {t} USING hnsw (vector vector_cosine_ops)"),
                format!("CREATE INDEX IF NOT EXISTS \"{bare}_metadata\" ON {t} USING gin (metadata jsonb_path_ops)"),
            ] {
                sqlx::query(AssertSqlSafe(statement)).execute(&mut *db).await.map_err(|e| fail(&e))?;
            }
        }
        Ok(())
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        let pool = self.pool().await?;
        let sql = format!(
            "INSERT INTO {} (id, vector, text, metadata) VALUES ($1, $2::vector, $3, $4::jsonb) ON CONFLICT (id) DO UPDATE SET \
             vector = EXCLUDED.vector, text = EXCLUDED.text, metadata = EXCLUDED.metadata",
            self.table
        );
        let mut tx = pool.begin().await.map_err(|e| fail(&e))?;
        for r in records {
            let args = vec![
                Arg::Text(r.id),
                Arg::Text(Self::vector(&r.vector)),
                Arg::Text(r.text.replace('\0', "")),
                Arg::Text(Value::Object(r.metadata).to_string()),
            ];
            bind(sql.clone(), args).execute(&mut *tx).await.map_err(|e| fail(&e))?;
        }
        tx.commit().await.map_err(|e| fail(&e))
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let mut args = vec![Arg::Text(Self::vector(vector))];
        let clause = Self::where_sql(wanted, &mut args, None);
        args.push(Arg::Int(k as i64));
        let sql = format!(
            "SELECT id, text, metadata::text AS metadata, 1 - (vector <=> $1::vector) AS score FROM {} {} ORDER BY vector <=> $1::vector LIMIT ${}",
            self.table,
            if clause.is_empty() { String::new() } else { format!("WHERE {clause}") },
            args.len()
        );
        let pool = self.pool().await?;
        let rows = bind(sql, args).fetch_all(&pool).await.map_err(|e| fail(&e))?;
        let mut hits = Vec::with_capacity(rows.len());
        for row in rows {
            let metadata: Option<String> = row.try_get("metadata").map_err(|e| fail(&e))?;
            let metadata = metadata.and_then(|m| serde_json::from_str::<Map<String, Value>>(&m).ok()).unwrap_or_default();
            hits.push(VectorHit {
                id: row.try_get("id").map_err(|e| fail(&e))?,
                score: row.try_get::<Option<f64>, _>("score").map_err(|e| fail(&e))?.unwrap_or(0.0),
                text: row.try_get::<Option<String>, _>("text").map_err(|e| fail(&e))?.unwrap_or_default(),
                metadata,
            });
        }
        Ok(hits)
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        let mut args = Vec::new();
        let clause = Self::where_sql(wanted, &mut args, ids);
        let pool = self.pool().await?;
        bind(format!("DELETE FROM {} WHERE {clause}", self.table), args).execute(&pool).await.map_err(|e| fail(&e))?;
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let pool = self.pool().await?;
        let n: i64 = sqlx::query_scalar(AssertSqlSafe(format!("SELECT COUNT(*) FROM {}", self.table))).fetch_one(&pool).await.map_err(|e| fail(&e))?;
        Ok(n.max(0) as usize)
    }

    async fn clear(&self) -> Result<()> {
        let pool = self.pool().await?;
        sqlx::query(AssertSqlSafe(format!("TRUNCATE {}", self.table))).execute(&pool).await.map_err(|e| fail(&e))?;
        Ok(())
    }

    async fn close(&self) {
        let pool = self.pool.lock().await.take();
        if let Some(pool) = pool {
            pool.close().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::base::conditions;

    #[test]
    fn urls_and_names() {
        let p = UrlParts::parse("pgvector://u:p@db.internal:5433/agents?table=handbook&sslmode=disable").unwrap();
        let store = PgVectorStore::from_url(p, false).unwrap();
        assert_eq!(store.table, "\"public\".\"handbook\"");
        assert_eq!((store.options.get_host(), store.options.get_port(), store.options.get_database()), ("db.internal", 5433, Some("agents")));
        assert!(PgVectorStore::from_url(UrlParts::default(), true).is_err());
        assert!(PgVectorStore::new("postgresql://h/db", "bad name", "public").is_err());
        assert!(PgVectorStore::new("pgvector://h/db", "ok_name", "public").is_ok());
        assert_eq!(PgVectorStore::vector(&[0.5, 1.0, -2.25]), "[0.5,1.0,-2.25]");
    }

    #[test]
    fn filters_become_sql() {
        let wanted = conditions(Some(&json!({"lang": "en", "year": {"$gte": 2020}, "tags": {"$in": ["a", 1]}, "kind": {"$ne": "x"}}))).unwrap();
        let mut args = vec![Arg::Text("[1.0]".into())];
        let sql = PgVectorStore::where_sql(&wanted, &mut args, Some(&["a".to_string()]));
        assert!(sql.starts_with("id = ANY($2::text[])"), "{sql}");
        assert!(sql.contains("(metadata @> $3::jsonb OR metadata @> $4::jsonb)"), "{sql}");
        assert!(sql.contains("(metadata ->> $5)::float8 >= $6"), "{sql}");
        assert!(sql.contains("(metadata -> $8) @> ANY($7::jsonb[])"), "{sql}");
        assert!(sql.contains("NOT (metadata @> $9::jsonb OR metadata @> $10::jsonb)"), "{sql}");
        assert_eq!(args[2], Arg::Text(r#"{"lang":"en"}"#.into()));
        assert_eq!(args[3], Arg::Text(r#"{"lang":["en"]}"#.into()));
        assert_eq!(args[6], Arg::Texts(vec![r#""a""#.into(), "1".into()]));
    }

    /// Runs against a real database with pgvector when `AH_LIVE_POSTGRES_URL` is set.
    #[tokio::test]
    async fn a_real_pgvector_holds_the_contract() {
        let Ok(url) = std::env::var("AH_LIVE_POSTGRES_URL") else { return };
        let table = format!("ah_vec_{}", crate::types::new_id(""));
        let store = PgVectorStore::new(&url, &table, "public").unwrap();
        let report = store.check(3).await;
        assert_eq!(report["ok"], true, "{report}");
        store.clear().await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
        sqlx::query(AssertSqlSafe(format!("DROP TABLE IF EXISTS {}", store.table))).execute(&store.pool().await.unwrap()).await.unwrap();
        store.close().await;
    }
}
