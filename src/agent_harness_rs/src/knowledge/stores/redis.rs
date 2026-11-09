//! Redis — Redis 8, Redis Stack, Redis Cloud, Azure Managed Redis, MemoryDB.
#![cfg(feature = "redis")]

use std::sync::Arc;

use async_trait::async_trait;
use indexmap::IndexMap;
use redis::Value as Reply;
use serde::Serialize;
use serde_json::{Map, Value};

use super::UrlParts;
use crate::errors::Result;
use crate::knowledge::base::{ALWAYS_FILTERABLE, Condition, VectorHit, VectorRecord, VectorStore, store_error};
use crate::memory::providers::redis::RedisLink;

const NAME: &str = "redis";

/// A vector store on a Redis with the query engine.
///
/// ```no_run
/// # use agent_harness::knowledge::vector_store;
/// let store = vector_store("redis://localhost:6379?index=handbook&filterable=source,year:numeric");
/// ```
///
/// Needs a Redis with the query engine — Redis 8, or Redis Stack before it.
///
/// Redis indexes the fields it is told about, so say which metadata is filtered
/// on, and how: `tag` for exact values, `numeric` for numbers and ranges.
/// Everything else in the metadata is kept and returned, but not filterable.
///
/// Each record is a hash at `{index}:{id}`: `vector` (float32 bytes), `ref`
/// (the id), `text`, `meta` (the metadata as JSON) and an `m_{field}` per
/// filterable field — the layout the Python harness writes.
#[derive(Debug)]
pub struct RedisVectorStore {
    link: Arc<RedisLink>,
    pub index: String,
    pub prefix: String,
    /// field → `tag` or `numeric`.
    pub filterable: IndexMap<String, String>,
}

impl RedisVectorStore {
    /// A store on `url` with the index `index` (`knowledge` by default).
    pub fn new(url: &str, index: &str) -> Result<Self> {
        let index = if index.is_empty() { "knowledge" } else { index };
        let link = RedisLink::open(url).map_err(|e| store_error(NAME, e.to_string()))?;
        Ok(Self {
            link: Arc::new(link),
            index: index.into(),
            prefix: format!("{index}:"),
            filterable: ALWAYS_FILTERABLE.iter().map(|k| (k.to_string(), "tag".to_string())).collect(),
        })
    }

    /// Declare more filterable fields: `[("source", "tag"), ("year", "numeric")]`.
    pub fn with_filterable<'a>(mut self, fields: impl IntoIterator<Item = (&'a str, &'a str)>) -> Result<Self> {
        for (key, kind) in fields {
            let kind = kind.to_lowercase();
            if kind != "tag" && kind != "numeric" {
                return Err(store_error(NAME, format!("'{key}' is '{kind}'; a filterable field is 'tag' or 'numeric'")));
            }
            self.filterable.insert(key.to_string(), kind);
        }
        Ok(self)
    }

    /// From `redis://host:6379/0?index=handbook` (`rediss://` for TLS), or the
    /// bare name `redis` for `redis://localhost:6379`.
    ///
    /// `filterable=source,year:numeric` declares fields beyond the ones every
    /// store filters on; a field without a kind is a tag.
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        let index = p.take("index").unwrap_or_else(|| "knowledge".into());
        let declared = p.take("filterable").unwrap_or_default();
        let url = if bare {
            "redis://localhost:6379".to_string()
        } else {
            let scheme = if p.secure || p.scheme.eq_ignore_ascii_case("rediss") { "rediss" } else { "redis" };
            format!("{scheme}://{}{}", p.netloc, p.raw_path)
        };
        let fields: Vec<(&str, &str)> = declared.split(',').map(str::trim).filter(|f| !f.is_empty()).map(|f| f.split_once(':').unwrap_or((f, "tag"))).collect();
        Self::new(&url, &index)?.with_filterable(fields)
    }

    /// The hash field a metadata key is indexed as.
    pub(crate) fn field(name: &str) -> String {
        format!("m_{}", name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect::<String>())
    }

    /// A value as a tag: booleans as `true`/`false`, everything else as Python's `str()`.
    pub(crate) fn tag(value: &Value) -> String {
        match value {
            Value::Bool(b) => if *b { "true" } else { "false" }.into(),
            Value::String(s) => s.clone(),
            Value::Number(n) => n.as_i64().map_or_else(|| py_float(n.as_f64().unwrap_or(0.0)), |i| i.to_string()),
            Value::Null => "None".into(),
            other => other.to_string(),
        }
    }

    /// The query-engine expression for a filter; `*` for none.
    pub(crate) fn expression(&self, wanted: &[Condition]) -> Result<String> {
        let mut parts: Vec<String> = Vec::new();
        for c in wanted {
            let Some(kind) = self.filterable.get(&c.field) else {
                let known: Vec<&str> = self.filterable.keys().map(String::as_str).collect();
                return Err(store_error(
                    NAME,
                    format!(
                        "'{0}' is not filterable in this Redis index — declare it: filterable={{'{0}': 'tag'}} (or 'numeric'). Filterable now: {1}",
                        c.field,
                        if known.is_empty() { "nothing".into() } else { known.join(", ") }
                    ),
                ));
            };
            let field = format!("@{}", Self::field(&c.field));
            if kind == "numeric" {
                if c.op == "$in" {
                    let mut options = Vec::new();
                    for v in c.value.as_array().into_iter().flatten() {
                        let x = py_float(number(v)?);
                        options.push(format!("{field}:[{x} {x}]"));
                    }
                    parts.push(format!("({})", options.join(" | ")));
                    continue;
                }
                let x = py_float(number(&c.value)?);
                let (low, high) = match c.op.as_str() {
                    "$eq" | "$ne" => (x.clone(), x),
                    "$gt" => (format!("({x}"), "+inf".into()),
                    "$gte" => (x, "+inf".into()),
                    "$lt" => ("-inf".into(), format!("({x}")),
                    _ => ("-inf".into(), x),
                };
                let clause = format!("{field}:[{low} {high}]");
                parts.push(if c.op == "$ne" { format!("-{clause}") } else { clause });
                continue;
            }
            if !matches!(c.op.as_str(), "$eq" | "$ne" | "$in") {
                return Err(store_error(NAME, format!("'{}' is a tag; a range needs it declared 'numeric'", c.field)));
            }
            let values: Vec<&Value> = if c.op == "$in" { c.value.as_array().map(|a| a.iter().collect()).unwrap_or_default() } else { vec![&c.value] };
            let tags: Vec<String> = values.iter().map(|v| escape(&Self::tag(v))).collect();
            let clause = format!("{field}:{{{}}}", tags.join("|"));
            parts.push(if c.op == "$ne" { format!("-{clause}") } else { clause });
        }
        Ok(if parts.is_empty() { "*".into() } else { parts.join(" ") })
    }

    /// The hash fields one record is stored as.
    pub(crate) fn fields(&self, r: &VectorRecord) -> Result<Vec<(String, Vec<u8>)>> {
        let vector: Vec<u8> = r.vector.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut fields = vec![
            ("vector".to_string(), vector),
            ("ref".to_string(), r.id.clone().into_bytes()),
            ("text".to_string(), r.text.clone().into_bytes()),
            ("meta".to_string(), python_json(&r.metadata).into_bytes()),
        ];
        for (key, kind) in &self.filterable {
            let Some(value) = r.metadata.get(key).filter(|v| !v.is_null()) else { continue };
            let stored = if kind == "numeric" {
                py_float(number(value)?)
            } else {
                match value {
                    Value::Array(items) => items.iter().map(Self::tag).collect::<Vec<_>>().join(","),
                    one => Self::tag(one),
                }
            };
            fields.push((Self::field(key), stored.into_bytes()));
        }
        Ok(fields)
    }
}

/// A number from a metadata value, as Python's `float()` would take it.
fn number(value: &Value) -> Result<f64> {
    match value {
        Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.trim().parse().map_err(|_| store_error(NAME, format!("could not convert string to float: '{s}'"))),
        other => Err(store_error(NAME, format!("a numeric field takes a number, not {other}"))),
    }
}

/// A float as Python's `repr` writes it: `2024.0`, `1.5`, `inf`.
fn py_float(x: f64) -> String {
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x.fract() == 0.0 && x.abs() < 1e16 { format!("{x:.1}") } else { format!("{x}") }
}

/// A tag with every character but letters, digits and `_` escaped.
fn escape(tag: &str) -> String {
    let mut out = String::with_capacity(tag.len());
    for c in tag.chars() {
        if !(c.is_ascii_alphanumeric() || c == '_') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// JSON the way Python's `json.dumps` writes it: `", "` and `": "` between items.
fn python_json(metadata: &Map<String, Value>) -> String {
    struct Spaced;
    impl serde_json::ser::Formatter for Spaced {
        fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()> {
            if first { Ok(()) } else { writer.write_all(b", ") }
        }
        fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()> {
            if first { Ok(()) } else { writer.write_all(b", ") }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
            writer.write_all(b": ")
        }
    }
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, Spaced);
    if metadata.serialize(&mut ser).is_err() {
        return "{}".into();
    }
    String::from_utf8(out).unwrap_or_else(|_| "{}".into())
}

fn text(v: &Reply) -> String {
    match v {
        Reply::BulkString(b) => String::from_utf8_lossy(b).into_owned(),
        Reply::SimpleString(s) => s.clone(),
        Reply::VerbatimString { text, .. } => text.clone(),
        Reply::Int(i) => i.to_string(),
        Reply::Double(f) => f.to_string(),
        _ => String::new(),
    }
}

fn items(v: &Reply) -> &[Reply] {
    match v {
        Reply::Array(a) | Reply::Set(a) => a,
        _ => &[],
    }
}

/// `[total, key, [field, value, …], key, …]` as a list of `(key, fields)`.
pub(crate) fn rows(answer: &Reply) -> Vec<(String, IndexMap<String, Reply>)> {
    let answer = items(answer);
    let mut out = Vec::new();
    let mut index = 1;
    while index < answer.len() {
        let mut row = IndexMap::new();
        if let Some(fields) = answer.get(index + 1) {
            for pair in items(fields).chunks(2) {
                if let [k, v] = pair {
                    row.insert(text(k), v.clone());
                }
            }
        }
        out.push((text(&answer[index]), row));
        index += 2;
    }
    out
}

fn fail(e: redis::RedisError) -> crate::errors::Error {
    store_error(NAME, format!("redis: {e}"))
}

#[async_trait]
impl VectorStore for RedisVectorStore {
    fn name(&self) -> &str {
        NAME
    }

    fn batch_size(&self) -> usize {
        500
    }

    fn filterable(&self) -> Option<Vec<String>> {
        Some(self.filterable.keys().cloned().collect())
    }

    async fn ensure(&self, dimension: usize) -> Result<()> {
        let mut conn = self.link.conn().await?;
        match redis::cmd("FT.INFO").arg(&self.index).query_async::<Reply>(&mut conn).await {
            Ok(_) => return Ok(()),
            Err(e) if e.to_string().to_lowercase().contains("unknown command") => {
                return Err(store_error(NAME, "this Redis has no query engine — it needs Redis 8 or Redis Stack"));
            }
            Err(_) => {}
        }
        let mut create = redis::cmd("FT.CREATE");
        create.arg(&self.index).arg("ON").arg("HASH").arg("PREFIX").arg(1).arg(&self.prefix).arg("SCHEMA");
        create.arg("vector").arg("VECTOR").arg("HNSW").arg(6).arg("TYPE").arg("FLOAT32").arg("DIM").arg(dimension).arg("DISTANCE_METRIC").arg("COSINE");
        for (key, kind) in &self.filterable {
            create.arg(Self::field(key)).arg(if kind == "tag" { "TAG" } else { "NUMERIC" });
        }
        match create.query_async::<Reply>(&mut conn).await {
            Err(e) if !e.to_string().to_lowercase().contains("already exists") => Err(fail(e)),
            _ => Ok(()),
        }
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        let mut conn = self.link.conn().await?;
        let mut pipe = redis::pipe();
        for r in &records {
            let key = format!("{}{}", self.prefix, r.id);
            // No fields left over from before.
            pipe.cmd("DEL").arg(&key).ignore();
            let hset = pipe.cmd("HSET").arg(&key);
            for (field, value) in self.fields(r)? {
                hset.arg(field).arg(value);
            }
            hset.ignore();
        }
        pipe.query_async::<()>(&mut conn).await.map_err(fail)
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let expression = self.expression(wanted)?;
        let mut conn = self.link.conn().await?;
        let blob: Vec<u8> = vector.iter().flat_map(|x| x.to_le_bytes()).collect();
        let answer: Reply = redis::cmd("FT.SEARCH")
            .arg(&self.index)
            .arg(format!("({expression})=>[KNN {k} @vector $vec AS dist]"))
            .arg("PARAMS")
            .arg(2)
            .arg("vec")
            .arg(blob)
            .arg("SORTBY")
            .arg("dist")
            .arg("RETURN")
            .arg(4)
            .arg("ref")
            .arg("text")
            .arg("meta")
            .arg("dist")
            .arg("LIMIT")
            .arg(0)
            .arg(k)
            .arg("DIALECT")
            .arg(2)
            .query_async(&mut conn)
            .await
            .map_err(fail)?;
        Ok(rows(&answer)
            .into_iter()
            .map(|(key, row)| {
                let meta = row.get("meta").map(text).and_then(|m| serde_json::from_str::<Map<String, Value>>(&m).ok()).unwrap_or_default();
                let id = row.get("ref").map(text).filter(|s| !s.is_empty()).unwrap_or_else(|| key.strip_prefix(&self.prefix).unwrap_or(&key).to_string());
                let dist: f64 = row.get("dist").map(text).and_then(|d| d.parse().ok()).unwrap_or(0.0);
                VectorHit { id, score: 1.0 - dist, text: row.get("text").map(text).unwrap_or_default(), metadata: meta }
            })
            .collect())
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        let mut conn = self.link.conn().await?;
        if wanted.is_empty() {
            let keys: Vec<String> = ids.unwrap_or_default().iter().map(|i| format!("{}{i}", self.prefix)).collect();
            if !keys.is_empty() {
                redis::cmd("DEL").arg(keys).query_async::<()>(&mut conn).await.map_err(fail)?;
            }
            return Ok(());
        }
        let expression = self.expression(wanted)?;
        for _ in 0..10_000 {
            let answer: Reply = redis::cmd("FT.SEARCH")
                .arg(&self.index)
                .arg(&expression)
                .arg("NOCONTENT")
                .arg("LIMIT")
                .arg(0)
                .arg(500)
                .arg("DIALECT")
                .arg(2)
                .query_async(&mut conn)
                .await
                .map_err(fail)?;
            let found = items(&answer);
            let keys: Vec<String> = found
                .iter()
                .skip(1)
                .map(text)
                .filter(|key| ids.is_none_or(|ids| ids.iter().any(|i| key.strip_prefix(&self.prefix) == Some(i.as_str()))))
                .collect();
            if keys.is_empty() {
                return Ok(());
            }
            redis::cmd("DEL").arg(keys).query_async::<()>(&mut conn).await.map_err(fail)?;
            if found.len().saturating_sub(1) < 500 || ids.is_some() {
                return Ok(());
            }
        }
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let mut conn = self.link.conn().await?;
        let info: Reply = redis::cmd("FT.INFO").arg(&self.index).query_async(&mut conn).await.map_err(fail)?;
        let pairs = items(&info);
        let found = pairs.chunks(2).find(|p| p.len() == 2 && text(&p[0]) == "num_docs").map(|p| text(&p[1])).unwrap_or_default();
        Ok(found.parse::<f64>().unwrap_or(0.0) as usize)
    }

    async fn clear(&self) -> Result<()> {
        let mut conn = self.link.conn().await?;
        // Nothing to drop is not a failure.
        let _ = redis::cmd("FT.DROPINDEX").arg(&self.index).arg("DD").query_async::<Reply>(&mut conn).await;
        Ok(())
    }

    async fn close(&self) {
        self.link.close().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn store() -> RedisVectorStore {
        RedisVectorStore::new("redis://localhost:6379", "kb").unwrap().with_filterable([("year", "numeric"), ("src", "tag")]).unwrap()
    }

    fn cond(field: &str, op: &str, value: Value) -> Condition {
        Condition { field: field.into(), op: op.into(), value }
    }

    #[test]
    fn filters_translate_like_python() {
        let s = store();
        assert_eq!(s.expression(&[]).unwrap(), "*");
        assert_eq!(s.expression(&[cond("doc", "$eq", json!("faq.md"))]).unwrap(), r"@m_doc:{faq\.md}");
        assert_eq!(s.expression(&[cond("src", "$ne", json!(true))]).unwrap(), "-@m_src:{true}");
        assert_eq!(s.expression(&[cond("scope", "$in", json!(["a b", "c"]))]).unwrap(), r"@m_scope:{a\ b|c}");
        assert_eq!(s.expression(&[cond("year", "$gte", json!(2024))]).unwrap(), "@m_year:[2024.0 +inf]");
        assert_eq!(s.expression(&[cond("year", "$lt", json!(2.5))]).unwrap(), "@m_year:[-inf (2.5]");
        assert_eq!(s.expression(&[cond("year", "$ne", json!(1))]).unwrap(), "-@m_year:[1.0 1.0]");
        assert_eq!(s.expression(&[cond("year", "$in", json!([1, 2]))]).unwrap(), "(@m_year:[1.0 1.0] | @m_year:[2.0 2.0])");
        assert_eq!(s.expression(&[cond("doc", "$eq", json!("x")), cond("year", "$gt", json!(3))]).unwrap(), "@m_doc:{x} @m_year:[(3.0 +inf]");
        assert!(s.expression(&[cond("lang", "$eq", json!("en"))]).is_err(), "undeclared");
        assert!(s.expression(&[cond("src", "$gt", json!(1))]).is_err(), "a range on a tag");
        assert!(RedisVectorStore::new("redis://x", "k").unwrap().with_filterable([("a", "text")]).is_err());
    }

    #[test]
    fn records_are_stored_like_python() {
        let s = store();
        let r = VectorRecord::new("a", vec![1.0, 0.5], "t", json!({"doc": "d", "year": 2024, "src": ["x", "y"], "n": 1.5, "é": "ü"}));
        let fields: IndexMap<String, Vec<u8>> = s.fields(&r).unwrap().into_iter().collect();
        assert_eq!(fields["vector"], [1.0f32.to_le_bytes(), 0.5f32.to_le_bytes()].concat());
        assert_eq!(fields["ref"], b"a");
        assert_eq!(String::from_utf8(fields["meta"].clone()).unwrap(), r#"{"doc": "d", "year": 2024, "src": ["x", "y"], "n": 1.5, "é": "ü"}"#);
        assert_eq!(fields["m_doc"], b"d");
        assert_eq!(fields["m_year"], b"2024.0");
        assert_eq!(fields["m_src"], b"x,y");
        assert_eq!(RedisVectorStore::field("a.b-c"), "m_a_b_c");
    }

    #[test]
    fn urls_and_replies() {
        let p = UrlParts::parse("rediss://:pw@host:6380/0?index=handbook&filterable=year:numeric,lang").unwrap();
        let s = RedisVectorStore::from_url(p, false).unwrap();
        assert_eq!((s.index.as_str(), s.prefix.as_str()), ("handbook", "handbook:"));
        assert_eq!(s.filterable.get("year").map(String::as_str), Some("numeric"));
        assert_eq!(s.filterable.get("lang").map(String::as_str), Some("tag"));
        let bare = RedisVectorStore::from_url(UrlParts { scheme: "redis".into(), ..Default::default() }, true).unwrap();
        assert_eq!(bare.index, "knowledge");

        let b = |s: &str| Reply::BulkString(s.as_bytes().to_vec());
        let answer = Reply::Array(vec![Reply::Int(1), b("kb:a"), Reply::Array(vec![b("ref"), b("a"), b("dist"), b("0.25")])]);
        let got = rows(&answer);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, "kb:a");
        assert_eq!(text(&got[0].1["dist"]), "0.25");
    }

    #[tokio::test]
    async fn live_check() {
        let Ok(url) = std::env::var("AH_LIVE_REDIS_URL") else { return };
        let index = format!("ah_check_{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        let store = RedisVectorStore::new(&url, &index).unwrap().with_filterable([("n", "numeric")]).unwrap();
        let report = store.check(8).await;
        store.clear().await.unwrap();
        assert_eq!(report["ok"], true, "{report}");
    }
}
