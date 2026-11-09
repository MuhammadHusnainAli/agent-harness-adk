//! MongoDB Atlas Vector Search — Atlas, or the Atlas local deployment.
#![cfg(feature = "mongodb")]

use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::TryStreamExt;
use mongodb::bson::{Bson, Document, doc};
use mongodb::{Collection, SearchIndexModel, SearchIndexType};
use serde_json::Value;

use super::UrlParts;
use crate::errors::Result;
use crate::knowledge::base::{ALWAYS_FILTERABLE, Condition, VectorHit, VectorRecord, VectorStore, store_error};
use crate::memory::providers::mongo::{MongoLink, doc_to_json, to_bson, to_document, to_json};

const NAME: &str = "mongodb";

/// A vector store on Atlas Vector Search.
///
/// ```no_run
/// # use agent_harness::knowledge::vector_store;
/// let store = vector_store("mongodb+srv://user:pass@cluster.mongodb.net?database=kb&collection=handbook");
/// ```
///
/// Needs a deployment with Atlas Search: Atlas itself, or
/// `mongodb/mongodb-atlas-local`. A plain community server has no vector search.
///
/// Atlas filters on the fields its index was told about: name them in
/// `filterable`. A new index takes a few seconds to become queryable; `ensure`
/// waits for it.
///
/// A record is `{_id, vector, text, meta}`; the index (`vector_index` by
/// default) is a `vectorSearch` index on `vector` (cosine) with a `filter`
/// field per `meta.{name}` — the layout the Python harness writes.
#[derive(Debug)]
pub struct MongoVectorStore {
    link: MongoLink,
    pub database: String,
    pub collection: String,
    pub index: String,
    pub filterable: Vec<String>,
    /// How long `ensure` waits for a new index to become queryable.
    pub ready_timeout: Duration,
}

impl MongoVectorStore {
    /// A store on `url`, in `knowledge.vectors` with the index `vector_index`.
    pub fn new(url: &str) -> Result<Self> {
        let link = MongoLink::new(url).map_err(|e| store_error(NAME, e.to_string()))?;
        Ok(Self {
            link,
            database: "knowledge".into(),
            collection: "vectors".into(),
            index: "vector_index".into(),
            filterable: ALWAYS_FILTERABLE.iter().map(|s| s.to_string()).collect(),
            ready_timeout: Duration::from_secs(120),
        })
    }

    /// Declare more metadata fields the index filters on.
    pub fn with_filterable<I: IntoIterator<Item = S>, S: Into<String>>(mut self, fields: I) -> Self {
        for f in fields {
            let f = f.into();
            if !f.is_empty() && !self.filterable.contains(&f) {
                self.filterable.push(f);
            }
        }
        self
    }

    /// From `mongodb://…` or `mongodb+srv://…`. `database`, `collection` and
    /// `index` in the query string name where the vectors go; `filterable=a,b`
    /// and `ready_timeout=` (seconds) set those; everything else stays on the
    /// connection string.
    pub fn from_url(mut p: UrlParts, bare: bool) -> Result<Self> {
        if bare {
            return Err(store_error(NAME, "MongoVectorStore needs a connection string: mongodb://… or mongodb+srv://…"));
        }
        let database = p.take("database");
        let collection = p.take("collection");
        let index = p.take("index");
        let filterable = p.take("filterable").unwrap_or_default();
        let timeout = p.take("ready_timeout").and_then(|t| t.parse::<f64>().ok()).filter(|t| t.is_finite() && *t >= 0.0);
        // `mongo://` and `atlas://` name the store; the driver wants `mongodb://`.
        let scheme = if p.scheme.eq_ignore_ascii_case("mongodb+srv") { "mongodb+srv" } else { "mongodb" };
        if p.secure {
            p.given.insert("tls".into(), "true".into());
        }
        let query = p.rest_query();
        let url = format!("{scheme}://{}{}{}", p.netloc, p.raw_path, if query.is_empty() { String::new() } else { format!("?{query}") });
        let mut store = Self::new(&url)?.with_filterable(filterable.split(',').map(str::trim).filter(|f| !f.is_empty()).map(String::from));
        if let Some(d) = database {
            store.database = d;
        }
        if let Some(c) = collection {
            store.collection = c;
        }
        if let Some(i) = index {
            store.index = i;
        }
        if let Some(t) = timeout {
            store.ready_timeout = Duration::from_secs_f64(t);
        }
        Ok(store)
    }

    async fn coll(&self) -> Result<Collection<Document>> {
        Ok(self.link.client().await.map_err(|e| store_error(NAME, e.to_string()))?.database(&self.database).collection(&self.collection))
    }

    async fn indexes(&self, coll: &Collection<Document>) -> Result<Vec<Document>> {
        let cursor = coll.list_search_indexes().name(&self.index).await.map_err(fail)?;
        cursor.try_collect().await.map_err(fail)
    }

    /// The search index definition for vectors of `dimension`.
    pub(crate) fn definition(&self, dimension: usize) -> Document {
        let mut fields: Vec<Bson> = vec![Bson::Document(doc! {"type": "vector", "path": "vector", "numDimensions": dimension as i32, "similarity": "cosine"})];
        fields.extend(self.filterable.iter().map(|name| Bson::Document(doc! {"type": "filter", "path": format!("meta.{name}")})));
        doc! {"fields": fields}
    }

    /// The `$vectorSearch` filter; empty for none.
    pub(crate) fn filter(&self, wanted: &[Condition]) -> Result<Document> {
        let mut parts: Vec<Document> = Vec::new();
        for c in wanted {
            if !self.filterable.contains(&c.field) {
                return Err(store_error(
                    NAME,
                    format!(
                        "'{}' is not filterable in this Atlas index — name it in filterable=[...]. Filterable now: {}",
                        c.field,
                        if self.filterable.is_empty() { "nothing".into() } else { self.filterable.join(", ") }
                    ),
                ));
            }
            let mut condition = Document::new();
            condition.insert(c.op.clone(), to_bson(&c.value));
            let mut part = Document::new();
            part.insert(format!("meta.{}", c.field), condition);
            parts.push(part);
        }
        Ok(match parts.len() {
            0 => Document::new(),
            1 => parts.remove(0),
            _ => doc! {"$and": parts},
        })
    }

    /// The `$vectorSearch` stage.
    pub(crate) fn stage(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Document> {
        let mut stage = doc! {
            "index": &self.index, "path": "vector",
            "queryVector": vector.iter().map(|x| Bson::Double(*x as f64)).collect::<Vec<_>>(),
            "numCandidates": (k * 15).clamp(100, 10_000) as i32,
            "limit": k as i32,
        };
        let found = self.filter(wanted)?;
        if !found.is_empty() {
            stage.insert("filter", found);
        }
        Ok(stage)
    }

    /// A plain delete filters on any field, declared or not.
    pub(crate) fn delete_query(ids: Option<&[String]>, wanted: &[Condition]) -> Document {
        let mut query = Document::new();
        if let Some(ids) = ids {
            query.insert("_id", doc! {"$in": ids.to_vec()});
        }
        for c in wanted {
            let key = format!("meta.{}", c.field);
            let mut ops = query.get_document(&key).cloned().unwrap_or_default();
            ops.insert(c.op.clone(), to_bson(&c.value));
            query.insert(key, ops);
        }
        query
    }

    /// A record as it is stored.
    pub(crate) fn document(r: &VectorRecord) -> Document {
        doc! {
            "_id": &r.id,
            "vector": r.vector.iter().map(|x| Bson::Double(*x as f64)).collect::<Vec<_>>(),
            "text": &r.text,
            "meta": to_document(&r.metadata),
        }
    }
}

fn fail(e: impl std::fmt::Display) -> crate::errors::Error {
    store_error(NAME, format!("mongodb: {e}").chars().take(400).collect::<String>())
}

#[async_trait]
impl VectorStore for MongoVectorStore {
    fn name(&self) -> &str {
        NAME
    }

    fn batch_size(&self) -> usize {
        500
    }

    fn filterable(&self) -> Option<Vec<String>> {
        Some(self.filterable.clone())
    }

    async fn ensure(&self, dimension: usize) -> Result<()> {
        let coll = self.coll().await?;
        let found = self.indexes(&coll).await.unwrap_or_default();
        if found.is_empty() {
            let db = self.link.client().await.map_err(|e| store_error(NAME, e.to_string()))?.database(&self.database);
            let names = db.list_collection_names().await.map_err(fail)?;
            if !names.contains(&self.collection) {
                db.create_collection(&self.collection).await.map_err(fail)?;
            }
            let model =
                SearchIndexModel::builder().definition(self.definition(dimension)).name(self.index.clone()).index_type(SearchIndexType::VectorSearch).build();
            if let Err(e) = coll.create_search_index(model).await
                && !e.to_string().to_lowercase().contains("already exists")
            {
                return Err(store_error(NAME, format!("mongodb could not create the vector index: {e}").chars().take(400).collect::<String>()));
            }
        }
        let deadline = Instant::now() + self.ready_timeout;
        while Instant::now() < deadline {
            let found = self.indexes(&coll).await?;
            if let Some(first) = found.first()
                && (first.get_bool("queryable").unwrap_or(false) || first.get_str("status").is_ok_and(|s| s == "READY"))
            {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        Err(store_error(NAME, format!("the vector index '{}' was not ready after {:.0}s", self.index, self.ready_timeout.as_secs_f64())))
    }

    async fn upsert_batch(&self, records: Vec<VectorRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let client = self.link.client().await.map_err(|e| store_error(NAME, e.to_string()))?;
        // One unordered `update` command of replace-or-insert statements — what
        // a bulk of `ReplaceOne(upsert=True)` sends, on any server version.
        let updates: Vec<Document> = records.iter().map(|r| doc! {"q": {"_id": &r.id}, "u": Self::document(r), "upsert": true, "multi": false}).collect();
        let reply = client.database(&self.database).run_command(doc! {"update": &self.collection, "updates": updates, "ordered": false}).await.map_err(fail)?;
        if let Ok(errors) = reply.get_array("writeErrors")
            && let Some(Bson::Document(first)) = errors.first()
        {
            return Err(fail(format!("{} of {} records were not written: {}", errors.len(), records.len(), first.get_str("errmsg").unwrap_or("write error"))));
        }
        Ok(())
    }

    async fn query_raw(&self, vector: &[f32], k: usize, wanted: &[Condition]) -> Result<Vec<VectorHit>> {
        let stage = self.stage(vector, k, wanted)?;
        let coll = self.coll().await?;
        let pipeline = vec![doc! {"$vectorSearch": stage}, doc! {"$project": {"text": 1, "meta": 1, "score": {"$meta": "vectorSearchScore"}}}];
        let cursor = coll.aggregate(pipeline).await.map_err(fail)?;
        let rows: Vec<Document> = cursor.try_collect().await.map_err(fail)?;
        // Atlas scores cosine as (1 + cosine) / 2.
        Ok(rows
            .into_iter()
            .map(|row| {
                let id = match row.get("_id") {
                    Some(Bson::String(s)) => s.clone(),
                    Some(other) => match to_json(other) {
                        Value::String(s) => s,
                        v => v.to_string(),
                    },
                    None => String::new(),
                };
                let score = match row.get("score") {
                    Some(Bson::Double(f)) => *f,
                    Some(Bson::Int32(i)) => *i as f64,
                    Some(Bson::Int64(i)) => *i as f64,
                    _ => 0.0,
                };
                VectorHit {
                    id,
                    score: 2.0 * score - 1.0,
                    text: row.get_str("text").unwrap_or_default().to_string(),
                    metadata: row.get_document("meta").map(doc_to_json).unwrap_or_default(),
                }
            })
            .collect())
    }

    async fn delete_raw(&self, ids: Option<&[String]>, wanted: &[Condition]) -> Result<()> {
        self.coll().await?.delete_many(Self::delete_query(ids, wanted)).await.map_err(fail)?;
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.coll().await?.count_documents(doc! {}).await.map_err(fail)? as usize)
    }

    async fn clear(&self) -> Result<()> {
        self.coll().await?.delete_many(doc! {}).await.map_err(fail)?;
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

    fn cond(field: &str, op: &str, value: Value) -> Condition {
        Condition { field: field.into(), op: op.into(), value }
    }

    fn store() -> MongoVectorStore {
        MongoVectorStore::new("mongodb://localhost:27017").unwrap().with_filterable(["year"])
    }

    #[test]
    fn filters_and_stages_match_python() {
        let s = store();
        assert_eq!(s.filter(&[]).unwrap(), doc! {});
        assert_eq!(s.filter(&[cond("doc", "$eq", json!("a"))]).unwrap(), doc! {"meta.doc": {"$eq": "a"}});
        assert_eq!(
            s.filter(&[cond("doc", "$eq", json!("a")), cond("year", "$gte", json!(2024))]).unwrap(),
            doc! {"$and": [{"meta.doc": {"$eq": "a"}}, {"meta.year": {"$gte": 2024}}]}
        );
        assert!(s.filter(&[cond("lang", "$eq", json!("en"))]).is_err());
        let stage = s.stage(&[1.0, 0.0], 5, &[cond("scope", "$in", json!(["b", "c"]))]).unwrap();
        assert_eq!(stage.get_str("index").unwrap(), "vector_index");
        assert_eq!(stage.get_i32("numCandidates").unwrap(), 100);
        assert_eq!(stage.get_i32("limit").unwrap(), 5);
        assert_eq!(stage.get_document("filter").unwrap(), &doc! {"meta.scope": {"$in": ["b", "c"]}});
        assert_eq!(s.stage(&[1.0], 1000, &[]).unwrap().get_i32("numCandidates").unwrap(), 10_000);
        let ids = vec!["a".to_string()];
        assert_eq!(
            MongoVectorStore::delete_query(Some(&ids), &[cond("n", "$gte", json!(1)), cond("n", "$lt", json!(3))]),
            doc! {"_id": {"$in": ["a"]}, "meta.n": {"$gte": 1, "$lt": 3}}
        );
    }

    #[test]
    fn documents_and_index_match_python() {
        let r = VectorRecord::new("a", vec![0.5, 1.0], "t", json!({"doc": "d", "n": 1}));
        assert_eq!(MongoVectorStore::document(&r), doc! {"_id": "a", "vector": [0.5, 1.0], "text": "t", "meta": {"doc": "d", "n": 1}});
        let def = store().definition(8);
        let fields = def.get_array("fields").unwrap();
        assert_eq!(fields[0], Bson::Document(doc! {"type": "vector", "path": "vector", "numDimensions": 8, "similarity": "cosine"}));
        assert_eq!(fields.len(), 1 + ALWAYS_FILTERABLE.len() + 1);
        assert_eq!(fields.last().unwrap(), &Bson::Document(doc! {"type": "filter", "path": "meta.year"}));
    }

    #[test]
    fn urls_come_apart() {
        let p = UrlParts::parse("mongodb+srv://u:p@c.example.net/admin?database=kb&collection=hb&index=ix&filterable=year&retryWrites=true").unwrap();
        let s = MongoVectorStore::from_url(p, false).unwrap();
        assert_eq!((s.database.as_str(), s.collection.as_str(), s.index.as_str()), ("kb", "hb", "ix"));
        assert!(s.filterable.contains(&"year".to_string()));
        assert!(MongoVectorStore::from_url(UrlParts { scheme: "mongodb".into(), ..Default::default() }, true).is_err());
        let atlas = MongoVectorStore::from_url(UrlParts::parse("atlas://localhost:27017").unwrap(), false).unwrap();
        assert_eq!(atlas.database, "knowledge");
    }

    #[tokio::test]
    async fn live_check() {
        let Ok(url) = std::env::var("AH_LIVE_MONGO_URL") else { return };
        let mut store = MongoVectorStore::new(&url).unwrap().with_filterable(["n"]);
        store.database = "ah_test".into();
        store.collection = format!("vectors_{}", &uuid::Uuid::new_v4().simple().to_string()[..8]);
        let report = store.check(8).await;
        let client = store.link.client().await.unwrap();
        client.database("ah_test").collection::<Document>(&store.collection).drop().await.unwrap();
        assert_eq!(report["ok"], true, "{report}");
    }
}
