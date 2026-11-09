//! Sessions in DynamoDB.
//!
//! An item may not exceed 400 KB, and a conversation easily does. So a
//! session is a small *head* item — who owns it, its version, how many pieces
//! — and its body, zlib-compressed and cut into pieces of its own. The head is
//! written last and only if the version still matches: until that one
//! conditional write succeeds, the new pieces are invisible, and the old ones
//! are removed only after it has.
//!
//! ```text
//! pk = _session#{id}    sk = head | body#{rev}#{n}
//! pk = _sessions#{tenant}#{user}  (and #_all)    sk = {id}      ← for listing
//! ```
//!
//! The items are the Python `DynamoDBSessionStore`'s, so one table serves both.

use std::io::{Read, Write};

use async_trait::async_trait;
use base64::Engine as _;
use serde_json::{Value, json};

use crate::errors::{Error, Result};
use crate::memory::providers::dynamodb::DynamoDbMemory;
use crate::runtime::session::{DurableBackend, ListFilter, Session};
use crate::types::new_id;

/// The largest piece of a compressed body, in bytes.
pub const PIECE: usize = 350_000;

fn b64() -> base64::engine::GeneralPurpose {
    base64::engine::general_purpose::STANDARD
}

/// Sessions in the DynamoDB memory backend's table (`pk` / `sk` keys).
///
/// ```no_run
/// # use agent_harness::sessions::session_provider;
/// let store = session_provider("dynamodb://agents?region=eu-west-1");
/// ```
#[derive(Debug, Clone)]
pub struct DynamoDbSessionStore {
    dynamo: DynamoDbMemory,
}

impl DynamoDbSessionStore {
    pub fn new(dynamo: DynamoDbMemory) -> Self {
        Self { dynamo }
    }

    fn table(&self) -> &str {
        &self.dynamo.table_name
    }

    fn pk(session_id: &str) -> Value {
        json!({"S": format!("_session#{session_id}")})
    }

    /// The listing partition of one owner.
    pub fn owner(tenant_id: Option<&str>, user_id: Option<&str>) -> String {
        let or = |v: Option<&str>| v.filter(|s| !s.is_empty()).unwrap_or("_").to_string();
        format!("_sessions#{}#{}", or(tenant_id), or(user_id))
    }

    fn piece_key(session_id: &str, rev: &str, n: usize) -> Value {
        json!({"pk": Self::pk(session_id), "sk": {"S": format!("body#{rev}#{n:05}")}})
    }

    async fn delete_key(&self, key: Value) -> Result<()> {
        self.dynamo.request("DeleteItem", json!({"TableName": self.table(), "Key": key})).await.map(|_| ())
    }

    async fn drop_pieces(&self, session_id: &str, rev: &str, pieces: usize) -> Result<()> {
        for n in 0..pieces {
            self.delete_key(Self::piece_key(session_id, rev, n)).await?;
        }
        Ok(())
    }

    async fn get(&self, key: Value) -> Result<Option<Value>> {
        let found = self.dynamo.request("GetItem", json!({"TableName": self.table(), "ConsistentRead": true, "Key": key})).await?;
        Ok(found.get("Item").filter(|i| i.is_object()).cloned())
    }

    async fn head(&self, session_id: &str) -> Result<Option<Value>> {
        self.get(json!({"pk": Self::pk(session_id), "sk": {"S": "head"}})).await
    }

    fn number(v: &Value) -> u64 {
        v["N"].as_str().and_then(|n| n.parse::<f64>().ok()).unwrap_or(0.0) as u64
    }

    async fn load(&self, session_id: &str) -> Result<Option<Session>> {
        let Some(head) = self.head(session_id).await? else { return Ok(None) };
        let rev = head["rev"]["S"].as_str().unwrap_or("").to_string();
        let mut body = Vec::new();
        for n in 0..Self::number(&head["pieces"]) as usize {
            // Replaced under us: the caller reads again.
            let Some(piece) = self.get(Self::piece_key(session_id, &rev, n)).await? else { return Ok(None) };
            body.extend(b64().decode(piece["body"]["B"].as_str().unwrap_or("")).unwrap_or_default());
        }
        let mut text = Vec::new();
        if flate2::read::ZlibDecoder::new(&body[..]).read_to_end(&mut text).is_err() {
            return Ok(None);
        }
        let Ok(mut found) = serde_json::from_slice::<Session>(&text) else { return Ok(None) };
        found.version = Self::number(&head["version"]);
        Ok(Some(found))
    }
}

#[async_trait]
impl DurableBackend for DynamoDbSessionStore {
    async fn read(&self, session_id: &str) -> Result<Option<Session>> {
        self.dynamo.ensure().await?;
        // A save can swap the pieces between the two reads; once more is
        // enough, because the head it then finds is the new one.
        match self.load(session_id).await? {
            Some(s) => Ok(Some(s)),
            None => self.load(session_id).await,
        }
    }

    async fn write(&self, session: &Session, expect: u64) -> Result<bool> {
        self.dynamo.ensure().await?;
        let rev = new_id("rev");
        let json_body = serde_json::to_vec(session).map_err(|e| Error::Io(e.to_string()))?;
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&json_body).map_err(|e| Error::Io(e.to_string()))?;
        let body = encoder.finish().map_err(|e| Error::Io(e.to_string()))?;
        let pieces: Vec<&[u8]> = if body.is_empty() { vec![&[][..]] } else { body.chunks(PIECE).collect() };
        let owner = Self::owner(session.tenant_id.as_deref(), session.user_id.as_deref());

        for (n, piece) in pieces.iter().enumerate() {
            let item = json!({"pk": Self::pk(&session.id), "sk": {"S": format!("body#{rev}#{n:05}")}, "body": {"B": b64().encode(piece)}});
            self.dynamo.request("PutItem", json!({"TableName": self.table(), "Item": item})).await?;
        }
        let head = json!({
            "pk": Self::pk(&session.id), "sk": {"S": "head"},
            "version": {"N": session.version.to_string()}, "rev": {"S": rev},
            "pieces": {"N": pieces.len().to_string()}, "owner": {"S": owner},
        });
        let mut put = json!({"TableName": self.table(), "Item": head, "ReturnValues": "ALL_OLD"});
        if expect == 0 {
            put["ConditionExpression"] = json!("attribute_not_exists(pk)");
        } else {
            put["ConditionExpression"] = json!("#v = :expect");
            put["ExpressionAttributeNames"] = json!({"#v": "version"});
            put["ExpressionAttributeValues"] = json!({":expect": {"N": expect.to_string()}});
        }
        let old = match self.dynamo.call("PutItem", &put).await? {
            Ok(old) => old,
            Err(r) if r.kind == "ConditionalCheckFailedException" => {
                self.drop_pieces(&session.id, &rev, pieces.len()).await?;
                return Ok(false);
            }
            Err(r) => {
                return Err(Error::VectorStore {
                    message: format!("dynamodb: saving session {}: {}: {}", session.id, r.kind, r.message),
                    store: "dynamodb".into(),
                    status: Some(400),
                    retryable: false,
                });
            }
        };
        if let Some(was) = old.get("Attributes").filter(|a| a.as_object().is_some_and(|o| !o.is_empty())) {
            self.drop_pieces(&session.id, was["rev"]["S"].as_str().unwrap_or(""), Self::number(&was["pieces"]) as usize).await?;
            if let Some(previous) = was["owner"]["S"].as_str()
                && previous != owner
            {
                self.delete_key(json!({"pk": {"S": previous}, "sk": {"S": session.id}})).await?;
            }
        }
        let agent = if session.agent.is_empty() { "_" } else { session.agent.as_str() };
        for pk in [owner.as_str(), "_sessions#_all"] {
            let row = json!({"pk": {"S": pk}, "sk": {"S": session.id}, "updated": {"N": session.updated.to_string()}, "agent": {"S": agent}});
            self.dynamo.request("PutItem", json!({"TableName": self.table(), "Item": row})).await?;
        }
        Ok(true)
    }

    async fn list_sessions(&self, filter: &ListFilter) -> Result<Vec<Session>> {
        self.dynamo.ensure().await?;
        let exact = filter.user_id.is_some() && filter.tenant_id.is_some();
        let pk = if exact { Self::owner(filter.tenant_id.as_deref(), filter.user_id.as_deref()) } else { "_sessions#_all".into() };
        let mut rows: Vec<Value> = Vec::new();
        let mut query = json!({"TableName": self.table(), "KeyConditionExpression": "pk = :pk", "ExpressionAttributeValues": {":pk": {"S": pk}}});
        loop {
            let page = self.dynamo.request("Query", query.clone()).await?;
            rows.extend(page["Items"].as_array().cloned().unwrap_or_default());
            match page.get("LastEvaluatedKey").filter(|k| k.is_object()) {
                Some(k) => query["ExclusiveStartKey"] = k.clone(),
                None => break,
            }
        }
        let updated = |r: &Value| r["updated"]["N"].as_str().and_then(|n| n.parse::<f64>().ok()).unwrap_or(0.0);
        rows.sort_by(|a, b| updated(b).total_cmp(&updated(a)));
        let mut out = Vec::new();
        for row in rows {
            if filter.agent.as_ref().is_some_and(|a| row["agent"]["S"].as_str() != Some(a.as_str())) {
                continue;
            }
            if let Some(found) = self.load(row["sk"]["S"].as_str().unwrap_or("")).await?
                && filter.user_id.as_ref().is_none_or(|u| found.user_id.as_ref() == Some(u))
                && filter.tenant_id.as_ref().is_none_or(|t| found.tenant_id.as_ref() == Some(t))
            {
                out.push(found);
                if out.len() >= filter.limit {
                    break;
                }
            }
        }
        Ok(out)
    }

    async fn remove(&self, session_id: &str) -> Result<()> {
        self.dynamo.ensure().await?;
        let Some(head) = self.head(session_id).await? else { return Ok(()) };
        self.delete_key(json!({"pk": Self::pk(session_id), "sk": {"S": "head"}})).await?;
        self.drop_pieces(session_id, head["rev"]["S"].as_str().unwrap_or(""), Self::number(&head["pieces"]) as usize).await?;
        for pk in [head["owner"]["S"].as_str().unwrap_or(""), "_sessions#_all"] {
            self.delete_key(json!({"pk": {"S": pk}, "sk": {"S": session_id}})).await?;
        }
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "DynamoDBSessionStore"
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Map;

    use super::*;
    use crate::memory::base::MemoryStore;
    use crate::memory::providers::dynamodb::fake;
    use crate::memory::providers::objects::testing::exercise_sessions;

    #[tokio::test]
    async fn a_table_holds_the_contract_and_clears_every_piece_behind_it() {
        let (table, items) = fake::table().await;
        let store = table.sessions(&Map::new()).unwrap().unwrap();
        assert_eq!(store.kind(), "DynamoDBSessionStore");
        exercise_sessions(store.as_ref(), true).await;
        // Every piece of every replaced body was cleared away behind it.
        assert!(items.lock().unwrap().is_empty(), "{:?}", items.lock().unwrap().keys().collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn a_body_is_zlib_in_pieces_python_can_read() {
        let (table, items) = fake::table().await;
        let store = DynamoDbSessionStore::new(table);
        let mut s = Session { user_id: Some("alice".into()), ..Session::new("bot") };
        crate::runtime::session::SessionStore::save(&store, &mut s).await.unwrap();
        let held = items.lock().unwrap();
        let head = &held[&(format!("_session#{}", s.id), "head".to_string())];
        assert_eq!(
            (head["version"]["N"].as_str(), head["pieces"]["N"].as_str(), head["owner"]["S"].as_str()),
            (Some("1"), Some("1"), Some("_sessions#_#alice"))
        );
        let rev = head["rev"]["S"].as_str().unwrap();
        let piece = &held[&(format!("_session#{}", s.id), format!("body#{rev}#00000"))];
        let raw = b64().decode(piece["body"]["B"].as_str().unwrap()).unwrap();
        assert_eq!(&raw[..1], &[0x78], "a zlib stream, as zlib.compress writes");
        assert!(held.contains_key(&("_sessions#_all".to_string(), s.id.clone())));
    }
}
