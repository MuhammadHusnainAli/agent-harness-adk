//! Audit trail: an immutable who-did-what record. What compliance will ask for.
//!
//! Every entry carries the hash of the one before it, so a tampered or deleted
//! record breaks the chain and `verify()` says exactly where.
//!
//! A hash chain proves order and integrity, but anyone able to rewrite the file
//! can rebuild the chain. Give the trail a [`Signer`] and every entry's hash is
//! also signed, so a rebuilt chain fails verification without the key.
//!
//! This is deliberately separate from the run journal. The journal is for
//! debugging and can be summarised, trimmed or thrown away; the audit trail is
//! append-only and answers "who did what, when, and was it allowed". The hashes
//! are computed over the same bytes Python hashes, so a trail written by one
//! verifies in the other.

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::pyjson::{Dumps, dumps_with};
use crate::types::now;

pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn genesis() -> String {
    GENESIS.into()
}

/// One immutable record. `hash` covers every field above it plus `prev_hash`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEntry {
    #[serde(default)]
    pub seq: u64,
    #[serde(default = "now")]
    pub ts: f64,
    /// Which agent, or "human".
    #[serde(default)]
    pub actor: String,
    /// tool_call | permission | run_start | run_end | stop | ...
    #[serde(default)]
    pub action: String,
    /// The tool, sub-agent or resource acted on.
    #[serde(default)]
    pub target: String,
    /// allow | deny | ask | ok | error
    #[serde(default)]
    pub decision: String,
    #[serde(default)]
    pub run_id: String,
    #[serde(default)]
    pub detail: Map<String, Value>,
    #[serde(default = "genesis")]
    pub prev_hash: String,
    #[serde(default)]
    pub hash: String,
    /// A signature over `hash`, and which key made it. Outside the hashed
    /// payload, so an unsigned trail and a signed one chain identically.
    #[serde(default)]
    pub signature: String,
    #[serde(default)]
    pub key_id: String,
}

impl AuditEntry {
    /// The exact bytes the hash covers. Sorted, so it is reproducible.
    pub fn payload(&self) -> String {
        let v = json!({
            "seq": self.seq, "ts": self.ts, "actor": self.actor, "action": self.action,
            "target": self.target, "decision": self.decision, "run_id": self.run_id,
            "detail": self.detail, "prev_hash": self.prev_hash,
        });
        dumps_with(&v, Dumps::canonical())
    }

    pub fn compute_hash(&self) -> String {
        Sha256::digest(self.payload().as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn line(&self) -> String {
        let target = if self.target.is_empty() { String::new() } else { format!(" {}", self.target) };
        format!("[{}] {} {}{target} → {}", stamp(self.ts), self.actor, self.action, self.decision)
    }
}

/// `YYYY-MM-DD HH:MM:SS`, UTC.
pub(crate) fn stamp(ts: f64) -> String {
    let (_, date) = crate::llm_providers::sigv4::amz_dates(ts);
    let secs = (ts.floor() as i64).rem_euclid(86_400);
    format!("{}-{}-{} {:02}:{:02}:{:02}", &date[..4], &date[4..6], &date[6..], secs / 3600, secs % 3600 / 60, secs % 60)
}

/// Signs entries: `key_id`, `sign(bytes) -> signature`, `verify(bytes, signature)`.
pub trait Signer: Send + Sync {
    fn key_id(&self) -> String;
    fn sign(&self, data: &[u8]) -> String;
    fn verify(&self, data: &[u8], signature: &str) -> bool;
}

/// `scrub(action, target, detail) -> (target, detail)`, applied before an entry
/// is hashed. Governance uses it to put personal data in the trail only as
/// per-person tokens, so erasing a person needs no rewrite.
pub type Scrub = Arc<dyn Fn(&str, &str, Map<String, Value>) -> (String, Map<String, Value>) + Send + Sync>;

struct Inner {
    path: Option<PathBuf>,
    redact: bool,
    signer: Option<Arc<dyn Signer>>,
    scrub: Option<Scrub>,
    entries: Vec<AuditEntry>,
}

/// Append-only, hash-chained. In memory by default; give it a path to persist.
/// Cheap to clone; clones are the same trail.
#[derive(Clone)]
pub struct AuditTrail {
    inner: Arc<Mutex<Inner>>,
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for AuditTrail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<AuditTrail {} entries>", self.len())
    }
}

impl AuditTrail {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(Inner { path: None, redact: true, signer: None, scrub: None, entries: vec![] })) }
    }

    /// A trail kept in a JSONL file — read back if it is already there.
    pub fn at(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let entries = std::fs::read_to_string(&path)
            .map(|t| t.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str(l).ok()).collect())
            .unwrap_or_default();
        let t = Self::new();
        {
            let mut i = t.lock();
            i.path = Some(path);
            i.entries = entries;
        }
        t
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn with_signer(self, signer: Arc<dyn Signer>) -> Self {
        self.lock().signer = Some(signer);
        self
    }

    pub fn set_signer(&self, signer: Option<Arc<dyn Signer>>) {
        self.lock().signer = signer;
    }

    /// The key entries are signed with, if any.
    pub fn signer(&self) -> Option<Arc<dyn Signer>> {
        self.lock().signer.clone()
    }

    /// The file the trail is kept in, if it is kept in one.
    pub fn path(&self) -> Option<PathBuf> {
        self.lock().path.clone()
    }

    pub fn set_scrub(&self, scrub: Option<Scrub>) {
        self.lock().scrub = scrub;
    }

    /// Keep details as given (by default keys that look secret are redacted
    /// and long strings trimmed).
    pub fn set_redact(&self, on: bool) {
        self.lock().redact = on;
    }

    /// Append one record. Synchronous on purpose — an audit write never waits.
    pub fn record(&self, actor: &str, action: &str, target: &str, decision: &str, run_id: &str, detail: Map<String, Value>) -> AuditEntry {
        let mut i = self.lock();
        // Scrubbed before it is trimmed: trimming first could cut an address in
        // half, and half an address is still personal data a scrubber misses.
        let (target, detail) = match &i.scrub {
            Some(s) => s(action, target, detail),
            None => (target.to_string(), detail),
        };
        let detail = if i.redact { clean(detail) } else { detail };
        let mut entry = AuditEntry {
            seq: i.entries.len() as u64 + 1,
            ts: now(),
            actor: actor.into(),
            action: action.into(),
            target,
            decision: if decision.is_empty() { "ok".into() } else { decision.into() },
            run_id: run_id.into(),
            detail,
            prev_hash: i.entries.last().map_or_else(genesis, |e| e.hash.clone()),
            hash: String::new(),
            signature: String::new(),
            key_id: String::new(),
        };
        entry.hash = entry.compute_hash();
        if let Some(s) = &i.signer {
            entry.signature = s.sign(entry.hash.as_bytes());
            entry.key_id = s.key_id();
        }
        if let Some(path) = &i.path
            && let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path)
        {
            let _ = writeln!(f, "{}", serde_json::to_string(&entry).unwrap_or_default());
        }
        i.entries.push(entry.clone());
        entry
    }

    /// `record` with no target, run or detail: `audit.note("human", "stop", "ok")`.
    pub fn note(&self, actor: &str, action: &str, decision: &str) -> AuditEntry {
        self.record(actor, action, "", decision, "", Map::new())
    }

    /// Walk the chain. Returns `(ok, reason)` and names the first bad entry.
    pub fn verify(&self) -> (bool, String) {
        let i = self.lock();
        let mut previous = GENESIS.to_string();
        for (index, e) in i.entries.iter().enumerate() {
            if e.seq != index as u64 + 1 {
                return (false, format!("entry {}: sequence is {}", index + 1, e.seq));
            }
            if e.prev_hash != previous {
                return (false, format!("entry {}: does not follow the previous entry", e.seq));
            }
            if e.hash != e.compute_hash() {
                return (false, format!("entry {}: contents do not match its hash", e.seq));
            }
            if let Some(s) = &i.signer {
                if e.signature.is_empty() {
                    return (false, format!("entry {}: is not signed", e.seq));
                }
                if !s.verify(e.hash.as_bytes(), &e.signature) {
                    return (false, format!("entry {}: signature does not verify", e.seq));
                }
            }
            previous = e.hash.clone();
        }
        (true, "chain intact".into())
    }

    pub fn entries(&self) -> Vec<AuditEntry> {
        self.lock().entries.clone()
    }

    pub fn for_run(&self, run_id: &str) -> Vec<AuditEntry> {
        self.lock().entries.iter().filter(|e| e.run_id == run_id).cloned().collect()
    }

    pub fn by_actor(&self, actor: &str) -> Vec<AuditEntry> {
        self.lock().entries.iter().filter(|e| e.actor == actor).cloned().collect()
    }

    pub fn denials(&self) -> Vec<AuditEntry> {
        self.lock().entries.iter().filter(|e| e.decision == "deny").cloned().collect()
    }

    pub fn render(&self, limit: Option<usize>) -> String {
        let i = self.lock();
        let skip = limit.map_or(0, |n| i.entries.len().saturating_sub(n));
        i.entries.iter().skip(skip).map(AuditEntry::line).collect::<Vec<_>>().join("\n")
    }

    pub fn export(&self) -> Vec<Value> {
        self.lock().entries.iter().map(|e| serde_json::to_value(e).unwrap_or_default()).collect()
    }

    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Change the entries directly — for tests that prove tampering is caught.
    #[doc(hidden)]
    pub fn tamper(&self, f: impl FnOnce(&mut Vec<AuditEntry>)) {
        f(&mut self.lock().entries);
    }
}

/// Keep the record small and free of anything that should not be kept.
fn clean(detail: Map<String, Value>) -> Map<String, Value> {
    detail
        .into_iter()
        .map(|(k, v)| {
            let lower = k.to_lowercase();
            let v = if ["key", "token", "secret", "password", "authorization"].iter().any(|w| lower.contains(w)) {
                json!("[redacted]")
            } else {
                match v {
                    Value::String(s) if s.chars().count() > 500 => {
                        json!(format!("{}... [{} chars]", s.chars().take(500).collect::<String>(), s.chars().count()))
                    }
                    other => other,
                }
            };
            (k, v)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chain_catches_tampering() {
        let t = AuditTrail::new();
        t.record("bot", "tool_call", "search", "allow", "run_1", json!({"api_key": "sk-1", "q": "x".repeat(600)}).as_object().cloned().unwrap());
        t.note("human", "stop", "ok");
        assert_eq!(t.verify(), (true, "chain intact".into()));
        let first = &t.entries()[0];
        assert_eq!(first.detail["api_key"], "[redacted]");
        assert!(first.detail["q"].as_str().unwrap().ends_with("... [600 chars]"));
        assert_eq!(first.prev_hash, GENESIS);
        t.tamper(|e| e[0].decision = "deny".into());
        assert_eq!(t.verify(), (false, "entry 1: contents do not match its hash".into()));
    }

    #[test]
    fn hashes_match_python() {
        // AuditEntry(seq=1, ts=1700000000.5, actor="bot", action="tool_call", target="search",
        //            decision="allow", run_id="r", detail={"n": 1, "é": 2.5}).compute_hash()
        let e = AuditEntry {
            seq: 1,
            ts: 1_700_000_000.5,
            actor: "bot".into(),
            action: "tool_call".into(),
            target: "search".into(),
            decision: "allow".into(),
            run_id: "r".into(),
            detail: json!({"n": 1, "é": 2.5}).as_object().cloned().unwrap(),
            prev_hash: GENESIS.into(),
            hash: String::new(),
            signature: String::new(),
            key_id: String::new(),
        };
        assert_eq!(
            e.payload(),
            r#"{"action":"tool_call","actor":"bot","decision":"allow","detail":{"n":1,"\u00e9":2.5},"prev_hash":"0000000000000000000000000000000000000000000000000000000000000000","run_id":"r","seq":1,"target":"search","ts":1700000000.5}"#
        );
        assert_eq!(e.compute_hash(), PYTHON_HASH);
        assert_eq!(stamp(1_700_000_000.5), "2023-11-14 22:13:20");
    }

    const PYTHON_HASH: &str = "dd82e47d07faf2d5dd1a4628bb7399db2e9a91d96d320aad34763bc04f6768be";

    #[test]
    fn persists_and_reloads() {
        let path = std::env::temp_dir().join(format!("ah_audit_{}.jsonl", uuid::Uuid::new_v4().simple()));
        let t = AuditTrail::at(&path);
        t.note("a", "run_start", "ok");
        t.note("a", "run_end", "ok");
        let again = AuditTrail::at(&path);
        assert_eq!(again.len(), 2);
        assert!(again.verify().0);
        let _ = std::fs::remove_file(path);
    }
}
