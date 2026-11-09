//! Records that hold up: signatures, per-subject keys, retention.
//!
//! **Signing.** [`AuditTrail`] already hash-chains its entries; a signer makes
//! the chain unforgeable by anyone without the key.
//!
//! ```
//! use agent_harness::governance::{Ed25519Signer, HMACSigner};
//!
//! let shared = HMACSigner::new(b"a-key-of-at-least-16-bytes", None).unwrap();   // a shared secret
//! let keypair = Ed25519Signer::generate();       // public-key: auditors verify
//! let auditor = Ed25519Signer::verifier(&keypair.public_key_pem()).unwrap();   // with the public key alone
//! ```
//!
//! **Crypto-shredding.** The audit trail must be immutable; people have a right
//! to erasure. Both hold if personal data never enters the trail in the clear:
//! values are replaced by tokens keyed per data subject, and erasing someone
//! destroys their key. The chain still verifies — every byte it covers is
//! unchanged — but nothing in it can be linked back to that person any more.
//!
//! **Retention.** Decision records are kept at least as long as the strictest
//! pack asks (EU AI Act deployers: six months). Conversations are kept no longer
//! than the shortest data-retention limit. [`RetentionSweeper`] enforces the
//! second and reports on the first.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use base64::Engine;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use super::{hex, hmac_hex, random_bytes, sha256_hex};
use crate::errors::{Error, Result};
use crate::runtime::audit::{AuditTrail, Signer};
use crate::runtime::session::{ListFilter, SessionStore};
use crate::types::now;

/// HMAC-SHA256. Anyone who can verify can also sign — keep the key close.
#[derive(Clone)]
pub struct HMACSigner {
    key: Vec<u8>,
    key_id: String,
}

impl std::fmt::Debug for HMACSigner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<HMACSigner {}>", self.key_id)
    }
}

impl HMACSigner {
    pub const ALGORITHM: &'static str = "hmac-sha256";

    /// At least 16 bytes of key. The key id defaults to `hmac:` and the first
    /// twelve hex digits of the key's sha256.
    pub fn new(key: &[u8], key_id: Option<&str>) -> Result<Self> {
        if key.len() < 16 {
            return Err(Error::config("an audit signing key needs at least 16 bytes"));
        }
        let key_id = key_id.map_or_else(|| format!("hmac:{}", &sha256_hex(key)[..12]), str::to_string);
        Ok(Self { key: key.to_vec(), key_id })
    }
}

impl Signer for HMACSigner {
    fn key_id(&self) -> String {
        self.key_id.clone()
    }

    fn sign(&self, data: &[u8]) -> String {
        hmac_hex(&self.key, data)
    }

    fn verify(&self, data: &[u8], signature: &str) -> bool {
        use subtle_compare::eq;
        eq(self.sign(data).as_bytes(), signature.as_bytes())
    }
}

mod subtle_compare {
    /// Compare without stopping at the first difference (`hmac.compare_digest`).
    pub fn eq(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
    }
}

// ---- Ed25519 -------------------------------------------------------------------

/// The DER around a raw Ed25519 key: PKCS#8 v1 for the private key (what
/// Python's `cryptography` writes) and SubjectPublicKeyInfo for the public one.
const PKCS8_PREFIX: [u8; 16] = [0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20];
const SPKI_PREFIX: [u8; 12] = [0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00];
const ED25519_OID: [u8; 5] = [0x06, 0x03, 0x2b, 0x65, 0x70];

fn pem(label: &str, der: &[u8]) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(der);
    let body: Vec<String> = b64.as_bytes().chunks(64).map(|c| String::from_utf8_lossy(c).into_owned()).collect();
    format!("-----BEGIN {label}-----\n{}\n-----END {label}-----\n", body.join("\n"))
}

fn unpem(text: &str) -> Result<Vec<u8>> {
    let body: String = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with("-----")).collect();
    base64::engine::general_purpose::STANDARD.decode(body).map_err(|e| Error::config(format!("not a PEM key: {e}")))
}

/// The 32 bytes after `marker`, somewhere after the Ed25519 algorithm id.
fn key_after(der: &[u8], marker: &[u8]) -> Option<[u8; 32]> {
    let oid = der.windows(ED25519_OID.len()).position(|w| w == ED25519_OID)?;
    let rest = &der[oid + ED25519_OID.len()..];
    let at = rest.windows(marker.len()).position(|w| w == marker)? + marker.len();
    rest.get(at..at + 32)?.try_into().ok()
}

/// Ed25519 (RustCrypto's `ed25519-dalek`).
///
/// Hand auditors `public_key_pem()`; `Ed25519Signer::verifier(pem)` checks a
/// trail with it and cannot sign anything. Keys are read and written as the PEM
/// Python's `cryptography` uses (PKCS#8 and SubjectPublicKeyInfo), so a key
/// made by one works in the other.
#[derive(Clone)]
pub struct Ed25519Signer {
    private: Option<ed25519_dalek::SigningKey>,
    public: ed25519_dalek::VerifyingKey,
    key_id: String,
}

impl std::fmt::Debug for Ed25519Signer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<Ed25519Signer {}{}>", self.key_id, if self.private.is_some() { "" } else { " (verify only)" })
    }
}

impl Ed25519Signer {
    pub const ALGORITHM: &'static str = "ed25519";

    fn with(private: Option<ed25519_dalek::SigningKey>, public: ed25519_dalek::VerifyingKey) -> Self {
        let key_id = format!("ed25519:{}", &sha256_hex(public.as_bytes())[..12]);
        Self { private, public, key_id }
    }

    /// A fresh key pair.
    pub fn generate() -> Self {
        let seed: [u8; 32] = random_bytes(32).try_into().expect("32 bytes");
        Self::from_seed(&seed)
    }

    /// The key pair for this 32-byte private key.
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let private = ed25519_dalek::SigningKey::from_bytes(seed);
        let public = private.verifying_key();
        Self::with(Some(private), public)
    }

    /// From a PKCS#8 PEM private key.
    pub fn from_private_pem(pem: &str) -> Result<Self> {
        let der = unpem(pem)?;
        let seed = key_after(&der, &[0x04, 0x20]).ok_or_else(|| Error::config("not an Ed25519 private key"))?;
        Ok(Self::from_seed(&seed))
    }

    /// Verify only, from a SubjectPublicKeyInfo PEM public key.
    pub fn verifier(public_key_pem: &str) -> Result<Self> {
        let der = unpem(public_key_pem)?;
        let raw = key_after(&der, &[0x03, 0x21, 0x00]).ok_or_else(|| Error::config("not an Ed25519 public key"))?;
        let public = ed25519_dalek::VerifyingKey::from_bytes(&raw).map_err(|e| Error::config(format!("not an Ed25519 public key: {e}")))?;
        Ok(Self::with(None, public))
    }

    /// Sign, or say why not: a verifier holds no private key.
    pub fn try_sign(&self, data: &[u8]) -> Result<String> {
        use ed25519_dalek::Signer as _;
        let private = self.private.as_ref().ok_or_else(|| Error::config("this Ed25519Signer holds only a public key"))?;
        Ok(base64::engine::general_purpose::STANDARD.encode(private.sign(data).to_bytes()))
    }

    pub fn public_key_pem(&self) -> String {
        let mut der = SPKI_PREFIX.to_vec();
        der.extend_from_slice(self.public.as_bytes());
        pem("PUBLIC KEY", &der)
    }

    pub fn private_key_pem(&self) -> Result<String> {
        let private = self.private.as_ref().ok_or_else(|| Error::config("this Ed25519Signer holds only a public key"))?;
        let mut der = PKCS8_PREFIX.to_vec();
        der.extend_from_slice(&private.to_bytes());
        Ok(pem("PRIVATE KEY", &der))
    }

    pub fn can_sign(&self) -> bool {
        self.private.is_some()
    }
}

impl Signer for Ed25519Signer {
    fn key_id(&self) -> String {
        self.key_id.clone()
    }

    /// The base64 signature. A verifier has no private key and returns an empty
    /// signature, which no verification accepts; use `try_sign` to get the error.
    fn sign(&self, data: &[u8]) -> String {
        self.try_sign(data).unwrap_or_default()
    }

    fn verify(&self, data: &[u8], signature: &str) -> bool {
        use ed25519_dalek::Verifier as _;
        let Ok(raw) = base64::engine::general_purpose::STANDARD.decode(signature) else { return false };
        let Ok(sig) = ed25519_dalek::Signature::from_slice(&raw) else { return false };
        self.public.verify(data, &sig).is_ok()
    }
}

// ---- the subject vault ------------------------------------------------------------

#[derive(Default)]
struct VaultData {
    keys: IndexMap<String, String>,
    index: IndexMap<String, IndexMap<String, Vec<String>>>,
    erased: IndexMap<String, f64>,
}

/// Per-subject keys and the index of where each subject's data lives.
///
/// Kept apart from the audit trail on purpose: the trail is immutable, this is
/// not. A file path makes it durable (the same JSON file Python keeps); without
/// one it lives in memory.
pub struct SubjectVault {
    pub path: Option<PathBuf>,
    data: Mutex<VaultData>,
}

impl std::fmt::Debug for SubjectVault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubjectVault").field("path", &self.path).finish_non_exhaustive()
    }
}

impl Default for SubjectVault {
    fn default() -> Self {
        Self { path: None, data: Mutex::new(VaultData::default()) }
    }
}

const OPEN: char = '\u{27e8}';
const CLOSE: char = '\u{27e9}';

impl SubjectVault {
    /// In memory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Kept in this file, read back if it is there.
    pub fn at(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut data = VaultData::default();
        if path.exists() {
            let text = std::fs::read_to_string(&path).map_err(|e| Error::config(format!("vault {}: {e}", path.display())))?;
            let v: Value =
                serde_json::from_str(if text.trim().is_empty() { "{}" } else { &text }).map_err(|e| Error::config(format!("vault {}: {e}", path.display())))?;
            data.keys = serde_json::from_value(v.get("keys").cloned().unwrap_or(json!({}))).unwrap_or_default();
            data.index = serde_json::from_value(v.get("index").cloned().unwrap_or(json!({}))).unwrap_or_default();
            data.erased = serde_json::from_value(v.get("erased").cloned().unwrap_or(json!({}))).unwrap_or_default();
        }
        Ok(Self { path: Some(path), data: Mutex::new(data) })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VaultData> {
        self.data.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn save(&self, data: &VaultData) {
        let Some(path) = &self.path else { return };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let body = crate::pyjson::dumps(&json!({"keys": data.keys, "index": data.index, "erased": data.erased}));
        let tmp = path.with_extension("tmp");
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        // The keys in here are what makes erasure work: owner-only.
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        if let Ok(mut f) = options.open(&tmp) {
            use std::io::Write;
            if f.write_all(body.as_bytes()).is_ok() {
                drop(f);
                let _ = std::fs::rename(&tmp, path);
            }
        }
    }

    /// The subject's key, made (and saved) the first time it is needed.
    fn key(&self, data: &mut VaultData, subject: &str) -> Vec<u8> {
        if !data.keys.contains_key(subject) {
            data.keys.insert(subject.to_string(), hex(&random_bytes(32)));
            self.save(data);
        }
        decode_hex(&data.keys[subject])
    }

    /// A stable, unlinkable-after-erasure stand-in for one personal value.
    pub fn token(&self, subject: &str, value: &str, kind: &str) -> String {
        let mut data = self.lock();
        if data.erased.contains_key(subject) {
            return format!("{OPEN}{kind}:erased{CLOSE}");
        }
        let key = self.key(&mut data, subject);
        format!("{OPEN}{kind}:{}{CLOSE}", &hmac_hex(&key, value.as_bytes())[..12])
    }

    pub fn has(&self, subject: &str) -> bool {
        self.lock().keys.contains_key(subject)
    }

    /// How a subject is named in records: a keyed hash, never the id itself.
    ///
    /// `create = false` answers "" for someone with no key yet, instead of
    /// making one — looking a person up must not register them.
    pub fn subject_ref(&self, subject: &str, create: bool) -> String {
        let mut data = self.lock();
        if subject.is_empty() || (!create && !data.keys.contains_key(subject)) {
            return String::new();
        }
        let key = self.key(&mut data, subject);
        format!("subj:{}", &hmac_hex(&key, b"subject-ref")[..16])
    }

    /// Remember that `subject` has data at `reference` (a session id, a run).
    pub fn note(&self, subject: &str, kind: &str, reference: &str) {
        let mut data = self.lock();
        if subject.is_empty() || data.erased.contains_key(subject) {
            return;
        }
        let refs = data.index.entry(subject.to_string()).or_default().entry(kind.to_string()).or_default();
        if !refs.iter().any(|r| r == reference) {
            refs.push(reference.to_string());
            self.save(&data);
        }
    }

    pub fn locations(&self, subject: &str) -> IndexMap<String, Vec<String>> {
        self.lock().index.get(subject).cloned().unwrap_or_default()
    }

    /// Destroy the subject's key and index. Returns what was there.
    pub fn erase(&self, subject: &str) -> Value {
        let mut data = self.lock();
        let had_key = data.keys.shift_remove(subject).is_some();
        let locations = data.index.shift_remove(subject).unwrap_or_default();
        data.erased.insert(subject.to_string(), now());
        self.save(&data);
        json!({"key_destroyed": had_key, "locations": locations})
    }

    /// Forget index entries that no longer exist (swept by retention).
    pub fn drop_refs(&self, kind: &str, refs: &HashSet<String>) {
        let mut data = self.lock();
        let mut changed = false;
        for locations in data.index.values_mut() {
            if let Some(list) = locations.get_mut(kind) {
                let before = list.len();
                list.retain(|r| !refs.contains(r));
                changed |= list.len() != before;
            }
        }
        if changed {
            self.save(&data);
        }
    }

    pub fn is_erased(&self, subject: &str) -> bool {
        self.lock().erased.contains_key(subject)
    }

    pub fn subjects(&self) -> Vec<String> {
        let data = self.lock();
        let mut all: Vec<String> = data.keys.keys().chain(data.index.keys()).cloned().collect();
        all.sort();
        all.dedup();
        all
    }
}

fn decode_hex(text: &str) -> Vec<u8> {
    (0..text.len() / 2).filter_map(|i| u8::from_str_radix(text.get(2 * i..2 * i + 2)?, 16).ok()).collect()
}

/// Deletes conversations older than the data-retention limit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RetentionSweeper {
    pub days: Option<i64>,
}

impl RetentionSweeper {
    pub fn new(days: Option<i64>) -> Self {
        Self { days }
    }

    pub fn cutoff(&self, now_ts: Option<f64>) -> Option<f64> {
        Some(now_ts.unwrap_or_else(now) - self.days? as f64 * 86_400.0)
    }

    /// Delete every session last updated before the cutoff, drop them from the
    /// vault's index, and record the sweep.
    pub async fn sweep(
        &self,
        sessions: &Arc<dyn SessionStore>,
        audit: Option<&AuditTrail>,
        now_ts: Option<f64>,
        vault: Option<&SubjectVault>,
    ) -> Result<Map<String, Value>> {
        let mut report = Map::new();
        report.insert("days".into(), json!(self.days));
        let Some(cutoff) = self.cutoff(now_ts) else {
            report.insert("sessions_deleted".into(), json!([]));
            return Ok(report);
        };
        let mut deleted: Vec<String> = vec![];
        for session in sessions.list(&ListFilter::limit(100_000)).await? {
            if session.updated < cutoff {
                sessions.delete(&session.id).await?;
                deleted.push(session.id);
            }
        }
        if let Some(v) = vault
            && !deleted.is_empty()
        {
            v.drop_refs("sessions", &deleted.iter().cloned().collect());
        }
        if let Some(a) = audit {
            a.record("governance", "retention_sweep", "", "ok", "", super::obj(vec![("deleted", json!(deleted.len())), ("days", json!(self.days))]));
        }
        report.insert("sessions_deleted".into(), json!(deleted));
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &[u8] = &[b'k'; 32];

    #[test]
    fn hmac_signer_matches_python() {
        let signer = HMACSigner::new(KEY, None).unwrap();
        assert_eq!(signer.key_id(), "hmac:5e318f8cf9cb");
        assert_eq!(signer.sign(b"x"), "5011cc61ff53aa5af4fdf0d44b117900222c1a355f08463571aa947afcb9ac17");
        assert!(signer.verify(b"x", &signer.sign(b"x")) && !signer.verify(b"y", &signer.sign(b"x")));
        assert!(HMACSigner::new(b"short", None).is_err());
    }

    #[test]
    fn ed25519_signs_verifies_and_round_trips_pem() {
        // RFC 8032, test 1.
        let seed: [u8; 32] = decode_hex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60").try_into().unwrap();
        let signer = Ed25519Signer::from_seed(&seed);
        assert_eq!(signer.key_id(), "ed25519:21fe31dfa154");
        let sig = base64::engine::general_purpose::STANDARD.decode(signer.sign(b"")).unwrap();
        assert_eq!(
            hex(&sig),
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        );
        assert_eq!(
            signer.public_key_pem(),
            "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEA11qYAYKxCrfVS/7TyWQHOg7hcvPapiMlrwIaaPcHURo=\n-----END PUBLIC KEY-----\n"
        );

        let private = Ed25519Signer::generate();
        let public = Ed25519Signer::verifier(&private.public_key_pem()).unwrap();
        assert!(public.verify(b"x", &private.sign(b"x")) && !public.verify(b"y", &private.sign(b"x")));
        assert!(public.try_sign(b"x").is_err() && public.private_key_pem().is_err());
        let again = Ed25519Signer::from_private_pem(&private.private_key_pem().unwrap()).unwrap();
        assert_eq!(again.key_id(), private.key_id());
    }

    #[test]
    fn the_vault_tokenises_and_shreds() {
        let path = std::env::temp_dir().join(format!("ah_vault_{}.json", uuid::Uuid::new_v4().simple()));
        std::fs::write(&path, r#"{"keys": {"acme/alice": "0000000000000000000000000000000000000000000000000000000000000000"}, "index": {}, "erased": {}}"#)
            .unwrap();
        let vault = SubjectVault::at(&path).unwrap();
        assert_eq!(vault.token("acme/alice", "alice@example.com", "email"), "\u{27e8}email:0cf19c8f102e\u{27e9}");
        assert_eq!(vault.subject_ref("acme/alice", true), "subj:33b34c58f8784a55");
        assert_eq!(vault.subject_ref("nobody", false), "");
        assert!(!vault.has("nobody"));
        vault.note("acme/alice", "sessions", "ses_1");
        vault.note("acme/alice", "sessions", "ses_1");
        assert_eq!(vault.locations("acme/alice")["sessions"], vec!["ses_1"]);
        let again = SubjectVault::at(&path).unwrap();
        assert_eq!(again.locations("acme/alice")["sessions"], vec!["ses_1"]);
        let gone = again.erase("acme/alice");
        assert_eq!(gone["key_destroyed"], true);
        assert_eq!(again.token("acme/alice", "x", "email"), "\u{27e8}email:erased\u{27e9}");
        assert!(again.is_erased("acme/alice"));
        let _ = std::fs::remove_file(path);
    }
}
