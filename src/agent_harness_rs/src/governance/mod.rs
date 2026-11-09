//! Governance: policy, identity, data, residency, oversight and evidence for agents.
//!
//! ```no_run
//! use agent_harness::Harness;
//! use agent_harness::governance::{Governance, GovernanceOptions};
//!
//! # fn main() -> agent_harness::Result<()> {
//! let gov = Governance::from_packs(
//!     &["eu-ai-act", "gdpr", "owasp-agentic"],
//!     GovernanceOptions { home: Some("eu".into()), signing_key: Some(std::env::var("AUDIT_KEY").unwrap_or_default().into_bytes()), ..Default::default() },
//! )?;
//! let mut harness = Harness::local(".harness")?;
//! gov.attach(&mut harness)?;
//! println!("{}", gov.report("eu-ai-act").markdown());
//! # Ok(()) }
//! ```
//!
//! Packs turn on what a law or framework asks for; your own policy adds rules on
//! top; every combination can only tighten. See [`list_packs`] for what ships —
//! EU (AI Act, GDPR, DORA, NIS2), UK, the Gulf (UAE, DIFC, ADGM, Saudi Arabia,
//! Qatar, Bahrain, Oman), Asia-Pacific (Singapore, India, China, Korea, Japan,
//! Vietnam), the US (NIST AI RMF, Colorado, Texas, California) and the standards
//! (ISO/IEC 42001, OWASP Top 10 for Agentic Applications).
//!
//! This supplies technical controls and the evidence for them. It is not legal
//! advice, and it does not by itself make anyone compliant with anything.

pub mod core;
pub mod data;
pub mod evidence;
pub mod expr;
mod frameworks;
pub mod identity;
pub mod incidents;
pub mod inventory;
pub mod monitor;
pub mod oversight;
pub mod packs;
pub mod policy;
pub mod records;
pub mod residency;
pub mod rights;
pub mod risk;
pub mod transparency;

pub use self::core::{AgentInfo, AuditArgs, Governance, GovernanceOptions, Mode};
pub use data::{Classification, DATA_CLASSES, DataClassifier, DataFinding, PERSONAL, Pseudonymizer, SPECIAL_CATEGORIES, known_classes};
pub use evidence::{CONTROLS, ComplianceReport, ControlStatus, RequirementStatus};
pub use expr::{ExpressionError, compile_expr};
pub use identity::{AgentIdentity, DelegationChain, Principal, RiskTier};
pub use incidents::{Deadline, Incident, IncidentDesk};
pub use inventory::AIInventory;
pub use monitor::{RunState, RuntimeMonitor};
pub use oversight::{ApprovalRequest, ApprovalVote, Approver, OversightDesk};
pub use packs::{Pack, Requirement, list_packs, load_pack};
pub use policy::{Decision, Match, Policy, PolicyEngine, PolicyRule, ResidencyConfig, TransparencyConfig};
pub use records::{Ed25519Signer, HMACSigner, RetentionSweeper, SubjectVault};
pub use residency::{ProviderInfo, Region, RegionResolver, resolve_region};
pub use rights::SubjectRights;
pub use risk::{RiskAssessment, assess, impact_assessment};
pub use transparency::ProvenanceManifest;

// ---------------------------------------------------------------------------
// small helpers shared by the modules: Python's spellings of things
// ---------------------------------------------------------------------------

use serde_json::Value;

/// Python's `fnmatch.fnmatchcase`: `*`, `?`, `[seq]`, `[!seq]`.
pub(crate) fn fnmatch(name: &str, pattern: &str) -> bool {
    match glob::Pattern::new(pattern) {
        Ok(p) => p.matches_with(name, glob::MatchOptions { case_sensitive: true, require_literal_separator: false, require_literal_leading_dot: false }),
        Err(_) => name == pattern,
    }
}

/// Lowercase hex of some bytes.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `hashlib.sha256(data).hexdigest()`.
pub(crate) fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    hex(&sha2::Sha256::digest(data))
}

/// `hmac.new(key, data, hashlib.sha256).hexdigest()`.
pub(crate) fn hmac_hex(key: &[u8], data: &[u8]) -> String {
    use hmac::{KeyInit, Mac};
    let mut mac = <hmac::Hmac<sha2::Sha256> as KeyInit>::new_from_slice(key).expect("HMAC takes a key of any length");
    mac.update(data);
    hex(&mac.finalize().into_bytes())
}

/// `n` bytes from the operating system's secure random source.
pub(crate) fn random_bytes(n: usize) -> Vec<u8> {
    use ring::rand::SecureRandom;
    let mut out = vec![0u8; n];
    ring::rand::SystemRandom::new().fill(&mut out).expect("the system random source works");
    out
}

/// `time.strftime(fmt, time.gmtime(ts))` for the two formats the reports use:
/// `iso` gives `2026-09-27T14:02:11Z`, otherwise `2026-09-27 14:02 UTC`.
pub(crate) fn utc(ts: f64, iso: bool) -> String {
    let secs = ts.floor() as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Howard Hinnant's days-to-civil.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let (h, mi, s) = (rem / 3600, rem % 3600 / 60, rem % 60);
    if iso { format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z") } else { format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02} UTC") }
}

/// Python's `str(value)` for a JSON value (`None`, `True`, `1.5`, `['a']`).
pub(crate) fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python's `repr(value)` for a JSON value — what an f-string shows of a dict.
pub(crate) fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) if !n.is_f64() => i.to_string(),
            (_, Some(u)) if !n.is_f64() => u.to_string(),
            _ => crate::pyjson::float_repr(n.as_f64().unwrap_or(0.0)),
        },
        Value::String(s) => py_str_repr(s),
        Value::Array(items) => format!("[{}]", items.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(map) => format!("{{{}}}", map.iter().map(|(k, v)| format!("{}: {}", py_str_repr(k), py_repr(v))).collect::<Vec<_>>().join(", ")),
    }
}

/// `repr(str)`: single quotes unless the text holds one and no double quote.
pub(crate) fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// A JSON object from `(key, value)` pairs, in order.
pub(crate) fn obj(pairs: Vec<(&str, Value)>) -> serde_json::Map<String, Value> {
    pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_spellings() {
        assert_eq!(utc(1_700_000_000.5, true), "2023-11-14T22:13:20Z");
        assert_eq!(utc(1_700_000_000.5, false), "2023-11-14 22:13 UTC");
        assert_eq!(utc(0.0, true), "1970-01-01T00:00:00Z");
        assert_eq!(py_repr(&json!({"deny": 1, "a": [true, null, "it's"]})), "{'deny': 1, 'a': [True, None, \"it's\"]}");
        assert!(fnmatch("pay_out", "pay*") && fnmatch("a/b", "a*") && !fnmatch("Pay", "pay*") && fnmatch("x1", "x?"));
        assert_eq!(hmac_hex(&[b'k'; 32], b"x"), "5011cc61ff53aa5af4fdf0d44b117900222c1a355f08463571aa947afcb9ac17");
    }
}
