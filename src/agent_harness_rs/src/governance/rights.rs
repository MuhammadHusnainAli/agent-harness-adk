//! Data-subject requests: access and erasure, across everything the harness keeps.
//!
//! ```no_run
//! # async fn demo(gov: agent_harness::governance::Governance) -> agent_harness::Result<()> {
//! use agent_harness::Trace;
//! let alice = Trace::user("alice").tenant("acme");
//! let export = gov.rights().access(&alice).await?;
//! let receipt = gov.rights().erase(&alice, "data subject").await?;
//! # Ok(()) }
//! ```
//!
//! Erasure covers the memory store (records and `user.md`), every session the
//! person's runs were saved in, the deliverables those runs produced, their
//! entries in the run journal and the trace export, pseudonym tokens held in
//! memory, and the audit trail — by crypto-shredding: the subject's key is
//! destroyed, so the tokens standing in for their data in the trail can never be
//! linked back to them, while the chain itself still verifies. The receipt says
//! what was removed and is signed when governance has a signer. The request and
//! its outcome are themselves recorded (without the personal data).

use std::collections::HashSet;

use serde_json::{Map, Value, json};

use super::core::Governance;
use super::{obj, utc};
use crate::errors::{Error, Result};
use crate::memory::Trace;
use crate::memory::base::RecordQuery;
use crate::memory::trace::Scope;
use crate::pyjson::{Dumps, dumps_with};
use crate::types::now;

fn subject(trace: &Trace) -> Result<(Trace, String)> {
    let user = trace.user_id.clone().filter(|u| !u.is_empty()).ok_or_else(|| Error::config("a data-subject request needs a user_id on the trace"))?;
    let subject = format!("{}/{user}", trace.tenant_id.clone().unwrap_or_default());
    Ok((trace.clone().scoped(Scope::User), subject))
}

/// Access and erasure for one governance layer. `gov.rights()`.
#[derive(Clone)]
pub struct SubjectRights {
    gov: Governance,
}

impl SubjectRights {
    pub fn new(gov: Governance) -> Self {
        Self { gov }
    }

    /// Everything held about one person, as JSON (GDPR Art 15).
    pub async fn access(&self, trace: &Trace) -> Result<Value> {
        let (user_trace, subject) = subject(trace)?;
        let h = self.gov.services()?;
        let records = h.memory_store.all(RecordQuery { trace: Some(user_trace.clone()), ..Default::default() }).await?;
        let vault = self.gov.vault();
        let locations = vault.locations(&subject);
        let mut sessions = vec![];
        for id in locations.get("sessions").into_iter().flatten() {
            let Ok(session) = h.sessions.load(id).await else { continue };
            sessions.push(json!({
                "id": session.id, "agent": session.agent, "created": session.created,
                "messages": session.messages.iter().map(|m| json!({"role": m.role, "text": m.text()})).collect::<Vec<_>>(),
            }));
        }
        let reference = vault.subject_ref(&subject, false);
        let runs: HashSet<String> = locations.get("runs").cloned().unwrap_or_default().into_iter().collect();
        let decisions: Vec<Value> = h
            .audit
            .entries()
            .into_iter()
            .filter(|e| (!reference.is_empty() && e.detail.get("subject").and_then(Value::as_str) == Some(reference.as_str())) || runs.contains(&e.run_id))
            .map(|e| serde_json::to_value(e).unwrap_or_default())
            .collect();
        let mut sorted_runs: Vec<&String> = runs.iter().collect();
        sorted_runs.sort();
        let deliverables: Vec<Value> = sorted_runs
            .iter()
            .flat_map(|r| h.deliverables.for_run(r))
            .map(|a| json!({"name": a.name, "version": a.extra.get("version").cloned().unwrap_or(json!(1)), "content": a.content}))
            .collect();
        h.audit.record(
            "governance",
            "governance.subject_access",
            "",
            "ok",
            "",
            obj(vec![("subject", json!(reference)), ("records", json!(records.len())), ("sessions", json!(sessions.len()))]),
        );
        Ok(json!({
            "subject": {"tenant": trace.tenant_id, "user": trace.user_id},
            "generated": utc(now(), true),
            "profile": h.memory_store.read_doc("user.md", Some(&user_trace)).await?,
            "memory": records.iter().map(|r| serde_json::to_value(r).unwrap_or_default()).collect::<Vec<_>>(),
            "sessions": sessions,
            "deliverables": deliverables,
            "decisions": decisions,
        }))
    }

    /// Erase one person everywhere (GDPR Art 17, DPDP s12, PDPL equivalents).
    pub async fn erase(&self, trace: &Trace, requested_by: &str) -> Result<Map<String, Value>> {
        let (user_trace, subject) = subject(trace)?;
        let h = self.gov.services()?;
        let vault = self.gov.vault();
        let reference = vault.subject_ref(&subject, false);

        let records = h.memory_store.all(RecordQuery { trace: Some(user_trace.clone()), ..Default::default() }).await?.len();
        h.memory_store.clear(None, Some(&user_trace)).await?;
        h.memory_store.write_doc("user.md", "", Some(&user_trace)).await?;

        let mut deleted_sessions = vec![];
        for id in vault.locations(&subject).get("sessions").cloned().unwrap_or_default() {
            if h.sessions.delete(&id).await.is_ok() {
                deleted_sessions.push(id);
            }
        }
        let locations = vault.locations(&subject);
        let runs: HashSet<String> = locations.get("runs").cloned().unwrap_or_default().into_iter().collect();
        let traces: HashSet<String> = locations.get("traces").cloned().unwrap_or_default().into_iter().collect();
        let deliverables = h.deliverables.forget_runs(&runs);
        let journal = h.journal.forget(&runs);
        let spans = h.tracer.forget(&traces);
        let tokens = self.gov.pseudonymizer().forget(&subject);
        let shredded = vault.erase(&subject);

        let mut receipt = obj(vec![
            ("subject", json!(reference)),
            ("erased_at", json!(utc(now(), true))),
            ("requested_by", json!(requested_by)),
            ("memory_records", json!(records)),
            ("sessions", json!(deleted_sessions)),
            ("deliverables", json!(deliverables)),
            ("journal_entries", json!(journal)),
            ("trace_spans", json!(spans)),
            ("pseudonyms_forgotten", json!(tokens)),
            ("audit_key_destroyed", shredded["key_destroyed"].clone()),
            ("audit_chain", json!(h.audit.verify().1)),
        ]);
        if let Some(signer) = self.gov.signer() {
            let payload = dumps_with(&Value::Object(receipt.clone()), Dumps { sort_keys: true, ..Default::default() });
            receipt.insert("signature".into(), json!(signer.sign(payload.as_bytes())));
            receipt.insert("key_id".into(), json!(signer.key_id()));
        }
        h.audit.record(
            "governance",
            "governance.erasure",
            "",
            "ok",
            "",
            obj(vec![
                ("subject", json!(reference)),
                ("requested_by", json!(requested_by)),
                ("memory_records", json!(records)),
                ("sessions", json!(deleted_sessions.len())),
            ]),
        );
        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use crate::governance::records::HMACSigner;
    use crate::runtime::audit::Signer;
    use serde_json::json;

    #[test]
    fn receipts_are_signed_over_python_json() {
        let receipt = json!({"subject": "subj:abc", "erased_at": "2026-01-01T00:00:00Z", "requested_by": "data subject",
            "memory_records": 1, "sessions": ["ses_1"], "deliverables": [], "journal_entries": 2,
            "trace_spans": 0, "pseudonyms_forgotten": 0, "audit_key_destroyed": true, "audit_chain": "chain intact"});
        let payload = crate::pyjson::dumps_with(&receipt, crate::pyjson::Dumps { sort_keys: true, ..Default::default() });
        let signer = HMACSigner::new(&[b'k'; 32], None).unwrap();
        assert_eq!(signer.sign(payload.as_bytes()), "8acb9237a6f92079b20e538ad5d2a7f623063d0cb1b1dbfe61ff3109e94b067c");
    }
}
