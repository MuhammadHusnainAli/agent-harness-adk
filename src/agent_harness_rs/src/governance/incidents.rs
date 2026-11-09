//! Incidents, and the regulatory clocks that start when one is opened.
//!
//! ```no_run
//! # async fn demo(gov: agent_harness::governance::Governance) {
//! let incident = gov.incidents().open("personal_data_breach", "high", "support agent emailed a card number to the wrong customer", "", "", Default::default()).await;
//! for d in &incident.deadlines {
//!     println!("{} {} {}", d.framework, d.notify, d.due_iso());
//! }
//! // gdpr        supervisory authority   2026-09-27T14:02:11Z   (72 h)
//! // ksa-pdpl    SDAIA                   2026-09-27T14:02:11Z   (72 h)
//! // dora        competent authority     2026-09-24T18:02:11Z   (4 h initial)
//! # }
//! ```
//!
//! Which clocks apply comes from the active packs; each pack lists its incident
//! kinds and deadlines. The desk opens incidents on its own for what the runtime
//! can see (a runaway loop, a tool whose breaker opened) and calls your `notify`
//! hooks — the reporting itself stays a human decision.

use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use super::packs::Clocks;
use crate::types::{new_id, now};

/// What each incident kind means.
pub const INCIDENT_KINDS: [(&str, &str); 6] = [
    ("personal_data_breach", "personal data disclosed, lost or altered without authority"),
    ("serious_incident", "an AI system caused or nearly caused serious harm (AI Act Art 73)"),
    ("ict_incident", "an ICT disruption affecting a financial service (DORA)"),
    ("security_incident", "a cyber-security incident (NIS2, national CERT rules)"),
    ("policy_violation", "the agent attempted something policy forbids"),
    ("runaway", "the agent looped, stormed a tool or ran away with spend"),
];

/// `low`, `medium`, `high`, `critical`.
pub const SEVERITIES: [&str; 4] = ["low", "medium", "high", "critical"];

fn severity_rank(s: &str) -> usize {
    SEVERITIES.iter().position(|x| *x == s).unwrap_or(1)
}

/// Who must be told, by when, and under which rule.
#[derive(Debug, Clone, PartialEq)]
pub struct Deadline {
    pub framework: String,
    pub notify: String,
    pub within_hours: f64,
    pub due: f64,
    pub basis: String,
}

impl Deadline {
    pub fn due_iso(&self) -> String {
        super::utc(self.due, true)
    }

    pub fn overdue(&self) -> bool {
        now() > self.due
    }
}

/// One incident and its clocks.
#[derive(Debug, Clone, PartialEq)]
pub struct Incident {
    pub kind: String,
    pub severity: String,
    pub title: String,
    pub detail: Map<String, Value>,
    pub agent: String,
    pub run_id: String,
    pub id: String,
    pub opened: f64,
    /// `open`, `reported` or `closed`.
    pub status: String,
    pub deadlines: Vec<Deadline>,
    pub notes: Vec<String>,
}

impl Incident {
    pub fn as_dict(&self) -> Value {
        json!({
            "id": self.id, "kind": self.kind, "severity": self.severity, "title": self.title,
            "agent": self.agent, "run_id": self.run_id, "opened": self.opened, "status": self.status,
            "detail": self.detail,
            "deadlines": self.deadlines.iter().map(|d| json!({"framework": d.framework, "notify": d.notify, "within_hours": d.within_hours, "due": d.due_iso(), "basis": d.basis})).collect::<Vec<_>>(),
            "notes": self.notes,
        })
    }
}

/// Told about each incident at or above the desk's minimum severity. An error
/// is noted on the incident (`notify failed: ...`) and never stops anything.
pub type Notifier = Arc<dyn Fn(Incident) -> BoxFuture<'static, std::result::Result<(), String>> + Send + Sync>;

/// A notifier from a plain function.
pub fn notifier(f: impl Fn(&Incident) -> std::result::Result<(), String> + Send + Sync + 'static) -> Notifier {
    let f = Arc::new(f);
    Arc::new(move |incident: Incident| {
        let f = f.clone();
        Box::pin(async move { f(&incident) })
    })
}

/// Called with every incident as it is opened (governance records it in the audit trail).
pub type IncidentRecord = Arc<dyn Fn(&Incident) + Send + Sync>;

/// Holds incidents and works out their deadlines from the active packs.
pub struct IncidentDesk {
    /// pack id → incident kind → clocks.
    pub clocks: IndexMap<String, Clocks>,
    pub notify: Vec<Notifier>,
    pub record: Option<IncidentRecord>,
    pub min_severity: String,
    incidents: Mutex<Vec<Incident>>,
}

impl std::fmt::Debug for IncidentDesk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<IncidentDesk {} incidents>", self.incidents().len())
    }
}

impl IncidentDesk {
    pub fn new(clocks: IndexMap<String, Clocks>, notify: Vec<Notifier>, record: Option<IncidentRecord>) -> Self {
        Self { clocks, notify, record, min_severity: "medium".into(), incidents: Mutex::new(vec![]) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Incident>> {
        self.incidents.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn deadlines_for(&self, kind: &str, severity: &str, opened: f64) -> Vec<Deadline> {
        let mut out = vec![];
        for (framework, kinds) in &self.clocks {
            for clock in kinds.get(kind).into_iter().flatten() {
                let floor = clock.get("min_severity").and_then(Value::as_str).unwrap_or("medium");
                if severity_rank(severity) < severity_rank(floor) {
                    continue;
                }
                let hours = clock.get("within").and_then(Value::as_f64).unwrap_or(0.0);
                out.push(Deadline {
                    framework: framework.clone(),
                    notify: clock.get("notify").map(super::py_str).unwrap_or_else(|| "authority".into()),
                    within_hours: hours,
                    due: opened + hours * 3600.0,
                    basis: clock.get("basis").map(super::py_str).unwrap_or_default(),
                });
            }
        }
        out.sort_by(|a, b| a.due.partial_cmp(&b.due).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Open one, start its clocks, record it, and tell the people who should know.
    pub async fn open(&self, kind: &str, severity: &str, title: &str, agent: &str, run_id: &str, detail: Map<String, Value>) -> Incident {
        let opened = now();
        let mut incident = Incident {
            kind: kind.into(),
            severity: severity.into(),
            title: title.into(),
            detail,
            agent: agent.into(),
            run_id: run_id.into(),
            id: new_id("inc"),
            opened,
            status: "open".into(),
            deadlines: self.deadlines_for(kind, severity, opened),
            notes: vec![],
        };
        self.lock().push(incident.clone());
        if let Some(r) = &self.record {
            r(&incident);
        }
        if severity_rank(severity) >= severity_rank(&self.min_severity) {
            for hook in &self.notify {
                // A pager that is down must not take the agent down with it.
                if let Err(e) = hook(incident.clone()).await {
                    incident.notes.push(format!("notify failed: {e}"));
                }
            }
            let mut all = self.lock();
            if let Some(stored) = all.iter_mut().find(|i| i.id == incident.id) {
                stored.notes = incident.notes.clone();
            }
        }
        incident
    }

    pub fn incidents(&self) -> Vec<Incident> {
        self.lock().clone()
    }

    pub fn open_incidents(&self) -> Vec<Incident> {
        self.lock().iter().filter(|i| i.status == "open").cloned().collect()
    }

    pub fn overdue(&self) -> Vec<(Incident, Deadline)> {
        self.open_incidents()
            .into_iter()
            .flat_map(|i| i.deadlines.iter().filter(|d| d.overdue()).cloned().map(|d| (i.clone(), d)).collect::<Vec<_>>())
            .collect()
    }

    pub fn close(&self, incident_id: &str, note: &str) -> Option<Incident> {
        let mut all = self.lock();
        let incident = all.iter_mut().find(|i| i.id == incident_id)?;
        incident.status = "closed".into();
        if !note.is_empty() {
            incident.notes.push(note.into());
        }
        Some(incident.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::governance::packs::load_pack;

    fn desk(packs: &[&str], notify: Vec<Notifier>) -> IncidentDesk {
        IncidentDesk::new(
            packs
                .iter()
                .map(|p| {
                    let p = load_pack(p).unwrap();
                    (p.id.clone(), p.incidents)
                })
                .collect(),
            notify,
            None,
        )
    }

    #[tokio::test]
    async fn incidents_carry_every_applicable_deadline() {
        let d = desk(&["gdpr", "dora", "ksa-pdpl"], vec![]);
        let breach = d.open("personal_data_breach", "high", "wrong recipient", "", "", Map::new()).await;
        let frameworks: std::collections::BTreeSet<&str> = breach.deadlines.iter().map(|d| d.framework.as_str()).collect();
        assert_eq!(frameworks, ["gdpr", "ksa-pdpl"].into());
        assert!(breach.deadlines.iter().all(|d| (d.within_hours - 72.0).abs() < 1e-9));
        let outage = d.open("ict_incident", "high", "provider down", "", "", Map::new()).await;
        assert_eq!(outage.deadlines.iter().map(|d| d.within_hours).collect::<Vec<_>>(), vec![4.0, 72.0, 720.0]);
        assert!(d.open("ict_incident", "low", "blip", "", "", Map::new()).await.deadlines.is_empty());
        assert_eq!(d.open_incidents().len(), 3);
        assert_eq!(d.close(&breach.id, "reported").unwrap().status, "closed");
    }

    #[tokio::test]
    async fn a_failing_pager_does_not_take_the_agent_down() {
        let d = desk(&["gdpr"], vec![notifier(|_| Err("ConnectionError: pager down".into()))]);
        let incident = d.open("personal_data_breach", "high", "x", "", "", Map::new()).await;
        assert!(!incident.deadlines.is_empty() && incident.notes[0].contains("pager down"));
    }
}
