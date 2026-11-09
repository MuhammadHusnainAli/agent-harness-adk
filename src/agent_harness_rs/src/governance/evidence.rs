//! The control catalog, and the report that proves each control is working.
//!
//! Sixteen controls (C1–C16) are what every pack's requirements reduce to. For
//! each one the report says whether it is **met**, **partial** or a **gap** on
//! *this* deployment — worked out from the live configuration and the audit
//! trail, not from a questionnaire — and gives the evidence and what to do next.
//!
//! Requirements a machine cannot meet (appoint a DPO, register in the EU
//! database, get a DPIA signed) are listed as **manual**. The report never
//! claims compliance; it shows what the technical controls cover and what is
//! left to people.
//!
//! ```no_run
//! # fn demo(gov: agent_harness::governance::Governance) {
//! println!("{}", gov.report("eu-ai-act").markdown());
//! gov.report_all().json();          // every active pack
//! # }
//! ```

use indexmap::IndexMap;
use serde_json::{Value, json};

use super::core::Governance;
use super::packs::{Pack, load_pack};
use super::{py_repr, utc};
use crate::types::now;

/// `met`, `partial`, `gap` or `manual`.
fn order(status: &str) -> usize {
    match status {
        "met" => 0,
        "manual" => 1,
        "partial" => 2,
        _ => 3,
    }
}

/// One control's state on this deployment.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlStatus {
    pub id: String,
    pub title: String,
    pub status: String,
    pub evidence: Vec<String>,
    pub actions: Vec<String>,
}

/// One control in the catalog.
#[derive(Clone, Copy)]
pub struct Control {
    pub id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub check: fn(&Governance) -> ControlStatus,
}

impl std::fmt::Debug for Control {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.id, self.title)
    }
}

fn title(id: &str) -> String {
    CONTROLS.iter().find(|c| c.id == id).map_or_else(|| id.to_string(), |c| c.title.to_string())
}

fn fixed(id: &str, status: &str, evidence: Vec<String>, actions: Vec<String>) -> ControlStatus {
    ControlStatus { id: id.into(), title: title(id), status: status.into(), evidence, actions }
}

fn status(id: &str, ok: usize, total: usize, evidence: Vec<String>, actions: Vec<String>) -> ControlStatus {
    let state = if total > 0 && ok == total {
        "met"
    } else if ok > 0 {
        "partial"
    } else {
        "gap"
    };
    fixed(id, state, evidence, if state == "met" { vec![] } else { actions })
}

fn decisions(gov: &Governance, action: Option<&str>) -> IndexMap<String, u64> {
    let mut out: IndexMap<String, u64> = IndexMap::new();
    for ((act, effect), n) in gov.stats() {
        if action.is_none_or(|a| a == act) {
            *out.entry(effect).or_default() += n;
        }
    }
    out
}

fn yes_no(b: bool, yes: &str, no: &str) -> String {
    if b { yes.into() } else { no.into() }
}

fn c1(gov: &Governance) -> ControlStatus {
    let ids = gov.identities();
    let ok: Vec<&String> = ids.iter().filter(|(_, i)| i.accountable()).map(|(n, _)| n).collect();
    let mut missing: Vec<&String> = ids.keys().filter(|n| !ok.contains(n)).collect();
    missing.sort();
    let action = if missing.is_empty() {
        "register agents on a governed harness".to_string()
    } else {
        format!("give {} an owner and a purpose", missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
    };
    status("C1", ok.len(), ids.len(), vec![format!("{} agents registered, {} with an owner and purpose", ids.len(), ok.len())], vec![action])
}

fn c2(gov: &Governance) -> ControlStatus {
    let ids = gov.identities();
    let scoped = ids.values().filter(|i| i.tools.is_some()).count();
    let depth = gov.policy().max_delegation_depth.is_some() || ids.values().any(|i| i.max_delegation_depth.is_some() || !i.may_delegate);
    let denied = decisions(gov, Some("delegate")).get("deny").copied().unwrap_or(0);
    let evidence = vec![
        format!("{scoped}/{} agents have a tool allowlist", ids.len()),
        format!("delegation depth limited: {}", yes_no(depth, "yes", "no")),
        format!("{denied} delegations refused"),
    ];
    let mut actions = vec![];
    if scoped < ids.len() {
        actions.push("declare `tools=` on every agent identity".to_string());
    }
    if !depth {
        actions.push("set `max_delegation_depth` (the owasp-agentic pack sets 3)".into());
    }
    status("C2", scoped + usize::from(depth), ids.len() + 1, evidence, actions)
}

fn c3(gov: &Governance) -> ControlStatus {
    let total: u64 = gov.stats().values().sum();
    let policy = gov.policy();
    let rules = policy.rules.len();
    status(
        "C3",
        usize::from(rules > 0) + usize::from(total > 0),
        2,
        vec![
            format!("policy {}@{} sha256:{}", policy.name, policy.version, &policy.hash()[..16]),
            format!("{rules} rules"),
            format!("{total} decisions recorded"),
        ],
        vec![if rules == 0 { "add rules or packs" } else { "run agents to record decisions" }.into()],
    )
}

fn c4(gov: &Governance) -> ControlStatus {
    let mut seen: Vec<(String, u64)> = gov.data_seen().into_iter().collect();
    seen.sort();
    let seen = if seen.is_empty() { "none yet".to_string() } else { seen.iter().map(|(k, v)| format!("{k}\u{d7}{v}")).collect::<Vec<_>>().join(", ") };
    fixed("C4", "met", vec![format!("deterministic classifier on every egress, tool call and memory write; classes observed: {seen}")], vec![])
}

fn c5(gov: &Governance) -> ControlStatus {
    let policy = gov.policy();
    let rules = policy.rules.iter().filter(|r| r.effect == "redact" || !r.redact.is_empty()).count();
    let purposes = !policy.purposes.is_empty();
    status(
        "C5",
        usize::from(rules > 0) + usize::from(purposes),
        2,
        vec![
            format!("{rules} redaction rules"),
            format!("purpose-based minimisation: {}", yes_no(purposes, "on", "off")),
            "credentials are always stripped from memory".into(),
        ],
        vec!["add a redact rule for egress or declare `purposes:` with the data classes each purpose may use".into()],
    )
}

fn c6(gov: &Governance) -> ControlStatus {
    let policy = gov.policy();
    let Some(residency) = policy.residency.as_ref().filter(|r| !r.transfers.is_empty()) else {
        return fixed("C6", "gap", vec!["no residency limits configured".into()], vec!["add a jurisdiction pack (gdpr, ksa-pdpl, ...) or `residency:`".into()]);
    };
    let transfers = gov.audit().map_or(0, |a| a.entries().iter().filter(|e| e.action == "governance.transfer").count());
    let egress = decisions(gov, Some("egress"));
    let mut limits: Vec<(&String, &Vec<String>)> = residency.transfers.iter().collect();
    limits.sort();
    let evidence = vec![
        format!("home: {}", residency.home.clone().unwrap_or_else(|| "(per principal)".into())),
        format!("limits: {}", limits.iter().map(|(o, d)| format!("{o} \u{2192} {}", d.join(", "))).collect::<Vec<_>>().join("; ")),
        format!("unknown regions: {}", residency.unknown),
        format!("egress decisions: {}", py_repr(&json!(egress))),
        format!("{transfers} cross-border transfers recorded"),
    ];
    let mut restricted: Vec<&String> = residency.transfers.iter().filter(|(_, d)| !d.iter().any(|x| x == "*")).map(|(o, _)| o).collect();
    if residency.home.is_none() && restricted.len() > 1 {
        restricted.sort();
        let names = restricted.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
        return fixed(
            "C6",
            "partial",
            evidence,
            vec![format!("set `home=` \u{2014} a principal with no `jurisdiction` tag is otherwise unrestricted ({names} all restrict transfers)")],
        );
    }
    let unknown: Vec<String> = gov.inventory().agents().values().filter(|a| a["region"] == "unknown").map(|a| super::py_str(&a["name"])).collect();
    if !unknown.is_empty() && residency.unknown == "allow" {
        return fixed("C6", "partial", evidence, vec![format!("declare regions for {} (`regions=`)", unknown.join(", "))]);
    }
    fixed("C6", "met", evidence, vec![])
}

fn c7(gov: &Governance) -> ControlStatus {
    let policy = gov.policy();
    let purposes = &policy.purposes;
    let ids = gov.identities();
    let ok = ids.values().filter(|i| !i.purpose.is_empty() && (purposes.is_empty() || purposes.contains_key(&i.purpose))).count();
    let mut declared: Vec<&String> = purposes.keys().collect();
    declared.sort();
    let declared = if declared.is_empty() { "none".to_string() } else { declared.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ") };
    status(
        "C7",
        ok + usize::from(!purposes.is_empty()),
        ids.len() + 1,
        vec![format!("declared purposes: {declared}"), format!("{ok}/{} agents run under a declared purpose", ids.len())],
        vec!["declare `purposes:` in the policy and give every agent one of them".into()],
    )
}

fn c8(gov: &Governance) -> ControlStatus {
    let rules = gov.policy().rules.iter().filter(|r| r.effect == "require_approval").count();
    let history = gov.oversight().history();
    let approved = history.iter().filter(|h| h["status"] == "approved").count();
    let high = gov.identities().values().any(|i| i.risk.as_deref() == Some("high"));
    let evidence = vec![
        format!("{rules} approval rules"),
        format!("{} approval requests, {approved} approved", history.len()),
        "fail-closed timeouts, quorum of distinct approvers, separation of duties".into(),
        "stop control: harness.stop() halts every run".into(),
    ];
    if rules == 0 {
        return fixed(
            "C8",
            if high { "gap" } else { "partial" },
            evidence,
            vec!["add `require_approval` rules for consequential or irreversible tools".into()],
        );
    }
    fixed("C8", "met", evidence, vec![])
}

fn c9(gov: &Governance) -> ControlStatus {
    let t = gov.policy().transparency.clone();
    let signed = if t.provenance && gov.signer().is_some() { " (signed)" } else { "" };
    status(
        "C9",
        usize::from(t.disclose) + usize::from(t.provenance),
        2,
        vec![
            format!("disclosure: {} ({})", yes_no(t.disclose, "on", "off"), t.languages.join(", ")),
            format!("provenance manifests: {}{signed}", yes_no(t.provenance, "on", "off")),
            format!("visible labels: {}", yes_no(t.label_output, "on", "off")),
        ],
        vec!["enable transparency.disclose and transparency.provenance (the eu-ai-act pack does)".into()],
    )
}

fn c10(gov: &Governance) -> ControlStatus {
    let Some(audit) = gov.audit() else {
        return fixed("C10", "gap", vec!["not attached".into()], vec!["attach governance to a harness".into()]);
    };
    let (ok, why) = audit.verify();
    let signer = gov.trail_key_id();
    let path = gov.audit_path();
    let retention = gov.policy().log_retention_days.map_or_else(|| "not set".to_string(), |d| d.to_string());
    let evidence = vec![
        format!("{} entries, chain: {why}", audit.len()),
        format!("signed: {}", signer.as_ref().map_or_else(|| "no".to_string(), |k| format!("yes ({k})"))),
        format!("persisted: {}", path.clone().unwrap_or_else(|| "no (in memory)".into())),
        format!("minimum retention: {retention} days"),
    ];
    let mut actions = vec![];
    if !ok {
        actions.push(format!("the audit chain is broken \u{2014} {why}; investigate before anything else"));
    }
    if signer.is_none() {
        actions.push("pass `signing_key=` (HMAC) or an Ed25519Signer".into());
    }
    if path.is_none() {
        actions.push("persist the audit trail (Harness.local does)".into());
    }
    let checks = usize::from(ok) + usize::from(signer.is_some()) + usize::from(path.is_some());
    let mut s = status("C10", checks, 3, evidence, actions);
    if !ok {
        s.status = "gap".into();
    }
    s
}

fn c11(gov: &Governance) -> ControlStatus {
    let durable = gov.vault().path.is_some();
    status(
        "C11",
        1 + usize::from(durable),
        2,
        vec![
            "access and erasure across memory, sessions and the audit trail (crypto-shredding)".into(),
            format!("subjects indexed: {}", gov.vault().subjects().len()),
            format!("subject index persisted: {}", yes_no(durable, "yes", "no")),
        ],
        vec!["give the vault a path so the subject index survives restarts".into()],
    )
}

fn c12(gov: &Governance) -> ControlStatus {
    let assessments = gov.assessments();
    let mut tiers: IndexMap<String, u64> = IndexMap::new();
    for a in assessments.values() {
        *tiers.entry(a.tier.clone()).or_default() += 1;
    }
    let unowned: Vec<String> = gov.identities().iter().filter(|(_, i)| i.risk.as_deref() == Some("high") && !i.accountable()).map(|(n, _)| n.clone()).collect();
    let tiers = if tiers.is_empty() { "no agents".to_string() } else { py_repr(&json!(tiers)) };
    let action = if unowned.is_empty() {
        "register agents to classify them".to_string()
    } else {
        format!("high-risk agents without an accountable owner: {}", unowned.join(", "))
    };
    status(
        "C12",
        assessments.len().saturating_sub(unowned.len()),
        assessments.len(),
        vec![format!("tiers: {tiers}"), "prohibited uses are refused at registration".into()],
        vec![action],
    )
}

fn c13(gov: &Governance) -> ControlStatus {
    let inv = gov.inventory();
    let pinned = !inv.pins().is_empty();
    let drift = if pinned { Some(inv.drift_value()) } else { None };
    let changed = drift.as_ref().is_some_and(|d| d.as_object().is_some_and(|m| m.values().any(|v| v.as_array().is_some_and(|a| !a.is_empty()))));
    let agents = inv.agents().len();
    let evidence = vec![
        format!("{agents} agents, {} tools fingerprinted", inv.fingerprints().len()),
        format!("pinned: {}", yes_no(pinned, "yes", "no")),
        format!("drift: {}", drift.as_ref().map_or_else(|| "n/a".to_string(), py_repr)),
        format!("third parties: {}", inv.third_parties().len()),
    ];
    let mut actions = vec![];
    if !pinned {
        actions.push("review the inventory and call `inventory.pin()`".to_string());
    }
    if changed {
        actions.push("review the drifted tools, then re-pin".into());
    }
    status("C13", usize::from(agents > 0) + usize::from(pinned && !changed), 2, evidence, actions)
}

fn c14(gov: &Governance) -> ControlStatus {
    let m = gov.monitor();
    let lim = |v: Option<u64>| v.filter(|n| *n > 0).map_or_else(|| "\u{221e}".to_string(), |n| n.to_string());
    fixed(
        "C14",
        "met",
        vec![
            format!(
                "loop detection at {} identical calls, {} calls per tool, {} delegations per run",
                m.max_identical_calls,
                lim(m.max_calls_per_tool),
                lim(m.max_delegations)
            ),
            "prompt-injection signals in tool results mark the run untrusted".into(),
            "health and tracing: harness.health, harness.tracer".into(),
        ],
        vec![],
    )
}

fn c15(gov: &Governance) -> ControlStatus {
    let clocks: usize = gov.packs().iter().map(|p| p.incidents.values().map(Vec::len).sum::<usize>()).sum();
    let desk = gov.incidents();
    let hooked = !desk.notify.is_empty();
    status(
        "C15",
        usize::from(clocks > 0) + usize::from(hooked),
        2,
        vec![
            format!("{clocks} regulatory clocks from active packs"),
            format!("notification hooks: {}", desk.notify.len()),
            format!("open incidents: {}, overdue: {}", desk.open_incidents().len(), desk.overdue().len()),
        ],
        vec![if hooked { "activate packs with incident clocks" } else { "pass `notify=[...]` so a person hears about incidents" }.into()],
    )
}

fn c16(_: &Governance) -> ControlStatus {
    fixed("C16", "met", vec!["this report, generated from live configuration and the audit trail".into()], vec![])
}

/// The sixteen controls every pack's requirements reduce to.
pub const CONTROLS: [Control; 16] = [
    Control { id: "C1", title: "Agent identity & accountability", description: "every agent has an owner, a purpose and a risk tier", check: c1 },
    Control { id: "C2", title: "Least privilege & delegation", description: "sub-agents act under a narrowed copy of their parent's authority", check: c2 },
    Control { id: "C3", title: "Policy as code", description: "versioned, hashed policy; every decision names the policy that made it", check: c3 },
    Control { id: "C4", title: "Data classification", description: "personal, special-category, credential and national-ID data detected", check: c4 },
    Control { id: "C5", title: "Minimisation & pseudonymisation", description: "data is redacted or tokenised before it leaves", check: c5 },
    Control {
        id: "C6",
        title: "Residency & cross-border transfer",
        description: "each model call's real region is checked against transfer limits",
        check: c6,
    },
    Control { id: "C7", title: "Purpose limitation", description: "a run carries a purpose and may use only the data it allows", check: c7 },
    Control { id: "C8", title: "Human oversight", description: "approvals with quorum and fail-closed timeouts; a stop switch", check: c8 },
    Control { id: "C9", title: "Transparency & provenance", description: "AI disclosure and machine-readable marking of output", check: c9 },
    Control { id: "C10", title: "Tamper-evident records", description: "signed, hash-chained, retained decision records", check: c10 },
    Control { id: "C11", title: "Data-subject rights", description: "access and erasure without breaking the audit trail", check: c11 },
    Control { id: "C12", title: "Risk classification", description: "per-framework risk tiers; prohibited uses refused", check: c12 },
    Control { id: "C13", title: "AI inventory & supply chain", description: "models, providers, tools and MCP servers recorded and pinned", check: c13 },
    Control { id: "C14", title: "Runtime monitoring", description: "loops, storms, cascades and injection signals detected", check: c14 },
    Control { id: "C15", title: "Incident management", description: "incidents opened with every applicable regulatory deadline", check: c15 },
    Control { id: "C16", title: "Evidence & reporting", description: "control status and evidence, per framework, on demand", check: c16 },
];

/// One requirement of one pack, and where it stands.
#[derive(Debug, Clone, PartialEq)]
pub struct RequirementStatus {
    pub pack: String,
    pub r#ref: String,
    pub title: String,
    pub status: String,
    pub controls: Vec<String>,
    pub note: String,
}

/// What the report always says about itself.
pub const DISCLAIMER: &str = "This report shows which technical controls are in place and the evidence for them. It is not legal advice and does not by itself establish compliance with any law; organisational measures and legal review are still required.";

/// The evidence report: every control, and every requirement of the packs asked about.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplianceReport {
    pub generated: f64,
    pub policy: String,
    pub mode: String,
    pub controls: IndexMap<String, ControlStatus>,
    pub requirements: Vec<RequirementStatus>,
    pub packs: Vec<Value>,
}

impl ComplianceReport {
    /// For one pack (active or not), or every active one.
    pub fn build(gov: &Governance, pack: Option<&str>) -> ComplianceReport {
        let active = gov.packs();
        let mut packs: Vec<Pack> = active.iter().filter(|p| pack.is_none_or(|id| p.id == id)).cloned().collect();
        if let Some(id) = pack
            && packs.is_empty()
            && let Ok(p) = load_pack(id)
        {
            packs.push(p); // report against a pack that is not active
        }
        let controls: IndexMap<String, ControlStatus> = CONTROLS.iter().map(|c| (c.id.to_string(), (c.check)(gov))).collect();
        let mut requirements = vec![];
        for p in &packs {
            for r in &p.requirements {
                let state = if r.manual || r.controls.is_empty() {
                    "manual".to_string()
                } else {
                    r.controls.iter().filter_map(|c| controls.get(c)).map(|c| c.status.clone()).max_by_key(|s| order(s)).unwrap_or_else(|| "gap".into())
                };
                requirements.push(RequirementStatus {
                    pack: p.id.clone(),
                    r#ref: r.r#ref.clone(),
                    title: r.title.clone(),
                    status: state,
                    controls: r.controls.clone(),
                    note: r.note.clone(),
                });
            }
        }
        let policy = gov.policy();
        ComplianceReport {
            generated: now(),
            policy: format!("{}@{} sha256:{}", policy.name, policy.version, &policy.hash()[..16]),
            mode: gov.mode().to_string(),
            controls,
            requirements,
            packs: packs
                .iter()
                .map(|p| json!({"id": p.id, "name": p.name, "jurisdiction": p.jurisdiction, "binding": p.binding, "status": p.status, "as_of": p.as_of, "active": active.iter().any(|a| a == p), "sources": p.sources}))
                .collect(),
        }
    }

    pub fn counts(&self) -> IndexMap<String, usize> {
        let mut out: IndexMap<String, usize> = ["met", "partial", "gap", "manual"].iter().map(|s| (s.to_string(), 0)).collect();
        for r in &self.requirements {
            *out.entry(r.status.clone()).or_default() += 1;
        }
        out
    }

    pub fn gaps(&self) -> Vec<&RequirementStatus> {
        self.requirements.iter().filter(|r| r.status == "gap" || r.status == "partial").collect()
    }

    pub fn as_dict(&self) -> Value {
        json!({
            "generated": utc(self.generated, true),
            "policy": self.policy, "mode": self.mode, "packs": self.packs,
            "summary": self.counts(),
            "controls": self.controls.iter().map(|(k, v)| (k.clone(), json!({"title": v.title, "status": v.status, "evidence": v.evidence, "actions": v.actions}))).collect::<serde_json::Map<_, _>>(),
            "requirements": self.requirements.iter().map(|r| json!({"pack": r.pack, "ref": r.r#ref, "title": r.title, "status": r.status, "controls": r.controls, "note": r.note})).collect::<Vec<_>>(),
            "disclaimer": DISCLAIMER,
        })
    }

    /// Indented JSON, text as UTF-8.
    pub fn json(&self) -> String {
        serde_json::to_string_pretty(&self.as_dict()).unwrap_or_default()
    }

    pub fn markdown(&self) -> String {
        let mut lines = vec![
            "# Governance evidence report".to_string(),
            String::new(),
            format!("Policy `{}` \u{b7} mode **{}** \u{b7} generated {}", self.policy, self.mode, utc(self.generated, false)),
            String::new(),
        ];
        let c = self.counts();
        lines.push(format!("**{} met \u{b7} {} partial \u{b7} {} gap \u{b7} {} manual**", c["met"], c["partial"], c["gap"], c["manual"]));
        lines.push(String::new());
        for p in &self.packs {
            let s = |k: &str| super::py_str(&p[k]);
            lines.push(format!("## {} (`{}`)", s("name"), s("id")));
            lines.push(String::new());
            lines.push(format!(
                "{} \u{b7} {} \u{b7} {} \u{b7} checked {}",
                s("jurisdiction"),
                if p["binding"] == true { "binding" } else { "voluntary" },
                s("status"),
                s("as_of")
            ));
            lines.push(String::new());
            lines.push("| Ref | Requirement | Controls | Status |".into());
            lines.push("|---|---|---|---|".into());
            for r in self.requirements.iter().filter(|r| r.pack == s("id")) {
                let controls = if r.controls.is_empty() { "\u{2014}".to_string() } else { r.controls.join(", ") };
                lines.push(format!("| {} | {} | {controls} | {} |", r.r#ref, r.title, badge(&r.status)));
            }
            lines.push(String::new());
        }
        lines.push("## Controls".into());
        lines.push(String::new());
        for c in self.controls.values() {
            lines.push(format!("### {} {} \u{2014} {}", c.id, c.title, badge(&c.status)));
            lines.extend(c.evidence.iter().map(|e| format!("- {e}")));
            lines.extend(c.actions.iter().map(|a| format!("- **Next:** {a}")));
            lines.push(String::new());
        }
        lines.extend(["---".to_string(), format!("_{DISCLAIMER}_"), String::new()]);
        lines.join("\n")
    }

    /// The Markdown report — what a terminal or a pull request shows best.
    pub fn render(&self) -> String {
        self.markdown()
    }

    pub fn html(&self) -> String {
        let mut body = String::new();
        for p in &self.packs {
            let id = super::py_str(&p["id"]);
            let rows: String = self
                .requirements
                .iter()
                .filter(|r| r.pack == id)
                .map(|r| {
                    format!(
                        "<tr><td>{}</td><td>{}</td><td>{}</td><td class='{}'>{}</td></tr>",
                        escape(&r.r#ref),
                        escape(&r.title),
                        escape(&r.controls.join(", ")),
                        r.status,
                        r.status
                    )
                })
                .collect();
            body.push_str(&format!(
                "<h2>{}</h2><table><tr><th>Ref</th><th>Requirement</th><th>Controls</th><th>Status</th></tr>{rows}</table>",
                escape(&super::py_str(&p["name"]))
            ));
        }
        let controls: String = self
            .controls
            .values()
            .map(|c| {
                format!(
                    "<h3>{} {} <span class='{}'>{}</span></h3><ul>{}{}</ul>",
                    c.id,
                    escape(&c.title),
                    c.status,
                    c.status,
                    c.evidence.iter().map(|e| format!("<li>{}</li>", escape(e))).collect::<String>(),
                    c.actions.iter().map(|a| format!("<li><b>Next:</b> {}</li>", escape(a))).collect::<String>()
                )
            })
            .collect();
        format!(
            "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Governance report</title><style>:root{{color-scheme:light dark;--fg:#1b1b1f;--bg:#fff;--line:#d7d7de}}@media (prefers-color-scheme:dark){{:root{{--fg:#e8e8ee;--bg:#15151a;--line:#33333d}}}}body{{font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:960px;padding:16px;color:var(--fg);background:var(--bg)}}table{{border-collapse:collapse;width:100%;display:block;overflow-x:auto}}td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}}.met{{color:#1a7f37}}.partial{{color:#9a6700}}.gap{{color:#cf222e}}.manual{{color:#6e7781}}</style></head><body><h1>Governance evidence report</h1><p>Policy <code>{}</code> \u{b7} mode {}</p>{body}<h2>Controls</h2>{controls}<p><em>{}</em></p></body></html>",
            escape(&self.policy),
            self.mode,
            escape(DISCLAIMER)
        )
    }
}

/// `html.escape(text)`.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&#x27;")
}

fn badge(status: &str) -> &'static str {
    match status {
        "met" => "\u{2705} met",
        "partial" => "\u{1f7e1} partial",
        "gap" => "\u{274c} gap",
        _ => "\u{1f4dd} manual",
    }
}
