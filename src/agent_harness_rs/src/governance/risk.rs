//! Risk classification, prohibited uses, and impact-assessment drafts.
//!
//! An agent declares the *domains* it works in; each framework's risk tiers are
//! then a lookup, not an opinion:
//!
//! ```
//! use agent_harness::governance::{AgentIdentity, assess};
//!
//! let hiring = assess(&AgentIdentity { domains: vec!["employment".into()], purpose: "rank CVs".into(), ..Default::default() });
//! assert_eq!(hiring.tier, "high");   // eu-ai-act: Annex III §4 · korea-ai-basic · colorado-adm · ...
//! ```
//!
//! A domain on the EU's Article 5 list makes the agent **prohibited** — the
//! governance layer refuses to register it at all, because no runtime control
//! makes a banned practice lawful.
//!
//! The impact-assessment drafts (DPIA, FRIA, Korea's AI impact assessment) are
//! filled from what the harness actually knows — model, providers, regions,
//! tools, data classes, oversight — and leave the questions only people can
//! answer marked as such. They are a first draft for your DPO, not a substitute.

use std::collections::BTreeSet;

use indexmap::IndexMap;
use serde_json::{Value, json};

use super::identity::{AgentIdentity, stricter};
use super::policy::Policy;

/// Domain vocabulary → which framework treats it how.
pub const DOMAINS: &[(&str, &[(&str, &str)])] = &[
    // EU AI Act Annex III areas.
    ("biometric_identification", &[("eu-ai-act", "Annex III \u{a7}1"), ("korea-ai-basic", "biometric")]),
    ("critical_infrastructure", &[("eu-ai-act", "Annex III \u{a7}2"), ("korea-ai-basic", "energy/water/transport")]),
    ("education", &[("eu-ai-act", "Annex III \u{a7}3"), ("korea-ai-basic", "student evaluation"), ("colorado-adm", "education")]),
    ("employment", &[("eu-ai-act", "Annex III \u{a7}4"), ("korea-ai-basic", "hiring"), ("colorado-adm", "employment"), ("ccpa-admt", "employment")]),
    (
        "credit",
        &[("eu-ai-act", "Annex III \u{a7}5(b)"), ("korea-ai-basic", "loan screening"), ("colorado-adm", "financial or lending"), ("ccpa-admt", "financial")],
    ),
    ("insurance", &[("eu-ai-act", "Annex III \u{a7}5(c)"), ("colorado-adm", "insurance")]),
    (
        "essential_services",
        &[("eu-ai-act", "Annex III \u{a7}5(a)"), ("colorado-adm", "essential government service"), ("korea-ai-basic", "public service decisions")],
    ),
    ("emergency_services", &[("eu-ai-act", "Annex III \u{a7}5(d)")]),
    ("healthcare", &[("korea-ai-basic", "healthcare"), ("colorado-adm", "health care"), ("ccpa-admt", "healthcare")]),
    ("medical_device", &[("eu-ai-act", "Annex I (MDR)"), ("korea-ai-basic", "medical devices")]),
    ("housing", &[("colorado-adm", "housing"), ("ccpa-admt", "housing")]),
    ("legal_services", &[("colorado-adm", "legal services")]),
    ("law_enforcement", &[("eu-ai-act", "Annex III \u{a7}6")]),
    ("migration", &[("eu-ai-act", "Annex III \u{a7}7")]),
    ("justice", &[("eu-ai-act", "Annex III \u{a7}8(a)")]),
    ("elections", &[("eu-ai-act", "Annex III \u{a7}8(b)")]),
    ("nuclear", &[("korea-ai-basic", "nuclear")]),
    // Everyday domains — limited risk at most.
    ("customer_support", &[]),
    ("coding", &[]),
    ("research", &[]),
    ("marketing", &[]),
    ("internal_operations", &[]),
    ("content_generation", &[]),
];

/// EU AI Act Art 5 (including the 2026 omnibus addition), Texas TRAIGA §552.
pub const PROHIBITED: &[(&str, &str)] = &[
    ("social_scoring", "Art 5(1)(c) social scoring"),
    ("manipulation", "Art 5(1)(a) subliminal or manipulative techniques"),
    ("exploiting_vulnerabilities", "Art 5(1)(b) exploiting age, disability or situation"),
    ("predictive_policing", "Art 5(1)(d) crime prediction from profiling alone"),
    ("facial_scraping", "Art 5(1)(e) untargeted scraping for facial recognition"),
    ("emotion_recognition_work", "Art 5(1)(f) emotion recognition at work or school"),
    ("biometric_categorisation", "Art 5(1)(g) inferring sensitive traits from biometrics"),
    ("realtime_remote_biometric_id", "Art 5(1)(h) real-time remote biometric ID in public"),
    ("non_consensual_intimate_imagery", "Art 5 (2026 omnibus) NCII and CSAM generation"),
    ("self_harm_incitement", "TRAIGA: inciting self-harm, harm to others or crime"),
];

fn domain(name: &str) -> Option<&'static [(&'static str, &'static str)]> {
    DOMAINS.iter().find(|(d, _)| *d == name).map(|(_, f)| *f)
}

fn treatment(name: &str, framework: &str) -> Option<&'static str> {
    domain(name)?.iter().find(|(f, _)| *f == framework).map(|(_, t)| *t)
}

/// One agent's classification under every framework we know.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RiskAssessment {
    pub agent: String,
    /// The strictest across frameworks.
    pub tier: String,
    pub frameworks: IndexMap<String, String>,
    /// framework → what it asks of a system in this tier.
    pub duties: IndexMap<String, Vec<String>>,
    pub reasons: Vec<String>,
}

impl RiskAssessment {
    pub fn obligations(&self) -> Vec<String> {
        self.duties.values().flatten().cloned().collect()
    }

    /// The same classification, reported under these frameworks only.
    pub fn only(&self, frameworks: &BTreeSet<String>) -> RiskAssessment {
        RiskAssessment {
            agent: self.agent.clone(),
            tier: self.tier.clone(),
            frameworks: self.frameworks.iter().filter(|(k, _)| frameworks.contains(*k)).map(|(k, v)| (k.clone(), v.clone())).collect(),
            duties: self.duties.iter().filter(|(k, _)| frameworks.contains(*k)).map(|(k, v)| (k.clone(), v.clone())).collect(),
            reasons: self.reasons.clone(),
        }
    }

    pub fn prohibited(&self) -> bool {
        self.tier == "prohibited"
    }

    pub fn as_dict(&self) -> Value {
        json!({"agent": self.agent, "tier": self.tier, "frameworks": self.frameworks, "duties": self.duties, "reasons": self.reasons})
    }
}

/// Classify one agent under every framework we know.
pub fn assess(identity: &AgentIdentity) -> RiskAssessment {
    let mut result = RiskAssessment { agent: identity.name.clone(), tier: "minimal".into(), ..Default::default() };
    let domains: BTreeSet<&str> = identity.domains.iter().map(String::as_str).collect();

    let banned: Vec<&str> = PROHIBITED.iter().filter(|(d, _)| domains.contains(d)).map(|(d, _)| *d).collect::<BTreeSet<_>>().into_iter().collect();
    if !banned.is_empty() {
        result.tier = "prohibited".into();
        result.frameworks.insert("eu-ai-act".into(), "prohibited".into());
        result.reasons.extend(banned.iter().filter_map(|d| PROHIBITED.iter().find(|(p, _)| p == d)).map(|(_, why)| why.to_string()));
        return result;
    }

    let under = |framework: &str| -> Vec<&'static str> {
        let mut out: Vec<&str> = domains.iter().filter_map(|d| treatment(d, framework)).collect();
        out.sort_unstable();
        out
    };
    let (eu, korea, colorado) = (under("eu-ai-act"), under("korea-ai-basic"), under("colorado-adm"));

    let mut tier = "minimal";
    if identity.user_facing || domains.contains("content_generation") {
        tier = "limited";
        result.frameworks.insert("eu-ai-act".into(), "limited (Art 50 transparency)".into());
        result.duties.entry("eu-ai-act".into()).or_default().push("disclose AI interaction; mark generated content (Art 50)".into());
    }
    if !eu.is_empty() {
        tier = "high";
        result.frameworks.insert("eu-ai-act".into(), format!("high-risk ({})", eu.join(", ")));
        result.duties.insert(
            "eu-ai-act".into(),
            [
                "risk management system (Art 9)",
                "data governance (Art 10)",
                "technical documentation (Art 11)",
                "automatic logging (Art 12)",
                "human oversight (Art 14)",
                "accuracy & robustness (Art 15)",
                "deployer: keep logs \u{2265} 6 months (Art 26(6)); FRIA where Art 27 applies",
            ]
            .map(String::from)
            .to_vec(),
        );
    }
    if !korea.is_empty() {
        tier = "high";
        result.frameworks.insert("korea-ai-basic".into(), format!("high-impact ({})", korea.join(", ")));
        result.duties.insert("korea-ai-basic".into(), vec!["risk management, impact assessment, human oversight, explanation, records".into()]);
    }
    if !colorado.is_empty() {
        tier = stricter(tier, "high");
        result.frameworks.insert("colorado-adm".into(), format!("consequential decision ({})", colorado.join(", ")));
        let duty = vec!["pre-use notice, explanation of adverse decisions, appeal route".to_string()];
        result.duties.insert("colorado-adm".into(), duty.clone());
        result.duties.insert("ccpa-admt".into(), duty);
    }
    if let Some(risk) = &identity.risk {
        // A declared tier can only raise the computed one.
        tier = stricter(tier, risk);
        result.reasons.push(format!("declared risk tier {risk}"));
    }
    if identity.autonomy == "autonomous" && tier == "high" {
        result.reasons.push("autonomous operation in a high-risk domain: approval gates on consequential actions strongly advised".into());
    }
    result.tier = tier.to_string();
    let unknown: Vec<&str> = domains.iter().filter(|d| domain(d).is_none()).copied().collect();
    if !unknown.is_empty() {
        result.reasons.push(format!("unrecognised domains (classify by hand): {}", unknown.join(", ")));
    }
    result
}

/// A Markdown draft of a DPIA (`dpia`), FRIA (`fria`) or Korean AI impact
/// assessment (`korea`). `inventory` is the agent's entry in the AI inventory.
pub fn impact_assessment(identity: &AgentIdentity, inventory: Option<&Value>, policy: Option<&Policy>, kind: &str, data_classes: &[String]) -> String {
    let risk = assess(identity);
    let empty = serde_json::Map::new();
    let inv = inventory.and_then(Value::as_object).unwrap_or(&empty);
    let title = match kind {
        "fria" => "Fundamental Rights Impact Assessment (EU AI Act Art 27)",
        "korea" => "AI Impact Assessment (Korea AI Basic Act)",
        _ => "Data Protection Impact Assessment",
    };
    let todo = "_To be completed by the accountable owner._";
    let get = |k: &str, default: &str| inv.get(k).map(super::py_str).unwrap_or_else(|| default.to_string());
    let packs = policy.map(|p| p.packs.join(", ")).unwrap_or_default();
    let tools: Vec<String> = inv
        .get("tools")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(|t| t.get("name").map(super::py_str).unwrap_or_default()).collect())
        .unwrap_or_default();
    let subagents: Vec<String> = inv.get("subagents").and_then(Value::as_array).map(|a| a.iter().map(super::py_str).collect()).unwrap_or_default();
    let mut classes: BTreeSet<String> = data_classes.iter().cloned().collect();
    classes.extend(identity.data.clone().unwrap_or_default());
    let or = |s: String, d: &str| if s.is_empty() { d.to_string() } else { s };
    let mut lines = vec![
        format!("# {title} \u{2014} draft"),
        String::new(),
        format!("**System:** `{}`  ", identity.name),
        format!("**Accountable owner:** {}  ", or(identity.owner.clone(), "**missing \u{2014} assign one**")),
        format!("**Purpose:** {}  ", or(identity.purpose.clone(), "**missing \u{2014} state it**")),
        format!("**Risk tier:** {}  ", risk.tier),
        format!("**Autonomy:** {}  ", identity.autonomy),
        format!("**Frameworks in force:** {}  ", or(packs.clone(), "none declared")),
        String::new(),
        "## 1. Description of the processing".into(),
        format!("- Model: {} via {} (region: {})", get("model", "n/a"), get("provider", "n/a"), get("region", "unknown")),
        format!("- Tools: {}", or(tools.join(", "), "none")),
        format!("- Sub-agents: {}", or(subagents.join(", "), "none")),
        format!("- Data classes observed or declared: {}", or(classes.into_iter().collect::<Vec<_>>().join(", "), "none recorded")),
        String::new(),
        "## 2. Classification".into(),
    ];
    lines.extend(risk.frameworks.iter().map(|(fw, tier)| format!("- **{fw}:** {tier}")));
    lines.extend(risk.reasons.iter().map(|r| format!("- {r}")));
    lines.extend([String::new(), "## 3. Obligations that follow".into()]);
    let obligations = risk.obligations();
    if obligations.is_empty() {
        lines.push("- none beyond general law".into());
    } else {
        lines.extend(obligations.iter().map(|o| format!("- {o}")));
    }
    let approvals = policy.map_or(0, |p| p.rules.iter().filter(|r| r.effect == "require_approval").count());
    lines.extend([
        String::new(),
        "## 4. Controls in place (from the governance layer)".into(),
        format!(
            "- Policy: `{}` v{}, packs: {}",
            policy.map_or("n/a".to_string(), |p| p.name.clone()),
            policy.map_or("?".to_string(), |p| p.version.clone()),
            or(packs, "none")
        ),
        format!("- Residency: {}", if policy.is_some_and(|p| p.residency.is_some()) { "configured" } else { "not configured" }),
        format!("- Human approval rules: {approvals}"),
        "- Decision records: hash-chained audit trail".into(),
        String::new(),
        "## 5. Necessity and proportionality".into(),
        todo.into(),
        String::new(),
        "## 6. Risks to the people affected".into(),
        todo.into(),
        String::new(),
        "## 7. Measures to address the risks".into(),
        todo.into(),
        String::new(),
        "## 8. Consultation and sign-off".into(),
        todo.into(),
    ]);
    lines.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_classification_per_framework() {
        let hiring = assess(&AgentIdentity { name: "cv".into(), domains: vec!["employment".into()], ..Default::default() });
        assert_eq!(hiring.tier, "high");
        assert!(hiring.frameworks["eu-ai-act"].contains("Annex III \u{a7}4"));
        assert!(hiring.frameworks["korea-ai-basic"].contains("hiring"));
        assert!(hiring.frameworks.contains_key("colorado-adm"));
        assert_eq!(assess(&AgentIdentity { name: "x".into(), user_facing: false, ..Default::default() }).tier, "minimal");
        assert!(assess(&AgentIdentity { name: "s".into(), domains: vec!["social_scoring".into()], ..Default::default() }).prohibited());
        let odd = assess(&AgentIdentity { risk: Some("high".into()), domains: vec!["astrology".into()], autonomy: "autonomous".into(), ..Default::default() });
        assert_eq!(odd.tier, "high");
        assert_eq!(odd.reasons.len(), 3);
    }

    #[test]
    fn impact_assessment_drafts() {
        let identity = AgentIdentity { name: "anon".into(), ..Default::default() };
        let draft = impact_assessment(&identity, None, Some(&Policy::default()), "dpia", &[]);
        assert!(draft.starts_with("# Data Protection Impact Assessment \u{2014} draft\n"));
        assert!(draft.contains("**Accountable owner:** **missing \u{2014} assign one**  "));
        assert!(draft.contains("- Policy: `policy` v1, packs: none"));
        assert!(draft.ends_with("_To be completed by the accountable owner._\n"));
    }
}
