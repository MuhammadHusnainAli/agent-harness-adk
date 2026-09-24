"""Risk classification, prohibited uses, and impact-assessment drafts.

An agent declares the *domains* it works in; each framework's risk tiers are
then a lookup, not an opinion:

    assess(AgentIdentity(domains=["employment"], purpose="rank CVs"))
    # eu-ai-act: high (Annex III §4) · korea-ai-basic: high-impact ·
    # colorado-adm: consequential decision · ...

A domain on the EU's Article 5 list makes the agent **prohibited** — the
governance layer refuses to register it at all, because no runtime control
makes a banned practice lawful.

The impact-assessment drafts (DPIA, FRIA, Korea's AI impact assessment) are
filled from what the harness actually knows — model, providers, regions,
tools, data classes, oversight — and leave the questions only people can
answer marked as such. They are a first draft for your DPO, not a substitute.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .identity import RISK_ORDER, AgentIdentity

__all__ = ["DOMAINS", "PROHIBITED", "RiskAssessment", "assess", "impact_assessment"]

#: Domain vocabulary → which framework treats it how.
DOMAINS: dict[str, dict[str, str]] = {
    # EU AI Act Annex III areas.
    "biometric_identification": {"eu-ai-act": "Annex III §1", "korea-ai-basic": "biometric"},
    "critical_infrastructure": {"eu-ai-act": "Annex III §2",
                                "korea-ai-basic": "energy/water/transport"},
    "education": {"eu-ai-act": "Annex III §3", "korea-ai-basic": "student evaluation",
                  "colorado-adm": "education"},
    "employment": {"eu-ai-act": "Annex III §4", "korea-ai-basic": "hiring",
                   "colorado-adm": "employment", "ccpa-admt": "employment"},
    "credit": {"eu-ai-act": "Annex III §5(b)", "korea-ai-basic": "loan screening",
               "colorado-adm": "financial or lending", "ccpa-admt": "financial"},
    "insurance": {"eu-ai-act": "Annex III §5(c)", "colorado-adm": "insurance"},
    "essential_services": {"eu-ai-act": "Annex III §5(a)",
                           "colorado-adm": "essential government service",
                           "korea-ai-basic": "public service decisions"},
    "emergency_services": {"eu-ai-act": "Annex III §5(d)"},
    "healthcare": {"korea-ai-basic": "healthcare", "colorado-adm": "health care",
                   "ccpa-admt": "healthcare"},
    "medical_device": {"eu-ai-act": "Annex I (MDR)", "korea-ai-basic": "medical devices"},
    "housing": {"colorado-adm": "housing", "ccpa-admt": "housing"},
    "legal_services": {"colorado-adm": "legal services"},
    "law_enforcement": {"eu-ai-act": "Annex III §6"},
    "migration": {"eu-ai-act": "Annex III §7"},
    "justice": {"eu-ai-act": "Annex III §8(a)"},
    "elections": {"eu-ai-act": "Annex III §8(b)"},
    "nuclear": {"korea-ai-basic": "nuclear"},
    # Everyday domains — limited risk at most.
    "customer_support": {}, "coding": {}, "research": {}, "marketing": {},
    "internal_operations": {}, "content_generation": {},
}

#: EU AI Act Art 5 (including the 2026 omnibus addition), Texas TRAIGA §552.
PROHIBITED: dict[str, str] = {
    "social_scoring": "Art 5(1)(c) social scoring",
    "manipulation": "Art 5(1)(a) subliminal or manipulative techniques",
    "exploiting_vulnerabilities": "Art 5(1)(b) exploiting age, disability or situation",
    "predictive_policing": "Art 5(1)(d) crime prediction from profiling alone",
    "facial_scraping": "Art 5(1)(e) untargeted scraping for facial recognition",
    "emotion_recognition_work": "Art 5(1)(f) emotion recognition at work or school",
    "biometric_categorisation": "Art 5(1)(g) inferring sensitive traits from biometrics",
    "realtime_remote_biometric_id": "Art 5(1)(h) real-time remote biometric ID in public",
    "non_consensual_intimate_imagery": "Art 5 (2026 omnibus) NCII and CSAM generation",
    "self_harm_incitement": "TRAIGA: inciting self-harm, harm to others or crime",
}


@dataclass
class RiskAssessment:
    agent: str
    tier: str                                   # the strictest across frameworks
    frameworks: dict[str, str] = field(default_factory=dict)
    obligations: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def prohibited(self) -> bool:
        return self.tier == "prohibited"

    def as_dict(self) -> dict[str, Any]:
        return {"agent": self.agent, "tier": self.tier, "frameworks": self.frameworks,
                "obligations": self.obligations, "reasons": self.reasons}


def assess(identity: AgentIdentity) -> RiskAssessment:
    """Classify one agent under every framework we know."""
    result = RiskAssessment(agent=identity.name, tier="minimal")
    domains = set(identity.domains)

    banned = sorted(domains & set(PROHIBITED))
    if banned:
        result.tier = "prohibited"
        result.frameworks["eu-ai-act"] = "prohibited"
        result.reasons += [PROHIBITED[d] for d in banned]
        return result

    eu = [DOMAINS[d]["eu-ai-act"] for d in domains
          if d in DOMAINS and "eu-ai-act" in DOMAINS[d]]
    korea = [DOMAINS[d]["korea-ai-basic"] for d in domains
             if d in DOMAINS and "korea-ai-basic" in DOMAINS[d]]
    colorado = [DOMAINS[d]["colorado-adm"] for d in domains
                if d in DOMAINS and "colorado-adm" in DOMAINS[d]]

    tier = "minimal"
    if identity.user_facing or "content_generation" in domains:
        tier = "limited"
        result.frameworks["eu-ai-act"] = "limited (Art 50 transparency)"
        result.obligations.append("disclose AI interaction; mark generated content")
    if eu:
        tier = "high"
        result.frameworks["eu-ai-act"] = "high-risk (" + ", ".join(sorted(eu)) + ")"
        result.obligations += [
            "risk management system (Art 9)", "data governance (Art 10)",
            "technical documentation (Art 11)", "automatic logging (Art 12)",
            "human oversight (Art 14)", "accuracy & robustness (Art 15)",
            "deployer: keep logs ≥ 6 months (Art 26(6)); FRIA where Art 27 applies",
        ]
    if korea:
        tier = "high"
        result.frameworks["korea-ai-basic"] = "high-impact (" + ", ".join(sorted(korea)) + ")"
        result.obligations.append("Korea: risk management, impact assessment, "
                                  "human oversight, explanation, records")
    if colorado:
        tier = max(tier, "high", key=RISK_ORDER.__getitem__)
        result.frameworks["colorado-adm"] = ("consequential decision ("
                                             + ", ".join(sorted(colorado)) + ")")
        result.obligations.append("Colorado/CCPA: pre-use ADMT notice, "
                                  "explanation of adverse decisions, appeal route")
    if identity.risk is not None:
        # A declared tier can only raise the computed one.
        tier = max(tier, identity.risk, key=RISK_ORDER.__getitem__)
        result.reasons.append(f"declared risk tier {identity.risk}")
    if identity.autonomy == "autonomous" and tier == "high":
        result.reasons.append("autonomous operation in a high-risk domain: approval "
                              "gates on consequential actions strongly advised")
    result.tier = tier
    unknown = sorted(d for d in domains if d not in DOMAINS)
    if unknown:
        result.reasons.append("unrecognised domains (classify by hand): "
                              + ", ".join(unknown))
    return result


def impact_assessment(identity: AgentIdentity, *, inventory: dict[str, Any] | None = None,
                      policy: Any = None, kind: str = "dpia",
                      data_classes: Iterable[str] = ()) -> str:
    """A Markdown draft of a DPIA, FRIA or Korean AI impact assessment."""
    risk = assess(identity)
    inv = inventory or {}
    titles = {"dpia": "Data Protection Impact Assessment (GDPR Art 35)",
              "fria": "Fundamental Rights Impact Assessment (EU AI Act Art 27)",
              "korea": "AI Impact Assessment (Korea AI Basic Act)"}
    todo = "_To be completed by the accountable owner._"
    lines = [
        f"# {titles.get(kind, titles['dpia'])} — draft",
        "",
        f"**System:** `{identity.name}`  ",
        f"**Accountable owner:** {identity.owner or '**missing — assign one**'}  ",
        f"**Purpose:** {identity.purpose or '**missing — state it**'}  ",
        f"**Risk tier:** {risk.tier}  ",
        f"**Autonomy:** {identity.autonomy}  ",
        "",
        "## 1. Description of the processing",
        f"- Model: {inv.get('model', 'n/a')} via {inv.get('provider', 'n/a')} "
        f"(region: {inv.get('region', 'unknown')})",
        f"- Tools: {', '.join(t['name'] for t in inv.get('tools', [])) or 'none'}",
        f"- Sub-agents: {', '.join(inv.get('subagents', [])) or 'none'}",
        f"- Data classes observed or declared: "
        f"{', '.join(sorted(set(data_classes) | set(identity.data or []))) or 'none recorded'}",
        "",
        "## 2. Classification",
        *[f"- **{fw}:** {tier}" for fw, tier in risk.frameworks.items()],
        *[f"- {reason}" for reason in risk.reasons],
        "",
        "## 3. Obligations that follow",
        *([f"- {o}" for o in risk.obligations] or ["- none beyond general law"]),
        "",
        "## 4. Controls in place (from the governance layer)",
        f"- Policy: `{getattr(policy, 'name', 'n/a')}` v{getattr(policy, 'version', '?')}, "
        f"packs: {', '.join(getattr(policy, 'packs', []) or []) or 'none'}",
        f"- Residency: {'configured' if getattr(policy, 'residency', None) else 'not configured'}",
        f"- Human approval rules: "
        f"{sum(1 for r in getattr(policy, 'rules', []) if r.effect == 'require_approval')}",
        "- Decision records: hash-chained audit trail",
        "",
        "## 5. Necessity and proportionality",
        todo,
        "",
        "## 6. Risks to the people affected",
        todo,
        "",
        "## 7. Measures to address the risks",
        todo,
        "",
        "## 8. Consultation and sign-off",
        todo,
    ]
    return "\n".join(lines) + "\n"
