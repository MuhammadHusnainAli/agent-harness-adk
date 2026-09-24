"""Governance: policy, identity, data, residency, oversight and evidence for agents.

    from agent_harness import Agent, Harness
    from agent_harness.governance import Governance, AgentIdentity

    gov = Governance.from_packs(["eu-ai-act", "gdpr", "owasp-agentic"], home="eu",
                                signing_key=os.environ["AUDIT_KEY"])
    harness = Harness.local(".harness", governance=gov)

    support = Agent("support", "Answer order questions.", harness=harness,
                    tools=[order_status, issue_refund],
                    identity=AgentIdentity(owner="ops@acme.com",
                                           purpose="customer_support",
                                           tools=["order_status", "issue_refund"]))

    print(gov.report("eu-ai-act").markdown())

Packs turn on what a law or framework asks for; your own policy adds rules on
top; every combination can only tighten. See `list_packs()` for what ships —
EU (AI Act, GDPR, DORA, NIS2), UK, the Gulf (UAE, DIFC, ADGM, Saudi Arabia,
Qatar, Bahrain), Asia-Pacific (Singapore, India, China, Korea, Japan, Vietnam),
the US (NIST AI RMF, Colorado, Texas, California) and the standards
(ISO/IEC 42001, OWASP Top 10 for Agentic Applications).

This supplies technical controls and the evidence for them. It is not legal
advice, and it does not by itself make anyone compliant with anything.
"""

from __future__ import annotations

from .core import Governance, Mode
from .data import (
    DATA_CLASSES,
    PERSONAL,
    SPECIAL_CATEGORIES,
    Classification,
    DataClassifier,
    DataFinding,
    Pseudonymizer,
)
from .evidence import CONTROLS, ComplianceReport
from .expr import ExpressionError, compile_expr
from .identity import AgentIdentity, DelegationChain, RiskTier
from .incidents import Deadline, Incident, IncidentDesk
from .inventory import AIInventory
from .monitor import RunState, RuntimeMonitor
from .oversight import ApprovalRequest, ApprovalVote, OversightDesk
from .packs import Pack, Requirement, list_packs, load_pack
from .policy import (
    Decision,
    Match,
    Policy,
    PolicyEngine,
    PolicyRule,
    ResidencyConfig,
    TransparencyConfig,
)
from .records import Ed25519Signer, HMACSigner, RetentionSweeper, SubjectVault
from .residency import Region, RegionResolver
from .risk import RiskAssessment, assess, impact_assessment
from .transparency import ProvenanceManifest

__all__ = [
    "Governance", "Mode",
    "Policy", "PolicyRule", "Match", "Decision", "PolicyEngine", "ResidencyConfig",
    "TransparencyConfig", "compile_expr", "ExpressionError",
    "AgentIdentity", "DelegationChain", "RiskTier",
    "DataClassifier", "Classification", "DataFinding", "Pseudonymizer",
    "DATA_CLASSES", "PERSONAL", "SPECIAL_CATEGORIES",
    "Region", "RegionResolver",
    "OversightDesk", "ApprovalRequest", "ApprovalVote",
    "HMACSigner", "Ed25519Signer", "SubjectVault", "RetentionSweeper",
    "ProvenanceManifest",
    "RiskAssessment", "assess", "impact_assessment",
    "AIInventory", "RuntimeMonitor", "RunState",
    "Incident", "IncidentDesk", "Deadline",
    "Pack", "Requirement", "load_pack", "list_packs",
    "CONTROLS", "ComplianceReport",
]
