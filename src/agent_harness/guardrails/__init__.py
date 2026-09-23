"""Guardrails, in two halves.

**Content** — `Guardrails` runs pattern rules over anything entering or leaving
an agent: API keys redacted, private keys blocked, injection attempts flagged,
oversized payloads trimmed. It is harness-wide by default.

**Behaviour** — `AgentGuardrails` is per agent: which tools it may touch, and
what has to be true of its answer before that answer is accepted. A failed check
sends the agent back round with a plain-English note about what is missing,
rather than failing the run.

    from agent_harness.guardrails import AgentGuardrails, RequireTools, MaxCost

    AgentGuardrails(RequireTools("order_status"), MaxCost(0.25),
                    forbid_tools=["issue_refund"], no_placeholders=True)
"""

from .agent import AgentGuardrails, OnViolation
from .checks import (
    Check,
    CompletionContext,
    Custom,
    ForbidTools,
    MaxCost,
    MaxSteps,
    MinLength,
    MustInclude,
    MustMatch,
    MustNotInclude,
    NoPlaceholders,
    RequireCitation,
    RequireJSON,
    RequireStructured,
    RequireTools,
    Violation,
    as_checks,
)
from .engine import Guardrails
from .rules import INJECTION_RULES, SECRET_RULES, Action, Rule

__all__ = [
    # content
    "Guardrails",
    "Rule",
    "Action",
    "SECRET_RULES",
    "INJECTION_RULES",
    # behaviour
    "AgentGuardrails",
    "OnViolation",
    "CompletionContext",
    "Violation",
    "Check",
    "as_checks",
    # checks
    "RequireTools",
    "ForbidTools",
    "MustInclude",
    "MustNotInclude",
    "MustMatch",
    "MinLength",
    "MaxSteps",
    "MaxCost",
    "RequireCitation",
    "RequireJSON",
    "RequireStructured",
    "NoPlaceholders",
    "Custom",
]
