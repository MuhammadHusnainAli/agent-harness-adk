"""Per-agent guardrails: what this agent must do, and when it may say it is done.

The content engine (`Guardrails`) polices *text* — secrets, injection, size.
This polices *behaviour*: which tools an agent may touch, and what has to be
true before its answer is accepted.

    support = Agent(
        "support",
        "Answer order questions.",
        tools=[order_status, issue_refund],
        guardrails=AgentGuardrails(
            require_tools=["order_status"],   # look it up, never guess
            forbid_tools=["issue_refund"],    # not this agent's job
            must_include=["order"],
            no_placeholders=True,
            on_violation="retry",             # tell it what is missing, let it fix it
        ),
    )

A failed check is not a crash. The agent is told, in words, what is missing and
gets another turn — which is usually all it needs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal

from ..errors import ConfigurationError
from .checks import (
    Check,
    CompletionContext,
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

__all__ = ["AgentGuardrails", "OnViolation"]

OnViolation = Literal["retry", "fail", "warn"]


class AgentGuardrails:
    """One object holding everything this agent is and is not allowed to do."""

    def __init__(
        self,
        *checks: Any,
        require_tools: Sequence[str] = (),
        forbid_tools: Sequence[str] = (),
        allow_tools: Sequence[str] | None = None,
        must_include: Sequence[str] = (),
        must_not_include: Sequence[str] = (),
        must_match: str | None = None,
        min_output_chars: int | None = None,
        max_steps: int | None = None,
        max_cost_usd: float | None = None,
        require_citation: bool = False,
        require_json: bool = False,
        require_structured: bool = False,
        no_placeholders: bool = False,
        content: Guardrails | None = None,
        on_violation: OnViolation = "retry",
        max_retries: int = 2,
    ) -> None:
        if on_violation not in ("retry", "fail", "warn"):
            raise ConfigurationError(
                f"on_violation must be retry, fail or warn — got {on_violation!r}"
            )
        if max_retries < 0:
            raise ConfigurationError("max_retries cannot be negative")

        self.on_violation: OnViolation = on_violation
        self.max_retries = max_retries
        self.content = content

        # Tool access is enforced before a call, not only judged after one.
        self.forbidden = list(forbid_tools)
        self.allowed = list(allow_tools) if allow_tools is not None else None

        built: list[Check] = list(as_checks(checks))
        if require_tools:
            built.append(RequireTools(*require_tools))
        # `forbid_tools` is enforced before the call, so it deliberately does not
        # also become a completion check — the tool never ran, and penalising the
        # answer for a call that was already blocked just confuses the agent.
        # `ForbidTools(...)` is still available as an explicit after-the-fact check.
        if must_include:
            built.append(MustInclude(*must_include))
        if must_not_include:
            built.append(MustNotInclude(*must_not_include))
        if must_match:
            built.append(MustMatch(must_match))
        if min_output_chars:
            built.append(MinLength(min_output_chars))
        if max_steps:
            built.append(MaxSteps(max_steps))
        if max_cost_usd:
            built.append(MaxCost(max_cost_usd))
        if require_citation:
            built.append(RequireCitation())
        if require_json:
            built.append(RequireJSON())
        if require_structured:
            built.append(RequireStructured())
        if no_placeholders:
            built.append(NoPlaceholders())
        self.checks: list[Check] = built
        self.violations: list[Violation] = []

    # ---- before a tool runs ------------------------------------------------
    def tool_allowed(self, name: str) -> tuple[bool, str]:
        """Checked before the call, so a forbidden tool never actually runs."""
        if name in self.forbidden:
            return False, f"{name} is not permitted for this agent"
        if self.allowed is not None and name not in self.allowed:
            permitted = ", ".join(self.allowed) or "none"
            return False, (f"{name} is not on this agent's allowlist "
                           f"(permitted: {permitted})")
        return True, ""

    # ---- before the answer is accepted --------------------------------------
    def check(self, ctx: CompletionContext) -> list[Violation]:
        """Run every check. Returns what is unmet, in the order declared."""
        found: list[Violation] = []
        for check in self.checks:
            try:
                violation = check(ctx)
            except Exception as exc:   # a broken check must not break the run
                violation = Violation(getattr(check, "name", "check"),
                                      f"the check itself failed: {exc}")
            if violation is not None:
                found.append(violation)
        self.violations.extend(found)
        return found

    @staticmethod
    def feedback(violations: Iterable[Violation]) -> str:
        """The message the agent is given so it can put the answer right."""
        rows = list(violations)
        lead = ("That answer does not meet this task's requirements yet:"
                if len(rows) > 1 else "That answer does not meet a requirement:")
        body = "\n".join(f"- {v.line()}" for v in rows)
        return f"{lead}\n{body}\n\nPut it right and answer again."

    # ---- plumbing -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.checks)

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        names = ", ".join(getattr(c, "name", "check") for c in self.checks)
        return f"<AgentGuardrails {names or 'empty'} on_violation={self.on_violation}>"

    def report(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for violation in self.violations:
            counts[violation.check] = counts.get(violation.check, 0) + 1
        return {"checks": len(self.checks), "violations": counts,
                "on_violation": self.on_violation}

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> AgentGuardrails | None:
        """Build from the serialisable form a `SubAgentSpec` can carry."""
        if not config:
            return None
        return cls(**config)
