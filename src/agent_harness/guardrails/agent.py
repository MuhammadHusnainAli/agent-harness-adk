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

import asyncio
import inspect
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
from .standard import (
    Grounded,
    NoInjection,
    NoPII,
    NoRepetition,
    NoSecrets,
    NotToxic,
)

__all__ = ["AgentGuardrails", "OnViolation"]

OnViolation = Literal["retry", "fail", "warn"]


def _needs_await(check: Any) -> bool:
    """Does running this check return a coroutine?

    A plain `async def` is one case; an object whose `__call__` is async — every
    `LLMGuard` — is the other. `iscoroutinefunction` alone misses the second.
    """
    if inspect.iscoroutinefunction(check):
        return True
    call = vars(type(check)).get("__call__")
    for klass in type(check).__mro__ if call is None else ():
        call = vars(klass).get("__call__")
        if call is not None:
            break
    return call is not None and inspect.iscoroutinefunction(call)


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
        no_pii: bool = False,
        no_secrets: bool = False,
        no_injection: bool = False,
        not_toxic: bool = False,
        no_repetition: bool = False,
        grounded: float | bool = False,
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
        if no_pii:
            built.append(NoPII())
        if no_secrets:
            built.append(NoSecrets())
        if no_injection:
            built.append(NoInjection())
        if not_toxic:
            built.append(NotToxic())
        if no_repetition:
            built.append(NoRepetition())
        if grounded:
            built.append(Grounded(0.6 if grounded is True else float(grounded)))
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
    @property
    def has_async_checks(self) -> bool:
        """True when a check must be awaited — an LLM judge, typically."""
        return any(_needs_await(c) for c in self.checks)

    def check(self, ctx: CompletionContext) -> list[Violation]:
        """Run the synchronous checks. Raises if an async one is present."""
        if self.has_async_checks:
            raise ConfigurationError(
                "these guardrails include a check that must be awaited (an LLM "
                "guard, most likely) — call `await rails.check_async(ctx)`")
        return self._collect([(c, self._safely(c, ctx)) for c in self.checks])

    async def check_async(self, ctx: CompletionContext) -> list[Violation]:
        """Run every check, awaiting the ones that need it.

        Sync checks run first, in order. If any of them already objects, the
        async ones are skipped — there is no reason to pay a model to confirm
        what a regex just proved.
        """
        sync: list[Check] = []
        awaited: list[Check] = []
        for check in self.checks:
            (awaited if _needs_await(check) else sync).append(check)

        found = self._collect([(c, self._safely(c, ctx)) for c in sync])
        if found or not awaited:
            return found

        results = await asyncio.gather(
            *(self._safely_async(c, ctx) for c in awaited))
        return self._collect(list(zip(awaited, results, strict=True)))

    @staticmethod
    def _safely(check: Check, ctx: CompletionContext) -> Any:
        try:
            return check(ctx)
        except Exception as exc:   # a broken check must not break the run
            return Violation(getattr(check, "name", "check"),
                             f"the check itself failed: {exc}")

    @staticmethod
    async def _safely_async(check: Check, ctx: CompletionContext) -> Any:
        try:
            return await check(ctx)
        except Exception as exc:
            return Violation(getattr(check, "name", "check"),
                             f"the check itself failed: {exc}")

    def _collect(self, pairs: list[tuple[Check, Any]]) -> list[Violation]:
        found = [v for _, v in pairs if isinstance(v, Violation)]
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
