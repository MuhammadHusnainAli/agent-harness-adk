"""The content engine: runs the rules over anything entering or leaving an agent."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..errors import GuardrailTripped
from .rules import INJECTION_RULES, SECRET_RULES, Rule

__all__ = ["Guardrails"]


class Guardrails:
    """A rule set plus the size cap. Runs in microseconds; put it on every path."""

    def __init__(
        self,
        rules: Iterable[Rule] | None = None,
        *,
        secrets: bool = True,
        injection: bool = True,
        max_chars: int = 500_000,
        blocklist: Iterable[str] = (),
        strict: bool = False,
    ) -> None:
        self.rules: list[Rule] = list(rules or [])
        if secrets:
            self.rules.extend(SECRET_RULES)
        if injection:
            self.rules.extend(INJECTION_RULES)
        self.rules.extend(
            Rule(f"blocked:{term}", re.escape(term), "block") for term in blocklist
        )
        self.max_chars = max_chars
        self.strict = strict  # in strict mode a `warn` becomes a `block`
        self.triggered: list[tuple[str, str]] = []

    def check(self, text: str, *, where: str = "output", label: str = "") -> str:
        """Return the (possibly redacted) text, or raise GuardrailTripped."""
        if not text:
            return text
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + "\n... [truncated by guardrail]"

        for rule in self.rules:
            if not rule.applies(where):
                continue
            assert isinstance(rule.pattern, re.Pattern)
            if not rule.pattern.search(text):
                continue
            if rule.check and not rule.check(text):
                continue
            self.triggered.append((rule.name, where))
            action = "block" if (rule.action == "warn" and self.strict) else rule.action
            if action == "block":
                raise GuardrailTripped(
                    f"guardrail {rule.name!r} blocked this {where}"
                    + (f" ({label})" if label else ""),
                    rule=rule.name, where=where,
                )
            if action == "redact":
                text = rule.pattern.sub(rule.replacement, text)
        return text

    def safe(self, text: str, *, where: str = "output") -> tuple[bool, str]:
        """Non-raising form: (ok, text_or_reason)."""
        try:
            return True, self.check(text, where=where)
        except GuardrailTripped as exc:
            return False, str(exc)

    def add(self, rule: Rule) -> None:
        self.rules.append(rule)

    def report(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name, _ in self.triggered:
            counts[name] = counts.get(name, 0) + 1
        return counts
