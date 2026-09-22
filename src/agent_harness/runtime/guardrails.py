"""Guardrails: injection, data-leak and output checks. The blast-radius limiter.

Checks run on the way in (tool results, retrieved documents, user input) and on
the way out (the agent's answer). A rule either redacts or blocks.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from ..errors import GuardrailTripped

__all__ = ["Rule", "Guardrails", "SECRET_RULES", "INJECTION_RULES"]

Action = Literal["block", "redact", "warn"]


@dataclass
class Rule:
    """A named pattern and what to do when it matches."""

    name: str
    pattern: re.Pattern[str] | str
    action: Action = "redact"
    replacement: str = "[redacted]"
    where: tuple[str, ...] = ("input", "output")
    check: Callable[[str], bool] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.pattern, str):
            self.pattern = re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)

    def applies(self, where: str) -> bool:
        return where in self.where


SECRET_RULES: list[Rule] = [
    Rule("anthropic_key", r"sk-ant-[A-Za-z0-9_\-]{16,}", "redact"),
    Rule("openai_key", r"\bsk-(?!ant-)[A-Za-z0-9]{20,}\b", "redact"),
    Rule("google_key", r"\bAIza[0-9A-Za-z_\-]{30,}\b", "redact"),
    Rule("aws_key", r"\bAKIA[0-9A-Z]{16}\b", "redact"),
    Rule("github_token", r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "redact"),
    Rule("private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "block"),
    Rule("bearer_token", r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}=*", "redact"),
]

INJECTION_RULES: list[Rule] = [
    Rule("ignore_instructions",
         r"ignore (?:all |any )?(?:previous|prior|above) instructions", "warn",
         where=("input",)),
    Rule("system_prompt_exfil",
         r"(?:reveal|print|repeat|show)(?: me)? (?:your |the )?system prompt", "warn",
         where=("input",)),
    Rule("role_override", r"you are now (?:a|an|the) [a-z ]{3,40}(?:\.|,|\n)", "warn",
         where=("input",)),
]


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
