"""Content rules: the patterns that get redacted, blocked or warned about."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

__all__ = ["Action", "Rule", "SECRET_RULES", "INJECTION_RULES"]


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
