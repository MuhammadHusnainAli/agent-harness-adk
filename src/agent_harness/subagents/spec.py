"""The blueprint a sub-agent is built from: serialisable, versionable."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..runtime.budget import Budget

__all__ = ["SubAgentSpec"]


_STOPWORDS = {
    "with", "from", "this", "that", "into", "your", "their", "them", "please",
    "would", "should", "about", "when", "what", "which", "where", "have", "been",
    "make", "made", "will", "must", "such", "than", "then", "they", "also", "each",
    "only", "very", "just", "over", "more", "most", "some", "here", "there", "were",
    "does", "done", "need", "needs", "want", "wants", "give", "take", "using", "use",
}


def _tokens(text: str) -> set[str]:
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {t for t in cleaned.split() if t}


def _matches_any(word: str, hay: set[str], *, prefix: int = 5) -> bool:
    if word in hay:
        return True
    head = word[:prefix]
    return any(token.startswith(head) or word.startswith(token[:prefix])
               for token in hay if len(token) >= 4)


class SubAgentSpec(BaseModel):
    """The blueprint a sub-agent is built from. Serialisable, versionable."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    instructions: str = ""
    tools: list[str] | None = None          # glob allowlist; None inherits the parent's
    skills: list[str] | None = None
    model: str | None = None
    tier: str | None = None
    effort: str | None = None
    max_steps: int = 12
    max_tokens: int = 8192
    temperature: float | None = None
    output_schema: dict[str, Any] | None = None
    workspace: Literal["isolated", "shared", "none"] = "none"
    allow_shell: bool = False
    allow: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    # What this sub-agent must do before it may call itself done. The keyword
    # form of AgentGuardrails, so a spec stays serialisable:
    #   {"require_tools": ["lookup"], "must_include": ["source"]}
    guardrails: dict[str, Any] | None = None
    budget: Budget | None = None
    memory: bool = False                     # sub-agents start clean by design
    version: str = "1"
    origin: Literal["bench", "factory"] = "bench"
    tags: list[str] = Field(default_factory=list)

    def matches(self, task: str) -> float:
        """A cheap capability score used when no embedder is configured.

        Token-level with a shared-prefix rule, so "validate" finds the validator
        but a stopword like "with" never staffs anybody.
        """
        hay = _tokens(f"{self.name} {self.description} {' '.join(self.tags)}")
        words = _tokens(task) - _STOPWORDS
        words = {w for w in words if len(w) > 3}
        if not words:
            return 0.0
        hits = sum(1 for w in words if _matches_any(w, hay))
        return hits / len(words)
