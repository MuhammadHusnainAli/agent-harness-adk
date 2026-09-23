"""Model router: the right model and effort tier for each task, not one model for all."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Literal

from ..providers.base import MODELS, model_info

__all__ = ["Effort", "RouteRule", "ModelRouter"]

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# A tier is a promise about capability, not a model id — so this stays portable.
DEFAULT_TIERS: dict[str, str] = {
    "fast": "claude-haiku-4-5",
    "balanced": "claude-sonnet-5",
    "deep": "claude-opus-5",
}

DEFAULT_EFFORT: dict[str, Effort] = {
    "fast": "low",
    "balanced": "medium",
    "deep": "high",
}


@dataclass
class RouteRule:
    """Match a task or agent name and pin it to a tier."""

    pattern: str
    tier: str = "balanced"
    model: str | None = None
    effort: Effort | None = None

    def matches(self, name: str) -> bool:
        return fnmatch(name.lower(), self.pattern.lower())


class ModelRouter:
    """Chooses `(model, effort)` per task. Explicit beats rule beats tier beats default."""

    def __init__(
        self,
        default: str | None = None,
        *,
        tiers: dict[str, str] | None = None,
        rules: Iterable[RouteRule] = (),
        effort: dict[str, Effort] | None = None,
        chooser: Callable[[str], str | None] | None = None,
        fallbacks: Iterable[str] = (),
        max_attempts: int | None = None,
    ) -> None:
        self.tiers = {**DEFAULT_TIERS, **(tiers or {})}
        self.default = default or self.tiers["balanced"]
        self.rules = list(rules)
        self.effort = {**DEFAULT_EFFORT, **(effort or {})}
        self.chooser = chooser
        #: Tried in order when the chosen model cannot be reached. A different
        #: provider is the useful fallback — a second Anthropic model does not
        #: help when Anthropic is the thing that is down.
        self.fallbacks = list(fallbacks)
        self.max_attempts = max_attempts

    def chain(self, model: str) -> list[str]:
        """The models to try, in order, starting with `model`."""
        seen, out = set(), []
        for candidate in [model, *self.fallbacks]:
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
        return out[: self.max_attempts] if self.max_attempts else out

    def pick(self, *, model: str | None = None, tier: str | None = None,
             task: str = "") -> tuple[str, Effort | None]:
        if model:
            return model, self._effort_for(model, tier)
        if self.chooser and task:
            chosen = self.chooser(task)
            if chosen:
                return chosen, self._effort_for(chosen, tier)
        if task:
            for rule in self.rules:
                if rule.matches(task):
                    picked = rule.model or self.tiers.get(rule.tier, self.default)
                    return picked, rule.effort or self.effort.get(rule.tier)
        if tier:
            return self.tiers.get(tier, self.default), self.effort.get(tier)
        return self.default, self._effort_for(self.default, tier)

    def _effort_for(self, model: str, tier: str | None) -> Effort | None:
        if tier:
            return self.effort.get(tier)
        info = model_info(model)
        return self.effort.get(info.tier) if info else None

    def tier_of(self, model: str) -> str:
        info = model_info(model)
        return info.tier if info else "balanced"

    def known_models(self) -> list[str]:
        return sorted(MODELS)
