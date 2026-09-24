"""Generation parameters: one vocabulary, validated once, planned per provider.

You set parameters in one vocabulary:

    temperature · top_p · top_k · min_p · frequency_penalty · presence_penalty
    repetition_penalty · seed · max_tokens · stop · effort · thinking · thinking_budget

and each provider turns that into what *it* accepts. That step is a
`ParameterPlan`: the values that will be sent, the ones dropped (with the
reason), and the ones adjusted to fit (with the reason). Nothing is dropped or
changed silently — `provider.explain(request)` shows the plan and the exact
payload, and every drop or adjustment is logged once on
``agent_harness.llm_providers``.

Two rules decide it:

- **A parameter a provider does not have is dropped, not translated.** Anthropic
  has no `seed`; inventing one would change what you asked for.
- **A value a provider would reject is fitted, not sent to fail.** Claude with
  extended thinking only takes `top_p` of 0.95–1, so 0.9 becomes 0.95; Gemini
  2.5 Pro cannot switch thinking off, so `effort="none"` becomes its minimum
  budget. The run keeps going, and the plan says what happened.

`effort` is the portable way to ask for reasoning depth. It becomes Anthropic's
`output_config.effort`, OpenAI's `reasoning_effort`, a Gemini 2.5 thinking
budget, a Gemini 3 thinking level, OpenRouter's `reasoning.effort` — each
mapped to the nearest level that model has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..errors import ConfigurationError

if TYPE_CHECKING:
    from .base import CompletionRequest

__all__ = [
    "Effort",
    "EFFORT_LEVELS",
    "SAMPLING_PARAMETERS",
    "GENERATION_PARAMETERS",
    "ParameterPlan",
    "validate_parameters",
    "nearest_effort",
]

#: Reasoning depth, shallowest first. `none` switches reasoning off where it can be.
Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
EFFORT_LEVELS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

#: The sampling knobs. A provider lists the ones it has in `Provider.parameters`.
SAMPLING_PARAMETERS: tuple[str, ...] = (
    "temperature", "top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty",
    "repetition_penalty", "seed",
)

#: Everything a ParameterPlan carries.
GENERATION_PARAMETERS: tuple[str, ...] = (
    *SAMPLING_PARAMETERS, "max_tokens", "stop", "effort", "thinking", "thinking_budget",
)

#: (lowest, highest, what it does) — the widest range any provider accepts.
RANGES: dict[str, tuple[float | None, float | None, str]] = {
    "temperature": (0.0, 2.0, "randomness: 0 is most deterministic"),
    "top_p": (0.0, 1.0, "nucleus sampling: the probability mass to sample from"),
    "top_k": (1, None, "sample from the k most likely tokens"),
    "min_p": (0.0, 1.0, "drop tokens below this fraction of the top token's probability"),
    "frequency_penalty": (-2.0, 2.0, "penalise tokens by how often they have appeared"),
    "presence_penalty": (-2.0, 2.0, "penalise tokens that have appeared at all"),
    "repetition_penalty": (0.0, None, "multiplicative repetition penalty; 1 is off"),
    "max_tokens": (1, None, "the longest answer, in tokens"),
    "thinking_budget": (-1, None, "tokens the model may spend thinking; -1 lets Gemini "
                                  "decide"),
}


def validate_parameters(**values: Any) -> dict[str, Any]:
    """Check generation parameters up front. Returns the ones that are set.

    Every problem is reported at once, so a misconfigured agent fails when it
    is built — not half-way through a run, after it has spent money.
    """
    problems: list[str] = []
    out: dict[str, Any] = {}
    for name, value in values.items():
        if value is None:
            continue
        if name == "effort":
            if value not in EFFORT_LEVELS:
                problems.append(f"effort={value!r}: use one of {', '.join(EFFORT_LEVELS)}")
        elif name == "thinking":
            if not isinstance(value, bool):
                problems.append(f"thinking={value!r}: must be True or False")
        elif name == "stop":
            if isinstance(value, str) or not all(isinstance(s, str) for s in value):
                problems.append("stop: must be a list of strings")
        elif name in RANGES:
            low, high, meaning = RANGES[name]
            integral = name in ("top_k", "max_tokens", "thinking_budget")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (
                    integral and not isinstance(value, int)):
                kind = "a whole number" if integral else "a number"
                problems.append(f"{name}={value!r}: must be {kind}")
                continue
            too_low = low is not None and (value <= low if name == "repetition_penalty"
                                           else value < low)
            if too_low or (high is not None and value > high):
                span = (f"between {low} and {high}" if high is not None
                        else f"above {low}" if name == "repetition_penalty"
                        else f"at least {low}")
                problems.append(f"{name}={value!r}: must be {span} ({meaning})")
        elif name == "seed":
            if isinstance(value, bool) or not isinstance(value, int):
                problems.append(f"seed={value!r}: must be a whole number")
        out[name] = value
    if problems:
        raise ConfigurationError("invalid generation parameters — " + "; ".join(problems))
    return out


def nearest_effort(level: str, supported: tuple[str, ...]) -> str:
    """The closest level a model has: the deepest one not beyond what was asked,
    or failing that the shallowest it offers."""
    rank = {name: i for i, name in enumerate(EFFORT_LEVELS)}
    ordered = sorted((s for s in supported if s in rank), key=rank.__getitem__)
    if not ordered:
        return level
    wanted = rank.get(level, rank["medium"])
    below = [s for s in ordered if rank[s] <= wanted]
    return below[-1] if below else ordered[0]


@dataclass
class ParameterPlan:
    """What one provider will do with one request's generation parameters."""

    provider: str
    model: str
    values: dict[str, Any] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)
    adjusted: dict[str, str] = field(default_factory=dict)

    @classmethod
    def of(cls, provider: str, req: CompletionRequest) -> ParameterPlan:
        values = {name: getattr(req, name, None) for name in GENERATION_PARAMETERS}
        values = {k: v for k, v in values.items() if v is not None and v != []}
        return cls(provider=provider, model=req.model, values=values)

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def drop(self, name: str, why: str) -> None:
        if name in self.values:
            self.values.pop(name)
            self.dropped[name] = why

    def adjust(self, name: str, value: Any, why: str) -> None:
        before = self.values.get(name)
        if before == value:
            return
        self.values[name] = value
        self.adjusted[name] = f"{before!r} → {value!r}: {why}"

    def set(self, name: str, value: Any) -> None:
        """A value the provider derived, not one that was asked for differently."""
        self.values[name] = value

    def as_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "sent": dict(self.values),
                "dropped": dict(self.dropped), "adjusted": dict(self.adjusted)}
