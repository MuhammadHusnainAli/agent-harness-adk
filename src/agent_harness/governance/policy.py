"""Policy as code: declarative rules, compiled once, decided in microseconds.

    rules:
      - id: refunds-need-a-human
        on: tool
        match: {tags: [payments]}
        when: "args.amount > 100"
        effect: require_approval
        approvers: 2
        timeout: 15m

Every rule that matches an action contributes; the strictest effect wins
(``deny`` > ``require_approval`` > ``reroute`` > ``redact`` > ``log`` >
``allow``) and the obligations of all of them are kept — two rules asking for
different data classes to be redacted both get their way. That is the
"deny-overrides" combination auditors expect, and it means adding a rule can
only ever make a policy stricter, never quietly loosen it.

A rule whose condition cannot be evaluated takes ``defaults.on_error``
(``deny`` unless you say otherwise). A policy engine that fails open is not a
control.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable, Mapping
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from ..errors import ConfigurationError
from .expr import Evaluator, compile_expr

__all__ = [
    "Action", "ACTIONS", "Effect", "STRICTNESS", "Match", "PolicyRule", "PolicyDefaults",
    "ResidencyConfig", "TransparencyConfig", "Policy", "Decision", "PolicyEngine",
    "parse_duration",
]

Action = Literal["run", "tool", "tool_result", "egress", "delegate", "memory", "output"]
ACTIONS: tuple[str, ...] = ("run", "tool", "tool_result", "egress", "delegate", "memory",
                            "output")

Effect = Literal["allow", "log", "redact", "reroute", "require_approval", "deny"]
STRICTNESS: dict[str, int] = {"allow": 0, "log": 1, "redact": 2, "reroute": 3,
                              "require_approval": 4, "deny": 5}

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)?\s*$")
_UNITS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, None: 1}


def parse_duration(value: float | int | str | None) -> float | None:
    """`90`, `"90s"`, `"15m"`, `"72h"`, `"30d"` → seconds."""
    if value is None or isinstance(value, (int, float)):
        return None if value is None else float(value)
    match = _DURATION.match(value)
    if not match:
        raise ConfigurationError(f"not a duration: {value!r} (try '15m', '72h', '30d')")
    return float(match.group(1)) * _UNITS[match.group(2)]


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


class Match(BaseModel):
    """Cheap pre-filters, checked before any condition. Globs; any-of per field.

    An empty field matches everything. `data` matches when any of the listed
    data classes is present in what is being decided about.
    """

    model_config = ConfigDict(extra="forbid")

    tool: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    agent: list[str] = Field(default_factory=list)
    provider: list[str] = Field(default_factory=list)
    model: list[str] = Field(default_factory=list)
    region: list[str] = Field(default_factory=list)
    data: list[str] = Field(default_factory=list)
    risk: list[str] = Field(default_factory=list)
    purpose: list[str] = Field(default_factory=list)
    jurisdiction: list[str] = Field(default_factory=list)

    @field_validator("*", mode="before")
    @classmethod
    def _as_list(cls, value: Any) -> list[str]:
        return _listify(value)

    @property
    def empty(self) -> bool:
        return not any(getattr(self, f) for f in type(self).model_fields)

    def matches(self, ctx: Mapping[str, Any]) -> bool:
        def one(patterns: list[str], value: Any) -> bool:
            if not patterns:
                return True
            if value is None:
                return False
            return any(fnmatch(str(value), p) for p in patterns)

        tool = ctx.get("tool") or {}
        if not one(self.tool, tool.get("name")):
            return False
        if self.tags and not set(self.tags) & set(tool.get("tags") or ()):
            return False
        agent = ctx.get("agent") or {}
        if not one(self.agent, agent.get("name")):
            return False
        if not one(self.risk, agent.get("risk")):
            return False
        if not one(self.purpose, ctx.get("purpose")):
            return False
        provider = ctx.get("provider") or {}
        if not one(self.provider, provider.get("name")):
            return False
        if not one(self.model, provider.get("model")):
            return False
        if not one(self.region, provider.get("region")):
            return False
        if not one(self.jurisdiction, (ctx.get("principal") or {}).get("jurisdiction")):
            return False
        if self.data:
            present = set((ctx.get("data") or {}).get("classes") or ())
            if not present & set(self.data):
                return False
        return True


class PolicyRule(BaseModel):
    """One rule. `on` says which actions it looks at."""

    model_config = ConfigDict(extra="forbid")

    id: str
    on: list[Action] = Field(default_factory=lambda: ["tool"])
    match: Match = Field(default_factory=Match)
    when: str | None = None
    effect: Effect = "deny"
    reason: str = ""
    #: require_approval: distinct humans needed, and how long to wait for them.
    approvers: int = Field(1, ge=1)
    timeout: float | None = None
    #: redact: which data classes to take out, and how.
    redact: list[str] = Field(default_factory=list)
    redact_mode: Literal["mask", "pseudonymize"] = "mask"
    #: Which catalog controls this rule implements, and which pack it came from.
    controls: list[str] = Field(default_factory=list)
    source: str = ""

    _test: Evaluator | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _yaml_on(cls, data: Any) -> Any:
        # YAML 1.1 reads a bare `on:` key as the boolean True (the "Norway
        # problem"). Everyone writes `on: tool`, so take it as meant.
        if isinstance(data, Mapping) and True in data:
            data = {("on" if k is True else k): v for k, v in data.items()}
        return data

    @field_validator("on", mode="before")
    @classmethod
    def _on(cls, value: Any) -> list[str]:
        return _listify(value)

    @field_validator("timeout", mode="before")
    @classmethod
    def _timeout(cls, value: Any) -> float | None:
        return parse_duration(value)

    def model_post_init(self, __context: Any) -> None:
        self._test = compile_expr(self.when) if self.when else None

    def evaluate(self, ctx: Mapping[str, Any]) -> bool:
        """True when this rule applies. Raises if its condition blows up."""
        if not self.match.matches(ctx):
            return False
        return True if self._test is None else bool(self._test(ctx))


class PolicyDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effect: Effect = "allow"
    on_error: Effect = "deny"


class ResidencyConfig(BaseModel):
    """Where data about people from each jurisdiction may be sent.

    `transfers` maps a data subject's jurisdiction to the destinations their
    personal data may reach; ``"*"`` means anywhere. The subject's jurisdiction
    comes from the run's principal, or `home` when none is set. A jurisdiction
    no pack speaks for is unrestricted.
    """

    model_config = ConfigDict(extra="forbid")

    home: str | None = None
    transfers: dict[str, list[str]] = Field(default_factory=dict)
    #: Which data triggers the restriction. Non-personal data goes anywhere.
    applies_to: list[str] = Field(default_factory=lambda: ["personal"])
    #: A self-hosted model never leaves the building.
    on_prem_always_allowed: bool = True
    #: A destination whose region cannot be resolved.
    unknown: Literal["deny", "allow"] = "deny"

    def merge(self, other: ResidencyConfig | None) -> ResidencyConfig:
        """Combine two packs' views. Where both restrict a jurisdiction, both must agree."""
        if other is None:
            return self
        transfers = {k: list(v) for k, v in self.transfers.items()}
        for origin, allowed in other.transfers.items():
            if origin not in transfers:
                transfers[origin] = list(allowed)
            elif "*" in transfers[origin]:
                transfers[origin] = list(allowed)
            elif "*" not in allowed:
                transfers[origin] = [d for d in transfers[origin] if d in allowed]
        return ResidencyConfig(
            home=self.home or other.home,
            transfers=transfers,
            applies_to=sorted(set(self.applies_to) | set(other.applies_to)),
            on_prem_always_allowed=(self.on_prem_always_allowed
                                    and other.on_prem_always_allowed),
            unknown="deny" if "deny" in (self.unknown, other.unknown) else "allow",
        )

    def destinations(self, origin: str | None) -> list[str] | None:
        """Where `origin`'s personal data may go. None means unrestricted."""
        if origin is None or origin not in self.transfers:
            return None
        allowed = self.transfers[origin]
        if "*" in allowed:
            return None
        return sorted({origin, *allowed})


class TransparencyConfig(BaseModel):
    """How people are told they are dealing with AI, and how output is marked."""

    model_config = ConfigDict(extra="forbid")

    #: Attach a disclosure to every top-level result (`result.disclosure`).
    disclose: bool = False
    #: Also put a visible label into the output text itself...
    label_output: bool = False
    #: ...for people in these jurisdictions (``"*"``: everyone). A pack that
    #: requires visible labels in China does not label answers to Germans.
    label_for: list[str] = Field(default_factory=lambda: ["*"])
    #: Attach a machine-readable provenance manifest (`result.provenance`).
    provenance: bool = False
    languages: list[str] = Field(default_factory=lambda: ["en"])

    def merge(self, other: TransparencyConfig | None) -> TransparencyConfig:
        if other is None:
            return self
        label_for = sorted({j for c in (self, other) if c.label_output
                            for j in c.label_for}) or ["*"]
        return TransparencyConfig(
            disclose=self.disclose or other.disclose,
            label_output=self.label_output or other.label_output,
            label_for=["*"] if "*" in label_for else label_for,
            provenance=self.provenance or other.provenance,
            languages=list(dict.fromkeys([*self.languages, *other.languages])),
        )


class Policy(BaseModel):
    """A complete, versioned policy. Packs are merged into one of these."""

    model_config = ConfigDict(extra="forbid")

    name: str = "policy"
    version: str = "1"
    description: str = ""
    defaults: PolicyDefaults = Field(default_factory=PolicyDefaults)
    rules: list[PolicyRule] = Field(default_factory=list)
    #: purpose → the data classes a run with that purpose may handle.
    purposes: dict[str, list[str]] = Field(default_factory=dict)
    residency: ResidencyConfig | None = None
    transparency: TransparencyConfig = Field(default_factory=TransparencyConfig)
    #: Minimum days decision records must be kept (the longest any pack asks for).
    log_retention_days: int | None = None
    #: Maximum days conversational data may be kept (the shortest any pack allows).
    data_retention_days: int | None = None
    #: Highest delegation depth any agent may reach.
    max_delegation_depth: int | None = None
    #: A tool whose schema changed since it was pinned: deny, log, or allow.
    tool_drift: Literal["deny", "log", "allow"] = "log"
    #: Where the rules came from, for the evidence report.
    packs: list[str] = Field(default_factory=list)

    @field_validator("rules", mode="before")
    @classmethod
    def _rules(cls, value: Any) -> Any:
        return value or []

    # ---- identity -----------------------------------------------------------
    @property
    def hash(self) -> str:
        """sha256 over the canonical JSON — every decision records it."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True,
                             separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    # ---- loading ------------------------------------------------------------
    @classmethod
    def load(cls, source: str | Path | Mapping[str, Any] | Policy | None) -> Policy:
        """A path to YAML/JSON, a YAML string, a dict, or a Policy."""
        if source is None:
            return cls()
        if isinstance(source, Policy):
            return source
        if isinstance(source, Mapping):
            return cls.model_validate(dict(source))
        text = str(source)
        path = Path(text)
        if "\n" not in text and path.suffix.lower() in {".yaml", ".yml", ".json"}:
            if not path.exists():
                raise ConfigurationError(f"policy file not found: {path}")
            text = path.read_text(encoding="utf-8")
        import yaml

        data = yaml.safe_load(text) or {}
        if not isinstance(data, Mapping):
            raise ConfigurationError("a policy must be a mapping at the top level")
        return cls.model_validate(dict(data))

    def merge(self, *others: Policy) -> Policy:
        """This policy plus others. Every combination only tightens."""
        merged = self.model_copy(deep=True)
        for other in others:
            merged.rules = [*merged.rules, *other.rules]
            for purpose, classes in other.purposes.items():
                if purpose in merged.purposes:
                    merged.purposes[purpose] = [c for c in merged.purposes[purpose]
                                                if c in classes]
                else:
                    merged.purposes[purpose] = list(classes)
            merged.residency = (other.residency if merged.residency is None
                                else merged.residency.merge(other.residency))
            merged.transparency = merged.transparency.merge(other.transparency)
            merged.log_retention_days = _pick(max, merged.log_retention_days,
                                              other.log_retention_days)
            merged.data_retention_days = _pick(min, merged.data_retention_days,
                                               other.data_retention_days)
            merged.max_delegation_depth = _pick(min, merged.max_delegation_depth,
                                                other.max_delegation_depth)
            order = ("allow", "log", "deny")
            merged.tool_drift = max(merged.tool_drift, other.tool_drift, key=order.index)
            if STRICTNESS[other.defaults.effect] > STRICTNESS[merged.defaults.effect]:
                merged.defaults.effect = other.defaults.effect
            merged.packs = list(dict.fromkeys([*merged.packs, *other.packs]))
        return merged


def _pick(fn: Any, a: int | None, b: int | None) -> int | None:
    values = [v for v in (a, b) if v is not None]
    return fn(values) if values else None


class Decision(BaseModel):
    """What the engine decided, why, and under which policy version."""

    action: str
    target: str = ""
    effect: Effect = "allow"
    rules: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    policy: str = ""                    # the policy hash, shortened
    approvers: int = 1
    timeout: float | None = None
    redact: list[str] = Field(default_factory=list)
    redact_mode: Literal["mask", "pseudonymize"] = "mask"
    enforced: bool = True               # False in monitor mode
    duration_us: float = 0.0

    @property
    def allowed(self) -> bool:
        return self.effect in ("allow", "log", "redact")

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons) or self.effect

    def tighten(self, effect: Effect, reason: str, *, rule: str = "",
                controls: Iterable[str] = ()) -> Decision:
        """Fold a built-in check (residency, delegation, drift) into the decision."""
        if STRICTNESS[effect] > STRICTNESS[self.effect]:
            self.effect = effect
        if rule:
            self.rules.append(rule)
        if reason:
            self.reasons.append(reason)
        self.controls = list(dict.fromkeys([*self.controls, *controls]))
        return self


class PolicyEngine:
    """A compiled policy: rules indexed by action, evaluated in order."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.hash = policy.hash
        self._by_action: dict[str, list[PolicyRule]] = {a: [] for a in ACTIONS}
        for rule in policy.rules:
            for action in rule.on:
                self._by_action[action].append(rule)

    def rules_for(self, action: str) -> list[PolicyRule]:
        return self._by_action.get(action, [])

    def decide(self, action: str, ctx: Mapping[str, Any], *, target: str = "") -> Decision:
        started = time.perf_counter()
        # Starts at the loosest effect so a matching `allow` rule can carve an
        # exception out of a deny-by-default policy; the default applies only
        # when nothing matched.
        decision = Decision(action=action, target=target, effect="allow",
                            policy=self.hash[:16])
        matched = False
        for rule in self._by_action.get(action, ()):
            try:
                applies = rule.evaluate(ctx)
            except Exception as exc:  # a broken condition takes on_error
                matched = True
                decision.tighten(self.policy.defaults.on_error,
                                 f"rule {rule.id} could not be evaluated ({exc})",
                                 rule=rule.id, controls=rule.controls)
                continue
            if not applies:
                continue
            # A `log` rule observes; it does not stand in for the default.
            matched = matched or rule.effect != "log"
            decision.tighten(rule.effect, rule.reason or f"rule {rule.id}",
                             rule=rule.id, controls=rule.controls)
            if rule.effect == "require_approval":
                decision.approvers = max(decision.approvers, rule.approvers)
                if rule.timeout is not None:
                    decision.timeout = (rule.timeout if decision.timeout is None
                                        else min(decision.timeout, rule.timeout))
            if rule.effect == "redact" or rule.redact:
                decision.redact = list(dict.fromkeys([*decision.redact, *rule.redact]))
                if rule.redact_mode == "pseudonymize":
                    decision.redact_mode = "pseudonymize"
        if not matched:
            default = self.policy.defaults.effect
            if STRICTNESS[default] > STRICTNESS[decision.effect]:
                decision.effect = default
            if default != "allow":
                decision.reasons.append(f"policy default is {decision.effect}")
        decision.duration_us = (time.perf_counter() - started) * 1e6
        return decision
