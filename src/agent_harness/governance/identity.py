"""Who an agent is, who answers for it, and what it may do.

    AgentIdentity(owner="ops@acme.com", purpose="customer_support", risk="limited",
                  tools=["order_*", "issue_refund"], data=["contact", "financial"])

Delegation never widens authority. When an agent hands work to a sub-agent,
the sub-agent acts under the *intersection* of its own identity and every
identity above it — a research agent spawned by a support agent cannot reach a
tool the support agent could not. That is the least-privilege rule the
Singapore agentic framework and OWASP ASI03 (identity & privilege abuse) ask
for, enforced rather than recommended.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["RiskTier", "RISK_ORDER", "Autonomy", "AgentIdentity", "DelegationChain"]

RiskTier = Literal["minimal", "limited", "high", "prohibited"]
RISK_ORDER: dict[str, int] = {"minimal": 0, "limited": 1, "high": 2, "prohibited": 3}
Autonomy = Literal["assistive", "supervised", "autonomous"]


class AgentIdentity(BaseModel):
    """The accountable description of one agent.

    Left-out fields are unrestricted, not forbidden — but the evidence report
    counts an agent with no `owner` or `purpose` as a gap, because every
    framework asks who is accountable for it and what it is for.
    """

    model_config = ConfigDict(extra="allow")

    name: str = ""
    #: The accountable human or team — an email, a group, an employee id.
    owner: str = ""
    purpose: str = ""
    description: str = ""
    risk: RiskTier | None = None      # None: classify from `domains`
    #: Areas it decides or advises on: "employment", "credit", "health", ...
    #: These drive risk classification (see `governance.risk`).
    domains: list[str] = Field(default_factory=list)
    autonomy: Autonomy = "supervised"
    #: Talks to members of the public (drives disclosure duties).
    user_facing: bool = True

    #: Allowed tool name globs or tags (``tag:payments``). None: any tool.
    tools: list[str] | None = None
    #: Data classes it may handle. None: any.
    data: list[str] | None = None
    #: Jurisdictions its model calls may go to. None: whatever residency allows.
    regions: list[str] | None = None
    may_delegate: bool = True
    max_delegation_depth: int | None = None

    labels: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def of(cls, value: AgentIdentity | Mapping[str, Any] | None, *,
           name: str = "") -> AgentIdentity:
        if isinstance(value, AgentIdentity):
            identity = value
        elif isinstance(value, Mapping):
            identity = cls.model_validate(dict(value))
        else:
            identity = cls()
        if not identity.name:
            identity = identity.model_copy(update={"name": name})
        return identity

    @property
    def accountable(self) -> bool:
        return bool(self.owner and self.purpose)

    def allows_tool(self, tool: str, tags: Iterable[str] = ()) -> bool:
        if self.tools is None:
            return True
        tag_set = set(tags)
        for pattern in self.tools:
            if pattern.startswith("tag:"):
                if pattern[4:] in tag_set:
                    return True
            elif fnmatch(tool, pattern):
                return True
        return False


@dataclass(frozen=True)
class DelegationChain:
    """The identities a run is acting under, root first."""

    identities: tuple[AgentIdentity, ...] = ()

    @property
    def depth(self) -> int:
        """0 for a top-level run, 1 for its sub-agent, and so on."""
        return max(0, len(self.identities) - 1)

    @property
    def head(self) -> AgentIdentity | None:
        return self.identities[-1] if self.identities else None

    @property
    def root(self) -> AgentIdentity | None:
        return self.identities[0] if self.identities else None

    @property
    def names(self) -> list[str]:
        return [i.name for i in self.identities]

    def child(self, identity: AgentIdentity) -> DelegationChain:
        return DelegationChain((*self.identities, identity))

    def allows_tool(self, tool: str, tags: Iterable[str] = ()) -> tuple[bool, str]:
        tags = list(tags)
        for identity in self.identities:
            if not identity.allows_tool(tool, tags):
                via = "" if identity is self.head else " (inherited)"
                return False, f"{identity.name or 'agent'} may not use {tool}{via}"
        return True, ""

    def allowed_data(self) -> set[str] | None:
        """Intersection of every level's data allowance. None: unrestricted."""
        allowed: set[str] | None = None
        for identity in self.identities:
            if identity.data is not None:
                mine = set(identity.data)
                allowed = mine if allowed is None else allowed & mine
        return allowed

    def allowed_regions(self) -> set[str] | None:
        allowed: set[str] | None = None
        for identity in self.identities:
            if identity.regions is not None:
                mine = set(identity.regions)
                allowed = mine if allowed is None else allowed & mine
        return allowed

    def may_delegate(self, max_depth: int | None = None) -> tuple[bool, str]:
        """May the head hand work to one more level?"""
        head = self.head
        if head is not None and not head.may_delegate:
            return False, f"{head.name} is not allowed to delegate"
        limits = [i.max_delegation_depth for i in self.identities
                  if i.max_delegation_depth is not None]
        if max_depth is not None:
            limits.append(max_depth)
        if limits and self.depth + 1 > min(limits):
            return False, (f"delegation depth {self.depth + 1} exceeds the limit of "
                           f"{min(limits)}")
        return True, ""

    @property
    def risk(self) -> str:
        """The highest risk anywhere in the chain."""
        tiers = [i.risk for i in self.identities if i.risk]
        return max(tiers, key=RISK_ORDER.__getitem__) if tiers else "minimal"
