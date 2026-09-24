"""What the governance layer knows about a run while it is happening.

`RunState` is the per-run memory decisions are made against: who the run acts
for, under which delegation chain, which data classes have entered its
context (they stay — once a patient's diagnosis has been read, every later
model call carries it), whether untrusted content with injection signals has
been read, and how often each tool has been called.

`RuntimeMonitor` turns that into the signals OWASP's agentic list is about:

* **ASI01 goal hijack / ASI06 context poisoning** — a tool result with
  injection signals marks the run `untrusted`; policies can then require a
  human before any sensitive tool runs (``when: "run.untrusted"``).
* **ASI08 cascading failures / ASI10 rogue agents** — the same call repeated,
  a tool stormed, delegation fanning out: stopped, and an incident opened.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from ..guardrails.detectors import InjectionDetector
from .identity import DelegationChain

__all__ = ["RunState", "RuntimeMonitor", "current_run"]

#: The run the current task is acting for. Flows into sub-agents and parallel
#: tool calls with the asyncio context, so a child run sees its parent.
current_run: ContextVar[RunState | None] = ContextVar("agent_harness_governance_run",
                                                      default=None)


@dataclass
class RunState:
    run_id: str
    agent: str
    chain: DelegationChain
    principal: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""
    parent: RunState | None = None
    classes: set[str] = field(default_factory=set)
    untrusted: bool = False
    tool_calls: Counter[str] = field(default_factory=Counter)
    signatures: Counter[str] = field(default_factory=Counter)
    delegations: int = 0
    denials: int = 0
    models: set[str] = field(default_factory=set)
    providers: set[str] = field(default_factory=set)
    regions: set[str] = field(default_factory=set)
    pseudonymized: bool = False

    @property
    def top_level(self) -> bool:
        return self.parent is None

    def taint(self, classes: Any) -> None:
        """Data classes seen are sticky — and flow up to the parent run too."""
        new = set(classes) - self.classes
        if not new:
            return
        self.classes |= new
        if self.parent is not None:
            self.parent.taint(new)

    def context(self) -> dict[str, Any]:
        return {"id": self.run_id, "untrusted": self.untrusted,
                "tool_calls": sum(self.tool_calls.values()),
                "delegations": self.delegations, "denials": self.denials,
                "top_level": self.top_level}


@dataclass
class RuntimeMonitor:
    #: The same tool with the same arguments this many times in one run: a loop.
    max_identical_calls: int = 5
    #: Calls to one tool in one run.
    max_calls_per_tool: int | None = 50
    #: Sub-agents started by one run.
    max_delegations: int | None = 25
    injection: InjectionDetector = field(default_factory=InjectionDetector)

    @staticmethod
    def signature(tool: str, args: dict[str, Any]) -> str:
        payload = json.dumps(args, sort_keys=True, default=str)
        return tool + ":" + hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()

    def on_tool(self, state: RunState, tool: str, args: dict[str, Any]) -> str | None:
        """Count the call. Returns why it must be stopped, or None."""
        sig = self.signature(tool, args)
        state.tool_calls[tool] += 1
        state.signatures[sig] += 1
        if state.signatures[sig] > self.max_identical_calls:
            return (f"{tool} called with identical arguments "
                    f"{state.signatures[sig]} times — the agent is looping")
        if self.max_calls_per_tool and state.tool_calls[tool] > self.max_calls_per_tool:
            return f"{tool} called {state.tool_calls[tool]} times in one run"
        return None

    def on_delegate(self, state: RunState) -> str | None:
        state.delegations += 1
        if self.max_delegations and state.delegations > self.max_delegations:
            return f"{state.delegations} sub-agents started by one run"
        return None

    def on_tool_result(self, state: RunState, content: str) -> float:
        """Injection score of a tool result; marks the run untrusted when high."""
        findings = self.injection.scan(content[:20_000])
        score = findings[0].score if findings else 0.0
        if score >= self.injection.threshold:
            state.untrusted = True
            node = state.parent
            while node is not None:
                node.untrusted = True
                node = node.parent
        return score
