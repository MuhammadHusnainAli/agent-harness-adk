"""Completion checks: what an agent must have done before it may call itself done.

A check looks at the finished run — the answer, the tools it called, what it
cost — and either passes or says what is missing. The agent is told, in words,
and gets another turn to put it right.

    guardrails = AgentGuardrails(
        RequireTools("order_status"),       # look it up, do not guess
        MustInclude("order"),
        NoPlaceholders(),                   # no "TODO", no "[insert name]"
        MaxCost(0.25),
    )
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CompletionContext",
    "Violation",
    "Check",
    "RequireTools",
    "ForbidTools",
    "MustInclude",
    "MustNotInclude",
    "MustMatch",
    "MinLength",
    "MaxSteps",
    "MaxCost",
    "RequireCitation",
    "RequireJSON",
    "RequireStructured",
    "NoPlaceholders",
    "Custom",
]


@dataclass
class CompletionContext:
    """Everything a check is allowed to look at."""

    agent: str = ""
    output: str = ""
    steps: int = 0
    cost_usd: float = 0.0
    tools_called: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    data: Any = None
    result: Any = None

    @property
    def lowered(self) -> str:
        return self.output.lower()


@dataclass
class Violation:
    """One unmet requirement, and what the agent should do about it."""

    check: str
    detail: str
    fix: str = ""

    def line(self) -> str:
        return f"{self.detail}{f' — {self.fix}' if self.fix else ''}"


@runtime_checkable
class Check(Protocol):
    """Anything that can inspect a finished run and object to it."""

    name: str

    def __call__(self, ctx: CompletionContext) -> Violation | None: ...


# --- what it must have done ---------------------------------------------------

@dataclass
class RequireTools:
    """The agent must have called these before answering. Stops it guessing."""

    tools: Sequence[str]
    name: str = "require_tools"

    def __init__(self, *tools: str, name: str = "require_tools") -> None:
        self.tools = [t for group in tools for t in
                      ([group] if isinstance(group, str) else group)]
        self.name = name

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        missing = [t for t in self.tools if t not in ctx.tools_called]
        if not missing:
            return None
        return Violation(
            self.name,
            f"you answered without calling {', '.join(missing)}",
            f"call {missing[0]} and answer from what it returns",
        )


@dataclass
class ForbidTools:
    """These may never be called. Enforced before the call, and checked after."""

    tools: Sequence[str]
    name: str = "forbid_tools"

    def __init__(self, *tools: str, name: str = "forbid_tools") -> None:
        self.tools = [t for group in tools for t in
                      ([group] if isinstance(group, str) else group)]
        self.name = name

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        used = [t for t in self.tools if t in ctx.tools_called]
        if not used:
            return None
        return Violation(self.name, f"you called {', '.join(used)}, which is not "
                                    "permitted for this task")


# --- what the answer must look like -------------------------------------------

@dataclass
class MustInclude:
    """The answer has to mention these. One missing phrase is enough to fail."""

    phrases: Sequence[str]
    case_sensitive: bool = False
    name: str = "must_include"

    def __init__(self, *phrases: str, case_sensitive: bool = False,
                 name: str = "must_include") -> None:
        self.phrases = [p for group in phrases for p in
                        ([group] if isinstance(group, str) else group)]
        self.case_sensitive = case_sensitive
        self.name = name

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        haystack = ctx.output if self.case_sensitive else ctx.lowered
        missing = [p for p in self.phrases
                   if (p if self.case_sensitive else p.lower()) not in haystack]
        if not missing:
            return None
        return Violation(self.name,
                         f"your answer does not mention {', '.join(repr(m) for m in missing)}",
                         "say it explicitly")


@dataclass
class MustNotInclude:
    """The answer must avoid these — a phrase blocklist for the final reply."""

    phrases: Sequence[str]
    case_sensitive: bool = False
    name: str = "must_not_include"

    def __init__(self, *phrases: str, case_sensitive: bool = False,
                 name: str = "must_not_include") -> None:
        self.phrases = [p for group in phrases for p in
                        ([group] if isinstance(group, str) else group)]
        self.case_sensitive = case_sensitive
        self.name = name

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        haystack = ctx.output if self.case_sensitive else ctx.lowered
        found = [p for p in self.phrases
                 if (p if self.case_sensitive else p.lower()) in haystack]
        if not found:
            return None
        return Violation(self.name,
                         f"your answer contains {', '.join(repr(f) for f in found)}",
                         "rewrite it without that")


@dataclass
class MustMatch:
    """The answer has to match a pattern — an order id, a verdict, a format."""

    pattern: str
    description: str = ""
    name: str = "must_match"

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        if re.search(self.pattern, ctx.output, re.IGNORECASE | re.DOTALL):
            return None
        return Violation(self.name,
                         self.description or f"your answer does not match /{self.pattern}/",
                         "answer in the form that was asked for")


@dataclass
class MinLength:
    """Guards against a one-word answer to a question that needed working through."""

    chars: int = 40
    name: str = "min_length"

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        if len(ctx.output.strip()) >= self.chars:
            return None
        return Violation(self.name,
                         f"your answer is {len(ctx.output.strip())} characters; "
                         f"this task needs at least {self.chars}",
                         "answer the question properly")


@dataclass
class RequireCitation:
    """Every claim needs a source. Checks the answer carries at least one."""

    markers: Sequence[str] = ("http://", "https://", "source:", "according to",
                              "[1]", "§")
    name: str = "require_citation"

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        if any(m.lower() in ctx.lowered for m in self.markers):
            return None
        return Violation(self.name, "your answer cites nothing",
                         "give the source for each claim, or say you could not find one")


@dataclass
class RequireJSON:
    """The answer must be parseable JSON — the cheap version of a contract."""

    name: str = "require_json"

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        text = ctx.output.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        for candidate in ([fenced.group(1)] if fenced else []) + [text]:
            try:
                json.loads(candidate)
                return None
            except (json.JSONDecodeError, TypeError):
                continue
        return Violation(self.name, "your answer is not valid JSON",
                         "reply with only the JSON object, no prose")


@dataclass
class RequireStructured:
    """Used with `output_type`: the contract must actually have been satisfied."""

    name: str = "require_structured"

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        if ctx.data is not None:
            return None
        return Violation(self.name, "you did not return the structured output "
                                    "this task requires",
                         "reply with only the JSON object the schema describes")


@dataclass
class NoPlaceholders:
    """Catches work handed back half-finished: TODO, [insert x], lorem ipsum."""

    patterns: Sequence[str] = (
        r"\bTODO\b", r"\bFIXME\b", r"\bTBD\b", r"lorem ipsum",
        r"\[insert[^\]]*\]", r"\[your[^\]]*\]", r"<placeholder>", r"xxx+",
    )
    name: str = "no_placeholders"

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        hits = [p for p in self.patterns
                if re.search(p, ctx.output, re.IGNORECASE)]
        if not hits:
            return None
        return Violation(self.name, "your answer still contains placeholder text",
                         "fill it in, or say plainly what you could not find")


# --- budgets, as a completion condition ----------------------------------------

@dataclass
class MaxSteps:
    """A soft step ceiling: the run is not killed, the answer is just rejected."""

    steps: int
    name: str = "max_steps"

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        if ctx.steps <= self.steps:
            return None
        return Violation(self.name,
                         f"this took {ctx.steps} steps, the limit is {self.steps}")


@dataclass
class MaxCost:
    """A soft spend ceiling, checked at the end rather than enforced mid-run."""

    usd: float
    name: str = "max_cost"

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        if ctx.cost_usd <= self.usd:
            return None
        return Violation(self.name,
                         f"this cost ${ctx.cost_usd:.4f}, the limit is ${self.usd:.4f}")


@dataclass
class Custom:
    """Your own rule. Return True, or False, or (False, "what is wrong")."""

    fn: Callable[[CompletionContext], Any]
    name: str = "custom"
    fix: str = ""

    def __call__(self, ctx: CompletionContext) -> Violation | None:
        verdict = self.fn(ctx)
        detail = ""
        if isinstance(verdict, tuple):
            verdict, detail = verdict[0], str(verdict[1])
        if verdict:
            return None
        return Violation(self.name, detail or f"the {self.name} check did not pass",
                         self.fix)


def as_checks(items: Iterable[Any]) -> list[Check]:
    """Accept checks, or bare callables, and normalise them."""
    out: list[Check] = []
    for item in items:
        if isinstance(item, Check):
            out.append(item)
        elif callable(item):
            out.append(Custom(item, name=getattr(item, "__name__", "custom")))
        else:
            raise TypeError(f"{item!r} is not a guardrail check")
    return out
