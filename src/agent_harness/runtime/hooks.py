"""Hook engine: pre and post automation around every model call and tool call.

    hooks = HookEngine()

    @hooks.on("pre_tool")
    def audit(ctx):
        if ctx.data["tool"] == "shell" and "rm -rf" in str(ctx.data["args"]):
            ctx.block("no recursive deletes")

A `pre_tool` handler can block the call or rewrite its arguments; `post_tool`
can rewrite the result before the model ever sees it.

`model_egress` fires once per provider actually tried — after the fallback
chain has picked the backend, so a handler sees where the data is really going
(provider, model, region). Blocking it skips that backend and moves on to the
next model in the chain; replacing it swaps the request sent to that backend
only, leaving the agent's own history untouched.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["HookEvent", "HookContext", "HookEngine"]

HookEvent = Literal[
    "run_start", "run_end", "step_start", "step_end",
    "pre_model", "post_model", "model_egress", "pre_tool", "post_tool",
    "subagent_start", "subagent_end", "memory_write", "error",
]

EVENTS: tuple[str, ...] = (
    "run_start", "run_end", "step_start", "step_end", "pre_model", "post_model",
    "model_egress", "pre_tool", "post_tool", "subagent_start", "subagent_end", "memory_write", "error",
)


@dataclass
class HookContext:
    """What a handler is given, and how it answers back."""

    event: str
    agent: str = ""
    run_id: str = ""
    step: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    reason: str = ""
    replacement: Any = None
    replaced: bool = False

    def block(self, reason: str = "blocked by hook") -> None:
        self.blocked = True
        self.reason = reason

    def replace(self, value: Any) -> None:
        """Swap the arguments (pre_tool) or the result (post_tool/post_model)."""
        self.replacement = value
        self.replaced = True

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class HookEngine:
    """Registry of handlers, fired in registration order."""

    def __init__(self, handlers: dict[str, Iterable[Callable]] | None = None) -> None:
        self._handlers: dict[str, list[Callable]] = {e: [] for e in EVENTS}
        for event, fns in (handlers or {}).items():
            for fn in fns:
                self.add(event, fn)

    def add(self, event: str, fn: Callable[[HookContext], Any]) -> Callable:
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(fn)
        return fn

    def on(self, event: str) -> Callable[[Callable], Callable]:
        """Decorator form: `@hooks.on("pre_tool")`."""
        def wrap(fn: Callable[[HookContext], Any]) -> Callable:
            self.add(event, fn)
            return fn
        return wrap

    def remove(self, event: str, fn: Callable) -> None:
        if fn in self._handlers.get(event, []):
            self._handlers[event].remove(fn)

    def count(self, event: str) -> int:
        return len(self._handlers.get(event, []))

    async def emit(self, event: str, *, agent: str = "", run_id: str = "", step: int = 0,
                   **data: Any) -> HookContext:
        """Fire every handler for `event` and return the (possibly mutated) context."""
        ctx = HookContext(event=event, agent=agent, run_id=run_id, step=step, data=data)
        for fn in self._handlers.get(event, ()):
            result = fn(ctx)
            if inspect.isawaitable(result):
                await result
            if ctx.blocked:
                break
        return ctx

    def merge(self, other: HookEngine | None) -> HookEngine:
        """Combine two engines — an agent's own hooks plus the harness-wide ones."""
        if other is None:
            return self
        merged = HookEngine()
        for event in set(self._handlers) | set(other._handlers):
            for fn in self._handlers.get(event, []) + other._handlers.get(event, []):
                merged.add(event, fn)
        return merged
