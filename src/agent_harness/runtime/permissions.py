"""Permission and policy gate: allow, ask, or deny — per action.

    gate = PolicyGate(default="allow", deny=["shell", "fs_write"], ask=["http_*"])

`ask` needs an approver. Without one, asking is denied — a harness should never
silently escalate itself.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Literal

from ..errors import PermissionDenied

__all__ = ["Decision", "Rule", "PolicyGate", "console_approver", "always_approve"]

Decision = Literal["allow", "ask", "deny"]


@dataclass
class Rule:
    """First matching rule wins. `pattern` globs the tool name."""

    pattern: str
    decision: Decision = "allow"
    reason: str = ""
    when: Callable[[dict[str, Any]], bool] | None = None

    def matches(self, tool: str, args: dict[str, Any]) -> bool:
        if not fnmatch(tool, self.pattern):
            return False
        return self.when is None or bool(self.when(args))


async def always_approve(tool: str, args: dict[str, Any], reason: str) -> bool:
    return True


async def console_approver(tool: str, args: dict[str, Any], reason: str) -> bool:
    """Blocking y/n prompt on stdin. Fine for a CLI, never for a server."""
    import asyncio
    prompt = f"\nAllow {tool}({args})? {reason}\n[y/N] "
    answer = await asyncio.to_thread(input, prompt)
    return answer.strip().lower() in {"y", "yes"}


class PolicyGate:
    """Decides whether an action may run, before it runs."""

    def __init__(
        self,
        default: Decision = "allow",
        *,
        rules: Iterable[Rule] = (),
        allow: Iterable[str] = (),
        ask: Iterable[str] = (),
        deny: Iterable[str] = (),
        approver: Callable[..., Any] | None = None,
        audit: Callable[[str, str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self.default = default
        # Deny wins over ask wins over allow when patterns overlap.
        self.rules = [
            *[Rule(p, "deny", "denied by policy") for p in deny],
            *[Rule(p, "ask", "needs approval") for p in ask],
            *[Rule(p, "allow") for p in allow],
            *rules,
        ]
        self.approver = approver
        self.audit = audit

    def decide(self, tool: str, args: dict[str, Any] | None = None) -> tuple[Decision, str]:
        for rule in self.rules:
            if rule.matches(tool, args or {}):
                return rule.decision, rule.reason
        return self.default, ""

    async def check(self, tool: str, args: dict[str, Any] | None = None, *,
                    tool_permission: Decision | None = None) -> None:
        """Raise PermissionDenied unless the action is allowed. Otherwise return."""
        decision, reason = self.decide(tool, args)
        # A tool's own declared permission can only tighten the policy, never loosen it.
        if tool_permission == "deny":
            decision, reason = "deny", "the tool declares itself deny-by-default"
        elif tool_permission == "ask" and decision == "allow":
            decision, reason = "ask", "the tool asks for confirmation"

        if decision == "allow":
            await self._audit(tool, "allow", args or {})
            return
        if decision == "deny":
            await self._audit(tool, "deny", args or {})
            raise PermissionDenied(f"{tool} is not permitted: {reason or 'policy'}",
                                   tool=tool, reason=reason)
        if self.approver is None:
            await self._audit(tool, "deny", args or {})
            raise PermissionDenied(
                f"{tool} needs approval but no approver is configured", tool=tool,
                reason=reason,
            )
        verdict = self.approver(tool, args or {}, reason)
        if inspect.isawaitable(verdict):
            verdict = await verdict
        await self._audit(tool, "allow" if verdict else "deny", args or {})
        if not verdict:
            raise PermissionDenied(f"{tool} was declined by the approver", tool=tool,
                                   reason=reason)

    async def _audit(self, tool: str, decision: str, args: dict[str, Any]) -> None:
        if self.audit is None:
            return
        result = self.audit(tool, decision, args)
        if inspect.isawaitable(result):
            await result
