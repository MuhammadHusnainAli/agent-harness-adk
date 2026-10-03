"""Handoff: another agent takes over the conversation.

    billing = Agent("billing", "Handle refunds and invoices.", tools=[refund])
    triage = Agent("triage", "Work out what the customer needs.", mode="chat",
                   handoffs=[billing])

    result = await triage.run("I was charged twice.")
    result.agent          # "billing" — who answered
    result.handoffs       # [triage → billing: "a billing question"]
    await triage.run("And the invoice?")      # billing still has it

Delegation and a handoff are different things. A sub-agent is given a task,
starts clean, and reports back to the agent that asked — which then answers. A
handoff gives the *conversation* away: the other agent sees what was said,
answers the user itself, and keeps answering on the turns that follow, until it
hands the conversation on or back.

A `handoff` tool appears on any agent that has somewhere to hand off to. What
the receiving agent sees is the `history` of the `Handoff`:

    Handoff(billing)                      # everything, tool calls included
    Handoff(billing, history="text")      # what was said, not what was looked up
    Handoff(billing, history="fresh")     # only the user's last message
    Handoff(billing, history=my_filter)   # your own: messages in, messages out

and that is the conversation from then on — a narrowed history is narrowed for
good, which is the point of narrowing it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .context import close_open_tool_calls
from .errors import ConfigurationError
from .types import (
    HandoffRecord,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

__all__ = ["Handoff", "HandoffRecord", "HISTORIES"]

#: The built-in answers to "what does the agent taking over get to see?"
HISTORIES: tuple[str, ...] = ("full", "text", "fresh")

HistoryFilter = Callable[[list[Message]], list[Message]]

# How many handoffs a session remembers the story of.
_TRAIL = 20


class Handoff:
    """Somewhere an agent may hand the conversation, and how it is handed over.

    Args:
        agent: the agent that takes over.
        description: when to hand off to it — what the model choosing reads.
            Defaults to the agent's own description.
        history: what the agent taking over sees. `"full"` is everything;
            `"text"` is what was said, without tool calls and their results;
            `"fresh"` is only the user's latest message; or a function from the
            messages so far to the messages it should have.
        sticky: True, and it keeps the conversation on the turns that follow.
            False, and it answers this turn only — the next one goes back to
            whoever had the conversation before.
        on_handoff: called with the `HandoffRecord` once the handoff is
            allowed, before the other agent starts. Raise to refuse it.
    """

    def __init__(self, agent: Any, *, description: str | None = None,
                 history: str | HistoryFilter = "full", sticky: bool = True,
                 on_handoff: Callable[[HandoffRecord], Any] | None = None) -> None:
        if not (hasattr(agent, "_drive") and hasattr(agent, "tools")):
            raise ConfigurationError(
                f"a handoff needs an Agent to hand off to — got {type(agent).__name__}")
        if not callable(history) and history not in HISTORIES:
            raise ConfigurationError(
                f"handoff history must be one of {', '.join(HISTORIES)}, or a "
                f"function — got {history!r}")
        self.agent = agent
        self.description = description or agent.description
        self.history = history
        self.sticky = sticky
        self.on_handoff = on_handoff

    @property
    def name(self) -> str:
        return self.agent.name

    @classmethod
    def of(cls, value: Any) -> Handoff:
        """An Agent, or a Handoff that already says how."""
        return value if isinstance(value, Handoff) else cls(value)

    async def announce(self, record: HandoffRecord) -> None:
        if self.on_handoff is None:
            return
        outcome = self.on_handoff(record)
        if inspect.isawaitable(outcome):
            await outcome

    def view(self, history: list[Message], *, source: Any,
             reason: str = "") -> tuple[list[Message], bool]:
        """The conversation as the agent taking over gets it.

        Returns the messages, and whether they are no longer simply the ones
        handed in — a narrowed history is a rewritten one.
        """
        target = self.agent
        how = self.history
        # A model given tool calls and no tools rejects the conversation.
        if how == "full" and not len(target.tools):
            how = "text"
        if callable(how):
            seen = how(list(history))
            if not isinstance(seen, list) or not all(
                    isinstance(m, Message) for m in seen):
                raise ConfigurationError(
                    f"the history filter for {target.name} must return a list of "
                    "Message")
            seen = close_open_tool_calls(seen)
        elif how == "text":
            seen = _spoken(history)
        elif how == "fresh":
            seen = _latest_request(history)
        else:
            seen = list(history)
        rewrote = how != "full"

        # A model's reasoning is signed for that model; another one refuses it.
        if target.model != source.model and any(
                isinstance(b, ThinkingBlock) for m in seen for b in m.content):
            seen = _without(seen, (ThinkingBlock,))
            rewrote = True

        # With the tool call gone, nothing in the conversation says it changed
        # hands — and a model cannot answer a conversation that ends on itself.
        if how != "full" and (not callable(how) or not seen
                              or seen[-1].role != "user"):
            why = f": {reason}" if reason else ""
            seen.append(Message.user(
                f"[{source.name} handed this conversation to {target.name}{why}. "
                f"You are {target.name}: carry on from here and answer the user "
                "directly.]"))
        return seen, rewrote


def _without(messages: list[Message], kinds: tuple[type, ...]) -> list[Message]:
    """The same conversation with some kinds of block taken out."""
    out: list[Message] = []
    for message in messages:
        kept = [b for b in message.content if not isinstance(b, kinds)]
        if len(kept) == len(message.content):
            out.append(message)
        elif kept:
            out.append(message.model_copy(update={"content": kept}))
    return out


def _spoken(messages: list[Message]) -> list[Message]:
    """What was said: no tool calls, no results, no reasoning. Turns that are
    left next to one another by the same speaker become one."""
    out: list[Message] = []
    for message in _without(messages, (ToolUseBlock, ToolResultBlock, ThinkingBlock)):
        if out and out[-1].role == message.role == "assistant":
            out[-1] = out[-1].model_copy(
                update={"content": [*out[-1].content, *message.content]})
        else:
            out.append(message)
    return out


def _latest_request(messages: list[Message]) -> list[Message]:
    """The last thing the user themselves said."""
    for message in reversed(messages):
        if message.role == "user" and not any(
                isinstance(b, ToolResultBlock) for b in message.content) and (
                message.media or any(isinstance(b, TextBlock) and b.text
                                     for b in message.content)):
            return [message]
    return []


@dataclass
class _Pending:
    """A handoff the model asked for, waiting for the step to finish."""

    handoff: Handoff
    reason: str
    keeper: str            # who kept the conversation before this was asked


@dataclass
class Relay:
    """What travels with a conversation while it changes hands within one run.

    One run of the agent you called may be several agents' work. Each of them
    drives its own loop; this is the part they share — who is next, who keeps
    the conversation afterwards, and the one session all of it is saved to.
    """

    entry: str
    limit: int
    store: Any
    persist: bool = True
    #: Who has the conversation on the next turn.
    keeper: str = ""
    hops: list[HandoffRecord] = field(default_factory=list)
    pending: _Pending | None = None
    #: This agent was handed the conversation mid-run: there is no new message
    #: from the user to add to it.
    takeover: bool = False
    rewrote: bool = False
    session: Any = None
    earlier: list[Message] = field(default_factory=list)
    trail: list[dict[str, Any]] | None = None
    #: The todo list and source ledger of the agent that last had one.
    notebook: Any = None

    def attach(self, session: Any) -> None:
        self.session = session
        if self.trail is None:
            kept = (session.metadata.get("handoff") or {}).get("trail") or []
            self.trail = list(kept)

    def owner_ok(self, agent: Any) -> bool:
        """May `agent` have this conversation's session — is it acting for the
        person the session belongs to?"""
        trace = getattr(agent, "trace", None)
        return self.session is None or self.session.owned_by(
            trace.user_id if trace is not None else None,
            trace.tenant_id if trace is not None else None)

    def claim(self, handoff: Handoff, reason: str) -> None:
        self.pending = _Pending(handoff, reason, self.keeper)

    def confirm(self, record: HandoffRecord) -> None:
        self.hops.append(record)
        if self.pending is not None and self.pending.handoff.sticky:
            self.keeper = record.target

    def release(self) -> None:
        """A handoff that was asked for and refused before it was recorded."""
        self.pending = None

    def cancel(self) -> None:
        """The run that asked for a handoff failed before it could happen."""
        if self.pending is None:
            return
        self.keeper = self.pending.keeper
        if self.hops and self.hops[-1].target == self.pending.handoff.name:
            self.hops.pop()
        self.pending = None

    def stamp(self, session: Any, agent: str, *, never_started: bool = False) -> None:
        """Write on the session who has the conversation now, and how it got
        there, so the next turn — in this process or another — goes to them."""
        if never_started and self.keeper == agent and agent != self.entry:
            # It has a conversation it could not even begin on. Leaving it
            # there would strand every turn that follows.
            self.keeper = (self.hops[-1].source if self.takeover and self.hops
                           else self.entry)
        if not self.hops and "handoff" not in session.metadata:
            return
        trail = [*(self.trail or []), *(h.model_dump(mode="json") for h in self.hops)]
        session.metadata["handoff"] = {
            "active": self.keeper or agent, "entry": self.entry,
            "trail": trail[-_TRAIL:],
        }
