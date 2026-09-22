"""Context assembly and compaction — what actually reaches the model.

The assembler builds the system prompt from four layers: operating instructions,
domain (skills the agent can pull in), context (tools, output contract) and
memory. The compactor keeps the message list under budget without ever breaking
a tool_use/tool_result pair, which is the usual way naive trimming corrupts a
conversation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from .prompts import sections
from .types import Message, TextBlock, ToolResultBlock, ToolUseBlock

__all__ = ["estimate_tokens", "ContextAssembler", "ContextCompactor"]

Summarizer = Callable[[str], Awaitable[str]]


def estimate_tokens(text: str) -> int:
    """Cheap, provider-agnostic estimate. Good to roughly ±15% on English prose."""
    return max(1, len(text) // 4)


def message_tokens(message: Message) -> int:
    total = 8  # per-message overhead the providers add
    for block in message.content:
        if isinstance(block, TextBlock):
            total += estimate_tokens(block.text)
        elif isinstance(block, ToolResultBlock):
            total += estimate_tokens(block.content)
        elif isinstance(block, ToolUseBlock):
            total += estimate_tokens(str(block.input)) + 8
        else:
            total += 16
    return total


def conversation_tokens(messages: Iterable[Message]) -> int:
    return sum(message_tokens(m) for m in messages)


class ContextAssembler:
    """Builds the system prompt. Stable parts first, so prompt caching works."""

    def __init__(
        self,
        identity: str = "",
        *,
        instructions: str = "",
        skills: Any = None,
        memory: Any = None,
        extra: Iterable[str] = (),
        include_memory: bool = True,
    ) -> None:
        self.identity = identity
        self.instructions = instructions
        self.skills = skills
        self.memory = memory
        self.extra = list(extra)
        self.include_memory = include_memory

    async def build(self, *, query: str = "", tool_names: Iterable[str] = (),
                    output_contract: str = "") -> str:
        """Assemble the system prompt for one call."""
        memory_blocks: list[str] = []
        if self.include_memory and self.memory is not None:
            memory_blocks = await self.memory.prompt_blocks(query)

        skills_index = self.skills.index() if self.skills is not None else ""
        names = list(tool_names)
        tool_note = (
            f"Tools you can call: {', '.join(names)}.\n"
            "Call a tool when it gets you a fact you do not have. Do not guess "
            "what a tool would have returned."
        ) if names else ""

        return sections(
            self.identity,
            self.instructions,
            skills_index,
            tool_note,
            ("Output contract", output_contract) if output_contract else None,
            *memory_blocks,
            *self.extra,
        )


class ContextCompactor:
    """Summarise · evict · pin — keeps the window under budget, pairing intact."""

    def __init__(
        self,
        *,
        max_tokens: int = 120_000,
        keep_last: int = 8,
        evict_tool_results_over: int = 2_000,
        summarize: Summarizer | None = None,
        target_ratio: float = 0.6,
    ) -> None:
        self.max_tokens = max_tokens
        self.keep_last = keep_last
        self.evict_over = evict_tool_results_over
        self.summarize = summarize
        self.target_ratio = target_ratio

    def should_compact(self, messages: list[Message], *, headroom: int = 0) -> bool:
        return conversation_tokens(messages) + headroom > self.max_tokens

    async def compact(self, messages: list[Message], *,
                      pinned: Iterable[str] = ()) -> list[Message]:
        """Return a shorter message list that still round-trips to the provider."""
        if not self.should_compact(messages):
            return messages

        target = int(self.max_tokens * self.target_ratio)
        working = [m.model_copy(deep=True) for m in messages]

        # 1. Evict fat tool results first — they are the cheapest tokens to lose,
        #    and blanking the content keeps every tool_use/tool_result pair intact.
        protected = set(range(max(0, len(working) - self.keep_last), len(working)))
        for idx, msg in enumerate(working):
            if idx in protected or conversation_tokens(working) <= target:
                continue
            for block in msg.content:
                if isinstance(block, ToolResultBlock) and len(block.content) > self.evict_over:
                    dropped = len(block.content)
                    block.content = (
                        f"{block.content[:400]}\n"
                        f"... [evicted {dropped - 400} characters to free context]"
                    )

        if conversation_tokens(working) <= target:
            return working

        # 2. Still too big: summarise the head, keep the tail.
        cut = self._safe_cut(working, max(0, len(working) - self.keep_last))
        head, tail = working[:cut], working[cut:]
        if not head:
            return working

        transcript = "\n".join(f"{m.role}: {m.text}" for m in head if m.text)
        pins = "\n".join(f"- {p}" for p in pinned)
        if self.summarize and transcript.strip():
            summary = await self.summarize(
                "Summarise this conversation so the assistant can continue without it. "
                "Keep decisions, facts established, open questions and anything the user "
                "asked for. Be dense and specific.\n\n" + transcript[:60_000]
            )
        else:
            summary = self._mechanical_summary(head)

        block = "## Earlier in this conversation (compacted)\n" + summary.strip()
        if pins:
            block += f"\n\n## Pinned facts\n{pins}"
        return [Message.user(block), *tail]

    @staticmethod
    def _safe_cut(messages: list[Message], desired: int) -> int:
        """Move the cut forward until the tail has no orphaned tool results."""
        cut = max(0, min(desired, len(messages)))
        while cut < len(messages):
            has_orphan = any(
                isinstance(b, ToolResultBlock) for b in messages[cut].content
            )
            if not has_orphan:
                break
            cut += 1
        return cut

    @staticmethod
    def _mechanical_summary(messages: list[Message]) -> str:
        """No model available: keep the shape of the conversation, not the bulk."""
        lines: list[str] = []
        for msg in messages:
            if msg.text:
                snippet = " ".join(msg.text.split())[:300]
                lines.append(f"- {msg.role}: {snippet}")
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    lines.append(f"- called {block.name}({_short(block.input)})")
        return "\n".join(lines[-80:])


def _short(data: dict[str, Any], limit: int = 120) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in data.items())
    return text[:limit] + ("..." if len(text) > limit else "")
