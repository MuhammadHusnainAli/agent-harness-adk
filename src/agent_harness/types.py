"""The value types that travel through the harness.

Everything here is a pydantic model so it validates at the edges, serialises to
JSON for the journal and the session store, and round-trips back again.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]

__all__ = [
    "Role",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ThinkingBlock",
    "ImageBlock",
    "ContentBlock",
    "Message",
    "Usage",
    "ModelResponse",
    "StreamEvent",
    "ToolCall",
    "ToolOutcome",
    "Artifact",
    "RunResult",
    "new_id",
]


def new_id(prefix: str = "") -> str:
    """Short, sortable-enough ids. Cheap: no crypto, no clock skew worries."""
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}_{raw}" if prefix else raw


class _Block(BaseModel):
    model_config = ConfigDict(extra="allow")


class TextBlock(_Block):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(_Block):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ToolUseBlock(_Block):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(_Block):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str = ""
    is_error: bool = False


class ImageBlock(_Block):
    type: Literal["image"] = "image"
    media_type: str = "image/png"
    data: str = ""  # base64
    url: str | None = None


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock | ImageBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    """One turn of the conversation, always stored as a list of blocks."""

    model_config = ConfigDict(extra="allow")

    role: Role
    content: list[ContentBlock] = Field(default_factory=list)
    name: str | None = None
    ts: float = Field(default_factory=time.time)

    # ---- constructors -------------------------------------------------
    @classmethod
    def user(cls, text: str, **kw: Any) -> Message:
        return cls(role="user", content=[TextBlock(text=text)], **kw)

    @classmethod
    def assistant(cls, content: str | list[ContentBlock], **kw: Any) -> Message:
        blocks = [TextBlock(text=content)] if isinstance(content, str) else list(content)
        return cls(role="assistant", content=blocks, **kw)

    @classmethod
    def system(cls, text: str, **kw: Any) -> Message:
        return cls(role="system", content=[TextBlock(text=text)], **kw)

    @classmethod
    def tool_results(cls, results: list[ToolResultBlock], **kw: Any) -> Message:
        """Tool output goes back as a *user* turn — that is what every provider expects."""
        return cls(role="user", content=list(results), **kw)

    # ---- accessors ----------------------------------------------------
    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock)).strip()

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    def __str__(self) -> str:  # pragma: no cover - debugging affordance
        return f"{self.role}: {self.text[:120]}"


class Usage(BaseModel):
    """Token and money accounting for a single call, or a whole run once summed."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 8),
            calls=self.calls + other.calls,
        )

    def __iadd__(self, other: Usage) -> Usage:
        merged = self + other
        for field in type(self).model_fields:
            setattr(self, field, getattr(merged, field))
        return self


StopReason = Literal[
    "end_turn", "tool_use", "max_tokens", "stop_sequence", "max_steps", "error", "stopped"
]


class ModelResponse(BaseModel):
    """What a provider hands back, normalised across Anthropic / OpenAI / Gemini."""

    model_config = ConfigDict(extra="allow")

    message: Message
    stop_reason: StopReason = "end_turn"
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0
    raw: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return self.message.tool_uses


class StreamEvent(BaseModel):
    """One beat of a streaming run: a token, a tool call, a step boundary."""

    model_config = ConfigDict(extra="allow")

    type: Literal[
        "run_start", "step_start", "text", "thinking", "tool_call", "tool_result",
        "step_end", "run_end", "error", "delegation",
    ]
    text: str = ""
    agent: str = ""
    step: int = 0
    data: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A tool invocation as it happened, for the journal and the audit trail."""

    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    agent: str = ""
    step: int = 0


class ToolOutcome(BaseModel):
    """The result of running a tool, plus everything the rails want to record."""

    model_config = ConfigDict(extra="allow")

    call_id: str
    name: str
    content: str = ""
    is_error: bool = False
    duration_ms: float = 0.0
    cached: bool = False
    value: Any = Field(default=None, exclude=True)

    def as_block(self) -> ToolResultBlock:
        return ToolResultBlock(
            tool_use_id=self.call_id, content=self.content, is_error=self.is_error
        )


class Artifact(BaseModel):
    """A file or document a run produced. Sub-agents hand these back, not transcripts."""

    model_config = ConfigDict(extra="allow")

    name: str
    content: str = ""
    path: str | None = None
    media_type: str = "text/plain"
    produced_by: str = ""
    ts: float = Field(default_factory=time.time)


class RunResult(BaseModel):
    """The one object a caller gets back from `agent.run()`."""

    model_config = ConfigDict(extra="allow")

    output: str = ""
    data: Any = None
    agent: str = ""
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    steps: int = 0
    stop_reason: StopReason = "end_turn"
    tool_calls: list[ToolCall] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    session_id: str = ""
    run_id: str = Field(default_factory=lambda: new_id("run"))
    trace_id: str = ""
    error: str | None = None
    children: list[RunResult] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def cost_usd(self) -> float:
        """This run plus everything it delegated."""
        return round(self.usage.cost_usd + sum(c.cost_usd for c in self.children), 8)

    def __str__(self) -> str:  # pragma: no cover - debugging affordance
        return self.output
