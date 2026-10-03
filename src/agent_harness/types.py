"""The value types that travel through the harness.

Everything here is a pydantic model so it validates at the edges, serialises to
JSON for the journal and the session store, and round-trips back again.
"""

from __future__ import annotations

import base64
import mimetypes
import time
import uuid
from pathlib import Path
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
    "AudioBlock",
    "VideoBlock",
    "DocumentBlock",
    "MediaBlock",
    "attach",
    "ContentBlock",
    "Message",
    "Usage",
    "ModelResponse",
    "StreamEvent",
    "ToolCall",
    "ToolOutcome",
    "Artifact",
    "Todo",
    "Source",
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


class MediaBlock(_Block):
    """Something that is not text: an image, a recording, a clip, a file.

    It is held one of three ways. `path` is a file on this machine, read when
    the message is sent — so a conversation that is saved stays small, and a
    large file is not copied into every session. `data` is the bytes themselves,
    base64, for something that exists only in memory. `url` is somewhere the
    provider can fetch it from.
    """

    media_type: str = "application/octet-stream"
    data: str = ""  # base64
    url: str | None = None
    path: str | None = None
    name: str = ""

    @classmethod
    def from_path(cls, path: str | Path, media_type: str | None = None,
                  **fields: Any) -> Any:
        file = Path(path).expanduser()
        return cls(path=str(file), name=fields.pop("name", file.name),
                   media_type=media_type or _guess(file.name, cls), **fields)

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str | None = None,
                   **fields: Any) -> Any:
        kind = media_type or cls.model_fields["media_type"].default
        return cls(data=base64.b64encode(data).decode(), media_type=kind, **fields)

    @classmethod
    def from_url(cls, url: str, media_type: str | None = None, **fields: Any) -> Any:
        name = fields.pop("name", url.rsplit("/", 1)[-1].split("?")[0])
        return cls(url=url, name=name,
                   media_type=media_type or _guess(name, cls), **fields)

    @property
    def inline(self) -> bool:
        """Can its bytes be sent, rather than only pointed at?"""
        return bool(self.data or self.path)

    def load(self) -> str:
        """The content as base64. Raises `FileNotFoundError` if `path` is gone."""
        if self.data:
            return self.data
        if self.path:
            return base64.b64encode(Path(self.path).read_bytes()).decode()
        raise FileNotFoundError(self.url or self.name or "attachment")

    def read(self) -> bytes:
        return base64.b64decode(self.load())

    def size(self) -> int:
        """Its size in bytes; 0 for one that is only a URL."""
        if self.data:
            return len(self.data) * 3 // 4 - self.data[-2:].count("=")
        if self.path:
            try:
                return Path(self.path).stat().st_size
            except OSError:
                return 0
        return 0

    @property
    def label(self) -> str:
        return self.name or self.url or self.type

    def __repr__(self) -> str:  # never the bytes
        where = self.path or self.url or f"{self.size()} bytes"
        return f"<{type(self).__name__} {self.media_type} {where}>"

    __str__ = __repr__


class ImageBlock(MediaBlock):
    type: Literal["image"] = "image"
    media_type: str = "image/png"


class AudioBlock(MediaBlock):
    """A recording. Models that listen take it as it is; for one that cannot,
    the harness transcribes it first, if it has been given something to
    transcribe with."""

    type: Literal["audio"] = "audio"
    media_type: str = "audio/wav"


class VideoBlock(MediaBlock):
    type: Literal["video"] = "video"
    media_type: str = "video/mp4"


class DocumentBlock(MediaBlock):
    """A file to read: a PDF, a Word document, a spreadsheet, a text file.

    A PDF reaches a model that reads PDFs as the PDF, pages and figures and all.
    Anything else — and a PDF for a model that does not — is read here and its
    text sent instead.
    """

    type: Literal["document"] = "document"
    media_type: str = "application/pdf"


_KINDS: dict[str, type[MediaBlock]] = {"image": ImageBlock, "audio": AudioBlock,
                                       "video": VideoBlock}


def _guess(name: str, cls: type[MediaBlock] | None = None) -> str:
    found = mimetypes.guess_type(name)[0]
    if found:
        return found
    return cls.model_fields["media_type"].default if cls else "application/octet-stream"


def attach(source: Any, media_type: str | None = None, **fields: Any) -> MediaBlock:
    """Whatever you have, as the block that carries it.

        attach("chart.png")                      # an image
        attach("call.mp3")                       # a recording
        attach("report.pdf")                     # a document
        attach("https://example.com/clip.mp4")   # a video the provider fetches
        attach(pcm_bytes, "audio/wav")           # bytes, with what they are

    The kind is read from the media type: `image/*`, `audio/*`, `video/*`, and
    everything else is a document.
    """
    if isinstance(source, MediaBlock):
        return source
    if isinstance(source, (bytes, bytearray)):
        if not media_type:
            raise ValueError("bytes need a media_type — attach(data, \"audio/wav\")")
        kind = _KINDS.get(media_type.split("/", 1)[0], DocumentBlock)
        return kind.from_bytes(bytes(source), media_type, **fields)
    text = str(source)
    kind = _KINDS.get((media_type or _guess(text.split("?")[0])).split("/", 1)[0],
                      DocumentBlock)
    if text.startswith(("http://", "https://", "gs://")):
        return kind.from_url(text, media_type, **fields)
    return kind.from_path(text, media_type, **fields)


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock | ImageBlock
    | AudioBlock | VideoBlock | DocumentBlock,
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
    def user(cls, text: str = "", *, attachments: Any = (), **kw: Any) -> Message:
        """A user turn. `attachments` are files, URLs or blocks; they go before
        the text, which is where models expect what the text is about."""
        blocks: list[Any] = [attach(item) for item in attachments]
        if text or not blocks:
            blocks.append(TextBlock(text=text))
        return cls(role="user", content=blocks, **kw)

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

    @property
    def media(self) -> list[MediaBlock]:
        return [b for b in self.content if isinstance(b, MediaBlock)]

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
    "end_turn", "tool_use", "max_tokens", "stop_sequence", "max_steps", "error",
    "stopped", "budget",
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
        "step_end", "run_end", "error", "delegation", "progress",
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


TodoStatus = Literal["pending", "in_progress", "done", "skipped"]


class Todo(BaseModel):
    """One item on the list an agent keeps while it works through a task."""

    content: str
    status: TodoStatus = "pending"
    note: str = ""

    @property
    def open(self) -> bool:
        return self.status in ("pending", "in_progress")

    def line(self) -> str:
        mark = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]",
                "skipped": "[-]"}[self.status]
        return f"{mark} {self.content}{f' — {self.note}' if self.note else ''}"


class Source(BaseModel):
    """Something an agent read and relied on. `id` is the number it is cited by."""

    id: int
    ref: str
    title: str = ""
    finding: str = ""
    agent: str = ""

    def line(self) -> str:
        label = f"{self.title} — {self.ref}" if self.title else self.ref
        return f"[{self.id}] {label}"


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
    budget_exceeded: str | None = None
    violations: list[str] = Field(default_factory=list)
    children: list[RunResult] = Field(default_factory=list)
    #: The mode the agent ran in, and what that mode kept while it worked.
    mode: str = ""
    depth: str = ""
    todos: list[Todo] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    #: The sandbox the run worked in, if it worked in one. With `session_id`,
    #: this is what picks the conversation back up where it left off.
    sandbox_id: str = ""
    #: Things that went wrong around the run without stopping it — a session
    #: that could not be saved, say. The answer is still the answer.
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def cost_usd(self) -> float:
        """This run plus everything it delegated."""
        return round(self.usage.cost_usd + sum(c.cost_usd for c in self.children), 8)

    def __str__(self) -> str:  # pragma: no cover - debugging affordance
        return self.output
