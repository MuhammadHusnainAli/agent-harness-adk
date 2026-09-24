"""Anthropic Messages API adapter.

The stream assembler here is shared by Bedrock and Vertex, which serve the same
event format over a different transport — so all three stream, and all three
keep thinking signatures intact for the next turn.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from ..errors import ProviderError
from ..types import (
    ImageBlock,
    Message,
    ModelResponse,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from .base import CompletionRequest, Provider, ProviderField, sse_events
from .resilience import classify

# These take `thinking: {"type": "adaptive"}`; older ones still want budget_tokens.
_ADAPTIVE_THINKING = ("claude-opus-5", "claude-opus-4", "claude-sonnet-5", "claude-fable",
                      "claude-mythos", "claude-sonnet-4-6")

_STOP_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "pause_turn": "tool_use",
    "refusal": "error",
}


#: What an SSE `error` event's type means, as the HTTP status it would have had.
_STREAM_ERROR_STATUS = {
    "overloaded_error": 529, "api_error": 500, "rate_limit_error": 429,
    "timeout_error": 504, "authentication_error": 401, "permission_error": 403,
    "not_found_error": 404, "request_too_large": 413, "invalid_request_error": 400,
}


class AnthropicStream:
    """Folds Anthropic stream events into StreamEvents and, at the end, a response.

    Transport-agnostic: the direct API feeds it from SSE, Bedrock from its binary
    event stream, Vertex from SSE again.
    """

    def __init__(self, provider: AnthropicProvider, model: str) -> None:
        self.provider = provider
        self.model = model
        self.blocks: dict[int, dict[str, Any]] = {}
        self.usage = Usage(calls=1)
        self.stop_reason = "end_turn"
        self.stop_detail = ""

    def feed(self, data: dict[str, Any]) -> list[StreamEvent]:
        etype = data.get("type")
        out: list[StreamEvent] = []
        if etype == "message_start":
            msg = data.get("message") or {}
            self.model = msg.get("model") or self.model
            self.usage = AnthropicProvider._usage(msg)
        elif etype == "content_block_start":
            slot = dict(data.get("content_block") or {})
            slot["_json"] = ""
            self.blocks[data.get("index", len(self.blocks))] = slot
        elif etype == "content_block_delta":
            delta = data.get("delta") or {}
            slot = self.blocks.setdefault(data.get("index", 0),
                                          {"type": "text", "text": "", "_json": ""})
            kind = delta.get("type")
            if kind == "text_delta":
                slot["text"] = slot.get("text", "") + delta.get("text", "")
                out.append(StreamEvent(type="text", text=delta.get("text", "")))
            elif kind == "thinking_delta":
                slot["thinking"] = slot.get("thinking", "") + delta.get("thinking", "")
                out.append(StreamEvent(type="thinking", text=delta.get("thinking", "")))
            elif kind == "signature_delta":
                # Without it the thinking block cannot be sent back next turn.
                slot["signature"] = slot.get("signature", "") + delta.get("signature", "")
            elif kind == "input_json_delta":
                slot["_json"] = slot.get("_json", "") + delta.get("partial_json", "")
        elif etype == "message_delta":
            delta = data.get("delta") or {}
            if delta.get("stop_reason"):
                self.stop_reason = _STOP_MAP.get(delta["stop_reason"], "end_turn")
            detail = delta.get("stop_details") or {}
            if isinstance(detail, dict) and detail.get("explanation"):
                self.stop_detail = detail["explanation"]
            u = data.get("usage") or {}
            # These counts are cumulative, so they replace rather than add.
            if "output_tokens" in u:
                self.usage.output_tokens = u["output_tokens"] or 0
            if u.get("input_tokens") is not None:
                self.usage.input_tokens = u["input_tokens"]
            if u.get("cache_read_input_tokens") is not None:
                self.usage.cache_read_tokens = u["cache_read_input_tokens"]
            if u.get("cache_creation_input_tokens") is not None:
                self.usage.cache_write_tokens = u["cache_creation_input_tokens"]
        elif etype == "error":
            raise self.provider._stream_error(data.get("error") or {})
        return out

    def finish(self) -> list[StreamEvent]:
        final: list[Any] = []
        out: list[StreamEvent] = []
        for _, slot in sorted(self.blocks.items()):
            if slot.get("_json"):
                try:
                    slot["input"] = json.loads(slot["_json"])
                except json.JSONDecodeError:
                    slot["input"] = {}
            slot.pop("_json", None)
            for block in AnthropicProvider._decode_blocks([slot]):
                final.append(block)
                if isinstance(block, ToolUseBlock):
                    out.append(StreamEvent(type="tool_call", data={
                        "id": block.id, "name": block.name, "input": block.input}))
        if self.stop_reason == "error":
            final.append(TextBlock(text=f"[refused] {self.stop_detail or 'request refused'}"))
        response = self.provider._finish(
            message=Message(role="assistant", content=final), stop_reason=self.stop_reason,
            usage=self.usage, model=self.model, raw={})
        out.append(StreamEvent(type="step_end",
                               data={"response": response.model_dump(mode="json")}))
        return out


class AnthropicProvider(Provider):
    name: ClassVar[str] = "anthropic"
    env_key: ClassVar[str] = "ANTHROPIC_API_KEY"
    default_model: ClassVar[str] = "claude-opus-5"
    BASE_URL: ClassVar[str] = "https://api.anthropic.com"
    api_version: ClassVar[str] = "2023-06-01"

    display_name: ClassVar[str] = "Anthropic"
    description: ClassVar[str] = "Claude models through the Anthropic Messages API."
    docs_url: ClassVar[str] = "https://docs.anthropic.com/en/api/messages"
    fields: ClassVar[tuple[ProviderField, ...]] = (
        ProviderField(name="api_key", type="secret", required=True,
                      env=("ANTHROPIC_API_KEY",), example="sk-ant-...",
                      description="An Anthropic API key from console.anthropic.com."),
        ProviderField(name="base_url", type="url", default="https://api.anthropic.com",
                      description="Override for a proxy or gateway."),
    )
    capabilities: ClassVar[frozenset[str]] = frozenset({
        "streaming", "tools", "vision", "thinking", "json_schema", "prompt_caching",
        "list_models"})

    def _auth_headers(self) -> dict[str, str]:
        self._require_key()
        return {"x-api-key": self.api_key, "anthropic-version": self.api_version}

    # ---- encoding -----------------------------------------------------
    @staticmethod
    def _encode_block(block: Any) -> dict[str, Any] | None:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}
        if isinstance(block, ThinkingBlock):
            redacted = getattr(block, "redacted", None)
            if redacted:
                return {"type": "redacted_thinking", "data": redacted}
            if not block.signature:
                return None     # unsigned thinking (another vendor's) is rejected
            return {"type": "thinking", "thinking": block.thinking,
                    "signature": block.signature}
        if isinstance(block, ToolUseBlock):
            return {"type": "tool_use", "id": block.id, "name": block.name,
                    "input": block.input}
        if isinstance(block, ToolResultBlock):
            return {"type": "tool_result", "tool_use_id": block.tool_use_id,
                    "content": block.content, "is_error": block.is_error}
        if isinstance(block, ImageBlock):
            source = ({"type": "url", "url": block.url} if block.url else
                      {"type": "base64", "media_type": block.media_type, "data": block.data})
            return {"type": "image", "source": source}
        return {"type": "text", "text": str(block)}

    def _encode_messages(self, messages: list[Message]) -> tuple[list[dict], list[str]]:
        """Anthropic keeps the system prompt out of the message list."""
        out: list[dict[str, Any]] = []
        systems: list[str] = []
        for msg in messages:
            if msg.role == "system":
                systems.append(msg.text)
                continue
            role = "assistant" if msg.role == "assistant" else "user"
            blocks = [e for e in (self._encode_block(b) for b in msg.content) if e]
            if not blocks:
                continue
            # Consecutive same-role turns are legal but merging keeps the cache prefix tidy.
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})
        return out, systems

    def _payload(self, req: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
        messages, inline_systems = self._encode_messages(req.messages)
        system_parts = [p for p in ([req.system] + inline_systems) if p]
        payload: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if req.tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in req.tools
            ]
            if req.tool_choice == "any":
                payload["tool_choice"] = {"type": "any"}
            elif req.tool_choice == "none":
                payload["tool_choice"] = {"type": "none"}
            elif isinstance(req.tool_choice, dict) and req.tool_choice.get("name"):
                payload["tool_choice"] = {"type": "tool", "name": req.tool_choice["name"]}
            elif req.tool_choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
        if req.stop:
            payload["stop_sequences"] = req.stop

        adaptive = req.model.startswith(_ADAPTIVE_THINKING)
        thinking_on = bool(req.thinking)
        if thinking_on:
            if adaptive:
                payload["thinking"] = {"type": "adaptive", "display": "summarized"}
            else:
                budget = req.thinking_budget or max(
                    1024, min(req.max_tokens - 1, req.max_tokens // 2))
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        # Sampling is rejected outright by the thinking-only models, so it is sent
        # only where it is accepted rather than being silently dropped by the API.
        if not adaptive:
            for field, key in (("temperature", "temperature"), ("top_p", "top_p"),
                               ("top_k", "top_k")):
                value = getattr(req, field)
                if value is not None:
                    payload[key] = value
        if req.effort:
            payload["output_config"] = {"effort": req.effort}
        if req.response_schema:
            oc = payload.setdefault("output_config", {})
            oc["format"] = {"type": "json_schema", "schema": req.response_schema}
        if req.parallel_tool_calls is False and req.tools:
            payload.setdefault("tool_choice", {"type": "auto"})
            payload["tool_choice"]["disable_parallel_tool_use"] = True
        if req.cache:
            payload["cache_control"] = {"type": "ephemeral"}
        if req.speed:
            payload["speed"] = req.speed
        if req.user or req.metadata:
            payload["metadata"] = {**req.metadata,
                                   **({"user_id": req.user} if req.user else {})}
        if stream:
            payload["stream"] = True
        payload.update(req.extra)
        return payload

    # ---- decoding -----------------------------------------------------
    @staticmethod
    def _decode_blocks(raw_blocks: list[dict[str, Any]]) -> list[Any]:
        blocks: list[Any] = []
        for b in raw_blocks:
            kind = b.get("type")
            if kind == "text":
                blocks.append(TextBlock(text=b.get("text", "")))
            elif kind == "thinking":
                blocks.append(ThinkingBlock(thinking=b.get("thinking", ""),
                                            signature=b.get("signature") or None))
            elif kind == "redacted_thinking":
                # Encrypted, but it must go back verbatim or the next turn fails.
                blocks.append(ThinkingBlock(thinking="", redacted=b.get("data", "")))
            elif kind == "tool_use":
                blocks.append(ToolUseBlock(id=b.get("id", ""), name=b.get("name", ""),
                                           input=b.get("input") or {}))
        return blocks

    @staticmethod
    def _usage(raw: dict[str, Any]) -> Usage:
        u = raw.get("usage") or {}
        return Usage(
            input_tokens=u.get("input_tokens") or 0,
            output_tokens=u.get("output_tokens") or 0,
            cache_read_tokens=u.get("cache_read_input_tokens") or 0,
            cache_write_tokens=u.get("cache_creation_input_tokens") or 0,
            calls=1,
        )

    def _decode(self, raw: dict[str, Any], req: CompletionRequest,
                latency_ms: float) -> ModelResponse:
        if raw.get("type") == "error":
            raise self._stream_error(raw.get("error") or {})
        blocks = self._decode_blocks(raw.get("content") or [])
        stop = _STOP_MAP.get(raw.get("stop_reason") or "end_turn", "end_turn")
        if stop == "error":
            detail = (raw.get("stop_details") or {}).get("explanation", "request refused")
            blocks.append(TextBlock(text=f"[refused] {detail}"))
        return self._finish(
            message=Message(role="assistant", content=blocks),
            stop_reason=stop,
            usage=self._usage(raw),
            model=raw.get("model") or req.model,
            raw=raw,
            latency_ms=latency_ms,
        )

    def _stream_error(self, error: dict[str, Any]) -> ProviderError:
        kind = str(error.get("type") or "api_error")
        return classify(self.name, _STREAM_ERROR_STATUS.get(kind, 500),
                        body=json.dumps({"error": error}), policy=self._policy())

    # ---- transport ----------------------------------------------------
    def _messages_path(self, req: CompletionRequest, stream: bool) -> str:
        return "/v1/messages"

    async def complete(self, req: CompletionRequest) -> ModelResponse:
        started = time.perf_counter()
        raw = await self._post(self._messages_path(req, False), self._payload(req),
                               timeout=req.timeout)
        return self._decode(raw, req, (time.perf_counter() - started) * 1000)

    async def _raw_events(self, req: CompletionRequest) -> AsyncIterator[dict[str, Any]]:
        """Decoded stream events, before assembly. Bedrock overrides this."""
        lines = self._stream_lines(self._messages_path(req, True),
                                   self._payload(req, stream=True), timeout=req.timeout)
        async for _, data in sse_events(lines):
            if data and data != "[DONE]":
                yield json.loads(data)

    async def _stream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        state = AnthropicStream(self, req.model)
        async for data in self._raw_events(req):
            for event in state.feed(data):
                yield event
        for event in state.finish():
            yield event

    async def list_models(self) -> list[str]:
        models: list[str] = []
        params: dict[str, str] = {"limit": "1000"}
        while True:
            raw = await self._get("/v1/models", params=params)
            models.extend(m["id"] for m in raw.get("data") or [] if m.get("id"))
            if not raw.get("has_more") or not raw.get("last_id"):
                return models
            params = {"limit": "1000", "after_id": raw["last_id"]}

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError(
            "Anthropic has no embeddings endpoint — use OpenAI/Gemini or a local embedder"
        )
