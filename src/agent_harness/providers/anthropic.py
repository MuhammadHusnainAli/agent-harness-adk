"""Anthropic Messages API adapter."""

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
from .base import CompletionRequest, Provider

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


class AnthropicProvider(Provider):
    name: ClassVar[str] = "anthropic"
    env_key: ClassVar[str] = "ANTHROPIC_API_KEY"
    default_model: ClassVar[str] = "claude-opus-5"
    BASE_URL: ClassVar[str] = "https://api.anthropic.com"
    api_version: ClassVar[str] = "2023-06-01"

    def _auth_headers(self) -> dict[str, str]:
        self._require_key()
        return {"x-api-key": self.api_key, "anthropic-version": self.api_version}

    # ---- encoding -----------------------------------------------------
    @staticmethod
    def _encode_block(block: Any) -> dict[str, Any]:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}
        if isinstance(block, ThinkingBlock):
            out: dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
            if block.signature:
                out["signature"] = block.signature
            return out
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
            blocks = [self._encode_block(b) for b in msg.content]
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
                                            signature=b.get("signature")))
            elif kind == "tool_use":
                blocks.append(ToolUseBlock(id=b.get("id", ""), name=b.get("name", ""),
                                           input=b.get("input") or {}))
        return blocks

    @staticmethod
    def _usage(raw: dict[str, Any]) -> Usage:
        u = raw.get("usage") or {}
        return Usage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_read_tokens=u.get("cache_read_input_tokens", 0),
            cache_write_tokens=u.get("cache_creation_input_tokens", 0),
            calls=1,
        )

    async def complete(self, req: CompletionRequest) -> ModelResponse:
        started = time.perf_counter()
        raw = await self._post("/v1/messages", self._payload(req))
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
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def stream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        payload = self._payload(req, stream=True)
        headers = {"content-type": "application/json", "accept": "text/event-stream",
                   **self._auth_headers(), **self.extra_headers}
        blocks: dict[int, dict[str, Any]] = {}
        usage = Usage(calls=1)
        stop_reason = "end_turn"
        model = req.model

        async with self.http.stream(
            "POST", f"{self.base_url}/v1/messages", json=payload, headers=headers
        ) as resp:
            if resp.status_code >= 300:
                body = (await resp.aread()).decode()[:2000]
                raise ProviderError(f"anthropic stream failed {resp.status_code}: {body}",
                                    provider=self.name, status=resp.status_code, body=body)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = json.loads(line[5:].strip())
                etype = data.get("type")
                if etype == "message_start":
                    msg = data.get("message") or {}
                    model = msg.get("model", model)
                    usage += self._usage(msg)
                elif etype == "content_block_start":
                    blocks[data["index"]] = dict(data.get("content_block") or {})
                    blocks[data["index"]].setdefault("_json", "")
                elif etype == "content_block_delta":
                    delta = data.get("delta") or {}
                    slot = blocks.setdefault(data["index"], {"type": "text", "text": ""})
                    if delta.get("type") == "text_delta":
                        slot["text"] = slot.get("text", "") + delta.get("text", "")
                        yield StreamEvent(type="text", text=delta.get("text", ""))
                    elif delta.get("type") == "thinking_delta":
                        slot["thinking"] = slot.get("thinking", "") + delta.get("thinking", "")
                        yield StreamEvent(type="thinking", text=delta.get("thinking", ""))
                    elif delta.get("type") == "input_json_delta":
                        slot["_json"] = slot.get("_json", "") + delta.get("partial_json", "")
                elif etype == "message_delta":
                    stop_reason = _STOP_MAP.get(
                        (data.get("delta") or {}).get("stop_reason") or stop_reason, stop_reason
                    )
                    u = data.get("usage") or {}
                    usage.output_tokens += u.get("output_tokens", 0)
                elif etype == "error":
                    raise ProviderError(str(data.get("error")), provider=self.name)

        final: list[Any] = []
        for _, slot in sorted(blocks.items()):
            if slot.get("type") == "text":
                final.append(TextBlock(text=slot.get("text", "")))
            elif slot.get("type") == "thinking":
                final.append(ThinkingBlock(thinking=slot.get("thinking", ""),
                                           signature=slot.get("signature")))
            elif slot.get("type") == "tool_use":
                args = slot.get("input") or {}
                if slot.get("_json"):
                    try:
                        args = json.loads(slot["_json"])
                    except json.JSONDecodeError:
                        args = {}
                block = ToolUseBlock(id=slot.get("id", ""), name=slot.get("name", ""),
                                     input=args)
                final.append(block)
                yield StreamEvent(type="tool_call",
                                  data={"id": block.id, "name": block.name, "input": args})

        response = self._finish(
            message=Message(role="assistant", content=final), stop_reason=stop_reason,
            usage=usage, model=model, raw={},
        )
        yield StreamEvent(type="step_end", data={"response": response.model_dump(mode="json")})

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError(
            "Anthropic has no embeddings endpoint — use OpenAI/Gemini or a local embedder"
        )

