"""OpenAI Chat Completions adapter.

Also works against anything that speaks the same wire format (Azure OpenAI,
Groq, Together, Ollama, vLLM) by passing a different `base_url`.
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
from .base import CompletionRequest, Provider

_STOP_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "error",
}

# Reasoning models reject `temperature` and want `max_completion_tokens`.
_REASONING = ("o1", "o3", "o4", "gpt-5")


class OpenAIProvider(Provider):
    name: ClassVar[str] = "openai"
    env_key: ClassVar[str] = "OPENAI_API_KEY"
    default_model: ClassVar[str] = "gpt-4.1"
    BASE_URL: ClassVar[str] = "https://api.openai.com/v1"
    embedding_model: ClassVar[str] = "text-embedding-3-small"

    def _auth_headers(self) -> dict[str, str]:
        self._require_key()
        return {"authorization": f"Bearer {self.api_key}"}

    # ---- encoding -----------------------------------------------------
    def _encode_messages(self, req: CompletionRequest) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if req.system:
            out.append({"role": "system", "content": req.system})
        for msg in req.messages:
            if msg.role == "system":
                out.append({"role": "system", "content": msg.text})
                continue

            # Tool results become their own `tool` messages, one per result.
            results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            for res in results:
                out.append({"role": "tool", "tool_call_id": res.tool_use_id,
                            "content": res.content or ("error" if res.is_error else "")})

            parts: list[dict[str, Any]] = []
            text_chunks: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_chunks.append(block.text)
                    parts.append({"type": "text", "text": block.text})
                elif isinstance(block, ThinkingBlock):
                    continue  # provider-specific; never replay across vendors
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append({
                        "id": block.id, "type": "function",
                        "function": {"name": block.name,
                                     "arguments": json.dumps(block.input)},
                    })
                elif isinstance(block, ImageBlock):
                    url = block.url or f"data:{block.media_type};base64,{block.data}"
                    parts.append({"type": "image_url", "image_url": {"url": url}})

            if msg.role == "assistant":
                if not text_chunks and not tool_calls:
                    continue
                entry: dict[str, Any] = {"role": "assistant",
                                         "content": "\n".join(text_chunks) or None}
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                out.append(entry)
            else:
                if not parts:
                    continue
                has_image = any(p["type"] == "image_url" for p in parts)
                out.append({"role": "user",
                            "content": parts if has_image else "\n".join(text_chunks)})
        return out

    def _payload(self, req: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
        reasoning = req.model.startswith(_REASONING)
        payload: dict[str, Any] = {
            "model": req.model,
            "messages": self._encode_messages(req),
        }
        payload["max_completion_tokens" if reasoning else "max_tokens"] = req.max_tokens
        if req.tools:
            payload["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.parameters}}
                for t in req.tools
            ]
            if req.tool_choice == "any":
                payload["tool_choice"] = "required"
            elif req.tool_choice in ("auto", "none"):
                payload["tool_choice"] = req.tool_choice
            elif isinstance(req.tool_choice, dict) and req.tool_choice.get("name"):
                payload["tool_choice"] = {
                    "type": "function", "function": {"name": req.tool_choice["name"]}
                }
        if not reasoning:
            if req.temperature is not None:
                payload["temperature"] = req.temperature
            if req.top_p is not None:
                payload["top_p"] = req.top_p
        elif req.effort:
            payload["reasoning_effort"] = {"xhigh": "high", "max": "high"}.get(
                req.effort, req.effort
            )
        if req.stop:
            payload["stop"] = req.stop
        if req.response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": req.response_schema,
                                "strict": False},
            }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        payload.update(req.extra)
        return payload

    # ---- decoding -----------------------------------------------------
    @staticmethod
    def _usage(raw: dict[str, Any]) -> Usage:
        u = raw.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        return Usage(
            input_tokens=max(u.get("prompt_tokens", 0) - cached, 0),
            output_tokens=u.get("completion_tokens", 0),
            cache_read_tokens=cached,
            calls=1,
        )

    async def complete(self, req: CompletionRequest) -> ModelResponse:
        started = time.perf_counter()
        raw = await self._post("/chat/completions", self._payload(req))
        choices = raw.get("choices") or []
        if not choices:
            raise ProviderError("openai returned no choices", provider=self.name)
        choice = choices[0]
        msg = choice.get("message") or {}
        blocks: list[Any] = []
        if msg.get("content"):
            blocks.append(TextBlock(text=msg["content"]))
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            blocks.append(ToolUseBlock(id=call.get("id", ""), name=fn.get("name", ""),
                                       input=args))
        return self._finish(
            message=Message(role="assistant", content=blocks),
            stop_reason=_STOP_MAP.get(choice.get("finish_reason") or "stop", "end_turn"),
            usage=self._usage(raw),
            model=raw.get("model") or req.model,
            raw=raw,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def stream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        payload = self._payload(req, stream=True)
        headers = {"content-type": "application/json", "accept": "text/event-stream",
                   **self._auth_headers(), **self.extra_headers}
        text = ""
        calls: dict[int, dict[str, Any]] = {}
        usage = Usage(calls=1)
        stop_reason = "end_turn"
        model = req.model

        async with self.http.stream(
            "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
        ) as resp:
            if resp.status_code >= 300:
                body = (await resp.aread()).decode()[:2000]
                raise ProviderError(f"openai stream failed {resp.status_code}: {body}",
                                    provider=self.name, status=resp.status_code, body=body)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                data = json.loads(chunk)
                model = data.get("model", model)
                if data.get("usage"):
                    usage += self._usage(data)
                for choice in data.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text += delta["content"]
                        yield StreamEvent(type="text", text=delta["content"])
                    for call in delta.get("tool_calls") or []:
                        idx = call.get("index", 0)
                        slot = calls.setdefault(idx, {"id": "", "name": "", "args": ""})
                        if call.get("id"):
                            slot["id"] = call["id"]
                        fn = call.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
                    if choice.get("finish_reason"):
                        stop_reason = _STOP_MAP.get(choice["finish_reason"], "end_turn")

        blocks: list[Any] = [TextBlock(text=text)] if text else []
        for _, slot in sorted(calls.items()):
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            block = ToolUseBlock(id=slot["id"], name=slot["name"], input=args)
            blocks.append(block)
            yield StreamEvent(type="tool_call",
                              data={"id": block.id, "name": block.name, "input": args})

        response = self._finish(message=Message(role="assistant", content=blocks),
                                stop_reason=stop_reason, usage=usage, model=model, raw={})
        yield StreamEvent(type="step_end", data={"response": response.model_dump(mode="json")})

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raw = await self._post("/embeddings", {"model": model or self.embedding_model,
                                               "input": texts})
        rows = sorted(raw.get("data") or [], key=lambda d: d.get("index", 0))
        return [row["embedding"] for row in rows]
