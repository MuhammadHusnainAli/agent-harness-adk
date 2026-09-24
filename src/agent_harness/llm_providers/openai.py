"""OpenAI Chat Completions adapter.

Also the base for everything that speaks the same wire format — Azure OpenAI and
the presets in `compatible.py` (Groq, Together, OpenRouter, DeepSeek, Mistral,
xAI, Ollama, vLLM, ...) — so they all share this encoding, streaming and retry path.
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

    display_name: ClassVar[str] = "OpenAI"
    description: ClassVar[str] = "GPT and o-series models through the Chat Completions API."
    docs_url: ClassVar[str] = "https://platform.openai.com/docs/api-reference/chat"
    fields: ClassVar[tuple[ProviderField, ...]] = (
        ProviderField(name="api_key", type="secret", required=True,
                      env=("OPENAI_API_KEY",), example="sk-...",
                      description="An OpenAI API key from platform.openai.com."),
        ProviderField(name="base_url", type="url", default="https://api.openai.com/v1",
                      description="Any OpenAI-compatible endpoint."),
        ProviderField(name="organization", env=("OPENAI_ORG_ID",),
                      description="Bill a specific organisation."),
        ProviderField(name="project", env=("OPENAI_PROJECT_ID",),
                      description="Bill a specific project."),
    )
    capabilities: ClassVar[frozenset[str]] = frozenset({
        "streaming", "tools", "vision", "thinking", "json_schema", "embeddings",
        "list_models"})

    #: False for local servers (Ollama, vLLM) that run without a key.
    key_required: ClassVar[bool] = True
    #: Ask for token usage at the end of a stream. Some servers reject the option.
    stream_usage: ClassVar[bool] = True
    #: Models that take `max_completion_tokens` and `reasoning_effort`.
    reasoning_prefixes: ClassVar[tuple[str, ...]] = _REASONING
    #: Payload keys this server does not accept, dropped before sending.
    drop_params: ClassVar[frozenset[str]] = frozenset()
    #: Payload keys this server spells differently: {"seed": "random_seed"}.
    rename_params: ClassVar[dict[str, str]] = {}

    def __init__(self, api_key: str | None = None, *, organization: str | None = None,
                 project: str | None = None, **kw: Any) -> None:
        super().__init__(api_key, **kw)
        import os

        self.organization = organization or (
            os.environ.get("OPENAI_ORG_ID", "") if type(self).name == "openai" else "")
        self.project = project or (
            os.environ.get("OPENAI_PROJECT_ID", "") if type(self).name == "openai" else "")

    def _auth_headers(self) -> dict[str, str]:
        if not self.api_key and not self.key_required:
            return {}
        self._require_key()
        headers = {"authorization": f"Bearer {self.api_key}"}
        if getattr(self, "organization", ""):
            headers["openai-organization"] = self.organization
        if getattr(self, "project", ""):
            headers["openai-project"] = self.project
        return headers

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
        reasoning = req.model.startswith(self.reasoning_prefixes)
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
            for field in ("temperature", "top_p", "frequency_penalty",
                          "presence_penalty", "seed"):
                value = getattr(req, field)
                if value is not None:
                    payload[field] = value
        elif req.effort:
            # OpenAI's reasoning models take low/medium/high; the two levels
            # above that map to the highest they have rather than erroring.
            payload["reasoning_effort"] = {"xhigh": "high", "max": "high"}.get(
                req.effort, req.effort
            )
        if req.parallel_tool_calls is not None and req.tools:
            payload["parallel_tool_calls"] = req.parallel_tool_calls
        if req.user:
            payload["user"] = req.user
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
            if self.stream_usage:
                payload["stream_options"] = {"include_usage": True}
        for key in self.drop_params:
            payload.pop(key, None)
        for old, new in self.rename_params.items():
            if old in payload:
                payload[new] = payload.pop(old)
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

    def _decode(self, raw: dict[str, Any], req: CompletionRequest,
                latency_ms: float = 0.0) -> ModelResponse:
        if raw.get("error"):
            raise self._body_error(raw["error"])
        choices = raw.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.name} returned no choices", provider=self.name,
                                body=json.dumps(raw)[:2000])
        choice = choices[0]
        msg = choice.get("message") or {}
        blocks: list[Any] = []
        # DeepSeek, vLLM and OpenRouter return the reasoning alongside the answer.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            blocks.append(ThinkingBlock(thinking=reasoning))
        if msg.get("content"):
            blocks.append(TextBlock(text=msg["content"]))
        elif msg.get("refusal"):
            blocks.append(TextBlock(text=f"[refused] {msg['refusal']}"))
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            blocks.append(_tool_use(call.get("id"), fn.get("name", ""), fn.get("arguments")))
        stop = _STOP_MAP.get(choice.get("finish_reason") or "stop", "end_turn")
        if msg.get("refusal") and not msg.get("content"):
            stop = "error"
        if any(isinstance(b, ToolUseBlock) for b in blocks) and stop == "end_turn":
            stop = "tool_use"     # some compatible servers say "stop" after a tool call
        return self._finish(
            message=Message(role="assistant", content=blocks),
            stop_reason=stop,
            usage=self._usage(raw),
            model=raw.get("model") or req.model,
            raw=raw,
            latency_ms=latency_ms,
        )

    def _body_error(self, error: Any) -> ProviderError:
        """An error delivered inside a 200 — mid-stream, or by a lenient proxy."""
        err = error if isinstance(error, dict) else {"message": str(error)}
        code = str(err.get("code") or err.get("type") or "")
        status = err.get("status") if isinstance(err.get("status"), int) else None
        if status is None:
            status = (429 if "rate_limit" in code else 400 if code in (
                "invalid_request_error", "context_length_exceeded") else 500)
        return classify(self.name, status, body=json.dumps({"error": err}),
                        policy=self._policy())

    # ---- transport ----------------------------------------------------
    def _chat_path(self, req: CompletionRequest) -> str:
        return "/chat/completions"

    async def complete(self, req: CompletionRequest) -> ModelResponse:
        started = time.perf_counter()
        raw = await self._post(self._chat_path(req), self._payload(req), timeout=req.timeout)
        return self._decode(raw, req, (time.perf_counter() - started) * 1000)

    async def _stream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        text = ""
        thinking = ""
        calls: dict[int, dict[str, Any]] = {}
        usage = Usage(calls=1)
        stop_reason = "end_turn"
        model = req.model

        lines = self._stream_lines(self._chat_path(req), self._payload(req, stream=True),
                                   timeout=req.timeout)
        async for _, chunk in sse_events(lines):
            if not chunk:
                continue
            if chunk.strip() == "[DONE]":
                break
            data = json.loads(chunk)
            if data.get("error"):
                raise self._body_error(data["error"])
            model = data.get("model") or model
            if data.get("usage"):
                usage = self._usage(data)
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    thinking += reasoning
                    yield StreamEvent(type="thinking", text=reasoning)
                if delta.get("content"):
                    text += delta["content"]
                    yield StreamEvent(type="text", text=delta["content"])
                for call in delta.get("tool_calls") or []:
                    idx = call.get("index", len(calls))
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

        blocks: list[Any] = [ThinkingBlock(thinking=thinking)] if thinking else []
        if text:
            blocks.append(TextBlock(text=text))
        for _, slot in sorted(calls.items()):
            block = _tool_use(slot["id"], slot["name"], slot["args"])
            blocks.append(block)
            yield StreamEvent(type="tool_call",
                              data={"id": block.id, "name": block.name, "input": block.input})
        if calls and stop_reason == "end_turn":
            stop_reason = "tool_use"

        response = self._finish(message=Message(role="assistant", content=blocks),
                                stop_reason=stop_reason, usage=usage, model=model, raw={})
        yield StreamEvent(type="step_end", data={"response": response.model_dump(mode="json")})

    def _embeddings_path(self, model: str) -> str:
        return "/embeddings"

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        name = model or self.embedding_model
        raw = await self._post(self._embeddings_path(name), {"model": name, "input": texts})
        rows = sorted(raw.get("data") or [], key=lambda d: d.get("index", 0))
        return [row["embedding"] for row in rows]

    async def list_models(self) -> list[str]:
        raw = await self._get("/models")
        return sorted(m["id"] for m in raw.get("data") or [] if m.get("id"))


def _tool_use(call_id: str | None, name: str, arguments: Any) -> ToolUseBlock:
    """Some compatible servers omit the call id; mint one so the result can match."""
    if call_id:
        return ToolUseBlock(id=call_id, name=name, input=_parse_args(arguments))
    return ToolUseBlock(name=name, input=_parse_args(arguments))


def _parse_args(arguments: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string — or, from some servers, already parsed."""
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
