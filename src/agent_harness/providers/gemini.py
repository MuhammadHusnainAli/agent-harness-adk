"""Google Gemini (Generative Language API) adapter."""

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
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "error",
    "RECITATION": "error",
    "OTHER": "end_turn",
}

# Gemini takes an OpenAPI subset and 400s on stray JSON Schema keywords.
_SCHEMA_DROP = {"$schema", "additionalProperties", "definitions", "$defs", "title",
                "default", "examples", "exclusiveMinimum", "exclusiveMaximum",
                "const", "allOf", "oneOf", "not", "patternProperties"}


def _clean_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        out = {k: _clean_schema(v) for k, v in schema.items() if k not in _SCHEMA_DROP}
        if out.get("type") == "object" and "properties" not in out:
            out["properties"] = {}
        return out
    if isinstance(schema, list):
        return [_clean_schema(v) for v in schema]
    return schema


class GeminiProvider(Provider):
    name: ClassVar[str] = "gemini"
    env_key: ClassVar[str] = "GEMINI_API_KEY"
    default_model: ClassVar[str] = "gemini-2.5-pro"
    BASE_URL: ClassVar[str] = "https://generativelanguage.googleapis.com/v1beta"
    embedding_model: ClassVar[str] = "text-embedding-004"

    def __init__(self, api_key: str | None = None, **kw: Any) -> None:
        super().__init__(api_key, **kw)
        if not self.api_key:
            import os
            self.api_key = os.environ.get("GOOGLE_API_KEY", "")

    def _auth_headers(self) -> dict[str, str]:
        self._require_key()
        return {"x-goog-api-key": self.api_key}

    # ---- encoding -----------------------------------------------------
    @staticmethod
    def _tool_names(messages: list[Message]) -> dict[str, str]:
        """Gemini's functionResponse is keyed by name, not by call id."""
        return {b.id: b.name for m in messages for b in m.content
                if isinstance(b, ToolUseBlock)}

    def _encode_contents(self, req: CompletionRequest) -> tuple[list[dict], list[str]]:
        names = self._tool_names(req.messages)
        contents: list[dict[str, Any]] = []
        systems: list[str] = []
        for msg in req.messages:
            if msg.role == "system":
                systems.append(msg.text)
                continue
            role = "model" if msg.role == "assistant" else "user"
            parts: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    if block.text:
                        parts.append({"text": block.text})
                elif isinstance(block, ThinkingBlock):
                    continue
                elif isinstance(block, ToolUseBlock):
                    parts.append({"functionCall": {"name": block.name, "args": block.input}})
                elif isinstance(block, ToolResultBlock):
                    parts.append({"functionResponse": {
                        "name": names.get(block.tool_use_id, "tool"),
                        "response": {"output": block.content,
                                     "error": block.is_error or None},
                    }})
                elif isinstance(block, ImageBlock):
                    parts.append({"inlineData": {"mimeType": block.media_type,
                                                 "data": block.data}})
            if not parts:
                continue
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})
        return contents, systems

    def _payload(self, req: CompletionRequest) -> dict[str, Any]:
        contents, inline_systems = self._encode_contents(req)
        system_parts = [p for p in ([req.system] + inline_systems) if p]
        gen: dict[str, Any] = {"maxOutputTokens": req.max_tokens}
        for field, key in (("temperature", "temperature"), ("top_p", "topP"),
                           ("top_k", "topK"), ("seed", "seed"),
                           ("frequency_penalty", "frequencyPenalty"),
                           ("presence_penalty", "presencePenalty")):
            value = getattr(req, field)
            if value is not None:
                gen[key] = value
        if req.stop:
            gen["stopSequences"] = req.stop
        if req.response_mime_type:
            gen["responseMimeType"] = req.response_mime_type
        if req.response_schema:
            gen["responseMimeType"] = "application/json"
            gen["responseSchema"] = _clean_schema(req.response_schema)
        if req.thinking is not None or req.effort or req.thinking_budget:
            budget = req.thinking_budget
            if budget is None and req.effort:
                # Gemini takes a token budget, not a level, so the levels are
                # mapped to budgets rather than being dropped.
                budget = {"low": 1024, "medium": 8192, "high": 16384,
                          "xhigh": 24576, "max": 32768}[req.effort]
            gen["thinkingConfig"] = {
                "includeThoughts": bool(req.thinking),
                **({"thinkingBudget": budget} if budget is not None else {}),
            }

        payload: dict[str, Any] = {"contents": contents, "generationConfig": gen}
        if req.safety_settings:
            payload["safetySettings"] = req.safety_settings
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if req.tools:
            payload["tools"] = [{"functionDeclarations": [
                {"name": t.name, "description": t.description,
                 "parameters": _clean_schema(t.parameters)}
                for t in req.tools
            ]}]
            mode = {"any": "ANY", "none": "NONE"}.get(str(req.tool_choice), "AUTO")
            cfg: dict[str, Any] = {"mode": mode}
            if isinstance(req.tool_choice, dict) and req.tool_choice.get("name"):
                cfg = {"mode": "ANY", "allowedFunctionNames": [req.tool_choice["name"]]}
            payload["toolConfig"] = {"functionCallingConfig": cfg}
        payload.update(req.extra)
        return payload

    # ---- decoding -----------------------------------------------------
    @staticmethod
    def _usage(raw: dict[str, Any]) -> Usage:
        u = raw.get("usageMetadata") or {}
        cached = u.get("cachedContentTokenCount", 0)
        return Usage(
            input_tokens=max(u.get("promptTokenCount", 0) - cached, 0),
            output_tokens=u.get("candidatesTokenCount", 0),
            cache_read_tokens=cached,
            calls=1,
        )

    @staticmethod
    def _decode_parts(parts: list[dict[str, Any]]) -> list[Any]:
        blocks: list[Any] = []
        for part in parts:
            if "text" in part and part["text"]:
                if part.get("thought"):
                    blocks.append(ThinkingBlock(thinking=part["text"]))
                else:
                    blocks.append(TextBlock(text=part["text"]))
            elif "functionCall" in part:
                call = part["functionCall"]
                blocks.append(ToolUseBlock(name=call.get("name", ""),
                                           input=call.get("args") or {}))
        return blocks

    async def complete(self, req: CompletionRequest) -> ModelResponse:
        started = time.perf_counter()
        raw = await self._post(f"/models/{req.model}:generateContent", self._payload(req))
        candidates = raw.get("candidates") or []
        if not candidates:
            reason = (raw.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ProviderError(f"gemini returned nothing: {reason}", provider=self.name)
        cand = candidates[0]
        blocks = self._decode_parts((cand.get("content") or {}).get("parts") or [])
        stop = _STOP_MAP.get(cand.get("finishReason") or "STOP", "end_turn")
        if any(isinstance(b, ToolUseBlock) for b in blocks):
            stop = "tool_use"
        return self._finish(
            message=Message(role="assistant", content=blocks),
            stop_reason=stop,
            usage=self._usage(raw),
            model=raw.get("modelVersion") or req.model,
            raw=raw,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    async def stream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        payload = self._payload(req)
        headers = {"content-type": "application/json", "accept": "text/event-stream",
                   **self._auth_headers(), **self.extra_headers}
        url = f"{self.base_url}/models/{req.model}:streamGenerateContent"
        blocks: list[Any] = []
        usage = Usage(calls=1)
        stop_reason = "end_turn"
        model = req.model

        async with self.http.stream("POST", url, json=payload, headers=headers,
                                    params={"alt": "sse"}) as resp:
            if resp.status_code >= 300:
                body = (await resp.aread()).decode()[:2000]
                raise ProviderError(f"gemini stream failed {resp.status_code}: {body}",
                                    provider=self.name, status=resp.status_code, body=body)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = json.loads(line[5:].strip())
                model = data.get("modelVersion", model)
                if data.get("usageMetadata"):
                    usage = self._usage(data)  # Gemini reports cumulative totals
                for cand in data.get("candidates") or []:
                    for block in self._decode_parts(
                        (cand.get("content") or {}).get("parts") or []
                    ):
                        blocks.append(block)
                        if isinstance(block, TextBlock):
                            yield StreamEvent(type="text", text=block.text)
                        elif isinstance(block, ThinkingBlock):
                            yield StreamEvent(type="thinking", text=block.thinking)
                        elif isinstance(block, ToolUseBlock):
                            yield StreamEvent(type="tool_call",
                                              data={"id": block.id, "name": block.name,
                                                    "input": block.input})
                    if cand.get("finishReason"):
                        stop_reason = _STOP_MAP.get(cand["finishReason"], "end_turn")

        if any(isinstance(b, ToolUseBlock) for b in blocks):
            stop_reason = "tool_use"
        # Merge the text fragments back into one block so history stays compact.
        merged: list[Any] = []
        for block in blocks:
            if (isinstance(block, TextBlock) and merged
                    and isinstance(merged[-1], TextBlock)):
                merged[-1] = TextBlock(text=merged[-1].text + block.text)
            else:
                merged.append(block)
        response = self._finish(message=Message(role="assistant", content=merged),
                                stop_reason=stop_reason, usage=usage, model=model, raw={})
        yield StreamEvent(type="step_end", data={"response": response.model_dump(mode="json")})

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        name = model or self.embedding_model
        payload = {"requests": [
            {"model": f"models/{name}", "content": {"parts": [{"text": t}]}} for t in texts
        ]}
        raw = await self._post(f"/models/{name}:batchEmbedContents", payload)
        return [e.get("values", []) for e in raw.get("embeddings") or []]
