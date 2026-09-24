"""Google Gemini (Generative Language API) adapter.

Decoding and streaming are shared with Gemini on Vertex AI, which serves the same
format behind a different URL and auth.
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
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "error",
    "RECITATION": "error",
    "BLOCKLIST": "error",
    "PROHIBITED_CONTENT": "error",
    "SPII": "error",
    "IMAGE_SAFETY": "error",
    "MALFORMED_FUNCTION_CALL": "error",
    "OTHER": "end_turn",
}

#: google.rpc status names, as the HTTP status they stand for.
_RPC_STATUS = {"RESOURCE_EXHAUSTED": 429, "UNAVAILABLE": 503, "INTERNAL": 500,
               "DEADLINE_EXCEEDED": 504, "INVALID_ARGUMENT": 400, "NOT_FOUND": 404,
               "PERMISSION_DENIED": 403, "UNAUTHENTICATED": 401}

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
    embedding_model: ClassVar[str] = "gemini-embedding-001"

    display_name: ClassVar[str] = "Google Gemini"
    description: ClassVar[str] = "Gemini models through the Generative Language API (AI Studio)."
    docs_url: ClassVar[str] = "https://ai.google.dev/api/generate-content"
    fields: ClassVar[tuple[ProviderField, ...]] = (
        ProviderField(name="api_key", type="secret", required=True,
                      env=("GEMINI_API_KEY", "GOOGLE_API_KEY"), example="AIza...",
                      description="A Gemini API key from aistudio.google.com."),
        ProviderField(name="base_url", type="url",
                      default="https://generativelanguage.googleapis.com/v1beta",
                      description="Override for a proxy or gateway."),
    )
    capabilities: ClassVar[frozenset[str]] = frozenset({
        "streaming", "tools", "vision", "thinking", "json_schema", "embeddings",
        "list_models"})

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
                part: dict[str, Any] | None = None
                if isinstance(block, TextBlock):
                    if block.text:
                        part = {"text": block.text}
                elif isinstance(block, ThinkingBlock):
                    continue
                elif isinstance(block, ToolUseBlock):
                    part = {"functionCall": {"name": block.name, "args": block.input}}
                if part is not None:
                    # Thinking models need their signature back on the next turn.
                    signature = getattr(block, "thought_signature", None)
                    if signature:
                        part["thoughtSignature"] = signature
                    parts.append(part)
                elif isinstance(block, ToolResultBlock):
                    parts.append({"functionResponse": {
                        "name": names.get(block.tool_use_id, "tool"),
                        "response": {"output": block.content,
                                     "error": block.is_error or None},
                    }})
                elif isinstance(block, ImageBlock):
                    if block.data:
                        parts.append({"inlineData": {"mimeType": block.media_type,
                                                     "data": block.data}})
                    elif block.url:
                        parts.append({"fileData": {"mimeType": block.media_type,
                                                   "fileUri": block.url}})
            if not parts:
                continue
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})
        return contents, systems

    def _payload(self, req: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
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
            extra = ({"thought_signature": part["thoughtSignature"]}
                     if part.get("thoughtSignature") else {})
            if "functionCall" in part:
                call = part["functionCall"]
                blocks.append(ToolUseBlock(**({"id": call["id"]} if call.get("id") else {}),
                                           name=call.get("name", ""),
                                           input=call.get("args") or {}, **extra))
            elif part.get("text"):
                if part.get("thought"):
                    blocks.append(ThinkingBlock(thinking=part["text"]))
                else:
                    blocks.append(TextBlock(text=part["text"], **extra))
            elif extra and blocks:
                # A bare signature part belongs to the block before it.
                blocks[-1].thought_signature = extra["thought_signature"]
        return blocks

    def _decode(self, raw: dict[str, Any], req: CompletionRequest,
                latency_ms: float = 0.0) -> ModelResponse:
        if raw.get("error"):
            raise self._body_error(raw["error"])
        candidates = raw.get("candidates") or []
        if not candidates:
            reason = (raw.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ProviderError(f"{self.name} returned nothing: {reason}", provider=self.name,
                                body=json.dumps(raw)[:2000])
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
            latency_ms=latency_ms,
        )

    def _body_error(self, error: Any) -> ProviderError:
        err = error if isinstance(error, dict) else {"message": str(error)}
        status = err.get("code") if isinstance(err.get("code"), int) else _RPC_STATUS.get(
            str(err.get("status")), 500)
        return classify(self.name, status, body=json.dumps({"error": err}),
                        policy=self._policy())

    # ---- transport ----------------------------------------------------
    def _model_path(self, model: str, stream: bool) -> str:
        name = model.removeprefix("models/")
        return f"/models/{name}:{'streamGenerateContent' if stream else 'generateContent'}"

    async def complete(self, req: CompletionRequest) -> ModelResponse:
        started = time.perf_counter()
        raw = await self._post(self._model_path(req.model, False), self._payload(req),
                               timeout=req.timeout)
        return self._decode(raw, req, (time.perf_counter() - started) * 1000)

    async def _stream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        blocks: list[Any] = []
        usage = Usage(calls=1)
        stop_reason = "end_turn"
        model = req.model

        lines = self._stream_lines(self._model_path(req.model, True),
                                   self._payload(req, stream=True),
                                   params={"alt": "sse"}, timeout=req.timeout)
        async for _, chunk in sse_events(lines):
            if not chunk:
                continue
            data = json.loads(chunk)
            if isinstance(data, list):          # a JSON array rather than SSE framing
                data = data[0] if data else {}
            if data.get("error"):
                raise self._body_error(data["error"])
            model = data.get("modelVersion") or model
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
            if (type(block) is TextBlock and merged and type(merged[-1]) is TextBlock
                    and not getattr(merged[-1], "thought_signature", None)):
                signature = getattr(block, "thought_signature", None)
                merged[-1] = TextBlock(text=merged[-1].text + block.text,
                                       **({"thought_signature": signature}
                                          if signature else {}))
            elif (type(block) is ThinkingBlock and merged
                    and type(merged[-1]) is ThinkingBlock):
                merged[-1] = ThinkingBlock(thinking=merged[-1].thinking + block.thinking)
            else:
                merged.append(block)
        response = self._finish(message=Message(role="assistant", content=merged),
                                stop_reason=stop_reason, usage=usage, model=model, raw={})
        yield StreamEvent(type="step_end", data={"response": response.model_dump(mode="json")})

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        name = (model or self.embedding_model).removeprefix("models/")
        payload = {"requests": [
            {"model": f"models/{name}", "content": {"parts": [{"text": t}]}} for t in texts
        ]}
        raw = await self._post(f"/models/{name}:batchEmbedContents", payload)
        return [e.get("values", []) for e in raw.get("embeddings") or []]

    async def list_models(self) -> list[str]:
        models: list[str] = []
        params: dict[str, str] = {"pageSize": "1000"}
        while True:
            raw = await self._get("/models", params=params)
            models.extend(m["name"].removeprefix("models/")
                          for m in raw.get("models") or [] if m.get("name"))
            token = raw.get("nextPageToken")
            if not token:
                return models
            params = {"pageSize": "1000", "pageToken": token}
