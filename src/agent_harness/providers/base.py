"""The provider contract every model backend implements.

One shape in, one shape out. The agent loop never learns which vendor it is
talking to — that is the whole point of this file.

Adapters speak raw HTTP over ``httpx`` rather than pulling in three vendor SDKs.
It keeps the wheel small and means Anthropic, OpenAI and Gemini all travel the
same retry, timeout and cost-accounting path.
"""

from __future__ import annotations

import asyncio
import os
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # httpx is imported on the first call, not on import
    import httpx

from ..errors import ProviderError, RateLimitError
from ..types import Message, ModelResponse, StreamEvent, Usage

__all__ = [
    "ToolSchema",
    "ModelInfo",
    "CompletionRequest",
    "Provider",
    "MODELS",
    "model_info",
    "estimate_cost",
    "register_model",
]


class ToolSchema(BaseModel):
    """A tool as the model sees it: a name, a reason to call it, and a shape."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class ModelInfo(BaseModel):
    """What the harness needs to know about a model to route and bill it."""

    id: str
    provider: str
    context_window: int = 200_000
    max_output: int = 8192
    input_cost: float = 0.0  # USD per 1M tokens
    output_cost: float = 0.0
    cache_read_cost: float | None = None
    cache_write_cost: float | None = None
    supports_tools: bool = True
    supports_thinking: bool = False
    tier: Literal["fast", "balanced", "deep"] = "balanced"


# Prices are USD per million tokens. Cached 2026-06; override with register_model().
MODELS: dict[str, ModelInfo] = {}


def register_model(info: ModelInfo) -> ModelInfo:
    """Teach the harness about a model it does not ship with."""
    MODELS[info.id] = info
    return info


def _m(**kw: Any) -> None:
    register_model(ModelInfo(**kw))


# --- Anthropic ---------------------------------------------------------------
_m(id="claude-opus-5", provider="anthropic", context_window=1_000_000, max_output=128_000,
   input_cost=5.0, output_cost=25.0, cache_read_cost=0.5, cache_write_cost=6.25,
   supports_thinking=True, tier="deep")
_m(id="claude-fable-5-1", provider="anthropic", context_window=1_000_000, max_output=128_000,
   input_cost=10.0, output_cost=50.0, cache_read_cost=0.25, cache_write_cost=12.5,
   supports_thinking=True, tier="deep")
_m(id="claude-opus-4-8", provider="anthropic", context_window=1_000_000, max_output=128_000,
   input_cost=5.0, output_cost=25.0, cache_read_cost=0.5, cache_write_cost=6.25,
   supports_thinking=True, tier="deep")
_m(id="claude-sonnet-5", provider="anthropic", context_window=1_000_000, max_output=128_000,
   input_cost=2.0, output_cost=10.0, cache_read_cost=0.2, cache_write_cost=2.5,
   supports_thinking=True, tier="balanced")
_m(id="claude-haiku-4-5", provider="anthropic", context_window=200_000, max_output=64_000,
   input_cost=1.0, output_cost=5.0, cache_read_cost=0.1, cache_write_cost=1.25,
   supports_thinking=True, tier="fast")

# --- OpenAI ------------------------------------------------------------------
_m(id="gpt-4.1", provider="openai", context_window=1_000_000, max_output=32_768,
   input_cost=2.0, output_cost=8.0, cache_read_cost=0.5, tier="balanced")
_m(id="gpt-4.1-mini", provider="openai", context_window=1_000_000, max_output=32_768,
   input_cost=0.4, output_cost=1.6, cache_read_cost=0.1, tier="fast")
_m(id="gpt-4o", provider="openai", context_window=128_000, max_output=16_384,
   input_cost=2.5, output_cost=10.0, cache_read_cost=1.25, tier="balanced")
_m(id="gpt-4o-mini", provider="openai", context_window=128_000, max_output=16_384,
   input_cost=0.15, output_cost=0.6, cache_read_cost=0.075, tier="fast")
_m(id="o3", provider="openai", context_window=200_000, max_output=100_000,
   input_cost=2.0, output_cost=8.0, supports_thinking=True, tier="deep")

# --- Google ------------------------------------------------------------------
_m(id="gemini-2.5-pro", provider="gemini", context_window=1_048_576, max_output=65_536,
   input_cost=1.25, output_cost=10.0, supports_thinking=True, tier="deep")
_m(id="gemini-2.5-flash", provider="gemini", context_window=1_048_576, max_output=65_536,
   input_cost=0.3, output_cost=2.5, supports_thinking=True, tier="balanced")
_m(id="gemini-2.0-flash", provider="gemini", context_window=1_048_576, max_output=8192,
   input_cost=0.1, output_cost=0.4, tier="fast")


def model_info(model: str) -> ModelInfo | None:
    """Exact match first, then longest known prefix — so dated snapshots still bill.

    Platform-prefixed ids (`anthropic.claude-opus-5` on Bedrock) resolve to the
    same model, so cost accounting works wherever it is served from.
    """
    for candidate in (model, normalise_model(model)):
        if candidate in MODELS:
            return MODELS[candidate]
    best: ModelInfo | None = None
    for candidate in (model, normalise_model(model)):
        for key, info in MODELS.items():
            if candidate.startswith(key) and (best is None or len(key) > len(best.id)):
                best = info
        if best is not None:
            return best
    return best


def normalise_model(model: str) -> str:
    """Strip the platform prefixes so pricing and routing still recognise a model.

    Bedrock and Vertex rename the same models: `anthropic.claude-opus-5`,
    `us.anthropic.claude-opus-5`, `claude-opus-5@20260401`. They are the same
    model and should cost and route the same.
    """
    name = model.split("@", 1)[0]                   # Vertex version suffix
    for prefix in ("anthropic.", "us.anthropic.", "eu.anthropic.",
                   "apac.anthropic.", "google.", "publishers/anthropic/models/",
                   "publishers/google/models/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name.startswith(("us.", "eu.", "apac.")):    # cross-region inference profiles
        name = name.split(".", 1)[1]
    return name


def provider_for_model(model: str) -> str:
    """Infer the vendor from the model id, so callers rarely have to say it."""
    info = model_info(model)
    if info:
        return info.provider
    low = model.lower()
    if low.startswith("claude"):
        return "anthropic"
    if low.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if low.startswith(("gemini", "models/gemini")):
        return "gemini"
    return "anthropic"


def estimate_cost(model: str, usage: Usage) -> float:
    """Dollars for one call. Unknown models cost nothing — never guess a price up."""
    info = model_info(model)
    if info is None:
        return 0.0
    cost = (
        usage.input_tokens * info.input_cost
        + usage.output_tokens * info.output_cost
        + usage.cache_read_tokens * (info.cache_read_cost or info.input_cost)
        + usage.cache_write_tokens * (info.cache_write_cost or info.input_cost)
    ) / 1_000_000
    return round(cost, 8)


Effort = Literal["low", "medium", "high", "xhigh", "max"]


class CompletionRequest(BaseModel):
    """Everything one model call needs, in a form every adapter accepts.

    A parameter a provider does not have is dropped rather than guessed at —
    Anthropic has no `seed`, OpenAI has no `top_k`, and inventing an equivalent
    would change what you asked for. `extra` goes into the payload untouched, so
    anything this does not cover is still reachable.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # --- what to send -------------------------------------------------------
    model: str
    messages: list[Message] = Field(default_factory=list)
    system: str | None = None
    tools: list[ToolSchema] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None  # auto | any | none | {"name": ...}

    # --- how much ------------------------------------------------------------
    max_tokens: int = 8192

    # --- how it should think --------------------------------------------------
    #: Reasoning depth. `low` for mechanical work, `high` for judgement, `max`
    #: when correctness matters more than cost.
    effort: Effort | None = None
    thinking: bool | None = None
    #: An explicit thinking-token ceiling, for models that take one instead of
    #: an effort level.
    thinking_budget: int | None = None

    # --- sampling --------------------------------------------------------------
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None                    # Anthropic, Gemini
    seed: int | None = None                     # OpenAI, Gemini
    frequency_penalty: float | None = None      # OpenAI, Gemini
    presence_penalty: float | None = None       # OpenAI, Gemini
    stop: list[str] = Field(default_factory=list)

    # --- shape of the answer -----------------------------------------------------
    response_schema: dict[str, Any] | None = None
    response_mime_type: str | None = None       # Gemini
    parallel_tool_calls: bool | None = None

    # --- cost and operations -------------------------------------------------------
    cache: bool | None = None                   # prompt caching, where supported
    speed: str | None = None                    # Anthropic fast mode
    user: str | None = None                     # end-user id, for abuse tracing
    metadata: dict[str, Any] = Field(default_factory=dict)
    safety_settings: list[dict[str, Any]] = Field(default_factory=list)  # Gemini
    timeout: float | None = None

    #: Merged into the payload verbatim. The escape hatch for anything above.
    extra: dict[str, Any] = Field(default_factory=dict)

    def merged(self, **overrides: Any) -> CompletionRequest:
        """A copy with some fields changed — used when walking a fallback chain."""
        return self.model_copy(update=overrides)


class Provider(ABC):
    """Base class for model backends. Subclasses implement `complete`."""

    name: ClassVar[str] = "provider"
    env_key: ClassVar[str] = ""
    default_model: ClassVar[str] = ""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key or (os.environ.get(self.env_key, "") if self.env_key else "")
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_headers = headers or {}
        self._client = client
        self._owns_client = client is None

    BASE_URL: ClassVar[str] = ""

    # ---- transport ----------------------------------------------------
    @property
    def http(self) -> httpx.AsyncClient:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        return {}

    def _require_key(self) -> None:
        if not self.api_key:
            raise ProviderError(
                f"No API key for {self.name}. Set ${self.env_key} or pass api_key=...",
                provider=self.name,
            )

    async def _post(self, path: str, payload: dict[str, Any], *,
                    params: dict[str, str] | None = None) -> dict[str, Any]:
        """POST with backoff on 429 and 5xx. Everything else fails fast."""
        url = f"{self.base_url}{path}"
        headers = {"content-type": "application/json", **self._auth_headers(),
                   **self.extra_headers}
        import httpx

        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.http.post(url, json=payload, headers=headers, params=params)
            except httpx.RequestError as exc:  # network flake — worth a retry
                last = ProviderError(f"{self.name} request failed: {exc}", provider=self.name)
                if attempt == self.max_retries:
                    raise last from exc
                await self._sleep(attempt)
                continue

            if resp.status_code < 300:
                return resp.json()
            body = resp.text[:2000]
            if resp.status_code == 429:
                last = RateLimitError(f"{self.name} rate limited", provider=self.name,
                                      status=429, body=body)
            elif resp.status_code >= 500:
                last = ProviderError(f"{self.name} server error {resp.status_code}",
                                     provider=self.name, status=resp.status_code, body=body)
            else:
                raise ProviderError(
                    f"{self.name} returned {resp.status_code}: {body}",
                    provider=self.name, status=resp.status_code, body=body,
                )
            if attempt == self.max_retries:
                raise last
            await self._sleep(attempt, resp.headers.get("retry-after"))
        raise last or ProviderError("unreachable", provider=self.name)

    async def _sleep(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                await asyncio.sleep(min(float(retry_after), 60.0))
                return
            except ValueError:
                pass
        await asyncio.sleep(min(2.0**attempt + random.uniform(0, 0.5), 30.0))

    # ---- the contract -------------------------------------------------
    @abstractmethod
    async def complete(self, req: CompletionRequest) -> ModelResponse:
        """One round trip: messages in, one assistant message out."""

    async def stream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Token stream. The default just wraps `complete` so every provider streams."""
        resp = await self.complete(req)
        if resp.text:
            yield StreamEvent(type="text", text=resp.text)
        for use in resp.tool_uses:
            yield StreamEvent(type="tool_call", data={"name": use.name, "input": use.input,
                                                      "id": use.id})
        yield StreamEvent(type="step_end", data={"response": resp.model_dump(mode="json")})

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError(f"{self.name} has no embedding endpoint wired up")

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _finish(self, *, message: Message, stop_reason: str, usage: Usage, model: str,
                raw: dict[str, Any], latency_ms: float = 0.0) -> ModelResponse:
        usage.calls = max(usage.calls, 1)
        usage.cost_usd = estimate_cost(model, usage)
        return ModelResponse(
            message=message, stop_reason=stop_reason, usage=usage, model=model,
            provider=self.name, raw=raw, latency_ms=latency_ms,
        )
