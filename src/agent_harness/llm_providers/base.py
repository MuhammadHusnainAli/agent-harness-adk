"""The provider contract every model backend implements.

One shape in, one shape out. The agent loop never learns which vendor it is
talking to — that is the whole point of this file.

Adapters speak raw HTTP over ``httpx`` rather than pulling in three vendor SDKs.
It keeps the wheel small and means every backend travels the same retry,
timeout, circuit-breaker and cost-accounting path: `Provider._request` is the
only place in the package that puts a request on the wire.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # httpx is imported on the first call, not on import
    import httpx

from ..errors import (
    AuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from ..types import Message, ModelResponse, StreamEvent, Usage
from .parameters import SAMPLING_PARAMETERS, Effort, ParameterPlan
from .resilience import (
    CircuitBreaker,
    ProviderStats,
    RetryEvent,
    RetryPolicy,
    classify,
    request_id,
)

logger = logging.getLogger("agent_harness.llm_providers")
logger.addHandler(logging.NullHandler())   # a library stays quiet until you configure logging

__all__ = [
    "ToolSchema",
    "ModelInfo",
    "CompletionRequest",
    "Provider",
    "ProviderField",
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
    if low.startswith("deepseek"):
        return "deepseek"
    if low.startswith("grok"):
        return "xai"
    if low.startswith(("mistral", "codestral", "magistral", "ministral", "devstral",
                       "pixtral")):
        return "mistral"
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
    max_tokens: int = Field(8192, ge=1)

    # --- how it should think --------------------------------------------------
    #: Reasoning depth: none · minimal · low · medium · high · xhigh · max.
    #: `low` for mechanical work, `high` for judgement, `max` when correctness
    #: matters more than cost, `none` to switch reasoning off where it can be.
    effort: Effort | None = None
    thinking: bool | None = None
    #: An explicit thinking-token ceiling, for models that take one instead of
    #: an effort level. Setting it asks for thinking. -1 lets Gemini decide.
    thinking_budget: int | None = Field(None, ge=-1)

    # --- sampling (see Provider.parameters for who takes which) -------------------
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=1)                  # Anthropic, Gemini, open models
    min_p: float | None = Field(None, ge=0.0, le=1.0)      # vLLM, Together, OpenRouter
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    repetition_penalty: float | None = Field(None, gt=0.0)  # vLLM, Together, OpenRouter
    seed: int | None = None                     # OpenAI, Gemini, most open-model hosts
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


class ProviderField(BaseModel):
    """One thing a provider needs (or can take) to connect.

    `list_llm_providers()` shows these, so someone wiring up a backend for the
    first time can see exactly what to supply and where it can come from.
    """

    model_config = ConfigDict(frozen=True)

    name: str                                   # the constructor keyword
    description: str = ""
    type: Literal["str", "secret", "url", "int", "float", "bool", "object"] = "str"
    required: bool = False
    #: Satisfied if *any* field sharing this group is set: "a key OR a credential".
    one_of: str | None = None
    env: tuple[str, ...] = ()                   # environment variables read, in order
    default: Any = None
    example: str | None = None

    @property
    def secret(self) -> bool:
        return self.type == "secret"


#: Settings every HTTP-backed provider takes, on top of its own fields.
COMMON_FIELDS: tuple[ProviderField, ...] = (
    ProviderField(name="timeout", type="float", default=600.0,
                  description="Seconds to wait for a response (per attempt)."),
    ProviderField(name="connect_timeout", type="float", default=10.0,
                  description="Seconds to wait to open the connection."),
    ProviderField(name="max_retries", type="int", default=3,
                  description="Retries on 429, 5xx, timeouts and dropped connections."),
    ProviderField(name="retry", type="object",
                  description="A RetryPolicy: backoff, jitter, wall-clock ceiling, "
                              "and the longest Retry-After to honour."),
    ProviderField(name="max_concurrency", type="int",
                  description="Cap on in-flight requests from this provider instance."),
    ProviderField(name="circuit_breaker", type="object", default=True,
                  description="A CircuitBreaker, True for the default, False for none."),
    ProviderField(name="on_retry", type="object",
                  description="Called with a RetryEvent before every retry."),
    ProviderField(name="headers", type="object",
                  description="Extra HTTP headers sent with every request."),
    ProviderField(name="client", type="object",
                  description="Your own httpx.AsyncClient (proxies, mTLS, transports)."),
)

OnRetry = Callable[[RetryEvent], "Awaitable[None] | None"]


class Provider(ABC):
    """Base class for model backends. Subclasses implement `complete`.

    What a subclass gets for free, by sending through `_post` / `_stream_lines`:

    - retries with exponential backoff and jitter on 408/409/425/429/5xx/529,
      timeouts and dropped connections, honouring every Retry-After variant;
    - typed errors (`RateLimitError`, `ContextWindowExceededError`, ...) that
      carry the vendor's request id;
    - a shared cool-down, so when one call is told to wait, the calls running
      alongside it wait too instead of hammering a throttled endpoint;
    - a circuit breaker, an optional concurrency cap, per-request timeouts,
      a token refresh on 401 for bearer-token backends, and `stats`.
    """

    name: ClassVar[str] = "provider"
    env_key: ClassVar[str] = ""
    default_model: ClassVar[str] = ""
    BASE_URL: ClassVar[str] = ""

    # ---- what `list_llm_providers()` shows --------------------------------
    display_name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    docs_url: ClassVar[str] = ""
    #: api_key · bearer · sigv4 · google-oauth · none
    auth_type: ClassVar[str] = "api_key"
    #: The fields specific to this backend. `COMMON_FIELDS` are added to these.
    fields: ClassVar[tuple[ProviderField, ...]] = ()
    #: streaming · tools · vision · thinking · json_schema · embeddings · list_models
    capabilities: ClassVar[frozenset[str]] = frozenset()
    #: The sampling parameters this backend accepts. Anything else is dropped.
    parameters: ClassVar[frozenset[str]] = frozenset(SAMPLING_PARAMETERS)

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 600.0,
        connect_timeout: float = 10.0,
        max_retries: int | None = None,
        retry: RetryPolicy | None = None,
        max_concurrency: int | None = None,
        circuit_breaker: CircuitBreaker | bool | None = True,
        on_retry: OnRetry | None = None,
        client: httpx.AsyncClient | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key or (os.environ.get(self.env_key, "") if self.env_key else "")
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        policy = retry or RetryPolicy()
        if max_retries is not None:
            from dataclasses import replace

            policy = replace(policy, max_retries=max(0, max_retries))
        self.retry = policy
        if circuit_breaker is True:
            circuit_breaker = CircuitBreaker()
        self.circuit: CircuitBreaker | None = circuit_breaker or None
        self.max_concurrency = max_concurrency
        self.on_retry = on_retry
        self.extra_headers = headers or {}
        self._client = client
        self._owns_client = client is None

    # Subclasses such as the replay provider skip __init__; these keep them working.
    @property
    def max_retries(self) -> int:
        return self._policy().max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        from dataclasses import replace

        self.retry = replace(self._policy(), max_retries=max(0, int(value)))

    def _policy(self) -> RetryPolicy:
        policy = getattr(self, "retry", None)
        if policy is None:
            policy = RetryPolicy()
            self.retry = policy
        return policy

    @property
    def stats(self) -> ProviderStats:
        stats = self.__dict__.get("_stats")
        if stats is None:
            stats = self.__dict__["_stats"] = ProviderStats()
        return stats

    def health(self) -> dict[str, Any]:
        """Counters plus the circuit's state — what to put on a dashboard."""
        circuit = getattr(self, "circuit", None)
        return {"provider": self.name, **self.stats.snapshot(),
                "circuit": circuit.state if circuit else "disabled"}

    # ---- transport ----------------------------------------------------
    @property
    def http(self) -> httpx.AsyncClient:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout,
                                      connect=getattr(self, "connect_timeout", 10.0)),
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        return {}

    def _require_key(self) -> None:
        if not self.api_key:
            hint = f"Set ${self.env_key} or pass" if self.env_key else "Pass"
            raise AuthenticationError(
                f"No API key for {self.name}. {hint} api_key=... "
                f"(see describe_llm_provider({self.name!r}))",
                provider=self.name,
            )

    async def _prepare_headers(self, method: str, url: str, body: bytes,
                               headers: dict[str, str]) -> dict[str, str]:
        """The final headers for one attempt. Called again on every retry, so a
        signature or a token is always fresh. Bedrock signs here; the bearer-token
        backends add their token here."""
        return {**headers, **self._auth_headers()}

    async def _refresh_auth(self) -> bool:
        """Drop a cached token after a 401. True if a retry could now succeed."""
        return False

    def _url(self, path: str) -> str:
        return path if path.startswith(("http://", "https://")) else f"{self.base_url}{path}"

    async def _post(self, path: str, payload: dict[str, Any], *,
                    params: dict[str, str] | None = None,
                    timeout: float | None = None) -> dict[str, Any]:
        """POST JSON, get JSON — with every retry and safety rail applied."""
        return await self._request("POST", self._url(path), payload=payload,
                                   params=params, timeout=timeout)

    async def _get(self, path: str, *, params: dict[str, str] | None = None,
                   timeout: float | None = None) -> dict[str, Any]:
        return await self._request("GET", self._url(path), params=params,
                                   timeout=timeout)

    async def _stream_lines(self, path: str, payload: dict[str, Any], *,
                            params: dict[str, str] | None = None,
                            timeout: float | None = None) -> AsyncIterator[str]:
        """POST and yield the response line by line (SSE). Opening the stream is
        retried like any call; a failure once bytes are flowing is raised, and
        `stream()` decides whether it is still safe to start over."""
        async for chunk in self._stream_raw(path, payload, params=params,
                                            timeout=timeout, lines=True):
            yield chunk  # type: ignore[misc]

    async def _stream_raw(self, path: str, payload: dict[str, Any], *,
                          params: dict[str, str] | None = None,
                          timeout: float | None = None,
                          lines: bool = False) -> AsyncIterator[str | bytes]:
        import httpx

        resp, release = await self._request("POST", self._url(path), payload=payload,
                                            params=params, timeout=timeout, stream=True)
        try:
            source = resp.aiter_lines() if lines else resp.aiter_bytes()
            async for chunk in source:
                yield chunk
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"{self.name} stream stalled: {exc or type(exc).__name__}",
                provider=self.name) from exc
        except httpx.RequestError as exc:
            raise ProviderConnectionError(
                f"{self.name} stream dropped: {exc or type(exc).__name__}",
                provider=self.name) from exc
        finally:
            await resp.aclose()
            release()

    async def _request(self, method: str, url: str, *, payload: dict[str, Any] | None = None,
                       params: dict[str, str] | None = None, timeout: float | None = None,
                       stream: bool = False) -> Any:
        """The one place a request goes out. Retries, classifies, and keeps score.

        Returns parsed JSON, or — with ``stream=True`` — ``(response, release)``,
        where the caller must close the response and call ``release()``.
        """
        import httpx

        policy = self._policy()
        stats = self.stats
        circuit: CircuitBreaker | None = getattr(self, "circuit", None)
        stats.calls += 1
        if circuit is not None and not circuit.allow():
            stats.circuit_rejections += 1
            stats.failed += 1
            wait = circuit.retry_in()
            raise ProviderUnavailableError(
                f"{self.name} circuit is open after {circuit.failures} consecutive "
                f"failures — not calling it for another {wait:.0f}s",
                provider=self.name, retry_after=wait, retryable=True)

        body = (json.dumps(payload, separators=(",", ":"), default=str).encode()
                if payload is not None else b"")
        base_headers = {"accept": "text/event-stream" if stream else "application/json",
                        "user-agent": _user_agent(),
                        **({"content-type": "application/json"} if payload is not None
                           else {}),
                        **getattr(self, "extra_headers", {})}
        per_try = None
        if timeout:
            per_try = httpx.Timeout(timeout, connect=min(timeout,
                                    getattr(self, "connect_timeout", 10.0)))
        started = time.monotonic()
        waited = 0.0          # back-off taken so far; wall time may not show it all
        resume_at = 0.0       # when our own back-off ends — no need to wait twice
        attempt = 0
        refreshed = False
        settled = False
        try:
            while True:
                await self._wait_for_cooldown(resume_at)
                release = await self._acquire_slot()
                error: ProviderError | None = None
                try:
                    headers = await self._prepare_headers(method, url, body, base_headers)
                    stats.requests += 1
                    request = self.http.build_request(
                        method, url, content=body or None, headers=headers, params=params,
                        **({"timeout": per_try} if per_try is not None else {}))
                    resp = await self.http.send(request, stream=stream)
                except httpx.TimeoutException as exc:
                    stats.timeouts += 1
                    error = ProviderTimeoutError(
                        f"{self.name} timed out after {time.monotonic() - started:.1f}s "
                        f"({type(exc).__name__})", provider=self.name)
                except httpx.RequestError as exc:
                    fatal = isinstance(exc, (httpx.UnsupportedProtocol,
                                             httpx.LocalProtocolError))
                    error = ProviderConnectionError(
                        f"{self.name} request failed: {exc or type(exc).__name__}",
                        provider=self.name, retryable=not fatal)
                except BaseException:
                    release()
                    raise
                else:
                    stats.by_status[resp.status_code] = stats.by_status.get(
                        resp.status_code, 0) + 1
                    rid = request_id(resp.headers)
                    if rid:
                        stats.last_request_id = rid
                    if resp.status_code < 300:
                        if stream:
                            self._settle(ok=True)
                            settled = True
                            return resp, release
                        try:
                            data = resp.json()
                        except ValueError:
                            error = ProviderUnavailableError(
                                f"{self.name} sent a {resp.status_code} that is not JSON: "
                                f"{resp.text[:200]!r}", provider=self.name,
                                status=resp.status_code, request_id=rid)
                        else:
                            release()
                            self._settle(ok=True)
                            settled = True
                            return data
                    else:
                        try:
                            raw = (await resp.aread()).decode("utf-8", errors="replace")
                        except httpx.HTTPError:
                            raw = ""                  # the status alone will have to do
                        finally:
                            if stream:
                                await resp.aclose()
                        error = classify(self.name, resp.status_code, headers=resp.headers,
                                         body=raw, policy=policy)
                if error is not None:
                    release()

                assert error is not None
                if isinstance(error, RateLimitError):
                    stats.rate_limited += 1
                if (isinstance(error, AuthenticationError) and not refreshed
                        and await self._refresh_auth()):
                    refreshed = True                  # a stale token, not a retry
                    continue
                attempt += 1
                error.attempts = attempt
                elapsed = max(time.monotonic() - started, waited)
                delay = self._next_delay(error, attempt, elapsed, policy)
                if delay is None:
                    stats.last_error = f"{type(error).__name__}: {error}"
                    error._final = True  # type: ignore[attr-defined]
                    # A 4xx still proves the provider is up; only the request was bad.
                    self._settle(ok=False,
                                 healthy=not error.retryable and error.status is not None)
                    settled = True
                    raise error
                if isinstance(error, RateLimitError) and error.retry_after:
                    self._cooldown_until = max(getattr(self, "_cooldown_until", 0.0),
                                               time.monotonic() + delay)
                await self._before_retry(RetryEvent(provider=self.name, attempt=attempt,
                                                    delay=delay, error=error, url=url))
                resume_at = time.monotonic() + delay
                waited += delay
                await self._pause(delay)
        finally:
            if not settled and circuit is not None:
                circuit._probing = False             # cancelled mid-probe: free the slot

    def _settle(self, *, ok: bool, healthy: bool | None = None) -> None:
        stats = self.stats
        if ok:
            stats.succeeded += 1
        else:
            stats.failed += 1
        circuit: CircuitBreaker | None = getattr(self, "circuit", None)
        if circuit is not None:
            if ok if healthy is None else healthy:
                circuit.record_success()
            else:
                circuit.record_failure()

    def _next_delay(self, error: ProviderError, attempt: int, elapsed: float,
                    policy: RetryPolicy) -> float | None:
        """Seconds before the next attempt, or None to give up now."""
        if not error.retryable or attempt > policy.max_retries:
            return None
        if isinstance(error, ProviderTimeoutError) and not policy.retry_on_timeout:
            return None
        if isinstance(error, ProviderConnectionError) and not policy.retry_on_connection_error:
            return None
        if error.retry_after is not None:
            if error.retry_after > policy.max_retry_after:
                return None                    # let a fallback model take over instead
            delay = error.retry_after
        else:
            delay = policy.backoff(attempt - 1)
        if policy.max_elapsed is not None and elapsed + delay > policy.max_elapsed:
            return None
        return delay

    async def _before_retry(self, event: RetryEvent) -> None:
        stats = self.stats
        stats.retries += 1
        stats.total_backoff_s += event.delay
        logger.warning("%s: %s — retry %d/%d in %.2fs", event.provider, event.reason,
                       event.attempt, self._policy().max_retries, event.delay)
        callback = getattr(self, "on_retry", None)
        if callback is not None:
            try:
                result = callback(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:                  # a broken observer must not break the call
                logger.exception("on_retry callback failed")

    async def _pause(self, seconds: float) -> None:
        """Every back-off goes through here — one seam to observe or speed up in tests."""
        await asyncio.sleep(seconds)

    async def _wait_for_cooldown(self, resume_at: float = 0.0) -> None:
        """Hold back while a rate limit someone else hit is still in force."""
        wait = getattr(self, "_cooldown_until", 0.0) - max(time.monotonic(), resume_at)
        if wait > 0:
            await self._pause(wait)

    async def _acquire_slot(self) -> Callable[[], None]:
        limit = getattr(self, "max_concurrency", None)
        if not limit:
            return _noop
        loop = asyncio.get_running_loop()
        # One semaphore per event loop: a cached provider can outlive an asyncio.run.
        if getattr(self, "_slot_loop", None) is not loop:
            self._slots = asyncio.Semaphore(limit)
            self._slot_loop = loop
        slots = self._slots
        await slots.acquire()
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                slots.release()

        return release

    # ---- generation parameters -----------------------------------------------
    def plan_parameters(self, req: CompletionRequest) -> ParameterPlan:
        """Decide what this provider sends for the request's generation parameters.

        Sampling knobs the backend lacks are dropped; values a model would reject
        are fitted; `effort` and `thinking` become whatever this model takes.
        Every decision is recorded on the plan with its reason.
        """
        plan = ParameterPlan.of(self.name, req)
        label = self.display_name or self.name
        for name in SAMPLING_PARAMETERS:
            if name in plan and name not in self.parameters:
                plan.drop(name, f"{label} has no {name}")
        self._plan(req, plan)
        _report(plan)
        return plan

    def _plan(self, req: CompletionRequest, plan: ParameterPlan) -> None:
        """Model-specific rules. Adapters override this."""
        return None

    def explain(self, req: CompletionRequest) -> dict[str, Any]:
        """What would be sent for this request — without sending it.

        ``sent``/``dropped``/``adjusted`` from the parameter plan, and the exact
        ``payload`` that would go on the wire.
        """
        out = self.plan_parameters(req).as_dict()
        payload = getattr(self, "_payload", None)
        if payload is not None:
            out["payload"] = payload(req)
        return out

    # ---- the contract -------------------------------------------------
    @abstractmethod
    async def complete(self, req: CompletionRequest) -> ModelResponse:
        """One round trip: messages in, one assistant message out."""

    async def _stream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Native token streaming. Override this, not `stream`, to get its retries."""
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator

    async def stream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Token stream, ending with a ``step_end`` event that carries the response.

        Backends without native streaming get one built from `complete`. Native
        streams are retried while it is still safe to — until the first event
        has been handed on. After that a failure is raised, because starting over
        would repeat text the caller has already shown.
        """
        if type(self)._stream is Provider._stream:
            resp = await self.complete(req)
            for event in _events_from(resp):
                yield event
            return

        policy = self._policy()
        started = time.monotonic()
        waited = 0.0
        attempt = 0
        while True:
            emitted = False
            try:
                async for event in self._stream(req):
                    emitted = True
                    yield event
                return
            except ProviderError as exc:
                if emitted or getattr(exc, "_final", False):
                    raise
                attempt += 1
                exc.attempts = attempt
                delay = self._next_delay(exc, attempt,
                                         max(time.monotonic() - started, waited), policy)
                if delay is None:
                    raise
                await self._before_retry(RetryEvent(provider=self.name, attempt=attempt,
                                                    delay=delay, error=exc))
                waited += delay
                await self._pause(delay)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError(f"{self.name} has no embedding endpoint wired up")

    async def list_models(self) -> list[str]:
        """Model ids this account can use, straight from the provider."""
        raise NotImplementedError(f"{self.name} cannot list its models")

    async def ping(self, model: str | None = None) -> dict[str, Any]:
        """Is this provider reachable, and does it accept our credentials?

        Uses the free model-listing endpoint where there is one, and a one-token
        completion otherwise. Never raises — the answer is in the result.
        """
        started = time.perf_counter()
        try:
            try:
                models = await self.list_models()
                detail: dict[str, Any] = {"models": len(models)}
            except NotImplementedError:
                target = model or getattr(self, "deployment", "") or self.default_model
                if not target:
                    raise ProviderError(f"{self.name}: pass model=... to ping it",
                                        provider=self.name) from None
                resp = await self.complete(CompletionRequest(
                    model=target, messages=[Message.user("ping")], max_tokens=1))
                detail = {"model": resp.model}
        except Exception as exc:
            return {"provider": self.name, "ok": False,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error": f"{type(exc).__name__}: {exc}"}
        return {"provider": self.name, "ok": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1), **detail}

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Provider:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def _finish(self, *, message: Message, stop_reason: str, usage: Usage, model: str,
                raw: dict[str, Any], latency_ms: float = 0.0) -> ModelResponse:
        usage.calls = max(usage.calls, 1)
        usage.cost_usd = estimate_cost(model, usage)
        return ModelResponse(
            message=message, stop_reason=stop_reason, usage=usage, model=model,
            provider=self.name, raw=raw, latency_ms=latency_ms,
        )


def _noop() -> None:
    return None


_REPORTED: set[tuple[str, str, str, str]] = set()


def _report(plan: ParameterPlan) -> None:
    """Log each drop or adjustment once per provider, model and parameter."""
    for kind, notes in (("dropped", plan.dropped), ("adjusted", plan.adjusted)):
        for name, why in notes.items():
            key = (plan.provider, plan.model, name, kind)
            if key in _REPORTED:
                continue
            if len(_REPORTED) > 10_000:
                _REPORTED.clear()
            _REPORTED.add(key)
            logger.info("%s/%s: %s %s — %s", plan.provider, plan.model, kind, name, why)


def _events_from(resp: ModelResponse) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    if resp.text:
        events.append(StreamEvent(type="text", text=resp.text))
    for use in resp.tool_uses:
        events.append(StreamEvent(type="tool_call",
                                  data={"name": use.name, "input": use.input, "id": use.id}))
    events.append(StreamEvent(type="step_end", data={"response": resp.model_dump(mode="json")}))
    return events


def _user_agent() -> str:
    try:
        from .. import __version__
    except ImportError:  # pragma: no cover - mid-import
        __version__ = "0"
    return f"agent-harness/{__version__}"


async def sse_events(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, str]]:
    """Server-sent events as ``(event, data)``, joining multi-line ``data:`` fields."""
    event, data = "", []
    async for line in lines:
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "", []
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if name == "event":
            event = value
        elif name == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)
