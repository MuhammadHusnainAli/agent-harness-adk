"""Retries, back-off, typed errors, circuits and streaming recovery — over MockTransport."""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from agent_harness import (
    AnthropicProvider,
    AuthenticationError,
    AzureOpenAIProvider,
    BedrockProvider,
    CircuitBreaker,
    ContextWindowExceededError,
    GeminiProvider,
    InvalidRequestError,
    ModelNotFoundError,
    OpenAIProvider,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
    RetryPolicy,
    VertexProvider,
)
from agent_harness.llm_providers._sigv4 import AWSCredentials
from agent_harness.llm_providers.base import CompletionRequest, Provider
from agent_harness.llm_providers.bedrock import encode_event
from agent_harness.llm_providers.resilience import classify, parse_duration, retry_after
from agent_harness.types import Message, ThinkingBlock, ToolUseBlock

ANTHROPIC_OK = {"model": "claude-opus-5", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 3, "output_tokens": 1}}
OPENAI_OK = {"model": "gpt-4.1", "choices": [{"finish_reason": "stop",
                                              "message": {"content": "hi"}}],
             "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


def req(model: str = "claude-opus-5", **kw) -> CompletionRequest:
    return CompletionRequest(model=model, messages=[Message.user("x")], **kw)


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    """Every back-off the providers take, recorded instead of slept."""
    seen: list[float] = []

    async def pause(self, seconds: float) -> None:
        seen.append(round(seconds, 3))

    monkeypatch.setattr(Provider, "_pause", pause)
    return seen


def scripted(*responses):
    """A client that replays responses (or raises exceptions) in order."""
    calls: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    return calls, httpx.AsyncClient(transport=httpx.MockTransport(handler))


def err(status: int, body: dict | str = "", **headers: str) -> httpx.Response:
    content = json.dumps(body) if isinstance(body, dict) else body
    return httpx.Response(status, content=content.encode(), headers=headers)


# --- what is retried, and how long it waits ---------------------------------------

async def test_a_429_waits_as_told_then_succeeds(waits):
    calls, client = scripted(err(429, {"error": {"message": "slow"}}, **{"retry-after": "2"}),
                             httpx.Response(200, json=ANTHROPIC_OK))
    seen = []
    provider = AnthropicProvider(api_key="k", client=client, on_retry=seen.append)
    response = await provider.complete(req())
    assert response.text == "hi"
    assert len(calls) == 2 and waits == [2.0]
    assert isinstance(seen[0].error, RateLimitError) and seen[0].attempt == 1
    assert provider.stats.retries == 1 and provider.stats.rate_limited == 1


@pytest.mark.parametrize("headers,body,expected", [
    ({"retry-after-ms": "1500"}, "", 1.5),
    ({"retry-after": "7"}, "", 7.0),
    ({"x-ratelimit-reset-requests": "1s", "x-ratelimit-reset-tokens": "6m0s"}, "", 360.0),
    ({}, '{"error":{"details":[{"retryDelay":"31s"}]}}', 31.0),
    ({}, "", None),
])
def test_every_retry_after_hint_is_read(headers, body, expected):
    assert retry_after(httpx.Headers(headers), body) == expected


def test_an_http_date_retry_after_is_understood():
    when = datetime.now(timezone.utc) + timedelta(seconds=30)
    wait = retry_after(httpx.Headers({"retry-after": format_datetime(when, usegmt=True)}))
    assert wait is not None and 25 <= wait <= 31


def test_bucket_reset_headers_only_count_for_a_rate_limit():
    headers = httpx.Headers({"x-ratelimit-reset-requests": "20s"})
    assert retry_after(headers, rate_limited=False) is None
    assert classify("openai", 500, headers=headers).retry_after is None


@pytest.mark.parametrize("text,seconds", [
    ("250ms", 0.25), ("1.5s", 1.5), ("6m0s", 360.0), ("1h2m3s", 3723.0), ("3", 3.0),
    ("soon", None), ("", None),
])
def test_durations(text, seconds):
    assert parse_duration(text) == seconds


async def test_without_a_hint_backoff_grows_exponentially_with_jitter(waits):
    calls, client = scripted(err(503, "down"))
    policy = RetryPolicy(max_retries=4, initial_delay=1.0, multiplier=2.0, jitter=0.25)
    provider = AnthropicProvider(api_key="k", client=client, retry=policy,
                                 circuit_breaker=False)
    with pytest.raises(ProviderUnavailableError) as info:
        await provider.complete(req())
    assert len(calls) == 5 and info.value.attempts == 5
    for attempt, wait in enumerate(waits):
        full = 2.0 ** attempt
        assert full * 0.75 <= wait <= full


async def test_529_overloaded_is_retried(waits):
    calls, client = scripted(
        err(529, {"type": "error", "error": {"type": "overloaded_error",
                                             "message": "Overloaded"}}),
        httpx.Response(200, json=ANTHROPIC_OK))
    provider = AnthropicProvider(api_key="k", client=client)
    assert (await provider.complete(req())).text == "hi"
    assert len(calls) == 2


async def test_a_server_asking_for_too_long_a_wait_fails_fast_for_the_fallback(waits):
    calls, client = scripted(err(429, "busy", **{"retry-after": "600"}))
    provider = AnthropicProvider(api_key="k", client=client,
                                 retry=RetryPolicy(max_retry_after=60))
    with pytest.raises(RateLimitError) as info:
        await provider.complete(req())
    assert len(calls) == 1 and waits == []
    assert info.value.retry_after == 600 and info.value.retryable


async def test_the_wall_clock_ceiling_stops_retrying(waits):
    calls, client = scripted(err(500, "x", **{"retry-after": "5"}))
    provider = AnthropicProvider(api_key="k", client=client,
                                 retry=RetryPolicy(max_retries=10, max_elapsed=12))
    with pytest.raises(ProviderUnavailableError):
        await provider.complete(req())
    assert waits == [5.0, 5.0]           # a third wait would pass twelve seconds


async def test_x_should_retry_is_believed(waits):
    calls, client = scripted(err(500, "no", **{"x-should-retry": "false"}))
    provider = OpenAIProvider(api_key="k", client=client)
    with pytest.raises(ProviderUnavailableError) as info:
        await provider.complete(req("gpt-4.1"))
    assert len(calls) == 1 and not info.value.retryable


# --- what is not retried, and what it is called ------------------------------------------

@pytest.mark.parametrize("status,body,kind", [
    (429, {"error": {"code": "insufficient_quota", "message": "You exceeded your "
                                                               "current quota"}},
     QuotaExceededError),
    (400, {"error": {"code": "context_length_exceeded", "message": "too long"}},
     ContextWindowExceededError),
    (400, {"type": "error", "error": {"type": "invalid_request_error",
                                      "message": "prompt is too long: 250000 tokens"}},
     ContextWindowExceededError),
    (413, {"error": {"type": "request_too_large"}}, ContextWindowExceededError),
    (401, {"error": {"message": "invalid x-api-key"}}, AuthenticationError),
    (400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                     "details": [{"reason": "API_KEY_INVALID"}]}}, AuthenticationError),
    (404, {"error": {"code": "model_not_found"}}, ModelNotFoundError),
    (422, {"detail": "bad"}, InvalidRequestError),
])
async def test_permanent_failures_are_typed_and_not_retried(waits, status, body, kind):
    calls, client = scripted(err(status, body, **{"request-id": "req_123"}))
    provider = OpenAIProvider(api_key="k", client=client)
    with pytest.raises(kind) as info:
        await provider.complete(req("gpt-4.1"))
    assert len(calls) == 1 and waits == []
    assert type(info.value) is kind and not info.value.retryable
    assert info.value.request_id == "req_123" and "req_123" in str(info.value)
    assert isinstance(info.value, ProviderError)       # one except still catches all


async def test_timeouts_are_retried_then_named(waits):
    calls, client = scripted(httpx.ReadTimeout("slow"))
    provider = AnthropicProvider(api_key="k", client=client, max_retries=2)
    with pytest.raises(ProviderTimeoutError):
        await provider.complete(req())
    assert len(calls) == 3 and provider.stats.timeouts == 3


async def test_a_dropped_connection_recovers(waits):
    calls, client = scripted(httpx.ConnectError("reset"),
                             httpx.Response(200, json=ANTHROPIC_OK))
    provider = AnthropicProvider(api_key="k", client=client)
    assert (await provider.complete(req())).text == "hi"
    assert len(calls) == 2


async def test_connection_retries_can_be_turned_off(waits):
    calls, client = scripted(httpx.ConnectError("reset"))
    provider = AnthropicProvider(api_key="k", client=client,
                                 retry=RetryPolicy(retry_on_connection_error=False))
    with pytest.raises(ProviderConnectionError):
        await provider.complete(req())
    assert len(calls) == 1


async def test_a_200_that_is_not_json_is_retried(waits):
    calls, client = scripted(httpx.Response(200, content=b"<html>proxy</html>"),
                             httpx.Response(200, json=ANTHROPIC_OK))
    provider = AnthropicProvider(api_key="k", client=client)
    assert (await provider.complete(req())).text == "hi"
    assert len(calls) == 2


async def test_the_per_request_timeout_reaches_the_wire():
    calls, client = scripted(httpx.Response(200, json=ANTHROPIC_OK))
    provider = AnthropicProvider(api_key="k", client=client)
    await provider.complete(req(timeout=12.5))
    assert calls[0].extensions["timeout"]["read"] == 12.5


async def test_a_missing_key_is_an_authentication_error_before_any_request():
    calls, client = scripted(httpx.Response(200, json=ANTHROPIC_OK))
    provider = AnthropicProvider(api_key="", client=client)
    provider.api_key = ""
    with pytest.raises(AuthenticationError, match="ANTHROPIC_API_KEY"):
        await provider.complete(req())
    assert calls == []


# --- the circuit breaker ---------------------------------------------------------------

async def test_the_circuit_opens_after_repeated_failures_and_stops_calling(waits):
    calls, client = scripted(err(503, "down"))
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout=30)
    provider = AnthropicProvider(api_key="k", client=client, max_retries=0,
                                 circuit_breaker=breaker)
    for _ in range(2):
        with pytest.raises(ProviderUnavailableError):
            await provider.complete(req())
    assert breaker.state == "open"

    with pytest.raises(ProviderUnavailableError, match="circuit is open") as info:
        await provider.complete(req())
    assert len(calls) == 2                       # the third call never left
    assert info.value.retryable                  # so the router falls back
    assert provider.stats.circuit_rejections == 1
    assert provider.health()["circuit"] == "open"


async def test_a_half_open_circuit_closes_on_a_good_probe(waits):
    calls, client = scripted(err(503, "down"), httpx.Response(200, json=ANTHROPIC_OK))
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout=30)
    provider = AnthropicProvider(api_key="k", client=client, max_retries=0,
                                 circuit_breaker=breaker)
    with pytest.raises(ProviderUnavailableError):
        await provider.complete(req())
    breaker.opened_at -= 31                      # time passes
    assert breaker.state == "half_open"
    assert (await provider.complete(req())).text == "hi"
    assert breaker.state == "closed"


async def test_a_bad_request_does_not_trip_the_circuit(waits):
    calls, client = scripted(err(400, {"error": {"message": "bad"}}))
    breaker = CircuitBreaker(failure_threshold=1)
    provider = AnthropicProvider(api_key="k", client=client, circuit_breaker=breaker)
    for _ in range(3):
        with pytest.raises(InvalidRequestError):
            await provider.complete(req())
    assert breaker.state == "closed" and len(calls) == 3


# --- concurrency and shared cool-down ------------------------------------------------------

async def test_max_concurrency_caps_requests_in_flight():
    live = {"now": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        await asyncio.sleep(0.01)
        live["now"] -= 1
        return httpx.Response(200, json=ANTHROPIC_OK)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="k", client=client, max_concurrency=2)
    await asyncio.gather(*(provider.complete(req()) for _ in range(8)))
    assert live["peak"] == 2


async def test_a_rate_limit_makes_every_caller_wait(waits):
    calls, client = scripted(err(429, "slow", **{"retry-after": "3"}),
                             httpx.Response(200, json=ANTHROPIC_OK))
    provider = AnthropicProvider(api_key="k", client=client)
    await provider.complete(req())
    assert waits == [3.0]                       # the caller that was told waits once
    await provider.complete(req())              # a second caller, inside the window,
    assert len(waits) == 2 and 2.5 < waits[1] <= 3.0   # holds back instead of hammering


# --- streaming: retried until the first token, never after ---------------------------------

def sse(*events: dict) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode()
                    for e in events)


START = {"type": "message_start", "message": {"model": "claude-opus-5",
                                              "usage": {"input_tokens": 10,
                                                        "output_tokens": 1}}}
TEXT = [{"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "hello"}}]
END = [{"type": "message_delta", "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 7}},
       {"type": "message_stop"}]
OVERLOADED = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}


async def collect(provider, request):
    return [e async for e in provider.stream(request)]


async def test_an_overloaded_stream_is_restarted_before_any_token(waits):
    calls, client = scripted(httpx.Response(200, content=sse(START, OVERLOADED)),
                             httpx.Response(200, content=sse(START, *TEXT, *END)))
    provider = AnthropicProvider(api_key="k", client=client)
    events = await collect(provider, req())
    assert [e.text for e in events if e.type == "text"] == ["hello"]   # no duplicates
    final = events[-1].data["response"]
    assert final["usage"]["output_tokens"] == 7          # cumulative, not added
    assert final["usage"]["input_tokens"] == 10
    assert len(calls) == 2 and len(waits) == 1


async def test_a_stream_that_fails_mid_answer_is_not_replayed(waits):
    calls, client = scripted(httpx.Response(200, content=sse(START, *TEXT, OVERLOADED)))
    provider = AnthropicProvider(api_key="k", client=client)
    with pytest.raises(ProviderUnavailableError):
        await collect(provider, req())
    assert len(calls) == 1


async def test_opening_a_stream_is_retried_like_any_call(waits):
    calls, client = scripted(err(429, "slow", **{"retry-after": "1"}),
                             httpx.Response(200, content=sse(START, *TEXT, *END)))
    provider = AnthropicProvider(api_key="k", client=client)
    events = await collect(provider, req())
    assert events[-1].type == "step_end" and len(calls) == 2


async def test_thinking_signatures_survive_streaming():
    events = [START,
              {"type": "content_block_start", "index": 0,
               "content_block": {"type": "thinking", "thinking": ""}},
              {"type": "content_block_delta", "index": 0,
               "delta": {"type": "thinking_delta", "thinking": "hmm"}},
              {"type": "content_block_delta", "index": 0,
               "delta": {"type": "signature_delta", "signature": "sig=="}},
              {"type": "content_block_start", "index": 1,
               "content_block": {"type": "tool_use", "id": "t1", "name": "look",
                                 "input": {}}},
              {"type": "content_block_delta", "index": 1,
               "delta": {"type": "input_json_delta", "partial_json": '{"q": 1}'}},
              {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
               "usage": {"output_tokens": 5}}]
    calls, client = scripted(httpx.Response(200, content=sse(*events)))
    provider = AnthropicProvider(api_key="k", client=client)
    out = await collect(provider, req(thinking=True))
    content = out[-1].data["response"]["message"]["content"]
    assert content[0] == {"type": "thinking", "thinking": "hmm", "signature": "sig=="}
    assert content[1]["input"] == {"q": 1}
    assert [e.type for e in out].count("tool_call") == 1


def test_unsigned_thinking_is_not_sent_to_anthropic_but_redacted_thinking_is():
    provider = AnthropicProvider(api_key="k")
    messages, _ = provider._encode_messages([Message(role="assistant", content=[
        ThinkingBlock(thinking="from another vendor"),
        ThinkingBlock(thinking="", redacted="opaque"),
        ToolUseBlock(id="t", name="n", input={}),
    ])])
    kinds = [b["type"] for b in messages[0]["content"]]
    assert kinds == ["redacted_thinking", "tool_use"]


async def test_openai_stream_errors_are_typed(waits):
    body = b'data: {"error": {"code": "rate_limit_exceeded", "message": "slow"}}\n\n'
    calls, client = scripted(httpx.Response(200, content=body))
    provider = OpenAIProvider(api_key="k", client=client, max_retries=0)
    with pytest.raises(RateLimitError):
        await collect(provider, req("gpt-4.1"))


async def test_openai_stream_reads_reasoning_and_tool_calls():
    chunks = [
        {"model": "deepseek-reasoner", "choices": [{"delta": {"reasoning_content": "think"}}]},
        {"choices": [{"delta": {"content": "ok"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {
            "name": "look", "arguments": '{"q":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
            "arguments": ' 2}'}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}},
    ]
    body = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"
    calls, client = scripted(httpx.Response(200, content=body))
    provider = OpenAIProvider(api_key="k", client=client)
    events = await collect(provider, req("gpt-4.1"))
    assert [e.type for e in events] == ["thinking", "text", "tool_call", "step_end"]
    final = events[-1].data["response"]
    assert final["stop_reason"] == "tool_use"
    assert final["message"]["content"][2]["input"] == {"q": 2}
    assert final["usage"]["output_tokens"] == 4


async def test_gemini_streams_over_sse_and_keeps_thought_signatures():
    chunks = [
        {"candidates": [{"content": {"parts": [{"text": "He"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "llo"}]}}]},
        {"candidates": [{"content": {"parts": [{"functionCall": {"name": "look",
                                                                  "args": {"q": 1}},
                                                 "thoughtSignature": "gsig"}]},
                         "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2}},
    ]
    body = b"".join(f"data: {json.dumps(c)}\r\n\r\n".encode() for c in chunks)
    calls, client = scripted(httpx.Response(200, content=body))
    provider = GeminiProvider(api_key="k", client=client)
    events = await collect(provider, req("gemini-2.5-pro"))
    assert "streamGenerateContent" in str(calls[0].url) and "alt=sse" in str(calls[0].url)
    final = events[-1].data["response"]
    assert final["message"]["content"][0]["text"] == "Hello"
    assert final["stop_reason"] == "tool_use"

    # and the signature goes back on the next turn
    from agent_harness.types import Message as M
    replay = M(**final["message"])
    payload = provider._payload(CompletionRequest(model="gemini-2.5-pro",
                                                  messages=[M.user("x"), replay]))
    assert payload["contents"][1]["parts"][1]["thoughtSignature"] == "gsig"


async def test_gemini_rpc_errors_in_a_200_are_typed(waits):
    calls, client = scripted(httpx.Response(200, json={"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}}))
    provider = GeminiProvider(api_key="k", client=client, max_retries=0)
    with pytest.raises(RateLimitError):
        await provider.complete(req("gemini-2.5-pro"))


# --- the platforms share all of it ----------------------------------------------------------

CREDS = AWSCredentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")


def frame(event: dict) -> bytes:
    payload = json.dumps({"bytes": base64.b64encode(json.dumps(event).encode()).decode()})
    return encode_event({":message-type": "event", ":event-type": "chunk",
                         ":content-type": "application/json"}, payload.encode())


async def test_bedrock_streams_its_binary_event_stream():
    body = b"".join(frame(e) for e in (START, *TEXT, *END))
    calls, client = scripted(httpx.Response(200, content=body))
    provider = BedrockProvider(region="us-east-1", client=client, credentials=CREDS)
    events = await collect(provider, req("anthropic.claude-opus-5"))
    assert str(calls[0].url).endswith("/model/anthropic.claude-opus-5/"
                                      "invoke-with-response-stream")
    assert calls[0].headers["authorization"].startswith("AWS4-HMAC-SHA256")
    assert calls[0].headers["accept"] == "application/vnd.amazon.eventstream"
    assert [e.text for e in events if e.type == "text"] == ["hello"]
    assert events[-1].data["response"]["usage"]["output_tokens"] == 7


async def test_bedrock_stream_throttling_is_a_rate_limit(waits):
    throttled = encode_event({":message-type": "exception",
                              ":exception-type": "throttlingException"},
                             b'{"message":"Too many requests"}')
    calls, client = scripted(httpx.Response(200, content=throttled),
                             httpx.Response(200, content=b"".join(
                                 frame(e) for e in (START, *TEXT, *END))))
    provider = BedrockProvider(region="us-east-1", client=client, credentials=CREDS)
    events = await collect(provider, req("anthropic.claude-opus-5"))
    assert events[-1].type == "step_end" and len(calls) == 2


async def test_a_corrupt_bedrock_frame_is_caught():
    good = frame(START)
    bad = good[:-1] + bytes([good[-1] ^ 0xFF])
    calls, client = scripted(httpx.Response(200, content=bad))
    provider = BedrockProvider(region="us-east-1", client=client, credentials=CREDS,
                               max_retries=0)
    with pytest.raises(ProviderError, match="CRC"):
        await collect(provider, req("anthropic.claude-opus-5"))


async def test_bedrock_signs_every_attempt_and_escapes_arns(waits):
    calls, client = scripted(err(503, "x"), httpx.Response(200, json=ANTHROPIC_OK))
    provider = BedrockProvider(region="us-east-1", client=client, credentials=CREDS)
    arn = "arn:aws:bedrock:us-east-1:123:inference-profile/us.anthropic.claude-opus-5"
    await provider.complete(req(arn))
    assert len(calls) == 2
    assert all(c.headers["authorization"].startswith("AWS4-HMAC-SHA256") for c in calls)
    assert "arn%3Aaws%3Abedrock" in str(calls[0].url)


async def test_a_bedrock_api_key_replaces_signing(monkeypatch):
    calls, client = scripted(httpx.Response(200, json=ANTHROPIC_OK))
    provider = BedrockProvider(region="us-east-1", client=client, api_key="bedrock-key")
    await provider.complete(req("anthropic.claude-opus-5"))
    assert calls[0].headers["authorization"] == "Bearer bedrock-key"


class RotatingToken:
    refreshable = True

    def __init__(self) -> None:
        self.issued = 0

    def invalidate(self) -> None:
        pass

    async def token(self) -> str:
        self.issued += 1
        return f"token-{self.issued}"


async def test_vertex_refreshes_a_stale_token_once(waits):
    calls, client = scripted(err(401, {"error": {"status": "UNAUTHENTICATED"}}),
                             httpx.Response(200, json=ANTHROPIC_OK))
    provider = VertexProvider(project="p", region="us-east5", access_token="t",
                              client=client)
    provider.auth = RotatingToken()
    assert (await provider.complete(req())).text == "hi"
    assert [c.headers["authorization"] for c in calls] == ["Bearer token-1",
                                                           "Bearer token-2"]
    assert waits == []                           # a refresh is not a back-off


async def test_vertex_is_retried_and_streams_from_its_own_url(waits):
    calls, client = scripted(err(503, "x"),
                             httpx.Response(200, content=sse(START, *TEXT, *END)))
    provider = VertexProvider(project="p", region="us-east5", access_token="t",
                              client=client)
    events = await collect(provider, req())
    assert str(calls[-1].url).endswith("claude-opus-5:streamRawPredict")
    assert json.loads(calls[-1].content)["stream"] is True
    assert "model" not in json.loads(calls[-1].content)
    assert events[-1].type == "step_end"


async def test_azure_streams_from_the_deployment_and_refreshes_entra_tokens(waits):
    class Credential:
        def __init__(self) -> None:
            self.n = 0

        def get_token(self, scope):
            self.n += 1
            return type("T", (), {"token": f"entra-{self.n}", "expires_on": 9e12})()

    body = (b'data: {"choices":[{"delta":{"content":"az"},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n")
    calls, client = scripted(err(401, {"error": {"code": "401"}}),
                             httpx.Response(200, content=body))
    provider = AzureOpenAIProvider(endpoint="https://r.openai.azure.com",
                                   deployment="gpt-prod", credential=Credential(),
                                   client=client)
    events = await collect(provider, req("gpt-4.1"))
    assert "/openai/deployments/gpt-prod/chat/completions" in str(calls[-1].url)
    assert calls[-1].headers["authorization"] == "Bearer entra-2"
    assert events[0].text == "az"


# --- list_models and ping ---------------------------------------------------------------------

async def test_anthropic_lists_models_across_pages():
    calls, client = scripted(
        httpx.Response(200, json={"data": [{"id": "a"}], "has_more": True, "last_id": "a"}),
        httpx.Response(200, json={"data": [{"id": "b"}], "has_more": False}))
    provider = AnthropicProvider(api_key="k", client=client)
    assert await provider.list_models() == ["a", "b"]
    assert "after_id=a" in str(calls[1].url)


async def test_ping_reports_rather_than_raises(waits):
    calls, client = scripted(err(401, {"error": {"message": "invalid x-api-key"}}))
    provider = AnthropicProvider(api_key="bad", client=client)
    result = await provider.ping()
    assert result["ok"] is False and "AuthenticationError" in result["error"]

    calls, client = scripted(httpx.Response(200, json={"data": [{"id": "gpt-4.1"}]}))
    result = await OpenAIProvider(api_key="k", client=client).ping()
    assert result == {**result, "ok": True, "models": 1}


async def test_an_error_body_that_cannot_be_read_still_classifies_and_releases(waits):
    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise httpx.ReadError("gone")
            yield b""  # pragma: no cover

    calls, client = scripted(httpx.Response(503, stream=Broken()),
                             httpx.Response(200, json=ANTHROPIC_OK))
    provider = AnthropicProvider(api_key="k", client=client, max_concurrency=1)
    assert (await provider.complete(req())).text == "hi"   # would hang if the slot leaked
    assert len(calls) == 2
