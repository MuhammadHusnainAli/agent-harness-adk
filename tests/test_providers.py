from __future__ import annotations

import json

import httpx
import pytest

from agent_harness import AnthropicProvider, GeminiProvider, OpenAIProvider
from agent_harness.errors import ProviderError, RateLimitError
from agent_harness.llm_providers.base import (
    CompletionRequest,
    ToolSchema,
    estimate_cost,
    model_info,
    provider_for_model,
)
from agent_harness.types import Message, TextBlock, ToolResultBlock, ToolUseBlock, Usage

TOOLS = [ToolSchema(name="lookup", description="Look something up",
                    parameters={"type": "object",
                                "properties": {"q": {"type": "string"}},
                                "required": ["q"],
                                "additionalProperties": False})]

CONVERSATION = [
    Message.user("find the total"),
    Message(role="assistant", content=[ToolUseBlock(id="c1", name="lookup",
                                                    input={"q": "total"})]),
    Message.tool_results([ToolResultBlock(tool_use_id="c1", content="42")]),
]


def capture(response_json, status=200):
    """A mock transport that records the request and replays a canned response."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(status, json=response_json)

    return seen, httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- Anthropic ---------------------------------------------------------------

async def test_anthropic_encodes_system_tools_and_tool_results():
    seen, client = capture({
        "model": "claude-sonnet-5", "stop_reason": "tool_use",
        "content": [{"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "c2", "name": "lookup",
                     "input": {"q": "net"}}],
        "usage": {"input_tokens": 100, "output_tokens": 20,
                  "cache_read_input_tokens": 10},
    })
    provider = AnthropicProvider(api_key="k", client=client)
    response = await provider.complete(CompletionRequest(
        model="claude-sonnet-5", messages=CONVERSATION, system="Be brief.", tools=TOOLS,
    ))

    body = seen["body"]
    assert seen["headers"]["x-api-key"] == "k"
    assert body["system"] == "Be brief."
    assert body["tools"][0]["input_schema"]["properties"]["q"]["type"] == "string"
    assert body["messages"][1]["content"][0]["type"] == "tool_use"
    assert body["messages"][2]["content"][0]["type"] == "tool_result"

    assert response.stop_reason == "tool_use"
    assert response.text == "checking"
    assert response.tool_uses[0].input == {"q": "net"}
    assert response.usage.cache_read_tokens == 10
    assert response.usage.cost_usd > 0


async def test_anthropic_uses_adaptive_thinking_on_current_models():
    seen, client = capture({"content": [], "stop_reason": "end_turn", "usage": {}})
    provider = AnthropicProvider(api_key="k", client=client)
    await provider.complete(CompletionRequest(model="claude-opus-5", thinking=True,
                                              effort="high", temperature=0.7,
                                              messages=[Message.user("hi")]))
    body = seen["body"]
    assert body["thinking"]["type"] == "adaptive"
    assert body["output_config"]["effort"] == "high"
    # Sampling knobs are rejected by the thinking models — they must not be sent.
    assert "temperature" not in body


async def test_anthropic_reports_a_refusal_rather_than_pretending():
    _, client = capture({
        "content": [], "stop_reason": "refusal",
        "stop_details": {"type": "refusal", "explanation": "declined"}, "usage": {},
    })
    provider = AnthropicProvider(api_key="k", client=client)
    response = await provider.complete(CompletionRequest(model="claude-opus-5",
                                                         messages=[Message.user("x")]))
    assert response.stop_reason == "error"
    assert "declined" in response.text


# --- OpenAI ------------------------------------------------------------------

async def test_openai_splits_tool_results_into_tool_messages():
    seen, client = capture({
        "model": "gpt-4.1",
        "choices": [{"finish_reason": "tool_calls", "message": {
            "content": None,
            "tool_calls": [{"id": "c9", "type": "function",
                            "function": {"name": "lookup",
                                         "arguments": '{"q": "net"}'}}],
        }}],
        "usage": {"prompt_tokens": 90, "completion_tokens": 12,
                  "prompt_tokens_details": {"cached_tokens": 40}},
    })
    provider = OpenAIProvider(api_key="k", client=client)
    response = await provider.complete(CompletionRequest(
        model="gpt-4.1", messages=CONVERSATION, system="Be brief.", tools=TOOLS,
    ))

    roles = [m["role"] for m in seen["body"]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert seen["body"]["messages"][3]["tool_call_id"] == "c1"
    assert seen["body"]["tools"][0]["function"]["name"] == "lookup"

    assert response.stop_reason == "tool_use"
    assert response.tool_uses[0].input == {"q": "net"}
    assert response.usage.input_tokens == 50  # cached tokens are billed separately
    assert response.usage.cache_read_tokens == 40


async def test_openai_reasoning_models_get_the_right_token_field():
    seen, client = capture({"choices": [{"finish_reason": "stop",
                                         "message": {"content": "hi"}}], "usage": {}})
    provider = OpenAIProvider(api_key="k", client=client)
    await provider.complete(CompletionRequest(model="o3", messages=[Message.user("x")],
                                              temperature=0.5, effort="xhigh"))
    assert "max_completion_tokens" in seen["body"]
    assert "temperature" not in seen["body"]
    assert seen["body"]["reasoning_effort"] == "high"


# --- Gemini ------------------------------------------------------------------

async def test_gemini_maps_tool_results_back_to_function_names():
    seen, client = capture({
        "modelVersion": "gemini-2.5-pro",
        "candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": "the total is 42"}]}}],
        "usageMetadata": {"promptTokenCount": 60, "candidatesTokenCount": 8},
    })
    provider = GeminiProvider(api_key="k", client=client)
    response = await provider.complete(CompletionRequest(
        model="gemini-2.5-pro", messages=CONVERSATION, system="Be brief.", tools=TOOLS,
    ))

    body = seen["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "Be brief."
    assert body["contents"][1]["parts"][0]["functionCall"]["name"] == "lookup"
    assert body["contents"][2]["parts"][0]["functionResponse"]["name"] == "lookup"
    # Gemini rejects stray JSON-Schema keywords, so they are stripped.
    declaration = body["tools"][0]["functionDeclarations"][0]
    assert "additionalProperties" not in declaration["parameters"]
    assert response.text == "the total is 42"
    assert response.usage.input_tokens == 60


async def test_gemini_marks_tool_use_as_the_stop_reason():
    _, client = capture({
        "candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"functionCall": {"name": "lookup", "args": {"q": "x"}}}]}}],
        "usageMetadata": {},
    })
    provider = GeminiProvider(api_key="k", client=client)
    response = await provider.complete(CompletionRequest(model="gemini-2.5-pro",
                                                         messages=[Message.user("x")]))
    assert response.stop_reason == "tool_use"


# --- shared behaviour ---------------------------------------------------------

async def test_rate_limits_are_retried_then_surfaced():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"error": "slow down"},
                              headers={"retry-after": "0"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="k", client=client, max_retries=2)
    with pytest.raises(RateLimitError):
        await provider.complete(CompletionRequest(model="claude-opus-5",
                                                  messages=[Message.user("x")]))
    assert calls["n"] == 3  # the first attempt plus two retries


async def test_client_errors_fail_fast():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="k", client=client, max_retries=3)
    with pytest.raises(ProviderError):
        await provider.complete(CompletionRequest(model="gpt-4.1",
                                                  messages=[Message.user("x")]))
    assert calls["n"] == 1


async def test_a_missing_key_is_reported_before_any_request():
    provider = AnthropicProvider(api_key="")
    with pytest.raises(ProviderError, match="No API key"):
        await provider.complete(CompletionRequest(model="claude-opus-5",
                                                  messages=[Message.user("x")]))


def test_model_routing_and_pricing():
    assert provider_for_model("claude-opus-5") == "anthropic"
    assert provider_for_model("gpt-4.1-mini") == "openai"
    assert provider_for_model("gemini-2.5-flash") == "gemini"
    assert model_info("claude-sonnet-5-20260101").id == "claude-sonnet-5"
    cost = estimate_cost("claude-opus-5", Usage(input_tokens=1_000_000,
                                                output_tokens=1_000_000))
    assert cost == pytest.approx(30.0)
    assert estimate_cost("some-unknown-model", Usage(input_tokens=999)) == 0.0


async def test_streaming_falls_back_to_one_shot_for_providers_without_sse():
    from agent_harness import FakeProvider

    provider = FakeProvider(["streamed answer"])
    events = [e async for e in provider.stream(
        CompletionRequest(model="fake-1", messages=[Message.user("x")])
    )]
    assert events[0].type == "text" and events[0].text == "streamed answer"
    assert events[-1].type == "step_end"


async def test_anthropic_streaming_assembles_text_and_tool_calls():
    chunks = [
        {"type": "message_start",
         "message": {"model": "claude-opus-5", "usage": {"input_tokens": 10}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "c3", "name": "lookup"}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"q": "net"}'}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
         "usage": {"output_tokens": 5}},
    ]
    body = "".join(f"event: {c['type']}\ndata: {json.dumps(c)}\n\n" for c in chunks)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body,
                              headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="k", client=client)
    texts, tools = [], []
    final = None
    async for event in provider.stream(CompletionRequest(model="claude-opus-5",
                                                         messages=[Message.user("x")])):
        if event.type == "text":
            texts.append(event.text)
        elif event.type == "tool_call":
            tools.append(event.data)
        elif event.type == "step_end":
            final = event.data["response"]
    assert "".join(texts) == "Hello"
    assert tools[0]["input"] == {"q": "net"}
    assert final["stop_reason"] == "tool_use"
    assert final["message"]["content"][0]["text"] == "Hello"


def test_message_helpers():
    message = Message.assistant([TextBlock(text="a"), ToolUseBlock(name="t")])
    assert message.text == "a"
    assert message.tool_uses[0].name == "t"


# --- every connection parameter, mapped per provider --------------------------

FULL = dict(
    max_tokens=4096, effort="max", thinking=True, thinking_budget=8192,
    temperature=0.3, top_p=0.9, top_k=40, seed=7,
    frequency_penalty=0.2, presence_penalty=0.1, stop=["END"],
    parallel_tool_calls=False, cache=True, user="user-42",
    metadata={"team": "support"}, response_mime_type="application/json",
)


async def test_anthropic_maps_what_it_supports_and_drops_what_it_does_not():
    seen, client = capture({"content": [], "stop_reason": "end_turn", "usage": {}})
    provider = AnthropicProvider(api_key="k", client=client)
    await provider.complete(CompletionRequest(
        model="claude-haiku-4-5", messages=[Message.user("x")], tools=TOOLS, **FULL))

    body = seen["body"]
    assert body["max_tokens"] == 4096
    assert body["output_config"]["effort"] == "max"
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert body["temperature"] == 0.3 and body["top_p"] == 0.9 and body["top_k"] == 40
    assert body["stop_sequences"] == ["END"]
    assert body["tool_choice"]["disable_parallel_tool_use"] is True
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["metadata"] == {"team": "support", "user_id": "user-42"}
    # Anthropic has no seed or penalties, so they are dropped rather than invented.
    for absent in ("seed", "frequency_penalty", "presence_penalty"):
        assert absent not in body


async def test_a_thinking_model_is_not_sent_sampling_it_would_reject():
    seen, client = capture({"content": [], "stop_reason": "end_turn", "usage": {}})
    provider = AnthropicProvider(api_key="k", client=client)
    await provider.complete(CompletionRequest(
        model="claude-opus-5", messages=[Message.user("x")], thinking=True,
        temperature=0.5, top_k=10, effort="xhigh"))

    body = seen["body"]
    assert body["thinking"]["type"] == "adaptive"
    assert body["output_config"]["effort"] == "xhigh"
    assert "temperature" not in body and "top_k" not in body


async def test_openai_maps_what_it_supports():
    seen, client = capture({"choices": [{"finish_reason": "stop",
                                         "message": {"content": "x"}}], "usage": {}})
    provider = OpenAIProvider(api_key="k", client=client)
    await provider.complete(CompletionRequest(
        model="gpt-4.1", messages=[Message.user("x")], tools=TOOLS, **FULL))

    body = seen["body"]
    assert body["temperature"] == 0.3 and body["top_p"] == 0.9
    assert body["seed"] == 7
    assert body["frequency_penalty"] == 0.2 and body["presence_penalty"] == 0.1
    assert body["parallel_tool_calls"] is False
    assert body["user"] == "user-42"
    assert body["stop"] == ["END"]
    assert "top_k" not in body                   # OpenAI has none


async def test_effort_maps_onto_openais_three_levels():
    for given, expected in (("low", "low"), ("medium", "medium"), ("high", "high"),
                            ("xhigh", "high"), ("max", "high")):
        seen, client = capture({"choices": [{"finish_reason": "stop",
                                             "message": {"content": "x"}}],
                                "usage": {}})
        provider = OpenAIProvider(api_key="k", client=client)
        await provider.complete(CompletionRequest(
            model="o3", messages=[Message.user("x")], effort=given))
        assert seen["body"]["reasoning_effort"] == expected, given


async def test_gemini_maps_what_it_supports_including_a_thinking_budget():
    seen, client = capture({"candidates": [{"finishReason": "STOP",
                                            "content": {"parts": [{"text": "x"}]}}],
                            "usageMetadata": {}})
    provider = GeminiProvider(api_key="k", client=client)
    await provider.complete(CompletionRequest(
        model="gemini-2.5-pro", messages=[Message.user("x")], tools=TOOLS,
        safety_settings=[{"category": "HARM_CATEGORY_HARASSMENT",
                          "threshold": "BLOCK_ONLY_HIGH"}],
        **FULL))

    gen = seen["body"]["generationConfig"]
    assert gen["temperature"] == 0.3 and gen["topP"] == 0.9 and gen["topK"] == 40
    assert gen["seed"] == 7
    assert gen["frequencyPenalty"] == 0.2 and gen["presencePenalty"] == 0.1
    assert gen["stopSequences"] == ["END"]
    assert gen["thinkingConfig"] == {"includeThoughts": True, "thinkingBudget": 8192}
    assert seen["body"]["safetySettings"][0]["threshold"] == "BLOCK_ONLY_HIGH"


async def test_an_effort_level_becomes_a_thinking_budget_on_gemini():
    for effort, budget in (("low", 1024), ("medium", 8192), ("high", 16384),
                           ("xhigh", 24576), ("max", 32768)):
        seen, client = capture({"candidates": [{"finishReason": "STOP",
                                                "content": {"parts": []}}],
                                "usageMetadata": {}})
        provider = GeminiProvider(api_key="k", client=client)
        await provider.complete(CompletionRequest(
            model="gemini-2.5-pro", messages=[Message.user("x")], effort=effort))
        config = seen["body"]["generationConfig"]["thinkingConfig"]
        assert config["thinkingBudget"] == budget, effort


async def test_extra_reaches_the_payload_untouched():
    seen, client = capture({"content": [], "stop_reason": "end_turn", "usage": {}})
    provider = AnthropicProvider(api_key="k", client=client)
    await provider.complete(CompletionRequest(
        model="claude-opus-5", messages=[Message.user("x")],
        extra={"betas": ["some-beta-2026-01-01"], "speed": "fast"}))
    assert seen["body"]["betas"] == ["some-beta-2026-01-01"]
    assert seen["body"]["speed"] == "fast"


def test_the_request_carries_every_documented_parameter():
    fields = set(CompletionRequest.model_fields)
    assert fields >= {
        "model", "messages", "system", "tools", "tool_choice", "max_tokens",
        "effort", "thinking", "thinking_budget", "temperature", "top_p", "top_k",
        "seed", "frequency_penalty", "presence_penalty", "stop",
        "response_schema", "response_mime_type", "parallel_tool_calls",
        "cache", "speed", "user", "metadata", "safety_settings", "timeout", "extra",
    }
