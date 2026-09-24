"""Generation parameters: validated once, planned per provider, never dropped silently."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from pydantic import ValidationError

from agent_harness import (
    Agent,
    AnthropicProvider,
    BedrockProvider,
    ConfigurationError,
    FakeProvider,
    GeminiProvider,
    GroqProvider,
    Harness,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
    SubAgentSpec,
    VLLMProvider,
    XAIProvider,
    describe_llm_provider,
    validate_parameters,
)
from agent_harness.llm_providers._sigv4 import AWSCredentials
from agent_harness.llm_providers.base import CompletionRequest
from agent_harness.llm_providers.parameters import nearest_effort
from agent_harness.subagents.builder import build_agent
from agent_harness.types import Message


def req(model: str, **kw) -> CompletionRequest:
    return CompletionRequest(model=model, messages=[Message.user("x")], **kw)


def plan(provider, model: str, **kw):
    return provider.plan_parameters(req(model, **kw))


def body(provider, model: str, **kw) -> dict:
    return provider.explain(req(model, **kw))["payload"]


ANTHROPIC = AnthropicProvider(api_key="k")
OPENAI = OpenAIProvider(api_key="k")
GEMINI = GeminiProvider(api_key="k")


# --- validation ---------------------------------------------------------------------

def test_good_values_pass_and_unset_ones_are_left_out():
    assert validate_parameters(temperature=0.7, top_p=None, effort="high", seed=3) == {
        "temperature": 0.7, "effort": "high", "seed": 3}


def test_every_bad_value_is_reported_at_once():
    with pytest.raises(ConfigurationError) as info:
        validate_parameters(temperature=3, top_p=1.5, top_k=0, presence_penalty=-5,
                            effort="extreme", max_tokens=0, repetition_penalty=0,
                            thinking="yes", seed=1.5)
    message = str(info.value)
    for name in ("temperature=3", "top_p=1.5", "top_k=0", "presence_penalty=-5",
                 "effort='extreme'", "max_tokens=0", "repetition_penalty=0",
                 "thinking='yes'", "seed=1.5"):
        assert name in message, name


def test_an_agent_with_a_bad_parameter_fails_when_built_not_mid_run():
    with pytest.raises(ConfigurationError, match="temperature=5"):
        Agent("a", provider=FakeProvider(), temperature=5, memory=False)
    with pytest.raises(ConfigurationError, match="frequency_penalty=9"):
        Agent("a", provider=FakeProvider(), model_options={"frequency_penalty": 9},
              memory=False)


def test_the_request_itself_refuses_out_of_range_values():
    with pytest.raises(ValidationError):
        req("gpt-4.1", temperature=2.5)
    with pytest.raises(ValidationError):
        req("gpt-4.1", effort="turbo")


@pytest.mark.parametrize("asked,levels,got", [
    ("high", ("low", "medium", "high"), "high"),
    ("max", ("low", "medium", "high"), "high"),
    ("xhigh", ("low", "high"), "high"),
    ("medium", ("low", "high"), "low"),
    ("none", ("low", "medium", "high"), "low"),
    ("minimal", ("minimal", "low"), "minimal"),
])
def test_effort_maps_to_the_nearest_level_a_model_has(asked, levels, got):
    assert nearest_effort(asked, levels) == got


# --- Anthropic, Bedrock, Vertex ------------------------------------------------------------

def test_claude_takes_temperature_or_top_p_and_clamps_temperature():
    p = plan(ANTHROPIC, "claude-haiku-4-5", temperature=1.4, top_p=0.8, top_k=20,
             seed=1, presence_penalty=0.5)
    assert p.values["temperature"] == 1.0 and "temperature" in p.adjusted
    assert "top_p" in p.dropped and p.values["top_k"] == 20
    assert set(p.dropped) >= {"seed", "presence_penalty"}
    assert "has no seed" in p.dropped["seed"]


def test_a_small_thinking_budget_is_raised_to_claudes_minimum():
    p = plan(ANTHROPIC, "claude-haiku-4-5", thinking_budget=200, max_tokens=4000)
    assert p.values["thinking"] is True                  # a budget asks for thinking
    assert p.values["thinking_budget"] == 1024


def test_adaptive_models_are_recognised_under_any_platform_name():
    provider = BedrockProvider(region="us-east-1", credentials=AWSCredentials("a", "b"))
    payload = body(provider, "us.anthropic.claude-opus-5", thinking=True,
                   temperature=0.2, thinking_budget=5000)
    assert payload["thinking"]["type"] == "adaptive"
    assert "temperature" not in payload and "budget_tokens" not in payload["thinking"]


def test_effort_none_turns_claude_thinking_off_and_minimal_becomes_low():
    payload = body(ANTHROPIC, "claude-opus-5", effort="none", thinking=True)
    assert "thinking" not in payload and "output_config" not in payload
    assert body(ANTHROPIC, "claude-opus-5", effort="minimal")["output_config"] == {
        "effort": "low"}


def test_claude_does_not_think_when_a_tool_is_forced():
    p = ANTHROPIC.plan_parameters(CompletionRequest(
        model="claude-haiku-4-5", messages=[Message.user("x")], thinking=True,
        tool_choice="any"))
    assert "thinking" in p.dropped


# --- OpenAI ------------------------------------------------------------------------------

def test_gpt_4_1_takes_sampling_but_has_no_effort():
    p = plan(OPENAI, "gpt-4.1", temperature=0.5, top_p=0.9, top_k=5, seed=3,
             frequency_penalty=0.4, presence_penalty=0.2, effort="high")
    assert p.values == {**p.values, "temperature": 0.5, "top_p": 0.9, "seed": 3,
                        "frequency_penalty": 0.4, "presence_penalty": 0.2}
    assert {"top_k", "effort"} <= set(p.dropped)


def test_a_reasoning_model_gets_effort_and_no_sampling_or_stop():
    payload = body(OPENAI, "o3", temperature=0.5, presence_penalty=0.1, stop=["x"],
                   effort="none")
    assert payload["reasoning_effort"] == "low"
    assert payload["max_completion_tokens"] == 8192
    for absent in ("temperature", "presence_penalty", "stop"):
        assert absent not in payload
    assert body(OPENAI, "gpt-5", effort="minimal")["reasoning_effort"] == "minimal"
    assert body(OPENAI, "gpt-5", thinking=False)["reasoning_effort"] == "minimal"
    assert body(OPENAI, "gpt-5", effort="max")["reasoning_effort"] == "high"


# --- the OpenAI-compatible hosts ----------------------------------------------------------------

def test_open_reasoning_models_get_reasoning_effort_wherever_they_are_hosted():
    groq = GroqProvider(api_key="k")
    assert body(groq, "openai/gpt-oss-120b", effort="xhigh")["reasoning_effort"] == "high"
    llama = plan(groq, "llama-3.3-70b-versatile", effort="high", top_k=5)
    assert {"effort", "top_k"} <= set(llama.dropped)


def test_openrouter_speaks_its_reasoning_object():
    router = OpenRouterProvider(api_key="k")
    payload = body(router, "anthropic/claude-sonnet-5", effort="high", top_k=40,
                   min_p=0.05, repetition_penalty=1.1)
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["top_k"] == 40 and payload["min_p"] == 0.05
    assert payload["repetition_penalty"] == 1.1
    assert body(router, "x/y", thinking_budget=2000, effort="low")["reasoning"] == {
        "max_tokens": 2000}
    assert body(router, "x/y", effort="none")["reasoning"] == {"enabled": False}


def test_vllm_passes_every_sampling_knob():
    payload = body(VLLMProvider(base_url="http://gpu:8000/v1"), "qwen3-32b",
                   temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                   repetition_penalty=1.05, presence_penalty=0.3, seed=11)
    assert {k: payload[k] for k in ("top_k", "min_p", "repetition_penalty", "seed")} == {
        "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.05, "seed": 11}


def test_grok_reasoning_models_lose_what_they_reject():
    xai = XAIProvider(api_key="k")
    payload = body(xai, "grok-4", presence_penalty=0.5, stop=["x"], temperature=0.4)
    assert "presence_penalty" not in payload and "stop" not in payload
    assert payload["temperature"] == 0.4
    assert body(xai, "grok-3-mini", effort="medium")["reasoning_effort"] == "low"


def test_mistral_seed_is_renamed_after_planning():
    assert body(MistralProvider(api_key="k"), "mistral-large-latest", seed=4)[
        "random_seed"] == 4


# --- Gemini ---------------------------------------------------------------------------------

def thinking_config(model: str, **kw) -> dict:
    return body(GEMINI, model, **kw)["generationConfig"].get("thinkingConfig", {})


def test_gemini_2_5_pro_cannot_switch_thinking_off():
    assert thinking_config("gemini-2.5-pro", effort="none")["thinkingBudget"] == 128
    assert thinking_config("gemini-2.5-pro", thinking=False)["thinkingBudget"] == 128
    assert thinking_config("gemini-2.5-flash", effort="none")["thinkingBudget"] == 0


def test_gemini_budgets_are_fitted_to_the_model():
    assert thinking_config("gemini-2.5-flash", thinking_budget=50_000)[
        "thinkingBudget"] == 24576
    assert thinking_config("gemini-2.5-flash", thinking_budget=-1)["thinkingBudget"] == -1


def test_gemini_3_takes_a_thinking_level():
    config = thinking_config("gemini-3-pro-preview", effort="medium", thinking=True)
    assert config == {"includeThoughts": True, "thinkingLevel": "low"}
    assert thinking_config("gemini-3-flash", effort="medium")["thinkingLevel"] == "medium"
    assert thinking_config("gemini-3-flash", effort="none")["thinkingLevel"] == "minimal"


def test_a_model_that_does_not_think_gets_no_thinking_config():
    p = plan(GEMINI, "gemini-2.0-flash", effort="high", thinking=True)
    assert {"effort", "thinking"} <= set(p.dropped)
    assert thinking_config("gemini-2.0-flash", effort="high") == {}


# --- explain and reporting -------------------------------------------------------------------

def test_explain_shows_the_plan_and_the_payload_without_sending():
    out = OPENAI.explain(req("o3", temperature=0.3, effort="max"))
    assert out["sent"]["effort"] == "high"
    assert "temperature" in out["dropped"] and "effort" in out["adjusted"]
    assert out["payload"]["reasoning_effort"] == "high"


def test_each_drop_is_logged_once(caplog):
    provider = AnthropicProvider(api_key="k")
    with caplog.at_level(logging.INFO, logger="agent_harness.llm_providers"):
        for _ in range(3):
            provider.plan_parameters(req("claude-haiku-4-5-log-once", seed=99, top_k=3))
    lines = [r.getMessage() for r in caplog.records if "dropped seed" in r.getMessage()]
    assert len(lines) == 1


async def test_the_planned_payload_is_what_goes_on_the_wire():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                      "message": {"content": "ok"}}]})

    provider = OpenAIProvider(api_key="k",
                              client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    request = req("gpt-4.1", temperature=0.2, top_k=4, frequency_penalty=0.3)
    await provider.complete(request)
    expected = provider.explain(request)["payload"]
    assert seen == json.loads(json.dumps(expected))


# --- the agent, its versions, its sub-agents ---------------------------------------------------

async def test_the_agent_sends_every_parameter_it_was_given():
    provider = FakeProvider(["ok"])
    agent = Agent("a", provider=provider, harness=Harness.testing(provider), memory=False,
                  temperature=0.4, top_p=0.9, top_k=30, min_p=0.1, frequency_penalty=0.5,
                  presence_penalty=0.25, repetition_penalty=1.1, seed=5, effort="low",
                  thinking=True, thinking_budget=2048, stop=["STOP"])
    await agent.run("hi")
    sent = provider.requests[0]
    assert (sent.temperature, sent.top_p, sent.top_k, sent.min_p) == (0.4, 0.9, 30, 0.1)
    assert (sent.frequency_penalty, sent.presence_penalty) == (0.5, 0.25)
    assert (sent.repetition_penalty, sent.seed, sent.effort) == (1.1, 5, "low")
    assert (sent.thinking, sent.thinking_budget, sent.stop) == (True, 2048, ["STOP"])


async def test_a_version_can_change_the_sampling():
    provider = FakeProvider(["ok"])
    agent = Agent("a", provider=provider, harness=Harness.testing(provider), memory=False,
                  top_p=0.9, version="precise",
                  versions={"precise": {"top_p": 0.5, "presence_penalty": 0.4,
                                        "effort": "high"}})
    await agent.run("hi")
    sent = provider.requests[0]
    assert (sent.top_p, sent.presence_penalty, sent.effort) == (0.5, 0.4, "high")


def test_a_sub_agent_spec_carries_the_full_vocabulary():
    parent = Agent("parent", provider=FakeProvider(), memory=False)
    child = build_agent(SubAgentSpec(name="c", description="d", top_k=12,
                                     frequency_penalty=0.3, thinking=True, seed=8),
                        parent)
    assert child.model_options["top_k"] == 12 and child.model_options["seed"] == 8
    assert child.model_options["frequency_penalty"] == 0.3 and child.thinking is True


def test_the_catalog_says_which_parameters_each_provider_takes():
    assert describe_llm_provider("anthropic").parameters == ["temperature", "top_p", "top_k"]
    assert "min_p" in describe_llm_provider("vllm").parameters
    assert "top_k" not in describe_llm_provider("openai").parameters
