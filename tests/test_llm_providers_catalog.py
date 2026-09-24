"""list_llm_providers and friends: what each backend needs, before connecting."""

from __future__ import annotations

import json

import httpx
import pytest

from agent_harness import (
    ConfigurationError,
    GroqProvider,
    MistralProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    Provider,
    ProviderField,
    check_llm_provider,
    describe_llm_provider,
    get_provider,
    list_llm_providers,
    ping_llm_provider,
    register_provider,
)
from agent_harness.cli import main
from agent_harness.llm_providers import PROVIDERS, provider_for_model
from agent_harness.llm_providers.base import CompletionRequest
from agent_harness.types import Message

ENV = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
       "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "GROQ_API_KEY",
       "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_BEARER_TOKEN_BEDROCK",
       "AWS_PROFILE", "VLLM_BASE_URL", "OLLAMA_HOST", "OLLAMA_BASE_URL"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


def test_every_registered_provider_is_listed_once_with_its_fields():
    specs = {s.name: s for s in list_llm_providers()}
    assert {"anthropic", "openai", "gemini", "bedrock", "vertex", "vertex-gemini",
            "azure", "azure-foundry", "openai-compatible", "openrouter", "groq",
            "together", "deepseek", "mistral", "xai", "fireworks", "cerebras",
            "ollama", "lmstudio", "vllm"} <= set(specs)
    assert "fake" not in specs and "claude" not in specs      # aliases fold in
    assert "claude" in specs["anthropic"].aliases
    for spec in specs.values():
        assert spec.display_name and spec.description and spec.docs_url, spec.name
        assert spec.capabilities, spec.name
        assert spec.common_fields, spec.name


def test_the_fake_provider_is_listed_only_on_request():
    assert "fake" in {s.name for s in list_llm_providers(include_testing=True)}


def test_required_fields_say_where_they_can_come_from():
    spec = describe_llm_provider("anthropic")
    key = spec.required_fields[0]
    assert key.name == "api_key" and key.secret and key.env == ("ANTHROPIC_API_KEY",)
    assert spec.configured is False and spec.missing == [
        "api_key (or set $ANTHROPIC_API_KEY)"]
    assert "claude-opus-5" in spec.models


def test_an_alias_describes_the_canonical_provider():
    assert describe_llm_provider("claude").name == "anthropic"
    assert describe_llm_provider("AWS").name == "bedrock"


def test_configured_is_read_from_the_environment(monkeypatch):
    assert "groq" not in {s.name for s in list_llm_providers(configured_only=True)}
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    ready = {s.name for s in list_llm_providers(configured_only=True)}
    assert "groq" in ready and "ollama" in ready and "anthropic" not in ready


def test_filtering_by_capability():
    names = {s.name for s in list_llm_providers(capability="embeddings")}
    assert {"openai", "gemini", "mistral"} <= names and "anthropic" not in names


def test_check_reports_sources_but_never_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    check = check_llm_provider("openai")
    assert check.ok and check.found == {"api_key": "$OPENAI_API_KEY"}
    assert "sk-secret-value" not in check.model_dump_json()
    assert "sk-secret-value" not in check.render()


def test_check_understands_either_or_auth():
    missing = check_llm_provider("azure")
    assert not missing.ok
    assert "endpoint (or set $AZURE_OPENAI_ENDPOINT)" in missing.missing
    assert any("api_key" in m and "credential" in m for m in missing.missing)

    ok = check_llm_provider("azure", endpoint="https://r.openai.azure.com",
                            credential=object())
    assert ok.ok and ok.found["credential"] == "argument"


def test_a_half_given_aws_key_pair_is_named(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKID")
    check = check_llm_provider("bedrock")
    assert not check.ok and any("secret_key" in m for m in check.missing)


def test_an_unknown_setting_is_caught_before_connecting():
    check = check_llm_provider("anthropic", api_key="k", endpont="typo")
    assert not check.ok and check.unknown == ["endpont"]
    with pytest.raises(ConfigurationError, match="It accepts: api_key, base_url"):
        get_provider("anthropic", api_key="k", endpont="typo")


def test_an_unknown_provider_points_at_the_catalog():
    with pytest.raises(ConfigurationError, match="list_llm_providers"):
        get_provider("nope")


def test_the_example_builds_the_provider_without_inlining_secrets():
    example = describe_llm_provider("azure").example()
    assert example.startswith("get_provider('azure', endpoint=")
    assert "api_key" not in example
    assert "region='eu-west-1'" in describe_llm_provider("bedrock").example()


def test_a_registered_provider_appears_with_its_own_fields():
    class Custom(Provider):
        name = "custom-llm"
        display_name = "Custom"
        description = "In-house gateway."
        docs_url = "https://example.invalid"
        capabilities = frozenset({"tools"})
        fields = (ProviderField(name="token", type="secret", required=True,
                                env=("CUSTOM_TOKEN",)),)

        async def complete(self, req):  # pragma: no cover - never called
            raise NotImplementedError

    register_provider("custom-llm", Custom)
    try:
        spec = describe_llm_provider("custom-llm")
        assert spec.fields[0].name == "token" and not spec.configured
    finally:
        PROVIDERS.pop("custom-llm")


async def test_ping_without_configuration_says_what_is_missing():
    result = await ping_llm_provider("anthropic")
    assert result["ok"] is False and "ANTHROPIC_API_KEY" in result["error"]


# --- the OpenAI-compatible presets ---------------------------------------------------------

def capture(response=None):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=response or {
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]})

    return seen, httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_a_preset_knows_its_url_and_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    seen, client = capture()
    await GroqProvider(client=client).complete(CompletionRequest(
        model="llama-3.3-70b-versatile", messages=[Message.user("hi")]))
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer gsk_x"


async def test_a_local_server_needs_no_key(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.5:11434")
    seen, client = capture()
    await OllamaProvider(client=client).complete(CompletionRequest(
        model="llama3.2", messages=[Message.user("hi")]))
    assert seen["url"] == "http://10.0.0.5:11434/v1/chat/completions"
    assert "authorization" not in seen["headers"]


async def test_mistral_gets_its_own_spelling():
    seen, client = capture()
    await MistralProvider(api_key="k", client=client).complete(CompletionRequest(
        model="mistral-large-latest", messages=[Message.user("hi")], seed=7,
        user="u1"))
    assert seen["body"]["random_seed"] == 7
    assert "seed" not in seen["body"] and "user" not in seen["body"]


async def test_a_compatible_server_that_omits_call_ids_still_gets_them():
    seen, client = capture({"choices": [{"finish_reason": "stop", "message": {
        "content": None, "tool_calls": [{"function": {"name": "look",
                                                      "arguments": {"q": 1}}}]}}]})
    provider = OpenAICompatibleProvider(base_url="http://gw/v1", client=client)
    response = await provider.complete(CompletionRequest(
        model="any", messages=[Message.user("hi")]))
    call = response.tool_uses[0]
    assert call.id and call.input == {"q": 1}
    assert response.stop_reason == "tool_use"


def test_the_generic_endpoint_insists_on_a_url():
    with pytest.raises(Exception, match="base_url"):
        OpenAICompatibleProvider()


@pytest.mark.parametrize("model,vendor", [
    ("deepseek-chat", "deepseek"), ("grok-4", "xai"), ("mistral-large-latest", "mistral"),
    ("codestral-latest", "mistral"), ("claude-opus-5", "anthropic"), ("gpt-4.1", "openai"),
])
def test_model_ids_route_to_their_vendor(model, vendor):
    assert provider_for_model(model) == vendor


# --- the CLI ---------------------------------------------------------------------------------

def test_the_cli_lists_providers_and_what_they_need(capsys):
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "anthropic" in out and "needs setup" in out and "ollama" in out
    assert "endpoint, api_key | credential" in out


def test_the_cli_describes_one_provider_as_json(capsys):
    assert main(["providers", "vertex", "--json"]) == 0
    spec = json.loads(capsys.readouterr().out)
    assert spec["name"] == "vertex"
    assert {f["name"] for f in spec["fields"]} >= {"project", "region", "access_token"}
