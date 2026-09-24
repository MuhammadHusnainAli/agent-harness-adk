"""The cloud platforms: Bedrock, Vertex AI and Azure — and SigV4.

These build real requests against a mock transport, so the URL, the headers and
the body are checked exactly. What is not checked here is the far end: no
network call is made, and the endpoints are the documented ones rather than
ones observed in this test run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from agent_harness.errors import ProviderError
from agent_harness.llm_providers import (
    AzureFoundryProvider,
    AzureOpenAIProvider,
    BedrockProvider,
    VertexGeminiProvider,
    VertexProvider,
    get_provider,
)
from agent_harness.llm_providers._sigv4 import AWSCredentials, sign, signing_key
from agent_harness.llm_providers.base import (
    CompletionRequest,
    estimate_cost,
    model_info,
    normalise_model,
)
from agent_harness.types import Message, Usage

CREDS = AWSCredentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
WHEN = datetime(2015, 8, 30, 12, 36, 0, tzinfo=timezone.utc)


def capture(response_json, status=200):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content) if request.content else {}
        return httpx.Response(status, json=response_json)

    return seen, httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- SigV4, against AWS's own vectors ---------------------------------------------

def test_the_signing_key_matches_the_published_derivation():
    """From the AWS documentation's worked example."""
    key = signing_key("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
                      "20150830", "us-east-1", "iam")
    assert key.hex() == (
        "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9")


def test_the_signature_matches_the_get_vanilla_test_vector():
    """From the AWS SigV4 test suite."""
    headers = sign(method="GET", url="https://example.amazonaws.com/",
                   region="us-east-1", service="service", credentials=CREDS,
                   now=WHEN, content_sha_header=False)
    assert headers["Authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/"
        "aws4_request, SignedHeaders=host;x-amz-date, Signature="
        "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31")


def test_signing_is_deterministic_and_input_sensitive():
    common = {"method": "POST", "url": "https://bedrock-runtime.us-east-1."
                                        "amazonaws.com/model/x/invoke",
              "region": "us-east-1", "service": "bedrock", "credentials": CREDS,
              "now": WHEN}
    first = sign(body=b'{"a":1}', **common)["Authorization"]
    assert first == sign(body=b'{"a":1}', **common)["Authorization"]
    assert first != sign(body=b'{"a":2}', **common)["Authorization"]
    assert first != sign(**{**common, "region": "eu-west-1"}, body=b'{"a":1}')[
        "Authorization"]


def test_a_session_token_is_signed_in():
    headers = sign(method="GET", url="https://x.amazonaws.com/", region="us-east-1",
                   service="s", credentials=AWSCredentials("k", "s", "session-token"),
                   now=WHEN)
    assert headers["x-amz-security-token"] == "session-token"
    assert "x-amz-security-token" in headers["Authorization"]


def test_query_parameters_are_canonicalised():
    a = sign(method="GET", url="https://x.amazonaws.com/?b=2&a=1", region="r",
             service="s", credentials=CREDS, now=WHEN)["Authorization"]
    b = sign(method="GET", url="https://x.amazonaws.com/?a=1&b=2", region="r",
             service="s", credentials=CREDS, now=WHEN)["Authorization"]
    assert a == b        # order in the URL must not change the signature


# --- Bedrock --------------------------------------------------------------------------

async def test_bedrock_signs_and_puts_the_model_in_the_url():
    seen, client = capture({
        "content": [{"type": "text", "text": "hello from bedrock"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 4},
    })
    provider = BedrockProvider(region="eu-west-1", client=client,
                               credentials=CREDS)
    response = await provider.complete(CompletionRequest(
        model="anthropic.claude-opus-5", system="Be brief.",
        messages=[Message.user("hello")], max_tokens=512))

    assert seen["url"] == ("https://bedrock-runtime.eu-west-1.amazonaws.com/"
                           "model/anthropic.claude-opus-5/invoke")
    assert seen["headers"]["authorization"].startswith("AWS4-HMAC-SHA256 ")
    assert "eu-west-1/bedrock/aws4_request" in seen["headers"]["authorization"]
    assert "x-amz-date" in seen["headers"]

    body = seen["body"]
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert "model" not in body                 # it is in the URL
    assert body["system"] == "Be brief."
    assert body["max_tokens"] == 512

    assert response.text == "hello from bedrock"
    assert response.usage.input_tokens == 12
    assert response.usage.cost_usd > 0         # priced as the model it is


async def test_bedrock_without_credentials_says_what_to_do():
    provider = BedrockProvider(region="us-east-1",
                               credentials=AWSCredentials("", ""))
    with pytest.raises(ProviderError, match="AWS_ACCESS_KEY_ID"):
        await provider.complete(CompletionRequest(model="anthropic.claude-opus-5",
                                                  messages=[Message.user("x")]))


async def test_bedrock_tools_survive_the_round_trip():
    from agent_harness.llm_providers.base import ToolSchema

    seen, client = capture({"content": [], "stop_reason": "end_turn", "usage": {}})
    provider = BedrockProvider(region="us-east-1", client=client, credentials=CREDS)
    await provider.complete(CompletionRequest(
        model="anthropic.claude-opus-5", messages=[Message.user("x")],
        tools=[ToolSchema(name="lookup", description="Look up",
                          parameters={"type": "object", "properties": {}})]))
    assert seen["body"]["tools"][0]["name"] == "lookup"


# --- Vertex AI --------------------------------------------------------------------------

async def test_vertex_builds_the_publisher_url_and_moves_the_version_into_the_body():
    seen, client = capture({
        "content": [{"type": "text", "text": "hello from vertex"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 9, "output_tokens": 3},
    })
    provider = VertexProvider(project="my-project", region="europe-west1",
                              access_token="a-token", client=client)
    response = await provider.complete(CompletionRequest(
        model="claude-opus-5", messages=[Message.user("hello")]))

    assert seen["url"] == (
        "https://europe-west1-aiplatform.googleapis.com/v1/projects/my-project/"
        "locations/europe-west1/publishers/anthropic/models/claude-opus-5:rawPredict")
    assert seen["headers"]["authorization"] == "Bearer a-token"
    assert seen["body"]["anthropic_version"] == "vertex-2023-10-16"
    assert "model" not in seen["body"]
    assert response.text == "hello from vertex"


async def test_gemini_on_vertex_uses_the_google_publisher():
    seen, client = capture({
        "candidates": [{"finishReason": "STOP",
                        "content": {"parts": [{"text": "gemini on vertex"}]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
    })
    provider = VertexGeminiProvider(project="p", region="us-central1",
                                    access_token="t", client=client)
    response = await provider.complete(CompletionRequest(
        model="gemini-2.5-pro", messages=[Message.user("hi")], system="Be brief."))

    assert "publishers/google/models/gemini-2.5-pro:generateContent" in seen["url"]
    assert seen["body"]["systemInstruction"]["parts"][0]["text"] == "Be brief."
    assert response.text == "gemini on vertex"


def test_the_global_region_uses_the_unprefixed_host():
    provider = VertexProvider(project="p", region="global", access_token="t")
    assert provider.endpoint.startswith("https://aiplatform.googleapis.com/")


def test_vertex_without_a_project_says_so():
    import os

    saved = os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
    try:
        with pytest.raises(ProviderError, match="needs a project"):
            VertexProvider(access_token="t")
    finally:
        if saved:
            os.environ["GOOGLE_CLOUD_PROJECT"] = saved


# --- Azure ---------------------------------------------------------------------------------

async def test_azure_openai_uses_a_deployment_and_an_api_key():
    seen, client = capture({
        "model": "gpt-4.1",
        "choices": [{"finish_reason": "stop",
                     "message": {"content": "hello from azure"}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    })
    provider = AzureOpenAIProvider(endpoint="https://my-resource.openai.azure.com",
                                   deployment="gpt-4.1-prod",
                                   api_version="2024-10-21",
                                   api_key="secret", client=client)
    response = await provider.complete(CompletionRequest(
        model="gpt-4.1", messages=[Message.user("hello")]))

    assert seen["url"] == ("https://my-resource.openai.azure.com/openai/deployments/"
                           "gpt-4.1-prod/chat/completions?api-version=2024-10-21")
    assert seen["headers"]["api-key"] == "secret"
    assert "model" not in seen["body"]          # the deployment is the model
    assert response.text == "hello from azure"
    assert response.usage.input_tokens == 7


async def test_the_model_id_is_used_when_no_deployment_is_given():
    seen, client = capture({"choices": [{"finish_reason": "stop",
                                         "message": {"content": "x"}}], "usage": {}})
    provider = AzureOpenAIProvider(endpoint="https://r.openai.azure.com",
                                   api_key="k", client=client)
    await provider.complete(CompletionRequest(model="my-deployment",
                                              messages=[Message.user("x")]))
    assert "/deployments/my-deployment/" in seen["url"]


async def test_entra_id_is_used_instead_of_a_key_when_given():
    class Credential:
        def __init__(self):
            self.calls = 0

        def get_token(self, scope):
            self.calls += 1
            return type("T", (), {"token": "entra-token",
                                  "expires_on": 9_999_999_999})()

    credential = Credential()
    seen, client = capture({"choices": [{"finish_reason": "stop",
                                         "message": {"content": "x"}}], "usage": {}})
    provider = AzureOpenAIProvider(endpoint="https://r.openai.azure.com",
                                   deployment="d", credential=credential,
                                   client=client)
    await provider.complete(CompletionRequest(model="d",
                                              messages=[Message.user("x")]))
    await provider.complete(CompletionRequest(model="d",
                                              messages=[Message.user("y")]))

    assert seen["headers"]["authorization"] == "Bearer entra-token"
    assert credential.calls == 1                # cached until it expires


async def test_azure_foundry_routes_differently_and_keeps_the_model_in_the_body():
    seen, client = capture({"choices": [{"finish_reason": "stop",
                                         "message": {"content": "from foundry"}}],
                            "usage": {}})
    provider = AzureFoundryProvider(endpoint="https://my.services.ai.azure.com",
                                    deployment="claude-opus-5", api_key="k",
                                    client=client)
    response = await provider.complete(CompletionRequest(
        model="claude-opus-5", messages=[Message.user("hi")]))

    assert "/models/chat/completions?api-version=" in seen["url"]
    assert seen["body"]["model"] == "claude-opus-5"
    assert response.text == "from foundry"


async def test_the_foundry_route_can_be_overridden():
    seen, client = capture({"choices": [{"finish_reason": "stop",
                                         "message": {"content": "x"}}], "usage": {}})
    provider = AzureFoundryProvider(
        endpoint="https://my.services.ai.azure.com", deployment="d", api_key="k",
        client=client, path_template="/openai/v1/chat/completions")
    await provider.complete(CompletionRequest(model="d",
                                              messages=[Message.user("x")]))
    assert seen["url"].endswith("/openai/v1/chat/completions")


def test_azure_without_an_endpoint_says_so():
    import os

    saved = os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
    try:
        with pytest.raises(ProviderError, match="needs an endpoint"):
            AzureOpenAIProvider(api_key="k")
    finally:
        if saved:
            os.environ["AZURE_OPENAI_ENDPOINT"] = saved


# --- model ids across platforms ----------------------------------------------------------

def test_the_same_model_is_recognised_whatever_it_is_called():
    for name in ("claude-opus-5", "anthropic.claude-opus-5",
                 "us.anthropic.claude-opus-5", "claude-opus-5@20260401"):
        assert normalise_model(name) == "claude-opus-5"
        assert model_info(name).id == "claude-opus-5"


def test_a_platform_prefixed_model_costs_the_same():
    usage = Usage(input_tokens=1_000_000, output_tokens=500_000)
    direct = estimate_cost("claude-opus-5", usage)
    assert estimate_cost("anthropic.claude-opus-5", usage) == direct
    assert estimate_cost("us.anthropic.claude-opus-5", usage) == direct
    assert direct > 0


def test_the_platforms_are_in_the_registry():
    for name in ("bedrock", "aws", "vertex", "gcp", "vertex-gemini",
                 "azure", "azure-openai", "azure-foundry", "foundry"):
        assert name in __import__("agent_harness.llm_providers", fromlist=["PROVIDERS"]
                                  ).PROVIDERS, name


def test_a_platform_provider_is_built_by_name():
    provider = get_provider("bedrock", cached=False, region="us-west-2",
                            credentials=CREDS)
    assert isinstance(provider, BedrockProvider)
    assert provider.region == "us-west-2"


def test_the_platforms_are_importable_from_the_package_root():
    """A user should not have to know which submodule a provider lives in."""
    import agent_harness as ah

    for name in ("BedrockProvider", "VertexProvider", "VertexGeminiProvider",
                 "AzureOpenAIProvider", "AzureFoundryProvider", "GoogleAuth"):
        assert hasattr(ah, name), name
        assert name in ah.__all__, name
