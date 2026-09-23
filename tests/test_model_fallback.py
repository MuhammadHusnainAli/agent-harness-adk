"""Model fallback: when the first model cannot be reached, try the next."""

from __future__ import annotations

import pytest

from agent_harness import Agent, FakeProvider, Harness, ModelRouter
from agent_harness.errors import ProviderError, RateLimitError
from agent_harness.providers.base import CompletionRequest, Provider
from agent_harness.types import Message, ModelResponse, Usage

MODEL = "claude-sonnet-5"


class Unreachable(Provider):
    """A provider that fails the way a real outage does."""

    name = "unreachable"
    BASE_URL = "http://none"

    def __init__(self, error: Exception, **kw):
        super().__init__(api_key="x", **kw)
        self.error = error
        self.calls = 0

    async def complete(self, req):
        self.calls += 1
        raise self.error


class Working(Provider):
    name = "working"
    BASE_URL = "http://none"

    def __init__(self, answer: str = "the fallback answered", **kw):
        super().__init__(api_key="x", **kw)
        self.answer = answer
        self.models: list[str] = []

    async def complete(self, req: CompletionRequest):
        self.models.append(req.model)
        return ModelResponse(message=Message.assistant(self.answer),
                             usage=Usage(calls=1), model=req.model)


class Flaky(Provider):
    """Fails on the first model it is given, answers for anything after."""

    name = "flaky"
    BASE_URL = "http://none"

    def __init__(self, bad: str, **kw):
        super().__init__(api_key="x", **kw)
        self.bad = bad
        self.seen: list[str] = []

    async def complete(self, req: CompletionRequest):
        self.seen.append(req.model)
        if req.model == self.bad:
            raise ProviderError("connection refused", provider="flaky")
        return ModelResponse(message=Message.assistant(f"answered by {req.model}"),
                             usage=Usage(calls=1), model=req.model)


# --- the chain ------------------------------------------------------------------

def test_the_chain_starts_with_the_chosen_model():
    router = ModelRouter(fallbacks=["claude-sonnet-5", "gpt-4.1"])
    assert router.chain("claude-opus-5") == ["claude-opus-5", "claude-sonnet-5",
                                             "gpt-4.1"]


def test_the_chain_never_repeats_a_model():
    router = ModelRouter(fallbacks=["gpt-4.1", "claude-opus-5"])
    assert router.chain("gpt-4.1") == ["gpt-4.1", "claude-opus-5"]


def test_the_chain_can_be_capped():
    router = ModelRouter(fallbacks=["a", "b", "c"], max_attempts=2)
    assert router.chain("first") == ["first", "a"]


def test_no_fallbacks_means_a_chain_of_one():
    assert ModelRouter().chain("claude-opus-5") == ["claude-opus-5"]


# --- in a run ---------------------------------------------------------------------

async def test_an_unreachable_model_falls_through_to_the_next():
    provider = Flaky(bad="claude-opus-5")
    harness = Harness.testing(provider)
    harness.router = ModelRouter(fallbacks=["claude-sonnet-5"])

    agent = Agent("resilient", provider=provider, model="claude-opus-5",
                  harness=harness, memory=False)
    result = await agent.run("hello")

    assert result.ok
    assert result.output == "answered by claude-sonnet-5"
    assert provider.seen == ["claude-opus-5", "claude-sonnet-5"]


async def test_the_fallback_is_recorded_so_you_know_it_happened():
    provider = Flaky(bad="claude-opus-5")
    harness = Harness.testing(provider)
    harness.router = ModelRouter(fallbacks=["claude-sonnet-5"])
    agent = Agent("resilient", provider=provider, model="claude-opus-5",
                  harness=harness, memory=False)

    await agent.run("hello")

    assert any(e.kind == "fallback" for e in harness.journal.entries)
    switched = [e for e in harness.audit.entries if e.action == "model_fallback"]
    assert switched and switched[0].target == "claude-opus-5"
    assert switched[0].detail["to"] == "claude-sonnet-5"
    assert harness.audit.verify()[0]


@pytest.mark.parametrize("error", [
    ProviderError("connection refused", provider="x"),
    ProviderError("gateway timeout", provider="x", status=504),
    RateLimitError("slow down", provider="x", status=429),
    ProviderError("server error", provider="x", status=503),
    OSError("network is unreachable"),
])
async def test_the_failures_worth_retrying_elsewhere(error):
    broken = Unreachable(error)
    good = Working()
    harness = Harness.testing(broken)
    harness.router = ModelRouter(fallbacks=["gpt-4.1"])

    agent = Agent("resilient", provider=broken, model="claude-opus-5",
                  harness=harness, memory=False)
    agent._fallback_providers["gpt-4.1"] = good
    agent._provider = None                     # let the fallback resolve its own
    agent._provider = broken

    # An explicit provider serves every model, so point the second attempt at
    # the working one by making the broken provider fail only once.
    calls = {"n": 0}

    async def complete(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise error
        return await good.complete(req)

    broken.complete = complete                  # type: ignore[method-assign]
    result = await agent.run("hello")

    assert result.ok and result.output == "the fallback answered"
    assert calls["n"] == 2


async def test_a_bad_request_is_not_retried_on_another_model():
    """A 400 means the request is wrong; the next model will reject it too."""
    provider = Unreachable(ProviderError("invalid tool schema", provider="x",
                                         status=400))
    harness = Harness.testing(provider)
    harness.router = ModelRouter(fallbacks=["gpt-4.1", "gemini-2.5-pro"])
    agent = Agent("resilient", provider=provider, model="claude-opus-5",
                  harness=harness, memory=False)

    result = await agent.run("hello")

    assert not result.ok and "invalid tool schema" in result.error
    assert provider.calls == 1                 # tried once, not three times


async def test_the_last_model_in_the_chain_surfaces_its_error():
    provider = Unreachable(ProviderError("everything is down", provider="x"))
    harness = Harness.testing(provider)
    harness.router = ModelRouter(fallbacks=["gpt-4.1"])
    agent = Agent("resilient", provider=provider, model="claude-opus-5",
                  harness=harness, memory=False)

    result = await agent.run("hello")
    assert not result.ok and "everything is down" in result.error
    assert provider.calls == 2                 # both attempts were made


async def test_without_fallbacks_nothing_changes():
    provider = Unreachable(ProviderError("down", provider="x"))
    harness = Harness.testing(provider)
    agent = Agent("plain", provider=provider, model=MODEL, harness=harness,
                  memory=False)

    result = await agent.run("hello")
    assert not result.ok and provider.calls == 1


async def test_health_records_the_model_that_failed():
    provider = Flaky(bad="claude-opus-5")
    harness = Harness.testing(provider)
    harness.router = ModelRouter(fallbacks=["claude-sonnet-5"])
    agent = Agent("resilient", provider=provider, model="claude-opus-5",
                  harness=harness, memory=False)

    await agent.run("hello")

    assert harness.health.component("claude-opus-5", "model").failure_rate == 1.0
    assert harness.health.component("claude-sonnet-5", "model").failure_rate == 0.0


async def test_a_fallback_model_resolves_its_own_provider_when_none_was_given():
    """With no explicit provider, each model in the chain gets the right backend."""
    harness = Harness.testing(FakeProvider(["unused"]))
    agent = Agent("router", model="claude-opus-5", harness=harness, memory=False)

    anthropic = agent._provider_for("claude-opus-5")
    openai = agent._provider_for("gpt-4.1")
    gemini = agent._provider_for("gemini-2.5-pro")

    assert anthropic.name == "anthropic"
    assert openai.name == "openai"
    assert gemini.name == "gemini"
    assert agent._provider_for("gpt-4.1") is openai      # cached, not rebuilt
