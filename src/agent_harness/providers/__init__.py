"""Model providers: one contract, four backends, resolved by name or model id."""

from __future__ import annotations

from typing import Any

from ..errors import ConfigurationError
from .anthropic import AnthropicProvider
from .azure import AzureFoundryProvider, AzureOpenAIProvider
from .base import (
    MODELS,
    CompletionRequest,
    ModelInfo,
    Provider,
    ToolSchema,
    estimate_cost,
    model_info,
    provider_for_model,
    register_model,
)
from .bedrock import BedrockProvider
from .fake import FakeProvider, hash_embedding, tool_call
from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .vertex import GoogleAuth, VertexGeminiProvider, VertexProvider

__all__ = [
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "FakeProvider",
    "BedrockProvider",
    "VertexProvider",
    "VertexGeminiProvider",
    "AzureOpenAIProvider",
    "AzureFoundryProvider",
    "GoogleAuth",
    "Provider",
    "CompletionRequest",
    "ToolSchema",
    "ModelInfo",
    "MODELS",
    "model_info",
    "provider_for_model",
    "estimate_cost",
    "register_model",
    "register_provider",
    "get_provider",
    "resolve_provider",
    "tool_call",
    "hash_embedding",
]

PROVIDERS: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "fake": FakeProvider,
    # The same models, served by a cloud platform.
    "bedrock": BedrockProvider,
    "aws": BedrockProvider,
    "vertex": VertexProvider,
    "vertex-gemini": VertexGeminiProvider,
    "gcp": VertexProvider,
    "azure": AzureOpenAIProvider,
    "azure-openai": AzureOpenAIProvider,
    "azure-foundry": AzureFoundryProvider,
    "foundry": AzureFoundryProvider,
}

_CACHE: dict[tuple[str, tuple], Provider] = {}


def register_provider(name: str, cls: type[Provider]) -> None:
    """Add your own backend: `register_provider("bedrock", MyProvider)`."""
    PROVIDERS[name.lower()] = cls


def get_provider(name: str, *, cached: bool = True, **kwargs: Any) -> Provider:
    """Build (or reuse) a provider by vendor name."""
    key = name.lower()
    cls = PROVIDERS.get(key)
    if cls is None:
        raise ConfigurationError(
            f"Unknown provider {name!r}. Known: {', '.join(sorted(set(PROVIDERS)))}"
        )
    if not cached or kwargs:
        return cls(**kwargs)
    cache_key = (key, ())
    if cache_key not in _CACHE:
        _CACHE[cache_key] = cls()
    return _CACHE[cache_key]


def resolve_provider(provider: Provider | str | None, model: str | None) -> Provider:
    """Accept a Provider, a vendor name, or nothing at all and work it out."""
    if isinstance(provider, Provider):
        return provider
    if isinstance(provider, str):
        return get_provider(provider)
    if model:
        return get_provider(provider_for_model(model))
    return get_provider("anthropic")


async def close_all() -> None:
    """Close every cached provider's HTTP client. Call it on shutdown."""
    for prov in list(_CACHE.values()):
        await prov.aclose()
    _CACHE.clear()
