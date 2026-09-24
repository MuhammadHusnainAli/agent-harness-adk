"""LLM providers: one contract, every major backend, resolved by name or model id.

    list_llm_providers()                      # what's supported, and what each needs
    describe_llm_provider("bedrock").render() # one provider's fields and env vars
    check_llm_provider("azure")               # is it configured? (offline)
    await ping_llm_provider("openai")         # do the credentials work? (live)

Every backend sends through the same transport, so every one gets the same
retries, backoff, Retry-After handling, typed errors, circuit breaker and stats.
"""

from __future__ import annotations

from .anthropic import AnthropicProvider
from .azure import AzureFoundryProvider, AzureOpenAIProvider
from .base import (
    COMMON_FIELDS,
    MODELS,
    CompletionRequest,
    ModelInfo,
    Provider,
    ProviderField,
    ToolSchema,
    estimate_cost,
    model_info,
    provider_for_model,
    register_model,
)
from .bedrock import BedrockProvider
from .catalog import (
    _CACHE,
    PROVIDERS,
    ProviderCheck,
    ProviderSpec,
    check_llm_provider,
    close_all,
    describe_llm_provider,
    get_provider,
    list_llm_providers,
    ping_llm_provider,
    register_provider,
    resolve_provider,
)
from .compatible import (
    CerebrasProvider,
    DeepSeekProvider,
    FireworksProvider,
    GroqProvider,
    LMStudioProvider,
    MistralProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    OpenRouterProvider,
    TogetherProvider,
    VLLMProvider,
    XAIProvider,
)
from .fake import FakeProvider, hash_embedding, tool_call
from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .parameters import (
    EFFORT_LEVELS,
    GENERATION_PARAMETERS,
    SAMPLING_PARAMETERS,
    Effort,
    ParameterPlan,
    nearest_effort,
    validate_parameters,
)
from .resilience import CircuitBreaker, ProviderStats, RetryEvent, RetryPolicy
from .vertex import GoogleAuth, VertexGeminiProvider, VertexProvider

__all__ = [
    # the contract
    "Provider",
    "CompletionRequest",
    "ToolSchema",
    "ProviderField",
    "COMMON_FIELDS",
    # direct APIs
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    # cloud platforms
    "BedrockProvider",
    "VertexProvider",
    "VertexGeminiProvider",
    "AzureOpenAIProvider",
    "AzureFoundryProvider",
    "GoogleAuth",
    # OpenAI-compatible
    "OpenAICompatibleProvider",
    "OpenRouterProvider",
    "GroqProvider",
    "TogetherProvider",
    "DeepSeekProvider",
    "MistralProvider",
    "XAIProvider",
    "FireworksProvider",
    "CerebrasProvider",
    "OllamaProvider",
    "LMStudioProvider",
    "VLLMProvider",
    # testing
    "FakeProvider",
    "tool_call",
    "hash_embedding",
    # generation parameters
    "Effort",
    "EFFORT_LEVELS",
    "SAMPLING_PARAMETERS",
    "GENERATION_PARAMETERS",
    "ParameterPlan",
    "validate_parameters",
    "nearest_effort",
    # resilience
    "RetryPolicy",
    "RetryEvent",
    "CircuitBreaker",
    "ProviderStats",
    # the catalog
    "PROVIDERS",
    "ProviderSpec",
    "ProviderCheck",
    "list_llm_providers",
    "describe_llm_provider",
    "check_llm_provider",
    "ping_llm_provider",
    "register_provider",
    "get_provider",
    "resolve_provider",
    "close_all",
    # models and pricing
    "ModelInfo",
    "MODELS",
    "model_info",
    "provider_for_model",
    "estimate_cost",
    "register_model",
]

# Re-exported for code that reached into the 0.1 module for it.
_ = _CACHE
