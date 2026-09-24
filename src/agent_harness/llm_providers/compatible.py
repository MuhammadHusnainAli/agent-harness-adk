"""Every vendor that speaks the OpenAI Chat Completions format, preconfigured.

Each preset is a few lines: where the API lives, which environment variable
holds the key, and any parameter the server spells differently. The encoding,
streaming, retries, circuit breaking and typed errors are all `OpenAIProvider`'s.

    Agent("fast", model="llama-3.3-70b-versatile", provider="groq")
    Agent("local", model="qwen3:8b", provider="ollama")
    get_provider("openai-compatible", base_url="https://my-gateway/v1", api_key="...")

Local servers (Ollama, LM Studio, vLLM) run without a key; the rest need one.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

from ..errors import ProviderError
from .base import ProviderField
from .openai import OpenAIProvider

__all__ = [
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
    "COMPATIBLE_PROVIDERS",
]

_CAPS = frozenset({"streaming", "tools", "json_schema", "list_models"})


def _key(env: str, example: str = "", *, required: bool = True,
         description: str = "") -> ProviderField:
    return ProviderField(name="api_key", type="secret", required=required, env=(env,),
                         example=example or None,
                         description=description or f"The API key (${env}).")


def _url(default: str, env: str = "", *, required: bool = False) -> ProviderField:
    return ProviderField(name="base_url", type="url", required=required,
                         env=(env,) if env else (), default=default or None,
                         description="Where the OpenAI-compatible API lives.")


class OpenAICompatibleProvider(OpenAIProvider):
    """Any server that speaks the OpenAI format — a gateway, a proxy, your own.

    Presets subclass this and fill in the class attributes; used directly it needs
    a `base_url`, and a key only if the server asks for one.
    """

    name: ClassVar[str] = "openai-compatible"
    env_key: ClassVar[str] = "OPENAI_COMPATIBLE_API_KEY"
    default_model: ClassVar[str] = ""
    BASE_URL: ClassVar[str] = ""
    #: An environment variable that overrides BASE_URL.
    base_url_env: ClassVar[str] = "OPENAI_COMPATIBLE_BASE_URL"
    key_required: ClassVar[bool] = False
    #: Only OpenAI's own models take `max_completion_tokens`.
    reasoning_prefixes: ClassVar[tuple[str, ...]] = ("o1", "o3", "o4", "gpt-5")
    embedding_model: ClassVar[str] = ""

    display_name: ClassVar[str] = "OpenAI-compatible"
    description: ClassVar[str] = ("Any endpoint that speaks the OpenAI Chat Completions "
                                  "format: gateways, proxies, self-hosted servers.")
    docs_url: ClassVar[str] = "https://platform.openai.com/docs/api-reference/chat"
    fields: ClassVar[tuple[ProviderField, ...]] = (
        _url("", "OPENAI_COMPATIBLE_BASE_URL", required=True),
        _key("OPENAI_COMPATIBLE_API_KEY", required=False,
             description="Sent as a bearer token, if the server wants one."),
    )
    capabilities: ClassVar[frozenset[str]] = _CAPS

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None,
                 **kw: Any) -> None:
        base_url = (base_url or (os.environ.get(self.base_url_env)
                                 if self.base_url_env else None) or self.BASE_URL)
        if not base_url:
            env = f" or set {self.base_url_env}" if self.base_url_env else ""
            raise ProviderError(f"{self.display_name} needs base_url=...{env}",
                                provider=self.name)
        super().__init__(api_key, base_url=base_url, **kw)


class OpenRouterProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "openrouter"
    env_key: ClassVar[str] = "OPENROUTER_API_KEY"
    BASE_URL: ClassVar[str] = "https://openrouter.ai/api/v1"
    base_url_env: ClassVar[str] = ""
    key_required: ClassVar[bool] = True
    default_model: ClassVar[str] = "anthropic/claude-sonnet-5"
    display_name: ClassVar[str] = "OpenRouter"
    description: ClassVar[str] = "Hundreds of models from every major lab behind one key."
    docs_url: ClassVar[str] = "https://openrouter.ai/docs"
    fields: ClassVar[tuple[ProviderField, ...]] = (
        _key("OPENROUTER_API_KEY", "sk-or-..."),
        ProviderField(name="app_url", description="Your site, sent as HTTP-Referer "
                                                  "for OpenRouter's rankings."),
        ProviderField(name="app_name", description="Your app's name, sent as X-Title."),
    )
    capabilities: ClassVar[frozenset[str]] = _CAPS | {"vision", "thinking"}

    def __init__(self, api_key: str | None = None, *, app_url: str | None = None,
                 app_name: str | None = None, **kw: Any) -> None:
        headers = dict(kw.pop("headers", None) or {})
        if app_url:
            headers.setdefault("HTTP-Referer", app_url)
        if app_name:
            headers.setdefault("X-Title", app_name)
        super().__init__(api_key, headers=headers, **kw)


class GroqProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "groq"
    env_key: ClassVar[str] = "GROQ_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.groq.com/openai/v1"
    base_url_env: ClassVar[str] = ""
    key_required: ClassVar[bool] = True
    default_model: ClassVar[str] = "llama-3.3-70b-versatile"
    display_name: ClassVar[str] = "Groq"
    description: ClassVar[str] = "Open-weight models on Groq's LPUs — very low latency."
    docs_url: ClassVar[str] = "https://console.groq.com/docs/api-reference"
    fields: ClassVar[tuple[ProviderField, ...]] = (_key("GROQ_API_KEY", "gsk_..."),)


class TogetherProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "together"
    env_key: ClassVar[str] = "TOGETHER_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.together.xyz/v1"
    base_url_env: ClassVar[str] = ""
    key_required: ClassVar[bool] = True
    default_model: ClassVar[str] = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    display_name: ClassVar[str] = "Together AI"
    description: ClassVar[str] = "Open-weight models hosted by Together AI."
    docs_url: ClassVar[str] = "https://docs.together.ai/reference/chat-completions-1"
    fields: ClassVar[tuple[ProviderField, ...]] = (_key("TOGETHER_API_KEY"),)
    capabilities: ClassVar[frozenset[str]] = _CAPS | {"embeddings"}


class DeepSeekProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "deepseek"
    env_key: ClassVar[str] = "DEEPSEEK_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.deepseek.com/v1"
    base_url_env: ClassVar[str] = ""
    key_required: ClassVar[bool] = True
    default_model: ClassVar[str] = "deepseek-chat"
    display_name: ClassVar[str] = "DeepSeek"
    description: ClassVar[str] = "DeepSeek's chat and reasoner models."
    docs_url: ClassVar[str] = "https://api-docs.deepseek.com/"
    fields: ClassVar[tuple[ProviderField, ...]] = (_key("DEEPSEEK_API_KEY", "sk-..."),)
    capabilities: ClassVar[frozenset[str]] = _CAPS | {"thinking"}


class MistralProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "mistral"
    env_key: ClassVar[str] = "MISTRAL_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.mistral.ai/v1"
    base_url_env: ClassVar[str] = ""
    key_required: ClassVar[bool] = True
    default_model: ClassVar[str] = "mistral-large-latest"
    embedding_model: ClassVar[str] = "mistral-embed"
    display_name: ClassVar[str] = "Mistral AI"
    description: ClassVar[str] = "Mistral's hosted models (La Plateforme)."
    docs_url: ClassVar[str] = "https://docs.mistral.ai/api/"
    fields: ClassVar[tuple[ProviderField, ...]] = (_key("MISTRAL_API_KEY"),)
    capabilities: ClassVar[frozenset[str]] = _CAPS | {"vision", "embeddings"}
    # Mistral names the seed differently and rejects fields it does not know.
    rename_params: ClassVar[dict[str, str]] = {"seed": "random_seed"}
    drop_params: ClassVar[frozenset[str]] = frozenset({"user", "stream_options"})
    stream_usage: ClassVar[bool] = False


class XAIProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "xai"
    env_key: ClassVar[str] = "XAI_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.x.ai/v1"
    base_url_env: ClassVar[str] = ""
    key_required: ClassVar[bool] = True
    default_model: ClassVar[str] = "grok-4"
    display_name: ClassVar[str] = "xAI"
    description: ClassVar[str] = "Grok models from xAI."
    docs_url: ClassVar[str] = "https://docs.x.ai/docs/api-reference"
    fields: ClassVar[tuple[ProviderField, ...]] = (_key("XAI_API_KEY", "xai-..."),)
    capabilities: ClassVar[frozenset[str]] = _CAPS | {"vision", "thinking"}


class FireworksProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "fireworks"
    env_key: ClassVar[str] = "FIREWORKS_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.fireworks.ai/inference/v1"
    base_url_env: ClassVar[str] = ""
    key_required: ClassVar[bool] = True
    default_model: ClassVar[str] = "accounts/fireworks/models/llama-v3p3-70b-instruct"
    display_name: ClassVar[str] = "Fireworks AI"
    description: ClassVar[str] = "Open-weight models hosted by Fireworks AI."
    docs_url: ClassVar[str] = "https://docs.fireworks.ai/api-reference"
    fields: ClassVar[tuple[ProviderField, ...]] = (_key("FIREWORKS_API_KEY", "fw_..."),)


class CerebrasProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "cerebras"
    env_key: ClassVar[str] = "CEREBRAS_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.cerebras.ai/v1"
    base_url_env: ClassVar[str] = ""
    key_required: ClassVar[bool] = True
    default_model: ClassVar[str] = "llama-3.3-70b"
    display_name: ClassVar[str] = "Cerebras"
    description: ClassVar[str] = "Open-weight models on Cerebras wafer-scale hardware."
    docs_url: ClassVar[str] = "https://inference-docs.cerebras.ai/api-reference"
    fields: ClassVar[tuple[ProviderField, ...]] = (_key("CEREBRAS_API_KEY", "csk-..."),)


class OllamaProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "ollama"
    env_key: ClassVar[str] = "OLLAMA_API_KEY"
    BASE_URL: ClassVar[str] = "http://localhost:11434/v1"
    base_url_env: ClassVar[str] = "OLLAMA_BASE_URL"
    default_model: ClassVar[str] = "llama3.2"
    display_name: ClassVar[str] = "Ollama"
    description: ClassVar[str] = "Models running locally under Ollama. No key needed."
    docs_url: ClassVar[str] = "https://github.com/ollama/ollama/blob/main/docs/openai.md"
    fields: ClassVar[tuple[ProviderField, ...]] = (
        _url("http://localhost:11434/v1", "OLLAMA_BASE_URL"),
        _key("OLLAMA_API_KEY", required=False,
             description="Only for an Ollama behind an authenticating proxy."),
    )
    capabilities: ClassVar[frozenset[str]] = _CAPS | {"vision", "embeddings"}
    embedding_model: ClassVar[str] = "nomic-embed-text"

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None,
                 **kw: Any) -> None:
        host = os.environ.get("OLLAMA_HOST", "")
        if not base_url and not os.environ.get(self.base_url_env) and host:
            host = host if host.startswith("http") else f"http://{host}"
            base_url = f"{host.rstrip('/')}/v1"
        super().__init__(api_key, base_url=base_url, **kw)


class LMStudioProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "lmstudio"
    env_key: ClassVar[str] = "LMSTUDIO_API_KEY"
    BASE_URL: ClassVar[str] = "http://localhost:1234/v1"
    base_url_env: ClassVar[str] = "LMSTUDIO_BASE_URL"
    display_name: ClassVar[str] = "LM Studio"
    description: ClassVar[str] = "Models served by LM Studio's local server. No key needed."
    docs_url: ClassVar[str] = "https://lmstudio.ai/docs/app/api/endpoints/openai"
    fields: ClassVar[tuple[ProviderField, ...]] = (
        _url("http://localhost:1234/v1", "LMSTUDIO_BASE_URL"),
    )
    capabilities: ClassVar[frozenset[str]] = _CAPS | {"embeddings"}


class VLLMProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "vllm"
    env_key: ClassVar[str] = "VLLM_API_KEY"
    BASE_URL: ClassVar[str] = ""
    base_url_env: ClassVar[str] = "VLLM_BASE_URL"
    display_name: ClassVar[str] = "vLLM"
    description: ClassVar[str] = "A self-hosted vLLM server's OpenAI-compatible API."
    docs_url: ClassVar[str] = "https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html"
    fields: ClassVar[tuple[ProviderField, ...]] = (
        _url("", "VLLM_BASE_URL", required=True),
        _key("VLLM_API_KEY", required=False,
             description="Only if the server was started with --api-key."),
    )
    capabilities: ClassVar[frozenset[str]] = _CAPS | {"thinking", "embeddings"}


COMPATIBLE_PROVIDERS: tuple[type[OpenAICompatibleProvider], ...] = (
    OpenAICompatibleProvider, OpenRouterProvider, GroqProvider, TogetherProvider,
    DeepSeekProvider, MistralProvider, XAIProvider, FireworksProvider, CerebrasProvider,
    OllamaProvider, LMStudioProvider, VLLMProvider,
)
