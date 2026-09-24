"""The provider registry, and what it tells you before you connect.

    for spec in list_llm_providers():
        print(spec.name, [f.name for f in spec.required_fields], spec.configured)

    print(describe_llm_provider("azure").render())     # every field, and where it's read
    check_llm_provider("bedrock", region="eu-west-1")   # what's missing, offline
    await ping_llm_provider("anthropic")                # does the key actually work?

Every backend declares its connection fields — which are required, which are
secret, which environment variables they fall back to — so a newcomer can see
exactly what a provider needs without reading its source.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from typing import Any

from pydantic import BaseModel, Field

from ..errors import ConfigurationError
from .anthropic import AnthropicProvider
from .azure import AzureFoundryProvider, AzureOpenAIProvider
from .base import COMMON_FIELDS, MODELS, Provider, ProviderField, provider_for_model
from .bedrock import BedrockProvider
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
from .fake import FakeProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .parameters import SAMPLING_PARAMETERS
from .vertex import VertexGeminiProvider, VertexProvider

__all__ = [
    "PROVIDERS",
    "ProviderSpec",
    "ProviderCheck",
    "register_provider",
    "get_provider",
    "resolve_provider",
    "close_all",
    "list_llm_providers",
    "describe_llm_provider",
    "check_llm_provider",
    "ping_llm_provider",
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
    # Everything that speaks the OpenAI format.
    "openai-compatible": OpenAICompatibleProvider,
    "openrouter": OpenRouterProvider,
    "groq": GroqProvider,
    "together": TogetherProvider,
    "deepseek": DeepSeekProvider,
    "mistral": MistralProvider,
    "xai": XAIProvider,
    "grok": XAIProvider,
    "fireworks": FireworksProvider,
    "cerebras": CerebrasProvider,
    "ollama": OllamaProvider,
    "lmstudio": LMStudioProvider,
    "vllm": VLLMProvider,
}

_CACHE: dict[tuple[str, tuple], Provider] = {}


def register_provider(name: str, cls: type[Provider]) -> None:
    """Add your own backend: `register_provider("bedrock", MyProvider)`.

    Declare `fields`, `description` and `capabilities` on the class and it shows
    up in `list_llm_providers()` like any built-in.
    """
    PROVIDERS[name.lower()] = cls


def _lookup(name: str) -> tuple[str, type[Provider]]:
    key = name.lower().strip()
    cls = PROVIDERS.get(key)
    if cls is None:
        raise ConfigurationError(
            f"Unknown provider {name!r}. Known: {', '.join(sorted(set(PROVIDERS)))}. "
            "See list_llm_providers()."
        )
    return key, cls


def get_provider(name: str, *, cached: bool = True, **kwargs: Any) -> Provider:
    """Build (or reuse) a provider by vendor name."""
    key, cls = _lookup(name)
    if not cached or kwargs:
        return _build(key, cls, kwargs)
    cache_key = (key, ())
    if cache_key not in _CACHE:
        _CACHE[cache_key] = _build(key, cls, {})
    return _CACHE[cache_key]


def _build(key: str, cls: type[Provider], kwargs: dict[str, Any]) -> Provider:
    try:
        return cls(**kwargs)
    except TypeError as exc:
        accepted = ", ".join(f.name for f in (*cls.fields, *COMMON_FIELDS))
        raise ConfigurationError(
            f"{key}: {exc}. It accepts: {accepted or 'no settings'}. "
            f"See describe_llm_provider({key!r})."
        ) from exc


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


# --- describing what a provider needs ------------------------------------------------

class ProviderCheck(BaseModel):
    """Whether a provider has what it needs to connect — decided offline."""

    provider: str
    ok: bool
    #: What is still needed, each with where it could come from.
    missing: list[str] = Field(default_factory=list)
    #: field → where it was found: "argument" or "$ENV_VAR". Values are never shown.
    found: dict[str, str] = Field(default_factory=dict)
    unknown: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    def render(self) -> str:
        lines = [f"{self.provider}: {'ready' if self.ok else 'not ready'}"]
        lines += [f"  missing  {m}" for m in self.missing]
        lines += [f"  found    {k} ← {v}" for k, v in self.found.items()]
        lines += [f"  unknown  {u}" for u in self.unknown]
        lines += [f"  note     {n}" for n in self.notes]
        return "\n".join(lines)


class ProviderSpec(BaseModel):
    """Everything there is to know about a backend before connecting to it."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    display_name: str
    description: str = ""
    docs_url: str = ""
    auth_type: str = ""
    class_name: str
    default_model: str = ""
    capabilities: list[str] = Field(default_factory=list)
    #: The sampling parameters it accepts; the rest are dropped with a note.
    #: `effort`, `thinking`, `thinking_budget`, `max_tokens` and `stop` are
    #: understood by every provider and mapped to what each model takes.
    parameters: list[str] = Field(default_factory=list)
    fields: list[ProviderField] = Field(default_factory=list)
    common_fields: list[ProviderField] = Field(default_factory=list)
    #: Models in the price table served by this provider.
    models: list[str] = Field(default_factory=list)
    #: True when everything required is already in the environment.
    configured: bool = False
    missing: list[str] = Field(default_factory=list)

    @property
    def required_fields(self) -> list[ProviderField]:
        return [f for f in self.fields if f.required or f.one_of]

    @property
    def optional_fields(self) -> list[ProviderField]:
        return [f for f in self.fields if not (f.required or f.one_of)]

    def example(self) -> str:
        """A line of code that builds this provider, with placeholders."""
        args = []
        for f in self.fields:
            if f.type in ("secret", "object") or f.one_of:
                continue                    # keys come from the environment, not code
            if f.required or f.example:
                args.append(f"{f.name}={(f.example or '...')!r}")
        return f"get_provider({self.name!r}{''.join(', ' + a for a in args)})"

    def render(self) -> str:
        lines = [f"{self.display_name} — provider={self.name!r}"
                 + (f" (aliases: {', '.join(self.aliases)})" if self.aliases else "")]
        if self.description:
            lines.append(f"  {self.description}")
        lines.append(f"  status   {'ready' if self.configured else 'needs setup'}"
                     + (f" — missing {', '.join(self.missing)}" if self.missing else ""))
        lines.append(f"  auth     {self.auth_type}")
        if self.default_model:
            lines.append(f"  default  {self.default_model}")
        if self.capabilities:
            lines.append(f"  can      {', '.join(self.capabilities)}")
        if self.parameters:
            lines.append(f"  params   {', '.join(self.parameters)}, plus effort, thinking, "
                         "max_tokens, stop")
        if self.fields:
            lines.append("  fields")
            for f in self.fields:
                need = ("required" if f.required else f"one of [{f.one_of}]" if f.one_of
                        else "optional")
                env = f" ← ${' / $'.join(f.env)}" if f.env else ""
                default = f" (default {f.default!r})" if f.default not in (None, "") else ""
                lines.append(f"    {f.name:<16} {need:<18} {f.type:<7}{env}{default}".rstrip())
                if f.description:
                    lines.append(f"    {'':<16} {f.description}")
        lines.append("  also     " + ", ".join(f.name for f in self.common_fields))
        lines.append(f"  example  {self.example()}")
        if self.docs_url:
            lines.append(f"  docs     {self.docs_url}")
        return "\n".join(lines)


def _is_set(value: Any) -> bool:
    return value is not None and value != "" and value is not False


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):       # the parent package is missing
        return False


def _ambient_auth(cls: type[Provider]) -> str | None:
    """A credential source that needs no argument: a credential chain, a CLI login."""
    if issubclass(cls, BedrockProvider):
        if _installed("botocore"):
            return "no explicit AWS credentials — botocore's chain (profiles, SSO, " \
                   "instance roles) will be used"
    if cls.auth_type == "google-oauth":
        if _installed("google.auth"):
            return "no access_token — google-auth application default credentials " \
                   "will be used"
        if shutil.which("gcloud"):
            return "no access_token — `gcloud auth print-access-token` will be used"
    return None


def _check(key: str, cls: type[Provider], kwargs: dict[str, Any]) -> ProviderCheck:
    found: dict[str, str] = {}
    for f in cls.fields:
        if _is_set(kwargs.get(f.name)):
            found[f.name] = "argument"
            continue
        for env in f.env:
            if os.environ.get(env):
                found[f.name] = f"${env}"
                break

    missing: list[str] = []
    notes: list[str] = []
    for f in cls.fields:
        if f.required and f.name not in found:
            where = f" (or set ${' / $'.join(f.env)})" if f.env else ""
            missing.append(f"{f.name}{where}")
    groups: dict[str, list[ProviderField]] = {}
    for f in cls.fields:
        if f.one_of:
            groups.setdefault(f.one_of, []).append(f)
    for members in groups.values():
        if any(m.name in found for m in members):
            continue
        ambient = _ambient_auth(cls)
        if ambient:
            notes.append(ambient)
            continue
        options = " | ".join(m.name + (f" (${m.env[0]})" if m.env else "") for m in members)
        missing.append(f"one of: {options}")

    # A half-given AWS key pair is a mistake worth naming.
    if "access_key" in found and "secret_key" not in found:
        missing.append("secret_key (or set $AWS_SECRET_ACCESS_KEY)")

    accepted = {f.name for f in (*cls.fields, *COMMON_FIELDS)}
    unknown = sorted(k for k in kwargs if k not in accepted)
    if unknown:
        notes.append(f"not a {key} setting: {', '.join(unknown)} — see "
                     f"describe_llm_provider({key!r})")
    return ProviderCheck(provider=key, ok=not missing and not unknown, missing=missing,
                         found=found, unknown=unknown, notes=notes)


def _spec(key: str, cls: type[Provider]) -> ProviderSpec:
    aliases = sorted(k for k, v in PROVIDERS.items() if v is cls and k != key)
    check = _check(key, cls, {})
    doc = (cls.__doc__ or "").strip().splitlines()
    return ProviderSpec(
        name=key,
        aliases=aliases,
        display_name=cls.display_name or cls.__name__,
        description=cls.description or (doc[0] if doc else ""),
        docs_url=cls.docs_url,
        auth_type=(cls.auth_type if getattr(cls, "key_required", True)
                   else "none (a key is optional)"),
        class_name=cls.__name__,
        default_model=cls.default_model,
        capabilities=sorted(cls.capabilities),
        parameters=[p for p in SAMPLING_PARAMETERS if p in cls.parameters],
        fields=list(cls.fields),
        common_fields=list(COMMON_FIELDS),
        models=sorted(m.id for m in MODELS.values() if m.provider == cls.name),
        configured=check.ok,
        missing=check.missing,
    )


def _canonical() -> list[tuple[str, type[Provider]]]:
    """One entry per class, under the name the class calls itself where registered."""
    seen: dict[type[Provider], str] = {}
    for key, cls in PROVIDERS.items():
        if cls not in seen or key == getattr(cls, "name", None):
            seen[cls] = key
    return [(key, cls) for cls, key in seen.items()]


def list_llm_providers(*, configured_only: bool = False, capability: str | None = None,
                       include_testing: bool = False) -> list[ProviderSpec]:
    """Every provider this harness can talk to, with the fields each one needs.

    ``configured_only``  just the ones whose required settings are already present
    ``capability``       just the ones that can, e.g. ``"embeddings"`` or ``"streaming"``
    ``include_testing``  include the scripted `fake` provider
    """
    specs = []
    for key, cls in _canonical():
        if cls is FakeProvider and not include_testing:
            continue
        spec = _spec(key, cls)
        if configured_only and not spec.configured:
            continue
        if capability and capability not in spec.capabilities:
            continue
        specs.append(spec)
    return specs


def describe_llm_provider(name: str) -> ProviderSpec:
    """One provider's full spec: fields, env vars, capabilities, an example."""
    key, cls = _lookup(name)
    canonical = next((k for k, c in _canonical() if c is cls), key)
    return _spec(canonical, cls)


def check_llm_provider(name: str, **kwargs: Any) -> ProviderCheck:
    """Would `get_provider(name, **kwargs)` have what it needs? Checked offline.

    Arguments and environment variables both count; secret values are never
    echoed back, only where they were found.
    """
    key, cls = _lookup(name)
    return _check(key, cls, kwargs)


async def ping_llm_provider(name: str, *, model: str | None = None,
                            **kwargs: Any) -> dict[str, Any]:
    """Connect for real and report whether the credentials work. Never raises."""
    check = check_llm_provider(name, **kwargs)
    if not check.ok:
        return {"provider": name, "ok": False,
                "error": "not configured: " + "; ".join(check.missing + check.unknown)}
    try:
        provider = get_provider(name, cached=False, **kwargs)
    except Exception as exc:
        return {"provider": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        return await provider.ping(model)
    finally:
        await provider.aclose()
