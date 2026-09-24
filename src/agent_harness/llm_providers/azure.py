"""Azure OpenAI and Azure AI Foundry.

Azure keeps the OpenAI wire format and changes the envelope: the model is a
*deployment* in the URL, the API version is a query parameter, and the key goes
in an `api-key` header rather than a bearer token.

    AzureOpenAIProvider(endpoint="https://my-resource.openai.azure.com",
                        deployment="gpt-4.1", api_version="2024-10-21")

    AzureFoundryProvider(endpoint="https://my-project.services.ai.azure.com",
                         deployment="claude-opus-5")

Both take `credential=` for Entra ID instead of a key: anything with a
`get_token` method, sync or async, which is what `azure-identity` hands you.
Completion, streaming and embeddings all go through the same retrying transport.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from typing import Any, ClassVar

from ..errors import ProviderError
from .base import CompletionRequest, ProviderField
from .openai import OpenAIProvider

__all__ = ["AzureOpenAIProvider", "AzureFoundryProvider"]

SCOPE = "https://cognitiveservices.azure.com/.default"


class AzureOpenAIProvider(OpenAIProvider):
    """An Azure OpenAI deployment."""

    name: ClassVar[str] = "azure"
    env_key: ClassVar[str] = "AZURE_OPENAI_API_KEY"
    default_model: ClassVar[str] = ""
    BASE_URL: ClassVar[str] = ""
    #: `{deployment}` and `{api_version}` are filled in per request.
    path_template: ClassVar[str] = (
        "/openai/deployments/{deployment}/chat/completions?api-version={api_version}")
    endpoint_env: ClassVar[str] = "AZURE_OPENAI_ENDPOINT"

    display_name: ClassVar[str] = "Azure OpenAI"
    description: ClassVar[str] = "OpenAI models deployed in your own Azure OpenAI resource."
    docs_url: ClassVar[str] = "https://learn.microsoft.com/azure/ai-services/openai/reference"
    auth_type: ClassVar[str] = "api_key or Entra ID"
    fields: ClassVar[tuple[ProviderField, ...]] = (
        ProviderField(name="endpoint", type="url", required=True,
                      env=("AZURE_OPENAI_ENDPOINT",),
                      example="https://my-resource.openai.azure.com",
                      description="The resource endpoint from the Azure portal."),
        ProviderField(name="deployment", env=("AZURE_OPENAI_DEPLOYMENT",),
                      example="gpt-4.1-prod",
                      description="The deployment name. Defaults to the model id."),
        ProviderField(name="api_version", default="2024-10-21",
                      description="The Azure OpenAI REST API version."),
        ProviderField(name="api_key", type="secret", one_of="auth",
                      env=("AZURE_OPENAI_API_KEY",),
                      description="The resource key (Keys and Endpoint blade)."),
        ProviderField(name="credential", type="object", one_of="auth",
                      description="An azure-identity credential, for Entra ID instead "
                                  "of a key."),
    )
    capabilities: ClassVar[frozenset[str]] = frozenset({
        "streaming", "tools", "vision", "thinking", "json_schema", "embeddings"})

    def __init__(
        self,
        *,
        endpoint: str = "",
        deployment: str = "",
        api_version: str = "2024-10-21",
        api_key: str | None = None,
        credential: Any = None,
        **kw: Any,
    ) -> None:
        endpoint = (endpoint or os.environ.get(self.endpoint_env, "")).rstrip("/")
        if not endpoint:
            raise ProviderError(
                f"{self.display_name} needs an endpoint — pass endpoint=... or set "
                f"{self.endpoint_env}", provider=self.name)
        super().__init__(api_key=api_key, base_url=endpoint, **kw)
        self.endpoint = endpoint
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        self.api_version = api_version
        self.credential = credential
        self._token = ""
        self._token_expires = 0.0
        self._lock: asyncio.Lock | None = None
        self._lock_loop: Any = None

    # ---- auth ---------------------------------------------------------------
    def _auth_headers(self) -> dict[str, str]:
        if self.credential is not None:
            return {}                  # a bearer token is added per attempt
        if not self.api_key:
            raise ProviderError(
                f"No API key for {self.name}. Set ${self.env_key}, pass "
                "api_key=..., or give credential= for Entra ID",
                provider=self.name)
        return {"api-key": self.api_key}

    async def _prepare_headers(self, method: str, url: str, body: bytes,
                               headers: dict[str, str]) -> dict[str, str]:
        out = {**headers, **self._auth_headers()}
        if self.credential is not None:
            out["authorization"] = f"Bearer {await self._bearer()}"
        return out

    async def _refresh_auth(self) -> bool:
        if self.credential is None:
            return False
        self._token, self._token_expires = "", 0.0
        return True

    def _token_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock, self._lock_loop = asyncio.Lock(), loop
        return self._lock

    async def _bearer(self) -> str:
        """An Entra ID token, cached until shortly before it expires."""
        if self._token and time.time() < self._token_expires - 60:
            return self._token
        async with self._token_lock():
            if self._token and time.time() < self._token_expires - 60:
                return self._token
            getter = self.credential.get_token
            if inspect.iscoroutinefunction(getter):
                token = await getter(SCOPE)
            else:
                token = await asyncio.to_thread(getter, SCOPE)
            self._token = token.token
            self._token_expires = float(getattr(token, "expires_on",
                                                time.time() + 3000))
            return self._token

    # ---- the request ---------------------------------------------------------
    def _deployment(self, model: str) -> str:
        deployment = self.deployment or model
        if not deployment:
            raise ProviderError(
                "Azure needs a deployment name — pass deployment=... or use the "
                "deployment as the model id", provider=self.name)
        return deployment

    def _chat_path(self, req: CompletionRequest) -> str:
        return self.path_template.format(deployment=self._deployment(req.model),
                                         api_version=self.api_version)

    # Kept for callers of the 0.1 API.
    _path = _chat_path

    def _payload(self, req: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
        payload = super()._payload(req, stream=stream)
        # The deployment is the model on Azure, and it is already in the URL.
        payload.pop("model", None)
        return payload

    def _embeddings_path(self, model: str) -> str:
        return (f"/openai/deployments/{model}/embeddings"
                f"?api-version={self.api_version}")

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        deployment = model or self.embedding_model
        raw = await self._post(self._embeddings_path(deployment), {"input": texts})
        rows = sorted(raw.get("data") or [], key=lambda d: d.get("index", 0))
        return [row["embedding"] for row in rows]

    async def list_models(self) -> list[str]:
        raise NotImplementedError("Azure deployments are listed in the portal")


class AzureFoundryProvider(AzureOpenAIProvider):
    """Azure AI Foundry, which serves many model families on one endpoint.

    The route differs from Azure OpenAI's, so it is a separate class rather
    than a flag — and `path_template` is overridable, because Foundry's routes
    have moved before and will again.
    """

    name: ClassVar[str] = "azure-foundry"
    env_key: ClassVar[str] = "AZURE_AI_API_KEY"
    endpoint_env: ClassVar[str] = "AZURE_AI_ENDPOINT"
    path_template: ClassVar[str] = (
        "/models/chat/completions?api-version={api_version}")

    display_name: ClassVar[str] = "Azure AI Foundry"
    description: ClassVar[str] = ("Many model families (Claude, Llama, Mistral, GPT, ...) "
                                  "served from one Azure AI Foundry endpoint.")
    docs_url: ClassVar[str] = "https://learn.microsoft.com/azure/ai-foundry/"
    fields: ClassVar[tuple[ProviderField, ...]] = (
        ProviderField(name="endpoint", type="url", required=True,
                      env=("AZURE_AI_ENDPOINT",),
                      example="https://my-project.services.ai.azure.com",
                      description="The Foundry project endpoint."),
        ProviderField(name="deployment", example="claude-opus-5",
                      description="The model deployment. Defaults to the model id."),
        ProviderField(name="api_version", default="2024-05-01-preview",
                      description="The Foundry inference API version."),
        ProviderField(name="path_template",
                      description="Override the route if Foundry moves it again."),
        ProviderField(name="api_key", type="secret", one_of="auth",
                      env=("AZURE_AI_API_KEY",), description="The project key."),
        ProviderField(name="credential", type="object", one_of="auth",
                      description="An azure-identity credential, for Entra ID."),
    )

    def __init__(self, *, endpoint: str = "", deployment: str = "",
                 api_version: str = "2024-05-01-preview",
                 path_template: str | None = None, **kw: Any) -> None:
        super().__init__(endpoint=endpoint, deployment=deployment,
                         api_version=api_version, **kw)
        if path_template is not None:
            self.path_template = path_template

    def _payload(self, req: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
        # Foundry routes by model in the body, unlike Azure OpenAI's deployments.
        payload = OpenAIProvider._payload(self, req, stream=stream)
        payload["model"] = self.deployment or req.model
        return payload

    def _chat_path(self, req: CompletionRequest) -> str:
        return self.path_template.format(deployment=self.deployment or req.model,
                                         api_version=self.api_version)

    _path = _chat_path

    def _embeddings_path(self, model: str) -> str:
        return f"/models/embeddings?api-version={self.api_version}"

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        name = model or self.embedding_model
        raw = await self._post(self._embeddings_path(name), {"model": name, "input": texts})
        rows = sorted(raw.get("data") or [], key=lambda d: d.get("index", 0))
        return [row["embedding"] for row in rows]
