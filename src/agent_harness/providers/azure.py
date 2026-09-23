"""Azure OpenAI and Azure AI Foundry.

Azure keeps the OpenAI wire format and changes the envelope: the model is a
*deployment* in the URL, the API version is a query parameter, and the key goes
in an `api-key` header rather than a bearer token.

    AzureOpenAIProvider(endpoint="https://my-resource.openai.azure.com",
                        deployment="gpt-4.1", api_version="2024-10-21")

    AzureFoundryProvider(endpoint="https://my-project.services.ai.azure.com",
                         deployment="claude-opus-5")

Both take `credential=` for Entra ID instead of a key: anything with a
`get_token` method, which is what `azure-identity` hands you.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, ClassVar

from ..errors import ProviderError
from .base import CompletionRequest
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
        endpoint = (endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
        if not endpoint:
            raise ProviderError(
                "Azure needs an endpoint — pass endpoint=... or set "
                "AZURE_OPENAI_ENDPOINT", provider=self.name)
        super().__init__(api_key=api_key, base_url=endpoint, **kw)
        self.endpoint = endpoint
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        self.api_version = api_version
        self.credential = credential
        self._token = ""
        self._token_expires = 0.0
        self._lock = asyncio.Lock()

    # ---- auth ---------------------------------------------------------------
    def _auth_headers(self) -> dict[str, str]:
        if self.credential is not None:
            return {}                  # filled in per request, asynchronously
        if not self.api_key:
            raise ProviderError(
                f"No API key for {self.name}. Set ${self.env_key}, pass "
                "api_key=..., or give credential= for Entra ID",
                provider=self.name)
        return {"api-key": self.api_key}

    async def _bearer(self) -> str:
        """An Entra ID token, cached until shortly before it expires."""
        if self._token and time.time() < self._token_expires - 60:
            return self._token
        async with self._lock:
            if self._token and time.time() < self._token_expires - 60:
                return self._token
            token = await asyncio.to_thread(self.credential.get_token, SCOPE)
            self._token = token.token
            self._token_expires = float(getattr(token, "expires_on",
                                                time.time() + 3000))
            return self._token

    # ---- the request ---------------------------------------------------------
    def _path(self, req: CompletionRequest) -> str:
        deployment = self.deployment or req.model
        if not deployment:
            raise ProviderError(
                "Azure needs a deployment name — pass deployment=... or use the "
                "deployment as the model id", provider=self.name)
        return self.path_template.format(deployment=deployment,
                                         api_version=self.api_version)

    def _payload(self, req: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
        payload = super()._payload(req, stream=stream)
        # The deployment is the model on Azure, and it is already in the URL.
        payload.pop("model", None)
        return payload

    async def _post(self, path: str, payload: dict[str, Any], *,
                    params: dict[str, str] | None = None) -> dict[str, Any]:
        if self.credential is not None:
            self.extra_headers = {**self.extra_headers,
                                  "authorization": f"Bearer {await self._bearer()}"}
        return await super()._post(path, payload, params=params)

    async def complete(self, req: CompletionRequest) -> Any:
        started = time.perf_counter()
        raw = await self._post(self._path(req), self._payload(req))
        response = self._decode(raw, req)
        response.latency_ms = (time.perf_counter() - started) * 1000
        return response

    def _decode(self, raw: dict[str, Any], req: CompletionRequest) -> Any:
        import json

        from ..types import Message, ToolUseBlock
        from .openai import _STOP_MAP

        choices = raw.get("choices") or []
        if not choices:
            raise ProviderError("azure returned no choices", provider=self.name)
        choice = choices[0]
        message = choice.get("message") or {}
        blocks: list[Any] = []
        if message.get("content"):
            from ..types import TextBlock

            blocks.append(TextBlock(text=message["content"]))
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            blocks.append(ToolUseBlock(id=call.get("id", ""), name=fn.get("name", ""),
                                       input=args))
        return self._finish(
            message=Message(role="assistant", content=blocks),
            stop_reason=_STOP_MAP.get(choice.get("finish_reason") or "stop",
                                      "end_turn"),
            usage=self._usage(raw), model=raw.get("model") or req.model, raw=raw)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        deployment = model or self.embedding_model
        path = (f"/openai/deployments/{deployment}/embeddings"
                f"?api-version={self.api_version}")
        raw = await self._post(path, {"input": texts})
        rows = sorted(raw.get("data") or [], key=lambda d: d.get("index", 0))
        return [row["embedding"] for row in rows]


class AzureFoundryProvider(AzureOpenAIProvider):
    """Azure AI Foundry, which serves many model families on one endpoint.

    The route differs from Azure OpenAI's, so it is a separate class rather
    than a flag — and `path_template` is overridable, because Foundry's routes
    have moved before and will again.
    """

    name: ClassVar[str] = "azure-foundry"
    env_key: ClassVar[str] = "AZURE_AI_API_KEY"
    path_template: ClassVar[str] = (
        "/models/chat/completions?api-version={api_version}")

    def __init__(self, *, endpoint: str = "", deployment: str = "",
                 api_version: str = "2024-05-01-preview",
                 path_template: str | None = None, **kw: Any) -> None:
        endpoint = endpoint or os.environ.get("AZURE_AI_ENDPOINT", "")
        if not endpoint:
            raise ProviderError(
                "Azure AI Foundry needs an endpoint — pass endpoint=... or set "
                "AZURE_AI_ENDPOINT", provider=self.name)
        super().__init__(endpoint=endpoint, deployment=deployment,
                         api_version=api_version, **kw)
        if path_template is not None:
            self.path_template = path_template

    def _payload(self, req: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
        # Foundry routes by model in the body, unlike Azure OpenAI's deployments.
        payload = OpenAIProvider._payload(self, req, stream=stream)
        payload["model"] = self.deployment or req.model
        return payload

    def _path(self, req: CompletionRequest) -> str:
        return self.path_template.format(deployment=self.deployment or req.model,
                                         api_version=self.api_version)
