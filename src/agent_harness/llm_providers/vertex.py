"""Claude and Gemini on Google Cloud Vertex AI.

Vertex serves both families, each in its own publisher path, and both in the
same wire format they use directly — so the encoding is shared with the direct
providers and only the URL and the auth change.

    VertexProvider(project="my-project", region="europe-west1")        # Claude
    VertexGeminiProvider(project="my-project", region="us-central1")   # Gemini

Auth is a Google access token: `google-auth` if it is installed (which covers
application default credentials, service accounts and workload identity), else
`gcloud auth print-access-token`, else one you pass in. A 401 drops the cached
token and fetches a new one, once. Completion and streaming both go through the
shared retrying transport.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import Any, ClassVar

from ..errors import ProviderError
from .anthropic import AnthropicProvider
from .base import CompletionRequest, ProviderField
from .gemini import GeminiProvider

__all__ = ["VertexProvider", "VertexGeminiProvider", "GoogleAuth"]

VERTEX_VERSION = "vertex-2023-10-16"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class GoogleAuth:
    """A Google access token, from whichever source is available.

    Tokens are cached until shortly before they expire, so a long run does not
    re-authenticate on every call.
    """

    def __init__(self, token: str | None = None, *, credentials: Any = None) -> None:
        self._static = token or os.environ.get("GOOGLE_ACCESS_TOKEN")
        self._credentials = credentials
        self._token = ""
        self._expires = 0.0
        self._lock: asyncio.Lock | None = None
        self._lock_loop: Any = None

    @property
    def refreshable(self) -> bool:
        return not self._static

    def invalidate(self) -> None:
        """Forget the cached token, so the next call fetches a fresh one."""
        self._token, self._expires = "", 0.0

    async def token(self) -> str:
        if self._static:
            return self._static
        if self._token and time.time() < self._expires - 60:
            return self._token
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock, self._lock_loop = asyncio.Lock(), loop
        async with self._lock:
            if self._token and time.time() < self._expires - 60:
                return self._token
            self._token, self._expires = await self._fetch()
            return self._token

    async def _fetch(self) -> tuple[str, float]:
        credentials = self._credentials
        if credentials is None:
            try:
                import google.auth

                credentials, _ = await asyncio.to_thread(google.auth.default,
                                                         scopes=[SCOPE])
            except ImportError:
                credentials = None
            except Exception as exc:
                raise ProviderError(f"google auth failed: {exc}",
                                    provider="vertex") from exc
        if credentials is not None:
            from google.auth.transport.requests import Request

            await asyncio.to_thread(credentials.refresh, Request())
            expiry = getattr(credentials, "expiry", None)
            return credentials.token, (expiry.timestamp() if expiry
                                       else time.time() + 3000)

        # No library: borrow the token the CLI already has.
        try:
            result = await asyncio.to_thread(
                subprocess.run, ["gcloud", "auth", "print-access-token"],
                capture_output=True, text=True, timeout=30)
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            raise ProviderError(
                "Vertex needs a Google access token — `pip install google-auth`, "
                "run `gcloud auth application-default login`, or pass "
                "access_token=...", provider="vertex") from exc
        if result.returncode != 0:
            raise ProviderError(
                f"gcloud could not produce a token: {result.stderr.strip()[:300]}",
                provider="vertex")
        return result.stdout.strip(), time.time() + 3000


_VERTEX_FIELDS: tuple[ProviderField, ...] = (
    ProviderField(name="project", required=True, env=("GOOGLE_CLOUD_PROJECT",),
                  example="my-project", description="The Google Cloud project id."),
    ProviderField(name="region", env=("GOOGLE_CLOUD_REGION", "CLOUD_ML_REGION"),
                  default="us-central1", example="europe-west1",
                  description="The Vertex region, or 'global'."),
    ProviderField(name="access_token", type="secret", one_of="google",
                  env=("GOOGLE_ACCESS_TOKEN",),
                  description="A ready-made OAuth token. Otherwise google-auth "
                              "(ADC, service accounts) or gcloud is used."),
    ProviderField(name="credentials", type="object", one_of="google",
                  description="A google.auth credentials object."),
)


class _VertexMixin:
    """The URL shape and auth both Vertex providers share."""

    publisher: ClassVar[str] = ""
    action: ClassVar[str] = "rawPredict"
    stream_action: ClassVar[str] = "streamRawPredict"
    auth_type: ClassVar[str] = "google-oauth"
    fields: ClassVar[tuple[ProviderField, ...]] = _VERTEX_FIELDS

    def _setup(self, project: str, region: str, access_token: str | None,
               credentials: Any) -> None:
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.region = (region or os.environ.get("GOOGLE_CLOUD_REGION")
                       or os.environ.get("CLOUD_ML_REGION") or "us-central1")
        if not self.project:
            raise ProviderError(
                "Vertex needs a project — pass project=... or set "
                "GOOGLE_CLOUD_PROJECT", provider="vertex")
        self.auth = GoogleAuth(access_token, credentials=credentials)

    @property
    def endpoint(self) -> str:
        host = ("aiplatform.googleapis.com" if self.region == "global"
                else f"{self.region}-aiplatform.googleapis.com")
        return f"https://{host}/v1/projects/{self.project}/locations/{self.region}"

    def _model_url(self, model: str, action: str | None = None) -> str:
        name = model.split("/")[-1]
        return (f"{self.endpoint}/publishers/{self.publisher}/models/"
                f"{name}:{action or self.action}")

    def _auth_headers(self) -> dict[str, str]:
        return {}

    async def _prepare_headers(self, method: str, url: str, body: bytes,
                               headers: dict[str, str]) -> dict[str, str]:
        return {**headers, "authorization": f"Bearer {await self.auth.token()}"}

    async def _refresh_auth(self) -> bool:
        if not self.auth.refreshable:
            return False
        self.auth.invalidate()
        return True


class VertexProvider(_VertexMixin, AnthropicProvider):
    """Claude on Vertex AI."""

    name: ClassVar[str] = "vertex"
    env_key: ClassVar[str] = ""
    default_model: ClassVar[str] = "claude-opus-5"
    publisher: ClassVar[str] = "anthropic"
    BASE_URL: ClassVar[str] = "https://aiplatform.googleapis.com"

    display_name: ClassVar[str] = "Google Vertex AI (Claude)"
    description: ClassVar[str] = "Claude served from your Google Cloud project on Vertex AI."
    docs_url: ClassVar[str] = "https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude"
    capabilities: ClassVar[frozenset[str]] = frozenset({
        "streaming", "tools", "vision", "thinking", "json_schema", "prompt_caching"})

    def __init__(self, *, project: str = "", region: str = "",
                 access_token: str | None = None, credentials: Any = None,
                 **kw: Any) -> None:
        super().__init__(api_key="vertex", **kw)
        self._setup(project, region, access_token, credentials)

    def _payload(self, req: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
        payload = super()._payload(req, stream=stream)
        # The model is in the URL, and the API version moves into the body.
        payload.pop("model", None)
        payload["anthropic_version"] = VERTEX_VERSION
        if not stream:
            payload.pop("stream", None)
        return payload

    def _messages_path(self, req: CompletionRequest, stream: bool) -> str:
        return self._model_url(req.model, self.stream_action if stream else self.action)

    async def list_models(self) -> list[str]:
        raise NotImplementedError("Vertex lists partner models in Model Garden")


class VertexGeminiProvider(_VertexMixin, GeminiProvider):
    """Gemini on Vertex AI."""

    name: ClassVar[str] = "vertex-gemini"
    env_key: ClassVar[str] = ""
    default_model: ClassVar[str] = "gemini-2.5-pro"
    publisher: ClassVar[str] = "google"
    action: ClassVar[str] = "generateContent"
    stream_action: ClassVar[str] = "streamGenerateContent"
    BASE_URL: ClassVar[str] = "https://aiplatform.googleapis.com"

    display_name: ClassVar[str] = "Google Vertex AI (Gemini)"
    description: ClassVar[str] = "Gemini served from your Google Cloud project on Vertex AI."
    docs_url: ClassVar[str] = "https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference"
    capabilities: ClassVar[frozenset[str]] = frozenset({
        "streaming", "tools", "vision", "thinking", "json_schema", "embeddings"})

    def __init__(self, *, project: str = "", region: str = "",
                 access_token: str | None = None, credentials: Any = None,
                 **kw: Any) -> None:
        super().__init__(api_key="vertex", **kw)
        self._setup(project, region, access_token, credentials)

    def _model_path(self, model: str, stream: bool) -> str:
        return self._model_url(model, self.stream_action if stream else self.action)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        url = self._model_url(model or self.embedding_model, "predict")
        raw = await self._post(url, {"instances": [{"content": t} for t in texts]})
        return [(p.get("embeddings") or {}).get("values", [])
                for p in raw.get("predictions") or []]

    async def list_models(self) -> list[str]:
        raise NotImplementedError("Vertex lists models in Model Garden")
