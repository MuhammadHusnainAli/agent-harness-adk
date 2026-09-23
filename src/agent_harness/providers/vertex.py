"""Claude and Gemini on Google Cloud Vertex AI.

Vertex serves both families, each in its own publisher path, and both in the
same wire format they use directly — so the encoding is shared with the direct
providers and only the URL and the auth change.

    VertexProvider(project="my-project", region="europe-west1")        # Claude
    VertexGeminiProvider(project="my-project", region="us-central1")   # Gemini

Auth is a Google access token: `google-auth` if it is installed (which covers
application default credentials, service accounts and workload identity), else
`gcloud auth print-access-token`, else one you pass in.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import Any, ClassVar

from ..errors import ProviderError
from ..types import ModelResponse
from .anthropic import AnthropicProvider
from .base import CompletionRequest
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
        self._lock = asyncio.Lock()

    async def token(self) -> str:
        if self._static:
            return self._static
        if self._token and time.time() < self._expires - 60:
            return self._token
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


class _VertexMixin:
    """The URL shape and auth both Vertex providers share."""

    publisher: ClassVar[str] = ""
    action: ClassVar[str] = "rawPredict"

    def _setup(self, project: str, region: str, access_token: str | None,
               credentials: Any) -> None:
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        self.region = region or os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
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
        return (f"{self.endpoint}/publishers/{self.publisher}/models/"
                f"{model}:{action or self.action}")

    async def _vertex_post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        token = await self.auth.token()
        headers = {"content-type": "application/json",
                   "authorization": f"Bearer {token}", **self.extra_headers}
        resp = await self.http.post(url, json=payload, headers=headers)
        if resp.status_code >= 300:
            raise ProviderError(f"vertex returned {resp.status_code}: "
                                f"{resp.text[:1000]}", provider=self.name,
                                status=resp.status_code, body=resp.text[:2000])
        return resp.json()


class VertexProvider(_VertexMixin, AnthropicProvider):
    """Claude on Vertex AI."""

    name: ClassVar[str] = "vertex"
    env_key: ClassVar[str] = ""
    default_model: ClassVar[str] = "claude-opus-5"
    publisher: ClassVar[str] = "anthropic"
    BASE_URL: ClassVar[str] = "https://aiplatform.googleapis.com"

    def __init__(self, *, project: str = "", region: str = "",
                 access_token: str | None = None, credentials: Any = None,
                 **kw: Any) -> None:
        super().__init__(api_key="vertex", **kw)
        self._setup(project, region, access_token, credentials)

    def _payload(self, req: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
        payload = super()._payload(req, stream=stream)
        # The model is in the URL, and the API version moves into the body.
        payload.pop("model", None)
        payload.pop("stream", None)
        payload["anthropic_version"] = VERTEX_VERSION
        return payload

    async def complete(self, req: CompletionRequest) -> ModelResponse:
        started = time.perf_counter()
        raw = await self._vertex_post(self._model_url(req.model), self._payload(req))
        from ..types import Message, TextBlock
        from .anthropic import _STOP_MAP

        blocks = self._decode_blocks(raw.get("content") or [])
        stop = _STOP_MAP.get(raw.get("stop_reason") or "end_turn", "end_turn")
        if stop == "error":
            blocks.append(TextBlock(text="[refused]"))
        return self._finish(message=Message(role="assistant", content=blocks),
                            stop_reason=stop, usage=self._usage(raw),
                            model=req.model, raw=raw,
                            latency_ms=(time.perf_counter() - started) * 1000)


class VertexGeminiProvider(_VertexMixin, GeminiProvider):
    """Gemini on Vertex AI."""

    name: ClassVar[str] = "vertex-gemini"
    env_key: ClassVar[str] = ""
    default_model: ClassVar[str] = "gemini-2.5-pro"
    publisher: ClassVar[str] = "google"
    action: ClassVar[str] = "generateContent"
    BASE_URL: ClassVar[str] = "https://aiplatform.googleapis.com"

    def __init__(self, *, project: str = "", region: str = "",
                 access_token: str | None = None, credentials: Any = None,
                 **kw: Any) -> None:
        super().__init__(api_key="vertex", **kw)
        self._setup(project, region, access_token, credentials)

    async def complete(self, req: CompletionRequest) -> ModelResponse:
        started = time.perf_counter()
        raw = await self._vertex_post(self._model_url(req.model), self._payload(req))
        from ..types import Message, ToolUseBlock
        from .gemini import _STOP_MAP

        candidates = raw.get("candidates") or []
        if not candidates:
            raise ProviderError("vertex returned no candidates", provider=self.name)
        cand = candidates[0]
        blocks = self._decode_parts((cand.get("content") or {}).get("parts") or [])
        stop = _STOP_MAP.get(cand.get("finishReason") or "STOP", "end_turn")
        if any(isinstance(b, ToolUseBlock) for b in blocks):
            stop = "tool_use"
        return self._finish(message=Message(role="assistant", content=blocks),
                            stop_reason=stop, usage=self._usage(raw),
                            model=req.model, raw=raw,
                            latency_ms=(time.perf_counter() - started) * 1000)
