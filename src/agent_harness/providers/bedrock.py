"""Claude on Amazon Bedrock.

Bedrock speaks the Anthropic Messages format, so the encoding and decoding are
shared with the direct provider — what differs is the endpoint, the fact that
the model lives in the URL rather than the body, and SigV4 signing instead of
an API key.

    BedrockProvider(region="eu-west-1")
    Agent("support", model="anthropic.claude-opus-5", provider=BedrockProvider())

Credentials come from the usual places: explicit arguments, the standard
`AWS_*` environment variables, or botocore if it happens to be installed (which
also covers instance roles and SSO).
"""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar

from ..errors import ProviderError
from ..types import ModelResponse
from ._sigv4 import AWSCredentials, resolve_credentials, sign
from .anthropic import AnthropicProvider
from .base import CompletionRequest

__all__ = ["BedrockProvider"]

#: Bedrock takes the API version in the body instead of a header.
BEDROCK_VERSION = "bedrock-2023-05-31"


class BedrockProvider(AnthropicProvider):
    """Anthropic models served by Bedrock, signed with SigV4."""

    name: ClassVar[str] = "bedrock"
    env_key: ClassVar[str] = ""          # signed, not keyed
    default_model: ClassVar[str] = "anthropic.claude-opus-5"
    BASE_URL: ClassVar[str] = ""
    service: ClassVar[str] = "bedrock"

    def __init__(
        self,
        *,
        region: str = "",
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        profile: str | None = None,
        credentials: AWSCredentials | None = None,
        base_url: str | None = None,
        **kw: Any,
    ) -> None:
        import os

        self.region = (region or os.environ.get("AWS_REGION")
                       or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
        endpoint = base_url or f"https://bedrock-runtime.{self.region}.amazonaws.com"
        super().__init__(api_key="bedrock", base_url=endpoint, **kw)
        self._credentials = credentials
        self._credential_args = (access_key, secret_key, session_token, profile)

    @property
    def credentials(self) -> AWSCredentials:
        if self._credentials is None:
            self._credentials = resolve_credentials(*self._credential_args)
        return self._credentials

    # ---- the payload differs in two places ---------------------------------
    def _payload(self, req: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
        payload = super()._payload(req, stream=stream)
        # The model is in the URL on Bedrock, and the API version moves into
        # the body because there is no header to put it in.
        payload.pop("model", None)
        payload.pop("stream", None)
        payload["anthropic_version"] = BEDROCK_VERSION
        return payload

    def _model_path(self, model: str, stream: bool = False) -> str:
        suffix = "invoke-with-response-stream" if stream else "invoke"
        return f"/model/{model}/{suffix}"

    # ---- signing ------------------------------------------------------------
    async def _post(self, path: str, payload: dict[str, Any], *,
                    params: dict[str, str] | None = None) -> dict[str, Any]:
        if not self.credentials:
            raise ProviderError(
                "Bedrock needs AWS credentials — set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY, pass them to BedrockProvider(), or "
                "install boto3 so the usual credential chain is used",
                provider=self.name)

        url = f"{self.base_url}{path}"
        body = json.dumps(payload, separators=(",", ":")).encode()
        headers = sign(method="POST", url=url, region=self.region,
                       service=self.service, body=body,
                       headers={"content-type": "application/json",
                                "accept": "application/json", **self.extra_headers},
                       credentials=self.credentials)

        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            resp = await self.http.post(url, content=body, headers=headers)
            if resp.status_code < 300:
                return resp.json()
            detail = resp.text[:2000]
            if resp.status_code in (408, 429) or resp.status_code >= 500:
                last = ProviderError(f"bedrock returned {resp.status_code}",
                                     provider=self.name, status=resp.status_code,
                                     body=detail)
                if attempt == self.max_retries:
                    raise last
                await self._sleep(attempt, resp.headers.get("retry-after"))
                continue
            raise ProviderError(f"bedrock returned {resp.status_code}: {detail}",
                                provider=self.name, status=resp.status_code,
                                body=detail)
        raise last or ProviderError("unreachable", provider=self.name)

    async def complete(self, req: CompletionRequest) -> ModelResponse:
        started = time.perf_counter()
        raw = await self._post(self._model_path(req.model), self._payload(req))
        return self._decode(raw, req, (time.perf_counter() - started) * 1000)

    def _decode(self, raw: dict[str, Any], req: CompletionRequest,
                latency_ms: float) -> ModelResponse:
        from ..types import Message, TextBlock
        from .anthropic import _STOP_MAP

        blocks = self._decode_blocks(raw.get("content") or [])
        stop = _STOP_MAP.get(raw.get("stop_reason") or "end_turn", "end_turn")
        if stop == "error":
            detail = (raw.get("stop_details") or {}).get("explanation",
                                                         "request refused")
            blocks.append(TextBlock(text=f"[refused] {detail}"))
        return self._finish(message=Message(role="assistant", content=blocks),
                            stop_reason=stop, usage=self._usage(raw),
                            model=req.model, raw=raw, latency_ms=latency_ms)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError(
            "embeddings on Bedrock use a different model family — call it directly")
