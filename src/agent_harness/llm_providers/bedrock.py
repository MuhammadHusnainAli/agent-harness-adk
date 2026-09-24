"""Claude on Amazon Bedrock.

Bedrock speaks the Anthropic Messages format, so the encoding and decoding are
shared with the direct provider — what differs is the endpoint, the fact that
the model lives in the URL rather than the body, and SigV4 signing instead of
an API key.

    BedrockProvider(region="eu-west-1")
    Agent("support", model="anthropic.claude-opus-5", provider=BedrockProvider())

Credentials come from the usual places: explicit arguments, the standard
`AWS_*` environment variables, or botocore if it happens to be installed (which
also covers instance roles and SSO). A Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`)
works too, and skips signing altogether.

Every attempt is signed afresh, so a retry after a long back-off never goes out
with a stale timestamp, and credentials from the chain are re-resolved once if
AWS says they have expired. Streaming decodes Bedrock's binary event stream —
CRC-checked — into the same events the direct API sends.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import zlib
from collections.abc import AsyncIterator
from typing import Any, ClassVar
from urllib.parse import quote

from ..errors import AuthenticationError, ProviderError
from ._sigv4 import AWSCredentials, resolve_credentials, sign
from .anthropic import AnthropicProvider
from .base import CompletionRequest, ProviderField
from .resilience import classify

__all__ = ["BedrockProvider", "decode_event_stream"]

#: Bedrock takes the API version in the body instead of a header.
BEDROCK_VERSION = "bedrock-2023-05-31"

#: Bedrock's stream exceptions, as the HTTP status they would have had.
_EXCEPTION_STATUS = {
    "throttlingException": 429, "serviceUnavailableException": 503,
    "internalServerException": 500, "modelStreamErrorException": 500,
    "modelTimeoutException": 504, "validationException": 400,
    "accessDeniedException": 403, "resourceNotFoundException": 404,
    "modelNotReadyException": 503,
}


class BedrockProvider(AnthropicProvider):
    """Anthropic models served by Bedrock, signed with SigV4."""

    name: ClassVar[str] = "bedrock"
    env_key: ClassVar[str] = ""          # signed, not keyed
    default_model: ClassVar[str] = "anthropic.claude-opus-5"
    BASE_URL: ClassVar[str] = ""
    service: ClassVar[str] = "bedrock"

    display_name: ClassVar[str] = "Amazon Bedrock"
    description: ClassVar[str] = "Claude on AWS Bedrock, SigV4-signed or with a Bedrock API key."
    docs_url: ClassVar[str] = "https://docs.aws.amazon.com/bedrock/latest/userguide/"
    auth_type: ClassVar[str] = "sigv4 or Bedrock API key"
    fields: ClassVar[tuple[ProviderField, ...]] = (
        ProviderField(name="region", env=("AWS_REGION", "AWS_DEFAULT_REGION"),
                      default="us-east-1", example="eu-west-1",
                      description="The AWS region the model is enabled in."),
        ProviderField(name="access_key", type="secret", one_of="aws",
                      env=("AWS_ACCESS_KEY_ID",),
                      description="An IAM access key id (with secret_key)."),
        ProviderField(name="secret_key", type="secret", env=("AWS_SECRET_ACCESS_KEY",),
                      description="The secret for access_key."),
        ProviderField(name="session_token", type="secret", env=("AWS_SESSION_TOKEN",),
                      description="For temporary credentials (STS, SSO)."),
        ProviderField(name="profile", one_of="aws", env=("AWS_PROFILE",),
                      description="A named profile from ~/.aws (needs boto3 installed)."),
        ProviderField(name="api_key", type="secret", one_of="aws",
                      env=("AWS_BEARER_TOKEN_BEDROCK",),
                      description="A Bedrock API key, used instead of SigV4."),
        ProviderField(name="credentials", type="object", one_of="aws",
                      description="An AWSCredentials instance."),
        ProviderField(name="base_url", type="url",
                      description="A VPC endpoint or other override."),
    )
    capabilities: ClassVar[frozenset[str]] = frozenset({
        "streaming", "tools", "vision", "thinking", "json_schema", "prompt_caching"})

    def __init__(
        self,
        *,
        region: str = "",
        access_key: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        profile: str | None = None,
        credentials: AWSCredentials | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kw: Any,
    ) -> None:
        self.region = (region or os.environ.get("AWS_REGION")
                       or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
        endpoint = base_url or f"https://bedrock-runtime.{self.region}.amazonaws.com"
        super().__init__(api_key="bedrock", base_url=endpoint, **kw)
        explicit = any((access_key, secret_key, credentials))
        #: A Bedrock API key — only used when no SigV4 credentials were given.
        self.bearer_token = api_key or (
            "" if explicit else os.environ.get("AWS_BEARER_TOKEN_BEDROCK", ""))
        self._credentials = credentials
        self._credentials_fixed = credentials is not None
        self._credential_args = (access_key, secret_key, session_token,
                                 profile or os.environ.get("AWS_PROFILE"))

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
        # ARNs and versioned ids carry ':' and '/', which must be escaped.
        return f"/model/{quote(model, safe='')}/{suffix}"

    def _messages_path(self, req: CompletionRequest, stream: bool) -> str:
        return self._model_path(req.model, stream)

    # ---- auth ---------------------------------------------------------------
    def _auth_headers(self) -> dict[str, str]:
        return {}

    async def _prepare_headers(self, method: str, url: str, body: bytes,
                               headers: dict[str, str]) -> dict[str, str]:
        streaming = headers.get("accept") == "text/event-stream"
        headers = {**headers, "accept": "application/json"}
        if streaming:
            headers["accept"] = "application/vnd.amazon.eventstream"
            headers["x-amzn-bedrock-accept"] = "application/json"
        if self.bearer_token:
            return {**headers, "authorization": f"Bearer {self.bearer_token}"}
        if not self.credentials:
            raise AuthenticationError(
                "Bedrock needs AWS credentials — set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY, pass them to BedrockProvider(), set "
                "AWS_BEARER_TOKEN_BEDROCK, or install boto3 so the usual credential "
                "chain is used", provider=self.name)
        return sign(method=method, url=url, region=self.region, service=self.service,
                    body=body, headers=headers, credentials=self.credentials)

    async def _refresh_auth(self) -> bool:
        """Chain credentials (SSO, instance roles) expire; resolve them again once."""
        if self._credentials_fixed or self.bearer_token or any(self._credential_args[:2]):
            return False
        self._credentials = None
        return True

    # ---- streaming ---------------------------------------------------------------
    async def _raw_events(self, req: CompletionRequest) -> AsyncIterator[dict[str, Any]]:
        chunks = self._stream_raw(self._messages_path(req, True),
                                  self._payload(req, stream=True), timeout=req.timeout)
        async for headers, payload in decode_event_stream(chunks):  # type: ignore[arg-type]
            kind = headers.get(":message-type", "event")
            if kind == "exception" or kind == "error":
                name = headers.get(":exception-type") or headers.get(":error-code") or ""
                raise classify(self.name, _EXCEPTION_STATUS.get(name, 500),
                               body=payload.decode("utf-8", errors="replace"),
                               policy=self._policy())
            if headers.get(":event-type") != "chunk":
                continue
            envelope = json.loads(payload or b"{}")
            if "bytes" in envelope:
                yield json.loads(base64.b64decode(envelope["bytes"]))

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise NotImplementedError(
            "embeddings on Bedrock use a different model family — call it directly")


async def decode_event_stream(
    chunks: AsyncIterator[bytes],
) -> AsyncIterator[tuple[dict[str, str], bytes]]:
    """AWS's `application/vnd.amazon.eventstream` framing → (headers, payload).

    Each message: total length, headers length, a CRC of those eight bytes, the
    headers, the payload, then a CRC of everything before it. Both CRCs are
    checked, so a corrupted frame is an error rather than garbled JSON.
    """
    buffer = b""
    async for chunk in chunks:
        buffer += chunk
        while len(buffer) >= 12:
            total, header_len, prelude_crc = struct.unpack(">III", buffer[:12])
            if zlib.crc32(buffer[:8]) != prelude_crc:
                raise ProviderError("bedrock event stream is corrupt (prelude CRC)",
                                    provider="bedrock", retryable=True)
            if total < 16:
                raise ProviderError("bedrock event stream is corrupt (length)",
                                    provider="bedrock", retryable=True)
            if len(buffer) < total:
                break
            message, buffer = buffer[:total], buffer[total:]
            (message_crc,) = struct.unpack(">I", message[-4:])
            if zlib.crc32(message[:-4]) != message_crc:
                raise ProviderError("bedrock event stream is corrupt (message CRC)",
                                    provider="bedrock", retryable=True)
            headers = _event_headers(message[12:12 + header_len])
            yield headers, message[12 + header_len:-4]


def _event_headers(raw: bytes) -> dict[str, str]:
    """Decode event-stream headers. Only strings matter here; the rest are skipped."""
    out: dict[str, str] = {}
    i = 0
    fixed = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}
    while i < len(raw):
        name_len = raw[i]
        name = raw[i + 1:i + 1 + name_len].decode("utf-8", errors="replace")
        i += 1 + name_len
        kind = raw[i]
        i += 1
        if kind in (6, 7):                          # bytes, string: 2-byte length
            (length,) = struct.unpack(">H", raw[i:i + 2])
            value = raw[i + 2:i + 2 + length]
            i += 2 + length
            if kind == 7:
                out[name] = value.decode("utf-8", errors="replace")
        elif kind in fixed:
            i += fixed[kind]
        else:
            break
    return out


def encode_event(headers: dict[str, str], payload: bytes) -> bytes:
    """Build one event-stream frame. The inverse of `decode_event_stream`, for tests."""
    raw_headers = b""
    for key, value in headers.items():
        name, data = key.encode(), value.encode()
        raw_headers += bytes([len(name)]) + name + b"\x07" + struct.pack(">H", len(data)) + data
    total = 12 + len(raw_headers) + len(payload) + 4
    prelude = struct.pack(">II", total, len(raw_headers))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    message = prelude + raw_headers + payload
    return message + struct.pack(">I", zlib.crc32(message))
