"""AWS Signature Version 4, from the standard library.

Bedrock will not take an API key, so a request has to be signed. This is the
whole of it — about sixty lines — which is cheaper than a dependency and is
checked against the signing vectors AWS publishes.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timezone
from urllib.parse import quote

__all__ = ["sign", "signing_key", "AWSCredentials", "resolve_credentials"]

ALGORITHM = "AWS4-HMAC-SHA256"


class AWSCredentials:
    """An access key, a secret, and optionally a session token."""

    def __init__(self, access_key: str, secret_key: str,
                 token: str | None = None) -> None:
        self.access_key = access_key
        self.secret_key = secret_key
        self.token = token

    def __bool__(self) -> bool:
        return bool(self.access_key and self.secret_key)


def resolve_credentials(access_key: str | None = None, secret_key: str | None = None,
                        token: str | None = None,
                        profile: str | None = None) -> AWSCredentials:
    """Explicit values, then the environment, then botocore if it is installed.

    botocore is only consulted last, and only if it is already there — it knows
    about instance roles, SSO and config files, and re-implementing that would
    be a bad trade.
    """
    access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    token = token or os.environ.get("AWS_SESSION_TOKEN")
    if access_key and secret_key:
        return AWSCredentials(access_key, secret_key, token)
    try:
        import botocore.session

        session = botocore.session.Session(profile=profile)
        frozen = session.get_credentials()
        if frozen is not None:
            frozen = frozen.get_frozen_credentials()
            return AWSCredentials(frozen.access_key, frozen.secret_key, frozen.token)
    except Exception:
        pass
    return AWSCredentials(access_key or "", secret_key or "", token)


def _hmac(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def signing_key(secret: str, date: str, region: str, service: str) -> bytes:
    """The four-step derivation: date, region, service, then the terminator."""
    return _hmac(_hmac(_hmac(_hmac(f"AWS4{secret}".encode(), date), region),
                       service), "aws4_request")


def sign(
    *,
    method: str,
    url: str,
    region: str,
    service: str,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
    credentials: AWSCredentials,
    now: datetime | None = None,
    content_sha_header: bool = True,
) -> dict[str, str]:
    """Return the headers a signed request needs, including `Authorization`."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.netloc
    path = quote(parsed.path or "/", safe="/-_.~")
    query = _canonical_query(parsed.query)

    moment = now or datetime.now(timezone.utc)
    amz_date = moment.strftime("%Y%m%dT%H%M%SZ")
    date = moment.strftime("%Y%m%d")

    signed_headers = {"host": host, "x-amz-date": amz_date}
    for key, value in (headers or {}).items():
        signed_headers[key.lower()] = value.strip()
    if credentials.token:
        signed_headers["x-amz-security-token"] = credentials.token

    payload_hash = hashlib.sha256(body).hexdigest()
    # Required by S3, accepted everywhere else. Off only so the signature can be
    # checked against AWS's published test vectors, which omit it.
    if content_sha_header:
        signed_headers["x-amz-content-sha256"] = payload_hash

    ordered = sorted(signed_headers)
    canonical_headers = "".join(f"{k}:{signed_headers[k]}\n" for k in ordered)
    header_list = ";".join(ordered)

    canonical_request = "\n".join([
        method.upper(), path, query, canonical_headers, header_list, payload_hash,
    ])
    scope = f"{date}/{region}/{service}/aws4_request"
    to_sign = "\n".join([
        ALGORITHM, amz_date, scope,
        hashlib.sha256(canonical_request.encode()).hexdigest(),
    ])
    signature = hmac.new(signing_key(credentials.secret_key, date, region, service),
                         to_sign.encode(), hashlib.sha256).hexdigest()

    out = dict(signed_headers)
    out["Authorization"] = (
        f"{ALGORITHM} Credential={credentials.access_key}/{scope}, "
        f"SignedHeaders={header_list}, Signature={signature}"
    )
    return out


def _canonical_query(query: str) -> str:
    """Parameters sorted by name, each encoded, joined with `&`."""
    if not query:
        return ""
    pairs: list[tuple[str, str]] = []
    for part in query.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        pairs.append((quote(key, safe="-_.~"), quote(value, safe="-_.~")))
    return "&".join(f"{k}={v}" for k, v in sorted(pairs))


