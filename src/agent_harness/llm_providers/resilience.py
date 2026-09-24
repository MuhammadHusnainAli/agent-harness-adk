"""What happens when a model call fails: retries, backoff, classification, circuits.

Every adapter sends through one transport (`Provider._send`), and that transport
leans on this module for four decisions:

1. **What went wrong.** `classify()` turns a status code, headers and body into a
   typed error — `RateLimitError`, `QuotaExceededError`, `ContextWindowExceededError`,
   `AuthenticationError`, `ProviderUnavailableError` and so on — using each
   vendor's own error codes, not just the status. A 429 that means "out of credit"
   is not retried; a 400 that means "prompt too long" says so.
2. **How long to wait.** `retry_after()` reads every hint a provider gives:
   `Retry-After` in seconds or as an HTTP date, `retry-after-ms`, OpenAI's
   `x-ratelimit-reset-*` durations, and Gemini's `RetryInfo.retryDelay` in the
   body. Without a hint, `RetryPolicy.backoff()` is exponential with jitter.
3. **Whether to try again at all.** `RetryPolicy` — a retry count, a wall-clock
   ceiling, and a cap on how long a server may ask us to wait. Ask for longer
   than the cap and the call fails at once with `retry_after` set, so a fallback
   model takes over instead of the run sleeping for five minutes.
4. **Whether to call at all.** `CircuitBreaker` stops sending to a provider that
   keeps failing, so a dead endpoint costs one fast error instead of a full
   retry cycle per call — and the router's fallback chain moves on immediately.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Literal

from ..errors import (
    AuthenticationError,
    ContextWindowExceededError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExceededError,
    RateLimitError,
)

__all__ = [
    "RetryPolicy",
    "RetryEvent",
    "CircuitBreaker",
    "ProviderStats",
    "classify",
    "retry_after",
    "parse_duration",
    "request_id",
]

#: Statuses worth sending again. 408 request timeout, 409 lock contention,
#: 425 too early, 429 rate limited, and every 5xx except the two that mean
#: "this will never work" (501 not implemented, 505 HTTP version).
RETRY_STATUSES = frozenset({408, 409, 425, 429})
NEVER_RETRY_5XX = frozenset({501, 505})


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to try before giving up on a call.

        RetryPolicy(max_retries=5, max_elapsed=120)     # patient
        RetryPolicy.none()                              # fail fast, e.g. in tests

    Waits grow as ``initial_delay * multiplier ** attempt``, capped at
    ``max_delay``, with ``jitter`` of that shaved off at random so a hundred
    parallel sub-agents that were throttled together do not retry together.
    """

    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 30.0
    multiplier: float = 2.0
    #: Fraction of each wait that is randomised: 0 is deterministic, 1 is "full jitter".
    jitter: float = 0.25
    #: The longest a server may ask us to wait. Longer than this and the call
    #: fails at once, carrying `retry_after`, so a fallback model can take over.
    max_retry_after: float = 60.0
    #: A ceiling on the whole call, retries and waits included. None: no ceiling.
    max_elapsed: float | None = None
    retry_on_timeout: bool = True
    retry_on_connection_error: bool = True
    retry_statuses: frozenset[int] = RETRY_STATUSES

    @classmethod
    def none(cls) -> RetryPolicy:
        return cls(max_retries=0)

    def should_retry_status(self, status: int) -> bool:
        if status in self.retry_statuses:
            return True
        return 500 <= status < 600 and status not in NEVER_RETRY_5XX

    def backoff(self, attempt: int) -> float:
        """The wait before retry number ``attempt + 1`` when the server gave no hint."""
        base = min(self.max_delay, self.initial_delay * self.multiplier ** attempt)
        spread = base * max(0.0, min(self.jitter, 1.0))
        return max(0.0, base - random.uniform(0.0, spread))


@dataclass
class RetryEvent:
    """Handed to `on_retry` before each wait, so a retry is never silent."""

    provider: str
    attempt: int                 # the attempt that just failed, from 1
    delay: float                 # seconds until the next one
    error: ProviderError
    url: str = ""

    @property
    def reason(self) -> str:
        return f"{type(self.error).__name__}: {self.error}"


@dataclass
class ProviderStats:
    """Running counters for one provider instance. `provider.stats.snapshot()`."""

    requests: int = 0            # HTTP attempts, retries included
    calls: int = 0               # logical calls (complete / stream / embed)
    succeeded: int = 0
    failed: int = 0
    retries: int = 0
    rate_limited: int = 0
    timeouts: int = 0
    circuit_rejections: int = 0
    total_backoff_s: float = 0.0
    last_error: str = ""
    last_request_id: str = ""
    by_status: dict[int, int] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "requests": self.requests, "calls": self.calls,
            "succeeded": self.succeeded, "failed": self.failed,
            "retries": self.retries, "rate_limited": self.rate_limited,
            "timeouts": self.timeouts, "circuit_rejections": self.circuit_rejections,
            "total_backoff_s": round(self.total_backoff_s, 3),
            "last_error": self.last_error, "last_request_id": self.last_request_id,
            "by_status": dict(self.by_status),
        }


class CircuitBreaker:
    """Stop calling a provider that keeps failing, then probe it back to health.

    ``closed``     calls flow; consecutive failures are counted
    ``open``       after `failure_threshold` of them, calls fail at once with
                   `ProviderUnavailableError` for `reset_timeout` seconds
    ``half_open``  then one call is let through; success closes the circuit,
                   failure opens it again

    Only failures that say the *provider* is unwell count — 429, 5xx, timeouts,
    dropped connections. A 400 is the caller's fault and proves the provider is
    up, so it counts as a success.
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.opened_at = 0.0
        self._state: Literal["closed", "open", "half_open"] = "closed"
        self._probing = False

    @property
    def state(self) -> Literal["closed", "open", "half_open"]:
        if self._state == "open" and time.monotonic() - self.opened_at >= self.reset_timeout:
            self._state = "half_open"
            self._probing = False
        return self._state

    def allow(self) -> bool:
        """May a call go out now? Reserves the probe slot when half open."""
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self._probing:
            self._probing = True
            return True
        return False

    def retry_in(self) -> float:
        return max(0.0, self.reset_timeout - (time.monotonic() - self.opened_at))

    def record_success(self) -> None:
        self.failures = 0
        self._state = "closed"
        self._probing = False

    def record_failure(self) -> None:
        self.failures += 1
        self._probing = False
        if self._state == "half_open" or self.failures >= self.failure_threshold:
            self._state = "open"
            self.opened_at = time.monotonic()

    def reset(self) -> None:
        self.record_success()


# --- reading what the provider told us --------------------------------------------

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def parse_duration(text: str) -> float | None:
    """`"1.5s"`, `"250ms"`, `"6m0s"`, `"1h2m"`, or a bare number of seconds."""
    text = (text or "").strip().lower()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    parts = _DURATION.findall(text)
    if not parts or "".join(n + u for n, u in parts) != text.replace(" ", ""):
        return None
    scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(n) * scale[u] for n, u in parts)


def _header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    try:
        return headers.get(name)
    except AttributeError:
        return None


def retry_after(headers: Any, body: str | None = None, *,
                rate_limited: bool = True) -> float | None:
    """Seconds the provider asked us to wait, from whichever hint it gave.

    OpenAI's bucket-reset headers ride on every response, so they are only read
    when the call was actually rate limited.
    """
    ms = _header(headers, "retry-after-ms")
    if ms:
        try:
            return max(0.0, float(ms) / 1000.0)
        except ValueError:
            pass
    value = _header(headers, "retry-after")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
                return max(0.0, when.timestamp() - time.time())
            except (TypeError, ValueError, IndexError, OverflowError):
                pass
    # OpenAI: how long until the exhausted bucket refills. Wait for the later one.
    if rate_limited:
        resets = [parse_duration(_header(headers, h) or "") for h in
                  ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")]
        found = [r for r in resets if r is not None]
        if found:
            return max(found)
    # Gemini / Vertex: google.rpc.RetryInfo inside the error body.
    if body:
        match = re.search(r'"retryDelay"\s*:\s*"([^"]+)"', body)
        if match:
            return parse_duration(match.group(1))
    return None


def request_id(headers: Any) -> str | None:
    for name in ("request-id", "x-request-id", "x-amzn-requestid",
                 "x-amz-request-id", "x-goog-request-id", "apim-request-id",
                 "x-ms-request-id", "cf-ray"):
        value = _header(headers, name)
        if value:
            return value
    return None


def _error_fields(body: str | None) -> tuple[str, str]:
    """(message, code) out of whatever error envelope the vendor uses."""
    if not body:
        return "", ""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return body.strip()[:500], ""
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return str(data)[:500], ""
    err = data.get("error", data)
    if isinstance(err, str):
        return err, str(data.get("code") or data.get("type") or "")
    if not isinstance(err, dict):
        return str(err)[:500], ""
    message = str(err.get("message") or data.get("message") or data.get("Message") or "")
    code = str(err.get("code") or err.get("type") or err.get("status")
               or data.get("__type") or "")
    for detail in err.get("details") or []:           # Google: ErrorInfo.reason
        if isinstance(detail, dict) and detail.get("reason"):
            code = f"{code}:{detail['reason']}" if code else str(detail["reason"])
    return message, code


_CONTEXT_MARKERS = (
    "context_length_exceeded", "context length", "context window",
    "prompt is too long", "input is too long", "too many tokens",
    "maximum number of tokens", "request_too_large",
)
_QUOTA_MARKERS = ("insufficient_quota", "billing", "credit balance",
                  "exceeded your current quota", "quota exceeded")
_AUTH_MARKERS = ("api_key_invalid", "invalid api key", "invalid x-api-key",
                 "unauthenticated", "permission_denied", "invalid_api_key")


def classify(provider: str, status: int, *, headers: Any = None,
             body: str | None = None, policy: RetryPolicy | None = None) -> ProviderError:
    """The right `ProviderError` subclass for an HTTP failure."""
    message, code = _error_fields(body)
    lowered = f"{message} {code}".lower()
    rid = request_id(headers)
    wait = retry_after(headers, body, rate_limited=status == 429)
    detail = message or (body or "").strip()[:300] or "no body"
    text = f"{provider} returned {status}: {detail}"
    if rid:
        text += f" (request id {rid})"
    kw: dict[str, Any] = dict(provider=provider, status=status, body=(body or "")[:2000],
                              retry_after=wait, request_id=rid, code=code or None)

    should = (_header(headers, "x-should-retry") or "").lower()

    cls: type[ProviderError]
    if status == 429 or "throttl" in lowered or "resource_exhausted" in lowered:
        cls = QuotaExceededError if any(m in lowered for m in _QUOTA_MARKERS) else RateLimitError
    elif any(m in lowered for m in _CONTEXT_MARKERS) or status == 413:
        cls = ContextWindowExceededError
    elif status in (401, 403) or any(m in lowered for m in _AUTH_MARKERS):
        cls = AuthenticationError
    elif status == 404 or "model_not_found" in lowered or "not_found_error" in lowered:
        cls = ModelNotFoundError
    elif status == 402:
        cls = QuotaExceededError
    elif status in (408, 504):
        cls = ProviderTimeoutError
    elif (policy or RetryPolicy()).should_retry_status(status) or "overloaded" in lowered:
        cls = ProviderUnavailableError
    else:
        cls = InvalidRequestError

    err = cls(text, **kw)
    # OpenAI and Anthropic say outright whether a retry is worthwhile. Believe them.
    if should == "true":
        err.retryable = True
    elif should == "false":
        err.retryable = False
    return err
