"""Every exception the harness raises, in one place."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for all agent-harness errors."""


class ConfigurationError(HarnessError):
    """Something was wired up wrong before the run ever started."""


class ProviderError(HarnessError):
    """A model provider returned something we could not use.

    Every provider failure is one of these, so one `except ProviderError` covers
    them all. The subclasses say *what kind* of failure it was, which is what
    decides whether it is worth trying again:

    ``retryable``    the same request may well succeed later (429, 5xx, timeout)
    ``retry_after``  seconds the provider asked us to wait, when it said
    ``request_id``   the vendor's id for the request — quote it to their support
    ``attempts``     how many times it was tried before giving up
    """

    #: The class default; an instance can override it (a 429 whose quota is
    #: exhausted is a rate limit that will not clear by waiting).
    retryable: bool = False

    def __init__(self, message: str, *, provider: str = "", status: int | None = None,
                 body: str | None = None, retryable: bool | None = None,
                 retry_after: float | None = None, request_id: str | None = None,
                 code: str | None = None, attempts: int = 1) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.body = body
        if retryable is not None:
            self.retryable = retryable
        self.retry_after = retry_after
        self.request_id = request_id
        self.code = code
        self.attempts = attempts


class RateLimitError(ProviderError):
    """The provider asked us to slow down (HTTP 429, or a throttling exception)."""

    retryable = True


class QuotaExceededError(RateLimitError):
    """The account is out of credit or quota. Waiting will not fix it."""

    retryable = False


class AuthenticationError(ProviderError):
    """The key, token or signature was refused (HTTP 401/403)."""


class InvalidRequestError(ProviderError):
    """The request itself is wrong (HTTP 400/404/422). Sending it again won't help."""


class ModelNotFoundError(InvalidRequestError):
    """The model id does not exist, or this account cannot use it."""


class ContextWindowExceededError(InvalidRequestError):
    """The prompt is longer than the model can read. Compact it, or pick a bigger model."""


class ProviderTimeoutError(ProviderError):
    """The provider did not answer in time."""

    retryable = True


class ProviderConnectionError(ProviderError):
    """The provider could not be reached at all: DNS, TLS, a dropped connection."""

    retryable = True


class ProviderUnavailableError(ProviderError):
    """The provider is overloaded or failing (HTTP 5xx, 529), or its circuit is open."""

    retryable = True


class ToolError(HarnessError):
    """A tool failed in a way the agent should hear about."""

    def __init__(self, message: str, *, tool: str = "", recoverable: bool = True) -> None:
        super().__init__(message)
        self.tool = tool
        self.recoverable = recoverable


class ToolNotFound(ToolError):
    """The model asked for a tool that is not on its allowlist."""


class PermissionDenied(HarnessError):
    """The policy gate refused an action."""

    def __init__(self, message: str, *, tool: str = "", reason: str = "") -> None:
        super().__init__(message)
        self.tool = tool
        self.reason = reason


class BudgetExceeded(HarnessError):
    """A spend ceiling, token cap, step cap or deadline was hit."""

    def __init__(self, message: str, *, kind: str = "", limit: float = 0.0,
                 spent: float = 0.0) -> None:
        super().__init__(message)
        self.kind = kind
        self.limit = limit
        self.spent = spent


class GuardrailTripped(HarnessError):
    """An input or output check blocked the content."""

    def __init__(self, message: str, *, rule: str = "", where: str = "") -> None:
        super().__init__(message)
        self.rule = rule
        self.where = where


class MaxStepsExceeded(HarnessError):
    """The loop ran out of steps before the agent finished."""


class OutputContractError(HarnessError):
    """The agent's final answer did not match the declared output type."""


class StopRequested(HarnessError):
    """A human pulled the stop control."""


class MCPError(HarnessError):
    """An MCP server misbehaved."""
