"""Every exception the harness raises, in one place."""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for all agent-harness errors."""


class ConfigurationError(HarnessError):
    """Something was wired up wrong before the run ever started."""


class ProviderError(HarnessError):
    """A model provider returned something we could not use."""

    def __init__(self, message: str, *, provider: str = "", status: int | None = None,
                 body: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.body = body


class RateLimitError(ProviderError):
    """The provider asked us to slow down."""


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
