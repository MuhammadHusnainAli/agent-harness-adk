//! Every error the harness returns, in one place.
//!
//! Python models these as an exception hierarchy. Rust has one [`Error`] enum;
//! provider failures carry a [`ProviderError`] whose [`ProviderErrorKind`] says
//! what kind of failure it was, which is what decides whether to retry.

use std::fmt;

/// The crate-wide result type.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// What kind of provider failure this was. Mirrors the Python subclasses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderErrorKind {
    /// Anything not covered below (`ProviderError` itself in Python).
    Other,
    /// HTTP 429 or a throttling exception.
    RateLimit,
    /// Out of credit or quota. A rate limit that waiting will not fix.
    QuotaExceeded,
    /// The key, token or signature was refused (401/403).
    Authentication,
    /// The request itself is wrong (400/404/422).
    InvalidRequest,
    /// The model id does not exist, or this account cannot use it.
    ModelNotFound,
    /// The prompt is longer than the model can read.
    ContextWindowExceeded,
    /// The provider did not answer in time.
    Timeout,
    /// DNS, TLS, a dropped connection.
    Connection,
    /// Overloaded or failing (5xx, 529), or its circuit is open.
    Unavailable,
}

impl ProviderErrorKind {
    /// The class default for `retryable`.
    pub fn default_retryable(self) -> bool {
        matches!(self, Self::RateLimit | Self::Timeout | Self::Connection | Self::Unavailable)
    }

    /// The Python class name, for messages and logs.
    pub fn name(self) -> &'static str {
        match self {
            Self::Other => "ProviderError",
            Self::RateLimit => "RateLimitError",
            Self::QuotaExceeded => "QuotaExceededError",
            Self::Authentication => "AuthenticationError",
            Self::InvalidRequest => "InvalidRequestError",
            Self::ModelNotFound => "ModelNotFoundError",
            Self::ContextWindowExceeded => "ContextWindowExceededError",
            Self::Timeout => "ProviderTimeoutError",
            Self::Connection => "ProviderConnectionError",
            Self::Unavailable => "ProviderUnavailableError",
        }
    }

    /// `isinstance(err, RateLimitError)` — a quota error is a rate limit too.
    pub fn is_rate_limit(self) -> bool {
        matches!(self, Self::RateLimit | Self::QuotaExceeded)
    }

    /// `isinstance(err, InvalidRequestError)`.
    pub fn is_invalid_request(self) -> bool {
        matches!(self, Self::InvalidRequest | Self::ModelNotFound | Self::ContextWindowExceeded)
    }
}

/// A model provider returned something we could not use.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    pub message: String,
    pub provider: String,
    pub status: Option<u16>,
    pub body: Option<String>,
    /// Whether the same request may succeed later.
    pub retryable: bool,
    /// Seconds the provider asked us to wait, when it said.
    pub retry_after: Option<f64>,
    /// The vendor's id for the request — quote it to their support.
    pub request_id: Option<String>,
    pub code: Option<String>,
    /// How many times it was tried before giving up.
    pub attempts: u32,
    /// Set once the transport has finished retrying; a stream will not retry it again.
    pub(crate) is_final: bool,
}

impl ProviderError {
    pub fn new(kind: ProviderErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            provider: String::new(),
            status: None,
            body: None,
            retryable: kind.default_retryable(),
            retry_after: None,
            request_id: None,
            code: None,
            attempts: 1,
            is_final: false,
        }
    }

    pub fn provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    pub fn retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    pub fn retry_after(mut self, seconds: Option<f64>) -> Self {
        self.retry_after = seconds;
        self
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProviderError {}

/// Every error the harness raises.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    /// Something was wired up wrong before the run ever started.
    #[error("{0}")]
    Configuration(String),

    /// A model provider failed. Boxed: it carries a lot, and errors travel often.
    #[error("{0}")]
    Provider(Box<ProviderError>),

    /// A tool failed in a way the agent should hear about.
    #[error("{message}")]
    Tool { message: String, tool: String, recoverable: bool },

    /// The model asked for a tool that is not on its allowlist.
    #[error("{message}")]
    ToolNotFound { message: String, tool: String },

    /// The policy gate refused an action.
    #[error("{message}")]
    PermissionDenied { message: String, tool: String, reason: String },

    /// A spend ceiling, token cap, step cap or deadline was hit.
    #[error("{message}")]
    BudgetExceeded { message: String, kind: String, limit: f64, spent: f64 },

    /// An input or output check blocked the content.
    #[error("{message}")]
    GuardrailTripped { message: String, rule: String, place: String },

    /// The loop ran out of steps before the agent finished.
    #[error("{0}")]
    MaxStepsExceeded(String),

    /// The final answer did not match the declared output type.
    #[error("{0}")]
    OutputContract(String),

    /// An approval could not be given, taken or resumed.
    #[error("{0}")]
    Approval(String),

    /// Someone else saved this session first.
    #[error("{message}")]
    SessionConflict { message: String, session_id: String },

    /// A human pulled the stop control.
    #[error("{0}")]
    StopRequested(String),

    /// An MCP server misbehaved.
    #[error("{0}")]
    Mcp(String),

    /// A declared workflow could not carry a step through.
    #[error("{0}")]
    Workflow(String),

    /// A file could not be read or written.
    #[error("{0}")]
    Io(String),

    /// A WebSocket closed, or was lost.
    #[error("websocket closed ({code}){}", if reason.is_empty() { String::new() } else { format!(": {reason}") })]
    ConnectionClosed { code: u16, reason: String },

    /// A browser refused a DevTools command, did not answer it, or is gone.
    #[error("{message}")]
    Cdp { message: String, code: Option<i64>, gone: bool },

    /// A vector database (or a memory backend) refused something, or could not be reached.
    #[error("{message}")]
    VectorStore { message: String, store: String, status: Option<u16>, retryable: bool },
}

impl Error {
    pub fn config(message: impl Into<String>) -> Self {
        Self::Configuration(message.into())
    }

    pub fn tool(message: impl Into<String>, tool: impl Into<String>) -> Self {
        Self::Tool { message: message.into(), tool: tool.into(), recoverable: true }
    }

    /// The Python class name, for messages like `"ToolError: ..."`.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Configuration(_) => "ConfigurationError",
            Self::Provider(e) => e.kind.name(),
            Self::Tool { .. } => "ToolError",
            Self::ToolNotFound { .. } => "ToolNotFound",
            Self::PermissionDenied { .. } => "PermissionDenied",
            Self::BudgetExceeded { .. } => "BudgetExceeded",
            Self::GuardrailTripped { .. } => "GuardrailTripped",
            Self::MaxStepsExceeded(_) => "MaxStepsExceeded",
            Self::OutputContract(_) => "OutputContractError",
            Self::Approval(_) => "ApprovalError",
            Self::SessionConflict { .. } => "SessionConflict",
            Self::StopRequested(_) => "StopRequested",
            Self::Mcp(_) => "MCPError",
            Self::Workflow(_) => "WorkflowError",
            Self::Io(_) => "IOError",
            Self::VectorStore { .. } => "VectorStoreError",
            Self::ConnectionClosed { .. } => "ConnectionClosed",
            Self::Cdp { .. } => "CDPError",
        }
    }

    /// The provider failure inside, if this is one.
    pub fn as_provider(&self) -> Option<&ProviderError> {
        match self {
            Self::Provider(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<ProviderError> for Error {
    fn from(e: ProviderError) -> Self {
        Self::Provider(Box::new(e))
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_defaults_follow_kind() {
        assert!(ProviderError::new(ProviderErrorKind::RateLimit, "x").retryable);
        assert!(!ProviderError::new(ProviderErrorKind::QuotaExceeded, "x").retryable);
        assert!(!ProviderError::new(ProviderErrorKind::InvalidRequest, "x").retryable);
        assert!(ProviderError::new(ProviderErrorKind::Unavailable, "x").retryable);
    }

    #[test]
    fn hierarchy_queries() {
        assert!(ProviderErrorKind::QuotaExceeded.is_rate_limit());
        assert!(ProviderErrorKind::ContextWindowExceeded.is_invalid_request());
        assert!(!ProviderErrorKind::Timeout.is_invalid_request());
    }

    #[test]
    fn provider_error_converts_and_displays() {
        let err: Error = ProviderError::new(ProviderErrorKind::Timeout, "slow").into();
        assert_eq!(err.to_string(), "slow");
        assert_eq!(err.kind_name(), "ProviderTimeoutError");
        assert!(err.as_provider().is_some());
    }
}
