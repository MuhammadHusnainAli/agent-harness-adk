//! Guardrails, in two halves.
//!
//! **Content** — [`Guardrails`] runs pattern rules over anything entering or
//! leaving an agent: API keys redacted, private keys blocked, injection attempts
//! flagged, oversized payloads trimmed. It is harness-wide by default.
//!
//! **Behaviour** — [`AgentGuardrails`] is per agent: which tools it may touch,
//! and what has to be true of its answer before that answer is accepted. A
//! failed check sends the agent back round with a plain-English note about what
//! is missing, rather than failing the run.
//!
//! ```
//! use agent_harness::guardrails::{AgentGuardrails, GuardrailOptions, MaxCost, RequireTools};
//! use std::sync::Arc;
//!
//! let rails = AgentGuardrails::from_options(
//!     vec![Arc::new(RequireTools::new(["order_status"])), Arc::new(MaxCost::new(0.25))],
//!     GuardrailOptions { forbid_tools: vec!["issue_refund".into()], no_placeholders: true, ..Default::default() },
//! )
//! .unwrap();
//! assert_eq!(rails.len(), 3);
//! ```

pub mod agent;
pub mod checks;
pub mod detectors;
pub mod engine;
pub mod llm;
pub mod rules;
pub mod standard;

// content
pub use engine::{Guardrails, GuardrailsConfig};
pub use rules::{Action, INJECTION_RULES, Rule, SECRET_RULES};
// behaviour
pub use agent::{AgentGuardrails, CheckResult, GuardrailOptions, GuardrailReport, OnViolation};
pub use checks::{Check, CompletionContext, Verdict, Violation};
// checks
pub use checks::{
    Custom, ForbidTools, MaxCost, MaxSteps, MinLength, MustInclude, MustMatch, MustNotInclude, NoPlaceholders, RequireCitation, RequireJSON, RequireStructured,
    RequireTools,
};
// detectors
pub use detectors::{
    Detector, Finding, GroundednessDetector, InjectionDetector, PIIDetector, RepetitionDetector, ScanContext, SecretDetector, Severity, ToxicityDetector,
    iban_valid, luhn_valid, shannon_entropy,
};
// detector-backed checks
pub use standard::{DetectorCheck, Grounded, NoInjection, NoPII, NoRepetition, NoSecrets, NotToxic};
// llm judges
pub use llm::{Judge, JudgeAnswer, LLMGuard, LLMGuardStats, LLMVerdict, OnError, POLICIES, ProviderJudge, policy};

/// `x` as Python's `f"{x:.{places}%}"` writes it.
pub(crate) fn pct(x: f64, places: usize) -> String {
    format!("{:.*}%", places, x * 100.0)
}

/// Python's `round(x, places)`: both round the exact binary value, ties to even.
pub(crate) fn round_to(x: f64, places: usize) -> f64 {
    format!("{x:.places$}").parse().unwrap_or(x)
}
