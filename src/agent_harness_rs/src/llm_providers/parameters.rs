//! Generation parameters: one vocabulary, validated once, planned per provider.
//!
//! You set parameters in one vocabulary (temperature, top_p, top_k, min_p,
//! frequency_penalty, presence_penalty, repetition_penalty, seed, max_tokens,
//! stop, effort, thinking, thinking_budget) and each provider turns that into
//! what *it* accepts. That step is a [`ParameterPlan`]: the values sent, the
//! ones dropped (with the reason), and the ones adjusted to fit (with the
//! reason). Nothing is dropped or changed silently.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::errors::{Error, Result};

/// Reasoning depth, shallowest first. `None` switches reasoning off where it can be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl Effort {
    pub const ALL: [Effort; 7] = [Effort::None, Effort::Minimal, Effort::Low, Effort::Medium, Effort::High, Effort::Xhigh, Effort::Max];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }

    pub fn parse(s: &str) -> Option<Effort> {
        Self::ALL.into_iter().find(|e| e.as_str() == s)
    }
}

impl std::fmt::Display for Effort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The sampling knobs. A provider lists the ones it has.
pub const SAMPLING_PARAMETERS: [&str; 8] = ["temperature", "top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty", "seed"];

/// Everything a ParameterPlan carries.
pub const GENERATION_PARAMETERS: [&str; 13] = [
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "max_tokens",
    "stop",
    "effort",
    "thinking",
    "thinking_budget",
];

/// (lowest, highest, what it does) — the widest range any provider accepts.
fn range(name: &str) -> Option<(Option<f64>, Option<f64>, &'static str)> {
    Some(match name {
        "temperature" => (Some(0.0), Some(2.0), "randomness: 0 is most deterministic"),
        "top_p" => (Some(0.0), Some(1.0), "nucleus sampling: the probability mass to sample from"),
        "top_k" => (Some(1.0), None, "sample from the k most likely tokens"),
        "min_p" => (Some(0.0), Some(1.0), "drop tokens below this fraction of the top token's probability"),
        "frequency_penalty" => (Some(-2.0), Some(2.0), "penalise tokens by how often they have appeared"),
        "presence_penalty" => (Some(-2.0), Some(2.0), "penalise tokens that have appeared at all"),
        "repetition_penalty" => (Some(0.0), None, "multiplicative repetition penalty; 1 is off"),
        "max_tokens" => (Some(1.0), None, "the longest answer, in tokens"),
        "thinking_budget" => (Some(-1.0), None, "tokens the model may spend thinking; -1 lets Gemini decide"),
        _ => return None,
    })
}

fn fmt_num(x: f64) -> String {
    if x.fract() == 0.0 { format!("{x:.1}") } else { x.to_string() }
}

/// Check generation parameters up front; every problem is reported at once.
/// Takes `(name, value)` pairs; `Value::Null` means unset. Returns the set ones.
pub fn validate_parameters(values: &[(&str, Value)]) -> Result<IndexMap<String, Value>> {
    let mut problems = Vec::new();
    let mut out = IndexMap::new();
    for (name, value) in values {
        if value.is_null() {
            continue;
        }
        let name = *name;
        match name {
            "effort" => {
                if value.as_str().and_then(Effort::parse).is_none() {
                    let levels: Vec<&str> = Effort::ALL.iter().map(|e| e.as_str()).collect();
                    problems.push(format!("effort={value}: use one of {}", levels.join(", ")));
                }
            }
            "thinking" => {
                if !value.is_boolean() {
                    problems.push(format!("thinking={value}: must be True or False"));
                }
            }
            "stop" => {
                if !value.as_array().is_some_and(|a| a.iter().all(Value::is_string)) {
                    problems.push("stop: must be a list of strings".into());
                }
            }
            "seed" => {
                if !(value.is_i64() || value.is_u64()) {
                    problems.push(format!("seed={value}: must be a whole number"));
                }
            }
            _ => {
                if let Some((low, high, meaning)) = range(name) {
                    let integral = matches!(name, "top_k" | "max_tokens" | "thinking_budget");
                    let Some(v) = value.as_f64().filter(|_| !integral || value.is_i64() || value.is_u64()) else {
                        let kind = if integral { "a whole number" } else { "a number" };
                        problems.push(format!("{name}={value}: must be {kind}"));
                        continue;
                    };
                    let too_low = low.is_some_and(|l| if name == "repetition_penalty" { v <= l } else { v < l });
                    if too_low || high.is_some_and(|h| v > h) {
                        let span = match (low, high) {
                            (Some(l), Some(h)) => format!("between {} and {}", fmt_num(l), fmt_num(h)),
                            (Some(l), None) if name == "repetition_penalty" => format!("above {}", fmt_num(l)),
                            (Some(l), None) if integral => format!("at least {}", l as i64),
                            (Some(l), None) => format!("at least {}", fmt_num(l)),
                            _ => String::new(),
                        };
                        problems.push(format!("{name}={value}: must be {span} ({meaning})"));
                    }
                }
            }
        }
        out.insert(name.to_string(), value.clone());
    }
    if !problems.is_empty() {
        return Err(Error::config(format!("invalid generation parameters — {}", problems.join("; "))));
    }
    Ok(out)
}

/// The closest level a model has: the deepest one not beyond what was asked,
/// or failing that the shallowest it offers.
pub fn nearest_effort(level: Effort, supported: &[Effort]) -> Effort {
    let mut ordered: Vec<Effort> = supported.to_vec();
    ordered.sort();
    ordered.dedup();
    if ordered.is_empty() {
        return level;
    }
    ordered.iter().rev().find(|s| **s <= level).copied().unwrap_or(ordered[0])
}

/// What one provider will do with one request's generation parameters.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ParameterPlan {
    pub provider: String,
    pub model: String,
    pub values: IndexMap<String, Value>,
    pub dropped: IndexMap<String, String>,
    pub adjusted: IndexMap<String, String>,
}

/// Python `repr()` of a JSON value, for adjustment notes.
fn repr(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "None".into(),
        Some(Value::String(s)) => format!("'{s}'"),
        Some(Value::Bool(b)) => if *b { "True" } else { "False" }.into(),
        Some(other) => other.to_string(),
    }
}

impl ParameterPlan {
    pub fn new(provider: &str, model: &str, values: IndexMap<String, Value>) -> Self {
        let values = values.into_iter().filter(|(_, v)| !v.is_null() && *v != json!([])).collect();
        Self { provider: provider.into(), model: model.into(), values, ..Default::default() }
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }

    pub fn f64(&self, name: &str) -> Option<f64> {
        self.values.get(name).and_then(Value::as_f64)
    }

    pub fn i64(&self, name: &str) -> Option<i64> {
        self.values.get(name).and_then(Value::as_i64)
    }

    pub fn bool(&self, name: &str) -> Option<bool> {
        self.values.get(name).and_then(Value::as_bool)
    }

    pub fn effort(&self) -> Option<Effort> {
        self.values.get("effort").and_then(Value::as_str).and_then(Effort::parse)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    pub fn drop(&mut self, name: &str, why: impl Into<String>) {
        if self.values.shift_remove(name).is_some() {
            self.dropped.insert(name.into(), why.into());
        }
    }

    pub fn adjust(&mut self, name: &str, value: Value, why: &str) {
        let before = self.values.get(name).cloned();
        if before.as_ref() == Some(&value) {
            return;
        }
        let note = format!("{} → {}: {why}", repr(before.as_ref()), repr(Some(&value)));
        self.values.insert(name.into(), value);
        self.adjusted.insert(name.into(), note);
    }

    /// A value the provider derived, not one that was asked for differently.
    pub fn set(&mut self, name: &str, value: Value) {
        self.values.insert(name.into(), value);
    }

    pub fn as_json(&self) -> Value {
        json!({"provider": self.provider, "model": self.model, "sent": self.values,
               "dropped": self.dropped, "adjusted": self.adjusted})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_all_problems_at_once() {
        let err =
            validate_parameters(&[("temperature", json!(3.0)), ("top_k", json!(0.5)), ("effort", json!("huge")), ("stop", json!("x")), ("seed", json!(1))])
                .unwrap_err()
                .to_string();
        assert!(err.contains("temperature=3.0: must be between 0.0 and 2.0"), "{err}");
        assert!(err.contains("top_k=0.5: must be a whole number"), "{err}");
        assert!(err.contains("effort=\"huge\""), "{err}");
        assert!(err.contains("stop: must be a list of strings"), "{err}");
    }

    #[test]
    fn validation_returns_set_values() {
        let ok = validate_parameters(&[("temperature", json!(0.2)), ("seed", Value::Null), ("max_tokens", json!(10))]).unwrap();
        assert_eq!(ok.len(), 2);
        assert!(validate_parameters(&[("repetition_penalty", json!(0))]).is_err());
        assert!(validate_parameters(&[("thinking_budget", json!(-1))]).is_ok());
    }

    #[test]
    fn nearest_effort_rounds_down_then_up() {
        use Effort::*;
        assert_eq!(nearest_effort(Xhigh, &[Low, Medium, High]), High);
        assert_eq!(nearest_effort(Minimal, &[Low, High]), Low);
        assert_eq!(nearest_effort(Max, &[]), Max);
    }

    #[test]
    fn plan_records_drops_and_adjustments() {
        let mut values = IndexMap::new();
        values.insert("seed".into(), json!(1));
        values.insert("top_p".into(), json!(0.9));
        values.insert("stop".into(), json!([]));
        let mut plan = ParameterPlan::new("anthropic", "claude", values);
        assert!(!plan.contains("stop"));
        plan.drop("seed", "Anthropic has no seed");
        plan.adjust("top_p", json!(0.95), "thinking needs 0.95-1");
        plan.adjust("top_p", json!(0.95), "again");
        assert_eq!(plan.dropped["seed"], "Anthropic has no seed");
        assert_eq!(plan.adjusted["top_p"], "0.9 → 0.95: thinking needs 0.95-1");
        assert_eq!(plan.as_json()["sent"]["top_p"], 0.95);
    }
}
