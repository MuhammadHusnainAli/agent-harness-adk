//! Model routing: which model (and how much reasoning) a task gets, and what
//! to fall back to when that model cannot be reached.
//!
//! Explicit beats rule beats tier beats default.

use std::collections::HashMap;
use std::sync::Arc;

use crate::llm_providers::base::{MODELS, Tier, model_info};
use crate::llm_providers::parameters::Effort;
use crate::tools::fnmatch;

fn tier_name(t: Tier) -> &'static str {
    match t {
        Tier::Fast => "fast",
        Tier::Balanced => "balanced",
        Tier::Deep => "deep",
    }
}

/// Match a task or agent name and pin it to a tier (or a model).
#[derive(Debug, Clone)]
pub struct RouteRule {
    pub pattern: String,
    pub tier: String,
    pub model: Option<String>,
    pub effort: Option<Effort>,
}

impl RouteRule {
    pub fn new(pattern: &str, tier: &str) -> Self {
        Self { pattern: pattern.into(), tier: tier.into(), model: None, effort: None }
    }

    pub fn matches(&self, name: &str) -> bool {
        fnmatch(&name.to_lowercase(), &self.pattern.to_lowercase())
    }
}

type Chooser = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// Chooses `(model, effort)` per task.
#[derive(Clone)]
pub struct ModelRouter {
    pub tiers: HashMap<String, String>,
    pub default: String,
    pub rules: Vec<RouteRule>,
    pub effort: HashMap<String, Effort>,
    pub chooser: Option<Chooser>,
    /// Tried in order when the chosen model cannot be reached. A different
    /// provider is the useful fallback.
    pub fallbacks: Vec<String>,
    pub max_attempts: Option<usize>,
}

impl std::fmt::Debug for ModelRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelRouter").field("default", &self.default).field("fallbacks", &self.fallbacks).finish_non_exhaustive()
    }
}

impl Default for ModelRouter {
    fn default() -> Self {
        let tiers: HashMap<String, String> =
            [("fast", "claude-haiku-4-5"), ("balanced", "claude-sonnet-5"), ("deep", "claude-opus-5")].into_iter().map(|(k, v)| (k.into(), v.into())).collect();
        let effort: HashMap<String, Effort> =
            [("fast", Effort::Low), ("balanced", Effort::Medium), ("deep", Effort::High)].into_iter().map(|(k, v)| (k.into(), v)).collect();
        Self { default: tiers["balanced"].clone(), tiers, rules: vec![], effort, chooser: None, fallbacks: vec![], max_attempts: None }
    }
}

impl ModelRouter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_default(mut self, model: &str) -> Self {
        self.default = model.into();
        self
    }

    pub fn with_fallbacks<I: IntoIterator<Item = S>, S: Into<String>>(mut self, models: I) -> Self {
        self.fallbacks = models.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_rule(mut self, rule: RouteRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_chooser(mut self, f: impl Fn(&str) -> Option<String> + Send + Sync + 'static) -> Self {
        self.chooser = Some(Arc::new(f));
        self
    }

    pub fn pick(&self, model: Option<&str>, tier: Option<&str>, task: &str) -> (String, Option<Effort>) {
        if let Some(m) = model.filter(|m| !m.is_empty()) {
            return (m.into(), self.effort_for(m, tier));
        }
        if let (Some(choose), false) = (&self.chooser, task.is_empty())
            && let Some(chosen) = choose(task)
        {
            let e = self.effort_for(&chosen, tier);
            return (chosen, e);
        }
        if !task.is_empty()
            && let Some(rule) = self.rules.iter().find(|r| r.matches(task))
        {
            let picked = rule.model.clone().unwrap_or_else(|| self.tiers.get(&rule.tier).cloned().unwrap_or_else(|| self.default.clone()));
            return (picked, rule.effort.or_else(|| self.effort.get(&rule.tier).copied()));
        }
        if let Some(t) = tier {
            return (self.tiers.get(t).cloned().unwrap_or_else(|| self.default.clone()), self.effort.get(t).copied());
        }
        (self.default.clone(), self.effort_for(&self.default, None))
    }

    fn effort_for(&self, model: &str, tier: Option<&str>) -> Option<Effort> {
        if let Some(t) = tier {
            return self.effort.get(t).copied();
        }
        model_info(model).and_then(|info| self.effort.get(tier_name(info.tier)).copied())
    }

    /// The models to try, in order, starting with `model`.
    pub fn chain(&self, model: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for c in std::iter::once(model.to_string()).chain(self.fallbacks.iter().cloned()) {
            if !c.is_empty() && !out.contains(&c) {
                out.push(c);
            }
        }
        if let Some(n) = self.max_attempts {
            out.truncate(n);
        }
        out
    }

    pub fn tier_of(&self, model: &str) -> &'static str {
        model_info(model).map(|i| tier_name(i.tier)).unwrap_or("balanced")
    }

    pub fn known_models(&self) -> Vec<String> {
        let mut m: Vec<String> = MODELS.read().unwrap_or_else(|p| p.into_inner()).keys().cloned().collect();
        m.sort();
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_in_order_of_precedence() {
        let r = ModelRouter::new().with_rule(RouteRule::new("research*", "deep"));
        assert_eq!(r.pick(Some("gpt-4.1"), None, "x"), ("gpt-4.1".into(), Some(Effort::Medium)));
        assert_eq!(r.pick(None, None, "research the market"), ("claude-opus-5".into(), Some(Effort::High)));
        assert_eq!(r.pick(None, Some("fast"), ""), ("claude-haiku-4-5".into(), Some(Effort::Low)));
        assert_eq!(r.pick(None, None, ""), ("claude-sonnet-5".into(), Some(Effort::Medium)));
        assert_eq!(r.pick(Some("mystery"), None, "").1, None);
        let chosen = ModelRouter::new().with_chooser(|t| t.contains("code").then(|| "gpt-4.1".to_string()));
        assert_eq!(chosen.pick(None, None, "write code").0, "gpt-4.1");
    }

    #[test]
    fn chain_dedupes_and_caps() {
        let r = ModelRouter { max_attempts: Some(2), ..ModelRouter::new().with_fallbacks(["a", "b", "a"]) };
        assert_eq!(r.chain("a"), vec!["a", "b"]);
        assert_eq!(ModelRouter::new().with_fallbacks(["x"]).chain("m"), vec!["m", "x"]);
    }
}
