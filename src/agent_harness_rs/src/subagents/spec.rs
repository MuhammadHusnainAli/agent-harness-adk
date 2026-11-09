//! The blueprint a sub-agent is built from: serialisable, versionable.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::llm_providers::parameters::Effort;
use crate::runtime::Budget;

const STOPWORDS: [&str; 48] = [
    "with", "from", "this", "that", "into", "your", "their", "them", "please", "would", "should", "about", "when", "what", "which", "where", "have", "been",
    "make", "made", "will", "must", "such", "than", "then", "they", "also", "each", "only", "very", "just", "over", "more", "most", "some", "here", "there",
    "were", "does", "done", "need", "needs", "want", "wants", "give", "take", "using", "use",
];

fn tokens(text: &str) -> HashSet<String> {
    let cleaned: String = text.chars().map(|c| if c.is_alphanumeric() { c.to_lowercase().next().unwrap_or(c) } else { ' ' }).collect();
    cleaned.split_whitespace().map(str::to_string).collect()
}

fn head(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

fn matches_any(word: &str, hay: &HashSet<String>) -> bool {
    if hay.contains(word) {
        return true;
    }
    let w = head(word, 5);
    hay.iter().filter(|t| t.chars().count() >= 4).any(|t| t.starts_with(w) || word.starts_with(head(t, 5)))
}

/// Where a sub-agent works.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkspaceKind {
    Isolated,
    Shared,
    #[default]
    None,
}

/// Where a spec came from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Origin {
    #[default]
    Bench,
    Factory,
}

/// The blueprint a sub-agent is built from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SubAgentSpec {
    pub name: String,
    pub description: String,
    pub instructions: String,
    /// Glob allowlist; `None` inherits the parent's tools.
    pub tools: Option<Vec<String>>,
    pub skills: Option<Vec<String>>,
    pub model: Option<String>,
    pub mode: Option<String>,
    pub depth: Option<String>,
    pub tier: Option<String>,
    pub effort: Option<Effort>,
    pub max_steps: u32,
    pub max_tokens: u32,
    pub temperature: Option<f64>,
    pub thinking: Option<bool>,
    pub thinking_budget: Option<i64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub min_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub output_schema: Option<Value>,
    pub workspace: WorkspaceKind,
    pub allow_shell: bool,
    pub allow: Vec<String>,
    pub ask: Vec<String>,
    pub deny: Vec<String>,
    /// What it must do before it may call itself done (ported with guardrails, M4).
    pub guardrails: Option<Value>,
    pub budget: Option<Budget>,
    pub identity: Option<Value>,
    /// Sub-agents start clean by design.
    pub memory: bool,
    pub version: String,
    pub origin: Origin,
    pub tags: Vec<String>,
}

impl Default for SubAgentSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            instructions: String::new(),
            tools: None,
            skills: None,
            model: None,
            mode: None,
            depth: None,
            tier: None,
            effort: None,
            max_steps: 12,
            max_tokens: 8192,
            temperature: None,
            thinking: None,
            thinking_budget: None,
            top_p: None,
            top_k: None,
            min_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            seed: None,
            output_schema: None,
            workspace: WorkspaceKind::None,
            allow_shell: false,
            allow: vec![],
            ask: vec![],
            deny: vec![],
            guardrails: None,
            budget: None,
            identity: None,
            memory: false,
            version: "1".into(),
            origin: Origin::Bench,
            tags: vec![],
        }
    }
}

impl SubAgentSpec {
    pub fn new(name: &str, description: &str, instructions: &str) -> Self {
        Self { name: name.into(), description: description.into(), instructions: instructions.into(), ..Default::default() }
    }

    /// A cheap capability score used when no embedder is configured.
    /// Token-level with a shared-prefix rule, so "validate" finds the
    /// validator but a stopword like "with" never staffs anybody.
    pub fn matches(&self, task: &str) -> f64 {
        let hay = tokens(&format!("{} {} {}", self.name, self.description, self.tags.join(" ")));
        let words: Vec<String> = tokens(task).into_iter().filter(|w| !STOPWORDS.contains(&w.as_str()) && w.chars().count() > 3).collect();
        if words.is_empty() {
            return 0.0;
        }
        let hits = words.iter().filter(|w| matches_any(w, &hay)).count();
        hits as f64 / words.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_by_shared_words() {
        let v = SubAgentSpec { tags: vec!["validate".into(), "check".into()], ..SubAgentSpec::new("validator", "Re-checks work.", "") };
        assert!(v.matches("please validate this invoice") > 0.4);
        assert_eq!(v.matches("with this"), 0.0);
        let parsed: SubAgentSpec = serde_json::from_value(serde_json::json!({"name": "x", "workspace": "shared"})).unwrap();
        assert_eq!(parsed.max_steps, 12);
        assert_eq!(parsed.workspace, WorkspaceKind::Shared);
    }
}
