//! The factory: a brand-new specialist, written during the run.

use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::bench::Bench;
use super::spec::{Origin, SubAgentSpec};
use crate::agent::{Agent, RunOptions, extract_json};
use crate::prompts::Prompt;
use crate::runtime::BudgetGuard;
use crate::types::new_id;

/// The prompt the factory's builder agent is given.
pub const FACTORY_PROMPT: &str = r#"Write the specification for a brand-new sub-agent that will do exactly one task.

TASK
{task}

TOOLS IT COULD BE GIVEN
{tools}

Decide, in order:
1. name — lower_snake_case, specific to the task, not a generic title.
2. description — one line, what it is accountable for.
3. instructions — its operating instructions. Address it as "you". Say what done
   looks like and what it must never do. No preamble, no flattery.
4. tools — the smallest set from the list above that can finish the task. Least
   privilege: if it only reads, give it no writing tools. Use [] for none.
5. tier — "fast" for mechanical work, "balanced" for normal judgement, "deep" for
   hard reasoning.
6. max_steps — a realistic ceiling for this task.
7. workspace — "isolated" if it writes files, "shared" if it must hand files to
   another sub-agent, otherwise "none".

Return only this JSON object:
{{"name": "...", "description": "...", "instructions": "...", "tools": ["..."],
  "tier": "fast|balanced|deep", "max_steps": 8, "workspace": "none"}}"#;

/// Writes a new sub-agent spec at run time when the bench has nothing that fits.
#[derive(Debug, Clone)]
pub struct SubAgentFactory {
    /// An agent used purely to author the spec.
    pub builder: Agent,
    pub bench: Arc<Mutex<Bench>>,
}

impl SubAgentFactory {
    pub fn new(builder: Agent) -> Self {
        Self { builder, bench: Arc::new(Mutex::new(Bench::standard())) }
    }

    /// A spec for `task`. Falls back to a safe generic spec on any problem.
    /// `guard` is the budget the drafting is charged to.
    pub async fn create(&self, task: &str, tools: &[String], keep: bool, guard: Option<Arc<BudgetGuard>>) -> SubAgentSpec {
        let catalogue =
            if tools.is_empty() { "(no tools available)".to_string() } else { tools.iter().map(|t| format!("- {t}")).collect::<Vec<_>>().join("\n") };
        let prompt = Prompt::new("subagent.factory", FACTORY_PROMPT).render(&json!({"task": task, "tools": catalogue})).unwrap_or_default();
        let mut spec = Self::fallback(task, tools);
        let options = RunOptions { messages: Some(vec![]), guard, ..Default::default() };
        if let Ok(result) = self.builder.run_with(prompt, options).await
            && result.error.is_none()
            && let Some(Value::Object(mut drafted)) = extract_json(&result.output)
        {
            let short: String = task.chars().take(80).collect();
            drafted.entry("description").or_insert_with(|| json!(format!("Purpose-built for: {short}")));
            drafted.insert("origin".into(), json!("factory"));
            if !tools.is_empty()
                && let Some(Value::Array(asked)) = drafted.get("tools").cloned()
            {
                let kept: Vec<Value> = asked.into_iter().filter(|t| t.as_str().is_some_and(|n| tools.iter().any(|a| a == n))).collect();
                drafted.insert("tools".into(), Value::Array(kept));
            }
            if let Ok(parsed) = serde_json::from_value::<SubAgentSpec>(Value::Object(drafted))
                && !parsed.name.is_empty()
            {
                spec = parsed;
            }
        }
        if keep {
            self.bench.lock().unwrap_or_else(|p| p.into_inner()).register(spec.clone());
        }
        spec
    }

    fn fallback(task: &str, tools: &[String]) -> SubAgentSpec {
        let short: String = task.chars().take(80).collect();
        SubAgentSpec {
            name: format!("task_{}", &new_id("")[..6]),
            description: format!("Purpose-built for: {short}"),
            instructions: "You have one task, stated below by the agent that assigned it. Finish it and report the result. Do not expand the scope. If you cannot finish it, say exactly what stopped you.".into(),
            tools: if tools.is_empty() { None } else { Some(tools.to_vec()) },
            origin: Origin::Factory,
            max_steps: 10,
            ..Default::default()
        }
    }
}
