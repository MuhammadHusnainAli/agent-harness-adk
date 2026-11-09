//! Spec compiler: an agent blueprint becomes the payload a provider accepts.
//!
//! A [`SubAgentSpec`] is a declaration. This turns it into what goes over the
//! wire — a resolved model, a system prompt, the tool schemas, the output
//! contract and the sampling settings — so you can inspect exactly what a
//! sub-agent will be sent *before* you spend anything on it:
//!
//! ```
//! use agent_harness::spec::SpecCompiler;
//! use agent_harness::{SubAgentSpec, ToolRegistry};
//!
//! let spec = SubAgentSpec::new("checker", "Checks things.", "Be strict.");
//! let compiled = SpecCompiler::default().compile(&spec, &ToolRegistry::default(), None, &[], "");
//! println!("{}", compiled.explain());
//! assert_eq!(compiled.model, "claude-sonnet-5");
//! ```

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::llm_providers::base::{CompletionRequest, model_info};
use crate::llm_providers::parameters::Effort;
use crate::prompts::{Section, sections};
use crate::runtime::ModelRouter;
use crate::skills::SkillRegistry;
use crate::subagents::{SubAgentSpec, WorkspaceKind};
use crate::tools::{ToolRegistry, ToolSchema};
use crate::types::Message;

/// A blueprint, resolved. Everything needed to make the call and nothing else.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledSpec {
    pub name: String,
    pub model: String,
    pub system: String,
    pub tools: Vec<ToolSchema>,
    pub tool_choice: Option<Value>,
    pub max_tokens: u32,
    pub temperature: Option<f64>,
    pub effort: Option<Effort>,
    pub thinking: Option<bool>,
    pub thinking_budget: Option<i64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub min_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub response_schema: Option<Value>,
    pub max_steps: u32,
    pub workspace: WorkspaceKind,
    pub allow_shell: bool,
    pub permissions: IndexMap<String, Vec<String>>,
    pub estimated_input_tokens: usize,
    pub estimated_cost_usd: f64,
}

impl CompiledSpec {
    /// The provider payload. This is the end of the compilation.
    pub fn to_request(&self, messages: Vec<Message>) -> CompletionRequest {
        let mut r = CompletionRequest::new(self.model.clone(), messages);
        r.system = if self.system.is_empty() { None } else { Some(self.system.clone()) };
        r.tools = self.tools.clone();
        r.tool_choice = self.tool_choice.clone();
        r.max_tokens = self.max_tokens;
        r.temperature = self.temperature;
        r.effort = self.effort;
        r.thinking = self.thinking;
        r.thinking_budget = self.thinking_budget;
        r.top_p = self.top_p;
        r.top_k = self.top_k;
        r.min_p = self.min_p;
        r.frequency_penalty = self.frequency_penalty;
        r.presence_penalty = self.presence_penalty;
        r.repetition_penalty = self.repetition_penalty;
        r.seed = self.seed;
        r.response_schema = self.response_schema.clone();
        r
    }

    /// What this sub-agent is, in the terms that matter: access and cost.
    pub fn explain(&self) -> String {
        let tools: Vec<&str> = self.tools.iter().map(|t| t.name.as_str()).collect();
        let effort = self.effort.map(|e| format!(" (effort {e})")).unwrap_or_default();
        let workspace = match self.workspace {
            WorkspaceKind::Isolated => "isolated",
            WorkspaceKind::Shared => "shared",
            WorkspaceKind::None => "none",
        };
        let mut lines = vec![
            self.name.clone(),
            format!("  model        {}{effort}", self.model),
            format!("  steps        up to {}, {} tokens per reply", self.max_steps, self.max_tokens),
            format!("  tools        {}", if tools.is_empty() { "none".to_string() } else { tools.join(", ") }),
            format!("  workspace    {workspace}{}", if self.allow_shell { " + shell" } else { "" }),
        ];
        for kind in ["allow", "ask", "deny"] {
            if let Some(entries) = self.permissions.get(kind).filter(|e| !e.is_empty()) {
                lines.push(format!("  {kind:<12} {}", entries.join(", ")));
            }
        }
        if let Some(schema) = &self.response_schema {
            let required: Vec<&str> = schema.get("required").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).collect();
            lines.push(format!("  must return  JSON with {}", if required.is_empty() { "-".to_string() } else { required.join(", ") }));
        }
        lines.push(format!("  system       {} chars, ~{} tokens", self.system.chars().count(), self.estimated_input_tokens));
        if self.estimated_cost_usd > 0.0 {
            lines.push(format!("  first call   ~${:.5}", self.estimated_cost_usd));
        }
        lines.join("\n")
    }
}

/// Blueprint in, provider payload out. No network, no side effects.
#[derive(Debug, Clone, Default)]
pub struct SpecCompiler {
    pub router: ModelRouter,
}

impl SpecCompiler {
    pub fn new(router: ModelRouter) -> Self {
        Self { router }
    }

    /// Resolve a spec into something you can send.
    pub fn compile(
        &self,
        spec: &SubAgentSpec,
        tools: &ToolRegistry,
        skills: Option<&SkillRegistry>,
        memory_blocks: &[String],
        extra_instructions: &str,
    ) -> CompiledSpec {
        let allowed = tools.select(spec.tools.as_deref());
        let task = format!("{} {}", spec.name, spec.description);
        let (model, routed) = self.router.pick(spec.model.as_deref(), spec.tier.as_deref(), &task);
        let effort = spec.effort.or(routed);

        let identity = if spec.description.is_empty() {
            format!("You are {}, an AI agent.", spec.name)
        } else {
            format!("You are {}, an AI agent. {}", spec.name, spec.description)
        };
        let names = allowed.names();
        let tool_note = if names.is_empty() {
            String::new()
        } else {
            format!(
                "Tools you can call: {}.\nCall a tool when it gets you a fact you do not have. Do not guess what a tool would have returned.",
                names.join(", ")
            )
        };
        let skills_index = skills.map(SkillRegistry::index).unwrap_or_default();
        let mut parts = vec![
            Section::Text(&identity),
            Section::Text(&spec.instructions),
            Section::Text(&skills_index),
            Section::Text(&tool_note),
            Section::Text(extra_instructions),
        ];
        parts.extend(memory_blocks.iter().map(|b| Section::Text(b)));
        let mut system = sections(&parts);
        if let Some(schema) = &spec.output_schema {
            system.push_str(&format!(
                "\n\nYour final message must be a single JSON object matching this schema, with no prose and no code fence around it:\n{}",
                serde_json::to_string_pretty(schema).unwrap_or_default()
            ));
        }
        let schemas = allowed.schemas();
        let estimated = (system.chars().count() / 4).max(1) + schemas.iter().map(|t| t.description.chars().count() / 4 + 20).sum::<usize>();
        let cost = model_info(&model).map_or(0.0, |i| ((estimated as f64 * i.input_cost / 1_000_000.0) * 1e8).round() / 1e8);
        let mut permissions = IndexMap::new();
        permissions.insert("allow".to_string(), spec.allow.clone());
        permissions.insert("ask".to_string(), spec.ask.clone());
        permissions.insert("deny".to_string(), spec.deny.clone());
        CompiledSpec {
            name: spec.name.clone(),
            model,
            system,
            tools: schemas,
            tool_choice: None,
            max_tokens: spec.max_tokens,
            temperature: spec.temperature,
            effort,
            thinking: spec.thinking,
            thinking_budget: spec.thinking_budget,
            top_p: spec.top_p,
            top_k: spec.top_k,
            min_p: spec.min_p,
            frequency_penalty: spec.frequency_penalty,
            presence_penalty: spec.presence_penalty,
            repetition_penalty: spec.repetition_penalty,
            seed: spec.seed,
            response_schema: spec.output_schema.clone(),
            max_steps: spec.max_steps,
            workspace: spec.workspace,
            allow_shell: spec.allow_shell,
            permissions,
            estimated_input_tokens: estimated,
            estimated_cost_usd: cost,
        }
    }

    /// Compile and hand back the request in one step.
    pub fn payload(&self, spec: &SubAgentSpec, tools: &ToolRegistry, messages: Vec<Message>) -> CompletionRequest {
        self.compile(spec, tools, None, &[], "").to_request(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{NoArgs, Tool};
    use serde_json::json;

    #[test]
    fn compiles_and_explains() {
        let tools = ToolRegistry::new([Tool::new("lookup", |_: NoArgs, _| async { Ok("") }), Tool::new("write", |_: NoArgs, _| async { Ok("") })]);
        let spec = SubAgentSpec {
            tools: Some(vec!["look*".into()]),
            tier: Some("deep".into()),
            deny: vec!["write".into()],
            output_schema: Some(json!({"type": "object", "required": ["verdict"]})),
            ..SubAgentSpec::new("checker", "Checks invoices.", "Be strict.")
        };
        let c = SpecCompiler::default().compile(&spec, &tools, None, &["## Memory\nx".into()], "");
        assert_eq!(c.model, "claude-opus-5");
        assert_eq!(c.effort, Some(Effort::High));
        assert_eq!(c.tools.len(), 1);
        assert!(c.system.starts_with("You are checker, an AI agent. Checks invoices.\n\nBe strict.\n\nTools you can call: lookup."));
        assert!(c.system.contains("## Memory\nx"));
        assert!(c.estimated_cost_usd > 0.0);
        let text = c.explain();
        assert!(text.contains("  model        claude-opus-5 (effort high)"), "{text}");
        assert!(text.contains("  deny         write"));
        assert!(text.contains("  must return  JSON with verdict"));
        let req = c.to_request(vec![Message::user("go")]);
        assert_eq!(req.tools[0].name, "lookup");
        assert_eq!(req.response_schema.unwrap()["required"][0], "verdict");
    }
}
