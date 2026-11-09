//! Context assembly and compaction — what actually reaches the model.
//!
//! The assembler builds the system prompt from layers: identity, how the
//! agent works, instructions, the skills index, the tools, the output contract.
//! The compactor keeps the message list under budget without ever breaking a
//! tool_use / tool_result pair, which is how naive trimming corrupts a chat.

use std::sync::Arc;

use futures::future::BoxFuture;
use serde_json::Value;

use crate::prompts::{Section, sections};
use crate::skills::SkillRegistry;
use crate::types::{ContentBlock, Message, Role, ToolResultBlock};

/// Turns a prompt into a summary — a cheap model call.
pub type Summarizer = Arc<dyn Fn(String) -> BoxFuture<'static, String> + Send + Sync>;

/// Cheap, provider-agnostic estimate. Good to roughly ±15% on English prose.
pub fn estimate_tokens(text: &str) -> usize {
    (text.chars().count() / 4).max(1)
}

pub fn message_tokens(message: &Message) -> usize {
    let mut total = 8; // per-message overhead the providers add
    for block in &message.content {
        total += match block {
            ContentBlock::Text(t) => estimate_tokens(&t.text),
            ContentBlock::ToolResult(r) => estimate_tokens(&r.content),
            ContentBlock::ToolUse(u) => estimate_tokens(&Value::Object(u.input.clone()).to_string()) + 8,
            // Deliberately not small: under-counting is what overflows.
            other => match other.as_media() {
                Some((_, m)) => 1_500usize.max((m.size() / 750) as usize),
                None => 16,
            },
        };
    }
    total
}

pub fn conversation_tokens(messages: &[Message]) -> usize {
    messages.iter().map(message_tokens).sum()
}

/// Answer every tool call that never got a result.
///
/// A run stopped mid-step leaves an assistant turn asking for tools and
/// nothing after it; providers reject that, so each orphan gets a result
/// saying it was interrupted.
pub fn close_open_tool_calls(messages: Vec<Message>) -> Vec<Message> {
    let mut out: Vec<Message> = Vec::with_capacity(messages.len());
    let mut iter = messages.into_iter().peekable();
    while let Some(message) = iter.next() {
        let calls: Vec<String> = if message.role == Role::Assistant { message.tool_uses().iter().map(|u| u.id.clone()).collect() } else { vec![] };
        out.push(message);
        if calls.is_empty() {
            continue;
        }
        let answered: Vec<String> = iter
            .peek()
            .map(|next| next.content.iter().filter_map(|b| if let ContentBlock::ToolResult(r) = b { Some(r.tool_use_id.clone()) } else { None }).collect())
            .unwrap_or_default();
        let missing: Vec<ContentBlock> = calls
            .into_iter()
            .filter(|id| !answered.contains(id))
            .map(|id| {
                ContentBlock::ToolResult(ToolResultBlock { tool_use_id: id, content: "[interrupted — this call did not finish]".into(), is_error: true })
            })
            .collect();
        if missing.is_empty() {
            continue;
        }
        if !answered.is_empty() {
            // Some were answered: the rest belong in that same turn.
            let next = iter.peek_mut().expect("answered implies a next message");
            next.content.splice(0..0, missing);
        } else {
            out.push(Message::new(Role::User, missing));
        }
    }
    out
}

/// Builds the system prompt. Stable parts first, so prompt caching works.
#[derive(Debug, Clone, Default)]
pub struct ContextAssembler {
    pub identity: String,
    /// How the agent's mode works — before the instructions, so what you
    /// wrote for this agent has the last word.
    pub mode: String,
    pub instructions: String,
    pub skills: Option<SkillRegistry>,
    pub extra: Vec<String>,
}

impl ContextAssembler {
    /// The system prompt for one call.
    pub fn build(&self, tool_names: &[String], output_contract: &str) -> String {
        self.build_with_memory(tool_names, output_contract, &[])
    }

    /// The system prompt for one call, with what memory brings to it (`user.md`,
    /// the job so far, relevant precedent) after the output contract.
    pub fn build_with_memory(&self, tool_names: &[String], output_contract: &str, memory_blocks: &[String]) -> String {
        let skills_index = self.skills.as_ref().map(SkillRegistry::index).unwrap_or_default();
        let tool_note = if tool_names.is_empty() {
            String::new()
        } else {
            format!(
                "Tools you can call: {}.\nCall a tool when it gets you a fact you do not have. Do not guess what a tool would have returned.",
                tool_names.join(", ")
            )
        };
        let mut parts = vec![
            Section::Text(&self.identity),
            Section::Titled("How you work", &self.mode),
            Section::Text(&self.instructions),
            Section::Text(&skills_index),
            Section::Text(&tool_note),
            Section::Titled("Output contract", output_contract),
        ];
        parts.extend(memory_blocks.iter().map(|b| Section::Text(b)));
        parts.extend(self.extra.iter().map(|e| Section::Text(e)));
        sections(&parts)
    }
}

/// Summarise · evict · pin — keeps the window under budget, pairing intact.
#[derive(Clone)]
pub struct ContextCompactor {
    pub max_tokens: usize,
    pub keep_last: usize,
    pub evict_over: usize,
    pub summarize: Option<Summarizer>,
    pub target_ratio: f64,
}

impl Default for ContextCompactor {
    fn default() -> Self {
        Self { max_tokens: 120_000, keep_last: 8, evict_over: 2_000, summarize: None, target_ratio: 0.6 }
    }
}

impl std::fmt::Debug for ContextCompactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContextCompactor").field("max_tokens", &self.max_tokens).field("keep_last", &self.keep_last).finish_non_exhaustive()
    }
}

impl ContextCompactor {
    pub fn should_compact(&self, messages: &[Message], headroom: usize) -> bool {
        conversation_tokens(messages) + headroom > self.max_tokens
    }

    /// A shorter message list that still round-trips to the provider, or
    /// `None` when nothing needed doing.
    pub async fn compact(&self, messages: &[Message], pinned: &[String]) -> Option<Vec<Message>> {
        if !self.should_compact(messages, 0) {
            return None;
        }
        let target = (self.max_tokens as f64 * self.target_ratio) as usize;
        let mut working: Vec<Message> = messages.to_vec();

        // 1. Evict fat tool results first — the cheapest tokens to lose, and
        //    blanking the content keeps every pair intact.
        let protect_from = working.len().saturating_sub(self.keep_last);
        for idx in 0..protect_from {
            if conversation_tokens(&working) <= target {
                break;
            }
            for block in &mut working[idx].content {
                if let ContentBlock::ToolResult(r) = block {
                    let n = r.content.chars().count();
                    if n > self.evict_over {
                        let head: String = r.content.chars().take(400).collect();
                        r.content = format!("{head}\n... [evicted {} characters to free context]", n - 400);
                    }
                }
            }
        }
        if conversation_tokens(&working) <= target {
            return Some(working);
        }

        // 2. Still too big: summarise the head, keep the tail.
        let cut = Self::safe_cut(&working, working.len().saturating_sub(self.keep_last));
        if cut == 0 {
            return Some(working);
        }
        let tail = working.split_off(cut);
        let head = working;
        let transcript: Vec<String> = head.iter().filter(|m| !m.text().is_empty()).map(|m| format!("{}: {}", role_name(m.role), m.text())).collect();
        let transcript = transcript.join("\n");
        let summary = match (&self.summarize, transcript.trim().is_empty()) {
            (Some(summarize), false) => {
                let clipped: String = transcript.chars().take(60_000).collect();
                summarize(format!(
                    "Summarise this conversation so the assistant can continue without it. Keep decisions, facts established, open questions and anything the user asked for. Be dense and specific.\n\n{clipped}"
                ))
                .await
            }
            _ => Self::mechanical_summary(&head),
        };
        let summary = if summary.trim().is_empty() { Self::mechanical_summary(&head) } else { summary };
        let mut block = format!("## Earlier in this conversation (compacted)\n{}", summary.trim());
        if !pinned.is_empty() {
            let pins: Vec<String> = pinned.iter().map(|p| format!("- {p}")).collect();
            block.push_str(&format!("\n\n## Pinned facts\n{}", pins.join("\n")));
        }
        let mut out = vec![Message::user(block)];
        out.extend(tail);
        Some(out)
    }

    /// Move the cut forward until the tail has no orphaned tool results.
    pub fn safe_cut(messages: &[Message], desired: usize) -> usize {
        let mut cut = desired.min(messages.len());
        while cut < messages.len() && messages[cut].content.iter().any(|b| matches!(b, ContentBlock::ToolResult(_))) {
            cut += 1;
        }
        cut
    }

    /// No model available: keep the shape of the conversation, not the bulk.
    pub fn mechanical_summary(messages: &[Message]) -> String {
        let mut lines = Vec::new();
        for msg in messages {
            let text = msg.text();
            if !text.is_empty() {
                let snippet: String = text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(300).collect();
                lines.push(format!("- {}: {snippet}", role_name(msg.role)));
            }
            for u in msg.tool_uses() {
                let args: Vec<String> = u.input.iter().map(|(k, v)| format!("{k}={v}")).collect();
                let mut short = args.join(", ");
                if short.chars().count() > 120 {
                    short = format!("{}...", short.chars().take(120).collect::<String>());
                }
                lines.push(format!("- called {}({short})", u.name));
            }
        }
        let start = lines.len().saturating_sub(80);
        lines[start..].join("\n")
    }
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolUseBlock;

    fn call(id: &str) -> ContentBlock {
        let mut u = ToolUseBlock::new("t", Default::default());
        u.id = id.into();
        ContentBlock::ToolUse(u)
    }

    fn result(id: &str, content: &str) -> ContentBlock {
        ContentBlock::ToolResult(ToolResultBlock { tool_use_id: id.into(), content: content.into(), is_error: false })
    }

    #[test]
    fn closes_orphans() {
        let msgs = vec![Message::user("go"), Message::assistant_blocks(vec![call("a"), call("b")]), Message::new(Role::User, vec![result("a", "ok")])];
        let fixed = close_open_tool_calls(msgs);
        assert_eq!(fixed.len(), 3);
        assert_eq!(fixed[2].content.len(), 2);
        match &fixed[2].content[0] {
            ContentBlock::ToolResult(r) => assert!(r.is_error && r.tool_use_id == "b"),
            other => panic!("{other:?}"),
        }
        let dangling = close_open_tool_calls(vec![Message::assistant_blocks(vec![call("x")])]);
        assert_eq!(dangling.len(), 2);
    }

    #[test]
    fn assembler_orders_sections() {
        let a = ContextAssembler { identity: "You are x.".into(), instructions: "Be kind.".into(), ..Default::default() };
        let s = a.build(&["add".into()], "JSON please");
        assert!(s.starts_with("You are x.\n\nBe kind.\n\nTools you can call: add."));
        assert!(s.ends_with("## Output contract\nJSON please"));
    }

    #[tokio::test]
    async fn compaction_evicts_then_summarises() {
        let c = ContextCompactor { max_tokens: 300, keep_last: 2, evict_over: 100, ..Default::default() };
        let big = "x".repeat(4000);
        let msgs = vec![
            Message::user("start"),
            Message::assistant_blocks(vec![call("1")]),
            Message::new(Role::User, vec![result("1", &big)]),
            Message::assistant("done"),
            Message::user("next"),
        ];
        let out = c.compact(&msgs, &["pin me".into()]).await.unwrap();
        assert!(conversation_tokens(&out) < conversation_tokens(&msgs));
        let tiny = ContextCompactor { max_tokens: 20, keep_last: 1, ..Default::default() };
        let out = tiny.compact(&msgs, &["pin me".into()]).await.unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].text().contains("## Pinned facts\n- pin me"));
        assert!(out[0].text().contains("- called t()"));
        assert!(c.compact(&[Message::user("hi")], &[]).await.is_none());
    }

    #[test]
    fn safe_cut_skips_results() {
        let msgs = vec![Message::assistant_blocks(vec![call("1")]), Message::new(Role::User, vec![result("1", "r")]), Message::user("x")];
        assert_eq!(ContextCompactor::safe_cut(&msgs, 1), 2);
    }
}
