//! Modes, end to end: chat threads, research ledgers, cowork questions.

use std::sync::Arc;

use agent_harness::llm_providers::{FakeProvider, Script, tool_call};
use agent_harness::modes::{self, Asker};
use agent_harness::types::{ContentBlock, StreamEventType, TodoStatus};
use agent_harness::{Agent, RunOptions, SubAgentSpec, Tool};
use futures::StreamExt;
use futures::future::FutureExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

fn fake(script: Vec<Script>) -> Arc<FakeProvider> {
    Arc::new(FakeProvider::new(script))
}

fn result_of(p: &FakeProvider, request: usize) -> String {
    let msg = p.requests()[request].messages.last().unwrap().clone();
    match &msg.content[0] {
        ContentBlock::ToolResult(r) => r.content.clone(),
        _ => msg.text(),
    }
}

/// Search the web.
#[derive(Deserialize, JsonSchema)]
struct Search {
    query: String,
}

fn search() -> Tool {
    Tool::new("search", |a: Search, _| async move { Ok(format!("Results for {}: https://example.com/a says the sky is blue; docs.rs/b agrees.", a.query)) })
}

#[tokio::test]
async fn chat_keeps_the_thread() {
    let p = fake(vec!["hi Ada".into(), "your name is Ada".into()]);
    let a = Agent::builder("chatty").provider(p.clone()).mode(modes::chat(Some("fast")).unwrap()).build().unwrap();
    assert!(a.conversational());
    assert_eq!(a.max_steps_hint(), 6);
    assert_eq!(a.model(), "claude-haiku-4-5", "naming a depth asks for its tier");
    let r1 = a.run("I'm Ada").await.unwrap();
    let r2 = a.run("what's my name?").await.unwrap();
    assert_eq!(r2.session_id, r1.session_id);
    assert_eq!(p.requests()[1].messages.len(), 3);
    assert_eq!(r2.mode, "chat");
    assert!(p.requests()[0].system.as_deref().unwrap().contains("## How you work\nYou are in a conversation."));
    a.new_session();
    a.run("again").await.unwrap();
    assert_eq!(p.requests()[2].messages.len(), 1);
}

#[tokio::test]
async fn research_records_cites_and_appends_sources() {
    let p = fake(vec![
        tool_call("todo_write", json!({"todos": [{"content": "find the colour", "status": "in_progress"}]})),
        tool_call("search", json!({"query": "sky colour"})),
        // A source nothing has mentioned is refused...
        tool_call("record_source", json!({"ref": "https://made-up.org", "finding": "x"})),
        tool_call("record_source", json!({"ref": "https://example.com/a", "finding": "the sky is blue", "title": "Example"})),
        tool_call("record_source", json!({"ref": "docs.rs/b", "finding": "agrees"})),
        // ...an answer citing an unrecorded number, with the todo still open, is sent back...
        "The sky is blue [1][3].".into(),
        tool_call("todo_write", json!({"todos": [{"content": "find the colour", "status": "done"}]})),
        // ...and the finished one is delivered with its sources.
        "The sky is blue [1], as others agree [2].".into(),
    ]);
    let a = Agent::builder("analyst").provider(p.clone()).tool(search()).mode(modes::research(Some("fast")).unwrap()).build().unwrap();
    assert_eq!(a.tools().names(), vec!["recall", "record_source", "remember", "search", "todo_write"]);
    let r = a.run("What colour is the sky?").await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert!(result_of(&p, 3).contains("nothing you have read in this run mentions 'https://made-up.org'"));
    assert_eq!(result_of(&p, 4), "Recorded as [1] — cite it as [1]. 1 sources recorded so far.");
    let feedback = result_of(&p, 6);
    assert!(feedback.starts_with("That is not finished yet:"), "{feedback}");
    assert!(feedback.contains("items on your todo list are still open: find the colour"));
    assert!(feedback.contains("you cite [3] but no such source is recorded"));
    assert_eq!(r.output, "The sky is blue [1], as others agree [2].\n\n## Sources\n- [1] Example — https://example.com/a\n- [2] docs.rs/b");
    assert_eq!(r.sources.len(), 2);
    assert_eq!(r.todos[0].status, TodoStatus::Done);
    assert!(r.violations.is_empty());
    assert_eq!(r.artifacts[0].name, "report.md");
    assert_eq!((r.mode.as_str(), r.depth.as_str()), ("research", "fast"));
}

#[tokio::test]
async fn research_needs_something_to_read_with() {
    let a = Agent::builder("empty").provider(fake(vec![])).mode(modes::research(None).unwrap()).build().unwrap();
    let r = a.run("anything").await.unwrap();
    assert!(r.error.unwrap().contains("is in research mode but has nothing to read with"));
}

#[tokio::test]
async fn helpers_share_the_ledger() {
    // Lead and helper share one scripted provider.
    let p = fake(vec![
        tool_call("delegate", json!({"agent_name": "reader", "task": "find the colour"})),
        tool_call("search", json!({"query": "sky"})),
        tool_call("record_source", json!({"ref": "https://example.com/a", "finding": "blue"})),
        "It is blue [1].".into(),
        "Blue [1].".into(),
    ]);
    let mode = agent_harness::modes::Mode { min_sources: 1, todos: false, ..modes::research(Some("fast")).unwrap() };
    let lead =
        Agent::builder("lead").provider(p.clone()).tool(search()).mode(mode).subagent_spec(SubAgentSpec::new("reader", "Reads.", "Read.")).build().unwrap();
    assert_eq!(lead.subagents()["reader"].tools().names(), vec!["record_source", "search"], "the lead-only todo list is not inherited");
    let r = lead.run("sky?").await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert!(p.requests()[1].messages[0].text().contains("## Sources\nRecord every source you rely on with `record_source`"));
    assert_eq!(r.sources.len(), 1, "the helper's source is in the lead's ledger");
    assert_eq!(r.output, "Blue [1].\n\n## Sources\n- [1] https://example.com/a");
}

#[tokio::test]
async fn cowork_asks_the_person() {
    let asker: Asker = Arc::new(|q: String, opts: Vec<String>| async move { format!("{q} -> {}", opts.join("/")) }.boxed());
    let mode = agent_harness::modes::Mode { max_questions: 1, ..modes::cowork(Some("fast"), Some(asker)).unwrap() };
    let p = fake(vec![
        tool_call("todo_write", json!({"todos": [{"content": "pick a colour", "status": "done"}]})),
        tool_call("ask_user", json!({"question": "Which colour?", "options": ["red", "blue"]})),
        tool_call("ask_user", json!({"question": "Again?"})),
        "Painted it blue.".into(),
    ]);
    let a = Agent::builder("helper").provider(p.clone()).mode(mode).build().unwrap();
    let r = a.run("paint the fence").await.unwrap();
    assert_eq!(r.output, "Painted it blue.");
    assert_eq!(result_of(&p, 2), "Which colour? -> red/blue");
    assert!(result_of(&p, 3).starts_with("You have asked 1 questions, which is the limit"));
    assert!(p.requests()[0].system.as_deref().unwrap().contains("`ask_user` is for a decision"));
}

#[tokio::test]
async fn the_last_step_hands_over() {
    let p = Arc::new(FakeProvider::new(vec![tool_call("search", json!({"query": "x"}))]).looping());
    let a = Agent::builder("w").provider(p.clone()).tool(search()).mode(modes::chat(None).unwrap()).max_steps(2).build().unwrap();
    let events: Vec<_> = a.stream("go", RunOptions::default()).await.unwrap().collect().await;
    let last = &p.requests()[1];
    assert_eq!(last.tool_choice, Some(json!("none")));
    assert!(last.system.as_deref().unwrap().ends_with("This is your last step. Answer now with what you have."));
    assert!(events.iter().any(|e| e.kind == StreamEventType::RunEnd));
}

#[tokio::test]
async fn progress_is_reported() {
    let p = fake(vec![tool_call("todo_write", json!({"todos": [{"content": "a", "status": "done"}]})), "done".into()]);
    let a = Agent::builder("w").provider(p).mode(modes::cowork(None, None).unwrap()).build().unwrap();
    let events: Vec<_> = a.stream("go", RunOptions::default()).await.unwrap().collect().await;
    let progress = events.iter().find(|e| e.kind == StreamEventType::Progress).unwrap();
    assert_eq!(progress.text, "1 items · 1 done");
    let bad = fake(vec![tool_call("todo_write", json!({"todos": [{"content": "a", "status": "skipped"}]})), "ok".into()]);
    let a = Agent::builder("w").provider(bad.clone()).mode(modes::cowork(None, None).unwrap()).build().unwrap();
    a.run("go").await.unwrap();
    assert_eq!(result_of(&bad, 1), "Error: item 1 is skipped without a `note` — say why it was not done");
}
