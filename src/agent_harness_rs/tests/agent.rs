//! The agent loop, end to end, against the scripted provider.

use std::sync::{Arc, Mutex};

use agent_harness::errors::{ProviderError, ProviderErrorKind};
use agent_harness::llm_providers::{FakeProvider, Script, tool_call};
use agent_harness::runtime::{Budget, HookEngine, ModelRouter, OnExceed, PolicyGate, Session};
use agent_harness::types::{ContentBlock, Role, StopReason, StreamEventType};
use agent_harness::{Agent, Error, Harness, NoArgs, Permission, RunOptions, Tool, ToolContext};
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

/// Add two numbers.
#[derive(Deserialize, JsonSchema)]
struct Add {
    a: i64,
    b: i64,
}

fn add() -> Tool {
    Tool::new("add", |x: Add, _: ToolContext| async move { Ok(x.a + x.b) })
}

fn fake(script: Vec<Script>) -> Arc<FakeProvider> {
    Arc::new(FakeProvider::new(script))
}

fn agent(provider: Arc<FakeProvider>) -> agent_harness::AgentBuilder {
    Agent::builder("calc").instructions("Do sums.").provider(provider)
}

#[tokio::test]
async fn answers_without_tools() {
    let p = fake(vec!["hello".into()]);
    let r = agent(p.clone()).build().unwrap().run("hi").await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert_eq!(r.output, "hello");
    assert_eq!(r.steps, 1);
    assert_eq!(r.stop_reason, StopReason::EndTurn);
    assert_eq!(r.messages.len(), 2);
    let sent = &p.requests()[0];
    let system = sent.system.as_deref().unwrap();
    assert!(system.starts_with("You are calc, an AI agent."), "{system}");
    assert!(system.contains("Do sums."));
    // No model named: the router's default, at its tier's effort.
    assert_eq!(sent.model, "claude-sonnet-5");
    assert_eq!(sent.effort.map(|e| e.as_str()), Some("medium"));
}

#[tokio::test]
async fn calls_tools_then_answers() {
    let p = fake(vec![tool_call("add", json!({"a": 2, "b": 3})), "5".into()]);
    let a = agent(p.clone()).tool(add()).build().unwrap();
    let r = a.run("2+3?").await.unwrap();
    assert_eq!(r.output, "5");
    assert_eq!(r.steps, 2);
    assert_eq!(r.tool_calls.len(), 1);
    assert_eq!(r.tool_calls[0].name, "add");
    let second = &p.requests()[1];
    let last = second.messages.last().unwrap();
    assert_eq!(last.role, Role::User);
    match &last.content[0] {
        ContentBlock::ToolResult(res) => assert_eq!(res.content, "5"),
        other => panic!("{other:?}"),
    }
    assert!(p.requests()[0].system.as_deref().unwrap().contains("Tools you can call: add, recall, remember."));
}

#[tokio::test]
async fn parallel_calls_and_unknown_tools() {
    let both = Script::Blocks(vec![
        ContentBlock::ToolUse(agent_harness::ToolUseBlock::new("add", json!({"a": 1, "b": 1}).as_object().unwrap().clone())),
        ContentBlock::ToolUse(agent_harness::ToolUseBlock::new("nope", Default::default())),
    ]);
    let p = fake(vec![both, "done".into()]);
    let r = agent(p.clone()).tool(add()).build().unwrap().run("go").await.unwrap();
    assert_eq!(r.output, "done");
    let requests = p.requests();
    let results = &requests[1].messages.last().unwrap().content;
    assert_eq!(results.len(), 2);
    match &results[1] {
        ContentBlock::ToolResult(res) => {
            assert!(res.is_error);
            assert!(res.content.starts_with("no tool named 'nope'"));
        }
        other => panic!("{other:?}"),
    }
}

#[derive(Deserialize, JsonSchema, Debug)]
struct Verdict {
    approved: bool,
}

#[tokio::test]
async fn output_contract_retries_then_parses() {
    let p = fake(vec!["sure!".into(), "```json\n{\"approved\": true}\n```".into()]);
    let r = agent(p.clone()).output_type::<Verdict>().build().unwrap().run("decide").await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert_eq!(r.data, json!({"approved": true}));
    let verdict: Verdict = serde_json::from_value(r.data.clone()).unwrap();
    assert!(verdict.approved);
    let retry = p.requests()[1].messages.last().unwrap().text();
    assert!(retry.starts_with("That did not satisfy the output contract (no JSON object found in the reply)"));
    assert!(p.requests()[0].system.as_deref().unwrap().contains("## Output contract"));
    assert!(p.requests()[0].response_schema.is_some());

    let stubborn = fake(vec!["no".into(), "no".into(), "no".into()]);
    let r = agent(stubborn).output_type::<Verdict>().build().unwrap().run("decide").await.unwrap();
    assert_eq!(r.error.as_deref(), Some("OutputContractError: no JSON object found in the reply"));
    assert_eq!(r.output, "no");
}

#[tokio::test]
async fn budget_stops_cleanly_or_raises() {
    let p = Arc::new(FakeProvider::new(vec![tool_call("add", json!({"a": 1, "b": 1}))]).looping());
    let r = agent(p).tool(add()).budget(Budget { max_tool_calls: Some(2), ..Default::default() }).build().unwrap().run("go").await.unwrap();
    assert!(r.ok());
    assert_eq!(r.stop_reason, StopReason::Budget);
    assert_eq!(r.budget_exceeded.as_deref(), Some("tool_calls"));
    assert!(r.output.contains("[The budget for this agent is exceeded — tool calls 3 of 2."), "{}", r.output);

    let p = Arc::new(FakeProvider::new(vec![tool_call("add", json!({"a": 1, "b": 1}))]).looping());
    let r =
        agent(p).tool(add()).budget(Budget { max_steps: Some(2), on_exceed: OnExceed::Raise, ..Default::default() }).build().unwrap().run("go").await.unwrap();
    assert_eq!(r.stop_reason, StopReason::Error);
    assert_eq!(r.error.as_deref(), Some("BudgetExceeded: step cap reached (2)"));
}

#[tokio::test]
async fn max_steps_is_an_error() {
    let p = Arc::new(FakeProvider::new(vec![tool_call("add", json!({"a": 1, "b": 1}))]).looping());
    let r = agent(p).tool(add()).max_steps(3).build().unwrap().run("go").await.unwrap();
    assert_eq!(r.error.as_deref(), Some("MaxStepsExceeded: calc did not finish within 3 steps"));
    assert_eq!(r.steps, 3);
}

#[tokio::test]
async fn hooks_block_and_rewrite() {
    let mut hooks = HookEngine::new();
    hooks.on("pre_tool", |c| {
        if c.data["args"]["a"] == 666 {
            c.block("no devil maths");
        }
    });
    hooks.on("post_tool", |c| c.replace(json!("rewritten")));
    let p = fake(vec![tool_call("add", json!({"a": 666, "b": 1})), tool_call("add", json!({"a": 1, "b": 1})), "ok".into()]);
    agent(p.clone()).tool(add()).hooks(hooks).build().unwrap().run("go").await.unwrap();
    let blocked = p.requests()[1].messages.last().unwrap().clone();
    match &blocked.content[0] {
        ContentBlock::ToolResult(r) => assert_eq!(r.content, "Blocked: no devil maths"),
        other => panic!("{other:?}"),
    }
    match &p.requests()[2].messages.last().unwrap().content[0] {
        ContentBlock::ToolResult(r) => assert_eq!(r.content, "rewritten"),
        other => panic!("{other:?}"),
    }

    let mut stop = HookEngine::new();
    stop.on("run_start", |c| c.block("closed today"));
    let r = agent(fake(vec![])).hooks(stop).build().unwrap().run("go").await.unwrap();
    assert_eq!(r.error.as_deref(), Some("PermissionDenied: calc may not run: closed today"));
}

#[tokio::test]
async fn policy_gate_refuses() {
    let p = fake(vec![tool_call("add", json!({"a": 1, "b": 1})), tool_call("wipe", json!({})), "ok".into()]);
    let wipe = Tool::new("wipe", |_: NoArgs, _| async { Ok("wiped") }).permission(Permission::Ask);
    let r = agent(p.clone()).tool(add()).tool(wipe).policy(PolicyGate::new(Permission::Allow).deny(["add"])).build().unwrap().run("go").await.unwrap();
    assert!(r.ok());
    let texts: Vec<String> = p.requests()[1..]
        .iter()
        .map(|req| match &req.messages.last().unwrap().content[0] {
            ContentBlock::ToolResult(r) => r.content.clone(),
            _ => String::new(),
        })
        .collect();
    assert_eq!(texts[0], "Not permitted: add is not permitted: denied by policy");
    assert_eq!(texts[1], "Not permitted: wipe needs approval but no approver is configured");
}

#[tokio::test]
async fn sessions_persist_and_continue() {
    let p = fake(vec!["first".into(), "second".into()]);
    let harness = Harness::new();
    let a = agent(p.clone()).harness(harness.clone()).build().unwrap();
    let r1 = a.run("one").await.unwrap();
    let saved = harness.sessions.load(&r1.session_id).await.unwrap();
    assert_eq!(saved.messages.len(), 2);
    assert_eq!(saved.title, "one");
    let r2 = a.run_with("two", RunOptions::session(r1.session_id.clone())).await.unwrap();
    assert_eq!(r2.session_id, r1.session_id);
    assert_eq!(p.requests()[1].messages.len(), 3);
    assert_eq!(harness.sessions.load(&r1.session_id).await.unwrap().messages.len(), 4);
    let missing = a.run_with("x", RunOptions::session("ses_nope")).await.unwrap_err();
    assert!(matches!(missing, Error::Configuration(_)));
}

#[tokio::test]
async fn a_stale_session_is_merged() {
    let harness = Harness::new();
    let mut s = Session::new("calc");
    harness.sessions.save(&mut s).await.unwrap();
    let stale = s.clone();
    let mut moved_on = s.clone();
    moved_on.add(agent_harness::Message::user("meanwhile"));
    harness.sessions.save(&mut moved_on).await.unwrap();
    let a = agent(fake(vec!["ok".into()])).harness(harness.clone()).build().unwrap();
    let r = a.run_with("hello", RunOptions::session(stale)).await.unwrap();
    let held = harness.sessions.load(&r.session_id).await.unwrap();
    let texts: Vec<String> = held.messages.iter().map(|m| m.text()).collect();
    assert_eq!(texts, vec!["meanwhile", "hello", "ok"]);
}

#[tokio::test]
async fn streams_tokens_and_ends_with_result() {
    let p = Arc::new(FakeProvider::new(vec![tool_call("add", json!({"a": 1, "b": 2})), "the answer is 3".into()]).streaming_words(std::time::Duration::ZERO));
    let a = agent(p).tool(add()).build().unwrap();
    let events: Vec<_> = a.stream("go", RunOptions::default()).await.unwrap().collect().await;
    let kinds: Vec<StreamEventType> = events.iter().map(|e| e.kind).collect();
    assert_eq!(kinds.first(), Some(&StreamEventType::RunStart));
    assert!(kinds.contains(&StreamEventType::ToolCall));
    assert!(kinds.contains(&StreamEventType::ToolResult));
    let text: String = events.iter().filter(|e| e.kind == StreamEventType::Text).map(|e| e.text.as_str()).collect();
    assert_eq!(text, "the answer is 3");
    let end = events.last().unwrap();
    assert_eq!(end.kind, StreamEventType::RunEnd);
    assert_eq!(end.data["result"]["output"], "the answer is 3");
}

#[tokio::test]
async fn falls_back_when_a_model_is_down() {
    let down = ProviderError::new(ProviderErrorKind::Unavailable, "overloaded").status(529);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let seen = calls.clone();
    let p = fake(vec![
        Script::Error(down.into()),
        Script::func(move |req| {
            seen.lock().unwrap().push(req.model.clone());
            Script::text("from backup")
        }),
    ]);
    let harness = Harness::new().with_router(ModelRouter::new().with_fallbacks(["backup-model"]));
    let r = agent(p).model("main-model").harness(harness).build().unwrap().run("go").await.unwrap();
    assert_eq!(r.output, "from backup");
    assert_eq!(*calls.lock().unwrap(), vec!["backup-model"]);
}

#[tokio::test]
async fn egress_hook_can_refuse_every_model() {
    let mut hooks = HookEngine::new();
    hooks.on("model_egress", |c| c.block("data must stay home"));
    let r = agent(fake(vec!["x".into()])).hooks(hooks).build().unwrap().run("go").await.unwrap();
    let err = r.error.unwrap();
    assert!(err.starts_with("PermissionDenied: no model in the chain may receive this request"), "{err}");
}

#[tokio::test]
async fn agent_as_tool_and_call_tool() {
    let helper = agent(fake(vec!["helped".into()])).build().unwrap();
    let tool = helper.as_tool(Some("helper"), Some("Helps."));
    assert_eq!(tool.parameters["required"], json!(["task"]));
    let boss_provider = fake(vec![tool_call("helper", json!({"task": "help me"})), "thanks".into()]);
    let boss = Agent::builder("boss").provider(boss_provider.clone()).tool(tool).build().unwrap();
    let r = boss.run("go").await.unwrap();
    assert_eq!(r.output, "thanks");
    match &boss_provider.requests()[1].messages.last().unwrap().content[0] {
        ContentBlock::ToolResult(res) => assert_eq!(res.content, "helped"),
        other => panic!("{other:?}"),
    }
    let a = agent(fake(vec![])).tool(add()).build().unwrap();
    let out = a.call_tool("add", json!({"a": 4, "b": 5}).as_object().unwrap().clone()).await;
    assert_eq!(out.content, "9");
}

#[tokio::test]
async fn bad_parameters_fail_at_build() {
    let err = agent(fake(vec![])).temperature(5.0).build().unwrap_err().to_string();
    assert!(err.starts_with("invalid generation parameters — temperature=5.0"), "{err}");
    let err = agent(fake(vec![])).compact_at(-3.0).build().unwrap_err().to_string();
    assert!(err.contains("compact_at must be a positive token count"));
}

#[test]
fn run_future_is_send() {
    fn is_send<T: Send>(_: &T) {}
    let a = agent(fake(vec![])).build().unwrap();
    let fut = a.run("x");
    is_send(&fut);
    let _: Value = json!(null);
}
