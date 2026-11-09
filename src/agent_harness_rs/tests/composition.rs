//! Sub-agents, run-time specialists, handoffs, versions and resume.

use std::sync::Arc;

use agent_harness::handoff::History;
use agent_harness::llm_providers::{FakeProvider, Script, tool_call};
use agent_harness::runtime::Budget;
use agent_harness::types::{ContentBlock, Role, StopReason, StreamEventType};
use agent_harness::{Agent, AgentVersion, Handoff, Harness, NoArgs, Permission, RunOptions, SubAgentSpec, Tool};
use futures::StreamExt;
use serde_json::json;

fn fake(script: Vec<Script>) -> Arc<FakeProvider> {
    Arc::new(FakeProvider::new(script))
}

fn last_result(p: &FakeProvider, request: usize) -> String {
    match &p.requests()[request].messages.last().unwrap().content[0] {
        ContentBlock::ToolResult(r) => r.content.clone(),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn delegates_to_a_subagent_and_collects_its_result() {
    let researcher_p = fake(vec!["the sky is blue".into()]);
    let researcher = Agent::builder("researcher").description("Finds facts.").provider(researcher_p.clone()).build().unwrap();
    let lead_p =
        fake(vec![tool_call("delegate", json!({"agent_name": "researcher", "task": "what colour is the sky?", "context": "ask nicely"})), "Blue.".into()]);
    let lead = Agent::builder("lead").provider(lead_p.clone()).subagent(researcher).build().unwrap();

    let tools = lead.tools();
    let schema = &tools.get("delegate").unwrap().parameters;
    assert_eq!(schema["properties"]["agent_name"]["enum"], json!(["researcher"]));
    assert_eq!(schema["properties"]["agent_name"]["description"], "researcher: Finds facts.");

    let r = lead.run("sky?").await.unwrap();
    assert_eq!(r.output, "Blue.");
    assert_eq!(r.children.len(), 1);
    assert_eq!(r.children[0].agent, "researcher");
    assert!((r.cost_usd() - (r.usage.cost_usd + r.children[0].usage.cost_usd)).abs() < 1e-9);
    assert_eq!(last_result(&lead_p, 1), "the sky is blue");
    let brief = researcher_p.requests()[0].messages[0].text();
    assert_eq!(brief, "what colour is the sky?\n\n## Context you were given\nask nicely");

    let unknown = fake(vec![tool_call("delegate", json!({"agent_name": "ghost", "task": "x"})), "ok".into()]);
    let lead = Agent::builder("lead").provider(unknown.clone()).subagent(Agent::builder("helper").provider(fake(vec![])).build().unwrap()).build().unwrap();
    lead.run("go").await.unwrap();
    assert_eq!(last_result(&unknown, 1), "No sub-agent named 'ghost'. Available: helper.");
}

#[tokio::test]
async fn subagent_cap_is_enforced() {
    let helper = Agent::builder("helper").provider(fake(vec![Script::text("done")]).clone()).build().unwrap();
    let p = fake(vec![
        tool_call("delegate", json!({"agent_name": "helper", "task": "a"})),
        tool_call("delegate", json!({"agent_name": "helper", "task": "b"})),
        "finished".into(),
    ]);
    let lead = Agent::builder("lead").provider(p.clone()).subagent(helper).budget(Budget { max_subagents: Some(1), ..Default::default() }).build().unwrap();
    let r = lead.run("go").await.unwrap();
    assert_eq!(r.output, "finished");
    assert!(last_result(&p, 2).contains("sub-agent cap reached (1)"), "{}", last_result(&p, 2));
}

#[tokio::test]
async fn builds_subagents_from_specs() {
    let read = Tool::new("read", |_: NoArgs, _| async { Ok("text") });
    let write = Tool::new("write", |_: NoArgs, _| async { Ok("written") });
    let spec = SubAgentSpec { tools: Some(vec!["read".into()]), deny: vec!["read".into()], ..SubAgentSpec::new("reader", "Reads.", "Only read.") };
    let lead = Agent::builder("lead").provider(fake(vec![])).tools([read, write]).subagent_spec(spec).build().unwrap();
    let reader = lead.subagents()["reader"].clone();
    assert_eq!(reader.tools().names(), vec!["read"]);
    let out = reader.call_tool("read", Default::default()).await;
    assert_eq!(out.content, "Not permitted: read is not permitted: denied by policy");
    // The explicit provider is shared down the tree.
    assert_eq!(reader.provider().name(), "fake");
    // Orchestration tools are never inherited.
    let inherit = lead.add_subagent_spec(SubAgentSpec::new("all", "Everything.", "")).unwrap();
    assert_eq!(inherit.tools().names(), vec!["read", "write"]);
}

#[tokio::test]
async fn spawns_a_specialist_written_by_the_factory() {
    let spec = json!({"name": "summariser", "description": "Summarises.", "instructions": "Summarise.", "tools": ["read", "not_allowed"], "tier": "fast", "max_steps": 3});
    // The lead, its factory and the specialist share one scripted provider.
    let p = fake(vec![
        tool_call("spawn_agent", json!({"task": "summarise the doc", "purpose": "One-line summaries."})),
        Script::text(format!("```json\n{spec}\n```")),
        Script::text("short summary"),
        Script::text("Here it is: short summary"),
    ]);
    let read = Tool::new("read", |_: NoArgs, _| async { Ok("doc") });
    let lead = Agent::builder("lead").provider(p.clone()).tool(read).runtime_agents(true).max_runtime_agents(1).build().unwrap();
    assert!(lead.tools().contains("delegate"));
    let r = lead.run("summarise").await.unwrap();
    assert_eq!(r.output, "Here it is: short summary");
    let child = &lead.subagents()["summariser"];
    assert_eq!(child.description(), "One-line summaries.");
    assert_eq!(child.tools().names(), vec!["read"]);
    assert_eq!(lead.total_spawned(), 1);
    assert_eq!(r.children[0].agent, "summariser");
    let err = Agent::builder("x").provider(fake(vec![])).runtime_agents(true).max_runtime_agents(0).build().unwrap_err();
    assert!(err.to_string().contains("max_runtime_agents is 0"));
}

#[tokio::test]
async fn hands_off_and_keeps_the_conversation() {
    let harness = Harness::new();
    let billing_p = fake(vec!["Refund issued.".into(), "Your invoice is attached.".into()]);
    let billing = Agent::builder("billing").description("Refunds and invoices.").provider(billing_p.clone()).harness(harness.clone()).build().unwrap();
    let triage_p = fake(vec![tool_call("handoff", json!({"agent_name": "billing", "reason": "a billing question"}))]);
    let triage = Agent::builder("triage").provider(triage_p.clone()).harness(harness.clone()).handoff(billing.clone()).build().unwrap();

    let events: Vec<_> = triage.stream("I was charged twice.", RunOptions::default()).await.unwrap().collect().await;
    let handoff = events.iter().find(|e| e.kind == StreamEventType::Handoff).unwrap();
    assert_eq!(handoff.data["to"], "billing");
    assert_eq!(events.iter().filter(|e| e.kind == StreamEventType::RunStart).count(), 1);
    let r: agent_harness::RunResult = serde_json::from_value(events.last().unwrap().data["result"].clone()).unwrap();
    assert_eq!(r.agent, "billing");
    assert_eq!(r.output, "Refund issued.");
    assert_eq!(r.handoffs.len(), 1);
    assert_eq!(r.handoffs[0].line(), "triage → billing: a billing question");
    assert_eq!(r.active_agent, "billing");
    assert_eq!(r.steps, 2);
    // Billing saw the whole conversation, ending on the handoff's result.
    let seen = &billing_p.requests()[0].messages;
    assert_eq!(seen[0].text(), "I was charged twice.");

    // The session says billing has it: the next turn goes straight there.
    let session = harness.sessions.load(&r.session_id).await.unwrap();
    assert_eq!(session.metadata["handoff"]["active"], "billing");
    let r2 = triage.run_with("And the invoice?", RunOptions::session(r.session_id.clone())).await.unwrap();
    assert_eq!(r2.agent, "billing");
    assert_eq!(r2.output, "Your invoice is attached.");
    assert_eq!(triage_p.requests().len(), 1, "triage was not asked again");
    assert_eq!(billing_p.requests()[1].messages.last().unwrap().text(), "And the invoice?");
}

#[tokio::test]
async fn handoff_history_views_and_refusals() {
    let target_p = fake(vec!["hello from support".into()]);
    let support = Agent::builder("support").provider(target_p.clone()).tool(Tool::new("noop", |_: NoArgs, _| async { Ok("") })).build().unwrap();
    let p = fake(vec![Script::Blocks(vec![
        ContentBlock::text("let me check"),
        ContentBlock::ToolUse(agent_harness::ToolUseBlock::new("handoff", json!({"agent_name": "support"}).as_object().unwrap().clone())),
    ])]);
    let front = Agent::builder("front").provider(p).handoff(Handoff::new(support).history(History::Fresh)).build().unwrap();
    front.run("my printer is on fire").await.unwrap();
    let seen = &target_p.requests()[0].messages;
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].text(), "my printer is on fire");
    assert!(seen[1].text().starts_with("[front handed this conversation to support. You are support"));

    // A refused handoff is an error the model reads; it answers itself.
    let mut hooks = agent_harness::runtime::HookEngine::new();
    hooks.on("handoff", |c| c.block("support is closed"));
    let p = fake(vec![tool_call("handoff", json!({"agent_name": "desk"})), "I'll help you myself.".into()]);
    let desk = Agent::builder("desk").provider(fake(vec![])).build().unwrap();
    let front = Agent::builder("front").provider(p.clone()).hooks(hooks).handoff(desk).build().unwrap();
    let r = front.run("help").await.unwrap();
    assert_eq!(r.agent, "front");
    assert_eq!(r.output, "I'll help you myself.");
    assert!(r.handoffs.is_empty());
    assert_eq!(last_result(&p, 1), "Error: the handoff to desk was not permitted: support is closed");

    let me = Agent::builder("me").provider(fake(vec![])).build().unwrap();
    assert!(me.add_handoff(me.clone()).unwrap_err().to_string().contains("cannot hand the conversation to itself"));
}

#[tokio::test]
async fn agents_that_hand_to_each_other() {
    let a = Agent::builder("a").provider(fake(vec![tool_call("handoff", json!({"agent_name": "b"}))]).clone()).build().unwrap();
    let b = Agent::builder("b").provider(Arc::new(FakeProvider::new(vec![tool_call("handoff", json!({"agent_name": "a"}))]).looping())).build().unwrap();
    a.add_handoff(b.clone()).unwrap();
    b.add_handoff(a.clone()).unwrap();
    assert_eq!(a.handoff_agent("b").unwrap().name(), "b");
    assert!(a.handoff_agent("zzz").is_none());
    let r = Agent::builder("entry")
        .provider(fake(vec![tool_call("handoff", json!({"agent_name": "a"}))]))
        .handoff(a.clone())
        .max_handoffs(2)
        .build()
        .unwrap()
        .run("go")
        .await
        .unwrap();
    // entry → a → b, then b's handoff back is over the limit: it must answer.
    assert_eq!(r.handoffs.len(), 2);
    assert_eq!(r.agent, "b");
}

#[tokio::test]
async fn versions_switch_by_name() {
    let p = fake(vec![Script::text("v2 says hi"), Script::text("v1 says hi")]);
    let order = Tool::new("order_status", |_: NoArgs, _| async { Ok("shipped") });
    let lookup = Tool::new("lookup", |_: NoArgs, _| async { Ok("found") });
    let agent = Agent::builder("support")
        .provider(p.clone())
        .tools([order, lookup])
        .version("v2")
        .add_version(
            "v1",
            AgentVersion { instructions: Some("Answer order questions.".into()), tools: Some(vec!["order_status".into()]), ..Default::default() },
        )
        .add_version("v2", AgentVersion { instructions: Some("Cite the order.".into()), model: Some("claude-opus-5".into()), ..Default::default() })
        .build()
        .unwrap();
    assert_eq!(agent.version(), "v2");
    assert_eq!(agent.model(), "claude-opus-5");
    assert_eq!(agent.available_versions(), vec!["v1", "v2"]);
    agent.run("hi").await.unwrap();
    let v1 = agent.use_version("v1").unwrap();
    assert_eq!(v1.tools().names(), vec!["order_status", "recall", "remember"]);
    assert_eq!(v1.model(), "claude-sonnet-5");
    v1.run("hi").await.unwrap();
    assert!(p.requests()[0].system.as_deref().unwrap().contains("Cite the order."));
    assert!(p.requests()[1].system.as_deref().unwrap().contains("Answer order questions."));
    let again = agent.use_version("v1").unwrap();
    assert_eq!(again.name(), v1.name());
    assert!(agent.use_version("v9").is_err());
    let bad = Agent::builder("x").provider(fake(vec![])).version("v3").add_version("v1", AgentVersion::default()).build().unwrap_err();
    assert_eq!(bad.to_string(), "x: version 'v3' is not defined; known: v1");
}

#[tokio::test]
async fn resume_follows_a_conversation() {
    let harness = Harness::new();
    let p = fake(vec!["first".into(), "second".into(), "third".into()]);
    let a = Agent::builder("chat").provider(p.clone()).harness(harness.clone()).build().unwrap();
    let r1 = a.run("one").await.unwrap();
    a.resume(&r1.session_id).await.unwrap();
    let r2 = a.run("two").await.unwrap();
    assert_eq!(r2.session_id, r1.session_id);
    assert_eq!(p.requests()[1].messages.len(), 3);
    a.run("three").await.unwrap();
    assert_eq!(p.requests()[2].messages.len(), 5);
    assert_eq!(a.thread().unwrap().messages.len(), 6);
    a.new_session();
    assert!(a.thread().is_none());
    let held = harness.sessions.load(&r1.session_id).await.unwrap();
    assert_eq!(held.messages.iter().filter(|m| m.role == Role::User).count(), 3);
    let _ = (StopReason::Handoff, Permission::Allow);
}
