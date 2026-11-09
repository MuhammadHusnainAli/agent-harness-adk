//! The rails, end to end: approvals that outlive the run, the audit trail,
//! traces, the result cache, the stop switch, checkpoints and time travel.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use agent_harness::llm_providers::{FakeProvider, Script, tool_call};
use agent_harness::runtime::{Approvals, Checkpointer};
use agent_harness::types::{ContentBlock, StopReason, StreamEventType};
use agent_harness::{Agent, Harness, Permission, RunOptions, SubAgentSpec, Tool};
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn fake(script: Vec<Script>) -> Arc<FakeProvider> {
    Arc::new(FakeProvider::new(script))
}

/// Refund an order.
#[derive(Deserialize, JsonSchema)]
struct Refund {
    order: String,
}

/// Look an order up.
#[derive(Deserialize, JsonSchema)]
struct Lookup {
    order: String,
}

fn tools(refunds: Arc<AtomicU32>, lookups: Arc<AtomicU32>) -> Vec<Tool> {
    let refund = Tool::new("refund", move |a: Refund, _| {
        let n = refunds.clone();
        async move {
            n.fetch_add(1, Ordering::SeqCst);
            Ok(format!("refunded {}", a.order))
        }
    })
    .permission(Permission::Ask);
    let lookup = Tool::new("lookup", move |a: Lookup, _| {
        let n = lookups.clone();
        async move {
            n.fetch_add(1, Ordering::SeqCst);
            Ok(format!("order {} paid twice", a.order))
        }
    })
    .cacheable(true);
    vec![refund, lookup]
}

fn both_at_once() -> Script {
    Script::Blocks(vec![
        ContentBlock::ToolUse(agent_harness::types::ToolUseBlock {
            id: "c_look".into(),
            name: "lookup".into(),
            input: json!({"order": "4182"}).as_object().cloned().unwrap(),
            thought_signature: None,
        }),
        ContentBlock::ToolUse(agent_harness::types::ToolUseBlock {
            id: "c_refund".into(),
            name: "refund".into(),
            input: json!({"order": "4182"}).as_object().cloned().unwrap(),
            thought_signature: None,
        }),
    ])
}

fn last_result(p: &FakeProvider, request: usize, at: usize) -> String {
    match &p.requests()[request].messages.last().unwrap().content[at] {
        ContentBlock::ToolResult(r) => r.content.clone(),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_run_stops_for_approval_and_resumes_from_that_step() {
    let (refunds, lookups) = (Arc::new(AtomicU32::new(0)), Arc::new(AtomicU32::new(0)));
    let p = fake(vec![both_at_once(), "Refunded.".into()]);
    let harness = Harness::testing(Some(p.clone())).with_approvals(Approvals::default());
    let a = Agent::builder("support").harness(harness.clone()).memory(false).tools(tools(refunds.clone(), lookups.clone())).build().unwrap();

    let events: Vec<_> = a.stream("Refund order 4182.", RunOptions::default()).await.unwrap().collect().await;
    let asked = events.iter().find(|e| e.kind == StreamEventType::ApprovalRequired).expect("an approval_required event");
    assert_eq!(asked.text, "support wants to run refund(order=\"4182\")");
    let end = events.last().unwrap();
    let r: agent_harness::types::RunResult = serde_json::from_value(end.data["result"].clone()).unwrap();
    assert_eq!(r.stop_reason, StopReason::Approval);
    assert_eq!(r.output, "Waiting for approval: support wants to run refund(order=\"4182\").");
    let id = r.approval.as_ref().unwrap()["id"].as_str().unwrap().to_string();
    assert_eq!((refunds.load(Ordering::SeqCst), lookups.load(Ordering::SeqCst)), (0, 1), "nothing is refunded before a person says yes");

    let desk = harness.approvals.clone().unwrap();
    assert!(a.resume_approval(&id).await.unwrap_err().to_string().contains("still waiting for an answer"));
    desk.approve(&id, "maria", "checked the order", None).await.unwrap();
    let done = a.resume_approval(&id).await.unwrap();
    assert!(done.ok(), "{:?}", done.error);
    assert_eq!(done.output, "Refunded.");
    assert_eq!((refunds.load(Ordering::SeqCst), lookups.load(Ordering::SeqCst)), (1, 1), "the lookup that already ran is not run again");
    assert_eq!(p.requests().len(), 2, "the model is not asked again for the approved call");
    assert_eq!(last_result(&p, 1, 0), "order 4182 paid twice");
    assert_eq!(last_result(&p, 1, 1), "refunded 4182");
    assert!(done.usage.calls >= 2, "the whole run is counted");
    assert_eq!(desk.get(&id).await.unwrap().status, "resumed");
    assert!(a.resume_approval(&id).await.unwrap_err().to_string().contains("was already resumed"));

    let allowed: Vec<_> = harness.audit.entries().into_iter().filter(|e| e.action == "tool_call" && e.target == "refund").collect();
    assert_eq!(allowed.iter().map(|e| e.decision.as_str()).collect::<Vec<_>>(), vec!["pending", "allow"]);
    assert_eq!(allowed[1].detail["approved_by"], "maria");
    assert!(harness.audit.verify().0);
}

#[tokio::test]
async fn a_declined_call_is_answered_with_the_reason() {
    let refunds = Arc::new(AtomicU32::new(0));
    let p = fake(vec![tool_call("refund", json!({"order": "9"})), "I could not refund it.".into()]);
    let harness = Harness::testing(Some(p.clone())).with_session_approvals();
    let a = Agent::builder("support").harness(harness.clone()).memory(false).tools(tools(refunds.clone(), Arc::default())).build().unwrap();
    let r = a.run("Refund order 9.").await.unwrap();
    let id = r.approval.unwrap()["id"].as_str().unwrap().to_string();
    harness.approvals.clone().unwrap().deny(&id, "maria", "over the limit", None).await.unwrap();
    let done = a.resume_approval(&id).await.unwrap();
    assert_eq!(done.output, "I could not refund it.");
    assert_eq!(last_result(&p, 1, 0), "Not permitted: declined by maria: over the limit");
    assert_eq!(refunds.load(Ordering::SeqCst), 0);
    // The session the run paused in holds the whole conversation once.
    let held = harness.sessions.load(&done.session_id).await.unwrap();
    assert_eq!(held.messages.len(), 4);
}

#[tokio::test]
async fn without_a_desk_an_ask_is_refused() {
    let p = fake(vec![tool_call("refund", json!({"order": "9"})), "ok".into()]);
    let a = Agent::builder("support").provider(p.clone()).memory(false).tools(tools(Arc::default(), Arc::default())).build().unwrap();
    let r = a.run("Refund order 9.").await.unwrap();
    assert_eq!(r.stop_reason, StopReason::EndTurn);
    assert!(last_result(&p, 1, 0).starts_with("Not permitted:"), "{}", last_result(&p, 1, 0));
    assert_eq!(a.harness().audit.denials().len(), 1);
}

#[tokio::test]
async fn traces_nest_and_results_are_cached() {
    let lookups = Arc::new(AtomicU32::new(0));
    let p = fake(vec![
        tool_call("lookup", json!({"order": "1"})),
        tool_call("delegate", json!({"agent_name": "helper", "task": "look again"})),
        tool_call("lookup", json!({"order": "1"})),
        "helper done".into(),
        "all done".into(),
    ]);
    let a = Agent::builder("lead")
        .provider(p.clone())
        .memory(false)
        .tools(tools(Arc::default(), lookups.clone()))
        .subagent_spec(SubAgentSpec::new("helper", "Helps.", "Help."))
        .build()
        .unwrap();
    let r = a.run("go").await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert_eq!(lookups.load(Ordering::SeqCst), 1, "the helper's identical lookup was served from the cache");
    let h = a.harness();
    assert_eq!(h.cache.stats()["hits"], 1);
    let tree = h.tracer.tree();
    assert!(tree.starts_with("run:agent:lead"), "{tree}");
    assert!(
        tree.contains("\n  tool:tool:lookup")
            && tree.contains("\n  tool:tool:delegate")
            && tree.contains("\n    subagent:subagent:helper")
            && tree.contains("\n      run:agent:helper"),
        "{tree}"
    );
    assert!(h.tracer.spans().iter().all(|s| s.trace_id == r.trace_id));
    assert!(h.journal.render(None).contains("lead assignment: go"));
    assert!(h.health.component("lookup", "tool").calls == 1);
}

#[tokio::test]
async fn the_stop_switch_ends_runs_at_the_next_step() {
    let p = fake(vec!["never".into()]);
    let a = Agent::builder("w").provider(p.clone()).memory(false).build().unwrap();
    a.harness().stop("customer withdrew", "drain", "ops");
    let r = a.run("go").await.unwrap();
    assert_eq!(r.stop_reason, StopReason::Stopped);
    assert_eq!(r.error.as_deref(), Some("StopRequested: run stopped by ops at w step 1: customer withdrew"));
    assert!(p.requests().is_empty());
    assert_eq!(a.harness().audit.entries()[0].action, "stop");
}

#[tokio::test]
async fn checkpoints_let_a_run_be_retried_from_a_step() {
    let p = fake(vec![tool_call("lookup", json!({"order": "1"})), "first answer".into(), "second answer".into()]);
    let harness = Harness::testing(Some(p.clone())).with_checkpoints(Checkpointer::new(1));
    let a = Agent::builder("w").harness(harness.clone()).memory(false).tools(tools(Arc::default(), Arc::default())).build().unwrap();
    let r = a.run("check order 1").await.unwrap();
    let timeline = harness.replayer().timeline(&r.run_id);
    assert_eq!(timeline.len(), 2);
    assert_eq!(timeline[0]["tools_called"], json!(["lookup"]));
    let again = harness.replayer().resume(&a, &r.run_id, Some(1), Some("check it once more")).await.unwrap();
    assert_eq!(again.output, "second answer");
    let sent = &p.requests()[2].messages;
    assert_eq!(sent.last().unwrap().text(), "check it once more");
    assert_eq!(sent.len(), 2, "the step-1 history without its unanswered tool call, then the new task");
}

#[tokio::test]
async fn a_local_harness_keeps_its_records() {
    let dir = std::env::temp_dir().join(format!("ah_rails_{}", uuid::Uuid::new_v4().simple()));
    let p = fake(vec![tool_call("refund", json!({"order": "7"})), "done".into()]);
    let a = Agent::builder("w").provider(p).harness(Harness::local(&dir).unwrap()).memory(false).tools(tools(Arc::default(), Arc::default())).build().unwrap();
    let r = a.run("refund 7").await.unwrap();
    assert_eq!(r.stop_reason, StopReason::Approval, "local() keeps an approvals desk on disk");
    let id = r.approval.unwrap()["id"].as_str().unwrap().to_string();
    // Another process: a fresh harness on the same folder.
    let later = Harness::local(&dir).unwrap();
    later.approvals.clone().unwrap().approve(&id, "maria", "", None).await.unwrap();
    let b = Agent::builder("w")
        .provider(fake(vec!["refunded".into()]))
        .harness(later.clone())
        .memory(false)
        .tools(tools(Arc::default(), Arc::default()))
        .build()
        .unwrap();
    assert_eq!(b.resume_approval(&id).await.unwrap().output, "refunded");
    for file in ["audit.jsonl", "journal.jsonl", "traces.jsonl"] {
        assert!(std::fs::metadata(dir.join(file)).is_ok_and(|m| m.len() > 0), "{file}");
    }
    assert!(agent_harness::runtime::AuditTrail::at(dir.join("audit.jsonl")).verify().0);
    let traced: Vec<Value> = std::fs::read_to_string(dir.join("traces.jsonl")).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert!(traced.iter().any(|s| s["kind"] == "run"));
    let _ = std::fs::remove_dir_all(dir);
}

// ---- guardrails ----------------------------------------------------------------

#[tokio::test]
async fn an_unmet_requirement_sends_the_agent_back_round() {
    use agent_harness::guardrails::AgentGuardrails;
    let lookups = Arc::new(AtomicU32::new(0));
    let p = fake(vec!["It was paid twice.".into(), tool_call("lookup", json!({"order": "1"})), "Order 1 was paid twice.".into()]);
    let rails = AgentGuardrails::from_value(&json!({"require_tools": ["lookup"]})).unwrap();
    let a = Agent::builder("w").provider(p.clone()).memory(false).tools(tools(Arc::default(), lookups.clone())).guardrails(rails).build().unwrap();
    let r = a.run("was order 1 paid twice?").await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert_eq!(r.output, "Order 1 was paid twice.");
    assert_eq!(lookups.load(Ordering::SeqCst), 1);
    let nudge = p.requests()[1].messages.last().unwrap().text();
    assert!(nudge.contains("lookup"), "{nudge}");
    assert!(r.violations.is_empty(), "a successful retry clears what the first attempt failed on");
    assert!(a.harness().journal.render(None).contains("guardrail:"));
}

#[tokio::test]
async fn a_forbidden_tool_never_runs_and_failing_checks_fail_the_run() {
    use agent_harness::guardrails::AgentGuardrails;
    let refunds = Arc::new(AtomicU32::new(0));
    let p = fake(vec![tool_call("refund", json!({"order": "1"})), "done".into()]);
    let rails = AgentGuardrails::from_value(&json!({"forbid_tools": ["refund"], "must_include": ["order"], "on_violation": "fail"})).unwrap();
    let a = Agent::builder("w").provider(p.clone()).memory(false).tools(tools(refunds.clone(), Arc::default())).guardrails(rails).build().unwrap();
    let r = a.run("refund it").await.unwrap();
    assert_eq!(refunds.load(Ordering::SeqCst), 0);
    assert!(last_result(&p, 1, 0).starts_with("Not permitted:"), "{}", last_result(&p, 1, 0));
    let error = r.error.unwrap();
    assert!(error.starts_with("GuardrailTripped: the answer did not meet this agent's guardrails:"), "{error}");
}

#[tokio::test]
async fn secrets_are_kept_out_of_what_goes_in_and_comes_out() {
    let key = format!("sk-ant-{}", "a".repeat(40));
    let p = fake(vec![Script::text(format!("your key is {key}"))]);
    let a = Agent::builder("w").provider(p.clone()).memory(false).build().unwrap();
    let r = a.run(format!("store {key} for me").as_str()).await.unwrap();
    assert!(!p.requests()[0].messages[0].text().contains(&key), "the task reached the model with the key in it");
    assert!(!r.output.contains(&key), "{}", r.output);
}

#[tokio::test]
async fn blueprints_name_a_guardrail_set() {
    let b = agent_harness::blueprint::Blueprint::from_text(
        "guardrails:\n  strict: {must_include: [\"order\"], on_violation: fail}\nagents:\n  support:\n    instructions: Help.\n    guardrails: strict\n  loose:\n    instructions: Help.\n    guardrails: missing\n",
        "yaml",
    )
    .unwrap();
    let h = Harness::testing(Some(fake(vec!["hello".into()])));
    let support = b.build("support", &[], Some(h.clone())).unwrap();
    assert!(support.guardrails().is_some_and(|g| g.has_checks()));
    assert!(support.run("hi").await.unwrap().error.unwrap().contains("did not meet this agent's guardrails"));
    assert!(b.build("loose", &[], Some(h)).unwrap_err().to_string().contains("guardrails 'missing' are not defined; known: strict"));
}

// ---- governance ----------------------------------------------------------------

#[tokio::test]
async fn a_governed_harness_registers_agents_and_answers_for_their_runs() {
    use agent_harness::governance::{Governance, GovernanceOptions};
    let gov = Governance::from_packs(&["gdpr"], GovernanceOptions::default()).unwrap();
    let harness = Harness::testing(Some(fake(vec!["Your order shipped.".into()]))).with_governance(gov.clone()).unwrap();
    let a = Agent::builder("support")
        .harness(harness.clone())
        .memory(false)
        .identity(json!({"owner": "ops@acme.com", "purpose": "customer_support"}))
        .trace("alice")
        .build()
        .unwrap();
    let r = a.run("Where is my order?").await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert!(r.extra.contains_key("governance"), "governance answers for the run: {:?}", r.extra.keys().collect::<Vec<_>>());
    let actions: Vec<String> = harness.audit.entries().iter().map(|e| e.action.clone()).collect();
    assert!(actions.iter().any(|a| a == "governance.attach"), "{actions:?}");
    assert!(harness.audit.verify().0);
    assert!(harness.report().await["governance"].is_object());
    assert!(!gov.report("gdpr").markdown().is_empty());
    assert!(Harness::testing(None).with_governance(gov).is_err(), "one governance governs one harness");
}
