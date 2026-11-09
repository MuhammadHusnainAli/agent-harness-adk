//! Declared workflows, end to end.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use agent_harness::llm_providers::{FakeProvider, Script};
use agent_harness::workflow::{StepStatus, Workflow, WorkflowStatus};
use agent_harness::{Agent, Harness, Tool};
use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

/// Find the charges on an order.
#[derive(Deserialize, JsonSchema)]
struct Order {
    order: String,
}

/// Refund an order.
#[derive(Deserialize, JsonSchema)]
struct Refund {
    order: String,
    amount: f64,
}

fn find_charges() -> Tool {
    Tool::new("find_charges", |a: Order, _| async move { Ok(json!({"order": a.order, "charges": [40.0, 40.0]})) })
}

fn issue_refund() -> Tool {
    Tool::new("issue_refund", |a: Refund, _| async move { Ok(format!("refunded {} on {}", a.amount, a.order)) })
}

fn agent(name: &str, replies: Vec<Script>) -> (Agent, Arc<FakeProvider>) {
    let p = Arc::new(FakeProvider::new(replies));
    (Agent::builder(name).provider(p.clone()).build().unwrap(), p)
}

const REFUNDS: &str = r#"
name: refunds
inputs:
  order: {required: true}
  message: {required: true}
steps:
  - id: charges
    tool: find_charges
    args: {order: "{{ inputs.order }}"}
  - id: classify
    agent: classifier
    input: "Charges: {{ steps.charges.output }}\nCustomer: {{ inputs.message }}"
    parse: json
    save: verdict
  - if: "state.verdict.duplicate"
    then:
      - tool: issue_refund
        args: {order: "{{ inputs.order }}", amount: "{{ state.verdict.amount }}"}
    else:
      - agent: support
        input: "Explain why order {{ inputs.order }} is not refunded."
output: "{{ previous }}"
"#;

#[tokio::test]
async fn the_refunds_example() {
    let (classifier, cp) = agent(
        "classifier",
        vec![
            r#"```json
{"duplicate": true, "amount": 40.0}
```"#
                .into(),
        ],
    );
    let (support, sp) = agent("support", vec![]);
    let wf = Workflow::from_text(REFUNDS, "yaml").unwrap().agents([classifier, support]).tools([find_charges(), issue_refund()]).build().unwrap();
    let r = wf.run(json!({"order": "4182", "message": "I was charged twice."})).await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert_eq!(r.output, json!("refunded 40 on 4182"));
    assert_eq!(r.state["verdict"], json!({"duplicate": true, "amount": 40.0}));
    assert_eq!(r.steps["classify"].data["amount"], 40.0);
    assert_eq!(r.steps["if_1"].branch, "then");
    assert_eq!(r.steps_run, 4);
    let asked = cp.requests()[0].messages[0].text();
    assert!(asked.starts_with("Charges: {\"order\":\"4182\",\"charges\":[40.0,40.0]}\nCustomer: I was charged twice."), "{asked}");
    assert!(sp.requests().is_empty());
    assert!(wf.describe().contains("- if_1: if state.verdict.duplicate"), "{}", wf.describe());

    let missing = wf.run(json!({"order": "1"})).await.unwrap_err();
    assert_eq!(missing.to_string(), "workflow 'refunds' needs the input 'message'");
    let extra = wf.run(json!({"order": "1", "message": "x", "other": 1})).await.unwrap_err();
    assert!(extra.to_string().starts_with("workflow 'refunds' does not take other"));
}

#[tokio::test]
async fn control_flow() {
    let spec = json!({
        "name": "flow",
        "state": {"n": 0},
        "steps": [
            {"id": "count", "loop": {"max": 10, "until": "state.n >= 3"}, "steps": [{"set": {"n": "{{ state.n + 1 }}"}}]},
            {"id": "each", "foreach": "{{ [1, 2, 3] }}", "as": "x", "concurrency": 2, "steps": [{"set": {"last": "{{ x * 10 }}"}}, {"return_like": null}]},
        ]
    });
    // An unknown key is refused, as Python's extra="forbid".
    assert!(Workflow::from_value(spec).is_err());

    let spec = json!({
        "name": "flow",
        "state": {"n": 0},
        "steps": [
            {"id": "count", "loop": {"max": 10, "until": "state.n >= 3"}, "steps": [{"set": {"n": "{{ state.n + 1 }}"}}]},
            {"id": "doubled", "foreach": "{{ [1, 2, 3] }}", "as": "x", "concurrency": 2, "steps": [{"set": {"seen": "{{ x }}"}}]},
            {"id": "pick", "switch": [
                {"when": "state.n > 5", "steps": [{"set": {"size": "big"}}]},
                {"steps": [{"set": {"size": "small"}}]}
            ]},
            {"id": "skipped", "when": "state.n > 100", "set": {"never": true}},
            {"id": "both", "parallel": [{"id": "a", "set": {"a": 1}}, {"id": "b", "set": {"b": 2}}]},
            {"id": "early", "return": "{{ state.size + ':' + str(state.n) }}"},
            {"id": "after", "set": {"unreachable": true}},
        ]
    });
    let wf = Workflow::from_value(spec).unwrap().build().unwrap();
    let r = wf.run(Value::Null).await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert_eq!(r.output, json!("small:3"));
    assert_eq!(r.steps["count"].iterations, 3);
    assert!(!r.steps["count"].exhausted);
    assert_eq!(r.steps["doubled"].iterations, 3);
    assert_eq!(r.steps["pick"].branch, "default");
    assert_eq!(r.steps["skipped"].status, StepStatus::Skipped);
    assert_eq!(r.steps["both"].output, json!({"a": {"a": 1}, "b": {"b": 2}}));
    assert!(!r.steps.contains_key("after"));
    assert!(r.state.get("unreachable").is_none());
}

#[tokio::test]
async fn graphs_run_as_soon_as_they_can() {
    let spec = json!({
        "name": "g",
        "graph": [
            {"id": "a", "set": {"a": 1}},
            {"id": "b", "set": {"b": 2}},
            {"id": "c", "needs": ["a", "b"], "return_value_is": null}
        ]
    });
    assert!(Workflow::from_value(spec).is_err());
    let spec = json!({
        "name": "g",
        "graph": [
            {"id": "a", "set": {"a": 1}},
            {"id": "broken", "fail": "nope", "on_error": "continue"},
            {"id": "joined", "needs": ["a", "broken"], "set": {"seen": "{{ previous }}"}},
            {"id": "either", "needs": ["a", "broken"], "join": "any", "set": {"seen_any": "{{ previous }}"}}
        ]
    });
    let r = Workflow::from_value(spec).unwrap().build().unwrap().run(Value::Null).await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert_eq!(r.steps["broken"].status, StepStatus::Failed);
    assert_eq!(r.steps["joined"].status, StepStatus::Skipped, "a need that failed skips an `all` node");
    assert_eq!(r.state["seen_any"], json!({"a": {"a": 1}}));
    let cyclic = json!({"name": "g", "graph": [{"id": "x", "needs": ["y"], "set": {}}, {"id": "y", "needs": ["x"], "set": {}}]});
    assert!(Workflow::from_value(cyclic).unwrap().build().unwrap_err().to_string().starts_with("these nodes wait on each other and can never start: x, y"));
}

#[tokio::test]
async fn retries_failures_and_timeouts() {
    let tries = Arc::new(AtomicU32::new(0));
    let seen = tries.clone();
    let flaky = Tool::new("flaky", move |_: agent_harness::NoArgs, _| {
        let seen = seen.clone();
        async move { if seen.fetch_add(1, Ordering::SeqCst) < 2 { Err(agent_harness::Error::config("not yet")) } else { Ok("finally") } }
    });
    let spec = json!({"name": "r", "steps": [{"id": "try", "tool": "flaky", "retry": 2}]});
    let r = Workflow::from_value(spec).unwrap().tools([flaky]).build().unwrap().run(Value::Null).await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert_eq!(r.steps["try"].attempts, 3);
    assert_eq!(r.output, json!("finally"));

    let spec = json!({"name": "f", "steps": [{"id": "outer", "steps": [{"id": "inner", "fail": "boom {{ 1 + 1 }}"}]}]});
    let r = Workflow::from_value(spec).unwrap().build().unwrap().run(Value::Null).await.unwrap();
    assert_eq!(r.status, WorkflowStatus::Failed);
    assert_eq!(r.error.as_deref(), Some("step 'inner' failed: boom 2"));
    assert_eq!(r.failed_step, "inner");

    let spec = json!({"name": "t", "steps": [{"id": "slow", "wait": 5, "timeout": 0.05}]});
    let r = Workflow::from_value(spec).unwrap().build().unwrap().run(Value::Null).await.unwrap();
    assert_eq!(r.error.as_deref(), Some("step 'slow' failed: step 'slow' timed out after 0.05s"));

    let spec = json!({"name": "loopy", "max_steps": 5, "steps": [{"loop": 100, "steps": [{"set": {}}]}]});
    let r = Workflow::from_value(spec).unwrap().build().unwrap().run(Value::Null).await.unwrap();
    assert!(r.error.unwrap().contains("ran 5 steps without finishing"));
}

#[test]
fn validation_catches_mistakes() {
    let bad = |spec: Value| Workflow::from_value(spec).and_then(|b| b.build()).unwrap_err().to_string();
    assert!(bad(json!({"name": "x"})).contains("needs `steps` or `graph`"));
    assert!(bad(json!({"steps": [{"id": "a"}]})).starts_with("step 'a' does nothing"));
    assert!(bad(json!({"steps": [{"id": "a", "set": {}, "fail": "x"}]})).starts_with("step 'a' is set and fail at once"));
    assert!(bad(json!({"steps": [{"id": "a", "set": {}}, {"id": "a", "set": {}}]})).starts_with("two steps are called 'a'"));
    assert!(bad(json!({"steps": [{"id": "a", "when": "steps.zzz.output", "set": {}}]})).contains("refers to steps.zzz, and no step is called that"));
    assert!(
        bad(json!({"inputs": {"order": null}, "steps": [{"id": "a", "set": {"x": "{{ inputs.ordr }}"}}]}))
            .contains("refers to inputs.ordr, which the workflow does not declare")
    );
    assert!(bad(json!({"steps": [{"id": "a", "set": {"x": "{{ mystery }}"}}]})).contains("uses 'mystery', which is not known here"));
    assert!(bad(json!({"steps": [{"id": "a", "tool": "missing"}]})).contains("no tool named 'missing'"));
    assert!(bad(json!({"steps": [{"id": "a", "agent": "ghost"}]})).contains("no agent named 'ghost'"));
    assert!(bad(json!({"steps": [{"id": "a", "set": {}, "needs": ["b"]}]})).contains("only a node of a `graph` can have"));
    assert!(bad(json!({"steps": [{"id": "a", "if": "true"}]})).contains("an `if` needs a `then`"));
    assert!(bad(json!({"steps": [{"id": "bad id", "set": {}}]})).contains("must be a plain name"));
}

#[tokio::test]
async fn declared_agents_threads_and_tools() {
    let harness = Harness::testing(Some(Arc::new(FakeProvider::new(vec![Script::text("first"), Script::text("second")]))));
    let spec = json!({
        "name": "chat",
        "agents": {"writer": {"instructions": "Write."}},
        "steps": [
            {"id": "one", "agent": "writer", "input": "draft it", "thread": true},
            {"id": "two", "agent": "writer", "input": "now shorten it", "thread": true},
            {"id": "inline", "agent": {"instructions": "Inline."}, "input": "hi"}
        ]
    });
    let wf = Workflow::from_value(spec).unwrap().harness(harness.clone()).build().unwrap();
    let r = wf.run(Value::Null).await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert_eq!(r.steps["two"].output, json!("second"));
    assert_eq!(r.steps["inline"].agent, "inline");

    let tool = wf.as_tool(Some("chatter"), None);
    assert_eq!(tool.parameters["required"], json!(["input"]));
    let events: Vec<_> = wf.stream(Value::Null).collect().await;
    let kinds: Vec<String> = events.iter().map(|e| e.as_ref().unwrap().event.clone()).collect();
    assert_eq!(kinds.first().map(String::as_str), Some("workflow_start"));
    assert_eq!(kinds.last().map(String::as_str), Some("workflow_end"));
    assert_eq!(kinds.iter().filter(|k| *k == "step_start").count(), 3);
}

#[tokio::test]
async fn one_input_can_be_handed_over_bare() {
    let spec = json!({"name": "echo", "inputs": {"text": {"type": "string"}}, "steps": [{"id": "s", "set": {"got": "{{ inputs.text }}"}}], "output": "{{ state.got }}"});
    let wf = Workflow::from_value(spec).unwrap().build().unwrap();
    assert_eq!(wf.run(json!("hello")).await.unwrap().output, json!("hello"));
    assert!(wf.run(json!({"text": 5})).await.unwrap_err().to_string().contains("input 'text' must be string — got int"));
}
