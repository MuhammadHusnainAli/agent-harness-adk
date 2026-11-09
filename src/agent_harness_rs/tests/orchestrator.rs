//! The orchestrator, end to end, with one scripted provider playing every role.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use agent_harness::llm_providers::base::CompletionRequest;
use agent_harness::llm_providers::{FakeProvider, Script};
use agent_harness::orchestrator::{Orchestrator, Plan, TaskStatus};
use agent_harness::runtime::Budget;
use agent_harness::{Error, Harness};
use serde_json::Value;

/// Who is being asked, from the system prompt.
fn role(req: &CompletionRequest) -> &'static str {
    let system = req.system.as_deref().unwrap_or("");
    for (marker, role) in [
        ("You are lead.planner", "planner"),
        ("You are lead.replanner", "replanner"),
        ("You are lead.checker", "checker"),
        ("You are lead.reviewer", "reviewer"),
        ("You are lead.factory", "factory"),
        ("You are lead,", "manager"),
    ] {
        if system.contains(marker) {
            return role;
        }
    }
    "worker"
}

fn last_text(req: &CompletionRequest) -> String {
    req.messages.last().map(|m| m.text()).unwrap_or_default()
}

type Responder = Arc<dyn Fn(&str, &CompletionRequest) -> String + Send + Sync>;

fn scripted(respond: Responder) -> (Arc<FakeProvider>, Harness) {
    let p = Arc::new(FakeProvider::new(vec![Script::func(move |req| Script::text(respond(role(req), req)))]).looping());
    (p.clone(), Harness::testing(Some(p)))
}

const PLAN: &str = r#"{"goal": "Assess the vendors", "definition_of_done": ["Each vendor is covered", "One is recommended"],
 "tasks": [
   {"id": "t1", "statement": "Research the vendor SLAs and sources", "done_when": "lists every SLA"},
   {"id": "t2", "statement": "Frobnicate the widget quotes", "done_when": ""},
   {"id": "t3", "statement": "Validate and check the comparison numbers", "depends_on": ["t1", "t2"], "done_when": "every number checked"}
 ]}"#;

#[tokio::test]
async fn a_whole_job() {
    let respond: Responder = Arc::new(|role, req| {
        match role {
        "planner" => PLAN.into(),
        "factory" => r#"{"name": "widget_frobnicator", "description": "Frobnicates quotes.", "instructions": "Frobnicate.", "tools": [], "tier": "fast", "max_steps": 3}"#.into(),
        "checker" => r#"{"met": true}"#.into(),
        "reviewer" => r#"{"accepted": true, "score": 0.9, "summary": "covers everything"}"#.into(),
        "manager" => "Vendor B is best.".into(),
        _ => {
            let task = last_text(req);
            if task.contains("## What earlier tasks produced") { "checked: all fine".into() } else { format!("done: {}", task.lines().nth(1).unwrap_or("")) }
        }
    }
    });
    let (_p, harness) = scripted(respond);
    let boss = Orchestrator::builder("lead").harness(harness.clone()).max_concurrency(2).build().unwrap();
    let plan = boss.plan("Compare the vendors.").await.unwrap();
    assert_eq!(plan.tasks.len(), 3);
    assert_eq!(plan.parallelism, 2);
    assert!(plan.estimate_usd > 0.0);

    let r = boss.run("Compare the vendors.").await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    assert_eq!(r.output, "Vendor B is best.");
    let plan: Plan = serde_json::from_value(r.data["plan"].clone()).unwrap();
    assert!(plan.tasks.iter().all(|t| t.status == TaskStatus::Done), "{:#?}", plan.tasks);
    let staffed: Vec<(&str, bool)> = plan.tasks.iter().map(|t| (t.agent.as_str(), t.reused)).collect();
    assert_eq!(staffed, vec![("research", true), ("widget_frobnicator", false), ("validator", true)]);
    assert_eq!(plan.tasks[0].checked, Some(true));
    assert_eq!(plan.tasks[1].checked, None, "nothing to check it against");
    assert_eq!(r.data["review"]["accepted"], true);
    assert_eq!(r.children.len(), 3);
    assert!(r.spend["total_usd"].as_f64().unwrap() >= r.spend["work_usd"].as_f64().unwrap());
    assert!(r.spend["by_agent"].as_object().unwrap().contains_key("lead.planner"));
    let names: Vec<&str> = r.artifacts.iter().map(|a| a.name.as_str()).collect();
    assert!(names.contains(&"plan.json") && names.contains(&"deliverable.md"));

    // The job was written down as it went, and a finished job resumes to its answer.
    let job = boss.job(&r.run_id).await.unwrap();
    assert_eq!(job["status"], "done");
    assert_eq!(job["stage"], "finished");
    let again = boss.resume(&r.run_id).await.unwrap();
    assert_eq!(again.output, "Vendor B is best.");
    assert!(harness.sessions.load(&format!("orchestrator_{}", r.run_id)).await.is_ok());
}

#[tokio::test]
async fn short_hand_backs_are_replanned() {
    let checks = Arc::new(AtomicU32::new(0));
    let seen = checks.clone();
    let briefs = Arc::new(Mutex::new(Vec::new()));
    let kept = briefs.clone();
    let respond: Responder = Arc::new(move |role, req| match role {
        "planner" => r#"{"goal": "g", "tasks": [{"id": "t1", "statement": "Research the market sources", "done_when": "names three vendors"}]}"#.into(),
        "checker" => {
            // The first task never meets its bar; its replacement does.
            if last_text(req).contains("TASK\nResearch the market sources") {
                seen.fetch_add(1, Ordering::SeqCst);
                r#"{"met": false, "missing": "only two vendors named"}"#.into()
            } else {
                r#"{"met": true}"#.into()
            }
        }
        "replanner" => {
            r#"{"tasks": [{"id": "a", "statement": "Research vendor one"}, {"id": "b", "statement": "Research vendor two", "depends_on": ["a"]}]}"#.into()
        }
        "reviewer" => r#"{"accepted": true, "score": 1}"#.into(),
        "manager" => "three vendors".into(),
        _ => {
            kept.lock().unwrap().push(last_text(req));
            "vendors A and B".into()
        }
    });
    let (_p, harness) = scripted(respond);
    let boss = Orchestrator::builder("lead").harness(harness).build().unwrap();
    let r = boss.run("Find vendors.").await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    let plan: Plan = serde_json::from_value(r.data["plan"].clone()).unwrap();
    let statuses: Vec<(&str, TaskStatus)> = plan.tasks.iter().map(|t| (t.id.as_str(), t.status)).collect();
    assert_eq!(statuses, vec![("t1", TaskStatus::Replaced), ("t1.1", TaskStatus::Done), ("t1.2", TaskStatus::Done)]);
    assert_eq!(plan.tasks[2].depends_on, vec!["t1.1"]);
    assert_eq!(checks.load(Ordering::SeqCst), 2, "first try and one retry");
    let briefs = briefs.lock().unwrap();
    assert!(briefs[1].contains("## What the last attempt was missing\nonly two vendors named"));
    assert!(briefs[2].contains("### An earlier attempt at this (Research the market sources) came back short"));
}

#[tokio::test]
async fn rejected_work_is_reworked_and_critics_combine() {
    let reviews = Arc::new(AtomicU32::new(0));
    let n = reviews.clone();
    let respond: Responder = Arc::new(move |role, req| match role {
        "planner" => r#"{"goal": "g", "tasks": [{"id": "t1", "statement": "Draft the email"}]}"#.into(),
        "reviewer" => {
            let round = n.fetch_add(1, Ordering::SeqCst);
            let strict = last_text(req).contains("for one thing in particular: tone");
            if round < 2 && strict {
                r#"{"accepted": false, "score": 0.3, "gaps": ["Make the tone warmer"], "summary": "cold"}"#.into()
            } else {
                r#"{"accepted": true, "score": 0.9, "summary": "fine"}"#.into()
            }
        }
        "manager" => "email v2".into(),
        _ => "an email".into(),
    });
    let (_p, harness) = scripted(respond);
    let boss = Orchestrator::builder("lead").harness(harness).critic_lenses(["accuracy", "tone"]).build().unwrap();
    let r = boss.run("Write the email.").await.unwrap();
    assert!(r.ok(), "{:?}", r.error);
    let plan: Plan = serde_json::from_value(r.data["plan"].clone()).unwrap();
    assert_eq!(plan.tasks.len(), 2, "the gap became a task");
    assert_eq!(plan.tasks[1].statement, "Make the tone warmer");
    assert_eq!(r.data["review"]["accepted"], true);
    assert_eq!(r.data["review"]["verdicts"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn an_unaffordable_job_does_not_start() {
    let respond: Responder = Arc::new(|role, _| match role {
        "planner" => r#"{"goal": "g", "tasks": [{"id": "t1", "statement": "Research one"}, {"id": "t2", "statement": "Research two"}]}"#.into(),
        _ => "x".into(),
    });
    let (_p, harness) = scripted(respond);
    let boss = Orchestrator::builder("lead")
        .harness(harness)
        .model("claude-opus-5")
        .budget(Budget { max_usd: Some(0.0000001), ..Default::default() })
        .on_over_estimate("stop")
        .build()
        .unwrap();
    let r = boss.run("Do it.").await.unwrap();
    let plan: Plan = serde_json::from_value(r.data["plan"].clone()).unwrap_or_default();
    // The planner's own call already spends past the tiny budget: the job either stops at
    // the budget while planning, or refuses to start tasks it cannot pay for.
    let refused = plan.tasks.iter().all(|t| t.status == TaskStatus::Skipped);
    assert!(r.error.is_some() || refused, "{r:#?}");
    let _ = Value::Null;
    assert!(matches!(Orchestrator::builder("x").on_over_estimate("panic").build(), Err(Error::Configuration(_))));
}
