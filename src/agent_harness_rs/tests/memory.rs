//! Memory and knowledge, end to end, through an agent.

use std::sync::Arc;

use agent_harness::knowledge::{KnowledgeBase, NewDocument};
use agent_harness::llm_providers::{FakeProvider, Script, tool_call};
use agent_harness::memory::{MemoryManager, MemoryOptions, Trace};
use agent_harness::runtime::HookEngine;
use agent_harness::types::ContentBlock;
use agent_harness::{Agent, Harness, SubAgentSpec};
use serde_json::{Value, json};

fn fake(script: Vec<Script>) -> Arc<FakeProvider> {
    Arc::new(FakeProvider::new(script))
}

fn system(p: &FakeProvider, request: usize) -> String {
    p.requests()[request].system.clone().unwrap_or_default()
}

fn last_result(p: &FakeProvider, request: usize) -> String {
    match &p.requests()[request].messages.last().unwrap().content[0] {
        ContentBlock::ToolResult(r) => r.content.clone(),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn what_it_remembers_is_in_the_next_prompt() {
    let p = fake(vec![tool_call("remember", json!({"fact": "The user bills in EUR", "kind": "preference"})), "noted".into(), "EUR it is".into()]);
    let a = Agent::builder("support").provider(p.clone()).trace("alice").build().unwrap();
    assert_eq!(a.trace().unwrap().user_id.as_deref(), Some("alice"));
    assert_eq!(a.trace().unwrap().agent.as_deref(), Some("support"));
    a.run("I bill in euros").await.unwrap();
    assert_eq!(last_result(&p, 1), "remembered: The user bills in EUR");
    a.run("which currency?").await.unwrap();
    assert!(system(&p, 2).contains("## What you already know about this user\n- [preference] The user bills in EUR"), "{}", system(&p, 2));
    assert_eq!(a.memory().unwrap().session().messages.len(), 6, "both runs are in the live session");
}

#[tokio::test]
async fn one_store_many_users() {
    let harness = Harness::testing(Some(fake(vec![])));
    let shared = MemoryManager::new(MemoryOptions { store: Some(harness.memory_store.clone()), ..Default::default() });
    let pa = fake(vec![tool_call("remember", json!({"fact": "Alice likes tea"})), "ok".into()]);
    let alice = Agent::builder("a").provider(pa).memory_manager(shared.clone()).trace("alice").build().unwrap();
    alice.run("note it").await.unwrap();
    let pb = fake(vec!["hi".into()]);
    let bob = Agent::builder("b").provider(pb.clone()).memory_manager(shared).trace("bob").build().unwrap();
    bob.run("hello").await.unwrap();
    assert!(!system(&pb, 0).contains("Alice likes tea"), "one user never sees another's memory");
    assert_eq!(alice.memory().unwrap().user.load().await.unwrap(), "- [fact] Alice likes tea");
}

#[tokio::test]
async fn memory_writes_pass_the_hook() {
    let mut hooks = HookEngine::new();
    hooks.on("memory_write", |ctx| {
        let text = ctx.get("text").and_then(Value::as_str).unwrap_or("").to_string();
        if text.contains("password") {
            ctx.block("no secrets in memory");
        } else {
            ctx.replace(json!(text.replace("4111", "****")));
        }
    });
    let p = fake(vec![tool_call("remember", json!({"fact": "card 4111"})), tool_call("remember", json!({"fact": "password is hunter2"})), "done".into()]);
    let a = Agent::builder("w").provider(p.clone()).hooks(hooks).build().unwrap();
    a.run("go").await.unwrap();
    assert_eq!(last_result(&p, 1), "remembered: card ****");
    assert!(last_result(&p, 2).contains("memory write refused: no secrets in memory"), "{}", last_result(&p, 2));
    assert_eq!(a.memory().unwrap().user.load().await.unwrap(), "- [fact] card ****");
}

#[tokio::test]
async fn memory_can_be_turned_off() {
    let p = fake(vec!["hi".into()]);
    let a = Agent::builder("w").provider(p.clone()).memory(false).build().unwrap();
    a.run("go").await.unwrap();
    assert!(a.memory().is_none());
    assert!(a.tools().is_empty());
    assert!(!system(&p, 0).contains("Tools you can call"));
}

#[tokio::test]
async fn delegation_is_a_digest_and_children_start_clean() {
    let p = fake(vec![tool_call("delegate", json!({"agent_name": "reader", "task": "read the file"})), "read it".into(), "all done".into()]);
    let lead = Agent::builder("lead").provider(p.clone()).subagent_spec(SubAgentSpec::new("reader", "Reads.", "Read.")).build().unwrap();
    assert!(lead.subagents()["reader"].memory().is_none(), "sub-agents start with no memory");
    assert!(!lead.subagents()["reader"].tools().contains("remember"));
    lead.run("do it").await.unwrap();
    let after = system(&p, 2);
    assert!(after.contains("## Job so far\nstaffing:\n  - read the file → reused reader\nspend so far: $"), "{after}");
}

#[tokio::test]
async fn runs_without_a_session_keep_one_record() {
    let p = fake(vec!["one".into(), "two".into()]);
    let a = Agent::builder("w").provider(p.clone()).build().unwrap();
    let r1 = a.run("first").await.unwrap();
    let r2 = a.run("second").await.unwrap();
    assert_eq!(r1.session_id, r2.session_id);
    assert_eq!(r1.session_id, a.memory().unwrap().session().id);
    assert_eq!(p.requests()[1].messages.len(), 1, "the record is kept, not replayed");
    let held = a.harness().sessions.load(&r1.session_id).await.unwrap();
    assert_eq!(held.messages.len(), 4);
    a.new_session();
    let r3 = a.run("third").await.unwrap();
    assert_ne!(r3.session_id, r1.session_id);
}

#[tokio::test]
async fn memory_survives_the_process_on_disk() {
    let dir = std::env::temp_dir().join(format!("ah_local_{}", uuid_like()));
    {
        let p = fake(vec![tool_call("remember", json!({"fact": "Ships on Fridays"})), "ok".into()]);
        let a = Agent::builder("w").provider(p).harness(Harness::local(&dir).unwrap()).trace(Trace::user("u1")).build().unwrap();
        a.run("remember").await.unwrap();
    }
    let p = fake(vec!["Friday".into()]);
    let a = Agent::builder("w").provider(p.clone()).harness(Harness::local(&dir).unwrap()).trace(Trace::user("u1")).build().unwrap();
    a.run("when do we ship?").await.unwrap();
    assert!(system(&p, 0).contains("- [fact] Ships on Fridays"));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn a_knowledge_base_as_a_tool() {
    let kb = KnowledgeBase::new("memory").unwrap();
    kb.add(NewDocument::text("# Refunds\n\nRefunds take five working days.").id("refunds").title("Refunds")).await.unwrap();
    kb.add(NewDocument::text("# Shipping\n\nOrders ship within two days.").id("shipping")).await.unwrap();
    let again = kb.add(NewDocument::text("# Refunds\n\nRefunds take five working days.").id("refunds").title("Refunds")).await.unwrap();
    assert!(again.unchanged);
    let p = fake(vec![tool_call("search_knowledge", json!({"query": "how long do refunds take", "limit": 1})), "Five days.".into()]);
    let a = Agent::builder("support").provider(p.clone()).memory(false).tool(kb.as_tool("search_knowledge")).build().unwrap();
    a.run("refund time?").await.unwrap();
    let found: Value = serde_json::from_str(&last_result(&p, 1)).unwrap();
    assert_eq!(found[0]["document"], "refunds");
    assert_eq!(found[0]["title"], "Refunds");
    assert!(found[0]["text"].as_str().unwrap().contains("five working days"));
    kb.delete("refunds").await.unwrap();
    assert_eq!(kb.count().await.unwrap(), 1);
}

fn uuid_like() -> String {
    format!("{:x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos())
}
