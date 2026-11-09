//! A2A: serving an agent to other frameworks, and calling theirs. Our client
//! against our server, over real HTTP on 127.0.0.1.

use std::sync::Arc;
use std::time::Duration;

use agent_harness::a2a::{
    A2AClient, A2AError, A2AEvent, A2AServer, A2AServerBuilder, MemoryTaskStore, Principal, RemoteAgent, SendOptions, Served, SessionTaskStore, TaskRecord,
    TaskStore,
};
use agent_harness::llm_providers::{FakeProvider, Script, tool_call};
use agent_harness::runtime::{ListFilter, SessionStore};
use agent_harness::types::ContentBlock;
use agent_harness::{Agent, Harness, Tool};
use axum::routing::{get, post};
use futures::StreamExt;
use serde_json::{Map, Value, json};
use tokio::sync::watch;

// ----------------------------------------------------------------------
// fixtures
// ----------------------------------------------------------------------

/// A tool that waits until the test lets it go.
#[derive(Clone)]
struct Gate {
    started: watch::Sender<bool>,
    go: watch::Sender<bool>,
}

impl Gate {
    fn new() -> Self {
        Self { started: watch::channel(false).0, go: watch::channel(false).0 }
    }

    fn tool(&self) -> Tool {
        let gate = self.clone();
        Tool::raw("slow", "Wait until the test lets it go.", json!({"type": "object", "properties": {}}), move |_, _| {
            let gate = gate.clone();
            async move {
                gate.started.send_replace(true);
                let mut go = gate.go.subscribe();
                let _ = go.wait_for(|v| *v).await;
                Ok(json!("waited"))
            }
        })
    }

    async fn started(&self) {
        let mut rx = self.started.subscribe();
        let _ = tokio::time::timeout(Duration::from_secs(10), rx.wait_for(|v| *v)).await;
    }

    fn go(&self) {
        self.go.send_replace(true);
    }
}

fn script(items: Vec<Script>) -> Arc<FakeProvider> {
    Arc::new(FakeProvider::new(items).streaming_words(Duration::ZERO))
}

fn desk(provider: Arc<FakeProvider>, harness: Harness, tools: Vec<Tool>, max_steps: Option<u32>) -> Agent {
    let mut b = Agent::builder("desk")
        .instructions("Answer the customer.")
        .description("A support desk.")
        .provider(provider)
        .harness(harness)
        .memory(false)
        .tools(tools);
    if let Some(n) = max_steps {
        b = b.max_steps(n);
    }
    b.build().unwrap()
}

struct Served1 {
    server: A2AServer,
    provider: Arc<FakeProvider>,
    url: String,
}

async fn serve(server: A2AServer) -> String {
    server.spawn_local().await.unwrap().0
}

async fn served_with(
    items: Vec<Script>,
    harness: Option<Harness>,
    tools: Vec<Tool>,
    max_steps: Option<u32>,
    f: impl FnOnce(A2AServerBuilder) -> A2AServerBuilder,
) -> Served1 {
    let provider = script(items);
    let agent = desk(provider.clone(), harness.unwrap_or_else(|| Harness::testing(None)), tools, max_steps);
    let server = f(A2AServer::builder(agent).poll_interval(0.01)).build().unwrap();
    let url = serve(server.clone()).await;
    Served1 { server, provider, url }
}

async fn served(items: &[&str]) -> Served1 {
    served_with(items.iter().map(|s| Script::text(*s)).collect(), None, vec![], None, |b| b).await
}

fn calling(url: &str) -> A2AClient {
    A2AClient::new(url).retries(0)
}

fn send() -> SendOptions {
    SendOptions::default()
}

async fn rpc(server: &A2AServer, method: &str, params: Value) -> Value {
    rpc_as(server, method, params, None).await
}

async fn rpc_as(server: &A2AServer, method: &str, params: Value, who: Option<&Principal>) -> Value {
    server.handle(json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}), who, None).await.unwrap().json().unwrap()
}

fn said(text: &str) -> Value {
    json!({"message": {"role": "user", "parts": [{"kind": "text", "text": text}], "messageId": uuid::Uuid::new_v4().to_string()}})
}

async fn collect(mut events: futures::stream::BoxStream<'static, Result<A2AEvent, A2AError>>) -> Vec<A2AEvent> {
    let mut out = vec![];
    while let Some(e) = events.next().await {
        out.push(e.unwrap());
    }
    out
}

fn http() -> reqwest::Client {
    agent_harness::install_tls();
    reqwest::Client::new()
}

// ----------------------------------------------------------------------
// the card
// ----------------------------------------------------------------------
#[tokio::test]
async fn the_card_says_what_the_agent_is_and_where() {
    let s = served_with(vec![], None, vec![], None, |b| b.version("2.1.0").provider(json!({"organization": "Acme", "url": "https://acme.test"}))).await;
    let client = calling(&s.url);
    let card = client.card(false).await.unwrap();
    let legacy: Value = http().get(format!("{}/.well-known/agent.json", s.url)).send().await.unwrap().json().await.unwrap();
    assert_eq!(card, legacy);
    assert_eq!((card["protocolVersion"].as_str(), card["preferredTransport"].as_str()), (Some("0.3.0"), Some("JSONRPC")));
    assert_eq!((&card["name"], &card["description"], &card["version"]), (&json!("desk"), &json!("A support desk."), &json!("2.1.0")));
    assert_eq!(card["url"], format!("{}/", s.url)); // read from the request
    assert_eq!(card["capabilities"], json!({"streaming": true, "pushNotifications": false, "stateTransitionHistory": false}));
    assert_eq!(card["skills"], json!([{"id": "desk", "name": "desk", "description": "A support desk.", "tags": ["agent"]}]));
    assert!(card.get("securitySchemes").is_none());
    assert_eq!(card["provider"]["organization"], "Acme");

    let locked = served_with(vec![], None, vec![], None, |b| b.token("sk-1").url("https://public.example/a2a/")).await;
    let card = locked.server.card(None, "");
    assert_eq!(card["url"], "https://public.example/a2a/");
    assert_eq!(card["security"], json!([{"bearer": []}]));
}

// ----------------------------------------------------------------------
// sending a message
// ----------------------------------------------------------------------
#[tokio::test]
async fn a_message_becomes_a_task_that_completes() {
    let s = served(&["Your order shipped on Thursday."]).await;
    let task = calling(&s.url).send("Where is order 4182?", &send()).await.unwrap();
    assert!(task.ok() && task.done());
    assert_eq!(task.text, "Your order shipped on Thursday.");
    let raw = &task.raw;
    assert_eq!((raw["kind"].as_str(), raw["status"]["state"].as_str()), (Some("task"), Some("completed")));
    assert_eq!(raw["artifacts"][0]["name"], "response");
    assert_eq!(raw["artifacts"][0]["parts"], json!([{"kind": "text", "text": "Your order shipped on Thursday."}]));
    let roles: Vec<&str> = raw["history"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["user", "agent"]);
    assert_eq!(raw["history"][0]["taskId"], json!(task.id));
    assert_eq!(raw["id"], json!(task.id));
    assert_eq!(raw["contextId"], json!(task.context_id));
    assert_eq!(s.provider.requests()[0].messages[0].text(), "Where is order 4182?");
    assert_eq!(s.server.health()["completed"], 1);
}

#[tokio::test]
async fn a_context_id_carries_the_conversation_on() {
    let s = served(&["Noted: order 4182.", "It was 4182."]).await;
    let client = calling(&s.url);
    let first = client.send("My order is 4182.", &send()).await.unwrap();
    let second = client.send("Which order?", &send().context_id(&first.context_id)).await.unwrap();
    let other = client.send("Which order?", &send()).await.unwrap();
    assert_eq!(second.context_id, first.context_id);
    assert_ne!(second.id, first.id);
    assert_eq!(second.text, "It was 4182.");
    let seen: Vec<Vec<String>> = s.provider.requests().iter().map(|r| r.messages.iter().map(|m| m.text()).collect()).collect();
    assert_eq!(seen[1], ["My order is 4182.", "Noted: order 4182.", "Which order?"]);
    assert_eq!(seen[2], ["Which order?"]);
    assert_ne!(other.context_id, first.context_id);
}

#[tokio::test]
async fn a_stream_is_the_task_as_it_happens() {
    let s = served(&["Your order shipped on Thursday."]).await;
    let events = collect(calling(&s.url).stream("Where is my order?", &send()).await.unwrap()).await;
    assert_eq!((events[0].kind.as_str(), events[0].state.as_str()), ("task", "submitted"));
    assert_eq!((events[1].kind.as_str(), events[1].state.as_str()), ("status-update", "working"));
    let last = events.last().unwrap();
    assert!(last.kind == "status-update" && last.is_final && last.state == "completed");
    let chunks: Vec<&A2AEvent> = events.iter().filter(|e| e.kind == "artifact-update").collect();
    assert_eq!(chunks.iter().map(|e| e.text.as_str()).collect::<String>(), "Your order shipped on Thursday.");
    assert!(chunks.len() > 3);
    assert_eq!(chunks[0].raw["append"], false);
    assert!(chunks[1..].iter().all(|c| c.raw["append"] == true));
    assert_eq!(chunks.last().unwrap().raw["lastChunk"], true);
    let ids: std::collections::HashSet<String> = chunks.iter().map(|c| c.raw["artifact"]["artifactId"].as_str().unwrap().to_string()).collect();
    assert_eq!(ids.len(), 1);
    // The task that was streamed is the task that is kept.
    let kept = rpc(&s.server, "tasks/get", json!({"id": events[0].task_id})).await;
    assert_eq!(agent_harness::a2a::protocol::answer_of(&kept["result"]), "Your order shipped on Thursday.");
}

#[tokio::test]
async fn what_was_kept_replaces_what_was_streamed_when_they_differ() {
    let mut harness = Harness::testing(None);
    harness.hooks.on("run_end", |ctx| {
        let mut result = ctx.data["result"].clone();
        result["output"] = json!("[withheld by policy]");
        ctx.replacement = Some(result);
    });
    let s = served_with(vec![Script::text("Write to ada@example.com today.")], Some(harness), vec![], None, |b| b).await;
    let events = collect(calling(&s.url).stream("Who do I write to?", &send()).await.unwrap()).await;
    let last = events.iter().rfind(|e| e.kind == "artifact-update").unwrap();
    assert_eq!((&last.raw["append"], &last.raw["lastChunk"]), (&json!(false), &json!(true)));
    assert_eq!(last.text, "[withheld by policy]");
}

#[tokio::test]
async fn a_message_can_be_sent_without_waiting_for_it() {
    let gate = Gate::new();
    let s = served_with(vec![tool_call("slow", json!({})), Script::text("Done.")], None, vec![gate.tool()], None, |b| b).await;
    let client = calling(&s.url);
    let task = client.send("Take your time.", &send().no_wait()).await.unwrap();
    assert!(["submitted", "working"].contains(&task.state.as_str()) && !task.done());
    gate.started().await;
    assert_eq!(client.get(&task.id, None).await.unwrap().state, "working");
    gate.go();
    let finished = client.wait(&task.id, 30.0, 0.01).await.unwrap();
    assert!(finished.ok());
    assert_eq!(finished.text, "Done.");
    assert_eq!(rpc(&s.server, "tasks/get", json!({"id": task.id, "historyLength": 1})).await["result"]["history"][0]["role"], "agent");
}

#[tokio::test]
async fn the_same_message_sent_twice_is_one_task() {
    let s = served(&["Refunded.", "never said"]).await;
    let client = calling(&s.url);
    let first = client.send("Refund order 4182.", &send().message_id("msg-1")).await.unwrap();
    let again = client.send("Refund order 4182.", &send().message_id("msg-1")).await.unwrap();
    let events = collect(client.stream("Refund order 4182.", &send().message_id("msg-1")).await.unwrap()).await;
    assert_eq!(again.id, first.id);
    assert_eq!(again.text, "Refunded.");
    assert_eq!(s.provider.requests().len(), 1);
    assert!(events[0].kind == "task" && events[0].is_final);
    assert_eq!(s.server.health()["deduplicated"], 2);
}

#[tokio::test]
async fn files_and_data_reach_the_agent() {
    use base64::Engine as _;
    let s = served(&["It says hello."]).await;
    let note = base64::engine::general_purpose::STANDARD.encode(b"hello from the file");
    let answer = rpc(
        &s.server,
        "message/send",
        json!({"message": {"role": "user", "messageId": "m1", "parts": [
            {"kind": "text", "text": "Read these."},
            {"kind": "data", "data": {"order": 4182}},
            {"kind": "file", "file": {"name": "note.txt", "mimeType": "text/plain", "bytes": note}}]}}),
    )
    .await;
    let task = &answer["result"];
    assert_eq!(task["status"]["state"], "completed");
    let sent = &s.provider.requests()[0].messages[0];
    assert!(sent.text().contains("Read these.") && sent.text().contains("{\"order\": 4182}"), "{}", sent.text());
    let all: String = sent
        .content
        .iter()
        .map(|b| match b {
            ContentBlock::Text(t) => t.text.clone(),
            _ => String::new(),
        })
        .collect();
    assert!(all.contains("hello from the file"), "{all}");
    // The record keeps that a file came, not the file.
    assert_eq!(task["history"][0]["parts"][2], json!({"kind": "file", "file": {"name": "note.txt", "mimeType": "text/plain"}}));
}

#[tokio::test]
async fn an_agent_that_fails_is_a_failed_task() {
    let s = served_with(vec![tool_call("nope", json!({})), tool_call("nope", json!({}))], None, vec![], Some(2), |b| b).await;
    let task = calling(&s.url).send("Go.", &send()).await.unwrap();
    assert_eq!(task.state, "failed");
    assert!(task.error().unwrap().contains("MaxStepsExceeded"), "{:?}", task.error());
    assert_eq!(task.raw["artifacts"], json!([]));
    assert_eq!(s.server.health()["failed"], 1);
}

// ----------------------------------------------------------------------
// what is refused
// ----------------------------------------------------------------------
#[tokio::test]
async fn requests_that_are_wrong_are_answered_as_json_rpc_says() {
    let s = served(&["Hello."]).await;
    let code = |v: Value| v["error"]["code"].as_i64().unwrap();
    assert_eq!(code(s.server.handle(json!([1]), None, None).await.unwrap().json().unwrap()), -32600);
    assert_eq!(code(rpc(&s.server, "tasks/nope", json!({})).await), -32601);
    assert_eq!(code(rpc(&s.server, "message/send", json!({})).await), -32602);
    assert_eq!(code(rpc(&s.server, "message/send", json!({"message": {"parts": []}})).await), -32602);
    assert_eq!(code(rpc(&s.server, "message/send", json!({"message": {"parts": [{"kind": "hologram"}]}})).await), -32005);
    assert_eq!(code(rpc(&s.server, "tasks/get", json!({"id": "ghost"})).await), -32001);
    assert_eq!(code(rpc(&s.server, "agent/getAuthenticatedExtendedCard", json!({})).await), -32007);
    assert_eq!(code(rpc(&s.server, "tasks/pushNotificationConfig/set", json!({"taskId": "x"})).await), -32001);

    let bad: Value = http().post(format!("{}/", s.url)).body("{not json").send().await.unwrap().json().await.unwrap();
    assert_eq!(bad["error"]["code"], -32700);
    assert_eq!(http().get(format!("{}/", s.url)).send().await.unwrap().status(), 405);
    assert_eq!(http().get(format!("{}/nowhere", s.url)).send().await.unwrap().status(), 404);
    let client = calling(&s.url);
    let done = client.send("Hi.", &send()).await.unwrap();
    let refused = client.send("And?", &send().task_id(&done.id)).await.unwrap_err();
    assert!(refused.message.contains("takes no more messages"), "{refused}");
    assert_eq!(refused.code, -32602);
}

#[tokio::test]
async fn a_body_that_is_too_large_is_refused() {
    let s = served_with(vec![Script::text("Hello.")], None, vec![], None, |b| b.max_body_bytes(200)).await;
    let refused = calling(&s.url).send("x".repeat(500), &send()).await.unwrap_err();
    assert!(refused.message.contains("larger than 200 bytes"), "{refused}");
    assert_eq!(refused.status, 413);
}

// ----------------------------------------------------------------------
// who is calling
// ----------------------------------------------------------------------
#[tokio::test]
async fn a_caller_must_be_known_and_has_only_what_is_theirs() {
    let harness = Harness::testing(None);
    let s = served_with(["Noted.", "Yours is 4182.", "I do not know."].map(Script::text).to_vec(), Some(harness.clone()), vec![], None, |b| {
        b.token_for("sk-ada", Principal::user("ada").tenant("acme")).token_for("sk-bob", Principal::of(&json!("bob")))
    })
    .await;

    let nobody = calling(&s.url);
    let refused = nobody.send("Hello?", &send()).await.unwrap_err();
    assert!(refused.message.contains("not authorised"));
    assert_eq!(refused.status, 401);
    assert_eq!(nobody.card(false).await.unwrap()["name"], "desk"); // the card is public
    assert!(calling(&s.url).token("sk-wrong").send("Hello?", &send()).await.unwrap_err().message.contains("not authorised"));

    let ada = calling(&s.url).token("sk-ada");
    let bob = calling(&s.url).token("sk-bob");
    let mine = ada.send("My order is 4182.", &send()).await.unwrap();
    assert_eq!(ada.send("Which?", &send().context_id(&mine.context_id)).await.unwrap().text, "Yours is 4182.");
    // Bob is told Ada's task and Ada's conversation do not exist.
    let hidden = bob.get(&mine.id, None).await.unwrap_err();
    assert!(hidden.message.contains("no task"));
    assert_eq!(hidden.code, -32001);
    assert!(bob.cancel(&mine.id).await.unwrap_err().message.contains("no task"));
    assert!(bob.send("Which order was it?", &send().context_id(&mine.context_id)).await.unwrap_err().message.contains("no conversation"));

    // The conversation is a session of the agent's, and it is Ada's.
    let filter = |user: &str, tenant: Option<&str>| ListFilter {
        user_id: Some(user.into()),
        tenant_id: tenant.map(str::to_string),
        agent: Some("desk".into()),
        limit: 10,
    };
    let chats = harness.sessions.list(&filter("ada", Some("acme"))).await.unwrap();
    assert_eq!(chats.len(), 1);
    assert_eq!(chats[0].messages.len(), 4);
    assert!(harness.sessions.list(&filter("bob", None)).await.unwrap().is_empty());
}

#[tokio::test]
async fn a_function_can_decide_who_is_calling() {
    let mut schemes = Map::new();
    schemes.insert("key".into(), json!({"type": "apiKey", "in": "header", "name": "X-API-Key"}));
    let s = served_with(vec![Script::text("Hello.")], None, vec![], None, move |b| {
        b.security_schemes(schemes).auth(|headers| {
            let key = headers.get("x-api-key").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
            async move { ["k1", "k2"].contains(&key.as_str()).then(|| Principal::of(&json!({"user_id": key, "plan": "gold"}))) }
        })
    })
    .await;
    assert_eq!(s.server.card(None, "")["security"], json!([{"key": []}]));
    assert!(calling(&s.url).header("X-API-Key", "k1").send("Hi.", &send()).await.unwrap().ok());
    assert!(calling(&s.url).header("X-API-Key", "zz").send("Hi.", &send()).await.unwrap_err().message.contains("not authorised"));
}

#[tokio::test]
async fn every_conversation_gets_an_agent_of_its_own() {
    let harness = Harness::testing(None);
    let built = Arc::new(std::sync::Mutex::new(Vec::<Option<String>>::new()));
    let (h, b) = (harness.clone(), built.clone());
    let factory = Served::factory(move |p: &Principal| {
        b.lock().unwrap().push(p.user_id.clone());
        Agent::builder("desk").provider(Arc::new(FakeProvider::new(vec![Script::text("Hello.")]))).memory(false).harness(h.clone()).build()
    });
    let server = A2AServer::builder(factory.clone())
        .token_for("tok-1", Principal::user("ada"))
        .token_for("tok-2", Principal::user("bob"))
        .store(Arc::new(SessionTaskStore::new(harness.sessions.clone())))
        .build()
        .unwrap();
    let url = serve(server).await;
    let (ada, bob) = (calling(&url).token("tok-1"), calling(&url).token("tok-2"));
    let first = ada.send("Hi.", &send()).await.unwrap();
    ada.send("Again.", &send().context_id(&first.context_id)).await.unwrap(); // the same agent
    bob.send("Hi.", &send()).await.unwrap();
    assert_eq!(*built.lock().unwrap(), vec![Some("ada".to_string()), Some("bob".to_string())]);

    let err = A2AServer::builder(factory).build().unwrap_err().to_string();
    assert!(err.contains("needs `store(...)`"), "{err}");
    assert!(A2AServer::builder_many(Vec::<(String, Served)>::new()).build().is_err());
}

// ----------------------------------------------------------------------
// cancelling
// ----------------------------------------------------------------------
#[tokio::test]
async fn a_running_task_can_be_cancelled() {
    let gate = Gate::new();
    let s = served_with(vec![tool_call("slow", json!({})), Script::text("never said")], None, vec![gate.tool()], None, |b| b).await;
    let client = calling(&s.url);
    let task = client.send("Wait.", &send().no_wait()).await.unwrap();
    gate.started().await;
    let cancelled = client.cancel(&task.id).await.unwrap();
    assert_eq!(cancelled.state, "canceled");
    assert_eq!(cancelled.error().as_deref(), Some("Canceled."));
    let twice = client.cancel(&task.id).await.unwrap_err();
    assert!(twice.message.contains("already canceled"));
    assert_eq!(twice.code, -32002);
    assert_eq!(s.provider.requests().len(), 1);
    assert_eq!(s.server.health()["running"], 0);
}

#[tokio::test]
async fn a_task_that_runs_too_long_is_stopped() {
    let gate = Gate::new();
    let s = served_with(vec![tool_call("slow", json!({})), Script::text("never")], None, vec![gate.tool()], None, |b| b.task_timeout(0.05)).await;
    let task = calling(&s.url).send("Wait.", &send()).await.unwrap();
    assert_eq!(task.state, "failed");
    assert!(task.error().unwrap().contains("ran longer than 0.05s"), "{:?}", task.error());
}

// ----------------------------------------------------------------------
// at scale: more than one replica
// ----------------------------------------------------------------------
/// Two servers as two processes would be: their own agents and harnesses, one
/// store of sessions and tasks between them.
async fn replicas(items: Vec<Script>, tools: Vec<Tool>, task_timeout: Option<f64>) -> ((A2AServer, String), (A2AServer, String), Arc<dyn SessionStore>) {
    let shared = Harness::testing(None).sessions;
    let provider = script(items);
    let mut out = vec![];
    for _ in 0..2 {
        let harness = Harness::testing(None).with_sessions(shared.clone());
        let agent = desk(provider.clone(), harness, tools.clone(), None);
        let mut b = A2AServer::builder(agent).poll_interval(0.01);
        if let Some(t) = task_timeout {
            b = b.task_timeout(t);
        }
        let server = b.build().unwrap();
        let url = serve(server.clone()).await;
        out.push((server, url));
    }
    let b = out.pop().unwrap();
    let a = out.pop().unwrap();
    (a, b, shared)
}

#[tokio::test]
async fn any_replica_answers_for_a_task_and_carries_on_a_conversation() {
    let ((_, a), (_, b), _) = replicas(vec![Script::text("Noted: 4182."), Script::text("It was 4182.")], vec![], None).await;
    let (on_a, on_b) = (calling(&a), calling(&b));
    let first = on_a.send("My order is 4182.", &send()).await.unwrap();
    assert_eq!(on_b.get(&first.id, None).await.unwrap().text, "Noted: 4182.");
    let second = on_b.send("Which order?", &send().context_id(&first.context_id)).await.unwrap();
    assert_eq!(second.text, "It was 4182.");
}

#[tokio::test]
async fn a_retry_that_lands_on_another_replica_waits_for_the_first() {
    let gate = Gate::new();
    let ((_, a), (b_server, b), _) = replicas(vec![tool_call("slow", json!({})), Script::text("Refunded.")], vec![gate.tool()], None).await;
    let (on_a, on_b) = (calling(&a), calling(&b));
    let started = on_a.send("Refund it.", &send().message_id("m-1").no_wait()).await.unwrap();
    gate.started().await;
    let retry = tokio::spawn(async move { on_b.send("Refund it.", &send().message_id("m-1")).await });
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(!retry.is_finished());
    gate.go();
    let finished = retry.await.unwrap().unwrap();
    assert_eq!(finished.id, started.id);
    assert_eq!(finished.text, "Refunded.");
    assert_eq!((b_server.health()["accepted"].as_u64(), b_server.health()["deduplicated"].as_u64()), (Some(0), Some(1)));
}

#[tokio::test]
async fn a_task_is_cancelled_from_a_replica_that_is_not_running_it() {
    let gate = Gate::new();
    let ((_, a), (_, b), _) = replicas(vec![tool_call("slow", json!({})), Script::text("never said")], vec![gate.tool()], None).await;
    let (on_a, on_b) = (calling(&a), calling(&b));
    let task = on_a.send("Wait.", &send().no_wait()).await.unwrap();
    gate.started().await;
    let asked = on_b.cancel(&task.id).await.unwrap();
    assert_eq!(asked.state, "working"); // marked; not b's to stop
    gate.go();
    let finished = on_a.wait(&task.id, 30.0, 0.01).await.unwrap();
    assert_eq!(finished.state, "canceled");
}

#[tokio::test]
async fn a_task_is_followed_from_a_replica_that_is_not_running_it() {
    let gate = Gate::new();
    let ((_, a), (_, b), _) = replicas(vec![tool_call("slow", json!({})), Script::text("All done.")], vec![gate.tool()], None).await;
    let on_a = calling(&a);
    let task = on_a.send("Wait.", &send().no_wait()).await.unwrap();
    gate.started().await;
    let here = on_a.resubscribe(&task.id).await.unwrap();
    let there = calling(&b).resubscribe(&task.id).await.unwrap();
    let (here, there) = (tokio::spawn(collect(here)), tokio::spawn(collect(there)));
    tokio::time::sleep(Duration::from_millis(30)).await;
    gate.go();
    for events in [here.await.unwrap(), there.await.unwrap()] {
        assert_eq!(events[0].kind, "task");
        let last = events.last().unwrap();
        assert!(last.is_final && last.state == "completed", "{last:?}");
        assert!(events.iter().map(|e| e.text.as_str()).collect::<String>().contains("All done."));
    }
}

#[tokio::test]
async fn a_task_whose_worker_died_is_reported_failed() {
    let ((_, _), (b_server, b), shared) = replicas(vec![Script::text("unused")], vec![], Some(0.05)).await;
    // Replica a took the task and died before working on it.
    let store = SessionTaskStore::new(shared);
    let mut record = TaskRecord {
        id: "t-lost".into(),
        agent: "desk".into(),
        task: json!({"kind": "task", "id": "t-lost", "contextId": "c", "status": {"state": "submitted"}, "history": [], "artifacts": []}),
        ..Default::default()
    };
    assert!(store.create(&mut record).await.unwrap());
    let on_b = calling(&b);
    assert_eq!(on_b.get("t-lost", None).await.unwrap().state, "submitted");
    let mut held = b_server.store().load("t-lost").await.unwrap().unwrap();
    held.updated -= 3600.0;
    b_server.store().save(&mut held).await.unwrap();
    let lost = on_b.get("t-lost", None).await.unwrap();
    assert_eq!(lost.state, "failed");
    assert!(lost.error().unwrap().contains("worker handling this task was lost"));
}

#[tokio::test]
async fn a_full_replica_says_so_and_the_client_comes_back() {
    let gate = Gate::new();
    let s = served_with(
        ["One.", "Two."].map(Script::text).into_iter().fold(vec![tool_call("slow", json!({}))], |mut v, x| {
            v.push(x);
            v
        }),
        None,
        vec![gate.tool()],
        None,
        |b| b.max_concurrency(1).max_queue(0),
    )
    .await;
    let client = calling(&s.url);
    let first = client.send("One?", &send().no_wait()).await.unwrap();
    let full = client.send("Two?", &send()).await.unwrap_err();
    assert!(full.message.contains("at capacity"));
    assert_eq!((full.status, full.code), (429, -32000));
    let raw = http().post(format!("{}/", s.url)).json(&json!({"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": said("x")})).send().await.unwrap();
    assert_eq!(raw.status(), 429);
    assert_eq!(raw.headers()["retry-after"], "2");
    assert_eq!(s.server.health()["rejected"], 2);

    let patient = calling(&s.url).retries(3);
    let waiting = tokio::spawn(async move { patient.send("Two?", &send()).await });
    tokio::time::sleep(Duration::from_millis(50)).await;
    gate.go();
    client.wait(&first.id, 30.0, 0.01).await.unwrap();
    assert!(waiting.await.unwrap().unwrap().ok());
}

#[tokio::test]
async fn a_server_that_is_closing_finishes_its_books() {
    let gate = Gate::new();
    let s = served_with(vec![tool_call("slow", json!({})), Script::text("never")], None, vec![gate.tool()], None, |b| b).await;
    let client = calling(&s.url);
    let task = client.send("Wait.", &send().no_wait()).await.unwrap();
    gate.started().await;
    s.server.close(0.0).await;
    let after = client.get(&task.id, None).await.unwrap();
    assert_eq!(after.state, "canceled");
    assert!(after.error().unwrap().contains("shut down"));
    let health = http().get(format!("{}/healthz", s.url)).send().await.unwrap();
    assert_eq!(health.status(), 503);
    assert_eq!(health.json::<Value>().await.unwrap()["status"], "draining");
    assert!(client.send("More?", &send()).await.unwrap_err().message.contains("shutting down"));
}

#[tokio::test]
async fn tasks_can_be_kept_somewhere_of_your_own() {
    let store = Arc::new(MemoryTaskStore::new());
    let s = served_with(vec![Script::text("Hello.")], None, vec![], None, {
        let store = store.clone();
        move |b| b.store(store)
    })
    .await;
    let task = calling(&s.url).send("Hi.", &send()).await.unwrap();
    let held = store.load(&task.id).await.unwrap().unwrap();
    assert_eq!((held.state(), held.agent.as_str()), ("completed", "desk"));
    // The default keeps them with the chats, but never lists them among them.
    let harness = Harness::testing(None);
    let other = served_with(vec![Script::text("Hello.")], Some(harness.clone()), vec![], None, |b| b).await;
    rpc(&other.server, "message/send", said("Hi.")).await;
    let mut agents: Vec<String> = harness.sessions.list(&ListFilter::limit(10)).await.unwrap().into_iter().map(|s| s.agent).collect();
    agents.sort();
    assert_eq!(agents, ["desk", "desk#a2a"]);
    assert_eq!(harness.sessions.list(&ListFilter { agent: Some("desk".into()), ..ListFilter::limit(10) }).await.unwrap().len(), 1);
}

// ----------------------------------------------------------------------
// push notifications
// ----------------------------------------------------------------------
#[tokio::test]
async fn a_webhook_is_told_when_the_task_is_over() {
    let received = Arc::new(std::sync::Mutex::new(Vec::<(String, Value)>::new()));
    let kept = received.clone();
    let hook = axum::Router::new().route(
        "/a2a",
        post(move |headers: axum::http::HeaderMap, body: String| {
            let kept = kept.clone();
            async move {
                let token = headers.get("x-a2a-notification-token").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
                kept.lock().unwrap().push((token, serde_json::from_str(&body).unwrap()));
                "ok"
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let hook_url = format!("http://{}/a2a", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, hook).await.unwrap() });

    let s = served_with(vec![Script::text("Refunded.")], None, vec![], None, |b| b.push_notifications_to(|url| url.starts_with("http://127.0.0.1:"))).await;
    assert_eq!(s.server.card(None, "")["capabilities"]["pushNotifications"], true);
    let client = calling(&s.url);
    let task = client.send("Refund it.", &send().push(json!({"url": hook_url, "token": "t-9"}))).await.unwrap();
    let listed = client.call("tasks/pushNotificationConfig/list", json!({"id": task.id})).await.unwrap();
    s.server.flush_notifications().await;
    let got = received.lock().unwrap().clone();
    let (token, body) = &got[0];
    assert_eq!(token, "t-9");
    assert_eq!(body["id"], json!(task.id));
    assert_eq!(body["status"]["state"], "completed");
    assert_eq!(listed[0]["pushNotificationConfig"]["url"], json!(hook_url));
}

#[tokio::test]
async fn webhooks_are_off_unless_allowed_and_never_point_inwards() {
    let with_push = |url: &str| {
        let mut p = said("Hi.");
        p["configuration"] = json!({"pushNotificationConfig": {"url": url}});
        p
    };
    let closed = served(&["Hi."]).await;
    assert_eq!(rpc(&closed.server, "message/send", with_push("https://hooks.example.com")).await["error"]["code"], -32003);

    let open = served_with(vec![Script::text("Hi.")], None, vec![], None, |b| b.push_notifications(true)).await;
    for url in ["http://hooks.example.com", "https://localhost/x", "https://10.0.0.8/x", "https://127.0.0.1/x", "https://db.internal/x"] {
        assert_eq!(rpc(&open.server, "message/send", with_push(url)).await["error"]["code"], -32602, "{url}");
    }

    let own = served_with(vec![Script::text("Hi.")], None, vec![], None, |b| b.push_notifications_to(|url| url.starts_with("http://ci."))).await;
    let task = rpc(&own.server, "message/send", said("Hi.")).await["result"].clone();
    let done =
        rpc(&own.server, "tasks/pushNotificationConfig/set", json!({"taskId": task["id"], "pushNotificationConfig": {"url": "http://ci.test/hook"}})).await;
    let config = done["result"]["pushNotificationConfig"].clone();
    assert_eq!(rpc(&own.server, "tasks/pushNotificationConfig/get", json!({"id": task["id"]})).await["result"]["pushNotificationConfig"], config);
    rpc(&own.server, "tasks/pushNotificationConfig/delete", json!({"id": task["id"], "pushNotificationConfigId": config["id"]})).await;
    assert_eq!(rpc(&own.server, "tasks/pushNotificationConfig/list", json!({"id": task["id"]})).await["result"], json!([]));
}

// ----------------------------------------------------------------------
// several agents on one server
// ----------------------------------------------------------------------
#[tokio::test]
async fn several_agents_are_served_under_their_names() {
    let harness = Harness::testing(None);
    let one = |name: &str, says: &str| {
        Agent::builder(name).provider(Arc::new(FakeProvider::new(vec![Script::text(says)]))).harness(harness.clone()).memory(false).build().unwrap()
    };
    let server =
        A2AServer::builder_many([("billing", Served::from(one("billing", "Refunded."))), ("orders", Served::from(one("orders", "Shipped.")))]).build().unwrap();
    let url = serve(server).await;
    let (to_billing, to_orders) = (calling(&format!("{url}/billing")), calling(&format!("{url}/orders")));
    assert_eq!(to_orders.card(false).await.unwrap()["url"], format!("{url}/orders/"));
    assert_eq!(to_billing.send("Refund.", &send()).await.unwrap().text, "Refunded.");
    assert_eq!(to_orders.send("Where?", &send()).await.unwrap().text, "Shipped.");
    let index: Value = http().get(format!("{url}/")).send().await.unwrap().json().await.unwrap();
    assert_eq!(index["agents"]["billing"], format!("{url}/billing/.well-known/agent-card.json"));
}

// ----------------------------------------------------------------------
// a remote agent, used as one of your own
// ----------------------------------------------------------------------
#[tokio::test]
async fn a_remote_agent_is_a_tool_and_a_delegate() {
    let s = served(&["The gold plan is 40 EUR.", "Still 40 EUR."]).await;
    let pricing = RemoteAgent::connect(&s.url).await.unwrap();
    assert_eq!((pricing.name(), pricing.description()), ("desk", "A support desk."));

    let manager = Agent::builder("manager")
        .memory(false)
        .tool(RemoteAgent::delegate_tool("delegate_remote", std::slice::from_ref(&pricing)))
        .tool(pricing.as_tool(Some("ask_pricing"), None))
        .provider(Arc::new(FakeProvider::new(vec![
            tool_call("delegate_remote", json!({"agent_name": "desk", "task": "Price of gold?"})),
            tool_call("ask_pricing", json!({"task": "And now?"})),
            Script::text("Gold is 40 EUR."),
        ])))
        .build()
        .unwrap();
    let result = manager.run("What does gold cost?").await.unwrap();
    assert_eq!(result.output, "Gold is 40 EUR.");
    let handed: Vec<String> = result
        .messages
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|b| match b {
            ContentBlock::ToolResult(r) => Some(r.content.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(handed, ["The gold plan is 40 EUR.", "Still 40 EUR."]);
    // One-offs: neither call continued the other's conversation.
    assert_eq!(s.provider.requests()[1].messages.len(), 1);
    assert_eq!(s.provider.requests()[0].messages[0].text(), "Price of gold?");
}

#[tokio::test]
async fn a_remote_agent_run_directly_keeps_its_conversation() {
    let s = served(&["Noted.", "It was 4182.", "Which one?"]).await;
    let pricing = RemoteAgent::builder(&s.url).name("pricing").connect().await.unwrap();
    assert_eq!(pricing.name(), "pricing");
    let first = pricing.run("My order is 4182.").await;
    let second = pricing.run("Which order?").await;
    assert_eq!(second.output, "It was 4182.");
    assert_eq!(second.session_id, first.session_id);
    pricing.new_session();
    assert_ne!(pricing.run("Which order?").await.session_id, first.session_id);
    assert_eq!(s.provider.requests()[1].messages.len(), 3);
}

#[tokio::test]
async fn a_remote_failure_is_a_result_not_an_error() {
    let s = served_with(vec![tool_call("nope", json!({})), tool_call("nope", json!({}))], None, vec![], Some(2), |b| b).await;
    let harness = Harness::testing(None);
    let far = RemoteAgent::builder(&s.url).harness(harness.clone()).connect().await.unwrap();
    let result = far.run("Go.").await;
    assert!(!result.ok());
    assert!(result.error.as_deref().unwrap().contains("MaxStepsExceeded"), "{:?}", result.error);
    assert_eq!(harness.audit.entries().last().unwrap().action, "a2a_call");

    // A gateway that only says 502.
    let gateway = axum::Router::new().fallback(|| async { (axum::http::StatusCode::BAD_GATEWAY, "bad gateway") });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bad = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, gateway).await.unwrap() });
    let gone = RemoteAgent::builder(&bad).name("gone").retries(0).build().unwrap();
    let result = gone.run("Hello?").await;
    assert!(!result.ok() && result.error.as_deref().unwrap().contains("A2AError"));
    let err = RemoteAgent::connect(&bad).await.unwrap_err();
    assert!(err.message.contains("no A2A agent card"), "{err}");
}

#[tokio::test]
async fn the_client_reads_an_agent_that_answers_with_a_message() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let card = json!({"name": "Echo Agent", "url": format!("{base}/rpc")});
    let plain = axum::Router::new().route("/.well-known/agent-card.json", get(move || async move { axum::Json(card) })).route(
        "/rpc",
        post(|| async {
            axum::Json(json!({"jsonrpc": "2.0", "id": 1, "result": {
                "kind": "message", "role": "agent", "messageId": "r1", "contextId": "c1", "parts": [{"kind": "text", "text": "echo"}]}}))
        }),
    );
    tokio::spawn(async move { axum::serve(listener, plain).await.unwrap() });
    let echo = RemoteAgent::connect(&base).await.unwrap();
    assert_eq!(echo.name(), "Echo_Agent");
    let result = echo.run("hi").await;
    assert!(result.ok());
    assert_eq!((result.output.as_str(), result.session_id.as_str()), ("echo", "c1"));
}

#[tokio::test]
async fn a_stream_carries_a_conversation_on_over_http() {
    let s = served(&["Your order shipped on Thursday.", "It arrives Monday."]).await;
    let client = A2AClient::new(&s.url);
    assert_eq!(client.card(false).await.unwrap()["url"], format!("{}/", s.url));
    let task = client.send("Where is my order?", &send()).await.unwrap();
    let events = collect(client.stream("When does it arrive?", &send().context_id(&task.context_id)).await.unwrap()).await;
    assert_eq!(task.text, "Your order shipped on Thursday.");
    assert_eq!(events.iter().map(|e| e.text.as_str()).collect::<String>(), "It arrives Monday.");
    let last = events.last().unwrap();
    assert!(last.is_final && last.state == "completed");
}
