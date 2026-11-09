//! MCP both ways: our client against our server, over HTTP and stdio.

use std::sync::Arc;

use agent_harness::llm_providers::{FakeProvider, Script, tool_call};
use agent_harness::mcp::{ExposeTools, McpAgentServer, McpClient, McpManager, McpServer};
use agent_harness::types::ContentBlock;
use agent_harness::{Agent, Harness, NoArgs, Tool};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

fn looping(name: &str, reply: &str) -> Agent {
    let p = FakeProvider::new(vec![Script::text(reply)]).looping();
    Agent::builder(name).description(format!("Handles {name}.")).provider(Arc::new(p)).build().unwrap()
}

fn args(v: Value) -> serde_json::Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}

#[tokio::test]
async fn http_keys_open_only_their_agents() {
    let server = McpAgentServer::builder([looping("billing", "refund issued"), looping("orders", "order shipped")])
        .agent_key("billing", "sk-billing-0001")
        .agent_key("orders", "sk-orders-0001")
        .build()
        .unwrap();
    let (url, _task) = server.spawn_local().await.unwrap();

    let billing = McpClient::new(McpServer::http("pay", &url).header("authorization", "Bearer sk-billing-0001"));
    billing.connect().await.unwrap();
    let tools = billing.list_tools(false).await.unwrap();
    let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert_eq!(names, vec!["billing"]);
    assert_eq!(tools[0]["inputSchema"]["required"], json!(["task"]));
    assert!(tools[0]["description"].as_str().unwrap().starts_with("Ask the billing agent. Handles billing."));
    assert_eq!(billing.call_tool("billing", args(json!({"task": "refund 42"}))).await.unwrap(), "refund issued");
    let hidden = billing.call_tool("orders", args(json!({"task": "x"}))).await.unwrap_err().to_string();
    assert!(hidden.contains("Unknown tool: 'orders'"), "{hidden}");

    let wrong = McpClient::new(McpServer::http("bad", &url).header("x-api-key", "sk-nope-00000"));
    let err = wrong.connect().await.unwrap_err().to_string();
    assert!(err.contains("returned 401") && err.contains("that API key is not valid"), "{err}");

    let one = McpClient::new(McpServer::http("one", &url.replace("/mcp", "/orders/mcp")).header("x-api-key", "sk-billing-0001"));
    assert!(one.connect().await.unwrap_err().to_string().contains("returned 403"));

    agent_harness::install_tls();
    let health: Value = reqwest::get(url.replace("/mcp", "/healthz")).await.unwrap().json().await.unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["refused"], 2);
}

#[tokio::test]
async fn an_agent_uses_another_through_mcp() {
    let server = McpAgentServer::builder([looping("weather", "sunny, 21C")]).build().unwrap();
    let (url, _task) = server.spawn_local().await.unwrap();
    let mcp = McpManager::new([McpServer::http("remote", &url)]).connect().await;
    assert!(mcp.errors.is_empty(), "{:?}", mcp.errors);
    let tools = mcp.tools();
    assert_eq!(tools[0].name, "remote_weather");
    assert!(tools[0].tags.contains("mcp"));

    let p = Arc::new(FakeProvider::new(vec![tool_call("remote_weather", json!({"task": "weather in Lahore"})), Script::text("It is sunny.")]));
    let boss = Agent::builder("boss").provider(p.clone()).tools(tools).build().unwrap();
    let r = boss.run("what's the weather?").await.unwrap();
    assert_eq!(r.output, "It is sunny.");
    match &p.requests()[1].messages.last().unwrap().content[0] {
        ContentBlock::ToolResult(res) => assert_eq!(res.content, "sunny, 21C"),
        other => panic!("{other:?}"),
    }
    mcp.close().await;
}

#[tokio::test]
async fn conversations_carry_on_between_calls() {
    let p = Arc::new(FakeProvider::new(vec![Script::text("noted"), Script::text("you said hi")]));
    let a = Agent::builder("chat").provider(p.clone()).build().unwrap();
    let server = McpAgentServer::builder([a]).build().unwrap();
    let (url, _task) = server.spawn_local().await.unwrap();
    let c = McpClient::new(McpServer::http("c", &url));
    c.connect().await.unwrap();
    c.call_tool("chat", args(json!({"task": "hi", "conversation_id": "conv-1"}))).await.unwrap();
    c.call_tool("chat", args(json!({"task": "what did I say?", "conversation_id": "conv-1"}))).await.unwrap();
    let second = &p.requests()[1];
    assert_eq!(second.messages.len(), 3, "the second call sees the first");
    let bad = c.call_tool("chat", args(json!({"task": "x", "conversation_id": 7}))).await.unwrap_err();
    assert!(bad.to_string().contains("`conversation_id` is a short string"));
}

#[tokio::test]
async fn progress_streams_as_server_sent_events() {
    let server = McpAgentServer::builder([looping("slow", "finished")]).build().unwrap();
    let (url, _task) = server.spawn_local().await.unwrap();
    let body = json!({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                      "params": {"name": "slow", "arguments": {"task": "go"}, "_meta": {"progressToken": "p1"}}});
    agent_harness::install_tls(); // our own reqwest client needs a TLS backend
    let resp = reqwest::Client::new().post(&url).header("accept", "application/json, text/event-stream").body(body.to_string()).send().await.unwrap();
    assert_eq!(resp.headers()["content-type"], "text/event-stream");
    let text = resp.text().await.unwrap();
    let events: Vec<Value> = text.lines().filter_map(|l| l.strip_prefix("data: ")).map(|d| serde_json::from_str(d).unwrap()).collect();
    assert_eq!(events[0]["method"], "notifications/progress");
    assert_eq!(events[0]["params"]["message"], "step 1");
    let last = events.last().unwrap();
    assert_eq!(last["id"], 9);
    assert_eq!(last["result"]["content"][0]["text"], "finished");
}

#[tokio::test]
async fn exposes_an_agents_own_tools() {
    let ping = Tool::new("ping", |_: NoArgs, _| async { Ok("pong") });
    let a = Agent::builder("ops").provider(Arc::new(FakeProvider::new(vec![]))).tool(ping).build().unwrap();
    let server = McpAgentServer::builder([a]).expose_tools(ExposeTools::All).build().unwrap();
    let names: Vec<String> = server.tools(None).iter().map(|t| t["name"].as_str().unwrap().to_string()).collect();
    assert_eq!(names, vec!["ops", "ops_ping"]);
    let reply = server.handle(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "ops_ping", "arguments": {}}})).await.unwrap();
    assert_eq!(reply["result"]["content"][0]["text"], "pong");
}

#[tokio::test]
async fn protocol_edges() {
    let server = McpAgentServer::builder([looping("a", "x")]).name("demo").instructions("Be nice.").build().unwrap();
    let init = server.handle(json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})).await.unwrap();
    assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
    assert_eq!(init["result"]["serverInfo"]["name"], "demo");
    assert_eq!(init["result"]["instructions"], "Be nice.");
    let newest = server.handle(json!({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "1999"}})).await.unwrap();
    assert_eq!(newest["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(server.handle(json!({"jsonrpc": "2.0", "id": 3, "method": "ping"})).await.unwrap()["result"], json!({}));
    assert_eq!(server.handle(json!({"jsonrpc": "2.0", "id": 4, "method": "nope"})).await.unwrap()["error"]["code"], -32601);
    assert_eq!(server.handle(json!({"id": 5, "method": "ping"})).await.unwrap()["error"]["code"], -32600);
    assert!(server.handle(json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).await.is_none());
    let no_task = server.handle(json!({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "a", "arguments": {}}})).await.unwrap();
    assert_eq!(no_task["error"]["message"], "`task` is required: what should the agent do?");

    assert!(McpAgentServer::builder([looping("a", "x")]).api_key("short").build().unwrap_err().to_string().contains("at least 8 characters"));
    let locked = McpAgentServer::builder([looping("a", "x"), looping("b", "y")]).agent_key("a", "sk-aaaaaaaa").build().unwrap_err();
    assert!(locked.to_string().starts_with("b would be served with no key that opens it"), "{locked}");
    McpAgentServer::builder([looping("a", "x"), looping("b", "y")]).agent_key("a", "sk-aaaaaaaa").public("b").build().unwrap();
}

#[tokio::test]
async fn stdio_server_over_pipes() {
    let server = McpAgentServer::builder([looping("helper", "done it")]).build().unwrap();
    let (client_end, server_end) = tokio::io::duplex(64 * 1024);
    let (server_read, server_write) = tokio::io::split(server_end);
    let serving = tokio::spawn(async move { server.serve_stdio(server_read, server_write, None).await });
    let (client_read, mut client_write) = tokio::io::split(client_end);
    let mut lines = BufReader::new(client_read).lines();
    client_write.write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}\nnot json\n").await.unwrap();
    client_write
        .write_all(
            format!("{}\n", json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "helper", "arguments": {"task": "go"}}})).as_bytes(),
        )
        .await
        .unwrap();
    let mut seen = Vec::new();
    while seen.len() < 3 {
        let line = lines.next_line().await.unwrap().unwrap();
        seen.push(serde_json::from_str::<Value>(&line).unwrap());
    }
    let by_id = |id: i64| seen.iter().find(|m| m["id"] == id).cloned().unwrap();
    assert_eq!(by_id(1)["result"]["tools"][0]["name"], "helper");
    assert_eq!(by_id(2)["result"]["content"][0]["text"], "done it");
    assert!(seen.iter().any(|m| m["error"]["code"] == -32700));
    client_write.shutdown().await.unwrap(); // EOF: the client has gone
    serving.await.unwrap().unwrap();
}

/// Not a test on its own: the child process `stdio_client_spawns_a_server` starts.
#[tokio::test]
async fn stdio_child() {
    if std::env::var("AH_MCP_CHILD").is_err() {
        return;
    }
    let server = McpAgentServer::builder([looping("echo", "child says hi")]).build().unwrap();
    // libtest has printed `test stdio_child ... ` with no newline; start clean.
    let mut out = tokio::io::stdout();
    out.write_all(
        b"
",
    )
    .await
    .unwrap();
    out.flush().await.unwrap();
    server.serve_stdio(tokio::io::stdin(), tokio::io::stdout(), None).await.unwrap();
}

#[tokio::test]
async fn stdio_client_spawns_a_server() {
    let exe = std::env::current_exe().unwrap();
    let server = McpServer::stdio("kid", exe.to_str().unwrap(), ["stdio_child", "--exact", "--nocapture", "--test-threads=1"]).env("AH_MCP_CHILD", "1");
    let client = McpClient::new(server);
    client.connect().await.unwrap();
    let tools = client.list_tools(false).await.unwrap();
    assert_eq!(tools[0]["name"], "echo");
    let tool = &client.as_tools()[0];
    assert_eq!(tool.name, "kid_echo");
    let out = tool.run("c1", args(json!({"task": "hello"})), Default::default()).await;
    assert_eq!(out.content, "child says hi");
    client.close().await;
    let _ = Harness::new();
}
