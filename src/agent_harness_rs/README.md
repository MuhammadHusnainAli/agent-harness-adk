# agent-harness-adk (Rust)

A fast, lightweight harness for building production AI agents in Rust: agents,
tools, skills, prompts, model providers, and the runtime rails underneath them
(permissions, budgets, hooks, sessions).

This is the Rust port of the Python package
[`agent-harness-adk`](https://pypi.org/project/agent-harness-adk/). The concepts
and names match, and so does the JSON: a session saved by one can be read by
the other. The port is happening milestone by milestone; [`track.md`](track.md)
shows what is done and what is pending.

```toml
[dependencies]
agent-harness-adk = "0.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
schemars = "1"
serde = { version = "1", features = ["derive"] }
```

```rust
use agent_harness::{Agent, Tool, ToolContext};
use schemars::JsonSchema;
use serde::Deserialize;

/// Look up the status of a customer order.
#[derive(Deserialize, JsonSchema)]
struct OrderStatus {
    /// the order number, digits only.
    order_id: String,
}

#[tokio::main]
async fn main() -> agent_harness::Result<()> {
    let order_status = Tool::new("order_status", |a: OrderStatus, _: ToolContext| async move {
        Ok(format!("order {} shipped", a.order_id))
    });
    let agent = Agent::builder("support")
        .instructions("Answer customer questions about orders. Look the order up before answering.")
        .tool(order_status)
        .build()?;

    let result = agent.run("Where is order 4182?").await?;
    println!("{} {} {}", result.output, result.cost_usd(), result.steps);
    Ok(())
}
```

Python reads a function's signature and docstring to build the tool schema. In
Rust the arguments are a struct deriving `Deserialize + JsonSchema`. Its doc
comments become the descriptions the model sees, and serde validates the
model's arguments before your function runs.

Run the example offline (scripted provider), or against Claude:

```bash
cargo run --example quickstart
ANTHROPIC_API_KEY=sk-ant-... cargo run --example quickstart -- --live
```

## Providers

`anthropic`, `openai`, `gemini`, and every OpenAI-compatible host: `openrouter`,
`groq`, `together`, `deepseek`, `mistral`, `xai`, `fireworks`, `cerebras`,
`ollama`, `lmstudio`, `vllm`, `openai-compatible`. They are pure HTTP with no
vendor SDKs, and every backend shares one transport: retries with backoff,
every `Retry-After` variant, typed errors, a circuit breaker, a concurrency cap
and stats. A `fake` provider is scripted, for tests.

```rust
let agent = Agent::builder("fast")
    .model("llama-3.3-70b-versatile")
    .provider_name("groq") // reads GROQ_API_KEY
    .build()?;
```

## Composition

- **Sub-agents.** `.subagent(agent)` or `.subagent_spec(spec)` gives the agent a
  `delegate` tool; `.runtime_agents(true)` lets it write and run new
  specialists (`spawn_agent`) during a run, under a sub-agent cap.
- **Handoffs.** `.handoff(billing)` lets an agent hand the *conversation* on.
  The next turn goes straight to whoever has it, and a handoff can pass the
  full history, only what was said, or only the latest message.
- **Modes.** `modes::chat`, `research` (a source ledger, citations checked
  against it, a report appended with its sources) and `cowork` (todos,
  questions to the person, helpers), at `fast` / `balanced` / `deep`.
- **Versions.** `.add_version("v1", …)` and `agent.use_version("v1")` let you
  compare versions on the same harness.
- **Blueprints.** A whole agent tree in one YAML/JSON file.
- **Workflows.** Declared steps (`agent`, `tool`, `parallel`, `foreach`,
  `loop`, `if`, `switch`, `graph`, …) with safe `{{ … }}` expressions, run as
  written.
- **Orchestrator.** Plan → staff (bench or factory) → run in waves → check each
  hand-back → re-plan → consolidate → independent critics → rework. Jobs are
  checkpointed and resumable.
- **MCP.** Use any MCP server's tools (`McpManager`, stdio or HTTP), and serve
  your agents as MCP tools (`McpAgentServer`, per-agent API keys, streamable
  HTTP with progress, or stdio).

## Memory & knowledge

- **Memory** is on by default. `user.md` (what is true about the person) is in
  every prompt, relevant precedent is recalled for each task, and the agent
  can `remember` and `recall` for itself. `.trace("alice")` (or a `Trace` with
  a tenant and session) keeps one user's memory from another's on one store.
  A `memory_write` hook sees every durable write, and can refuse or rewrite it.
- **Stores**: in memory, files (`Harness::local`), SQLite, Postgres, MySQL,
  Redis, MongoDB, DynamoDB, Elasticsearch, S3, GCS, Azure Blob, or your own
  HTTP API — `memory_provider(url)`. `Harness::on(url)` keeps memory and chat
  sessions in one database.
- **Knowledge bases**: `KnowledgeBase::new("qdrant://host:6333/handbook")?`,
  `kb.add(NewDocument::path("refunds.md"))`, `kb.as_tool("search_knowledge")`.
  Passages are cut at headings and sentences, searched by meaning and by
  shared words, over 17 vector databases (`vector_stores()`).

Database drivers are Cargo features: `sqlite` (default), `postgres`, `mysql`,
`redis`, `mongodb`. Everything spoken to over HTTP is always built in.

## Rails

- **Budgets.** `Budget { max_usd, max_tokens, max_steps, max_tool_calls, max_seconds, .. }`.
  By default hitting a ceiling ends the run cleanly with what it produced.
- **Permissions.** `PolicyGate::new(Permission::Allow).deny(["shell"]).ask(["http_*"])`.
- **Hooks.** `pre_model`, `model_egress`, `pre_tool`, `post_tool`, `run_start`, … block or rewrite.
- **Sessions.** In memory or one JSON file each; versioned, so a stale save is merged or forked rather than lost.
- **Output contracts.** `.output_type::<T>()`: the answer must deserialise into `T`, with retries.
- **Streaming.** `agent.stream(task, opts)` yields tokens, tool calls and results, then `run_end`.

## License

MIT
