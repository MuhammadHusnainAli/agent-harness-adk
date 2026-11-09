//! The 60-second example from the README.
//!
//! Runs offline against the scripted provider by default. Set
//! `ANTHROPIC_API_KEY` (or pass `--live`) to talk to a real model.
//!
//!     cargo run --example quickstart
//!     ANTHROPIC_API_KEY=sk-ant-... cargo run --example quickstart -- --live

use std::sync::Arc;

use agent_harness::llm_providers::{FakeProvider, Script, tool_call};
use agent_harness::{Agent, Tool, ToolContext};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

/// Look up the status of a customer order.
#[derive(Deserialize, JsonSchema)]
struct OrderStatus {
    /// the order number, digits only.
    order_id: String,
}

#[tokio::main]
async fn main() -> agent_harness::Result<()> {
    let order_status = Tool::new("order_status", |args: OrderStatus, _ctx: ToolContext| async move {
        Ok(format!("Order {} shipped on Tuesday and arrives Friday.", args.order_id))
    });

    let live = std::env::args().any(|a| a == "--live");
    let mut builder = Agent::builder("support").instructions("Answer customer questions about orders. Look the order up before answering.").tool(order_status);
    if !live {
        builder = builder.provider(Arc::new(FakeProvider::new(vec![
            tool_call("order_status", json!({"order_id": "4182"})),
            Script::text("Order 4182 shipped on Tuesday and should arrive on Friday."),
        ])));
    }
    let agent = builder.build()?;

    let result = agent.run("Where is order 4182?").await?;
    println!("{}", result.output);
    println!("steps={} cost=${:.6} tools={:?}", result.steps, result.cost_usd(), result.tool_calls.iter().map(|c| &c.name).collect::<Vec<_>>());
    Ok(())
}
