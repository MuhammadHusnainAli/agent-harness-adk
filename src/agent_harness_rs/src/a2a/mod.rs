//! Agent-to-agent (A2A): serve your agents to others, and call theirs.
//!
//! ```no_run
//! use agent_harness::a2a::{A2AClient, A2AServer, RemoteAgent, SendOptions};
//! # async fn demo(agent: agent_harness::Agent) -> Result<(), Box<dyn std::error::Error>> {
//! let server = A2AServer::builder(agent).token("sk-…").build()?;
//! let (url, _running) = server.spawn_local().await?;
//!
//! let task = A2AClient::new(&url).token("sk-…").send("What does it cost?", &SendOptions::default()).await?;
//! let remote = RemoteAgent::builder(&url).token("sk-…").connect().await?;
//! let manager = agent_harness::Agent::builder("manager").tool(remote.as_tool(None, None)).build()?;
//! # Ok(()) }
//! ```
//!
//! Both sides speak the JSON-RPC binding of A2A 0.3, with no SDK: the server
//! is an `axum` router and the client is `reqwest`. The wire format, field
//! names and error codes are the Python harness's, so either side talks to the
//! other language's.

pub mod client;
pub mod protocol;
pub mod server;
pub mod store;

pub use client::{A2AClient, A2AEvent, Outgoing, RemoteAgent, RemoteAgentBuilder, RemoteTask, SendOptions};
pub use protocol::{A2AError, PROTOCOL_VERSION};
pub use server::{A2AAuthFn, A2AServer, A2AServerBuilder, AgentFactory, Principal, Reply, Served};
pub use store::{MemoryTaskStore, SessionTaskStore, TaskRecord, TaskStore};
