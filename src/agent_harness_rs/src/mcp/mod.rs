//! Model Context Protocol, both ways: use other servers' tools
//! ([`McpManager`]), and serve your agents as tools to any MCP client
//! ([`McpAgentServer`]). Stdio and streamable HTTP, no SDK needed.

pub mod client;
pub mod server;

pub use client::{McpClient, McpManager, McpServer, Transport};
pub use server::{ApiKey, Claims, ExposeTools, McpAgentServer, McpServerBuilder};
