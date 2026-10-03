"""Model Context Protocol, both ways: use other servers' tools (`MCPManager`),
and serve your agents as tools to any MCP client (`MCPAgentServer`). Stdio and
streamable HTTP, no SDK needed."""

from .client import MCPClient, MCPManager, MCPServer
from .server import MCPAgentServer

__all__ = ["MCPServer", "MCPClient", "MCPManager", "MCPAgentServer"]
