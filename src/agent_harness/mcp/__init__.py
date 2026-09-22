"""Model Context Protocol support: stdio and streamable HTTP, no SDK needed."""

from .client import MCPClient, MCPManager, MCPServer

__all__ = ["MCPServer", "MCPClient", "MCPManager"]
