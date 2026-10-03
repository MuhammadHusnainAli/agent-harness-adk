"""Agent-to-agent (A2A): serve your agents to others, and call theirs.

    from agent_harness.a2a import A2AServer, A2AClient, RemoteAgent

    app = A2AServer(agent, auth="sk-…")            # an ASGI app, or `await app.serve()`

    task = await A2AClient("https://host/agent").send("What does it cost?")
    remote = await RemoteAgent.connect("https://host/agent")
    manager = Agent("manager", subagents=[remote])

Both sides speak the JSON-RPC binding of A2A 0.3, with no SDK: the server is a
plain ASGI application and the client is `httpx`.
"""

from .client import A2AClient, A2AEvent, RemoteAgent, RemoteTask
from .protocol import PROTOCOL_VERSION, A2AError
from .server import A2AServer, Principal
from .store import MemoryTaskStore, SessionTaskStore, TaskRecord, TaskStore

__all__ = [
    "A2AServer",
    "A2AClient",
    "RemoteAgent",
    "RemoteTask",
    "A2AEvent",
    "A2AError",
    "Principal",
    "TaskStore",
    "TaskRecord",
    "MemoryTaskStore",
    "SessionTaskStore",
    "PROTOCOL_VERSION",
]
