"""Sandboxed compute: run code, get the answer, without it touching your machine.

The agent writes Python, it runs inside the workspace, and only what it printed
comes back. In a sandbox (`agent_harness.sandboxes`) or the Docker workspace the
process is contained. With the local backend it is a plain subprocess in a
jailed directory, which is weaker: use a sandbox for anything you did not write.

On this machine the tool asks for approval before every run. That is
deliberate. Inside a sandbox, which is its own machine, it does not.
"""

from __future__ import annotations

from typing import Any

from ..errors import ToolError
from ..tools import Tool, tool

__all__ = ["make_python_tool"]

PREAMBLE = (
    "import json, math, statistics, re, csv, datetime\n"
    "from pathlib import Path\n"
)


def make_python_tool(workspace: Any, *, name: str = "run_python",
                     timeout: float = 60.0, preamble: str = PREAMBLE) -> Tool:
    """A sandboxed Python tool bound to a workspace.

    Args:
        workspace: the `Workspace` the code runs inside.
        name: the tool name the model sees.
        timeout: seconds before the process is killed.
        preamble: imports made available to the snippet.
    """
    if workspace is None:
        raise ToolError("run_python needs a workspace to run inside",
                        tool=name)

    @tool(name=name, tags=["builtin", "compute"],
          permission="allow" if getattr(workspace, "isolated", False) else "ask")
    async def run_python(code: str, timeout_s: float = timeout) -> dict[str, Any]:
        """Run Python inside the workspace and return what it printed.

        The snippet starts in the workspace directory with json, math,
        statistics, re, csv, datetime and pathlib already imported. Print what
        you want back — nothing else is returned.

        Args:
            code: the Python to run.
            timeout_s: seconds before it is killed.
        """
        if not code.strip():
            raise ToolError("no code to run", tool=name)
        script = f"{preamble}\n{code}\n"
        await workspace.awrite("_snippet.py", script)
        result = await workspace.shell(f"{workspace.python} _snippet.py",
                                       timeout=timeout_s)
        if result["returncode"] != 0:
            return {
                "ok": False,
                "stdout": result["stdout"],
                "error": result["stderr"][-4000:] or "the snippet exited non-zero",
                "returncode": result["returncode"],
            }
        return {"ok": True, "stdout": result["stdout"],
                "stderr": result["stderr"][-2000:] or None}

    return run_python
