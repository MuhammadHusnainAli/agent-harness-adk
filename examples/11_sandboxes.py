"""Cowork inside a sandbox: the agent writes a script, runs it, hands the file back.

Runs in a Docker container when Docker and the image are to hand, and in a plain
local shell otherwise — the agent, and everything it calls, is the same in both.
Swap the one line for `sandbox("e2b")`, `sandbox("daytona")` or `sandbox("modal")`
to do the same work in a hosted one.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile

from _common import pick_provider

from agent_harness import Agent, Harness, available_sandboxes, sandbox, tool_call

IMAGE = "alpine:latest"

SCRIPT = [
    tool_call("todo_write", todos=[{"content": "Write the script", "status": "in_progress"},
                                   {"content": "Run it", "status": "pending"}]),
    tool_call("fs_write", path="count.sh",
              content="for word in plan staff run review; do echo $word; done | wc -l\n"),
    tool_call("shell", command="sh count.sh > stages.txt && cat stages.txt"),
    tool_call("todo_write", todos=[{"content": "Write the script", "status": "done"},
                                   {"content": "Run it", "status": "done"}]),
    "There are 4 stages. The script is count.sh and its output is in stages.txt.",
]


def have_image() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "image", "inspect", IMAGE],
                              capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


async def main() -> None:
    provider, model = pick_provider(SCRIPT)
    ready = [name for name, ok in available_sandboxes().items() if ok]
    print(f"sandboxes ready here: {', '.join(ready)}\n")

    if have_image():
        workspace = sandbox("docker", image=IMAGE)          # no network, 2 CPUs, 2 GB
    else:
        # Not contained by anything — a stand-in so the example runs anywhere.
        workspace = sandbox("command", workdir=tempfile.mkdtemp(), isolated=True)

    report = await workspace.check()
    print(f"{report['sandbox']} sandbox {report['id']}: "
          f"{'works' if report['ok'] else 'does not work'} ({report['seconds']}s)")

    harness = Harness()
    if provider is not None:
        harness.provider = provider
    agent = Agent("colleague", mode="cowork", depth="fast", model=model,
                  workspace=workspace, harness=harness)

    result = await agent.run("Count the stages of the pipeline with a shell script.")
    print(f"\n{result.output}")
    for artifact in result.artifacts:
        print(f"  {artifact.name}: {artifact.content.strip()!r}  → {artifact.path}")

    await harness.aclose()          # stops the sandbox
    print(f"\nsandbox stopped: {not workspace.sandbox.info()['running']}")


if __name__ == "__main__":
    asyncio.run(main())
