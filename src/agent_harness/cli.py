"""The `agent-harness` command: run an agent, chat with one, inspect what happened."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .agent import Agent
from .harness import Harness
from .providers.base import MODELS
from .runtime.permissions import console_approver
from .runtime.tracing import console_exporter

__all__ = ["main"]


def _harness(args: argparse.Namespace) -> Harness:
    harness = (Harness.local(args.state) if args.state else Harness())
    if args.trace:
        harness.tracer.add_exporter(console_exporter())
    if args.approve:
        harness.policy.approver = console_approver
    return harness


def _agent(args: argparse.Namespace, harness: Harness) -> Agent:
    tools: list[Any] = []
    if args.tools:
        from .toolkits import basic_tools, http_fetch
        tools = [*basic_tools(), http_fetch]
    return Agent(
        args.name,
        args.instructions or "",
        model=args.model,
        provider=args.provider,
        harness=harness,
        tools=tools,
        skills=args.skills,
        memory=not args.no_memory,
        max_steps=args.max_steps,
    )


async def _run(args: argparse.Namespace) -> int:
    harness = _harness(args)
    agent = _agent(args, harness)
    result = None
    try:
        if args.stream:
            async for event in agent.stream(args.task, session=args.session):
                if event.type == "text":
                    print(event.text, end="", flush=True)
                elif event.type == "tool_result":
                    print(f"\n  · {event.data.get('tool')} → {event.text[:80]}",
                          file=sys.stderr)
                elif event.type == "run_end":
                    result = event.data["result"]
            print()
        else:
            result = await agent.run(args.task, session=args.session)
            print(result.output)

        if result is None:
            print("the run produced no result", file=sys.stderr)
            return 1
        if result.error:
            print(f"\nerror: {result.error}", file=sys.stderr)
        if args.json:
            print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
        if args.report:
            print(json.dumps(harness.report(), indent=2), file=sys.stderr)
        print(f"\n[{result.steps} steps · ${result.cost_usd:.4f} · "
              f"session {result.session_id}]", file=sys.stderr)
        return 1 if result.error else 0
    finally:
        await harness.aclose()


async def _chat(args: argparse.Namespace) -> int:
    harness = _harness(args)
    agent = _agent(args, harness)
    session = args.session
    print(f"agent-harness {__version__} · {agent.name} on {agent.model}")
    print("Type /exit to leave, /new for a fresh session, /cost for the bill.\n")
    try:
        while True:
            try:
                line = await asyncio.to_thread(input, "you › ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue
            if line in {"/exit", "/quit"}:
                break
            if line == "/new":
                session = None
                print("(new session)")
                continue
            if line == "/cost":
                print(json.dumps(harness.report()["budget"], indent=2))
                continue
            result = await agent.run(line, session=session)
            session = result.session_id
            print(f"\n{agent.name} › {result.output or result.error}\n")
        summary = await agent.close_session()
        if summary:
            print("(memory updated)", file=sys.stderr)
        return 0
    finally:
        await harness.aclose()


def _models(args: argparse.Namespace) -> int:
    rows = sorted(MODELS.values(), key=lambda m: (m.provider, m.id))
    print(f"{'model':<26} {'provider':<10} {'tier':<9} {'in $/M':>8} {'out $/M':>8} "
          f"{'context':>10}")
    for info in rows:
        print(f"{info.id:<26} {info.provider:<10} {info.tier:<9} "
              f"{info.input_cost:>8.2f} {info.output_cost:>8.2f} "
              f"{info.context_window:>10,}")
    return 0


async def _sessions(args: argparse.Namespace) -> int:
    harness = Harness.local(args.state or ".harness")
    rows = await harness.sessions.list(limit=args.limit)
    if args.show:
        session = await harness.sessions.load(args.show)
        for message in session.messages:
            print(f"{message.role}: {message.text[:2000]}")
        return 0
    if not rows:
        print("no sessions yet")
        return 0
    for session in rows:
        print(f"{session.id}  {session.agent:<16} {len(session.messages):>3} messages  "
              f"${session.usage.cost_usd:.4f}")
    return 0


async def _mcp(args: argparse.Namespace) -> int:
    from .mcp import MCPManager, MCPServer

    if args.url:
        server = MCPServer(name=args.server_name, url=args.url, transport="http")
    else:
        server = MCPServer(name=args.server_name, command=args.command[0],
                           args=args.command[1:])
    async with MCPManager([server]) as manager:
        if manager.errors:
            print("\n".join(manager.errors), file=sys.stderr)
            return 1
        for entry in manager.tools():
            print(f"{entry.name}\n    {entry.description}")
    return 0


def _journal(args: argparse.Namespace) -> int:
    from .runtime.journal import RunJournal

    path = Path(args.state or ".harness") / "journal.jsonl"
    if not path.exists():
        print(f"no journal at {path}", file=sys.stderr)
        return 1
    print(RunJournal.load(path).render(limit=args.limit))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-harness",
        description="Run and inspect agents built with agent-harness.",
    )
    parser.add_argument("--version", action="version", version=f"agent-harness {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--name", default="assistant", help="what to call the agent")
        p.add_argument("--model", default=None, help="model id (routed by default)")
        p.add_argument("--provider", default=None,
                       help="anthropic | openai | gemini (inferred from the model)")
        p.add_argument("--instructions", default="", help="the agent's instructions")
        p.add_argument("--skills", default=None, help="a directory of skills to load")
        p.add_argument("--tools", action="store_true",
                       help="give it the built-in tools (time, maths, web fetch)")
        p.add_argument("--no-memory", action="store_true", help="run without memory")
        p.add_argument("--max-steps", type=int, default=20)
        p.add_argument("--state", default=None,
                       help="persist sessions, memory and traces under this directory")
        p.add_argument("--trace", action="store_true", help="print spans as they close")
        p.add_argument("--approve", action="store_true",
                       help="ask before any tool that needs approval")
        p.add_argument("--session", default=None, help="resume a session by id")

    run = sub.add_parser("run", help="run one task and print the answer")
    run.add_argument("task")
    run.add_argument("--stream", action="store_true", help="stream the answer")
    run.add_argument("--json", action="store_true", help="also print the full result")
    run.add_argument("--report", action="store_true", help="print the run report")
    common(run)

    chat = sub.add_parser("chat", help="an interactive session")
    common(chat)

    sub.add_parser("models", help="list the models the harness knows and their prices")

    sessions = sub.add_parser("sessions", help="list or show stored sessions")
    sessions.add_argument("--state", default=None)
    sessions.add_argument("--limit", type=int, default=20)
    sessions.add_argument("--show", default=None, help="print one session's transcript")

    mcp = sub.add_parser("mcp", help="list the tools an MCP server offers")
    mcp.add_argument("command", nargs="*", help="the server command and its arguments")
    mcp.add_argument("--url", default=None, help="an HTTP MCP endpoint instead")
    mcp.add_argument("--server-name", default="server")

    journal = sub.add_parser("journal", help="print the run journal")
    journal.add_argument("--state", default=None)
    journal.add_argument("--limit", type=int, default=50)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_run(args))
        if args.command == "chat":
            return asyncio.run(_chat(args))
        if args.command == "models":
            return _models(args)
        if args.command == "sessions":
            return asyncio.run(_sessions(args))
        if args.command == "mcp":
            return asyncio.run(_mcp(args))
        if args.command == "journal":
            return _journal(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
