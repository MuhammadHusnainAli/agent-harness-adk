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
from .llm_providers.base import MODELS
from .modes import DEPTHS, MODES
from .runtime.permissions import console_approver
from .runtime.tracing import console_exporter

__all__ = ["main"]


def _harness(args: argparse.Namespace) -> Harness:
    stores = {"sessions": args.sessions} if getattr(args, "sessions", None) else {}
    harness = (Harness.local(args.state, **stores) if args.state
               else Harness(**stores))
    if args.trace:
        harness.tracer.add_exporter(console_exporter())
    if args.approve:
        harness.policy.approver = console_approver
    return harness


async def _console_ask(question: str, options: list[str]) -> str:
    """Put an agent's question to whoever is at the terminal."""
    choices = "".join(f"\n  {n}. {o}" for n, o in enumerate(options, 1))
    answer = (await asyncio.to_thread(input, f"\n? {question}{choices}\n> ")).strip()
    if answer.isdigit() and 0 < int(answer) <= len(options):
        return options[int(answer) - 1]
    return answer


def _agent(args: argparse.Namespace, harness: Harness) -> Agent:
    tools: list[Any] = []
    if args.tools:
        from .toolkits import basic_tools, http_fetch
        tools = [*basic_tools(), http_fetch]
    for spec in getattr(args, "openapi", None) or []:
        tools += _api(spec, args).tools
    mode: Any = args.mode
    if mode == "cowork" and sys.stdin.isatty():
        from . import modes

        mode = modes.cowork(args.depth, ask=_console_ask)
    workspace: Any = None
    if args.sandbox:
        from .sandboxes import sandbox

        options: dict[str, Any] = {}
        if args.sandbox_id:
            options["id"] = args.sandbox_id
        if args.keep_sandbox:
            options["keep"] = True
        if args.workspace:
            # The folder is mounted into the container, so the work lands in it.
            if args.sandbox.partition("://")[0].lower() not in ("docker", "podman"):
                raise SystemExit(
                    "--workspace with --sandbox only applies to docker and podman, "
                    "which can mount a folder; other sandboxes keep their own files")
            options["mount"] = args.workspace
        workspace = sandbox(args.sandbox, **options)
    elif args.workspace:
        from .runtime.workspace import Workspace

        workspace = Workspace(args.workspace)
    elif args.mode == "cowork" and not args.state:
        print("note: no --workspace, so this works in a temporary folder that is "
              "removed on exit", file=sys.stderr)
    return Agent(
        args.name,
        args.instructions or "",
        model=args.model,
        provider=args.provider,
        mode=mode,
        depth=args.depth if isinstance(mode, (str, type(None))) else None,
        harness=harness,
        tools=tools,
        skills=args.skills,
        memory=not args.no_memory,
        max_steps=args.max_steps,
        workspace=workspace,
        trace=({"user_id": args.user, "tenant_id": args.tenant}
               if args.user or args.tenant else None),
    )


def _api(spec: str, args: argparse.Namespace) -> Any:
    """An OpenAPI document named on the command line, as a toolkit."""
    import os

    from .toolkits.openapi import OpenAPIToolkit

    return OpenAPIToolkit(
        spec, base_url=getattr(args, "base_url", None),
        token=getattr(args, "openapi_token", None) or os.environ.get("OPENAPI_TOKEN"),
        api_key=getattr(args, "api_key", None),
        writes="ask" if getattr(args, "approve", False) else "allow")


async def _mcp_serve(args: argparse.Namespace) -> int:
    import os

    from .mcp import MCPAgentServer

    harness = _harness(args)
    keys = [*args.api_key, *filter(None, [os.environ.get("MCP_API_KEY")])]
    try:
        if args.blueprint:
            from .blueprint import Blueprint

            per_agent: dict[str, Any] = {}
            for pair in args.key_env:
                name, sep, variable = pair.partition("=")
                if not sep or not os.environ.get(variable):
                    raise SystemExit(f"--key-env takes AGENT=ENV_VAR, with the "
                                     f"variable set — got {pair!r}")
                per_agent[name] = os.environ[variable]
            server = Blueprint.from_file(args.blueprint).mcp_server(
                harness=harness, api_key=keys or None, api_keys=per_agent or None,
                expose_tools=args.expose_tools)
        else:
            server = MCPAgentServer(_agent(args, harness), api_key=keys or None,
                                    expose_tools=args.expose_tools)
        if args.stdio:
            await server.serve_stdio()
            return 0
        await server.serve(args.host, args.port, ready=lambda url: print(
            f"serving {', '.join(server.agents)} over MCP at {url}/mcp\n"
            + "".join(f"  {name:<14} {url}/{name}/mcp\n" for name in server.agents)
            + f"  keys           {len(server._keys) or 'none — anyone may call'}\n"
            "This is the development server; for production run the ASGI app "
            "under uvicorn.", file=sys.stderr))
        return 0
    finally:
        await harness.aclose()


async def _openapi(args: argparse.Namespace) -> int:
    toolkit = _api(args.spec, args)
    try:
        if not args.call:
            if args.json:
                print(json.dumps([{"name": t.name, "description": t.description,
                                   "parameters": t.parameters, **t.operation}
                                  for t in toolkit], indent=2))
            else:
                print(toolkit.describe())
            return 0
        arguments: dict[str, Any] = {}
        for pair in args.arg:
            key, sep, value = pair.partition("=")
            if not sep:
                raise SystemExit(f"--arg takes name=value — got {pair!r}")
            kind = toolkit.get(args.call).parameters["properties"].get(key, {}).get("type")
            arguments[key] = value if kind == "string" else _value(value)
        answer = await toolkit.call(args.call, **arguments)
        print(answer if isinstance(answer, str)
              else json.dumps(answer, indent=2, default=str))
        return 0
    finally:
        await toolkit.aclose()


def _footer(result: Any) -> str:
    """What a mode left behind, in a line or two for stderr."""
    lines: list[str] = []
    if result.todos:
        done = sum(1 for t in result.todos if t.status == "done")
        lines.append(f"todos: {done} of {len(result.todos)} done")
    if result.sources:
        lines.append(f"sources: {len(result.sources)}")
    files = [a.path or a.name for a in result.artifacts]
    if files:
        lines.append("files: " + ", ".join(files[:12])
                     + (f" (+{len(files) - 12} more)" if len(files) > 12 else ""))
    if result.violations:
        lines.extend(f"unmet: {v}" for v in result.violations)
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    harness = _harness(args)
    agent = _agent(args, harness)
    result = None
    try:
        if args.stream:
            async for event in agent.stream(args.task, session=args.session,
                                            attachments=args.attach):
                if event.type == "text":
                    print(event.text, end="", flush=True)
                elif event.type == "tool_result":
                    print(f"\n  · {event.data.get('tool')} → {event.text[:80]}",
                          file=sys.stderr)
                elif event.type == "progress":
                    print(f"\n  · {event.text} · {event.data.get('sources', 0)} "
                          "sources", file=sys.stderr)
                elif event.type == "run_end":
                    result = event.data["result"]
            print()
        else:
            result = await agent.run(args.task, session=args.session,
                                     attachments=args.attach)
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
        if _footer(result):
            print(f"\n{_footer(result)}", file=sys.stderr)
        where = f" · sandbox {result.sandbox_id}" if result.sandbox_id else ""
        print(f"\n[{result.steps} steps · ${result.cost_usd:.4f} · "
              f"session {result.session_id}{where}]", file=sys.stderr)
        return 1 if result.error else 0
    finally:
        await harness.aclose()


async def _voice(args: argparse.Namespace) -> int:
    """Talk to an agent: from a WAV file, or live through the microphone."""
    from .voice import OpenAISpeech, RealtimeAgent, VoiceAgent, duration_ms, unwav, wav

    args.mode = args.mode or "voice"
    harness = _harness(args)
    agent = _agent(args, harness)
    rate = 16_000
    try:
        if args.realtime:
            talker: Any = RealtimeAgent(agent, args.realtime, rate=rate,
                                        session=args.session)
        else:
            talker = VoiceAgent(
                agent, rate=rate, session=args.session, greeting=args.greeting,
                speech=OpenAISpeech(voice=args.voice, language=args.language))
        out_rate = talker.output_rate

        if args.input:
            pcm, file_rate = unwav(Path(args.input).read_bytes())
            if file_rate != rate:
                from .voice import resample
                pcm = resample(pcm, file_rate, rate)

            async def source() -> Any:
                # A little quiet at the end, so the last word is heard to end.
                audio = pcm + bytes(rate * 2)
                for offset in range(0, len(audio), 640):
                    yield audio[offset:offset + 640]
                    await asyncio.sleep(0)

            spoken = bytearray()
            async for event in talker.run(source()):
                if event.type == "transcript" and not event.data.get("partial"):
                    print(f"you   › {event.text}")
                elif event.type == "audio":
                    spoken += event.audio
                elif event.type == "turn_end" and event.text:
                    first = event.data.get("first_audio_ms")
                    took = f"  [first sound after {first:.0f} ms]" if first else ""
                    print(f"agent › {event.text}{took}")
                elif event.type == "error":
                    print(f"error: {event.text}", file=sys.stderr)
            if args.output and spoken:
                Path(args.output).write_bytes(wav(bytes(spoken), out_rate))
                print(f"\nwrote {args.output} "
                      f"({duration_ms(bytes(spoken), out_rate) / 1000:.1f}s)",
                      file=sys.stderr)
            print(f"[session {talker.session_id}]", file=sys.stderr)
            return 0

        try:
            import sounddevice
        except ImportError:
            print("live voice needs a microphone library: pip install sounddevice\n"
                  "or pass --input question.wav to talk from a file", file=sys.stderr)
            return 2

        loop = asyncio.get_running_loop()
        heard: asyncio.Queue[bytes] = asyncio.Queue()

        def captured(data: Any, frames: int, time_info: Any, status: Any) -> None:
            loop.call_soon_threadsafe(heard.put_nowait, bytes(data))

        async def microphone() -> Any:
            while True:
                yield await heard.get()

        print(f"agent-harness {__version__} · talking to {agent.name}. "
              "Ctrl-C to hang up.\n")
        with sounddevice.RawInputStream(samplerate=rate, channels=1, dtype="int16",
                                        blocksize=320, callback=captured), \
                sounddevice.RawOutputStream(samplerate=out_rate, channels=1,
                                            dtype="int16") as speaker:
            async for event in talker.run(microphone()):
                if event.type == "audio":
                    await asyncio.to_thread(speaker.write, event.audio)
                elif event.type == "interrupted":
                    speaker.abort()
                    speaker.start()
                elif event.type == "transcript" and not event.data.get("partial"):
                    print(f"you   › {event.text}")
                elif event.type == "turn_end" and event.text:
                    print(f"agent › {event.text}")
                elif event.type == "error":
                    print(f"error: {event.text}", file=sys.stderr)
        return 0
    finally:
        await harness.aclose()


async def _chat(args: argparse.Namespace) -> int:
    harness = _harness(args)
    agent = _agent(args, harness)
    session = args.session
    how = f" · {agent.mode.name} ({agent.mode.depth})" if agent.mode else ""
    print(f"agent-harness {__version__} · {agent.name} on {agent.model}{how}")
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
                agent.new_session()
                print("(new session)")
                continue
            if line == "/cost":
                print(json.dumps(harness.report()["budget"], indent=2))
                continue
            result = await agent.run(line, session=session)
            session = result.session_id
            for hop in result.handoffs:
                print(f"  · {hop.line()}", file=sys.stderr)
            print(f"\n{result.agent or agent.name} › {result.output or result.error}\n")
            if _footer(result):
                print(f"{_footer(result)}\n", file=sys.stderr)
        summary = await agent.close_session()
        if summary:
            print("(memory updated)", file=sys.stderr)
        return 0
    finally:
        await harness.aclose()


def _value(text: str) -> Any:
    """`--input amount=40` is the number 40; anything JSON cannot read is text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def _workflow(args: argparse.Namespace) -> int:
    from .blueprint import _import_tool
    from .workflow import Workflow

    tools: list[Any] = [_import_tool(path) for path in args.tool]
    if args.tools:
        from .toolkits import basic_tools, http_fetch
        tools += [*basic_tools(), http_fetch]
    harness = _harness(args)
    try:
        workflow = Workflow.from_file(args.file, tools=tools, harness=harness)
        if args.check:
            print(workflow.describe())
            return 0
        inputs: dict[str, Any] = {}
        declared = workflow.spec.input_specs()
        for pair in args.input:
            key, sep, value = pair.partition("=")
            if not sep:
                raise SystemExit(f"--input takes name=value — got {pair!r}")
            # An input declared as text stays text: an order number is not
            # a number.
            inputs[key] = (value if declared.get(key, {}).get("type") == "string"
                           else _value(value))

        def show(event: Any) -> None:
            if event.type == "step_start":
                print(f"  ▸ {event.step} ({event.kind})", file=sys.stderr)
            elif event.type == "step_skipped":
                print(f"  - {event.step} skipped", file=sys.stderr)
            elif event.type in ("step_failed", "step_retry"):
                print(f"  ✗ {event.step}: {event.text}", file=sys.stderr)

        result = await workflow.run(inputs or args.text, on_event=show)
        if args.json:
            print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
        else:
            print(result.output if isinstance(result.output, str)
                  else json.dumps(result.output, indent=2, default=str))
        if result.error:
            print(f"\nerror: {result.error}", file=sys.stderr)
        print(f"\n[{result.status} · {result.steps_run} steps · "
              f"${result.cost_usd:.4f} · {result.duration_ms / 1000:.1f}s]",
              file=sys.stderr)
        return 0 if result.ok else 1
    finally:
        await harness.aclose()


async def _a2a(args: argparse.Namespace) -> int:
    from .a2a import A2AClient, A2AServer

    if args.action == "serve":
        harness = _harness(args)
        agent = _agent(args, harness)
        server = A2AServer(agent, url=args.public_url, auth=args.token or None,
                           max_concurrency=args.max_concurrency)
        try:
            await server.serve(args.host, args.port, ready=lambda url: print(
                f"{agent.name} is served over A2A at {url}\n"
                f"  card   {url}/.well-known/agent-card.json\n"
                f"  auth   {'bearer token' if args.token else 'none — anyone may call'}\n"
                "This is the development server; for production run the ASGI app "
                "under uvicorn.", file=sys.stderr))
        finally:
            await harness.aclose()
        return 0

    if not args.target:
        raise SystemExit(f"a2a {args.action} needs the agent's URL")
    async with A2AClient(args.target, token=(args.token or [None])[0]) as client:
        if args.action == "card":
            print(json.dumps(await client.card(), indent=2))
            return 0
        if not args.text:
            raise SystemExit("a2a send needs something to send")
        if args.stream:
            state = ""
            async for event in client.stream(args.text, context_id=args.context):
                if event.kind == "artifact-update":
                    print(event.text, end="", flush=True)
                state = event.state or state
            print()
            return 0 if state == "completed" else 1
        task = await client.send(args.text, context_id=args.context)
        print(task.text or task.error or "")
        print(f"\n[{task.state} · task {task.id} · context {task.context_id}]",
              file=sys.stderr)
        return 0 if task.ok else 1


def _models(args: argparse.Namespace) -> int:
    rows = sorted(MODELS.values(), key=lambda m: (m.provider, m.id))
    print(f"{'model':<26} {'provider':<10} {'tier':<9} {'in $/M':>8} {'out $/M':>8} "
          f"{'context':>10}")
    for info in rows:
        print(f"{info.id:<26} {info.provider:<10} {info.tier:<9} "
              f"{info.input_cost:>8.2f} {info.output_cost:>8.2f} "
              f"{info.context_window:>10,}")
    return 0


async def _providers(args: argparse.Namespace) -> int:
    from .llm_providers import (
        describe_llm_provider,
        list_llm_providers,
        ping_llm_provider,
    )

    if args.name:
        spec = describe_llm_provider(args.name)
        result = await ping_llm_provider(args.name) if args.ping else None
        if args.json:
            print(json.dumps({**spec.model_dump(mode="json"),
                              **({"ping": result} if result else {})}, indent=2))
        else:
            print(spec.render())
            if result:
                state = "ok" if result["ok"] else f"failed — {result.get('error')}"
                print(f"  ping     {state} ({result.get('latency_ms', 0)} ms)")
        return 0 if result is None or result["ok"] else 1

    specs = list_llm_providers(configured_only=args.ready)
    if args.json:
        print(json.dumps([s.model_dump(mode="json") for s in specs], indent=2))
        return 0
    print(f"{'provider':<18} {'status':<12} needs")
    for spec in specs:
        needs = []
        groups: dict[str, list[str]] = {}
        for field in spec.required_fields:
            if field.one_of:
                groups.setdefault(field.one_of, []).append(field.name)
            else:
                needs.append(field.name)
        needs += [" | ".join(names) for names in groups.values()]
        status = "ready" if spec.configured else "needs setup"
        print(f"{spec.name:<18} {status:<12} {', '.join(needs) or '—'}")
    print("\nagent-harness providers <name> for every field; --ping to test the "
          "credentials.")
    return 0


async def _sandboxes(args: argparse.Namespace) -> int:
    from .sandboxes import describe_sandboxes, sandbox

    if args.name and args.check:
        workspace = sandbox(args.name)
        try:
            report = await workspace.check()
        finally:
            await workspace.aclose()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            for row in report["steps"]:
                mark = "ok  " if row["ok"] else "FAIL"
                detail = f"  {row['detail']}" if row["detail"] else ""
                print(f"{mark} {row['step']:<16} {row['seconds']:>6.2f}s{detail}")
            verdict = "works" if report["ok"] else "does not work"
            print(f"\n{report['sandbox']} {verdict} ({report['seconds']}s)")
        return 0 if report["ok"] else 1

    rows = describe_sandboxes()
    if args.name:
        wanted = args.name.partition("://")[0].lower()
        rows = [r for r in rows if r["name"] == wanted] or rows
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'sandbox':<12} {'status':<12} what it is")
    for row in rows:
        status = "ready" if row["ready"] else "needs setup"
        print(f"{row['name']:<12} {status:<12} {row['summary']}")
        for gap in row["missing"]:
            print(f"{'':<25} needs: {gap}")
    print("\nagent-harness sandboxes <name> --check starts one and proves it works.")
    return 0


async def _sessions(args: argparse.Namespace) -> int:
    from .sessions import session_backends, session_provider

    if args.backends:
        for name, ready in session_backends().items():
            print(f"{name:<10} {'ready' if ready else 'needs its driver'}")
        return 0
    store = (session_provider(args.store) if args.store
             else Harness.local(args.state or ".harness").sessions)
    try:
        if args.check:
            report = await store.check()
            for row in report["steps"]:
                mark = "ok  " if row["ok"] else "FAIL"
                detail = f"  {row['detail']}" if row["detail"] else ""
                print(f"{mark} {row['step']:<20} {row['seconds']:>7.3f}s{detail}")
            print(f"\n{report['store']} "
                  f"{'works' if report['ok'] else 'does not work'}")
            return 0 if report["ok"] else 1
        if args.show:
            session = await store.load(args.show)
            for message in session.messages:
                print(f"{message.role}: {message.text[:2000]}")
            return 0
        if args.delete:
            await store.delete(args.delete)
            print(f"deleted {args.delete}")
            return 0
        rows = await store.list(limit=args.limit, user_id=args.user,
                                tenant_id=args.tenant, agent=args.agent)
        if args.json:
            print(json.dumps([s.summary() for s in rows], indent=2))
            return 0
        if not rows:
            print("no sessions yet")
            return 0
        for session in rows:
            owner = "/".join(p for p in (session.tenant_id, session.user_id) if p)
            print(f"{session.id}  {session.agent:<14} {len(session.messages):>3} messages  "
                  f"${session.usage.cost_usd:.4f}  {owner or '-':<16} "
                  f"{session.title[:48]}")
        return 0
    finally:
        await store.aclose()


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


def _governance(args: argparse.Namespace) -> int:
    import os

    from .governance import Governance, HMACSigner, list_packs, load_pack
    from .runtime.audit import AuditTrail

    if args.action == "packs":
        rows = [load_pack(p) for p in list_packs()]
        if args.json:
            print(json.dumps([{"id": p.id, "name": p.name, "jurisdiction": p.jurisdiction,
                               "kind": p.kind, "binding": p.binding, "as_of": p.as_of}
                              for p in rows], indent=2))
            return 0
        for p in rows:
            kind = p.kind if p.binding else f"{p.kind}, voluntary"
            print(f"{p.id:<19} {p.jurisdiction:<8} {kind:<22} {p.name}")
        return 0

    if args.action == "pack":
        if not args.target:
            print("usage: agent-harness governance pack <id>", file=sys.stderr)
            return 2
        p = load_pack(args.target)
        print(f"{p.name}\n  {p.jurisdiction} · {p.kind} · "
              f"{'binding' if p.binding else 'voluntary'} · checked {p.as_of}")
        print(f"  status   {' '.join(p.status.split())}")
        print("\nrequirements")
        for r in p.requirements:
            where = "manual" if r.manual else ", ".join(r.controls)
            print(f"  {r.ref:<28} {r.title}  [{where}]")
        if p.policy.rules:
            print("\nrules")
            for rule in p.policy.rules:
                print(f"  {rule.id}: {rule.effect} on {', '.join(rule.on)} — {rule.reason}")
        if p.policy.residency and p.policy.residency.transfers:
            print("\nresidency")
            for origin, dest in p.policy.residency.transfers.items():
                print(f"  {origin} → {', '.join(dest)}")
        if p.incidents:
            print("\nincident clocks")
            for kind, clocks in p.incidents.items():
                for c in clocks:
                    print(f"  {kind}: {c.get('notify')} within {c.get('within')} h "
                          f"({c.get('basis', '')})")
        print("\nsources\n" + "\n".join(f"  {u}" for u in p.sources))
        return 0

    if args.action == "verify":
        path = Path(args.target or Path(args.state or ".harness") / "audit.jsonl")
        if not path.exists():
            print(f"no audit trail at {path}", file=sys.stderr)
            return 1
        signer = None
        if args.key_env:
            if not os.environ.get(args.key_env):
                print(f"${args.key_env} is not set", file=sys.stderr)
                return 1
            signer = HMACSigner(os.environ[args.key_env])
        trail = AuditTrail.load(path, signer=signer)
        ok, why = trail.verify()
        signed = "signed, signatures checked" if signer else "signatures not checked"
        print(f"{path}: {len(trail)} entries, {why} ({signed})")
        return 0 if ok else 1

    if args.action == "check":
        if not args.config:
            print("usage: agent-harness governance check --config governance.yaml",
                  file=sys.stderr)
            return 2
        gov = Governance.from_file(args.config)     # raises on anything invalid
        policy = gov.policy
        print(f"ok   {args.config}: policy {policy.name}@{policy.version} "
              f"sha256:{gov.engine.hash[:16]} · mode {gov.mode}")
        print(f"     packs: {', '.join(p.id for p in gov.packs) or 'none'}")
        print(f"     rules: {len(policy.rules)} · purposes: "
              f"{', '.join(sorted(policy.purposes)) or 'none'}")
        residency = policy.residency
        if residency and residency.transfers:
            print(f"     residency: home {residency.home or '(per principal)'}; "
                  + "; ".join(f"{o} → {', '.join(d)}"
                              for o, d in sorted(residency.transfers.items())))
        warnings = []
        if residency and residency.home is None and sum(
                "*" not in d for d in residency.transfers.values()) > 1:
            warnings.append("no `home:` — a person with no jurisdiction tag is unrestricted")
        if gov.signer is None:
            warnings.append("no signing key — the audit trail is hash-chained but unsigned")
        if gov.vault.path is None:
            warnings.append("no `vault:` path — the subject index is lost on restart, "
                            "so erasure cannot find earlier sessions")
        for warning in warnings:
            print(f"warn {warning}")
        return 0

    if args.action in ("inventory", "dsar"):
        if not args.config:
            print(f"usage: agent-harness governance {args.action} --config governance.yaml",
                  file=sys.stderr)
            return 2
        gov = Governance.from_file(args.config)

    if args.action == "inventory":
        if not args.agents:
            print("inventory needs --agents agents.yaml (a blueprint)", file=sys.stderr)
            return 2
        from .blueprint import Blueprint

        harness = Harness.testing(governance=gov)
        Blueprint.from_file(args.agents).build_all(harness=harness)
        if args.format == "json":
            print(json.dumps(gov.inventory.to_cyclonedx(), indent=2))
            return 0
        for entry in gov.inventory.agents.values():
            identity = entry["identity"]
            print(f"{entry['name']}  model={entry['model']}  provider={entry['provider']}"
                  f"  risk={identity.get('risk')}  owner={identity.get('owner') or '-'}")
            for tool_row in entry["tools"]:
                print(f"    {tool_row['name']:<24} {tool_row['fingerprint'][:12]}  "
                      f"{','.join(tool_row['tags'])}")
            if entry["mcp_servers"]:
                print(f"    mcp: {', '.join(entry['mcp_servers'])}")
        return 0

    if args.action == "dsar":
        if args.target not in ("access", "erase") or not args.user:
            print("usage: agent-harness governance dsar access|erase --user ID "
                  "[--tenant ID] --state DIR --config governance.yaml", file=sys.stderr)
            return 2
        from .memory.trace import Trace

        harness = Harness.local(args.state or ".harness", trace=False, governance=gov)
        trace = Trace(user_id=args.user, tenant_id=args.tenant)
        if args.target == "access":
            result = asyncio.run(gov.rights.access(trace))
        else:
            result = asyncio.run(gov.rights.erase(trace, requested_by=args.requested_by))
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0

    if args.action == "report":
        if not args.config:
            print("usage: agent-harness governance report --config governance.yaml",
                  file=sys.stderr)
            return 2
        gov = Governance.from_file(args.config)
        gov.attach(Harness.testing())
        report = gov.report(args.pack)
        print({"json": report.json, "html": report.html}.get(args.format,
                                                             report.markdown)())
        return 0
    return 2


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
                       help="any name from `agent-harness providers` "
                            "(inferred from the model)")
        p.add_argument("--instructions", default="", help="the agent's instructions")
        p.add_argument("--skills", default=None, help="a directory of skills to load")
        p.add_argument("--tools", action="store_true",
                       help="give it the built-in tools (time, maths, web fetch)")
        p.add_argument("--openapi", action="append", default=[], metavar="SPEC",
                       help="an OpenAPI file or URL whose operations become tools "
                            "(repeatable)")
        p.add_argument("--openapi-token", default=None,
                       help="the bearer token for --openapi (or $OPENAPI_TOKEN)")
        p.add_argument("--no-memory", action="store_true", help="run without memory")
        p.add_argument("--mode", default=None, choices=sorted(MODES),
                       help="how the agent works: chat keeps the thread, research "
                            "writes a cited report, cowork carries a task through "
                            "in a workspace")
        p.add_argument("--depth", default=None, choices=list(DEPTHS),
                       help="how hard the mode works, and the model tier it asks for")
        p.add_argument("--workspace", default=None,
                       help="a directory the agent may read and write files in")
        p.add_argument("--sandbox", default=None, metavar="NAME",
                       help="work inside a sandbox instead: docker, podman, e2b, "
                            "daytona, modal, kubernetes, ssh — or a URL such as "
                            "docker://node:22 (see `agent-harness sandboxes`)")
        p.add_argument("--sandbox-id", default=None, metavar="ID",
                       help="pick up the sandbox an earlier run reported, instead "
                            "of starting a new one (with --session, not needed: "
                            "the session remembers its sandbox)")
        p.add_argument("--keep-sandbox", action="store_true",
                       help="leave the sandbox running at exit, to continue in "
                            "it later")
        p.add_argument("--max-steps", type=int, default=None,
                       help="the step ceiling (the mode's own, or 20)")
        p.add_argument("--state", default=None,
                       help="persist sessions, memory and traces under this directory")
        p.add_argument("--trace", action="store_true", help="print spans as they close")
        p.add_argument("--approve", action="store_true",
                       help="ask before any tool that needs approval")
        p.add_argument("--session", default=None, help="resume a session by id")
        p.add_argument("--sessions", default=None, metavar="URL",
                       help="keep chats in a database: postgresql://…, mongodb://…, "
                            "redis://…, sqlite:///chats.db, azure://container")
        p.add_argument("--user", default=None,
                       help="who the agent is acting for; their chats are theirs")
        p.add_argument("--tenant", default=None, help="the tenant they belong to")

    run = sub.add_parser("run", help="run one task and print the answer")
    run.add_argument("task")
    run.add_argument("--stream", action="store_true", help="stream the answer")
    run.add_argument("--attach", action="append", default=[], metavar="FILE",
                     help="a file or URL to send with the task: an image, a PDF, a "
                          "recording, a video, a document (repeatable)")
    run.add_argument("--json", action="store_true", help="also print the full result")
    run.add_argument("--report", action="store_true", help="print the run report")
    common(run)

    chat = sub.add_parser("chat", help="an interactive session")
    common(chat)

    voice = sub.add_parser(
        "voice", help="talk to an agent: live, or from a WAV file")
    voice.add_argument("--input", default=None, metavar="WAV",
                       help="a recording to answer, instead of the microphone")
    voice.add_argument("--output", default=None, metavar="WAV",
                       help="with --input: where to write the spoken answer")
    voice.add_argument("--realtime", default=None, choices=["openai", "gemini"],
                       help="use a speech-to-speech model instead of listen, "
                            "think, speak")
    voice.add_argument("--voice", default="alloy", help="the voice to speak in")
    voice.add_argument("--language", default=None,
                       help="the language being spoken, e.g. en")
    voice.add_argument("--greeting", default=None, help="said when the call opens")
    common(voice)

    flow = sub.add_parser(
        "workflow", help="run a declared workflow from a YAML or JSON file")
    flow.add_argument("file", help="the workflow file")
    flow.add_argument("text", nargs="?", default=None,
                      help="the input, for a workflow that takes one")
    flow.add_argument("--input", action="append", default=[], metavar="NAME=VALUE",
                      help="one input; the value is read as JSON if it is JSON "
                           "(repeatable)")
    flow.add_argument("--tool", action="append", default=[], metavar="PATH",
                      help="a tool the steps may call, as package.module:name "
                           "(repeatable)")
    flow.add_argument("--tools", action="store_true",
                      help="give it the built-in tools (time, maths, web fetch)")
    flow.add_argument("--check", action="store_true",
                      help="validate the file and print its outline; run nothing")
    flow.add_argument("--json", action="store_true", help="print the full result")
    flow.add_argument("--state", default=None,
                      help="persist sessions, memory and traces under this directory")
    flow.add_argument("--trace", action="store_true", help="print spans as they close")
    flow.add_argument("--approve", action="store_true",
                      help="ask before any tool that needs approval")

    a2a = sub.add_parser(
        "a2a", help="serve an agent over the agent-to-agent protocol, or call one")
    a2a.add_argument("action", choices=["serve", "card", "send"],
                     help="serve: serve an agent · card URL: print an agent's card · "
                          "send URL TEXT: send it a message")
    a2a.add_argument("target", nargs="?", default=None, help="the agent's URL")
    a2a.add_argument("text", nargs="?", default=None, help="what to send")
    a2a.add_argument("--host", default="127.0.0.1")
    a2a.add_argument("--port", type=int, default=8000)
    a2a.add_argument("--public-url", default=None,
                     help="the URL others reach this agent at, for its card")
    a2a.add_argument("--token", action="append", default=[],
                     help="serve: a bearer token callers must send (repeatable) · "
                          "card, send: the token to send")
    a2a.add_argument("--max-concurrency", type=int, default=64,
                     help="serve: how many tasks run at once")
    a2a.add_argument("--stream", action="store_true", help="send: stream the answer")
    a2a.add_argument("--context", default=None,
                     help="send: the context id of a conversation to carry on")
    common(a2a)

    serve = sub.add_parser(
        "mcp-serve", help="serve an agent, or a blueprint's agents, as an MCP server")
    serve.add_argument("--blueprint", default=None, metavar="FILE",
                       help="serve the agents of an agents.yaml; each may name its "
                            "own key with `mcp_key_env`")
    serve.add_argument("--api-key", action="append", default=[],
                       help="a key that opens every agent (repeatable; or "
                            "$MCP_API_KEY)")
    serve.add_argument("--key-env", action="append", default=[], metavar="AGENT=VAR",
                       help="with --blueprint: an agent's own key, read from an "
                            "environment variable (repeatable)")
    serve.add_argument("--expose-tools", action="store_true",
                       help="serve the agents' own tools as well")
    serve.add_argument("--stdio", action="store_true",
                       help="serve over stdin/stdout, for a client that starts "
                            "the server itself")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    common(serve)

    api = sub.add_parser(
        "openapi", help="show the tools an OpenAPI document becomes, or call one")
    api.add_argument("spec", help="the OpenAPI file or URL")
    api.add_argument("--base-url", default=None, help="where the API is, if the "
                                                      "document does not say")
    api.add_argument("--openapi-token", "--token", dest="openapi_token", default=None,
                     help="a bearer token (or $OPENAPI_TOKEN)")
    api.add_argument("--api-key", default=None,
                     help="the key for the document's apiKey scheme")
    api.add_argument("--call", default=None, metavar="TOOL",
                     help="call this operation and print what it returns")
    api.add_argument("--arg", action="append", default=[], metavar="NAME=VALUE",
                     help="an argument for --call (repeatable)")
    api.add_argument("--json", action="store_true",
                     help="the tools as JSON: names, schemas, requests")

    sub.add_parser("models", help="list the models the harness knows and their prices")

    providers = sub.add_parser(
        "providers", help="list the LLM providers and what each needs to connect")
    providers.add_argument("name", nargs="?", default=None,
                           help="describe one provider in full")
    providers.add_argument("--ping", action="store_true",
                           help="with a name: connect and check the credentials work")
    providers.add_argument("--ready", action="store_true",
                           help="only the providers already configured")
    providers.add_argument("--json", action="store_true", help="machine-readable output")

    sandboxes = sub.add_parser(
        "sandboxes", help="list the sandboxes an agent can work in, or test one")
    sandboxes.add_argument("name", nargs="?", default=None,
                           help="one sandbox, by name or URL (docker://alpine)")
    sandboxes.add_argument("--check", action="store_true",
                           help="with a name: start it and prove it works — run a "
                                "command, write and read a file, see a change")
    sandboxes.add_argument("--json", action="store_true",
                           help="machine-readable output")

    sessions = sub.add_parser("sessions", help="list or show stored sessions")
    sessions.add_argument("--state", default=None)
    sessions.add_argument("--store", default=None, metavar="URL",
                          help="a database instead of --state: postgresql://…, "
                               "mongodb://…, redis://…, azure://container, …")
    sessions.add_argument("--limit", type=int, default=20)
    sessions.add_argument("--user", default=None, help="only this user's sessions")
    sessions.add_argument("--tenant", default=None, help="only this tenant's")
    sessions.add_argument("--agent", default=None, help="only this agent's")
    sessions.add_argument("--show", default=None, help="print one session's transcript")
    sessions.add_argument("--delete", default=None, help="delete one session")
    sessions.add_argument("--check", action="store_true",
                          help="prove the store works: save, load, list by owner, "
                               "refuse a stale save, delete")
    sessions.add_argument("--backends", action="store_true",
                          help="which stores have their driver installed")
    sessions.add_argument("--json", action="store_true", help="machine-readable")

    mcp = sub.add_parser("mcp", help="list the tools an MCP server offers")
    mcp.add_argument("command", nargs="*", help="the server command and its arguments")
    mcp.add_argument("--url", default=None, help="an HTTP MCP endpoint instead")
    mcp.add_argument("--server-name", default="server")

    journal = sub.add_parser("journal", help="print the run journal")
    journal.add_argument("--state", default=None)
    journal.add_argument("--limit", type=int, default=50)

    gov = sub.add_parser("governance",
                         help="jurisdiction packs, evidence reports, audit verification")
    gov.add_argument("action",
                     choices=["packs", "pack", "check", "report", "inventory", "verify",
                              "dsar"],
                     help="packs: list them · pack <id>: one in full · check: validate a "
                          "config · report: evidence for a config · inventory: the AI "
                          "inventory of a blueprint · verify: check an audit trail · "
                          "dsar access|erase: a data-subject request")
    gov.add_argument("target", nargs="?", default=None,
                     help="the pack id (pack), an audit file (verify), or access|erase (dsar)")
    gov.add_argument("--agents", default=None, help="a blueprint agents.yaml (inventory)")
    gov.add_argument("--user", default=None, help="the data subject's user id (dsar)")
    gov.add_argument("--tenant", default=None, help="the data subject's tenant id (dsar)")
    gov.add_argument("--requested-by", default="data subject",
                     help="who asked for the erasure, for the record (dsar)")
    gov.add_argument("--config", default=None, help="a governance.yaml (report)")
    gov.add_argument("--pack", default=None, help="report on one pack only")
    gov.add_argument("--format", choices=["md", "json", "html"], default="md")
    gov.add_argument("--state", default=None, help="the harness directory (verify, dsar)")
    gov.add_argument("--key-env", default=None,
                     help="environment variable holding the HMAC audit key (verify)")
    gov.add_argument("--json", action="store_true", help="machine-readable (packs)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_run(args))
        if args.command == "chat":
            return asyncio.run(_chat(args))
        if args.command == "voice":
            return asyncio.run(_voice(args))
        if args.command == "workflow":
            return asyncio.run(_workflow(args))
        if args.command == "a2a":
            return asyncio.run(_a2a(args))
        if args.command == "openapi":
            return asyncio.run(_openapi(args))
        if args.command == "mcp-serve":
            return asyncio.run(_mcp_serve(args))
        if args.command == "models":
            return _models(args)
        if args.command == "providers":
            return asyncio.run(_providers(args))
        if args.command == "sandboxes":
            return asyncio.run(_sandboxes(args))
        if args.command == "sessions":
            return asyncio.run(_sessions(args))
        if args.command == "mcp":
            return asyncio.run(_mcp(args))
        if args.command == "journal":
            return _journal(args)
        if args.command == "governance":
            return _governance(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
