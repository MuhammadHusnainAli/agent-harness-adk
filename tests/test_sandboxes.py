"""Sandboxes: a workspace that is somewhere else.

Three layers are tested here. The contract — everything derived from `exec` —
runs for real against a local shell, and against busybox in a container when
Docker is to hand. Docker itself runs for real. The hosted ones (E2B, Daytona,
Modal) and the ones needing a cluster or a remote host are tested against
stand-ins for their SDK or CLI, which proves what we send, not that the service
answers.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import types
from typing import Any

import pytest

from agent_harness import (
    Agent,
    Blueprint,
    ConfigurationError,
    ExecResult,
    FakeProvider,
    Harness,
    Sandbox,
    SandboxWorkspace,
    SubAgentSpec,
    ToolError,
    Workspace,
    WorkspaceBroker,
    available_sandboxes,
    register_sandbox,
    sandbox,
    tool_call,
)
from agent_harness import sandboxes as registry
from agent_harness.sandboxes.base import run_process
from agent_harness.sandboxes.command import KubernetesSandbox, SSHSandbox
from agent_harness.sandboxes.docker import DockerSandbox
from agent_harness.toolkits import make_chart_tool, make_document_tool, make_python_tool

IMAGE = "alpine:latest"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "image", "inspect", IMAGE],
                              capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


needs_docker = pytest.mark.skipif(
    not _docker_ready(), reason=f"needs docker and a local {IMAGE} image")


class ShellOnly(Sandbox):
    """The smallest possible backend: `_exec`, and nothing else. Everything a
    workspace does on it goes through the shell forms in the base class."""

    name = "shell-only"
    isolated = False

    def __init__(self, root: str, *, prefix: tuple[str, ...] = ("sh", "-c"),
                 **kw: Any) -> None:
        super().__init__(workdir=root, **kw)
        self.prefix = list(prefix)
        self.commands: list[str] = []
        self.stopped = 0

    async def _exec(self, command, *, cwd, env, timeout):
        self.commands.append(command)
        code, out, err = await run_process(
            [*self.prefix, self.script(command, cwd=cwd, env=env)], None, timeout)
        return ExecResult(code, out.decode(errors="replace"),
                          err.decode(errors="replace"))

    async def _stop(self) -> None:
        self.stopped += 1


@pytest.fixture
def box(tmp_path) -> SandboxWorkspace:
    return SandboxWorkspace(ShellOnly(str(tmp_path / "ws")),
                            export_dir=tmp_path / "out")


# --- the contract: everything from exec ------------------------------------------

async def test_a_backend_with_only_exec_passes_the_whole_self_check(box):
    report = await box.check()
    assert report["ok"], report["steps"]
    assert [s["step"] for s in report["steps"]] == [
        "start", "exec", "write and read", "list", "detect changes", "delete"]


async def test_files_round_trip_through_the_shell_whatever_is_in_them(box):
    text = "héllo 'quoted' $HOME `tick` \\n and a real\nnewline\n"
    assert await box.awrite("notes/a b.txt", text) == len(text)
    assert await box.aread("notes/a b.txt") == text

    # Larger than one command can carry, so it goes over in pieces.
    blob = os.urandom(200_000)
    await box.awrite("big.bin", blob)
    assert await box.aread_bytes("big.bin") == blob

    await box.awrite("log.txt", "one\n")
    await box.awrite("log.txt", "two\n", append=True)
    assert await box.aread("log.txt") == "one\ntwo\n"

    await box.awrite("empty.txt", "")
    assert await box.aread("empty.txt") == ""


async def test_listing_deleting_and_asking_what_is_there(box):
    await box.awrite("a.txt", "x")
    await box.awrite("sub/b.txt", "y")
    assert await box.alistdir(".") == ["a.txt", "sub/"]
    assert await box.aexists("sub/b.txt") and not await box.aexists("nope")

    await box.aremove("sub")
    assert await box.alistdir(".") == ["a.txt"]

    with pytest.raises(ToolError, match="no such file"):
        await box.aread("gone.txt")
    with pytest.raises(ToolError, match="not a directory"):
        await box.alistdir("a.txt")
    with pytest.raises(ToolError, match="refusing to delete the workspace root"):
        await box.aremove(".")


async def test_what_changed_is_what_was_written_since(box):
    await box.awrite("keep.txt", "same")
    await box.awrite("edit.txt", "before")
    before = await box.asnapshot()
    assert set(before) == {"keep.txt", "edit.txt"}

    await asyncio.sleep(0.01)
    await box.awrite("edit.txt", "after!")
    await box.awrite("new/deep/file.txt", "hi")
    await box.awrite(".hidden/secret", "no")
    await box.awrite("_snippet.py", "scratch")

    assert await box.achanged(before) == ["edit.txt", "new/deep/file.txt"]


async def test_a_file_too_large_is_refused_not_downloaded(tmp_path):
    ws = SandboxWorkspace(ShellOnly(str(tmp_path / "ws")), max_file_bytes=100)
    await ws.awrite("big.txt", "x" * 500)
    with pytest.raises(ToolError, match="larger than the 100-byte limit"):
        await ws.aread("big.txt")


async def test_paths_cannot_climb_out_of_the_working_directory(box):
    for path in ("../outside.txt", "/etc/passwd", "a/../../b"):
        with pytest.raises(ToolError, match="outside the workspace"):
            await box.aread(path)
    with pytest.raises(ToolError, match="outside the workspace"):
        await box.awrite("../x", "no")
    assert str(box.resolve("a/./b/../c.txt")).endswith("/ws/a/c.txt")


async def test_a_read_only_sandbox_refuses_writes(tmp_path):
    ws = SandboxWorkspace(ShellOnly(str(tmp_path / "ws")), read_only=True)
    with pytest.raises(ToolError, match="read-only"):
        await ws.awrite("a.txt", "x")
    with pytest.raises(ToolError, match="read-only"):
        await ws.aremove("a.txt")


def test_the_blocking_methods_say_to_await_instead(box):
    for call in (lambda: box.read("a"), lambda: box.write("a", "b"),
                 lambda: box.listdir(), lambda: box.snapshot(),
                 lambda: box.exists("a")):
        with pytest.raises(ToolError, match="await the async form"):
            call()


async def test_commands_run_in_the_working_directory_with_the_environment(tmp_path):
    ws = SandboxWorkspace(ShellOnly(str(tmp_path / "ws"), env={"GREETING": "hi"}))
    result = await ws.shell("pwd && echo $GREETING && exit 3")
    assert result["returncode"] == 3
    assert result["stdout"].split() == [str(tmp_path / "ws"), "hi"]


async def test_a_command_that_overruns_is_a_tool_error(tmp_path):
    ws = SandboxWorkspace(sandbox_backend_command(tmp_path))
    with pytest.raises(ToolError, match="timed out"):
        await ws.shell("sleep 20", timeout=1)


def sandbox_backend_command(tmp_path):
    return registry.sandbox_backend("command", workdir=str(tmp_path / "ws"))


async def test_a_sandbox_starts_once_and_stops_once(tmp_path):
    backend = ShellOnly(str(tmp_path / "ws"))
    ws = SandboxWorkspace(backend)
    await asyncio.gather(*(ws.awrite(f"{n}.txt", "x") for n in range(5)))
    assert sum(c.startswith("mkdir -p") and "printf" not in c
               for c in backend.commands) == 1

    await ws.aclose()
    await ws.aclose()
    assert backend.stopped == 1
    with pytest.raises(ToolError, match="has been stopped"):
        await ws.aread("0.txt")


async def test_a_kept_sandbox_is_left_running(tmp_path):
    backend = ShellOnly(str(tmp_path / "ws"), keep=True)
    async with SandboxWorkspace(backend) as ws:
        await ws.awrite("a.txt", "x")
    assert backend.stopped == 0


async def test_a_working_directory_left_unsaid_is_settled_at_start(tmp_path):
    class Homed(ShellOnly):
        async def _exec(self, command, *, cwd, env, timeout):
            return await super()._exec(command, cwd=cwd or str(tmp_path), env=env,
                                       timeout=timeout)

    backend = Homed(str(tmp_path))
    backend.workdir = None
    ws = SandboxWorkspace(backend)
    await ws.awrite("a.txt", "x")
    assert backend.workdir == f"{tmp_path}/workspace"
    assert (tmp_path / "workspace" / "a.txt").read_text() == "x"


# --- the workspace an agent holds ---------------------------------------------------

def test_a_sandbox_comes_with_its_shell_and_a_local_one_still_asks(tmp_path):
    local = SandboxWorkspace(ShellOnly(str(tmp_path / "a")))
    tools = {t.name: t for t in local.tools()}
    assert set(tools) == {"fs_read", "fs_write", "fs_list", "fs_delete", "shell"}
    assert tools["shell"].permission == "ask"          # not isolated: this machine

    class Walled(ShellOnly):
        isolated = True

    walled = {t.name: t for t in SandboxWorkspace(Walled(str(tmp_path / "b"))).tools()}
    assert walled["shell"].permission == "allow"
    assert make_python_tool(SandboxWorkspace(Walled(str(tmp_path / "c")))
                            ).permission == "allow"
    assert make_python_tool(Workspace(tmp_path / "d")).permission == "ask"

    quiet = SandboxWorkspace(Walled(str(tmp_path / "e")), allow_shell=False)
    assert "shell" not in {t.name for t in quiet.tools()}


async def test_cowork_in_a_sandbox_hands_back_files_that_outlive_it(tmp_path):
    class Walled(ShellOnly):
        isolated = True

    backend = Walled(str(tmp_path / "ws"))
    provider = FakeProvider([
        tool_call("todo_write", todos=[{"content": "Build it", "status": "in_progress"}]),
        tool_call("fs_write", path="app/main.py", content="print(6 * 7)\n"),
        tool_call("shell", command="echo 42 > app/out.txt && printf '\\211PNG\\377' > logo.png"),
        tool_call("todo_write", todos=[{"content": "Build it", "status": "done"}]),
        "Built. See app/main.py.",
    ])
    harness = Harness.testing(provider)
    agent = Agent("colleague", mode="cowork", depth="fast", harness=harness,
                  workspace=SandboxWorkspace(backend, export_dir=tmp_path / "out"))
    assert "`shell` runs commands" in agent.assembler.mode

    result = await agent.run("Build the thing.")

    assert result.ok and result.violations == []
    by_name = {a.name: a for a in result.artifacts}
    assert set(by_name) == {"app/main.py", "app/out.txt", "logo.png"}
    assert by_name["app/main.py"].content == "print(6 * 7)\n"
    assert by_name["app/out.txt"].content == "42\n"
    # Brought down to this machine, so they are still there once it is gone.
    assert by_name["app/main.py"].path == str(tmp_path / "out" / "app" / "main.py")
    assert by_name["logo.png"].content == ""
    assert by_name["logo.png"].media_type == "image/png"

    await harness.aclose()
    assert backend.stopped == 1
    assert open(by_name["logo.png"].path, "rb").read() == b"\x89PNG\xff"


async def test_a_sandbox_that_cannot_start_ends_the_run_before_it_costs_anything():
    class Broken(Sandbox):
        name = "broken"

        async def _start(self) -> None:
            raise ConfigurationError("the broken sandbox needs an API key")

        async def _exec(self, command, *, cwd, env, timeout):  # pragma: no cover
            raise AssertionError("never reached")

    provider = FakeProvider(["should not be asked"])
    agent = Agent("colleague", mode="cowork", workspace=sandbox(Broken()),
                  harness=Harness.testing(provider))
    result = await agent.run("go")
    assert "needs an API key" in (result.error or "")
    assert provider.requests == []


async def test_an_agent_takes_a_sandbox_by_name_url_or_mapping(tmp_path, harness):
    register_sandbox("shellonly", __name__, "ShellOnly", summary="for tests")
    ShellOnly.url_option = "root"
    try:
        by_url = Agent("a", workspace=f"shellonly://{tmp_path}/u", harness=harness)
        by_map = Agent("b", harness=harness,
                       workspace={"sandbox": "shellonly", "root": f"{tmp_path}/m",
                                  "allow_shell": False})
        bare = Agent("c", workspace=ShellOnly(f"{tmp_path}/c"), harness=harness)
        for agent in (by_url, by_map, bare):
            assert isinstance(agent.workspace, SandboxWorkspace)
        assert by_url.workspace.sandbox.workdir == f"{tmp_path}/u"
        assert "shell" in by_url.tools and "shell" not in by_map.tools

        # The words that have always meant "a folder" still do.
        assert type(Agent("d", workspace="true", harness=harness).workspace) is Workspace
        assert Agent("e", workspace="none", harness=harness).workspace is None
        with pytest.raises(ConfigurationError, match="unknown sandbox"):
            Agent("f", workspace="nowhere", harness=harness)

        # Whoever built it, the harness closes it.
        await harness.aclose()
        assert bare.workspace.sandbox._stopped and by_url.workspace.sandbox._stopped
    finally:
        registry.BACKENDS.pop("shellonly")
        ShellOnly.url_option = ""


def test_every_version_of_an_agent_works_in_the_one_sandbox(tmp_path, harness):
    agent = Agent("a", harness=harness, version="v1",
                  workspace=ShellOnly(str(tmp_path / "ws")),
                  versions={"v1": {"mode": "chat"}, "v2": {"mode": "cowork"}})
    assert agent.use("v2").workspace is agent.workspace


async def test_a_broker_hands_every_agent_its_own_sandbox(tmp_path):
    register_sandbox("shellonly", __name__, "ShellOnly")
    try:
        counter = iter(range(100))

        class Broker(WorkspaceBroker):
            def _make(self, name, **kw):
                return super()._make(name, root=f"{tmp_path}/{next(counter)}", **kw)

        harness = Harness.testing(workspaces=Broker(sandbox="shellonly"))
        lead = Agent("lead", workspace=True, harness=harness)
        helper = lead.add_subagent(SubAgentSpec(name="helper", workspace="isolated"))
        sharer = lead.add_subagent(SubAgentSpec(name="sharer", workspace="shared"))

        boxes = [lead.workspace, helper.workspace, sharer.workspace]
        assert all(isinstance(b, SandboxWorkspace) for b in boxes)
        assert len({id(b.sandbox) for b in boxes}) == 3
        assert harness.workspaces.shared() is sharer.workspace

        await lead.workspace.awrite("a.txt", "x")
        await harness.aclose()
        assert lead.workspace.sandbox.stopped == 1
        # One never started is not something to stop.
        assert helper.workspace.sandbox.stopped == 0
    finally:
        registry.BACKENDS.pop("shellonly")


def test_a_broker_only_makes_a_folder_when_something_asks_for_one(tmp_path):
    broker = WorkspaceBroker()
    assert broker._root is None
    assert broker.acquire("a").root.exists() and broker._root is not None
    broker.cleanup()

    assert WorkspaceBroker(tmp_path / "given").root.is_dir()
    assert WorkspaceBroker(backend="e2b", template="t").sandbox == "e2b"
    assert WorkspaceBroker(sandbox="docker", image="node:22").options == {
        "image": "node:22"}
    with pytest.raises(ConfigurationError, match="sandbox options"):
        WorkspaceBroker(template="t")


def test_a_blueprint_declares_the_sandbox(tmp_path, harness):
    register_sandbox("shellonly", __name__, "ShellOnly")
    try:
        blueprint = Blueprint.from_text(f"""
agents:
  colleague:
    mode: cowork
    workspace: {{sandbox: shellonly, root: {tmp_path}/ws}}
  local:
    mode: cowork
""")
        colleague = blueprint.build("colleague", harness=harness)
        assert isinstance(colleague.workspace, SandboxWorkspace)
        assert "shell" in colleague.tools
        local = blueprint.build("local", harness=harness)
        assert type(local.workspace) is Workspace and "shell" not in local.tools
    finally:
        registry.BACKENDS.pop("shellonly")


async def test_the_toolkits_work_on_a_sandbox_as_they_do_on_a_folder(box):
    box.sandbox.python = sys.executable
    ran = await make_python_tool(box).invoke({"code": "print(6 * 7)"})
    assert ran == {"ok": True, "stdout": "42\n", "stderr": None}

    await box.awrite("data.csv", "name,score\nada,9\n")
    parsed = await make_document_tool(box).invoke({"path": "data.csv"})
    assert "ada | 9" in parsed["text"]

    wrote = await make_chart_tool(box).invoke(
        {"kind": "bar", "title": "Scores", "data": {"ada": 9}, "path": "c.svg"})
    assert wrote.startswith("wrote c.svg")
    assert (await box.aread("c.svg")).startswith("<svg")


# --- the registry ----------------------------------------------------------------------

def test_the_registry_knows_the_sandboxes_and_what_each_needs(monkeypatch):
    assert set(available_sandboxes()) >= {
        "docker", "podman", "kubernetes", "ssh", "command", "e2b", "daytona", "modal"}
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setattr(registry, "_installed", lambda module: False)
    assert registry.missing("e2b") == ["pip install e2b", "set E2B_API_KEY"]
    assert registry.missing("command") == []
    assert not available_sandboxes()["e2b"]

    monkeypatch.setattr(registry, "_installed", lambda module: True)
    monkeypatch.setenv("E2B_API_KEY", "k")
    assert available_sandboxes()["e2b"]
    rows = {r["name"]: r for r in registry.describe_sandboxes()}
    assert rows["e2b"]["ready"] and rows["e2b"]["summary"]


def test_a_sandbox_is_named_by_word_url_or_mapping():
    assert sandbox("docker://node:22", network=True).sandbox.image == "node:22"
    assert sandbox("podman").sandbox.cli == "podman"
    assert sandbox("k8s://python:3.13").sandbox.image == "python:3.13"
    assert sandbox("ssh://agent@build-7").sandbox.host == "agent@build-7"
    assert sandbox("e2b://base").sandbox.template == "base"
    assert sandbox({"sandbox": "docker", "image": "a", "read_only": True}).read_only

    with pytest.raises(ConfigurationError, match="unknown sandbox"):
        sandbox("nowhere")
    with pytest.raises(ConfigurationError, match="docker sandbox"):
        sandbox("docker", no_such_option=1)
    with pytest.raises(ConfigurationError, match="takes no URL"):
        sandbox("command://x", workdir="/w")
    with pytest.raises(ConfigurationError, match="needs `sandbox:`"):
        sandbox({"image": "a"})
    with pytest.raises(ConfigurationError, match="needs a `workdir`"):
        sandbox("command")


# --- docker: the arguments, without docker ------------------------------------------------

def test_a_container_is_started_closed_by_default():
    argv = DockerSandbox("python:3.12-slim").run_argv()
    assert argv[:4] == ["docker", "run", "-d", "--rm"]
    for flag, value in (("--network", "none"), ("--memory", "2g"), ("--cpus", "2"),
                        ("--pids-limit", "512"),
                        ("--security-opt", "no-new-privileges"), ("-w", "/workspace")):
        assert argv[argv.index(flag) + 1] == value
    assert argv[-3:] == ["python:3.12-slim", "sleep", "14400"]
    assert "--user" not in argv and "-v" not in argv


def test_a_container_opens_up_only_as_far_as_it_is_told(tmp_path):
    argv = DockerSandbox("node:22", network=True, runtime="runsc", ttl=None,
                         mount=tmp_path, env={"CI": "1"}, memory=None,
                         args=["--gpus", "all"]).run_argv()
    assert "--network" not in argv and "--memory" not in argv
    assert argv[argv.index("--runtime") + 1] == "runsc"
    assert argv[argv.index("-v") + 1] == f"{tmp_path}:/workspace"
    # A mounted folder is written as you, not as root.
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "CI=1" in argv
    assert argv[-6:] == ["--gpus", "all", "node:22", "tail", "-f", "/dev/null"]

    read_only = DockerSandbox("a", mount=tmp_path, mount_read_only=True).run_argv()
    assert read_only[read_only.index("-v") + 1].endswith(":/workspace:ro")


async def test_a_container_it_did_not_start_is_left_running():
    calls: list[list[str]] = []

    async def runner(argv, data, timeout):
        calls.append(argv)
        return 0, b"true\n" if "inspect" in argv else b"", b""

    box = DockerSandbox(container="mine", runner=runner)
    await box.start()
    await box.stop()
    assert box.keep and box.id == "mine"
    assert not any("run" in c[:2] or "rm" in c[:2] for c in calls)
    assert calls[1][:4] == ["docker", "exec", "-i", "mine"]


# --- docker: for real -----------------------------------------------------------------------

def _containers() -> set[str]:
    out = subprocess.run(["docker", "ps", "-a", "--filter", "label=agent-harness=sandbox",
                          "--format", "{{.Names}}"], capture_output=True, text=True)
    return set(out.stdout.split())


@needs_docker
async def test_docker_passes_the_self_check_and_cleans_up_after_itself():
    ws = sandbox(f"docker://{IMAGE}")
    report = await ws.check()
    assert report["ok"], report["steps"]
    name = ws.sandbox.container
    assert name in _containers()
    await ws.aclose()
    assert name not in _containers()


@needs_docker
async def test_a_container_keeps_its_state_between_commands_and_has_no_network():
    async with sandbox(f"docker://{IMAGE}") as ws:
        await ws.shell("echo kept > /tmp/state && export GONE=1")
        assert (await ws.shell("cat /tmp/state"))["stdout"] == "kept\n"
        # Only loopback: the default is no network at all.
        assert (await ws.shell("ls /sys/class/net"))["stdout"].split() == ["lo"]
        blob = os.urandom(300_000)
        await ws.awrite("big.bin", blob)
        assert await ws.aread_bytes("big.bin") == blob


@needs_docker
async def test_a_command_that_overruns_is_killed_inside_the_container():
    async with sandbox(f"docker://{IMAGE}") as ws:
        with pytest.raises(ToolError, match="timed out"):
            await ws.shell("sleep 300", timeout=1)
        assert ws.sandbox._has_timeout
        # Killed where it ran, not merely abandoned by the client.
        assert "sleep 300" not in (await ws.shell("ps"))["stdout"]


@needs_docker
async def test_a_mounted_folder_is_worked_in_and_the_files_are_yours(tmp_path):
    (tmp_path / "given.txt").write_text("from the host")
    async with sandbox("docker", image=IMAGE, mount=tmp_path) as ws:
        assert await ws.aread("given.txt") == "from the host"
        await ws.awrite("made/here.txt", "from the container")
    made = tmp_path / "made" / "here.txt"
    assert made.read_text() == "from the container"
    assert made.stat().st_uid == os.getuid()


@needs_docker
async def test_the_shell_forms_hold_on_busybox():
    """The exec-only path, inside alpine: no GNU coreutils to lean on."""
    host = DockerSandbox(IMAGE)
    await host.start()
    try:
        inner = ShellOnly("/workspace/inner",
                          prefix=("docker", "exec", "-i", host.container, "sh", "-c"))
        report = await SandboxWorkspace(inner).check()
        assert report["ok"], report["steps"]
    finally:
        await host.stop()


@needs_docker
async def test_an_agent_does_cowork_inside_a_real_container(tmp_path):
    provider = FakeProvider([
        tool_call("fs_write", path="hello.sh", content="echo hello from $(hostname)\n"),
        tool_call("shell", command="sh hello.sh > out.txt && cat out.txt"),
        "Ran it; the output is in out.txt.",
    ])
    harness = Harness.testing(provider)
    agent = Agent("colleague", mode="cowork", depth="fast", harness=harness,
                  workspace=sandbox("docker", image=IMAGE, export_dir=tmp_path))
    result = await agent.run("Write and run a script.")
    name = agent.workspace.sandbox.container

    assert result.ok, result.error
    assert sorted(a.name for a in result.artifacts) == ["hello.sh", "out.txt"]
    out = next(a for a in result.artifacts if a.name == "out.txt")
    assert out.content.startswith("hello from ") and (tmp_path / "out.txt").exists()
    shell_reply = provider.requests[2].messages[-1].content[0]
    assert not shell_reply.is_error and "hello from" in shell_reply.content

    await harness.aclose()
    assert name not in _containers()


# --- ssh and kubernetes: what is sent ---------------------------------------------------------

class Recorder:
    def __init__(self, replies: dict[str, bytes] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.replies = replies or {}

    async def __call__(self, argv, data, timeout):
        self.calls.append(list(argv))
        joined = " ".join(argv)
        for needle, reply in self.replies.items():
            if needle in joined:
                return 0, reply, b""
        return 0, b"", b""


async def test_ssh_carries_the_script_as_one_quoted_word():
    run = Recorder()
    box = SSHSandbox("agent@build-7", port=2222, key="~/.ssh/id", workdir="/srv/w",
                     options=["StrictHostKeyChecking=accept-new"], runner=run)
    assert not box.isolated            # a real machine, unless you say otherwise
    await box.start()
    result = await box.exec("echo 'it works'")
    await box.stop()

    assert result.ok
    first = run.calls[0]
    assert first[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    assert first[first.index("-p") + 1] == "2222"
    assert first[first.index("-i") + 1] == os.path.expanduser("~/.ssh/id")
    assert first[-2:] == ["agent@build-7", "sh -lc true"]
    sent = run.calls[-1][-1]
    assert sent.startswith("sh -lc ") and "cd /srv/w" in sent and "it works" in sent
    assert SSHSandbox("h", isolated=True, workdir="/w").isolated
    with pytest.raises(ConfigurationError, match="needs a `host`"):
        SSHSandbox()


async def test_kubernetes_makes_a_pod_with_nothing_it_does_not_need():
    run = Recorder()
    box = KubernetesSandbox("python:3.13", namespace="agents", context="dev", runner=run)
    await box.start()
    await box.exec("echo hi")
    await box.stop()

    created = run.calls[0]
    assert created[:6] == ["kubectl", "--context", "dev", "-n", "agents", "run"]
    assert "--image=python:3.13" in created and "--restart=Never" in created
    spec = json.loads(next(a for a in created if a.startswith("--overrides=")
                           ).split("=", 1)[1])["spec"]
    assert spec["automountServiceAccountToken"] is False
    assert spec["activeDeadlineSeconds"] == 14_400
    container = spec["containers"][0]
    assert container["name"] == box.pod and container["command"] == ["sleep", "14400"]
    assert container["resources"]["limits"] == {"cpu": "2", "memory": "2Gi"}
    assert container["securityContext"] == {"allowPrivilegeEscalation": False}

    assert run.calls[1][5:8] == ["wait", "--for=condition=Ready", f"pod/{box.pod}"]
    executed = next(c for c in run.calls if "echo hi" in c[-1])
    assert executed[5:9] == ["exec", "-i", box.pod, "--"]
    assert run.calls[-1][5:8] == ["delete", "pod", box.pod]


async def test_a_pod_it_was_pointed_at_is_not_deleted():
    run = Recorder()
    box = KubernetesSandbox(pod="mine", container="main", runner=run)
    await box.start()
    await box.exec("true")
    await box.stop()
    assert not any("run" in c or "delete" in c for c in run.calls)
    assert run.calls[-1][:6] == ["kubectl", "exec", "-i", "mine", "-c", "main"]


async def test_a_cli_that_is_not_installed_says_so(tmp_path):
    box = DockerSandbox("a", cli="definitely-not-a-container-cli")
    with pytest.raises(ConfigurationError, match="is not on PATH"):
        await box.start()


# --- e2b, daytona, modal: what is asked of the SDK ----------------------------------------------

def _fake(monkeypatch, name: str, **members: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _Local:
    """Runs what a fake SDK is asked to run, in a local shell."""

    @staticmethod
    async def run(script: str) -> tuple[int, str, str]:
        code, out, err = await run_process(["sh", "-c", script], None, 30)
        return code, out.decode(), err.decode()


async def test_e2b_is_created_with_what_was_asked_and_killed_at_the_end(
        monkeypatch, tmp_path):
    seen: dict[str, Any] = {"killed": 0, "files": {}}

    class CommandExitException(Exception):
        def __init__(self, code, out, err):
            super().__init__(f"exit {code}")
            self.exit_code, self.stdout, self.stderr = code, out, err

    class Commands:
        async def run(self, cmd, cwd=None, envs=None, timeout=60):
            seen.setdefault("runs", []).append({"cwd": cwd, "envs": envs})
            script = Sandbox.script(cmd, cwd=cwd, env=envs)
            code, out, err = await _Local.run(script)
            if code != 0:
                raise CommandExitException(code, out, err)
            return types.SimpleNamespace(exit_code=0, stdout=out, stderr=err)

    class Files:
        async def exists(self, path):
            return os.path.exists(path)

        async def read(self, path, format="text"):
            assert format == "bytes"
            return bytearray(open(path, "rb").read())

        async def write(self, path, data):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(data)
            seen["files"][path] = len(data)

    class AsyncSandbox:
        sandbox_id = "sbx-123"
        commands, files = Commands(), Files()

        @classmethod
        async def create(cls, **options):
            seen["create"] = options
            return cls()

        @classmethod
        async def connect(cls, sandbox_id, **options):
            seen["connect"] = (sandbox_id, options)
            return cls()

        async def kill(self):
            seen["killed"] += 1

    _fake(monkeypatch, "e2b", AsyncSandbox=AsyncSandbox)
    monkeypatch.setenv("E2B_API_KEY", "key-1")

    ws = sandbox("e2b://my-template", ttl=900, network=False, env={"A": "1"},
                 metadata={"job": "7"}, workdir=str(tmp_path / "w"))
    report = await ws.check()
    assert report["ok"], report["steps"]
    assert seen["create"] == {"timeout": 900, "api_key": "key-1",
                              "template": "my-template", "metadata": {"job": "7"},
                              "envs": {"A": "1"}, "allow_internet_access": False}
    assert ws.sandbox.id == "sbx-123"
    # Its own file API for whole files; a non-zero exit is an answer, not a crash.
    assert seen["files"]
    assert (await ws.shell("exit 7"))["returncode"] == 7
    assert seen["runs"][-1] == {"cwd": str(tmp_path / "w"), "envs": {"A": "1"}}
    await ws.aclose()
    assert seen["killed"] == 1

    # Nothing asked for, nothing passed — an older SDK is not handed new options.
    plain = sandbox("e2b", workdir=str(tmp_path / "p"))
    await plain.start()
    assert seen["create"] == {"timeout": 3600, "api_key": "key-1"}

    attached = sandbox("e2b", sandbox_id="sbx-9", api_key="other",
                       workdir=str(tmp_path / "a"))
    await attached.start()
    await attached.aclose()
    assert seen["connect"] == ("sbx-9", {"timeout": 3600, "api_key": "other"})
    assert seen["killed"] == 1 and not attached.sandbox.replaced


async def test_e2b_without_its_sdk_or_its_key_says_which(monkeypatch):
    monkeypatch.setitem(sys.modules, "e2b", None)
    with pytest.raises(ConfigurationError, match="pip install e2b"):
        await sandbox("e2b").start()
    _fake(monkeypatch, "e2b", AsyncSandbox=object)
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="set E2B_API_KEY"):
        await sandbox("e2b").start()


async def test_daytona_is_created_from_an_image_or_a_snapshot_and_deleted(
        monkeypatch, tmp_path):
    seen: dict[str, Any] = {"deleted": 0, "closed": 0}

    class Params:
        def __init__(self, **fields):
            self.kind, self.fields = type(self).__name__, fields

    class CreateSandboxFromImageParams(Params):
        pass

    class CreateSandboxFromSnapshotParams(Params):
        pass

    class Process:
        async def exec(self, command, cwd=None, env=None, timeout=None):
            seen["exec"] = {"cwd": cwd, "env": env, "timeout": timeout}
            code, out, err = await _Local.run(
                Sandbox.script(command, cwd=cwd or str(tmp_path), env=env))
            return types.SimpleNamespace(exit_code=code, result=out + err)

    class Box:
        id = "dt-1"
        process = Process()

    class AsyncDaytona:
        def __init__(self, config):
            seen["config"] = config

        async def create(self, params, timeout=60):
            seen["params"], seen["timeout"] = params, timeout
            return Box()

        async def get(self, sandbox_id):
            seen["get"] = sandbox_id
            return Box()

        async def delete(self, box):
            seen["deleted"] += 1

        async def close(self):
            seen["closed"] += 1

    _fake(monkeypatch, "daytona", AsyncDaytona=AsyncDaytona,
          DaytonaConfig=lambda **f: f,
          CreateSandboxFromImageParams=CreateSandboxFromImageParams,
          CreateSandboxFromSnapshotParams=CreateSandboxFromSnapshotParams)
    monkeypatch.setenv("DAYTONA_API_KEY", "dk")

    ws = sandbox("daytona://python:3.12-slim", ttl=1800, network=False,
                 target="eu", labels={"team": "x"})
    report = await ws.check()
    assert report["ok"], report["steps"]
    assert seen["config"] == {"api_key": "dk", "target": "eu"}
    assert seen["params"].kind == "CreateSandboxFromImageParams"
    assert seen["params"].fields == {
        "image": "python:3.12-slim", "auto_stop_interval": 30,
        "labels": {"team": "x"}, "network_block_all": True}
    # No working directory was named, so it is a folder under where it starts.
    assert ws.sandbox.workdir == f"{tmp_path}/workspace"
    assert seen["exec"]["cwd"] == f"{tmp_path}/workspace"
    await ws.aclose()
    assert (seen["deleted"], seen["closed"]) == (1, 1)

    snap = sandbox("daytona", snapshot="base-snap")
    await snap.start()
    assert seen["params"].kind == "CreateSandboxFromSnapshotParams"
    assert seen["params"].fields == {"auto_stop_interval": 60, "snapshot": "base-snap"}

    attached = sandbox("daytona", sandbox_id="dt-9")
    await attached.start()
    await attached.aclose()
    assert seen["get"] == "dt-9" and seen["deleted"] == 1
    assert not attached.sandbox.replaced

    with pytest.raises(ConfigurationError, match="not both"):
        sandbox("daytona", image="a", snapshot="b")
    monkeypatch.delenv("DAYTONA_API_KEY")
    with pytest.raises(ConfigurationError, match="set DAYTONA_API_KEY"):
        await sandbox("daytona").start()


async def test_modal_is_created_under_an_app_and_terminated(monkeypatch, tmp_path):
    seen: dict[str, Any] = {"terminated": 0}

    def aio(fn):
        return types.SimpleNamespace(aio=fn)

    class Stream:
        def __init__(self, text):
            self.read = aio(self._read)
            self._text = text

        async def _read(self):
            return self._text

    class Box:
        object_id = "sb-1"

        def __init__(self):
            self.exec = aio(self._exec)
            self.terminate = aio(self._terminate)
            self.poll = aio(self._poll)

        async def _poll(self):
            return None                      # still running

        async def _exec(self, *argv, timeout=None):
            seen["argv"], seen["exec_timeout"] = argv[:2], timeout
            code, out, err = await _Local.run(argv[2])

            async def wait():
                return code

            return types.SimpleNamespace(stdout=Stream(out), stderr=Stream(err),
                                         wait=aio(wait))

        async def _terminate(self):
            seen["terminated"] += 1

    async def lookup(name, create_if_missing=False):
        seen["app"] = (name, create_if_missing)
        return "the-app"

    async def create(**options):
        seen["create"] = options
        return Box()

    async def from_id(sandbox_id):
        seen["from_id"] = sandbox_id
        return Box()

    _fake(monkeypatch, "modal",
          App=types.SimpleNamespace(lookup=aio(lookup)),
          Image=types.SimpleNamespace(from_registry=lambda ref: f"registry:{ref}",
                                      debian_slim=lambda: "debian-slim"),
          Sandbox=types.SimpleNamespace(create=aio(create), from_id=aio(from_id)))

    ws = sandbox("modal://python:3.12-slim", app="jobs", ttl=600, network=False,
                 cpu=2, memory=4096, workdir=str(tmp_path / "w"))
    report = await ws.check()
    assert report["ok"], report["steps"]
    assert seen["app"] == ("jobs", True)
    assert seen["create"] == {"app": "the-app", "image": "registry:python:3.12-slim",
                              "timeout": 600, "block_network": True, "cpu": 2,
                              "memory": 4096}
    assert seen["argv"] == ("sh", "-c") and ws.sandbox.id == "sb-1"
    result = await ws.shell("echo out; echo err >&2; exit 2", timeout=45)
    assert result == {"returncode": 2, "stdout": "out\n", "stderr": "err\n"}
    assert seen["exec_timeout"] == 45
    await ws.aclose()
    assert seen["terminated"] == 1

    plain = sandbox("modal", workdir=str(tmp_path / "p"))
    await plain.start()
    assert seen["create"] == {"app": "the-app", "image": "debian-slim",
                              "timeout": 3600}

    attached = sandbox("modal", sandbox_id="sb-9", workdir=str(tmp_path / "a"))
    await attached.start()
    await attached.aclose()
    assert seen["from_id"] == "sb-9" and seen["terminated"] == 1
    assert not attached.sandbox.replaced


# --- the command line ----------------------------------------------------------------------------

def test_the_cli_lists_sandboxes_and_takes_one_for_a_run(capsys):
    from agent_harness.cli import build_parser, main

    assert main(["sandboxes"]) == 0
    listing = capsys.readouterr().out
    assert "docker" in listing and "e2b" in listing and "--check" in listing

    assert main(["sandboxes", "--json"]) == 0
    assert {r["name"] for r in json.loads(capsys.readouterr().out)} >= {"docker", "modal"}

    args = build_parser().parse_args(
        ["run", "do it", "--mode", "cowork", "--sandbox", "docker://node:22"])
    assert args.sandbox == "docker://node:22"


@needs_docker
def test_the_cli_proves_a_sandbox_works(capsys):
    from agent_harness.cli import main

    assert main(["sandboxes", f"docker://{IMAGE}", "--check"]) == 0
    out = capsys.readouterr().out
    assert "ok   write and read" in out and "docker works" in out
