"""Picking a conversation back up: the chat by its id, and the sandbox with it."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

import pytest

from agent_harness import (
    Agent,
    ConfigurationError,
    ExecResult,
    FakeProvider,
    Harness,
    Sandbox,
    Workspace,
    sandbox,
    tool_call,
)
from agent_harness.sandboxes.base import run_process
from agent_harness.sandboxes.docker import DockerSandbox

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


class Cloud(Sandbox):
    """A hosted sandbox in miniature: each one is a directory, alive until it is
    stopped, and found again by its id while it is."""

    name = "cloud"
    live: dict[str, str] = {}
    made = 0

    def __init__(self, base: Any, **kw: Any) -> None:
        super().__init__(**kw)
        self.base = str(base)

    async def _start(self) -> None:
        if self.attach_id:
            if self.attach_id not in Cloud.live:
                raise RuntimeError(f"404: sandbox {self.attach_id} not found")
            self.id = self.attach_id
        else:
            Cloud.made += 1
            self.id = f"cloud-{Cloud.made}"
            Cloud.live[self.id] = f"{self.base}/{self.id}"
            os.makedirs(Cloud.live[self.id])
        self.workdir = f"{Cloud.live[self.id]}/w"

    async def _exec(self, command, *, cwd, env, timeout):
        code, out, err = await run_process(
            ["sh", "-c", self.script(command, cwd=cwd, env=env)], None, timeout)
        return ExecResult(code, out.decode(), err.decode())

    async def _stop(self) -> None:
        Cloud.live.pop(self.id, None)


@pytest.fixture(autouse=True)
def _fresh_cloud():
    Cloud.live, Cloud.made = {}, 0
    yield


def colleague(tmp_path, script: list, *, state: str = "state", **box: Any) -> Agent:
    """A cowork agent as a new process would build it: its own harness, the
    same state directory, a sandbox of the same kind."""
    harness = Harness.local(tmp_path / state, trace=False,
                            provider=FakeProvider(script))
    return Agent("colleague", mode="cowork", depth="fast", harness=harness,
                 workspace=sandbox(Cloud(tmp_path / "cloud", **box),
                                   export_dir=tmp_path / "out"))


FIRST = [
    tool_call("fs_write", path="app/main.py", content="print('v1')\n"),
    tool_call("shell", command="printf '\\211PNG' > logo.png && echo built > build.log"),
    "Built v1.",
]


# --- the chat and its sandbox, picked up together -----------------------------------

async def test_a_run_reports_the_two_ids_that_pick_it_back_up(tmp_path):
    agent = colleague(tmp_path, FIRST, keep=True)
    result = await agent.run("Build the app.")

    assert result.session_id and result.sandbox_id == "cloud-1"
    saved = await agent.harness.sessions.load(result.session_id)
    assert saved.metadata["sandbox"] == {
        "sandbox": "cloud", "id": "cloud-1", "workdir": f"{tmp_path}/cloud/cloud-1/w"}
    # What it wrote is remembered with the conversation; what is not text, by name.
    assert saved.metadata["sandbox_files"] == {
        "app/main.py": "print('v1')\n", "build.log": "built\n", "logo.png": None}

    await agent.harness.aclose()
    assert "cloud-1" in Cloud.live          # kept, for next time


async def test_the_chat_id_alone_brings_back_the_conversation_and_its_sandbox(tmp_path):
    first = colleague(tmp_path, FIRST, keep=True)
    before = await first.run("Build the app.")
    await first.harness.aclose()

    later = colleague(tmp_path, [tool_call("fs_read", path="app/main.py"),
                                 "It prints v1."])
    after = await later.run("What does main.py print?", session=before.session_id)

    assert after.ok and after.session_id == before.session_id
    assert after.sandbox_id == "cloud-1" and Cloud.made == 1
    sent = later.harness.provider.requests[0].messages
    # The earlier turns are there, and nothing had to be explained away.
    assert sent[0].text == "Build the app." and sent[-1].text == "What does main.py print?"
    read = later.harness.provider.requests[1].messages[-1].content[0]
    assert read.content == "print('v1')\n" and not read.is_error
    # One it picked up is left for the time after this, too.
    await later.harness.aclose()
    assert "cloud-1" in Cloud.live


async def test_both_ids_can_be_passed_by_hand(tmp_path):
    first = colleague(tmp_path, FIRST, keep=True)
    before = await first.run("Build the app.")
    other = colleague(tmp_path, ["a different job"], state="elsewhere", keep=True)
    await other.run("Something else.")
    assert Cloud.made == 2

    # To the run...
    by_run = colleague(tmp_path, ["ok"])
    result = await by_run.run("Carry on.", session=before.session_id,
                              sandbox_id="cloud-2")
    assert result.sandbox_id == "cloud-2"

    # ...or to the sandbox itself, which is the same thing said earlier.
    by_build = colleague(tmp_path, ["ok"], id="cloud-1")
    result = await by_build.run("Carry on.", session=before.session_id)
    assert result.sandbox_id == "cloud-1" and Cloud.made == 2


async def test_resume_follows_a_conversation_from_then_on(tmp_path):
    first = colleague(tmp_path, FIRST, keep=True)
    before = await first.run("Build the app.")

    later = colleague(tmp_path, ["second", "third"])
    session = await later.resume(before.session_id)
    assert session.id == before.session_id
    await later.run("Second turn.")
    third = await later.run("Third turn.")

    assert third.session_id == before.session_id and third.sandbox_id == "cloud-1"
    texts = [m.text for m in later.harness.provider.requests[1].messages if m.text]
    assert texts[0] == "Build the app." and texts[-2:] == ["second", "Third turn."]

    with pytest.raises(ConfigurationError, match="no session"):
        await later.resume("ses_never_was")


async def test_resume_works_for_an_agent_with_no_mode_and_no_sandbox(tmp_path):
    harness = Harness.local(tmp_path / "s", trace=False,
                            provider=FakeProvider(["hello Ada"]))
    first = await Agent("plain", harness=harness).run("I am Ada.")
    assert first.sandbox_id == ""

    again = Harness.local(tmp_path / "s", trace=False,
                          provider=FakeProvider(["You are Ada.", "Still Ada."]))
    agent = Agent("plain", harness=again, workspace=Workspace(tmp_path / "w"))
    await agent.resume(first.session_id)
    await agent.run("Who am I?")
    await agent.run("And now?")
    assert [m.text for m in again.provider.requests[1].messages] == [
        "I am Ada.", "hello Ada", "Who am I?", "You are Ada.", "And now?"]

    with pytest.raises(ConfigurationError, match="not a sandbox"):
        await agent.resume(first.session_id, sandbox_id="cloud-1")
    result = await agent.run("x", sandbox_id="cloud-1")
    assert "not a sandbox" in (result.error or "")


# --- when the sandbox is gone -----------------------------------------------------------

async def test_a_sandbox_that_is_gone_is_replaced_and_its_files_put_back(tmp_path):
    first = colleague(tmp_path, FIRST)                 # not kept
    before = await first.run("Build the app.")
    await first.harness.aclose()
    assert Cloud.live == {}

    later = colleague(tmp_path, [tool_call("fs_read", path="app/main.py"),
                                 "Still v1; the logo needs remaking."])
    after = await later.run("Check the app.", session=before.session_id)

    assert after.ok and after.sandbox_id == "cloud-2"
    box = later.workspace.sandbox
    assert box.replaced == "cloud-1" and "404" in box.missing_reason
    # The model is told, in the conversation, exactly what it has and has not got.
    told = later.harness.provider.requests[0].messages[-1].text
    assert told.endswith("Check the app.")
    assert "sandbox this conversation was working in (cloud-1) is no longer" in told
    assert "Put back from the conversation's record: app/main.py, build.log." in told
    assert "Could not be put back (not text, or too large): logo.png." in told
    read = later.harness.provider.requests[1].messages[-1].content[0]
    assert read.content == "print('v1')\n"

    # The conversation now lives in the new one, and says so only once.
    saved = await later.harness.sessions.load(before.session_id)
    assert saved.metadata["sandbox"]["id"] == "cloud-2"
    assert "logo.png" not in saved.metadata["sandbox_files"]
    assert any(e.action == "sandbox_replaced" for e in later.harness.audit.entries)
    again = await later.run("And again.")
    assert again.sandbox_id == "cloud-2"
    assert "Workspace notice" not in later.harness.provider.requests[-1].messages[-1].text

    # It was not being kept before, so its replacement is not either.
    await later.harness.aclose()
    assert Cloud.live == {}


async def test_on_missing_error_refuses_to_carry_on_somewhere_else(tmp_path):
    first = colleague(tmp_path, FIRST)
    before = await first.run("Build the app.")
    await first.harness.aclose()

    strict = colleague(tmp_path, ["never asked"], on_missing="error")
    result = await strict.run("Carry on.", session=before.session_id)

    assert "could not be picked back up" in (result.error or "")
    assert strict.harness.provider.requests == [] and Cloud.made == 1


async def test_an_agent_already_in_one_sandbox_will_not_serve_another_conversation(
        tmp_path):
    a = colleague(tmp_path, FIRST, keep=True)
    conversation_a = await a.run("Build A.")
    b = colleague(tmp_path, ["b", "never asked"], state="other", keep=True)
    await b.run("Build B.")

    # b is in cloud-2; conversation A's files are in cloud-1.
    b.harness.sessions = a.harness.sessions
    result = await b.run("Carry on with A.", session=conversation_a.session_id)
    assert "build an agent for each conversation" in (result.error or "")
    assert len(b.harness.provider.requests) == 1


async def test_the_record_follows_what_is_actually_in_the_sandbox(tmp_path, monkeypatch):
    agent = colleague(tmp_path, [
        tool_call("fs_write", path="keep.txt", content="k"),
        tool_call("fs_write", path="drop.txt", content="d"),
        "wrote two",
        tool_call("shell", command="rm drop.txt && echo more >> keep.txt"),
        "tidied",
    ], keep=True)
    first = await agent.run("Write two files.")
    saved = await agent.harness.sessions.load(first.session_id)
    assert set(saved.metadata["sandbox_files"]) == {"keep.txt", "drop.txt"}

    await agent.run("Tidy up.")
    saved = await agent.harness.sessions.load(first.session_id)
    assert saved.metadata["sandbox_files"] == {"keep.txt": "kmore\n"}

    # Past the cap a file is remembered by name, not by content.
    monkeypatch.setattr("agent_harness.agent._SESSION_FILE_CHARS", 3)
    capped = colleague(tmp_path, [
        tool_call("fs_write", path="small.txt", content="ab"),
        tool_call("fs_write", path="large.txt", content="x" * 50),
        "done",
    ], state="capped")
    result = await capped.run("Write.")
    saved = await capped.harness.sessions.load(result.session_id)
    assert saved.metadata["sandbox_files"] == {"small.txt": "ab", "large.txt": None}


async def test_a_local_folder_is_already_where_it_was(tmp_path):
    """No sandbox, nothing to pick up: the files never went anywhere."""
    script = [tool_call("fs_write", path="a.txt", content="x"), "wrote it"]
    harness = Harness.local(tmp_path / "s", trace=False, provider=FakeProvider(script))
    agent = Agent("colleague", mode="cowork", depth="fast", harness=harness,
                  workspace=Workspace(tmp_path / "project"))
    first = await agent.run("Write a.txt.")
    saved = await harness.sessions.load(first.session_id)
    assert first.sandbox_id == "" and "sandbox" not in saved.metadata

    again = Harness.local(tmp_path / "s", trace=False, provider=FakeProvider(
        [tool_call("fs_read", path="a.txt"), "x"]))
    later = Agent("colleague", mode="cowork", depth="fast", harness=again,
                  workspace=Workspace(tmp_path / "project"))
    result = await later.run("Read it back.", session=first.session_id)
    assert result.ok and again.provider.requests[1].messages[-1].content[0].content == "x"


# --- the sandbox itself --------------------------------------------------------------------

async def test_keeping_destroying_and_picking_up(tmp_path):
    kept = sandbox(Cloud(tmp_path, keep=True))
    await kept.awrite("a.txt", "x")
    assert kept.sandbox_id == "cloud-1" and kept.ref()["sandbox"] == "cloud"
    await kept.aclose()
    assert "cloud-1" in Cloud.live

    again = sandbox(Cloud(tmp_path, id="cloud-1"))
    assert again.sandbox.keep                    # picked up, so left as it was found
    assert await again.aread("a.txt") == "x"
    await again.destroy()                        # unless the conversation is over
    assert Cloud.live == {}

    box = Cloud(tmp_path)
    await box.start()
    with pytest.raises(Exception, match="already running"):
        box.attach("cloud-9")
    with pytest.raises(ConfigurationError, match="on_missing"):
        Cloud(tmp_path, on_missing="shrug")


def test_every_backend_takes_the_same_id(tmp_path):
    for name, alias in (("docker", "container"), ("e2b", "sandbox_id"),
                        ("daytona", "sandbox_id"), ("modal", "sandbox_id"),
                        ("kubernetes", "pod")):
        by_id = sandbox(name, id="abc").sandbox
        by_alias = sandbox(name, **{alias: "abc"}).sandbox
        assert by_id.attach_id == by_alias.attach_id == "abc", name
        assert by_id.keep and by_alias.keep
        assert not sandbox(name).sandbox.keep
        assert sandbox(name, id="abc", keep=False).sandbox.keep is False


def test_a_kept_container_is_not_removed_when_it_exits():
    assert "--rm" in DockerSandbox("a").run_argv()
    assert "--rm" not in DockerSandbox("a", keep=True).run_argv()


def test_the_cli_takes_the_ids():
    from agent_harness.cli import build_parser

    args = build_parser().parse_args([
        "run", "carry on", "--mode", "cowork", "--sandbox", "docker",
        "--session", "ses_1", "--sandbox-id", "agent-harness-abc", "--keep-sandbox"])
    assert (args.session, args.sandbox_id, args.keep_sandbox) == (
        "ses_1", "agent-harness-abc", True)


# --- docker, for real -------------------------------------------------------------------------

def _state(name: str) -> str:
    out = subprocess.run(["docker", "inspect", "-f", "{{.State.Status}}", name],
                         capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "gone"


@needs_docker
async def test_a_conversation_is_picked_up_in_its_real_container(tmp_path):
    def agent(script: list, **options: Any) -> Agent:
        harness = Harness.local(tmp_path / "state", trace=False,
                                provider=FakeProvider(script))
        return Agent("colleague", mode="cowork", depth="fast", harness=harness,
                     workspace=sandbox("docker", image=IMAGE,
                                       export_dir=tmp_path / "out", **options))

    first = agent([
        tool_call("fs_write", path="notes.md", content="# v1\n"),
        tool_call("shell", command="echo installed > /opt/tool && echo ok"),
        "Wrote the notes.",
    ], keep=True)
    before = await first.run("Start the notes.")
    name = before.sandbox_id
    await first.harness.aclose()
    try:
        assert _state(name) == "running"

        # Stopped in between — a reboot, say. Its files, and what was installed
        # outside the workspace, are still in it.
        subprocess.run(["docker", "stop", "-t", "0", name], capture_output=True)
        assert _state(name) == "exited"

        second = agent([
            tool_call("shell", command="cat notes.md /opt/tool"),
            "It is all still here.",
        ])
        after = await second.run("Is it all still there?", session=before.session_id)
        assert after.ok and after.sandbox_id == name
        assert not second.workspace.sandbox.replaced
        seen = second.harness.provider.requests[1].messages[-1].content[0].content
        assert "# v1" in seen and "installed" in seen
        await second.harness.aclose()
        assert _state(name) == "running"             # picked up, so left running

        # The conversation is over: end it for good.
        third = agent(["bye"])
        await third.resume(before.session_id)
        await third.run("That is all.")
        await third.workspace.destroy()
        assert _state(name) == "gone"

        # And if someone comes back anyway, they get a new one with the notes in it.
        fourth = agent([tool_call("fs_read", path="notes.md"), "Notes are back."])
        last = await fourth.run("One more thing.", session=before.session_id)
        assert last.ok and last.sandbox_id != name
        assert fourth.workspace.sandbox.replaced == name
        read = fourth.harness.provider.requests[1].messages[-1].content[0]
        assert read.content == "# v1\n"
        name = last.sandbox_id
        await fourth.harness.aclose()
        assert _state(name) == "gone"                # never asked to be kept
    finally:
        subprocess.run(["docker", "rm", "-f", name, before.sandbox_id],
                       capture_output=True)
