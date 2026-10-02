"""Workspace broker: an isolated place for a sub-agent to work.

Every path a workspace tool touches is resolved inside the workspace root — a
symlink or a `../..` that escapes is refused, not clamped. Shell access is off
unless you ask for it, and the shell tool asks for approval even then.

Two backends ship here: `local` (a directory per agent) and `docker` (the same
directory bind-mounted into a container, so the command itself is contained).
For a workspace that lives somewhere else entirely — a long-lived container, an
E2B, Daytona or Modal sandbox, a pod, a machine over SSH — see
`agent_harness.sandboxes`.

Every operation has an async twin (`aread`, `awrite`, `asnapshot`, ...). The
tools, and everything else in the library, go through those, so a workspace
whose files are a network call away behaves exactly like one on local disk.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError, ToolError
from ..tools import Tool, tool
from ..types import new_id

__all__ = ["Workspace", "DockerWorkspace", "WorkspaceBroker"]

# On Python 3.10 asyncio.TimeoutError is a distinct class from the builtin.
_Timeout = (TimeoutError, asyncio.TimeoutError)

# Files the harness writes for its own use — never something a run produced.
_SCRATCH = {"_snippet.py"}


def _release(proc: Any) -> None:
    """Close a finished subprocess's transport.

    Without this the transport is closed by `__del__` whenever the garbage
    collector gets to it — which, on Python 3.10, is usually after the event
    loop that owns it has closed, and it raises "Event loop is closed" from a
    destructor where nothing can catch it.
    """
    transport = getattr(proc, "_transport", None)
    if transport is None:
        return
    try:
        transport.close()
    except (RuntimeError, AttributeError):  # already closed, or loop gone
        pass


class Workspace:
    """A jailed directory plus the tools that operate inside it."""

    def __init__(
        self,
        root: str | Path,
        *,
        id: str | None = None,
        network: bool = False,
        read_only: bool = False,
        allow_shell: bool = False,
        env: dict[str, str] | None = None,
        timeout: float = 120.0,
        max_file_bytes: int = 8_000_000,
        ephemeral: bool = False,
    ) -> None:
        self.id = id or new_id("ws")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.network = network
        self.read_only = read_only
        self.allow_shell = allow_shell
        self.env = env or {}
        self.timeout = timeout
        self.max_file_bytes = max_file_bytes
        self.ephemeral = ephemeral

    #: Commands here run on the machine the harness runs on. A workspace that
    #: is its own machine says True, and its shell does not need asking for.
    isolated: bool = False

    @property
    def python(self) -> str:
        """The interpreter a snippet run in this workspace is given to."""
        return sys.executable

    # ---- path safety ---------------------------------------------------
    def resolve(self, path: str | Path) -> Path:
        """Resolve inside the jail. Anything that escapes raises."""
        candidate = Path(path)
        target = (self.root / candidate).resolve() if not candidate.is_absolute() \
            else candidate.resolve()
        if target != self.root and self.root not in target.parents:
            raise ToolError(f"path {path!r} is outside the workspace", tool="workspace")
        return target

    def _writable(self) -> None:
        if self.read_only:
            raise ToolError("this workspace is read-only", tool="workspace")

    # ---- file operations -----------------------------------------------
    def read(self, path: str) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"no such file: {path}", tool="fs_read")
        if target.stat().st_size > self.max_file_bytes:
            raise ToolError(f"{path} is larger than the {self.max_file_bytes}-byte limit",
                            tool="fs_read")
        return target.read_text(encoding="utf-8", errors="replace")

    def write(self, path: str, content: str, *, append: bool = False) -> int:
        self._writable()
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a" if append else "w", encoding="utf-8") as fh:
            fh.write(content)
        return len(content)

    def listdir(self, path: str = ".") -> list[str]:
        target = self.resolve(path)
        if not target.is_dir():
            raise ToolError(f"not a directory: {path}", tool="fs_list")
        return sorted(
            f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir()
        )

    def remove(self, path: str) -> None:
        self._writable()
        target = self.resolve(path)
        if target == self.root:
            raise ToolError("refusing to delete the workspace root", tool="fs_delete")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)

    def exists(self, path: str) -> bool:
        try:
            return self.resolve(path).exists()
        except ToolError:
            return False

    # ---- what changed ------------------------------------------------------
    def snapshot(self) -> dict[str, tuple[int, int]]:
        """Every file in the workspace, as (modified, size) by relative path."""
        seen: dict[str, tuple[int, int]] = {}
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            try:
                if path.is_file() and not path.is_symlink():
                    stat = path.stat()
                    seen[relative.as_posix()] = (stat.st_mtime_ns, stat.st_size)
            except OSError:  # removed while we were walking
                continue
        return seen

    def changed(self, since: dict[str, tuple[int, int]]) -> list[str]:
        """The files created or modified since `snapshot()` was taken."""
        return sorted(name for name, stamp in self.snapshot().items()
                      if since.get(name) != stamp and name not in _SCRATCH)

    # ---- the same, awaited ---------------------------------------------------
    # What the tools call. Here they are the methods above; a remote workspace
    # overrides these, and leaves the blocking ones refusing to block.
    async def aread(self, path: str) -> str:
        return self.read(path)

    async def aread_bytes(self, path: str) -> bytes:
        target = self.resolve(path)
        if not target.is_file():
            raise ToolError(f"no such file: {path}", tool="fs_read")
        return target.read_bytes()

    async def awrite(self, path: str, content: str, *, append: bool = False) -> int:
        return self.write(path, content, append=append)

    async def alistdir(self, path: str = ".") -> list[str]:
        return self.listdir(path)

    async def aremove(self, path: str) -> None:
        self.remove(path)

    async def aexists(self, path: str) -> bool:
        return self.exists(path)

    async def asnapshot(self) -> dict[str, tuple[int, int]]:
        return self.snapshot()

    async def achanged(self, since: dict[str, tuple[int, int]]) -> list[str]:
        return self.changed(since)

    async def materialize(self, path: str) -> Path:
        """A local file holding `path` — for anything that must open it here.

        On local disk that is the file itself. A remote workspace downloads it.
        """
        return self.resolve(path)

    async def aclose(self) -> None:
        """Release whatever the workspace holds. Safe to call twice."""
        self.cleanup()

    # ---- execution -------------------------------------------------------
    async def shell(self, command: str, *, timeout: float | None = None) -> dict[str, Any]:
        """Run a command with the workspace as its working directory."""
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            **self.env,
        }
        proc = await asyncio.create_subprocess_shell(
            command, cwd=str(self.root), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            try:
                out, err = await asyncio.wait_for(proc.communicate(),
                                                  timeout or self.timeout)
            except _Timeout:
                proc.kill()
                await proc.wait()
                raise ToolError(f"command timed out after {timeout or self.timeout}s",
                                tool="shell") from None
            return {
                "returncode": proc.returncode,
                "stdout": out.decode(errors="replace")[-20_000:],
                "stderr": err.decode(errors="replace")[-8_000:],
            }
        finally:
            _release(proc)

    # ---- tools ------------------------------------------------------------
    def tools(self) -> list[Tool]:
        """Filesystem tools bound to this workspace (plus shell, if allowed)."""
        ws = self

        @tool(name="fs_read", tags=["builtin", "fs"])
        async def fs_read(path: str) -> str:
            """Read a text file from the workspace.

            Args:
                path: path relative to the workspace root.
            """
            return await ws.aread(path)

        @tool(name="fs_write", tags=["builtin", "fs"], permission="allow")
        async def fs_write(path: str, content: str, append: bool = False) -> str:
            """Write a text file in the workspace.

            Args:
                path: path relative to the workspace root.
                content: the full text to write.
                append: append instead of replacing the file.
            """
            written = await ws.awrite(path, content, append=append)
            return f"wrote {written} characters to {path}"

        @tool(name="fs_list", tags=["builtin", "fs"])
        async def fs_list(path: str = ".") -> list[str]:
            """List a directory in the workspace.

            Args:
                path: directory relative to the workspace root.
            """
            return await ws.alistdir(path)

        @tool(name="fs_delete", tags=["builtin", "fs"], permission="ask")
        async def fs_delete(path: str) -> str:
            """Delete a file or directory in the workspace.

            Args:
                path: path relative to the workspace root.
            """
            await ws.aremove(path)
            return f"deleted {path}"

        built = [fs_read, fs_write, fs_list, fs_delete]

        if ws.allow_shell:
            # Inside its own machine a command can only hurt the sandbox, and
            # asking before each one would make the sandbox pointless. On this
            # machine it asks, every time.
            @tool(name="shell", tags=["builtin", "exec"],
                  permission="allow" if ws.isolated else "ask")
            async def shell_tool(command: str, timeout: float = 60.0) -> dict[str, Any]:
                """Run a shell command inside the workspace.

                Args:
                    command: the command line to run.
                    timeout: seconds before the command is killed.
                """
                return await ws.shell(command, timeout=timeout)

            built.append(shell_tool)
        return built

    def cleanup(self) -> None:
        if self.ephemeral and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<Workspace {self.id} at {self.root}>"


class DockerWorkspace(Workspace):
    """Same directory, but commands run inside a container.

    Requires the `docker` CLI on PATH. Falls back to nothing — if docker is
    missing the constructor raises rather than quietly running on the host.
    """

    def __init__(self, root: str | Path, *, image: str = "python:3.12-slim",
                 docker_args: list[str] | None = None, **kw: Any) -> None:
        super().__init__(root, **kw)
        if shutil.which("docker") is None:
            raise ToolError("docker is not on PATH; use the local workspace backend",
                            tool="workspace")
        self.image = image
        self.docker_args = docker_args or []

    @property
    def python(self) -> str:
        return "python3"

    async def shell(self, command: str, *, timeout: float | None = None) -> dict[str, Any]:
        args = [
            "docker", "run", "--rm", "-i",
            "-v", f"{self.root}:/work", "-w", "/work",
            "--memory", "2g", "--cpus", "2",
        ]
        if not self.network:
            args += ["--network", "none"]
        if self.read_only:
            args += ["--read-only"]
        args += [*self.docker_args, self.image, "sh", "-lc", command]

        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            try:
                out, err = await asyncio.wait_for(proc.communicate(),
                                                  timeout or self.timeout)
            except _Timeout:
                proc.kill()
                await proc.wait()
                raise ToolError(f"container timed out after {timeout or self.timeout}s",
                                tool="shell") from None
            return {
                "returncode": proc.returncode,
                "stdout": out.decode(errors="replace")[-20_000:],
                "stderr": err.decode(errors="replace")[-8_000:],
            }
        finally:
            _release(proc)


class WorkspaceBroker:
    """Hands out workspaces: one shared, one per isolated sub-agent.

        WorkspaceBroker()                                   # a folder each
        WorkspaceBroker(sandbox="docker", image="node:22")  # a container each
        WorkspaceBroker(sandbox="e2b", template="base")     # a micro-VM each

    With `sandbox=`, every workspace it hands out is its own sandbox, built
    with the options given here; see `agent_harness.sandboxes`.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        backend: str = "local",
        sandbox: str | dict[str, Any] | None = None,
        image: str | None = None,
        network: bool | None = None,
        allow_shell: bool = False,
        ephemeral: bool | None = None,
        **options: Any,
    ) -> None:
        # Any backend that is not one of the two built in here is a sandbox.
        if sandbox is None and backend not in ("local", "docker"):
            sandbox = backend
        if options and sandbox is None:
            raise ConfigurationError(
                f"{', '.join(sorted(options))}: these are sandbox options — say "
                "which sandbox they are for with sandbox=")
        self.sandbox = sandbox
        # What was said here about the image and the network is said to the
        # sandbox too; what was left unsaid is the sandbox's own default.
        self.options = {**({"image": image} if image is not None else {}),
                        **({"network": network} if network is not None else {}),
                        **options}
        self.ephemeral = ephemeral if ephemeral is not None else root is None
        # A sandbox keeps its files in itself, so only a local workspace needs a
        # root — and a temporary one is not made until something asks for it.
        self._root = Path(root) if root else None
        if self._root is not None and sandbox is None:
            self._root.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.image = image or "python:3.12-slim"
        self.network = bool(network)
        self.allow_shell = allow_shell
        self._open: dict[str, Workspace] = {}

    @property
    def root(self) -> Path:
        """Where local workspaces live. Made on first use."""
        if self._root is None:
            self._root = Path(tempfile.mkdtemp(prefix="agent-harness-"))
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def _make(self, name: str, **kw: Any) -> Workspace:
        if self.sandbox is not None:
            from ..sandboxes import sandbox

            return sandbox(self.sandbox, **{**self.options, **kw})
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = self.root / safe
        if self.backend == "docker":
            return DockerWorkspace(path, image=self.image, network=self.network,
                                   allow_shell=self.allow_shell, **kw)
        return Workspace(path, network=self.network, allow_shell=self.allow_shell, **kw)

    def acquire(self, name: str = "shared", *, isolated: bool = True,
                **kw: Any) -> Workspace:
        """An isolated workspace per name, or the shared one that files pass through."""
        key = name if isolated else "shared"
        if key in self._open:
            return self._open[key]
        workspace = self._make(key, **kw)
        self._open[key] = workspace
        return workspace

    def adopt(self, workspace: Workspace) -> Workspace:
        """Take charge of a workspace built elsewhere, so it is closed with the
        rest. An agent handed a sandbox does this for you."""
        if not any(held is workspace for held in self._open.values()):
            self._open[f"adopted:{workspace.id}"] = workspace
        return workspace

    def shared(self) -> Workspace:
        """Where parallel sub-agents hand files to each other."""
        return self.acquire("shared", isolated=False)

    def release(self, name: str) -> None:
        workspace = self._open.pop(name, None)
        if workspace:
            workspace.cleanup()

    def cleanup(self) -> None:
        for workspace in list(self._open.values()):
            workspace.cleanup()
        self._open.clear()
        if self.ephemeral and self._root is not None and self._root.exists():
            shutil.rmtree(self._root, ignore_errors=True)

    async def aclose(self) -> None:
        """Stop every sandbox and remove what was temporary. A sandbox that
        will not stop does not keep the others running."""
        for workspace in list(self._open.values()):
            try:
                await workspace.aclose()
            except Exception:  # noqa: S110 - closing must reach every one of them
                pass
        self.cleanup()
