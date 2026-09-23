"""Workspace broker: an isolated place for a sub-agent to work.

Every path a workspace tool touches is resolved inside the workspace root — a
symlink or a `../..` that escapes is refused, not clamped. Shell access is off
unless you ask for it, and the shell tool asks for approval even then.

Two backends ship: `local` (a directory per agent) and `docker` (the same
directory bind-mounted into a container, so the command itself is contained).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..errors import ToolError
from ..tools import Tool, tool
from ..types import new_id

__all__ = ["Workspace", "DockerWorkspace", "WorkspaceBroker"]

# On Python 3.10 asyncio.TimeoutError is a distinct class from the builtin.
_Timeout = (TimeoutError, asyncio.TimeoutError)


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
        def fs_read(path: str) -> str:
            """Read a text file from the workspace.

            Args:
                path: path relative to the workspace root.
            """
            return ws.read(path)

        @tool(name="fs_write", tags=["builtin", "fs"], permission="allow")
        def fs_write(path: str, content: str, append: bool = False) -> str:
            """Write a text file in the workspace.

            Args:
                path: path relative to the workspace root.
                content: the full text to write.
                append: append instead of replacing the file.
            """
            written = ws.write(path, content, append=append)
            return f"wrote {written} characters to {path}"

        @tool(name="fs_list", tags=["builtin", "fs"])
        def fs_list(path: str = ".") -> list[str]:
            """List a directory in the workspace.

            Args:
                path: directory relative to the workspace root.
            """
            return ws.listdir(path)

        @tool(name="fs_delete", tags=["builtin", "fs"], permission="ask")
        def fs_delete(path: str) -> str:
            """Delete a file or directory in the workspace.

            Args:
                path: path relative to the workspace root.
            """
            ws.remove(path)
            return f"deleted {path}"

        built = [fs_read, fs_write, fs_list, fs_delete]

        if ws.allow_shell:
            @tool(name="shell", tags=["builtin", "exec"], permission="ask")
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
    """Hands out workspaces: one shared, one per isolated sub-agent."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        backend: str = "local",
        image: str = "python:3.12-slim",
        network: bool = False,
        allow_shell: bool = False,
        ephemeral: bool | None = None,
    ) -> None:
        self.ephemeral = ephemeral if ephemeral is not None else root is None
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="agent-harness-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.image = image
        self.network = network
        self.allow_shell = allow_shell
        self._open: dict[str, Workspace] = {}

    def _make(self, path: Path, **kw: Any) -> Workspace:
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
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        workspace = self._make(self.root / safe, **kw)
        self._open[key] = workspace
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
        if self.ephemeral and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
