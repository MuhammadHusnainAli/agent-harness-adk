"""The sandbox contract, and the workspace that sits on top of one.

A `Sandbox` is somewhere else to run things: a container, a micro-VM, a pod, a
machine over SSH. The contract is deliberately one method wide —

    class MySandbox(Sandbox):
        name = "mine"

        async def _exec(self, command, *, cwd, env, timeout):
            out = await my_platform.run(command, cwd=cwd)
            return ExecResult(out.code, out.stdout, out.stderr)

— because everything else can be said in shell. Reading a file is `base64`,
writing one is `base64 -d`, listing is `ls`, and "what changed" is `find` and
`stat`, all written to run on busybox as well as GNU. A backend with a real file
API overrides `read_bytes` and `write_bytes` and gets faster; one without still
works on day one.

`SandboxWorkspace` is what an agent holds. It is a `Workspace`, so the file
tools, the shell tool, `run_python`, `parse_document` and cowork's file
collection work on it unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import posixpath
import shlex
import tempfile
import time
import zlib
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from ..errors import ConfigurationError, ToolError
from ..runtime.workspace import Workspace, _release
from ..types import new_id

__all__ = ["ExecResult", "Sandbox", "ProcessSandbox", "SandboxWorkspace",
           "run_process"]

# On Python 3.10 asyncio.TimeoutError is a distinct class from the builtin.
_Timeout = (TimeoutError, asyncio.TimeoutError)

_MISSING = "__AH_MISSING__"
_TOO_BIG = "__AH_TOO_BIG__"
# One argument to exec may not exceed 128 KiB on Linux, and the whole command
# travels as one. 48 KiB of bytes is 64 KiB of base64, comfortably inside it.
_CHUNK = 48 * 1024
# How long past its own deadline a command is given before we stop waiting.
_GRACE = 10.0

q = shlex.quote


@dataclass
class ExecResult:
    """What a command did."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class Sandbox(ABC):
    """Somewhere else to run commands. Started on first use, stopped once."""

    name: ClassVar[str] = "sandbox"
    #: Commands here cannot touch the machine the harness runs on.
    isolated: ClassVar[bool] = True
    #: What a URL's remainder means: `docker://python:3.12` sets `image`.
    url_option: ClassVar[str] = ""
    driver_hint: ClassVar[str] = ""

    def __init__(self, *, id: str | None = None, workdir: str | None = None,
                 env: dict[str, str] | None = None, timeout: float = 120.0,
                 start_timeout: float = 300.0, keep: bool | None = None,
                 on_missing: str = "new", python: str = "python3") -> None:
        if on_missing not in ("new", "error"):
            raise ConfigurationError(
                f"on_missing must be \"new\" or \"error\" — got {on_missing!r}")
        #: The sandbox to pick back up instead of creating one. Whatever a run
        #: reported as `result.sandbox_id` goes here.
        self.attach_id = id or None
        #: Where the agent works, inside the sandbox. None means a `workspace`
        #: folder under wherever the sandbox starts, settled at start.
        self.workdir = workdir
        self.env = dict(env or {})
        self.timeout = timeout
        self.start_timeout = start_timeout
        #: Leave it running at `stop()`, to be picked back up later by its id.
        #: One that was itself picked up is kept unless told otherwise.
        self.keep = keep if keep is not None else self.attach_id is not None
        #: When the sandbox to pick up is gone: start a new one in its place
        #: ("new"), or fail ("error").
        self.on_missing = on_missing
        #: The id of the sandbox this one had to replace, and why it was gone.
        self.replaced = ""
        self.missing_reason = ""
        self._keep_before = self.keep
        self.python = python
        self.id = ""
        self.started_at = 0.0
        self._started = False
        self._stopped = False
        self._lock = asyncio.Lock()

    # ---- what a backend provides ---------------------------------------------
    async def _start(self) -> None:  # noqa: B027 - optional, not abstract
        """Create the sandbox — or, when `self.attach_id` is set, connect to
        that one and raise if it is not there. Set `self.id`."""

    @abstractmethod
    async def _exec(self, command: str, *, cwd: str | None, env: dict[str, str],
                    timeout: float) -> ExecResult:
        """Run a shell command and wait for it. `cwd=None` is the sandbox's own."""

    async def _stop(self) -> None:  # noqa: B027 - optional, not abstract
        """Destroy the sandbox."""

    # ---- lifecycle -------------------------------------------------------------
    async def start(self) -> Sandbox:
        if self._started:
            return self
        async with self._lock:
            if self._started:
                return self
            if self._stopped:
                raise ToolError(f"this {self.name} sandbox has been stopped",
                                tool="sandbox")
            try:
                await self._begin()
            except ConfigurationError:
                raise          # a missing key or driver: a new one would fail too
            except Exception as exc:
                if self.attach_id is None or self.on_missing == "error":
                    raise
                # The one to pick up is gone — expired, deleted, never there.
                # Work goes on in a new one, and the caller is told which.
                self.replaced, self.attach_id, self.id = self.attach_id, None, ""
                self.missing_reason = str(exc)[:300]
                # The new one is ours: whether it is kept is as it was set up,
                # not as it would have been for one we had only picked up.
                self.keep = self._keep_before
                await self._begin()
            self.id = self.id or new_id("sbx")
            if self.workdir is None:
                here = await self._exec("pwd", cwd=None, env={}, timeout=30)
                base = here.stdout.strip().splitlines()[-1] if here.stdout.strip() else ""
                self.workdir = posixpath.join(base or "/", "workspace")
            made = await self._exec(f"mkdir -p {q(self.workdir)}", cwd=None, env={},
                                    timeout=30)
            if not made.ok:
                raise ToolError(
                    f"the {self.name} sandbox started but {self.workdir} could not "
                    f"be created: {(made.stderr or made.stdout).strip()[:300]}",
                    tool="sandbox")
            self.started_at = time.time()
            self._started = True
        return self

    @property
    def running(self) -> bool:
        return self._started

    async def _begin(self) -> None:
        """Start it, with whatever went wrong said as one kind of error — a
        platform's own exception is not something a run knows how to survive."""
        try:
            await asyncio.wait_for(self._start(), self.start_timeout)
        except _Timeout:
            raise ToolError(
                f"the {self.name} sandbox did not start within "
                f"{self.start_timeout:.0f}s", tool="sandbox") from None
        except (ConfigurationError, ToolError):
            raise
        except Exception as exc:
            what = (f"{self.name} sandbox {self.attach_id} could not be picked up"
                    if self.attach_id else f"the {self.name} sandbox could not start")
            raise ToolError(f"{what}: {exc}", tool="sandbox") from exc

    def attach(self, id: str) -> Sandbox:
        """Pick up the sandbox with this id when this one starts, and leave it
        running afterwards. Only before the first use."""
        if self._started:
            raise ToolError(
                f"this {self.name} sandbox is already running as {self.id}; it "
                "cannot become another one", tool="sandbox")
        self.attach_id = id
        self._keep_before, self.keep = self.keep, True
        return self

    async def stop(self, *, destroy: bool = False) -> None:
        """Destroy the sandbox, unless it was asked to be kept. Safe to repeat.

        `destroy=True` ends a kept one too — the conversation is over.
        """
        if self._stopped:
            return
        self._stopped = True
        if self._started and (destroy or not self.keep):
            await self._stop()
        self._started = False

    async def __aenter__(self) -> Sandbox:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ---- running things ----------------------------------------------------------
    async def exec(self, command: str, *, cwd: str | None = None,
                   env: dict[str, str] | None = None,
                   timeout: float | None = None) -> ExecResult:
        """Run a shell command in the sandbox. A timeout is a result, not a raise."""
        await self.start()
        limit = float(timeout or self.timeout)
        try:
            return await asyncio.wait_for(
                self._exec(command, cwd=cwd or self.workdir,
                           env={**self.env, **(env or {})}, timeout=limit),
                limit + _GRACE)
        except _Timeout:
            return ExecResult(124, "", f"timed out after {limit:.0f}s", timed_out=True)

    @staticmethod
    def script(command: str, *, cwd: str | None = None,
               env: dict[str, str] | None = None) -> str:
        """`command` with its directory and environment said in shell — for a
        backend whose exec call takes neither."""
        parts: list[str] = []
        if cwd:
            parts.append(f"cd {q(cwd)}")
        for key, value in (env or {}).items():
            parts.append(f"export {key}={q(str(value))}")
        parts.append(command)
        return " && ".join(parts)

    async def _sh(self, script: str, *, what: str) -> ExecResult:
        result = await self.exec(script)
        if result.timed_out:
            raise ToolError(f"{what}: {result.stderr}", tool="sandbox")
        return result

    # ---- files, in shell -----------------------------------------------------------
    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        limit = max_bytes if max_bytes is not None else 2**62
        result = await self._sh(
            f"if [ ! -f {q(path)} ]; then echo {_MISSING}; "
            f"elif [ \"$(wc -c < {q(path)})\" -gt {limit} ]; then echo {_TOO_BIG}; "
            f"else base64 < {q(path)} 2>/dev/null; fi", what=f"reading {path}")
        text = result.stdout.strip()
        if _MISSING in text:
            raise FileNotFoundError(path)
        if _TOO_BIG in text:
            raise ToolError(f"{path} is larger than the {limit}-byte limit",
                            tool="fs_read")
        try:
            return base64.b64decode("".join(text.split()), validate=True)
        except ValueError:
            raise ToolError(
                f"could not read {path}: "
                f"{(result.stderr or result.stdout).strip()[:300]}",
                tool="fs_read") from None

    async def write_bytes(self, path: str, data: bytes, *, append: bool = False) -> None:
        parent = posixpath.dirname(path) or "/"
        first = ">>" if append else ">"
        if not data:
            result = await self._sh(f"mkdir -p {q(parent)} && : {first} {q(path)}",
                                    what=f"writing {path}")
            if not result.ok:
                raise ToolError(f"could not write {path}: "
                                f"{(result.stderr or result.stdout).strip()[:300]}",
                                tool="fs_write")
            return
        for offset in range(0, len(data), _CHUNK):
            chunk = base64.b64encode(data[offset:offset + _CHUNK]).decode()
            redirect = first if offset == 0 else ">>"
            prefix = f"mkdir -p {q(parent)} && " if offset == 0 else ""
            result = await self._sh(
                f"{prefix}printf %s {q(chunk)} | base64 -d {redirect} {q(path)}",
                what=f"writing {path}")
            if not result.ok:
                raise ToolError(f"could not write {path}: "
                                f"{(result.stderr or result.stdout).strip()[:300]}",
                                tool="fs_write")

    async def list_dir(self, path: str) -> list[str]:
        result = await self._sh(
            f"if [ -d {q(path)} ]; then ls -1Ap {q(path)}; else echo {_MISSING}; fi",
            what=f"listing {path}")
        if _MISSING in result.stdout:
            raise NotADirectoryError(path)
        return sorted(line for line in result.stdout.splitlines() if line.strip())

    async def remove(self, path: str) -> None:
        result = await self._sh(f"rm -rf -- {q(path)}", what=f"deleting {path}")
        if not result.ok:
            raise ToolError(f"could not delete {path}: "
                            f"{(result.stderr or result.stdout).strip()[:300]}",
                            tool="fs_delete")

    async def exists(self, path: str) -> bool:
        result = await self._sh(f"test -e {q(path)} && echo yes || echo no",
                                what=f"checking {path}")
        return result.stdout.strip().endswith("yes")

    async def snapshot(self, root: str) -> dict[str, tuple[int, int]]:
        """Every file under `root` as (a stamp of its modified time, its size)."""
        result = await self._sh(
            f"cd {q(root)} && find . -type f ! -path '*/.*' "
            "-exec stat -c '%s|%y|%n' {} + 2>/dev/null", what="looking at the files")
        seen: dict[str, tuple[int, int]] = {}
        for line in result.stdout.splitlines():
            size, _, rest = line.partition("|")
            stamp, _, name = rest.partition("|")
            if not size.strip().isdigit() or not name:
                continue
            seen[name[2:] if name.startswith("./") else name] = (
                zlib.crc32(stamp.encode()), int(size))
        return seen

    def info(self) -> dict[str, Any]:
        return {"sandbox": self.name, "id": self.id, "workdir": self.workdir,
                "isolated": self.isolated, "running": self._started,
                "kept": self.keep, "replaced": self.replaced}

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        state = "running" if self._started else "stopped" if self._stopped else "idle"
        return f"<{type(self).__name__} {self.id or '-'} {state}>"


# ---------------------------------------------------------------------------
# sandboxes reached through a command line
# ---------------------------------------------------------------------------
Runner = Callable[[list[str], "bytes | None", float], Awaitable["tuple[int, bytes, bytes]"]]


class ProcessTimeout(Exception):
    """The local client process outlived its deadline and was killed."""


async def run_process(argv: list[str], data: bytes | None = None,
                      timeout: float = 120.0) -> tuple[int, bytes, bytes]:
    """Run a local command to completion: (exit code, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        raise ConfigurationError(f"{argv[0]} is not on PATH") from None
    try:
        try:
            out, err = await asyncio.wait_for(proc.communicate(data), timeout)
        except _Timeout:
            proc.kill()
            await proc.wait()
            raise ProcessTimeout(argv[0]) from None
        return proc.returncode or 0, out, err
    finally:
        _release(proc)


class ProcessSandbox(Sandbox):
    """A sandbox driven through a CLI — `docker exec`, `ssh`, `kubectl exec`.

    A subclass says how a shell script is carried inside (`_wrap`). Files then
    travel over the command's own stdin and stdout, which is binary-safe and
    needs no chunking.
    """

    def __init__(self, *, runner: Runner | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self._run: Runner = runner or run_process
        #: Whether the sandbox has `timeout(1)`, found out at start. With it a
        #: command that overruns is killed inside the sandbox, not just here.
        self._has_timeout = False

    @abstractmethod
    def _wrap(self, script: str) -> list[str]:
        """The local argv that runs `script` in a shell inside the sandbox."""

    async def _probe(self) -> None:
        try:
            code, out, _ = await self._run(
                self._wrap("command -v timeout >/dev/null 2>&1 && echo yes"), None, 30)
        except ProcessTimeout:
            return
        self._has_timeout = code == 0 and b"yes" in out

    async def start(self) -> Sandbox:
        fresh = not self._started
        await super().start()
        if fresh:
            await self._probe()
        return self

    async def _exec(self, command: str, *, cwd: str | None, env: dict[str, str],
                    timeout: float) -> ExecResult:
        body = self.script(command, cwd=cwd, env=env)
        if self._has_timeout:
            body = f"timeout -s KILL {max(1, int(timeout))} sh -c {q(body)}"
        started = time.monotonic()
        try:
            code, out, err = await self._run(self._wrap(body), None, timeout + _GRACE / 2)
        except ProcessTimeout:
            return ExecResult(124, "", f"timed out after {timeout:.0f}s", timed_out=True)
        killed = (self._has_timeout and code == 137
                  and time.monotonic() - started >= timeout - 0.5)
        return ExecResult(code, out.decode(errors="replace"),
                          f"timed out after {timeout:.0f}s" if killed
                          else err.decode(errors="replace"), timed_out=killed)

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        await self.start()
        limit = max_bytes if max_bytes is not None else 2**62
        code, out, err = await self._run(self._wrap(
            f"if [ ! -f {q(path)} ]; then exit 44; "
            f"elif [ \"$(wc -c < {q(path)})\" -gt {limit} ]; then exit 45; "
            f"else cat {q(path)}; fi"), None, self.timeout)
        if code == 44:
            raise FileNotFoundError(path)
        if code == 45:
            raise ToolError(f"{path} is larger than the {limit}-byte limit",
                            tool="fs_read")
        if code != 0:
            raise ToolError(f"could not read {path}: "
                            f"{err.decode(errors='replace').strip()[:300]}",
                            tool="fs_read")
        return out

    async def write_bytes(self, path: str, data: bytes, *, append: bool = False) -> None:
        await self.start()
        parent = posixpath.dirname(path) or "/"
        code, _, err = await self._run(self._wrap(
            f"mkdir -p {q(parent)} && cat {'>>' if append else '>'} {q(path)}"),
            data, self.timeout)
        if code != 0:
            raise ToolError(f"could not write {path}: "
                            f"{err.decode(errors='replace').strip()[:300]}",
                            tool="fs_write")


# ---------------------------------------------------------------------------
# the workspace an agent holds
# ---------------------------------------------------------------------------
class SandboxWorkspace(Workspace):
    """A workspace whose files and commands live in a sandbox.

    Paths are jailed to the sandbox's working directory by name — `..` cannot
    climb out of it. That is tidiness rather than security here: the sandbox
    itself is the wall, and nothing inside it can reach this machine.
    """

    def __init__(self, sandbox: Sandbox, *, id: str | None = None,
                 read_only: bool = False, allow_shell: bool = True,
                 timeout: float | None = None, max_file_bytes: int = 8_000_000,
                 export_dir: str | Path | None = None) -> None:
        # Deliberately not calling Workspace.__init__: there is no local root.
        self.sandbox = sandbox
        self.id = id or new_id("ws")
        self.network = True
        self.read_only = read_only
        self.allow_shell = allow_shell
        self.env = sandbox.env
        self.timeout = timeout or sandbox.timeout
        self.max_file_bytes = max_file_bytes
        self.ephemeral = not sandbox.keep
        self._export = Path(export_dir) if export_dir else None

    @property
    def isolated(self) -> bool:  # type: ignore[override]
        return self.sandbox.isolated

    @property
    def python(self) -> str:
        return self.sandbox.python

    @property
    def root(self) -> PurePosixPath:  # type: ignore[override]
        """The working directory inside the sandbox."""
        return PurePosixPath(self.sandbox.workdir or "workspace")

    # ---- paths -----------------------------------------------------------------
    def resolve(self, path: str | Path) -> PurePosixPath:  # type: ignore[override]
        root = self.root
        candidate = PurePosixPath(str(path))
        joined = candidate if candidate.is_absolute() else root / candidate
        target = PurePosixPath(posixpath.normpath(str(joined)))
        if target != root and root not in target.parents:
            raise ToolError(f"path {path!r} is outside the workspace", tool="workspace")
        return target

    def uri(self, path: str = "") -> str:
        """Where a file is, for something that cannot be handed a local path."""
        target = self.resolve(path) if path else self.root
        return f"{self.sandbox.name}://{self.sandbox.id}{target}"

    def _remote(self, *_: Any, **__: Any) -> Any:
        raise ToolError(
            f"this workspace is a {self.sandbox.name} sandbox, so its files are a "
            "call away — await the async form (aread, awrite, alistdir, aremove, "
            "aexists, asnapshot, achanged)", tool="workspace")

    read = write = listdir = remove = snapshot = changed = _remote  # type: ignore[assignment]

    def exists(self, path: str) -> bool:
        self._remote()
        return False  # pragma: no cover

    # ---- files -------------------------------------------------------------------
    async def aread_bytes(self, path: str) -> bytes:
        await self.sandbox.start()
        target = self.resolve(path)
        try:
            return await self.sandbox.read_bytes(str(target),
                                                 max_bytes=self.max_file_bytes)
        except FileNotFoundError:
            raise ToolError(f"no such file: {path}", tool="fs_read") from None

    async def aread(self, path: str) -> str:
        return (await self.aread_bytes(path)).decode("utf-8", errors="replace")

    async def awrite(self, path: str, content: str | bytes, *,
                     append: bool = False) -> int:
        self._writable()
        await self.sandbox.start()
        data = content if isinstance(content, bytes) else content.encode("utf-8")
        await self.sandbox.write_bytes(str(self.resolve(path)), data, append=append)
        return len(content)

    async def alistdir(self, path: str = ".") -> list[str]:
        await self.sandbox.start()
        try:
            return await self.sandbox.list_dir(str(self.resolve(path)))
        except NotADirectoryError:
            raise ToolError(f"not a directory: {path}", tool="fs_list") from None

    async def aremove(self, path: str) -> None:
        self._writable()
        await self.sandbox.start()
        target = self.resolve(path)
        if target == self.root:
            raise ToolError("refusing to delete the workspace root", tool="fs_delete")
        await self.sandbox.remove(str(target))

    async def aexists(self, path: str) -> bool:
        await self.sandbox.start()
        try:
            return await self.sandbox.exists(str(self.resolve(path)))
        except ToolError:
            return False

    async def asnapshot(self) -> dict[str, tuple[int, int]]:
        await self.sandbox.start()
        return await self.sandbox.snapshot(str(self.root))

    async def achanged(self, since: dict[str, tuple[int, int]]) -> list[str]:
        now = await self.asnapshot()
        return sorted(name for name, stamp in now.items()
                      if since.get(name) != stamp and name != "_snippet.py")

    async def materialize(self, path: str) -> Path:
        """Download `path` and return where it landed on this machine."""
        target = self.resolve(path)
        data = await self.aread_bytes(path)
        if self._export is None:
            self._export = Path(tempfile.mkdtemp(prefix="agent-harness-sandbox-"))
        local = self._export / target.relative_to(self.root)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        return local

    # ---- execution -----------------------------------------------------------------
    async def shell(self, command: str, *, timeout: float | None = None) -> dict[str, Any]:
        limit = timeout or self.timeout
        result = await self.sandbox.exec(command, timeout=limit)
        if result.timed_out:
            raise ToolError(f"command timed out after {limit}s", tool="shell")
        return {"returncode": result.returncode, "stdout": result.stdout[-20_000:],
                "stderr": result.stderr[-8_000:]}

    # ---- lifecycle -------------------------------------------------------------------
    async def start(self) -> SandboxWorkspace:
        await self.sandbox.start()
        return self

    def cleanup(self) -> None:
        """Nothing that can be done without awaiting — see `aclose()`."""

    async def aclose(self) -> None:
        await self.sandbox.stop()

    async def destroy(self) -> None:
        """End the sandbox even if it was being kept for later."""
        await self.sandbox.stop(destroy=True)

    @property
    def sandbox_id(self) -> str:
        """The id to pick this sandbox back up with. "" until it has started."""
        return self.sandbox.id

    def ref(self) -> dict[str, str]:
        """Which sandbox this is, as plain data — what a session remembers."""
        return {"sandbox": self.sandbox.name, "id": self.sandbox.id,
                "workdir": self.sandbox.workdir or ""}

    async def __aenter__(self) -> SandboxWorkspace:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def check(self) -> dict[str, Any]:
        """Start the sandbox and prove each thing an agent will ask of it.

        Returns every step with whether it held, so a misconfigured sandbox is
        found here rather than halfway through a run.
        """
        steps: list[dict[str, Any]] = []
        started = time.monotonic()
        probe = f".ah-check-{new_id()}"
        payload = bytes(range(256)) * 4

        async def step(name: str, action: Callable[[], Awaitable[Any]]) -> bool:
            began = time.monotonic()
            try:
                detail = await action()
                steps.append({"step": name, "ok": True, "detail": str(detail or ""),
                              "seconds": round(time.monotonic() - began, 2)})
                return True
            except Exception as exc:
                steps.append({"step": name, "ok": False,
                              "detail": f"{type(exc).__name__}: {exc}",
                              "seconds": round(time.monotonic() - began, 2)})
                return False

        async def run() -> str:
            result = await self.sandbox.exec("echo sandbox-ok")
            if "sandbox-ok" not in result.stdout:
                raise ToolError(f"exit {result.returncode}: "
                                f"{(result.stderr or result.stdout).strip()[:200]}")
            return f"in {self.root}"

        async def roundtrip() -> str:
            await self.sandbox.write_bytes(f"{self.root}/{probe}/dir/bytes.bin", payload)
            back = await self.sandbox.read_bytes(f"{self.root}/{probe}/dir/bytes.bin")
            if back != payload:
                raise ToolError(f"wrote {len(payload)} bytes, read back {len(back)}")
            return f"{len(payload)} bytes, binary-safe"

        async def listing() -> str:
            names = await self.sandbox.list_dir(f"{self.root}/{probe}")
            if names != ["dir/"]:
                raise ToolError(f"expected ['dir/'], got {names}")
            return "directories are marked"

        async def changes() -> str:
            before = await self.sandbox.snapshot(f"{self.root}/{probe}")
            await self.sandbox.write_bytes(f"{self.root}/{probe}/new.txt", b"x")
            after = await self.sandbox.snapshot(f"{self.root}/{probe}")
            fresh = sorted(set(after) - set(before))
            if fresh != ["new.txt"] or "dir/bytes.bin" not in before:
                raise ToolError(f"expected ['new.txt'], saw {fresh}")
            return "new files are seen"

        async def tidy() -> str:
            await self.sandbox.remove(f"{self.root}/{probe}")
            if await self.sandbox.exists(f"{self.root}/{probe}"):
                raise ToolError("the probe directory is still there")
            return ""

        async def begin() -> str:
            await self.sandbox.start()
            return self.sandbox.id

        if await step("start", begin):
            for name, action in (("exec", run), ("write and read", roundtrip),
                                 ("list", listing), ("detect changes", changes),
                                 ("delete", tidy)):
                await step(name, action)
        return {**self.sandbox.info(), "ok": all(s["ok"] for s in steps),
                "steps": steps, "seconds": round(time.monotonic() - started, 2)}

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<SandboxWorkspace {self.sandbox.name} {self.sandbox.id or '-'}>"
