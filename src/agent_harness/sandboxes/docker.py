"""A long-lived container: Docker, Podman, or anything with the same CLI.

    sandbox("docker", image="python:3.12-slim")
    sandbox("docker", image="node:22", mount="./project", network=True)
    sandbox("podman", image="python:3.12-slim")
    sandbox("docker", image="python:3.12-slim", runtime="runsc")      # gVisor

Unlike `DockerWorkspace`, which starts a fresh container for every command, the
container here lives as long as the sandbox does — so what the agent installs in
one step is still there in the next.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from ..errors import ConfigurationError, ToolError
from ..types import new_id
from .base import ProcessSandbox, ProcessTimeout, q, run_process

__all__ = ["DockerSandbox"]


class DockerSandbox(ProcessSandbox):
    """One container, kept running, with every command `exec`ed into it.

    Defaults lean closed: no network, 2 CPUs, 2 GB, a process limit, and no way
    to gain privileges. The container removes itself when it stops, is stopped
    when the sandbox is, and has a lifetime (`ttl`) so that one orphaned by a
    crash does not run for ever.

    Args:
        image: the image to run. Pulled on first use if it is not present.
        mount: a directory on this machine to work in. It appears at `workdir`,
            and what the agent writes there stays after the container is gone.
            The container then runs as you, so the files are yours.
        network: let the container reach the network.
        container: attach to a container that is already running, and leave it
            running afterwards.
        runtime: an OCI runtime — "runsc" for gVisor, "kata" for Kata.
        ttl: seconds until the container ends by itself. None is no limit.
        args: anything else `docker run` should be given.
        cli: "docker", "podman", "nerdctl" — whichever is on PATH.
    """

    name: ClassVar[str] = "docker"
    url_option: ClassVar[str] = "image"

    def __init__(self, image: str = "python:3.12-slim", *,
                 mount: str | Path | None = None, mount_read_only: bool = False,
                 network: bool = False, container: str | None = None,
                 memory: str | None = "2g", cpus: float | str | None = 2,
                 pids: int | None = 512, user: str | None = None,
                 runtime: str | None = None, ttl: int | None = 14_400,
                 args: Sequence[str] = (), cli: str = "docker",
                 workdir: str | None = "/workspace", **kw: Any) -> None:
        # A container we were pointed at is somebody else's to remove.
        kw.setdefault("keep", container is not None)
        super().__init__(workdir=workdir, **kw)
        self.image = image
        self.mount = Path(mount).expanduser().resolve() if mount else None
        self.mount_read_only = mount_read_only
        self.network = network
        self.attached = container is not None
        self.container = container or f"agent-harness-{new_id()}"
        self.memory, self.cpus, self.pids = memory, cpus, pids
        self.user = user
        self.runtime = runtime
        self.ttl = ttl
        self.args = [str(a) for a in args]
        self.cli = cli

    def _wrap(self, script: str) -> list[str]:
        return [self.cli, "exec", "-i", self.container, "sh", "-c", script]

    def run_argv(self) -> list[str]:
        """The `docker run` this sandbox starts its container with."""
        argv = [self.cli, "run", "-d", "--rm", "--name", self.container,
                "--label", "agent-harness=sandbox", "-w", self.workdir or "/workspace",
                "--security-opt", "no-new-privileges"]
        if not self.network:
            argv += ["--network", "none"]
        if self.memory:
            argv += ["--memory", str(self.memory)]
        if self.cpus:
            argv += ["--cpus", str(self.cpus)]
        if self.pids:
            argv += ["--pids-limit", str(self.pids)]
        if self.runtime:
            argv += ["--runtime", self.runtime]
        user = self.user
        if self.mount is not None:
            mode = ":ro" if self.mount_read_only else ""
            argv += ["-v", f"{self.mount}:{self.workdir}{mode}"]
            # Otherwise everything written lands on the host owned by root.
            if user is None and hasattr(os, "getuid") and self.cli == "docker":
                user = f"{os.getuid()}:{os.getgid()}"
        if user:
            argv += ["--user", user, "-e", "HOME=/tmp"]
        for key, value in self.env.items():
            argv += ["-e", f"{key}={value}"]
        idle = ["sleep", str(int(self.ttl))] if self.ttl else ["tail", "-f", "/dev/null"]
        return [*argv, *self.args, self.image, *idle]

    async def _cli(self, argv: list[str], *, timeout: float, what: str) -> str:
        try:
            code, out, err = await self._run(argv, None, timeout)
        except ProcessTimeout:
            raise ToolError(f"{self.cli} {what} did not finish in {timeout:.0f}s",
                            tool="sandbox") from None
        if code != 0:
            raise ToolError(f"{self.cli} {what} failed: "
                            f"{err.decode(errors='replace').strip()[:400]}",
                            tool="sandbox")
        return out.decode(errors="replace").strip()

    async def _start(self) -> None:
        if self._run is run_process and shutil.which(self.cli) is None:
            raise ConfigurationError(
                f"{self.cli} is not on PATH — install it, or pick another sandbox")
        if self.attached:
            state = await self._cli(
                [self.cli, "inspect", "-f", "{{.State.Running}}", self.container],
                timeout=30, what="inspect")
            if state != "true":
                raise ToolError(f"container {self.container} is not running",
                                tool="sandbox")
        else:
            if self.mount is not None:
                self.mount.mkdir(parents=True, exist_ok=True)
            await self._cli(self.run_argv(), timeout=self.start_timeout, what="run")
            if self.user and self.mount is None:
                # Docker made the working directory as root; hand it over.
                await self._cli(
                    [self.cli, "exec", "-u", "0", self.container, "sh", "-c",
                     f"mkdir -p {q(self.workdir or '/workspace')} && "
                     f"chown {q(self.user)} {q(self.workdir or '/workspace')}"],
                    timeout=30, what="exec")
            # A crash, a Ctrl-C, a forgotten aclose(): the container still goes.
            atexit.register(self._remove_now)
        self.id = self.container

    def _remove_now(self) -> None:
        if self.keep or self._stopped:
            return
        try:
            subprocess.run([self.cli, "rm", "-f", self.container], timeout=20,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
        except (OSError, subprocess.SubprocessError):
            pass

    async def _stop(self) -> None:
        if self.attached:
            return
        atexit.unregister(self._remove_now)
        try:
            await self._run([self.cli, "rm", "-f", self.container], None, 30)
        except ProcessTimeout:
            pass
