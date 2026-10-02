"""Modal: a sandbox container on Modal. Needs `modal` and a Modal token."""

from __future__ import annotations

from typing import Any, ClassVar

from ..errors import ConfigurationError
from .base import ExecResult, Sandbox

__all__ = ["ModalSandbox"]


class ModalSandbox(Sandbox):
    """A Modal sandbox.

        sandbox("modal")                                    # debian-slim
        sandbox("modal", image="python:3.12-slim", network=False, cpu=2)
        sandbox("modal", sandbox_id="sb-...")               # one already running

    Modal authenticates with its own token — `modal token new`, or
    `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

    Args:
        image: a registry image. None is Modal's debian-slim.
        app: the Modal app the sandbox is created under; made if missing.
        ttl: seconds the sandbox lives before Modal ends it.
        network: False blocks all network access.
        cpu, memory: cores, and MiB.
        sandbox_id: connect to an existing sandbox, and leave it afterwards.
    """

    name: ClassVar[str] = "modal"
    url_option: ClassVar[str] = "image"
    driver_hint: ClassVar[str] = "pip install modal"

    def __init__(self, image: str | None = None, *, app: str = "agent-harness",
                 ttl: int = 3600, network: bool = True, cpu: float | None = None,
                 memory: int | None = None, sandbox_id: str | None = None,
                 box: Any = None, workdir: str | None = "/workspace",
                 **kw: Any) -> None:
        kw.setdefault("keep", sandbox_id is not None)
        super().__init__(workdir=workdir, **kw)
        self.image = image or None
        self.app = app
        self.ttl = ttl
        self.network = network
        self.cpu, self.memory = cpu, memory
        self.sandbox_id = sandbox_id
        self._box = box

    async def _start(self) -> None:
        if self._box is None:
            try:
                import modal
            except ImportError as exc:
                raise ConfigurationError(
                    "the modal sandbox needs its SDK — " + self.driver_hint) from exc
            if self.sandbox_id:
                self._box = await modal.Sandbox.from_id.aio(self.sandbox_id)
            else:
                app = await modal.App.lookup.aio(self.app, create_if_missing=True)
                image = (modal.Image.from_registry(self.image) if self.image
                         else modal.Image.debian_slim())
                options: dict[str, Any] = {"app": app, "image": image,
                                           "timeout": int(self.ttl)}
                if not self.network:
                    options["block_network"] = True
                if self.cpu is not None:
                    options["cpu"] = self.cpu
                if self.memory is not None:
                    options["memory"] = self.memory
                self._box = await modal.Sandbox.create.aio(**options)
        self.id = str(getattr(self._box, "object_id", "") or self.sandbox_id or "")

    async def _exec(self, command: str, *, cwd: str | None, env: dict[str, str],
                    timeout: float) -> ExecResult:
        # Directory and environment go in the script: that is the one form
        # every version of the SDK's exec agrees on.
        process = await self._box.exec.aio(
            "sh", "-c", self.script(command, cwd=cwd, env=env),
            timeout=max(1, int(timeout)))
        stdout = await process.stdout.read.aio()
        stderr = await process.stderr.read.aio()
        code = await process.wait.aio()
        return ExecResult(int(code or 0), stdout or "", stderr or "")

    async def _stop(self) -> None:
        await self._box.terminate.aio()
