"""E2B: a Firecracker micro-VM per sandbox. Needs `e2b` and an API key."""

from __future__ import annotations

import os
from typing import Any, ClassVar

from ..errors import ConfigurationError
from .base import ExecResult, Sandbox

__all__ = ["E2BSandbox"]


class E2BSandbox(Sandbox):
    """An E2B sandbox.

        sandbox("e2b")                              # E2B_API_KEY from the environment
        sandbox("e2b", template="my-template", ttl=1800)
        sandbox("e2b", sandbox_id="i1a2b3...")      # one that is already running

    Args:
        template: the sandbox template. None is E2B's base image.
        api_key: defaults to `E2B_API_KEY`.
        ttl: seconds the sandbox lives. E2B ends it after that whatever happens
            here, so one we lose track of does not bill for ever.
        network: False cuts the sandbox off from the internet.
        sandbox_id: connect to a running sandbox instead of creating one. It is
            left running afterwards.
        metadata: labels shown in the E2B dashboard.
    """

    name: ClassVar[str] = "e2b"
    url_option: ClassVar[str] = "template"
    driver_hint: ClassVar[str] = "pip install e2b"

    def __init__(self, template: str | None = None, *, api_key: str | None = None,
                 ttl: int = 3600, network: bool = True,
                 sandbox_id: str | None = None,
                 metadata: dict[str, str] | None = None, client: Any = None,
                 workdir: str | None = "/home/user/workspace", **kw: Any) -> None:
        kw.setdefault("keep", sandbox_id is not None)
        super().__init__(workdir=workdir, **kw)
        self.template = template or None
        self.api_key = api_key
        self.ttl = ttl
        self.network = network
        self.sandbox_id = sandbox_id
        self.metadata = metadata
        self._box = client

    async def _start(self) -> None:
        if self._box is None:
            try:
                from e2b import AsyncSandbox
            except ImportError as exc:
                raise ConfigurationError(
                    "the e2b sandbox needs its SDK — " + self.driver_hint) from exc
            key = self.api_key or os.environ.get("E2B_API_KEY")
            if not key:
                raise ConfigurationError(
                    "the e2b sandbox needs an API key — set E2B_API_KEY, or pass "
                    "api_key=")
            if self.sandbox_id:
                self._box = await AsyncSandbox.connect(self.sandbox_id, api_key=key)
            else:
                # Only what was asked for is passed, so an older SDK that does
                # not know a newer option is not handed it.
                options: dict[str, Any] = {"timeout": int(self.ttl), "api_key": key}
                if self.template:
                    options["template"] = self.template
                if self.metadata:
                    options["metadata"] = self.metadata
                if self.env:
                    options["envs"] = self.env
                if not self.network:
                    options["allow_internet_access"] = False
                self._box = await AsyncSandbox.create(**options)
        self.id = str(getattr(self._box, "sandbox_id", "") or self.sandbox_id or "")

    async def _exec(self, command: str, *, cwd: str | None, env: dict[str, str],
                    timeout: float) -> ExecResult:
        options: dict[str, Any] = {"timeout": timeout}
        if cwd:
            options["cwd"] = cwd
        if env:
            options["envs"] = env
        try:
            done = await self._box.commands.run(command, **options)
        except Exception as exc:
            kind = type(exc).__name__
            # A non-zero exit is an answer, not a failure of the sandbox.
            if hasattr(exc, "exit_code"):
                return ExecResult(int(exc.exit_code), getattr(exc, "stdout", "") or "",
                                  getattr(exc, "stderr", "") or str(exc))
            if "Timeout" in kind:
                return ExecResult(124, "", f"timed out after {timeout:.0f}s",
                                  timed_out=True)
            raise
        return ExecResult(int(done.exit_code or 0), done.stdout or "",
                          done.stderr or "")

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        await self.start()
        if max_bytes is not None or not await self._box.files.exists(path):
            # The shell form knows how to say "missing" and "too big".
            return await super().read_bytes(path, max_bytes=max_bytes)
        return bytes(await self._box.files.read(path, format="bytes"))

    async def write_bytes(self, path: str, data: bytes, *, append: bool = False) -> None:
        if append:
            return await super().write_bytes(path, data, append=True)
        await self.start()
        await self._box.files.write(path, data)

    async def _stop(self) -> None:
        await self._box.kill()
