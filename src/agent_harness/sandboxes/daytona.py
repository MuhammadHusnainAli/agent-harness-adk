"""Daytona: a sandbox from an image or a snapshot. Needs `daytona` and a key."""

from __future__ import annotations

import os
from typing import Any, ClassVar

from ..errors import ConfigurationError
from .base import ExecResult, Sandbox

__all__ = ["DaytonaSandbox"]


class DaytonaSandbox(Sandbox):
    """A Daytona sandbox.

        sandbox("daytona")                                  # DAYTONA_API_KEY
        sandbox("daytona", image="python:3.12-slim", network=False)
        sandbox("daytona", snapshot="my-snapshot")
        sandbox("daytona", sandbox_id="...")                # one already running

    Args:
        image: build the sandbox from this image.
        snapshot: or start it from this snapshot. Neither is Daytona's default.
        api_key, api_url, target: default to `DAYTONA_API_KEY`,
            `DAYTONA_API_URL` and `DAYTONA_TARGET`.
        ttl: seconds of idleness after which Daytona stops the sandbox itself.
        network: False blocks all network access.
        sandbox_id: connect to an existing sandbox, and leave it afterwards.
        labels: labels on the sandbox.

    Daytona returns a command's stdout and stderr as one stream, so both arrive
    in `stdout`.
    """

    name: ClassVar[str] = "daytona"
    url_option: ClassVar[str] = "image"
    driver_hint: ClassVar[str] = "pip install daytona"

    def __init__(self, image: str | None = None, *, snapshot: str | None = None,
                 api_key: str | None = None, api_url: str | None = None,
                 target: str | None = None, ttl: int = 3600, network: bool = True,
                 sandbox_id: str | None = None,
                 labels: dict[str, str] | None = None, client: Any = None,
                 box: Any = None, **kw: Any) -> None:
        if image and snapshot:
            raise ConfigurationError(
                "a daytona sandbox starts from an image or a snapshot, not both")
        kw.setdefault("keep", sandbox_id is not None)
        super().__init__(**kw)
        self.image = image or None
        # Not `self.snapshot`: that is the method that lists the files.
        self.snapshot_name = snapshot
        self.api_key, self.api_url, self.target = api_key, api_url, target
        self.ttl = ttl
        self.network = network
        self.sandbox_id = sandbox_id
        self.labels = labels
        self._client = client
        self._box = box

    async def _start(self) -> None:
        if self._box is None:
            try:
                import daytona
            except ImportError as exc:
                raise ConfigurationError(
                    "the daytona sandbox needs its SDK — " + self.driver_hint) from exc
            if self._client is None:
                key = self.api_key or os.environ.get("DAYTONA_API_KEY")
                if not key:
                    raise ConfigurationError(
                        "the daytona sandbox needs an API key — set "
                        "DAYTONA_API_KEY, or pass api_key=")
                config: dict[str, Any] = {"api_key": key}
                if self.api_url:
                    config["api_url"] = self.api_url
                if self.target:
                    config["target"] = self.target
                self._client = daytona.AsyncDaytona(daytona.DaytonaConfig(**config))
            if self.sandbox_id:
                self._box = await self._client.get(self.sandbox_id)
            else:
                options: dict[str, Any] = {
                    # Daytona counts idleness in minutes.
                    "auto_stop_interval": max(1, int(self.ttl) // 60),
                }
                if self.env:
                    options["env_vars"] = self.env
                if self.labels:
                    options["labels"] = self.labels
                if not self.network:
                    options["network_block_all"] = True
                if self.image:
                    params = daytona.CreateSandboxFromImageParams(image=self.image,
                                                                  **options)
                else:
                    if self.snapshot_name:
                        options["snapshot"] = self.snapshot_name
                    params = daytona.CreateSandboxFromSnapshotParams(**options)
                self._box = await self._client.create(params,
                                                      timeout=self.start_timeout)
        self.id = str(getattr(self._box, "id", "") or self.sandbox_id or "")

    async def _exec(self, command: str, *, cwd: str | None, env: dict[str, str],
                    timeout: float) -> ExecResult:
        options: dict[str, Any] = {"timeout": max(1, int(timeout))}
        if cwd:
            options["cwd"] = cwd
        if env:
            options["env"] = env
        done = await self._box.process.exec(command, **options)
        return ExecResult(int(done.exit_code or 0), done.result or "", "")

    async def _stop(self) -> None:
        try:
            await self._client.delete(self._box)
        finally:
            close = getattr(self._client, "close", None)
            if close is not None:
                await close()
