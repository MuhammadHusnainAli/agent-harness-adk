"""Sandboxes reached by wrapping a command: your own, SSH, or a Kubernetes pod.

`CommandSandbox` is the escape hatch. Anything that can run `sh -c <script>`
somewhere — nsjail, bubblewrap, firejail, `lxc exec`, `nerdctl exec`, a wrapper
script of your own — becomes a sandbox by saying what to put in front:

    sandbox("command", wrap=["bwrap", "--ro-bind", "/", "/", "--bind", "/tmp/w",
                             "/work", "--unshare-all", "sh", "-c"],
            workdir="/work", isolated=True)
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any, ClassVar

from ..errors import ConfigurationError, ToolError
from ..types import new_id
from .base import ProcessSandbox, ProcessTimeout, q

__all__ = ["CommandSandbox", "SSHSandbox", "KubernetesSandbox"]


class CommandSandbox(ProcessSandbox):
    """Runs every script as `[*wrap, script]`. With the default wrap that is a
    plain local shell — contained by nothing, and it says so."""

    name: ClassVar[str] = "command"
    isolated: ClassVar[bool] = False

    def __init__(self, wrap: Sequence[str] = ("sh", "-c"), *,
                 workdir: str | None = None, isolated: bool = False,
                 **kw: Any) -> None:
        if not wrap:
            raise ConfigurationError("a command sandbox needs a `wrap` — the "
                                     "command a shell script is appended to")
        if workdir is None:
            raise ConfigurationError(
                "a command sandbox needs a `workdir` — the directory, as the "
                "wrapped shell sees it, that the agent works in")
        super().__init__(workdir=workdir, **kw)
        self.wrap = [str(part) for part in wrap]
        # What the wrap contains is yours to vouch for.
        self.isolated = isolated  # type: ignore[misc]

    def _wrap(self, script: str) -> list[str]:
        return [*self.wrap, script]

    async def _start(self) -> None:
        # The same place every time, so a conversation finds its files again.
        self.id = self.workdir or ""


class SSHSandbox(ProcessSandbox):
    """A machine over SSH — a VM you already have, a build box, a lab host.

        sandbox("ssh", host="agent@build-7", key="~/.ssh/agent_ed25519")

    Authentication is your SSH configuration's business: keys, agents and
    `~/.ssh/config` all apply, and a password prompt is refused rather than
    waited on. The machine is a real one, so this does not claim to be isolated
    unless you say it is (`isolated=True`, for a throwaway VM).
    """

    name: ClassVar[str] = "ssh"
    isolated: ClassVar[bool] = False
    url_option: ClassVar[str] = "host"

    def __init__(self, host: str = "", *, port: int | None = None,
                 key: str | None = None, options: Sequence[str] = (),
                 isolated: bool = False, ssh: str = "ssh", **kw: Any) -> None:
        if not host:
            raise ConfigurationError("an ssh sandbox needs a `host` — user@machine")
        super().__init__(**kw)
        self.host = host
        self.isolated = isolated  # type: ignore[misc]
        self._base = [ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
        if port:
            self._base += ["-p", str(port)]
        if key:
            self._base += ["-i", os.path.expanduser(key)]
        for option in options:
            self._base += ["-o", option]

    def _wrap(self, script: str) -> list[str]:
        # ssh joins its arguments with spaces and hands them to the remote
        # shell, so the script has to arrive as one quoted word.
        return [*self._base, self.host, f"sh -lc {q(script)}"]

    async def _start(self) -> None:
        code, _, err = await self._run(self._wrap("true"), None, self.start_timeout)
        if code != 0:
            raise ToolError(f"could not reach {self.host} over ssh: "
                            f"{err.decode(errors='replace').strip()[:300]}",
                            tool="sandbox")
        self.id = self.host

    async def _stop(self) -> None:
        """The machine is not ours to switch off."""


class KubernetesSandbox(ProcessSandbox):
    """A pod, driven with `kubectl`.

        sandbox("kubernetes", image="python:3.12-slim", namespace="agents")
        sandbox("kubernetes", pod="agent-box-7", namespace="agents")   # one you run

    A pod it creates has no service-account token mounted, a CPU and memory
    limit, and a lifetime (`ttl`) after which it ends on its own. Whether it can
    reach the network is the cluster's NetworkPolicy to decide, not ours.
    """

    name: ClassVar[str] = "kubernetes"
    url_option: ClassVar[str] = "image"

    def __init__(self, image: str = "python:3.12-slim", *, pod: str | None = None,
                 namespace: str | None = None, container: str | None = None,
                 context: str | None = None, cpu: str = "2", memory: str = "2Gi",
                 ttl: int | None = 14_400, labels: dict[str, str] | None = None,
                 service_account: str | None = None, kubectl: str = "kubectl",
                 workdir: str | None = "/workspace", **kw: Any) -> None:
        if pod is not None:
            kw.setdefault("id", pod)
        super().__init__(workdir=workdir, **kw)
        self.image = image
        self.pod: str | None = self.attach_id
        self.container = container
        self.cpu, self.memory, self.ttl = cpu, memory, ttl
        self.labels = {"app.kubernetes.io/managed-by": "agent-harness",
                       **(labels or {})}
        self.service_account = service_account
        self._base = [kubectl]
        if context:
            self._base += ["--context", context]
        if namespace:
            self._base += ["-n", namespace]

    def _wrap(self, script: str) -> list[str]:
        argv = [*self._base, "exec", "-i", self.pod or ""]
        if self.container:
            argv += ["-c", self.container]
        return [*argv, "--", "sh", "-c", script]

    def manifest(self) -> dict[str, Any]:
        """The pod it creates, as `kubectl run --overrides` takes it."""
        name = self.pod or ""
        spec: dict[str, Any] = {
            "automountServiceAccountToken": False,
            "restartPolicy": "Never",
            "containers": [{
                "name": name, "image": self.image,
                "command": ["sleep", str(self.ttl)] if self.ttl
                else ["tail", "-f", "/dev/null"],
                "workingDir": "/",
                "resources": {"limits": {"cpu": self.cpu, "memory": self.memory}},
                "securityContext": {"allowPrivilegeEscalation": False},
            }],
        }
        if self.ttl:
            spec["activeDeadlineSeconds"] = int(self.ttl)
        if self.service_account:
            spec["serviceAccountName"] = self.service_account
        return {"metadata": {"labels": self.labels}, "spec": spec}

    async def _kubectl(self, *args: str, timeout: float) -> str:
        try:
            code, out, err = await self._run([*self._base, *args], None, timeout)
        except ProcessTimeout:
            raise ToolError(f"kubectl {args[0]} did not finish in {timeout:.0f}s",
                            tool="sandbox") from None
        if code != 0:
            raise ToolError(f"kubectl {args[0]} failed: "
                            f"{err.decode(errors='replace').strip()[:400]}",
                            tool="sandbox")
        return out.decode(errors="replace")

    async def _start(self) -> None:
        if self.attach_id:
            self.pod = self.attach_id
        else:
            self.pod = f"agent-harness-{new_id()}"
            await self._kubectl(
                "run", self.pod, f"--image={self.image}", "--restart=Never",
                f"--overrides={json.dumps(self.manifest())}", timeout=60)
        await self._kubectl("wait", "--for=condition=Ready", f"pod/{self.pod}",
                            f"--timeout={int(self.start_timeout)}s",
                            timeout=self.start_timeout + 10)
        self.id = self.pod or ""

    async def _stop(self) -> None:
        if self.pod:
            await self._kubectl("delete", "pod", self.pod, "--grace-period=0",
                                "--force", "--wait=false", "--ignore-not-found",
                                timeout=60)
