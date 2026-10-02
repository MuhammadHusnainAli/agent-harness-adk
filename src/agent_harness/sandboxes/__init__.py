"""Sandboxes: somewhere other than this machine for an agent to work.

    from agent_harness import Agent, sandbox

    Agent("colleague", mode="cowork", workspace=sandbox("docker"))
    Agent("colleague", mode="cowork", workspace=sandbox("e2b", template="base"))
    Agent("colleague", mode="cowork", workspace="docker://node:22")

A sandbox is a workspace, so everything that works on a local folder — the file
tools, the shell, `run_python`, `parse_document`, cowork handing back the files
it wrote — works inside one, unchanged.

| name | what it is | needs |
|---|---|---|
| `docker` | a long-lived container | the `docker` CLI |
| `podman` | the same, with Podman | the `podman` CLI |
| `kubernetes` | a pod, created or attached to | `kubectl` |
| `ssh` | a machine you can already reach | `ssh` |
| `e2b` | a Firecracker micro-VM | `pip install e2b`, `E2B_API_KEY` |
| `daytona` | a sandbox from an image or snapshot | `pip install daytona`, `DAYTONA_API_KEY` |
| `modal` | a container on Modal | `pip install modal`, a Modal token |
| `command` | anything you can put in front of `sh -c` | — |

Nothing is imported until you ask for it, and a sandbox that is not ready says
exactly what is missing. Your own is one method — see `Sandbox` — and
`register_sandbox` gives it a name.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
from typing import TYPE_CHECKING, Any

from ..errors import ConfigurationError
from .base import ExecResult, ProcessSandbox, Sandbox, SandboxWorkspace

if TYPE_CHECKING:  # pragma: no cover - import-time cost is the whole point
    from .command import CommandSandbox, KubernetesSandbox, SSHSandbox
    from .daytona import DaytonaSandbox
    from .docker import DockerSandbox
    from .e2b import E2BSandbox
    from .modal import ModalSandbox

__all__ = [
    "Sandbox",
    "ProcessSandbox",
    "SandboxWorkspace",
    "ExecResult",
    "DockerSandbox",
    "CommandSandbox",
    "SSHSandbox",
    "KubernetesSandbox",
    "E2BSandbox",
    "DaytonaSandbox",
    "ModalSandbox",
    "sandbox",
    "sandbox_backend",
    "register_sandbox",
    "available_sandboxes",
    "describe_sandboxes",
    "BACKENDS",
]

_HERE = "agent_harness.sandboxes"

#: name -> (module, class, constructor defaults). Imported on first use.
BACKENDS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "docker": (f"{_HERE}.docker", "DockerSandbox", {}),
    "podman": (f"{_HERE}.docker", "DockerSandbox", {"cli": "podman"}),
    "kubernetes": (f"{_HERE}.command", "KubernetesSandbox", {}),
    "ssh": (f"{_HERE}.command", "SSHSandbox", {}),
    "command": (f"{_HERE}.command", "CommandSandbox", {}),
    "e2b": (f"{_HERE}.e2b", "E2BSandbox", {}),
    "daytona": (f"{_HERE}.daytona", "DaytonaSandbox", {}),
    "modal": (f"{_HERE}.modal", "ModalSandbox", {}),
}

ALIASES: dict[str, str] = {"k8s": "kubernetes", "kube": "kubernetes",
                           "container": "docker"}

#: What each one needs before it can start: a Python package, a program on
#: PATH, an environment variable. Any one of the listed env vars is enough.
NEEDS: dict[str, dict[str, list[str]]] = {
    "docker": {"binary": ["docker"]},
    "podman": {"binary": ["podman"]},
    "kubernetes": {"binary": ["kubectl"]},
    "ssh": {"binary": ["ssh"]},
    "command": {},
    "e2b": {"package": ["e2b"], "env": ["E2B_API_KEY"]},
    "daytona": {"package": ["daytona"], "env": ["DAYTONA_API_KEY"]},
    "modal": {"package": ["modal"]},
}

SUMMARIES: dict[str, str] = {
    "docker": "a long-lived container on this machine",
    "podman": "a long-lived container, with Podman",
    "kubernetes": "a pod, created or attached to",
    "ssh": "a machine reached over SSH",
    "command": "anything you can put in front of `sh -c`",
    "e2b": "a Firecracker micro-VM on E2B",
    "daytona": "a Daytona sandbox, from an image or a snapshot",
    "modal": "a sandbox container on Modal",
}

#: Options that belong to the workspace wrapped round a sandbox, not to it.
_WORKSPACE_OPTIONS = ("read_only", "allow_shell", "max_file_bytes", "export_dir")

_EXPORTS = {cls: module for module, cls, _ in BACKENDS.values()}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


def register_sandbox(name: str, module: str, cls: str, *,
                     package: list[str] | None = None,
                     binary: list[str] | None = None,
                     env: list[str] | None = None, summary: str = "",
                     **defaults: Any) -> None:
    """Add your own: `register_sandbox("mine", "myapp.sandboxes", "MySandbox")`."""
    key = name.strip().lower()
    BACKENDS[key] = (module, cls, dict(defaults))
    NEEDS[key] = {k: v for k, v in (("package", package), ("binary", binary),
                                    ("env", env)) if v}
    SUMMARIES[key] = summary
    _EXPORTS[cls] = module


def _name(name: str) -> str:
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key not in BACKENDS:
        raise ConfigurationError(
            f"unknown sandbox {name!r}; known: {', '.join(sorted(BACKENDS))}")
    return key


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def missing(name: str) -> list[str]:
    """What stands between this sandbox and starting. Empty if nothing does."""
    needs = NEEDS.get(_name(name), {})
    gaps: list[str] = []
    for package in needs.get("package", []):
        if not _installed(package):
            gaps.append(f"pip install {package}")
    for binary in needs.get("binary", []):
        if shutil.which(binary) is None:
            gaps.append(f"`{binary}` on PATH")
    variables = needs.get("env", [])
    if variables and not any(os.environ.get(v) for v in variables):
        gaps.append("set " + " or ".join(variables))
    return gaps


def available_sandboxes() -> dict[str, bool]:
    """Which sandboxes could start right now, without installing or setting
    anything. A True here is not a promise the service will answer — for that,
    `await sandbox(name).check()`."""
    return {name: not missing(name) for name in sorted(BACKENDS)}


def describe_sandboxes() -> list[dict[str, Any]]:
    """Every sandbox, what it is, and what it still needs."""
    return [{"name": name, "summary": SUMMARIES.get(name, ""),
             "ready": not missing(name), "missing": missing(name),
             "needs": NEEDS.get(name, {})} for name in sorted(BACKENDS)]


def sandbox_backend(name: str, **options: Any) -> Sandbox:
    """The bare `Sandbox`, for when you want to drive one yourself."""
    module, cls, defaults = BACKENDS[_name(name)]
    try:
        return getattr(importlib.import_module(module), cls)(**{**defaults, **options})
    except TypeError as exc:
        raise ConfigurationError(f"{name} sandbox: {exc}") from None


def sandbox(spec: str | dict[str, Any] | Sandbox = "docker",
            **options: Any) -> SandboxWorkspace:
    """A workspace inside a sandbox — what `Agent(workspace=...)` takes.

        sandbox("docker", image="python:3.12-slim")
        sandbox("docker://node:22", network=True)        # the URL names the image
        sandbox("e2b://my-template")
        sandbox("ssh://agent@build-7")
        sandbox({"sandbox": "daytona", "image": "python:3.12-slim"})
        sandbox(MySandbox())

    It starts on first use and is stopped by `harness.aclose()`, or sooner with
    `async with sandbox(...) as workspace:`.
    """
    if isinstance(spec, dict):
        merged = {**spec, **options}
        name = merged.pop("sandbox", None) or merged.pop("name", None)
        if not name:
            raise ConfigurationError(
                "a sandbox given as a mapping needs `sandbox:` — its name")
        return sandbox(str(name), **merged)

    wrapping = {k: options.pop(k) for k in _WORKSPACE_OPTIONS if k in options}
    if isinstance(spec, Sandbox):
        if options:
            raise ConfigurationError(
                f"{', '.join(sorted(options))}: pass sandbox options to the "
                "sandbox itself")
        return SandboxWorkspace(spec, **wrapping)

    name, _, rest = str(spec).partition("://")
    key = _name(name)
    if rest:
        module, cls, _ = BACKENDS[key]
        option = getattr(getattr(importlib.import_module(module), cls),
                         "url_option", "")
        if not option:
            raise ConfigurationError(
                f"the {key} sandbox takes no URL — pass its options by name")
        options.setdefault(option, rest)
    return SandboxWorkspace(sandbox_backend(key, **options), **wrapping)
