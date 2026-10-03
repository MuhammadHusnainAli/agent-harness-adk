"""The A2A wire format: what goes over the line, and how it maps to a run.

This is the JSON-RPC binding of A2A 0.3. Everything on the wire is a plain
mapping, built and read here, so the server and the client agree by sharing
one set of functions rather than two readings of the specification.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any

from ..errors import HarnessError
from ..types import Artifact, MediaBlock, RunResult, attach

__all__ = ["A2AError", "PROTOCOL_VERSION", "TERMINAL", "CARD_PATH", "LEGACY_CARD_PATH"]

PROTOCOL_VERSION = "0.3.0"
CARD_PATH = "/.well-known/agent-card.json"
#: Where the card lived before 0.3. Still served, still looked for.
LEGACY_CARD_PATH = "/.well-known/agent.json"

#: A task in one of these states is over: it takes no more messages.
TERMINAL = frozenset({"completed", "canceled", "failed", "rejected"})
#: The answer of a run, among the artefacts of its task.
ANSWER = "response"

# JSON-RPC's own errors, and the ones A2A adds.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
SERVER_BUSY = -32000
TASK_NOT_FOUND = -32001
TASK_NOT_CANCELABLE = -32002
PUSH_NOT_SUPPORTED = -32003
UNSUPPORTED_OPERATION = -32004
CONTENT_TYPE_NOT_SUPPORTED = -32005
INVALID_AGENT_RESPONSE = -32006
NO_EXTENDED_CARD = -32007

#: The largest file an artefact carries inline. Larger ones are named, not sent.
MAX_INLINE_BYTES = 1_000_000


class A2AError(HarnessError):
    """An A2A request that was refused, or a reply that was an error.

    `code` is the JSON-RPC error code; `status` the HTTP status that goes with
    it when the refusal is about the request rather than the task.
    """

    def __init__(self, message: str, *, code: int = INTERNAL_ERROR, data: Any = None,
                 status: int = 200, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data
        self.status = status
        self.retry_after = retry_after

    def body(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.data is not None:
            error["data"] = self.data
        return error


def now() -> str:
    """A timestamp as A2A writes one: ISO 8601, UTC."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def new_uuid() -> str:
    return str(uuid.uuid4())


# ----------------------------------------------------------------------
# parts and messages
# ----------------------------------------------------------------------
def text_part(text: str) -> dict[str, Any]:
    return {"kind": "text", "text": text}


def message(role: str, parts: list[dict[str, Any]] | str, *, task_id: str = "",
            context_id: str = "", message_id: str = "",
            metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": "message", "role": role, "messageId": message_id or new_uuid(),
        "parts": [text_part(parts)] if isinstance(parts, str) else parts,
    }
    if task_id:
        out["taskId"] = task_id
    if context_id:
        out["contextId"] = context_id
    if metadata:
        out["metadata"] = metadata
    return out


def file_part(block: MediaBlock) -> dict[str, Any]:
    """An attachment as a file part: its bytes, or where to fetch it."""
    file: dict[str, Any] = {"mimeType": block.media_type}
    if block.name:
        file["name"] = block.name
    if block.inline:
        file["bytes"] = block.load()
    else:
        file["uri"] = block.url
    return {"kind": "file", "file": file}


def parts_of(text: str, attachments: Any = ()) -> list[dict[str, Any]]:
    parts = [file_part(attach(item)) for item in attachments or ()]
    if text or not parts:
        parts.append(text_part(text))
    return parts


def read_parts(parts: Any) -> tuple[str, list[MediaBlock]]:
    """What a message says, and what came with it.

    Text parts are the text. A data part is structured input: it is written out
    as JSON, where the model can read it. A file part becomes an attachment.
    """
    if not isinstance(parts, list) or not parts:
        raise A2AError("a message needs at least one part", code=INVALID_PARAMS)
    texts: list[str] = []
    files: list[MediaBlock] = []
    for part in parts:
        if not isinstance(part, dict):
            raise A2AError("a part must be an object", code=INVALID_PARAMS)
        kind = part.get("kind") or part.get("type")
        if kind == "text":
            texts.append(str(part.get("text") or ""))
        elif kind == "data":
            texts.append(json.dumps(part.get("data"), ensure_ascii=False, default=str))
        elif kind == "file":
            file = part.get("file") or {}
            media_type = file.get("mimeType") or "application/octet-stream"
            name = file.get("name") or ""
            try:
                if file.get("bytes"):
                    files.append(attach(base64.b64decode(file["bytes"], validate=True),
                                        media_type, name=name))
                elif file.get("uri"):
                    uri = str(file["uri"])
                    if not uri.startswith(("http://", "https://", "gs://")):
                        raise ValueError("only http(s) and gs URIs are fetched")
                    files.append(attach(uri, file.get("mimeType"),
                                        **({"name": name} if name else {})))
                else:
                    raise ValueError("a file part needs `bytes` or `uri`")
            except ValueError as exc:
                raise A2AError(f"a file part could not be read: {exc}",
                               code=INVALID_PARAMS) from None
        else:
            raise A2AError(f"a part of kind {kind!r} is not supported",
                           code=CONTENT_TYPE_NOT_SUPPORTED)
    return "\n\n".join(t for t in texts if t), files


def text_of(parts: Any) -> str:
    """The text of some parts, for whoever only wants to read it."""
    out: list[str] = []
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("kind") or part.get("type")
        if kind == "text" and part.get("text"):
            out.append(str(part["text"]))
        elif kind == "data":
            out.append(json.dumps(part.get("data"), ensure_ascii=False, default=str))
    return "\n".join(out)


# ----------------------------------------------------------------------
# tasks
# ----------------------------------------------------------------------
def status(state: str, text: str = "", *, task_id: str = "",
           context_id: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"state": state, "timestamp": now()}
    if text:
        out["message"] = message("agent", text, task_id=task_id, context_id=context_id)
    return out


def status_event(task: dict[str, Any], *, final: bool = False) -> dict[str, Any]:
    return {"kind": "status-update", "taskId": task["id"],
            "contextId": task["contextId"], "status": task["status"], "final": final}


def artifact_event(task: dict[str, Any], artifact: dict[str, Any], *,
                   append: bool = False, last: bool = False) -> dict[str, Any]:
    return {"kind": "artifact-update", "taskId": task["id"],
            "contextId": task["contextId"], "artifact": artifact,
            "append": append, "lastChunk": last}


def trimmed(task: dict[str, Any], history_length: Any = None) -> dict[str, Any]:
    """The task as it is sent: with as much history as was asked for."""
    if not isinstance(history_length, int) or isinstance(history_length, bool):
        return task
    history = task.get("history") or []
    return {**task, "history": history[-history_length:] if history_length > 0 else []}


def artifacts_of(result: RunResult, *, answer_id: str = "") -> list[dict[str, Any]]:
    """What a run produced, as A2A artefacts: the answer first, then its files."""
    parts: list[dict[str, Any]] = []
    if result.output:
        parts.append(text_part(result.output))
    if result.data is not None:
        data = (result.data.model_dump(mode="json")
                if hasattr(result.data, "model_dump") else result.data)
        parts.append({"kind": "data",
                      "data": data if isinstance(data, dict) else {"value": data}})
    out: list[dict[str, Any]] = []
    if parts:
        out.append({"artifactId": answer_id or new_uuid(), "name": ANSWER,
                    "parts": parts})
    for produced in result.artifacts:
        out.append(_artifact(produced))
    return out


def _artifact(produced: Artifact) -> dict[str, Any]:
    content = produced.content or ""
    kind = produced.media_type or "text/plain"
    entry: dict[str, Any] = {"artifactId": new_uuid(), "name": produced.name}
    if len(content.encode()) > MAX_INLINE_BYTES:
        entry["parts"] = [text_part(f"[{produced.name}: too large to send inline]")]
        entry["metadata"] = {"truncated": True}
    elif kind.startswith("text/") and not produced.path:
        entry["parts"] = [text_part(content)]
    else:
        entry["parts"] = [{"kind": "file", "file": {
            "name": produced.name, "mimeType": kind,
            "bytes": base64.b64encode(content.encode()).decode()}}]
    return entry


def answer_of(task: dict[str, Any]) -> str:
    """What a task came to, as text — wherever the server put it."""
    artifacts = task.get("artifacts") or []
    named = [a for a in artifacts if a.get("name") == ANSWER] or artifacts
    for artifact in named:
        text = text_of(artifact.get("parts"))
        if text:
            return text
    said = text_of(((task.get("status") or {}).get("message") or {}).get("parts"))
    if said:
        return said
    for entry in reversed(task.get("history") or []):
        if entry.get("role") == "agent":
            return text_of(entry.get("parts"))
    return ""
