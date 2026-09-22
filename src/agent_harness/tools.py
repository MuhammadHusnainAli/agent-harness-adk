"""Tools: plain Python functions the model is allowed to call.

    @tool
    async def search(query: str, limit: int = 5) -> list[str]:
        '''Search the corpus.

        Args:
            query: what to look for
            limit: how many hits to return
        '''
        ...

The decorator reads the signature and the docstring and produces the JSON Schema
the provider needs. Arguments coming back from the model are validated before
your function ever sees them.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

from .errors import ToolError, ToolNotFound
from .providers.base import ToolSchema
from .types import ToolOutcome

__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "tool",
    "Permission",
    "render_result",
]

Permission = Literal["allow", "ask", "deny"]

# Only these *names* mean "inject the run context" — `context` is too common a
# domain word to claim. An explicit ToolContext annotation always works.
_CTX_NAMES = {"ctx", "_ctx"}


@dataclass
class ToolContext:
    """Everything a tool may want to know about the run calling it.

    Declare a parameter named `ctx` (or annotate it `ToolContext`) and the
    harness fills it in. The model never sees it.
    """

    agent: str = ""
    run_id: str = ""
    session_id: str = ""
    step: int = 0
    workspace: Any = None
    memory: Any = None
    harness: Any = None
    state: dict[str, Any] = field(default_factory=dict)
    emit: Callable[..., Any] | None = None

    def log(self, message: str, **data: Any) -> None:
        if self.emit:
            self.emit(message, **data)


def _resolve_annotation(annotation: Any, fn: Callable[..., Any]) -> Any:
    """Best-effort resolution of a stringified annotation."""
    if not isinstance(annotation, str):
        return annotation
    try:
        return eval(annotation, getattr(fn, "__globals__", {}), {})  # noqa: S307
    except Exception:
        return Any


def _split_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return (summary, {param: description}) from a Google-style docstring."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    summary: list[str] = []
    params: dict[str, str] = {}
    section = "summary"
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        header = stripped.rstrip(":").lower()
        if header in {"args", "arguments", "params", "parameters"} and stripped.endswith(":"):
            section = "args"
            continue
        if header in {"returns", "raises", "yields", "examples", "example", "notes"}:
            section = "other"
            continue
        if section == "summary":
            summary.append(stripped)
        elif section == "args":
            match = re.match(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", stripped)
            if match:
                current = match.group(1).lstrip("*")
                params[current] = match.group(2).strip()
            elif current and stripped:
                params[current] = f"{params[current]} {stripped}".strip()
    return "\n".join(summary).strip(), params


def render_result(value: Any, *, limit: int = 40_000) -> str:
    """Turn whatever a tool returned into something a model can read."""
    if value is None:
        return "(no output)"
    if isinstance(value, str):
        text = value
    elif isinstance(value, BaseModel):
        text = value.model_dump_json(indent=2)
    elif isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, indent=2, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    else:
        text = str(value)
    if len(text) > limit:
        cut = len(text) - limit
        text = f"{text[:limit]}\n... [truncated {cut} characters]"
    return text


class Tool:
    """A callable exposed to the model, with a validated schema around it."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        permission: Permission = "allow",
        tags: Iterable[str] = (),
        cacheable: bool = False,
        max_output_chars: int = 40_000,
        returns_direct: bool = False,
    ) -> None:
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "tool")
        self.permission: Permission = permission
        self.tags = set(tags)
        self.cacheable = cacheable
        self.max_output_chars = max_output_chars
        self.returns_direct = returns_direct
        self.is_async = inspect.iscoroutinefunction(fn)

        summary, param_docs = _split_docstring(inspect.getdoc(fn))
        self.description = description or summary or f"Call {self.name}."
        self._ctx_param: str | None = None
        self._accepts_kwargs = False
        self._model = self._build_model(param_docs)
        self.parameters = parameters or self._schema_from_model()

    # ---- schema -------------------------------------------------------
    def _build_model(self, param_docs: dict[str, str]) -> type[BaseModel] | None:
        sig = inspect.signature(self.fn)
        try:
            hints = get_type_hints(self.fn)
        except Exception:
            # Unresolvable forward refs (a model defined inside a function, say).
            # Resolve what we can and fall back to a loose type for the rest, so a
            # tool still works instead of failing at import time.
            raw = getattr(self.fn, "__annotations__", {})
            hints = {k: _resolve_annotation(v, self.fn) for k, v in raw.items()}

        fields: dict[str, Any] = {}
        from pydantic import Field as PField

        for pname, param in sig.parameters.items():
            if param.kind is param.VAR_KEYWORD:
                self._accepts_kwargs = True
                continue
            if param.kind is param.VAR_POSITIONAL:
                continue
            annotation = hints.get(pname, param.annotation)
            is_ctx = (pname in _CTX_NAMES or annotation is ToolContext
                      or ToolContext in getattr(annotation, "__args__", ()))
            if is_ctx:
                self._ctx_param = pname
                continue
            if annotation is inspect.Parameter.empty:
                annotation = str
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[pname] = (annotation, PField(default, description=param_docs.get(pname)))

        if not fields:
            return None
        return create_model(f"{self.name.title().replace('_', '')}Args", **fields)

    def _schema_from_model(self) -> dict[str, Any]:
        if self._model is None:
            return {"type": "object", "properties": {}}
        schema = self._model.model_json_schema()
        schema.pop("title", None)
        for prop in (schema.get("properties") or {}).values():
            prop.pop("title", None)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        return schema

    def to_schema(self) -> ToolSchema:
        return ToolSchema(name=self.name, description=self.description,
                          parameters=self.parameters)

    # ---- execution ----------------------------------------------------
    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Coerce the model's arguments, or pass them through for **kwargs tools."""
        supplied = dict(args or {})
        if self._model is None:
            # Tools declared with an explicit schema (MCP, for one) take **kwargs;
            # the remote side owns validation, so hand the arguments straight over.
            return supplied if self._accepts_kwargs else {}
        try:
            validated = self._model(**supplied)
            # Read the fields off the instance rather than model_dump()ing it, so a
            # nested pydantic argument reaches the function as a model, not a dict.
            coerced = {name: getattr(validated, name)
                       for name in type(validated).model_fields}
            if self._accepts_kwargs:
                extras = {k: v for k, v in supplied.items() if k not in coerced}
                coerced.update(extras)
            return coerced
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            raise ToolError(f"invalid arguments for {self.name} — {problems}",
                            tool=self.name) from exc

    async def invoke(self, args: dict[str, Any] | None = None,
                     ctx: ToolContext | None = None) -> Any:
        """Validate, call, and await if needed. Errors raise ToolError."""
        kwargs = self.validate(args or {})
        if self._ctx_param:
            kwargs[self._ctx_param] = ctx or ToolContext()
        try:
            result = self.fn(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except ToolError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ToolError(f"{self.name} failed: {exc}", tool=self.name) from exc

    async def run(self, call_id: str, args: dict[str, Any] | None = None,
                  ctx: ToolContext | None = None) -> ToolOutcome:
        """Invoke and package the result for the transcript. Never raises."""
        started = time.perf_counter()
        try:
            value = await self.invoke(args, ctx)
            return ToolOutcome(
                call_id=call_id, name=self.name, value=value,
                content=render_result(value, limit=self.max_output_chars),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ToolOutcome(
                call_id=call_id, name=self.name, content=f"Error: {exc}", is_error=True,
                duration_ms=(time.perf_counter() - started) * 1000,
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calling the decorated object still calls your function."""
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<Tool {self.name}>"


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    permission: Permission = "allow",
    tags: Iterable[str] = (),
    cacheable: bool = False,
    max_output_chars: int = 40_000,
    returns_direct: bool = False,
) -> Any:
    """Turn a function into a Tool. Works bare (`@tool`) or called (`@tool(...)`)."""

    def wrap(func: Callable[..., Any]) -> Tool:
        return Tool(func, name=name, description=description, parameters=parameters,
                    permission=permission, tags=tags, cacheable=cacheable,
                    max_output_chars=max_output_chars, returns_direct=returns_direct)

    return wrap(fn) if callable(fn) else wrap


class ToolRegistry:
    """The set of tools an agent may reach for. Glob-filterable for allowlists."""

    def __init__(self, tools: Iterable[Tool | Callable[..., Any]] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self.extend(tools)

    def add(self, item: Tool | Callable[..., Any], *, replace: bool = True) -> Tool:
        entry = item if isinstance(item, Tool) else Tool(item)
        if not replace and entry.name in self._tools:
            raise ToolError(f"tool {entry.name!r} is already registered", tool=entry.name)
        self._tools[entry.name] = entry
        return entry

    def extend(self, items: Iterable[Tool | Callable[..., Any]]) -> ToolRegistry:
        for item in items or ():
            self.add(item)
        return self

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(sorted(self._tools)) or "none"
            raise ToolNotFound(
                f"no tool named {name!r}; available: {known}", tool=name
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def select(self, patterns: Iterable[str] | None) -> ToolRegistry:
        """A sub-registry matching glob patterns — this is the tool allowlist."""
        if patterns is None:
            return self
        pats = list(patterns)
        if not pats:
            return ToolRegistry()
        picked = [t for t in self._tools.values()
                  if any(fnmatch(t.name, p) or p in t.tags for p in pats)]
        return ToolRegistry(picked)

    def schemas(self) -> list[ToolSchema]:
        return [t.to_schema() for t in self._tools.values()]

    def merge(self, other: ToolRegistry | Iterable[Tool]) -> ToolRegistry:
        self.extend(list(other))
        return self
