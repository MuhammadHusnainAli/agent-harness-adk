"""Trace: whose memory this is.

A trace carries the identity a memory belongs to — a user, a session, a tenant —
and every backend uses it two ways: to *stamp* records as they are written, and
to *scope* reads so one user never sees another's.

    agent = Agent("support", trace=Trace(user_id="alice", session_id="s-42"))

`scope` decides what documents like `user.md` are namespaced by. The default,
`"user"`, is what you almost always want: preferences follow the person across
sessions. `"session"` gives each conversation its own; `"tenant"` shares one
file across a whole organisation; `"global"` is a single shared file.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Trace", "Scope"]

Scope = Literal["global", "tenant", "user", "session"]

# The identity axes, in the order they nest. Narrower to the right.
AXES: tuple[tuple[str, str], ...] = (
    ("tenant_id", "t"),
    ("user_id", "u"),
    ("session_id", "s"),
)

_SCOPE_DEPTH: dict[str, int] = {"global": 0, "tenant": 1, "user": 2, "session": 3}
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class Trace(BaseModel):
    """Who a memory belongs to, and how widely it is shared."""

    model_config = ConfigDict(extra="allow", frozen=False)

    user_id: str | None = None
    session_id: str | None = None
    tenant_id: str | None = None
    agent: str | None = None
    run_id: str | None = None
    scope: Scope = "user"
    tags: dict[str, str] = Field(default_factory=dict)

    # ---- identity ---------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        """True when nothing identifies this trace — a single shared memory."""
        return not any(getattr(self, field) for field, _ in AXES)

    def axes(self, *, depth: int | None = None) -> list[tuple[str, str]]:
        """The set identity axes, narrowed to `scope` (or an explicit depth)."""
        limit = _SCOPE_DEPTH[self.scope] if depth is None else depth
        return [(short, getattr(self, field))
                for index, (field, short) in enumerate(AXES)
                if index < limit and getattr(self, field)]

    @property
    def namespace(self) -> str:
        """The stable key documents are stored under, e.g. `t=acme/u=alice`."""
        return "/".join(f"{short}={value}" for short, value in self.axes())

    @property
    def slug(self) -> str:
        """`namespace`, safe for a filename, an object key or a table value."""
        if not self.namespace:
            return "_shared"
        return "__".join(f"{short}-{_UNSAFE.sub('_', str(value))}"
                         for short, value in self.axes())

    # ---- writing ----------------------------------------------------------
    def stamp(self, record: Any) -> Any:
        """Put this identity on a record before it is stored."""
        for field, _ in AXES:
            value = getattr(self, field)
            if value and getattr(record, field, None) is None:
                setattr(record, field, value)
        if self.agent and not getattr(record, "source", ""):
            record.source = self.agent
        return record

    # ---- reading ----------------------------------------------------------
    def filters(self) -> dict[str, str]:
        """The equality filters a query must apply. Empty means unscoped."""
        return {field: getattr(self, field)
                for field, _ in AXES if getattr(self, field)}

    def matches(self, record: Any) -> bool:
        """Does this record belong to this trace? Used by in-memory backends.

        A record with no value on an axis is shared: it was written before the
        trace existed, or written deliberately without one, and hiding it would
        silently lose memory rather than isolate it.
        """
        for field, value in self.filters().items():
            found = getattr(record, field, None)
            if found is not None and found != value:
                return False
        return True

    # ---- deriving ---------------------------------------------------------
    def child(self, **overrides: Any) -> Trace:
        """A narrower trace — the same user, a new session, say."""
        return self.model_copy(update=overrides)

    def at(self, scope: Scope) -> Trace:
        """The same identity, namespaced more or less widely."""
        return self.model_copy(update={"scope": scope})

    @classmethod
    def of(cls, value: Trace | str | dict[str, Any] | None, **kw: Any) -> Trace:
        """Accept a Trace, a bare user id, a dict, or nothing at all."""
        if isinstance(value, Trace):
            return value.child(**kw) if kw else value
        if isinstance(value, str):
            return cls(user_id=value, **kw)
        if isinstance(value, dict):
            return cls(**{**value, **kw})
        return cls(**kw)

    def __str__(self) -> str:
        return self.namespace or "(shared)"

    def __bool__(self) -> bool:
        return not self.is_empty
