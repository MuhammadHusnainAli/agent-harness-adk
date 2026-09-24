"""A small, safe expression language for policy conditions.

    when: "args.amount > 100 and 'payments' in tool.tags"

Compliance people read these, so they look like Python and behave like it —
but they are parsed with `ast`, checked against a short list of allowed nodes
and compiled once into closures. Nothing is ever passed to `eval`, there is no
attribute access to anything starting with an underscore, and no function can
be called except the handful below.

Two rules make policies safe to write against loose data:

* a missing name or field is ``None``, never an error;
* a comparison that cannot be made (``None > 100``) is ``False``.

A policy that silently raised mid-run would fail open or closed at random;
this makes the result predictable instead.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any

from ..errors import ConfigurationError

__all__ = ["ExpressionError", "compile_expr", "FUNCTIONS"]

Evaluator = Callable[[Mapping[str, Any]], Any]


class ExpressionError(ConfigurationError):
    """A policy condition that does not parse, or uses something not allowed."""


def _matches(value: Any, pattern: str) -> bool:
    return value is not None and re.search(pattern, str(value)) is not None


def _contains(container: Any, item: Any) -> bool:
    try:
        return container is not None and item in container
    except TypeError:
        return False


#: Everything a condition may call.
FUNCTIONS: dict[str, Callable[..., Any]] = {
    "len": lambda v: len(v) if v is not None else 0,
    "lower": lambda v: str(v).lower() if v is not None else "",
    "upper": lambda v: str(v).upper() if v is not None else "",
    "str": lambda v: "" if v is None else str(v),
    "int": int,
    "float": float,
    "abs": abs,
    "min": min,
    "max": max,
    "any": any,
    "all": all,
    "matches": _matches,
    "contains": _contains,
    "startswith": lambda v, p: v is not None and str(v).startswith(p),
    "endswith": lambda v, p: v is not None and str(v).endswith(p),
}

_NAMES: dict[str, Any] = {"true": True, "false": False, "null": None, "none": None,
                          "True": True, "False": False, "None": None}

_COMPARE: dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: _contains(b, a),
    ast.NotIn: lambda a, b: not _contains(b, a),
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
}

_BINARY: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod,
}


def _field(value: Any, key: Any) -> Any:
    """`a.b` and `a['b']`: dicts by key, objects by public attribute, else None."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    if isinstance(key, int) and isinstance(value, Sequence) and not isinstance(value, str):
        return value[key] if -len(value) <= key < len(value) else None
    if isinstance(key, str) and not key.startswith("_"):
        return getattr(value, key, None)
    return None


def _compile(node: ast.AST, source: str) -> Evaluator:
    if isinstance(node, ast.Expression):
        return _compile(node.body, source)

    if isinstance(node, ast.Constant):
        value = node.value
        return lambda ctx: value

    if isinstance(node, ast.Name):
        name = node.id
        if name in _NAMES:
            constant = _NAMES[name]
            return lambda ctx: constant
        return lambda ctx: ctx.get(name)

    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            raise ExpressionError(f"{source!r}: private attribute {node.attr!r}")
        inner, attr = _compile(node.value, source), node.attr
        return lambda ctx: _field(inner(ctx), attr)

    if isinstance(node, ast.Subscript):
        inner, key = _compile(node.value, source), _compile(node.slice, source)
        return lambda ctx: _field(inner(ctx), key(ctx))

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_compile(e, source) for e in node.elts]
        kind = {ast.List: list, ast.Tuple: tuple, ast.Set: frozenset}[type(node)]
        return lambda ctx: kind(i(ctx) for i in items)

    if isinstance(node, ast.BoolOp):
        parts = [_compile(v, source) for v in node.values]
        if isinstance(node.op, ast.And):
            return lambda ctx: all(bool(p(ctx)) for p in parts)
        return lambda ctx: any(bool(p(ctx)) for p in parts)

    if isinstance(node, ast.UnaryOp):
        operand = _compile(node.operand, source)
        if isinstance(node.op, ast.Not):
            return lambda ctx: not operand(ctx)
        if isinstance(node.op, ast.USub):
            return lambda ctx: -operand(ctx)
        raise ExpressionError(f"{source!r}: operator {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        left = _compile(node.left, source)
        ops = []
        for op in node.ops:
            fn = _COMPARE.get(type(op))
            if fn is None:
                raise ExpressionError(f"{source!r}: comparison {type(op).__name__}")
            ops.append(fn)
        rights = [_compile(c, source) for c in node.comparators]

        def compare(ctx: Mapping[str, Any]) -> bool:
            current = left(ctx)
            for fn, right in zip(ops, rights, strict=True):
                other = right(ctx)
                try:
                    if not fn(current, other):
                        return False
                except TypeError:
                    return False
                current = other
            return True
        return compare

    if isinstance(node, ast.BinOp):
        fn = _BINARY.get(type(node.op))
        if fn is None:
            raise ExpressionError(f"{source!r}: operator {type(node.op).__name__}")
        lhs, rhs = _compile(node.left, source), _compile(node.right, source)

        def binary(ctx: Mapping[str, Any]) -> Any:
            try:
                return fn(lhs(ctx), rhs(ctx))
            except (TypeError, ZeroDivisionError):
                return None
        return binary

    if isinstance(node, ast.IfExp):
        test, body, other = (_compile(node.test, source), _compile(node.body, source),
                             _compile(node.orelse, source))
        return lambda ctx: body(ctx) if test(ctx) else other(ctx)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = getattr(node.func, "id", type(node.func).__name__)
            raise ExpressionError(
                f"{source!r}: {name}() is not an allowed function; allowed: "
                f"{', '.join(sorted(FUNCTIONS))}")
        if node.keywords:
            raise ExpressionError(f"{source!r}: keyword arguments are not allowed")
        fn = FUNCTIONS[node.func.id]
        args = [_compile(a, source) for a in node.args]

        def call(ctx: Mapping[str, Any]) -> Any:
            try:
                return fn(*(a(ctx) for a in args))
            except (TypeError, ValueError):
                return None
        return call

    raise ExpressionError(f"{source!r}: {type(node).__name__} is not allowed in a policy")


@lru_cache(maxsize=1024)
def compile_expr(source: str) -> Evaluator:
    """Parse and compile once. The result takes a context mapping."""
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"{source!r}: {exc.msg}") from None
    return _compile(tree, source)
