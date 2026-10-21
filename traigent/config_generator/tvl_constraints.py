"""Translate generated structural constraints into loader-accepted TVL.

The generator stores structural constraints as Python-flavoured code strings
(``require(...)``, ``implies(...)`` or a ``lambda config: ...``).  The TVL
loader (``traigent.tvl.load_tvl_spec``) only accepts ``expr`` or ``when`` +
``then`` entries, so emitting ``code`` produced specs the SDK could not load
(#2647).  This module converts the known shapes and leaves everything else
out, so an unknown shape is never emitted as unloadable text.
"""

from __future__ import annotations

import ast
import keyword
import math
import textwrap
import warnings
from typing import Any

from traigent.config_generator.types import StructuralConstraintSpec

# Names the loader's expression context always binds; a tvar with one of these
# names cannot be referenced bare, so such constraints are skipped.
_RESERVED_NAMES = frozenset(
    {"params", "metrics", "math", "len", "min", "max", "sum", "abs", "any", "all"}
)

_ATOM_METHODS: dict[str, str] = {
    "gte": ">=",
    "lte": "<=",
    "gt": ">",
    "lt": "<",
    "equals": "==",
    "not_equals": "!=",
    "is_in": "in",
    "not_in": "not in",
}

_COMPARE_OPS: dict[type[ast.cmpop], str] = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}


class _UnknownShape(Exception):
    """Raised internally when code is outside the known template grammar."""


def _is_tvar_name(name: str) -> bool:
    return (
        name.isidentifier()
        and not keyword.iskeyword(name)
        and not name.startswith("__")
        and name not in _RESERVED_NAMES
    )


def _scalar(value: object) -> str:
    """Render one scalar, refusing values the grammar cannot express."""
    if isinstance(value, float) and not math.isfinite(value):
        # ``repr`` gives ``inf``/``nan``, an undefined name at evaluation.
        raise _UnknownShape
    return repr(value)


def _literal(node: ast.expr) -> str:
    """Render a scalar, list or tuple literal without changing its type."""
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (bool, int, float, str)
    ):
        return _scalar(node.value)
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        parts = []
        for elt in node.elts:
            if not (
                isinstance(elt, ast.Constant)
                and isinstance(elt.value, (bool, int, float, str))
            ):
                raise _UnknownShape
            parts.append(_scalar(elt.value))
        if isinstance(node, ast.Tuple):
            # The loader grammar accepts tuples; a one-element tuple needs
            # its trailing comma to stay a tuple.
            return "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"
        return "[" + ", ".join(parts) + "]"
    raise _UnknownShape


def _operand(node: ast.expr) -> str:
    """Render a ``config['x']`` reference or a literal."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
        and _is_tvar_name(node.slice.value)
    ):
        return node.slice.value
    return _literal(node)


def _atom(node: ast.Call) -> str:
    """Render ``name.gte(x)`` style atoms."""
    func = node.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.attr in _ATOM_METHODS
        and _is_tvar_name(func.value.id)
        and len(node.args) == 1
        and not node.keywords
    ):
        value = _literal(node.args[0])
        if func.attr in ("is_in", "not_in") and not value.startswith(("[", "(")):
            raise _UnknownShape
        return f"{func.value.id} {_ATOM_METHODS[func.attr]} {value}"
    raise _UnknownShape


def _boolean(node: ast.expr) -> str:
    """Render a boolean tree of atoms, ``&``/``|``/``~`` and comparisons."""
    if isinstance(node, ast.Call):
        return f"({_atom(node)})"
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        joiner = "and" if isinstance(node.op, ast.BitAnd) else "or"
        return f"({_boolean(node.left)} {joiner} {_boolean(node.right)})"
    if isinstance(node, ast.BoolOp):
        joiner = "and" if isinstance(node.op, ast.And) else "or"
        return "(" + f" {joiner} ".join(_boolean(v) for v in node.values) + ")"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Invert, ast.Not)):
        return f"(not {_boolean(node.operand)})"
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op = _COMPARE_OPS.get(type(node.ops[0]))
        if op is None:
            raise _UnknownShape
        return f"({_operand(node.left)} {op} {_operand(node.comparators[0])})"
    raise _UnknownShape


def _strip_parens(text: str) -> str:
    """Drop one redundant outer pair of parentheses."""
    if text.startswith("(") and text.endswith(")"):
        depth = 0
        for i, ch in enumerate(text):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i < len(text) - 1:
                return text
        return text[1:-1]
    return text


def translate_constraint_code(code: str) -> dict[str, str] | None:
    """Translate one generated constraint code string to a TVL entry.

    Returns ``{"expr": ...}`` or ``{"when": ..., "then": ...}``, or ``None``
    when the code is not one of the known shapes.
    """
    try:
        tree = ast.parse(textwrap.dedent(code).strip(), mode="eval")
    except SyntaxError:
        return None
    body = tree.body
    try:
        if isinstance(body, ast.Lambda):
            if [a.arg for a in body.args.args] != ["config"]:
                return None
            return {"expr": _strip_parens(_boolean(body.body))}
        if (
            isinstance(body, ast.Call)
            and isinstance(body.func, ast.Name)
            and not body.keywords
        ):
            if body.func.id == "require" and len(body.args) == 1:
                return {"expr": _strip_parens(_boolean(body.args[0]))}
            if body.func.id == "implies" and len(body.args) == 2:
                return {
                    "when": _strip_parens(_boolean(body.args[0])),
                    "then": _strip_parens(_boolean(body.args[1])),
                }
    except _UnknownShape:
        return None
    return None


def structural_constraints_to_tvl(
    constraints: tuple[StructuralConstraintSpec, ...],
) -> list[dict[str, Any]]:
    """Convert structural constraints to loader-accepted TVL entries.

    Unknown shapes are skipped with a value-free warning that names only the
    constraint id (``structural_<position>``), never its code or values.
    """
    entries: list[dict[str, Any]] = []
    for index, sc in enumerate(constraints):
        constraint_id = f"structural_{index}"
        translated = translate_constraint_code(sc.constraint_code)
        if translated is None:
            warnings.warn(
                f"Skipped structural constraint '{constraint_id}': its shape is "
                "not expressible in TVL expr/when-then form, so it is not "
                "included in the generated spec.",
                UserWarning,
                stacklevel=3,
            )
            continue
        entry: dict[str, Any] = {"id": constraint_id, **translated}
        if sc.description:
            entry["error_message"] = sc.description
        entries.append(entry)
    return entries
