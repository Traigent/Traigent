"""Shared safety helpers for hardened example workflows.

Provides three primitives used by the examples and example runners to address
high-severity findings from issue #846:

- ``safe_arithmetic`` evaluates an arithmetic expression using an explicit AST
  allowlist (no ``eval``, no ``__builtins__``).
- ``wrap_untrusted`` isolates untrusted user content inside delimiter sentinels,
  caps length, and strips control characters so that examples cannot
  accidentally treat prompt input as instructions.
- ``resolve_within`` resolves a target path strictly under an allowed root and
  rejects path traversal and (by default) symlink escapes.

These helpers are intentionally dependency-free so any example or script in
this repo can import them without affecting the public ``traigent`` package.
"""

from __future__ import annotations

import ast
import operator
import os
import re
from pathlib import Path
from typing import Final

Number = int | float

DEFAULT_UNTRUSTED_CHAR_BUDGET: Final[int] = 4000
DEFAULT_ARITHMETIC_CHAR_BUDGET: Final[int] = 256

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_POW_EXPONENT: Final[int] = 64
_MAX_POW_BASE_ABS: Final[float] = 1e6


class UnsafeExpressionError(ValueError):
    """Raised when an expression contains constructs outside the allowlist."""


class UntrustedPathError(ValueError):
    """Raised when a target path escapes the allowed root."""


def safe_arithmetic(
    expression: str,
    *,
    max_chars: int = DEFAULT_ARITHMETIC_CHAR_BUDGET,
) -> Number:
    """Safely evaluate a numeric arithmetic expression.

    Allows: integer/float literals, parentheses, unary +/-, and binary
    + - * / // % **. Rejects names, attribute access, calls, subscripts,
    comprehensions, comparisons, boolean ops, and every other AST node.

    ``^`` is rewritten to ``**`` for convenience, matching the historical
    behavior of the examples that used ``eval``.

    Raises ``UnsafeExpressionError`` on any disallowed construct or on an
    expression that exceeds ``max_chars``.
    """
    if not isinstance(expression, str):
        raise UnsafeExpressionError("Expression must be a string")
    if len(expression) > max_chars:
        raise UnsafeExpressionError(f"Expression exceeds {max_chars} characters")

    rewritten = expression.replace("^", "**")

    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Invalid expression: {exc}") from exc

    return _eval_arithmetic_node(tree.body)


def _eval_arithmetic_node(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnsafeExpressionError(
                f"Only numeric literals are permitted, got {type(value).__name__}"
            )
        return value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        op = _BINOPS.get(op_type)
        if op is None:
            raise UnsafeExpressionError(f"Operator {op_type.__name__} is not allowed")
        left = _eval_arithmetic_node(node.left)
        right = _eval_arithmetic_node(node.right)
        if op_type is ast.Pow:
            _guard_power(left, right)
        try:
            return op(left, right)
        except ZeroDivisionError as exc:
            raise UnsafeExpressionError("Division by zero") from exc

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise UnsafeExpressionError(
                f"Unary operator {type(node.op).__name__} is not allowed"
            )
        return op(_eval_arithmetic_node(node.operand))

    raise UnsafeExpressionError(
        f"AST node {type(node).__name__} is not allowed in safe arithmetic"
    )


def _guard_power(base: Number, exponent: Number) -> None:
    """Reject pathological exponents that could trivially DoS the interpreter."""
    if isinstance(exponent, float) and not exponent.is_integer():
        return  # fractional exponents are computed as float, bounded by IEEE
    exponent_int = int(exponent)
    if exponent_int > _MAX_POW_EXPONENT:
        raise UnsafeExpressionError(
            f"Exponent {exponent_int} exceeds limit {_MAX_POW_EXPONENT}"
        )
    if exponent_int >= 0 and abs(float(base)) > _MAX_POW_BASE_ABS:
        raise UnsafeExpressionError(
            f"Base magnitude {base} exceeds limit {_MAX_POW_BASE_ABS} for ** operator"
        )


# Match all ASCII control characters EXCEPT \t (0x09) and \n (0x0A); also drop
# the C1 range used by some prompt-injection payloads.
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def wrap_untrusted(
    label: str,
    content: object,
    *,
    max_chars: int = DEFAULT_UNTRUSTED_CHAR_BUDGET,
) -> str:
    """Wrap ``content`` as an untrusted-data block usable inside an LLM prompt.

    The returned string starts and ends with explicit sentinel markers so
    downstream prompt templates can instruct the model not to treat the
    enclosed text as instructions. Content longer than ``max_chars`` is
    truncated with a visible marker; control characters are stripped to
    avoid hidden directives.
    """
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label or "data"))[:64] or "data"
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        text = str(content)

    cleaned = _CONTROL_CHAR_PATTERN.sub("", text)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "...[truncated]"

    sentinel = f"untrusted_{safe_label}"
    cleaned = cleaned.replace(f"<{sentinel}>", f"<{sentinel}_literal>").replace(
        f"</{sentinel}>", f"</{sentinel}_literal>"
    )
    return f"<{sentinel}>\n{cleaned}\n</{sentinel}>"


def resolve_within(
    root: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    allow_symlinks: bool = False,
    must_exist: bool = False,
) -> Path:
    """Resolve ``target`` strictly under ``root``.

    Rejects:
    - empty target,
    - targets containing NUL bytes,
    - targets that resolve outside ``root``,
    - symlink components when ``allow_symlinks`` is False,
    - missing targets when ``must_exist`` is True.

    Returns the resolved absolute path.
    """
    if target is None:
        raise UntrustedPathError("Target path is required")
    raw_target = os.fspath(target)
    if not raw_target:
        raise UntrustedPathError("Target path is required")
    if "\x00" in raw_target:
        raise UntrustedPathError("Target path contains NUL byte")

    root_path = Path(os.fspath(root)).resolve(strict=False)
    target_path = Path(raw_target)
    if not target_path.is_absolute():
        target_path = root_path / target_path

    resolved = target_path.resolve(strict=False)

    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise UntrustedPathError(
            f"Path {raw_target!r} escapes allowed root {root_path}"
        ) from exc

    if not allow_symlinks:
        cursor = resolved
        while True:
            if cursor.is_symlink():
                raise UntrustedPathError(
                    f"Path {raw_target!r} traverses a symlink: {cursor}"
                )
            if cursor == root_path or cursor.parent == cursor:
                break
            cursor = cursor.parent

    if must_exist and not resolved.exists():
        raise UntrustedPathError(f"Path {raw_target!r} does not exist")

    return resolved


__all__ = [
    "DEFAULT_ARITHMETIC_CHAR_BUDGET",
    "DEFAULT_UNTRUSTED_CHAR_BUDGET",
    "UnsafeExpressionError",
    "UntrustedPathError",
    "resolve_within",
    "safe_arithmetic",
    "wrap_untrusted",
]
