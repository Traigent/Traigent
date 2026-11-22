"""Scan ``traigent/`` for the static path words the SDK uses for backend routes.

A word is a path segment made only of lowercase ASCII letters, ``_`` and ``-`` in a
string (or f-string) that either contains ``/api/`` or is a whole relative route such
as ``/evaluators/{}/execute``, or in a ``*_PATH``/``*_ENDPOINT``/``*_ROUTE`` constant.
F-string placeholders render as ``{}`` and are never words. Docstrings are skipped.
The egress package itself is excluded (its allowlist is what is being checked).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_WORD = re.compile(r"[a-z_-]+")
_RELATIVE_ROUTE = re.compile(r"/[a-z][a-z0-9_{}/-]*(\?.*)?")
_PATH_NAME = re.compile(r"(PATH|ENDPOINT|ROUTE)S?$", re.IGNORECASE)
_MAX_LEN = 80
#: Filesystem roots: a relative "/etc/..." string is a file path, not a backend route.
_FS_ROOTS = frozenset(
    {"etc", "proc", "sys", "var", "dev", "root", "boot", "tmp", "usr", "home", "bin"}
)


def _render(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value if isinstance(v, ast.Constant) else "{}" for v in node.values
        )
    return None


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def _target_name(node: ast.Assign | ast.AnnAssign) -> str:
    target = node.targets[0] if isinstance(node, ast.Assign) else node.target
    return str(getattr(target, "id", None) or getattr(target, "attr", None) or "")


def scan_route_words(package_dir: Path) -> dict[str, str]:
    """Return ``{word: first "file:line" that uses it}``."""
    found: dict[str, str] = {}
    for path in sorted(package_dir.rglob("*.py")):
        if "egress" in path.relative_to(package_dir).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = _docstring_ids(tree)
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, (ast.Constant, ast.JoinedStr)) and id(node) not in docs:
                text = _render(node)
                if text and "\n" not in text and len(text) < _MAX_LEN * 4:
                    is_relative = len(text) < _MAX_LEN and _RELATIVE_ROUTE.fullmatch(
                        text
                    )
                    if is_relative and text.split("/")[1] in _FS_ROOTS:
                        continue
                    if "/api/" in text or is_relative:
                        candidates.append(text)
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                text = _render(node.value)
                if (
                    text
                    and text.startswith("/")
                    and _PATH_NAME.search(_target_name(node))
                ):
                    candidates.append(text)
            for text in candidates:
                for segment in text.split("?", 1)[0].split("/"):
                    if _WORD.fullmatch(segment):
                        found.setdefault(
                            segment, f"{path}:{getattr(node, 'lineno', 0)}"
                        )
    return found
