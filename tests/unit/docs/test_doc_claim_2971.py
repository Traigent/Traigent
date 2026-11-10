"""Doc claim matches code (#2971)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _raises_names(source: str, func: str) -> list[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            doc = ast.get_docstring(node) or ""
            block = doc.split("Raises:", 1)[1].split("\n\n", 1)[0]
            return re.findall(r"^\s*(\w+):", block, re.M)
    raise AssertionError(f"{func} not found")


def test_optimize_docstring_raises_only_real_exceptions_2971() -> None:
    import traigent.utils.exceptions as exc_mod

    names = _raises_names(_read("traigent/api/decorators.py"), "optimize")
    assert names
    unresolved = [n for n in names if not hasattr(exc_mod, n)]
    assert not unresolved, unresolved
