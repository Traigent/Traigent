"""Regression tests for example-script defect #3203."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "examples"


# --- #3203: objectives must name a metric the SDK measures -------------------

_OBJECTIVE_CALLS = {"create_default_objectives", "ObjectiveDefinition"}


def _call_name(node: ast.Call) -> str:
    func = node.func
    return getattr(func, "id", None) or getattr(func, "attr", "")


def _bad_objective_sites() -> list[str]:
    bad = []
    for path in sorted(EXAMPLES.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) in _OBJECTIVE_CALLS:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and sub.value == "response_time":
                        bad.append(f"{path.relative_to(REPO_ROOT)}:{sub.lineno}")
    return bad


def test_no_example_declares_unmatched_response_time_objective():
    assert _bad_objective_sites() == []
