"""Regression tests for example-script defect #3209."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "examples"


# --- #3209: traigent must be imported before litellm -------------------------

LITELLM_ORDER_FILES = [
    "examples/core/multi-objective-tradeoff/run_many_providers.py",
    "examples/tvl/reusable_safety/llm_utils.py",
    "walkthrough/real/advanced/04_multi_agent_bedrock.py",
]


def _first_import_line(tree: ast.AST, root: str) -> int | None:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == root for a in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == root:
                lines.append(node.lineno)
    return min(lines) if lines else None


@pytest.mark.parametrize("rel", LITELLM_ORDER_FILES)
def test_traigent_imported_before_litellm(rel):
    tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
    traigent_line = _first_import_line(tree, "traigent")
    litellm_line = _first_import_line(tree, "litellm")
    assert litellm_line is not None
    assert traigent_line is not None and traigent_line < litellm_line
