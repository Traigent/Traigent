"""Tests for static enum-derived parametrize resolution in the KG extractor."""

from __future__ import annotations

import ast
from pathlib import Path

from tests.optimizer_validation.viewer import knowledge_graph as kg
from traigent.config.types import ExecutionMode


def _eval_module_constant(source: str, name: str):
    parser = kg.TestParser(Path("."))
    tree = ast.parse(source)
    return parser._build_module_context(tree).get(name, kg.UNRESOLVED)


IMPORT = "from traigent.config.types import ExecutionMode\n"


def test_enum_value_comprehension_resolves_to_member_values() -> None:
    result = _eval_module_constant(
        IMPORT + "MODES = [mode.value for mode in ExecutionMode]\n", "MODES"
    )
    assert result == [mode.value for mode in ExecutionMode]
    assert "hybrid_api" in result


def test_filtered_comprehension_stays_unresolved() -> None:
    src = IMPORT + "MODES = [m.value for m in ExecutionMode if m.value != 'x']\n"
    assert _eval_module_constant(src, "MODES") is kg.UNRESOLVED


def test_unknown_or_unimported_enum_stays_unresolved() -> None:
    assert (
        _eval_module_constant("MODES = [m.value for m in ExecutionMode]\n", "MODES")
        is kg.UNRESOLVED
    )
    src = "from other_pkg import Color\nC = [c.value for c in Color]\n"
    assert _eval_module_constant(src, "C") is kg.UNRESOLVED


def test_other_comprehension_shapes_stay_unresolved() -> None:
    for expr in (
        "[m.name for m in ExecutionMode]",
        "[m.value for m in OTHER]",
        "[m.value for m in ExecutionMode for _ in ExecutionMode]",
    ):
        assert _eval_module_constant(IMPORT + f"X = {expr}\n", "X") is kg.UNRESOLVED
