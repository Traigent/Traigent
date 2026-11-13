"""#2904: /evaluate items must satisfy the EvaluationItem contract (objects, no null)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from traigent.evaluators.hybrid_api import HybridAPIEvaluator


def _build(expected, outputs):
    ev = HybridAPIEvaluator(transport=MagicMock(), tunable_id="cap", keep_alive=False)
    batch = [{"input": {"q": 1}, "expected_output": expected}]
    inputs = [{"example_id": "a", "data": {"q": 1}}]
    return ev._build_evaluation_payloads(batch, inputs, SimpleNamespace(outputs=outputs))


@pytest.mark.parametrize("expected", ["positive", 42, 1.5, True, None, [1, 2]])
def test_scalar_and_list_targets_are_wrapped(expected):
    (item,) = _build(expected, [{"example_id": "a", "output": {"x": 1}}])
    assert item["target"] == {"value": expected}


def test_object_target_is_unchanged():
    (item,) = _build({"label": "p"}, [{"example_id": "a", "output": {"x": 1}}])
    assert item["target"] == {"label": "p"}


def test_target_id_is_not_wrapped():
    (item,) = _build({"target_id": "t1"}, [{"example_id": "a", "output_id": "o1"}])
    assert item["target_id"] == "t1"
    assert "target" not in item


def test_output_is_omitted_when_execute_carried_none():
    (item,) = _build("positive", [{"example_id": "a"}])
    assert "output" not in item
    assert "output_id" not in item
