"""``select_failure_basis``: which field decided each failing example."""

from __future__ import annotations

import pytest

from traigent.utils.outcome_matrix import (
    select_failing_example_ids,
    select_failure_basis,
)


def _matrix(rows: dict[str, dict[str, dict]]) -> dict:
    return {
        "trials": [
            {"trial_id": "t0", "config": {"m": 1}},
            {"trial_id": "t1", "config": {"m": 2}},
        ],
        "examples": [{"example_id": k, "cells": v} for k, v in rows.items()],
    }


def test_basis_names_score_accuracy_error_and_mixed() -> None:
    matrix = _matrix(
        {
            "by_score": {"t0": {"score": 0.1}, "t1": {"score": 0.2}},
            "by_accuracy": {"t0": {"accuracy": 0.1}, "t1": {"accuracy": 0.0}},
            "by_error": {"t0": {"error": "x"}, "t1": {"success": False}},
            "mixed": {"t0": {"error": "x"}, "t1": {"score": 0.1}},
            "passes": {"t0": {"score": 0.9}, "t1": {"score": 0.1}},
        }
    )
    failing = select_failing_example_ids(matrix, mode="all_configs")
    assert set(failing) == {"by_score", "by_accuracy", "by_error", "mixed"}
    assert select_failure_basis(matrix, failing, mode="all_configs") == {
        "by_score": ("score", "score"),
        "by_accuracy": ("score", "accuracy"),
        "by_error": ("error", "error"),
        "mixed": ("mixed", "mixed"),
    }


def test_basis_uses_only_the_winning_columns_and_the_threshold() -> None:
    matrix = _matrix({"x": {"t0": {"score": 0.6}, "t1": {"error": "boom"}}})
    assert select_failure_basis(
        matrix, ["x"], mode="best_config", best_config={"m": 2}
    ) == {"x": ("error", "error")}
    assert select_failure_basis(matrix, ["x"], mode="all_configs", threshold=0.5) == {}
    with pytest.raises(ValueError):
        select_failure_basis(matrix, ["x"], mode="nope")
