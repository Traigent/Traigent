"""Tests for objective direction utilities."""

from __future__ import annotations

import pytest

from traigent.utils.objectives import is_minimization_objective


def test_is_minimization_objective_for_common_cost_latency_names() -> None:
    assert is_minimization_objective("cost") is True
    assert is_minimization_objective("p95_latency_ms", orientation="minimize") is True
    assert is_minimization_objective("error_rate") is True


def test_is_minimization_objective_for_maximize_style_name() -> None:
    assert is_minimization_objective("accuracy") is False


def test_is_minimization_objective_rejects_compound_name_guess() -> None:
    with pytest.raises(ValueError, match="has no declared orientation"):
        is_minimization_objective("accuracy_cost_ratio")


def test_explicit_orientation_wins_for_misleading_name() -> None:
    assert is_minimization_objective("cost_savings", orientation="maximize") is False
