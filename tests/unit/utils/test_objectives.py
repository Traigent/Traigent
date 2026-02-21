"""Tests for objective direction utilities."""

from __future__ import annotations

from traigent.utils.objectives import is_minimization_objective


def test_is_minimization_objective_for_common_cost_latency_names() -> None:
    assert is_minimization_objective("cost") is True
    assert is_minimization_objective("p95_latency_ms") is True
    assert is_minimization_objective("error_rate") is True


def test_is_minimization_objective_for_maximize_style_name() -> None:
    assert is_minimization_objective("accuracy") is False


def test_is_minimization_objective_uses_substring_heuristic() -> None:
    # Heuristic fallback: compound names containing minimize tokens are treated
    # as minimize objectives.
    assert is_minimization_objective("accuracy_cost_ratio") is True
