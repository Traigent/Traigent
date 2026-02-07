"""Tests for optimization_pipeline.py (collect_orchestrator_kwargs)."""

from __future__ import annotations

from typing import Any

import pytest

from traigent.core.optimization_pipeline import collect_orchestrator_kwargs

# ---------------------------------------------------------------------------
# collect_orchestrator_kwargs: invocations_per_example
# ---------------------------------------------------------------------------


class TestCollectOrchestratorKwargsInvocations:
    """Test that invocations_per_example flows through correctly."""

    def _call(self, *, invocations: int = 1, **extra: Any) -> dict[str, Any]:
        algorithm_kwargs: dict[str, Any] = {"cache_policy": "allow_repeats"}
        algorithm_kwargs.update(extra)
        return collect_orchestrator_kwargs(
            algorithm_kwargs,
            samples_include_pruned_value=False,
            default_config=None,
            constraints=None,
            agents=None,
            agent_prefixes=None,
            agent_measures=None,
            global_measures=None,
            promotion_gate=None,
            invocations_per_example=invocations,
        )

    def test_includes_invocations_in_output(self) -> None:
        result = self._call(invocations=3)
        assert result["invocations_per_example"] == 3

    def test_default_invocations_is_1(self) -> None:
        result = self._call()
        assert result["invocations_per_example"] == 1

    def test_does_not_pop_from_algorithm_kwargs(self) -> None:
        """invocations_per_example arrives as explicit param, NOT popped from kwargs."""
        algorithm_kwargs: dict[str, Any] = {
            "cache_policy": "allow_repeats",
            "invocations_per_example": 999,  # should NOT be touched
        }
        result = collect_orchestrator_kwargs(
            algorithm_kwargs,
            samples_include_pruned_value=False,
            default_config=None,
            constraints=None,
            agents=None,
            agent_prefixes=None,
            agent_measures=None,
            global_measures=None,
            promotion_gate=None,
            invocations_per_example=5,
        )
        # The explicit param wins
        assert result["invocations_per_example"] == 5
        # algorithm_kwargs was NOT mutated
        assert algorithm_kwargs["invocations_per_example"] == 999

    def test_backward_compat_without_param(self) -> None:
        """Calling without invocations_per_example defaults to 1."""
        result = collect_orchestrator_kwargs(
            {"cache_policy": "allow_repeats"},
            samples_include_pruned_value=False,
            default_config=None,
            constraints=None,
            agents=None,
            agent_prefixes=None,
            agent_measures=None,
            global_measures=None,
            promotion_gate=None,
        )
        assert result["invocations_per_example"] == 1
