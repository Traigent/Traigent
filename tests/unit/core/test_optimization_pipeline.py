"""Tests for optimization_pipeline.py (collect_orchestrator_kwargs)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.core.optimization_pipeline import (
    collect_orchestrator_kwargs,
    create_workflow_traces_tracker,
)

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


class TestCollectOrchestratorKwargsRemovedBudgetKeys:
    """Regression: removed budget_* keys must raise TypeError, not be silently dropped."""

    def _call(self, **extra: Any) -> dict[str, Any]:
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
        )

    def test_budget_limit_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="budget_limit"):
            self._call(budget_limit=10)

    def test_budget_metric_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="budget_metric"):
            self._call(budget_metric="examples_attempted")

    def test_budget_include_pruned_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="budget_include_pruned"):
            self._call(budget_include_pruned=True)

    def test_multiple_budget_keys_raise_together(self) -> None:
        with pytest.raises(TypeError, match="budget_limit"):
            self._call(budget_limit=5, budget_metric="total_cost")

    def test_valid_keys_still_pass_through(self) -> None:
        """Sanity check: cost_limit and metric_limit are the correct replacements."""
        result = self._call(cost_limit=1.5, metric_limit=0.9, metric_name="accuracy")
        assert result["cost_limit"] == 1.5
        assert result["metric_limit"] == 0.9
        assert result["metric_name"] == "accuracy"


@pytest.fixture
def workflow_trace_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "https://backend.example")
    monkeypatch.setenv("TRAIGENT_API_KEY", "api-key")
    monkeypatch.delenv("TRAIGENT_TRACE_ENABLED", raising=False)


def _patch_workflow_traces_tracker(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    tracker = object()
    tracker_factory = MagicMock(return_value=tracker)
    monkeypatch.setattr(
        "traigent.integrations.observability.workflow_traces.WorkflowTracesTracker",
        tracker_factory,
    )
    return tracker, tracker_factory


class TestCreateWorkflowTracesTrackerTraceEnv:
    """Workflow trace tracker follows the same trace enablement env contract."""

    def test_canonical_only_enabled_creates_tracker(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workflow_trace_backend_env: None,
    ) -> None:
        monkeypatch.setenv("TRAIGENT_TRACE_ENABLED", "true")
        tracker, tracker_factory = _patch_workflow_traces_tracker(monkeypatch)

        result = create_workflow_traces_tracker(None)  # type: ignore[arg-type]

        assert result is tracker
        tracker_factory.assert_called_once_with(
            backend_url="https://backend.example",
            auth_token="api-key",
        )

    def test_neither_set_defaults_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
        workflow_trace_backend_env: None,
    ) -> None:
        _, tracker_factory = _patch_workflow_traces_tracker(monkeypatch)

        result = create_workflow_traces_tracker(None)  # type: ignore[arg-type]

        assert result is None
        tracker_factory.assert_not_called()
