"""Round 17 (#2517): budgets charge the known spend floor; span cost pinned."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig
from traigent.core.cost_estimator import CostEstimator
from traigent.core.execution_budget import ExecutionBudget
from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.evaluators.base import BaseEvaluator, EvaluationResult
from traigent.evaluators.metrics_tracker import MetricsTracker, extract_llm_metrics

_COST_ENV = (
    "TRAIGENT_STRICT_COST_ACCOUNTING",
    "TRAIGENT_REQUIRE_COST_TRACKING",
    "TRAIGENT_RUN_COST_LIMIT",
)


def _trial(metrics: dict, **kwargs) -> TrialResult:
    return TrialResult(
        trial_id=kwargs.pop("trial_id", "t"),
        config={},
        status=kwargs.pop("status", TrialStatus.COMPLETED),
        duration=1.0,
        timestamp=datetime.now(UTC),
        metrics=metrics,
        **kwargs,
    )


def test_cost_limit_is_reached_by_flagged_trials_with_a_known_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in _COST_ENV:
        monkeypatch.delenv(key, raising=False)
    enforcer = CostEnforcer(CostEnforcerConfig(limit=2.0))
    enforcer.waive_unknown_cost_trial_cap()
    refused_at = None
    for i in range(8):
        trial = _trial(
            {"accuracy": 0.9, "cost_unmeasured": 1.0, "cost_lower_bound": 1.90},
            trial_id=f"t{i}",
        )
        permit = enforcer.acquire_permit()
        if not permit.active:
            refused_at = i
            break
        enforcer.track_cost(cost=CostEstimator.extract_trial_cost(trial), permit=permit)
    assert refused_at is not None and refused_at < 8
    assert enforcer.accumulated_cost == pytest.approx(1.90 * refused_at)
    assert enforcer.accumulated_cost > 0


def test_partial_usage_trial_charges_at_least_the_priced_spend() -> None:
    """9 priced gpt-4o calls plus 1 fast path: the 9 are charged, not $0."""

    class _Usage:
        prompt_tokens = 100000
        completion_tokens = 50000
        total_tokens = 150000

    class _Response:
        def __init__(self) -> None:
            self.usage = _Usage()
            self.model = "gpt-4o"
            self.choices: list = []

    tracker = MetricsTracker()
    tracker.start_tracking()
    for _ in range(9):
        tracker.add_example_metrics(
            extract_llm_metrics(response=_Response(), model_name="gpt-4o")
        )
    tracker.add_example_metrics(
        extract_llm_metrics(response="fast-path canned answer", model_name="gpt-4o")
    )
    tracker.end_tracking()
    metrics = {k: v for k, v in tracker.format_for_backend().items() if v is not None}
    assert metrics["cost_unmeasured"] == 1.0

    charged = CostEstimator.extract_trial_cost(_trial(metrics))

    assert charged is not None
    assert charged >= metrics["cost_lower_bound"] > 0


def test_execution_budget_charges_the_floor_and_stays_a_minimum() -> None:
    result = EvaluationResult(
        config={},
        aggregated_metrics={
            "accuracy": 0.9,
            "cost": None,
            "cost_unmeasured": 1.0,
            "cost_lower_bound": 1.25,
        },
        total_examples=1,
        successful_examples=1,
        duration=1.0,
    )
    budget = ExecutionBudget(max_cost_usd=10.0)

    BaseEvaluator._finalize_execution_budget_evaluation(budget, result, None)

    snapshot = budget.snapshot()
    assert snapshot.as_dict()["consumed_cost"] == pytest.approx(1.25)
    assert budget._untracked_trials == 1  # the figure is a floor, not exact


def test_execution_budget_flagged_without_a_floor_is_unknown() -> None:
    assert (
        BaseEvaluator._execution_budget_cost(
            SimpleNamespace(aggregated_metrics={"cost": 0.0, "cost_unmeasured": 1.0})
        )
        is None
    )


# --------------------------------------------------------------------------- #
# Workflow span cost line (trial_lifecycle._collect_workflow_span)             #
# --------------------------------------------------------------------------- #
def _collect_span(trial: TrialResult):
    spans: list = []
    orchestrator = SimpleNamespace(
        _workflow_traces_tracker=object(),
        _optimization_id="opt-r17",
        traigent_config=None,
        collect_workflow_span=spans.append,
    )
    lifecycle = TrialLifecycle.__new__(TrialLifecycle)
    lifecycle._orchestrator = orchestrator
    lifecycle._collect_workflow_span("run-1", trial, 1.0, 2.0)
    assert len(spans) == 1
    return spans[0]


def test_failed_trial_without_a_cost_uploads_a_span_with_no_cost() -> None:
    span = _collect_span(
        _trial({"accuracy": 0.0}, status=TrialStatus.FAILED, error_message="boom")
    )
    assert span.cost_usd is None


def test_flagged_trial_span_has_no_cost_and_carries_the_floor() -> None:
    span = _collect_span(
        _trial({"cost_unmeasured": 1.0, "cost_lower_bound": 0.4, "total_cost": 0.4})
    )
    assert span.cost_usd is None
    assert span.cost_lower_bound == pytest.approx(0.4)


def test_measured_zero_cost_span_keeps_zero() -> None:
    assert _collect_span(_trial({"total_cost": 0.0})).cost_usd == 0.0
