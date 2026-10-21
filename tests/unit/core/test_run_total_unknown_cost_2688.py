"""Run-level cost totals are honest about unmeasured trials (#2688).

An unknown cost is never summed as a measured number: the run total is
unknown and the exact floor travels in its own field with a count.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.metric_registry import MetricRegistry
from traigent.core.metrics_aggregator import aggregate_metrics
from traigent.evaluators.base import SimpleScoringEvaluator
from traigent.utils.trial_costs import COST_FLOOR_WIRE_KEY


def _trial(trial_id: str, metrics: dict[str, Any]) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config={"m": trial_id},
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
        metadata={},
    )


def _registry() -> MetricRegistry:
    return MetricRegistry()


def test_mixed_run_total_unknown_with_floor_and_count() -> None:
    trials = [
        _trial("a", {"accuracy": 0.9, "total_cost": 0.10}),
        _trial(
            "b", {"accuracy": 0.8, "cost_unmeasured": 1.0, "cost_lower_bound": 0.04}
        ),
        _trial("c", {"accuracy": 0.7, "cost_unmeasured": 1.0}),
    ]
    agg = aggregate_metrics(trials, _registry())
    assert agg.total_cost is None
    assert agg.cost_unmeasured_trials == 2
    assert agg.cost_floor_usd == pytest.approx(0.14)
    assert "total_cost" not in agg.processed_metrics
    assert agg.processed_metrics[COST_FLOOR_WIRE_KEY] == pytest.approx(0.14)
    assert agg.processed_metrics["cost_unmeasured_trials"] == 2.0


def test_trial_with_no_cost_data_is_unmeasured_not_zero() -> None:
    agg = aggregate_metrics(
        [
            _trial("a", {"accuracy": 0.9, "total_cost": 0.10}),
            _trial("b", {"accuracy": 0.8}),
        ],
        _registry(),
    )
    assert agg.total_cost is None
    assert agg.cost_unmeasured_trials == 1
    assert agg.cost_floor_usd == pytest.approx(0.10)


def test_all_measured_reports_normal_total() -> None:
    agg = aggregate_metrics(
        [
            _trial("a", {"accuracy": 0.9, "total_cost": 0.10}),
            _trial("b", {"accuracy": 0.8, "total_cost": 0.25}),
        ],
        _registry(),
    )
    assert agg.total_cost == pytest.approx(0.35)
    assert agg.cost_unmeasured_trials == 0
    assert agg.processed_metrics["total_cost"] == pytest.approx(0.35)
    assert COST_FLOOR_WIRE_KEY not in agg.processed_metrics


def test_simple_scoring_llm_metrics_omit_placeholder_costs() -> None:
    for flag in ("unmeasured", "unpriced"):
        cost = SimpleNamespace(
            total_cost=0.0,
            input_cost=0.0,
            output_cost=0.0,
            known_cost=0.02,
            unpriced=flag == "unpriced",
            unmeasured=flag == "unmeasured",
        )
        obj = SimpleNamespace(
            tokens=SimpleNamespace(
                total_tokens=5, prompt_tokens=3, completion_tokens=2
            ),
            cost=cost,
            response=SimpleNamespace(response_time_ms=1, tokens_per_second=1),
        )
        evaluator = SimpleScoringEvaluator.__new__(SimpleScoringEvaluator)
        llm = evaluator._build_llm_metrics_dict(obj, "m")
        assert "total_cost" not in llm
        assert "input_cost" not in llm
        assert "output_cost" not in llm
        assert llm["known_cost"] == pytest.approx(0.02)

        example: dict[str, Any] = {}
        evaluator._add_llm_metrics_to_example(example, llm)
        assert example["cost_unmeasured"] == 1.0
        assert "total_cost" not in example
        assert example["known_cost"] == pytest.approx(0.02)


def test_simple_scoring_measured_costs_unchanged() -> None:
    cost = SimpleNamespace(
        total_cost=0.03,
        input_cost=0.01,
        output_cost=0.02,
        known_cost=0.0,
        unpriced=False,
        unmeasured=False,
    )
    obj = SimpleNamespace(
        tokens=SimpleNamespace(total_tokens=5, prompt_tokens=3, completion_tokens=2),
        cost=cost,
        response=SimpleNamespace(response_time_ms=1, tokens_per_second=1),
    )
    evaluator = SimpleScoringEvaluator.__new__(SimpleScoringEvaluator)
    llm = evaluator._build_llm_metrics_dict(obj, "m")
    assert llm["total_cost"] == pytest.approx(0.03)
    example: dict[str, Any] = {}
    evaluator._add_llm_metrics_to_example(example, llm)
    assert example["total_cost"] == pytest.approx(0.03)
    assert "cost_unmeasured" not in example


def test_stability_spend_raises_floor_but_total_stays_unknown() -> None:
    from traigent.api.types import OptimizationResult
    from traigent.core.orchestrator import OptimizationOrchestrator

    result = OptimizationResult(
        trials=[],
        best_config={},
        best_score=0.0,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status=__import__(
            "traigent.api.types", fromlist=["x"]
        ).OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="a",
        timestamp=datetime.now(UTC),
        total_cost=None,
        metrics={COST_FLOOR_WIRE_KEY: 0.14, "cost_unmeasured_trials": 2.0},
    )
    OptimizationOrchestrator._add_winner_stability_cost_to_result(result, 0.05)
    assert result.total_cost is None
    assert "total_cost" not in result.metrics
    assert result.metrics[COST_FLOOR_WIRE_KEY] == pytest.approx(0.19)


def _result_with_total(total: float | None, metrics: dict[str, Any]) -> Any:
    from traigent.api.types import OptimizationResult, OptimizationStatus

    return OptimizationResult(
        trials=[],
        best_config={},
        best_score=0.0,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="a",
        timestamp=datetime.now(UTC),
        total_cost=total,
        metrics=metrics,
    )


def test_unmeasured_rerun_makes_measured_run_total_unknown() -> None:
    """Measured $0.10 run + unmeasured rerun with $0.04 floor is NOT $0.14 total."""
    from traigent.core.orchestrator import OptimizationOrchestrator

    result = _result_with_total(0.10, {"total_cost": 0.10})
    OptimizationOrchestrator._add_winner_stability_cost_to_result(
        result, 0.04, unmeasured_reruns=1
    )
    assert result.total_cost is None
    assert "total_cost" not in result.metrics
    assert result.metrics[COST_FLOOR_WIRE_KEY] == pytest.approx(0.14)
    assert result.metrics["cost_unmeasured_trials"] == 1.0


def test_unmeasured_rerun_adds_to_existing_unmeasured_count() -> None:
    from traigent.core.orchestrator import OptimizationOrchestrator

    result = _result_with_total(
        None, {COST_FLOOR_WIRE_KEY: 0.14, "cost_unmeasured_trials": 2.0}
    )
    OptimizationOrchestrator._add_winner_stability_cost_to_result(
        result, 0.0, unmeasured_reruns=1
    )
    assert result.total_cost is None
    assert result.metrics[COST_FLOOR_WIRE_KEY] == pytest.approx(0.14)
    assert result.metrics["cost_unmeasured_trials"] == 3.0


def test_measured_rerun_still_adds_to_measured_total() -> None:
    from traigent.core.orchestrator import OptimizationOrchestrator

    result = _result_with_total(0.10, {"total_cost": 0.10})
    OptimizationOrchestrator._add_winner_stability_cost_to_result(result, 0.04)
    assert result.total_cost == pytest.approx(0.14)
    assert result.metrics["total_cost"] == pytest.approx(0.14)


def _cancelled(trial_id: str, metrics: dict[str, Any]) -> TrialResult:
    trial = _trial(trial_id, metrics)
    trial.status = TrialStatus.CANCELLED
    return trial


def test_cancelled_unmeasured_trial_keeps_run_total_unknown_with_floor() -> None:
    """Measured completed trial + cancelled unmeasured trial: no numeric total."""
    agg = aggregate_metrics(
        [
            _trial("a", {"accuracy": 0.9, "total_cost": 0.10}),
            _cancelled("b", {"cost_unmeasured": 1.0, "cost_lower_bound": 0.03}),
        ],
        _registry(),
    )
    assert agg.total_cost is None
    assert agg.cost_unmeasured_trials == 1
    assert agg.cost_floor_usd == pytest.approx(0.13)
    assert "total_cost" not in agg.processed_metrics


def test_cancelled_trial_without_cost_data_is_unknown_not_zero() -> None:
    agg = aggregate_metrics(
        [_trial("a", {"accuracy": 0.9, "total_cost": 0.10}), _cancelled("b", {})],
        _registry(),
    )
    assert agg.total_cost is None
    assert agg.cost_unmeasured_trials == 1
    assert agg.cost_floor_usd == pytest.approx(0.10)


def test_cancelled_trial_with_measured_cost_counts_in_total() -> None:
    agg = aggregate_metrics(
        [
            _trial("a", {"accuracy": 0.9, "total_cost": 0.10}),
            _cancelled("b", {"total_cost": 0.02}),
        ],
        _registry(),
    )
    assert agg.total_cost == pytest.approx(0.12)
    assert agg.cost_unmeasured_trials == 0


class _RaisingEnforcer:
    """Cost enforcer whose tracking raises, like strict/explicit-limit mode."""

    async def track_cost_async(self, **_kwargs: Any) -> None:
        from traigent.core.cost_enforcement import CostTrackingRequiredError

        raise CostTrackingRequiredError("strict cost tracking aborted the rerun")


@pytest.mark.asyncio
async def test_enforcement_abort_still_records_unmeasured_rerun_floor() -> None:
    """Enforcement raising mid-rerun must not lose the rerun's floor/unknownness."""
    import asyncio

    from traigent.core.cost_enforcement import CostTrackingRequiredError
    from traigent.core.orchestrator import OptimizationOrchestrator

    orch = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orch._state_lock = asyncio.Lock()
    orch.cost_enforcer = _RaisingEnforcer()
    orch.execution_budget = None
    orch._winner_stability_accounted_cost = 0.0
    orch._winner_stability_unmeasured_reruns = 0
    orch._winner_stability_recorded_ids = {}

    rerun = _trial(
        "winner_stability_o_1",
        {"cost_unmeasured": 1.0, "cost_lower_bound": 0.04},
    )

    # Fail closed: enforcement still aborts the accounting call...
    with pytest.raises(CostTrackingRequiredError):
        await orch._account_winner_stability_rerun_attempt(
            rerun_trial=rerun,
            rerun_dispatched=True,
            permit=None,
            trial_id=rerun.trial_id,
        )
    # ...a retried accounting (the orchestrator's fallback) never double counts.
    with pytest.raises(CostTrackingRequiredError):
        await orch._account_winner_stability_rerun_attempt(
            rerun_trial=rerun,
            rerun_dispatched=True,
            permit=None,
            trial_id=rerun.trial_id,
        )

    result = _result_with_total(0.10, {"total_cost": 0.10})
    OptimizationOrchestrator._add_winner_stability_cost_to_result(
        result,
        orch._winner_stability_accounted_cost,
        orch._winner_stability_unmeasured_reruns,
    )
    assert result.total_cost is None
    assert "total_cost" not in result.metrics
    assert result.metrics[COST_FLOOR_WIRE_KEY] == pytest.approx(0.14)
    assert result.metrics["cost_unmeasured_trials"] == 1.0
