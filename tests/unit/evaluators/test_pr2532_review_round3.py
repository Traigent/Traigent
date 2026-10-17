"""Round-3 sol findings on merged PR 2532: tracked spend is the one authority.

Owner ruling A: a partly priced row is UNKNOWN cost (floor only); budget debit
and every cost summary come from tracker data (``tracked_spend``), never from a
user's custom ``cost``/``total_cost`` score. A path with no tracker data is
untracked, not a user-score debit.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.api.types import (
    ExampleResult,
    OptimizationResult,
    TrialResult,
    TrialStatus,
)
from traigent.core.cost_estimator import CostEstimator
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.core.execution_budget import ExecutionBudget
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    Dataset,
    EvaluationExample,
    SimpleScoringEvaluator,
)
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    ResponseMetrics,
    TokenMetrics,
)
from traigent.utils.langchain_interceptor import capture_langchain_response

from tests.unit.evaluators.test_pr2532_review_findings import (
    _completion,
    _trial_via_factory,
)


def _dataset() -> Dataset:
    return Dataset([EvaluationExample({"q": "a"}, 42)])


def _example_result(example: Any, actual: Any, metrics: dict) -> ExampleResult:
    return ExampleResult(
        example_id="e0",
        input_data=example.input_data,
        expected_output=example.expected_output,
        actual_output=actual,
        metrics=metrics,
        execution_time=0.0,
        success=True,
        error_message=None,
        metadata={},
    )


# 1 -- wrapper: a priced call plus an unpriced call is unknown -----------------
async def test_wrapper_priced_plus_unpriced_call_is_unknown(monkeypatch) -> None:
    async def custom(func, config, example):
        return _example_result(example, func(**example.input_data), {"accuracy": 1.0})

    evaluator = CustomEvaluatorWrapper(custom, metrics=["accuracy", "cost"])
    partly_priced = ExampleMetrics(
        tokens=TokenMetrics(input_tokens=100, output_tokens=50, total_tokens=150),
        response=ResponseMetrics(response_time_ms=10),
        cost=CostMetrics(input_cost=0.20, output_cost=0.0, unpriced=True),
        success=True,
    )
    monkeypatch.setattr(
        evaluator, "_capture_llm_metrics_for_example", lambda *a, **k: partly_priced
    )
    budget = ExecutionBudget(max_cost_usd=100.0)
    result = await evaluator.evaluate(
        lambda q: 42, {"model": "gpt-4o"}, _dataset(), budget=budget
    )

    row = result.example_results[0].metrics
    assert row.get("cost_unmeasured") == 1.0
    assert "total_cost" not in row
    assert row.get("cost_lower_bound") == pytest.approx(0.20)
    agg = result.aggregated_metrics
    assert agg.get("cost_unmeasured") == 1.0
    assert agg.get("total_cost") is None
    assert agg.get("cost") is None  # excluded from cost ranking
    assert result.execution_budget["cost_tracking"] == "untracked"
    assert budget.consumed_cost == pytest.approx(0.20)  # the floor only
    trial = _trial_via_factory(result)
    assert CostEstimator.extract_trial_cost(trial) == pytest.approx(0.20)


# 2 -- SimpleScoringEvaluator: snapshot built from the tracker only ------------
async def test_simple_scoring_no_captured_calls_custom_total_cost_not_debited() -> None:
    budget = ExecutionBudget(max_cost_usd=1000.0)
    evaluator = SimpleScoringEvaluator(
        metric_functions={"total_cost": lambda *a, **k: 200.0}
    )
    result = await evaluator.evaluate(
        lambda q: 42, {"model": "gpt-4o-mini"}, _dataset(), budget=budget
    )
    assert budget.consumed_cost == 0.0
    assert result.execution_budget["cost_tracking"] == "untracked"
    assert result.tracked_spend["unmeasured"] is True


async def test_simple_scoring_usage_less_response_custom_cost_not_a_floor() -> None:
    def func(q: str) -> int:
        capture_langchain_response(_completion("gpt-4o-mini", usage=False))
        return 42

    budget = ExecutionBudget(max_cost_usd=1000.0)
    evaluator = SimpleScoringEvaluator(
        metric_functions={"total_cost": lambda *a, **k: 200.0}
    )
    result = await evaluator.evaluate(
        func, {"model": "gpt-4o-mini"}, _dataset(), budget=budget
    )
    assert budget.consumed_cost != pytest.approx(200.0)
    assert result.tracked_spend["lower_bound"] == 0.0
    assert result.execution_budget["cost_tracking"] == "untracked"


# 3 -- wrapper, capture off: custom cost never debits --------------------------
# Carve-out (owner default A, 2026-10-17, pending explicit confirmation): with
# no tracker data at all, a custom evaluator's own reported cost counts and the
# run is not marked untracked. This keeps cost-objective runs of custom
# evaluators working (see test_cost_objective_run_survives_failed_rows_
# reporting_zero_attempts).
async def test_wrapper_capture_off_custom_cost_counts_when_no_tracker_data() -> None:
    async def custom(func, config, example):
        return _example_result(
            example,
            func(**example.input_data),
            {"accuracy": 1.0, "cost": 42.0, "total_cost": 99.0},
        )

    budget = ExecutionBudget(max_cost_usd=1000.0)
    evaluator = CustomEvaluatorWrapper(
        custom, metrics=["accuracy", "cost"], capture_llm_metrics=False
    )
    result = await evaluator.evaluate(lambda q: 42, {}, _dataset(), budget=budget)

    assert budget.consumed_cost == pytest.approx(42.0)
    assert result.execution_budget["cost_tracking"] != "untracked"
    trial = _trial_via_factory(result)
    assert CostEstimator.extract_trial_cost(trial) is not None


# 4 -- persistence: tracked_spend is a durable fact ----------------------------
def _tracked_trial() -> TrialResult:
    trial = TrialResult(
        trial_id="t-tracked",
        config={"model": "cheap"},
        metrics={"accuracy": 0.5, "total_cost": 200.0, "cost": 200.0},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime(2026, 4, 1, 9, 15, 0, tzinfo=UTC),
    )
    trial.tracked_spend = {"unmeasured": False, "cost": 0.20, "lower_bound": 0.0}
    return trial


def test_tracked_spend_is_classified_in_the_trial_manifest() -> None:
    from traigent.utils.optimization_result_persistence import (
        TRIAL_RESET,
        TRIAL_RESTORE,
    )

    assert "tracked_spend" in (TRIAL_RESTORE | TRIAL_RESET)
    assert "tracked_spend" in TRIAL_RESTORE


def test_config_state_reload_restores_tracked_spend(tmp_path) -> None:
    from tests.unit.utils.test_trial_result_round_trip_2047 import (
        _config_state_round_trip,
    )

    loaded = _config_state_round_trip(tmp_path, [_tracked_trial()])
    (trial,) = loaded.trials
    assert trial.tracked_spend == {
        "unmeasured": False,
        "cost": 0.20,
        "lower_bound": 0.0,
    }
    assert CostEstimator.extract_trial_cost(trial) == pytest.approx(0.20)


def test_persistence_manager_reload_restores_tracked_spend(tmp_path) -> None:
    from tests.unit.utils.test_trial_result_round_trip_2047 import (
        _persistence_round_trip,
    )

    loaded = _persistence_round_trip(tmp_path, [_tracked_trial()])
    (trial,) = loaded.trials
    assert trial.tracked_spend is not None
    assert CostEstimator.extract_trial_cost(trial) == pytest.approx(0.20)


# 5 -- orchestrator: a custom zero is preserved for the optimizer --------------
def test_objective_values_keep_custom_zero_cost() -> None:
    orch = SimpleNamespace(optimizer=SimpleNamespace(objectives=["cost"]))
    values = OptimizationOrchestrator._extract_objective_values(
        orch, {"cost": 0.0, "total_cost": 0.25}
    )
    assert values == [0.0]


# 6 -- summary floor consults tracked_spend ------------------------------------
def test_experiment_stats_floor_uses_tracked_spend_not_custom_cost() -> None:
    trial = TrialResult(
        trial_id="t1",
        config={"id": "a"},
        metrics={"accuracy": 0.5, "cost": 100.0},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )
    trial.tracked_spend = {"unmeasured": True, "cost": None, "lower_bound": 0.50}
    result = OptimizationResult(
        trials=[trial],
        best_config={"id": "a"},
        best_score=0.5,
        optimization_id="o",
        duration=1.0,
        convergence_info={},
        status="completed",
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(),
    )
    stats = result.experiment_stats
    assert stats.total_cost is None
    assert stats.cost_lower_bound == pytest.approx(0.50)


# 7 -- an unpriced judge call after a priced agent call: detail rows withdrawn -
@pytest.mark.parametrize("judge_tokens", [30, 0])
def test_unpriced_judge_call_withdraws_detail_row_cost(
    monkeypatch, judge_tokens: int
) -> None:
    import traigent.evaluators.local as local_module

    evaluator = LocalEvaluator(detailed=True, metrics=["accuracy"])
    example_metric = ExampleMetrics(
        tokens=TokenMetrics(input_tokens=100, output_tokens=50, total_tokens=150),
        response=ResponseMetrics(response_time_ms=10),
        cost=CostMetrics(input_cost=0.20, output_cost=0.0),
        success=True,
    )
    example_metric.cost.total_cost = 0.20
    row = SimpleNamespace(
        metrics={
            "accuracy": 1.0,
            "input_cost": 0.20,
            "output_cost": 0.0,
            "total_cost": 0.20,
            "cost": 0.20,
        }
    )
    example_metric.custom_metrics.update(
        {"input_cost": 0.20, "output_cost": 0.0, "total_cost": 0.20}
    )
    judge = ExampleMetrics(
        tokens=TokenMetrics(
            input_tokens=judge_tokens, output_tokens=0, total_tokens=judge_tokens
        ),
        response=ResponseMetrics(response_time_ms=1),
        cost=CostMetrics(unpriced=True),
        success=True,
    )
    judge.measured = True
    monkeypatch.setattr(local_module, "get_all_captured_responses", lambda: [object()])
    monkeypatch.setattr(local_module, "clear_captured_responses", lambda: None)
    monkeypatch.setattr(local_module, "extract_llm_metrics", lambda **k: judge)

    evaluator._fold_metric_function_llm_cost(example_metric, example_result=row)

    for metrics in (row.metrics, example_metric.custom_metrics):
        for key in ("cost", "total_cost", "input_cost", "output_cost"):
            assert key not in metrics, key
    assert row.metrics.get("cost_unmeasured") == 1.0
    floor = row.metrics.get("known_cost", row.metrics.get("cost_lower_bound"))
    assert floor == pytest.approx(0.20)
