"""Regression tests for the static review of merged PR 2532 (cost honesty).

Each test reproduces one reviewed finding. Semantics kept: unavailable cost
stays unavailable (never $0); a measured/custom zero stays zero.
"""

from __future__ import annotations

import pytest
from openai.types.chat import ChatCompletion

from traigent.core.execution_budget import ExecutionBudget
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
    SimpleScoringEvaluator,
)
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    ResponseMetrics,
    TokenMetrics,
)
from traigent.utils.cost_calculator import cost_from_tokens
from traigent.utils.langchain_interceptor import capture_langchain_response


def _completion(model: str, usage: bool) -> ChatCompletion:
    payload = {
        "id": "test",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }
    if usage:
        payload["usage"] = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
    return ChatCompletion.model_validate(payload)


def _partial_row(priced: float = 0.20) -> ExampleMetrics:
    """One example: ``priced`` dollars priced plus unknown-price usage."""
    return ExampleMetrics(
        tokens=TokenMetrics(input_tokens=100, output_tokens=50, total_tokens=150),
        response=ResponseMetrics(response_time_ms=10),
        cost=CostMetrics(input_cost=priced, output_cost=0.0, unpriced=True),
        success=True,
    )


def _unknown_row() -> ExampleMetrics:
    return ExampleMetrics(
        response=ResponseMetrics(response_time_ms=10),
        cost=CostMetrics(unmeasured=True),
        success=True,
    )


# P1-1 ----------------------------------------------------------------------
def test_custom_cost_metric_does_not_hide_measured_spend_from_budget() -> None:
    tracker = MetricsTracker()
    tracker.example_metrics.append(
        ExampleMetrics(
            tokens=TokenMetrics(input_tokens=10, output_tokens=20, total_tokens=30),
            response=ResponseMetrics(response_time_ms=10),
            cost=CostMetrics(input_cost=0.30, output_cost=0.20),
            success=True,
        )
    )
    comprehensive = tracker.format_for_backend()
    assert comprehensive["cost"] == pytest.approx(0.50)

    evaluator = LocalEvaluator(
        metrics=["cost"], metric_functions={"cost": lambda *a, **k: 0.0}
    )
    aggregated: dict = {"cost": 0.0}  # the user's custom score
    evaluator._merge_comprehensive_metrics(
        aggregated, comprehensive, preserve_authoritative_accuracy=True
    )

    assert aggregated["cost"] == 0.0  # custom score preserved
    result = EvaluationResult(
        config={},
        aggregated_metrics=aggregated,
        total_examples=1,
        successful_examples=1,
        duration=1.0,
        tracked_spend=BaseEvaluator._tracked_spend(comprehensive, "cost"),
    )
    assert BaseEvaluator._execution_budget_cost(result) == pytest.approx(0.50)


# P1-2 / P2-6 ---------------------------------------------------------------
def test_partially_priced_row_is_flagged_incomplete_with_a_floor() -> None:
    tracker = MetricsTracker()
    tracker.example_metrics.append(_partial_row(0.20))
    metrics = tracker.format_for_backend()

    assert metrics["cost_unpriced"] == 1.0  # flagged incomplete
    assert metrics["cost_lower_bound"] == pytest.approx(0.20)  # the floor
    # owner ruling A, partly priced cost is unknown: no cost value, only a floor
    assert metrics["cost"] is None
    assert metrics["cost_unmeasured"] == 1.0


def test_partially_priced_row_is_not_budget_complete() -> None:
    tracker = MetricsTracker()
    tracker.example_metrics.append(_partial_row(0.20))
    result = EvaluationResult(
        config={},
        aggregated_metrics=tracker.format_for_backend(),
        total_examples=1,
        successful_examples=1,
        duration=1.0,
    )
    budget = ExecutionBudget(max_cost_usd=10.0)
    BaseEvaluator._finalize_execution_budget_evaluation(budget, result, None)
    assert budget.snapshot().as_dict()["consumed_cost"] == pytest.approx(0.20)
    assert budget._untracked_trials == 1  # a floor, not an exact total


def test_measured_zero_cost_budget_debit_stays_tracked() -> None:
    result = EvaluationResult(
        config={},
        aggregated_metrics={"cost": 0.0, "cost_unpriced": 0.0},
        total_examples=1,
        successful_examples=1,
        duration=1.0,
    )
    budget = ExecutionBudget(max_cost_usd=10.0)
    BaseEvaluator._finalize_execution_budget_evaluation(budget, result, None)
    assert budget._untracked_trials == 0


def test_summary_lower_bound_does_not_double_count_partial_row() -> None:
    tracker = MetricsTracker()
    tracker.example_metrics.append(_partial_row(0.20))
    tracker.example_metrics.append(_unknown_row())
    summary = tracker.format_as_summary_stats()["metrics"]
    assert summary["cost_unmeasured"] == 1.0
    assert summary["cost_lower_bound"] == pytest.approx(0.20)


# P1-3 ----------------------------------------------------------------------
async def test_simple_scoring_unmeasured_first_call_keeps_aggregate_unmeasured() -> (
    None
):
    paid = sum(cost_from_tokens(10, 20, "gpt-4o-mini"))

    def func(q: str) -> str:
        capture_langchain_response(_completion("gpt-4o-mini", usage=False))
        capture_langchain_response(_completion("gpt-4o-mini", usage=True))
        return 42  # non-text output: no answer text to price the first call

    evaluator = SimpleScoringEvaluator(metrics=["accuracy"])
    result = await evaluator.evaluate(
        func,
        {"model": "gpt-4o-mini"},
        Dataset([EvaluationExample({"q": "a"}, 42)]),
    )
    agg = result.aggregated_metrics
    assert agg.get("cost_unmeasured") == 1.0
    assert not agg.get("total_cost")  # never the paid-only total as complete
    row = result.example_results[0].metrics
    assert row.get("total_cost") is None
    assert row.get("cost_unmeasured") == 1.0
    assert row.get("known_cost") == pytest.approx(paid)


@pytest.mark.parametrize("detailed", [False, True])
async def test_local_unmeasured_first_call_keeps_aggregate_unmeasured(
    detailed: bool,
) -> None:
    paid = sum(cost_from_tokens(10, 20, "gpt-4o-mini"))

    def func(q: str) -> str:
        capture_langchain_response(_completion("gpt-4o-mini", usage=False))
        capture_langchain_response(_completion("gpt-4o-mini", usage=True))
        return 42  # non-text output: no answer text to price the first call

    result = await LocalEvaluator(
        metrics=["accuracy", "cost"], detailed=detailed
    ).evaluate(
        func,
        {"model": "gpt-4o-mini"},
        Dataset([EvaluationExample({"q": "a"}, 42)]),
    )
    agg = result.aggregated_metrics
    assert agg.get("cost_unmeasured") == 1.0
    assert agg.get("cost") is None
    assert agg.get("cost_lower_bound") == pytest.approx(paid)


# Owner ruling A: a partly priced trial is UNKNOWN cost -----------------------
def _trial_from_tracker(trial_id: str, tracker: MetricsTracker, **extra: float):
    from datetime import UTC, datetime

    from traigent.api.types import TrialResult, TrialStatus

    keys = ("cost", "cost_unmeasured", "cost_lower_bound", "cost_unpriced")
    formatted = tracker.format_for_backend()
    metrics = {k: formatted[k] for k in keys if formatted.get(k) is not None}
    metrics.update(extra)
    return TrialResult(
        trial_id=trial_id,
        config={"id": trial_id},
        metrics=metrics,
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime.now(UTC),
    )


def _measured_tracker(cost: float) -> MetricsTracker:
    tracker = MetricsTracker()
    tracker.example_metrics.append(
        ExampleMetrics(
            tokens=TokenMetrics(input_tokens=10, output_tokens=20, total_tokens=30),
            response=ResponseMetrics(response_time_ms=10),
            cost=CostMetrics(input_cost=cost, output_cost=0.0),
            success=True,
        )
    )
    return tracker


def test_partly_priced_trial_does_not_beat_fully_measured_on_cost_objective() -> None:
    from traigent.core.result_selection import select_best_configuration

    partial_tracker = MetricsTracker()
    partial_tracker.example_metrics.append(_partial_row(0.20))
    partial = _trial_from_tracker("partial", partial_tracker)
    measured = _trial_from_tracker("measured", _measured_tracker(0.30))
    assert "cost" not in partial.metrics  # unknown, not $0.20

    res = select_best_configuration(
        [partial, measured],
        primary_objective="cost",
        config_space_keys=["id"],
        aggregate_configs=False,
        comparability_mode="legacy",
    )
    assert res.best_trial_id == "measured"


async def test_orchestrator_debit_for_partly_priced_trial_is_lower_bound_only() -> None:
    from types import SimpleNamespace

    from traigent.core.cost_estimator import CostEstimator
    from traigent.core.orchestrator import OptimizationOrchestrator

    tracker = MetricsTracker()
    tracker.example_metrics.append(_partial_row(0.20))
    trial = _trial_from_tracker("partial", tracker)
    budget = ExecutionBudget(max_cost_usd=100.0)

    async def _noop(**_kwargs) -> None:
        return None

    orch = SimpleNamespace(
        cost_enforcer=SimpleNamespace(track_cost_async=_noop),
        execution_budget=budget,
        _extract_trial_cost=CostEstimator.extract_trial_cost,
        _examples_of=lambda t: 1,
    )
    await OptimizationOrchestrator._account_trial_spend(orch, trial)
    snap = budget.snapshot()
    assert snap.consumed_cost == pytest.approx(0.20)  # the floor
    assert snap.untracked_trials == 1  # not an exact total


def test_summary_stats_for_single_partly_priced_row_are_not_complete() -> None:
    tracker = MetricsTracker()
    tracker.example_metrics.append(_partial_row(0.20))
    summary = tracker.format_as_summary_stats()["metrics"]
    assert summary["cost_unmeasured"] == 1.0
    assert summary["cost_lower_bound"] == pytest.approx(0.20)
    assert summary["total_cost"] is None  # no numeric statistics as complete
    assert summary.get("cost_unpriced") in (None, 1.0)


# Custom scores never feed the budget (P1 review) -----------------------------
def _two_calls_paid_only(unmeasured_first: bool):
    def func(q: str) -> str:
        if unmeasured_first:
            capture_langchain_response(_completion("gpt-4o-mini", usage=False))
        capture_langchain_response(_completion("gpt-4o-mini", usage=True))
        return 42

    return func


@pytest.mark.parametrize("evaluator_cls", [LocalEvaluator, SimpleScoringEvaluator])
async def test_custom_zero_cost_scores_do_not_hide_tracked_spend(evaluator_cls) -> None:
    paid = sum(cost_from_tokens(10, 20, "gpt-4o-mini"))
    budget = ExecutionBudget(max_cost_usd=10.0)
    evaluator = evaluator_cls(
        metric_functions={
            "cost": lambda *a, **k: 0.0,
            "total_cost": lambda *a, **k: 0.0,
        }
    )
    result = await evaluator.evaluate(
        _two_calls_paid_only(False),
        {"model": "gpt-4o-mini"},
        Dataset([EvaluationExample({"q": "a"}, 42)]),
        budget=budget,
    )
    assert result.aggregated_metrics["cost"] == 0.0  # custom score preserved
    assert budget.consumed_cost == pytest.approx(paid)
    assert result.execution_budget["cost_tracking"] == "complete"


@pytest.mark.parametrize("evaluator_cls", [LocalEvaluator, SimpleScoringEvaluator])
async def test_custom_cost_score_never_inflates_unmeasured_floor(evaluator_cls) -> None:
    paid = sum(cost_from_tokens(10, 20, "gpt-4o-mini"))
    budget = ExecutionBudget(max_cost_usd=1000.0)
    evaluator = evaluator_cls(metric_functions={"cost": lambda *a, **k: 100.0})
    result = await evaluator.evaluate(
        _two_calls_paid_only(True),
        {"model": "gpt-4o-mini"},
        Dataset([EvaluationExample({"q": "a"}, 42)]),
        budget=budget,
    )
    assert result.aggregated_metrics.get("cost_unmeasured") == 1.0
    assert budget.consumed_cost == pytest.approx(paid)  # never the custom 100
    assert result.execution_budget["cost_tracking"] == "untracked"


def test_custom_cost_score_never_inflates_floor_after_merge() -> None:
    """Unmeasured trial, $0.50 floor, custom ``cost`` score 100: debit $0.50."""
    tracker = MetricsTracker()
    tracker.example_metrics.append(_unknown_row())
    tracker.example_metrics[0].cost.known_cost = 0.50
    comprehensive = tracker.format_for_backend()
    assert comprehensive["cost_unmeasured"] == 1.0
    assert comprehensive["cost_lower_bound"] == pytest.approx(0.50)

    evaluator = LocalEvaluator(
        metrics=["cost"], metric_functions={"cost": lambda *a, **k: 100.0}
    )
    aggregated: dict = {"cost": 100.0}  # the user's custom score
    evaluator._merge_comprehensive_metrics(
        aggregated, comprehensive, preserve_authoritative_accuracy=True
    )
    assert aggregated["cost"] == 100.0  # custom score preserved
    result = EvaluationResult(
        config={},
        aggregated_metrics=aggregated,
        total_examples=1,
        successful_examples=1,
        duration=1.0,
        tracked_spend=BaseEvaluator._tracked_spend(comprehensive, "cost"),
    )
    assert BaseEvaluator._execution_budget_cost(result) == pytest.approx(0.50)
    budget = ExecutionBudget(max_cost_usd=1000.0)
    BaseEvaluator._finalize_execution_budget_evaluation(budget, result, None)
    assert budget.consumed_cost == pytest.approx(0.50)
    assert budget._untracked_trials == 1


# Round-2 review findings -----------------------------------------------------
def _trial_via_factory(eval_result):
    from traigent.core.trial_result_factory import build_success_result

    return build_success_result(
        trial_id="t1",
        evaluation_config={},
        eval_result=eval_result,
        duration=1.0,
        examples_attempted=1,
        total_cost=None,
        optuna_trial_id=None,
    )


def _result_with_tracker(metrics: dict, tracker_agg: dict) -> EvaluationResult:
    return EvaluationResult(
        config={},
        aggregated_metrics=metrics,
        total_examples=1,
        successful_examples=1,
        duration=1.0,
        tracked_spend=BaseEvaluator._tracked_spend(tracker_agg, "total_cost"),
    )


def test_orchestrator_debit_ignores_custom_zero_cost_scores() -> None:
    from traigent.core.cost_estimator import CostEstimator

    result = _result_with_tracker(
        {"cost": 0.0, "total_cost": 0.0},
        {"cost": 0.0000135, "total_cost": 0.0000135},
    )
    trial = _trial_via_factory(result)
    assert CostEstimator.extract_trial_cost(trial) == pytest.approx(0.0000135)


def test_orchestrator_debit_ignores_custom_cost_on_unknown_usage() -> None:
    from traigent.core.cost_estimator import CostEstimator

    result = _result_with_tracker(
        {"cost": 100.0, "cost_unmeasured": 1.0},
        {"cost_unmeasured": 1.0, "cost_lower_bound": 0.0000135},
    )
    trial = _trial_via_factory(result)
    assert CostEstimator.extract_trial_cost(trial) == pytest.approx(0.0000135)


async def test_capture_off_custom_cost_never_debits_budget() -> None:
    budget = ExecutionBudget(max_cost_usd=1000.0)
    evaluator = SimpleScoringEvaluator(
        capture_llm_metrics=False,
        metric_functions={
            "cost": lambda *a, **k: 100.0,
            "total_cost": lambda *a, **k: 100.0,
        },
    )
    result = await evaluator.evaluate(
        lambda q: 42,
        {},
        Dataset([EvaluationExample({"q": "a"}, 42)]),
        budget=budget,
    )
    assert budget.consumed_cost != pytest.approx(100.0)
    assert result.execution_budget["cost_tracking"] != "complete"


def test_custom_zero_cost_preserved_in_trial_metrics() -> None:
    result = _result_with_tracker(
        {"cost": 0.0, "total_cost": 0.0000135},
        {"cost": 0.0000135, "total_cost": 0.0000135},
    )
    trial = _trial_via_factory(result)
    assert trial.metrics["cost"] == 0.0
    assert trial.metrics["total_cost"] == pytest.approx(0.0000135)


def _partly_priced_metric() -> ExampleMetrics:
    return ExampleMetrics(
        tokens=TokenMetrics(input_tokens=20, output_tokens=40, total_tokens=60),
        response=ResponseMetrics(response_time_ms=10),
        cost=CostMetrics(input_cost=1.5e-06, output_cost=1.2e-05, unpriced=True),
        success=True,
    )


def test_partly_priced_row_withdraws_cost_in_detail_and_progress() -> None:
    from types import SimpleNamespace

    evaluator = LocalEvaluator(detailed=True, metrics=["accuracy"])
    metric = _partly_priced_metric()
    # The row already carries the priced subtotal under every cost key.
    row = SimpleNamespace(
        metrics={
            "accuracy": 1.0,
            "input_cost": 1.5e-06,
            "output_cost": 1.2e-05,
            "total_cost": 1.35e-05,
            "cost": 1.35e-05,
        }
    )
    evaluator._add_standard_metrics_to_custom(metric)
    evaluator._transfer_token_metrics_to_example_result(row, metric)
    payload = evaluator._build_local_progress_payload(metric, row, 42)

    for metrics in (row.metrics, payload["metrics"]):
        assert metrics.get("cost_unmeasured") == 1.0
        assert metrics.get("known_cost") == pytest.approx(1.35e-05)
        for key in ("cost", "total_cost", "input_cost", "output_cost"):
            assert key not in metrics, key
