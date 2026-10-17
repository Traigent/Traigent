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
    )
    assert BaseEvaluator._execution_budget_cost(result) == pytest.approx(0.50)


# P1-2 / P2-6 ---------------------------------------------------------------
def test_partially_priced_row_is_flagged_incomplete_with_a_floor() -> None:
    tracker = MetricsTracker()
    tracker.example_metrics.append(_partial_row(0.20))
    metrics = tracker.format_for_backend()

    assert metrics["cost_unpriced"] == 1.0  # flagged incomplete
    assert metrics["cost_lower_bound"] == pytest.approx(0.20)  # the floor


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
