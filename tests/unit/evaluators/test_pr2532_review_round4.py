"""Round-4 sol findings on merged PR 2532: mixed capture follows tracked spend."""

from __future__ import annotations

import pytest

from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.core.execution_budget import ExecutionBudget
from traigent.evaluators.base import (
    Dataset,
    EvaluationExample,
    SimpleScoringEvaluator,
)
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    ResponseMetrics,
    TokenMetrics,
)
from traigent.utils.langchain_interceptor import capture_langchain_response

from tests.unit.evaluators.test_pr2532_review_findings import _completion
from tests.unit.evaluators.test_pr2532_review_round3 import _example_result

_FLOOR = 0.0000135


def _two_examples() -> Dataset:
    return Dataset(
        [EvaluationExample({"q": "a"}, 42), EvaluationExample({"q": "b"}, 42)]
    )


def _assert_public_metrics_unknown(metrics: dict, tracked: dict) -> None:
    assert tracked["unmeasured"] is True
    assert tracked["lower_bound"] == pytest.approx(_FLOOR)
    assert metrics.get("cost_unmeasured") == 1.0
    assert metrics.get("cost") is None
    assert metrics.get("total_cost") is None
    assert metrics.get("cost_lower_bound") == pytest.approx(_FLOOR)


async def test_wrapper_mixed_capture_custom_zero_cost_is_unknown(monkeypatch) -> None:
    async def custom(func, config, example):
        return _example_result(
            example, func(**example.input_data), {"accuracy": 1.0, "cost": 0.0}
        )

    evaluator = CustomEvaluatorWrapper(custom, metrics=["accuracy", "cost"])
    paid = ExampleMetrics(
        tokens=TokenMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        response=ResponseMetrics(response_time_ms=10),
        cost=CostMetrics(input_cost=_FLOOR, output_cost=0.0),
        success=True,
    )
    calls = iter([paid, None])
    monkeypatch.setattr(
        evaluator, "_capture_llm_metrics_for_example", lambda *a, **k: next(calls)
    )
    budget = ExecutionBudget(max_cost_usd=100.0)
    result = await evaluator.evaluate(
        lambda q: 42, {"model": "gpt-4o"}, _two_examples(), budget=budget
    )
    _assert_public_metrics_unknown(result.aggregated_metrics, result.tracked_spend)
    assert result.execution_budget["cost_tracking"] == "untracked"


async def test_simple_scoring_mixed_capture_custom_zero_cost_is_unknown() -> None:
    seen = {"n": 0}

    def func(q: str) -> int:
        seen["n"] += 1
        if seen["n"] == 1:
            capture_langchain_response(_completion("gpt-4o-mini", usage=True))
        return 42

    evaluator = SimpleScoringEvaluator(metric_functions={"cost": lambda *a, **k: 0.0})
    budget = ExecutionBudget(max_cost_usd=100.0)
    result = await evaluator.evaluate(
        func, {"model": "gpt-4o-mini"}, _two_examples(), budget=budget
    )
    assert result.tracked_spend["unmeasured"] is True
    assert result.aggregated_metrics.get("cost_unmeasured") == 1.0
    assert result.aggregated_metrics.get("cost") is None
    assert result.aggregated_metrics.get("total_cost") is None
    assert result.execution_budget["cost_tracking"] == "untracked"
