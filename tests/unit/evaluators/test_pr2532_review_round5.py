"""Round-5 sol findings on merged PR 2532: unknown tracked cost drops custom keys.

Owner ruling A: when tracked spend is unknown, no finite cost/total_cost/
input_cost/output_cost survives in aggregates, returned rows or progress
payloads, including a user's custom cost metrics. Only ``cost_unmeasured`` and
``cost_lower_bound`` remain.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.core.execution_budget import ExecutionBudget
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

from tests.unit.evaluators.test_pr2532_review_findings import _completion
from tests.unit.evaluators.test_pr2532_review_round3 import _example_result

_FLOOR = 0.0000135
_COST_KEYS = ("cost", "total_cost", "input_cost", "output_cost")


def _two_examples() -> Dataset:
    return Dataset(
        [EvaluationExample({"q": "a"}, 42), EvaluationExample({"q": "b"}, 42)]
    )


def _custom_cost_functions() -> dict[str, Any]:
    return {
        "cost": lambda *a, **k: 100.0,
        "total_cost": lambda *a, **k: 200.0,
    }


def _mixed_func():
    seen = {"n": 0}

    def func(q: str) -> int:
        seen["n"] += 1
        if seen["n"] == 1:
            capture_langchain_response(_completion("gpt-4o-mini", usage=True))
        return 42

    return func


def _assert_no_cost_keys(metrics: dict, label: str) -> None:
    for key in _COST_KEYS:
        assert key not in metrics, f"{label}: {key}={metrics.get(key)}"
    assert metrics.get("cost_unmeasured") == 1.0, label


def _assert_no_custom_cost(metrics: dict, label: str) -> None:
    # A row's own tracker-measured cost may stay; the custom 100/200 may not.
    for key, custom in (("cost", 100.0), ("total_cost", 200.0)):
        assert metrics.get(key) != custom, f"{label}: {key}={custom}"
    for key in ("cost", "total_cost"):
        value = metrics.get(key)
        assert value is None or value == pytest.approx(_FLOOR), f"{label}: {key}"


def _assert_aggregate(agg: dict, tracked: dict) -> None:
    assert tracked["unmeasured"] is True
    assert tracked["cost"] is None
    _assert_no_cost_keys(agg, "aggregate")
    assert agg.get("cost_lower_bound") == pytest.approx(_FLOOR)


async def test_local_detailed_custom_cost_dropped_from_unknown_aggregate() -> None:
    result = await LocalEvaluator(
        metrics=["accuracy", "cost", "total_cost"],
        detailed=True,
        metric_functions=_custom_cost_functions(),
    ).evaluate(_mixed_func(), {"model": "gpt-4o-mini"}, _two_examples())
    _assert_aggregate(result.aggregated_metrics, result.tracked_spend)
    if result.metrics is not result.aggregated_metrics:
        _assert_aggregate(result.metrics, result.tracked_spend)


async def test_simple_scoring_custom_cost_dropped_from_rows_and_progress() -> None:
    progress: list[dict] = []
    evaluator = SimpleScoringEvaluator(
        metric_functions=_custom_cost_functions(), detailed=True
    )
    result = await evaluator.evaluate(
        _mixed_func(),
        {"model": "gpt-4o-mini"},
        _two_examples(),
        progress_callback=lambda i, payload: progress.append(payload),
    )
    _assert_aggregate(result.aggregated_metrics, result.tracked_spend)
    for n, row in enumerate(result.example_results):
        _assert_no_custom_cost(row.metrics, f"row{n}")
    for n, payload in enumerate(progress):
        _assert_no_custom_cost(payload.get("metrics") or {}, f"progress{n}")


async def test_wrapper_custom_cost_dropped_from_rows_and_progress(
    monkeypatch,
) -> None:
    async def custom(func, config, example):
        return _example_result(
            example,
            func(**example.input_data),
            {"accuracy": 1.0, "cost": 100.0, "total_cost": 200.0},
        )

    evaluator = CustomEvaluatorWrapper(
        custom, metrics=["accuracy", "cost", "total_cost"]
    )
    paid = ExampleMetrics(
        tokens=TokenMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        response=ResponseMetrics(response_time_ms=10),
        cost=CostMetrics(input_cost=_FLOOR, output_cost=0.0),
        success=True,
    )
    unpriced = ExampleMetrics(
        tokens=TokenMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        response=ResponseMetrics(response_time_ms=10),
        cost=CostMetrics(unpriced=True),
        success=True,
    )
    calls = iter([paid, unpriced])
    monkeypatch.setattr(
        evaluator, "_capture_llm_metrics_for_example", lambda *a, **k: next(calls)
    )
    progress: list[dict] = []
    result = await evaluator.evaluate(
        lambda q: 42,
        {"model": "gpt-4o"},
        _two_examples(),
        budget=ExecutionBudget(max_cost_usd=100.0),
        progress_callback=lambda i, payload: progress.append(payload),
    )
    _assert_aggregate(result.aggregated_metrics, result.tracked_spend)
    for n, row in enumerate(result.example_results):
        _assert_no_custom_cost(row.metrics, f"row{n}")
    for n, payload in enumerate(progress):
        _assert_no_custom_cost(payload.get("metrics") or {}, f"progress{n}")
