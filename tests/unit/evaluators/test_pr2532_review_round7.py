"""Round-7 sol findings on merged PR 2532 (owner ruling A, tracked spend).

Every cost number follows the tracker when ``tracked_spend`` is present:
trial totals, token-less captured charges and progress payloads copied at
call time.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.api.types import TrialStatus
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.core.execution_budget import ExecutionBudget
from traigent.core.mandatory_metrics import MandatoryMetricsCollector
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
from tests.unit.evaluators.test_pr2532_review_round5 import (
    _custom_cost_functions,
    _two_examples,
)

_FLOOR = 0.0000135


def _no_custom(metrics: dict, label: str, allowed: float | None = None) -> None:
    for key, custom in (("cost", 100.0), ("total_cost", 200.0)):
        value = metrics.get(key)
        assert value != custom, f"{label}: {key}={value}"
        assert value is None or value == pytest.approx(
            _FLOOR if allowed is None else allowed
        ), f"{label}: {key}={value}"


# 1 -- trial totals follow tracked spend ---------------------------------------
def _trial(metrics: dict, tracked: Any) -> SimpleNamespace:
    return SimpleNamespace(
        trial_id="t1",
        status=TrialStatus.COMPLETED,
        duration=1.0,
        metrics=metrics,
        metadata={},
        tracked_spend=tracked,
    )


def test_totals_use_tracker_cost_not_custom_total_cost() -> None:
    collector = MandatoryMetricsCollector()
    collector.accumulate(
        _trial(
            {"total_cost": 200.0, "cost": 200.0},
            {"unmeasured": False, "cost": _FLOOR, "lower_bound": 0.0},
        )
    )
    assert collector.totals().total_cost == pytest.approx(_FLOOR)


def test_totals_unknown_tracker_uses_floor_only() -> None:
    collector = MandatoryMetricsCollector()
    collector.accumulate(
        _trial(
            {"total_cost": 200.0},
            {"unmeasured": True, "cost": None, "lower_bound": 0.2},
        )
    )
    assert collector.totals().total_cost == pytest.approx(0.2)


def test_totals_without_tracker_keep_reported_cost() -> None:
    collector = MandatoryMetricsCollector()
    collector.accumulate(_trial({"total_cost": 3.0}, None))
    assert collector.totals().total_cost == pytest.approx(3.0)


async def test_local_detailed_result_total_cost_is_tracker_value() -> None:
    evaluator = LocalEvaluator(
        metrics=["accuracy", "cost", "total_cost"],
        detailed=True,
        metric_functions={
            "cost": lambda *a, **k: 200.0,
            "total_cost": lambda *a, **k: 200.0,
        },
    )

    def func(q: str) -> int:
        capture_langchain_response(_completion("gpt-4o-mini", usage=True))
        return 42

    result = await evaluator.evaluate(func, {"model": "gpt-4o-mini"}, _two_examples())
    assert result.tracked_spend["unmeasured"] is False
    collector = MandatoryMetricsCollector()
    collector.accumulate(
        _trial(dict(result.aggregated_metrics), dict(result.tracked_spend))
    )
    assert collector.totals().total_cost == pytest.approx(result.tracked_spend["cost"])
    assert collector.totals().total_cost < 1.0


# 2 -- token-less captured charge keeps cleanup on --------------------------------
def _tokenless_charge_func(charge: float = 0.2):
    seen = {"n": 0}

    def func(q: str) -> int:
        seen["n"] += 1
        if seen["n"] == 1:
            response = _completion("gpt-4o-mini", usage=True)
            response.usage.prompt_tokens = 0
            response.usage.completion_tokens = 0
            response.usage.total_tokens = 0
            object.__setattr__(response.usage, "cost", charge)
            capture_langchain_response(response)
        return 42

    return func


async def test_tokenless_captured_charge_drops_custom_cost() -> None:
    result = await LocalEvaluator(
        metrics=["accuracy", "cost", "total_cost"],
        detailed=True,
        metric_functions=_custom_cost_functions(),
    ).evaluate(_tokenless_charge_func(), {"model": "gpt-4o-mini"}, _two_examples())
    assert result.tracked_spend["unmeasured"] is True
    assert result.tracked_spend["lower_bound"] == pytest.approx(0.2)
    for label, metrics in (
        ("aggregate", result.aggregated_metrics),
        *((f"row{n}", r.metrics) for n, r in enumerate(result.example_results)),
    ):
        _no_custom(
            metrics, label, allowed=0.2
        )  # 3 -- local progress payload for a partly priced row


def test_local_progress_copy_drops_custom_cost_for_partly_priced_row() -> None:
    evaluator = LocalEvaluator(
        detailed=True,
        metrics=["accuracy", "cost", "total_cost"],
        metric_functions=_custom_cost_functions(),
    )
    metric = ExampleMetrics(
        tokens=TokenMetrics(input_tokens=20, output_tokens=40, total_tokens=60),
        response=ResponseMetrics(response_time_ms=10),
        cost=CostMetrics(input_cost=_FLOOR, output_cost=0.0, unpriced=True),
        success=True,
    )
    metric.custom_metrics.update({"cost": 100.0, "total_cost": 200.0})
    row = SimpleNamespace(metrics={"accuracy": 1.0, "cost": 100.0, "total_cost": 200.0})
    payload = evaluator._build_local_progress_payload(metric, row, 42)
    seen = copy.deepcopy(payload["metrics"])
    assert seen.get("cost_unmeasured") == 1.0
    for key in ("cost", "total_cost", "input_cost", "output_cost"):
        assert key not in seen, f"{key}={seen.get(key)}"


# 4 -- wrapper progress payload ----------------------------------------------------
async def test_wrapper_progress_copy_drops_custom_cost_for_partly_priced_row(
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
    partly = ExampleMetrics(
        tokens=TokenMetrics(input_tokens=10, output_tokens=5, total_tokens=15),
        response=ResponseMetrics(response_time_ms=10),
        cost=CostMetrics(input_cost=_FLOOR, output_cost=0.0, unpriced=True),
        success=True,
    )
    calls = iter([paid, partly])
    monkeypatch.setattr(
        evaluator, "_capture_llm_metrics_for_example", lambda *a, **k: next(calls)
    )
    copies: list[dict[str, Any]] = []
    await evaluator.evaluate(
        lambda q: 42,
        {"model": "gpt-4o"},
        _two_examples(),
        budget=ExecutionBudget(max_cost_usd=100.0),
        progress_callback=lambda i, payload: copies.append(copy.deepcopy(payload)),
    )
    assert copies
    unknown = [p["metrics"] for p in copies if p["metrics"].get("cost_unmeasured")]
    assert unknown, "scenario produced no unknown-cost progress row"
    for n, metrics in enumerate(unknown):
        for key in ("cost", "total_cost", "input_cost", "output_cost"):
            assert key not in metrics, f"progress{n}: {key}={metrics.get(key)}"


async def test_tokenless_explicit_zero_charge_drops_custom_cost() -> None:
    result = await LocalEvaluator(
        metrics=["accuracy", "cost", "total_cost"],
        detailed=True,
        metric_functions=_custom_cost_functions(),
    ).evaluate(_tokenless_charge_func(0.0), {"model": "gpt-4o-mini"}, _two_examples())
    assert result.tracked_spend["unmeasured"] is True
    for label, metrics in (
        ("aggregate", result.aggregated_metrics),
        *((f"row{n}", r.metrics) for n, r in enumerate(result.example_results)),
    ):
        for key, custom in (("cost", 100.0), ("total_cost", 200.0)):
            assert metrics.get(key) != custom, f"{label}: {key}"
