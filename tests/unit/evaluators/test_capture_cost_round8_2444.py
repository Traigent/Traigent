"""Final row ownership and budget debits must use final cost evidence."""

import asyncio
import copy
import threading
from typing import Any

import pytest

from tests.unit.evaluators.test_capture_cost_round6_2444 import _timeout_after_event
from tests.unit.evaluators.test_local_capture_ownership_2444 import (
    MODEL,
    _Resp,
    _cost,
    _dataset,
)
from traigent.api.types import ExampleResult
from traigent.core.execution_budget import ExecutionBudget
from traigent.evaluators.base import BaseEvaluator
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
)
from traigent.utils.langchain_interceptor import capture_langchain_response

SDK_COST_FIELDS = {"cost", "input_cost", "output_cost", "total_cost", "evaluation_cost"}


def test_out_of_order_cached_response_final_ownership_replaces_preview() -> None:
    shared = _Resp(1000, 500)
    first_started = asyncio.Event()
    second_previewed = asyncio.Event()
    payloads: list[tuple[int, Any]] = []

    async def agent(text: str, **config: Any) -> Any:
        if text == "0":
            first_started.set()
            await asyncio.wait_for(second_previewed.wait(), timeout=5)
        else:
            await asyncio.wait_for(first_started.wait(), timeout=5)
        return {"text": "ok", "raw_response": shared}

    def progress(index: int, payload: Any) -> None:
        payloads.append((index, copy.deepcopy(payload)))
        if index == 1:
            second_previewed.set()

    result = asyncio.run(
        LocalEvaluator(
            metrics=["accuracy"], detailed=True, execution_mode="local", max_workers=2
        ).evaluate(agent, {}, _dataset(2), progress_callback=progress)
    )
    assert [i for i, _ in payloads] == [1, 0, 0, 1]
    rows = [row.metrics for row in result.example_results]
    for metrics in rows + [p["metrics"] for _, p in payloads]:
        if metrics.get("cost_unmeasured"):
            assert all(metrics.get(key) is None for key in SDK_COST_FIELDS), metrics
    # Preview ownership is tentative; dataset-order final accounting wins.
    for lane in (rows, [p["metrics"] for _, p in payloads[2:]]):
        assert lane[0]["total_cost"] == pytest.approx(_cost(shared))
        assert not lane[0].get("cost_unmeasured")
        assert "cost_lower_bound" not in lane[0]
        assert lane[1]["cost_unmeasured"] == 1.0
        assert lane[1].get("total_cost") is None
        assert "cost_lower_bound" not in lane[1]
    for lane in (
        rows,
        [p["metrics"] for _, p in payloads[:2]],
        [p["metrics"] for _, p in payloads[2:]],
    ):
        assert sum(
            m.get("total_cost") or m.get("cost_lower_bound", 0) for m in lane
        ) == pytest.approx(_cost(shared))
    assert result.aggregated_metrics["cost_lower_bound"] == pytest.approx(_cost(shared))


@pytest.mark.parametrize("measured", [True, False])
def test_final_row_discards_preview_cost_flags_and_floor(
    measured: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = ExampleResult(
        example_id="example_0",
        execution_time=0.0,
        success=True,
        input_data={},
        expected_output=None,
        actual_output="ok",
        metrics={
            "cost": 9.0,
            "total_cost": 9.0,
            "cost_unmeasured": 1.0,
            "cost_unpriced": 1.0,
            "cost_lower_bound": 9.0,
            "evaluation_cost": 77.0,
            "business_cost": 88.0,
        },
    )
    final = ExampleMetrics(cost=CostMetrics(input_cost=0.1, unmeasured=not measured))
    if not measured:
        final.cost.known_cost = 0.2
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=True)
    # Model an SDK preview write explicitly: pre-existing evaluator metrics
    # are authoritative and must not be mistaken for preview-owned evidence.
    preview = {
        key: row.metrics.pop(key)
        for key in tuple(row.metrics)
        if key not in {"evaluation_cost", "business_cost"}
    }

    def write_preview(*args: Any, **kwargs: Any) -> Any:
        row.metrics.update(preview)
        return {}, None

    monkeypatch.setattr(BaseEvaluator, "_extract_response_metrics", write_preview)
    evaluator._extract_response_metrics("example_0", None, row, {})
    evaluator._update_example_metric_from_result(final, row, 0)
    evaluator._add_standard_metrics_to_custom(final)
    evaluator._enforce_row_cost_invariant(final, row)
    payload = evaluator._build_local_progress_payload(final, row, "ok")
    for metrics in (row.metrics, final.custom_metrics, payload["metrics"]):
        assert metrics["evaluation_cost"] == 77.0
        assert metrics["business_cost"] == 88.0
        assert not metrics.get("cost_unpriced")
        if measured:
            assert not metrics.get("cost_unmeasured")
            assert "cost_lower_bound" not in metrics
            assert metrics["total_cost"] == pytest.approx(0.1)
            if "cost" in metrics:
                assert metrics["cost"] == pytest.approx(0.1)
        else:
            assert metrics["cost_unmeasured"] == 1.0
            assert metrics["cost_lower_bound"] == pytest.approx(0.2)
            assert all(
                metrics.get(key) is None
                for key in SDK_COST_FIELDS - {"evaluation_cost"}
            )


def test_late_summary_spend_exhausts_direct_execution_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    captured = threading.Event()
    sibling_captured = threading.Event()
    _timeout_after_event(monkeypatch, sibling_captured)
    measured = _Resp(2000, 700)
    late = _Resp(9000, 3000)
    earlier_floor = _cost(measured)
    final_floor = earlier_floor + _cost(late)
    budget = ExecutionBudget(max_cost_usd=(earlier_floor + final_floor) / 2)
    original_summary = MetricsTracker.format_as_summary_stats

    def agent(text: str, **config: Any) -> str:
        if text == "0":
            assert release.wait(5)
            capture_langchain_response(late)
            captured.set()
        else:
            capture_langchain_response(measured)
            sibling_captured.set()
        return "ok"

    def summary(tracker: MetricsTracker, **kwargs: Any) -> Any:
        release.set()
        assert captured.wait(5)
        return original_summary(tracker, **kwargs)

    monkeypatch.setattr(MetricsTracker, "format_as_summary_stats", summary)
    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        detailed=True,
        max_workers=2,
        timeout=10.0,
        execution_mode="local",
    )
    try:
        result = asyncio.run(
            evaluator.evaluate(agent, {"model": MODEL}, _dataset(2), budget=budget)
        )
        assert "timed out" in result.errors[0]
        for metrics in (result.aggregated_metrics, result.summary_stats["metrics"]):
            assert metrics["cost_unmeasured"] == 1.0
            assert metrics.get("cost") is None
            assert metrics["cost_lower_bound"] == pytest.approx(final_floor)
        snapshot = budget.snapshot()
        assert snapshot.consumed_cost == pytest.approx(final_floor)
        assert snapshot.remaining_cost == 0
        assert snapshot.exhausted_dimension == "cost"
        assert snapshot.trials == 1
        assert snapshot.untracked_trials == 1
        assert snapshot.consumed_examples == 2
        assert result.execution_budget["consumed_cost"] == pytest.approx(final_floor)
        assert result.execution_budget["remaining_cost"] == 0
        assert result.execution_budget["exhausted_dimension"] == "cost"
        assert result.execution_budget["trials"] == 1
        assert result.execution_budget_exhausted
        assert result.stop_reason == "execution_budget"
        blocked = asyncio.run(evaluator.evaluate(agent, {}, _dataset(1), budget=budget))
        assert blocked.execution_budget_exhausted
        assert blocked.total_examples == 0
        assert budget.snapshot().trials == 1
    finally:
        release.set()
        assert captured.wait(5)
