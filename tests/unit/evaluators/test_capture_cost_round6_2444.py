"""Regression probes for capture ownership and honest cost, review round 6."""

import asyncio
import copy
import threading
from typing import Any

import pytest

from tests.unit.evaluators.test_local_capture_ownership_2444 import (
    MODEL,
    _Resp,
    _cost,
    _dataset,
)
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
)
from traigent.utils.langchain_interceptor import (
    capture_langchain_response,
    get_all_captured_responses,
)


@pytest.mark.parametrize("metric_name", ["business_cost", "evaluation_cost"])
def test_user_cost_metric_survives_unknown_agent_and_priced_judge(
    metric_name: str,
) -> None:
    def agent(text: str, **config: Any) -> str:
        capture_langchain_response(_Resp(1000, 500, model=None))
        return "ok"

    def metric(output: Any, expected: Any, llm_metrics: Any = None) -> float:
        capture_langchain_response(_Resp(2000, 700))
        return 77.0

    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        metric_functions={metric_name: metric},
        detailed=True,
        execution_mode="local",
    )
    result = asyncio.run(evaluator.evaluate(agent, {}, _dataset(1)))
    row = result.example_results[0].metrics
    assert row["cost_unmeasured"] == 1.0
    assert row[metric_name] == 77.0
    assert result.aggregated_metrics[metric_name] == 77.0
    assert result.summary_stats["metrics"][metric_name]["mean"] == 77.0


@pytest.mark.parametrize("scenario", ["modelless", "response-model", "unknown-sibling"])
def test_early_progress_uses_all_owned_calls_and_response_models(
    scenario: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_STRICT_COST_ACCOUNTING", raising=False)
    priced = _Resp(2000, 700)
    calls = {
        "modelless": [_Resp(1000, 500, model=None)],
        "response-model": [priced],
        "unknown-sibling": [_Resp(1000, 500, model=None), priced],
    }[scenario]
    payloads: list[Any] = []
    inputs: list[Any] = []

    async def agent(text: str, **config: Any) -> str:
        for response in calls:
            capture_langchain_response(response)
        return "ok"

    def metric(output: Any, expected: Any, llm_metrics: Any = None) -> float:
        inputs.append(copy.deepcopy(llm_metrics))
        return 1.0

    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        metric_functions={"quality": metric},
        detailed=True,
        execution_mode="local",
    )
    result = asyncio.run(
        evaluator.evaluate(
            agent,
            {"model": "gpt-4o"} if scenario == "response-model" else {},
            _dataset(1),
            progress_callback=lambda i, p: payloads.append(copy.deepcopy(p)),
        )
    )
    assert len(payloads) >= 2  # early detailed progress AND final local progress
    assert len(inputs) >= 2  # progress-time metric function AND final metric pass
    for metrics in [p["metrics"] for p in payloads] + inputs:
        if scenario == "response-model":
            assert metrics["total_cost"] == pytest.approx(_cost(priced))
            assert not metrics.get("cost_unmeasured")
        else:
            assert metrics.get("total_cost") is None, metrics
            assert metrics["cost_unmeasured"] == 1.0
            assert metrics.get("cost_lower_bound", 0) == pytest.approx(
                _cost(priced) if scenario == "unknown-sibling" else 0.0
            )
    assert result.summary_stats is not None


def test_final_metric_input_carries_unknown_flag_and_exact_floor() -> None:
    priced = _Resp(2000, 700)
    inputs: list[Any] = []

    def agent(text: str, **config: Any) -> str:
        capture_langchain_response(_Resp(1000, 500, model=None))
        capture_langchain_response(priced)
        return "ok"

    def metric(output: Any, expected: Any, llm_metrics: Any = None) -> float:
        inputs.append(llm_metrics)
        return 1.0

    evaluator = LocalEvaluator(
        metrics=["accuracy"], metric_functions={"quality": metric}
    )
    asyncio.run(evaluator.evaluate(agent, {}, _dataset(1)))
    assert len(inputs) == 1
    assert inputs[0]["total_cost"] is None
    assert inputs[0]["cost_unmeasured"] == 1.0
    assert inputs[0]["cost_lower_bound"] == pytest.approx(_cost(priced))


def test_user_evaluation_cost_aggregate_excludes_sdk_samples_and_missing_values() -> (
    None
):
    def agent(text: str, **config: Any) -> Any:
        capture_langchain_response(
            _Resp(1000, 500, model=None if text == "0" else MODEL)
        )
        return ("ok", {"evaluation_cost": 77.0}) if text == "0" else "ok"

    def judge(output: Any, expected: Any, llm_metrics: Any = None) -> float:
        capture_langchain_response(_Resp(2000, 700))
        return 1.0

    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        metric_functions={"judge": judge},
        detailed=True,
        execution_mode="local",
    )
    result = asyncio.run(evaluator.evaluate(agent, {}, _dataset(2)))
    assert result.example_results[0].user_metrics["evaluation_cost"] == 77.0
    assert result.example_results[1].metrics["evaluation_cost"] == pytest.approx(
        _cost(_Resp(2000, 700))
    )
    assert result.aggregated_metrics["cost_unmeasured"] == 1.0
    assert result.aggregated_metrics["evaluation_cost"] == 77.0
    summary = result.summary_stats["metrics"]["evaluation_cost"]
    assert summary["mean"] == 77.0
    assert summary["count"] == 1


def _timeout_after_event(
    monkeypatch: pytest.MonkeyPatch, ready: threading.Event
) -> None:
    """Trigger the first worker's timeout only once its sibling has captured.

    Exercise the real timeout handling without a scheduling deadline for the
    ordinary example. The suite timeout remains the deadlock guard.
    """
    calls = 0

    async def wait_for(awaitable: Any, timeout: Any = None) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            while not ready.is_set():
                await asyncio.sleep(0)
            raise TimeoutError
        return await awaitable

    monkeypatch.setattr(asyncio, "wait_for", wait_for)


@pytest.mark.parametrize("during_summary", [False, True])
def test_timeout_plus_measured_example_is_unknown_and_scope_closes(
    during_summary: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    captured = threading.Event()
    sibling_captured = threading.Event()
    _timeout_after_event(monkeypatch, sibling_captured)
    measured = _Resp(2000, 700)
    late = _Resp(9000, 3000)
    original = MetricsTracker.format_as_summary_stats

    def agent(text: str, **config: Any) -> str:
        if text == "0":
            assert release.wait(5), "test must release the timed-out worker"
            capture_langchain_response(late)
            captured.set()
        else:
            capture_langchain_response(measured)
            sibling_captured.set()
        return "ok"

    def summary(tracker: MetricsTracker, **kwargs: Any) -> Any:
        release.set()
        assert captured.wait(5)
        # Sealed late captures are separate from normal per-example reads.
        assert get_all_captured_responses() == []
        return original(tracker, **kwargs)

    if during_summary:
        monkeypatch.setattr(MetricsTracker, "format_as_summary_stats", summary)
    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        detailed=True,
        max_workers=2,
        timeout=10.0,
        execution_mode="local",
    )
    try:
        result = asyncio.run(evaluator.evaluate(agent, {"model": MODEL}, _dataset(2)))
        assert "timed out" in result.errors[0]
        assert result.example_results[0].metrics["cost_unmeasured"] == 1.0
        assert result.example_results[1].metrics["total_cost"] == pytest.approx(
            _cost(measured)
        )
        assert result.aggregated_metrics["cost_unmeasured"] == 1.0
        assert result.aggregated_metrics.get("cost") is None
        expected_floor = _cost(measured) + (_cost(late) if during_summary else 0.0)
        assert result.aggregated_metrics["cost_lower_bound"] == pytest.approx(
            expected_floor
        )
        assert result.summary_stats["metrics"]["cost_unmeasured"] == 1.0
        assert result.summary_stats["metrics"]["cost_lower_bound"] == pytest.approx(
            expected_floor
        )
    finally:
        release.set()
        assert captured.wait(5)


def test_tracker_unmeasured_timeout_row_prevents_a_measured_trial() -> None:
    tracker = MetricsTracker()
    tracker.add_example_metrics(
        ExampleMetrics(measured=False, cost=CostMetrics(unmeasured=True))
    )
    tracker.add_example_metrics(ExampleMetrics(cost=CostMetrics(input_cost=0.1)))
    metrics = tracker.format_for_backend()
    assert metrics["cost_unmeasured"] == 1.0
    assert metrics.get("cost") is None
    assert metrics["cost_lower_bound"] == pytest.approx(0.1)
    summary = tracker.format_as_summary_stats()["metrics"]
    assert summary["cost_unmeasured"] == 1.0
    assert summary["cost_lower_bound"] == pytest.approx(0.1)
