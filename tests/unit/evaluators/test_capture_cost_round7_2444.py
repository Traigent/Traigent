"""Round 7: previews preserve identity and finalization preserves late spend."""

import asyncio
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tests.unit.evaluators.test_capture_cost_round6_2444 import _timeout_after_event
from tests.unit.evaluators.test_local_capture_ownership_2444 import (
    MODEL,
    _Resp,
    _Usageless,
    _cost,
    _dataset,
)
from traigent.evaluators.local import LocalEvaluator
from traigent.evaluators.metrics_tracker import MetricsTracker
from traigent.utils import langchain_interceptor as interceptor


@pytest.mark.parametrize("captured_first", [False, True])
def test_cached_response_is_charged_once_in_each_progress_and_final_lane(
    captured_first: bool,
) -> None:
    shared = _Resp(1000, 500)
    evaluator = LocalEvaluator(
        metrics=["accuracy"], detailed=True, execution_mode="local", max_workers=1
    )

    async def evaluate() -> Any:
        payloads: list[Any] = []
        inputs: list[Any] = []

        async def agent(text: str, **config: Any) -> Any:
            if captured_first and text == "0":
                interceptor.capture_langchain_response(shared)
            return {"text": "ok", "raw_response": shared}

        def metric(output: Any, expected: Any, llm_metrics: Any = None) -> float:
            inputs.append(copy.deepcopy(llm_metrics))
            return 1.0

        evaluator.metric_functions = {"quality": metric}
        result = await evaluator.evaluate(
            agent,
            {},
            _dataset(2),
            progress_callback=lambda i, p: payloads.append(copy.deepcopy(p)),
        )
        assert len(inputs) == 4
        assert len(payloads) == 4
        for lane in (
            inputs[:2],
            inputs[2:],
            [p["metrics"] for p in payloads[:2]],
            [p["metrics"] for p in payloads[2:]],
        ):
            assert lane[0]["total_cost"] == pytest.approx(_cost(shared))
            assert lane[1].get("total_cost") is None
            assert lane[1]["cost_unmeasured"] == 1.0
            assert sum(
                m.get("total_cost") or m.get("cost_lower_bound", 0) for m in lane
            ) == pytest.approx(_cost(shared))
        assert result.aggregated_metrics["cost_lower_bound"] == pytest.approx(
            _cost(shared)
        )
        return result

    # Reuse both evaluator and cached object: identity history is evaluation scoped.
    asyncio.run(evaluate())
    asyncio.run(evaluate())


@pytest.mark.parametrize(
    "late",
    [_Usageless(), _Resp(9000, 3000, model="unpriced-test-model"), _Resp(9000, 3000)],
    ids=["no-usage", "unpriced", "priced"],
)
def test_sealed_capture_keeps_trial_unknown_and_exact_floor(
    late: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured = _Resp(2000, 700)
    original = MetricsTracker.format_as_summary_stats

    def summary(tracker: MetricsTracker, **kwargs: Any) -> Any:
        interceptor.capture_langchain_response(late)
        return original(tracker, **kwargs)

    monkeypatch.setattr(MetricsTracker, "format_as_summary_stats", summary)

    def agent(text: str, **config: Any) -> str:
        interceptor.capture_langchain_response(measured)
        return "ok"

    result = asyncio.run(
        LocalEvaluator(metrics=["accuracy"], execution_mode="local").evaluate(
            agent, {}, _dataset(1)
        )
    )
    for metrics in (result.aggregated_metrics, result.summary_stats["metrics"]):
        assert metrics["cost_unmeasured"] == 1.0
        assert metrics.get("cost") is None
        expected_floor = _cost(measured) + (
            _cost(late)
            if isinstance(late, _Resp)
            and late.response_metadata.get("model_name") == MODEL
            else 0.0
        )
        assert metrics["cost_lower_bound"] == pytest.approx(expected_floor)


@pytest.mark.parametrize("finished_before_seal", [False, True])
def test_seal_checks_actual_worker_completion_and_absorbs_captures(
    finished_before_seal: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.evaluators import local

    started = threading.Event()
    release = threading.Event()
    captured = threading.Event()
    futures: list[Any] = []
    priced = _Resp(1000, 500)
    original_submit = ThreadPoolExecutor.submit
    original_seal = local.seal_captured_responses
    original_summary = MetricsTracker.format_as_summary_stats
    sealed: list[bool] = []
    _timeout_after_event(monkeypatch, started)

    def submit(executor: Any, *args: Any, **kwargs: Any) -> Any:
        future = original_submit(executor, *args, **kwargs)
        futures.append(future)
        return future

    def seal() -> bool:
        if finished_before_seal:
            release.set()
            assert futures[0].result(timeout=5) == "ok"
        incomplete = original_seal()
        sealed.append(incomplete)
        return incomplete

    def summary(tracker: MetricsTracker, **kwargs: Any) -> Any:
        if not finished_before_seal:
            release.set()
            assert futures[0].result(timeout=5) == "ok"
        return original_summary(tracker, **kwargs)

    def agent(text: str, **config: Any) -> str:
        started.set()
        assert release.wait(5)
        interceptor.capture_langchain_response(priced)
        captured.set()
        return "ok"

    monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)
    monkeypatch.setattr(local, "seal_captured_responses", seal)
    monkeypatch.setattr(MetricsTracker, "format_as_summary_stats", summary)
    try:
        result = asyncio.run(
            LocalEvaluator(
                metrics=["accuracy"], timeout=10.0, execution_mode="local"
            ).evaluate(agent, {"model": MODEL}, _dataset(1))
        )
        assert sealed == [not finished_before_seal]
        assert "timed out" in result.errors[0]
        for metrics in (result.aggregated_metrics, result.summary_stats["metrics"]):
            # Completed calls without a row remain unattributed, with exact spend.
            assert metrics["cost_unmeasured"] == 1.0
            assert metrics["cost_lower_bound"] == pytest.approx(_cost(priced))
    finally:
        release.set()
        assert captured.wait(5)


def test_capture_during_late_batch_pricing_is_folded_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured = _Resp(1000, 500)
    late = _Resp(2000, 700)
    later = _Resp(3000, 900)
    original_summary = MetricsTracker.format_as_summary_stats
    original_price = LocalEvaluator._price_one_call

    def summary(tracker: MetricsTracker, **kwargs: Any) -> Any:
        interceptor.capture_langchain_response(late)
        return original_summary(tracker, **kwargs)

    def price(response: Any, **kwargs: Any) -> Any:
        if response is late:
            # A second worker can complete while the first late batch is priced.
            interceptor.capture_langchain_response(later)
        return original_price(response, **kwargs)

    async def agent(text: str, **config: Any) -> str:
        interceptor.capture_langchain_response(measured)
        return "ok"

    monkeypatch.setattr(MetricsTracker, "format_as_summary_stats", summary)
    monkeypatch.setattr(LocalEvaluator, "_price_one_call", staticmethod(price))
    result = asyncio.run(
        LocalEvaluator(metrics=["accuracy"], execution_mode="local").evaluate(
            agent, {}, _dataset(1)
        )
    )
    for metrics in (result.aggregated_metrics, result.summary_stats["metrics"]):
        assert metrics["cost_unmeasured"] == 1.0
        assert metrics.get("cost") is None
        assert metrics["cost_lower_bound"] == pytest.approx(
            _cost(measured) + _cost(late) + _cost(later)
        )
