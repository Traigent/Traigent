"""Timing metric canonicalization regressions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.api.types import ExampleResult
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.core.metadata_helpers import _build_single_measure_full
from traigent.evaluators.metrics_tracker import ExampleMetrics, MetricsTracker


def _example_result(*, execution_time: float = 0.25) -> ExampleResult:
    return ExampleResult(
        example_id="ex-1",
        input_data={"question": "q"},
        expected_output="a",
        actual_output="a",
        metrics={"accuracy": 1.0},
        execution_time=execution_time,
        success=True,
        metadata={},
    )


def _llm_metrics(response_time_ms: float) -> SimpleNamespace:
    return SimpleNamespace(
        tokens=SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
        ),
        cost=SimpleNamespace(
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
        ),
        response=SimpleNamespace(response_time_ms=response_time_ms),
    )


def test_custom_evaluator_emits_canonical_timing_keys_with_legacy_compat() -> None:
    wrapper = CustomEvaluatorWrapper(lambda *_args, **_kwargs: None)
    result = _example_result(execution_time=0.0)

    wrapper._enhance_result_with_llm_metrics(
        result,
        _llm_metrics(response_time_ms=1234.5),
        per_example_duration=0.25,
    )

    assert result.metrics["response_time_ms"] == pytest.approx(1234.5)
    assert result.metadata["response_time_ms"] == pytest.approx(1234.5)
    assert result.metrics["execution_time_ms"] == pytest.approx(250.0)
    assert result.metadata["execution_time_ms"] == pytest.approx(250.0)

    # Legacy seconds keys remain only for one compatibility window.
    assert result.metrics["model_response_time"] == pytest.approx(1.2345)
    assert result.metadata["model_response_time"] == pytest.approx(1.2345)
    assert result.metrics["function_duration"] == pytest.approx(0.25)
    assert result.metadata["function_duration"] == pytest.approx(0.25)


def test_measure_payload_keeps_model_latency_distinct_from_execution_time() -> None:
    result = _example_result(execution_time=0.25)
    result.metrics["response_time_ms"] = 1234.5
    result.metrics["execution_time_ms"] = 250.0

    measure = _build_single_measure_full(
        result,
        idx=0,
        dataset_hash="dataset",
        primary_objective="accuracy",
    )

    assert measure is not None
    metrics = measure["metrics"]
    assert metrics["response_time_ms"] == pytest.approx(1234.5)
    assert metrics["execution_time_ms"] == pytest.approx(250.0)
    assert metrics["response_time"] == pytest.approx(0.25)


def test_measure_payload_does_not_route_execution_time_to_response_time_ms() -> None:
    result = _example_result(execution_time=0.25)

    measure = _build_single_measure_full(
        result,
        idx=0,
        dataset_hash="dataset",
        primary_objective="accuracy",
    )

    assert measure is not None
    metrics = measure["metrics"]
    assert "response_time_ms" not in metrics
    assert metrics["execution_time_ms"] == pytest.approx(250.0)
    assert metrics["response_time"] == pytest.approx(0.25)


def test_metrics_tracker_adds_execution_time_ms_for_backend_format() -> None:
    tracker = MetricsTracker()
    tracker.start_tracking()
    tracker.add_example_metrics(ExampleMetrics(success=True))
    tracker.end_tracking()

    formatted = tracker.format_for_backend()

    assert "duration" in formatted
    assert "execution_time_ms" in formatted
    assert formatted["execution_time_ms"] == pytest.approx(
        formatted["duration"] * 1000.0
    )
