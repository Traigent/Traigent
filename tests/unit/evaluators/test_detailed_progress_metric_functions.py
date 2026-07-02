"""Regression tests for detailed progress payload objective metrics."""

from __future__ import annotations

import pytest

from traigent.api.types import ExampleResult
from traigent.core.trial_lifecycle import TrialLifecycle
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


def test_detailed_progress_payload_uses_metric_function_with_empty_expected_output():
    calls: list[tuple[dict[str, float], dict[str, object], dict[str, float]]] = []

    def metric(
        output: dict[str, float],
        expected: dict[str, object],
        metrics: dict[str, float],
    ) -> float:
        calls.append((output, expected, metrics))
        return float(output["acc"])

    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        detailed=True,
        metric_functions={"accuracy": metric},
    )
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"row": 0}, expected_output={}),
            EvaluationExample(input_data={"row": 1}, expected_output={}),
        ],
        name="empty-expected-custom-metric",
    )
    result = ExampleResult(
        example_id="example_0",
        input_data=dataset.examples[0].input_data,
        expected_output=dataset.examples[0].expected_output,
        actual_output={"acc": 75.0},
        metrics={"total_tokens": 12.0},
        execution_time=0.01,
        success=True,
        error_message=None,
        metadata={},
    )

    payload = evaluator._build_progress_payload(
        "example_0",
        dataset.examples[0],
        result,
        {"level": 9},
        None,
        dataset,
        0,
    )

    assert payload["metrics"]["accuracy"] == pytest.approx(75.0)
    assert calls == [({"acc": 75.0}, {}, {"total_tokens": 12.0})]


@pytest.mark.asyncio
async def test_detailed_progress_running_score_uses_raw_metric_scale():
    def metric(output: dict[str, float], expected: dict[str, object]) -> float:
        assert expected == {}
        return float(output["acc"])

    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        detailed=True,
        metric_functions={"accuracy": metric},
    )
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"acc": 75.0}, expected_output={}),
            EvaluationExample(input_data={"acc": 95.0}, expected_output={}),
        ],
        name="raw-scale-custom-metric",
    )
    reports: list[tuple[int, dict[str, object]]] = []

    def capture(index: int, payload: dict[str, object]) -> None:
        if "metrics" in payload:
            reports.append((index, payload))

    def emit(acc: float) -> dict[str, float]:
        return {"acc": acc}

    result = await evaluator.evaluate(
        emit,
        {},
        dataset,
        progress_callback=capture,
    )

    detailed_reports = reports[: len(dataset.examples)]
    streamed_values = [
        TrialLifecycle._extract_progress_value(
            dict(payload),  # classmethod treats the payload as read-only
            "accuracy",
        )
        for _, payload in detailed_reports
    ]

    assert streamed_values == [75.0, 95.0]
    assert sum(streamed_values) / len(streamed_values) == pytest.approx(85.0)
    assert result.metrics["accuracy"] == pytest.approx(85.0)
