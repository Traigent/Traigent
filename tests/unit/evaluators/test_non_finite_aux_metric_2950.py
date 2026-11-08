"""Traigent#2950: a NaN/inf from a non-objective metric_function must not fail the trial.

The non-objective value is recorded as unmeasured for that example (never 0),
a warning names the metric (not the example content), and the summary
statistics are computed only over the examples that did measure it.
An objective metric returning a non-finite value still fails closed.
"""

from __future__ import annotations

import logging
import math

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.exceptions import EvaluationError


def _dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"text": "q1"}, expected_output="1"),
            EvaluationExample(input_data={"text": "q2"}, expected_output="2"),
            EvaluationExample(input_data={"text": "q3"}, expected_output="3"),
        ],
        name="non_finite_aux_2950",
    )


async def _func(text: str) -> str:
    return "1"


def _evaluator(aux, metrics=("accuracy",)) -> LocalEvaluator:
    return LocalEvaluator(
        metrics=list(metrics),
        metric_functions={"aux": aux},
        detailed=True,
        execution_mode="local",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
async def test_non_objective_non_finite_does_not_fail_trial(bad, caplog) -> None:
    def aux(output, expected_output=None, **kwargs):
        return bad if expected_output == "1" else 1.0

    with caplog.at_level(logging.WARNING):
        result = await _evaluator(aux).evaluate(_func, {}, _dataset())

    assert result.successful_examples == 3
    # mean over the 2 finite examples only: no fabricated zero, no inf
    assert result.metrics["aux"] == 1.0
    records = [r for r in result.metric_errors if r["metric_name"] == "aux"]
    assert [r["example_index"] for r in records] == [0]
    assert records[0]["failure_mode"] == "non_finite_return"
    assert records[0]["is_objective"] is False
    assert any("aux" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_all_non_finite_non_objective_is_unmeasured() -> None:
    def aux(output, expected_output=None, **kwargs):
        return float("nan")

    result = await _evaluator(aux).evaluate(_func, {}, _dataset())

    assert result.successful_examples == 3
    assert "aux" not in result.metrics or result.metrics["aux"] is None
    assert len(result.metric_errors) == 3


@pytest.mark.asyncio
async def test_summary_stats_skip_unmeasured_rows() -> None:
    def aux(output, expected_output=None, **kwargs):
        return float("nan") if expected_output == "1" else 3.0

    result = await _evaluator(aux).evaluate(_func, {}, _dataset())
    stats = result.summary_stats["metrics"]["aux"]

    assert stats["count"] == 2
    assert stats["mean"] == 3.0
    assert stats["min"] == 3.0
    assert all(v is None or math.isfinite(v) for v in stats.values())


@pytest.mark.asyncio
async def test_mapping_sub_metric_non_finite_does_not_fail_trial() -> None:
    def aux(output, expected_output=None, **kwargs):
        return {"aux_a": float("nan") if expected_output == "1" else 2.0}

    result = await _evaluator(aux).evaluate(_func, {}, _dataset())

    assert result.successful_examples == 3
    assert result.metrics["aux_a"] == 2.0


@pytest.mark.asyncio
async def test_objective_non_finite_still_fails_closed() -> None:
    def aux(output, expected_output=None, **kwargs):
        return float("nan")

    with pytest.raises(EvaluationError) as exc_info:
        await _evaluator(aux, metrics=("aux",)).evaluate(_func, {}, _dataset())

    assert exc_info.value.details["metric_name"] == "aux"
    assert exc_info.value.details["is_objective"] is True
    assert exc_info.value.details["failure_mode"] == "non_finite_return"


@pytest.mark.asyncio
async def test_default_non_detailed_mode_does_not_fail_and_stats_skip_row() -> None:
    def aux(output, expected_output=None, **kwargs):
        return float("nan") if expected_output == "1" else 3.0

    evaluator = LocalEvaluator(
        metrics=["accuracy"], metric_functions={"aux": aux}, execution_mode="local"
    )
    result = await evaluator.evaluate(_func, {}, _dataset())

    assert result.successful_examples == 3
    stats = result.summary_stats["metrics"]["aux"]
    assert stats["count"] == 2
    assert stats["mean"] == 3.0
