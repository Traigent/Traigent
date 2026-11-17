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


@pytest.mark.asyncio
async def test_duplicate_key_producers_non_finite_wins_over_earlier_and_later() -> None:
    """99 (func 1), NaN then 3 (func 2) on the first row: unmeasured, not 51."""

    def first(output, expected_output=None, **kwargs):
        return {"shared": 99.0}

    def second(output, expected_output=None, **kwargs):
        return {"shared": float("nan") if expected_output == "1" else 3.0}

    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        metric_functions={"first": first, "second": second},
        detailed=True,
        execution_mode="local",
    )
    result = await evaluator.evaluate(_func, {}, _dataset())

    stats = result.summary_stats["metrics"]["shared"]
    # row 0 unmeasured; rows 1 and 2 are 99 then overridden by 3.0
    assert stats["count"] == 2
    assert stats["mean"] == 3.0
    assert "shared" not in result.example_results[0].metrics
    assert result.example_results[0].metadata["unmeasured_custom_keys"] == ["shared"]


def _resume_dataset() -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"text": "q1"}, expected_output="1"),
            EvaluationExample(input_data={"text": "q2"}, expected_output="2"),
        ],
        name="resume_2950",
    )


async def _resume_run(stored_metadata: dict | None):
    from traigent.core.resume import ResumePrefill, config_key
    from traigent.evaluators.base import bind_resume_prefill, reset_resume_prefill

    dataset = _resume_dataset()
    row = {
        "input_data": dataset.examples[0].input_data,
        "expected_output": "1",
        "actual_output": "1",
        "metrics": {"total_cost": 0.0, "input_tokens": 1},
        "execution_time": 0.01,
    }
    if stored_metadata is not None:
        row["metadata"] = stored_metadata

    def aux(output, expected_output=None, **kwargs):
        return 3.0

    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        metric_functions={"aux": aux},
        detailed=True,
        execution_mode="local",
    )
    token = bind_resume_prefill(ResumePrefill({config_key({}): {"example_0": row}}))
    try:
        result = await evaluator.evaluate(_func, {}, dataset)
    finally:
        reset_resume_prefill(token)
    return result


@pytest.mark.asyncio
async def test_resumed_unmeasured_row_keeps_marker() -> None:
    result = await _resume_run({"unmeasured_custom_keys": ["aux"]})

    stats = result.summary_stats["metrics"]["aux"]
    assert stats["count"] == 1
    assert stats["mean"] == 3.0
    resumed = result.example_results[0]
    assert resumed.metadata["unmeasured_custom_keys"] == ["aux"]  # survives re-save


@pytest.mark.asyncio
async def test_old_checkpoint_without_marker_still_loads() -> None:
    result = await _resume_run(None)

    assert result.successful_examples == 2
    # legacy behaviour for a row that never recorded the key: unchanged
    assert result.summary_stats["metrics"]["aux"]["count"] == 2
