"""Non-objective ExampleResult.metrics keys reach trial metrics (#2873).

The custom_evaluator lane must surface user keys the same way the tuple and
metric_functions lanes do, without changing the score or overwriting
evaluator-reserved names.
"""

import pytest

from traigent.api.types import ExampleResult
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.metrics_tracker import TOTAL_MEASURES_CEILING


def _dataset():
    return Dataset(
        examples=[
            EvaluationExample(input_data={"v": i}, expected_output=i) for i in range(4)
        ],
        name="user-metrics",
    )


async def _identity(v):
    return v


def _wrapper(extra_for_row, *, fail_rows=()):
    async def evaluator(func, config, example):
        index = example.input_data["v"]
        metrics = {"accuracy": 1.0}
        metrics.update(extra_for_row(index))
        return ExampleResult(
            example_id=f"r{index}",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=await func(**example.input_data),
            metrics=metrics,
            execution_time=0.0,
            success=index not in fail_rows,
            error_message=None,
            metadata={},
        )

    return CustomEvaluatorWrapper(
        evaluator, metrics=["accuracy"], capture_llm_metrics=False
    )


@pytest.mark.asyncio
async def test_non_objective_key_reaches_trial_metrics_and_score_unchanged():
    base = await _wrapper(lambda i: {}).evaluate(_identity, {}, _dataset())
    result = await _wrapper(
        lambda i: {"part_class": 0.5, "part_step": 0.25 * i}
    ).evaluate(_identity, {}, _dataset())
    assert result.aggregated_metrics["part_class"] == pytest.approx(0.5)
    assert result.aggregated_metrics["part_step"] == pytest.approx(0.375)
    assert result.metrics["part_class"] == pytest.approx(0.5)
    assert result.aggregated_metrics["accuracy"] == base.aggregated_metrics["accuracy"]


@pytest.mark.asyncio
async def test_failed_rows_do_not_contribute_user_keys():
    result = await _wrapper(
        lambda i: {"part_class": 1.0 if i != 3 else 100.0}, fail_rows=(3,)
    ).evaluate(_identity, {}, _dataset())
    assert result.aggregated_metrics["part_class"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_reserved_names_cannot_be_overwritten_by_user_keys():
    # "success_rate" is reserved; "accuracy" is the objective.
    result = await _wrapper(
        lambda i: {"success_rate": 0.123, "total_cost": 99.0, "ok": 1.0}
    ).evaluate(_identity, {}, _dataset())
    assert result.aggregated_metrics["accuracy"] == pytest.approx(1.0)
    assert result.aggregated_metrics.get("success_rate") != pytest.approx(0.123)
    assert result.aggregated_metrics.get("total_cost") != pytest.approx(99.0)
    assert result.aggregated_metrics["ok"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_registered_evaluator_metric_name_cannot_be_overwritten():
    wrapper = _wrapper(lambda i: {"f1": 0.9, "part_class": 0.5})
    wrapper.metric_functions = {"f1": lambda *a, **k: 0.2}
    result = await wrapper.evaluate(_identity, {}, _dataset())
    assert "f1" not in result.aggregated_metrics or result.aggregated_metrics[
        "f1"
    ] != pytest.approx(0.9)
    assert result.aggregated_metrics["part_class"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_user_keys_respect_total_ceiling():
    result = await _wrapper(
        lambda i: {f"k{n:03d}": 1.0 for n in range(TOTAL_MEASURES_CEILING + 20)}
    ).evaluate(_identity, {}, _dataset())
    assert len(result.aggregated_metrics) <= TOTAL_MEASURES_CEILING
    assert result.aggregated_metrics["accuracy"] == pytest.approx(1.0)
