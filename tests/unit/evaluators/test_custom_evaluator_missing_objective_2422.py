"""A custom_evaluator that omits an objective key fails the trial instead of
scoring a silent 0.0 (#2422)."""

import pytest

from traigent.api.types import ExampleResult
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import EvaluationError


def _dataset():
    return Dataset(
        examples=[
            EvaluationExample(input_data={"v": i}, expected_output=i) for i in range(3)
        ],
        name="missing-objective",
    )


async def _identity(v):
    return v


def _wrapper(key_for_row):
    async def evaluator(func, config, example):
        index = example.input_data["v"]
        key = key_for_row(index)
        return ExampleResult(
            example_id=f"r{index}",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=await func(**example.input_data),
            metrics={key: 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    return CustomEvaluatorWrapper(
        evaluator, metrics=["accuracy"], capture_llm_metrics=False
    )


@pytest.mark.asyncio
async def test_wrong_key_fails_trial_with_content_free_message():
    wrapper = _wrapper(lambda i: "score")
    with pytest.raises(EvaluationError) as info:
        await wrapper.evaluate(_identity, {}, _dataset())
    text = str(info.value)
    assert "'accuracy'" in text and "3 of 3" in text


@pytest.mark.asyncio
async def test_one_row_missing_key_also_fails():
    wrapper = _wrapper(lambda i: "score" if i == 1 else "accuracy")
    with pytest.raises(EvaluationError) as info:
        await wrapper.evaluate(_identity, {}, _dataset())
    assert "1 of 3" in str(info.value)


@pytest.mark.asyncio
async def test_correct_key_still_succeeds():
    wrapper = _wrapper(lambda i: "accuracy")
    result = await wrapper.evaluate(_identity, {}, _dataset())
    assert result.aggregated_metrics["accuracy"] == pytest.approx(1.0)
