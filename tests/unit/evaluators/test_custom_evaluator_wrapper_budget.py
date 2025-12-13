import pytest

from traigent.api.types import ExampleResult
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.core.sample_budget import SampleBudgetManager
from traigent.evaluators.base import Dataset, EvaluationExample


@pytest.mark.asyncio
async def test_custom_evaluator_wrapper_respects_sample_budget() -> None:
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(5)
        ],
        name="custom-budget-test",
    )

    manager = SampleBudgetManager(total_budget=2)
    lease = manager.create_lease("trial-custom", ceiling=2)

    async def identity(value: int) -> int:
        return value

    async def custom_evaluator(func, config, example):
        output = await func(**example.input_data)
        return ExampleResult(
            example_id=example.metadata.get("example_id", "example"),
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=output,
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata=example.metadata.copy() if example.metadata else {},
        )

    evaluator = CustomEvaluatorWrapper(custom_evaluator, metrics=["accuracy"])
    result = await evaluator.evaluate(identity, {}, dataset, sample_lease=lease)

    closure = lease.finalize()

    assert result.total_examples == 2
    assert len(result.outputs or []) == 2
    assert result.sample_budget_exhausted is True
    assert result.examples_consumed == 2
    assert closure.consumed == 2
    assert closure.exhausted is True
    assert manager.remaining() == 0
