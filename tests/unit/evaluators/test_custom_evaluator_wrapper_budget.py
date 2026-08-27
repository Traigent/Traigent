import asyncio

import pytest

from traigent.api.types import ExampleResult
from traigent.core.evaluator_wrapper import CustomEvaluatorWrapper
from traigent.core.execution_budget import ExecutionBudget
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


@pytest.mark.asyncio
async def test_custom_evaluator_wrapper_respects_execution_budget() -> None:
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(5)
        ],
        name="custom-execution-budget-test",
    )

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

    budget = ExecutionBudget(max_examples=2)
    evaluator = CustomEvaluatorWrapper(custom_evaluator, metrics=["accuracy"])
    result = await evaluator.evaluate(identity, {}, dataset, budget=budget)

    snapshot = budget.snapshot()
    assert result.total_examples == 2
    assert result.examples_consumed == 2
    assert result.sample_budget_exhausted is True
    assert result.execution_budget is not None
    assert result.execution_budget["consumed_examples"] == 2
    assert snapshot.runs == 1
    assert snapshot.trials == 1
    assert snapshot.consumed_examples == 2
    assert snapshot.exhausted_dimension == "examples"


@pytest.mark.asyncio
async def test_custom_evaluator_wrapper_concurrent_budget_admission_is_atomic() -> None:
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(2)
        ],
        name="custom-concurrent-execution-budget-test",
    )
    first_example_barrier = asyncio.Barrier(2)

    async def identity(value: int) -> int:
        return value

    async def custom_evaluator(func, config, example):
        await first_example_barrier.wait()
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

    budget = ExecutionBudget(max_examples=2)
    evaluator = CustomEvaluatorWrapper(custom_evaluator, metrics=["accuracy"])
    results = await asyncio.gather(
        evaluator.evaluate(identity, {}, dataset, budget=budget),
        evaluator.evaluate(identity, {}, dataset, budget=budget),
    )

    snapshot = budget.snapshot()
    assert sorted(result.examples_consumed for result in results) == [1, 1]
    assert sum(result.examples_consumed for result in results) == 2
    assert snapshot.consumed_examples == 2
    assert snapshot.trials == 2


@pytest.mark.asyncio
async def test_custom_evaluator_wrapper_cancellation_refunds_admitted_examples() -> (
    None
):
    dataset = Dataset(
        examples=[EvaluationExample(input_data={"value": 1}, expected_output=1)],
        name="custom-cancellation-execution-budget-test",
    )
    started = asyncio.Event()

    async def identity(value: int) -> int:
        return value

    async def custom_evaluator(func, config, example):
        started.set()
        await asyncio.Event().wait()

    budget = ExecutionBudget(max_examples=1)
    evaluator = CustomEvaluatorWrapper(custom_evaluator, metrics=["accuracy"])
    evaluation = asyncio.create_task(
        evaluator.evaluate(identity, {}, dataset, budget=budget)
    )
    await started.wait()
    evaluation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await evaluation

    snapshot = budget.snapshot()
    assert snapshot.consumed_examples == 0
    assert snapshot.trials == 0
