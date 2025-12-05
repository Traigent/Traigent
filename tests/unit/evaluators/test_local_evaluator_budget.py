import pytest

from traigent.core.sample_budget import SampleBudgetManager
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator


@pytest.mark.asyncio
async def test_local_evaluator_respects_sample_budget_sequential():
    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1)
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(5)
        ],
        name="budget-test",
        description="Sample budget dataset",
    )

    manager = SampleBudgetManager(total_budget=2)
    lease = manager.create_lease("trial-seq", ceiling=2)

    async def identity(value: int) -> int:
        return value

    result = await evaluator.evaluate(
        identity,
        {},
        dataset,
        sample_lease=lease,
    )

    closure = lease.finalize()

    assert result.total_examples == 2
    assert len(result.outputs or []) == 2
    assert result.sample_budget_exhausted is True
    assert result.examples_consumed == 2
    assert closure.consumed == 2
    assert closure.exhausted is True
    assert manager.remaining() == 0


@pytest.mark.asyncio
async def test_local_evaluator_respects_sample_budget_parallel():
    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=4)
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(6)
        ],
        name="budget-test-parallel",
        description="Sample budget dataset parallel",
    )

    manager = SampleBudgetManager(total_budget=3)
    lease = manager.create_lease("trial-parallel")

    async def identity(value: int) -> int:
        return value

    result = await evaluator.evaluate(
        identity,
        {},
        dataset,
        sample_lease=lease,
    )

    closure = lease.finalize()

    assert result.total_examples == 3
    assert len(result.outputs or []) == 3
    assert result.sample_budget_exhausted is True
    assert result.examples_consumed == 3
    assert closure.consumed == 3
    assert closure.exhausted is True
    assert manager.remaining() == 0
