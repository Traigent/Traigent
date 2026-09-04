import asyncio
import threading

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
        await asyncio.wait_for(first_example_barrier.wait(), timeout=1.0)
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
    await asyncio.wait_for(started.wait(), timeout=1.0)
    evaluation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(evaluation, timeout=1.0)

    snapshot = budget.snapshot()
    assert snapshot.consumed_examples == 0
    assert snapshot.trials == 0


@pytest.mark.asyncio
async def test_custom_evaluator_cancellation_retains_completed_examples() -> None:
    started_third = asyncio.Event()
    never = asyncio.Event()
    calls = 0

    async def custom_evaluator(func, config, example):
        nonlocal calls
        calls += 1
        if calls == 3:
            started_third.set()
            await never.wait()
        return ExampleResult(
            example_id="example",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=example.expected_output,
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(3)
        ],
        name="custom-partial-cancellation-budget-test",
    )
    budget = ExecutionBudget(max_examples=3)
    evaluator = CustomEvaluatorWrapper(custom_evaluator, metrics=["accuracy"])
    evaluation = asyncio.create_task(
        evaluator.evaluate(lambda value: value, {}, dataset, budget=budget)
    )
    await asyncio.wait_for(started_third.wait(), timeout=1.0)
    evaluation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(evaluation, timeout=1.0)

    snapshot = budget.snapshot()
    assert snapshot.consumed_examples == 2
    assert snapshot.trials == 0


@pytest.mark.asyncio
async def test_custom_evaluator_wrapper_does_not_refund_running_worker_thread() -> None:
    started = threading.Event()
    release = threading.Event()
    settled = threading.Event()
    calls = 0

    def blocking_custom_evaluator(func, config, example):
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=5)
        settled.set()
        return ExampleResult(
            example_id="example",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=example.expected_output,
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    dataset = Dataset(
        examples=[EvaluationExample(input_data={"value": 1}, expected_output=1)],
        name="custom-thread-cancellation-budget-test",
    )
    budget = ExecutionBudget(max_examples=1)
    evaluator = CustomEvaluatorWrapper(blocking_custom_evaluator, metrics=["accuracy"])
    first = asyncio.create_task(
        evaluator.evaluate(lambda value: value, {}, dataset, budget=budget)
    )
    await asyncio.wait_for(asyncio.to_thread(started.wait, 1), timeout=2.0)
    first.cancel()
    await asyncio.sleep(0)

    second = await evaluator.evaluate(lambda value: value, {}, dataset, budget=budget)
    assert second.execution_budget_exhausted is True
    assert calls == 1
    assert budget.snapshot().consumed_examples == 1

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, timeout=2.0)
    assert await asyncio.wait_for(asyncio.to_thread(settled.wait, 1), timeout=1.0)
    assert budget.snapshot().consumed_examples == 0


@pytest.mark.asyncio
async def test_custom_evaluator_cancellation_refunds_external_sample_lease() -> None:
    started = asyncio.Event()
    never = asyncio.Event()

    async def custom_evaluator(func, config, example):
        started.set()
        await never.wait()

    manager = SampleBudgetManager(total_budget=1)
    lease = manager.create_lease("custom-cancelled")
    evaluator = CustomEvaluatorWrapper(custom_evaluator, metrics=["accuracy"])
    dataset = Dataset(
        examples=[EvaluationExample(input_data={"value": 1}, expected_output=1)],
        name="custom-external-cancellation-budget-test",
    )
    evaluation = asyncio.create_task(
        evaluator.evaluate(
            lambda value: value,
            {},
            dataset,
            sample_lease=lease,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    evaluation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(evaluation, timeout=1.0)

    assert manager.snapshot().consumed == 0
    assert lease.completed == 0
    lease.finalize()


@pytest.mark.asyncio
async def test_custom_evaluator_rejects_budget_and_sample_lease_together() -> None:
    dataset = Dataset(
        examples=[EvaluationExample(input_data={"value": 1}, expected_output=1)],
        name="custom-ambiguous-budget-test",
    )
    budget = ExecutionBudget(max_examples=1)
    manager = SampleBudgetManager(total_budget=1)
    lease = manager.create_lease("external")
    evaluator = CustomEvaluatorWrapper(lambda *args: None, metrics=["accuracy"])

    with pytest.raises(ValueError, match="either budget or sample_lease"):
        await evaluator.evaluate(
            lambda value: value,
            {},
            dataset,
            sample_lease=lease,
            budget=budget,
        )

    assert budget.snapshot().consumed_examples == 0
    assert manager.remaining() == 1


@pytest.mark.asyncio
async def test_execution_budget_leases_share_one_atomic_example_authority() -> None:
    budget = ExecutionBudget(max_examples=1)
    first = budget._create_example_lease()
    second = budget._create_example_lease()

    assert first.try_take()
    assert not second.try_take()
    first.finalize()
    second.finalize()
    assert budget.snapshot().consumed_examples == 1


def test_execution_budget_lease_metrics_report_refunded_work() -> None:
    budget = ExecutionBudget(max_examples=2)
    lease = budget._create_example_lease()

    assert lease.try_take(2)
    lease.mark_completed()
    assert lease.rollback_uncompleted() == 1

    metrics = lease._manager.snapshot()  # noqa: SLF001
    assert metrics.consumed == 1
    assert metrics.wasted == 1
    assert metrics.efficiency == 0.5


def test_rollback_after_finalize_does_not_reopen_global_budget() -> None:
    manager = SampleBudgetManager(total_budget=3)
    lease = manager.create_lease("finalized-lease")
    assert lease.try_take(2)
    lease.mark_completed(1)
    closure = lease.finalize()

    assert closure.consumed == 2
    assert lease.rollback_uncompleted() == 0
    assert manager.consumed() == 2


@pytest.mark.asyncio
async def test_custom_evaluator_with_unbounded_examples_still_accounts_results() -> (
    None
):
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(2)
        ],
        name="custom-unbounded-examples-budget-test",
    )

    async def custom_evaluator(func, config, example):
        return ExampleResult(
            example_id="example",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output=example.expected_output,
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
            error_message=None,
            metadata={},
        )

    budget = ExecutionBudget(max_cost_usd=1.0)
    evaluator = CustomEvaluatorWrapper(custom_evaluator, metrics=["accuracy"])
    result = await evaluator.evaluate(lambda value: value, {}, dataset, budget=budget)

    assert result.total_examples == 2
    assert budget.snapshot().consumed_examples == 2
    assert budget.snapshot().trials == 1
