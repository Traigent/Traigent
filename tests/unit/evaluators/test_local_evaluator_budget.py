import asyncio
import threading

import pytest

from traigent.core.execution_budget import ExecutionBudget
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


@pytest.mark.asyncio
async def test_local_evaluator_cancellation_refunds_execution_budget_admission():
    started = asyncio.Event()
    never = asyncio.Event()

    async def blocking(value: int) -> int:
        started.set()
        await never.wait()
        return value

    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1)
    dataset = Dataset(
        examples=[EvaluationExample(input_data={"value": 1}, expected_output=1)],
        name="local-cancellation-budget-test",
    )
    budget = ExecutionBudget(max_examples=1)
    evaluation = asyncio.create_task(
        evaluator.evaluate(blocking, {}, dataset, budget=budget)
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    evaluation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(evaluation, timeout=1.0)

    snapshot = budget.snapshot()
    assert snapshot.consumed_examples == 0
    assert snapshot.trials == 0


@pytest.mark.asyncio
async def test_local_evaluator_cancellation_refunds_external_sample_lease() -> None:
    started = asyncio.Event()
    never = asyncio.Event()

    async def blocking(value: int) -> int:
        started.set()
        await never.wait()
        return value

    manager = SampleBudgetManager(total_budget=1)
    lease = manager.create_lease("local-cancelled")
    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1)
    dataset = Dataset(
        examples=[EvaluationExample(input_data={"value": 1}, expected_output=1)],
        name="local-external-cancellation-budget-test",
    )
    evaluation = asyncio.create_task(
        evaluator.evaluate(blocking, {}, dataset, sample_lease=lease)
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    evaluation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(evaluation, timeout=1.0)

    assert manager.snapshot().consumed == 0
    assert lease.completed == 0
    lease.finalize()


@pytest.mark.asyncio
async def test_local_timeout_keeps_sync_worker_lane_occupied_until_settled() -> None:
    """A timed-out sync call cannot overlap the next example on one worker."""
    first_started = threading.Event()
    first_release = threading.Event()
    second_started = threading.Event()
    active_lock = threading.Lock()
    active = 0
    peak_active = 0

    def blocking(value: int) -> int:
        nonlocal active, peak_active
        with active_lock:
            active += 1
            peak_active = max(peak_active, active)
        try:
            if value == 1:
                first_started.set()
                first_release.wait(timeout=2.0)
            else:
                second_started.set()
            return value
        finally:
            with active_lock:
                active -= 1

    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1, timeout=0.05)
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": 1}, expected_output=1),
            EvaluationExample(input_data={"value": 2}, expected_output=2),
        ],
        name="local-timeout-worker-capacity-test",
    )
    evaluation = asyncio.create_task(evaluator.evaluate(blocking, {}, dataset))

    await asyncio.wait_for(asyncio.to_thread(first_started.wait, 1.0), timeout=1.0)
    await asyncio.sleep(0.1)  # bounded: longer than the evaluator timeout
    assert not second_started.is_set()
    assert peak_active == 1

    first_release.set()
    result = await asyncio.wait_for(evaluation, timeout=2.0)

    assert second_started.is_set()
    assert peak_active == 1
    assert result.total_examples == 2
