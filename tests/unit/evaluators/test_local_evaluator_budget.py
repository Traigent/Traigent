import asyncio
import threading
import time

import pytest

from traigent.api.types import ExampleResult
from traigent.core.execution_budget import ExecutionBudget
from traigent.core.sample_budget import SampleBudgetManager
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.utils.langchain_interceptor import (
    capture_langchain_response,
    clear_captured_responses,
    get_captured_response_by_key,
)


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
    first_finished = threading.Event()
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
            if value == 1:
                first_finished.set()

    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1, timeout=0.05)
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": 1}, expected_output=1),
            EvaluationExample(input_data={"value": 2}, expected_output=2),
        ],
        name="local-timeout-worker-capacity-test",
    )
    evaluation = asyncio.create_task(evaluator.evaluate(blocking, {}, dataset))

    try:
        await asyncio.wait_for(asyncio.to_thread(first_started.wait, 1.0), timeout=1.0)
        await asyncio.sleep(0.1)  # bounded: longer than the evaluator timeout
        assert not second_started.is_set()
        assert peak_active == 1

        result = await asyncio.wait_for(evaluation, timeout=2.0)

        assert not second_started.is_set()
        assert peak_active == 1
        assert result.total_examples == 2
        assert result.errors[1] == (
            "Sync evaluator worker lane unavailable after 0.05s"
        )
    finally:
        first_release.set()
        assert await asyncio.to_thread(first_finished.wait, 1.0)


@pytest.mark.asyncio
async def test_local_sync_timeout_returns_promptly_but_retains_worker_lane() -> None:
    """Timeout is wall-clock bounded while the timed-out worker owns its lane."""
    first_started = threading.Event()
    first_release = threading.Event()
    second_started = threading.Event()

    def blocking(value: int) -> int:
        if value == 1:
            first_started.set()
            first_release.wait(timeout=2.0)
        else:
            second_started.set()
        return value

    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1, timeout=0.05)
    first = asyncio.create_task(evaluator._execute_function(blocking, {}, {"value": 1}))
    await asyncio.wait_for(asyncio.to_thread(first_started.wait, 1.0), timeout=1.0)

    started_timing = time.monotonic()
    output, error = await asyncio.wait_for(first, timeout=0.5)
    assert time.monotonic() - started_timing < 0.5
    assert output is None
    assert error == "Function call timed out after 0.05s"

    second = asyncio.create_task(
        evaluator._execute_function(blocking, {}, {"value": 2})
    )
    await asyncio.sleep(0.1)
    assert not second_started.is_set()

    first_release.set()
    await asyncio.wait_for(second, timeout=1.0)
    assert second_started.is_set()


@pytest.mark.asyncio
async def test_local_sync_timeout_bounds_wait_for_wedged_worker_lane() -> None:
    """A never-settling worker must not make later examples hang forever."""
    first_started = threading.Event()
    release_wedged_worker = threading.Event()
    worker_done = threading.Event()
    second_started = threading.Event()

    def blocking(value: int) -> int:
        try:
            if value == 1:
                first_started.set()
                release_wedged_worker.wait()
            else:
                second_started.set()
            return value
        finally:
            if value == 1:
                worker_done.set()

    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1, timeout=0.05)
    dataset = Dataset(
        examples=[
            EvaluationExample(input_data={"value": 1}, expected_output=1),
            EvaluationExample(input_data={"value": 2}, expected_output=2),
        ],
        name="local-wedged-worker-timeout-test",
    )

    try:
        start = time.monotonic()
        evaluation = await asyncio.wait_for(
            evaluator.evaluate(blocking, {}, dataset), timeout=0.5
        )
        elapsed = time.monotonic() - start

        assert first_started.is_set()
        assert not second_started.is_set()
        assert elapsed < 0.4
        assert evaluation.total_examples == 2
        assert evaluation.errors[0] == "Function call timed out after 0.05s"
        assert evaluation.errors[1] == (
            "Sync evaluator worker lane unavailable after 0.05s"
        )
        assert evaluator._sync_worker_slots.acquire(blocking=False) is False
    finally:
        release_wedged_worker.set()

    assert await asyncio.to_thread(worker_done.wait, 1.0)
    assert evaluator._sync_worker_slots.acquire(blocking=False) is True
    evaluator._sync_worker_slots.release()


@pytest.mark.asyncio
async def test_local_sync_custom_evaluator_preserves_capture_key_in_worker() -> None:
    """Sync custom evaluators must correlate captures from their worker thread."""
    clear_captured_responses()
    observed: dict[str, object | None] = {}
    worker_thread_name: list[str] = []
    response = object()

    def custom_evaluator(
        func: object, config: dict[str, object], example: EvaluationExample
    ) -> ExampleResult:
        worker_thread_name.append(threading.current_thread().name)
        capture_langchain_response(response)
        observed["keyed_response"] = get_captured_response_by_key("example-42")
        return ExampleResult(
            example_id="example-42",
            input_data=example.input_data,
            expected_output=example.expected_output,
            actual_output="ok",
            metrics={"accuracy": 1.0},
            execution_time=0.0,
            success=True,
        )

    evaluator = LocalEvaluator(
        metrics=["accuracy"],
        max_workers=1,
        detailed=True,
        custom_eval_func=custom_evaluator,
    )
    dataset = Dataset(
        examples=[
            EvaluationExample(
                input_data={"value": "input"},
                expected_output="ok",
                metadata={"example_id": "example-42"},
            )
        ],
        name="local-capture-key-worker-test",
    )

    try:
        result = await evaluator.evaluate(lambda value: value, {}, dataset)
    finally:
        clear_captured_responses()

    assert result.successful_examples == 1
    assert observed["keyed_response"] is response
    assert worker_thread_name
    assert worker_thread_name[0] != threading.current_thread().name


@pytest.mark.asyncio
async def test_local_sync_cancel_returns_promptly_but_retains_worker_lane() -> None:
    """Cancellation is prompt while the cancelled worker still owns its lane."""
    first_started = threading.Event()
    first_release = threading.Event()
    second_started = threading.Event()

    def blocking(value: int) -> int:
        if value == 1:
            first_started.set()
            first_release.wait(timeout=2.0)
        else:
            second_started.set()
        return value

    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1)
    first = asyncio.create_task(evaluator._execute_function(blocking, {}, {"value": 1}))
    await asyncio.wait_for(asyncio.to_thread(first_started.wait, 1.0), timeout=1.0)

    started_cancelling = time.monotonic()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, timeout=0.5)
    assert time.monotonic() - started_cancelling < 0.5

    second = asyncio.create_task(
        evaluator._execute_function(blocking, {}, {"value": 2})
    )
    await asyncio.sleep(0.1)
    assert not second_started.is_set()

    first_release.set()
    await asyncio.wait_for(second, timeout=1.0)
    assert second_started.is_set()
