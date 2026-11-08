"""A timed-out sync example must not poison later examples (#2951)."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from traigent.evaluators.base import (
    Dataset,
    EvaluationExample,
    count_timed_out_examples,
)
from traigent.evaluators.local import LocalEvaluator


def _dataset(n: int) -> Dataset:
    return Dataset(
        examples=[
            EvaluationExample(input_data={"value": i}, expected_output=i)
            for i in range(n)
        ],
        name="stuck-worker-2951",
    )


def _count(errors: list) -> int:
    return count_timed_out_examples(
        [SimpleNamespace(error_message=e) for e in errors if e is not None]
    )


class _Gate:
    """Deterministic slow function: values in ``hang`` block on an Event."""

    def __init__(self, hang: set[int]) -> None:
        self.hang = hang
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.started: list[int] = []
        self.finished = threading.Event()

    def __call__(self, value: int) -> int:
        with self.lock:
            self.started.append(value)
        if value in self.hang:
            self.release.wait(timeout=5.0)
        return value


@pytest.mark.asyncio
async def test_fast_examples_after_timeout_get_a_healthy_lane() -> None:
    gate = _Gate({1})
    evaluator = LocalEvaluator(metrics=["accuracy"], max_workers=1, timeout=0.05)
    try:
        result = await asyncio.wait_for(
            evaluator.evaluate(gate, {}, _dataset(5)), timeout=3.0
        )
        assert result.errors[1] == "Function call timed out after 0.05s"
        assert all(result.errors[i] is None for i in (0, 2, 3, 4))
        assert sorted(gate.started) == [0, 1, 2, 3, 4]
        assert _count(result.errors) == 1
    finally:
        gate.release.set()


@pytest.mark.asyncio
async def test_stuck_workers_are_bounded_and_skips_are_counted_as_timeouts() -> None:
    gate = _Gate({0, 1})
    evaluator = LocalEvaluator(
        metrics=["accuracy"], max_workers=1, timeout=0.05, max_stuck_workers=1
    )
    try:
        result = await asyncio.wait_for(
            evaluator.evaluate(gate, {}, _dataset(4)), timeout=3.0
        )
        assert result.errors[0] == "Function call timed out after 0.05s"
        assert result.errors[1] == "Function call timed out after 0.05s"
        for i in (2, 3):
            assert result.errors[i].startswith(
                "Sync evaluator worker lane unavailable after 0.05s"
            )
            assert "1 timed-out worker(s) still running" in result.errors[i]
        # Bound: 1 lane + 1 retired stuck worker => never more than 2 threads.
        assert sorted(gate.started) == [0, 1]
        assert _count(result.errors) == 4
    finally:
        gate.release.set()


@pytest.mark.asyncio
async def test_stuck_count_settles_when_the_worker_finishes() -> None:
    gate = _Gate({0})
    evaluator = LocalEvaluator(
        metrics=["accuracy"], max_workers=1, timeout=0.05, max_stuck_workers=1
    )
    await asyncio.wait_for(evaluator.evaluate(gate, {}, _dataset(2)), timeout=3.0)
    assert evaluator._stuck_workers == 1
    gate.release.set()
    for _ in range(200):
        if evaluator._stuck_workers == 0:
            break
        await asyncio.sleep(0.01)
    assert evaluator._stuck_workers == 0
    # Lane accounting stays exact: exactly max_workers slots are available.
    assert evaluator._sync_worker_slots.acquire(blocking=False) is True
    assert evaluator._sync_worker_slots.acquire(blocking=False) is False
    evaluator._sync_worker_slots.release()
