"""A failing worker cleanup callback must not kill the worker thread (S5754)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading

import pytest

from traigent.evaluators.base import _WorkerInterruptSlot
from traigent.evaluators.local import LocalEvaluator


def _done_future() -> concurrent.futures.Future:
    fut: concurrent.futures.Future = concurrent.futures.Future()
    fut.set_result(1)
    return fut


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_callback_interrupt_is_recorded_not_raised_in_worker(interrupt) -> None:
    ev = LocalEvaluator(metrics=["accuracy"])
    slot = _WorkerInterruptSlot()

    def cleanup() -> None:
        raise interrupt()

    # Must not escape the done-callback (it would kill the executor thread).
    ev._on_sync_worker_done(_done_future(), cleanup, slot)

    assert isinstance(slot.close(), interrupt)


def test_callback_exception_is_logged_not_recorded() -> None:
    ev = LocalEvaluator(metrics=["accuracy"])
    slot = _WorkerInterruptSlot()

    def cleanup() -> None:
        raise RuntimeError("cleanup failed")

    ev._on_sync_worker_done(_done_future(), cleanup, slot)
    assert slot.close() is None


def test_closed_slot_refuses_late_record() -> None:
    slot = _WorkerInterruptSlot()
    assert slot.close() is None
    assert slot.record(KeyboardInterrupt()) is False
    assert slot.close() is None


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interrupt_reaches_awaiting_caller_and_worker_survives(interrupt) -> None:
    ev = LocalEvaluator(metrics=["accuracy"])

    def cleanup() -> None:
        raise interrupt()

    async def go() -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(interrupt):
                await ev._execute_sync_in_thread(
                    lambda *a, **k: "ok",
                    {},
                    (),
                    {},
                    pool,
                    None,
                    worker_done_callback=cleanup,
                    apply_timeout=False,
                )
            # The single worker thread is still alive and usable.
            assert pool.submit(lambda: 42).result(timeout=5) == 42

    asyncio.run(go())


def test_late_interrupt_after_timeout_does_not_hit_unrelated_call() -> None:
    ev = LocalEvaluator(metrics=["accuracy"], timeout=0.2)
    release = threading.Event()
    cleanup_ran = threading.Event()

    def slow(*_a, **_k):
        release.wait(10)
        return "late"

    def cleanup() -> None:
        cleanup_ran.set()
        raise KeyboardInterrupt()

    async def go() -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            _out, err = await ev._execute_sync_in_thread(
                slow, {}, (), {}, pool, None, worker_done_callback=cleanup
            )
            assert err is not None  # first call timed out and left
            release.set()
            assert await asyncio.to_thread(cleanup_ran.wait, 5)
            # An unrelated later call must not inherit the late interrupt.
            out, err2 = await ev._execute_sync_in_thread(
                lambda *a, **k: "ok2", {}, (), {}, pool, None, apply_timeout=False
            )
            assert (out, err2) == ("ok2", None)

    asyncio.run(go())
