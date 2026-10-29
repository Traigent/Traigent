"""A failing worker cleanup callback must not kill the worker thread (S5754)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading

import pytest

from traigent.evaluators.local import LocalEvaluator


def _done_future() -> concurrent.futures.Future:
    fut: concurrent.futures.Future = concurrent.futures.Future()
    fut.set_result(1)
    return fut


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_callback_interrupt_is_recorded_not_raised_in_worker(interrupt) -> None:
    ev = LocalEvaluator(metrics=["accuracy"])

    def cleanup() -> None:
        raise interrupt()

    # Must not escape the done-callback (it would kill the executor thread).
    ev._on_sync_worker_done(_done_future(), cleanup)

    recorded = ev._worker_cleanup_interrupts
    assert len(recorded) == 1
    assert isinstance(recorded[0], interrupt)


def test_callback_exception_is_logged_not_recorded() -> None:
    ev = LocalEvaluator(metrics=["accuracy"])

    def cleanup() -> None:
        raise RuntimeError("cleanup failed")

    ev._on_sync_worker_done(_done_future(), cleanup)
    assert ev._worker_cleanup_interrupts == []


def test_interrupt_reaches_awaiting_caller_and_worker_survives() -> None:
    ev = LocalEvaluator(metrics=["accuracy"])
    cleanup_ran = threading.Event()

    def cleanup() -> None:
        cleanup_ran.set()
        raise KeyboardInterrupt()

    async def go() -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            raised = False
            try:
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
            except KeyboardInterrupt:
                raised = True
            if not raised:
                # The cleanup callback ran on the worker thread after the
                # awaiter resumed; the next observation must surface it.
                assert await asyncio.to_thread(cleanup_ran.wait, 5)
                with pytest.raises(KeyboardInterrupt):
                    await ev._execute_sync_in_thread(
                        lambda *a, **k: "ok2",
                        {},
                        (),
                        {},
                        pool,
                        None,
                        apply_timeout=False,
                    )
            # The single worker thread is still alive and usable.
            assert pool.submit(lambda: 42).result(timeout=5) == 42

    asyncio.run(go())
