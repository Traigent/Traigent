"""``_resolve_maybe_awaitable`` thread bridge keeps value/exception semantics."""

from __future__ import annotations

import asyncio
import threading

import pytest

from traigent.core.backend_session_manager import BackendSessionManager

resolve = BackendSessionManager._resolve_maybe_awaitable


class _Interrupt(BaseException):
    """Stand-in for a non-Exception BaseException."""


async def _value(v):
    return v


async def _raise(exc):
    raise exc


async def _thread_name():
    return threading.current_thread().name


def _in_loop(make_coro):
    async def go():
        return resolve(make_coro())

    return asyncio.run(go())


def test_non_awaitable_passthrough() -> None:
    assert resolve(5) == 5


def test_no_loop_value() -> None:
    assert resolve(_value(3)) == 3


def test_no_loop_exceptions_propagate() -> None:
    with pytest.raises(ValueError):
        resolve(_raise(ValueError("x")))
    with pytest.raises(SystemExit):
        resolve(_raise(SystemExit(2)))


def test_running_loop_value() -> None:
    assert _in_loop(lambda: _value({"prune": False})) == {"prune": False}


@pytest.mark.parametrize(
    "exc", [ValueError("x"), _Interrupt(), SystemExit(3), KeyboardInterrupt()]
)
def test_running_loop_exceptions_propagate_to_caller(exc) -> None:
    with pytest.raises(type(exc)):
        _in_loop(lambda: _raise(exc))


def test_running_loop_uses_named_worker_thread() -> None:
    name = _in_loop(_thread_name)
    assert name.startswith("traigent-smart-pruning-report")
    assert name != threading.current_thread().name


def test_running_loop_blocks_until_complete() -> None:
    done = []

    async def slow():
        await asyncio.sleep(0.05)
        done.append(True)
        return "ok"

    assert _in_loop(slow) == "ok"
    assert done == [True]


def test_external_interrupt_while_waiting_does_not_wait_for_stuck_report(
    monkeypatch,
) -> None:
    """An interrupt during ``result()`` must unwind without joining the worker."""
    import concurrent.futures
    import time

    release = threading.Event()

    async def stuck():
        # Blocks the report thread's loop until released.
        await asyncio.get_running_loop().run_in_executor(None, release.wait, 10)
        return "late"

    def interrupted_result(self, timeout=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(concurrent.futures.Future, "result", interrupted_result)
    threading.Timer(3.0, release.set).start()

    async def go():
        start = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            resolve(stuck())
        return time.monotonic() - start

    try:
        elapsed = asyncio.run(go())
    finally:
        release.set()
    assert elapsed < 1.5
