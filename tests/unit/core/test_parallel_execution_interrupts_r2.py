"""KeyboardInterrupt/SystemExit in a trial coroutine must not be swallowed (S5754)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.core.cost_enforcement import Permit
from traigent.core.parallel_execution_manager import ParallelExecutionManager


def _enforcer() -> MagicMock:
    enforcer = MagicMock()
    enforcer.acquire_permit_async = AsyncMock(
        return_value=Permit(id=7, amount=0.5, active=True)
    )
    enforcer.release_permit_async = AsyncMock()
    return enforcer


def _run(manager: ParallelExecutionManager, coro):
    async def go():
        return await manager.run_with_cost_permits([coro])

    return asyncio.run(go())


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interrupt_releases_permit_then_reraises(interrupt) -> None:
    enforcer = _enforcer()
    manager = ParallelExecutionManager(1, cost_enforcer=enforcer)

    async def trial():
        raise interrupt()

    with pytest.raises(interrupt):
        _run(manager, trial())

    enforcer.release_permit_async.assert_awaited_once()
    assert enforcer.release_permit_async.await_args.args[0].id == 7


def test_ordinary_exception_still_returned_as_value() -> None:
    enforcer = _enforcer()
    manager = ParallelExecutionManager(1, cost_enforcer=enforcer)

    async def trial():
        raise ValueError("boom")

    results, cancelled = _run(manager, trial())

    assert cancelled == 0
    assert isinstance(results[0].result, ValueError)
    assert results[0].permit.id == -1
    enforcer.release_permit_async.assert_awaited_once()
