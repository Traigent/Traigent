"""Validation tests for BackendSynchronizer."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from traigent.cloud.backend_synchronizer import BackendSynchronizer
from traigent.utils.exceptions import NonRetryableError
from traigent.utils.exceptions import ValidationError as ValidationException


def test_backend_synchronizer_rejects_non_positive_concurrency():
    with pytest.raises(ValidationException):
        BackendSynchronizer(max_concurrent_syncs=0)


def test_backend_synchronizer_rejects_non_positive_batch_size():
    with pytest.raises(ValidationException):
        BackendSynchronizer(batch_size=0)


def test_backend_synchronizer_rejects_non_positive_interval():
    with pytest.raises(ValidationException):
        BackendSynchronizer(sync_interval=0)


def test_backend_synchronizer_initializes_with_valid_values():
    sync = BackendSynchronizer(
        max_concurrent_syncs=2,
        batch_size=10,
        sync_interval=0.1,
        enable_auto_sync=False,
    )
    assert sync.max_concurrent_syncs == 2


@pytest.mark.asyncio
async def test_sync_session_state_fails_closed_without_backend_transport():
    sync = BackendSynchronizer(enable_auto_sync=False, sync_interval=0.1)

    result = await sync.sync_session_state(
        "session-1",
        {
            "status": "completed",
            "function_name": "test_function",
            "objectives": ["accuracy"],
        },
    )

    assert result.success is False
    assert result.items_synced == 0
    assert result.retries == 0
    assert result.error_message is not None
    assert "not implemented" in result.error_message
    assert "unexpected keyword argument" not in result.error_message
    assert sync._stats["failed_syncs"] == 1
    assert sync._stats["successful_syncs"] == 0


@pytest.mark.asyncio
async def test_sync_trial_states_fails_closed_without_backend_transport():
    sync = BackendSynchronizer(enable_auto_sync=False, sync_interval=0.1)

    result = await sync.sync_trial_states(
        "session-1",
        [{"trial_id": "trial-1", "status": "completed"}],
    )

    assert result.success is False
    assert result.items_synced == 0
    assert result.retries == 0
    assert result.error_message is not None
    assert "not implemented" in result.error_message
    assert "unexpected keyword argument" not in result.error_message
    assert sync._stats["failed_syncs"] == 1
    assert sync._stats["successful_syncs"] == 0


@pytest.mark.asyncio
async def test_private_sync_methods_do_not_fabricate_backend_success():
    sync = BackendSynchronizer(enable_auto_sync=False, sync_interval=0.1)

    with pytest.raises(NonRetryableError, match="session sync is not implemented"):
        await sync._sync_session_to_backend(
            "session-1",
            {
                "status": "completed",
                "function_name": "test_function",
                "objectives": ["accuracy"],
            },
        )

    with pytest.raises(NonRetryableError, match="trial sync is not implemented"):
        await sync._sync_trials_to_backend(
            "session-1",
            [{"trial_id": "trial-1", "status": "completed"}],
        )


@pytest.mark.asyncio
async def test_process_sync_queue_runs_queued_sessions(monkeypatch):
    sync = BackendSynchronizer(
        max_concurrent_syncs=1,
        batch_size=5,
        sync_interval=0.1,
        enable_auto_sync=False,
    )

    session_data = {
        "status": "completed",
        "function_name": "test_function",
        "objectives": ["accuracy"],
    }

    fake_result = SimpleNamespace(success=True, result={"status": "synced"}, attempts=1)
    sync._retry_handler.execute_async_with_result = AsyncMock(return_value=fake_result)  # type: ignore[assignment]

    sync.queue_session_sync("session-1", session_data=session_data)

    await sync._process_sync_queue()

    assert sync._stats["total_sync_attempts"] == 1
    assert sync._stats["successful_syncs"] == 1
    assert sync._sync_queue.qsize() == 0


@pytest.mark.asyncio
async def test_process_sync_queue_skips_when_no_snapshot():
    sync = BackendSynchronizer(
        max_concurrent_syncs=1,
        batch_size=5,
        sync_interval=0.1,
        enable_auto_sync=False,
    )

    sync.queue_session_sync("session-2")

    await sync._process_sync_queue()

    assert sync._stats["total_sync_attempts"] == 0
    assert sync._sync_queue.qsize() == 0


@pytest.mark.asyncio
async def test_background_sync_loop_reraises_cancellation(monkeypatch):
    sync = BackendSynchronizer(
        max_concurrent_syncs=1,
        batch_size=5,
        sync_interval=0.1,
        enable_auto_sync=True,
    )

    async def raise_cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", raise_cancelled)

    try:
        with pytest.raises(asyncio.CancelledError):
            await sync._background_sync_loop()
    finally:
        await sync.cleanup()


@pytest.mark.asyncio
async def test_stop_background_sync_awaits_cancelled_background_task():
    sync = BackendSynchronizer(enable_auto_sync=False, sync_interval=0.1)
    cancelled = asyncio.Event()

    async def pending_background_loop():
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    task = asyncio.create_task(pending_background_loop(), name="background_sync")
    sync._background_sync_task = task
    await asyncio.sleep(0)

    await sync.stop_background_sync()

    assert cancelled.is_set()
    assert task.done()
    assert task.cancelled()


@pytest.mark.asyncio
async def test_cleanup_awaits_cancelled_sync_tasks_before_clearing():
    sync = BackendSynchronizer(enable_auto_sync=False, sync_interval=0.1)
    cancelled = asyncio.Event()
    loop = asyncio.get_running_loop()
    exception_contexts = []
    previous_handler = loop.get_exception_handler()

    def capture_exception_context(_loop, context):
        exception_contexts.append(context)

    async def pending_sync():
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    task = asyncio.create_task(pending_sync(), name="queued_sync_session-1")
    sync._sync_tasks["session-1"] = task
    await asyncio.sleep(0)

    loop.set_exception_handler(capture_exception_context)
    try:
        await sync.cleanup()
        del task
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert cancelled.is_set()
    assert sync._sync_tasks == {}
    assert not any(
        context.get("message") == "Task was destroyed but it is pending!"
        for context in exception_contexts
    )
