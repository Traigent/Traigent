"""Validation tests for BackendSynchronizer."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from traigent.cloud.backend_synchronizer import BackendSynchronizer
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
