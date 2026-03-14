"""Safe unit tests for JS bridge shutdown logic.

These tests do not spawn Node.js. They only exercise mocked bridge and pool
shutdown paths that are otherwise hard to cover without the JS runtime.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from traigent.bridges.js_bridge import JSBridge, JSBridgeConfig
from traigent.bridges.process_pool import JSProcessPool, JSProcessPoolConfig


def _make_bridge(**config_overrides: object) -> JSBridge:
    config = JSBridgeConfig(module_path="./fake.js", **config_overrides)
    return JSBridge(config)


@pytest.mark.asyncio
async def test_stop_cancels_active_trial_before_shutdown() -> None:
    bridge = _make_bridge()
    bridge._started = True
    bridge._process = MagicMock(returncode=None)

    parent = MagicMock()
    bridge.cancel_active_trial = AsyncMock()
    bridge._send_request = AsyncMock(return_value={"status": "success", "payload": {}})
    bridge._wait_for_process_exit = AsyncMock(return_value=True)
    bridge._finalize_shutdown = AsyncMock()

    parent.attach_mock(bridge.cancel_active_trial, "cancel_active_trial")
    parent.attach_mock(bridge._send_request, "send_request")
    parent.attach_mock(bridge._wait_for_process_exit, "wait_for_process_exit")
    parent.attach_mock(bridge._finalize_shutdown, "finalize_shutdown")

    await bridge.stop(timeout=3.0)

    assert parent.mock_calls == [
        call.cancel_active_trial(),
        call.send_request(action="shutdown", payload={}, timeout=3.0),
        call.wait_for_process_exit(3.0),
        call.finalize_shutdown(),
    ]


@pytest.mark.asyncio
async def test_stop_terminates_when_process_does_not_exit() -> None:
    bridge = _make_bridge()
    bridge._started = True
    bridge._process = MagicMock(returncode=None)

    parent = MagicMock()
    bridge.cancel_active_trial = AsyncMock()
    bridge._send_request = AsyncMock(return_value={"status": "success", "payload": {}})
    bridge._wait_for_process_exit = AsyncMock(return_value=False)
    bridge._terminate_process = AsyncMock()
    bridge._finalize_shutdown = AsyncMock()

    parent.attach_mock(bridge.cancel_active_trial, "cancel_active_trial")
    parent.attach_mock(bridge._send_request, "send_request")
    parent.attach_mock(bridge._wait_for_process_exit, "wait_for_process_exit")
    parent.attach_mock(bridge._terminate_process, "terminate_process")
    parent.attach_mock(bridge._finalize_shutdown, "finalize_shutdown")

    await bridge.stop(timeout=3.0)

    assert parent.mock_calls == [
        call.cancel_active_trial(),
        call.send_request(action="shutdown", payload={}, timeout=3.0),
        call.wait_for_process_exit(3.0),
        call.terminate_process(),
        call.finalize_shutdown(),
    ]


@pytest.mark.asyncio
async def test_run_trial_clears_active_trial_id_after_completion() -> None:
    bridge = _make_bridge()
    bridge._started = True
    bridge._process = MagicMock()
    bridge._send_request = AsyncMock(
        return_value={
            "status": "success",
            "payload": {
                "trial_id": "trial-123",
                "status": "completed",
                "metrics": {"accuracy": 0.9},
                "duration": 1.0,
            },
        }
    )

    result = await bridge.run_trial(
        {"trial_id": "trial-123", "config": {}},
        timeout=2.0,
    )

    assert result.trial_id == "trial-123"
    assert bridge._active_trial_id is None
    bridge._send_request.assert_awaited_once_with(
        action="run_trial",
        payload={"trial_id": "trial-123", "config": {}, "timeout_ms": 2000},
        timeout=7.0,
    )


@pytest.mark.asyncio
async def test_finalize_shutdown_clears_state() -> None:
    bridge = _make_bridge()
    bridge._started = True
    bridge._process = MagicMock()
    pending = asyncio.get_running_loop().create_future()
    bridge._pending_requests["req-1"] = pending
    bridge._active_trial_id = "trial-123"
    bridge._reader_task = asyncio.create_task(asyncio.sleep(3600))
    bridge._stderr_task = asyncio.create_task(asyncio.sleep(3600))

    await bridge._finalize_shutdown()

    assert isinstance(pending.exception(), Exception)
    assert bridge._pending_requests == {}
    assert bridge._process is None
    assert bridge._reader_task is None
    assert bridge._stderr_task is None
    assert bridge._active_trial_id is None
    assert bridge._started is False


@pytest.mark.asyncio
async def test_terminate_process_uses_kill_after_timeout() -> None:
    bridge = _make_bridge()
    bridge._process = MagicMock(returncode=None)
    bridge._process.wait = AsyncMock(side_effect=[TimeoutError(), 0])

    await bridge._terminate_process()

    bridge._process.terminate.assert_called_once()
    bridge._process.kill.assert_called_once()
    assert bridge._process.wait.await_count == 2


@pytest.mark.asyncio
async def test_process_pool_shutdown_requests_cancel_before_stop() -> None:
    pool = JSProcessPool(JSProcessPoolConfig(module_path="./fake.js"))
    worker_one = MagicMock()
    worker_one.cancel_active_trial = AsyncMock()
    worker_one.stop = AsyncMock()
    worker_two = MagicMock()
    worker_two.cancel_active_trial = AsyncMock()
    worker_two.stop = AsyncMock()

    pool._workers = [worker_one, worker_two]
    pool._started = True

    await pool.shutdown(timeout=1.0)

    worker_one.cancel_active_trial.assert_awaited_once()
    worker_two.cancel_active_trial.assert_awaited_once()
    worker_one.stop.assert_awaited_once()
    worker_two.stop.assert_awaited_once()
