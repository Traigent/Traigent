"""Safe unit tests for JS bridge shutdown logic.

These tests do not spawn Node.js. They only exercise mocked bridge and pool
shutdown paths that are otherwise hard to cover without the JS runtime.
"""

from __future__ import annotations

import asyncio
import os
import signal
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from traigent.bridges.js_bridge import (
    JSBridge,
    JSBridgeConfig,
    JSProcessError,
    JSTrialTimeoutError,
)
from traigent.bridges.process_pool import JSProcessPool, JSProcessPoolConfig


def _make_bridge(**config_overrides: object) -> JSBridge:
    config = JSBridgeConfig(module_path="./fake.js", **config_overrides)
    return JSBridge(config)


def _make_mock_process() -> MagicMock:
    process = MagicMock(returncode=None)
    process.pid = 12345
    process.stdin = MagicMock()
    process.stdin.write = MagicMock()
    process.stdin.drain = AsyncMock()
    process.stdout = MagicMock()
    process.stdout.readline = AsyncMock(return_value=b"")
    process.stderr = MagicMock()
    process.stderr.readline = AsyncMock(return_value=b"")
    process.wait = AsyncMock(return_value=0)
    process.terminate = MagicMock()
    process.kill = MagicMock()
    return process


_HAS_POSIX_PROCESS_GROUPS = (
    os.name == "posix"
    and hasattr(os, "getpgid")
    and hasattr(os, "getpgrp")
    and hasattr(os, "killpg")
)


@pytest.mark.asyncio
async def test_start_tracks_background_tasks_without_spawning_node() -> None:
    bridge = _make_bridge()
    bridge.ping = AsyncMock(return_value={"ok": True})
    process = _make_mock_process()

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
        await bridge.start()

    assert bridge._process is process
    assert bridge._reader_task is not None
    assert bridge._stderr_task is not None

    await bridge._finalize_shutdown()


@pytest.mark.skipif(
    not _HAS_POSIX_PROCESS_GROUPS, reason="POSIX process groups are unavailable"
)
@pytest.mark.asyncio
async def test_start_captures_process_group_before_health_check() -> None:
    bridge = _make_bridge()
    bridge.ping = AsyncMock(return_value={"ok": True})
    process = _make_mock_process()
    create_process = AsyncMock(return_value=process)

    with (
        patch("asyncio.create_subprocess_exec", create_process),
        patch("os.getpgid", return_value=67890) as getpgid,
        patch("os.getpgrp", return_value=11111),
    ):
        await bridge.start()

    assert create_process.call_args.kwargs["start_new_session"] is True
    assert bridge._pgid == 67890
    getpgid.assert_called_once_with(process.pid)

    await bridge._finalize_shutdown()


@pytest.mark.asyncio
async def test_start_raises_process_error_when_ping_fails() -> None:
    bridge = _make_bridge()
    bridge.ping = AsyncMock(side_effect=RuntimeError("ping failed"))
    bridge._terminate = AsyncMock()
    process = _make_mock_process()

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
        with pytest.raises(JSProcessError, match="failed health check"):
            await bridge.start()

    bridge._terminate.assert_awaited_once()


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
async def test_stop_finalizes_state_after_successful_shutdown() -> None:
    bridge = _make_bridge()
    bridge._started = True
    bridge._process = _make_mock_process()
    bridge._reader_task = asyncio.create_task(asyncio.sleep(3600))
    bridge._stderr_task = asyncio.create_task(asyncio.sleep(3600))
    bridge._send_request = AsyncMock(return_value={"status": "success", "payload": {}})

    await bridge.stop(timeout=3.0)

    assert bridge._process is None
    assert bridge._reader_task is None
    assert bridge._stderr_task is None
    assert bridge._started is False


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
async def test_stop_terminates_after_failed_shutdown_command() -> None:
    bridge = _make_bridge()
    bridge._started = True
    bridge._process = MagicMock(returncode=None)

    parent = MagicMock()
    bridge.cancel_active_trial = AsyncMock()
    bridge._send_request = AsyncMock(side_effect=RuntimeError("shutdown failed"))
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
async def test_stop_finalizes_shutdown_even_when_wait_raises() -> None:
    bridge = _make_bridge()
    bridge._started = True
    bridge._process = MagicMock(returncode=None)
    bridge.cancel_active_trial = AsyncMock()
    bridge._send_request = AsyncMock(return_value={"status": "success", "payload": {}})
    bridge._wait_for_process_exit = AsyncMock(side_effect=RuntimeError("wait failed"))
    bridge._finalize_shutdown = AsyncMock()

    with pytest.raises(RuntimeError, match="wait failed"):
        await bridge.stop(timeout=3.0)

    bridge._finalize_shutdown.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_handles_process_exit_during_shutdown() -> None:
    bridge = _make_bridge()
    bridge._started = True
    bridge._process = _make_mock_process()
    bridge._process.returncode = 0
    bridge._reader_task = asyncio.create_task(asyncio.sleep(3600))
    bridge._stderr_task = asyncio.create_task(asyncio.sleep(3600))
    bridge.cancel_active_trial = AsyncMock()
    bridge._send_request = AsyncMock(side_effect=JSProcessError("process exited"))
    bridge._terminate_process = AsyncMock()

    await bridge.stop(timeout=3.0)

    bridge.cancel_active_trial.assert_awaited_once()
    bridge._terminate_process.assert_not_awaited()
    assert bridge._process is None
    assert bridge._reader_task is None
    assert bridge._stderr_task is None


@pytest.mark.asyncio
async def test_cancel_active_trial_returns_when_bridge_not_running() -> None:
    bridge = _make_bridge()
    bridge.cancel = AsyncMock()

    await bridge.cancel_active_trial()

    bridge.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_active_trial_returns_when_trial_id_missing() -> None:
    bridge = _make_bridge()
    bridge._started = True
    bridge._process = _make_mock_process()
    bridge.cancel = AsyncMock()

    await bridge.cancel_active_trial()

    bridge.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_active_trial_forwards_current_trial_id() -> None:
    bridge = _make_bridge()
    bridge._started = True
    bridge._process = _make_mock_process()
    bridge._active_trial_id = "trial-123"
    bridge.cancel = AsyncMock(return_value={"status": "success"})

    await bridge.cancel_active_trial()

    bridge.cancel.assert_awaited_once_with("trial-123")


@pytest.mark.asyncio
async def test_cancel_active_trial_swallows_cancel_errors() -> None:
    bridge = _make_bridge()
    bridge._started = True
    bridge._process = _make_mock_process()
    bridge._active_trial_id = "trial-123"
    bridge.cancel = AsyncMock(side_effect=RuntimeError("cancel failed"))

    await bridge.cancel_active_trial()

    bridge.cancel.assert_awaited_once_with("trial-123")


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
async def test_run_trial_timeout_clears_active_trial_id_after_cancel() -> None:
    bridge = _make_bridge()
    bridge._started = True
    bridge._process = _make_mock_process()
    bridge._send_request = AsyncMock(side_effect=TimeoutError())
    bridge.cancel = AsyncMock(return_value={"status": "cancelled"})

    with pytest.raises(JSTrialTimeoutError, match="timed out"):
        await bridge.run_trial({"trial_id": "trial-123", "config": {}}, timeout=2.0)

    bridge.cancel.assert_awaited_once_with("trial-123")
    assert bridge._active_trial_id is None


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
    assert bridge._pgid is None
    assert bridge._started is False


@pytest.mark.skipif(
    not _HAS_POSIX_PROCESS_GROUPS, reason="POSIX process groups are unavailable"
)
@pytest.mark.asyncio
async def test_terminate_process_uses_process_group_after_timeout() -> None:
    bridge = _make_bridge()
    bridge._process = _make_mock_process()
    bridge._pgid = 67890
    bridge._process.wait = AsyncMock(side_effect=[TimeoutError(), 0])

    with patch("os.killpg") as killpg:
        await bridge._terminate_process()

    killpg.assert_has_calls(
        [call(67890, signal.SIGTERM), call(67890, signal.SIGKILL)]
    )
    bridge._process.terminate.assert_not_called()
    bridge._process.kill.assert_not_called()
    assert bridge._process.wait.await_count == 2


@pytest.mark.asyncio
async def test_terminate_process_falls_back_to_child_without_pgid() -> None:
    bridge = _make_bridge()
    bridge._process = _make_mock_process()
    bridge._process.wait = AsyncMock(side_effect=[TimeoutError(), 0])

    await bridge._terminate_process()

    bridge._process.terminate.assert_called_once()
    bridge._process.kill.assert_called_once()
    assert bridge._process.wait.await_count == 2


@pytest.mark.asyncio
async def test_terminate_process_raises_after_failed_force_kill() -> None:
    bridge = _make_bridge()
    bridge._process = _make_mock_process()
    bridge._process.wait = AsyncMock(side_effect=TimeoutError())

    with pytest.raises(JSProcessError, match="did not exit after force kill"):
        await bridge._terminate_process()

    bridge._process.terminate.assert_called_once()
    bridge._process.kill.assert_called_once()
    assert bridge._process.wait.await_count == 2


@pytest.mark.asyncio
async def test_terminate_process_ignores_direct_process_lookup_error() -> None:
    bridge = _make_bridge()
    bridge._process = _make_mock_process()
    bridge._process.terminate.side_effect = ProcessLookupError()

    await bridge._terminate_process()

    bridge._process.kill.assert_not_called()


@pytest.mark.asyncio
async def test_wait_for_process_exit_returns_true_when_already_exited() -> None:
    bridge = _make_bridge()
    assert await bridge._wait_for_process_exit(0.1) is True

    bridge._process = _make_mock_process()
    bridge._process.returncode = 0
    assert await bridge._wait_for_process_exit(0.1) is True


@pytest.mark.asyncio
async def test_wait_for_process_exit_returns_false_on_timeout() -> None:
    bridge = _make_bridge()
    bridge._process = _make_mock_process()
    bridge._process.wait = AsyncMock(side_effect=TimeoutError())

    assert await bridge._wait_for_process_exit(0.1) is False


@pytest.mark.asyncio
async def test_wait_for_process_exit_returns_true_on_process_lookup_error() -> None:
    bridge = _make_bridge()
    bridge._process = _make_mock_process()
    bridge._process.wait = AsyncMock(side_effect=ProcessLookupError())

    assert await bridge._wait_for_process_exit(0.1) is True


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
