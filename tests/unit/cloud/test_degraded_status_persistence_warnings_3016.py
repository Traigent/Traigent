"""#3016: failed config-run status updates and local-persistence failures must warn.

A failed status PUT after a successful trial POST, an unwritable local store for the
fallback session, and a failed terminal-status persist all used to log at DEBUG only,
so a default-configured user saw a clean run. Offline: mocked client, no network.
"""

from __future__ import annotations

import logging
import threading
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.session_operations import SessionOperations
from traigent.cloud.trial_operations import TrialOperations


@pytest.mark.asyncio
async def test_status_update_failure_warns_and_marks_metadata(caplog):
    client = Mock()
    client._update_config_run_status = AsyncMock(return_value=False)
    client._update_config_run_measures = AsyncMock(return_value=True)
    ops = TrialOperations(client)
    response = AsyncMock()
    response.json = AsyncMock(return_value={"continue_optimization": True})
    result_data: dict = {"metadata": {}}

    with caplog.at_level(logging.WARNING, logger="traigent.cloud.trial_operations"):
        result = await ops._handle_trial_success_response(
            response=response,
            session_id="session_1",
            trial_id="trial_1",
            backend_status="COMPLETED",
            result_data=result_data,
            clean_metrics={"accuracy": 0.9},
        )

    # The trial POST itself succeeded, so the overall result stays True.
    assert result is True
    assert result_data["metadata"]["status_update_degraded"] is True
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("trial_1" in r.getMessage() for r in warnings)


@pytest.mark.asyncio
async def test_status_update_success_has_no_degraded_marker():
    client = Mock()
    client._update_config_run_status = AsyncMock(return_value=True)
    client._update_config_run_measures = AsyncMock(return_value=True)
    ops = TrialOperations(client)
    response = AsyncMock()
    response.json = AsyncMock(return_value={"continue_optimization": True})
    result_data: dict = {"metadata": {}}
    await ops._handle_trial_success_response(
        response=response,
        session_id="session_1",
        trial_id="trial_1",
        backend_status="COMPLETED",
        result_data=result_data,
        clean_metrics={"accuracy": 0.9},
    )
    assert "status_update_degraded" not in result_data["metadata"]


def test_local_fallback_storage_failure_warns(caplog):
    client = Mock()
    client.local_storage.create_session.side_effect = OSError("/secret/path full")
    ops = SessionOperations(client)
    with caplog.at_level(logging.WARNING, logger="traigent.cloud.session_operations"):
        session_id = ops._create_local_fallback_session("f", {}, "maximize", None)
    assert session_id.startswith("local_session_")
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "OSError" in message
    # The exception text can quote a full local path: never log it at WARNING.
    assert "/secret/path" not in message


@pytest.mark.asyncio
async def test_terminal_status_persist_failure_warns(caplog):
    client = Mock()
    client._active_sessions_lock = threading.Lock()
    client._active_sessions = {}
    client.local_storage.finalize_session.side_effect = OSError("/secret/path ro")
    ops = SessionOperations(client)
    with (
        patch.object(
            SessionOperations, "_recover_finalization_mapping", return_value=None
        ),
        caplog.at_level(logging.WARNING, logger="traigent.cloud.session_operations"),
    ):
        await ops.finalize_session("session_1", include_full_history=True)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "session_1" in message and "OSError" in message
    assert "/secret/path" not in message
