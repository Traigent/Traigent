"""SyncManager sends ``X-Project-Id`` on every /sessions* write when configured.

Backend develop answers 422 PROJECT_CONTEXT_REQUIRED to a write with no explicit
project, so session create, next-trial, results and finalize must carry it.
"""

from unittest.mock import Mock, patch

import pytest

from traigent.cloud.sync_manager import SyncManager
from traigent.config.types import TraigentConfig


def _resp(payload=None, status=201):
    response = Mock(status_code=status, text="ok")
    response.json.return_value = payload if payload is not None else {}
    return response


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    config = TraigentConfig(
        offline=True, local_storage_path=str(tmp_path), minimal_logging=True
    )
    return SyncManager(config, "test_api_key_123")  # pragma: allowlist secret


def _run_all(manager):
    """Drive create, next-trial, results and finalize; return every POST call."""
    runs = [
        {
            "trial_id": "t0",
            "experiment_parameters": {"a": 1},
            "measures": {"score": 0.5},
            "status": "COMPLETED",
        }
    ]

    def route(url, **kwargs):
        if url.endswith("/next-trial"):
            return _resp({"suggestion": {"trial_id": "bt-1"}}, 200)
        if url.endswith("/sessions"):
            return _resp({"session_id": "sess-id", "metadata": {}})
        return _resp({})

    with patch.object(manager.session, "post") as mock_post:
        mock_post.side_effect = route
        manager._sync_create_session({"function_name": "fn"})
        manager._sync_session_results("sess-id", runs)
        manager._sync_finalize_session("sess-id", "run-id")
    return mock_post.call_args_list


def test_project_header_sent_on_every_sessions_write(manager, monkeypatch):
    monkeypatch.setenv("TRAIGENT_PROJECT_ID", "proj-123")
    calls = _run_all(manager)
    urls = [c.args[0] for c in calls]
    assert any(u.endswith("/sessions") for u in urls)
    assert any(u.endswith("/next-trial") for u in urls)
    assert any(u.endswith("/results") for u in urls)
    assert any(u.endswith("/finalize") for u in urls)
    for call in calls:
        assert call.kwargs.get("headers") == {"X-Project-Id": "proj-123"}, call.args[0]


def test_no_project_header_when_unconfigured(manager, monkeypatch):
    monkeypatch.delenv("TRAIGENT_PROJECT_ID", raising=False)
    calls = _run_all(manager)
    assert calls
    for call in calls:
        assert "headers" not in call.kwargs, call.args[0]


def test_session_project_header_wins_over_env(manager, monkeypatch):
    monkeypatch.setenv("TRAIGENT_PROJECT_ID", "proj-B")
    manager.session.headers["x-project-id"] = "proj-A"
    calls = _run_all(manager)
    assert calls
    for call in calls:
        assert "headers" not in call.kwargs, call.args[0]
    assert manager.session.headers["x-project-id"] == "proj-A"
