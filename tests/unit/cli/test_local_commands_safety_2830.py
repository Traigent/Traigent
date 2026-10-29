"""Regression tests for #2830: destructive/misleading ``traigent local`` input handling.

Every test runs against a sandbox store (``TRAIGENT_RESULTS_FOLDER``); no user data is touched.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from traigent.cli.local_commands import (
    cleanup_sessions,
    list_sessions,
    sync_to_cloud,
)
from traigent.storage.local_storage import LocalStorageManager

MOD = "traigent.cli.local_commands"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))
    return LocalStorageManager(str(tmp_path))


def _make(store: LocalStorageManager, name: str, status: str, age_days: int) -> str:
    sid = store.create_session(name)
    session = store.load_session(sid)
    session.status = status
    session.created_at = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
    store._save_session(session)
    return sid


@pytest.mark.parametrize("days", ["0", "-1", "-5"])
def test_cleanup_rejects_non_positive_days(store, days):
    sid = _make(store, "f", "completed", 40)
    result = CliRunner().invoke(cleanup_sessions, ["--days", days, "--yes"])
    assert result.exit_code == 2
    assert "Invalid value" in result.output
    assert store.load_session(sid) is not None


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_list_rejects_non_positive_limit(store, limit):
    _make(store, "f", "completed", 1)
    result = CliRunner().invoke(list_sessions, ["--limit", limit])
    assert result.exit_code == 2
    assert "Invalid value" in result.output


def test_cleanup_prompts_and_abort_deletes_nothing(store):
    sid = _make(store, "f", "completed", 40)
    result = CliRunner().invoke(cleanup_sessions, ["--days", "30"], input="n\n")
    assert result.exit_code != 0
    assert "Permanently delete" in result.output
    assert store.load_session(sid) is not None


def test_cleanup_yes_deletes_finished_but_keeps_active_sessions(store):
    done = _make(store, "done", "completed", 40)
    failed = _make(store, "failed", "failed", 40)
    running = _make(store, "running", "running", 40)
    pending = _make(store, "pending", "pending", 40)
    fresh = _make(store, "fresh", "completed", 1)

    result = CliRunner().invoke(cleanup_sessions, ["--days", "30", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Cleaned up 2 sessions" in result.output
    assert store.load_session(done) is None
    assert store.load_session(failed) is None
    assert store.load_session(running) is not None
    assert store.load_session(pending) is not None
    assert store.load_session(fresh) is not None


def test_cleanup_dry_run_does_not_list_active_sessions(store):
    _make(store, "done", "completed", 40)
    _make(store, "running", "running", 40)
    result = CliRunner().invoke(cleanup_sessions, ["--days", "30", "--dry-run"])
    assert result.exit_code == 0
    assert "Would delete 1 sessions" in result.output
    assert "(done)" in result.output and "(running)" not in result.output


def _sync_run(args, one=None, all_=None):
    manager = MagicMock()
    manager.get_sync_status.return_value = {
        "total_sessions": 1,
        "completed_sessions": 1,
        "sync_eligible": 1,
        "total_trials": 1,
    }
    manager.sync_session_to_cloud.return_value = one
    manager.sync_all_sessions.return_value = all_
    with (
        patch(f"{MOD}.TraigentConfig"),
        patch(f"{MOD}._offline_requested", return_value=False),
        patch(f"{MOD}.SyncManager", return_value=manager),
    ):
        return CliRunner().invoke(sync_to_cloud, args, input="y\n"), manager


def test_local_sync_single_failure_exits_nonzero():
    result, _ = _sync_run(
        ["--api-key", "k", "--session-id", "s"],
        one={"status": "failed", "errors": ["boom"]},
    )
    assert result.exit_code == 1
    assert "Sync failed" in result.output


def test_local_sync_all_with_errors_exits_nonzero():
    result, _ = _sync_run(
        ["--api-key", "k"],
        all_={
            "total_sessions": 1,
            "synced_successfully": 0,
            "sync_errors": 1,
            "session_results": [{"session_id": "s", "status": "failed"}],
        },
    )
    assert result.exit_code == 1


def test_local_sync_success_still_exits_zero():
    result, _ = _sync_run(
        ["--api-key", "k", "--session-id", "s"],
        one={"status": "success", "cloud_url": "u"},
    )
    assert result.exit_code == 0


def test_local_sync_cleanup_with_session_id_is_a_usage_error():
    result, manager = _sync_run(["--api-key", "k", "--cleanup", "--session-id", "s"])
    assert result.exit_code == 2
    assert "--cleanup" in result.output and "--session-id" in result.output
    manager.sync_session_to_cloud.assert_not_called()
