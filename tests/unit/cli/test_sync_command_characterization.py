"""Extra characterization of the ``traigent sync`` command flow."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

from click.testing import CliRunner

from traigent.cli.main import cli


def _mgr():
    m = Mock()
    m.get_sync_status.return_value = {
        "completed_sessions": 0,
        "synced": 0,
        "unsynced": 0,
        "partial": 0,
        "failed": 0,
        "sync_eligible": 0,
    }
    return m


def _run(manager, *args):
    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        return CliRunner().invoke(cli, ["sync", "--api-key", "k", *args])


def test_session_id_wins_over_all_and_forwards_flags():
    m = _mgr()
    m.sync_session_to_cloud.return_value = {"session_id": "s", "status": "success"}
    r = _run(m, "s", "--all", "--force", "--dry-run")
    assert r.exit_code == 0
    m.sync_session_to_cloud.assert_called_once_with("s", dry_run=True, force=True)
    m.sync_all_sessions.assert_not_called()
    assert "Ready to sync s (0 trials)" in r.output


def test_all_forwards_dry_run_positionally_and_prints_summary():
    m = _mgr()
    m.sync_all_sessions.return_value = {
        "synced_successfully": 1,
        "sync_errors": 0,
        "eligible_sessions": 2,
        "session_results": [
            {"session_id": "a", "status": "success"},
            {"session_id": "b", "status": "already_synced"},
        ],
    }
    r = _run(m, "--all", "--force")
    assert r.exit_code == 0
    m.sync_all_sessions.assert_called_once_with(False, force=True)
    assert "Synced 1, skipped 0, errors 0 (of 2 eligible)" in r.output
    assert "✅ Synced a" in r.output and "Skipped b" in r.output


def test_all_json_output_and_error_exit():
    m = _mgr()
    payload = {
        "synced_successfully": 0,
        "sync_errors": 2,
        "eligible_sessions": 2,
        "session_results": [],
    }
    m.sync_all_sessions.return_value = payload
    r = _run(m, "--all", "--json")
    assert r.exit_code == 1
    assert json.loads(r.output) == payload


def test_all_errors_in_dry_run_do_not_fail():
    m = _mgr()
    m.sync_all_sessions.return_value = {
        "synced_successfully": 0,
        "sync_errors": 3,
        "eligible_sessions": 3,
        "session_results": [],
    }
    assert _run(m, "--all", "--dry-run").exit_code == 0


def test_clean_not_run_on_dry_run_and_cleanup_skipped_when_nothing_verified():
    m = _mgr()
    m.sync_session_to_cloud.return_value = {"session_id": "s", "status": "success"}
    _run(m, "s", "--dry-run", "--clean")
    m.storage.load_session.assert_not_called()
    m.cleanup_after_sync.assert_not_called()

    m2 = _mgr()
    m2.sync_session_to_cloud.return_value = {"session_id": "s", "status": "success"}
    m2.storage.load_session.return_value = None
    r = _run(m2, "s", "--clean")
    assert r.exit_code == 0
    m2.cleanup_after_sync.assert_not_called()


def test_clean_with_failed_session_still_exits_nonzero_without_cleanup():
    m = _mgr()
    m.sync_session_to_cloud.return_value = {"session_id": "s", "status": "error"}
    r = _run(m, "s", "--clean")
    assert r.exit_code == 1
    m.cleanup_after_sync.assert_not_called()


def test_clean_message_and_backup_flag():
    m = _mgr()
    m.sync_session_to_cloud.return_value = {"session_id": "s", "status": "success"}
    m.storage.load_session.return_value = Mock(sync_state={"status": "synced"})
    m.cleanup_after_sync.return_value = {"sessions_deleted": 1}
    r = _run(m, "s", "--clean")
    assert r.exit_code == 0
    m.cleanup_after_sync.assert_called_once_with(["s"], keep_backup=True)
    assert "Cleaned 1 synced run(s) (backed up first)." in r.output


def test_exceptions_are_wrapped_with_clean_message():
    m = _mgr()
    m.sync_session_to_cloud.side_effect = RuntimeError("boom")
    r = _run(m, "s")
    assert r.exit_code == 1
    assert "Error syncing to portal: boom" in r.output
