"""Tests for the top-level ``traigent sync`` CLI command."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli


def _patched_manager(**status):
    manager = Mock()
    manager.get_sync_status.return_value = {
        "completed_sessions": status.get("completed", 2),
        "synced": status.get("synced", 1),
        "unsynced": status.get("unsynced", 1),
        "partial": 0,
        "failed": 0,
        "sync_eligible": status.get("pending", 1),
    }
    return manager


def test_sync_no_args_prints_status_only():
    """`traigent sync` with no target reports status and uploads nothing."""
    manager = _patched_manager()
    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        result = CliRunner().invoke(cli, ["sync", "--api-key", "k"])

    assert result.exit_code == 0
    assert "Traigent sync status" in result.output
    assert "synced" in result.output
    manager.get_sync_status.assert_called_once()
    manager.sync_session_to_cloud.assert_not_called()
    manager.sync_all_sessions.assert_not_called()


def test_sync_status_json():
    manager = _patched_manager()
    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        result = CliRunner().invoke(cli, ["sync", "--json", "--api-key", "k"])

    assert result.exit_code == 0
    assert '"synced"' in result.output


def test_sync_single_session_idempotent_skip():
    manager = _patched_manager()
    manager.sync_session_to_cloud.return_value = {
        "session_id": "s1",
        "status": "already_synced",
        "cloud_experiment_id": "e1",
    }
    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        result = CliRunner().invoke(cli, ["sync", "s1", "--api-key", "k"])

    assert result.exit_code == 0
    assert "Skipped s1" in result.output
    manager.sync_session_to_cloud.assert_called_once()


def test_sync_dry_run_does_not_require_api_key():
    manager = _patched_manager()
    manager.sync_session_to_cloud.return_value = {
        "session_id": "s1",
        "status": "success",
        "trials_converted": 3,
    }
    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        result = CliRunner().invoke(cli, ["sync", "s1", "--dry-run"])

    assert result.exit_code == 0
    assert "Ready to sync s1" in result.output


def test_sync_single_session_success_prints_finalization_warning():
    manager = _patched_manager()
    warning = (
        "Experiment finalization skipped: Experiment e1 is FAILED; offline sync "
        "will not advance it to COMPLETED because this SDK sync does not own "
        "terminal-state recovery transitions"
    )
    manager.sync_session_to_cloud.return_value = {
        "session_id": "s1",
        "status": "success",
        "cloud_experiment_id": "e1",
        "finalization_status": "skipped_terminal_not_owned",
        "finalization_current_status": "FAILED",
        "warnings": [warning],
    }
    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        result = CliRunner().invoke(cli, ["sync", "s1", "--api-key", "k"])

    assert result.exit_code == 0
    assert "Synced s1" in result.output
    assert warning in result.output
    manager.sync_session_to_cloud.assert_called_once()


def test_sync_status_needs_no_api_key():
    """Status-only `traigent sync` must work without an API key."""
    manager = _patched_manager()
    with (
        patch("traigent.cli.sync_commands.SyncManager", return_value=manager),
        patch("traigent.cli.sync_commands._resolve_api_key", return_value=None),
    ):
        result = CliRunner().invoke(cli, ["sync"])

    assert result.exit_code == 0
    assert "Traigent sync status" in result.output


def test_sync_requires_api_key_when_not_dry_run():
    # No api key resolvable and not a dry run → clean error, exit 1.
    with patch("traigent.cli.sync_commands._resolve_api_key", return_value=None):
        result = CliRunner().invoke(cli, ["sync", "s1"])

    assert result.exit_code == 1
    assert "API key required" in result.output


def test_sync_single_session_failure_exits_nonzero():
    """A failed real sync (non dry-run) must exit non-zero so CI detects it.

    Regression test for issue #1420 sub-bug (b): `traigent sync` was exiting 0
    on failure, causing CI/scripts to believe the session was uploaded.
    """
    manager = _patched_manager()
    manager.sync_session_to_cloud.return_value = {
        "session_id": "s1",
        "status": "error",
        "errors": ["Experiment sync failed: HTTP 409: EXPERIMENT_HAS_NO_RUNS"],
    }
    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        result = CliRunner().invoke(cli, ["sync", "s1", "--api-key", "k"])

    assert result.exit_code == 1, (
        f"Expected exit code 1 on sync failure but got {result.exit_code}"
    )


def test_sync_all_failure_exits_nonzero():
    """sync --all with any error exits non-zero."""
    manager = _patched_manager()
    manager.sync_all_sessions.return_value = {
        "total_sessions": 2,
        "eligible_sessions": 2,
        "synced_successfully": 0,
        "skipped": 0,
        "sync_errors": 2,
        "dry_run": False,
        "session_results": [
            {"session_id": "s1", "status": "error", "errors": ["409"]},
            {"session_id": "s2", "status": "error", "errors": ["409"]},
        ],
        "overall_status": "failed",
    }
    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        result = CliRunner().invoke(cli, ["sync", "--all", "--api-key", "k"])

    assert result.exit_code == 1, (
        f"Expected exit code 1 when all syncs fail but got {result.exit_code}"
    )


@pytest.mark.parametrize("status", ["success", "already_synced"])
def test_sync_clean_uses_resolved_session_id(status):
    """--clean must verify and delete the id SyncManager synced, not the raw argument.

    Regression test for issue #2030: the single-session branch recorded the
    user-supplied argument, so when SyncManager resolved it to a different id the
    sync_state lookup missed and --clean silently deleted nothing. Both accepted
    statuses take that path, so both must use the resolved id.
    """
    manager = _patched_manager()
    manager.sync_session_to_cloud.return_value = {
        "session_id": "resolved-1",
        "status": status,
        "cloud_experiment_id": "e1",
        "trials_converted": 3,
    }
    synced_session = Mock()
    synced_session.sync_state = {"status": "synced"}
    manager.storage.load_session.side_effect = lambda sid: (
        synced_session if sid == "resolved-1" else None
    )
    manager.cleanup_after_sync.return_value = {"sessions_deleted": 1}

    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        result = CliRunner().invoke(
            cli, ["sync", "alias-1", "--clean", "--api-key", "k"]
        )

    assert result.exit_code == 0
    # Verification must actually happen, and against the resolved id — not the
    # raw argument and not a blind pass-through of synced_ids.
    manager.storage.load_session.assert_called_once_with("resolved-1")
    manager.cleanup_after_sync.assert_called_once_with(["resolved-1"], keep_backup=True)
    assert "Cleaned 1" in result.output


@pytest.mark.parametrize(
    ("label", "sync_state"),
    [
        ("failed", {"status": "failed"}),
        ("partial", {"status": "partial"}),
        ("missing", None),
    ],
)
def test_sync_clean_skips_when_persisted_state_is_not_synced(label, sync_state):
    """--clean deletes nothing unless the persisted sync_state confirms a sync.

    Pins ``_verified_synced``: a stored session whose sync_state is not "synced"
    (or that cannot be loaded at all) must never reach ``cleanup_after_sync``.
    """
    manager = _patched_manager()
    manager.sync_session_to_cloud.return_value = {
        "session_id": "resolved-1",
        "status": "success",
        "cloud_experiment_id": "e1",
        "trials_converted": 3,
    }
    if sync_state is None:
        session = None
    else:
        session = Mock()
        session.sync_state = sync_state
    manager.storage.load_session.return_value = session

    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        result = CliRunner().invoke(
            cli, ["sync", "alias-1", "--clean", "--api-key", "k"]
        )

    assert result.exit_code == 0
    manager.storage.load_session.assert_called_once_with("resolved-1")
    manager.cleanup_after_sync.assert_not_called()
    assert "Cleaned" not in result.output


# A result that claims success without a usable id is a contract violation: the
# synced id is unknowable, so #2030's silent no-op must become a loud failure.
_UNUSABLE_ID_RESULTS = [
    ("absent", {}),
    ("none", {"session_id": None}),
    ("empty", {"session_id": ""}),
]


@pytest.mark.parametrize(("label", "id_fields"), _UNUSABLE_ID_RESULTS)
@pytest.mark.parametrize("status", ["success", "already_synced"])
def test_sync_single_session_without_usable_id_fails_loudly(label, id_fields, status):
    """No usable session_id → error + exit 1, never a silent fallback to the argument."""
    manager = _patched_manager()
    manager.sync_session_to_cloud.return_value = {"status": status, **id_fields}
    manager.storage.load_session.return_value = None

    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        result = CliRunner().invoke(
            cli, ["sync", "alias-1", "--clean", "--api-key", "k"]
        )

    assert result.exit_code == 1, (
        f"Expected exit 1 for a {label} session_id but got {result.exit_code}"
    )
    assert "session_id" in result.output
    assert status in result.output
    # Never fall back to the raw argument, and never "clean" on an unverified id.
    manager.storage.load_session.assert_not_called()
    manager.cleanup_after_sync.assert_not_called()


@pytest.mark.parametrize(("label", "id_fields"), _UNUSABLE_ID_RESULTS)
def test_sync_all_without_usable_id_fails_loudly(label, id_fields):
    """--all applies the same strict contract as the single-session branch."""
    manager = _patched_manager()
    manager.sync_all_sessions.return_value = {
        "total_sessions": 1,
        "eligible_sessions": 1,
        "synced_successfully": 1,
        "skipped": 0,
        "sync_errors": 0,
        "dry_run": False,
        "session_results": [{"status": "success", **id_fields}],
        "overall_status": "completed",
    }
    manager.storage.load_session.return_value = None

    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        result = CliRunner().invoke(cli, ["sync", "--all", "--clean", "--api-key", "k"])

    assert result.exit_code == 1, (
        f"Expected exit 1 for a {label} session_id but got {result.exit_code}"
    )
    assert "session_id" in result.output
    manager.cleanup_after_sync.assert_not_called()


def test_sync_failure_in_dry_run_exits_zero():
    """--dry-run is non-destructive; failures in dry mode don't set exit code 1."""
    manager = _patched_manager()
    manager.sync_session_to_cloud.return_value = {
        "session_id": "s1",
        "status": "error",
        "errors": ["Dry-run detected ordering issue"],
        "dry_run": True,
    }
    with patch("traigent.cli.sync_commands.SyncManager", return_value=manager):
        result = CliRunner().invoke(cli, ["sync", "s1", "--dry-run"])

    # Dry-run exit code is 0 even when it predicts failure (it's informational).
    assert result.exit_code == 0
