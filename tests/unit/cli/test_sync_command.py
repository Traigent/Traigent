"""Tests for the top-level ``traigent sync`` CLI command."""

from __future__ import annotations

from unittest.mock import Mock, patch

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
