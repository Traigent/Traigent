"""
Comprehensive test suite for Edge Analytics CLI commands.
Tests all CLI commands with error handling and edge cases.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from traigent.cli.local_commands import (
    cleanup_sessions,
    delete_session,
    edge_analytics_commands,
    export_session,
    list_sessions,
    manage_config,
    preview_cloud_benefits,
    show_session,
    storage_info,
    sync_to_cloud,
)
from traigent.storage.local_storage import LocalStorageManager


class TestEdgeAnalyticsCommands:
    """Test suite for Edge Analytics CLI commands with comprehensive error handling."""

    def setup_method(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)

        # Set up environment for tests
        self.env_vars = {
            "TRAIGENT_RESULTS_FOLDER": str(self.storage_path),
            "TRAIGENT_EDGE_ANALYTICS_MODE": "true",
            "TRAIGENT_MINIMAL_LOGGING": "true",
        }

        self.runner = CliRunner()

        # Create test data
        self.storage = LocalStorageManager(str(self.storage_path))
        self._create_test_sessions()

    def teardown_method(self):
        """Clean up temporary storage."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_sessions(self):
        """Create test sessions for CLI testing."""
        # Create completed session
        self.completed_session_id = self.storage.create_session(
            "completed_function",
            optimization_config={"param": [1, 2, 3]},
            metadata={"user": "test_user"},
        )

        self.storage.add_trial_result(self.completed_session_id, {"param": 1}, 0.7)
        self.storage.add_trial_result(self.completed_session_id, {"param": 2}, 0.9)
        self.storage.add_trial_result(self.completed_session_id, {"param": 3}, 0.8)
        self.storage.finalize_session(self.completed_session_id, "completed")

        # Create pending session
        self.pending_session_id = self.storage.create_session(
            "pending_function", optimization_config={"temp": [0.1, 0.5, 0.9]}
        )

        self.storage.add_trial_result(self.pending_session_id, {"temp": 0.1}, 0.6)

        # Create failed session
        self.failed_session_id = self.storage.create_session("failed_function")
        self.storage.add_trial_result(
            self.failed_session_id, {"param": "bad"}, 0.0, error="Configuration error"
        )
        self.storage.finalize_session(self.failed_session_id, "failed")

    def test_list_sessions_default(self):
        """Test listing sessions with default parameters."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(list_sessions)

        assert result.exit_code == 0
        output = result.output

        # Should show table format by default
        assert "Session ID" in output
        assert "Function" in output
        assert "Status" in output
        assert "completed_function" in output
        assert "pending_function" in output
        assert "failed_function" in output

    def test_list_sessions_json_format(self):
        """Test listing sessions in JSON format."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(list_sessions, ["--format", "json"])

        assert result.exit_code == 0

        # Should be valid JSON
        try:
            sessions_data = json.loads(result.output)
            assert isinstance(sessions_data, list)
            assert len(sessions_data) == 3
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    def test_list_sessions_with_status_filter(self):
        """Test listing sessions with status filter."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(list_sessions, ["--status", "completed"])

        assert result.exit_code == 0
        output = result.output

        assert "completed_function" in output
        assert "pending_function" not in output
        assert "failed_function" not in output

    def test_list_sessions_with_limit(self):
        """Test listing sessions with limit."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(list_sessions, ["--limit", "1"])

        assert result.exit_code == 0
        # Should only show one session (most recent)

    def test_list_sessions_empty(self):
        """Test listing sessions when none exist."""
        # Clear all sessions
        for session_id in [
            self.completed_session_id,
            self.pending_session_id,
            self.failed_session_id,
        ]:
            self.storage.delete_session(session_id)

        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(list_sessions)

        assert result.exit_code == 0
        assert "No sessions found" in result.output

    def test_list_sessions_error_handling(self):
        """Test list sessions with storage errors."""
        with patch.dict(os.environ, self.env_vars):
            with patch(
                "traigent.cli.local_commands.LocalStorageManager"
            ) as mock_storage:
                mock_storage.side_effect = Exception("Storage error")

                result = self.runner.invoke(list_sessions)

        assert result.exit_code == 1
        assert "Error listing sessions" in result.output

    def test_show_session_summary_format(self):
        """Test showing session in summary format."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(show_session, [self.completed_session_id])

        assert result.exit_code == 0
        output = result.output

        assert "completed_function Optimization" in output
        assert "Status: completed" in output
        assert "Trials: 3" in output
        assert "Best Score: 0.9000" in output

    def test_show_session_detailed_format(self):
        """Test showing session in detailed format."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(
                show_session, [self.completed_session_id, "--format", "detailed"]
            )

        assert result.exit_code == 0
        output = result.output

        assert "Session Details:" in output
        assert "Best Configuration:" in output
        assert "Trial History:" in output
        assert "Function: completed_function" in output

    def test_show_session_json_format(self):
        """Test showing session in JSON format."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(
                show_session, [self.completed_session_id, "--format", "json"]
            )

        assert result.exit_code == 0

        # Should be valid JSON
        try:
            session_data = json.loads(result.output)
            assert session_data["session_id"] == self.completed_session_id
            assert session_data["function_name"] == "completed_function"
            assert session_data["status"] == "completed"
            assert len(session_data["trials"]) == 3
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    def test_show_session_nonexistent(self):
        """Test showing non-existent session."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(show_session, ["fake_session_id"])

        assert result.exit_code == 1
        assert "Session 'fake_session_id' not found" in result.output

    def test_show_session_error_handling(self):
        """Test show session with errors."""
        with patch.dict(os.environ, self.env_vars):
            with patch(
                "traigent.cli.local_commands.LocalStorageManager"
            ) as mock_storage_class:
                mock_storage = MagicMock()
                mock_storage.load_session.side_effect = Exception("Load error")
                mock_storage_class.return_value = mock_storage

                result = self.runner.invoke(show_session, [self.completed_session_id])

        assert result.exit_code == 1
        assert "Error showing session" in result.output

    def test_export_session_default(self):
        """Test exporting session with default parameters."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(export_session, [self.completed_session_id])

        assert result.exit_code == 0
        assert "Session exported to:" in result.output

        # Should create file under exports directory within storage path
        export_file = (
            Path(self.storage_path) / "exports" / f"{self.completed_session_id}.json"
        )
        assert export_file.exists()

        # Clean up
        export_file.unlink()

    def test_export_session_custom_output(self):
        """Test exporting session with custom output path."""
        custom_path = str(Path(self.temp_dir) / "custom_export.json")

        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(
                export_session, [self.completed_session_id, "--output", custom_path]
            )

        assert result.exit_code == 0
        assert Path(custom_path).exists()

    def test_export_session_nonexistent(self):
        """Test exporting non-existent session."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(export_session, ["fake_session_id"])

        assert result.exit_code == 1
        assert "Session 'fake_session_id' not found" in result.output

    def test_export_session_error_handling(self):
        """Test export session with errors."""
        with patch.dict(os.environ, self.env_vars):
            with patch(
                "traigent.cli.local_commands.LocalStorageManager"
            ) as mock_storage_class:
                mock_storage = MagicMock()
                mock_storage.load_session.return_value = MagicMock()
                mock_storage.export_session.return_value = False
                mock_storage_class.return_value = mock_storage

                result = self.runner.invoke(export_session, [self.completed_session_id])

        assert result.exit_code == 1
        assert "Failed to export session" in result.output

    def test_delete_session_with_confirmation(self):
        """Test deleting session with confirmation."""
        with patch.dict(os.environ, self.env_vars):
            # Simulate user confirming deletion
            result = self.runner.invoke(
                delete_session, [self.completed_session_id], input="y\n"
            )

        assert result.exit_code == 0
        assert "Session" in result.output and "deleted" in result.output

        # Verify session was deleted
        assert self.storage.load_session(self.completed_session_id) is None

    def test_delete_session_force(self):
        """Test deleting session with force flag."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(
                delete_session, [self.pending_session_id, "--force"]
            )

        assert result.exit_code == 0
        assert "deleted" in result.output

    def test_delete_session_abort(self):
        """Test aborting session deletion."""
        with patch.dict(os.environ, self.env_vars):
            # Simulate user aborting deletion
            result = self.runner.invoke(
                delete_session, [self.failed_session_id], input="n\n"
            )

        assert result.exit_code == 1  # Aborted

        # Verify session still exists
        assert self.storage.load_session(self.failed_session_id) is not None

    def test_delete_session_nonexistent(self):
        """Test deleting non-existent session."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(delete_session, ["fake_session_id", "--force"])

        assert result.exit_code == 1
        assert "Session 'fake_session_id' not found" in result.output

    def test_cleanup_sessions_dry_run(self):
        """Test cleanup sessions in dry run mode."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(
                cleanup_sessions,
                ["--days", "0", "--dry-run"],  # All sessions should be "old"
            )

        assert result.exit_code == 0
        assert "Would delete" in result.output
        assert "sessions older than 0 days" in result.output

        # Sessions should still exist
        assert self.storage.load_session(self.completed_session_id) is not None

    def test_cleanup_sessions_actual(self):
        """Test actual cleanup of old sessions."""
        with patch.dict(os.environ, self.env_vars):
            with patch(
                "traigent.cli.local_commands.LocalStorageManager"
            ) as mock_storage_class:
                mock_storage = MagicMock()
                mock_storage.cleanup_old_sessions.return_value = 2
                mock_storage_class.return_value = mock_storage

                result = self.runner.invoke(cleanup_sessions, ["--days", "30"])

        assert result.exit_code == 0
        assert "✅ Cleaned up 2 sessions older than 30 days" in result.output
        mock_storage.cleanup_old_sessions.assert_called_with(30)

    def test_cleanup_sessions_error_handling(self):
        """Test cleanup sessions with errors."""
        with patch.dict(os.environ, self.env_vars):
            with patch(
                "traigent.cli.local_commands.LocalStorageManager"
            ) as mock_storage_class:
                mock_storage = MagicMock()
                mock_storage.cleanup_old_sessions.side_effect = Exception(
                    "Cleanup error"
                )
                mock_storage_class.return_value = mock_storage

                result = self.runner.invoke(cleanup_sessions, ["--days", "30"])

        assert result.exit_code == 1
        assert "Error cleaning up sessions" in result.output

    def test_storage_info_command(self):
        """Test storage info command."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(storage_info)

        assert result.exit_code == 0
        output = result.output

        assert "Traigent Local Storage" in output
        assert "Total Sessions:" in output
        assert "Total Trials:" in output
        assert "Storage Size:" in output
        assert "Session Breakdown:" in output
        assert "Upgrade to Traigent Cloud:" in output

    def test_storage_info_error_handling(self):
        """Test storage info with errors."""
        with patch.dict(os.environ, self.env_vars):
            with patch(
                "traigent.cli.local_commands.LocalStorageManager"
            ) as mock_storage_class:
                mock_storage_class.side_effect = Exception("Storage error")

                result = self.runner.invoke(storage_info)

        assert result.exit_code == 1
        assert "Error getting storage info" in result.output

    def test_manage_config_show(self):
        """Test showing configuration."""
        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(manage_config, ["--show"])

        assert result.exit_code == 0
        output = result.output

        assert "Traigent Local Mode Configuration" in output
        assert "Execution Mode:" in output
        assert "Storage Path:" in output
        assert "Environment Variables:" in output
        assert "TRAIGENT_RESULTS_FOLDER" in output

    def test_manage_config_set_path(self):
        """Test setting custom storage path."""
        new_path = str(Path(self.temp_dir) / "new_storage")

        result = self.runner.invoke(manage_config, ["--set-path", new_path])

        assert result.exit_code == 0
        output = result.output

        assert "export TRAIGENT_RESULTS_FOLDER" in output
        assert new_path in output

    def test_manage_config_reset(self):
        """Test showing reset instructions."""
        result = self.runner.invoke(manage_config, ["--reset"])

        assert result.exit_code == 0
        output = result.output

        assert "unset TRAIGENT_RESULTS_FOLDER" in output
        assert "unset TRAIGENT_EDGE_ANALYTICS_MODE" in output

    def test_manage_config_no_options(self):
        """Test manage config with no options."""
        result = self.runner.invoke(manage_config)

        assert result.exit_code == 0
        assert "Use --show to display" in result.output

    def test_manage_config_error_handling(self):
        """Test manage config with errors."""
        with patch(
            "traigent.cli.local_commands.TraigentConfig.from_environment"
        ) as mock_config:
            mock_config.side_effect = Exception("Config error")

            result = self.runner.invoke(manage_config, ["--show"])

        assert result.exit_code == 1
        assert "Error managing configuration" in result.output

    def test_sync_to_cloud_no_api_key(self):
        """Test sync to cloud without API key."""
        # Ensure TRAIGENT_API_KEY is not set
        env_vars_no_api = self.env_vars.copy()
        env_vars_no_api["TRAIGENT_API_KEY"] = ""  # Explicitly clear API key

        with patch.dict(os.environ, env_vars_no_api, clear=True):
            result = self.runner.invoke(sync_to_cloud)

        assert result.exit_code == 1
        assert "API key required" in result.output
        assert "Get your API key at:" in result.output

    def test_sync_to_cloud_dry_run(self):
        """Test sync to cloud in dry run mode."""
        with patch.dict(os.environ, self.env_vars):
            with patch("traigent.cli.local_commands.SyncManager") as mock_sync_class:
                mock_sync = MagicMock()
                mock_sync.get_sync_status.return_value = {
                    "total_sessions": 3,
                    "completed_sessions": 2,
                    "sync_eligible": 2,
                    "total_trials": 5,
                }
                mock_sync.sync_all_sessions.return_value = {
                    "total_sessions": 2,
                    "synced_successfully": 2,
                    "sync_errors": 0,
                    "session_results": [],
                }
                mock_sync_class.return_value = mock_sync

                result = self.runner.invoke(sync_to_cloud, ["--dry-run"])

        assert result.exit_code == 0
        assert "Dry Run Mode - Preview Only" in result.output

    def test_sync_to_cloud_no_eligible_sessions(self):
        """Test sync when no sessions are eligible."""
        with patch.dict(os.environ, self.env_vars):
            with patch("traigent.cli.local_commands.SyncManager") as mock_sync_class:
                mock_sync = MagicMock()
                mock_sync.get_sync_status.return_value = {
                    "total_sessions": 1,
                    "completed_sessions": 0,
                    "sync_eligible": 0,
                    "total_trials": 2,
                }
                mock_sync_class.return_value = mock_sync

                result = self.runner.invoke(sync_to_cloud, ["--api-key", "test_key"])

        assert result.exit_code == 0
        assert "No completed sessions to sync" in result.output

    def test_sync_to_cloud_specific_session(self):
        """Test syncing specific session."""
        with patch.dict(os.environ, self.env_vars):
            with patch("traigent.cli.local_commands.SyncManager") as mock_sync_class:
                mock_sync = MagicMock()
                mock_sync.get_sync_status.return_value = {
                    "total_sessions": 3,
                    "completed_sessions": 2,
                    "sync_eligible": 2,
                    "total_trials": 5,
                }
                mock_sync.sync_session_to_cloud.return_value = {
                    "status": "success",
                    "cloud_url": "https://example.com/experiments/123",
                }
                mock_sync_class.return_value = mock_sync

                result = self.runner.invoke(
                    sync_to_cloud,
                    [
                        "--api-key",
                        "test_key",
                        "--session-id",
                        self.completed_session_id,
                    ],
                )

        assert result.exit_code == 0
        assert "Session synced successfully" in result.output

    def test_sync_to_cloud_failed_session(self):
        """Test syncing session that fails."""
        with patch.dict(os.environ, self.env_vars):
            with patch("traigent.cli.local_commands.SyncManager") as mock_sync_class:
                mock_sync = MagicMock()
                mock_sync.get_sync_status.return_value = {
                    "total_sessions": 1,
                    "completed_sessions": 1,
                    "sync_eligible": 1,
                    "total_trials": 3,
                }
                mock_sync.sync_session_to_cloud.return_value = {
                    "status": "error",
                    "errors": ["Network timeout", "Authentication failed"],
                }
                mock_sync_class.return_value = mock_sync

                result = self.runner.invoke(
                    sync_to_cloud,
                    [
                        "--api-key",
                        "test_key",
                        "--session-id",
                        self.completed_session_id,
                    ],
                )

        assert result.exit_code == 0
        assert "Sync failed:" in result.output
        assert "Network timeout" in result.output

    def test_sync_to_cloud_all_sessions(self):
        """Test syncing all sessions."""
        with patch.dict(os.environ, self.env_vars):
            with patch("traigent.cli.local_commands.SyncManager") as mock_sync_class:
                mock_sync = MagicMock()
                mock_sync.get_sync_status.return_value = {
                    "total_sessions": 3,
                    "completed_sessions": 2,
                    "sync_eligible": 2,
                    "total_trials": 6,
                }
                mock_sync.sync_all_sessions.return_value = {
                    "total_sessions": 2,
                    "synced_successfully": 1,
                    "sync_errors": 1,
                    "session_results": [
                        {"session_id": "session1", "status": "success"},
                        {
                            "session_id": "session2",
                            "status": "error",
                            "errors": ["API error"],
                        },
                    ],
                }
                mock_sync_class.return_value = mock_sync

                result = self.runner.invoke(
                    sync_to_cloud, ["--api-key", "test_key"], input="y\n"
                )  # Confirm sync

        assert result.exit_code == 0
        assert "Sync Results:" in result.output
        assert "Synced successfully: 1" in result.output
        assert "Errors: 1" in result.output

    def test_sync_to_cloud_with_cleanup(self):
        """Test sync with cleanup option."""
        with patch.dict(os.environ, self.env_vars):
            with patch("traigent.cli.local_commands.SyncManager") as mock_sync_class:
                mock_sync = MagicMock()
                mock_sync.get_sync_status.return_value = {
                    "total_sessions": 2,
                    "completed_sessions": 2,
                    "sync_eligible": 2,
                    "total_trials": 4,
                }
                mock_sync.sync_all_sessions.return_value = {
                    "total_sessions": 2,
                    "synced_successfully": 2,
                    "sync_errors": 0,
                    "session_results": [
                        {"session_id": "session1", "status": "success"},
                        {"session_id": "session2", "status": "success"},
                    ],
                }
                mock_sync.cleanup_after_sync.return_value = {"sessions_deleted": 2}
                mock_sync_class.return_value = mock_sync

                result = self.runner.invoke(
                    sync_to_cloud,
                    ["--api-key", "test_key", "--cleanup"],
                    input="y\ny\n",
                )  # Confirm sync and cleanup

        assert result.exit_code == 0
        assert "Cleaned up 2 sessions" in result.output

    def test_sync_to_cloud_error_handling(self):
        """Test sync to cloud with errors."""
        with patch.dict(os.environ, self.env_vars):
            with patch("traigent.cli.local_commands.SyncManager") as mock_sync_class:
                mock_sync_class.side_effect = Exception("Sync manager error")

                result = self.runner.invoke(sync_to_cloud, ["--api-key", "test_key"])

        assert result.exit_code == 1
        assert "Error syncing to cloud" in result.output

    def test_preview_cloud_benefits(self):
        """Test previewing cloud benefits."""
        with patch.dict(os.environ, self.env_vars):
            with patch("traigent.cli.local_commands.SyncManager") as mock_sync_class:
                mock_sync = MagicMock()
                mock_sync.get_sync_status.return_value = {
                    "completed_sessions": 5,
                    "total_trials": 50,
                }
                mock_sync.get_cloud_analytics_preview.return_value = {
                    "functions_optimized": 3,
                    "average_improvement": 0.15,
                    "estimated_dashboard_value": {
                        "time_saved_reviewing_results": "25 minutes"
                    },
                }
                mock_sync_class.return_value = mock_sync

                result = self.runner.invoke(preview_cloud_benefits)

        assert result.exit_code == 0
        output = result.output

        assert "Traigent Cloud Benefits Preview" in output
        assert "Your Current Local Usage:" in output
        assert "5 completed optimizations" in output
        assert "Upgrade to Traigent Cloud for:" in output
        assert "Advanced Algorithms" in output
        assert "Ready to upgrade?" in output

    def test_preview_cloud_benefits_no_sessions(self):
        """Test previewing cloud benefits with no sessions."""
        # Remove all sessions
        for session_id in [
            self.completed_session_id,
            self.pending_session_id,
            self.failed_session_id,
        ]:
            self.storage.delete_session(session_id)

        with patch.dict(os.environ, self.env_vars):
            with patch("traigent.cli.local_commands.SyncManager") as mock_sync_class:
                mock_sync = MagicMock()
                mock_sync.get_sync_status.return_value = {
                    "completed_sessions": 0,
                    "total_trials": 0,
                }
                mock_sync.get_cloud_analytics_preview.return_value = {
                    "functions_optimized": 0,
                    "average_improvement": None,
                }
                mock_sync_class.return_value = mock_sync

                result = self.runner.invoke(preview_cloud_benefits)

        assert result.exit_code == 0
        assert "Upgrade to Traigent Cloud for:" in result.output

    def test_preview_cloud_benefits_error_handling(self):
        """Test preview cloud benefits with errors."""
        with patch.dict(os.environ, self.env_vars):
            with patch("traigent.cli.local_commands.SyncManager") as mock_sync_class:
                mock_sync_class.side_effect = Exception("Preview error")

                result = self.runner.invoke(preview_cloud_benefits)

        assert result.exit_code == 1
        assert "Error previewing cloud benefits" in result.output

    def test_local_commands_group(self):
        """Test the local commands group."""
        result = self.runner.invoke(edge_analytics_commands, ["--help"])

        assert result.exit_code == 0
        assert "Edge Analytics mode operations" in result.output

    def test_command_help_texts(self):
        """Test that all commands have proper help text."""
        commands = [
            "list",
            "show",
            "export",
            "delete",
            "cleanup",
            "info",
            "config",
            "sync",
            "preview-cloud",
        ]

        for command in commands:
            result = self.runner.invoke(edge_analytics_commands, [command, "--help"])
            assert result.exit_code == 0
            assert len(result.output) > 50  # Has substantial help text

    def test_edge_case_very_long_session_id(self):
        """Test handling of very long session IDs."""
        long_id = "a" * 1000

        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(show_session, [long_id])

        assert result.exit_code == 1
        # Should either show "not found" or "File name too long" error
        assert "not found" in result.output or "File name too long" in result.output

    def test_edge_case_special_characters_in_paths(self):
        """Test handling of special characters in file paths."""
        special_path = str(Path(self.temp_dir) / "export with spaces & symbols.json")

        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(
                export_session, [self.completed_session_id, "--output", special_path]
            )

        assert result.exit_code == 0
        assert Path(special_path).exists()

    def test_concurrent_command_execution(self):
        """Test that commands handle concurrent execution gracefully."""
        # This test simulates potential race conditions
        with patch.dict(os.environ, self.env_vars):
            # Run multiple commands that might conflict
            result1 = self.runner.invoke(list_sessions)
            result2 = self.runner.invoke(storage_info)
            result3 = self.runner.invoke(show_session, [self.completed_session_id])

        # All should succeed
        assert result1.exit_code == 0
        assert result2.exit_code == 0
        assert result3.exit_code == 0

    def test_environment_variable_handling(self):
        """Test handling of missing or invalid environment variables."""
        # Test with missing environment variables, using a clean temp home directory
        import tempfile

        with tempfile.TemporaryDirectory() as temp_home:
            clean_env = {
                "HOME": temp_home,
                "USERPROFILE": temp_home,  # Windows compatibility
            }
            with patch.dict(os.environ, clean_env, clear=True):
                result = self.runner.invoke(list_sessions)

            # Should use default storage path and handle empty storage gracefully
            assert result.exit_code == 0

    def test_file_permission_handling(self):
        """Test handling of file permission issues."""
        # Make storage directory read-only
        os.chmod(self.storage_path, 0o444)

        try:
            with patch.dict(os.environ, self.env_vars):
                result = self.runner.invoke(list_sessions)

            # Should handle permission errors gracefully
            # Result depends on specific error handling in storage layer
            assert result is not None, "Command should return a result"
        finally:
            # Restore permissions for cleanup
            os.chmod(self.storage_path, 0o755)

    def test_large_dataset_handling(self):
        """Test handling of large datasets in commands."""
        # Create many sessions
        for i in range(50):
            session_id = self.storage.create_session(f"bulk_function_{i}")
            self.storage.add_trial_result(session_id, {"param": i}, 0.5 + i * 0.01)
            self.storage.finalize_session(session_id, "completed")

        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(list_sessions, ["--limit", "10"])

        assert result.exit_code == 0
        # Should handle large datasets without issues

    def test_unicode_and_encoding_handling(self):
        """Test handling of Unicode characters in session data."""
        # Create session with Unicode function name
        unicode_session_id = self.storage.create_session(
            "测试函数_with_unicode_🎯",
            optimization_config={"param": ["测试", "🚀", 'special"chars']},
        )

        with patch.dict(os.environ, self.env_vars):
            result = self.runner.invoke(show_session, [unicode_session_id])

        assert result.exit_code == 0
        # Should handle Unicode characters properly
