"""
Comprehensive test suite for LocalStorageManager.
Covers edge cases, error handling, and file I/O operations.
"""

import json
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from traigent.storage.local_storage import (
    LocalStorageManager,
    OptimizationSession,
    TrialResult,
)
from traigent.utils.exceptions import TraigentStorageError


class TestLocalStorageManager:
    """Test suite for LocalStorageManager with comprehensive edge cases."""

    def setup_method(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)
        self.storage = LocalStorageManager(str(self.storage_path))

    def teardown_method(self):
        """Clean up temporary storage."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_default_path(self):
        """Test storage manager initialization with default path."""
        storage = LocalStorageManager()
        assert storage.storage_path.exists()
        assert storage.storage_path.name == ".traigent"

    def test_initialization_custom_path(self):
        """Test storage manager initialization with custom path."""
        assert self.storage.storage_path == self.storage_path
        assert (self.storage_path / "sessions").exists()
        assert (self.storage_path / "cache").exists()

    def test_initialization_env_variable(self):
        """Test storage path from environment variable."""
        env_path = str(Path(self.temp_dir) / "custom_env_path")
        with patch.dict("os.environ", {"TRAIGENT_RESULTS_FOLDER": env_path}):
            storage = LocalStorageManager()
            assert str(storage.storage_path) == env_path

    def test_create_session_basic(self):
        """Test basic session creation."""
        session_id = self.storage.create_session(
            function_name="test_func",
            optimization_config={"test": "config"},
            metadata={"key": "value"},
        )

        assert isinstance(session_id, str)
        assert len(session_id) > 0

        # Verify session file exists
        session_file = self.storage_path / "sessions" / f"{session_id}.json"
        assert session_file.exists()

    def test_create_session_minimal_params(self):
        """Test session creation with minimal parameters."""
        session_id = self.storage.create_session("minimal_func")

        session = self.storage.load_session(session_id)
        assert session.function_name == "minimal_func"
        assert session.optimization_config is None
        assert session.metadata == {}
        assert session.status == "pending"

    def test_create_session_with_metadata(self):
        """Test session creation with comprehensive metadata."""
        metadata = {
            "version": "1.0.0",
            "environment": "test",
            "user": "test_user",
            "nested": {"data": [1, 2, 3]},
        }

        session_id = self.storage.create_session(
            function_name="meta_func",
            optimization_config={"param": "value"},
            metadata=metadata,
        )

        session = self.storage.load_session(session_id)
        assert session.metadata == metadata

    def test_load_session_existing(self):
        """Test loading an existing session."""
        session_id = self.storage.create_session("load_test")
        loaded_session = self.storage.load_session(session_id)

        assert loaded_session is not None
        assert loaded_session.session_id == session_id
        assert loaded_session.function_name == "load_test"

    def test_load_session_nonexistent(self):
        """Test loading a non-existent session."""
        fake_id = "non_existent_session"
        session = self.storage.load_session(fake_id)
        assert session is None

    def test_load_session_corrupted_file(self):
        """Test loading session with corrupted JSON file."""
        session_id = str(uuid.uuid4())
        session_file = self.storage_path / "sessions" / f"{session_id}.json"

        # Create corrupted JSON file
        session_file.write_text("invalid json content {")

        session = self.storage.load_session(session_id)
        assert session is None

    def test_add_trial_result_success(self):
        """Test adding successful trial result."""
        session_id = self.storage.create_session("trial_test")

        self.storage.add_trial_result(
            session_id=session_id,
            config={"param1": "value1"},
            score=0.85,
            metadata={"duration": 1.5},
        )

        session = self.storage.load_session(session_id)
        assert len(session.trials) == 1

        trial = session.trials[0]
        assert trial.config == {"param1": "value1"}
        assert trial.score == 0.85
        assert trial.error is None
        assert trial.metadata == {"duration": 1.5}

    def test_add_trial_result_with_error(self):
        """Test adding trial result with error."""
        session_id = self.storage.create_session("error_test")

        self.storage.add_trial_result(
            session_id=session_id,
            config={"param": "bad_value"},
            score=0.0,
            error="Configuration failed validation",
        )

        session = self.storage.load_session(session_id)
        trial = session.trials[0]
        assert trial.error == "Configuration failed validation"
        assert trial.score == 0.0

    def test_add_trial_result_nonexistent_session(self):
        """Test adding trial to non-existent session."""
        with pytest.raises(TraigentStorageError, match="Session .* not found"):
            self.storage.add_trial_result(session_id="fake_id", config={}, score=0.5)

    def test_add_trial_result_updates_best(self):
        """Test that best score and config are updated correctly."""
        session_id = self.storage.create_session("best_test")

        # Add first trial
        self.storage.add_trial_result(session_id, {"x": 1}, 0.7)
        session = self.storage.load_session(session_id)
        assert session.best_score == 0.7
        assert session.best_config == {"x": 1}

        # Add better trial
        self.storage.add_trial_result(session_id, {"x": 2}, 0.9)
        session = self.storage.load_session(session_id)
        assert session.best_score == 0.9
        assert session.best_config == {"x": 2}

        # Add worse trial (shouldn't update best)
        self.storage.add_trial_result(session_id, {"x": 3}, 0.6)
        session = self.storage.load_session(session_id)
        assert session.best_score == 0.9
        assert session.best_config == {"x": 2}

    def test_finalize_session_success(self):
        """Test successful session finalization."""
        session_id = self.storage.create_session("finalize_test")
        self.storage.add_trial_result(session_id, {"x": 1}, 0.8)

        finalized = self.storage.finalize_session(session_id, "completed")

        assert finalized.status == "completed"
        assert finalized.updated_at != finalized.created_at

    def test_finalize_session_nonexistent(self):
        """Test finalizing non-existent session."""
        result = self.storage.finalize_session("fake_id", "completed")
        assert result is None

    def test_list_sessions_empty(self):
        """Test listing sessions when none exist."""
        sessions = self.storage.list_sessions()
        assert sessions == []

    def test_list_sessions_multiple(self):
        """Test listing multiple sessions."""
        session_ids = []
        for i in range(3):
            session_id = self.storage.create_session(f"func_{i}")
            session_ids.append(session_id)

        sessions = self.storage.list_sessions()
        assert len(sessions) == 3

        # Should be sorted by creation time (newest first)
        assert sessions[0].session_id == session_ids[2]
        assert sessions[1].session_id == session_ids[1]
        assert sessions[2].session_id == session_ids[0]

    def test_list_sessions_with_status_filter(self):
        """Test listing sessions with status filter."""
        # Create sessions with different statuses
        completed_id = self.storage.create_session("completed_func")
        self.storage.finalize_session(completed_id, "completed")

        pending_id = self.storage.create_session("pending_func")

        # Test filtering
        completed_sessions = self.storage.list_sessions(status="completed")
        assert len(completed_sessions) == 1
        assert completed_sessions[0].session_id == completed_id

        pending_sessions = self.storage.list_sessions(status="pending")
        assert len(pending_sessions) == 1
        assert pending_sessions[0].session_id == pending_id

    def test_get_session_summary_complete(self):
        """Test getting summary of completed session."""
        session_id = self.storage.create_session("summary_test")

        # Add multiple trials
        self.storage.add_trial_result(session_id, {"x": 1}, 0.6)
        self.storage.add_trial_result(session_id, {"x": 2}, 0.8, error="minor error")
        self.storage.add_trial_result(session_id, {"x": 3}, 0.9)

        # Set baseline score
        session = self.storage.load_session(session_id)
        session.baseline_score = 0.5
        self.storage._save_session(session)

        self.storage.finalize_session(session_id, "completed")

        summary = self.storage.get_session_summary(session_id)

        assert summary["session_id"] == session_id
        assert summary["function_name"] == "summary_test"
        assert summary["status"] == "completed"
        assert summary["completed_trials"] == 3
        assert summary["successful_trials"] == 2  # One had error
        assert summary["best_score"] == 0.9
        assert summary["improvement"] == 0.8  # (0.9 - 0.5) / 0.5

    def test_get_session_summary_nonexistent(self):
        """Test getting summary of non-existent session."""
        summary = self.storage.get_session_summary("fake_id")
        assert summary is None

    def test_delete_session_success(self):
        """Test successful session deletion."""
        session_id = self.storage.create_session("delete_test")

        # Verify session exists
        assert self.storage.load_session(session_id) is not None

        # Delete session
        result = self.storage.delete_session(session_id)
        assert result is True

        # Verify session no longer exists
        assert self.storage.load_session(session_id) is None

    def test_delete_session_nonexistent(self):
        """Test deleting non-existent session."""
        result = self.storage.delete_session("fake_id")
        assert result is False

    def test_cleanup_old_sessions(self):
        """Test cleaning up old sessions."""
        # Create sessions with different ages
        old_session_id = self.storage.create_session("old_func")
        new_session_id = self.storage.create_session("new_func")

        # Manually modify creation time for old session
        old_session = self.storage.load_session(old_session_id)
        old_time = (datetime.now(timezone.utc) - timedelta(days=35)).isoformat()
        old_session.created_at = old_time
        self.storage._save_session(old_session)

        # Clean up sessions older than 30 days
        deleted_count = self.storage.cleanup_old_sessions(30)

        assert deleted_count == 1
        assert self.storage.load_session(old_session_id) is None
        assert self.storage.load_session(new_session_id) is not None

    def test_export_session_json(self):
        """Test exporting session to JSON file."""
        session_id = self.storage.create_session("export_test")
        self.storage.add_trial_result(session_id, {"x": 1}, 0.8)

        export_path = Path(self.temp_dir) / "exported_session.json"
        result = self.storage.export_session(session_id, str(export_path), "json")

        assert result is True
        assert export_path.exists()

        # Verify exported content
        with open(export_path) as f:
            exported_data = json.load(f)

        assert exported_data["session_id"] == session_id
        assert exported_data["function_name"] == "export_test"
        assert len(exported_data["trials"]) == 1

    def test_export_session_unsupported_format(self):
        """Test exporting session with unsupported format."""
        session_id = self.storage.create_session("export_test")
        export_path = Path(self.temp_dir) / "exported.xml"

        result = self.storage.export_session(session_id, str(export_path), "xml")
        assert result is False

    def test_get_storage_info(self):
        """Test getting storage information."""
        # Create some test data
        session_id = self.storage.create_session("info_test")
        self.storage.add_trial_result(session_id, {"x": 1}, 0.8)

        info = self.storage.get_storage_info()

        assert "storage_path" in info
        assert "total_sessions" in info
        assert "total_trials" in info
        assert "storage_size_mb" in info

        assert info["total_sessions"] == 1
        assert info["total_trials"] == 1
        assert info["storage_size_mb"] > 0

    def test_file_io_error_handling(self):
        """Test handling of file I/O errors."""
        # Test with read-only directory
        readonly_dir = Path(self.temp_dir) / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        try:
            # Should raise when trying to create directories
            with pytest.raises(
                TraigentStorageError, match="Failed to create storage directories"
            ):
                LocalStorageManager(str(readonly_dir))
        finally:
            readonly_dir.chmod(0o755)  # Restore permissions for cleanup

    def test_concurrent_access_simulation(self):
        """Test handling of simulated concurrent access."""
        session_id = self.storage.create_session("concurrent_test")

        # Simulate multiple rapid trial additions
        for i in range(10):
            self.storage.add_trial_result(session_id, {"trial": i}, 0.5 + i * 0.05)

        session = self.storage.load_session(session_id)
        assert len(session.trials) == 10
        assert session.best_score == 0.95  # Last trial had highest score

    def test_edge_case_large_data(self):
        """Test handling of large configuration and metadata."""
        large_config = {f"param_{i}": f"value_{i}" for i in range(1000)}
        large_metadata = {f"meta_{i}": list(range(100)) for i in range(10)}

        session_id = self.storage.create_session(
            "large_data_test", optimization_config=large_config, metadata=large_metadata
        )

        # Should handle large data without issues
        session = self.storage.load_session(session_id)
        assert len(session.optimization_config) == 1000
        assert len(session.metadata) == 10

    def test_edge_case_special_characters(self):
        """Test handling of special characters in data."""
        special_config = {
            "unicode": "测试数据 🎯",
            "newlines": "line1\nline2\r\nline3",
            "quotes": "both \"double\" and 'single' quotes",
            "json_chars": '{"nested": "json", "array": [1,2,3]}',
        }

        session_id = self.storage.create_session(
            "special_chars_test", optimization_config=special_config
        )

        session = self.storage.load_session(session_id)
        assert session.optimization_config == special_config

    def test_edge_case_empty_values(self):
        """Test handling of empty and None values."""
        session_id = self.storage.create_session(
            "", optimization_config={}, metadata={}  # Empty function name
        )

        # Add trial with empty config
        self.storage.add_trial_result(session_id, {}, 0.0)

        session = self.storage.load_session(session_id)
        assert session.function_name == ""
        assert session.optimization_config == {}
        assert len(session.trials) == 1

    def test_session_persistence_across_instances(self):
        """Test that sessions persist across storage manager instances."""
        # Create session with first instance
        session_id = self.storage.create_session("persistence_test")
        self.storage.add_trial_result(session_id, {"x": 1}, 0.8)

        # Create new storage manager instance
        new_storage = LocalStorageManager(str(self.storage_path))

        # Should be able to load session
        session = new_storage.load_session(session_id)
        assert session is not None
        assert session.function_name == "persistence_test"
        assert len(session.trials) == 1


class TestOptimizationSession:
    """Test OptimizationSession dataclass."""

    def test_session_creation(self):
        """Test creating OptimizationSession instance."""
        session = OptimizationSession(
            session_id="test_id",
            function_name="test_func",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            status="pending",
            total_trials=0,
            completed_trials=0,
            optimization_config={"param": "value"},
            metadata={"key": "value"},
        )

        assert session.session_id == "test_id"
        assert session.function_name == "test_func"
        assert session.status == "pending"
        assert session.trials == []
        assert session.best_score is None
        assert session.best_config is None

    def test_session_serialization(self):
        """Test session to_dict and from_dict methods."""
        original_session = OptimizationSession(
            session_id="serialize_test",
            function_name="serialize_func",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            status="pending",
            total_trials=0,
            completed_trials=0,
            optimization_config={"test": "config"},
            metadata={"meta": "data"},
        )

        # Convert to dict
        session_dict = original_session.to_dict()

        # Convert back to session
        restored_session = OptimizationSession.from_dict(session_dict)

        assert restored_session.session_id == original_session.session_id
        assert restored_session.function_name == original_session.function_name
        assert (
            restored_session.optimization_config == original_session.optimization_config
        )
        assert restored_session.metadata == original_session.metadata


class TestTrialResult:
    """Test TrialResult dataclass."""

    def test_trial_creation(self):
        """Test creating TrialResult instance."""
        trial = TrialResult(
            trial_id="trial_123",
            config={"param": "value"},
            score=0.85,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"duration": 1.5},
        )

        assert trial.trial_id == "trial_123"
        assert trial.config == {"param": "value"}
        assert trial.score == 0.85
        assert trial.error is None
        assert trial.metadata == {"duration": 1.5}

    def test_trial_with_error(self):
        """Test creating TrialResult with error."""
        trial = TrialResult(
            trial_id="error_trial",
            config={"param": "bad_value"},
            score=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            error="Configuration validation failed",
        )

        assert trial.error == "Configuration validation failed"
        assert trial.score == 0.0

    def test_trial_serialization(self):
        """Test trial to_dict and from_dict methods."""
        original_trial = TrialResult(
            trial_id="serialize_trial",
            config={"test": "config"},
            score=0.75,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"test": "metadata"},
        )

        # Convert to dict
        trial_dict = original_trial.to_dict()

        # Convert back to trial
        restored_trial = TrialResult.from_dict(trial_dict)

        assert restored_trial.trial_id == original_trial.trial_id
        assert restored_trial.config == original_trial.config
        assert restored_trial.score == original_trial.score
        assert restored_trial.metadata == original_trial.metadata
