"""Unit tests for local_analytics.py.

Tests for privacy-safe analytics collection and cloud incentive generation.
This module collects aggregated, non-sensitive usage statistics in Edge Analytics mode.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Observability CONC-Quality-Performance FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import tempfile
import uuid
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from traigent.api.types import OptimizationStatus
from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.storage.local_storage import LocalStorageManager, OptimizationSession
from traigent.utils.local_analytics import (
    DEFAULT_ANALYTICS_ENDPOINT,
    LocalAnalytics,
    collect_and_submit_analytics,
    get_enhanced_cloud_incentives,
)


class TestLocalAnalyticsInitialization:
    """Tests for LocalAnalytics initialization."""

    @pytest.fixture
    def temp_storage_path(self) -> Generator[str, None, None]:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def edge_analytics_config(self, temp_storage_path: str) -> TraigentConfig:
        """Create edge analytics configuration."""
        return TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
            analytics_endpoint="https://test.traigent.ai/analytics",
            anonymous_user_id=None,
        )

    def test_initialization_with_edge_analytics_enabled(
        self, edge_analytics_config: TraigentConfig
    ) -> None:
        """Test initialization with edge analytics mode enabled."""
        analytics = LocalAnalytics(edge_analytics_config)

        assert analytics.config == edge_analytics_config
        assert analytics.enabled is True
        assert analytics.analytics_endpoint == "https://test.traigent.ai/analytics"
        assert isinstance(analytics.storage, LocalStorageManager)
        assert analytics.user_id is not None
        assert len(analytics.user_id) > 0

    def test_initialization_with_analytics_disabled(
        self, temp_storage_path: str
    ) -> None:
        """Test initialization with analytics disabled."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=False,
            local_storage_path=temp_storage_path,
        )
        analytics = LocalAnalytics(config)

        assert analytics.enabled is False

    def test_initialization_with_cloud_mode(self, temp_storage_path: str) -> None:
        """Test initialization in cloud mode disables local analytics."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.CLOUD.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )
        analytics = LocalAnalytics(config)

        assert analytics.enabled is False

    def test_initialization_with_default_endpoint(self, temp_storage_path: str) -> None:
        """Test initialization uses default analytics endpoint when not specified."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
            analytics_endpoint=None,
        )
        analytics = LocalAnalytics(config)

        assert analytics.analytics_endpoint == DEFAULT_ANALYTICS_ENDPOINT

    def test_initialization_with_provided_user_id(self, temp_storage_path: str) -> None:
        """Test initialization uses provided anonymous user ID."""
        user_id = "test-user-12345"
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
            anonymous_user_id=user_id,
        )
        analytics = LocalAnalytics(config)

        assert analytics.user_id == user_id

    def test_initialization_creates_user_id(self, temp_storage_path: str) -> None:
        """Test initialization creates new anonymous user ID if not provided."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
            anonymous_user_id=None,
        )
        analytics = LocalAnalytics(config)

        # Should be a valid UUID
        assert analytics.user_id is not None
        try:
            uuid.UUID(analytics.user_id)
        except ValueError:
            pytest.fail("User ID should be a valid UUID")

    def test_user_id_persistence(self, temp_storage_path: str) -> None:
        """Test that user ID is persisted and reused."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
            anonymous_user_id=None,
        )

        # Create first instance
        analytics1 = LocalAnalytics(config)
        user_id1 = analytics1.user_id

        # Create second instance with same config
        analytics2 = LocalAnalytics(config)
        user_id2 = analytics2.user_id

        # Should reuse the same user ID
        assert user_id1 == user_id2

    def test_user_id_persistence_file_read_error(self, temp_storage_path: str) -> None:
        """Test user ID creation when analytics file cannot be read."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
            anonymous_user_id=None,
        )

        with patch(
            "pathlib.Path.read_text", side_effect=PermissionError("Access denied")
        ):
            analytics = LocalAnalytics(config)
            # Should create new user ID despite read error
            assert analytics.user_id is not None

    def test_user_id_persistence_file_write_error(self, temp_storage_path: str) -> None:
        """Test user ID creation when analytics file cannot be written."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
            anonymous_user_id=None,
        )

        with patch(
            "pathlib.Path.write_text", side_effect=PermissionError("Access denied")
        ):
            analytics = LocalAnalytics(config)
            # Should still create user ID but fail to persist
            assert analytics.user_id is not None


class TestCollectUsageStats:
    """Tests for collect_usage_stats method."""

    @pytest.fixture
    def analytics_instance(self, temp_storage_path: str) -> LocalAnalytics:
        """Create LocalAnalytics instance for testing."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
            anonymous_user_id="test-user-123",
        )
        return LocalAnalytics(config)

    @pytest.fixture
    def temp_storage_path(self) -> Generator[str, None, None]:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_collect_usage_stats_disabled(self, temp_storage_path: str) -> None:
        """Test that collect_usage_stats returns empty dict when disabled."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=False,
            local_storage_path=temp_storage_path,
        )
        analytics = LocalAnalytics(config)

        stats = analytics.collect_usage_stats()
        assert stats == {}

    def test_collect_usage_stats_with_no_sessions(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test collect_usage_stats with no sessions in storage."""
        with patch.object(analytics_instance.storage, "list_sessions", return_value=[]):
            stats = analytics_instance.collect_usage_stats()

            assert stats["total_sessions"] == 0
            assert stats["completed_sessions"] == 0
            assert stats["failed_sessions"] == 0
            assert stats["total_trials"] == 0
            assert stats["unique_functions_optimized"] == 0
            assert stats["avg_trials_per_session"] == 0
            assert stats["anonymous_user_id"] == "test-user-123"

    def test_collect_usage_stats_with_completed_sessions(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test collect_usage_stats with completed sessions."""
        sessions = [
            OptimizationSession(
                session_id="session1",
                function_name="test_func1",
                created_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
                status=OptimizationStatus.COMPLETED.value,
                total_trials=10,
                completed_trials=10,
                best_score=0.95,
            ),
            OptimizationSession(
                session_id="session2",
                function_name="test_func2",
                created_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
                status=OptimizationStatus.COMPLETED.value,
                total_trials=20,
                completed_trials=20,
                best_score=0.85,
            ),
        ]

        with patch.object(
            analytics_instance.storage, "list_sessions", return_value=sessions
        ):
            with patch.object(
                analytics_instance.storage, "get_session_summary", return_value={}
            ):
                stats = analytics_instance.collect_usage_stats()

                assert stats["total_sessions"] == 2
                assert stats["completed_sessions"] == 2
                assert stats["failed_sessions"] == 0
                assert stats["total_trials"] == 30
                assert stats["total_completed_trials"] == 30
                assert stats["unique_functions_optimized"] == 2
                assert abs(stats["avg_trials_per_session"] - 15.0) < 0.01

    def test_collect_usage_stats_with_failed_sessions(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test collect_usage_stats with failed sessions."""
        sessions = [
            OptimizationSession(
                session_id="session1",
                function_name="test_func",
                created_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
                status=OptimizationStatus.FAILED.value,
                total_trials=5,
                completed_trials=3,
            ),
        ]

        with patch.object(
            analytics_instance.storage, "list_sessions", return_value=sessions
        ):
            stats = analytics_instance.collect_usage_stats()

            assert stats["completed_sessions"] == 0
            assert stats["failed_sessions"] == 1

    def test_collect_usage_stats_with_config_space(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test collect_usage_stats calculates config space size."""
        sessions = [
            OptimizationSession(
                session_id="session1",
                function_name="test_func",
                created_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
                status=OptimizationStatus.COMPLETED.value,
                total_trials=10,
                completed_trials=10,
                optimization_config={
                    "configuration_space": {
                        "param1": [1, 2],
                        "param2": [3, 4],
                        "param3": [5, 6],
                    }
                },
            ),
        ]

        with patch.object(
            analytics_instance.storage, "list_sessions", return_value=sessions
        ):
            with patch.object(
                analytics_instance.storage, "get_session_summary", return_value={}
            ):
                stats = analytics_instance.collect_usage_stats()

                assert abs(stats["avg_config_space_size"] - 3.0) < 0.01

    def test_collect_usage_stats_with_improvements(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test collect_usage_stats calculates improvement percentages."""
        sessions = [
            OptimizationSession(
                session_id="session1",
                function_name="test_func",
                created_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
                status=OptimizationStatus.COMPLETED.value,
                total_trials=10,
                completed_trials=10,
            ),
        ]

        with patch.object(
            analytics_instance.storage, "list_sessions", return_value=sessions
        ):
            with patch.object(
                analytics_instance.storage,
                "get_session_summary",
                return_value={"improvement": 0.25},
            ):
                stats = analytics_instance.collect_usage_stats()

                assert abs(stats["avg_improvement_percent"] - 25.0) < 0.01

    def test_collect_usage_stats_recent_sessions(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test collect_usage_stats filters recent sessions."""
        now = datetime.now(UTC)
        old_date = (now - timedelta(days=60)).isoformat()
        recent_date = (now - timedelta(days=10)).isoformat()

        sessions = [
            OptimizationSession(
                session_id="old_session",
                function_name="test_func",
                created_at=old_date,
                updated_at=old_date,
                status=OptimizationStatus.COMPLETED.value,
                total_trials=10,
                completed_trials=10,
            ),
            OptimizationSession(
                session_id="recent_session",
                function_name="test_func",
                created_at=recent_date,
                updated_at=recent_date,
                status=OptimizationStatus.COMPLETED.value,
                total_trials=15,
                completed_trials=15,
            ),
        ]

        with patch.object(
            analytics_instance.storage, "list_sessions", return_value=sessions
        ):
            with patch.object(
                analytics_instance.storage, "get_session_summary", return_value={}
            ):
                stats = analytics_instance.collect_usage_stats()

                assert stats["total_sessions"] == 2
                assert stats["sessions_last_30_days"] == 1

    def test_collect_usage_stats_error_handling(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test collect_usage_stats handles errors gracefully."""
        with patch.object(
            analytics_instance.storage,
            "list_sessions",
            side_effect=Exception("Storage error"),
        ):
            stats = analytics_instance.collect_usage_stats()

            # Should return empty dict on error
            assert stats == {}

    def test_collect_usage_stats_includes_sdk_version(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test collect_usage_stats includes SDK version."""
        with patch.object(analytics_instance.storage, "list_sessions", return_value=[]):
            stats = analytics_instance.collect_usage_stats()

            assert "sdk_version" in stats
            assert stats["sdk_version"] == "1.1.0"

    def test_collect_usage_stats_includes_execution_mode(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test collect_usage_stats includes execution mode."""
        with patch.object(analytics_instance.storage, "list_sessions", return_value=[]):
            stats = analytics_instance.collect_usage_stats()

            assert stats["execution_mode"] == ExecutionMode.EDGE_ANALYTICS.value

    def test_collect_usage_stats_includes_timestamp(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test collect_usage_stats includes current timestamp."""
        with patch.object(analytics_instance.storage, "list_sessions", return_value=[]):
            before = datetime.now(UTC)
            stats = analytics_instance.collect_usage_stats()
            after = datetime.now(UTC)

            timestamp = datetime.fromisoformat(stats["timestamp"])
            assert before <= timestamp <= after

    def test_collect_usage_stats_no_sensitive_data(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test that collect_usage_stats does not include sensitive information."""
        sessions = [
            OptimizationSession(
                session_id="session1",
                function_name="my_secret_function",
                created_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
                status=OptimizationStatus.COMPLETED.value,
                total_trials=10,
                completed_trials=10,
                metadata={
                    "api_key": "secret-key-123",
                    "user_email": "user@example.com",
                },
            ),
        ]

        with patch.object(
            analytics_instance.storage, "list_sessions", return_value=sessions
        ):
            with patch.object(
                analytics_instance.storage, "get_session_summary", return_value={}
            ):
                stats = analytics_instance.collect_usage_stats()

                # Should not contain function names
                assert "my_secret_function" not in str(stats)
                # Should not contain metadata
                assert "api_key" not in stats
                assert "user_email" not in stats


class TestGetDaysSinceFirstUse:
    """Tests for _get_days_since_first_use method."""

    @pytest.fixture
    def analytics_instance(self, temp_storage_path: str) -> LocalAnalytics:
        """Create LocalAnalytics instance for testing."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )
        return LocalAnalytics(config)

    @pytest.fixture
    def temp_storage_path(self) -> Generator[str, None, None]:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_days_since_first_use_no_sessions(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _get_days_since_first_use with no sessions."""
        with patch.object(analytics_instance.storage, "list_sessions", return_value=[]):
            days = analytics_instance._get_days_since_first_use()
            assert days == 0

    def test_days_since_first_use_with_sessions(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _get_days_since_first_use calculates correctly."""
        now = datetime.now(UTC)
        first_use = now - timedelta(days=15)

        sessions = [
            OptimizationSession(
                session_id="session1",
                function_name="test_func",
                created_at=first_use.isoformat(),
                updated_at=first_use.isoformat(),
                status=OptimizationStatus.COMPLETED.value,
                total_trials=10,
                completed_trials=10,
            ),
            OptimizationSession(
                session_id="session2",
                function_name="test_func",
                created_at=(first_use + timedelta(days=5)).isoformat(),
                updated_at=(first_use + timedelta(days=5)).isoformat(),
                status=OptimizationStatus.COMPLETED.value,
                total_trials=10,
                completed_trials=10,
            ),
        ]

        with patch.object(
            analytics_instance.storage, "list_sessions", return_value=sessions
        ):
            days = analytics_instance._get_days_since_first_use()
            assert days == 15

    def test_days_since_first_use_error_handling(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _get_days_since_first_use handles errors gracefully."""
        with patch.object(
            analytics_instance.storage,
            "list_sessions",
            side_effect=Exception("Storage error"),
        ):
            days = analytics_instance._get_days_since_first_use()
            assert days == 0


class TestSubmitUsageStats:
    """Tests for submit_usage_stats method."""

    @pytest.fixture
    def analytics_instance(self, temp_storage_path: str) -> LocalAnalytics:
        """Create LocalAnalytics instance for testing."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )
        return LocalAnalytics(config)

    @pytest.fixture
    def temp_storage_path(self) -> Generator[str, None, None]:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.mark.asyncio
    async def test_submit_usage_stats_disabled(self, temp_storage_path: str) -> None:
        """Test submit_usage_stats returns error when analytics disabled."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=False,
            local_storage_path=temp_storage_path,
        )
        analytics = LocalAnalytics(config)

        result = await analytics.submit_usage_stats()

        assert result["success"] is False
        assert result["reason"] == "Analytics disabled"

    @pytest.mark.asyncio
    async def test_submit_usage_stats_not_due(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test submit_usage_stats skips when submission not due."""
        with patch.object(analytics_instance, "_is_submission_due", return_value=False):
            result = await analytics_instance.submit_usage_stats()

            assert result["success"] is False
            assert result["reason"] == "Submission not due yet"

    @pytest.mark.asyncio
    async def test_submit_usage_stats_force_submission(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test submit_usage_stats with force=True bypasses due check."""
        with patch.object(analytics_instance, "_is_submission_due", return_value=False):
            with patch.object(
                analytics_instance, "collect_usage_stats", return_value={}
            ):
                result = await analytics_instance.submit_usage_stats(force=True)

                # Should attempt submission even if not due
                assert result["success"] is False
                assert result["reason"] == "No stats to submit"

    @pytest.mark.asyncio
    async def test_submit_usage_stats_no_stats(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test submit_usage_stats handles empty stats."""
        with patch.object(analytics_instance, "_is_submission_due", return_value=True):
            with patch.object(
                analytics_instance, "collect_usage_stats", return_value={}
            ):
                result = await analytics_instance.submit_usage_stats()

                assert result["success"] is False
                assert result["reason"] == "No stats to submit"

    @pytest.mark.asyncio
    async def test_submit_usage_stats_no_api_key(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test submit_usage_stats handles missing API key."""
        with patch.object(analytics_instance, "_is_submission_due", return_value=True):
            with patch.object(
                analytics_instance,
                "collect_usage_stats",
                return_value={"total_sessions": 5},
            ):
                with patch(
                    "traigent.config.backend_config.BackendConfig.get_api_key",
                    return_value=None,
                ):
                    result = await analytics_instance.submit_usage_stats()

                    assert result["success"] is False
                    assert result["reason"] == "No API key available"

    @pytest.mark.asyncio
    async def test_submit_usage_stats_invalid_api_key(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test submit_usage_stats handles invalid API key."""
        with patch.object(analytics_instance, "_is_submission_due", return_value=True):
            with patch.object(
                analytics_instance,
                "collect_usage_stats",
                return_value={"total_sessions": 5},
            ):
                with patch(
                    "traigent.config.backend_config.BackendConfig.get_api_key",
                    return_value="short",
                ):
                    result = await analytics_instance.submit_usage_stats()

                    assert result["success"] is False
                    assert result["reason"] == "API key invalid"

    @pytest.mark.asyncio
    async def test_submit_usage_stats_timeout(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test submit_usage_stats handles timeout error."""
        with patch.object(analytics_instance, "_is_submission_due", return_value=True):
            with patch.object(
                analytics_instance,
                "collect_usage_stats",
                return_value={"total_sessions": 5},
            ):
                with patch(
                    "traigent.config.backend_config.BackendConfig.get_api_key",
                    return_value="tg_" + "a" * 61,
                ):
                    with patch(
                        "traigent.cloud.backend_client.get_backend_client"
                    ) as mock_client:
                        mock_client.return_value.__aenter__.side_effect = TimeoutError()

                        result = await analytics_instance.submit_usage_stats()

                        assert result["success"] is False
                        assert result["reason"] == "Request timeout"

    @pytest.mark.asyncio
    async def test_submit_usage_stats_import_error(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test submit_usage_stats handles ImportError."""
        with patch.object(analytics_instance, "_is_submission_due", return_value=True):
            with patch.object(
                analytics_instance,
                "collect_usage_stats",
                return_value={"total_sessions": 5},
            ):
                with patch(
                    "traigent.config.backend_config.BackendConfig.get_api_key",
                    return_value="tg_" + "a" * 61,
                ):
                    with patch(
                        "traigent.cloud.backend_client.get_backend_client",
                        side_effect=ImportError("Module not found"),
                    ):
                        result = await analytics_instance.submit_usage_stats()

                        assert result["success"] is False
                        assert result["reason"] == "Backend client not available"


class TestSubmissionDueCheck:
    """Tests for _is_submission_due method."""

    @pytest.fixture
    def analytics_instance(self, temp_storage_path: str) -> LocalAnalytics:
        """Create LocalAnalytics instance for testing."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )
        return LocalAnalytics(config)

    @pytest.fixture
    def temp_storage_path(self) -> Generator[str, None, None]:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_is_submission_due_no_previous_submission(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _is_submission_due returns True when no previous submission."""
        assert analytics_instance._is_submission_due() is True

    def test_is_submission_due_recent_submission(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _is_submission_due returns False for recent submission."""
        # Simulate recent submission
        last_submission_file = (
            Path(analytics_instance.storage.storage_path) / ".last_analytics"
        )
        last_submission_file.write_text(datetime.now(UTC).isoformat())

        assert analytics_instance._is_submission_due() is False

    def test_is_submission_due_old_submission(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _is_submission_due returns True for old submission."""
        # Simulate old submission (more than 1 day ago)
        old_date = datetime.now(UTC) - timedelta(days=2)
        last_submission_file = (
            Path(analytics_instance.storage.storage_path) / ".last_analytics"
        )
        last_submission_file.write_text(old_date.isoformat())

        assert analytics_instance._is_submission_due() is True

    def test_is_submission_due_naive_datetime(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _is_submission_due handles naive datetime correctly."""
        # Simulate submission with naive datetime (no timezone)
        old_date = datetime.now() - timedelta(days=2)
        last_submission_file = (
            Path(analytics_instance.storage.storage_path) / ".last_analytics"
        )
        last_submission_file.write_text(old_date.isoformat())

        # Should handle gracefully and return True
        assert analytics_instance._is_submission_due() is True

    def test_is_submission_due_file_read_error(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _is_submission_due handles file read errors."""
        last_submission_file = (
            Path(analytics_instance.storage.storage_path) / ".last_analytics"
        )
        last_submission_file.write_text("invalid-date-format")

        # Should return True on error
        assert analytics_instance._is_submission_due() is True


class TestUpdateLastSubmission:
    """Tests for _update_last_submission method."""

    @pytest.fixture
    def analytics_instance(self, temp_storage_path: str) -> LocalAnalytics:
        """Create LocalAnalytics instance for testing."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )
        return LocalAnalytics(config)

    @pytest.fixture
    def temp_storage_path(self) -> Generator[str, None, None]:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_update_last_submission(self, analytics_instance: LocalAnalytics) -> None:
        """Test _update_last_submission creates/updates timestamp file."""
        before = datetime.now(UTC)
        analytics_instance._update_last_submission()
        after = datetime.now(UTC)

        last_submission_file = (
            Path(analytics_instance.storage.storage_path) / ".last_analytics"
        )
        assert last_submission_file.exists()

        timestamp = datetime.fromisoformat(last_submission_file.read_text().strip())
        assert before <= timestamp <= after

    def test_update_last_submission_file_write_error(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _update_last_submission handles file write errors gracefully."""
        with patch(
            "pathlib.Path.write_text", side_effect=PermissionError("Access denied")
        ):
            # Should not raise exception
            analytics_instance._update_last_submission()


class TestCloudIncentiveData:
    """Tests for get_cloud_incentive_data method."""

    @pytest.fixture
    def analytics_instance(self, temp_storage_path: str) -> LocalAnalytics:
        """Create LocalAnalytics instance for testing."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )
        return LocalAnalytics(config)

    @pytest.fixture
    def temp_storage_path(self) -> Generator[str, None, None]:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_get_cloud_incentive_data_no_stats(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test get_cloud_incentive_data returns empty dict when no stats."""
        with patch.object(analytics_instance, "collect_usage_stats", return_value={}):
            incentive_data = analytics_instance.get_cloud_incentive_data()
            assert incentive_data == {}

    def test_get_cloud_incentive_data_basic_usage(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test get_cloud_incentive_data with basic usage data."""
        mock_stats = {
            "total_sessions": 5,
            "total_trials": 50,
            "avg_trials_per_session": 10.0,
            "avg_config_space_size": 3.0,
            "days_since_first_use": 7,
        }

        with patch.object(
            analytics_instance, "collect_usage_stats", return_value=mock_stats
        ):
            incentive_data = analytics_instance.get_cloud_incentive_data()

            assert "usage_summary" in incentive_data
            assert "cloud_benefits" in incentive_data
            assert "personalized_message" in incentive_data
            assert "upgrade_urgency" in incentive_data

            # Check basic benefits are present
            assert "algorithm_upgrade" in incentive_data["cloud_benefits"]
            assert "trial_limit" in incentive_data["cloud_benefits"]
            assert "analytics" in incentive_data["cloud_benefits"]

    def test_get_cloud_incentive_data_power_user(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test get_cloud_incentive_data for power users."""
        mock_stats = {
            "total_sessions": 15,
            "total_trials": 300,
            "avg_trials_per_session": 20.0,
            "avg_config_space_size": 8.0,
            "days_since_first_use": 30,
        }

        with patch.object(
            analytics_instance, "collect_usage_stats", return_value=mock_stats
        ):
            incentive_data = analytics_instance.get_cloud_incentive_data()

            # Should include team collaboration benefit
            assert "team_collaboration" in incentive_data["cloud_benefits"]
            # Should include complex optimization benefit
            assert "complex_optimization" in incentive_data["cloud_benefits"]

    def test_generate_personalized_message_power_user(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _generate_personalized_message for power users."""
        stats = {
            "total_sessions": 25,
            "total_trials": 100,
            "avg_improvement_percent": None,
        }
        message = analytics_instance._generate_personalized_message(stats)
        assert "Power user" in message or "20" in message

    def test_generate_personalized_message_high_trials(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _generate_personalized_message for high trial counts."""
        stats = {
            "total_sessions": 5,
            "total_trials": 150,
            "avg_improvement_percent": None,
        }
        message = analytics_instance._generate_personalized_message(stats)
        assert "trials" in message.lower()

    def test_generate_personalized_message_good_improvement(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _generate_personalized_message for good improvements."""
        stats = {
            "total_sessions": 5,
            "total_trials": 50,
            "avg_improvement_percent": 25.0,
        }
        message = analytics_instance._generate_personalized_message(stats)
        assert "improvement" in message.lower() or "results" in message.lower()

    def test_generate_personalized_message_moderate_usage(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _generate_personalized_message for moderate usage."""
        stats = {
            "total_sessions": 7,
            "total_trials": 70,
            "avg_improvement_percent": None,
        }
        message = analytics_instance._generate_personalized_message(stats)
        assert len(message) > 0

    def test_generate_personalized_message_new_user(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _generate_personalized_message for new users."""
        stats = {
            "total_sessions": 1,
            "total_trials": 10,
            "avg_improvement_percent": None,
        }
        message = analytics_instance._generate_personalized_message(stats)
        assert "start" in message.lower() or "cloud" in message.lower()

    def test_calculate_upgrade_urgency_high_trials(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _calculate_upgrade_urgency with high trial counts."""
        stats = {
            "total_sessions": 5,
            "avg_trials_per_session": 18.0,
            "days_since_first_use": 5,
        }
        urgency = analytics_instance._calculate_upgrade_urgency(stats)
        assert urgency == "high"

    def test_calculate_upgrade_urgency_heavy_usage(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _calculate_upgrade_urgency with heavy usage."""
        stats = {
            "total_sessions": 20,
            "avg_trials_per_session": 10.0,
            "days_since_first_use": 10,
        }
        urgency = analytics_instance._calculate_upgrade_urgency(stats)
        assert urgency == "high"

    def test_calculate_upgrade_urgency_medium(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _calculate_upgrade_urgency for medium urgency."""
        stats = {
            "total_sessions": 7,
            "avg_trials_per_session": 10.0,
            "days_since_first_use": 10,
        }
        urgency = analytics_instance._calculate_upgrade_urgency(stats)
        assert urgency == "medium"

    def test_calculate_upgrade_urgency_low(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _calculate_upgrade_urgency for low urgency."""
        stats = {
            "total_sessions": 4,
            "avg_trials_per_session": 5.0,
            "days_since_first_use": 3,
        }
        urgency = analytics_instance._calculate_upgrade_urgency(stats)
        assert urgency == "low"

    def test_calculate_upgrade_urgency_none(
        self, analytics_instance: LocalAnalytics
    ) -> None:
        """Test _calculate_upgrade_urgency for no urgency."""
        stats = {
            "total_sessions": 1,
            "avg_trials_per_session": 5.0,
            "days_since_first_use": 1,
        }
        urgency = analytics_instance._calculate_upgrade_urgency(stats)
        assert urgency == "none"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def temp_storage_path(self) -> Generator[str, None, None]:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_collect_and_submit_analytics_disabled(
        self, temp_storage_path: str
    ) -> None:
        """Test collect_and_submit_analytics when analytics disabled."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=False,
            local_storage_path=temp_storage_path,
        )

        # Should not raise exception
        collect_and_submit_analytics(config)

    def test_collect_and_submit_analytics_wrong_mode(
        self, temp_storage_path: str
    ) -> None:
        """Test collect_and_submit_analytics in wrong execution mode."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.CLOUD.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )

        # Should not raise exception
        collect_and_submit_analytics(config)

    def test_collect_and_submit_analytics_in_async_context(
        self, temp_storage_path: str
    ) -> None:
        """Test collect_and_submit_analytics from async context."""
        import asyncio

        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )

        async def test_async() -> None:
            # Should not raise exception even in async context
            collect_and_submit_analytics(config)
            # Add await to make this properly async
            await asyncio.sleep(0)

        asyncio.run(test_async())

    def test_collect_and_submit_analytics_error_handling(
        self, temp_storage_path: str
    ) -> None:
        """Test collect_and_submit_analytics handles errors gracefully."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )

        with patch(
            "traigent.utils.local_analytics.LocalAnalytics",
            side_effect=Exception("Init error"),
        ):
            # Should not raise exception
            collect_and_submit_analytics(config)

    def test_get_enhanced_cloud_incentives(self, temp_storage_path: str) -> None:
        """Test get_enhanced_cloud_incentives function."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )

        with patch(
            "traigent.utils.local_analytics.LocalAnalytics.get_cloud_incentive_data",
            return_value={"test": "data"},
        ):
            result = get_enhanced_cloud_incentives(config)
            assert result == {"test": "data"}


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture
    def temp_storage_path(self) -> Generator[str, None, None]:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_collect_usage_stats_with_invalid_improvement_type(
        self, temp_storage_path: str
    ) -> None:
        """Test collect_usage_stats handles invalid improvement data types."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )
        analytics = LocalAnalytics(config)

        sessions = [
            OptimizationSession(
                session_id="session1",
                function_name="test_func",
                created_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
                status=OptimizationStatus.COMPLETED.value,
                total_trials=10,
                completed_trials=10,
            ),
        ]

        with patch.object(analytics.storage, "list_sessions", return_value=sessions):
            with patch.object(
                analytics.storage,
                "get_session_summary",
                return_value={"improvement": "not-a-number"},
            ):
                stats = analytics.collect_usage_stats()

                # Should handle invalid type gracefully
                assert stats["avg_improvement_percent"] is None

    def test_collect_usage_stats_with_none_improvement(
        self, temp_storage_path: str
    ) -> None:
        """Test collect_usage_stats handles None improvement values."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )
        analytics = LocalAnalytics(config)

        sessions = [
            OptimizationSession(
                session_id="session1",
                function_name="test_func",
                created_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
                status=OptimizationStatus.COMPLETED.value,
                total_trials=10,
                completed_trials=10,
            ),
        ]

        with patch.object(analytics.storage, "list_sessions", return_value=sessions):
            with patch.object(
                analytics.storage,
                "get_session_summary",
                return_value={"improvement": None},
            ):
                stats = analytics.collect_usage_stats()

                # Should handle None gracefully
                assert stats["avg_improvement_percent"] is None

    def test_collect_usage_stats_with_invalid_config_space(
        self, temp_storage_path: str
    ) -> None:
        """Test collect_usage_stats handles invalid config space."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )
        analytics = LocalAnalytics(config)

        sessions = [
            OptimizationSession(
                session_id="session1",
                function_name="test_func",
                created_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
                status=OptimizationStatus.COMPLETED.value,
                total_trials=10,
                completed_trials=10,
                optimization_config={"configuration_space": "not-a-dict"},
            ),
        ]

        with patch.object(analytics.storage, "list_sessions", return_value=sessions):
            with patch.object(
                analytics.storage, "get_session_summary", return_value={}
            ):
                stats = analytics.collect_usage_stats()

                # Should handle invalid config space gracefully
                assert abs(stats["avg_config_space_size"] - 0.0) < 0.01

    def test_collect_usage_stats_with_missing_optimization_config(
        self, temp_storage_path: str
    ) -> None:
        """Test collect_usage_stats when optimization_config is None."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=True,
            local_storage_path=temp_storage_path,
        )
        analytics = LocalAnalytics(config)

        sessions = [
            OptimizationSession(
                session_id="session1",
                function_name="test_func",
                created_at=datetime.now(UTC).isoformat(),
                updated_at=datetime.now(UTC).isoformat(),
                status=OptimizationStatus.COMPLETED.value,
                total_trials=10,
                completed_trials=10,
                optimization_config=None,
            ),
        ]

        with patch.object(analytics.storage, "list_sessions", return_value=sessions):
            with patch.object(
                analytics.storage, "get_session_summary", return_value={}
            ):
                stats = analytics.collect_usage_stats()

                # Should handle missing config gracefully
                assert abs(stats["avg_config_space_size"] - 0.0) < 0.01
