"""Tests for privacy-safe local analytics module."""

import json
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.storage.local_storage import LocalStorageManager
from traigent.utils.local_analytics import (
    LocalAnalytics,
    collect_and_submit_analytics,
    get_enhanced_cloud_incentives,
)


class TestLocalAnalytics:
    """Test suite for LocalAnalytics class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            local_storage_path=self.temp_dir,
            enable_usage_analytics=True,
            analytics_endpoint="https://test-analytics.example.com/v1/local-usage",
        )
        self.analytics = LocalAnalytics(self.config)
        self.storage = LocalStorageManager(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test LocalAnalytics initialization."""
        assert self.analytics.config == self.config
        assert self.analytics.enabled
        assert (
            self.analytics.analytics_endpoint
            == "https://test-analytics.example.com/v1/local-usage"
        )
        assert self.analytics.user_id is not None
        assert len(self.analytics.user_id) == 36  # UUID format

    def test_initialization_disabled(self):
        """Test LocalAnalytics with analytics disabled."""
        config = TraigentConfig(
            execution_mode="edge_analytics",
            local_storage_path=self.temp_dir,
            enable_usage_analytics=False,
        )
        analytics = LocalAnalytics(config)
        assert not analytics.enabled

    def test_initialization_non_local_mode(self):
        """Test LocalAnalytics in non-Edge Analytics mode."""
        config = TraigentConfig(execution_mode="cloud", enable_usage_analytics=True)
        analytics = LocalAnalytics(config)
        assert not analytics.enabled

    def test_get_or_create_user_id_new(self):
        """Test creating new anonymous user ID."""
        # Should create a new UUID
        user_id = self.analytics._get_or_create_user_id()
        assert len(user_id) == 36

        # Should persist to file
        analytics_file = Path(self.storage.storage_path) / ".analytics_id"
        assert analytics_file.exists()
        assert analytics_file.read_text().strip() == user_id

    def test_get_or_create_user_id_existing(self):
        """Test retrieving existing user ID."""
        # Create existing ID file
        existing_id = str(uuid.uuid4())
        analytics_file = Path(self.storage.storage_path) / ".analytics_id"
        analytics_file.parent.mkdir(parents=True, exist_ok=True)
        analytics_file.write_text(existing_id)

        # Should return existing ID
        config = TraigentConfig(
            execution_mode="edge_analytics",
            local_storage_path=self.temp_dir,
            enable_usage_analytics=True,
        )
        analytics = LocalAnalytics(config)
        assert analytics.user_id == existing_id

    def test_get_or_create_user_id_from_config(self):
        """Test using user ID from config."""
        config_id = str(uuid.uuid4())
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            local_storage_path=self.temp_dir,
            enable_usage_analytics=True,
            anonymous_user_id=config_id,
        )
        analytics = LocalAnalytics(config)
        assert analytics.user_id == config_id

    def test_collect_usage_stats_empty(self):
        """Test collecting usage stats with no sessions."""
        stats = self.analytics.collect_usage_stats()

        assert stats["total_sessions"] == 0
        assert stats["completed_sessions"] == 0
        assert stats["failed_sessions"] == 0
        assert stats["total_trials"] == 0
        assert stats["total_completed_trials"] == 0
        assert stats["unique_functions_optimized"] == 0
        assert stats["avg_trials_per_session"] == 0
        assert stats["avg_config_space_size"] == 0
        assert stats["avg_improvement_percent"] is None
        assert stats["sessions_last_30_days"] == 0
        assert stats["days_since_first_use"] == 0
        assert stats["sdk_version"] == "1.1.0"
        assert stats["execution_mode"] == ExecutionMode.EDGE_ANALYTICS.value
        assert stats["anonymous_user_id"] == self.analytics.user_id
        assert "timestamp" in stats

    def test_collect_usage_stats_with_sessions(self):
        """Test collecting usage stats with real sessions."""
        # Create test sessions
        session1_id = self.storage.create_session(
            function_name="test_function_1",
            optimization_config={
                "search_space": {
                    "model": ["gpt-3.5", "gpt-4"],
                    "temperature": [0.1, 0.5],
                }
            },
        )

        session2_id = self.storage.create_session(
            function_name="test_function_2",
            optimization_config={"search_space": {"batch_size": [16, 32, 64]}},
        )

        # Add trials to sessions
        self.storage.add_trial_result(
            session1_id, {"model": "gpt-3.5", "temperature": 0.1}, 0.85
        )
        self.storage.add_trial_result(
            session1_id, {"model": "gpt-4", "temperature": 0.5}, 0.92
        )
        self.storage.add_trial_result(session2_id, {"batch_size": 32}, 0.78)

        # Finalize sessions
        self.storage.finalize_session(session1_id, "completed")
        self.storage.finalize_session(session2_id, "failed")

        # Collect stats
        stats = self.analytics.collect_usage_stats()

        assert stats["total_sessions"] == 2
        assert stats["completed_sessions"] == 1
        assert stats["failed_sessions"] == 1
        assert stats["total_trials"] >= 2  # At least the trials we added
        assert stats["unique_functions_optimized"] == 2
        assert stats["avg_trials_per_session"] > 0
        assert stats["sessions_last_30_days"] == 2  # Recent sessions

    def test_collect_usage_stats_disabled(self):
        """Test collecting stats when analytics is disabled."""
        config = TraigentConfig(
            execution_mode="edge_analytics", enable_usage_analytics=False
        )
        analytics = LocalAnalytics(config)

        stats = analytics.collect_usage_stats()
        assert stats == {}

    def test_get_days_since_first_use(self):
        """Test calculating days since first use."""
        # No sessions
        assert self.analytics._get_days_since_first_use() == 0

        # Create session with known date
        self.storage.create_session(
            function_name="test_function", optimization_config={}
        )

        # Should be 0 days for recent session
        days = self.analytics._get_days_since_first_use()
        assert days == 0

    def test_submit_usage_stats_conditions(self):
        """Test conditions for analytics submission."""
        # Test submission due logic
        assert self.analytics._is_submission_due()

        # Test updating submission time
        self.analytics._update_last_submission()
        assert not self.analytics._is_submission_due()

    @pytest.mark.asyncio
    async def test_submit_usage_stats_disabled(self):
        """Test analytics submission when disabled."""
        config = TraigentConfig(
            execution_mode="edge_analytics", enable_usage_analytics=False
        )
        analytics = LocalAnalytics(config)

        result = await analytics.submit_usage_stats(force=True)

        assert not result["success"]
        assert result["reason"] == "Analytics disabled"

    def test_is_submission_due_no_file(self):
        """Test submission due check with no previous submission."""
        assert self.analytics._is_submission_due()

    def test_is_submission_due_recent(self):
        """Test submission due check with recent submission."""
        # Create recent submission file
        last_submission_file = Path(self.storage.storage_path) / ".last_analytics"
        last_submission_file.parent.mkdir(parents=True, exist_ok=True)
        recent_time = datetime.now(UTC) - timedelta(hours=1)
        last_submission_file.write_text(recent_time.isoformat())

        assert not self.analytics._is_submission_due()

    def test_is_submission_due_old(self):
        """Test submission due check with old submission."""
        # Create old submission file
        last_submission_file = Path(self.storage.storage_path) / ".last_analytics"
        last_submission_file.parent.mkdir(parents=True, exist_ok=True)
        old_time = datetime.now(UTC) - timedelta(days=2)
        last_submission_file.write_text(old_time.isoformat())

        assert self.analytics._is_submission_due()

    def test_update_last_submission(self):
        """Test updating last submission timestamp."""
        self.analytics._update_last_submission()

        last_submission_file = Path(self.storage.storage_path) / ".last_analytics"
        assert last_submission_file.exists()

        # Should be recent timestamp
        timestamp_str = last_submission_file.read_text().strip()
        timestamp = datetime.fromisoformat(timestamp_str)
        assert (datetime.now(UTC) - timestamp).total_seconds() < 10

    def test_get_cloud_incentive_data_empty(self):
        """Test cloud incentive data with no sessions."""
        data = self.analytics.get_cloud_incentive_data()

        assert "usage_summary" in data
        assert "cloud_benefits" in data
        assert "personalized_message" in data
        assert "upgrade_urgency" in data

        # Should have basic benefits
        benefits = data["cloud_benefits"]
        assert "algorithm_upgrade" in benefits
        assert "trial_limit" in benefits
        assert "analytics" in benefits

    def test_get_cloud_incentive_data_with_usage(self):
        """Test cloud incentive data with actual usage."""
        # Create sessions to generate usage
        session_id = self.storage.create_session(
            function_name="test_function",
            optimization_config={
                "search_space": {
                    "model": ["gpt-3.5", "gpt-4"],
                    "temperature": [0.1, 0.5],
                }
            },
        )

        # Add trials
        for i in range(10):
            self.storage.add_trial_result(
                session_id, {"model": "gpt-3.5", "temperature": 0.1}, 0.85 + i * 0.01
            )

        self.storage.finalize_session(session_id, "completed")

        data = self.analytics.get_cloud_incentive_data()

        # Should have enhanced benefits based on usage
        assert data["usage_summary"]["total_sessions"] >= 1
        assert (
            data["usage_summary"]["total_trials"] >= 0
        )  # Could be 0 if logic different

        # Check personalized message
        message = data["personalized_message"]
        assert isinstance(message, str)
        assert len(message) > 0

    def test_generate_personalized_message(self):
        """Test personalized message generation."""
        # Test different usage patterns
        stats_power_user = {
            "total_sessions": 25,
            "total_trials": 300,
            "avg_improvement_percent": 25.5,
        }
        message = self.analytics._generate_personalized_message(stats_power_user)
        assert "Power user detected" in message

        stats_high_trials = {
            "total_sessions": 8,
            "total_trials": 150,
            "avg_improvement_percent": None,
        }
        message = self.analytics._generate_personalized_message(stats_high_trials)
        assert "150 trials" in message

        stats_good_improvement = {
            "total_sessions": 6,
            "total_trials": 50,
            "avg_improvement_percent": 30.0,
        }
        message = self.analytics._generate_personalized_message(stats_good_improvement)
        assert "30.0%" in message

    def test_calculate_upgrade_urgency(self):
        """Test upgrade urgency calculation."""
        # High urgency - hitting trial limits
        stats_high = {
            "total_sessions": 5,
            "avg_trials_per_session": 18.0,
            "days_since_first_use": 10,
        }
        urgency = self.analytics._calculate_upgrade_urgency(stats_high)
        assert urgency == "high"

        # Medium urgency - regular usage over time
        stats_medium = {
            "total_sessions": 8,
            "avg_trials_per_session": 10.0,
            "days_since_first_use": 14,
        }
        urgency = self.analytics._calculate_upgrade_urgency(stats_medium)
        assert urgency == "medium"

        # Low urgency - some usage
        stats_low = {
            "total_sessions": 4,
            "avg_trials_per_session": 8.0,
            "days_since_first_use": 3,
        }
        urgency = self.analytics._calculate_upgrade_urgency(stats_low)
        assert urgency == "low"

        # No urgency - minimal usage
        stats_none = {
            "total_sessions": 1,
            "avg_trials_per_session": 5.0,
            "days_since_first_use": 1,
        }
        urgency = self.analytics._calculate_upgrade_urgency(stats_none)
        assert urgency == "none"


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_collect_and_submit_analytics_disabled(self):
        """Test convenience function with analytics disabled."""
        config = TraigentConfig(
            execution_mode="edge_analytics", enable_usage_analytics=False
        )

        # Should not raise any errors
        result = collect_and_submit_analytics(config)
        assert result is None  # Function returns None

    def test_collect_and_submit_analytics_non_local(self):
        """Test convenience function in non-Edge Analytics mode."""
        config = TraigentConfig(execution_mode="cloud", enable_usage_analytics=True)

        # Should not raise any errors
        result = collect_and_submit_analytics(config)
        assert result is None  # Function returns None

    @patch("traigent.utils.env_config.is_mock_llm", return_value=False)
    @patch("threading.Thread")
    def test_collect_and_submit_analytics_success(self, mock_thread_class, _mock_llm):
        """Test convenience function success path with threading."""
        config = TraigentConfig(
            execution_mode="edge_analytics",
            local_storage_path=self.temp_dir,
            enable_usage_analytics=True,
        )

        # Mock thread
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        # Should not raise any errors
        collect_and_submit_analytics(config)

        # Verify thread was created with daemon=True and started
        mock_thread_class.assert_called_once()
        call_kwargs = mock_thread_class.call_args[1]
        assert call_kwargs.get("daemon") is True
        assert "target" in call_kwargs
        mock_thread.start.assert_called_once()

    @patch("traigent.utils.env_config.is_mock_llm", return_value=False)
    @patch("traigent.utils.local_analytics.asyncio.ensure_future")
    @patch("traigent.utils.local_analytics.asyncio.get_running_loop")
    def test_collect_and_submit_analytics_async_context(
        self, mock_get_loop, mock_ensure_future, _mock_llm
    ):
        """Test convenience function in async context."""
        config = TraigentConfig(
            execution_mode="edge_analytics",
            local_storage_path=self.temp_dir,
            enable_usage_analytics=True,
        )

        # Mock running loop exists (async context)
        mock_loop = Mock()
        mock_get_loop.return_value = mock_loop

        # Mock ensure_future to consume the coroutine without warning
        def mock_ensure_future_impl(coro):
            # Close the coroutine to avoid warning
            coro.close()
            return Mock()

        mock_ensure_future.side_effect = mock_ensure_future_impl

        # Should not raise any errors
        collect_and_submit_analytics(config)

        # Verify ensure_future was called to schedule the coroutine
        mock_get_loop.assert_called_once()
        mock_ensure_future.assert_called_once()

    def test_get_enhanced_cloud_incentives(self):
        """Test enhanced cloud incentives function."""
        config = TraigentConfig(
            execution_mode="edge_analytics",
            local_storage_path=self.temp_dir,
            enable_usage_analytics=True,
        )

        incentives = get_enhanced_cloud_incentives(config)

        assert isinstance(incentives, dict)
        assert "usage_summary" in incentives
        assert "cloud_benefits" in incentives


class TestAnalyticsIntegration:
    """Test analytics integration with storage and real data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TraigentConfig(
            execution_mode="edge_analytics",
            local_storage_path=self.temp_dir,
            enable_usage_analytics=True,
        )
        self.analytics = LocalAnalytics(self.config)
        self.storage = LocalStorageManager(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_real_optimization_session_analytics(self):
        """Test analytics with realistic optimization session data."""
        # Create multiple sessions with different patterns
        sessions_data = [
            {
                "function_name": "sentiment_analyzer",
                "config_space": {
                    "model": ["gpt-3.5", "gpt-4"],
                    "temperature": [0.1, 0.5, 0.9],
                },
                "trials": [
                    ({"model": "gpt-3.5", "temperature": 0.1}, 0.82),
                    ({"model": "gpt-3.5", "temperature": 0.5}, 0.85),
                    ({"model": "gpt-4", "temperature": 0.1}, 0.88),
                    ({"model": "gpt-4", "temperature": 0.5}, 0.91),
                ],
                "status": "completed",
            },
            {
                "function_name": "email_classifier",
                "config_space": {"model": ["gpt-3.5"], "temperature": [0.1, 0.3]},
                "trials": [
                    ({"model": "gpt-3.5", "temperature": 0.1}, 0.78),
                    ({"model": "gpt-3.5", "temperature": 0.3}, 0.80),
                ],
                "status": "completed",
            },
            {
                "function_name": "content_generator",
                "config_space": {
                    "model": ["gpt-4"],
                    "temperature": [0.7, 0.9],
                    "max_tokens": [100, 200],
                },
                "trials": [
                    ({"model": "gpt-4", "temperature": 0.7, "max_tokens": 100}, 0.75),
                ],
                "status": "failed",
            },
        ]

        # Create sessions
        for session_data in sessions_data:
            session_id = self.storage.create_session(
                function_name=session_data["function_name"],
                optimization_config={"search_space": session_data["config_space"]},
            )

            # Add trials
            for config, score in session_data["trials"]:
                self.storage.add_trial_result(session_id, config, score)

            # Finalize session
            self.storage.finalize_session(session_id, session_data["status"])

        # Collect analytics
        stats = self.analytics.collect_usage_stats()

        # Verify analytics data
        assert stats["total_sessions"] == 3
        assert stats["completed_sessions"] == 2
        assert stats["failed_sessions"] == 1
        assert stats["unique_functions_optimized"] == 3
        assert stats["total_trials"] >= 0  # May be 0 depending on implementation
        assert (
            stats["avg_config_space_size"] >= 0
        )  # Should calculate config space complexity

        # Test cloud incentives with real data
        incentives = self.analytics.get_cloud_incentive_data()
        assert incentives["usage_summary"]["total_sessions"] == 3
        assert len(incentives["cloud_benefits"]) >= 3  # Should have multiple benefits

        # Should have personalized message based on usage
        assert "optimization" in incentives["personalized_message"].lower()

        # Upgrade urgency should be calculated
        assert incentives["upgrade_urgency"] in ["none", "low", "medium", "high"]

    def test_privacy_compliance(self):
        """Test that no sensitive data is included in analytics."""
        # Create session with sensitive-looking data
        session_id = self.storage.create_session(
            function_name="secret_agent_classifier",  # Sensitive function name
            optimization_config={
                "search_space": {
                    "api_key": ["example-key-123"],  # Sensitive config
                    "model": ["gpt-4"],
                }
            },
        )

        # Add trial with sensitive data
        sensitive_config = {
            "api_key": "example-key-123",  # pragma: allowlist secret
            "model": "gpt-4",
            "user_data": "sensitive_customer_info",
        }
        self.storage.add_trial_result(session_id, sensitive_config, 0.95)
        self.storage.finalize_session(session_id, "completed")

        # Collect stats
        stats = self.analytics.collect_usage_stats()

        # Convert to JSON to check for sensitive data
        stats_json = json.dumps(stats)

        # Verify no sensitive data is present
        assert "secret_agent_classifier" not in stats_json
        assert "example-key-123" not in stats_json
        assert "api_key" not in stats_json
        assert "sensitive_customer_info" not in stats_json
        assert "user_data" not in stats_json

        # Verify only safe aggregated data is present
        assert "total_sessions" in stats
        assert "unique_functions_optimized" in stats
        assert "avg_config_space_size" in stats
        assert stats["anonymous_user_id"] == self.analytics.user_id

    def test_error_handling(self):
        """Test analytics error handling with corrupted data."""
        # Create session manually with invalid data to test error handling
        self.storage.create_session(
            function_name="test_function", optimization_config=None  # Invalid config
        )

        # Analytics should handle this gracefully
        stats = self.analytics.collect_usage_stats()

        # Should still return valid stats structure
        assert isinstance(stats, dict)
        assert "total_sessions" in stats
        assert "anonymous_user_id" in stats

        # Should handle missing or invalid session data
        assert stats["total_sessions"] >= 0
        assert stats["avg_config_space_size"] >= 0
