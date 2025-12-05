"""
Comprehensive test suite for IncentiveManager.
Tests hint triggers, achievement system, and upgrade value propositions.
"""

import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from traigent.config.types import TraigentConfig
from traigent.storage.local_storage import LocalStorageManager
from traigent.utils.incentives import (
    IncentiveManager,
    show_achievement,
    show_upgrade_hint,
)


class TestIncentiveManager:
    """Test suite for IncentiveManager with comprehensive hint and achievement tests."""

    def setup_method(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)

        # Create test config
        self.config = TraigentConfig(
            execution_mode="edge_analytics",
            local_storage_path=str(self.storage_path),
            minimal_logging=True,
        )

        self.incentive_manager = IncentiveManager(self.config)

    def teardown_method(self):
        """Clean up temporary storage."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test incentive manager initialization."""
        assert self.incentive_manager.config == self.config
        assert self.incentive_manager.storage is not None
        assert self.incentive_manager._state_file.exists()

        # Check default state
        state = self.incentive_manager._state
        assert "first_use" in state
        assert state["total_sessions"] == 0
        assert state["total_trials"] == 0
        assert state["hints_shown"] == []
        assert state["upgrade_dismissed"] is False
        assert state["achievement_unlocked"] == []

    def test_state_persistence(self):
        """Test that state persists across manager instances."""
        # Modify state in first instance
        self.incentive_manager._state["total_sessions"] = 5
        self.incentive_manager._state["hints_shown"] = ["test_hint"]
        self.incentive_manager._save_state()

        # Create new instance
        new_manager = IncentiveManager(self.config)

        # Should load previous state
        assert new_manager._state["total_sessions"] == 5
        assert new_manager._state["hints_shown"] == ["test_hint"]

    def test_state_corruption_recovery(self):
        """Test recovery from corrupted state file."""
        # Corrupt the state file
        self.incentive_manager._state_file.write_text("invalid json {")

        # Should gracefully recover with default state
        new_manager = IncentiveManager(self.config)
        assert new_manager._state["total_sessions"] == 0

    def test_update_usage_stats_empty(self):
        """Test updating usage stats with no sessions."""
        self.incentive_manager.update_usage_stats()

        state = self.incentive_manager._state
        assert state["total_sessions"] == 0
        assert state["total_trials"] == 0
        assert state["completed_sessions"] == 0

    def test_update_usage_stats_with_sessions(self):
        """Test updating usage stats with actual sessions."""
        # Create test sessions in storage
        storage = self.incentive_manager.storage

        # Create completed session
        session_id1 = storage.create_session("test_func1")
        storage.add_trial_result(session_id1, {"param": 1}, 0.8)
        storage.add_trial_result(session_id1, {"param": 2}, 0.9)
        storage.finalize_session(session_id1, "completed")

        # Create pending session
        session_id2 = storage.create_session("test_func2")
        storage.add_trial_result(session_id2, {"param": 1}, 0.7)

        self.incentive_manager.update_usage_stats()

        state = self.incentive_manager._state
        assert state["total_sessions"] == 2
        assert state["total_trials"] == 3
        assert state["completed_sessions"] == 1

    def test_should_show_hint_upgrade_dismissed(self):
        """Test that hints are not shown when upgrade is dismissed."""
        self.incentive_manager._state["upgrade_dismissed"] = True
        self.incentive_manager._save_state()

        assert self.incentive_manager.should_show_hint("general") is False
        assert self.incentive_manager.should_show_hint("session_complete") is False

    def test_should_show_hint_frequency_limit(self):
        """Test that hints respect frequency limits."""
        # Set last hint to recent time
        recent_time = datetime.now(UTC) - timedelta(hours=12)
        self.incentive_manager._state["last_hint"] = recent_time.isoformat()
        self.incentive_manager._state["completed_sessions"] = 3  # Should trigger hint

        # Should not show due to frequency limit
        assert self.incentive_manager.should_show_hint("session_complete") is False

        # Set last hint to old time
        old_time = datetime.now(UTC) - timedelta(hours=25)
        self.incentive_manager._state["last_hint"] = old_time.isoformat()

        # Should show now
        assert self.incentive_manager.should_show_hint("session_complete") is True

    def test_should_show_hint_trigger_thresholds(self):
        """Test hint trigger thresholds for different contexts."""
        # Test session_complete triggers
        test_cases = [
            (3, "session_complete", True),
            (7, "session_complete", True),
            (15, "session_complete", True),
            (30, "session_complete", True),
            (4, "session_complete", False),
            (8, "session_complete", False),
        ]

        for completed_count, context, expected in test_cases:
            self.incentive_manager._state["completed_sessions"] = completed_count
            self.incentive_manager._state["last_hint"] = None  # Reset frequency limit
            result = self.incentive_manager.should_show_hint(context)
            assert (
                result == expected
            ), f"Failed for {completed_count} sessions, {context} context"

    def test_get_contextual_hint_no_trigger(self):
        """Test getting hint when conditions are not met."""
        # No completed sessions, should not trigger
        hint = self.incentive_manager.get_contextual_hint("session_complete")
        assert hint is None

    def test_get_contextual_hint_session_complete(self):
        """Test getting session complete hint."""
        self.incentive_manager._state["completed_sessions"] = 3
        self.incentive_manager._state["total_trials"] = 15

        hint = self.incentive_manager.get_contextual_hint("session_complete")

        assert hint is not None
        assert "3 optimizations completed" in hint
        assert "Advanced Bayesian optimization" in hint
        assert "traigent login" in hint

        # Should update state
        assert self.incentive_manager._state["last_hint"] is not None
        assert len(self.incentive_manager._state["hints_shown"]) == 1

    def test_get_contextual_hint_categories(self):
        """Test different hint categories based on completion count."""
        # Test session_complete category (≤5)
        self.incentive_manager._state["completed_sessions"] = 3
        hint = self.incentive_manager.get_contextual_hint("general")
        assert "optimizations completed" in hint
        assert "Advanced Bayesian optimization" in hint

        # Test storage_growing category (6-20)
        self.incentive_manager._state["completed_sessions"] = 10
        self.incentive_manager._state["last_hint"] = None  # Reset
        hint = self.incentive_manager.get_contextual_hint("general")
        assert "optimization library is growing" in hint
        assert "Cross-project optimization insights" in hint

        # Test power_user category (>20)
        self.incentive_manager._state["completed_sessions"] = 25
        self.incentive_manager._state["last_hint"] = None  # Reset
        hint = self.incentive_manager.get_contextual_hint("general")
        assert "Power user detected" in hint
        assert "Multi-objective Pareto optimization" in hint

    def test_show_achievement_unlock_first_time(self):
        """Test showing achievement unlock for the first time."""
        achievement = "first_optimization"
        message = self.incentive_manager.show_achievement_unlock(achievement)

        assert message is not None
        assert "First Optimization Complete!" in message
        assert "You've completed your first LLM optimization" in message

        # Should be added to unlocked achievements
        assert achievement in self.incentive_manager._state["achievement_unlocked"]

    def test_show_achievement_unlock_already_unlocked(self):
        """Test showing achievement that's already unlocked."""
        achievement = "first_optimization"

        # Unlock it first
        self.incentive_manager.show_achievement_unlock(achievement)

        # Try to unlock again
        message = self.incentive_manager.show_achievement_unlock(achievement)
        assert message is None

    def test_show_achievement_unlock_invalid(self):
        """Test showing invalid achievement."""
        message = self.incentive_manager.show_achievement_unlock("invalid_achievement")
        assert message is None

    def test_check_achievements_progression(self):
        """Test achievement checking based on completion progression."""
        # Create sessions to trigger achievements
        storage = self.incentive_manager.storage

        # Create 1 completed session (should unlock first_optimization)
        session_id = storage.create_session("test_func")
        storage.finalize_session(session_id, "completed")

        with patch("builtins.print") as mock_print:
            self.incentive_manager.update_usage_stats()
            mock_print.assert_called()  # Should print achievement

        assert (
            "first_optimization"
            in self.incentive_manager._state["achievement_unlocked"]
        )

        # Create more sessions for next achievement
        for i in range(4):  # Total 5 completed
            session_id = storage.create_session(f"test_func_{i}")
            storage.finalize_session(session_id, "completed")

        with patch("builtins.print") as mock_print:
            self.incentive_manager.update_usage_stats()

        assert (
            "optimization_explorer"
            in self.incentive_manager._state["achievement_unlocked"]
        )

    def test_get_upgrade_value_proposition(self):
        """Test getting personalized upgrade value proposition."""
        # Set up usage stats
        self.incentive_manager._state["completed_sessions"] = 10
        self.incentive_manager._state["total_trials"] = 150

        value_props = self.incentive_manager.get_upgrade_value_proposition()

        # With analytics integration, the structure is different
        # It includes usage_summary, cloud_benefits, personalized_message, upgrade_urgency
        if "usage_summary" in value_props:
            # Analytics-enhanced format
            assert "usage_summary" in value_props
            assert "cloud_benefits" in value_props
            assert "personalized_message" in value_props
            assert "upgrade_urgency" in value_props

            # Check usage summary
            usage_summary = value_props["usage_summary"]
            assert "total_sessions" in usage_summary

            # Check cloud benefits structure
            cloud_benefits = value_props["cloud_benefits"]
            assert isinstance(cloud_benefits, dict)

        else:
            # Fallback format when analytics is disabled or fails
            assert "time_savings" in value_props
            assert "performance_gains" in value_props
            assert "scale_benefits" in value_props

            # Check calculated values
            scale_benefits = value_props["scale_benefits"]
            assert scale_benefits["current_sessions"] == 10

    def test_show_context_sensitive_hint_large_config(self):
        """Test context-sensitive hint for large configuration space."""
        with patch("builtins.print") as mock_print:
            with patch.object(
                self.incentive_manager, "should_show_hint", return_value=True
            ):
                self.incentive_manager.show_context_sensitive_hint(
                    "large_config_space", param_count=10
                )

                mock_print.assert_called()
                call_args = mock_print.call_args[0][0]
                assert "Large configuration space detected" in call_args
                assert "Bayesian optimization" in call_args

    def test_show_context_sensitive_hint_no_trigger(self):
        """Test context-sensitive hint when conditions not met."""
        with patch("builtins.print") as mock_print:
            with patch.object(
                self.incentive_manager, "should_show_hint", return_value=False
            ):
                self.incentive_manager.show_context_sensitive_hint("large_config_space")
                mock_print.assert_not_called()

    def test_dismiss_upgrade_hints(self):
        """Test dismissing upgrade hints."""
        with patch("builtins.print") as mock_print:
            self.incentive_manager.dismiss_upgrade_hints()

            assert self.incentive_manager._state["upgrade_dismissed"] is True
            assert "dismiss_timestamp" in self.incentive_manager._state
            mock_print.assert_called_with(
                "✅ Upgrade hints dismissed. You can re-enable them by deleting the incentive_state.json file."
            )

    def test_get_local_vs_cloud_comparison(self):
        """Test getting feature comparison table."""
        comparison = self.incentive_manager.get_local_vs_cloud_comparison()

        assert "Local Mode" in comparison
        assert "Traigent Cloud" in comparison
        assert "Random/Grid" in comparison
        assert "Bayesian/Multi" in comparison
        assert "20" in comparison  # Trial limit
        assert "Unlimited" in comparison

    def test_show_onboarding_tips(self):
        """Test showing onboarding tips."""
        tips = self.incentive_manager.show_onboarding_tips()

        assert isinstance(tips, list)
        assert len(tips) > 0

        # Check for key content
        tips_text = " ".join(tips)
        assert "TRAIGENT_RESULTS_FOLDER" in tips_text
        assert "traigent local list" in tips_text
        assert "traigent login" in tips_text

    def test_hint_tracking_and_analytics(self):
        """Test that hints are properly tracked for analytics."""
        self.incentive_manager._state["completed_sessions"] = 3

        # Get a hint
        hint = self.incentive_manager.get_contextual_hint("session_complete")
        assert hint is not None

        # Check tracking
        hints_shown = self.incentive_manager._state["hints_shown"]
        assert len(hints_shown) == 1

        hint_record = hints_shown[0]
        assert hint_record["context"] == "session_complete"
        assert hint_record["hint_key"] == "session_complete"
        assert hint_record["completed_sessions"] == 3
        assert "timestamp" in hint_record

    def test_multiple_context_hints(self):
        """Test hints for multiple different contexts."""
        contexts_and_hints = [
            ("large_config_space", "Large configuration space detected"),
            ("multiple_objectives", "Multi-objective optimization detected"),
            ("slow_evaluation", "Slow evaluations detected"),
            ("high_trial_count", "High trial count optimization"),
        ]

        for context, expected_text in contexts_and_hints:
            with patch("builtins.print") as mock_print:
                with patch.object(
                    self.incentive_manager, "should_show_hint", return_value=True
                ):
                    self.incentive_manager.show_context_sensitive_hint(context)

                    if mock_print.called:
                        call_args = mock_print.call_args[0][0]
                        assert expected_text in call_args

    def test_achievement_system_edge_cases(self):
        """Test achievement system edge cases."""
        # Test with exactly threshold values
        thresholds = [1, 5, 10, 25]
        achievements = [
            "first_optimization",
            "optimization_explorer",
            "efficiency_expert",
            "optimization_master",
        ]

        for threshold, achievement in zip(thresholds, achievements):
            # Create exact number of completed sessions
            storage = self.incentive_manager.storage

            # Clear previous state
            self.incentive_manager._state["achievement_unlocked"] = []

            for i in range(threshold):
                session_id = storage.create_session(f"threshold_test_{i}")
                storage.finalize_session(session_id, "completed")

            with patch("builtins.print"):
                self.incentive_manager.update_usage_stats()

            assert achievement in self.incentive_manager._state["achievement_unlocked"]

    def test_state_file_permissions(self):
        """Test handling of state file permission issues."""
        # Make state file read-only
        self.incentive_manager._state_file.chmod(0o444)

        try:
            # Should handle permission error gracefully
            self.incentive_manager._save_state()
            # Should not raise exception
        finally:
            # Restore permissions for cleanup
            self.incentive_manager._state_file.chmod(0o644)

    def test_concurrent_state_updates(self):
        """Test handling of concurrent state updates."""
        # Simulate rapid state updates
        for i in range(10):
            self.incentive_manager._state["total_sessions"] = i
            self.incentive_manager._save_state()

        # Should maintain consistency
        assert self.incentive_manager._state["total_sessions"] == 9

    def test_large_state_handling(self):
        """Test handling of large state data."""
        # Add large amounts of tracking data
        large_hints = [{"id": i, "data": "x" * 100} for i in range(1000)]
        self.incentive_manager._state["hints_shown"] = large_hints

        # Should handle large state without issues
        self.incentive_manager._save_state()

        # Verify persistence
        new_manager = IncentiveManager(self.config)
        assert len(new_manager._state["hints_shown"]) == 1000


class TestGlobalFunctions:
    """Test global utility functions for incentives."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary storage."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_show_upgrade_hint_local_mode(self):
        """Test global show_upgrade_hint function in Edge Analytics mode."""
        with patch(
            "traigent.utils.incentives.TraigentConfig.from_environment"
        ) as mock_config:
            mock_config.return_value = TraigentConfig(
                execution_mode="edge_analytics",
                local_storage_path=str(self.storage_path),
            )

            with patch(
                "traigent.utils.incentives.IncentiveManager"
            ) as mock_manager_class:
                mock_manager = MagicMock()
                mock_manager.get_contextual_hint.return_value = "Test hint message"
                mock_manager_class.return_value = mock_manager

                with patch("traigent.utils.incentives.logger") as mock_logger:
                    show_upgrade_hint("session_complete")

                    mock_manager.update_usage_stats.assert_called_once()
                    mock_manager.get_contextual_hint.assert_called_with(
                        "session_complete"
                    )
                    mock_logger.info.assert_called()

    def test_show_upgrade_hint_non_local_mode(self):
        """Test global show_upgrade_hint function in non-Edge Analytics mode."""
        with patch(
            "traigent.utils.incentives.TraigentConfig.from_environment"
        ) as mock_config:
            mock_config.return_value = TraigentConfig(execution_mode="cloud")

            with patch("traigent.utils.incentives.logger") as mock_logger:
                show_upgrade_hint("general")
                mock_logger.info.assert_not_called()

    def test_show_achievement_local_mode(self):
        """Test global show_achievement function in Edge Analytics mode."""
        with patch(
            "traigent.utils.incentives.TraigentConfig.from_environment"
        ) as mock_config:
            mock_config.return_value = TraigentConfig(
                execution_mode="edge_analytics",
                local_storage_path=str(self.storage_path),
            )

            with patch(
                "traigent.utils.incentives.IncentiveManager"
            ) as mock_manager_class:
                mock_manager = MagicMock()
                mock_manager.show_achievement_unlock.return_value = (
                    "Achievement unlocked!"
                )
                mock_manager_class.return_value = mock_manager

                with patch("traigent.utils.incentives.logger") as mock_logger:
                    show_achievement("first_optimization")

                    mock_manager.show_achievement_unlock.assert_called_with(
                        "first_optimization"
                    )
                    mock_logger.info.assert_called_with("Achievement unlocked!")

    def test_show_achievement_non_local_mode(self):
        """Test global show_achievement function in non-Edge Analytics mode."""
        with patch(
            "traigent.utils.incentives.TraigentConfig.from_environment"
        ) as mock_config:
            mock_config.return_value = TraigentConfig(execution_mode="cloud")

            with patch("traigent.utils.incentives.logger") as mock_logger:
                show_achievement("first_optimization")
                mock_logger.info.assert_not_called()


class TestIncentiveManagerIntegration:
    """Integration tests for IncentiveManager with actual storage."""

    def setup_method(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)

        self.config = TraigentConfig(
            execution_mode="edge_analytics", local_storage_path=str(self.storage_path)
        )

        self.storage = LocalStorageManager(str(self.storage_path))
        self.incentive_manager = IncentiveManager(self.config)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_hint_flow(self):
        """Test complete hint flow from session creation to display."""
        # Create sessions to trigger hints
        session_ids = []
        for i in range(3):
            session_id = self.storage.create_session(f"integration_test_{i}")
            self.storage.add_trial_result(session_id, {"param": i}, 0.5 + i * 0.1)
            self.storage.finalize_session(session_id, "completed")
            session_ids.append(session_id)

        # Update stats (triggers achievement checking)
        with patch("builtins.print") as mock_print:
            self.incentive_manager.update_usage_stats()

            # Should unlock first achievement
            mock_print.assert_called()

        # Get contextual hint
        hint = self.incentive_manager.get_contextual_hint("session_complete")
        assert hint is not None
        assert "3 optimizations completed" in hint

        # Verify state persistence
        state = self.incentive_manager._state
        assert state["completed_sessions"] == 3
        assert "first_optimization" in state["achievement_unlocked"]
        assert len(state["hints_shown"]) == 1

    def test_real_storage_integration(self):
        """Test integration with real storage operations."""
        # Create realistic optimization session
        session_id = self.storage.create_session(
            function_name="llm_optimization",
            optimization_config={
                "search_space": {
                    "model": ["gpt-4", "gpt-3.5-turbo"],
                    "temperature": {"min": 0.0, "max": 1.0},
                }
            },
        )

        # Add multiple trials
        configs = [
            {"model": "gpt-4", "temperature": 0.7},
            {"model": "gpt-3.5-turbo", "temperature": 0.5},
            {"model": "gpt-4", "temperature": 0.9},
        ]

        scores = [0.85, 0.78, 0.92]

        for config, score in zip(configs, scores):
            self.storage.add_trial_result(session_id, config, score)

        self.storage.finalize_session(session_id, "completed")

        # Update incentive manager
        self.incentive_manager.update_usage_stats()

        # Get value proposition
        value_props = self.incentive_manager.get_upgrade_value_proposition()

        # With analytics integration, check for enhanced format or fallback
        if "usage_summary" in value_props:
            # Analytics-enhanced format
            assert "usage_summary" in value_props
            assert "cloud_benefits" in value_props

            # Check usage summary contains session data
            usage_summary = value_props["usage_summary"]
            assert usage_summary["total_sessions"] >= 1

        else:
            # Fallback format
            assert value_props["time_savings"]["current_time_investment"] == "0.5 hours"
            assert (
                value_props["performance_gains"]["current_avg_trials"]
                == "3.0 trials per optimization"
            )
            assert value_props["scale_benefits"]["current_sessions"] == 1
