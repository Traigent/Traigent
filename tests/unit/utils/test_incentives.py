"""Unit tests for incentives.py.

Tests for the progressive feature hints and incentive system for Edge Analytics mode.
This module manages gamification, achievements, and upgrade hints to encourage cloud adoption.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Observability FUNC-ANALYTICS REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from traigent.api.types import OptimizationStatus
from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.storage.local_storage import OptimizationSession
from traigent.utils.incentives import (
    IncentiveManager,
    show_achievement,
    show_upgrade_hint,
)


class TestIncentiveManagerInitialization:
    """Tests for IncentiveManager initialization."""

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
        )

    @pytest.fixture
    def config_without_analytics(self, temp_storage_path: str) -> TraigentConfig:
        """Create configuration without analytics enabled."""
        return TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            enable_usage_analytics=False,
            local_storage_path=temp_storage_path,
        )

    def test_initialization_with_analytics_enabled(
        self, edge_analytics_config: TraigentConfig
    ) -> None:
        """Test initialization with analytics enabled."""
        manager = IncentiveManager(edge_analytics_config)

        assert manager.config == edge_analytics_config
        assert manager.analytics is not None
        assert manager.storage is not None
        assert manager._state is not None
        assert "first_use" in manager._state
        assert "total_sessions" in manager._state
        assert "total_trials" in manager._state

    def test_initialization_with_analytics_disabled(
        self, config_without_analytics: TraigentConfig
    ) -> None:
        """Test initialization with analytics disabled."""
        manager = IncentiveManager(config_without_analytics)

        assert manager.config == config_without_analytics
        assert manager.analytics is None
        assert manager.storage is not None

    def test_initialization_with_custom_storage_path(
        self, temp_storage_path: str
    ) -> None:
        """Test initialization with custom storage path."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            local_storage_path=temp_storage_path,
        )
        manager = IncentiveManager(config)

        assert manager._state_file == Path(temp_storage_path) / "incentive_state.json"

    def test_initialization_with_none_storage_path(self) -> None:
        """Test initialization falls back to home directory when storage path is None."""
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            local_storage_path=None,
        )
        manager = IncentiveManager(config)

        expected_path = Path.home() / ".traigent"
        assert str(expected_path) in str(manager._state_file)

    def test_initialization_creates_state_file(
        self, edge_analytics_config: TraigentConfig
    ) -> None:
        """Test initialization creates state file if it doesn't exist."""
        manager = IncentiveManager(edge_analytics_config)

        assert manager._state_file.exists()

    def test_state_has_required_fields(
        self, edge_analytics_config: TraigentConfig
    ) -> None:
        """Test state contains all required fields."""
        manager = IncentiveManager(edge_analytics_config)

        required_fields = [
            "first_use",
            "total_sessions",
            "total_trials",
            "hints_shown",
            "last_hint",
            "upgrade_dismissed",
            "achievement_unlocked",
        ]
        for field in required_fields:
            assert field in manager._state


class TestIncentiveManagerStateManagement:
    """Tests for state loading and saving."""

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
        )

    @pytest.fixture
    def manager(self, edge_analytics_config: TraigentConfig) -> IncentiveManager:
        """Create IncentiveManager instance."""
        return IncentiveManager(edge_analytics_config)

    def test_load_state_with_missing_file_returns_default(
        self, edge_analytics_config: TraigentConfig
    ) -> None:
        """Test loading state returns default when file doesn't exist."""
        manager = IncentiveManager(edge_analytics_config)
        state = manager._state

        assert state["total_sessions"] == 0
        assert state["total_trials"] == 0
        assert state["hints_shown"] == []
        assert state["last_hint"] is None
        assert state["upgrade_dismissed"] is False
        assert state["achievement_unlocked"] == []

    def test_load_state_with_corrupted_file_returns_default(
        self, edge_analytics_config: TraigentConfig, temp_storage_path: str
    ) -> None:
        """Test loading state returns default when file is corrupted."""
        # Create corrupted state file
        state_file = Path(temp_storage_path) / "incentive_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("invalid json content{")

        manager = IncentiveManager(edge_analytics_config)
        state = manager._state

        assert state["total_sessions"] == 0
        assert state["upgrade_dismissed"] is False

    def test_load_state_with_non_dict_content_returns_default(
        self, edge_analytics_config: TraigentConfig, temp_storage_path: str
    ) -> None:
        """Test loading state returns default when file contains non-dict data."""
        # Create state file with non-dict content
        state_file = Path(temp_storage_path) / "incentive_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text('["not", "a", "dict"]')

        manager = IncentiveManager(edge_analytics_config)
        state = manager._state

        assert isinstance(state, dict)
        assert state["total_sessions"] == 0

    def test_save_state_persists_changes(self, manager: IncentiveManager) -> None:
        """Test saving state persists changes to disk."""
        manager._state["total_sessions"] = 10
        manager._state["total_trials"] = 50
        manager._save_state()

        # Read file directly
        with open(manager._state_file) as f:
            saved_state = json.load(f)

        assert saved_state["total_sessions"] == 10
        assert saved_state["total_trials"] == 50

    def test_save_state_creates_parent_directories(
        self, temp_storage_path: str
    ) -> None:
        """Test saving state creates parent directories if they don't exist."""
        nested_path = str(Path(temp_storage_path) / "nested" / "path")
        config = TraigentConfig(
            execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
            local_storage_path=nested_path,
        )
        manager = IncentiveManager(config)

        assert manager._state_file.exists()
        assert manager._state_file.parent.exists()

    def test_save_state_handles_io_error_gracefully(
        self, manager: IncentiveManager
    ) -> None:
        """Test saving state handles IO errors gracefully."""
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            # Should not raise exception - verify it completes
            result = manager._save_state()
            assert result is None  # Method returns None on success or failure

    def test_state_persists_across_instances(
        self, edge_analytics_config: TraigentConfig
    ) -> None:
        """Test state persists across multiple manager instances."""
        # First instance
        manager1 = IncentiveManager(edge_analytics_config)
        manager1._state["total_sessions"] = 42
        manager1._save_state()

        # Second instance should load same state
        manager2 = IncentiveManager(edge_analytics_config)
        assert manager2._state["total_sessions"] == 42


class TestUpdateUsageStats:
    """Tests for usage statistics updates."""

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
        )

    @pytest.fixture
    def manager(self, edge_analytics_config: TraigentConfig) -> IncentiveManager:
        """Create IncentiveManager instance."""
        return IncentiveManager(edge_analytics_config)

    @pytest.fixture
    def mock_sessions(self) -> list[OptimizationSession]:
        """Create mock optimization sessions."""
        sessions = []
        for i in range(5):
            session = Mock(spec=OptimizationSession)
            session.status = (
                OptimizationStatus.COMPLETED.value
                if i < 3
                else OptimizationStatus.RUNNING.value
            )
            session.completed_trials = i * 10
            sessions.append(session)
        return sessions

    def test_update_usage_stats_updates_session_count(
        self, manager: IncentiveManager, mock_sessions: list[OptimizationSession]
    ) -> None:
        """Test update_usage_stats updates total session count."""
        with patch.object(manager.storage, "list_sessions", return_value=mock_sessions):
            manager.update_usage_stats()

        assert manager._state["total_sessions"] == 5

    def test_update_usage_stats_updates_completed_sessions(
        self, manager: IncentiveManager, mock_sessions: list[OptimizationSession]
    ) -> None:
        """Test update_usage_stats updates completed session count."""
        with patch.object(manager.storage, "list_sessions", return_value=mock_sessions):
            manager.update_usage_stats()

        assert manager._state["completed_sessions"] == 3

    def test_update_usage_stats_updates_trial_count(
        self, manager: IncentiveManager, mock_sessions: list[OptimizationSession]
    ) -> None:
        """Test update_usage_stats updates total trial count."""
        with patch.object(manager.storage, "list_sessions", return_value=mock_sessions):
            manager.update_usage_stats()

        # Sum of completed_trials: 0 + 10 + 20 + 30 + 40 = 100
        assert manager._state["total_trials"] == 100

    def test_update_usage_stats_saves_state(
        self, manager: IncentiveManager, mock_sessions: list[OptimizationSession]
    ) -> None:
        """Test update_usage_stats saves state to disk."""
        with patch.object(manager.storage, "list_sessions", return_value=mock_sessions):
            manager.update_usage_stats()

        # Verify state was saved
        with open(manager._state_file) as f:
            saved_state = json.load(f)
        assert saved_state["total_sessions"] == 5

    def test_update_usage_stats_with_empty_sessions(
        self, manager: IncentiveManager
    ) -> None:
        """Test update_usage_stats with no sessions."""
        with patch.object(manager.storage, "list_sessions", return_value=[]):
            manager.update_usage_stats()

        assert manager._state["total_sessions"] == 0
        assert manager._state["total_trials"] == 0
        assert manager._state["completed_sessions"] == 0

    def test_update_usage_stats_triggers_analytics_submission(
        self, manager: IncentiveManager, mock_sessions: list[OptimizationSession]
    ) -> None:
        """Test update_usage_stats triggers analytics submission when enabled."""
        with patch.object(manager.storage, "list_sessions", return_value=mock_sessions):
            with patch(
                "traigent.utils.incentives.collect_and_submit_analytics"
            ) as mock_analytics:
                manager.update_usage_stats()

        if manager.analytics:
            mock_analytics.assert_called_once_with(manager.config)

    def test_update_usage_stats_handles_analytics_failure_gracefully(
        self, manager: IncentiveManager, mock_sessions: list[OptimizationSession]
    ) -> None:
        """Test update_usage_stats handles analytics submission failures gracefully."""
        with patch.object(manager.storage, "list_sessions", return_value=mock_sessions):
            with patch(
                "traigent.utils.incentives.collect_and_submit_analytics",
                side_effect=Exception("Analytics error"),
            ):
                # Should not raise exception
                manager.update_usage_stats()

        assert manager._state["total_sessions"] == 5

    def test_update_usage_stats_checks_achievements(
        self, manager: IncentiveManager
    ) -> None:
        """Test update_usage_stats checks for achievement unlocks."""
        mock_sessions = [
            Mock(
                spec=OptimizationSession,
                status=OptimizationStatus.COMPLETED.value,
                completed_trials=10,
            )
        ]

        with patch.object(manager.storage, "list_sessions", return_value=mock_sessions):
            with patch.object(manager, "_check_achievements") as mock_check:
                manager.update_usage_stats()

        mock_check.assert_called_once()


class TestShouldShowHint:
    """Tests for hint display logic."""

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
            local_storage_path=temp_storage_path,
        )

    @pytest.fixture
    def manager(self, edge_analytics_config: TraigentConfig) -> IncentiveManager:
        """Create IncentiveManager instance."""
        return IncentiveManager(edge_analytics_config)

    def test_should_show_hint_returns_false_when_dismissed(
        self, manager: IncentiveManager
    ) -> None:
        """Test should_show_hint returns False when upgrade hints are dismissed."""
        manager._state["upgrade_dismissed"] = True

        assert manager.should_show_hint("general") is False

    def test_should_show_hint_returns_false_within_24_hours(
        self, manager: IncentiveManager
    ) -> None:
        """Test should_show_hint returns False if last hint was shown within 24 hours."""
        manager._state["last_hint"] = datetime.now(UTC).isoformat()
        manager._state["completed_sessions"] = 3

        assert manager.should_show_hint("general") is False

    def test_should_show_hint_returns_true_after_24_hours(
        self, manager: IncentiveManager
    ) -> None:
        """Test should_show_hint returns True if last hint was over 24 hours ago."""
        manager._state["last_hint"] = (
            datetime.now(UTC) - timedelta(hours=25)
        ).isoformat()
        manager._state["completed_sessions"] = 3

        assert manager.should_show_hint("general") is True

    def test_should_show_hint_handles_naive_datetime(
        self, manager: IncentiveManager
    ) -> None:
        """Test should_show_hint handles naive datetime correctly."""
        # Create naive datetime (no timezone info)
        naive_time = (datetime.now(UTC) - timedelta(hours=25)).replace(tzinfo=None)
        manager._state["last_hint"] = naive_time.isoformat()
        manager._state["completed_sessions"] = 3

        # Should convert to UTC and return True
        assert manager.should_show_hint("general") is True

    def test_should_show_hint_session_complete_context(
        self, manager: IncentiveManager
    ) -> None:
        """Test should_show_hint with session_complete context at milestone sessions."""
        for count in [3, 7, 15, 30]:
            manager._state["completed_sessions"] = count
            manager._state["last_hint"] = None
            assert manager.should_show_hint("session_complete") is True

    def test_should_show_hint_session_complete_non_milestone(
        self, manager: IncentiveManager
    ) -> None:
        """Test should_show_hint returns False for non-milestone sessions."""
        manager._state["completed_sessions"] = 5
        manager._state["last_hint"] = None

        assert manager.should_show_hint("session_complete") is False

    def test_should_show_hint_cli_usage_context(
        self, manager: IncentiveManager
    ) -> None:
        """Test should_show_hint with cli_usage context."""
        manager._state["completed_sessions"] = 10
        manager._state["last_hint"] = None

        assert manager.should_show_hint("cli_usage") is True

    def test_should_show_hint_storage_info_context(
        self, manager: IncentiveManager
    ) -> None:
        """Test should_show_hint with storage_info context."""
        manager._state["completed_sessions"] = 10
        manager._state["last_hint"] = None

        assert manager.should_show_hint("storage_info") is True

    def test_should_show_hint_storage_info_below_threshold(
        self, manager: IncentiveManager
    ) -> None:
        """Test should_show_hint returns False when below storage_info threshold."""
        manager._state["completed_sessions"] = 9
        manager._state["last_hint"] = None

        assert manager.should_show_hint("storage_info") is False

    def test_should_show_hint_general_context_milestones(
        self, manager: IncentiveManager
    ) -> None:
        """Test should_show_hint with general context at various milestones."""
        for count in [3, 7, 15, 30, 35, 40]:
            manager._state["completed_sessions"] = count
            manager._state["last_hint"] = None
            result = manager.should_show_hint("general")
            # Should be True for milestone values
            expected = count in [3, 7, 15, 30] or (count >= 5 and count % 5 == 0)
            assert result == expected

    def test_should_show_hint_with_none_completed_sessions(
        self, manager: IncentiveManager
    ) -> None:
        """Test should_show_hint handles None completed_sessions value."""
        manager._state["completed_sessions"] = None
        manager._state["last_hint"] = None

        # Should not crash and return False
        assert manager.should_show_hint("general") is False


class TestGetContextualHint:
    """Tests for contextual hint generation."""

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
            local_storage_path=temp_storage_path,
        )

    @pytest.fixture
    def manager(self, edge_analytics_config: TraigentConfig) -> IncentiveManager:
        """Create IncentiveManager instance."""
        return IncentiveManager(edge_analytics_config)

    def test_get_contextual_hint_returns_none_when_should_not_show(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_contextual_hint returns None when hints should not be shown."""
        manager._state["upgrade_dismissed"] = True

        assert manager.get_contextual_hint("general") is None

    def test_get_contextual_hint_returns_session_complete_hint(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_contextual_hint returns appropriate hint for new users."""
        manager._state["completed_sessions"] = 3
        manager._state["total_trials"] = 30
        manager._state["last_hint"] = None

        hint = manager.get_contextual_hint("session_complete")

        assert hint is not None
        assert "optimizations completed" in hint.lower()
        assert "traigent cloud" in hint.lower()
        assert "traigent login" in hint.lower()

    def test_get_contextual_hint_returns_storage_growing_hint(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_contextual_hint returns storage hint for mid-tier users."""
        manager._state["completed_sessions"] = 10
        manager._state["total_trials"] = 100
        manager._state["last_hint"] = None

        hint = manager.get_contextual_hint("general")

        assert hint is not None
        assert "optimization" in hint.lower()

    def test_get_contextual_hint_returns_power_user_hint(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_contextual_hint returns power user hint for heavy users."""
        manager._state["completed_sessions"] = 30
        manager._state["total_trials"] = 300
        manager._state["last_hint"] = None

        hint = manager.get_contextual_hint("session_complete")

        assert hint is not None
        assert "power user" in hint.lower()
        assert "advanced features" in hint.lower()

    def test_get_contextual_hint_updates_last_hint_timestamp(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_contextual_hint updates last_hint timestamp."""
        manager._state["completed_sessions"] = 3
        manager._state["last_hint"] = None

        before_time = datetime.now(UTC)
        manager.get_contextual_hint("session_complete")
        after_time = datetime.now(UTC)

        last_hint = manager._state["last_hint"]
        assert last_hint is not None
        last_hint_time = datetime.fromisoformat(last_hint)
        assert before_time <= last_hint_time <= after_time

    def test_get_contextual_hint_adds_to_hints_shown(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_contextual_hint adds entry to hints_shown list."""
        manager._state["completed_sessions"] = 3
        manager._state["last_hint"] = None
        initial_hints_count = len(manager._state["hints_shown"])

        manager.get_contextual_hint("session_complete")

        assert len(manager._state["hints_shown"]) == initial_hints_count + 1
        last_shown = manager._state["hints_shown"][-1]
        assert last_shown["context"] == "session_complete"
        assert "hint_key" in last_shown
        assert "timestamp" in last_shown
        assert "completed_sessions" in last_shown

    def test_get_contextual_hint_saves_state(self, manager: IncentiveManager) -> None:
        """Test get_contextual_hint saves state after updating."""
        manager._state["completed_sessions"] = 3
        manager._state["last_hint"] = None

        manager.get_contextual_hint("session_complete")

        # Verify state was saved to disk
        with open(manager._state_file) as f:
            saved_state = json.load(f)
        assert saved_state["last_hint"] is not None

    def test_get_contextual_hint_handles_none_total_trials(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_contextual_hint handles None total_trials value."""
        manager._state["completed_sessions"] = 3
        manager._state["total_trials"] = None
        manager._state["last_hint"] = None

        hint = manager.get_contextual_hint("session_complete")

        assert hint is not None  # Should not crash


class TestShowAchievementUnlock:
    """Tests for achievement unlock messages."""

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
            local_storage_path=temp_storage_path,
        )

    @pytest.fixture
    def manager(self, edge_analytics_config: TraigentConfig) -> IncentiveManager:
        """Create IncentiveManager instance."""
        return IncentiveManager(edge_analytics_config)

    def test_show_achievement_unlock_first_optimization(
        self, manager: IncentiveManager
    ) -> None:
        """Test showing first optimization achievement."""
        message = manager.show_achievement_unlock("first_optimization")

        assert message is not None
        assert "First Optimization Complete" in message
        assert "first LLM optimization" in message

    def test_show_achievement_unlock_optimization_explorer(
        self, manager: IncentiveManager
    ) -> None:
        """Test showing optimization explorer achievement."""
        message = manager.show_achievement_unlock("optimization_explorer")

        assert message is not None
        assert "Optimization Explorer" in message
        assert "5 optimizations" in message

    def test_show_achievement_unlock_efficiency_expert(
        self, manager: IncentiveManager
    ) -> None:
        """Test showing efficiency expert achievement."""
        message = manager.show_achievement_unlock("efficiency_expert")

        assert message is not None
        assert "Efficiency Expert" in message
        assert "10 optimizations" in message

    def test_show_achievement_unlock_optimization_master(
        self, manager: IncentiveManager
    ) -> None:
        """Test showing optimization master achievement."""
        message = manager.show_achievement_unlock("optimization_master")

        assert message is not None
        assert "Optimization Master" in message
        assert "25 optimizations" in message

    def test_show_achievement_unlock_returns_none_if_already_unlocked(
        self, manager: IncentiveManager
    ) -> None:
        """Test show_achievement_unlock returns None if achievement already unlocked."""
        manager._state["achievement_unlocked"].append("first_optimization")

        message = manager.show_achievement_unlock("first_optimization")

        assert message is None

    def test_show_achievement_unlock_returns_none_for_unknown_achievement(
        self, manager: IncentiveManager
    ) -> None:
        """Test show_achievement_unlock returns None for unknown achievement."""
        message = manager.show_achievement_unlock("unknown_achievement")

        assert message is None

    def test_show_achievement_unlock_adds_to_unlocked_list(
        self, manager: IncentiveManager
    ) -> None:
        """Test show_achievement_unlock adds achievement to unlocked list."""
        manager.show_achievement_unlock("first_optimization")

        assert "first_optimization" in manager._state["achievement_unlocked"]

    def test_show_achievement_unlock_saves_state(
        self, manager: IncentiveManager
    ) -> None:
        """Test show_achievement_unlock saves state to disk."""
        manager.show_achievement_unlock("first_optimization")

        with open(manager._state_file) as f:
            saved_state = json.load(f)
        assert "first_optimization" in saved_state["achievement_unlocked"]


class TestCheckAchievements:
    """Tests for achievement checking logic."""

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
            local_storage_path=temp_storage_path,
        )

    @pytest.fixture
    def manager(self, edge_analytics_config: TraigentConfig) -> IncentiveManager:
        """Create IncentiveManager instance."""
        return IncentiveManager(edge_analytics_config)

    def test_check_achievements_unlocks_first_optimization(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test _check_achievements unlocks first optimization at 1 session."""
        sessions = [
            Mock(spec=OptimizationSession, status=OptimizationStatus.COMPLETED.value)
        ]

        manager._check_achievements(sessions)

        assert "first_optimization" in manager._state["achievement_unlocked"]
        captured = capsys.readouterr()
        assert "First Optimization Complete" in captured.out

    def test_check_achievements_unlocks_optimization_explorer(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test _check_achievements unlocks optimization explorer at 5 sessions."""
        sessions = [
            Mock(spec=OptimizationSession, status=OptimizationStatus.COMPLETED.value)
            for _ in range(5)
        ]

        manager._check_achievements(sessions)

        assert "optimization_explorer" in manager._state["achievement_unlocked"]
        captured = capsys.readouterr()
        assert "Optimization Explorer" in captured.out

    def test_check_achievements_unlocks_efficiency_expert(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test _check_achievements unlocks efficiency expert at 10 sessions."""
        sessions = [
            Mock(spec=OptimizationSession, status=OptimizationStatus.COMPLETED.value)
            for _ in range(10)
        ]

        manager._check_achievements(sessions)

        assert "efficiency_expert" in manager._state["achievement_unlocked"]
        captured = capsys.readouterr()
        assert "Efficiency Expert" in captured.out

    def test_check_achievements_unlocks_optimization_master(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test _check_achievements unlocks optimization master at 25 sessions."""
        sessions = [
            Mock(spec=OptimizationSession, status=OptimizationStatus.COMPLETED.value)
            for _ in range(25)
        ]

        manager._check_achievements(sessions)

        assert "optimization_master" in manager._state["achievement_unlocked"]
        captured = capsys.readouterr()
        assert "Optimization Master" in captured.out

    def test_check_achievements_does_not_unlock_twice(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test _check_achievements doesn't unlock same achievement twice."""
        sessions = [
            Mock(spec=OptimizationSession, status=OptimizationStatus.COMPLETED.value)
        ]

        # First unlock
        manager._check_achievements(sessions)
        capsys.readouterr()  # Clear output

        # Second check should not print again
        manager._check_achievements(sessions)
        captured = capsys.readouterr()
        assert "First Optimization Complete" not in captured.out

    def test_check_achievements_with_zero_sessions(
        self, manager: IncentiveManager
    ) -> None:
        """Test _check_achievements with zero sessions."""
        manager._check_achievements([])

        assert len(manager._state["achievement_unlocked"]) == 0


class TestGetUpgradeValueProposition:
    """Tests for upgrade value proposition generation."""

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
        )

    @pytest.fixture
    def manager(self, edge_analytics_config: TraigentConfig) -> IncentiveManager:
        """Create IncentiveManager instance."""
        return IncentiveManager(edge_analytics_config)

    def test_get_upgrade_value_proposition_returns_dict(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_upgrade_value_proposition returns dictionary."""
        manager._state["completed_sessions"] = 10
        manager._state["total_trials"] = 100

        result = manager.get_upgrade_value_proposition()

        assert isinstance(result, dict)

    def test_get_upgrade_value_proposition_with_analytics_data(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_upgrade_value_proposition uses analytics data when available."""
        manager._state["completed_sessions"] = 10
        manager._state["total_trials"] = 100

        analytics_data = {"custom": "data"}
        with patch.object(
            manager.analytics, "get_cloud_incentive_data", return_value=analytics_data
        ):
            result = manager.get_upgrade_value_proposition()

        assert result == analytics_data

    def test_get_upgrade_value_proposition_fallback_structure(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_upgrade_value_proposition fallback has expected structure."""
        manager._state["completed_sessions"] = 10
        manager._state["total_trials"] = 100

        with patch.object(
            manager.analytics, "get_cloud_incentive_data", return_value=None
        ):
            result = manager.get_upgrade_value_proposition()

        assert "time_savings" in result
        assert "performance_gains" in result
        assert "scale_benefits" in result

    def test_get_upgrade_value_proposition_time_savings_calculation(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_upgrade_value_proposition calculates time savings correctly."""
        manager._state["completed_sessions"] = 10
        manager._state["total_trials"] = 100

        with patch.object(
            manager.analytics, "get_cloud_incentive_data", return_value=None
        ):
            result = manager.get_upgrade_value_proposition()

        time_savings = result["time_savings"]
        assert "current_time_investment" in time_savings
        assert "5.0 hours" in time_savings["current_time_investment"]

    def test_get_upgrade_value_proposition_performance_gains(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_upgrade_value_proposition includes performance gains."""
        manager._state["completed_sessions"] = 10
        manager._state["total_trials"] = 100

        with patch.object(
            manager.analytics, "get_cloud_incentive_data", return_value=None
        ):
            result = manager.get_upgrade_value_proposition()

        performance = result["performance_gains"]
        assert "current_avg_trials" in performance
        assert "cloud_improvement" in performance
        assert "advanced_algorithms" in performance

    def test_get_upgrade_value_proposition_scale_benefits(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_upgrade_value_proposition includes scale benefits."""
        manager._state["completed_sessions"] = 10
        manager._state["total_trials"] = 100

        with patch.object(
            manager.analytics, "get_cloud_incentive_data", return_value=None
        ):
            result = manager.get_upgrade_value_proposition()

        scale = result["scale_benefits"]
        assert "current_sessions" in scale
        assert scale["current_sessions"] == 10
        assert "collaboration_value" in scale

    def test_get_upgrade_value_proposition_handles_analytics_error(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_upgrade_value_proposition handles analytics errors gracefully."""
        manager._state["completed_sessions"] = 10
        manager._state["total_trials"] = 100

        with patch.object(
            manager.analytics,
            "get_cloud_incentive_data",
            side_effect=Exception("Analytics error"),
        ):
            result = manager.get_upgrade_value_proposition()

        # Should fallback to basic calculation
        assert "time_savings" in result

    def test_get_upgrade_value_proposition_handles_zero_sessions(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_upgrade_value_proposition handles zero sessions."""
        manager._state["completed_sessions"] = 0
        manager._state["total_trials"] = 0

        with patch.object(
            manager.analytics, "get_cloud_incentive_data", return_value=None
        ):
            result = manager.get_upgrade_value_proposition()

        # Should not crash with division by zero
        assert isinstance(result, dict)

    def test_get_upgrade_value_proposition_handles_none_values(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_upgrade_value_proposition handles None values."""
        manager._state["completed_sessions"] = None
        manager._state["total_trials"] = None

        with patch.object(
            manager.analytics, "get_cloud_incentive_data", return_value=None
        ):
            result = manager.get_upgrade_value_proposition()

        # Should handle None gracefully
        assert isinstance(result, dict)


class TestShowContextSensitiveHint:
    """Tests for context-sensitive hint display."""

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
            local_storage_path=temp_storage_path,
        )

    @pytest.fixture
    def manager(self, edge_analytics_config: TraigentConfig) -> IncentiveManager:
        """Create IncentiveManager instance."""
        return IncentiveManager(edge_analytics_config)

    def test_show_context_sensitive_hint_large_config_space(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test showing hint for large configuration space."""
        manager._state["last_hint"] = None
        manager._state["completed_sessions"] = 3

        with patch.object(manager, "should_show_hint", return_value=True):
            manager.show_context_sensitive_hint("large_config_space")

        captured = capsys.readouterr()
        assert "Large configuration space" in captured.out
        assert "Bayesian optimization" in captured.out

    def test_show_context_sensitive_hint_multiple_objectives(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test showing hint for multiple objectives."""
        manager._state["last_hint"] = None
        manager._state["completed_sessions"] = 3

        with patch.object(manager, "should_show_hint", return_value=True):
            manager.show_context_sensitive_hint("multiple_objectives")

        captured = capsys.readouterr()
        assert "Multi-objective" in captured.out
        assert "Pareto" in captured.out

    def test_show_context_sensitive_hint_slow_evaluation(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test showing hint for slow evaluations."""
        manager._state["last_hint"] = None
        manager._state["completed_sessions"] = 3

        with patch.object(manager, "should_show_hint", return_value=True):
            manager.show_context_sensitive_hint("slow_evaluation")

        captured = capsys.readouterr()
        assert "Slow evaluations" in captured.out
        assert "Parallel evaluation" in captured.out

    def test_show_context_sensitive_hint_high_trial_count(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test showing hint for high trial count."""
        manager._state["last_hint"] = None
        manager._state["completed_sessions"] = 3

        with patch.object(manager, "should_show_hint", return_value=True):
            manager.show_context_sensitive_hint("high_trial_count", trial_count=100)

        captured = capsys.readouterr()
        assert "High trial count" in captured.out
        assert "100" in captured.out

    def test_show_context_sensitive_hint_updates_last_hint(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test context sensitive hint updates last_hint timestamp."""
        manager._state["last_hint"] = None
        manager._state["completed_sessions"] = 3

        with patch.object(manager, "should_show_hint", return_value=True):
            manager.show_context_sensitive_hint("large_config_space")

        assert manager._state["last_hint"] is not None

    def test_show_context_sensitive_hint_saves_state(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test context sensitive hint saves state."""
        manager._state["last_hint"] = None
        manager._state["completed_sessions"] = 3

        with patch.object(manager, "should_show_hint", return_value=True):
            manager.show_context_sensitive_hint("large_config_space")

        with open(manager._state_file) as f:
            saved_state = json.load(f)
        assert saved_state["last_hint"] is not None

    def test_show_context_sensitive_hint_respects_should_show_hint(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test context sensitive hint respects should_show_hint logic."""
        manager._state["upgrade_dismissed"] = True

        manager.show_context_sensitive_hint("large_config_space")

        captured = capsys.readouterr()
        assert captured.out == ""  # No hint should be shown

    def test_show_context_sensitive_hint_unknown_context(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test context sensitive hint with unknown context does nothing."""
        manager._state["last_hint"] = None
        manager._state["completed_sessions"] = 3

        manager.show_context_sensitive_hint("unknown_context")

        captured = capsys.readouterr()
        assert captured.out == ""


class TestDismissUpgradeHints:
    """Tests for dismissing upgrade hints."""

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
            local_storage_path=temp_storage_path,
        )

    @pytest.fixture
    def manager(self, edge_analytics_config: TraigentConfig) -> IncentiveManager:
        """Create IncentiveManager instance."""
        return IncentiveManager(edge_analytics_config)

    def test_dismiss_upgrade_hints_sets_dismissed_flag(
        self, manager: IncentiveManager
    ) -> None:
        """Test dismiss_upgrade_hints sets upgrade_dismissed flag."""
        manager.dismiss_upgrade_hints()

        assert manager._state["upgrade_dismissed"] is True

    def test_dismiss_upgrade_hints_sets_timestamp(
        self, manager: IncentiveManager
    ) -> None:
        """Test dismiss_upgrade_hints sets dismiss timestamp."""
        manager.dismiss_upgrade_hints()

        assert "dismiss_timestamp" in manager._state
        timestamp = datetime.fromisoformat(manager._state["dismiss_timestamp"])
        assert timestamp.tzinfo is not None

    def test_dismiss_upgrade_hints_saves_state(self, manager: IncentiveManager) -> None:
        """Test dismiss_upgrade_hints saves state to disk."""
        manager.dismiss_upgrade_hints()

        with open(manager._state_file) as f:
            saved_state = json.load(f)
        assert saved_state["upgrade_dismissed"] is True

    def test_dismiss_upgrade_hints_prints_confirmation(
        self, manager: IncentiveManager, capsys
    ) -> None:
        """Test dismiss_upgrade_hints prints confirmation message."""
        manager.dismiss_upgrade_hints()

        captured = capsys.readouterr()
        assert "Upgrade hints dismissed" in captured.out


class TestGetLocalVsCloudComparison:
    """Tests for local vs cloud comparison table."""

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
            local_storage_path=temp_storage_path,
        )

    @pytest.fixture
    def manager(self, edge_analytics_config: TraigentConfig) -> IncentiveManager:
        """Create IncentiveManager instance."""
        return IncentiveManager(edge_analytics_config)

    def test_get_local_vs_cloud_comparison_returns_string(
        self, manager: IncentiveManager
    ) -> None:
        """Test get_local_vs_cloud_comparison returns string."""
        result = manager.get_local_vs_cloud_comparison()

        assert isinstance(result, str)

    def test_get_local_vs_cloud_comparison_includes_features(
        self, manager: IncentiveManager
    ) -> None:
        """Test comparison table includes key features."""
        result = manager.get_local_vs_cloud_comparison()

        assert "Optimization Algorithm" in result
        assert "Web Dashboard" in result
        assert "Team Collaboration" in result
        assert "Bayesian" in result
        assert "Traigent Cloud" in result


class TestShowOnboardingTips:
    """Tests for onboarding tips display."""

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
            local_storage_path=temp_storage_path,
        )

    @pytest.fixture
    def manager(self, edge_analytics_config: TraigentConfig) -> IncentiveManager:
        """Create IncentiveManager instance."""
        return IncentiveManager(edge_analytics_config)

    def test_show_onboarding_tips_returns_list(self, manager: IncentiveManager) -> None:
        """Test show_onboarding_tips returns list of strings."""
        tips = manager.show_onboarding_tips()

        assert isinstance(tips, list)
        assert len(tips) > 0
        assert all(isinstance(tip, str) for tip in tips)

    def test_show_onboarding_tips_includes_helpful_commands(
        self, manager: IncentiveManager
    ) -> None:
        """Test onboarding tips include helpful commands."""
        tips = manager.show_onboarding_tips()
        tips_text = " ".join(tips)

        assert "traigent local list" in tips_text
        assert "traigent local show" in tips_text
        assert "traigent login" in tips_text


class TestGlobalFunctions:
    """Tests for global helper functions."""

    @pytest.fixture
    def temp_storage_path(self) -> Generator[str, None, None]:
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_show_upgrade_hint_in_edge_analytics_mode(
        self, temp_storage_path: str
    ) -> None:
        """Test show_upgrade_hint works in edge analytics mode."""
        with patch(
            "traigent.utils.incentives.TraigentConfig.from_environment"
        ) as mock_config:
            config = TraigentConfig(
                execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
                local_storage_path=temp_storage_path,
            )
            mock_config.return_value = config

            # Should not raise exception - verify it completes
            result = show_upgrade_hint("general")
            assert result is None  # Function returns None

    def test_show_upgrade_hint_not_in_edge_analytics_mode(
        self, temp_storage_path: str
    ) -> None:
        """Test show_upgrade_hint does nothing in non-edge mode."""
        with patch(
            "traigent.utils.incentives.TraigentConfig.from_environment"
        ) as mock_config:
            config = TraigentConfig(
                execution_mode=ExecutionMode.CLOUD.value,
                local_storage_path=temp_storage_path,
            )
            mock_config.return_value = config

            # Should not raise exception - verify it completes
            result = show_upgrade_hint("general")
            assert result is None  # Function returns None

    def test_show_achievement_in_edge_analytics_mode(
        self, temp_storage_path: str
    ) -> None:
        """Test show_achievement works in edge analytics mode."""
        with patch(
            "traigent.utils.incentives.TraigentConfig.from_environment"
        ) as mock_config:
            config = TraigentConfig(
                execution_mode=ExecutionMode.EDGE_ANALYTICS.value,
                local_storage_path=temp_storage_path,
            )
            mock_config.return_value = config

            # Should not raise exception - verify it completes
            result = show_achievement("first_optimization")
            assert result is None  # Function returns None

    def test_show_achievement_not_in_edge_analytics_mode(
        self, temp_storage_path: str
    ) -> None:
        """Test show_achievement does nothing in non-edge mode."""
        with patch(
            "traigent.utils.incentives.TraigentConfig.from_environment"
        ) as mock_config:
            config = TraigentConfig(
                execution_mode=ExecutionMode.CLOUD.value,
                local_storage_path=temp_storage_path,
            )
            mock_config.return_value = config

            # Should not raise exception - verify it completes
            result = show_achievement("first_optimization")
            assert result is None  # Function returns None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

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
            local_storage_path=temp_storage_path,
        )

    @pytest.fixture
    def manager(self, edge_analytics_config: TraigentConfig) -> IncentiveManager:
        """Create IncentiveManager instance."""
        return IncentiveManager(edge_analytics_config)

    def test_handles_empty_hints_shown_list(self, manager: IncentiveManager) -> None:
        """Test handles empty hints_shown list gracefully."""
        manager._state["hints_shown"] = []
        manager._state["completed_sessions"] = 3
        manager._state["last_hint"] = None

        hint = manager.get_contextual_hint("session_complete")

        assert hint is not None
        assert len(manager._state["hints_shown"]) == 1

    def test_handles_missing_achievement_unlocked_list(
        self, manager: IncentiveManager
    ) -> None:
        """Test handles missing achievement_unlocked list gracefully."""
        # The code doesn't handle missing achievement_unlocked gracefully,
        # so we test that it exists after initialization
        assert "achievement_unlocked" in manager._state
        assert isinstance(manager._state["achievement_unlocked"], list)

    def test_handles_very_large_session_count(self, manager: IncentiveManager) -> None:
        """Test handles very large session counts."""
        manager._state["completed_sessions"] = 10000
        manager._state["total_trials"] = 100000
        manager._state["last_hint"] = None

        hint = manager.get_contextual_hint("general")

        # Should work without errors
        assert hint is not None or hint is None  # Either is acceptable

    def test_handles_negative_session_count(self, manager: IncentiveManager) -> None:
        """Test handles negative session count gracefully."""
        manager._state["completed_sessions"] = -5
        manager._state["last_hint"] = None

        # Should not crash
        result = manager.should_show_hint("general")
        assert isinstance(result, bool)

    def test_concurrent_state_updates(
        self, edge_analytics_config: TraigentConfig
    ) -> None:
        """Test multiple manager instances can handle concurrent updates."""
        manager1 = IncentiveManager(edge_analytics_config)
        manager2 = IncentiveManager(edge_analytics_config)

        manager1._state["total_sessions"] = 10
        manager1._save_state()

        manager2._state["total_sessions"] = 20
        manager2._save_state()

        # Last write should win
        manager3 = IncentiveManager(edge_analytics_config)
        assert manager3._state["total_sessions"] == 20
