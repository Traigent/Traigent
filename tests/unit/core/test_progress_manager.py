"""Unit tests for ProgressManager.

Tests cover:
- ProgressState: State tracking and property calculations
- create_progress_info: ProgressInfo creation from state
- log_progress: Progress logging
- ProgressManager: Progress tracking coordination
"""

import time
from unittest.mock import patch

from traigent.core.progress_manager import (
    ProgressManager,
    ProgressState,
    create_progress_info,
    log_progress,
)
from traigent.utils.callbacks import ProgressInfo


class TestProgressState:
    """Test ProgressState dataclass and properties."""

    def test_default_values(self):
        """Test default state values."""
        state = ProgressState()
        assert state.total_trials is None
        assert state.completed_trials == 0
        assert state.successful_trials == 0
        assert state.failed_trials == 0
        assert state.start_time is None
        assert state.best_score is None
        assert state.best_config is None
        assert state.algorithm_name == "Unknown"
        assert state.objectives == []

    def test_elapsed_time_no_start(self):
        """Test elapsed_time returns 0 when start_time is None."""
        state = ProgressState()
        assert state.elapsed_time == 0.0

    def test_elapsed_time_with_start(self):
        """Test elapsed_time calculation with start_time."""
        state = ProgressState(start_time=time.time() - 5.0)
        elapsed = state.elapsed_time
        assert 5.0 <= elapsed < 6.0

    def test_avg_trial_time_no_trials(self):
        """Test avg_trial_time returns 0 when no trials completed."""
        state = ProgressState()
        assert state.avg_trial_time == 0.0

    def test_avg_trial_time_with_trials(self):
        """Test avg_trial_time calculation."""
        state = ProgressState(
            start_time=time.time() - 10.0,
            completed_trials=5,
        )
        avg = state.avg_trial_time
        assert 1.9 <= avg <= 2.1  # ~10s / 5 trials = ~2s

    def test_estimated_remaining_no_total(self):
        """Test estimated_remaining returns None when total_trials is None."""
        state = ProgressState(completed_trials=5)
        assert state.estimated_remaining is None

    def test_estimated_remaining_all_complete(self):
        """Test estimated_remaining returns 0 when all trials complete."""
        state = ProgressState(
            total_trials=5,
            completed_trials=5,
            start_time=time.time() - 10.0,
        )
        assert state.estimated_remaining == 0.0

    def test_estimated_remaining_calculation(self):
        """Test estimated_remaining calculation."""
        state = ProgressState(
            total_trials=10,
            completed_trials=5,
            start_time=time.time() - 10.0,
        )
        # Avg trial time = 10s / 5 = 2s
        # Remaining trials = 10 - 5 = 5
        # Estimated remaining = 2s * 5 = 10s
        remaining = state.estimated_remaining
        assert remaining is not None
        assert 9.5 <= remaining <= 10.5

    def test_success_rate_no_trials(self):
        """Test success_rate returns 0 when no trials completed."""
        state = ProgressState()
        assert state.success_rate == 0.0

    def test_success_rate_calculation(self):
        """Test success_rate calculation."""
        state = ProgressState(
            completed_trials=10,
            successful_trials=8,
        )
        assert state.success_rate == 0.8

    def test_success_rate_all_successful(self):
        """Test success_rate is 1.0 when all trials successful."""
        state = ProgressState(
            completed_trials=5,
            successful_trials=5,
        )
        assert state.success_rate == 1.0


class TestCreateProgressInfo:
    """Test create_progress_info function."""

    def test_basic_progress_info(self):
        """Test creating basic ProgressInfo."""
        state = ProgressState(
            total_trials=10,
            completed_trials=5,
            successful_trials=4,
            failed_trials=1,
            algorithm_name="Bayesian",
        )
        info = create_progress_info(current_trial=6, state=state)

        assert isinstance(info, ProgressInfo)
        assert info.current_trial == 6
        assert info.total_trials == 10
        assert info.completed_trials == 5
        assert info.successful_trials == 4
        assert info.failed_trials == 1
        assert info.current_algorithm == "Bayesian"

    def test_progress_info_with_best_score(self):
        """Test ProgressInfo with best score and config."""
        state = ProgressState(
            total_trials=10,
            completed_trials=5,
            best_score=0.95,
            best_config={"model": "gpt-4"},
            start_time=time.time() - 5.0,
        )
        info = create_progress_info(current_trial=6, state=state)

        assert info.best_score == 0.95
        assert info.best_config == {"model": "gpt-4"}
        assert 5.0 <= info.elapsed_time < 6.0

    def test_progress_info_none_total_trials(self):
        """Test ProgressInfo when total_trials is None."""
        state = ProgressState(total_trials=None)
        info = create_progress_info(current_trial=1, state=state)
        assert info.total_trials == 0


class TestLogProgress:
    """Test log_progress function."""

    def test_log_progress_basic(self):
        """Test basic progress logging."""
        state = ProgressState(
            completed_trials=10,
            successful_trials=8,
            best_score=0.85,
            start_time=time.time() - 10.0,
        )
        with patch("traigent.core.progress_manager.logger") as mock_logger:
            log_progress(trial_count=10, state=state)
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "10 trials" in call_args
            assert "0.8500" in call_args  # best score
            assert "80.00%" in call_args  # success rate

    def test_log_progress_no_best_score(self):
        """Test progress logging with no best score."""
        state = ProgressState(
            completed_trials=5,
            successful_trials=0,
        )
        with patch("traigent.core.progress_manager.logger") as mock_logger:
            log_progress(trial_count=5, state=state)
            call_args = mock_logger.info.call_args[0][0]
            assert "N/A" in call_args


class TestProgressManager:
    """Test ProgressManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = ProgressManager(
            total_trials=100,
            algorithm_name="Random",
            objectives=["accuracy", "cost"],
            log_interval=5,
        )
        assert manager.state.total_trials == 100
        assert manager.state.algorithm_name == "Random"
        assert manager.state.objectives == ["accuracy", "cost"]
        assert manager.log_interval == 5

    def test_initialization_defaults(self):
        """Test manager initialization with defaults."""
        manager = ProgressManager()
        assert manager.state.total_trials is None
        assert manager.state.algorithm_name == "Unknown"
        assert manager.state.objectives == []
        assert manager.log_interval == 10

    def test_start(self):
        """Test start method initializes tracking."""
        manager = ProgressManager(total_trials=10)
        manager.state._state_completed_trials = 5  # Simulate previous state

        manager.start()

        assert manager.state.start_time is not None
        assert manager.state.completed_trials == 0
        assert manager.state.successful_trials == 0
        assert manager.state.failed_trials == 0

    def test_record_trial_completion_success(self):
        """Test recording successful trial."""
        manager = ProgressManager(objectives=["accuracy"])
        manager.start()

        manager.record_trial_completion(
            success=True,
            metrics={"accuracy": 0.9},
            config={"model": "gpt-4"},
        )

        assert manager.state.completed_trials == 1
        assert manager.state.successful_trials == 1
        assert manager.state.failed_trials == 0
        assert manager.state.best_score == 0.9
        assert manager.state.best_config == {"model": "gpt-4"}

    def test_record_trial_completion_failure(self):
        """Test recording failed trial."""
        manager = ProgressManager()
        manager.start()

        manager.record_trial_completion(success=False)

        assert manager.state.completed_trials == 1
        assert manager.state.successful_trials == 0
        assert manager.state.failed_trials == 1

    def test_record_trial_updates_best_score(self):
        """Test that better scores update best_score."""
        manager = ProgressManager(objectives=["accuracy"])
        manager.start()

        manager.record_trial_completion(
            success=True,
            metrics={"accuracy": 0.8},
            config={"model": "gpt-3.5"},
        )
        assert manager.state.best_score == 0.8

        manager.record_trial_completion(
            success=True,
            metrics={"accuracy": 0.9},
            config={"model": "gpt-4"},
        )
        assert manager.state.best_score == 0.9
        assert manager.state.best_config == {"model": "gpt-4"}

    def test_record_trial_keeps_best_score(self):
        """Test that worse scores don't update best_score."""
        manager = ProgressManager(objectives=["accuracy"])
        manager.start()

        manager.record_trial_completion(
            success=True,
            metrics={"accuracy": 0.9},
            config={"model": "gpt-4"},
        )

        manager.record_trial_completion(
            success=True,
            metrics={"accuracy": 0.7},
            config={"model": "gpt-3.5"},
        )

        assert manager.state.best_score == 0.9
        assert manager.state.best_config == {"model": "gpt-4"}

    def test_update_best_result(self):
        """Test direct update of best result."""
        manager = ProgressManager()
        manager.update_best_result(score=0.95, config={"temp": 0.5})

        assert manager.state.best_score == 0.95
        assert manager.state.best_config == {"temp": 0.5}

    def test_get_progress_info(self):
        """Test get_progress_info method."""
        manager = ProgressManager(
            total_trials=10,
            algorithm_name="Grid",
        )
        manager.start()
        manager.record_trial_completion(success=True)

        info = manager.get_progress_info(current_trial=1)

        assert isinstance(info, ProgressInfo)
        assert info.current_trial == 1
        assert info.total_trials == 10
        assert info.completed_trials == 1
        assert info.current_algorithm == "Grid"

    def test_should_log_at_interval(self):
        """Test should_log at log interval."""
        manager = ProgressManager(log_interval=5)

        assert manager.should_log(0) is False  # Never log at 0
        assert manager.should_log(1) is False
        assert manager.should_log(5) is True
        assert manager.should_log(10) is True
        assert manager.should_log(7) is False

    def test_log_progress_method(self):
        """Test log_progress method."""
        manager = ProgressManager()
        manager.start()
        manager.record_trial_completion(success=True)

        with patch("traigent.core.progress_manager.logger") as mock_logger:
            manager.log_progress(trial_count=1)
            mock_logger.info.assert_called_once()

    def test_log_if_interval_logs(self):
        """Test log_if_interval logs at interval."""
        manager = ProgressManager(log_interval=5)
        manager.start()

        with patch("traigent.core.progress_manager.logger") as mock_logger:
            manager.log_if_interval(5)
            mock_logger.info.assert_called_once()

    def test_log_if_interval_skips(self):
        """Test log_if_interval skips non-interval."""
        manager = ProgressManager(log_interval=5)
        manager.start()

        with patch("traigent.core.progress_manager.logger") as mock_logger:
            manager.log_if_interval(3)
            mock_logger.info.assert_not_called()

    def test_get_summary(self):
        """Test get_summary method."""
        manager = ProgressManager(
            total_trials=10,
            algorithm_name="Bayesian",
        )
        manager.start()
        manager.record_trial_completion(success=True)
        manager.record_trial_completion(success=False)
        manager.update_best_result(score=0.85, config={})

        summary = manager.get_summary()

        assert summary["total_trials"] == 10
        assert summary["completed_trials"] == 2
        assert summary["successful_trials"] == 1
        assert summary["failed_trials"] == 1
        assert summary["success_rate"] == 0.5
        assert summary["best_score"] == 0.85
        assert summary["algorithm"] == "Bayesian"
        assert "elapsed_time" in summary
        assert "avg_trial_time" in summary


class TestProgressManagerEdgeCases:
    """Test edge cases for ProgressManager."""

    def test_record_trial_no_objectives(self):
        """Test recording trial when no objectives configured."""
        manager = ProgressManager(objectives=[])
        manager.start()

        manager.record_trial_completion(
            success=True,
            metrics={"accuracy": 0.9},
        )

        # Should not update best_score without objectives
        assert manager.state.best_score is None

    def test_record_trial_missing_metric(self):
        """Test recording trial when metric not in results."""
        manager = ProgressManager(objectives=["accuracy"])
        manager.start()

        manager.record_trial_completion(
            success=True,
            metrics={"cost": 0.01},  # Different metric
        )

        # Should not update best_score
        assert manager.state.best_score is None

    def test_state_immutability_check(self):
        """Test that state property returns internal state."""
        manager = ProgressManager()
        state = manager.state
        state.completed_trials = 100  # Modify returned state

        # Should be same object (not a copy)
        assert manager.state.completed_trials == 100

    def test_multiple_start_calls_reset(self):
        """Test that start() resets all counters."""
        manager = ProgressManager(total_trials=10)
        manager.start()
        manager.record_trial_completion(success=True)
        manager.record_trial_completion(success=False)

        # Call start again
        manager.start()

        assert manager.state.completed_trials == 0
        assert manager.state.successful_trials == 0
        assert manager.state.failed_trials == 0
