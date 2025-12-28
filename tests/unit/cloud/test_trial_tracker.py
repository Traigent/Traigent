"""Tests for Traigent cloud trial tracking module."""

import time

import pytest

from traigent.cloud.models import TrialStatus, TrialSuggestion
from traigent.cloud.trial_tracker import TrialState, TrialTracker
from traigent.utils.exceptions import ValidationError


class TestTrialState:
    """Test TrialState dataclass."""

    def test_trial_state_creation(self):
        """Test creating a trial state."""
        trial_state = TrialState(
            trial_id="trial-123",
            session_id="session-456",
            trial_number=1,
            config={"param1": "value1"},
            status=TrialStatus.PENDING,
            suggestion_time=time.time(),
        )

        assert trial_state.trial_id == "trial-123"
        assert trial_state.session_id == "session-456"
        assert trial_state.trial_number == 1
        assert trial_state.config == {"param1": "value1"}
        assert trial_state.status == TrialStatus.PENDING
        assert trial_state.start_time is None
        assert trial_state.completion_time is None
        assert trial_state.duration is None
        assert trial_state.metrics is None
        assert trial_state.error_message is None
        assert trial_state.backend_sync_status == "pending"

    def test_get_duration_with_explicit_duration(self):
        """Test get_duration when duration is explicitly set."""
        trial_state = TrialState(
            trial_id="trial-123",
            session_id="session-456",
            trial_number=1,
            config={},
            status=TrialStatus.COMPLETED,
            suggestion_time=time.time(),
            duration=5.5,
        )

        assert trial_state.get_duration() == 5.5

    def test_get_duration_calculated_from_times(self):
        """Test get_duration calculated from start and completion times."""
        start_time = time.time()
        completion_time = start_time + 3.0

        trial_state = TrialState(
            trial_id="trial-123",
            session_id="session-456",
            trial_number=1,
            config={},
            status=TrialStatus.COMPLETED,
            suggestion_time=start_time,
            start_time=start_time,
            completion_time=completion_time,
        )

        duration = trial_state.get_duration()
        assert duration is not None
        assert abs(duration - 3.0) < 0.1  # Allow for small timing differences

    def test_get_duration_running_trial(self):
        """Test get_duration for a running trial."""
        start_time = time.time() - 2.0  # Started 2 seconds ago

        trial_state = TrialState(
            trial_id="trial-123",
            session_id="session-456",
            trial_number=1,
            config={},
            status=TrialStatus.RUNNING,
            suggestion_time=start_time,
            start_time=start_time,
        )

        duration = trial_state.get_duration()
        assert duration is not None
        assert duration >= 1.5  # At least 1.5 seconds should have passed

    def test_get_duration_no_times(self):
        """Test get_duration when no timing information is available."""
        trial_state = TrialState(
            trial_id="trial-123",
            session_id="session-456",
            trial_number=1,
            config={},
            status=TrialStatus.PENDING,
            suggestion_time=time.time(),
        )

        assert trial_state.get_duration() is None

    def test_is_completed(self):
        """Test is_completed method."""
        completed_trial = TrialState(
            trial_id="trial-123",
            session_id="session-456",
            trial_number=1,
            config={},
            status=TrialStatus.COMPLETED,
            suggestion_time=time.time(),
        )
        assert completed_trial.is_completed()

        failed_trial = TrialState(
            trial_id="trial-456",
            session_id="session-456",
            trial_number=2,
            config={},
            status=TrialStatus.FAILED,
            suggestion_time=time.time(),
        )
        assert failed_trial.is_completed()

        pending_trial = TrialState(
            trial_id="trial-789",
            session_id="session-456",
            trial_number=3,
            config={},
            status=TrialStatus.PENDING,
            suggestion_time=time.time(),
        )
        assert not pending_trial.is_completed()

    def test_is_running(self):
        """Test is_running method."""
        running_trial = TrialState(
            trial_id="trial-123",
            session_id="session-456",
            trial_number=1,
            config={},
            status=TrialStatus.RUNNING,
            suggestion_time=time.time(),
        )
        assert running_trial.is_running()

        pending_trial = TrialState(
            trial_id="trial-456",
            session_id="session-456",
            trial_number=2,
            config={},
            status=TrialStatus.PENDING,
            suggestion_time=time.time(),
        )
        assert not pending_trial.is_running()

    def test_get_summary(self):
        """Test get_summary method."""
        trial_state = TrialState(
            trial_id="trial-123",
            session_id="session-456",
            trial_number=1,
            config={"param1": "value1"},
            status=TrialStatus.COMPLETED,
            suggestion_time=time.time(),
            start_time=time.time() - 5.0,
            completion_time=time.time(),
            metrics={"accuracy": 0.95},
            backend_sync_status="synced",
        )

        summary = trial_state.get_summary()
        assert summary["trial_id"] == "trial-123"
        assert summary["session_id"] == "session-456"
        assert summary["trial_number"] == 1
        assert summary["status"] == "completed"
        assert summary["has_metrics"] is True
        assert summary["has_error"] is False
        assert summary["backend_sync_status"] == "synced"
        assert "duration" in summary
        assert "suggestion_time" in summary
        assert "start_time" in summary
        assert "completion_time" in summary


class TestTrialTracker:
    """Test TrialTracker class."""

    @pytest.fixture
    def trial_tracker(self):
        """Create a trial tracker instance for testing."""
        return TrialTracker(max_trials_per_session=100)

    def test_invalid_max_trials_per_session_raises(self):
        """Ensure tracker validates the maximum trials constraint."""
        with pytest.raises(ValidationError):
            TrialTracker(max_trials_per_session=0)
        with pytest.raises(ValidationError):
            TrialTracker(max_trials_per_session=-5)
        with pytest.raises(ValidationError):
            TrialTracker(max_trials_per_session="not-a-number")  # type: ignore[arg-type]

    @pytest.fixture
    def sample_suggestion(self):
        """Create a sample trial suggestion."""
        return TrialSuggestion(
            trial_id="trial-123",
            session_id="session-456",
            trial_number=1,
            config={"param1": "value1", "param2": 42},
            dataset_subset=None,  # Mock dataset subset
            exploration_type="exploration",
        )

    def test_trial_tracker_initialization(self, trial_tracker):
        """Test trial tracker initialization."""
        assert trial_tracker.max_trials_per_session == 100
        assert len(trial_tracker._trials_by_session) == 0
        assert len(trial_tracker._trial_access_times) == 0

        stats = trial_tracker.get_statistics()
        assert stats["total_trials_created"] == 0
        assert stats["trials_completed"] == 0
        assert stats["trials_failed"] == 0
        assert stats["trials_running"] == 0
        assert stats["total_trials_tracked"] == 0
        assert stats["sessions_with_trials"] == 0

    def test_register_trial_suggestion(self, trial_tracker, sample_suggestion):
        """Test registering a trial suggestion."""
        trial_state = trial_tracker.register_trial_suggestion(
            "session-456", sample_suggestion
        )

        assert trial_state.trial_id == "trial-123"
        assert trial_state.session_id == "session-456"
        assert trial_state.trial_number == 1
        assert trial_state.config == {"param1": "value1", "param2": 42}
        assert trial_state.status == TrialStatus.PENDING
        assert trial_state.backend_sync_status == "pending"

        # Check it's tracked in the trial tracker
        assert len(trial_tracker._trials_by_session) == 1
        assert "session-456" in trial_tracker._trials_by_session
        assert "trial-123" in trial_tracker._trials_by_session["session-456"]

        # Check statistics were updated
        stats = trial_tracker.get_statistics()
        assert stats["total_trials_created"] == 1

    def test_register_trial_suggestion_with_backend_id(
        self, trial_tracker, sample_suggestion
    ):
        """Test registering a trial suggestion with backend config run ID."""
        trial_state = trial_tracker.register_trial_suggestion(
            "session-456", sample_suggestion, backend_config_run_id="backend-run-123"
        )

        assert trial_state.backend_config_run_id == "backend-run-123"

    def test_register_duplicate_trial(self, trial_tracker, sample_suggestion):
        """Test registering a duplicate trial ID."""
        trial_tracker.register_trial_suggestion("session-456", sample_suggestion)

        with pytest.raises(ValueError, match="Trial trial-123 already exists"):
            trial_tracker.register_trial_suggestion("session-456", sample_suggestion)

    def test_register_trial_capacity_exceeded(self, sample_suggestion):
        """Test registering trials beyond capacity."""
        trial_tracker = TrialTracker(max_trials_per_session=2)

        # Create different trial suggestions
        suggestion1 = TrialSuggestion(
            trial_id="trial-1",
            session_id="session-456",
            trial_number=1,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )
        suggestion2 = TrialSuggestion(
            trial_id="trial-2",
            session_id="session-456",
            trial_number=2,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )
        suggestion3 = TrialSuggestion(
            trial_id="trial-3",
            session_id="session-456",
            trial_number=3,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )

        trial_tracker.register_trial_suggestion("session-456", suggestion1)
        trial_tracker.register_trial_suggestion("session-456", suggestion2)

        with pytest.raises(ValueError, match="Maximum trials per session exceeded"):
            trial_tracker.register_trial_suggestion("session-456", suggestion3)

    def test_start_trial(self, trial_tracker, sample_suggestion):
        """Test starting a trial."""
        trial_tracker.register_trial_suggestion("session-456", sample_suggestion)

        success = trial_tracker.start_trial("session-456", "trial-123")
        assert success is True

        trial_state = trial_tracker.get_trial_state("session-456", "trial-123")
        assert trial_state.status == TrialStatus.RUNNING
        assert trial_state.start_time is not None

        # Check statistics were updated
        stats = trial_tracker.get_statistics()
        assert stats["trials_running"] == 1
        assert stats["status_transitions"] == 1

    def test_start_nonexistent_trial(self, trial_tracker):
        """Test starting a non-existent trial."""
        success = trial_tracker.start_trial("session-456", "nonexistent-trial")
        assert success is False

    def test_start_trial_wrong_status(self, trial_tracker, sample_suggestion):
        """Test starting a trial that's not in pending status."""
        trial_tracker.register_trial_suggestion("session-456", sample_suggestion)
        trial_tracker.start_trial("session-456", "trial-123")

        # Try to start again
        success = trial_tracker.start_trial("session-456", "trial-123")
        assert success is False

    def test_complete_trial(self, trial_tracker, sample_suggestion):
        """Test completing a trial."""
        trial_tracker.register_trial_suggestion("session-456", sample_suggestion)
        trial_tracker.start_trial("session-456", "trial-123")

        metrics = {"accuracy": 0.95, "loss": 0.05}
        success = trial_tracker.complete_trial("session-456", "trial-123", metrics)
        assert success is True

        trial_state = trial_tracker.get_trial_state("session-456", "trial-123")
        assert trial_state.status == TrialStatus.COMPLETED
        assert trial_state.completion_time is not None
        assert trial_state.metrics == metrics
        assert trial_state.duration is not None

        # Check statistics were updated
        stats = trial_tracker.get_statistics()
        assert stats["trials_completed"] == 1
        assert stats["trials_running"] == 0
        assert stats["status_transitions"] == 2  # start + complete

    def test_complete_trial_with_explicit_duration(
        self, trial_tracker, sample_suggestion
    ):
        """Test completing a trial with explicit duration."""
        trial_tracker.register_trial_suggestion("session-456", sample_suggestion)
        trial_tracker.start_trial("session-456", "trial-123")

        metrics = {"accuracy": 0.95}
        duration = 10.5
        success = trial_tracker.complete_trial(
            "session-456", "trial-123", metrics, duration
        )
        assert success is True

        trial_state = trial_tracker.get_trial_state("session-456", "trial-123")
        assert trial_state.duration == duration

    def test_complete_nonexistent_trial(self, trial_tracker):
        """Test completing a non-existent trial."""
        success = trial_tracker.complete_trial("session-456", "nonexistent-trial", {})
        assert success is False

    def test_complete_trial_wrong_status(self, trial_tracker, sample_suggestion):
        """Test completing a trial that's not running."""
        trial_tracker.register_trial_suggestion("session-456", sample_suggestion)

        # Try to complete without starting
        success = trial_tracker.complete_trial(
            "session-456", "trial-123", {"accuracy": 0.95}
        )
        assert success is False

    def test_fail_trial(self, trial_tracker, sample_suggestion):
        """Test failing a trial."""
        trial_tracker.register_trial_suggestion("session-456", sample_suggestion)
        trial_tracker.start_trial("session-456", "trial-123")

        error_message = "Out of memory error"
        success = trial_tracker.fail_trial("session-456", "trial-123", error_message)
        assert success is True

        trial_state = trial_tracker.get_trial_state("session-456", "trial-123")
        assert trial_state.status == TrialStatus.FAILED
        assert trial_state.completion_time is not None
        assert trial_state.error_message == error_message
        assert trial_state.duration is not None

        # Check statistics were updated
        stats = trial_tracker.get_statistics()
        assert stats["trials_failed"] == 1
        assert stats["trials_running"] == 0
        assert stats["status_transitions"] == 2  # start + fail

    def test_fail_trial_from_pending(self, trial_tracker, sample_suggestion):
        """Test failing a trial from pending status."""
        trial_tracker.register_trial_suggestion("session-456", sample_suggestion)

        error_message = "Configuration error"
        success = trial_tracker.fail_trial("session-456", "trial-123", error_message)
        assert success is True

        trial_state = trial_tracker.get_trial_state("session-456", "trial-123")
        assert trial_state.status == TrialStatus.FAILED
        assert trial_state.error_message == error_message

    def test_fail_nonexistent_trial(self, trial_tracker):
        """Test failing a non-existent trial."""
        success = trial_tracker.fail_trial("session-456", "nonexistent-trial", "Error")
        assert success is False

    def test_get_trial_state(self, trial_tracker, sample_suggestion):
        """Test getting trial state."""
        trial_tracker.register_trial_suggestion("session-456", sample_suggestion)

        trial_state = trial_tracker.get_trial_state("session-456", "trial-123")
        assert trial_state is not None
        assert trial_state.trial_id == "trial-123"

        # Test non-existent trial
        trial_state = trial_tracker.get_trial_state("session-456", "nonexistent-trial")
        assert trial_state is None

        # Test non-existent session
        trial_state = trial_tracker.get_trial_state("nonexistent-session", "trial-123")
        assert trial_state is None

    def test_get_session_trials(self, trial_tracker):
        """Test getting all trials for a session."""
        # Create multiple trial suggestions
        suggestion1 = TrialSuggestion(
            trial_id="trial-1",
            session_id="session-456",
            trial_number=1,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )
        suggestion2 = TrialSuggestion(
            trial_id="trial-2",
            session_id="session-456",
            trial_number=2,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )

        trial_tracker.register_trial_suggestion("session-456", suggestion1)
        trial_tracker.register_trial_suggestion("session-456", suggestion2)

        session_trials = trial_tracker.get_session_trials("session-456")
        assert len(session_trials) == 2
        assert "trial-1" in session_trials
        assert "trial-2" in session_trials

        # Test non-existent session
        empty_trials = trial_tracker.get_session_trials("nonexistent-session")
        assert len(empty_trials) == 0

    def test_get_active_trials(self, trial_tracker):
        """Test getting active (running) trials."""
        # Create and start some trials
        suggestion1 = TrialSuggestion(
            trial_id="trial-1",
            session_id="session-456",
            trial_number=1,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )
        suggestion2 = TrialSuggestion(
            trial_id="trial-2",
            session_id="session-456",
            trial_number=2,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )

        trial_tracker.register_trial_suggestion("session-456", suggestion1)
        trial_tracker.register_trial_suggestion("session-456", suggestion2)

        trial_tracker.start_trial("session-456", "trial-1")
        # Leave trial-2 in pending state

        active_trials = trial_tracker.get_active_trials("session-456")
        assert len(active_trials) == 1
        assert active_trials[0].trial_id == "trial-1"
        assert active_trials[0].status == TrialStatus.RUNNING

    def test_get_completed_trials(self, trial_tracker):
        """Test getting completed trials."""
        # Create, start, and complete some trials
        suggestion1 = TrialSuggestion(
            trial_id="trial-1",
            session_id="session-456",
            trial_number=1,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )
        suggestion2 = TrialSuggestion(
            trial_id="trial-2",
            session_id="session-456",
            trial_number=2,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )

        trial_tracker.register_trial_suggestion("session-456", suggestion1)
        trial_tracker.register_trial_suggestion("session-456", suggestion2)

        trial_tracker.start_trial("session-456", "trial-1")
        trial_tracker.complete_trial("session-456", "trial-1", {"accuracy": 0.95})

        trial_tracker.start_trial("session-456", "trial-2")
        # Leave trial-2 running

        completed_trials = trial_tracker.get_completed_trials("session-456")
        assert len(completed_trials) == 1
        assert completed_trials[0].trial_id == "trial-1"
        assert completed_trials[0].status == TrialStatus.COMPLETED

    def test_get_failed_trials(self, trial_tracker):
        """Test getting failed trials."""
        # Create and fail some trials
        suggestion1 = TrialSuggestion(
            trial_id="trial-1",
            session_id="session-456",
            trial_number=1,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )
        suggestion2 = TrialSuggestion(
            trial_id="trial-2",
            session_id="session-456",
            trial_number=2,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )

        trial_tracker.register_trial_suggestion("session-456", suggestion1)
        trial_tracker.register_trial_suggestion("session-456", suggestion2)

        trial_tracker.start_trial("session-456", "trial-1")
        trial_tracker.fail_trial("session-456", "trial-1", "Error occurred")

        # Leave trial-2 in pending state

        failed_trials = trial_tracker.get_failed_trials("session-456")
        assert len(failed_trials) == 1
        assert failed_trials[0].trial_id == "trial-1"
        assert failed_trials[0].status == TrialStatus.FAILED

    def test_get_session_trial_summary(self, trial_tracker):
        """Test getting session trial summary."""
        # Create trials in different states
        suggestions = []
        for i in range(1, 6):
            suggestions.append(
                TrialSuggestion(
                    trial_id=f"trial-{i}",
                    session_id="session-456",
                    trial_number=i,
                    config={},
                    dataset_subset=None,
                    exploration_type="exploration",
                )
            )

        for suggestion in suggestions:
            trial_tracker.register_trial_suggestion("session-456", suggestion)

        # trial-1: pending
        # trial-2: running
        trial_tracker.start_trial("session-456", "trial-2")

        # trial-3: completed
        trial_tracker.start_trial("session-456", "trial-3")
        trial_tracker.complete_trial("session-456", "trial-3", {"accuracy": 0.95})

        # trial-4: failed
        trial_tracker.start_trial("session-456", "trial-4")
        trial_tracker.fail_trial("session-456", "trial-4", "Error")

        # trial-5: completed
        trial_tracker.start_trial("session-456", "trial-5")
        trial_tracker.complete_trial(
            "session-456", "trial-5", {"accuracy": 0.90}, duration=5.0
        )

        summary = trial_tracker.get_session_trial_summary("session-456")
        assert summary["session_id"] == "session-456"
        assert summary["total_trials"] == 5
        assert summary["active_trials"] == 1  # trial-2
        assert summary["completed_trials"] == 2  # trial-3, trial-5
        assert summary["failed_trials"] == 1  # trial-4
        assert summary["pending_trials"] == 1  # trial-1
        assert summary["has_trials"] is True
        assert summary["average_duration"] is not None

    def test_remove_session_trials(self, trial_tracker):
        """Test removing all trials for a session."""
        # Create some trials
        suggestion1 = TrialSuggestion(
            trial_id="trial-1",
            session_id="session-456",
            trial_number=1,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )
        suggestion2 = TrialSuggestion(
            trial_id="trial-2",
            session_id="session-456",
            trial_number=2,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )

        trial_tracker.register_trial_suggestion("session-456", suggestion1)
        trial_tracker.register_trial_suggestion("session-456", suggestion2)

        # Verify trials exist
        assert len(trial_tracker.get_session_trials("session-456")) == 2

        # Remove session trials
        removed_count = trial_tracker.remove_session_trials("session-456")
        assert removed_count == 2

        # Verify trials are gone
        assert len(trial_tracker.get_session_trials("session-456")) == 0
        assert "session-456" not in trial_tracker._trials_by_session

        # Check access times were cleaned up
        assert "trial-1" not in trial_tracker._trial_access_times
        assert "trial-2" not in trial_tracker._trial_access_times

    def test_cleanup_old_trials(self, trial_tracker):
        """Test cleaning up old trials."""
        # Create multiple trials
        suggestions = []
        for i in range(1, 6):
            suggestions.append(
                TrialSuggestion(
                    trial_id=f"trial-{i}",
                    session_id="session-456",
                    trial_number=i,
                    config={},
                    dataset_subset=None,
                    exploration_type="exploration",
                )
            )

        for suggestion in suggestions:
            trial_tracker.register_trial_suggestion("session-456", suggestion)

        # Set some access times manually to simulate age
        base_time = time.time()
        trial_tracker._trial_access_times["trial-1"] = base_time - 100
        trial_tracker._trial_access_times["trial-2"] = base_time - 80
        trial_tracker._trial_access_times["trial-3"] = base_time - 60
        trial_tracker._trial_access_times["trial-4"] = base_time - 40
        trial_tracker._trial_access_times["trial-5"] = base_time - 20

        # Start trial-5 (should not be removed)
        trial_tracker.start_trial("session-456", "trial-5")

        # Clean up keeping only 3 trials
        removed_count = trial_tracker.cleanup_old_trials("session-456", max_trials=3)

        # Should remove 2 oldest trials (trial-1, trial-2) but not running trial-5
        assert removed_count >= 1  # At least some cleanup happened

        remaining_trials = trial_tracker.get_session_trials("session-456")
        assert len(remaining_trials) <= 3

        # Running trial should still be there
        assert "trial-5" in remaining_trials

    def test_get_statistics(self, trial_tracker):
        """Test getting trial tracker statistics."""
        # Create and execute some trials
        suggestion1 = TrialSuggestion(
            trial_id="trial-1",
            session_id="session-456",
            trial_number=1,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )
        suggestion2 = TrialSuggestion(
            trial_id="trial-2",
            session_id="session-789",
            trial_number=1,
            config={},
            dataset_subset=None,
            exploration_type="exploration",
        )

        trial_tracker.register_trial_suggestion("session-456", suggestion1)
        trial_tracker.register_trial_suggestion("session-789", suggestion2)

        trial_tracker.start_trial("session-456", "trial-1")
        trial_tracker.complete_trial("session-456", "trial-1", {"accuracy": 0.95})

        trial_tracker.start_trial("session-789", "trial-2")
        trial_tracker.fail_trial("session-789", "trial-2", "Error")

        stats = trial_tracker.get_statistics()
        assert stats["total_trials_created"] == 2
        assert stats["trials_completed"] == 1
        assert stats["trials_failed"] == 1
        assert stats["trials_running"] == 0
        assert stats["status_transitions"] == 4  # 2 starts + 1 complete + 1 fail
        assert stats["total_trials_tracked"] == 2
        assert stats["sessions_with_trials"] == 2
        assert stats["avg_trials_per_session"] == 1.0
