"""Trial State Tracking for Traigent SDK.

Focused component responsible for tracking trial states and lifecycle,
extracted from the SessionLifecycleManager to follow Single Responsibility Principle.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from traigent.utils.logging import get_logger
from traigent.utils.validation import CoreValidators, validate_or_raise

from .models import TrialStatus, TrialSuggestion

logger = get_logger(__name__)


@dataclass
class TrialState:
    """State of an individual trial."""

    trial_id: str
    session_id: str
    trial_number: int
    config: dict[str, Any]
    status: TrialStatus
    suggestion_time: float
    start_time: float | None = None
    completion_time: float | None = None
    duration: float | None = None
    metrics: dict[str, float] | None = None
    error_message: str | None = None
    backend_config_run_id: str | None = None
    backend_sync_status: str = "pending"  # pending, synced, failed

    def get_duration(self) -> float | None:
        """Calculate trial duration."""
        if self.duration is not None:
            return self.duration

        if self.start_time and self.completion_time:
            return self.completion_time - self.start_time

        if self.start_time and self.status == TrialStatus.RUNNING:
            return time.time() - self.start_time

        return None

    def is_completed(self) -> bool:
        """Check if trial is completed (successfully or failed)."""
        return self.status in [TrialStatus.COMPLETED, TrialStatus.FAILED]

    def is_running(self) -> bool:
        """Check if trial is currently running."""
        return self.status == TrialStatus.RUNNING

    def get_summary(self) -> dict[str, Any]:
        """Get trial summary information."""
        return {
            "trial_id": self.trial_id,
            "session_id": self.session_id,
            "trial_number": self.trial_number,
            "status": self.status.value,
            "duration": self.get_duration(),
            "has_metrics": bool(self.metrics),
            "has_error": bool(self.error_message),
            "backend_sync_status": self.backend_sync_status,
            "suggestion_time": self.suggestion_time,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
        }


class TrialTracker:
    """Manages trial states and lifecycle operations."""

    def __init__(self, max_trials_per_session: int = 10000) -> None:
        """Initialize trial tracker.

        Args:
            max_trials_per_session: Maximum trials to track per session
        """
        validate_or_raise(
            CoreValidators.validate_positive_int(
                max_trials_per_session, "max_trials_per_session"
            )
        )
        self.max_trials_per_session = max_trials_per_session
        self._trials_by_session: dict[str, dict[str, TrialState]] = {}
        self._trial_access_times: dict[str, float] = {}

        # Statistics
        self._stats: dict[str, Any] = {
            "total_trials_created": 0,
            "trials_completed": 0,
            "trials_failed": 0,
            "trials_running": 0,
            "status_transitions": 0,
        }

    def register_trial_suggestion(
        self,
        session_id: str,
        suggestion: TrialSuggestion,
        backend_config_run_id: str | None = None,
    ) -> TrialState:
        """Register a new trial suggestion.

        Args:
            session_id: Session ID
            suggestion: Trial suggestion
            backend_config_run_id: Backend configuration run ID

        Returns:
            Created trial state
        """
        # Validate inputs
        validate_or_raise(
            CoreValidators.validate_string_non_empty(session_id, "session_id")
        )
        validate_or_raise(
            CoreValidators.validate_string_non_empty(suggestion.trial_id, "trial_id")
        )

        # Initialize session trials if needed
        if session_id not in self._trials_by_session:
            self._trials_by_session[session_id] = {}

        session_trials = self._trials_by_session[session_id]

        # Check if trial already exists
        if suggestion.trial_id in session_trials:
            raise ValueError(
                f"Trial {suggestion.trial_id} already exists in session {session_id}"
            )

        # Check capacity
        if len(session_trials) >= self.max_trials_per_session:
            raise ValueError(
                f"Maximum trials per session exceeded: {self.max_trials_per_session}"
            )

        # Create trial state
        trial_state = TrialState(
            trial_id=suggestion.trial_id,
            session_id=session_id,
            trial_number=suggestion.trial_number,
            config=suggestion.config.copy(),
            status=TrialStatus.PENDING,
            suggestion_time=time.time(),
            backend_config_run_id=backend_config_run_id,
        )

        session_trials[suggestion.trial_id] = trial_state
        self._trial_access_times[suggestion.trial_id] = time.time()

        self._stats["total_trials_created"] += 1

        logger.debug(f"Registered trial {suggestion.trial_id} for session {session_id}")
        return trial_state

    def start_trial(self, session_id: str, trial_id: str) -> bool:
        """Mark trial as started.

        Args:
            session_id: Session ID
            trial_id: Trial ID

        Returns:
            True if trial was started successfully
        """
        trial_state = self.get_trial_state(session_id, trial_id)
        if not trial_state:
            logger.warning(f"Trial {trial_id} not found in session {session_id}")
            return False

        if trial_state.status != TrialStatus.PENDING:
            logger.warning(
                f"Cannot start trial {trial_id} in status {trial_state.status}"
            )
            return False

        trial_state.status = TrialStatus.RUNNING
        trial_state.start_time = time.time()
        self._trial_access_times[trial_id] = trial_state.start_time

        self._stats["status_transitions"] += 1
        self._stats["trials_running"] += 1

        logger.debug(f"Started trial {trial_id}")
        return True

    def complete_trial(
        self,
        session_id: str,
        trial_id: str,
        metrics: dict[str, float],
        duration: float | None = None,
    ) -> bool:
        """Mark trial as completed with results.

        Args:
            session_id: Session ID
            trial_id: Trial ID
            metrics: Trial evaluation metrics
            duration: Optional execution duration

        Returns:
            True if trial was completed successfully
        """
        trial_state = self.get_trial_state(session_id, trial_id)
        if not trial_state:
            logger.warning(f"Trial {trial_id} not found in session {session_id}")
            return False

        if trial_state.status != TrialStatus.RUNNING:
            logger.warning(
                f"Cannot complete trial {trial_id} in status {trial_state.status}"
            )
            return False

        # Validate metrics
        validate_or_raise(CoreValidators.validate_type(metrics, dict, "metrics"))

        trial_state.status = TrialStatus.COMPLETED
        trial_state.completion_time = time.time()
        trial_state.metrics = metrics.copy()

        if duration is not None:
            trial_state.duration = duration
        elif trial_state.start_time:
            trial_state.duration = trial_state.completion_time - trial_state.start_time

        self._trial_access_times[trial_id] = trial_state.completion_time

        self._stats["status_transitions"] += 1
        self._stats["trials_completed"] += 1
        self._stats["trials_running"] = max(0, self._stats["trials_running"] - 1)

        logger.debug(f"Completed trial {trial_id} with metrics: {metrics}")
        return True

    def fail_trial(
        self,
        session_id: str,
        trial_id: str,
        error_message: str,
        duration: float | None = None,
    ) -> bool:
        """Mark trial as failed.

        Args:
            session_id: Session ID
            trial_id: Trial ID
            error_message: Error description
            duration: Optional execution duration

        Returns:
            True if trial was marked as failed successfully
        """
        trial_state = self.get_trial_state(session_id, trial_id)
        if not trial_state:
            logger.warning(f"Trial {trial_id} not found in session {session_id}")
            return False

        if trial_state.status not in [TrialStatus.PENDING, TrialStatus.RUNNING]:
            logger.warning(
                f"Cannot fail trial {trial_id} in status {trial_state.status}"
            )
            return False

        # Capture original status before changing it
        was_running = trial_state.status == TrialStatus.RUNNING

        trial_state.status = TrialStatus.FAILED
        trial_state.completion_time = time.time()
        trial_state.error_message = error_message

        if duration is not None:
            trial_state.duration = duration
        elif trial_state.start_time:
            trial_state.duration = trial_state.completion_time - trial_state.start_time

        self._trial_access_times[trial_id] = trial_state.completion_time

        self._stats["status_transitions"] += 1
        self._stats["trials_failed"] += 1
        if was_running:
            self._stats["trials_running"] = max(0, self._stats["trials_running"] - 1)

        logger.warning(f"Failed trial {trial_id}: {error_message}")
        return True

    def get_trial_state(self, session_id: str, trial_id: str) -> TrialState | None:
        """Get trial state by session and trial ID.

        Args:
            session_id: Session ID
            trial_id: Trial ID

        Returns:
            Trial state or None if not found
        """
        session_trials = self._trials_by_session.get(session_id, {})
        trial_state = session_trials.get(trial_id)

        if trial_state:
            self._trial_access_times[trial_id] = time.time()

        return trial_state

    def get_session_trials(self, session_id: str) -> dict[str, TrialState]:
        """Get all trials for a session.

        Args:
            session_id: Session ID

        Returns:
            Dictionary mapping trial IDs to trial states
        """
        return self._trials_by_session.get(session_id, {}).copy()

    def get_active_trials(self, session_id: str) -> list[TrialState]:
        """Get trials that are currently running.

        Args:
            session_id: Session ID

        Returns:
            List of running trial states
        """
        session_trials = self._trials_by_session.get(session_id, {})
        return [
            trial
            for trial in session_trials.values()
            if trial.status == TrialStatus.RUNNING
        ]

    def get_completed_trials(self, session_id: str) -> list[TrialState]:
        """Get trials that have completed successfully.

        Args:
            session_id: Session ID

        Returns:
            List of completed trial states
        """
        session_trials = self._trials_by_session.get(session_id, {})
        return [
            trial
            for trial in session_trials.values()
            if trial.status == TrialStatus.COMPLETED
        ]

    def get_failed_trials(self, session_id: str) -> list[TrialState]:
        """Get trials that have failed.

        Args:
            session_id: Session ID

        Returns:
            List of failed trial states
        """
        session_trials = self._trials_by_session.get(session_id, {})
        return [
            trial
            for trial in session_trials.values()
            if trial.status == TrialStatus.FAILED
        ]

    def get_session_trial_summary(self, session_id: str) -> dict[str, Any]:
        """Get comprehensive trial summary for a session.

        Args:
            session_id: Session ID

        Returns:
            Dictionary with trial summary information
        """
        session_trials = self._trials_by_session.get(session_id, {})

        active_trials = self.get_active_trials(session_id)
        completed_trials = self.get_completed_trials(session_id)
        failed_trials = self.get_failed_trials(session_id)

        # Calculate average duration for completed trials
        completed_durations: list[float] = [
            d for trial in completed_trials if (d := trial.get_duration()) is not None
        ]
        avg_duration = (
            sum(completed_durations) / len(completed_durations)
            if completed_durations
            else None
        )

        return {
            "session_id": session_id,
            "total_trials": len(session_trials),
            "active_trials": len(active_trials),
            "completed_trials": len(completed_trials),
            "failed_trials": len(failed_trials),
            "pending_trials": len(session_trials)
            - len(active_trials)
            - len(completed_trials)
            - len(failed_trials),
            "average_duration": avg_duration,
            "has_trials": len(session_trials) > 0,
        }

    def remove_session_trials(self, session_id: str) -> int:
        """Remove all trials for a session.

        Args:
            session_id: Session ID

        Returns:
            Number of trials removed
        """
        session_trials = self._trials_by_session.get(session_id, {})
        trial_count = len(session_trials)

        # Remove trial access times
        for trial_id in session_trials.keys():
            self._trial_access_times.pop(trial_id, None)

        # Remove session trials
        if session_id in self._trials_by_session:
            del self._trials_by_session[session_id]

        if trial_count > 0:
            logger.info(f"Removed {trial_count} trials for session {session_id}")

        return trial_count

    def get_statistics(self) -> dict[str, Any]:
        """Get trial tracker statistics."""
        total_trials = sum(len(trials) for trials in self._trials_by_session.values())
        total_sessions = len(self._trials_by_session)

        return {
            **self._stats,
            "total_trials_tracked": total_trials,
            "sessions_with_trials": total_sessions,
            "avg_trials_per_session": (
                total_trials / total_sessions if total_sessions > 0 else 0
            ),
        }

    def cleanup_old_trials(self, session_id: str, max_trials: int = 1000) -> int:
        """Clean up old trials if session exceeds maximum.

        Args:
            session_id: Session ID
            max_trials: Maximum trials to keep

        Returns:
            Number of trials cleaned up
        """
        session_trials = self._trials_by_session.get(session_id, {})

        if len(session_trials) <= max_trials:
            return 0

        # Sort trials by access time (oldest first)
        trials_by_access = sorted(
            session_trials.items(), key=lambda x: self._trial_access_times.get(x[0], 0)
        )

        # Remove oldest trials
        trials_to_remove = len(session_trials) - max_trials
        removed_count = 0

        for trial_id, _ in trials_by_access[:trials_to_remove]:
            # Don't remove running trials
            if session_trials[trial_id].status != TrialStatus.RUNNING:
                del session_trials[trial_id]
                self._trial_access_times.pop(trial_id, None)
                removed_count += 1

        if removed_count > 0:
            logger.info(
                f"Cleaned up {removed_count} old trials for session {session_id}"
            )

        return removed_count
