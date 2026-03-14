"""Progress management for optimization trials.

This module provides the ProgressManager class that handles progress tracking,
reporting, and callback invocation for optimization runs.

Extracted from OptimizationOrchestrator to reduce class complexity
and improve testability.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from traigent.utils.callbacks import ProgressInfo
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class ProgressState:
    """Tracks progress state for an optimization run."""

    total_trials: int | None = None
    completed_trials: int = 0
    successful_trials: int = 0
    failed_trials: int = 0
    start_time: float | None = None
    best_score: float | None = None
    best_config: dict[str, Any] | None = None
    algorithm_name: str = "Unknown"
    objectives: list[str] = field(default_factory=list)

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time since start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def avg_trial_time(self) -> float:
        """Calculate average time per trial."""
        if self.completed_trials == 0:
            return 0.0
        return self.elapsed_time / self.completed_trials

    @property
    def estimated_remaining(self) -> float | None:
        """Estimate remaining time based on average trial time."""
        if self.total_trials is None:
            return None
        remaining_trials = self.total_trials - self.completed_trials
        if remaining_trials <= 0:
            return 0.0
        return self.avg_trial_time * remaining_trials

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.completed_trials == 0:
            return 0.0
        return self.successful_trials / self.completed_trials


def create_progress_info(
    current_trial: int,
    state: ProgressState,
) -> ProgressInfo:
    """Create a ProgressInfo object from progress state.

    Args:
        current_trial: Current trial number
        state: Progress state

    Returns:
        ProgressInfo with current state
    """
    return ProgressInfo(
        current_trial=current_trial,
        total_trials=state.total_trials or 0,
        completed_trials=state.completed_trials,
        successful_trials=state.successful_trials,
        failed_trials=state.failed_trials,
        best_score=state.best_score,
        best_config=state.best_config,
        elapsed_time=state.elapsed_time,
        estimated_remaining=state.estimated_remaining,
        current_algorithm=state.algorithm_name,
    )


def log_progress(
    trial_count: int,
    state: ProgressState,
) -> None:
    """Log optimization progress.

    Args:
        trial_count: Current trial count
        state: Progress state
    """
    logger.info(
        f"Progress: {trial_count} trials, "
        f"best score: {'N/A' if state.best_score is None else f'{state.best_score:.4f}'}, "
        f"success rate: {state.success_rate:.2%}, "
        f"elapsed: {state.elapsed_time:.1f}s"
    )


class ProgressManager:
    """Manages optimization progress tracking and reporting.

    This class coordinates progress tracking, including:
    - Progress state management
    - Callback invocation
    - Progress logging

    Designed for injection into OptimizationOrchestrator.
    """

    def __init__(
        self,
        total_trials: int | None = None,
        algorithm_name: str = "Unknown",
        objectives: list[str] | None = None,
        log_interval: int = 10,
    ) -> None:
        """Initialize progress manager.

        Args:
            total_trials: Total expected trials (None for unlimited)
            algorithm_name: Name of the optimization algorithm
            objectives: List of objective names
            log_interval: Interval for logging progress
        """
        self._state = ProgressState(
            total_trials=total_trials,
            algorithm_name=algorithm_name,
            objectives=objectives or [],
        )
        self.log_interval = log_interval

    @property
    def state(self) -> ProgressState:
        """Get current progress state."""
        return self._state

    def start(self) -> None:
        """Start tracking progress."""
        self._state.start_time = time.time()
        self._state.completed_trials = 0
        self._state.successful_trials = 0
        self._state.failed_trials = 0

    def record_trial_completion(
        self,
        success: bool,
        metrics: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Record a completed trial.

        Args:
            success: Whether the trial was successful
            metrics: Trial metrics (for updating best score)
            config: Trial configuration
        """
        self._state.completed_trials += 1
        if success:
            self._state.successful_trials += 1
        else:
            self._state.failed_trials += 1

        # Update best score if provided
        if metrics and self._state.objectives:
            primary_objective = self._state.objectives[0]
            score = metrics.get(primary_objective)
            if score is not None:
                if self._state.best_score is None or score > self._state.best_score:
                    self._state.best_score = score
                    self._state.best_config = config

    def update_best_result(
        self,
        score: float | None,
        config: dict[str, Any] | None,
    ) -> None:
        """Update the best result directly.

        Args:
            score: Best score achieved
            config: Configuration that achieved best score
        """
        self._state.best_score = score
        self._state.best_config = config

    def get_progress_info(self, current_trial: int) -> ProgressInfo:
        """Get progress information for callbacks.

        Args:
            current_trial: Current trial number

        Returns:
            ProgressInfo with current state
        """
        return create_progress_info(current_trial, self._state)

    def should_log(self, trial_count: int) -> bool:
        """Check if progress should be logged at this trial count.

        Args:
            trial_count: Current trial count

        Returns:
            True if progress should be logged
        """
        return trial_count > 0 and trial_count % self.log_interval == 0

    def log_progress(self, trial_count: int) -> None:
        """Log current progress.

        Args:
            trial_count: Current trial count
        """
        log_progress(trial_count, self._state)

    def log_if_interval(self, trial_count: int) -> None:
        """Log progress if at log interval.

        Args:
            trial_count: Current trial count
        """
        if self.should_log(trial_count):
            self.log_progress(trial_count)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of progress.

        Returns:
            Dictionary with progress summary
        """
        return {
            "total_trials": self._state.total_trials,
            "completed_trials": self._state.completed_trials,
            "successful_trials": self._state.successful_trials,
            "failed_trials": self._state.failed_trials,
            "success_rate": self._state.success_rate,
            "elapsed_time": self._state.elapsed_time,
            "avg_trial_time": self._state.avg_trial_time,
            "best_score": self._state.best_score,
            "algorithm": self._state.algorithm_name,
        }
