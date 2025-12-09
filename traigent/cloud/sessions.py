"""Session management for stateful optimization.

This module provides session management capabilities for maintaining
optimization state across multiple client-server interactions.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from traigent.cloud.models import (
    DatasetSubsetIndices,
    OptimizationSession,
    OptimizationSessionStatus,
    TrialResultSubmission,
    TrialStatus,
    TrialSuggestion,
)
from traigent.utils.exceptions import SessionError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Import validation utilities for enhanced state management
try:
    from traigent.utils.validation import CoreValidators, validate_or_raise

    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    logger.debug("Validation utilities not available")


class SessionStorage(Protocol):
    """Protocol for session storage implementations."""

    async def create(self, session: OptimizationSession) -> None:
        """Store a new session."""
        ...

    async def get(self, session_id: str) -> OptimizationSession | None:
        """Retrieve a session by ID."""
        ...

    async def update(self, session: OptimizationSession) -> None:
        """Update an existing session."""
        ...

    async def delete(self, session_id: str) -> None:
        """Delete a session."""
        ...

    async def list_active(
        self, user_id: str | None = None
    ) -> list[OptimizationSession]:
        """List active sessions, optionally filtered by user."""
        ...


class InMemorySessionStorage:
    """Simple in-memory session storage for development/testing."""

    def __init__(self) -> None:
        self._sessions: dict[str, OptimizationSession] = {}
        self._user_sessions: dict[str, list[str]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def create(self, session: OptimizationSession) -> None:
        """Store a new session."""
        async with self._lock:
            if session.session_id in self._sessions:
                raise SessionError(
                    f"Session {session.session_id} already exists"
                ) from None

            self._sessions[session.session_id] = session

            # Track by user if available
            user_id = session.metadata.get("user_id")
            if user_id:
                self._user_sessions[user_id].append(session.session_id)

    async def get(self, session_id: str) -> OptimizationSession | None:
        """Retrieve a session by ID."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def update(self, session: OptimizationSession) -> None:
        """Update an existing session."""
        async with self._lock:
            if session.session_id not in self._sessions:
                raise SessionError(f"Session {session.session_id} not found")

            session.updated_at = datetime.now(UTC)
            self._sessions[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        """Delete a session."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if not session:
                raise SessionError(f"Session {session_id} not found")

            # Remove from user tracking
            user_id = session.metadata.get("user_id")
            if user_id and session_id in self._user_sessions[user_id]:
                self._user_sessions[user_id].remove(session_id)

    async def list_active(
        self, user_id: str | None = None
    ) -> list[OptimizationSession]:
        """List active sessions, optionally filtered by user."""
        async with self._lock:
            sessions = []

            if user_id:
                # Get sessions for specific user
                session_ids = self._user_sessions.get(user_id, [])
                for session_id in session_ids:
                    session = self._sessions.get(session_id)
                    if session and session.status == OptimizationSessionStatus.ACTIVE:
                        sessions.append(session)
            else:
                # Get all active sessions
                for session in self._sessions.values():
                    if session.status == OptimizationSessionStatus.ACTIVE:
                        sessions.append(session)

            return sessions


class SessionState:
    """Core session state management without lifecycle complexities."""

    def __init__(
        self,
        session: OptimizationSession,
        mapping: (
            Any | None
        ) = None,  # SessionExperimentMapping - optional for compatibility
    ) -> None:
        """Initialize session state.

        Args:
            session: Optimization session
            mapping: Session-to-experiment mapping (optional for compatibility)
        """
        # Validate inputs
        if not session:
            raise ValueError("Session cannot be None")

        self.session = session
        self.mapping = mapping
        self.created_at = time.time()
        self.last_updated = self.created_at

        # State tracking
        self._is_dirty = False
        self._version = 0

    def update_session_status(self, status: OptimizationSessionStatus) -> bool:
        """Update session status with validation.

        Args:
            status: New session status

        Returns:
            True if status was updated successfully
        """
        try:
            # Validate status transition
            if not self._is_valid_status_transition(self.session.status, status):
                logger.warning(
                    f"Invalid status transition from {self.session.status} to {status} "
                    f"for session {self.session.session_id}"
                )
                return False

            old_status = self.session.status
            self.session.status = status
            self.session.updated_at = datetime.now(UTC)
            self.last_updated = time.time()
            self._mark_dirty()

            logger.info(
                f"Session {self.session.session_id} status changed: {old_status} -> {status}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update session status: {e}")
            return False

    def update_best_results(
        self, config: dict[str, Any], metrics: dict[str, float]
    ) -> bool:
        """Update session's best results.

        Args:
            config: Best configuration found
            metrics: Best metrics achieved

        Returns:
            True if results were updated successfully
        """
        try:
            # Validate inputs if validation is available
            if VALIDATION_AVAILABLE:
                validate_or_raise(CoreValidators.validate_type(config, dict, "config"))
                validate_or_raise(
                    CoreValidators.validate_type(metrics, dict, "metrics")
                )
            else:
                # Basic validation
                if not isinstance(config, dict) or not isinstance(metrics, dict):
                    raise ValueError(
                        "Config and metrics must be dictionaries"
                    ) from None

            # Check if this is actually better
            if self._is_better_result(metrics):
                self.session.best_config = config.copy()
                self.session.best_metrics = metrics.copy()
                self.session.updated_at = datetime.now(UTC)
                self.last_updated = time.time()
                self._mark_dirty()

                logger.info(
                    f"Updated best results for session {self.session.session_id}: {metrics}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to update best results: {e}")
            return False

    def increment_completed_trials(self) -> None:
        """Increment completed trials counter."""
        self.session.completed_trials += 1
        self.session.updated_at = datetime.now(UTC)
        self.last_updated = time.time()
        self._mark_dirty()

    def get_session_summary(self) -> dict[str, Any]:
        """Get comprehensive session summary.

        Returns:
            Dictionary with session summary information
        """
        return {
            "session_id": self.session.session_id,
            "status": self.session.status.value,
            "function_name": self.session.function_name,
            "objectives": self.session.objectives,
            "completed_trials": self.session.completed_trials,
            "best_config": self.session.best_config,
            "best_metrics": self.session.best_metrics,
            "created_at": self.session.created_at,
            "updated_at": self.session.updated_at,
            "backend_experiment_id": (
                getattr(self.mapping, "experiment_id", None) if self.mapping else None
            ),
            "backend_experiment_run_id": (
                getattr(self.mapping, "experiment_run_id", None)
                if self.mapping
                else None
            ),
            "state_version": self._version,
            "is_dirty": self._is_dirty,
        }

    def _is_valid_status_transition(
        self, current: OptimizationSessionStatus, new: OptimizationSessionStatus
    ) -> bool:
        """Validate if status transition is allowed.

        Args:
            current: Current status
            new: New status

        Returns:
            True if transition is valid
        """
        # Define valid transitions
        valid_transitions = {
            OptimizationSessionStatus.PENDING: [
                OptimizationSessionStatus.ACTIVE,
                OptimizationSessionStatus.CANCELLED,
            ],
            OptimizationSessionStatus.ACTIVE: [
                OptimizationSessionStatus.PAUSED,
                OptimizationSessionStatus.COMPLETED,
                OptimizationSessionStatus.FAILED,
                OptimizationSessionStatus.CANCELLED,
            ],
            OptimizationSessionStatus.PAUSED: [
                OptimizationSessionStatus.ACTIVE,
                OptimizationSessionStatus.CANCELLED,
                OptimizationSessionStatus.FAILED,
            ],
            OptimizationSessionStatus.COMPLETED: [],  # Terminal state
            OptimizationSessionStatus.FAILED: [],  # Terminal state
            OptimizationSessionStatus.CANCELLED: [],  # Terminal state
        }

        return new in valid_transitions.get(current, [])

    def _is_better_result(self, new_metrics: dict[str, float]) -> bool:
        """Check if new metrics are better than current best.

        Args:
            new_metrics: New metrics to compare

        Returns:
            True if new metrics are better
        """
        if not self.session.best_metrics:
            return True  # First result is always better

        if not self.session.objectives:
            return True  # No objectives defined, accept any result

        # Simple comparison using first objective (primary)
        primary_objective = self.session.objectives[0]

        current_value = self.session.best_metrics.get(primary_objective)
        new_value = new_metrics.get(primary_objective)

        if current_value is None or new_value is None:
            return new_value is not None  # Prefer having a value

        # Assume higher values are better (can be made configurable)
        return new_value > current_value

    def _mark_dirty(self) -> None:
        """Mark state as dirty (needs synchronization)."""
        self._is_dirty = True
        self._version += 1

    def mark_clean(self) -> None:
        """Mark state as clean (synchronized)."""
        self._is_dirty = False

    def is_dirty(self) -> bool:
        """Check if state needs synchronization."""
        return self._is_dirty

    def get_version(self) -> int:
        """Get current state version."""
        return self._version


class SessionStateManager:
    """Manages multiple session states with efficient operations."""

    def __init__(self, max_sessions: int = 1000) -> None:
        """Initialize session state manager.

        Args:
            max_sessions: Maximum number of sessions to track
        """
        self.max_sessions = max_sessions
        self._sessions: dict[str, SessionState] = {}
        self._session_access_times: dict[str, float] = {}

        # Statistics
        self._stats: dict[str, Any] = {
            "total_sessions_created": 0,
            "active_sessions": 0,
            "status_transitions": 0,
            "best_result_updates": 0,
        }

    def create_session_state(
        self, session: OptimizationSession, mapping: Any | None = None
    ) -> SessionState:
        """Create new session state.

        Args:
            session: Optimization session
            mapping: Session-to-experiment mapping (optional)

        Returns:
            Created session state
        """
        # Validate inputs
        if VALIDATION_AVAILABLE:
            validate_or_raise(
                CoreValidators.validate_string_non_empty(
                    session.session_id, "session_id"
                )
            )
        else:
            if not session.session_id or not session.session_id.strip():
                raise ValueError("Session ID cannot be empty") from None

        if session.session_id in self._sessions:
            raise ValueError(f"Session {session.session_id} already exists")

        # Check capacity
        if len(self._sessions) >= self.max_sessions:
            self._evict_oldest_session()

        # Create session state
        session_state = SessionState(session, mapping)
        self._sessions[session.session_id] = session_state
        self._session_access_times[session.session_id] = time.time()

        self._stats["total_sessions_created"] += 1
        self._stats["active_sessions"] += 1

        logger.info(f"Created session state for {session.session_id}")
        return session_state

    def get_session_state(self, session_id: str) -> SessionState | None:
        """Get session state by ID.

        Args:
            session_id: Session ID

        Returns:
            Session state or None if not found
        """
        session_state = self._sessions.get(session_id)
        if session_state:
            self._session_access_times[session_id] = time.time()
        return session_state

    def remove_session_state(self, session_id: str) -> bool:
        """Remove session state.

        Args:
            session_id: Session ID

        Returns:
            True if session was removed
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._session_access_times.pop(session_id, None)
            self._stats["active_sessions"] = max(0, self._stats["active_sessions"] - 1)

            logger.info(f"Removed session state for {session_id}")
            return True
        return False

    def list_session_ids(self) -> list[str]:
        """Get list of all session IDs."""
        return list(self._sessions.keys())

    def list_dirty_sessions(self) -> list[str]:
        """Get list of session IDs that need synchronization."""
        return [
            session_id
            for session_id, state in self._sessions.items()
            if state.is_dirty()
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get session state manager statistics."""
        return {
            **self._stats,
            "current_sessions": len(self._sessions),
            "capacity_utilization": len(self._sessions) / self.max_sessions,
            "dirty_sessions": len(self.list_dirty_sessions()),
        }

    def _evict_oldest_session(self) -> None:
        """Evict the oldest accessed session to make room."""
        if not self._session_access_times:
            return

        # Find oldest session
        oldest_session_id = min(self._session_access_times.items(), key=lambda x: x[1])[
            0
        ]

        self.remove_session_state(oldest_session_id)
        logger.info(f"Evicted oldest session {oldest_session_id} to make room")

    def cleanup_completed_sessions(self, ttl_seconds: int = 3600) -> int:
        """Clean up completed sessions older than TTL.

        Args:
            ttl_seconds: Time to live for completed sessions

        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        sessions_to_remove = []

        for session_id, session_state in self._sessions.items():
            if session_state.session.status in [
                OptimizationSessionStatus.COMPLETED,
                OptimizationSessionStatus.FAILED,
                OptimizationSessionStatus.CANCELLED,
            ]:
                last_access = self._session_access_times.get(session_id, current_time)
                if current_time - last_access > ttl_seconds:
                    sessions_to_remove.append(session_id)

        # Remove expired sessions
        for session_id in sessions_to_remove:
            self.remove_session_state(session_id)

        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} completed sessions")

        return len(sessions_to_remove)


class SessionManager:
    """Manages optimization sessions with lifecycle and state management."""

    def __init__(
        self,
        storage: SessionStorage | None = None,
        session_timeout_hours: int = 24,
        max_sessions_per_user: int = 10,
    ) -> None:
        """Initialize session manager.

        Args:
            storage: Session storage backend (defaults to in-memory)
            session_timeout_hours: Hours before inactive sessions expire
            max_sessions_per_user: Maximum concurrent sessions per user
        """
        self.storage = storage or InMemorySessionStorage()
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.max_sessions_per_user = max_sessions_per_user

        # Track trial history per session
        self._trial_history: dict[str, list[TrialResultSubmission]] = defaultdict(list)

        # Track suggested trials
        self._pending_trials: dict[str, TrialSuggestion] = {}

    async def create_session(
        self,
        function_name: str,
        configuration_space: dict[str, Any],
        objectives: list[str],
        max_trials: int,
        optimization_strategy: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OptimizationSession:
        """Create a new optimization session.

        Args:
            function_name: Name of function being optimized
            configuration_space: Parameter search space
            objectives: Optimization objectives
            max_trials: Maximum number of trials
            optimization_strategy: Optional optimization strategy
            metadata: Optional metadata including user_id

        Returns:
            Created OptimizationSession

        Raises:
            SessionError: If user has too many active sessions
        """
        # Check user session limit
        user_id = metadata.get("user_id") if metadata else None
        if user_id:
            active_sessions = await self.storage.list_active(user_id)
            if len(active_sessions) >= self.max_sessions_per_user:
                raise SessionError(
                    f"User {user_id} has reached maximum of "
                    f"{self.max_sessions_per_user} active sessions"
                )

        # Create session
        session = OptimizationSession(
            session_id=str(uuid.uuid4()),
            function_name=function_name,
            configuration_space=configuration_space,
            objectives=objectives,
            max_trials=max_trials,
            status=OptimizationSessionStatus.ACTIVE,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            optimization_strategy=optimization_strategy or self._default_strategy(),
            metadata=metadata or {},
        )

        await self.storage.create(session)

        logger.info(f"Created optimization session {session.session_id}")
        return session

    async def get_session(self, session_id: str) -> OptimizationSession:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            OptimizationSession

        Raises:
            SessionError: If session not found or expired
        """
        session = await self.storage.get(session_id)
        if not session:
            raise SessionError(f"Session {session_id} not found")

        # Check if session has expired
        if self._is_expired(session):
            await self._expire_session(session)
            raise SessionError(f"Session {session_id} has expired")

        return session

    async def suggest_next_trial(
        self, session_id: str, dataset_size: int
    ) -> TrialSuggestion | None:
        """Generate suggestion for the next trial.

        Args:
            session_id: Session identifier
            dataset_size: Total size of the dataset

        Returns:
            TrialSuggestion or None if optimization is complete

        Raises:
            SessionError: If session not found or not active
        """
        session = await self.get_session(session_id)

        if not session.can_continue():
            return None

        # Get trial history
        history = self._trial_history.get(session_id, [])

        # Generate configuration based on history
        config = await self._suggest_configuration(session, history)

        # Generate dataset subset indices
        subset_indices = await self._suggest_subset_indices(
            session, history, dataset_size
        )

        # Create trial suggestion
        trial_number = session.completed_trials + 1
        suggestion = TrialSuggestion(
            trial_id=f"{session_id}_trial_{trial_number}",
            session_id=session_id,
            trial_number=trial_number,
            config=config,
            dataset_subset=subset_indices,
            exploration_type=self._determine_exploration_type(session, history),
            priority=1,
            estimated_duration=self._estimate_duration(session, history),
        )

        # Store pending trial
        self._pending_trials[suggestion.trial_id] = suggestion

        logger.debug(f"Generated trial suggestion {suggestion.trial_id}")
        return suggestion

    async def submit_trial_result(self, result: TrialResultSubmission) -> None:
        """Submit results from a completed trial.

        Args:
            result: Trial result submission

        Raises:
            SessionError: If session not found or trial not recognized
        """
        session = await self.get_session(result.session_id)

        # Verify trial was suggested
        if result.trial_id not in self._pending_trials:
            logger.warning(f"Unknown trial {result.trial_id}")
        else:
            del self._pending_trials[result.trial_id]

        # Store result in history
        self._trial_history[result.session_id].append(result)

        # Update session
        session.completed_trials += 1

        # Update best results if improved
        if result.status == TrialStatus.COMPLETED and result.metrics:
            if self._is_better(
                result.metrics, session.best_metrics, session.objectives
            ):
                # Get config from the suggestion
                suggestion = self._pending_trials.get(result.trial_id)
                if suggestion:
                    session.best_config = suggestion.config
                session.best_metrics = result.metrics

        # Check if optimization should complete
        if session.completed_trials >= session.max_trials:
            session.status = OptimizationSessionStatus.COMPLETED

        await self.storage.update(session)

        logger.info(
            f"Submitted result for trial {result.trial_id}: "
            f"{session.completed_trials}/{session.max_trials} completed"
        )

    async def finalize_session(self, session_id: str) -> OptimizationSession:
        """Finalize an optimization session.

        Args:
            session_id: Session identifier

        Returns:
            Final OptimizationSession with results

        Raises:
            SessionError: If session not found
        """
        session = await self.get_session(session_id)

        if session.status == OptimizationSessionStatus.ACTIVE:
            session.status = OptimizationSessionStatus.COMPLETED
            await self.storage.update(session)

        logger.info(
            f"Finalized session {session_id}: "
            f"{session.completed_trials} trials, "
            f"best metrics: {session.best_metrics}"
        )

        return session

    async def cancel_session(self, session_id: str) -> None:
        """Cancel an active session.

        Args:
            session_id: Session identifier

        Raises:
            SessionError: If session not found
        """
        session = await self.get_session(session_id)

        if session.status == OptimizationSessionStatus.ACTIVE:
            session.status = OptimizationSessionStatus.CANCELLED
            await self.storage.update(session)

        logger.info(f"Cancelled session {session_id}")

    def get_trial_history(self, session_id: str) -> list[TrialResultSubmission]:
        """Get trial history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of trial results
        """
        return self._trial_history.get(session_id, [])

    def _is_expired(self, session: OptimizationSession) -> bool:
        """Check if session has expired."""
        if session.status != OptimizationSessionStatus.ACTIVE:
            return False

        age = datetime.now(UTC) - session.updated_at
        return age > self.session_timeout

    async def _expire_session(self, session: OptimizationSession) -> None:
        """Mark session as expired."""
        session.status = OptimizationSessionStatus.FAILED
        session.metadata["failure_reason"] = "Session expired due to inactivity"
        await self.storage.update(session)

    def _default_strategy(self) -> dict[str, Any]:
        """Get default optimization strategy."""
        return {
            "exploration_ratio": 0.3,
            "min_examples_per_trial": 10,
            "max_examples_per_trial": 100,
            "adaptive_sampling": True,
            "early_stopping": True,
        }

    async def _suggest_configuration(
        self, session: OptimizationSession, history: list[TrialResultSubmission]
    ) -> dict[str, Any]:
        """Suggest next configuration based on history.

        This is a simple implementation. In production, this would
        integrate with optimization algorithms.
        """
        import random

        config = {}
        for param, space in session.configuration_space.items():
            if isinstance(space, list):
                # Categorical parameter
                config[param] = random.choice(space)
            elif isinstance(space, tuple) and len(space) == 2:
                # Continuous parameter
                min_val, max_val = space
                if isinstance(min_val, int) and isinstance(max_val, int):
                    config[param] = random.randint(min_val, max_val)
                else:
                    config[param] = random.uniform(min_val, max_val)
            else:
                # Default
                config[param] = space

        return config

    async def _suggest_subset_indices(
        self,
        session: OptimizationSession,
        history: list[TrialResultSubmission],
        dataset_size: int,
    ) -> DatasetSubsetIndices:
        """Suggest dataset subset indices based on optimization progress."""
        strategy = session.optimization_strategy or {}

        # Determine subset size based on trial number
        trial_number = session.completed_trials + 1

        if trial_number <= 3:
            # Early exploration: small subsets
            subset_size = min(strategy.get("min_examples_per_trial", 10), dataset_size)
            selection_strategy = "diverse_sampling"
            confidence = 0.3
        elif trial_number <= 10:
            # Mid exploration: medium subsets
            subset_size = min(
                int(dataset_size * 0.2), strategy.get("max_examples_per_trial", 100)
            )
            selection_strategy = "representative_sampling"
            confidence = 0.6
        else:
            # Exploitation: larger subsets
            subset_size = min(
                int(dataset_size * 0.5), strategy.get("max_examples_per_trial", 100)
            )
            selection_strategy = "high_confidence_sampling"
            confidence = 0.9

        # Generate random indices for now
        # In production, use smarter selection strategies
        import random

        indices = sorted(random.sample(range(dataset_size), subset_size))

        return DatasetSubsetIndices(
            indices=indices,
            selection_strategy=selection_strategy,
            confidence_level=confidence,
            estimated_representativeness=confidence,
        )

    def _determine_exploration_type(
        self, session: OptimizationSession, history: list[TrialResultSubmission]
    ) -> str:
        """Determine exploration vs exploitation based on progress."""
        strategy = session.optimization_strategy or {}
        exploration_ratio = strategy.get("exploration_ratio", 0.3)

        if session.completed_trials < session.max_trials * exploration_ratio:
            return "exploration"
        elif session.completed_trials > session.max_trials * 0.9:
            return "verification"
        else:
            return "exploitation"

    def _estimate_duration(
        self, session: OptimizationSession, history: list[TrialResultSubmission]
    ) -> float | None:
        """Estimate duration for next trial based on history."""
        if not history:
            return None

        # Average of last 5 trials
        recent = history[-5:]
        durations = [r.duration for r in recent if r.duration > 0]

        if durations:
            return sum(durations) / len(durations)
        return None

    def _is_better(
        self,
        new_metrics: dict[str, float],
        best_metrics: dict[str, float] | None,
        objectives: list[str],
    ) -> bool:
        """Check if new metrics are better than current best."""
        if not best_metrics:
            return True

        # Simple comparison: check if primary objective improved
        if (
            objectives
            and objectives[0] in new_metrics
            and objectives[0] in best_metrics
        ):
            primary = objectives[0]
            # Assume higher is better for now
            # In production, handle minimization objectives
            return new_metrics[primary] > best_metrics[primary]

        return False


# Enhanced Session Lifecycle Management
# This provides advanced session lifecycle capabilities with backend integration


class RefactoredSessionLifecycleManager:
    """Enhanced session lifecycle manager with advanced state management.

    This class provides more sophisticated session lifecycle management
    capabilities while maintaining compatibility with the existing SessionManager.
    """

    def __init__(
        self,
        sync_interval: float = 5.0,
        enable_auto_sync: bool = True,
        max_concurrent_syncs: int = 3,
        max_sessions_cache: int = 1000,
        session_ttl: int = 3600,
        max_events_per_session: int = 100,
    ) -> None:
        """Initialize enhanced session lifecycle manager.

        Args:
            sync_interval: Interval for automatic synchronization (seconds)
            enable_auto_sync: Enable automatic synchronization
            max_concurrent_syncs: Maximum concurrent sync operations
            max_sessions_cache: Maximum number of sessions to keep in memory
            session_ttl: Time-to-live for inactive sessions (seconds)
            max_events_per_session: Maximum events to keep per session
        """
        # Initialize core session management components
        self.session_manager = SessionStateManager(max_sessions=max_sessions_cache)

        # Configuration
        self.session_ttl = session_ttl
        self.enable_auto_sync = enable_auto_sync
        self.sync_interval = sync_interval
        self.max_concurrent_syncs = max_concurrent_syncs
        self.max_events_per_session = max_events_per_session

        # Simple event tracking for basic functionality
        self._session_events: dict[str, list[dict[str, Any]]] = {}

        # Cleanup tracking
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 300  # 5 minutes

        logger.info("Enhanced session lifecycle manager initialized")

    def register_session(
        self,
        session: OptimizationSession,
        mapping: Any | None = None,  # SessionExperimentMapping
    ) -> None:
        """Register a new optimization session.

        Args:
            session: Optimization session
            mapping: Session-to-experiment mapping
        """
        logger.info(f"Registering session {session.session_id}")

        # Create session state
        self.session_manager.create_session_state(session, mapping)

        # Record event
        self._record_event(
            session.session_id,
            "session_created",
            {"function_name": session.function_name, "objectives": session.objectives},
        )

        # Perform maintenance if needed
        self._perform_maintenance()

        logger.info(f"Registered session {session.session_id}")

    def start_session(self, session_id: str) -> bool:
        """Mark session as started.

        Args:
            session_id: Session ID

        Returns:
            True if session was started successfully
        """
        session_state = self.session_manager.get_session_state(session_id)
        if not session_state:
            logger.warning(f"Session {session_id} not found")
            return False

        success = session_state.update_session_status(OptimizationSessionStatus.ACTIVE)

        if success:
            self._record_event(session_id, "session_started", {})
            logger.info(f"Started session {session_id}")

        return success

    def complete_session(
        self, session_id: str, final_results: dict[str, Any] | None = None
    ) -> bool:
        """Mark session as completed.

        Args:
            session_id: Session ID
            final_results: Optional final optimization results

        Returns:
            True if session was completed successfully
        """
        session_state = self.session_manager.get_session_state(session_id)
        if not session_state:
            return False

        success = session_state.update_session_status(
            OptimizationSessionStatus.COMPLETED
        )

        if success and final_results:
            session_state.update_best_results(
                final_results.get("best_config", {}),
                final_results.get("best_metrics", {}),
            )

        if success:
            self._record_event(
                session_id,
                "session_completed",
                {"results": final_results if final_results else {}},
            )
            logger.info(f"Completed session {session_id}")

        return success

    def fail_session(self, session_id: str, error_message: str) -> bool:
        """Mark session as failed.

        Args:
            session_id: Session ID
            error_message: Error description

        Returns:
            True if session was marked as failed successfully
        """
        session_state = self.session_manager.get_session_state(session_id)
        if not session_state:
            return False

        success = session_state.update_session_status(OptimizationSessionStatus.FAILED)

        if success:
            self._record_event(
                session_id, "session_failed", {"error_message": error_message}
            )
            logger.error(f"Failed session {session_id}: {error_message}")

        return success

    def cancel_session(self, session_id: str) -> bool:
        """Cancel an active session.

        Args:
            session_id: Session ID

        Returns:
            True if session was cancelled successfully
        """
        session_state = self.session_manager.get_session_state(session_id)
        if not session_state:
            return False

        success = session_state.update_session_status(
            OptimizationSessionStatus.CANCELLED
        )

        if success:
            self._record_event(session_id, "session_cancelled", {})
            logger.info(f"Cancelled session {session_id}")

        return success

    def get_session_state(self, session_id: str) -> SessionState | None:
        """Get complete session state.

        Args:
            session_id: Session ID

        Returns:
            Session state or None if not found
        """
        return self.session_manager.get_session_state(session_id)

    def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get session summary information.

        Args:
            session_id: Session ID

        Returns:
            Session summary or None if not found
        """
        session_state = self.session_manager.get_session_state(session_id)
        if not session_state:
            return None

        # Get basic session summary
        summary = session_state.get_session_summary()

        # Add event information
        events = self._session_events.get(session_id, [])
        summary["total_events"] = len(events)
        summary["last_event_time"] = events[-1]["timestamp"] if events else None

        return summary

    def get_active_sessions(self) -> list[str]:
        """Get list of active session IDs."""
        active_sessions = []

        for session_id in self.session_manager.list_session_ids():
            session_state = self.session_manager.get_session_state(session_id)
            if (
                session_state
                and session_state.session.status == OptimizationSessionStatus.ACTIVE
            ):
                active_sessions.append(session_id)

        return active_sessions

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive lifecycle manager statistics."""
        return {
            "session_manager": self.session_manager.get_statistics(),
            "total_events": sum(
                len(events) for events in self._session_events.values()
            ),
            "sessions_with_events": len(self._session_events),
            "enable_auto_sync": self.enable_auto_sync,
            "session_ttl": self.session_ttl,
        }

    def _record_event(
        self, session_id: str, event_type: str, data: dict[str, Any]
    ) -> None:
        """Record an event for a session.

        Args:
            session_id: Session ID
            event_type: Type of event
            data: Event data
        """
        if session_id not in self._session_events:
            self._session_events[session_id] = []

        event = {"timestamp": time.time(), "event_type": event_type, "data": data}

        events = self._session_events[session_id]
        events.append(event)

        # Keep only recent events to prevent memory growth
        if len(events) > self.max_events_per_session:
            self._session_events[session_id] = events[-self.max_events_per_session :]

    def _perform_maintenance(self) -> None:
        """Perform regular maintenance tasks."""
        now = time.time()

        # Only run maintenance periodically
        if now - self._last_cleanup_time < self._cleanup_interval:
            return

        logger.debug("Performing session lifecycle maintenance")

        # Clean up completed sessions
        cleaned = self.session_manager.cleanup_completed_sessions(self.session_ttl)

        # Clean up old events for completed sessions
        for session_id in list(self._session_events.keys()):
            session_state = self.session_manager.get_session_state(session_id)
            if not session_state or session_state.session.status in [
                OptimizationSessionStatus.COMPLETED,
                OptimizationSessionStatus.FAILED,
                OptimizationSessionStatus.CANCELLED,
            ]:
                # Clear events for completed sessions after TTL
                if (
                    not session_state
                    or now - session_state.last_updated > self.session_ttl
                ):
                    del self._session_events[session_id]

        self._last_cleanup_time = now

        logger.debug(f"Maintenance complete: {cleaned} sessions cleaned")

    def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Enhanced session lifecycle manager cleaned up")


# Backward compatibility aliases
SessionLifecycleManager = RefactoredSessionLifecycleManager

# Global lifecycle manager instance (optional - for backward compatibility)
lifecycle_manager = RefactoredSessionLifecycleManager()
