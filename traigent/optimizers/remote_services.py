"""Remote optimization service interfaces and abstractions.

This module provides the core interfaces for integrating with remote optimization
services, enabling hybrid local/cloud optimization architectures.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability CONC-Quality-Compatibility FUNC-OPT-ALGORITHMS FUNC-CLOUD-HYBRID REQ-OPT-ALG-004 REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, cast

from traigent.api.types import TrialResult
from traigent.config.types import TraigentConfig
from traigent.evaluators.base import EvaluationExample
from traigent.utils.exceptions import ServiceError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class ServiceStatus(str, Enum):
    """Status of a remote service connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class OptimizationSessionStatus(str, Enum):
    """Status of an optimization session."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ServiceInfo:
    """Information about a remote optimization service."""

    name: str
    version: str
    supported_algorithms: list[str]
    max_concurrent_sessions: int
    capabilities: dict[str, Any] = field(default_factory=dict)
    endpoints: dict[str, str] = field(default_factory=dict)
    status: ServiceStatus = ServiceStatus.DISCONNECTED


@dataclass
class OptimizationSession:
    """Represents an active optimization session with a remote service."""

    session_id: str
    service_name: str
    config_space: dict[str, Any]
    objectives: list[str]
    algorithm: str
    status: OptimizationSessionStatus
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    # Session configuration
    max_trials: int | None = None
    timeout: float | None = None
    optimization_strategy: OptimizationStrategy | None = None

    # Progress tracking
    trials_completed: int = 0
    evaluations_performed: int = 0  # Total example evaluations (can be > trials)
    total_cost: float = 0.0  # Total cost spent
    total_time: float = 0.0  # Total time spent
    best_score: float | None = None
    best_config: dict[str, Any] | None = None

    # Smart optimization tracking
    pareto_frontier: list[dict[str, Any]] = field(default_factory=list)
    confidence_in_optimum: float = 0.0  # How confident we are we found the optimum


@dataclass
class ServiceMetrics:
    """Metrics about remote service performance."""

    response_time_ms: float
    success_rate: float
    total_requests: int
    failed_requests: int
    last_request_time: datetime
    average_session_duration: float = 0.0
    active_sessions: int = 0


@dataclass
class DatasetSubset:
    """Represents a strategically selected subset of evaluation examples."""

    examples: list[EvaluationExample]
    selection_strategy: (
        str  # e.g., "random", "diverse", "hard", "easy", "representative"
    )
    confidence_level: (
        float  # 0.0 to 1.0, how confident we are this subset is representative
    )
    subset_id: str  # Unique identifier for tracking
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of examples in this subset."""
        return len(self.examples)

    @property
    def indices(self) -> list[int]:
        """Original indices of examples (if available in metadata)."""
        return cast(
            list[int],
            self.metadata.get("original_indices", list(range(len(self.examples)))),
        )


@dataclass
class SmartTrialSuggestion:
    """Enhanced trial suggestion with configuration AND dataset subset."""

    config: dict[str, Any]  # Configuration to evaluate
    dataset_subset: DatasetSubset  # Which examples to use for evaluation

    # Strategic information
    exploration_type: (
        str  # e.g., "exploration", "exploitation", "verification", "refinement"
    )
    expected_value: float | None = None  # Expected performance (if available)
    uncertainty: float | None = None  # Uncertainty about this config

    # Resource optimization
    estimated_cost: float | None = None  # Estimated evaluation cost
    estimated_duration: float | None = None  # Estimated evaluation time
    priority: int = 0  # Higher numbers = higher priority

    # Metadata
    suggestion_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationStrategy:
    """Configuration for how the remote service should optimize."""

    # Resource constraints
    max_total_evaluations: int | None = None  # Total evaluation budget
    max_cost_budget: float | None = None  # Total cost budget
    max_time_budget: float | None = None  # Total time budget (seconds)

    # Exploration strategy
    exploration_ratio: float = 0.3  # 0.0 = pure exploitation, 1.0 = pure exploration
    early_stopping_patience: int = 10  # Stop if no improvement for N trials
    confidence_threshold: float = 0.95  # Stop when this confident about optimum

    # Dataset usage strategy
    min_examples_per_trial: int = 5  # Minimum examples for any evaluation
    max_examples_per_trial: int | None = None  # Maximum examples for any evaluation
    adaptive_sample_size: bool = True  # Increase sample size for promising configs

    # Multi-objective preferences
    pareto_preference: str | None = None  # "balanced", "speed", "accuracy", etc.
    objective_weights: dict[str, float] = field(default_factory=dict)

    # Metadata
    strategy_name: str = "smart_optimization"
    metadata: dict[str, Any] = field(default_factory=dict)


class RemoteOptimizationService(ABC):
    """Abstract interface for remote optimization services.

    This interface defines the contract for communicating with remote optimization
    services, whether they are TraiGent Cloud services or third-party providers.
    """

    def __init__(
        self,
        service_name: str,
        endpoint: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """Initialize remote optimization service.

        Args:
            service_name: Human-readable name for the service
            endpoint: Service endpoint URL
            api_key: Optional API key for authentication
            timeout: Default timeout for requests in seconds
            **kwargs: Additional service-specific configuration
        """
        self.service_name = service_name
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.config = kwargs

        # State tracking
        self._status = ServiceStatus.DISCONNECTED
        self._service_info: ServiceInfo | None = None
        self._active_sessions: dict[str, OptimizationSession] = {}

        # Memory bounds to prevent unbounded growth
        self._max_active_sessions = 100  # Maximum concurrent sessions
        self._metrics = ServiceMetrics(
            response_time_ms=0.0,
            success_rate=1.0,
            total_requests=0,
            failed_requests=0,
            last_request_time=datetime.now(UTC),
        )

    # Core Service Management

    @abstractmethod
    async def connect(self) -> ServiceInfo:
        """Connect to the remote service and retrieve service information.

        Returns:
            ServiceInfo containing service capabilities and status

        Raises:
            ServiceError: If connection fails
        """
        raise NotImplementedError

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the remote service and cleanup resources."""
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check service health and availability.

        Returns:
            Dictionary containing health status and metrics
        """
        raise NotImplementedError

    # Optimization Session Management

    @abstractmethod
    async def create_session(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        algorithm: str = "bayesian",
        max_trials: int | None = None,
        timeout: float | None = None,
        optimization_strategy: OptimizationStrategy | None = None,
        context: TraigentConfig | None = None,
        **session_kwargs: Any,
    ) -> OptimizationSession:
        """Create a new optimization session.

        Args:
            config_space: Parameter search space definition
            objectives: List of objectives to optimize
            algorithm: Optimization algorithm to use
            max_trials: Maximum number of trials for the session
            timeout: Session timeout in seconds
            optimization_strategy: Strategy for smart optimization
            context: Optional TraigentConfig context
            **session_kwargs: Additional session-specific parameters

        Returns:
            OptimizationSession representing the created session

        Raises:
            ServiceError: If session creation fails
            OptimizationError: If parameters are invalid
        """
        raise NotImplementedError

    @abstractmethod
    async def get_session(self, session_id: str) -> OptimizationSession:
        """Retrieve information about an existing session.

        Args:
            session_id: Unique session identifier

        Returns:
            OptimizationSession with current status

        Raises:
            ServiceError: If session not found or retrieval fails
        """
        raise NotImplementedError

    @abstractmethod
    async def close_session(self, session_id: str) -> None:
        """Close an optimization session and release resources.

        Args:
            session_id: Unique session identifier

        Raises:
            ServiceError: If session closure fails
        """
        raise NotImplementedError

    # Optimization Operations

    @abstractmethod
    async def suggest_configuration(
        self,
        session_id: str,
        trial_history: list[TrialResult],
        remote_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get next configuration suggestion from the remote service.

        Args:
            session_id: Active optimization session ID
            trial_history: List of previous trial results
            remote_context: Optional context data from the service

        Returns:
            Dictionary containing suggested parameter values

        Raises:
            ServiceError: If suggestion request fails
            OptimizationError: If session is invalid or optimization complete
        """
        raise NotImplementedError

    @abstractmethod
    async def report_trial_result(
        self, session_id: str, trial_result: TrialResult
    ) -> None:
        """Report the result of a trial back to the remote service.

        Args:
            session_id: Active optimization session ID
            trial_result: Result of the completed trial

        Raises:
            ServiceError: If result reporting fails
        """
        raise NotImplementedError

    @abstractmethod
    async def should_stop_optimization(
        self, session_id: str, trial_history: list[TrialResult]
    ) -> bool:
        """Check if optimization should stop according to remote service.

        Args:
            session_id: Active optimization session ID
            trial_history: List of previous trial results

        Returns:
            True if optimization should stop, False otherwise

        Raises:
            ServiceError: If stop check fails
        """
        raise NotImplementedError

    # Smart Optimization Operations (Enhanced Interface)

    async def suggest_smart_trial(
        self,
        session_id: str,
        trial_history: list[TrialResult],
        full_dataset: list[EvaluationExample],
        remote_context: dict[str, Any] | None = None,
    ) -> SmartTrialSuggestion:
        """Get smart trial suggestion with both config and dataset subset.

        This is the enhanced method that returns both what configuration to try
        AND which subset of examples to evaluate it on, optimizing the exploration-
        exploitation tradeoff while minimizing evaluation costs.

        Args:
            session_id: Active optimization session ID
            trial_history: List of previous trial results
            full_dataset: Complete dataset of evaluation examples
            remote_context: Optional context data from the service

        Returns:
            SmartTrialSuggestion with config and strategic dataset subset

        Raises:
            ServiceError: If suggestion request fails
            OptimizationError: If session is invalid or optimization complete
        """
        # Default implementation: fallback to simple suggestion with random subset
        session = await self.get_session(session_id)
        strategy = session.optimization_strategy or OptimizationStrategy()

        # Get simple configuration suggestion
        config = await self.suggest_configuration(
            session_id, trial_history, remote_context
        )

        # Create simple dataset subset
        import random

        subset_size = min(strategy.min_examples_per_trial, len(full_dataset))
        if strategy.max_examples_per_trial:
            subset_size = min(subset_size, strategy.max_examples_per_trial)

        selected_examples = random.sample(full_dataset, subset_size)
        subset = DatasetSubset(
            examples=selected_examples,
            selection_strategy="random_fallback",
            confidence_level=0.5,  # Medium confidence for random selection
            subset_id=f"fallback_{uuid.uuid4().hex[:8]}",
        )

        return SmartTrialSuggestion(
            config=config,
            dataset_subset=subset,
            exploration_type="exploration",  # Conservative default
            priority=1,
        )

    async def suggest_multiple_smart_trials(
        self,
        session_id: str,
        trial_history: list[TrialResult],
        full_dataset: list[EvaluationExample],
        num_suggestions: int,
        remote_context: dict[str, Any] | None = None,
    ) -> list[SmartTrialSuggestion]:
        """Get multiple smart trial suggestions for parallel evaluation.

        Args:
            session_id: Active optimization session ID
            trial_history: List of previous trial results
            full_dataset: Complete dataset of evaluation examples
            num_suggestions: Number of smart suggestions to generate
            remote_context: Optional context data from the service

        Returns:
            List of SmartTrialSuggestion objects

        Raises:
            ServiceError: If batch suggestion fails
        """
        # Default implementation calls suggest_smart_trial multiple times
        suggestions = []
        for _ in range(num_suggestions):
            try:
                suggestion = await self.suggest_smart_trial(
                    session_id, trial_history, full_dataset, remote_context
                )
                suggestions.append(suggestion)
            except Exception as e:
                logger.warning(f"Failed to get smart suggestion in batch: {e}")
                break

        return suggestions

    async def update_optimization_strategy(
        self, session_id: str, strategy: OptimizationStrategy
    ) -> None:
        """Update the optimization strategy for an active session.

        Args:
            session_id: Active optimization session ID
            strategy: New optimization strategy to apply

        Raises:
            ServiceError: If strategy update fails
        """
        # Default implementation: store in session (if service supports it)
        try:
            session = await self.get_session(session_id)
            session.optimization_strategy = strategy
            logger.info(f"Updated optimization strategy for session {session_id}")
        except Exception as e:
            logger.warning(f"Could not update strategy for session {session_id}: {e}")

    # Batch Operations

    async def suggest_multiple_configurations(
        self,
        session_id: str,
        trial_history: list[TrialResult],
        num_suggestions: int,
        remote_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Get multiple configuration suggestions for parallel evaluation.

        Args:
            session_id: Active optimization session ID
            trial_history: List of previous trial results
            num_suggestions: Number of configurations to suggest
            remote_context: Optional context data from the service

        Returns:
            List of suggested configurations

        Raises:
            ServiceError: If batch suggestion fails
        """
        # Default implementation calls suggest_configuration multiple times
        suggestions = []
        for _ in range(num_suggestions):
            try:
                suggestion = await self.suggest_configuration(
                    session_id, trial_history, remote_context
                )
                suggestions.append(suggestion)
            except Exception as e:
                logger.warning(f"Failed to get suggestion in batch: {e}")
                break

        return suggestions

    async def report_multiple_trial_results(
        self, session_id: str, trial_results: list[TrialResult]
    ) -> None:
        """Report multiple trial results in batch.

        Args:
            session_id: Active optimization session ID
            trial_results: List of trial results to report

        Raises:
            ServiceError: If batch reporting fails
        """
        # Default implementation reports results individually
        for trial_result in trial_results:
            try:
                await self.report_trial_result(session_id, trial_result)
            except Exception as e:
                logger.error(
                    f"Failed to report trial result {trial_result.trial_id}: {e}"
                )

    # Service Information and Monitoring

    @property
    def status(self) -> ServiceStatus:
        """Get current service connection status."""
        return self._status

    @property
    def service_info(self) -> ServiceInfo | None:
        """Get service information if connected."""
        return self._service_info

    @property
    def metrics(self) -> ServiceMetrics:
        """Get service performance metrics."""
        return self._metrics

    def get_active_sessions(self) -> list[OptimizationSession]:
        """Get list of currently active sessions."""
        return list(self._active_sessions.values())

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._active_sessions)

    # Protected helper methods

    def _update_metrics(self, request_start: float, success: bool) -> None:
        """Update service performance metrics.

        Args:
            request_start: Start time of the request
            success: Whether the request was successful
        """
        response_time = (time.time() - request_start) * 1000  # Convert to ms

        self._metrics.total_requests += 1
        if not success:
            self._metrics.failed_requests += 1

        # Update moving averages
        self._metrics.response_time_ms = (
            self._metrics.response_time_ms * 0.9 + response_time * 0.1
        )
        self._metrics.success_rate = (
            self._metrics.total_requests - self._metrics.failed_requests
        ) / max(1, self._metrics.total_requests)
        self._metrics.last_request_time = datetime.now(UTC)
        self._metrics.active_sessions = len(self._active_sessions)

    def _create_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"{self.service_name}_{uuid.uuid4().hex[:8]}"

    async def _validate_session(self, session_id: str) -> OptimizationSession:
        """Validate that a session exists and is active.

        Args:
            session_id: Session ID to validate

        Returns:
            OptimizationSession if valid

        Raises:
            ServiceError: If session is invalid or inactive
        """
        if session_id not in self._active_sessions:
            # Try to fetch from remote service
            try:
                session = await self.get_session(session_id)

                # Enforce max sessions limit before adding
                if len(self._active_sessions) >= self._max_active_sessions:
                    # Find and remove the oldest inactive session
                    oldest_session_id = None
                    oldest_time = float("inf")

                    for sid, sess in self._active_sessions.items():
                        if sess.status in [
                            OptimizationSessionStatus.COMPLETED,
                            OptimizationSessionStatus.FAILED,
                        ]:
                            if sess.created_at.timestamp() < oldest_time:
                                oldest_time = sess.created_at.timestamp()
                                oldest_session_id = sid

                    # If no inactive sessions, remove the oldest active one
                    if oldest_session_id is None:
                        oldest_session_id = min(
                            self._active_sessions.keys(),
                            key=lambda s: self._active_sessions[
                                s
                            ].created_at.timestamp(),
                        )

                    del self._active_sessions[oldest_session_id]
                    logger.debug(
                        f"Removed session {oldest_session_id} to stay within active session limit"
                    )

                self._active_sessions[session_id] = session
                return session
            except Exception as e:
                raise ServiceError(
                    f"Invalid or inactive session {session_id}: {e}"
                ) from None

        session = self._active_sessions[session_id]
        if session.status not in [
            OptimizationSessionStatus.ACTIVE,
            OptimizationSessionStatus.INITIALIZING,
        ]:
            raise ServiceError(
                f"Session {session_id} is not active (status: {session.status})"
            )

        return session


class MockRemoteService(RemoteOptimizationService):
    """Mock implementation of RemoteOptimizationService for testing and development.

    This service simulates remote optimization behavior locally, useful for:
    - Testing remote optimization workflows
    - Development without requiring actual remote services
    - Fallback when remote services are unavailable
    """

    def __init__(
        self,
        service_name: str = "MockTraiGentService",
        endpoint: str = "mock://localhost",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(service_name, endpoint, api_key, **kwargs)
        self._suggestion_count = 0

    async def connect(self) -> ServiceInfo:
        """Connect to mock service (instantaneous)."""
        await asyncio.sleep(0.01)  # Simulate network delay

        self._status = ServiceStatus.CONNECTED
        self._service_info = ServiceInfo(
            name=self.service_name,
            version="1.0.0-mock",
            supported_algorithms=["random", "grid", "bayesian"],
            max_concurrent_sessions=10,
            capabilities={
                "batch_suggestions": True,
                "session_persistence": False,
                "real_time_updates": True,
            },
            endpoints={
                "suggest": f"{self.endpoint}/suggest",
                "report": f"{self.endpoint}/report",
                "sessions": f"{self.endpoint}/sessions",
            },
            status=ServiceStatus.CONNECTED,
        )

        logger.info(f"Connected to mock service: {self.service_name}")
        return self._service_info

    async def disconnect(self) -> None:
        """Disconnect from mock service."""
        await asyncio.sleep(0.01)  # Simulate network delay

        # Close all active sessions
        for session_id in list(self._active_sessions.keys()):
            await self.close_session(session_id)

        self._status = ServiceStatus.DISCONNECTED
        self._service_info = None
        logger.info(f"Disconnected from mock service: {self.service_name}")

    async def health_check(self) -> dict[str, Any]:
        """Mock health check - always healthy."""
        await asyncio.sleep(0.01)  # Simulate network delay

        return {
            "status": "healthy",
            "uptime": 99.9,
            "active_sessions": len(self._active_sessions),
            "total_requests": self._metrics.total_requests,
            "response_time_ms": self._metrics.response_time_ms,
        }

    async def create_session(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        algorithm: str = "bayesian",
        max_trials: int | None = None,
        timeout: float | None = None,
        optimization_strategy: OptimizationStrategy | None = None,
        context: TraigentConfig | None = None,
        **session_kwargs: Any,
    ) -> OptimizationSession:
        """Create a mock optimization session."""
        request_start = time.time()

        try:
            await asyncio.sleep(0.1)  # Simulate session creation delay

            session_id = self._create_session_id()
            session = OptimizationSession(
                session_id=session_id,
                service_name=self.service_name,
                config_space=config_space,
                objectives=objectives,
                algorithm=algorithm,
                status=OptimizationSessionStatus.ACTIVE,
                created_at=datetime.now(UTC),
                max_trials=max_trials,
                timeout=timeout,
                optimization_strategy=optimization_strategy,
                metadata={
                    "mock_service": True,
                    "context": context.custom_params if context else {},
                    **session_kwargs,
                },
            )

            # Enforce max sessions limit
            if len(self._active_sessions) >= self._max_active_sessions:
                # Find and remove the oldest inactive session
                oldest_session_id = None
                oldest_time = float("inf")

                for sid, sess in self._active_sessions.items():
                    if sess.status in [
                        OptimizationSessionStatus.COMPLETED,
                        OptimizationSessionStatus.FAILED,
                    ]:
                        if sess.created_at.timestamp() < oldest_time:
                            oldest_time = sess.created_at.timestamp()
                            oldest_session_id = sid

                # If no inactive sessions, remove the oldest active one
                if oldest_session_id is None:
                    oldest_session_id = min(
                        self._active_sessions.keys(),
                        key=lambda s: self._active_sessions[s].created_at.timestamp(),
                    )

                del self._active_sessions[oldest_session_id]
                logger.debug(
                    f"Removed session {oldest_session_id} to stay within active session limit"
                )

            self._active_sessions[session_id] = session

            logger.info(f"Created mock session: {session_id}")
            self._update_metrics(request_start, True)
            return session

        except Exception as e:
            self._update_metrics(request_start, False)
            raise ServiceError(f"Failed to create mock session: {e}") from None

    async def get_session(self, session_id: str) -> OptimizationSession:
        """Get mock session information."""
        request_start = time.time()

        try:
            await asyncio.sleep(0.01)  # Simulate network delay

            if session_id not in self._active_sessions:
                raise ServiceError(f"Session {session_id} not found")

            session = self._active_sessions[session_id]
            self._update_metrics(request_start, True)
            return session

        except Exception:
            self._update_metrics(request_start, False)
            raise

    async def close_session(self, session_id: str) -> None:
        """Close mock session."""
        request_start = time.time()

        try:
            await asyncio.sleep(0.01)  # Simulate network delay

            if session_id in self._active_sessions:
                self._active_sessions[session_id].status = (
                    OptimizationSessionStatus.COMPLETED
                )
                del self._active_sessions[session_id]
                logger.info(f"Closed mock session: {session_id}")

            self._update_metrics(request_start, True)

        except Exception as e:
            self._update_metrics(request_start, False)
            raise ServiceError(f"Failed to close mock session: {e}") from None

    async def suggest_configuration(
        self,
        session_id: str,
        trial_history: list[TrialResult],
        remote_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate mock configuration suggestion."""
        request_start = time.time()

        try:
            await asyncio.sleep(0.05)  # Simulate computation delay

            session = await self._validate_session(session_id)

            # Simple mock suggestion logic
            config = {}
            self._suggestion_count += 1

            for param_name, param_def in session.config_space.items():
                if isinstance(param_def, list):
                    # Categorical parameter
                    import random

                    config[param_name] = random.choice(param_def)
                elif isinstance(param_def, tuple) and len(param_def) == 2:
                    # Continuous parameter
                    import random

                    low, high = param_def
                    if isinstance(low, int) and isinstance(high, int):
                        config[param_name] = random.randint(low, high)
                    else:
                        config[param_name] = random.uniform(low, high)
                else:
                    # Fixed parameter
                    config[param_name] = param_def

            # Add mock metadata
            config["_mock_suggestion_id"] = self._suggestion_count
            if remote_context:
                config["_remote_context"] = remote_context

            logger.debug(f"Mock service suggested config for {session_id}: {config}")
            self._update_metrics(request_start, True)
            return config

        except Exception as e:
            self._update_metrics(request_start, False)
            raise ServiceError(f"Failed to suggest configuration: {e}") from None

    async def report_trial_result(
        self, session_id: str, trial_result: TrialResult
    ) -> None:
        """Accept mock trial result report."""
        request_start = time.time()

        try:
            await asyncio.sleep(0.02)  # Simulate network delay

            session = await self._validate_session(session_id)

            # Update session with trial result
            session.trials_completed += 1

            if trial_result.is_successful:
                primary_objective = session.objectives[0]
                score = trial_result.metrics.get(primary_objective, 0.0)

                if session.best_score is None or score > session.best_score:
                    session.best_score = score
                    session.best_config = trial_result.config.copy()

            logger.debug(
                f"Mock service received trial result for {session_id}: score={trial_result.metrics}"
            )
            self._update_metrics(request_start, True)

        except Exception as e:
            self._update_metrics(request_start, False)
            raise ServiceError(f"Failed to report trial result: {e}") from None

    async def should_stop_optimization(
        self, session_id: str, trial_history: list[TrialResult]
    ) -> bool:
        """Mock stopping condition check."""
        request_start = time.time()

        try:
            await asyncio.sleep(0.01)  # Simulate network delay

            session = await self._validate_session(session_id)

            # Simple stopping logic: stop after max_trials or if no improvement for 10 trials
            if session.max_trials and len(trial_history) >= session.max_trials:
                logger.info(f"Mock service stopping {session_id}: max trials reached")
                self._update_metrics(request_start, True)
                return True

            # Stop if no improvement in last 10 trials (mock convergence)
            if len(trial_history) >= 10:
                recent_scores = []
                primary_objective = session.objectives[0]

                for trial in trial_history[-10:]:
                    if trial.is_successful:
                        score = trial.metrics.get(primary_objective, 0.0)
                        recent_scores.append(score)

                if recent_scores and max(recent_scores) <= min(recent_scores) + 0.01:
                    logger.info(f"Mock service stopping {session_id}: converged")
                    self._update_metrics(request_start, True)
                    return True

            self._update_metrics(request_start, True)
            return False

        except Exception as e:
            self._update_metrics(request_start, False)
            raise ServiceError(f"Failed to check stopping condition: {e}") from None

    # Enhanced Smart Optimization Implementation

    async def suggest_smart_trial(
        self,
        session_id: str,
        trial_history: list[TrialResult],
        full_dataset: list[EvaluationExample],
        remote_context: dict[str, Any] | None = None,
    ) -> SmartTrialSuggestion:
        """Mock implementation of smart trial suggestion."""
        request_start = time.time()

        try:
            await asyncio.sleep(0.1)  # Simulate smart computation delay

            session = await self._validate_session(session_id)
            strategy = session.optimization_strategy or OptimizationStrategy()

            # Generate configuration using existing logic
            config = await self.suggest_configuration(
                session_id, trial_history, remote_context
            )

            # Smart dataset subset selection
            dataset_subset = self._select_smart_dataset_subset(
                full_dataset, trial_history, strategy, len(trial_history)
            )

            # Determine exploration type based on trial count and performance
            exploration_type = self._determine_exploration_type(trial_history, strategy)

            # Estimate cost and priority
            estimated_cost = dataset_subset.size * 0.01  # Mock cost per example
            estimated_duration = dataset_subset.size * 0.05  # Mock duration per example

            # Priority based on exploration type and uncertainty
            priority = self._calculate_priority(
                exploration_type, dataset_subset.confidence_level
            )

            suggestion = SmartTrialSuggestion(
                config=config,
                dataset_subset=dataset_subset,
                exploration_type=exploration_type,
                estimated_cost=estimated_cost,
                estimated_duration=estimated_duration,
                priority=priority,
                metadata={
                    "mock_service": True,
                    "trial_count": len(trial_history),
                    "strategy": strategy.strategy_name,
                },
            )

            logger.debug(
                f"Mock smart suggestion for {session_id}: {exploration_type} "
                f"with {dataset_subset.size} examples"
            )

            self._update_metrics(request_start, True)
            return suggestion

        except Exception as e:
            self._update_metrics(request_start, False)
            raise ServiceError(
                f"Failed to generate smart trial suggestion: {e}"
            ) from None

    def _select_smart_dataset_subset(
        self,
        full_dataset: list[EvaluationExample],
        trial_history: list[TrialResult],
        strategy: OptimizationStrategy,
        trial_count: int,
    ) -> DatasetSubset:
        """Select strategic dataset subset based on optimization state."""
        import random

        total_examples = len(full_dataset)
        min_size = strategy.min_examples_per_trial
        max_size = strategy.max_examples_per_trial or total_examples

        # Adaptive sizing based on trial count and strategy
        if trial_count < 3:
            # Early exploration: use small subsets to test many configs quickly
            subset_size = min_size
            selection_strategy = "diverse_sampling"
            confidence = 0.3
        elif trial_count < 10:
            # Mid exploration: moderate subsets for better estimates
            subset_size = min(min_size * 2, max_size)
            selection_strategy = "representative_sampling"
            confidence = 0.6
        else:
            # Later stages: larger subsets for exploitation and verification
            subset_size = min(min_size * 3, max_size, total_examples // 2)
            selection_strategy = "high_confidence_sampling"
            confidence = 0.8

        # Ensure we don't exceed available examples
        subset_size = min(subset_size, total_examples)

        # Mock intelligent selection (in real service, this would be much more sophisticated)
        if selection_strategy == "diverse_sampling":
            # Try to get diverse examples across different difficulty levels
            selected = random.sample(full_dataset, subset_size)
        elif selection_strategy == "representative_sampling":
            # Balanced selection representing the full dataset
            selected = random.sample(full_dataset, subset_size)
        else:  # high_confidence_sampling
            # Focus on examples that give reliable signals
            selected = random.sample(full_dataset, subset_size)

        return DatasetSubset(
            examples=selected,
            selection_strategy=selection_strategy,
            confidence_level=confidence,
            subset_id=f"smart_{uuid.uuid4().hex[:8]}",
            metadata={
                "original_indices": list(range(len(selected))),  # Mock indices
                "total_dataset_size": total_examples,
                "selection_reason": f"Trial {trial_count}: {selection_strategy}",
            },
        )

    def _determine_exploration_type(
        self, trial_history: list[TrialResult], strategy: OptimizationStrategy
    ) -> str:
        """Determine what type of exploration this trial should be."""
        trial_count = len(trial_history)

        if trial_count < 5:
            return "exploration"  # Pure exploration early on
        elif trial_count < 15:
            # Mixed exploration/exploitation based on recent performance
            if trial_history:
                recent_improvements = 0
                for i in range(min(5, len(trial_history))):
                    if i > 0 and trial_history[-i].metrics:
                        primary_obj = list(trial_history[-i].metrics.keys())[0]
                        current_score = trial_history[-i].metrics.get(primary_obj, 0)
                        prev_score = trial_history[-i - 1].metrics.get(primary_obj, 0)
                        if current_score > prev_score:
                            recent_improvements += 1

                if recent_improvements >= 2:
                    return "exploitation"  # We're on a good track
                else:
                    return "exploration"  # Need to explore more
            return "exploration"
        else:
            # Later stages: focus on verification and refinement
            return "verification" if trial_count % 3 == 0 else "refinement"

    def _calculate_priority(self, exploration_type: str, confidence: float) -> int:
        """Calculate priority for trial suggestion."""
        base_priority = {
            "exploration": 1,
            "exploitation": 3,
            "verification": 2,
            "refinement": 4,
        }.get(exploration_type, 1)

        # Higher confidence = higher priority
        confidence_bonus = int(confidence * 2)

        return base_priority + confidence_bonus
