"""RemoteOptimizer implementation with fallback support.

This module provides the RemoteOptimizer class that integrates with remote
optimization services while maintaining fallback to local optimization for
reliability and seamless operation.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability CONC-Quality-Performance FUNC-OPT-ALGORITHMS FUNC-CLOUD-HYBRID REQ-OPT-ALG-004 REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import UTC
from typing import Any, TypeVar

from traigent.api.types import TrialResult
from traigent.config.types import TraigentConfig
from traigent.evaluators.base import EvaluationExample
from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.remote_services import (
    DatasetSubset,
    OptimizationSession,
    OptimizationStrategy,
    RemoteOptimizationService,
    SmartTrialSuggestion,
)
from traigent.utils.exceptions import OptimizationError, ServiceError
from traigent.utils.logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


def _run_async_safely(coro: Coroutine[Any, Any, T]) -> T:
    """Run async coroutine from sync context, ensuring no event loop is running.

    Args:
        coro: The coroutine to execute

    Returns:
        The result of the coroutine

    Raises:
        RuntimeError: If called from within a running event loop
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "Cannot call synchronous method from an async context. "
            "Please use the '_async' version of this method instead "
            "(e.g., await optimizer.suggest_next_trial_async())."
        )

    return asyncio.run(coro)


class CloudOptimizer(BaseOptimizer):
    """Optimizer that delegates to cloud optimization service with local fallback.

    This optimizer provides the best of both worlds:
    - Advanced optimization algorithms from cloud services
    - Reliability through local fallback when cloud services are unavailable
    - Smart dataset subset selection for cost optimization
    - Enhanced exploration-exploitation strategies

    Note:
        This is distinct from RemoteOptimizer (in remote.py) which is a simpler
        privacy-aware scaffold. CloudOptimizer provides full session management,
        smart suggestions, and optimization strategies.
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        remote_service: RemoteOptimizationService,
        fallback_optimizer: BaseOptimizer | None = None,
        optimization_strategy: OptimizationStrategy | None = None,
        context: TraigentConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize remote optimizer with fallback support.

        Args:
            config_space: Dictionary defining parameter search space
            objectives: List of objective names to optimize
            remote_service: Remote optimization service to use
            fallback_optimizer: Local optimizer to use if remote service fails
            optimization_strategy: Strategy for smart optimization
            context: Optional TraigentConfig for accessing global configuration
            **kwargs: Additional algorithm-specific configuration
        """
        super().__init__(config_space, objectives, context, **kwargs)

        self.remote_service = remote_service
        self.fallback_optimizer = fallback_optimizer
        self.optimization_strategy = optimization_strategy or OptimizationStrategy()

        # State tracking
        self.session_id: str | None = None
        self.session: OptimizationSession | None = None
        self._using_fallback = False
        self._fallback_reason: str | None = None
        self._smart_suggestions: list[SmartTrialSuggestion] = []

        # Performance tracking
        self._remote_successes = 0
        self._remote_failures = 0
        self._fallback_uses = 0

        logger.info(
            f"Created CloudOptimizer with {remote_service.service_name} "
            f"and {fallback_optimizer.__class__.__name__ if fallback_optimizer else 'no'} fallback"
        )

    async def initialize_session(self) -> OptimizationSession:
        """Initialize optimization session with remote service.

        Returns:
            OptimizationSession if successful

        Raises:
            ServiceError: If session creation fails and no fallback available
        """
        if self.session_id:
            logger.warning("Session already initialized, returning existing session")
            if self.session is None:
                raise ServiceError("session_id exists but session is None")
            return self.session

        try:
            # Ensure remote service is connected
            if self.remote_service.status.value != "connected":
                await self.remote_service.connect()

            # Create optimization session
            session = await self.remote_service.create_session(
                config_space=self.config_space,
                objectives=self.objectives,
                optimization_strategy=self.optimization_strategy,
                context=self.context,
            )

            self.session_id = session.session_id
            self.session = session
            self._using_fallback = False

            logger.info(f"Initialized remote session: {self.session_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to initialize remote session: {e}")
            self._handle_remote_failure("session_creation", e)

            if not self.fallback_optimizer:
                raise ServiceError(f"No fallback optimizer available: {e}") from None

            # Switch to fallback mode immediately for session creation failures
            self._using_fallback = True
            self._fallback_reason = f"session_creation: {e}"

            # Return a mock session for fallback mode
            from datetime import datetime

            from traigent.optimizers.remote_services import OptimizationSessionStatus

            fallback_session = OptimizationSession(
                session_id="fallback_session",
                service_name="LocalFallback",
                config_space=self.config_space,
                objectives=self.objectives,
                algorithm="fallback",
                status=OptimizationSessionStatus.ACTIVE,
                created_at=datetime.now(UTC),
                optimization_strategy=self.optimization_strategy,
            )

            self.session = fallback_session
            self.session_id = fallback_session.session_id

            return fallback_session

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Synchronous interface for backward compatibility."""
        return _run_async_safely(self.suggest_next_trial_async(history))

    async def suggest_next_trial_async(
        self, history: list[TrialResult], remote_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Suggest next configuration using remote service or fallback.

        Args:
            history: List of previous trial results
            remote_context: Optional context data from the service

        Returns:
            Dictionary containing suggested parameter values
        """
        # Initialize session if needed
        if not self.session_id:
            await self.initialize_session()

        if self.session_id is None:
            raise ServiceError("Session not initialized")

        # Try remote service first
        if not self._using_fallback:
            try:
                config = await self.remote_service.suggest_configuration(
                    self.session_id, history, remote_context
                )

                self._remote_successes += 1
                self._trial_count += 1

                logger.debug(f"Remote suggestion successful: {config}")
                return config

            except Exception as e:
                logger.warning(f"Remote suggestion failed: {e}")
                self._handle_remote_failure("suggestion", e)

        # Use fallback optimizer
        if self.fallback_optimizer:
            try:
                config = await self.fallback_optimizer.suggest_next_trial_async(
                    history, remote_context
                )

                self._fallback_uses += 1
                self._trial_count += 1

                logger.debug(f"Fallback suggestion used: {config}")
                return config

            except Exception as e:
                logger.error(f"Fallback suggestion also failed: {e}")
                raise OptimizationError(
                    f"Both remote and fallback suggestions failed: {e}"
                ) from e

        raise OptimizationError("No available optimization method")

    async def suggest_smart_trial(
        self,
        history: list[TrialResult],
        full_dataset: list[EvaluationExample],
        remote_context: dict[str, Any] | None = None,
    ) -> SmartTrialSuggestion:
        """Get smart trial suggestion with configuration and dataset subset.

        This is the enhanced method that leverages the remote service's ability
        to select both optimal configurations AND strategic dataset subsets.

        Args:
            history: List of previous trial results
            full_dataset: Complete dataset of evaluation examples
            remote_context: Optional context data from the service

        Returns:
            SmartTrialSuggestion with config and strategic dataset subset
        """
        # Initialize session if needed
        if not self.session_id:
            await self.initialize_session()

        if self.session_id is None:
            raise ServiceError("Session not initialized")

        # Try remote service smart suggestion first
        if not self._using_fallback:
            try:
                suggestion = await self.remote_service.suggest_smart_trial(
                    self.session_id, history, full_dataset, remote_context
                )

                # Store suggestion for tracking
                self._smart_suggestions.append(suggestion)

                self._remote_successes += 1
                self._trial_count += 1

                logger.info(
                    f"Smart suggestion: {suggestion.exploration_type} with "
                    f"{suggestion.dataset_subset.size} examples (confidence: "
                    f"{suggestion.dataset_subset.confidence_level:.2f})"
                )

                return suggestion

            except Exception as e:
                logger.warning(f"Remote smart suggestion failed: {e}")
                self._handle_remote_failure("smart_suggestion", e)

        # Fallback to basic suggestion with simple dataset subset
        if self.fallback_optimizer:
            try:
                # Get basic configuration from fallback
                config = await self.fallback_optimizer.suggest_next_trial_async(
                    history, remote_context
                )

                # Create simple dataset subset
                subset_size = min(
                    self.optimization_strategy.min_examples_per_trial, len(full_dataset)
                )

                import random

                selected_examples = random.sample(full_dataset, subset_size)

                dataset_subset = DatasetSubset(
                    examples=selected_examples,
                    selection_strategy="random_fallback",
                    confidence_level=0.3,  # Low confidence for random selection
                    subset_id=f"fallback_{self._trial_count}",
                )

                suggestion = SmartTrialSuggestion(
                    config=config,
                    dataset_subset=dataset_subset,
                    exploration_type="exploration",  # Conservative default
                    priority=1,
                    metadata={"fallback": True, "reason": self._fallback_reason},
                )

                self._fallback_uses += 1
                self._trial_count += 1

                logger.info(f"Fallback smart suggestion with {subset_size} examples")
                return suggestion

            except Exception as e:
                logger.error(f"Fallback smart suggestion also failed: {e}")
                raise OptimizationError(
                    f"Both remote and fallback smart suggestions failed: {e}"
                ) from e

        raise OptimizationError("No available smart optimization method")

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Synchronous interface for backward compatibility."""
        return _run_async_safely(self.should_stop_async(history))

    async def should_stop_async(
        self, history: list[TrialResult], remote_context: dict[str, Any] | None = None
    ) -> bool:
        """Check if optimization should stop using remote service or fallback.

        Args:
            history: List of previous trial results
            remote_context: Optional context data from the service

        Returns:
            True if optimization should stop, False otherwise
        """
        # Check strategy-based stopping conditions first
        if self._check_strategy_stopping_conditions(history):
            return True

        # Initialize session if needed for remote service check
        if not self.session_id and not self._using_fallback:
            try:
                await self.initialize_session()
            except Exception as e:
                logger.warning(f"Failed to initialize session for stopping check: {e}")

        # Try remote service stopping check
        if not self._using_fallback and self.session_id:
            try:
                should_stop = await self.remote_service.should_stop_optimization(
                    self.session_id, history
                )

                if should_stop:
                    logger.info("Remote service recommends stopping optimization")
                    return True

            except Exception as e:
                logger.warning(f"Remote stopping check failed: {e}")
                self._handle_remote_failure("stopping_check", e)

        # Use fallback stopping logic
        if self.fallback_optimizer:
            return await self.fallback_optimizer.should_stop_async(
                history, remote_context
            )

        # Default stopping condition
        return len(history) >= 100  # Conservative default

    async def report_trial_result(self, trial_result: TrialResult) -> None:
        """Report trial result to remote service (if available).

        Args:
            trial_result: Result of the completed trial
        """
        if not self._using_fallback and self.session_id:
            try:
                await self.remote_service.report_trial_result(
                    self.session_id, trial_result
                )
                logger.debug(
                    f"Reported trial result to remote service: {trial_result.trial_id}"
                )

            except Exception as e:
                logger.warning(f"Failed to report trial result: {e}")
                # Don't switch to fallback for reporting failures

    async def close_session(self) -> None:
        """Close optimization session and cleanup resources."""
        if self.session_id and not self._using_fallback:
            try:
                await self.remote_service.close_session(self.session_id)
                logger.info(f"Closed remote session: {self.session_id}")

            except Exception as e:
                logger.warning(f"Failed to close remote session: {e}")

        self.session_id = None
        self.session = None
        self._smart_suggestions.clear()

    async def generate_candidates_async(
        self, max_candidates: int, remote_context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Generate multiple candidate configurations using cloud service.

        This method leverages the remote service's batch suggestion capabilities
        for more efficient candidate generation compared to sequential calls.

        Args:
            max_candidates: Maximum number of candidates to generate
            remote_context: Optional context from remote optimization service

        Returns:
            List of candidate configurations
        """
        # Initialize session if needed
        if not self.session_id:
            await self.initialize_session()

        # Try remote service first
        if not self._using_fallback:
            candidates = await self._generate_candidates_remote(
                max_candidates, remote_context
            )
            if candidates:
                return candidates

        # Use fallback optimizer
        return await self._generate_candidates_fallback(max_candidates, remote_context)

    async def _generate_candidates_remote(
        self, max_candidates: int, remote_context: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Generate candidates using remote service."""
        if self.session_id is None:
            raise ServiceError("Session not initialized")
        try:
            # Try batch suggestions first
            if hasattr(self.remote_service, "suggest_batch"):
                candidates = await self.remote_service.suggest_batch(
                    self.session_id, max_candidates, [], remote_context
                )
                if candidates:
                    self._remote_successes += 1
                    logger.debug(
                        f"Generated {len(candidates)} candidates via remote batch"
                    )
                    return list(candidates)  # type: ignore[return-value]

            # Fall back to sequential suggestions
            return await self._generate_candidates_sequential(
                max_candidates, remote_context
            )

        except Exception as e:
            logger.warning(f"Remote candidate generation failed: {e}")
            self._handle_remote_failure("generate_candidates", e)
            return []

    async def _generate_candidates_sequential(
        self, max_candidates: int, remote_context: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Generate candidates sequentially from remote service."""
        if self.session_id is None:
            raise ServiceError("Session not initialized")
        candidates: list[dict[str, Any]] = []
        history: list[TrialResult] = []

        for _ in range(max_candidates):
            try:
                config = await self.remote_service.suggest_configuration(
                    self.session_id, history, remote_context
                )
                candidates.append(config)
                self._remote_successes += 1
            except Exception as e:
                logger.debug(
                    f"Remote candidate generation stopped after "
                    f"{len(candidates)} candidates: {e}"
                )
                break

        return candidates

    async def _generate_candidates_fallback(
        self, max_candidates: int, remote_context: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Generate candidates using fallback optimizer or base class."""
        if self.fallback_optimizer:
            try:
                candidates = await self.fallback_optimizer.generate_candidates_async(
                    max_candidates, remote_context
                )
                self._fallback_uses += 1
                logger.debug(f"Generated {len(candidates)} candidates via fallback")
                return candidates

            except Exception as e:
                logger.error(f"Fallback candidate generation also failed: {e}")
                raise OptimizationError(
                    f"Both remote and fallback candidate generation failed: {e}"
                ) from e

        # No fallback available, use base class implementation
        return await super().generate_candidates_async(max_candidates, remote_context)

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get optimization performance statistics.

        Returns:
            Dictionary with performance metrics and service usage stats
        """
        total_remote_attempts = self._remote_successes + self._remote_failures
        remote_success_rate = (
            self._remote_successes / total_remote_attempts
            if total_remote_attempts > 0
            else 0.0
        )

        return {
            "remote_service": self.remote_service.service_name,
            "using_fallback": self._using_fallback,
            "fallback_reason": self._fallback_reason,
            "remote_successes": self._remote_successes,
            "remote_failures": self._remote_failures,
            "remote_success_rate": remote_success_rate,
            "fallback_uses": self._fallback_uses,
            "total_trials": self._trial_count,
            "smart_suggestions_count": len(self._smart_suggestions),
            "session_id": self.session_id,
        }

    def get_smart_suggestions_history(self) -> list[SmartTrialSuggestion]:
        """Get history of smart trial suggestions.

        Returns:
            List of SmartTrialSuggestion objects used in this optimization
        """
        return self._smart_suggestions.copy()

    # Private helper methods

    def _handle_remote_failure(self, operation: str, error: Exception) -> None:
        """Handle remote service failure and potentially switch to fallback.

        Args:
            operation: Name of the operation that failed
            error: The exception that occurred
        """
        self._remote_failures += 1

        # Switch to fallback after multiple failures
        failure_threshold = 3
        if self._remote_failures >= failure_threshold and self.fallback_optimizer:
            if not self._using_fallback:
                logger.warning(
                    f"Switching to fallback optimizer after {self._remote_failures} "
                    f"remote failures (last: {operation})"
                )
                self._using_fallback = True
                self._fallback_reason = f"Remote {operation} failed: {error}"

        logger.error(f"Remote {operation} failed: {error}")

    def _check_strategy_stopping_conditions(self, history: list[TrialResult]) -> bool:
        """Check strategy-based stopping conditions.

        Args:
            history: List of previous trial results

        Returns:
            True if should stop based on strategy conditions
        """
        strategy = self.optimization_strategy

        # Check total evaluation budget
        if strategy.max_total_evaluations:
            total_evaluations = sum(
                len(suggestion.dataset_subset.examples)
                for suggestion in self._smart_suggestions
            )
            if total_evaluations >= strategy.max_total_evaluations:
                logger.info(
                    f"Stopping: reached evaluation budget ({total_evaluations})"
                )
                return True

        # Check cost budget
        if strategy.max_cost_budget:
            total_cost = sum(
                suggestion.estimated_cost or 0 for suggestion in self._smart_suggestions
            )
            if total_cost >= strategy.max_cost_budget:
                logger.info(f"Stopping: reached cost budget ({total_cost})")
                return True

        # Check time budget
        if strategy.max_time_budget and self.session:
            from datetime import datetime

            elapsed = (datetime.now(UTC) - self.session.created_at).total_seconds()
            if elapsed >= strategy.max_time_budget:
                logger.info(f"Stopping: reached time budget ({elapsed}s)")
                return True

        # Check early stopping patience
        if (
            strategy.early_stopping_patience
            and len(history) >= strategy.early_stopping_patience
        ):
            # Look for improvement in recent trials
            primary_obj = self.objectives[0]
            recent_scores = []

            for trial in history[-strategy.early_stopping_patience :]:
                if trial.is_successful and primary_obj in trial.metrics:
                    recent_scores.append(trial.metrics[primary_obj])

            if recent_scores:
                best_recent = max(recent_scores)
                # Find best overall score
                all_scores = [
                    trial.metrics.get(primary_obj, 0)
                    for trial in history
                    if trial.is_successful and primary_obj in trial.metrics
                ]

                if all_scores:
                    best_overall = max(all_scores)
                    # Stop if no significant improvement
                    if best_recent < best_overall - 0.01:
                        logger.info(
                            f"Stopping: no improvement in {strategy.early_stopping_patience} trials"
                        )
                        return True

        return False

    def get_algorithm_info(self) -> dict[str, Any]:
        """Get information about this remote optimization algorithm."""
        info = super().get_algorithm_info()
        info.update(
            {
                "remote_service": self.remote_service.service_name,
                "fallback_optimizer": (
                    self.fallback_optimizer.__class__.__name__
                    if self.fallback_optimizer
                    else None
                ),
                "supports_smart_suggestions": True,
                "supports_adaptive_datasets": True,
                "using_fallback": self._using_fallback,
                "optimization_strategy": self.optimization_strategy.strategy_name,
            }
        )
        return info
