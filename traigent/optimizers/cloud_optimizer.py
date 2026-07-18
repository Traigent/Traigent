"""CloudOptimizer scaffold for future remote optimization services.

This module provides the CloudOptimizer class that integrates with remote
optimization services. Cloud remote execution is not implemented yet; use
hybrid for portal-tracked local execution.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability CONC-Quality-Performance FUNC-OPT-ALGORITHMS FUNC-CLOUD-HYBRID REQ-OPT-ALG-004 REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Coroutine
from datetime import UTC, datetime
from random import SystemRandom
from typing import TYPE_CHECKING, Any, TypeVar, cast

from traigent.api.types import TrialResult
from traigent.config.types import TraigentConfig
from traigent.evaluators.base import EvaluationExample
from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.remote_services import (
    DatasetSubset,
    OptimizationSession,
    OptimizationSessionStatus,
    OptimizationStrategy,
    RemoteOptimizationService,
    SmartTrialSuggestion,
)
from traigent.utils.exceptions import OptimizationError, ServiceError
from traigent.utils.logging import get_logger
from traigent.utils.objectives import (
    coerce_finite_objective_score,
    is_minimization_objective,
)

if TYPE_CHECKING:
    # Annotation-only: the optimizers layer must not import traigent.core at
    # runtime (traigent.core.orchestrator imports optimizers).
    from traigent.core.objectives import ObjectiveSchema

_SECURE_RANDOM = SystemRandom()

T = TypeVar("T")
logger = get_logger(__name__)

# Error message constants
_SESSION_NOT_INITIALIZED = "Session not initialized"


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
    """Optimizer scaffold that delegates to a future remote optimization service.

    This optimizer provides:
    - Advanced optimization hooks for future remote services
    - Smart dataset subset selection for cost optimization
    - Enhanced exploration-exploitation strategies

    Note:
        Cloud remote execution is not implemented yet.
        This optimizer is reserved for future enterprise features.
        This is distinct from RemoteOptimizer (in remote.py) which is a simpler
        scaffold. CloudOptimizer provides full session management,
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
        objective_schema: ObjectiveSchema | None = None,
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
            objective_schema: Optional declared objective schema. When supplied,
                its ``orientation`` is authoritative for early-stopping
                direction; name-pattern heuristics are only used for objectives
                the schema does not declare.
            **kwargs: Additional algorithm-specific configuration
        """
        super().__init__(config_space, objectives, context, **kwargs)

        self.remote_service = remote_service
        self.fallback_optimizer = fallback_optimizer
        self.optimization_strategy = optimization_strategy or OptimizationStrategy()
        self.objective_schema = objective_schema

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

            # Build a fallback session for local execution. The session_id MUST
            # be unique (no synthetic "fallback_session" constant) and the
            # is_fallback flag MUST be True so callers cannot mistake this for
            # a real remote session — see workspace CLAUDE.md SDK rule:
            # "Cloud failures fail closed by default. ... Never construct a
            #  synthetic session ID on remote failure."
            fallback_session = OptimizationSession(
                session_id=f"local_fallback_{uuid.uuid4().hex[:12]}",
                service_name="LocalFallback",
                config_space=self.config_space,
                objectives=self.objectives,
                algorithm="fallback",
                status=OptimizationSessionStatus.ACTIVE,
                created_at=datetime.now(UTC),
                optimization_strategy=self.optimization_strategy,
                is_fallback=True,
                metadata={
                    "fallback_reason": self._fallback_reason,
                    "remote_service_name": self.remote_service.service_name,
                },
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
            raise ServiceError(_SESSION_NOT_INITIALIZED)

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
                return cast(dict[str, Any], config)

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
            raise ServiceError(_SESSION_NOT_INITIALIZED)

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

                selected_examples = _SECURE_RANDOM.sample(full_dataset, subset_size)

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
            return bool(
                await self.fallback_optimizer.should_stop_async(history, remote_context)
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
        """Generate multiple candidate configurations using a remote service.

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
            raise ServiceError(_SESSION_NOT_INITIALIZED)
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
            raise ServiceError(_SESSION_NOT_INITIALIZED)
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
                return cast(list[dict[str, Any]], candidates)

            except Exception as e:
                logger.error(f"Fallback candidate generation also failed: {e}")
                raise OptimizationError(
                    f"Both remote and fallback candidate generation failed: {e}"
                ) from e

        # No fallback available, use base class implementation
        return cast(
            list[dict[str, Any]],
            await super().generate_candidates_async(max_candidates, remote_context),
        )

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

    def _primary_objective_orientation(self, objective_name: str) -> str | None:
        """Return the declared orientation for *objective_name*, if any.

        Returns None when no schema was supplied or the schema does not declare
        this objective, in which case callers fall back to name-pattern
        heuristics for backward compatibility with string-only objective flows.
        """
        if self.objective_schema is None:
            return None
        orientation: str | None = self.objective_schema.get_orientation(objective_name)
        return orientation

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
            and self.objectives
            and len(history) >= strategy.early_stopping_patience
        ):
            # Look for improvement in recent trials. Resolve orientation once so
            # the "best" aggregation and the "no improvement" test respect a
            # minimize primary objective instead of the old hard-coded maximize
            # assumption (#1915, sibling of #1466/#1852). The declared schema
            # orientation wins; name patterns are only a fallback for objectives
            # no schema declares.
            primary_obj = self.objectives[0]
            orientation = self._primary_objective_orientation(primary_obj)

            # A ``band`` (target-range) primary objective is not directional:
            # "best" means closeness to a target interval, so neither the
            # maximize (``best_recent <= baseline_best + delta``) nor the
            # minimize (``best_recent >= baseline_best - delta``) plateau test
            # is meaningful. Running the raw maximize arithmetic here would treat a
            # run correctly parked inside the band as "no improvement" and stop
            # it (or refuse to stop a drifting run). Bypass ONLY this patience
            # plateau gate; the budget checks above and the remote/fallback stop
            # decisions in ``should_stop_async`` still apply.
            if orientation == "band":
                return False

            minimize = is_minimization_objective(
                primary_obj,
                orientation=orientation,
            )
            min_delta = strategy.early_stopping_min_delta

            # Coerce every metric through ``coerce_finite_objective_score`` so
            # NaN / infinities / bool / str / None values are dropped instead of
            # poisoning or crashing the min/max comparisons below: a single NaN
            # makes every comparison False (silently disabling stopping), and a
            # str raises TypeError inside min()/max().
            def _valid_scores(trials: list[TrialResult]) -> list[float]:
                scores: list[float] = []
                for trial in trials:
                    if trial.is_successful and primary_obj in trial.metrics:
                        coerced = coerce_finite_objective_score(
                            trial.metrics[primary_obj]
                        )
                        if coerced is not None:
                            scores.append(coerced)
                return scores

            recent_scores = _valid_scores(history[-strategy.early_stopping_patience :])

            if recent_scores:
                # Baseline = best score BEFORE the patience window. The window
                # must be compared against the pre-window prefix: comparing it
                # against the best of ALL history (which contains the window
                # itself) made a flat plateau read as "still at the best"
                # (best_recent == best_overall), so the canonical no-improvement
                # case never stopped and only a material REGRESSION did — and a
                # larger min_delta made stopping LESS likely instead of more.
                baseline_scores = _valid_scores(
                    history[: -strategy.early_stopping_patience]
                )

                if baseline_scores:
                    if minimize:
                        best_recent = min(recent_scores)
                        baseline_best = min(baseline_scores)
                        # No improvement: the window never got more than
                        # min_delta below the pre-window best.
                        no_improvement = best_recent >= baseline_best - min_delta
                    else:
                        best_recent = max(recent_scores)
                        baseline_best = max(baseline_scores)
                        # No improvement: the window never got more than
                        # min_delta above the pre-window best.
                        no_improvement = best_recent <= baseline_best + min_delta
                    if no_improvement:
                        logger.info(
                            f"Stopping: no improvement in {strategy.early_stopping_patience} trials"
                        )
                        return True

        return False

    def get_algorithm_info(self) -> dict[str, Any]:
        """Get information about this remote optimization algorithm."""
        info: dict[str, Any] = super().get_algorithm_info()
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
