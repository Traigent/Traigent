"""Interactive optimizer for client-side execution with remote guidance.

This optimizer separates the suggestion generation from execution, allowing
the client to execute functions locally while receiving configuration and
dataset subset suggestions from a remote service.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability CONC-Quality-Performance FUNC-OPT-ALGORITHMS FUNC-CLOUD-HYBRID REQ-OPT-ALG-004 REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from traigent.api.types import TrialResult
from traigent.config.types import TraigentConfig
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import OptimizationError
from traigent.utils.logging import get_logger

# Cloud models - required at runtime for this optimizer
try:
    from traigent.cloud.models import (
        NextTrialRequest,
        NextTrialResponse,
        OptimizationFinalizationRequest,
        OptimizationFinalizationResponse,
        OptimizationSession,
        OptimizationSessionStatus,
        SessionCreationRequest,
        SessionCreationResponse,
        TrialResultSubmission,
        TrialStatus,
        TrialSuggestion,
    )

    _CLOUD_MODELS_AVAILABLE = True
except ModuleNotFoundError as err:
    # Check .name to distinguish missing cloud vs broken transitive dependency
    if err.name and err.name.startswith("traigent.cloud"):
        _CLOUD_MODELS_AVAILABLE = False
    else:
        raise  # Re-raise for broken dependencies like missing pydantic
    if TYPE_CHECKING:
        from traigent.cloud.models import (
            NextTrialRequest,
            NextTrialResponse,
            OptimizationFinalizationRequest,
            OptimizationFinalizationResponse,
            OptimizationSession,
            OptimizationSessionStatus,
            SessionCreationRequest,
            SessionCreationResponse,
            TrialResultSubmission,
            TrialStatus,
            TrialSuggestion,
        )

logger = get_logger(__name__)


def _require_cloud_models() -> None:
    """Raise FeatureNotAvailableError if cloud models are not available."""
    if not _CLOUD_MODELS_AVAILABLE:
        from traigent.utils.exceptions import FeatureNotAvailableError

        raise FeatureNotAvailableError(
            "Interactive optimization with remote guidance",
            plugin_name="traigent-cloud",
            install_hint="pip install traigent[cloud]",
        )


class RemoteGuidanceService(Protocol):
    """Protocol for remote guidance services."""

    async def create_session(
        self, request: SessionCreationRequest
    ) -> SessionCreationResponse:
        """Create a new optimization session."""
        ...

    async def get_next_trial(self, request: NextTrialRequest) -> NextTrialResponse:
        """Get next trial suggestion."""
        ...

    async def submit_result(self, result: TrialResultSubmission) -> None:
        """Submit trial result."""
        ...

    async def finalize_session(
        self, request: OptimizationFinalizationRequest
    ) -> OptimizationFinalizationResponse:
        """Finalize optimization session."""
        ...


class InteractiveOptimizer(BaseOptimizer):
    """Optimizer that uses remote guidance for configuration selection.

    This optimizer enables a hybrid approach where:
    - Configuration and dataset subset suggestions come from a remote service
    - Actual function execution happens on the client side
    - Results are reported back to the service for next suggestion

    This approach is ideal when:
    - The function cannot be serialized or executed remotely
    - Data privacy requirements prevent sending full datasets
    - Low latency execution is required
    """

    def __init__(
        self,
        config_space: dict[str, Any],
        objectives: list[str],
        remote_service: RemoteGuidanceService,
        dataset_metadata: dict[str, Any] | None = None,
        optimization_strategy: dict[str, Any] | None = None,
        context: TraigentConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize interactive optimizer.

        Args:
            config_space: Configuration space for optimization
            objectives: List of objectives to optimize
            remote_service: Service providing optimization guidance
            dataset_metadata: Metadata about the dataset (size, type, etc.)
            optimization_strategy: Strategy for optimization
            context: Optional TraigentConfig for global settings
            **kwargs: Additional optimizer configuration

        Raises:
            FeatureNotAvailableError: If cloud models are not installed
        """
        _require_cloud_models()
        super().__init__(config_space, objectives, context, **kwargs)

        self.remote_service = remote_service
        self.dataset_metadata = dataset_metadata or {}
        self.optimization_strategy = optimization_strategy

        # Session management
        self.session: OptimizationSession | None = None
        self.session_id: str | None = None

        # Tracking
        self._pending_trials: dict[str, TrialSuggestion] = {}
        self._completed_trials: list[TrialResultSubmission] = []
        self._start_time: float | None = None

    async def initialize_session(
        self,
        function_name: str,
        max_trials: int,
        user_id: str | None = None,
        billing_tier: str = "standard",
    ) -> OptimizationSession:
        """Initialize optimization session with remote service.

        Args:
            function_name: Name of function being optimized
            max_trials: Maximum number of trials
            user_id: Optional user identifier
            billing_tier: Billing tier for the user

        Returns:
            Created OptimizationSession

        Raises:
            OptimizationError: If session creation fails
        """
        try:
            request = SessionCreationRequest(
                function_name=function_name,
                configuration_space=self.config_space,
                objectives=self.objectives,
                dataset_metadata=self.dataset_metadata,
                max_trials=max_trials,
                optimization_strategy=self.optimization_strategy,
                user_id=user_id,
                billing_tier=billing_tier,
            )

            response = await self.remote_service.create_session(request)

            # Create local session object
            self.session = OptimizationSession(
                session_id=response.session_id,
                function_name=function_name,
                configuration_space=self.config_space,
                objectives=self.objectives,
                max_trials=max_trials,
                status=response.status,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                optimization_strategy=response.optimization_strategy,
                metadata=response.metadata,
            )

            self.session_id = response.session_id
            self._start_time = time.time()

            logger.info(
                f"Initialized interactive optimization session {self.session_id} "
                f"for {function_name}"
            )

            return self.session

        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
            raise OptimizationError(f"Session initialization failed: {e}") from None

    async def get_next_suggestion(
        self,
        dataset_size: int,
        previous_results: list[TrialResultSubmission] | None = None,
    ) -> TrialSuggestion | None:
        """Get next trial suggestion from remote service.

        Args:
            dataset_size: Total size of the dataset
            previous_results: Optional list of previous results

        Returns:
            TrialSuggestion or None if optimization is complete

        Raises:
            OptimizationError: If no session or request fails
        """
        if not self.session_id:
            raise OptimizationError("No active session. Call initialize_session first.")

        try:
            request = NextTrialRequest(
                session_id=self.session_id,
                previous_results=previous_results or self._completed_trials[-5:],
                request_metadata={
                    "dataset_size": dataset_size,
                    "completed_trials": len(self._completed_trials),
                },
            )

            response = await self.remote_service.get_next_trial(request)

            # Update local session status
            if self.session:
                self.session.status = response.session_status

            if not response.should_continue or not response.suggestion:
                logger.info(f"Optimization complete: {response.reason}")
                return None

            suggestion = response.suggestion

            # Store pending trial
            self._pending_trials[suggestion.trial_id] = suggestion

            logger.info(
                f"Got suggestion {suggestion.trial_id}: "
                f"{suggestion.exploration_type} with "
                f"{len(suggestion.dataset_subset.indices)} examples"
            )

            return suggestion

        except Exception as e:
            logger.error(f"Failed to get next suggestion: {e}")
            raise OptimizationError(f"Failed to get suggestion: {e}") from None

    async def report_results(
        self,
        trial_id: str,
        metrics: dict[str, float],
        duration: float,
        status: TrialStatus = TrialStatus.COMPLETED,
        outputs_sample: list[Any] | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Report trial results back to remote service.

        Args:
            trial_id: Trial identifier
            metrics: Computed metrics
            duration: Execution duration
            status: Trial status
            outputs_sample: Optional sample of outputs
            error_message: Error message if failed
            metadata: Additional metadata

        Raises:
            OptimizationError: If no session or submission fails
        """
        if not self.session_id:
            raise OptimizationError("No active session.")

        try:
            result = TrialResultSubmission(
                session_id=self.session_id,
                trial_id=trial_id,
                metrics=metrics,
                duration=duration,
                status=status,
                outputs_sample=outputs_sample,
                error_message=error_message,
                metadata=metadata or {},
            )

            await self.remote_service.submit_result(result)

            # Update local tracking
            self._completed_trials.append(result)

            # Update session if successful
            if self.session and status == TrialStatus.COMPLETED:
                self.session.completed_trials += 1

                # Update best if improved
                if self._is_better_result(metrics):
                    # Get config from suggestion
                    suggestion = self._pending_trials.get(trial_id)
                    if suggestion:
                        self.session.best_config = suggestion.config
                    self.session.best_metrics = metrics

            # Clean up pending trial
            self._pending_trials.pop(trial_id, None)

            logger.info(
                f"Reported results for {trial_id}: status={status}, metrics={metrics}"
            )

        except Exception as e:
            logger.error(f"Failed to report results: {e}")
            raise OptimizationError(f"Failed to report results: {e}") from None

    async def get_optimization_status(self) -> dict[str, Any]:
        """Get current optimization status.

        Returns:
            Dictionary with status information
        """
        if not self.session:
            return {
                "status": "not_initialized",
                "session_id": None,
                "completed_trials": 0,
                "best_metrics": None,
            }

        elapsed = time.time() - self._start_time if self._start_time else 0

        return {
            "status": self.session.status.value,
            "session_id": self.session_id,
            "completed_trials": self.session.completed_trials,
            "max_trials": self.session.max_trials,
            "progress": self.session.completed_trials / self.session.max_trials,
            "best_config": self.session.best_config,
            "best_metrics": self.session.best_metrics,
            "elapsed_time": elapsed,
            "trials_per_minute": (
                self.session.completed_trials / (elapsed / 60) if elapsed > 0 else 0
            ),
        }

    async def finalize_optimization(
        self, include_full_history: bool = False
    ) -> OptimizationFinalizationResponse:
        """Finalize the optimization session.

        Args:
            include_full_history: Whether to include full trial history

        Returns:
            Finalization response with results

        Raises:
            OptimizationError: If no session or finalization fails
        """
        if not self.session_id:
            raise OptimizationError("No active session to finalize.")

        try:
            request = OptimizationFinalizationRequest(
                session_id=self.session_id,
                include_full_history=include_full_history,
                metadata={
                    "client_completed_trials": len(self._completed_trials),
                    "total_duration": (
                        time.time() - self._start_time if self._start_time else 0
                    ),
                },
            )

            response = await self.remote_service.finalize_session(request)

            # Update local session
            if self.session:
                self.session.status = OptimizationSessionStatus.COMPLETED
                self.session.best_config = response.best_config
                self.session.best_metrics = response.best_metrics

            logger.info(
                f"Finalized session {self.session_id}: "
                f"{response.successful_trials}/{response.total_trials} successful, "
                f"cost savings: {response.cost_savings * 100:.1f}%"
            )

            return response

        except Exception as e:
            logger.error(f"Failed to finalize session: {e}")
            raise OptimizationError(f"Failed to finalize: {e}") from None

    # BaseOptimizer interface methods (for compatibility)

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Synchronous interface for backward compatibility."""
        raise NotImplementedError(
            "InteractiveOptimizer requires async usage. "
            "Use get_next_suggestion() instead."
        )

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Check if optimization should stop."""
        if not self.session:
            return True

        return not self.session.can_continue()

    def _is_better_result(self, metrics: dict[str, float]) -> bool:
        """Check if metrics are better than current best."""
        if not self.session or not self.session.best_metrics:
            return True

        # Simple comparison on primary objective
        if self.objectives and self.objectives[0] in metrics:
            primary = self.objectives[0]
            current = metrics.get(primary, 0)
            best = self.session.best_metrics.get(primary, 0)

            # Assume higher is better for now
            return current > best

        return False
