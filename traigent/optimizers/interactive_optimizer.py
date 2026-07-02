"""Interactive optimizer for client-side execution with remote guidance.

This optimizer separates the suggestion generation from execution, allowing
the client to execute functions locally while receiving configuration and
dataset subset suggestions from a remote service.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Reliability CONC-Quality-Performance FUNC-OPT-ALGORITHMS FUNC-CLOUD-HYBRID REQ-OPT-ALG-004 REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

from traigent.api.types import TrialResult
from traigent.api.types import TrialStatus as SDKTrialStatus
from traigent.config.types import TraigentConfig
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import OptimizationError
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.cloud.models import TrialStatus

    TrialStatusLike: TypeAlias = SDKTrialStatus | TrialStatus
else:
    TrialStatusLike: TypeAlias = Any

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
except (
    ModuleNotFoundError
) as err:  # pragma: no cover - only runs when cloud not installed
    # Check .name to distinguish missing cloud vs broken transitive dependency
    missing_module = getattr(err, "name", "") or ""
    if missing_module == "traigent.cloud" or missing_module.startswith(
        "traigent.cloud."
    ):
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

_DEFAULT_OPTIMIZER_READY_TIMEOUT_SECONDS = 60.0
_OPTIMIZER_READY_TIMEOUT_ENV = "TRAIGENT_OPTIMIZER_READY_TIMEOUT_SECONDS"
_LEGACY_OPTIMIZER_READY_TIMEOUT_ENV = "TRAIGENT_CLOUD_OPTIMIZER_READY_TIMEOUT"


def _require_cloud_models() -> None:
    """Raise FeatureNotAvailableError if cloud models are not available."""
    if not _CLOUD_MODELS_AVAILABLE:
        from traigent.utils.exceptions import FeatureNotAvailableError

        raise FeatureNotAvailableError(
            "Interactive optimization with remote guidance",
            plugin_name="traigent-cloud",
            install_hint="pip install traigent[cloud]",
        )


def _coerce_trial_status(status: TrialStatusLike) -> Any:
    """Convert SDK trial statuses to cloud trial statuses when needed."""
    if _CLOUD_MODELS_AVAILABLE and isinstance(status, SDKTrialStatus):
        return TrialStatus(status.value)
    return status


def _is_completed_status(status: TrialStatusLike) -> bool:
    """Check completion across SDK and cloud status enums."""
    return bool(getattr(status, "value", status) == "completed")


def _string_sequence(value: Any) -> tuple[str, ...]:
    """Return string tuple metadata only from sequence-like values."""
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item) for item in value if item)


def _resolve_optimizer_ready_timeout(value: Any = None) -> float:
    """Resolve the cloud optimizer readiness timeout in seconds."""

    raw = value
    if raw is None:
        raw = os.getenv(_OPTIMIZER_READY_TIMEOUT_ENV)
    if raw is None:
        raw = os.getenv(_LEGACY_OPTIMIZER_READY_TIMEOUT_ENV)
    if raw is None:
        return _DEFAULT_OPTIMIZER_READY_TIMEOUT_SECONDS

    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "optimizer_ready_timeout must be a positive number of seconds"
        ) from exc

    if timeout <= 0:
        raise ValueError("optimizer_ready_timeout must be a positive number of seconds")
    return timeout


class CloudBrainOptimizationComplete(RuntimeError):
    """Normal cloud-brain terminal signal from the next-trial API."""

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason or "cloud brain completed optimization"
        super().__init__(self.reason)

    @property
    def stop_reason(self) -> str:
        normalized = self.reason.lower().replace("-", "_").replace(" ", "_")
        if "max" in normalized and "trial" in normalized:
            return "max_trials_reached"
        if "conver" in normalized:
            return "convergence"
        return "optimizer"


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
        artifact_fingerprints: dict[str, str | None] | None = None,
        fingerprint_meta: dict[str, Any] | None = None,
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
        optimizer_ready_timeout = kwargs.pop("optimizer_ready_timeout", None)
        if optimizer_ready_timeout is None:
            optimizer_ready_timeout = kwargs.pop("cloud_optimizer_ready_timeout", None)
        super().__init__(config_space, objectives, context, **kwargs)

        self.remote_service = remote_service
        self.dataset_metadata = dataset_metadata or {}
        self.optimization_strategy = optimization_strategy
        self.artifact_fingerprints = artifact_fingerprints
        self.fingerprint_meta = fingerprint_meta
        self.optimizer_ready_timeout = _resolve_optimizer_ready_timeout(
            optimizer_ready_timeout
        )

        # Session management
        self.session: OptimizationSession | None = None
        self.session_id: str | None = None

        # Tracking
        self._pending_trials: dict[str, TrialSuggestion] = {}
        self._completed_trials: list[TrialResultSubmission] = []
        self._start_time: float | None = None
        self._completion_reason: str | None = None

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
                artifact_fingerprints=self.artifact_fingerprints,
                fingerprint_meta=self.fingerprint_meta,
            )

            response = await self._await_optimizer_service(
                "session-create",
                self.remote_service.create_session(request),
            )

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

        request = NextTrialRequest(
            session_id=self.session_id,
            previous_results=previous_results or self._completed_trials[-5:],
            request_metadata={
                "dataset_size": dataset_size,
                "completed_trials": len(self._completed_trials),
            },
        )

        try:
            response = await self._await_optimizer_service(
                "next-trial readiness",
                self.remote_service.get_next_trial(request),
            )
        except Exception as e:
            logger.error(f"Failed to get next suggestion: {e}")
            raise OptimizationError(f"Failed to get suggestion: {e}") from e

        # Update local session status
        if self.session:
            self.session.status = response.session_status

        if not response.should_continue:
            self._completion_reason = (
                response.stop_reason
                or response.reason
                or "cloud brain completed optimization"
            )
            logger.info(f"Optimization complete: {self._completion_reason}")
            return None

        if not response.suggestion:
            raise OptimizationError(
                "Cloud brain returned no suggestion while should_continue=True"
            )

        self._completion_reason = None
        suggestion = response.suggestion

        # Store pending trial
        self._pending_trials[suggestion.trial_id] = suggestion

        logger.info(
            f"Got suggestion {suggestion.trial_id}: "
            f"{suggestion.exploration_type} with "
            f"{len(suggestion.dataset_subset.indices)} examples"
        )

        return suggestion

    async def suggest_next_trial_async(
        self,
        history: list[TrialResult],
        remote_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async BaseOptimizer interface backed by the cloud next-trial API."""

        context = remote_context or {}
        dataset_size = int(
            context.get("dataset_size") or self.dataset_metadata.get("size") or 0
        )
        suggestion = await self.get_next_suggestion(
            dataset_size=max(dataset_size, 1),
        )
        if suggestion is None:
            raise CloudBrainOptimizationComplete(self._completion_reason)

        config = dict(suggestion.config or {})
        config["_traigent_backend_trial_id"] = suggestion.trial_id
        config["_traigent_cloud_trial_number"] = suggestion.trial_number
        subset = getattr(suggestion, "dataset_subset", None)
        subset_indices = getattr(subset, "indices", None)
        if subset_indices:
            config["__subset_indices__"] = list(subset_indices)
        return config

    async def _await_optimizer_service(self, stage: str, awaitable: Any) -> Any:
        """Await a cloud optimizer operation with a bounded readiness deadline."""

        try:
            return await asyncio.wait_for(
                awaitable,
                timeout=self.optimizer_ready_timeout,
            )
        except TimeoutError as exc:
            raise OptimizationError(
                f"Cloud optimizer service did not become available during {stage} "
                f"within {self.optimizer_ready_timeout:g}s. "
                "Bayesian and other managed/cloud optimization algorithms require "
                "the backend optimizer service and cannot fall back to the local SDK. "
                "Check that the optimizer service is deployed and healthy "
                "(BE#1831/#1146), then retry."
            ) from exc

    async def report_results(
        self,
        trial_id: str,
        metrics: dict[str, float],
        duration: float,
        status: TrialStatusLike = SDKTrialStatus.COMPLETED,
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
            result_metadata = self._metadata_with_tvar_observation(
                trial_id=trial_id,
                metrics=metrics,
                metadata=metadata,
            )
            result = TrialResultSubmission(
                session_id=self.session_id,
                trial_id=trial_id,
                metrics=metrics,
                duration=duration,
                status=_coerce_trial_status(status),
                outputs_sample=outputs_sample,
                error_message=error_message,
                metadata=result_metadata,
            )

            await self.remote_service.submit_result(result)

            # Update local tracking
            self._completed_trials.append(result)

            # Update session if successful
            if self.session and _is_completed_status(status):
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
            current = float(metrics.get(primary, 0.0))
            best = float(self.session.best_metrics.get(primary, 0.0))

            # Assume higher is better for now
            return current > best

        return False

    def _metadata_with_tvar_observation(
        self,
        *,
        trial_id: str,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Attach a content-free TVAR observation when suggestion context exists."""
        result_metadata = dict(metadata or {})
        suggestion = self._pending_trials.get(trial_id)
        if suggestion is None or not self.session_id:
            return result_metadata

        try:
            from traigent.tuned_variables.observation import (
                build_tvar_observation,
                merge_tvar_observation_metadata,
            )

            suggestion_metadata = suggestion.metadata or {}
            session_metadata = self.session.metadata if self.session else {}
            catalog_entry_ids = _string_sequence(
                suggestion_metadata.get("catalog_entry_ids")
                or result_metadata.get("catalog_entry_ids")
                or session_metadata.get("catalog_entry_ids")
            )
            primary_metric = self.objectives[0] if self.objectives else "score"
            observation = build_tvar_observation(
                session_id=self.session_id,
                trial_id=trial_id,
                config=suggestion.config,
                metrics=metrics,
                primary_metric=primary_metric,
                comparability={
                    "scope": "trial",
                    "n": len(suggestion.dataset_subset.indices),
                },
                catalog_entry_ids=catalog_entry_ids,
                agent_type=(
                    suggestion_metadata.get("agent_type")
                    or result_metadata.get("agent_type")
                    or session_metadata.get("agent_type")
                    or self.dataset_metadata.get("agent_type")
                ),
                config_space_id=(
                    suggestion_metadata.get("config_space_id")
                    or result_metadata.get("config_space_id")
                    or session_metadata.get("config_space_id")
                ),
                effectuation_events=result_metadata.get("effectuation_events"),
            )
            return cast(
                dict[str, Any],
                merge_tvar_observation_metadata(result_metadata, observation),
            )
        except Exception as exc:
            logger.debug(
                "Skipping TVAR observation metadata for trial %s: %s",
                trial_id,
                exc,
            )
            return result_metadata
