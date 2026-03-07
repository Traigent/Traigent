"""Backend session lifecycle management."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Security FUNC-CLOUD-HYBRID FUNC-ORCH-LIFECYCLE REQ-CLOUD-009 REQ-ORCH-003 SYNC-CloudHybrid

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from traigent.api.types import (
    AgentConfiguration,
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.config.types import TraigentConfig

if TYPE_CHECKING:
    from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.core.metadata_helpers import build_backend_metadata
from traigent.core.objectives import ObjectiveSchema
from traigent.core.session_context import SessionContext
from traigent.evaluators.base import Dataset
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.env_config import is_backend_offline
from traigent.utils.function_identity import FunctionDescriptor
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Track warnings that have already been shown to reduce log noise
_warned_missing_mapping: set[str] = set()
_warned_no_api_key: bool = False


class BackendSessionManager:
    """Manages backend session lifecycle and trial synchronization.

    Owns all backend-related operations including:
    - Session creation and finalization
    - Trial submission with metadata
    - Weighted score updates for multi-objective optimization
    - Session aggregation and summary generation
    """

    def __init__(
        self,
        backend_client: BackendIntegratedClient | None,
        traigent_config: TraigentConfig,
        objectives: list[str],
        objective_schema: ObjectiveSchema | None,
        optimizer: BaseOptimizer,
        optimization_id: str,
        optimization_status: OptimizationStatus,
    ) -> None:
        """Initialize backend session manager.

        Args:
            backend_client: Backend API client (None disables backend sync)
            traigent_config: Global configuration
            objectives: List of optimization objectives
            objective_schema: Multi-objective schema with weights
            optimizer: Optimizer instance (for objectives and config_space)
            optimization_id: Unique optimization run identifier
            optimization_status: Current optimization status
        """
        self._backend_client: BackendIntegratedClient | None = backend_client
        self._traigent_config = traigent_config
        self._objectives = objectives
        self._objective_schema = objective_schema
        self._optimizer = optimizer
        self._optimization_id = optimization_id
        self._optimization_status = optimization_status

    @staticmethod
    def create_backend_client(
        traigent_config: TraigentConfig,
    ) -> BackendIntegratedClient | None:
        """Initialize backend client if cloud features are available.

        Returns None when cloud plugin is not installed (graceful degradation).
        Raises FeatureNotAvailableError only when cloud mode is explicitly
        requested but plugin is missing.

        Args:
            traigent_config: Global configuration for execution mode and storage

        Returns:
            BackendIntegratedClient if available, None otherwise
        """
        # Try to import cloud module - may not be available if cloud plugin not installed
        try:
            from traigent.cloud.backend_client import (
                BackendClientConfig,
                BackendIntegratedClient,
            )
            from traigent.config.backend_config import BackendConfig
        except ModuleNotFoundError as err:
            # Cloud module not installed - check if this was the cloud module itself
            missing_module = getattr(err, "name", "") or ""
            if missing_module.startswith("traigent.cloud"):
                if traigent_config.execution_mode == "cloud":
                    # User explicitly requested cloud mode but plugin not installed
                    from traigent.utils.exceptions import FeatureNotAvailableError

                    raise FeatureNotAvailableError(
                        "Cloud execution mode",
                        plugin_name="traigent-cloud",
                        install_hint="pip install traigent[cloud]",
                    ) from err
                # For edge_analytics or other modes, gracefully degrade to local-only
                logger.info(
                    f"Cloud module not available for {traigent_config.execution_mode} mode. "
                    "Continuing with local storage only."
                )
                return None
            # Re-raise if it's a different missing module (broken install)
            raise

        backend_url = BackendConfig.get_backend_url()
        api_key = BackendConfig.get_api_key()

        if traigent_config.is_edge_analytics_mode() or BackendConfig.is_local_backend():
            logger.info(
                f"Configuring for {traigent_config.execution_mode} mode "
                f"with backend at {backend_url} (fallback enabled)"
            )
        else:
            logger.info(
                f"Configuring for {traigent_config.execution_mode} mode "
                f"with backend at {backend_url}"
            )

        backend_config = BackendClientConfig(
            backend_base_url=backend_url,
            enable_session_sync=True,
        )
        local_storage_path = traigent_config.get_local_storage_path()

        try:
            client = BackendIntegratedClient(
                api_key=api_key,
                backend_config=backend_config,
                enable_fallback=True,
                local_storage_path=local_storage_path,
            )
            logger.info(
                f"Backend client initialized for {traigent_config.execution_mode} mode - "
                f"session endpoints at {backend_config.backend_base_url}"
            )
            return client
        except Exception as exc:
            logger.warning(
                "Backend initialization warning. Continuing with local storage only. "
                "Results will not appear in backend UI.",
                exc_info=exc,
            )
            return BackendIntegratedClient(
                api_key=None,
                backend_config=backend_config,
                enable_fallback=True,
                local_storage_path=local_storage_path,
            )

    def _should_suppress_backend_warnings(self) -> bool:
        """Check if backend-related warnings should be suppressed.

        Returns True if:
        - Offline mode is enabled, OR
        - No API key is configured (user hasn't set up backend access)

        This prevents noisy warnings for users evaluating the SDK without
        backend access or running in local-only mode.
        """
        if is_backend_offline():
            return True

        # Check if backend client has API key configured
        if self._backend_client:
            auth_manager = getattr(self._backend_client, "auth", None)
            has_api_key = bool(
                auth_manager
                and hasattr(auth_manager, "has_api_key")
                and auth_manager.has_api_key()
            )
            if not has_api_key:
                return True

        return False

    def create_session(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        function_descriptor: FunctionDescriptor,
        max_trials: int | None,
        start_time: float,
        max_total_examples: int | None = None,
        agent_configuration: AgentConfiguration | None = None,
    ) -> SessionContext:
        """Create backend session and return context.

        Args:
            func: Function being optimized
            dataset: Evaluation dataset
            function_descriptor: Descriptor for the function being optimized
            max_trials: Maximum number of trials
            start_time: Optimization start timestamp
            max_total_examples: Maximum total examples across all trials
            agent_configuration: Multi-agent configuration for parameter grouping

        Returns:
            SessionContext with session_id (or None if backend disabled)
        """
        session_id = None
        function_identifier = function_descriptor.identifier
        function_display_name = function_descriptor.display_name
        function_slug = function_descriptor.slug

        if self._backend_client:
            evaluation_set_name = (
                dataset.name if hasattr(dataset, "name") else "default_evaluation"
            )

            max_trials_value = max_trials if max_trials is not None else 10
            max_samples_value = (
                max_total_examples if max_total_examples is not None else None
            )
            logger.info(
                "Creating backend session with max_trials=%s for %s (remote_name=%s)",
                max_trials_value,
                function_identifier,
                function_slug,
            )

            # Build metadata including agent configuration if present
            session_metadata: dict[str, Any] = {
                "optimization_id": self._optimization_id,
                "max_trials": max_trials_value,
                "max_total_examples": max_samples_value,
                "dataset_size": len(dataset),
                "function_name": function_identifier,
                "function_display_name": function_display_name,
                "function_module": function_descriptor.module,
                "function_relative_path": function_descriptor.relative_path,
                "function_slug": function_slug,
                "evaluation_set": evaluation_set_name,
            }
            if agent_configuration is not None:
                session_metadata["agent_configuration"] = agent_configuration.to_dict()

            session_id = self._backend_client.create_session(
                function_name=function_slug,
                search_space=getattr(self._optimizer, "config_space", {}),
                optimization_goal="maximize",
                metadata=session_metadata,
            )
            logger.info("Created backend session: %s", session_id)

            # Verify that the backend recorded a mapping; absence indicates we fell
            # back to local storage even though a session identifier was returned.
            session_mapping = None
            get_mapping = getattr(self._backend_client, "get_session_mapping", None)
            if callable(get_mapping):
                try:
                    session_mapping = get_mapping(session_id)
                except Exception as exc:  # pragma: no cover - defensive logging only
                    logger.debug(
                        "Unable to resolve backend session mapping for %s: %s",
                        session_id,
                        exc,
                    )

            if session_mapping is None:
                # Only warn once per session; suppress in offline mode (expected)
                if (
                    session_id not in _warned_missing_mapping
                    and not is_backend_offline()
                ):
                    logger.debug(
                        "Backend mapping missing for session %s (%s). "
                        "This strongly suggests the Traigent API rejected the session "
                        "creation request and the SDK fell back to local tracking. "
                        "Trials will remain local unless the session endpoint is reachable.",
                        session_id,
                        function_identifier,
                    )
                    _warned_missing_mapping.add(session_id)
                elif is_backend_offline():
                    logger.debug(
                        "Backend mapping missing for session %s (offline mode - expected)",
                        session_id,
                    )

        dataset_name = (
            dataset.name if hasattr(dataset, "name") else "default_evaluation"
        )

        return SessionContext(
            session_id=session_id,
            dataset_name=dataset_name,
            function_name=function_identifier,
            optimization_id=self._optimization_id,
            start_time=start_time,
        )

    def update_backend_client(
        self, backend_client: BackendIntegratedClient | None
    ) -> None:
        """Swap the backend client while preserving existing session state."""

        self._backend_client = backend_client

    async def submit_trial(
        self,
        trial_result: TrialResult,
        session_id: str | None,
        dataset_name: str = "dataset",
        content_scores: dict[str, dict[int, float]] | None = None,
    ) -> bool:
        """Submit trial to backend.

        Args:
            trial_result: Completed trial result
            session_id: Backend session identifier
            dataset_name: Name of the dataset (for stable example ID generation)
            content_scores: Optional dict with keys "uniqueness", "novelty" mapping
                           example_index -> score (0.0-1.0)

        Returns:
            True if submission succeeded
        """
        if not self._backend_client or not session_id:
            return False

        primary_objective = (
            self._optimizer.objectives[0] if self._optimizer.objectives else "score"
        )

        score = trial_result.get_metric(primary_objective)
        trial_metadata = build_backend_metadata(
            trial_result,
            primary_objective,
            self._traigent_config,
            dataset_name,
            content_scores,
        )

        await self._log_trial_to_backend(
            session_id=session_id,
            trial_result=trial_result,
            score=score,
            metadata=trial_metadata,
        )

        return True

    async def _log_trial_to_backend(
        self,
        session_id: str,
        trial_result: TrialResult,
        score: float | None,
        metadata: dict[str, Any],
    ) -> None:
        """Persist trial outcome locally and submit metrics to backend when possible."""

        if not self._backend_client or not session_id:
            return

        sanitized_score = float(score) if score is not None else None
        metadata_payload = dict(metadata)
        if score is None:
            metadata_payload["primary_objective_missing"] = True

        # Always record locally so analytics remain available even without backend.
        try:
            self._backend_client.submit_result(
                session_id=session_id,
                config=trial_result.config,
                score=sanitized_score,
                metadata=metadata_payload,
            )
        except Exception as exc:
            logger.debug(
                "Local trial logging failed for session %s trial %s: %s",
                session_id,
                trial_result.trial_id,
                exc,
            )

        # Skip remote submission when no API key is configured (offline/local only).
        auth_manager = getattr(self._backend_client, "auth_manager", None)
        has_api_key = bool(
            auth_manager
            and hasattr(auth_manager, "has_api_key")
            and auth_manager.has_api_key()
        )
        global _warned_no_api_key
        if not has_api_key:
            # Only warn once; suppress in offline mode to reduce log noise
            if not _warned_no_api_key and not is_backend_offline():
                logger.debug(
                    "Skipping backend trial submissions (no API key detected). "
                    "Results will be saved locally only."
                )
                _warned_no_api_key = True
            else:
                logger.debug(
                    "Skipping backend trial submission for session %s trial %s (no API key)",
                    session_id,
                    trial_result.trial_id,
                )
            return
        else:
            logger.info(
                "✅ API key detected, submitting trial %s for session %s",
                trial_result.trial_id,
                session_id,
            )

        # If the backend session was never registered (e.g., API fallback), avoid
        # posting to the SaaS endpoint to prevent 400 errors for unknown sessions.
        session_mapping = None
        get_mapping = getattr(self._backend_client, "get_session_mapping", None)
        if callable(get_mapping):
            try:
                session_mapping = get_mapping(session_id)
            except Exception as exc:  # pragma: no cover - defensive logging only
                logger.debug(
                    "Failed to retrieve backend session mapping for %s: %s",
                    session_id,
                    exc,
                )

        if session_mapping is None:
            logger.info(
                "No backend session mapping for %s; assuming offline fallback. "
                "Skipping remote submission for trial %s.",
                session_id,
                trial_result.trial_id,
            )
            return

        # Build payload for backend session endpoint.
        metrics_payload: dict[str, Any] = dict(trial_result.metrics or {})

        # Surface summary stats / measures captured in metadata so the backend can persist them.
        if "summary_stats" in metadata and "summary_stats" not in metrics_payload:
            metrics_payload["summary_stats"] = metadata["summary_stats"]
            logger.debug(
                "Added summary_stats to metrics payload for trial %s",
                trial_result.trial_id,
            )
        if "measures" in metadata and "measures" not in metrics_payload:
            metrics_payload["measures"] = metadata["measures"]
            logger.info(
                "📊 Added %d measures to metrics payload for trial %s (status=%s)",
                len(metadata["measures"]),
                trial_result.trial_id,
                trial_result.status.value,
            )
        elif trial_result.status == TrialStatus.PRUNED:
            logger.warning(
                "⚠️ No measures found for PRUNED trial %s - metadata keys: %s",
                trial_result.trial_id,
                list(metadata.keys()),
            )

        if "score" not in metrics_payload and score is not None:
            metrics_payload["score"] = sanitized_score

        # Map SDK TrialStatus to backend status string.
        # PRUNED is a success case (early stopping for efficiency), not a failure.
        status_mapping = {
            TrialStatus.COMPLETED: "COMPLETED",
            TrialStatus.FAILED: "FAILED",
            TrialStatus.PRUNED: "PRUNED",
            TrialStatus.CANCELLED: "CANCELLED",
            TrialStatus.RUNNING: "RUNNING",
            TrialStatus.PENDING: "PENDING",
            TrialStatus.NOT_STARTED: "PENDING",
        }
        status = status_mapping.get(trial_result.status, "FAILED")

        try:
            try:
                await self._backend_client.register_trial_start(
                    session_id=session_id,
                    trial_id=trial_result.trial_id,
                    config=trial_result.config,
                )
            except Exception as exc:
                logger.debug(
                    "Trial start registration failed for session %s trial %s: %s",
                    session_id,
                    trial_result.trial_id,
                    exc,
                )

            submitted = await self._backend_client._submit_trial_result_via_session(
                session_id=session_id,
                trial_id=trial_result.trial_id,
                config=trial_result.config,
                metrics=metrics_payload,
                status=status,
                error_message=trial_result.error_message,
                execution_mode=self._traigent_config.execution_mode,
            )
            if not submitted:
                logger.warning(
                    "Backend session endpoint rejected trial %s for session %s",
                    trial_result.trial_id,
                    session_id,
                )
            else:
                logger.info(
                    "Submitted trial %s for session %s (status=%s, metrics_keys=%s)",
                    trial_result.trial_id,
                    session_id,
                    status,
                    sorted(metrics_payload.keys()),
                )
        except Exception as exc:
            logger.warning(
                "Failed to submit trial %s for session %s to backend: %s",
                trial_result.trial_id,
                session_id,
                exc,
            )

    async def update_weighted_scores(
        self,
        result: OptimizationResult,
        session_id: str | None,
    ) -> int:
        """Update backend with weighted scores for multi-objective runs.

        Args:
            result: Optimization result with all trials
            session_id: Backend session identifier

        Returns:
            Number of trials successfully updated
        """
        if not self._backend_client or session_id is None or len(self._objectives) <= 1:
            return 0

        backend_client = self._backend_client  # Capture for use in nested function

        try:
            logger.info("Calculating weighted scores for multi-objective optimization")

            if self._objective_schema:
                objective_weights = dict(self._objective_schema.weights_normalized)
            else:
                objective_weights = {
                    obj: 1.0 / len(self._objectives) for obj in self._objectives
                }

            weighted_results = result.calculate_weighted_scores(
                objective_weights=objective_weights
            )

            weighted_scores_data = list(weighted_results.get("weighted_scores", []))
            successful_trials_iter = iter(
                getattr(result, "successful_trials", result.trials)
            )

            weighted_updates: list[tuple[TrialResult, float]] = []

            for weighted_entry in weighted_scores_data:
                weighted_trial = None
                weighted_score = None

                if isinstance(weighted_entry, tuple):
                    candidate_trial = weighted_entry[0]
                    if hasattr(candidate_trial, "trial_id"):
                        weighted_trial = candidate_trial
                        if len(weighted_entry) > 1:
                            weighted_score = weighted_entry[1]
                    if weighted_score is None and len(weighted_entry) > 0:
                        weighted_score = weighted_entry[-1]
                else:
                    weighted_score = weighted_entry

                if weighted_trial is None:
                    try:
                        weighted_trial = next(successful_trials_iter)
                    except StopIteration:
                        logger.warning(
                            "No matching successful trial available for weighted score entry %s; skipping submission",
                            weighted_entry,
                        )
                        continue

                if weighted_score is None:
                    logger.debug(
                        "Weighted score entry %s lacked explicit score; defaulting to 0.0",
                        weighted_entry,
                    )
                    weighted_score = 0.0

                try:
                    weighted_score_value = float(weighted_score)
                except (TypeError, ValueError):
                    logger.warning(
                        "Unable to coerce weighted score %s for trial %s; skipping submission",
                        weighted_score,
                        getattr(weighted_trial, "trial_id", "unknown"),
                    )
                    continue

                weighted_updates.append((weighted_trial, weighted_score_value))

            attempted_updates = len(weighted_updates)

            if attempted_updates == 0:
                logger.info("No weighted score updates to submit")
                if hasattr(result, "metadata"):
                    if not result.metadata:
                        result.metadata = {}
                    result.metadata["weighted_results"] = {
                        "best_weighted_config": weighted_results.get(
                            "best_weighted_config"
                        ),
                        "best_weighted_score": weighted_results.get(
                            "best_weighted_score"
                        ),
                        "objective_weights": objective_weights,
                        "trials_updated": 0,
                    }
                return 0

            concurrency_limit = getattr(
                backend_client, "weighted_update_concurrency", 8
            )
            try:
                concurrency_limit_int = int(concurrency_limit)
            except (TypeError, ValueError):
                concurrency_limit_int = 8
            if concurrency_limit_int <= 0:
                concurrency_limit_int = 1

            semaphore = asyncio.Semaphore(concurrency_limit_int)
            normalization_info = weighted_results.get("normalization_ranges")

            async def submit_weighted_score(
                trial: TrialResult, score_value: float
            ) -> bool:
                async with semaphore:
                    trial_identifier = getattr(trial, "trial_id", "unknown")
                    try:
                        success = await backend_client.update_trial_weighted_scores(
                            trial_id=trial_identifier,
                            weighted_score=score_value,
                            normalization_info=normalization_info,
                            objective_weights=objective_weights,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        if self._should_suppress_backend_warnings():
                            logger.debug(
                                "Exception while updating weighted score for trial %s: %s (backend unavailable)",
                                trial_identifier,
                                exc,
                            )
                        else:
                            logger.warning(
                                "Exception while updating weighted score for trial %s: %s",
                                trial_identifier,
                                exc,
                            )
                        return False

                    if success:
                        logger.debug(
                            "Updated trial %s with weighted score %.4f",
                            trial_identifier,
                            score_value,
                        )
                        return True

                    # Suppress warnings in mock mode or when no API key configured
                    # This prevents noisy output for users evaluating SDK without backend
                    if self._should_suppress_backend_warnings():
                        logger.debug(
                            "Failed to update weighted score for trial %s (backend unavailable or mock mode)",
                            trial_identifier,
                        )
                    else:
                        logger.warning(
                            "Failed to update weighted score for trial %s",
                            trial_identifier,
                        )
                    return False

            update_results = await asyncio.gather(
                *(
                    submit_weighted_score(trial, score)
                    for trial, score in weighted_updates
                )
            )
            update_count = sum(1 for result_flag in update_results if result_flag)

            logger.info(
                "Updated %s/%s trials with weighted scores (concurrency=%s)",
                update_count,
                attempted_updates,
                concurrency_limit_int,
            )

            if hasattr(result, "metadata"):
                if not result.metadata:
                    result.metadata = {}
                result.metadata["weighted_results"] = {
                    "best_weighted_config": weighted_results.get(
                        "best_weighted_config"
                    ),
                    "best_weighted_score": weighted_results.get("best_weighted_score"),
                    "objective_weights": objective_weights,
                    "trials_updated": update_count,
                }

            return update_count

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error updating weighted scores: %s", exc)
            return 0

    def submit_session_aggregation(
        self, result: OptimizationResult, session_id: str | None
    ) -> None:
        """Submit aggregated summary for non-edge modes.

        Args:
            result: Final optimization result
            session_id: Backend session identifier
        """
        if (
            not self._backend_client
            or session_id is None
            or self._traigent_config.is_edge_analytics_mode()
        ):
            return

        session_summary = (
            result.metadata.get("session_summary")
            if hasattr(result, "metadata") and result.metadata
            else None
        )

        if not session_summary:
            return

        aggregated_trial_id = f"{session_id}_AGG_SUMMARY"

        samples_per_config = session_summary.get("samples_per_config", {})
        total_examples = sum(samples_per_config.values()) if samples_per_config else 0

        summary_stats_with_aggregation = {
            "metrics": session_summary.get("metrics", {}),
            "execution_time": result.duration,
            "total_examples": total_examples,
            "metadata": {
                "aggregation_level": "session",
                "aggregation_summary": session_summary,
                "trial_id": aggregated_trial_id,
                "sdk_version": "2.0.0",
            },
        }

        # Include statistical significance badges if computed
        stat_sig = (
            result.metadata.get("statistical_significance") if result.metadata else None
        )
        if stat_sig:
            summary_stats_with_aggregation["metadata"][
                "statistical_significance"
            ] = stat_sig

        try:
            successful_trials = len([t for t in result.trials if t.is_successful])
            overlay_metrics: dict[str, float] = {
                "run_trials_completed": len(result.trials),
                "run_successful_trials": successful_trials,
                "run_success_rate": result.success_rate,
            }
            if isinstance(result.metrics, dict):
                for key, value in result.metrics.items():
                    if isinstance(value, (int, float)):
                        overlay_metrics[f"run_{key}"] = value
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Failed building overlay run metrics: %s", exc)
            overlay_metrics = {}

        self._backend_client.submit_result(
            session_id=session_id,
            config=result.best_config,
            score=result.best_score,
            metadata={
                "summary_stats": summary_stats_with_aggregation,
                "trial_id": aggregated_trial_id,
                **overlay_metrics,
            },
        )
        logger.debug(
            "Submitted session aggregation with overlay metrics: %s",
            list(overlay_metrics.keys()),
        )

    def finalize_session(
        self,
        session_id: str | None,
        optimization_status: OptimizationStatus,
    ) -> dict[str, Any] | None:
        """Finalize backend session and return summary.

        Args:
            session_id: Backend session identifier
            optimization_status: Final optimization status

        Returns:
            Session summary metadata (or None if backend disabled)
        """
        if not self._backend_client or session_id is None:
            return None

        final_status = (
            "completed"
            if optimization_status == OptimizationStatus.COMPLETED
            else "failed"
        )

        if hasattr(self._backend_client, "finalize_session_sync"):
            result: dict[str, Any] | None = self._backend_client.finalize_session_sync(  # type: ignore[assignment]
                session_id, final_status == "completed"
            )
            return result

        result = self._backend_client.finalize_session(  # type: ignore[assignment]
            session_id, final_status == "completed"
        )
        return result

    def attach_session_metadata(
        self,
        result: OptimizationResult,
        session_id: str | None,
        session_summary: dict[str, Any] | None,
    ) -> None:
        """Attach session identifiers and summary to result metadata.

        Args:
            result: Optimization result to update
            session_id: Backend session identifier
            session_summary: Session summary from backend
        """
        if session_id is None or not hasattr(result, "metadata"):
            return

        if not result.metadata:
            result.metadata = {}
        update_payload: dict[str, Any] = {"local_session_id": session_id}
        if session_summary is not None:
            update_payload["local_session_summary"] = session_summary

        # Add experiment_id from session mapping if available
        if self._backend_client is not None:
            try:
                mapping = self._backend_client.get_session_mapping(session_id)
                if mapping is not None:
                    update_payload["experiment_id"] = mapping.experiment_id
                    update_payload["experiment_run_id"] = mapping.experiment_run_id
            except Exception:
                pass  # Silently ignore if mapping not available

        result.metadata.update(update_payload)
