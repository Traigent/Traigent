"""Trial lifecycle management.

This module contains the TrialLifecycle class which coordinates individual trial
execution and post-processing. It was extracted from OptimizationOrchestrator
to improve testability and reduce class complexity.

Classes:
    TrialLifecycle: Coordinates trial execution, budget management, and result building.
"""

# Traceability: CONC-Layer-Core FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from traigent.config.context import TrialContext
from traigent.core.orchestrator_helpers import (
    enforce_constraints,
    extract_cost_from_results,
    extract_optuna_trial_id,
    prepare_evaluation_config,
)
from traigent.core.pruning_progress_tracker import PruningProgressTracker
from traigent.core.sample_budget import LeaseClosure, SampleBudgetLease
from traigent.core.trial_result_factory import (
    build_failed_result,
    build_pruned_result,
    build_success_result,
)
from traigent.core.types import TrialResult, TrialStatus
from traigent.evaluators.base import Dataset
from traigent.utils.error_handler import APIKeyError
from traigent.utils.exceptions import (
    OptimizationError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
    TrialPrunedError,
    TVLConstraintError,
    VendorPauseError,
)
from traigent.utils.logging import get_logger

from .tracing import record_trial_result, trial_span

if TYPE_CHECKING:
    from traigent.core.orchestrator import OptimizationOrchestrator

from traigent.core.cost_enforcement import Permit

logger = get_logger(__name__)


class TrialLifecycle:
    """Coordinates individual trial execution and post-processing.

    This class encapsulates the logic for:
    - Running individual trials (sequential and parallel)
    - Generating trial IDs
    - Setting up progress tracking and budget leases
    - Building trial results (success, failure, pruned)
    - Managing budget metadata

    The class maintains a reference to the parent OptimizationOrchestrator
    to access shared state like the optimizer, evaluator, and budget manager.
    """

    def __init__(self, orchestrator: OptimizationOrchestrator) -> None:
        """Initialize TrialLifecycle with parent orchestrator reference.

        Args:
            orchestrator: Parent OptimizationOrchestrator instance
        """
        self._orchestrator = orchestrator

    # =========================================================================
    # Sequential Trial Execution
    # =========================================================================

    async def run_sequential_trial(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        session_id: str | None,
        function_name: str | None,
        trial_count: int,
    ) -> tuple[int, str]:
        """Run a single sequential trial in the optimization loop.

        Args:
            func: Function to evaluate
            dataset: Evaluation dataset
            session_id: Session ID for tracking
            function_name: Name of the function being optimized
            trial_count: Current trial count

        Returns:
            Tuple of (updated_trial_count, action) where action is "continue" or "break"
        """
        orchestrator = self._orchestrator

        if (
            orchestrator.max_trials is not None
            and trial_count >= orchestrator.max_trials
        ):
            logger.info(f"Trial limit reached: {orchestrator.max_trials}")
            orchestrator._stop_reason = "max_trials_reached"
            return trial_count, "break"

        config = orchestrator._consume_default_config()
        if config is None:
            try:
                config = orchestrator.optimizer.suggest_next_trial(orchestrator._trials)
            except OptimizationError:
                raise
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.exception("Optimizer failed to suggest the next trial: %s", e)
                return trial_count, "break"

        optuna_trial_id = (
            config.get("_optuna_trial_id") if isinstance(config, dict) else None
        )

        cache_policy = orchestrator.config.get("cache_policy", "allow_repeats")
        if cache_policy != "allow_repeats":
            dataset_name = getattr(dataset, "name", "unnamed_dataset")
            filtered_configs = orchestrator.cache_policy_handler.apply_policy(  # type: ignore[has-type]
                [config],
                cache_policy,
                function_name or "unknown_function",
                dataset_name,
            )
            if not filtered_configs:
                logger.info("Configuration already evaluated, skipping")
                if hasattr(orchestrator, "_abandon_optuna_trial"):
                    orchestrator._abandon_optuna_trial(  # type: ignore[attr-defined]
                        optuna_trial_id,
                        reason=f"trial_filtered_by_cache_policy:{cache_policy}",
                        config=config,
                        status=TrialStatus.PRUNED,
                        pruned_step=0,
                    )
                return trial_count, "continue"
            config = filtered_configs[0]
            optuna_trial_id = (
                config.get("_optuna_trial_id") if isinstance(config, dict) else None
            )

        config_for_run = prepare_evaluation_config(config)  # type: ignore[arg-type]

        # Phase 2: Enforce pre-evaluation constraints
        try:
            enforce_constraints(
                config_for_run,
                None,
                orchestrator._constraints_pre_eval,
                stage="pre",
            )
        except TVLConstraintError as e:
            # Constraint failed - skip this config without consuming a trial slot
            # (Fix for issue #27: constraint-rejected configs should not reduce actual evaluations)
            logger.info(
                "Pre-constraint failed for config %s: %s (not counting toward max_trials)",
                config_for_run,
                e,
            )
            # Give back the trial slot the optimizer consumed in suggest_next_trial
            if hasattr(orchestrator.optimizer, "_trial_count"):
                orchestrator.optimizer._trial_count = max(
                    0, orchestrator.optimizer._trial_count - 1
                )
            if hasattr(orchestrator, "_abandon_optuna_trial"):
                orchestrator._abandon_optuna_trial(  # type: ignore[attr-defined]
                    optuna_trial_id,
                    reason=f"trial_rejected_by_constraint:{e}",
                    config=config_for_run,
                    status=TrialStatus.PRUNED,
                    pruned_step=0,
                )
            return trial_count, "continue"

        # Phase 2.1: Acquire cost permit before sequential trial execution
        # This ensures sequential trials respect cost limits proactively
        permit: Permit | None = None
        if orchestrator.cost_enforcer is not None:
            permit = await orchestrator.cost_enforcer.acquire_permit_async()
            if not permit.is_granted:
                # Interactive pause: let user raise the budget limit
                prompt_adapter = getattr(orchestrator, "_prompt_adapter", None)
                if prompt_adapter is not None:
                    import asyncio

                    decision = await asyncio.to_thread(
                        prompt_adapter.prompt_budget_pause,
                        orchestrator.cost_enforcer.get_status().accumulated_cost_usd,
                        orchestrator.cost_enforcer.config.limit,
                    )
                    if decision.startswith("raise:"):
                        new_limit = float(decision.split(":")[1])
                        orchestrator.cost_enforcer.update_limit(new_limit)
                        return trial_count, "continue"

                logger.info(
                    "Sequential trial cancelled due to cost limit reached (config=%s)",
                    config_for_run,
                )
                orchestrator._stop_reason = "cost_limit"
                if hasattr(orchestrator, "_abandon_optuna_trial"):
                    orchestrator._abandon_optuna_trial(  # type: ignore[attr-defined]
                        optuna_trial_id,
                        reason="trial_cancelled_by_cost_limit",
                        config=config_for_run,
                        status=TrialStatus.CANCELLED,
                        pruned_step=0,
                    )
                return trial_count, "break"

        orchestrator.callback_manager.on_trial_start(trial_count, config_for_run)

        try:
            trial_result = await self.run_trial(
                func,
                config_for_run,
                dataset,
                trial_count,
                session_id,
                optuna_trial_id=optuna_trial_id,
            )

            trial_count = await orchestrator._handle_trial_result(
                trial_result=trial_result,
                optimizer_config=config,
                current_trial_index=trial_count,
                session_id=session_id,
                optuna_trial_id=optuna_trial_id,
                log_on_success=True,
                permit=permit,
            )
        except Exception:
            # Release permit on exception to prevent stranding
            if (
                permit is not None
                and permit.active
                and orchestrator.cost_enforcer is not None
            ):
                await orchestrator.cost_enforcer.release_permit_async(permit)
                logger.debug(
                    "Released permit %d after exception in run_sequential_trial",
                    permit.id,
                )
            raise

        return trial_count, "continue"

    # =========================================================================
    # Core Trial Execution
    # =========================================================================

    async def run_trial(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        trial_number: int,
        session_id: str | None = None,
        optuna_trial_id: int | None = None,
        sample_ceiling: int | None = None,
    ) -> TrialResult:
        """Run a single optimization trial.

        This method coordinates the full trial lifecycle:
        1. Setup trial context (ID generation, config preparation)
        2. Setup progress tracking and budget lease
        3. Execute evaluation
        4. Enforce post-evaluation constraints
        5. Build and return trial result

        Args:
            func: Function to evaluate
            config: Configuration to test
            dataset: Evaluation dataset
            trial_number: Trial number for tracking
            session_id: Session ID for hash generation (optional)
            optuna_trial_id: Optuna trial ID for pruning integration
            sample_ceiling: Optional per-trial budget ceiling (for parallel allocation)

        Returns:
            TrialResult with evaluation results
        """
        # Phase 1: Setup trial context
        optuna_trial_id = extract_optuna_trial_id(config, optuna_trial_id)
        trial_id = self._generate_trial_id(
            config, trial_number, session_id, dataset, optuna_trial_id
        )

        start_time = time.time()
        logger.debug(f"Running trial {trial_id}: {config}")
        evaluation_config = prepare_evaluation_config(config)

        # Phase 2: Setup progress tracking and budget
        progress_callback, progress_state = self._create_progress_tracking(
            optuna_trial_id, dataset, trial_id
        )
        lease = self._setup_trial_budget_lease(dataset, trial_id, sample_ceiling)

        # Wrap trial execution in tracing span
        with trial_span(trial_id, trial_number, evaluation_config) as span:
            return await self._execute_trial_with_tracing(
                func=func,
                dataset=dataset,
                trial_id=trial_id,
                evaluation_config=evaluation_config,
                start_time=start_time,
                optuna_trial_id=optuna_trial_id,
                progress_callback=progress_callback,
                progress_state=progress_state,
                lease=lease,
                span=span,
            )

    async def _execute_trial_with_tracing(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        trial_id: str,
        evaluation_config: dict[str, Any],
        start_time: float,
        optuna_trial_id: int | None,
        progress_callback: Any,
        progress_state: Any,
        lease: SampleBudgetLease | None,
        span: Any,
    ) -> TrialResult:
        """Execute trial with tracing span active."""
        orchestrator = self._orchestrator
        closure: LeaseClosure | None = None

        try:
            # Phase 3: Execute evaluation within TrialContext
            # This sets the trial context so get_trial_config() works inside the user's function
            evaluate_kwargs: dict[str, Any] = {}
            if progress_callback is not None:
                evaluate_kwargs["progress_callback"] = progress_callback
            if lease is not None:
                evaluate_kwargs["sample_lease"] = lease

            async with TrialContext(
                trial_id=trial_id,
                metadata={
                    "config": evaluation_config,
                    "optuna_trial_id": optuna_trial_id,
                },
            ):
                eval_result = await orchestrator.evaluator.evaluate(
                    func, evaluation_config, dataset, **evaluate_kwargs
                )
            budget_exhausted = bool(
                getattr(eval_result, "sample_budget_exhausted", False)
            )

            # Phase 4: Enforce constraints and extract results
            metrics_payload = getattr(eval_result, "metrics", {}) or {}
            enforce_constraints(
                evaluation_config,
                metrics_payload,
                orchestrator._constraints_post_eval,
                stage="post",
            )

            duration = time.time() - start_time
            examples_attempted, total_cost = extract_cost_from_results(
                eval_result, progress_state, trial_id
            )

            # Phase 5: Finalize budget and build result
            closure = self._finalize_budget_lease(lease)
            if closure:
                examples_attempted = closure.consumed
                if closure.exhausted:
                    budget_exhausted = True

            result = build_success_result(
                trial_id=trial_id,
                evaluation_config=evaluation_config,
                eval_result=eval_result,
                duration=duration,
                examples_attempted=examples_attempted,
                total_cost=total_cost,
                optuna_trial_id=optuna_trial_id,
            )
            self._apply_budget_metadata(result, closure, budget_exhausted)

            # Record success in tracing span
            record_trial_result(
                span,
                status="completed",
                metrics=result.metrics,
            )

            # Collect workflow span for backend visualization
            end_time = time.time()
            self._collect_workflow_span(trial_id, result, start_time, end_time)

            return result

        except TrialPrunedError as prune_error:
            result = self._build_trial_error_result(
                trial_id,
                evaluation_config,
                start_time,
                lease,
                progress_state,
                optuna_trial_id,
                prune_error,
                is_pruned=True,
            )
            record_trial_result(span, status="pruned", error=str(prune_error))
            end_time = time.time()
            self._collect_workflow_span(trial_id, result, start_time, end_time)
            return result

        except TVLConstraintError as constraint_error:
            result = self._build_trial_error_result(
                trial_id,
                evaluation_config,
                start_time,
                lease,
                progress_state,
                optuna_trial_id,
                constraint_error,
            )
            record_trial_result(span, status="failed", error=str(constraint_error))
            end_time = time.time()
            self._collect_workflow_span(trial_id, result, start_time, end_time)
            return result

        except APIKeyError:
            # Fail fast on API key errors - don't continue running trials
            # Re-raise to stop the entire optimization loop
            raise

        except asyncio.CancelledError:
            # SonarQube S7497: CancelledError must always be re-raised
            raise

        except (
            RateLimitError,
            QuotaExceededError,
            ServiceUnavailableError,
        ) as vendor_exc:
            # Re-raise as VendorPauseError so the orchestrator can prompt
            # the user to resume or stop, instead of silently failing.
            from traigent.core.exception_handler import classify_vendor_error

            category = classify_vendor_error(vendor_exc)
            if category is not None:
                raise VendorPauseError(
                    str(vendor_exc),
                    original_error=vendor_exc,
                    category=category.value,
                ) from vendor_exc
            # Defensive: if classify returns None, fall through to generic
            logger.exception("Trial %s vendor error not classified", trial_id)
            result = self._build_trial_error_result(
                trial_id,
                evaluation_config,
                start_time,
                lease,
                progress_state,
                optuna_trial_id,
                vendor_exc,
            )
            record_trial_result(span, status="failed", error=str(vendor_exc))
            end_time = time.time()
            self._collect_workflow_span(trial_id, result, start_time, end_time)
            return result

        except Exception as exc:
            logger.exception("Trial %s execution failed unexpectedly", trial_id)
            result = self._build_trial_error_result(
                trial_id,
                evaluation_config,
                start_time,
                lease,
                progress_state,
                optuna_trial_id,
                exc,
            )
            record_trial_result(span, status="failed", error=str(exc))
            end_time = time.time()
            self._collect_workflow_span(trial_id, result, start_time, end_time)
            return result

        finally:
            if lease is not None and closure is None:
                closure = self._finalize_budget_lease(lease)

    # =========================================================================
    # Trial ID Generation
    # =========================================================================

    def _generate_trial_id(
        self,
        config: dict[str, Any] | Any,
        trial_number: int,
        session_id: str | None,
        dataset: Dataset,
        optuna_trial_id: int | None,
    ) -> str:
        """Generate a deterministic trial ID based on config or fallback to sequential ID.

        Args:
            config: Configuration to test (may already exclude Optuna keys)
            trial_number: Trial number for tracking
            session_id: Session ID for hash generation
            dataset: Evaluation dataset
            optuna_trial_id: Optuna trial identifier (ensures unique hashes)

        Returns:
            Generated trial ID
        """
        orchestrator = self._orchestrator

        if session_id:
            from traigent.utils.hashing import generate_trial_hash

            dataset_name = dataset.name if hasattr(dataset, "name") else ""
            config_for_hash: dict[str, Any] | Any = config
            if isinstance(config, dict) and optuna_trial_id is not None:
                config_for_hash = dict(config)
                config_for_hash.setdefault("_optuna_trial_id", optuna_trial_id)

            trial_id: str = generate_trial_hash(
                session_id=session_id,
                config=config_for_hash,
                dataset_name=dataset_name,
            )
            logger.debug(
                f"Generated hash-based trial_id: {trial_id} for config: {config}"
            )
            return trial_id
        else:
            # Fallback to sequential ID for backwards compatibility
            return f"{orchestrator._optimization_id}_{trial_number}"

    # =========================================================================
    # Progress Tracking
    # =========================================================================

    def _create_progress_tracking(
        self,
        optuna_trial_id: int | None,
        dataset: Dataset,
        trial_id: str,
    ) -> tuple[Callable[[int, dict[str, Any]], None] | None, dict[str, Any] | None]:
        """Create progress callback and state for pruning if applicable.

        Returns (None, None) early if no Optuna trial ID is available, so evaluators
        don't receive a callback that would fail.

        Args:
            optuna_trial_id: Optuna trial ID for pruning integration
            dataset: Evaluation dataset
            trial_id: Trial identifier for logging

        Returns:
            Tuple of (progress_callback, progress_state) or (None, None) if no pruning
        """
        orchestrator = self._orchestrator

        # Early return if no Optuna trial ID or optimizer lacks reporting capability
        if optuna_trial_id is None or not hasattr(
            orchestrator.optimizer, "report_intermediate_value"
        ):
            return None, None

        # Extract band target for banded objective pruning
        band_target = None
        if (
            orchestrator.objective_schema is not None
            and orchestrator.objective_schema.objectives
        ):
            primary_obj = orchestrator.objective_schema.objectives[0]
            if hasattr(primary_obj, "band") and primary_obj.band is not None:
                band_target = primary_obj.band

        tracker = PruningProgressTracker(
            optimizer=orchestrator.optimizer,
            dataset=dataset,
            trial_id=trial_id,
            optuna_trial_id=optuna_trial_id,
            band_target=band_target,
        )
        return tracker.callback, tracker.state

    # =========================================================================
    # Budget Management
    # =========================================================================

    def _setup_trial_budget_lease(
        self,
        dataset: Dataset,
        trial_id: str,
        sample_ceiling: int | None,
    ) -> SampleBudgetLease | None:
        """Setup budget lease for trial execution.

        Args:
            dataset: Evaluation dataset
            trial_id: Trial identifier
            sample_ceiling: Optional ceiling from parallel allocation

        Returns:
            SampleBudgetLease if budget manager is configured, None otherwise
        """
        orchestrator = self._orchestrator

        if orchestrator._sample_budget_manager is None:
            return None

        ceiling_value = sample_ceiling
        if ceiling_value is None:
            remaining_float = orchestrator._sample_budget_manager.remaining()
            dataset_size = len(dataset)
            if math.isfinite(remaining_float):
                ceiling_value = min(dataset_size, int(remaining_float))
            else:
                ceiling_value = dataset_size

        return orchestrator._sample_budget_manager.create_lease(
            trial_id=trial_id,
            ceiling=ceiling_value,
        )

    def _finalize_budget_lease(
        self, lease: SampleBudgetLease | None
    ) -> LeaseClosure | None:
        """Finalize a budget lease and update orchestrator state.

        Args:
            lease: Budget lease to finalize (may be None)

        Returns:
            LeaseClosure with consumption details, or None if no lease
        """
        if lease is None:
            return None

        orchestrator = self._orchestrator
        closure = lease.finalize()

        if orchestrator._sample_budget_manager is not None:
            orchestrator._consumed_examples = int(
                orchestrator._sample_budget_manager.consumed()
            )

        logger.debug(
            "Finalized sample lease for trial %s: consumed=%s, remaining=%s, exhausted=%s",
            lease.trial_id,
            closure.consumed,
            closure.global_remaining,
            closure.exhausted,
        )

        if closure.exhausted:
            orchestrator._examples_capped += 1
            if not orchestrator._stop_reason:
                orchestrator._stop_reason = "max_samples_reached"

        return closure

    def _apply_budget_metadata(
        self,
        trial_result: TrialResult,
        closure: LeaseClosure | None,
        budget_exhausted: bool,
    ) -> None:
        """Apply budget metadata to trial result.

        Args:
            trial_result: Trial result to update
            closure: Lease closure with consumption details
            budget_exhausted: Whether budget was exhausted during evaluation
        """
        metadata = dict(trial_result.metadata or {})
        metrics = dict(trial_result.metrics or {})

        if closure is not None:
            metadata.setdefault("examples_attempted", closure.consumed)
            metadata["sample_budget_remaining"] = closure.global_remaining
            metadata["sample_budget_exhausted"] = budget_exhausted or closure.exhausted
            metadata["sample_budget_consumed"] = closure.consumed
            metadata["sample_budget_wasted"] = closure.wasted
            metrics.setdefault("examples_attempted", closure.consumed)
        else:
            metadata["sample_budget_exhausted"] = budget_exhausted

        if budget_exhausted or (closure is not None and closure.exhausted):
            metadata.setdefault("stop_reason", "sample_budget_exhausted")

        trial_result.metadata = metadata
        trial_result.metrics = metrics

    def _collect_workflow_span(
        self,
        trial_id: str,
        trial_result: TrialResult,
        start_time: float,
        end_time: float,
    ) -> None:
        """Create and collect a workflow span for the completed trial.

        Args:
            trial_id: Trial identifier (used as configuration_run_id)
            trial_result: The trial result containing metrics and status
            start_time: Trial start timestamp
            end_time: Trial end timestamp
        """
        orchestrator = self._orchestrator

        # Only collect if tracker is configured (use getattr for mock compatibility)
        tracker = getattr(orchestrator, "_workflow_traces_tracker", None)
        if tracker is None:
            return

        try:
            # Lazy import to avoid circular imports
            from traigent.integrations.observability.workflow_traces import (
                SpanPayload,
                SpanStatus,
                SpanType,
            )

            # Determine span status based on trial status
            if trial_result.status == TrialStatus.COMPLETED:
                span_status = SpanStatus.COMPLETED.value
            elif trial_result.status == TrialStatus.PRUNED:
                span_status = SpanStatus.REJECTED.value
            else:
                span_status = SpanStatus.FAILED.value

            # Create span payload
            # Note: trial_id IS the configuration_run_id (backend creates it at /next-trial)
            # Use timezone-aware datetime with UTC; .isoformat() includes +00:00 offset
            # DO NOT add "Z" suffix - that would create invalid "+00:00Z" format
            span = SpanPayload(
                span_id=uuid.uuid4().hex[:16],
                trace_id=orchestrator._optimization_id,  # Use optimization ID as trace
                configuration_run_id=trial_id,
                span_name=f"trial_{trial_result.trial_id}",
                span_type=SpanType.NODE.value,
                start_time=datetime.fromtimestamp(start_time, UTC).isoformat(),
                end_time=datetime.fromtimestamp(end_time, UTC).isoformat(),
                status=span_status,
                node_id="optimization_run",  # Links span to workflow graph node
                error_message=trial_result.error_message,
                input_tokens=(
                    trial_result.metadata.get("input_tokens", 0)
                    if trial_result.metadata
                    else 0
                ),
                output_tokens=(
                    trial_result.metadata.get("output_tokens", 0)
                    if trial_result.metadata
                    else 0
                ),
                cost_usd=(
                    trial_result.metrics.get("total_cost", 0.0)
                    if trial_result.metrics
                    else 0.0
                ),
                input_data={"config": trial_result.config},
                output_data={"metrics": trial_result.metrics},
                metadata={
                    "trial_number": (
                        trial_result.metadata.get("trial_number")
                        if trial_result.metadata
                        else None
                    ),
                    "examples_attempted": (
                        trial_result.metadata.get("examples_attempted")
                        if trial_result.metadata
                        else None
                    ),
                },
            )

            # Collect the span via orchestrator
            orchestrator.collect_workflow_span(span)
            logger.debug(f"Collected workflow span for trial {trial_id}")

        except Exception as exc:
            logger.debug(f"Failed to collect workflow span for trial {trial_id}: {exc}")

    # =========================================================================
    # Error Result Building
    # =========================================================================

    def _build_trial_error_result(
        self,
        trial_id: str,
        evaluation_config: dict[str, Any],
        start_time: float,
        lease: SampleBudgetLease | None,
        progress_state: dict[str, Any] | None,
        optuna_trial_id: int | None,
        error: Exception,
        is_pruned: bool = False,
    ) -> TrialResult:
        """Build error result for failed or pruned trial.

        Args:
            trial_id: Trial identifier
            evaluation_config: Configuration used for evaluation
            start_time: Trial start time (for duration calculation)
            lease: Budget lease (may be None)
            progress_state: Progress state dict (may be None)
            optuna_trial_id: Optuna trial ID (may be None)
            error: Exception that caused the failure
            is_pruned: Whether this is a pruned trial (vs failed)

        Returns:
            TrialResult with appropriate status and error information
        """
        duration = time.time() - start_time
        closure = self._finalize_budget_lease(lease)
        budget_exhausted = bool(closure.exhausted) if closure else False

        if is_pruned and isinstance(error, TrialPrunedError):
            trial_result = build_pruned_result(
                trial_id=trial_id,
                evaluation_config=evaluation_config,
                duration=duration,
                prune_error=error,
                progress_state=progress_state or {},
                optuna_trial_id=optuna_trial_id,
            )
        else:
            trial_result = build_failed_result(
                trial_id=trial_id,
                evaluation_config=evaluation_config,
                duration=duration,
                error=error,
                progress_state=progress_state or {},
                optuna_trial_id=optuna_trial_id,
            )

        self._apply_budget_metadata(trial_result, closure, budget_exhausted)
        return trial_result
