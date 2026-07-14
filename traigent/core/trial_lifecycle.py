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
import inspect
import math
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from traigent.config.context import TrialContext, WorkflowTraceContext
from traigent.core.optimization_pipeline import (
    build_surrogate_descriptor,
    get_surrogate_evaluator,
    get_surrogate_evaluator_name,
)
from traigent.core.orchestrator_helpers import (
    enforce_constraints,
    extract_cost_from_results,
    extract_optuna_trial_id,
    prepare_evaluation_config,
)
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
    InsufficientFundsError,
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


def _resolve_primary_objective(orchestrator: Any) -> str | None:
    """Return the run's primary objective name, or None when unavailable.

    Used to populate :attr:`TrialResult.score` with the optimization signal
    (issue #1845). Defensive: custom optimizers may not expose ``objectives``.
    """
    optimizer = getattr(orchestrator, "optimizer", None)
    objectives = getattr(optimizer, "objectives", None) if optimizer else None
    if objectives:
        first = objectives[0]
        if isinstance(first, str) and first:
            return first
    return None


# ---------------------------------------------------------------------------
# Surrogate (pre-screen) scoring
#
# A configured surrogate is a cheap SECOND scorer over the outputs the primary
# evaluator already captured. It NEVER re-executes the decorated function: the
# helpers below only read ``example_result.actual_output``. All scoring is
# fail-soft — any error drops the surrogate field for that example/trial and is
# logged at most once per trial; primary metrics are never perturbed.
# ---------------------------------------------------------------------------


def _build_surrogate_kwargs(
    surrogate: Callable[..., Any], output: Any, expected: Any, example: Any
) -> dict[str, Any] | None:
    """Map (output, expected, example) onto the surrogate's parameter names.

    Mirrors the ``scoring_function`` calling convention (``output``/``actual``,
    ``expected``): the surrogate is a scorer of outputs, not a func-executor.
    Returns None to signal "call positionally with the single output" when the
    signature cannot be inspected or exposes no recognised parameter names.
    """
    try:
        params = inspect.signature(surrogate).parameters
    except (TypeError, ValueError):
        return None

    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    names = set(params)
    kwargs: dict[str, Any] = {}
    if "output" in names or has_var_kw:
        kwargs["output"] = output
    elif "actual" in names:
        kwargs["actual"] = output
    if "expected_output" in names or has_var_kw:
        kwargs["expected_output"] = expected
    elif "expected" in names:
        kwargs["expected"] = expected
    if "example" in names or has_var_kw:
        kwargs["example"] = example
    return kwargs or None


async def _invoke_surrogate(
    surrogate: Callable[..., Any], output: Any, expected: Any, example: Any
) -> Any:
    """Invoke the surrogate over a single captured output (sync or async)."""
    kwargs = _build_surrogate_kwargs(surrogate, output, expected, example)
    result = surrogate(output) if kwargs is None else surrogate(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def _coerce_surrogate_score(raw: Any) -> float | None:
    """Coerce a surrogate return value to a float in [0, 1], or None to drop it.

    Out-of-range (``<0`` / ``>1``) and non-finite values are DROPPED, never
    clipped: the backend evaluator-tensor reader rejects out-of-range surrogate
    scores, so pre-clipping here would launder garbage past that guard. A dropped
    score simply omits the surrogate field for that example (fail-soft).
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, dict):
        raw = raw.get("surrogate_score", raw.get("score"))
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    if value < 0.0 or value > 1.0:
        return None
    return value


async def apply_surrogate_scoring(
    eval_result: Any, surrogate: Callable[..., Any], trial_id: str
) -> None:
    """Score already-captured per-example outputs with the surrogate, in place.

    Injects ``surrogate_score`` into each scored example's metrics and the mean
    onto the flat trial metrics. Skips (single warning) when per-example outputs
    are unavailable — it never triggers a second execution pass.
    """
    example_results = getattr(eval_result, "example_results", None)
    if not isinstance(example_results, list) or not example_results:
        logger.warning(
            "Surrogate evaluator configured but trial %s has no per-example "
            "outputs; skipping surrogate scoring (no re-execution).",
            trial_id,
        )
        return

    scored: list[float] = []
    errored = 0
    for example_result in example_results:
        output = getattr(example_result, "actual_output", None)
        metrics = getattr(example_result, "metrics", None)
        if output is None or not isinstance(metrics, dict):
            # No captured output (e.g. a failed example) — nothing to score.
            continue
        try:
            raw = await _invoke_surrogate(
                surrogate,
                output,
                getattr(example_result, "expected_output", None),
                example_result,
            )
            value = _coerce_surrogate_score(raw)
        except asyncio.CancelledError:
            raise
        except Exception:
            errored += 1
            continue
        if value is None:
            errored += 1
            continue
        metrics["surrogate_score"] = value
        scored.append(value)

    if errored:
        logger.warning(
            "Surrogate evaluator dropped surrogate_score for %d example(s) in "
            "trial %s (per-example error, non-numeric, or out-of-range [0,1] "
            "result).",
            errored,
            trial_id,
        )

    if scored:
        aggregate = getattr(eval_result, "metrics", None)
        if isinstance(aggregate, dict):
            aggregate["surrogate_score"] = sum(scored) / len(scored)


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
        self._constraint_rejection_count = 0

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
        optimizer_state_before_suggest = self._snapshot_optimizer_state()

        if (
            orchestrator.max_trials is not None
            and trial_count >= orchestrator.max_trials
        ):
            logger.info(f"Trial limit reached: {orchestrator.max_trials}")
            orchestrator._stop_reason = "max_trials_reached"
            return trial_count, "break"

        config = orchestrator._consume_default_config()
        if config is not None:
            # baseline trials get the same Fixed/CVAR injection (RFC 0001)
            config = orchestrator._apply_knob_resolution(config)
        if config is None:
            try:
                suggest_next = getattr(orchestrator, "_suggest_next_trial_config", None)
                if callable(suggest_next):
                    config = await suggest_next(dataset)
                else:
                    config = orchestrator.optimizer.suggest_next_trial(
                        orchestrator._trials
                    )
                    config = orchestrator._apply_knob_resolution(config)
            except OptimizationError as e:
                stop_reason = self._terminal_optimizer_stop_reason(e)
                if stop_reason is None:
                    logger.exception(
                        "Optimizer failed to suggest the next trial: %s", e
                    )
                    raise
                logger.info("Optimizer could not suggest another trial: %s", e)
                orchestrator._stop_reason = cast(Any, stop_reason)
                return trial_count, "break"
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.exception("Optimizer failed to suggest the next trial: %s", e)
                return trial_count, "break"

        if config is None:
            logger.info("Optimizer did not suggest a trial configuration")
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
                return trial_count, "continue"
            config = filtered_configs[0]
            optuna_trial_id = (
                config.get("_optuna_trial_id") if isinstance(config, dict) else None
            )

        config_for_run = prepare_evaluation_config(config)  # type: ignore[arg-type]
        config_for_trial = cast(dict[str, Any], config)

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
            # (Fix for the tracked fix: constraint-rejected configs should not reduce actual evaluations)
            logger.info(
                "Pre-constraint failed for config %s: %s (not counting toward max_trials)",
                config_for_run,
                e,
            )
            try:
                self._record_pre_constraint_pruned_result(
                    config=config_for_trial,
                    evaluation_config=config_for_run,
                    dataset=dataset,
                    trial_number=trial_count,
                    session_id=session_id,
                    optuna_trial_id=optuna_trial_id,
                    error=e,
                )
            finally:
                self._restore_optimizer_state(optimizer_state_before_suggest)
            return trial_count, "continue"

        # Phase 2.1: Acquire cost permit before sequential trial execution
        # This ensures sequential trials respect cost limits proactively
        permit: Permit | None = None
        if orchestrator.cost_enforcer is not None:
            permit = await orchestrator.cost_enforcer.acquire_permit_async()
            if not permit.is_granted:
                logger.info(
                    "Sequential trial cancelled due to cost limit reached (config=%s)",
                    config_for_run,
                )
                # Mid-run cost limits are graceful by contract: return a partial
                # result and surface result.stop_reason == "cost_limit".
                orchestrator._stop_reason = "cost_limit"
                return trial_count, "break"

        orchestrator.callback_manager.on_trial_start(trial_count, config_for_run)

        try:
            trial_result = await self.run_trial(
                func,
                config_for_trial,
                dataset,
                trial_count,
                session_id,
                optuna_trial_id=optuna_trial_id,
            )

            trial_count = await orchestrator._handle_trial_result(
                trial_result=trial_result,
                optimizer_config=config_for_trial,
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

    def _terminal_optimizer_stop_reason(self, error: OptimizationError) -> str | None:
        """Classify explicit optimizer terminal states without swallowing defects."""

        message = str(error)
        if "Maximum trials" in message:
            return "max_trials_reached"
        if message.startswith("Config space exhausted"):
            return "space_exhausted"
        if message == "All grid combinations have been evaluated":
            return "space_exhausted"
        return None

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
        backend_trial_id = (
            config.get("_traigent_backend_trial_id")
            if isinstance(config, dict)
            else None
        )
        trial_id = (
            str(backend_trial_id)
            if isinstance(backend_trial_id, str) and backend_trial_id
            else self._generate_trial_id(
                config, trial_number, session_id, dataset, optuna_trial_id
            )
        )

        start_time = time.time()
        logger.debug(f"Running trial {trial_id}: {config}")
        evaluation_config = prepare_evaluation_config(config)

        # Phase 2: Setup progress tracking and budget
        progress_callback, progress_state = self._create_progress_tracking(
            optuna_trial_id, dataset, trial_id, session_id
        )
        lease = self._setup_trial_budget_lease(dataset, trial_id, sample_ceiling)

        # Wrap trial execution in tracing span
        with trial_span(trial_id, trial_number, evaluation_config) as span:
            return await self._execute_trial_with_tracing(
                func=func,
                dataset=dataset,
                trial_id=trial_id,
                backend_trial_id=(
                    backend_trial_id if isinstance(backend_trial_id, str) else None
                ),
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
        backend_trial_id: str | None,
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
            evaluate_kwargs = self._build_evaluate_kwargs(progress_callback, lease)

            async with (
                TrialContext(
                    trial_id=trial_id,
                    metadata={
                        "config": evaluation_config,
                        "optuna_trial_id": optuna_trial_id,
                    },
                ),
                WorkflowTraceContext(
                    {
                        "configuration_run_id": trial_id,
                        "workflow_trace_id": orchestrator._optimization_id,
                        "workflow_trace_manager": getattr(
                            orchestrator, "_workflow_trace_manager", None
                        ),
                    }
                ),
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

            # Surrogate (pre-screen) scoring: cheap second scorer over the
            # already-captured outputs. Runs after constraint enforcement so it
            # never perturbs primary metrics/constraints; fully fail-soft. Mutates
            # eval_result in place BEFORE build_success_result so the per-example
            # and aggregate surrogate_score flow into the trial payload.
            surrogate_descriptor = await self._score_surrogate(eval_result, trial_id)

            result = build_success_result(
                trial_id=trial_id,
                evaluation_config=evaluation_config,
                eval_result=eval_result,
                duration=duration,
                examples_attempted=examples_attempted,
                total_cost=total_cost,
                optuna_trial_id=optuna_trial_id,
                # Thread the evaluator's runtime-only computable names (registry +
                # RAGAS + user metric functions) into the final-union cap so an
                # evaluator-computed objective like ``f1`` is never dropped as a
                # user key. Guard with getattr: custom evaluators may not subclass
                # BaseEvaluator and so may lack this method.
                extra_reserved=getattr(
                    orchestrator.evaluator,
                    "_evaluator_computable_metric_names",
                    lambda: frozenset(),
                )(),
                # Primary objective drives TrialResult.score population (#1845).
                primary_objective=_resolve_primary_objective(orchestrator),
            )
            if surrogate_descriptor is not None:
                result.metadata["surrogate_evaluator"] = surrogate_descriptor

            self._mark_backend_trial_id_acquired(result, backend_trial_id)
            self._apply_budget_metadata(result, closure, budget_exhausted)
            self._apply_effectuation_metadata(result, func)

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
            self._mark_backend_trial_id_acquired(result, backend_trial_id)
            self._apply_effectuation_metadata(result, func)
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
            self._mark_backend_trial_id_acquired(result, backend_trial_id)
            self._apply_effectuation_metadata(result, func)
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
            InsufficientFundsError,
            ServiceUnavailableError,
        ) as vendor_exc:
            # Re-raise as VendorPauseError so the orchestrator can prompt
            # the user to resume or stop, instead of silently failing.
            # classify_vendor_error is guaranteed to match for these types.
            from traigent.core.exception_handler import classify_vendor_error

            category = classify_vendor_error(vendor_exc)
            raise VendorPauseError(
                str(vendor_exc),
                original_error=vendor_exc,
                category=category,
            ) from vendor_exc

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
            self._mark_backend_trial_id_acquired(result, backend_trial_id)
            self._apply_effectuation_metadata(result, func)
            record_trial_result(span, status="failed", error=str(exc))
            end_time = time.time()
            self._collect_workflow_span(trial_id, result, start_time, end_time)
            return result

        finally:
            if lease is not None and closure is None:
                closure = self._finalize_budget_lease(lease)

    @staticmethod
    def _mark_backend_trial_id_acquired(
        result: TrialResult,
        backend_trial_id: str | None,
    ) -> None:
        if not isinstance(backend_trial_id, str) or not backend_trial_id:
            return
        metadata = dict(result.metadata or {})
        metadata["backend_trial_id_acquired"] = True
        metadata["backend_trial_id_source"] = "cloud_brain"
        result.metadata = metadata

    def _apply_effectuation_metadata(
        self,
        result: TrialResult,
        func: Callable[..., Any],
    ) -> None:
        events = self._effectuation_events(func)
        if not events:
            return
        metadata = dict(result.metadata or {})
        metadata["effectuation_events"] = events
        result.metadata = metadata

    def _effectuation_events(self, func: Callable[..., Any]) -> list[dict[str, Any]]:
        try:
            from traigent.effectuation import EFFECTUATION_EVENTS_ATTR

            emitter = getattr(func, EFFECTUATION_EVENTS_ATTR, None)
            if not callable(emitter):
                return []
            raw_events = emitter()
        except Exception:
            logger.debug("Failed to collect effectuation events", exc_info=True)
            return []

        events: list[dict[str, Any]] = []
        for event in raw_events or ():
            if isinstance(event, dict):
                events.append(dict(event))
        return events

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

    def _snapshot_optimizer_trial_count(self) -> int | None:
        trial_count = getattr(self._orchestrator.optimizer, "_trial_count", None)
        if isinstance(trial_count, int):
            return trial_count
        return None

    def _snapshot_optimizer_state(self) -> tuple[int | None, set[str] | None]:
        """Snapshot optimizer counters that suggestion may consume."""

        tried_hashes = getattr(
            self._orchestrator.optimizer, "_tried_config_hashes", None
        )
        if isinstance(tried_hashes, set):
            tried_snapshot = set(tried_hashes)
        else:
            tried_snapshot = None
        return self._snapshot_optimizer_trial_count(), tried_snapshot

    def _restore_optimizer_trial_count(self, previous_trial_count: int | None) -> None:
        """Restore the optimizer's consumed slot for a rejected pre-constraint."""
        if previous_trial_count is None:
            return
        current_trial_count = getattr(
            self._orchestrator.optimizer, "_trial_count", None
        )
        if (
            isinstance(current_trial_count, int)
            and current_trial_count > 0
            and current_trial_count > previous_trial_count
        ):
            self._orchestrator.optimizer._trial_count = previous_trial_count

    def _restore_optimizer_state(
        self, previous_state: tuple[int | None, set[str] | None]
    ) -> None:
        """Restore optimizer state for a rejected pre-constraint."""

        previous_trial_count, previous_tried_hashes = previous_state
        self._restore_optimizer_trial_count(previous_trial_count)
        if previous_tried_hashes is not None and hasattr(
            self._orchestrator.optimizer, "_tried_config_hashes"
        ):
            self._orchestrator.optimizer._tried_config_hashes = previous_tried_hashes

    def _generate_constraint_rejection_trial_id(
        self,
        config: dict[str, Any],
        trial_number: int,
        session_id: str | None,
        dataset: Dataset,
        optuna_trial_id: int | None,
    ) -> str:
        """Generate a unique, non-consuming trial ID for a constraint rejection."""
        self._constraint_rejection_count += 1
        base_trial_id = self._generate_trial_id(
            config,
            trial_number,
            session_id,
            dataset,
            optuna_trial_id,
        )
        return f"{base_trial_id}_rej{self._constraint_rejection_count}"

    # =========================================================================
    # Progress Tracking
    # =========================================================================

    def _create_progress_tracking(
        self,
        optuna_trial_id: int | None,
        dataset: Dataset,
        trial_id: str,
        session_id: str | None,
    ) -> tuple[Callable[[int, dict[str, Any]], None] | None, dict[str, Any] | None]:
        """Return progress callback state.

        Local Optuna-backed intermediate reporting remains evicted. Cloud
        smart-pruning reporting is enabled only when the backend session manager
        confirms this run is managed/cloud, backend-tracked, and configured with
        smart_pruning.
        """
        orchestrator = self._orchestrator
        manager = getattr(orchestrator, "backend_session_manager", None)
        should_report = getattr(manager, "should_report_intermediate_progress", None)
        if not callable(should_report) or not should_report(session_id):
            return None, None

        try:
            total_examples = len(dataset)
        except TypeError:
            total_examples = 0

        objective_name = self._primary_objective_name()
        state: dict[str, Any] = {
            "session_id": session_id,
            "trial_id": trial_id,
            "optuna_trial_id": optuna_trial_id,
            "evaluated": 0,
            "total_examples": total_examples,
            "correct_sum": 0.0,
            "metric_sum": 0.0,
            "last_running_score": 0.0,
            "partial_cost_usd": 0.0,
            "objective_name": objective_name,
        }

        def _progress_callback(step: int, data: dict[str, Any]) -> None:
            state["last_step"] = step
            if not isinstance(data, dict):
                data = {}
            if data.get("stop_reason"):
                return

            state["evaluated"] = int(state.get("evaluated", 0)) + 1
            metric_value = self._extract_progress_value(data, objective_name)
            if metric_value is None:
                success = data.get("success")
                if isinstance(success, bool):
                    metric_value = 1.0 if success else 0.0
            if metric_value is None:
                metric_value = float(state.get("last_running_score", 0.0))

            state["metric_sum"] = float(state.get("metric_sum", 0.0)) + metric_value
            state["correct_sum"] = float(state.get("correct_sum", 0.0)) + metric_value
            evaluated = max(1, int(state["evaluated"]))
            running_score = float(state["metric_sum"]) / evaluated
            state["last_running_score"] = running_score

            cost_delta = self._extract_progress_cost(data)
            if cost_delta is not None:
                state["partial_cost_usd"] = (
                    float(state.get("partial_cost_usd", 0.0)) + cost_delta
                )

            payload: dict[str, Any] = {
                "session_id": session_id,
                "trial_id": trial_id,
                "running_score": running_score,
                "examples_attempted": int(state["evaluated"]),
            }
            if objective_name:
                payload["objective_name"] = objective_name
            partial_cost = float(state.get("partial_cost_usd", 0.0))
            if partial_cost > 0:
                payload["partial_cost_usd"] = partial_cost

            report = getattr(manager, "report_intermediate_progress", None)
            decision = report(payload) if callable(report) else {}
            if isinstance(decision, dict) and decision.get("prune") is True:
                reason = decision.get("prune_reason") or "cloud smart pruning"
                logger.info(
                    "Cloud smart pruning requested trial pruning after %s examples: %s",
                    state["evaluated"],
                    reason,
                )
                raise TrialPrunedError(str(reason), step=int(state["evaluated"]))

        return _progress_callback, state

    def _primary_objective_name(self) -> str | None:
        objectives = getattr(self._orchestrator, "objectives", None)
        if objectives:
            first = objectives[0]
            if first is not None:
                return str(first)
        optimizer_objectives = getattr(
            getattr(self._orchestrator, "optimizer", None), "objectives", None
        )
        if optimizer_objectives:
            return str(optimizer_objectives[0])
        return None

    @staticmethod
    def _coerce_progress_float(value: Any) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    @classmethod
    def _extract_progress_value(
        cls, data: dict[str, Any], objective_name: str | None
    ) -> float | None:
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            candidate_names = [
                name
                for name in (objective_name, "score", "accuracy")
                if isinstance(name, str) and name
            ]
            for name in candidate_names:
                value = cls._coerce_progress_float(metrics.get(name))
                if value is not None:
                    return value
        for key in (objective_name, "score", "accuracy"):
            if isinstance(key, str) and key:
                value = cls._coerce_progress_float(data.get(key))
                if value is not None:
                    return value
        return None

    @classmethod
    def _extract_progress_cost(cls, data: dict[str, Any]) -> float | None:
        metrics = data.get("metrics")
        sources = [data]
        if isinstance(metrics, dict):
            sources.insert(0, metrics)
        for source in sources:
            for key in ("partial_cost_usd", "total_example_cost", "total_cost", "cost"):
                value = cls._coerce_progress_float(source.get(key))
                if value is not None and value >= 0:
                    return value
        return None

    def _record_pre_constraint_pruned_result(
        self,
        *,
        config: dict[str, Any],
        evaluation_config: dict[str, Any],
        dataset: Dataset,
        trial_number: int,
        session_id: str | None,
        optuna_trial_id: int | None,
        error: TVLConstraintError,
    ) -> None:
        """Record a constraint-rejected config without consuming a trial slot."""
        orchestrator = self._orchestrator
        trial_id = self._generate_constraint_rejection_trial_id(
            config,
            trial_number,
            session_id,
            dataset,
            optuna_trial_id,
        )
        try:
            total_examples = len(dataset)
        except TypeError:
            total_examples = 0

        result = build_pruned_result(
            trial_id=trial_id,
            evaluation_config=evaluation_config,
            duration=0.0,
            prune_error=TrialPrunedError(
                f"trial_rejected_by_constraint: {error}",
                step=0,
            ),
            progress_state={"evaluated": 0, "total_examples": total_examples},
            optuna_trial_id=optuna_trial_id,
        )
        result.error_message = str(error)
        metadata = dict(result.metadata or {})
        metadata.update(
            {
                "constraint_rejected": True,
                "stop_reason": "trial_rejected_by_constraint",
            }
        )
        result.metadata = metadata

        orchestrator._trials.append(result)
        log_trial = getattr(orchestrator, "_log_trial", None)
        if callable(log_trial):
            log_trial(result)

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

    @staticmethod
    def _build_evaluate_kwargs(
        progress_callback: Any, lease: SampleBudgetLease | None
    ) -> dict[str, Any]:
        """Build keyword arguments for evaluator.evaluate()."""
        kwargs: dict[str, Any] = {}
        if progress_callback is not None:
            kwargs["progress_callback"] = progress_callback
        if lease is not None:
            kwargs["sample_lease"] = lease
        return kwargs

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

    async def _score_surrogate(
        self, eval_result: Any, trial_id: str
    ) -> dict[str, Any] | None:
        """Apply the configured surrogate scorer to captured outputs, fail-soft.

        Returns the surrogate descriptor to attach to the trial metadata when a
        surrogate is configured, or None otherwise. Any surrogate failure is
        swallowed here: the surrogate fields are dropped and the trial proceeds
        with its primary metrics untouched.
        """
        surrogate = get_surrogate_evaluator(self._orchestrator.evaluator)
        if surrogate is None:
            return None

        surrogate_name = get_surrogate_evaluator_name(self._orchestrator.evaluator)

        # Build the descriptor FIRST and require a valid source fingerprint. A
        # surrogate whose source cannot be fingerprinted is unidentifiable: the
        # descriptor's ``config.fingerprint_source`` would be None (violating the
        # backend's ``^fp1:[0-9a-f]{64}$`` contract), and a scored-but-
        # unidentifiable evaluator would corrupt the server-side score tensor.
        # In that case we drop the WHOLE descriptor AND skip scoring (fail-soft),
        # so no surrogate_score is emitted for this trial.
        try:
            descriptor: dict[str, Any] | None = build_surrogate_descriptor(
                surrogate, name=surrogate_name
            )
        except Exception as descriptor_error:
            logger.warning(
                "Surrogate descriptor could not be built for trial %s: %s "
                "(surrogate scoring skipped; primary metrics unaffected).",
                trial_id,
                descriptor_error,
            )
            return None

        if not isinstance(descriptor, dict) or not isinstance(
            descriptor.get("config"), dict
        ):
            return None
        if descriptor["config"].get("fingerprint_source") is None:
            logger.warning(
                "Surrogate evaluator source could not be fingerprinted for trial "
                "%s; dropping the surrogate descriptor and all surrogate scores "
                "(an unidentifiable scorer would corrupt the server tensor).",
                trial_id,
            )
            return None

        try:
            await apply_surrogate_scoring(eval_result, surrogate, trial_id)
        except asyncio.CancelledError:
            raise
        except Exception as surrogate_error:
            logger.warning(
                "Surrogate evaluation failed for trial %s: %s (surrogate fields "
                "dropped; primary metrics unaffected).",
                trial_id,
                surrogate_error,
            )
        return descriptor

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
            trial_result.error_message = str(error)
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
