"""Utilities for constructing trial results."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from traigent.api.types import TrialResult, TrialStatus
from traigent.utils.exceptions import TrialPrunedError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def build_success_result(
    *,
    trial_id: str,
    evaluation_config: dict[str, Any],
    eval_result: Any,
    duration: float,
    examples_attempted: int | None,
    total_cost: float | None,
    optuna_trial_id: int | None,
) -> TrialResult:
    """Create a successful :class:`TrialResult` instance."""

    trial_metadata: dict[str, Any] = {
        "success_rate": getattr(eval_result, "success_rate", None),
        "has_errors": getattr(eval_result, "has_errors", None),
        "output_count": (
            len(eval_result.outputs)
            if getattr(eval_result, "outputs", None) is not None
            else 0
        ),
        "evaluation_result": eval_result,
    }

    example_results = getattr(eval_result, "example_results", None)
    if example_results:
        trial_metadata["example_results"] = example_results

    if examples_attempted is not None:
        trial_metadata["examples_attempted"] = int(examples_attempted)

    if total_cost is not None:
        try:
            trial_metadata["total_example_cost"] = float(total_cost)
        except (TypeError, ValueError):
            pass

    trial_result = TrialResult(
        trial_id=trial_id,
        config=evaluation_config,
        metrics=getattr(eval_result, "metrics", {}) or {},
        status=TrialStatus.COMPLETED,
        duration=duration,
        timestamp=datetime.now(UTC),
        metadata=trial_metadata,
    )

    if optuna_trial_id is not None:
        trial_result.metadata.setdefault("optuna_trial_id", optuna_trial_id)

    if examples_attempted is not None:
        try:
            trial_result.metrics.setdefault(
                "examples_attempted", int(examples_attempted)
            )
        except (TypeError, ValueError) as e:
            logger.warning(
                "Failed to convert examples_attempted to int for trial %s: %s (value: %s)",
                trial_id,
                e,
                examples_attempted,
            )

    if total_cost is not None:
        try:
            trial_result.metrics.setdefault("total_cost", float(total_cost))
        except (TypeError, ValueError) as e:
            logger.warning(
                "Failed to convert total_cost to float for trial %s: %s (value: %s)",
                trial_id,
                e,
                total_cost,
            )

    if getattr(eval_result, "summary_stats", None):
        trial_result.summary_stats = eval_result.summary_stats  # type: ignore[attr-defined]

    logger.debug(
        "Trial %s completed: %s, duration: %.2fs",
        trial_id,
        getattr(eval_result, "metrics", {}),
        duration,
    )

    return trial_result


def build_pruned_result(
    *,
    trial_id: str,
    evaluation_config: dict[str, Any],
    duration: float,
    prune_error: TrialPrunedError,
    progress_state: dict[str, Any] | None,
    optuna_trial_id: int | None,
) -> TrialResult:
    """Create a pruned :class:`TrialResult` instance."""

    metadata: dict[str, Any] = {
        "pruned": True,
        "pruned_step": prune_error.step,
    }
    if optuna_trial_id is not None:
        metadata["optuna_trial_id"] = optuna_trial_id

    metrics: dict[str, float] = {}
    if progress_state:
        evaluated = int(progress_state.get("evaluated", 0))
        total_examples = int(progress_state.get("total_examples", 0))
        if total_examples > 0:
            evaluated = min(evaluated, total_examples)

        if evaluated:
            correct_sum = float(progress_state.get("correct_sum", 0.0))
            metrics["accuracy"] = correct_sum / evaluated
            metadata["examples_attempted"] = evaluated

        total_cost_value = progress_state.get("total_cost")
        if total_cost_value is not None:
            try:
                cost_value = float(total_cost_value)
                metrics["total_cost"] = cost_value
                metrics.setdefault("cost", cost_value)
                metadata["total_example_cost"] = cost_value
            except (TypeError, ValueError) as e:
                logger.warning(
                    "Failed to convert total_cost to float for pruned trial %s: %s (value: %s)",
                    trial_id,
                    e,
                    total_cost_value,
                )

    return TrialResult(
        trial_id=trial_id,
        config=evaluation_config,
        metrics=metrics,
        status=TrialStatus.PRUNED,
        duration=duration,
        timestamp=datetime.now(UTC),
        metadata=metadata,
    )


def build_failed_result(
    *,
    trial_id: str,
    evaluation_config: dict[str, Any],
    duration: float,
    error: Exception,
    progress_state: dict[str, Any] | None,
    optuna_trial_id: int | None,
) -> TrialResult:
    """Create a failed :class:`TrialResult` instance."""

    logger.warning("Trial %s failed: %s", trial_id, error)

    metadata: dict[str, Any] = (
        {"optuna_trial_id": optuna_trial_id} if optuna_trial_id is not None else {}
    )
    metrics: dict[str, float] = {}
    if progress_state:
        evaluated = progress_state.get("evaluated", 0)
        metadata["examples_attempted"] = int(evaluated)
        total_cost_value = progress_state.get("total_cost")
        if total_cost_value is not None:
            try:
                cost_value = float(total_cost_value)
                metrics["total_cost"] = cost_value
                metadata["total_example_cost"] = cost_value
            except (TypeError, ValueError) as e:
                logger.warning(
                    "Failed to convert total_cost to float for failed trial %s: %s (value: %s)",
                    trial_id,
                    e,
                    total_cost_value,
                )

    return TrialResult(
        trial_id=trial_id,
        config=evaluation_config,
        metrics=metrics,
        status=TrialStatus.FAILED,
        duration=duration,
        timestamp=datetime.now(UTC),
        error_message=str(error),
        metadata=metadata,
    )


__all__ = [
    "build_success_result",
    "build_pruned_result",
    "build_failed_result",
]
