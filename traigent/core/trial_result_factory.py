"""Utilities for constructing trial results."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from traigent.api.types import (
    ComparabilityInfo,
    MetricCoverage,
    TrialResult,
    TrialStatus,
)
from traigent.utils.exceptions import TrialPrunedError
from traigent.utils.logging import get_logger
from traigent.utils.objectives import classify_objective

logger = get_logger(__name__)


def _coerce_non_negative_int(value: Any) -> int:
    """Best-effort integer conversion with non-negative fallback."""
    if value is None or isinstance(value, bool):
        return 0
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _build_fallback_comparability(eval_result: Any) -> dict[str, Any]:
    """Build comparability metadata when evaluator did not provide one."""
    total_examples = _coerce_non_negative_int(getattr(eval_result, "total_examples", 0))
    example_results = getattr(eval_result, "example_results", None)
    if isinstance(example_results, list) and total_examples <= 0:
        total_examples = len(example_results)

    per_metric_counts: dict[str, int] = {}
    has_detailed_example_results = (
        isinstance(example_results, list) and len(example_results) > 0
    )
    if has_detailed_example_results and total_examples > 0 and example_results:
        for example_result in example_results:
            metrics = getattr(example_result, "metrics", {}) or {}
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    per_metric_counts[metric_name] = (
                        per_metric_counts.get(metric_name, 0) + 1
                    )
    else:
        agg_metrics = getattr(eval_result, "metrics", {}) or {}
        if isinstance(agg_metrics, dict) and total_examples > 0:
            for metric_name, value in agg_metrics.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    per_metric_counts[metric_name] = total_examples

    per_metric_coverage = {
        metric_name: MetricCoverage(
            present=count,
            total=total_examples,
            ratio=(count / total_examples) if total_examples > 0 else 0.0,
        )
        for metric_name, count in per_metric_counts.items()
    }

    primary_objective = "unknown"
    for preferred in ("accuracy", "score"):
        if preferred in per_metric_counts:
            primary_objective = preferred
            break
    if primary_objective == "unknown":
        accuracy_like = sorted(
            metric_name
            for metric_name in per_metric_counts
            if metric_name.endswith("_accuracy")
        )
        if accuracy_like:
            primary_objective = accuracy_like[0]
    if primary_objective == "unknown":
        for preferred in ("total_cost", "cost", "latency", "response_time_ms"):
            if preferred in per_metric_counts:
                primary_objective = preferred
                break
    if primary_objective == "unknown" and per_metric_counts:
        primary_objective = sorted(per_metric_counts)[0]

    if has_detailed_example_results:
        evaluation_mode = "evaluated"
    elif per_metric_counts:
        objective_classes = {
            classify_objective(metric_name) for metric_name in per_metric_counts
        }
        evaluation_mode = (
            "execute_only" if objective_classes == {"operational"} else "evaluated"
        )
    else:
        evaluation_mode = "unknown"

    examples_with_primary_metric = per_metric_counts.get(primary_objective, 0)
    coverage_ratio = (
        examples_with_primary_metric / total_examples
        if total_examples > 0 and examples_with_primary_metric > 0
        else 0.0
    )
    ranking_eligible = (
        total_examples > 0
        and examples_with_primary_metric == total_examples
        and (
            classify_objective(primary_objective) != "quality"
            or evaluation_mode == "evaluated"
        )
    )
    if total_examples <= 0:
        warning_codes = ["MCI-001"]
    elif examples_with_primary_metric <= 0:
        warning_codes = ["MCI-004"]
    elif coverage_ratio < 1.0:
        warning_codes = ["MCI-002"]
    else:
        warning_codes = []

    return ComparabilityInfo(
        primary_objective=primary_objective,
        evaluation_mode=evaluation_mode,
        total_examples=total_examples,
        examples_with_primary_metric=examples_with_primary_metric,
        coverage_ratio=coverage_ratio,
        derivation_path="none",
        ranking_eligible=ranking_eligible,
        warning_codes=warning_codes,
        per_metric_coverage=per_metric_coverage,
        missing_example_ids=[],
    ).to_dict()


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

    comparability_payload = getattr(eval_result, "comparability", None)
    if not isinstance(comparability_payload, dict):
        summary_stats = getattr(eval_result, "summary_stats", None)
        if isinstance(summary_stats, dict):
            metadata = summary_stats.get("metadata")
            if isinstance(metadata, dict):
                summary_comp = metadata.get("comparability")
                if isinstance(summary_comp, dict):
                    comparability_payload = summary_comp
    if not isinstance(comparability_payload, dict):
        comparability_payload = _build_fallback_comparability(eval_result)
    trial_result.metadata["comparability"] = comparability_payload

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
    """Create a pruned :class:`TrialResult` instance.

    Captures partial example results from the TrialPrunedError if available,
    ensuring that metrics from evaluated examples are preserved for pruned trials.
    """

    metadata: dict[str, Any] = {
        "pruned": True,
        "pruned_step": prune_error.step,
    }
    if optuna_trial_id is not None:
        metadata["optuna_trial_id"] = optuna_trial_id

    pruned_example_count = len(prune_error.example_results or [])
    progress_total_examples = _coerce_non_negative_int(
        (progress_state or {}).get("total_examples")
    )
    comparability_total_examples = max(pruned_example_count, progress_total_examples)

    metadata["comparability"] = ComparabilityInfo(
        primary_objective="unknown",
        evaluation_mode="unknown",
        total_examples=comparability_total_examples,
        examples_with_primary_metric=0,
        coverage_ratio=0.0,
        derivation_path="none",
        ranking_eligible=False,
        warning_codes=["MCI-007"],
        per_metric_coverage={},
        missing_example_ids=[],
    ).to_dict()

    # Include partial example_results from the pruned trial
    # These are captured by the evaluator before raising TrialPrunedError
    if prune_error.example_results:
        metadata["example_results"] = prune_error.example_results
        logger.info(
            "📊 Captured %d partial example results for pruned trial %s",
            len(prune_error.example_results),
            trial_id,
        )
    else:
        logger.warning(
            "⚠️ No example_results in TrialPrunedError for pruned trial %s (step=%s)",
            trial_id,
            prune_error.step,
        )

    metrics: dict[str, float] = {}
    if progress_state:
        logger.info(
            "📊 Pruned trial %s progress_state: evaluated=%s, total=%s, correct_sum=%s",
            trial_id,
            progress_state.get("evaluated"),
            progress_state.get("total_examples"),
            progress_state.get("correct_sum"),
        )
        evaluated = int(progress_state.get("evaluated", 0))
        total_examples = int(progress_state.get("total_examples", 0))
        if total_examples > 0:
            evaluated = min(evaluated, total_examples)

        if evaluated:
            correct_sum = float(progress_state.get("correct_sum", 0.0))
            metrics["accuracy"] = correct_sum / evaluated
            metadata["examples_attempted"] = evaluated
            # Also add to metrics for backend submission
            metrics["examples_attempted"] = float(evaluated)

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
    metadata["comparability"] = ComparabilityInfo(
        primary_objective="unknown",
        evaluation_mode="unknown",
        total_examples=0,
        examples_with_primary_metric=0,
        coverage_ratio=0.0,
        derivation_path="none",
        ranking_eligible=False,
        warning_codes=["MCI-007"],
        per_metric_coverage={},
        missing_example_ids=[],
    ).to_dict()
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
