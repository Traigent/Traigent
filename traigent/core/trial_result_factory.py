"""Utilities for constructing trial results."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Any

from traigent.api.types import (
    ComparabilityInfo,
    MetricCoverage,
    TrialError,
    TrialResult,
    TrialStatus,
)
from traigent.security.redaction import redact_sensitive_data, redact_sensitive_text
from traigent.utils.exceptions import TrialPrunedError
from traigent.utils.logging import get_logger
from traigent.utils.objectives import classify_objective

logger = get_logger(__name__)


def _to_redactable_payload(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = None
        if payload is not None:
            return payload
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)  # type: ignore[arg-type]
    return value


def _to_redactable_payloads(values: list[Any]) -> list[Any]:
    return [_to_redactable_payload(value) for value in values]


def _coerce_non_negative_int(value: Any) -> int:
    """Best-effort integer conversion with non-negative fallback."""
    if value is None or isinstance(value, bool):
        return 0
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _infer_total_examples(eval_result: Any) -> tuple[int, Any, bool]:
    """Infer total example count and whether detailed example results are present."""
    total_examples = _coerce_non_negative_int(getattr(eval_result, "total_examples", 0))
    example_results = getattr(eval_result, "example_results", None)
    has_detailed_example_results = (
        isinstance(example_results, list) and len(example_results) > 0
    )
    if isinstance(example_results, list) and total_examples <= 0:
        total_examples = len(example_results)
    return total_examples, example_results, has_detailed_example_results


def _collect_per_metric_counts_from_examples(
    example_results: list[Any],
) -> dict[str, int]:
    """Count how often each numeric metric appears across detailed example results."""
    per_metric_counts: dict[str, int] = {}
    for example_result in example_results:
        metrics = getattr(example_result, "metrics", {}) or {}
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            per_metric_counts[metric_name] = per_metric_counts.get(metric_name, 0) + 1
    return per_metric_counts


def _collect_per_metric_counts_from_aggregate(
    eval_result: Any,
    total_examples: int,
) -> dict[str, int]:
    """Assume aggregate numeric metrics apply to all evaluated examples."""
    per_metric_counts: dict[str, int] = {}
    agg_metrics = getattr(eval_result, "metrics", {}) or {}
    if not isinstance(agg_metrics, dict) or total_examples <= 0:
        return per_metric_counts
    for metric_name, value in agg_metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        per_metric_counts[metric_name] = total_examples
    return per_metric_counts


def _collect_per_metric_counts(
    eval_result: Any,
    total_examples: int,
    example_results: Any,
    has_detailed_example_results: bool,
) -> dict[str, int]:
    """Collect per-metric coverage counts from detailed or aggregate outputs."""
    if has_detailed_example_results and isinstance(example_results, list):
        return _collect_per_metric_counts_from_examples(example_results)
    return _collect_per_metric_counts_from_aggregate(eval_result, total_examples)


def _select_primary_objective(per_metric_counts: dict[str, int]) -> str:
    """Choose a reasonable primary objective from available metric counts."""
    for preferred in ("accuracy", "score"):
        if preferred in per_metric_counts:
            return preferred

    accuracy_like = sorted(
        metric_name
        for metric_name in per_metric_counts
        if metric_name.endswith("_accuracy")
    )
    if accuracy_like:
        return accuracy_like[0]

    for preferred in ("total_cost", "cost", "latency", "response_time_ms"):
        if preferred in per_metric_counts:
            return preferred

    if per_metric_counts:
        return sorted(per_metric_counts)[0]
    return "unknown"


def _infer_evaluation_mode(
    has_detailed_example_results: bool,
    per_metric_counts: dict[str, int],
) -> str:
    """Infer whether results came from evaluated or execute-only mode."""
    if has_detailed_example_results:
        return "evaluated"
    if not per_metric_counts:
        return "unknown"
    objective_classes = {
        classify_objective(metric_name) for metric_name in per_metric_counts
    }
    return "execute_only" if objective_classes == {"operational"} else "evaluated"


def _build_warning_codes(
    total_examples: int,
    examples_with_primary_metric: int,
    coverage_ratio: float,
) -> list[str]:
    """Derive comparability warning codes from coverage information."""
    if total_examples <= 0:
        return ["MCI-001"]
    if examples_with_primary_metric <= 0:
        return ["MCI-004"]
    if coverage_ratio < 1.0:
        return ["MCI-002"]
    return []


def _build_fallback_comparability(eval_result: Any) -> dict[str, Any]:
    """Build comparability metadata when evaluator did not provide one."""
    total_examples, example_results, has_detailed_example_results = (
        _infer_total_examples(eval_result)
    )
    per_metric_counts = _collect_per_metric_counts(
        eval_result,
        total_examples,
        example_results,
        has_detailed_example_results,
    )

    per_metric_coverage = {
        metric_name: MetricCoverage(
            present=count,
            total=total_examples,
            ratio=(count / total_examples) if total_examples > 0 else 0.0,
        )
        for metric_name, count in per_metric_counts.items()
    }

    primary_objective = _select_primary_objective(per_metric_counts)
    evaluation_mode = _infer_evaluation_mode(
        has_detailed_example_results,
        per_metric_counts,
    )

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
    warning_codes = _build_warning_codes(
        total_examples,
        examples_with_primary_metric,
        coverage_ratio,
    )

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


def _build_success_trial_metadata(
    eval_result: Any,
    examples_attempted: int | None,
    total_cost: float | None,
) -> dict[str, Any]:
    """Build metadata for a successful trial result."""
    evaluation_result_payload = (
        eval_result.to_dict()
        if hasattr(eval_result, "to_dict") and callable(eval_result.to_dict)
        else eval_result
    )
    trial_metadata: dict[str, Any] = {
        "success_rate": getattr(eval_result, "success_rate", None),
        "has_errors": getattr(eval_result, "has_errors", None),
        "output_count": (
            len(eval_result.outputs)
            if getattr(eval_result, "outputs", None) is not None
            else 0
        ),
        "evaluation_result": redact_sensitive_data(evaluation_result_payload),
    }

    example_results = getattr(eval_result, "example_results", None)
    if isinstance(example_results, list) and example_results:
        trial_metadata["example_results"] = redact_sensitive_data(
            _to_redactable_payloads(example_results)
        )

    if examples_attempted is not None:
        trial_metadata["examples_attempted"] = int(examples_attempted)

    if total_cost is not None:
        try:
            trial_metadata["total_example_cost"] = float(total_cost)
        except (TypeError, ValueError):
            pass

    return trial_metadata


def _set_metric_if_convertible(
    metrics: dict[str, Any],
    key: str,
    raw_value: Any,
    converter: type[int] | type[float],
    trial_id: str,
) -> None:
    """Set a metric after conversion, logging instead of raising on invalid values."""
    try:
        metrics.setdefault(key, converter(raw_value))
    except (TypeError, ValueError) as error:
        logger.warning(
            "Failed to convert %s to %s for trial %s: %s (value: %s)",
            key,
            converter.__name__,
            trial_id,
            error,
            raw_value,
        )


def _extract_success_comparability(eval_result: Any) -> dict[str, Any]:
    """Resolve comparability metadata from evaluator output or fallback synthesis."""
    comparability_payload = _validated_comparability_payload(
        getattr(eval_result, "comparability", None)
    )
    if comparability_payload is not None:
        return comparability_payload

    summary_stats = getattr(eval_result, "summary_stats", None)
    if isinstance(summary_stats, dict):
        metadata = summary_stats.get("metadata")
        if isinstance(metadata, dict):
            summary_comp = _validated_comparability_payload(
                metadata.get("comparability")
            )
            if summary_comp is not None:
                return summary_comp

    return _build_fallback_comparability(eval_result)


def _validated_comparability_payload(
    comparability_payload: Any,
) -> dict[str, Any] | None:
    """Return comparability payloads only when ranking_eligible is absent or boolean."""
    if not isinstance(comparability_payload, dict):
        return None

    ranking_eligible = comparability_payload.get("ranking_eligible")
    if "ranking_eligible" in comparability_payload and not isinstance(
        ranking_eligible, bool
    ):
        logger.warning(
            "Ignoring comparability payload with non-boolean ranking_eligible: %r",
            ranking_eligible,
        )
        return None

    return comparability_payload


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
    trial_metadata = _build_success_trial_metadata(
        eval_result,
        examples_attempted,
        total_cost,
    )

    trial_result = TrialResult(
        trial_id=trial_id,
        config=copy.deepcopy(evaluation_config),
        metrics=getattr(eval_result, "metrics", {}) or {},
        status=TrialStatus.COMPLETED,
        duration=duration,
        timestamp=datetime.now(UTC),
        metadata=trial_metadata,
    )

    if optuna_trial_id is not None:
        trial_result.metadata.setdefault("optuna_trial_id", optuna_trial_id)

    if examples_attempted is not None:
        _set_metric_if_convertible(
            trial_result.metrics,
            "examples_attempted",
            examples_attempted,
            int,
            trial_id,
        )

    if total_cost is not None:
        _set_metric_if_convertible(
            trial_result.metrics,
            "total_cost",
            total_cost,
            float,
            trial_id,
        )

    if getattr(eval_result, "summary_stats", None):
        trial_result.summary_stats = redact_sensitive_data(  # type: ignore[attr-defined]
            eval_result.summary_stats
        )

    trial_result.metadata["comparability"] = _extract_success_comparability(eval_result)

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
        metadata["example_results"] = redact_sensitive_data(
            _to_redactable_payloads(list(prune_error.example_results))
        )
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
        config=copy.deepcopy(evaluation_config),
        metrics=metrics,
        status=TrialStatus.PRUNED,
        duration=duration,
        timestamp=datetime.now(UTC),
        metadata=redact_sensitive_data(metadata),
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
    error_details = TrialError.from_exception(error, config=evaluation_config)
    error_details.message = redact_sensitive_text(error_details.message)
    error_details.traceback = redact_sensitive_text(error_details.traceback)

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
        config=copy.deepcopy(evaluation_config),
        metrics=metrics,
        status=TrialStatus.FAILED,
        duration=duration,
        timestamp=datetime.now(UTC),
        error_message=redact_sensitive_text(str(error)),
        metadata=redact_sensitive_data(metadata),
        error=error_details,
    )


__all__ = [
    "build_success_result",
    "build_pruned_result",
    "build_failed_result",
]
