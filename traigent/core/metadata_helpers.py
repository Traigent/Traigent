"""Metadata construction helpers for trial and session results."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

import json
from typing import Any

from traigent.api.types import OptimizationResult, TrialResult
from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def merge_run_metrics_into_session_summary(result: OptimizationResult) -> None:
    """Merge optimization result metrics into session summary.

    Ensures session_summary includes aggregated metrics from the optimization run.
    This mutates result.metadata["session_summary"]["metrics"] in place.

    Args:
        result: OptimizationResult with metadata containing session_summary

    Side Effects:
        Mutates result.metadata["session_summary"]["metrics"] if present
    """
    try:
        if (
            hasattr(result, "metadata")
            and result.metadata
            and "session_summary" in result.metadata
        ):
            session_sum = result.metadata["session_summary"]
            if isinstance(session_sum, dict):
                session_metrics = session_sum.get("metrics", {})
                if not isinstance(session_metrics, dict):
                    session_metrics = {}
                if isinstance(result.metrics, dict):
                    for key, value in result.metrics.items():
                        if not isinstance(value, (int, float)):
                            continue
                        prefixed_key = key if key.startswith("run_") else f"run_{key}"
                        session_metrics[prefixed_key] = value
                session_sum["metrics"] = session_metrics
                result.metadata["session_summary"] = session_sum
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug(
            "Failed to merge run metrics into session_summary: %s",
            exc,
        )


def build_backend_metadata(
    trial_result: TrialResult,
    primary_objective: str,
    traigent_config: TraigentConfig,
) -> dict[str, Any]:
    """Build metadata dictionary for backend trial submission.

    Constructs comprehensive metadata payload including:
    - Basic trial info (duration, timestamp, execution mode)
    - Summary statistics (if available and not in cloud mode)
    - Per-example measures (respecting privacy settings)
    - Additional metrics beyond primary objective

    Args:
        trial_result: Completed trial result
        primary_objective: Primary optimization objective name
        traigent_config: Global configuration for execution mode and privacy

    Returns:
        Metadata dict ready for backend submission
    """
    trial_metadata: dict[str, Any] = {
        "duration": trial_result.duration,
        "trial_id": trial_result.trial_id,
        "execution_mode": traigent_config.execution_mode,
    }

    if not traigent_config.minimal_logging:
        trial_metadata["timestamp"] = trial_result.timestamp.isoformat()
        trial_metadata["all_metrics"] = trial_result.metrics

    # Add summary stats for non-cloud modes
    mode_enum = traigent_config.execution_mode_enum

    summary_stats_available = (
        hasattr(trial_result, "summary_stats") and trial_result.summary_stats
    )
    if summary_stats_available and mode_enum is not ExecutionMode.CLOUD:
        enhanced_summary_stats = trial_result.summary_stats.copy()
        if "metadata" not in enhanced_summary_stats:
            enhanced_summary_stats["metadata"] = {}
        enhanced_summary_stats["metadata"]["aggregation_level"] = "trial"
        enhanced_summary_stats["metadata"]["sdk_version"] = "2.0.0"
        trial_metadata["summary_stats"] = enhanced_summary_stats
        logger.debug(
            "Adding summary_stats for %s mode with aggregation_level=trial",
            mode_enum.value,
        )

    privacy_on = getattr(traigent_config, "privacy_enabled", False) or (
        mode_enum is ExecutionMode.PRIVACY
    )

    example_results = trial_result.metadata.get("example_results")

    # Build per-example measures based on privacy settings
    include_full_measures = not privacy_on and mode_enum in {
        ExecutionMode.EDGE_ANALYTICS,
        ExecutionMode.STANDARD,
        ExecutionMode.HYBRID,
    }
    if include_full_measures and example_results:
        measures = _build_measures_full(example_results, primary_objective)
        trial_metadata["measures"] = measures
        logger.debug(
            "Collected %s per-example MeasureResults for %s mode",
            len(measures),
            traigent_config.execution_mode,
        )
        if measures:
            logger.debug("First measure content: %s", measures[0])
            try:
                logger.debug(
                    "All measures: %s",
                    json.dumps(measures, indent=2),
                )
            except (TypeError, ValueError):
                pass

    elif privacy_on and example_results:
        measures = _build_measures_privacy(example_results, primary_objective)
        trial_metadata["measures"] = measures
        logger.debug(
            "Using sanitized MeasureResults for privacy mode: %s examples",
            len(measures),
        )
    else:
        trial_metadata.pop("measures", None)

    # Add additional metrics
    if trial_result.metrics:
        for metric_key, metric_value in trial_result.metrics.items():
            if metric_key != primary_objective:
                trial_metadata[metric_key] = metric_value

    # Add aggregation summary to summary_stats
    try:
        if "summary_stats" in trial_metadata:
            trial_aggregation_summary = {
                "primary_objective": primary_objective,
                "metrics": trial_result.metrics or {},
                "sanitized": True,
            }
            summary_metadata = trial_metadata["summary_stats"].setdefault(
                "metadata", {}
            )
            summary_metadata["aggregation_summary"] = trial_aggregation_summary
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("Failed to add aggregation summary (trial): %s", exc)

    # Remove example_results in privacy mode
    if privacy_on and "example_results" in trial_metadata:
        trial_metadata.pop("example_results", None)

    return trial_metadata


def _build_measures_full(
    example_results: list[Any],
    primary_objective: str,
) -> list[dict[str, Any]]:
    """Build full per-example measures (non-privacy mode).

    Args:
        example_results: List of example evaluation results
        primary_objective: Primary optimization objective name

    Returns:
        List of measure dicts with all available metrics
    """
    measures = []
    for idx, example_result in enumerate(example_results):
        measure_result: dict[str, Any] = {}

        # Extract score from metrics
        metrics = getattr(example_result, "metrics", {}) or {}
        example_score: float | None = None
        if metrics:
            logger.debug(
                "Example %s metrics: %s",
                idx,
                metrics,
            )
            for candidate in (
                primary_objective,
                "score",
                "accuracy",
            ):
                if candidate and candidate in metrics:
                    value = metrics[candidate]
                    if value is not None:
                        example_score = float(value)
                        break

            # Add all scalar metrics
            for metric_key, metric_value in metrics.items():
                if isinstance(metric_value, (int, float, str)) or metric_value is None:
                    measure_result[metric_key] = metric_value

        # Fallback score calculation
        if example_score is None:
            expected = getattr(example_result, "expected_output", None)
            actual = getattr(example_result, "actual_output", None)
            if expected is not None and actual is not None:
                example_score = 1.0 if actual == expected else 0.0

        if example_score is not None:
            measure_result["score"] = float(example_score)

        # Add execution time
        if hasattr(example_result, "execution_time"):
            measure_result["response_time"] = example_result.execution_time

        logger.debug(
            "Measure %s being sent: %s",
            idx,
            measure_result,
        )
        measures.append(measure_result)

    return measures


def _build_measures_privacy(
    example_results: list[Any],
    primary_objective: str,
) -> list[dict[str, Any]]:
    """Build sanitized per-example measures (privacy mode).

    Only includes score, response time, tokens, and cost metrics.

    Args:
        example_results: List of example evaluation results
        primary_objective: Primary optimization objective name

    Returns:
        List of sanitized measure dicts
    """
    measures = []
    for example_result in example_results:
        measure_result: dict[str, Any] = {}

        # Extract score from metrics
        metrics = getattr(example_result, "metrics", {}) or {}
        example_score: float | None = None
        if metrics:
            for candidate in (
                primary_objective,
                "score",
                "accuracy",
            ):
                if candidate and candidate in metrics:
                    value = metrics[candidate]
                    if value is not None:
                        example_score = float(value)
                        break

        # Fallback score calculation
        if example_score is None:
            expected = getattr(example_result, "expected_output", None)
            actual = getattr(example_result, "actual_output", None)
            if expected is not None and actual is not None:
                example_score = 1.0 if actual == expected else 0.0

        if example_score is not None:
            measure_result["score"] = float(example_score)

        # Add execution time
        if hasattr(example_result, "execution_time"):
            measure_result["response_time"] = example_result.execution_time

        # Add token metrics (privacy-safe)
        if metrics:
            for token_key in ("input_tokens", "output_tokens", "total_tokens"):
                if token_key in metrics:
                    measure_result[token_key] = metrics[token_key]

            # Add cost metrics (privacy-safe)
            for cost_key in ("input_cost", "output_cost", "total_cost"):
                if cost_key in metrics:
                    measure_result[cost_key] = metrics[cost_key]

        measures.append(measure_result)

    return measures
