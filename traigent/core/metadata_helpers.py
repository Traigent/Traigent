"""Metadata construction helpers for trial and session results."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

import json
from typing import Any

from traigent.api.types import OptimizationResult, TrialResult
from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.utils.example_id import compute_dataset_hash, generate_stable_example_id
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def _validate_example_id(measure: dict[str, Any], example_index: int) -> None:
    """Validate example_id field in measure dict."""
    if "example_id" not in measure:
        raise ValueError(
            f"Example {example_index}: Missing required 'example_id' field"
        )
    if not isinstance(measure["example_id"], str):
        raise ValueError(
            f"Example {example_index}: 'example_id' must be string, "
            f"got {type(measure['example_id']).__name__}"
        )


def _validate_metrics_field(measure: dict[str, Any], example_index: int) -> dict:
    """Validate metrics field exists and is a dict. Returns metrics dict."""
    if "metrics" not in measure:
        raise ValueError(f"Example {example_index}: Missing required 'metrics' field")
    if not isinstance(measure["metrics"], dict):
        raise ValueError(
            f"Example {example_index}: 'metrics' must be dict, "
            f"got {type(measure['metrics']).__name__}"
        )
    return measure["metrics"]


def _validate_metric_entry(key: str, value: Any, example_index: int) -> None:
    """Validate a single metric key-value pair."""
    if not key.isidentifier():
        raise ValueError(
            f"Example {example_index}: Invalid metric key '{key}', "
            f"must be Python identifier (^[a-zA-Z_][a-zA-Z0-9_]*$)"
        )
    if value is not None and not isinstance(value, (int, float)):
        raise ValueError(
            f"Example {example_index}: Metric '{key}' has invalid type "
            f"{type(value).__name__}, must be int, float, or None"
        )


def _validate_measure_dict(measure: dict[str, Any], example_index: int) -> None:
    """Validate measure against nested MeasuresDict format from TraigentSchema.

    Expected nested format:
        {
            "example_id": "ex_a3f4b2c8_0",
            "metrics": {"score": 0.85, "cost": 0.05, ...}
        }

    Constraints:
    1. Must have "example_id" (string) at top level
    2. Must have "metrics" (dict) at top level
    3. metrics keys must be valid Python identifiers (^[a-zA-Z_][a-zA-Z0-9_]*$)
    4. metrics values must be numeric (int, float) or None
    5. Max 50 keys in metrics

    Args:
        measure: Measure dict to validate (nested format)
        example_index: Index of the example (for error messages)

    Raises:
        ValueError: If validation fails
    """
    _validate_example_id(measure, example_index)
    metrics = _validate_metrics_field(measure, example_index)

    if len(metrics) > 50:
        raise ValueError(
            f"Example {example_index}: metrics has {len(metrics)} keys, "
            f"exceeds limit of 50"
        )

    for key, value in metrics.items():
        _validate_metric_entry(key, value, example_index)


def _get_session_summary(result: OptimizationResult) -> dict[str, Any] | None:
    """Extract session_summary dict from result metadata if available."""
    if not hasattr(result, "metadata") or not result.metadata:
        return None
    if "session_summary" not in result.metadata:
        return None
    session_sum = result.metadata["session_summary"]
    return session_sum if isinstance(session_sum, dict) else None


def _merge_metrics_with_prefix(
    session_metrics: dict[str, Any], result_metrics: dict[str, Any]
) -> None:
    """Merge result metrics into session metrics with 'run_' prefix."""
    for key, value in result_metrics.items():
        if not isinstance(value, (int, float)):
            continue
        prefixed_key = key if key.startswith("run_") else f"run_{key}"
        session_metrics[prefixed_key] = value


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
        session_sum = _get_session_summary(result)
        if session_sum is None:
            return

        session_metrics = session_sum.get("metrics", {})
        if not isinstance(session_metrics, dict):
            session_metrics = {}

        if isinstance(result.metrics, dict):
            _merge_metrics_with_prefix(session_metrics, result.metrics)

        session_sum["metrics"] = session_metrics
        result.metadata["session_summary"] = session_sum
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug(
            "Failed to merge run metrics into session_summary: %s",
            exc,
        )


def _add_summary_stats(
    trial_metadata: dict[str, Any],
    trial_result: TrialResult,
    mode_enum: ExecutionMode,
) -> None:
    """Add summary stats to trial metadata for non-cloud modes."""
    summary_stats_available = (
        hasattr(trial_result, "summary_stats") and trial_result.summary_stats  # type: ignore[attr-defined]
    )
    if not summary_stats_available or mode_enum is ExecutionMode.CLOUD:
        return

    enhanced_summary_stats = trial_result.summary_stats.copy()  # type: ignore[attr-defined]
    if "metadata" not in enhanced_summary_stats:
        enhanced_summary_stats["metadata"] = {}
    enhanced_summary_stats["metadata"]["aggregation_level"] = "trial"
    enhanced_summary_stats["metadata"]["sdk_version"] = "2.0.0"
    trial_metadata["summary_stats"] = enhanced_summary_stats
    logger.debug(
        "Adding summary_stats for %s mode with aggregation_level=trial",
        mode_enum.value,
    )


def _add_measures_to_metadata(
    trial_metadata: dict[str, Any],
    example_results: list[Any] | None,
    primary_objective: str,
    dataset_name: str,
    privacy_on: bool,
    mode_enum: ExecutionMode,
    execution_mode: str,
) -> None:
    """Add measures to metadata based on privacy settings."""
    if not example_results:
        trial_metadata.pop("measures", None)
        return

    include_full_measures = not privacy_on and mode_enum in {
        ExecutionMode.EDGE_ANALYTICS,
        ExecutionMode.STANDARD,
        ExecutionMode.HYBRID,
    }

    if include_full_measures:
        measures = _build_measures_full(
            example_results, primary_objective, dataset_name
        )
        trial_metadata["measures"] = measures
        logger.debug(
            "Collected %s per-example MeasureResults for %s mode",
            len(measures),
            execution_mode,
        )
        _log_measures_debug(measures)
    elif privacy_on:
        measures = _build_measures_privacy(
            example_results, primary_objective, dataset_name
        )
        trial_metadata["measures"] = measures
        logger.debug(
            "Using sanitized MeasureResults for privacy mode: %s examples",
            len(measures),
        )
    else:
        trial_metadata.pop("measures", None)


def _log_measures_debug(measures: list[dict[str, Any]]) -> None:
    """Log measures content for debugging."""
    if not measures:
        return
    logger.debug("First measure content: %s", measures[0])
    try:
        logger.debug("All measures: %s", json.dumps(measures, indent=2))
    except (TypeError, ValueError):
        pass


def _add_aggregation_summary(
    trial_metadata: dict[str, Any],
    primary_objective: str,
    metrics: dict[str, Any] | None,
) -> None:
    """Add aggregation summary to summary_stats."""
    try:
        if "summary_stats" not in trial_metadata:
            return
        trial_aggregation_summary = {
            "primary_objective": primary_objective,
            "metrics": metrics or {},
            "sanitized": True,
        }
        summary_metadata = trial_metadata["summary_stats"].setdefault("metadata", {})
        summary_metadata["aggregation_summary"] = trial_aggregation_summary
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("Failed to add aggregation summary (trial): %s", exc)


def build_backend_metadata(
    trial_result: TrialResult,
    primary_objective: str,
    traigent_config: TraigentConfig,
    dataset_name: str = "dataset",
    content_scores: dict[str, dict[int, float]] | None = None,
) -> dict[str, Any]:
    """Build metadata dictionary for backend trial submission.

    Constructs comprehensive metadata payload including:
    - Basic trial info (duration, timestamp, execution mode)
    - Summary statistics (if available and not in cloud mode)
    - Per-example measures (respecting privacy settings)
    - Stable example IDs (format: ex_{hash}_{index})
    - Additional metrics beyond primary objective

    Args:
        trial_result: Completed trial result
        primary_objective: Primary optimization objective name
        traigent_config: Global configuration for execution mode and privacy
        dataset_name: Name of the dataset (used for stable example ID generation)
        content_scores: Deprecated and ignored. Content analytics are uploaded
            once per run via the dedicated feature-upload path.

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

    mode_enum = traigent_config.execution_mode_enum
    _add_summary_stats(trial_metadata, trial_result, mode_enum)

    comparability_payload = trial_result.metadata.get("comparability")
    if isinstance(comparability_payload, dict):
        trial_metadata["comparability"] = comparability_payload

    privacy_on = getattr(traigent_config, "privacy_enabled", False) or (
        mode_enum is ExecutionMode.PRIVACY
    )
    example_results = trial_result.metadata.get("example_results")

    _add_measures_to_metadata(
        trial_metadata,
        example_results,
        primary_objective,
        dataset_name,
        privacy_on,
        mode_enum,
        traigent_config.execution_mode,
    )

    # Add additional metrics (excluding primary objective)
    if trial_result.metrics:
        for metric_key, metric_value in trial_result.metrics.items():
            if metric_key != primary_objective:
                trial_metadata[metric_key] = metric_value

    _add_aggregation_summary(trial_metadata, primary_objective, trial_result.metrics)

    # Remove example_results in privacy mode
    if privacy_on and "example_results" in trial_metadata:
        trial_metadata.pop("example_results", None)

    return trial_metadata


def _extract_score_from_metrics(
    eval_metrics: dict[str, Any], primary_objective: str
) -> float | None:
    """Extract score from evaluation metrics using priority order."""
    for candidate in (primary_objective, "score", "accuracy"):
        if candidate and candidate in eval_metrics:
            value = eval_metrics[candidate]
            if value is not None:
                return float(value)
    return None


def _calculate_fallback_score(example_result: Any) -> float | None:
    """Calculate fallback score from expected vs actual output."""
    expected = getattr(example_result, "expected_output", None)
    actual = getattr(example_result, "actual_output", None)
    if expected is not None and actual is not None:
        return 1.0 if actual == expected else 0.0
    return None


def _build_single_measure_full(
    example_result: Any,
    idx: int,
    dataset_hash: str,
    primary_objective: str,
) -> dict[str, Any]:
    """Build a single full measure for non-privacy mode."""
    example_id = generate_stable_example_id(dataset_hash, idx)
    metrics_dict: dict[str, Any] = {}
    eval_metrics = getattr(example_result, "metrics", {}) or {}

    if eval_metrics:
        logger.debug("Example %s metrics: %s", idx, eval_metrics)
        # Add all scalar metrics (numeric only per MeasuresDict constraints)
        for metric_key, metric_value in eval_metrics.items():
            if isinstance(metric_value, (int, float)) or metric_value is None:
                metrics_dict[metric_key] = metric_value

    # Extract or calculate score
    example_score = _extract_score_from_metrics(eval_metrics, primary_objective)
    if example_score is None:
        example_score = _calculate_fallback_score(example_result)
    if example_score is not None:
        metrics_dict["score"] = float(example_score)

    # Add execution time
    if hasattr(example_result, "execution_time"):
        metrics_dict["response_time"] = example_result.execution_time

    measure_result = {"example_id": example_id, "metrics": metrics_dict}
    _validate_measure_dict(measure_result, idx)
    logger.debug("Measure %s being sent: %s", idx, measure_result)
    return measure_result


def _build_measures_full(
    example_results: list[Any],
    primary_objective: str,
    dataset_name: str = "dataset",
) -> list[dict[str, Any]]:
    """Build full per-example measures with stable IDs (non-privacy mode).

    Returns nested format where example_id is at top level and all numeric
    metrics are in a 'metrics' sub-object:
        {
            "example_id": "ex_a3f4b2c8_0",
            "metrics": {"score": 0.85, "cost": 0.05, ...}
        }

    Args:
        example_results: List of example evaluation results
        primary_objective: Primary optimization objective name
        dataset_name: Name of the dataset (for stable example ID generation)
    Returns:
        List of nested measure dicts with example_id and metrics sub-object

    Raises:
        ValueError: If measures violate MeasuresDict constraints
    """
    dataset_hash = compute_dataset_hash(dataset_name)
    return [
        _build_single_measure_full(example_result, idx, dataset_hash, primary_objective)
        for idx, example_result in enumerate(example_results)
    ]


_PRIVACY_TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")
_PRIVACY_COST_KEYS = ("input_cost", "output_cost", "total_cost")


def _add_privacy_safe_metrics(
    metrics_dict: dict[str, Any], eval_metrics: dict[str, Any]
) -> None:
    """Add token and cost metrics which are privacy-safe."""
    for token_key in _PRIVACY_TOKEN_KEYS:
        if token_key in eval_metrics:
            metrics_dict[token_key] = eval_metrics[token_key]
    for cost_key in _PRIVACY_COST_KEYS:
        if cost_key in eval_metrics:
            metrics_dict[cost_key] = eval_metrics[cost_key]


def _build_single_measure_privacy(
    example_result: Any,
    idx: int,
    dataset_hash: str,
    primary_objective: str,
) -> dict[str, Any]:
    """Build a single sanitized measure for privacy mode."""
    example_id = generate_stable_example_id(dataset_hash, idx)
    metrics_dict: dict[str, Any] = {}
    eval_metrics = getattr(example_result, "metrics", {}) or {}

    # Extract or calculate score
    example_score = _extract_score_from_metrics(eval_metrics, primary_objective)
    if example_score is None:
        example_score = _calculate_fallback_score(example_result)
    if example_score is not None:
        metrics_dict["score"] = float(example_score)

    # Add execution time
    if hasattr(example_result, "execution_time"):
        metrics_dict["response_time"] = example_result.execution_time

    # Add privacy-safe token and cost metrics
    if eval_metrics:
        _add_privacy_safe_metrics(metrics_dict, eval_metrics)

    measure_result = {"example_id": example_id, "metrics": metrics_dict}
    _validate_measure_dict(measure_result, idx)
    return measure_result


def _build_measures_privacy(
    example_results: list[Any],
    primary_objective: str,
    dataset_name: str = "dataset",
) -> list[dict[str, Any]]:
    """Build sanitized per-example measures with stable IDs (privacy mode).

    Returns nested format where example_id is at top level and privacy-safe
    metrics are in a 'metrics' sub-object:
        {
            "example_id": "ex_a3f4b2c8_0",
            "metrics": {"score": 0.85, "response_time": 1.2, ...}
        }

    Only includes in metrics:
    - Score, response time, tokens, and cost metrics

    Args:
        example_results: List of example evaluation results
        primary_objective: Primary optimization objective name
        dataset_name: Name of the dataset (for stable example ID generation)
    Returns:
        List of nested sanitized measure dicts

    Raises:
        ValueError: If measures violate MeasuresDict constraints
    """
    dataset_hash = compute_dataset_hash(dataset_name)
    return [
        _build_single_measure_privacy(
            example_result, idx, dataset_hash, primary_objective
        )
        for idx, example_result in enumerate(example_results)
    ]
