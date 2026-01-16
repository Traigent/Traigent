"""Metadata construction helpers for trial and session results."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

import json
from typing import Any

from traigent.api.types import OptimizationResult, TrialResult
from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.utils.example_id import compute_dataset_hash, generate_stable_example_id
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def _validate_measure_dict(measure: dict[str, Any], example_index: int) -> None:
    """Validate measure against MeasuresDict constraints from TraigentSchema.

    MeasuresDict constraints (from traigent_schema/schemas/measures/):
    1. Max 50 keys per measure
    2. Keys must be valid Python identifiers (^[a-zA-Z_][a-zA-Z0-9_]*$)
    3. Values must be numeric (int, float) or None (except example_id which can be str)

    Args:
        measure: Measure dict to validate
        example_index: Index of the example (for error messages)

    Raises:
        ValueError: If validation fails
    """
    # Constraint 1: Max 50 keys
    if len(measure) > 50:
        raise ValueError(
            f"Example {example_index}: Measure has {len(measure)} keys, "
            f"exceeds MeasuresDict limit of 50"
        )

    # Constraint 2: Keys must be Python identifiers
    for key in measure.keys():
        if not key.isidentifier():
            raise ValueError(
                f"Example {example_index}: Invalid measure key '{key}', "
                f"must be Python identifier (^[a-zA-Z_][a-zA-Z0-9_]*$)"
            )

    # Constraint 3: Values must be numeric or None (except example_id)
    for key, value in measure.items():
        if value is not None:
            # example_id is allowed to be string
            if key == "example_id" and isinstance(value, str):
                continue
            # All other values must be numeric
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Example {example_index}: Measure key '{key}' has invalid type "
                    f"{type(value).__name__}, must be int, float, or None"
                )


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
    dataset_name: str = "dataset",
    content_scores: dict[str, dict[int, float]] | None = None,
) -> dict[str, Any]:
    """Build metadata dictionary for backend trial submission.

    Constructs comprehensive metadata payload including:
    - Basic trial info (duration, timestamp, execution mode)
    - Summary statistics (if available and not in cloud mode)
    - Per-example measures (respecting privacy settings)
    - Stable example IDs (format: ex_{hash}_{index})
    - Content-based scores (uniqueness, novelty)
    - Additional metrics beyond primary objective

    Args:
        trial_result: Completed trial result
        primary_objective: Primary optimization objective name
        traigent_config: Global configuration for execution mode and privacy
        dataset_name: Name of the dataset (used for stable example ID generation)
        content_scores: Optional dict with keys "uniqueness", "novelty" mapping
                       example_index -> score (0.0-1.0). Pre-computed by ContentScorer.

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
        hasattr(trial_result, "summary_stats") and trial_result.summary_stats  # type: ignore[attr-defined]
    )
    if summary_stats_available and mode_enum is not ExecutionMode.CLOUD:
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
        measures = _build_measures_full(
            example_results, primary_objective, dataset_name, content_scores
        )
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
        measures = _build_measures_privacy(
            example_results, primary_objective, dataset_name, content_scores
        )
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
    dataset_name: str = "dataset",
    content_scores: dict[str, dict[int, float]] | None = None,
) -> list[dict[str, Any]]:
    """Build full per-example measures with stable IDs and content scores (non-privacy mode).

    Args:
        example_results: List of example evaluation results
        primary_objective: Primary optimization objective name
        dataset_name: Name of the dataset (for stable example ID generation)
        content_scores: Optional dict with keys "uniqueness", "novelty" mapping
                       example_index -> score

    Returns:
        List of measure dicts with all available metrics

    Raises:
        ValueError: If measures violate MeasuresDict constraints
    """
    dataset_hash = compute_dataset_hash(dataset_name)
    measures = []

    for idx, example_result in enumerate(example_results):
        measure_result: dict[str, Any] = {}

        # Generate stable example_id (format: ex_{hash}_{index})
        example_id = generate_stable_example_id(dataset_hash, idx)
        measure_result["example_id"] = example_id

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

        # Add content scores if available (privacy-safe: numeric only)
        if content_scores:
            if "uniqueness" in content_scores:
                measure_result["content_uniqueness"] = content_scores["uniqueness"].get(
                    idx, 0.5
                )
            if "novelty" in content_scores:
                measure_result["content_novelty"] = content_scores["novelty"].get(
                    idx, 0.5
                )

        # Validate against MeasuresDict constraints
        _validate_measure_dict(measure_result, idx)

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
    dataset_name: str = "dataset",
    content_scores: dict[str, dict[int, float]] | None = None,
) -> list[dict[str, Any]]:
    """Build sanitized per-example measures with stable IDs (privacy mode).

    Only includes:
    - Stable example_id (privacy-safe: hash-based, no PII)
    - Score, response time, tokens, and cost metrics
    - Content scores (privacy-safe: numeric only, no raw text)

    Args:
        example_results: List of example evaluation results
        primary_objective: Primary optimization objective name
        dataset_name: Name of the dataset (for stable example ID generation)
        content_scores: Optional dict with keys "uniqueness", "novelty" mapping
                       example_index -> score

    Returns:
        List of sanitized measure dicts

    Raises:
        ValueError: If measures violate MeasuresDict constraints
    """
    dataset_hash = compute_dataset_hash(dataset_name)
    measures = []

    for idx, example_result in enumerate(example_results):
        measure_result: dict[str, Any] = {}

        # Stable example_id (privacy-safe: just a hash, no content)
        example_id = generate_stable_example_id(dataset_hash, idx)
        measure_result["example_id"] = example_id

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

        # Content scores are privacy-safe (just floats, no raw text)
        if content_scores:
            if "uniqueness" in content_scores:
                measure_result["content_uniqueness"] = content_scores["uniqueness"].get(
                    idx, 0.5
                )
            if "novelty" in content_scores:
                measure_result["content_novelty"] = content_scores["novelty"].get(
                    idx, 0.5
                )

        # Validate against MeasuresDict constraints
        _validate_measure_dict(measure_result, idx)

        measures.append(measure_result)

    return measures
