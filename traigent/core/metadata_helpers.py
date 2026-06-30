"""Metadata construction helpers for trial and session results."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

import json
from collections.abc import Mapping
from typing import Any, cast

from traigent._version import get_version
from traigent.api.types import OptimizationResult, TrialResult
from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.utils.example_id import compute_dataset_hash, generate_stable_example_id
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Producer-side cap on the per-example ``measures[]`` array submitted to the
# backend on ``POST /sessions/{id}/results``. The backend enforces
# ``MAX_EXAMPLES_PER_RUN = 1000`` and, on overflow, rejects the WHOLE submission
# (returning an empty list) -- which silently loses ALL of a trial's measures.
# We mirror that cap here so the SDK never emits a payload the server rejects
# wholesale: keep the first ``MAX_EXAMPLES_PER_RUN`` measures and WARN loudly
# about what was dropped, rather than losing everything. Keep in sync with the
# backend ``MAX_EXAMPLES_PER_RUN`` and ``configuration_run_schema`` maxItems.
MAX_EXAMPLES_PER_RUN = 1000


def _cap_measures(
    measures: list[dict[str, Any]], execution_mode: str
) -> list[dict[str, Any]]:
    """Cap per-example measures to the backend limit with a loud warning.

    The backend rejects the entire submission when ``measures`` exceeds
    :data:`MAX_EXAMPLES_PER_RUN`, so an uncapped array would lose ALL measures
    for the trial. Truncate to the limit (keeping the first
    ``MAX_EXAMPLES_PER_RUN`` entries) and surface the truncation via a WARNING
    so the data loss is never silent.

    Args:
        measures: Per-example measure dicts (may exceed the backend cap).
        execution_mode: Execution mode label, for the warning message.

    Returns:
        The measures list, truncated to at most ``MAX_EXAMPLES_PER_RUN`` items.
    """
    if len(measures) <= MAX_EXAMPLES_PER_RUN:
        return measures

    dropped = len(measures) - MAX_EXAMPLES_PER_RUN
    logger.warning(
        "Per-example measures (%s) exceed the backend cap of %s for %s mode; "
        "truncating to the first %s and dropping %s. The backend rejects the "
        "whole submission above this cap, so the trial would otherwise lose "
        "ALL measures. Set max_examples<=%s to control which examples are "
        "kept.",
        len(measures),
        MAX_EXAMPLES_PER_RUN,
        execution_mode,
        MAX_EXAMPLES_PER_RUN,
        dropped,
        MAX_EXAMPLES_PER_RUN,
    )
    return measures[:MAX_EXAMPLES_PER_RUN]


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
    if not measure["metrics"]:
        raise ValueError(f"Example {example_index}: 'metrics' must not be empty")
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
    if not summary_stats_available:
        return

    enhanced_summary_stats = trial_result.summary_stats.copy()  # type: ignore[attr-defined]
    if "metadata" not in enhanced_summary_stats:
        enhanced_summary_stats["metadata"] = {}
    enhanced_summary_stats["metadata"]["aggregation_level"] = "trial"
    enhanced_summary_stats["metadata"]["sdk_version"] = get_version()
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

    if not privacy_on:
        measures = _cap_measures(
            _build_measures_full(example_results, primary_objective, dataset_name),
            execution_mode,
        )
        if measures:
            trial_metadata["measures"] = measures
            logger.debug(
                "Collected %s per-example MeasureResults for %s mode",
                len(measures),
                execution_mode,
            )
            _log_measures_debug(measures)
        else:
            trial_metadata.pop("measures", None)
    elif privacy_on:
        measures = _cap_measures(
            _build_measures_privacy(example_results, primary_objective, dataset_name),
            execution_mode,
        )
        if measures:
            trial_metadata["measures"] = measures
            logger.debug(
                "Using sanitized MeasureResults for privacy mode: %s examples",
                len(measures),
            )
        else:
            trial_metadata.pop("measures", None)
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


def _metadata_string_sequence(metadata: dict[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item) for item in value if item)


def _observation_comparability(trial_metadata: dict[str, Any]) -> dict[str, Any]:
    payload = trial_metadata.get("comparability")
    n = 0
    if isinstance(payload, dict):
        for key in ("n", "examples_with_primary_metric", "total_examples"):
            try:
                n = int(payload.get(key, 0))
            except (TypeError, ValueError):
                n = 0
            if n:
                break
    return {"scope": "trial", "n": max(0, n)}


def _add_tvar_observation(
    trial_metadata: dict[str, Any],
    trial_result: TrialResult,
    primary_objective: str,
    session_id: str | None,
) -> dict[str, Any]:
    effective_session_id = session_id or trial_metadata.get("session_id")
    if not effective_session_id:
        return trial_metadata

    try:
        source_metadata = getattr(trial_result, "metadata", {}) or {}
        if not isinstance(source_metadata, dict):
            source_metadata = {}
        from traigent.tuned_variables.observation import (
            build_tvar_observation,
            merge_tvar_observation_metadata,
        )

        observation = build_tvar_observation(
            session_id=str(effective_session_id),
            trial_id=str(trial_result.trial_id),
            config=getattr(trial_result, "config", {}) or {},
            metrics=trial_result.metrics or {},
            primary_metric=primary_objective,
            comparability=_observation_comparability(trial_metadata),
            catalog_entry_ids=_metadata_string_sequence(
                source_metadata, "catalog_entry_ids"
            ),
            agent_type=source_metadata.get("agent_type"),
            config_space_id=source_metadata.get("config_space_id"),
            effectuation_events=source_metadata.get("effectuation_events"),
        )
        return cast(
            dict[str, Any],
            merge_tvar_observation_metadata(trial_metadata, observation),
        )
    except Exception as exc:
        logger.debug(
            "Skipping TVAR observation metadata for trial %s: %s",
            getattr(trial_result, "trial_id", "<unknown>"),
            exc,
        )
        return trial_metadata


def build_backend_metadata(
    trial_result: TrialResult,
    primary_objective: str,
    traigent_config: TraigentConfig,
    dataset_name: str = "dataset",
    content_scores: dict[str, dict[int, float]] | None = None,
    session_id: str | None = None,
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
        session_id: Optional backend session identifier for TVAR observation emission.

    Returns:
        Metadata dict ready for backend submission
    """
    _ = content_scores

    trial_metadata: dict[str, Any] = {
        "duration": trial_result.duration,
        "trial_id": trial_result.trial_id,
    }

    if not traigent_config.minimal_logging:
        trial_metadata["timestamp"] = trial_result.timestamp.isoformat()
        trial_metadata["all_metrics"] = trial_result.metrics

    mode_enum = traigent_config.execution_mode_enum
    _add_summary_stats(trial_metadata, trial_result, mode_enum)

    comparability_payload = trial_result.metadata.get("comparability")
    if isinstance(comparability_payload, dict):
        trial_metadata["comparability"] = comparability_payload

    privacy_on = getattr(traigent_config, "privacy_enabled", False)
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

    return _add_tvar_observation(
        trial_metadata,
        trial_result,
        primary_objective,
        session_id,
    )


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


def _example_field(example_result: Any, name: str, default: Any = None) -> Any:
    """Read a field from an example result object OR its dict payload form.

    Trial metadata stores example results as redacted ``to_dict()`` payloads
    (see ``trial_result_factory._to_redactable_payloads``), so the measure
    builders must read plain dicts as well as ``ExampleResult`` objects.
    """
    if isinstance(example_result, Mapping):
        return example_result.get(name, default)
    return getattr(example_result, name, default)


def _calculate_fallback_score(example_result: Any) -> float | None:
    """Calculate fallback score from expected vs actual output."""
    expected = _example_field(example_result, "expected_output")
    actual = _example_field(example_result, "actual_output")
    if expected is not None and actual is not None:
        return 1.0 if actual == expected else 0.0
    return None


def _add_execution_time_metrics(
    metrics_dict: dict[str, Any], example_result: Any
) -> None:
    """Add explicit and legacy timing metrics for compatibility.

    ``execution_time`` on example results is tracked internally in seconds. Its
    canonical optimization payload key is ``execution_time_ms``. The legacy
    ``response_time`` seconds key remains for one compatibility window.
    """
    execution_time = _example_field(example_result, "execution_time")
    if execution_time is None or not isinstance(execution_time, (int, float)):
        return

    execution_time_seconds = float(execution_time)
    metrics_dict.setdefault("execution_time_ms", execution_time_seconds * 1000.0)
    metrics_dict["response_time"] = execution_time_seconds


def _is_measure_metric_value(value: Any) -> bool:
    """Return True for MeasuresDict metric values accepted by the SDK."""
    return value is None or (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _coerce_non_empty_example_id(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _metadata_example_id(metadata: Any) -> str | None:
    if not isinstance(metadata, Mapping):
        return None

    id_keys = ("example_id", "dataset_example_id", "input_id", "row_id", "id")
    for key in id_keys:
        example_id = _coerce_non_empty_example_id(metadata.get(key))
        if example_id is not None:
            return example_id

    for container_key in ("dataset", "source", "provenance"):
        nested = metadata.get(container_key)
        if isinstance(nested, Mapping):
            for key in id_keys:
                nested_example_id = _coerce_non_empty_example_id(nested.get(key))
                if nested_example_id is not None:
                    return nested_example_id

    return None


def _resolve_measure_example_id(
    example_result: Any, idx: int, dataset_hash: str
) -> str:
    """Prefer the evaluator/dataset example_id, falling back to synthetic IDs."""
    for key in ("example_id", "dataset_example_id", "input_id"):
        example_id = _coerce_non_empty_example_id(_example_field(example_result, key))
        if example_id is not None:
            return example_id

    metadata_example_id = _metadata_example_id(
        _example_field(example_result, "metadata")
    )
    if metadata_example_id is not None:
        return metadata_example_id

    return generate_stable_example_id(dataset_hash, idx)


def _build_single_measure_full(
    example_result: Any,
    idx: int,
    dataset_hash: str,
    primary_objective: str,
) -> dict[str, Any] | None:
    """Build a single full measure for non-privacy mode."""
    example_id = _resolve_measure_example_id(example_result, idx, dataset_hash)
    metrics_dict: dict[str, Any] = {}
    eval_metrics = _example_field(example_result, "metrics") or {}

    if eval_metrics:
        logger.debug("Example %s metrics: %s", idx, eval_metrics)
        # Add all scalar metrics (numeric only per MeasuresDict constraints)
        for metric_key, metric_value in eval_metrics.items():
            if _is_measure_metric_value(metric_value):
                metrics_dict[metric_key] = metric_value

    # Extract or calculate score
    example_score = _extract_score_from_metrics(eval_metrics, primary_objective)
    if example_score is None:
        example_score = _calculate_fallback_score(example_result)
    if example_score is not None:
        metrics_dict["score"] = float(example_score)

    _add_execution_time_metrics(metrics_dict, example_result)

    if not metrics_dict:
        logger.debug("Skipping measure %s because it has no numeric metrics", idx)
        return None

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
    measures: list[dict[str, Any]] = []
    for idx, example_result in enumerate(example_results):
        measure = _build_single_measure_full(
            example_result, idx, dataset_hash, primary_objective
        )
        if measure is not None:
            measures.append(measure)
    return measures


_PRIVACY_TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")
_PRIVACY_COST_KEYS = ("input_cost", "output_cost", "total_cost")


def _add_privacy_safe_metrics(
    metrics_dict: dict[str, Any], eval_metrics: dict[str, Any]
) -> None:
    """Add token and cost metrics which are privacy-safe."""
    for token_key in _PRIVACY_TOKEN_KEYS:
        if token_key in eval_metrics and _is_measure_metric_value(
            eval_metrics[token_key]
        ):
            metrics_dict[token_key] = eval_metrics[token_key]
    for cost_key in _PRIVACY_COST_KEYS:
        if cost_key in eval_metrics and _is_measure_metric_value(
            eval_metrics[cost_key]
        ):
            metrics_dict[cost_key] = eval_metrics[cost_key]


def _build_single_measure_privacy(
    example_result: Any,
    idx: int,
    dataset_hash: str,
    primary_objective: str,
) -> dict[str, Any] | None:
    """Build a single sanitized measure for privacy mode."""
    example_id = _resolve_measure_example_id(example_result, idx, dataset_hash)
    metrics_dict: dict[str, Any] = {}
    eval_metrics = _example_field(example_result, "metrics") or {}

    # Extract or calculate score
    example_score = _extract_score_from_metrics(eval_metrics, primary_objective)
    if example_score is None:
        example_score = _calculate_fallback_score(example_result)
    if example_score is not None:
        metrics_dict["score"] = float(example_score)

    _add_execution_time_metrics(metrics_dict, example_result)

    # Add privacy-safe token and cost metrics
    if eval_metrics:
        _add_privacy_safe_metrics(metrics_dict, eval_metrics)

    if not metrics_dict:
        logger.debug(
            "Skipping sanitized measure %s because it has no numeric metrics", idx
        )
        return None

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
            "metrics": {"score": 0.85, "response_time_ms": 1200.0, ...}
        }

    Only includes in metrics:
    - Score, explicit response_time_ms, execution_time_ms, legacy response_time,
      tokens, and cost metrics

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
    measures: list[dict[str, Any]] = []
    for idx, example_result in enumerate(example_results):
        measure = _build_single_measure_privacy(
            example_result, idx, dataset_hash, primary_objective
        )
        if measure is not None:
            measures.append(measure)
    return measures
