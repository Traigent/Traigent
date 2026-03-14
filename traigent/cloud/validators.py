"""Validation functions for Traigent Backend integration.

This module contains validation functions for measures, summary statistics,
and configuration run submissions according to Traigent schema specifications.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

import re
from typing import Any

_VALID_SUMMARY_STAT_KEYS = {
    "count",
    "mean",
    "std",
    "min",
    "25%",
    "50%",
    "75%",
    "max",
}


def _validate_summary_stat_value(
    metric_key: str, stat_key: str, stat_value: Any
) -> bool:
    """Validate a single statistics entry within summary_stats.metrics."""
    if not isinstance(stat_value, (int, float)) and stat_value is not None:
        raise ValueError(
            f"summary_stats.metrics['{metric_key}']['{stat_key}'] must be a number, got {type(stat_value)}"
        )
    return True


def _validate_summary_metric_value(metric_key: str, value: Any) -> bool:
    """Validate a summary_stats.metrics value."""
    if value is None or isinstance(value, (int, float, str)):
        return True

    if not isinstance(value, dict):
        raise ValueError(
            f"summary_stats.metrics['{metric_key}'] must be number, string, null, or statistics dict, got {type(value)}"
        )

    invalid_keys = set(value.keys()) - _VALID_SUMMARY_STAT_KEYS
    if invalid_keys:
        raise ValueError(
            f"summary_stats.metrics['{metric_key}'] has invalid statistics keys: {invalid_keys}"
        )

    return all(
        _validate_summary_stat_value(metric_key, stat_key, stat_value)
        for stat_key, stat_value in value.items()
    )


def validate_comparability_metadata(comparability: Any) -> bool:
    """Validate comparability metadata payload when present."""
    if not isinstance(comparability, dict):
        raise ValueError(
            f"comparability metadata must be an object, got {type(comparability)}"
        )

    numeric_int_fields = ("total_examples", "examples_with_primary_metric")
    for field_name in numeric_int_fields:
        if field_name in comparability and not isinstance(
            comparability[field_name], int
        ):
            raise ValueError(
                f"comparability.{field_name} must be an integer, got {type(comparability[field_name])}"
            )

    if "coverage_ratio" in comparability:
        ratio = comparability["coverage_ratio"]
        if not isinstance(ratio, (int, float)):
            raise ValueError(
                f"comparability.coverage_ratio must be numeric, got {type(ratio)}"
            )
        if ratio < 0 or ratio > 1:
            raise ValueError("comparability.coverage_ratio must be between 0 and 1")

    if "ranking_eligible" in comparability and not isinstance(
        comparability["ranking_eligible"], bool
    ):
        raise ValueError("comparability.ranking_eligible must be boolean when provided")

    for list_field in ("warning_codes", "missing_example_ids"):
        if list_field in comparability:
            list_value = comparability[list_field]
            if not isinstance(list_value, list):
                raise ValueError(
                    f"comparability.{list_field} must be an array, got {type(list_value)}"
                )
            if not all(isinstance(item, str) for item in list_value):
                raise ValueError(
                    f"comparability.{list_field} must contain only strings"
                )

    if "per_metric_coverage" in comparability:
        coverage = comparability["per_metric_coverage"]
        if not isinstance(coverage, dict):
            raise ValueError(
                f"comparability.per_metric_coverage must be an object, got {type(coverage)}"
            )
        for metric_name, metric_coverage in coverage.items():
            if not isinstance(metric_name, str):
                raise ValueError(
                    "comparability.per_metric_coverage keys must be strings"
                )
            if not isinstance(metric_coverage, dict):
                raise ValueError(
                    "comparability.per_metric_coverage entries must be objects"
                )
            present = metric_coverage.get("present")
            total = metric_coverage.get("total")
            ratio = metric_coverage.get("ratio")
            if present is not None and not isinstance(present, int):
                raise ValueError(
                    "comparability.per_metric_coverage.present must be integer"
                )
            if total is not None and not isinstance(total, int):
                raise ValueError(
                    "comparability.per_metric_coverage.total must be integer"
                )
            if ratio is not None and not isinstance(ratio, (int, float)):
                raise ValueError(
                    "comparability.per_metric_coverage.ratio must be numeric"
                )

    return True


def validate_measure_results(measure: Any) -> bool:
    """Validate a single MeasureResults object according to Traigent schema.

    Supports two formats:
    1. Flat format (legacy): {metric_key: value, ...}
    2. Nested format: {example_id: str, metrics: {metric_key: value, ...}}

    Args:
        measure: The measure object to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    if not isinstance(measure, dict):
        raise ValueError(
            f"MeasureResults must be a dict, got {type(measure)}"
        ) from None

    # Validate that all keys match the pattern [a-zA-Z_][a-zA-Z0-9_]*
    pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

    # Check if this is nested format (has example_id and metrics)
    is_nested = "example_id" in measure and "metrics" in measure

    if is_nested:
        # Nested format validation
        example_id = measure.get("example_id")
        if example_id is not None and not isinstance(example_id, str):
            raise ValueError(f"example_id must be a string, got {type(example_id)}")

        metrics = measure.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics must be a dict, got {type(metrics)}")

        # Validate metrics dict
        for key, value in metrics.items():
            if not pattern.match(key):
                raise ValueError(
                    f"Invalid metric key '{key}': must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$"
                )
            if value is not None and not isinstance(value, (int, float, str)):
                raise ValueError(
                    f"Metric value for '{key}' must be number, string, or null, got {type(value)}"
                )
    else:
        # Flat format validation (legacy)
        for key, value in measure.items():
            # Check key format
            if not pattern.match(key):
                raise ValueError(
                    f"Invalid measure key '{key}': must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$"
                )

            # Check value type (must be number, string, or null)
            if value is not None and not isinstance(value, (int, float, str)):
                raise ValueError(
                    f"Measure value for '{key}' must be number, string, or null, got {type(value)}"
                )

    return True


def validate_measures_array(measures: Any) -> bool:
    """Validate measures array according to Traigent schema.

    Args:
        measures: The measures array to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    if measures is None:
        return True  # null is allowed

    if not isinstance(measures, list):
        raise ValueError(f"measures must be an array or null, got {type(measures)}")

    def _validate_measure_at_index(index: int, measure: Any) -> bool:
        try:
            return validate_measure_results(measure)
        except ValueError as e:
            raise ValueError(f"Invalid measure at index {index}: {e}") from None

    return all(
        _validate_measure_at_index(index, measure)
        for index, measure in enumerate(measures)
    )


def validate_summary_stats(summary_stats: Any) -> bool:
    """Validate summary_stats according to Traigent schema.

    Args:
        summary_stats: The summary_stats object to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    if summary_stats is None:
        return True  # null is allowed

    if not isinstance(summary_stats, dict):
        raise ValueError(
            f"summary_stats must be an object or null, got {type(summary_stats)}"
        )

    # Check allowed properties
    allowed_props = {"metrics", "execution_time", "total_examples", "metadata"}
    extra_props = set(summary_stats.keys()) - allowed_props
    if extra_props:
        raise ValueError(f"summary_stats has unexpected properties: {extra_props}")

    # Validate metrics if present
    metrics_valid = True
    if "metrics" in summary_stats:
        metrics = summary_stats["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError(
                f"summary_stats.metrics must be an object, got {type(metrics)}"
            )

        metrics_valid = all(
            _validate_summary_metric_value(key, value) for key, value in metrics.items()
        )

    # Validate execution_time if present
    if "execution_time" in summary_stats:
        exec_time = summary_stats["execution_time"]
        if exec_time is not None and not isinstance(exec_time, (int, float)):
            raise ValueError(
                f"summary_stats.execution_time must be number or null, got {type(exec_time)}"
            )

    # Validate total_examples if present
    if "total_examples" in summary_stats:
        total = summary_stats["total_examples"]
        if total is not None and not isinstance(total, int):
            raise ValueError(
                f"summary_stats.total_examples must be integer or null, got {type(total)}"
            )

    # metadata can be any object, so no validation needed beyond type check
    comparability_valid = True
    if "metadata" in summary_stats:
        metadata = summary_stats["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(
                f"summary_stats.metadata must be an object, got {type(summary_stats['metadata'])}"
            )
        comparability = metadata.get("comparability")
        if comparability is not None:
            comparability_valid = validate_comparability_metadata(comparability)

    return metrics_valid and comparability_valid


def validate_configuration_run_submission(data: dict[str, Any]) -> bool:
    """Validate a complete configuration run submission.

    Args:
        data: The submission data to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    # Validate measures if present
    if "measures" in data:
        validate_measures_array(data["measures"])

    # Validate summary_stats if present
    if "summary_stats" in data:
        validate_summary_stats(data["summary_stats"])

    if "metadata" in data:
        metadata = data["metadata"]
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(
                f"metadata must be an object or null, got {type(metadata)}"
            )
        if isinstance(metadata, dict):
            comparability = metadata.get("comparability")
            if comparability is not None:
                validate_comparability_metadata(comparability)

    # Validate metrics should NOT contain measures or summary_stats
    if "metrics" in data and isinstance(data["metrics"], dict):
        if "measures" in data["metrics"]:
            raise ValueError(
                "metrics dict should not contain 'measures' - it should be a separate field"
            )
        if "summary_stats" in data["metrics"]:
            raise ValueError(
                "metrics dict should not contain 'summary_stats' - it should be a separate field"
            )

    return True
