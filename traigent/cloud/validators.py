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


def _validate_comparability_coverage_ratio(comparability: dict) -> None:
    """Validate comparability.coverage_ratio field."""
    if "coverage_ratio" not in comparability:
        return
    ratio = comparability["coverage_ratio"]
    if not isinstance(ratio, (int, float)):
        raise ValueError(
            f"comparability.coverage_ratio must be numeric, got {type(ratio)}"
        )
    if ratio < 0 or ratio > 1:
        raise ValueError("comparability.coverage_ratio must be between 0 and 1")


def _validate_comparability_string_lists(comparability: dict) -> None:
    """Validate comparability list fields (warning_codes, missing_example_ids)."""
    for list_field in ("warning_codes", "missing_example_ids"):
        if list_field not in comparability:
            continue
        list_value = comparability[list_field]
        if not isinstance(list_value, list):
            raise ValueError(
                f"comparability.{list_field} must be an array, got {type(list_value)}"
            )
        if not all(isinstance(item, str) for item in list_value):
            raise ValueError(f"comparability.{list_field} must contain only strings")


def _validate_per_metric_coverage_entry(metric_name: str, metric_coverage: Any) -> None:
    """Validate a single per_metric_coverage entry."""
    if not isinstance(metric_name, str):
        raise ValueError("comparability.per_metric_coverage keys must be strings")
    if not isinstance(metric_coverage, dict):
        raise ValueError("comparability.per_metric_coverage entries must be objects")
    for field, expected in (("present", int), ("total", int), ("ratio", (int, float))):
        value = metric_coverage.get(field)
        if value is not None and not isinstance(value, expected):
            type_label = "integer" if expected is int else "numeric"
            raise ValueError(
                f"comparability.per_metric_coverage.{field} must be {type_label}"
            )


def _validate_per_metric_coverage(comparability: dict) -> None:
    """Validate comparability.per_metric_coverage field."""
    if "per_metric_coverage" not in comparability:
        return
    coverage = comparability["per_metric_coverage"]
    if not isinstance(coverage, dict):
        raise ValueError(
            f"comparability.per_metric_coverage must be an object, got {type(coverage)}"
        )
    for metric_name, metric_coverage in coverage.items():
        _validate_per_metric_coverage_entry(metric_name, metric_coverage)


def validate_comparability_metadata(comparability: Any) -> bool:
    """Validate comparability metadata payload when present."""
    if not isinstance(comparability, dict):
        raise ValueError(
            f"comparability metadata must be an object, got {type(comparability)}"
        )

    for field_name in ("total_examples", "examples_with_primary_metric"):
        if field_name in comparability and not isinstance(
            comparability[field_name], int
        ):
            raise ValueError(
                f"comparability.{field_name} must be an integer, got {type(comparability[field_name])}"
            )

    _validate_comparability_coverage_ratio(comparability)

    if "ranking_eligible" in comparability and not isinstance(
        comparability["ranking_eligible"], bool
    ):
        raise ValueError("comparability.ranking_eligible must be boolean when provided")

    _validate_comparability_string_lists(comparability)
    _validate_per_metric_coverage(comparability)

    return True


_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_metric_entries(entries: dict, key_label: str, value_label: str) -> None:
    """Validate key/value pairs in a metrics dict.

    Args:
        entries: The dict of metric key-value pairs.
        key_label: Label for error messages about keys (e.g. "measure", "metric").
        value_label: Label for error messages about values.
    """
    for key, value in entries.items():
        if not _IDENTIFIER_PATTERN.match(key):
            raise ValueError(
                f"Invalid {key_label} key '{key}': must match pattern ^[a-zA-Z_][a-zA-Z0-9_]*$"
            )
        if value is not None and not isinstance(value, (int, float, str)):
            raise ValueError(
                f"{value_label} value for '{key}' must be number, string, or null, got {type(value)}"
            )


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

    # Check if this is nested format (has example_id and metrics)
    if "example_id" in measure and "metrics" in measure:
        example_id = measure.get("example_id")
        if example_id is not None and not isinstance(example_id, str):
            raise ValueError(f"example_id must be a string, got {type(example_id)}")

        metrics = measure.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics must be a dict, got {type(metrics)}")

        _validate_metric_entries(metrics, "metric", "Metric")
    else:
        _validate_metric_entries(measure, "measure", "Measure")

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


def _validate_submission_metadata(metadata: Any) -> None:
    """Validate top-level metadata field in a configuration run submission."""
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be an object or null, got {type(metadata)}")
    comparability = metadata.get("comparability")
    if comparability is not None:
        validate_comparability_metadata(comparability)


def _validate_metrics_no_misplaced_fields(metrics: Any) -> None:
    """Ensure metrics dict does not contain fields that belong at the top level."""
    if not isinstance(metrics, dict):
        return
    for field in ("measures", "summary_stats"):
        if field in metrics:
            raise ValueError(
                f"metrics dict should not contain '{field}' - it should be a separate field"
            )


def validate_configuration_run_submission(data: dict[str, Any]) -> bool:
    """Validate a complete configuration run submission.

    Args:
        data: The submission data to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    if "measures" in data:
        validate_measures_array(data["measures"])

    if "summary_stats" in data:
        validate_summary_stats(data["summary_stats"])

    if "metadata" in data:
        _validate_submission_metadata(data["metadata"])

    if "metrics" in data:
        _validate_metrics_no_misplaced_fields(data["metrics"])

    return True
