"""Validation functions for OptiGen Backend integration.

This module contains validation functions for measures, summary statistics,
and configuration run submissions according to OptiGen schema specifications.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

import re
from typing import Any


def validate_measure_results(measure: Any) -> bool:
    """Validate a single MeasureResults object according to OptiGen schema.

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
    """Validate measures array according to OptiGen schema.

    Args:
        measures: The measures array to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    if measures is None:
        return True  # null is allowed

    if not isinstance(measures, list):
        raise ValueError(f"measures must be an array or null, got {type(measures)}")

    for i, measure in enumerate(measures):
        try:
            validate_measure_results(measure)
        except ValueError as e:
            raise ValueError(f"Invalid measure at index {i}: {e}") from None

    return True


def validate_summary_stats(summary_stats: Any) -> bool:
    """Validate summary_stats according to OptiGen schema.

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
    if "metrics" in summary_stats:
        metrics = summary_stats["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError(
                f"summary_stats.metrics must be an object, got {type(metrics)}"
            )

        # Validate each metric value
        # Values can be:
        # 1. Simple value: number, string, or null (original format)
        # 2. pandas.describe dict with statistics (enhanced format for privacy mode)
        for key, value in metrics.items():
            if value is None:
                continue  # null is allowed
            elif isinstance(value, (int, float, str)):
                continue  # Simple values are allowed
            elif isinstance(value, dict):
                # Check if it's a valid pandas.describe-style statistics dict
                valid_stat_keys = {
                    "count",
                    "mean",
                    "std",
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max",
                }
                if not all(k in valid_stat_keys for k in value.keys()):
                    invalid_keys = set(value.keys()) - valid_stat_keys
                    raise ValueError(
                        f"summary_stats.metrics['{key}'] has invalid statistics keys: {invalid_keys}"
                    )
                # Validate each statistic value
                for stat_key, stat_value in value.items():
                    if (
                        not isinstance(stat_value, (int, float))
                        and stat_value is not None
                    ):
                        raise ValueError(
                            f"summary_stats.metrics['{key}']['{stat_key}'] must be a number, got {type(stat_value)}"
                        )
            else:
                raise ValueError(
                    f"summary_stats.metrics['{key}'] must be number, string, null, or statistics dict, got {type(value)}"
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
    if "metadata" in summary_stats:
        if not isinstance(summary_stats["metadata"], dict):
            raise ValueError(
                f"summary_stats.metadata must be an object, got {type(summary_stats['metadata'])}"
            )

    return True


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
