"""Core utility functions for TraiGent optimization system.

This module provides utility functions that are used across multiple core
components. These functions handle common operations like safe attribute access,
bounds validation, metrics aggregation, and type conversion.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any

from traigent.core.constants import (
    AGGREGATION_METHODS,
    COST_METRICS,
    DEFAULT_EXECUTION_TIME,
    RESPONSE_TIME_METRIC,
    TOKEN_METRICS,
    VALIDATION_ERROR_CODES,
)

logger = logging.getLogger(__name__)

# =============================================================================
# SAFE ATTRIBUTE ACCESS UTILITIES
# =============================================================================


def safe_get_nested_attr(obj: Any, path: str, default: Any | None = None) -> Any:
    """Safely access nested attributes with dot notation.

    Args:
        obj: Object to access attributes on
        path: Dot-separated path to the attribute (e.g., 'response.metadata.model')
        default: Default value to return if path doesn't exist

    Returns:
        Attribute value or default if path doesn't exist

    Example:
        >>> response = type('Response', (), {'metadata': type('Meta', (), {'model': 'gpt-4'})})()
        >>> safe_get_nested_attr(response, 'metadata.model', 'unknown')
        'gpt-4'
        >>> safe_get_nested_attr(response, 'metadata.missing', 'default')
        'default'
    """
    if obj is None:
        return default

    try:
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            elif isinstance(obj, dict) and attr in obj:
                obj = obj[attr]
            elif isinstance(obj, (list, tuple)) and attr.isdigit():
                obj = obj[int(attr)]
            else:
                return default
        return obj
    except (AttributeError, KeyError, IndexError, ValueError):
        return default


def safe_get_attr(obj: Any, attr: str, default: Any | None = None) -> Any:
    """Safely get an attribute from an object.

    Args:
        obj: Object to get attribute from
        attr: Attribute name
        default: Default value if attribute doesn't exist

    Returns:
        Attribute value or default
    """
    try:
        return getattr(obj, attr, default)
    except (AttributeError, TypeError):
        return default


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


def validate_bounds(
    bounds: tuple[Any, ...] | list[Any], param_type: str, param_name: str = "parameter"
) -> bool:
    """Validate bounds for different parameter types.

    Args:
        bounds: Bounds to validate
        param_type: Type of parameter ('float', 'integer', 'categorical')
        param_name: Name of parameter for error messages

    Returns:
        True if bounds are valid

    Raises:
        ValueError: If bounds are invalid for the parameter type
    """
    if param_type in ("float", "integer"):
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(
                f"{param_type.capitalize()} parameter '{param_name}' must have tuple bounds (min, max)"
            )

        if param_type == "float":
            if not all(isinstance(x, (int, float)) for x in bounds):
                raise ValueError(
                    f"Float parameter '{param_name}' bounds must be numeric"
                )
        elif param_type == "integer":
            if not all(isinstance(x, int) for x in bounds):
                raise ValueError(
                    f"Integer parameter '{param_name}' bounds must be integers"
                )

        if bounds[0] >= bounds[1]:
            raise ValueError(
                f"Invalid bounds for '{param_name}': min ({bounds[0]}) must be less than max ({bounds[1]})"
            )

    elif param_type == "categorical":
        if not isinstance(bounds, (list, tuple)) or len(bounds) == 0:
            raise ValueError(
                f"Categorical parameter '{param_name}' must have non-empty list of choices"
            )

    return True


def validate_positive_number(value: int | float, name: str) -> None:
    """Validate that a value is a positive number.

    Args:
        value: Value to validate
        name: Name of the value for error messages

    Raises:
        ValueError: If value is not positive
        TypeError: If value is not numeric
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_range(
    value: int | float, min_val: int | float, max_val: int | float, name: str
) -> None:
    """Validate that a value is within a specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages

    Raises:
        ValueError: If value is outside the range
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


# =============================================================================
# METRICS AGGREGATION UTILITIES
# =============================================================================


def aggregate_metrics(
    metrics_list: list[dict[str, Any]], aggregation_type: str = "mean"
) -> dict[str, float]:
    """Aggregate metrics across multiple examples.

    Args:
        metrics_list: List of metric dictionaries from different examples
        aggregation_type: Type of aggregation ('mean', 'sum', 'max', 'min', 'median')

    Returns:
        Dictionary with aggregated metrics

    Raises:
        ValueError: If aggregation_type is not supported
    """
    if not metrics_list:
        return {}

    if aggregation_type not in AGGREGATION_METHODS:
        raise ValueError(
            f"Unsupported aggregation type '{aggregation_type}'. Supported: {AGGREGATION_METHODS}"
        )

    # Filter out None metrics
    valid_metrics = [m for m in metrics_list if m is not None]
    if not valid_metrics:
        return {}

    aggregated = {}

    # Get all unique metric names
    all_keys: set[str] = set()
    for metrics in valid_metrics:
        all_keys.update(metrics.keys())

    for key in all_keys:
        values = [
            m.get(key, 0.0) for m in valid_metrics if key in m and m[key] is not None
        ]

        if not values:
            continue

        if aggregation_type == "mean":
            aggregated[key] = sum(values) / len(values)
        elif aggregation_type == "sum":
            aggregated[key] = sum(values)
        elif aggregation_type == "max":
            aggregated[key] = max(values)
        elif aggregation_type == "min":
            aggregated[key] = min(values)
        elif aggregation_type == "median":
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n % 2 == 0:
                aggregated[key] = (
                    sorted_values[n // 2 - 1] + sorted_values[n // 2]
                ) / 2
            else:
                aggregated[key] = sorted_values[n // 2]

    return aggregated


def aggregate_tokens(metrics_list: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate token metrics across examples.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Dictionary with aggregated token counts
    """
    aggregated = aggregate_metrics(metrics_list, "sum")
    # Keep only token-related metrics
    return {k: int(v) for k, v in aggregated.items() if k in TOKEN_METRICS}


def aggregate_costs(metrics_list: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate cost metrics across examples.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Dictionary with aggregated costs
    """
    aggregated = aggregate_metrics(metrics_list, "sum")
    # Keep only cost-related metrics
    return {k: v for k, v in aggregated.items() if k in COST_METRICS}


def aggregate_response_times(metrics_list: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate response time metrics across examples.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Dictionary with aggregated response times
    """
    aggregated = aggregate_metrics(metrics_list, "mean")
    # Keep only response time metrics
    return {k: v for k, v in aggregated.items() if RESPONSE_TIME_METRIC in k}


# =============================================================================
# TYPE CONVERSION UTILITIES
# =============================================================================


def extract_examples_attempted(
    trial_result: Any,
    *,
    default: int | None = None,
    check_example_results: bool = True,
) -> int | None:
    """Extract examples_attempted from trial metrics or metadata.

    Checks in order:
    1. trial_result.metrics.get("examples_attempted")
    2. trial_result.metadata.get("examples_attempted")
    3. len(trial_result.example_results) if check_example_results=True

    Args:
        trial_result: TrialResult object
        default: Value to return if not found (None or int)
        check_example_results: Whether to check example_results as fallback

    Returns:
        Number of examples attempted, or default
    """
    metrics = getattr(trial_result, "metrics", None) or {}
    value = metrics.get("examples_attempted")

    if value is None:
        metadata = getattr(trial_result, "metadata", None) or {}
        value = metadata.get("examples_attempted")

    if value is None and check_example_results:
        example_results = getattr(trial_result, "example_results", None)
        if example_results:
            value = len(example_results)

    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        trial_id = getattr(trial_result, "trial_id", "unknown")
        logger.debug("Unable to parse examples_attempted for trial %s", trial_id)
        return default


def safe_float_convert(value: Any, default: float = 0.0) -> float | None:
    """Safely convert a value to float.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float representation of value or default (None if TRAIGENT_STRICT_METRICS_NULLS=true)
    """
    # Check for strict metrics nulls mode
    strict_nulls = os.environ.get("TRAIGENT_STRICT_METRICS_NULLS", "").lower() in (
        "true",
        "1",
        "yes",
    )
    actual_default = None if strict_nulls else default

    try:
        if value is None:
            return actual_default
        return float(value)
    except (ValueError, TypeError):
        return actual_default


def safe_int_convert(value: Any, default: int = 0) -> int:
    """Safely convert a value to int.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Int representation of value or default
    """
    try:
        if value is None:
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_str_convert(value: Any, default: str = "") -> str:
    """Safely convert a value to string.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        String representation of value or default
    """
    try:
        if value is None:
            return default
        return str(value)
    except (ValueError, TypeError):
        return default


# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================


def merge_configs(
    base_config: dict[str, Any], override_config: dict[str, Any]
) -> dict[str, Any]:
    """Merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Configuration to override base values

    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged


def validate_config_keys(
    config: dict[str, Any],
    required_keys: list[str],
    optional_keys: list[str] | None = None,
) -> list[str]:
    """Validate that a configuration has required keys.

    Args:
        config: Configuration dictionary to validate
        required_keys: List of required key names
        optional_keys: List of optional key names

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required keys
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required configuration key: '{key}'")

    # Check for unexpected keys if optional_keys is specified
    if optional_keys is not None:
        allowed_keys = set(required_keys + optional_keys)
        for key in config.keys():
            if key not in allowed_keys:
                errors.append(f"Unexpected configuration key: '{key}'")

    return errors


# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================


def create_error_message(error_code: str, message: str, **kwargs) -> str:
    """Create a standardized error message.

    Args:
        error_code: Error code from VALIDATION_ERROR_CODES
        message: Error message
        **kwargs: Additional context for message formatting

    Returns:
        Formatted error message
    """
    if error_code not in VALIDATION_ERROR_CODES:
        error_code = "UNKNOWN_ERROR"

    formatted_message = message.format(**kwargs) if kwargs else message
    return f"[{error_code}] {formatted_message}"


def truncate_error_message(message: str, max_length: int = 500) -> str:
    """Truncate an error message to a maximum length.

    Args:
        message: Original error message
        max_length: Maximum allowed length

    Returns:
        Truncated message with ellipsis if needed
    """
    if len(message) <= max_length:
        return message

    return message[: max_length - 3] + "..."


# =============================================================================
# PERFORMANCE UTILITIES
# =============================================================================


def calculate_execution_time(start_time: float, end_time: float | None = None) -> float:
    """Calculate execution time with fallback to default.

    Args:
        start_time: Start time (from time.time())
        end_time: End time (from time.time()), uses current time if None

    Returns:
        Execution time in seconds
    """
    if end_time is None:
        end_time = time.time()

    execution_time = end_time - start_time

    # Ensure positive execution time
    if execution_time <= 0:
        return DEFAULT_EXECUTION_TIME

    return execution_time


# =============================================================================
# DEBUGGING UTILITIES
# =============================================================================


def debug_print_object(obj: Any, max_depth: int = 2, current_depth: int = 0) -> str:
    """Create a debug string representation of an object.

    Args:
        obj: Object to debug
        max_depth: Maximum depth to traverse
        current_depth: Current recursion depth

    Returns:
        Debug string representation
    """
    if current_depth >= max_depth:
        return "..."

    indent = "  " * current_depth

    if obj is None:
        return f"{indent}None"

    if isinstance(obj, dict):
        if not obj:
            return f"{indent}{{}}"
        result = f"{indent}{{\n"
        for key, value in obj.items():
            result += f"{indent}  {key}: {debug_print_object(value, max_depth, current_depth + 1)}\n"
        result += f"{indent}}}"
        return result

    if isinstance(obj, (list, tuple)):
        if not obj:
            return f"{indent}[]"
        result = f"{indent}[\n"
        for item in obj[:5]:  # Limit to first 5 items
            result += (
                f"{indent}  {debug_print_object(item, max_depth, current_depth + 1)}\n"
            )
        if len(obj) > 5:
            result += f"{indent}  ... ({len(obj) - 5} more items)\n"
        result += f"{indent}]"
        return result

    # For other objects, try to get a reasonable representation
    try:
        if hasattr(obj, "__dict__"):
            attrs = vars(obj)
            if attrs:
                result = f"{indent}{type(obj).__name__}(\n"
                for key, value in list(attrs.items())[
                    :3
                ]:  # Limit to first 3 attributes
                    result += f"{indent}  {key}: {debug_print_object(value, max_depth, current_depth + 1)}\n"
                if len(attrs) > 3:
                    result += f"{indent}  ... ({len(attrs) - 3} more attributes)\n"
                result += f"{indent})"
                return result
    except Exception as e:
        # If we fail to introspect the object, log it and fall through to string representation
        logger.debug(
            "Failed to debug print object of type %s: %s", type(obj).__name__, e
        )

    # Fallback to string representation
    return f"{indent}{str(obj)}"


# =============================================================================
# DECORATOR UTILITIES
# =============================================================================


def timing_decorator(func):
    """Decorator to measure and log function execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function that logs execution time
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.4f}s")

    return wrapper


def validation_decorator(validator_func):
    """Create a decorator that validates function inputs.

    Args:
        validator_func: Function that validates inputs and returns error messages

    Returns:
        Decorator function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            errors = validator_func(*args, **kwargs)
            if errors:
                error_msg = "; ".join(errors)
                raise ValueError(
                    f"Validation failed for {func.__name__}: {error_msg}"
                ) from None
            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# VALIDATION UTILITIES (for config_builder.py)
# =============================================================================


def create_validation_result(
    is_valid: bool, errors: list[str] | None = None, warnings: list[str] | None = None
) -> Any:
    """Create a validation result object.

    Args:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages

    Returns:
        ValidationResult object
    """
    # Import here to avoid circular imports
    from traigent.core.types_ext import ValidationResult

    return ValidationResult(
        is_valid=is_valid, errors=errors or [], warnings=warnings or []
    )


# Duplicate function removed - see definition above
