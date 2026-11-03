"""Shared finite-domain helpers for stepped search ranges."""

from __future__ import annotations

import math
import sys
from typing import Any

FLOAT_GRID_ROUND_DIGITS = 6
_FLOAT_STEP_TOLERANCE = 1e-9


def float_step_divides_span(low: float, high: float, step: float) -> bool:
    """Return True when ``step`` lands exactly on ``high`` within tolerance."""

    if step <= 0:
        return False
    span = high - low
    if span < 0:
        return False
    intervals = span / step
    nearest = round(intervals)
    tolerance = _FLOAT_STEP_TOLERANCE * max(1.0, abs(intervals))
    return abs(intervals - nearest) <= tolerance


def stepped_int_values(low: int, high: int, step: int) -> tuple[int, ...]:
    """Build the integer grid, preserving the inclusive high endpoint."""

    if step <= 0 or low > high:
        return ()
    values = tuple(range(low, high + 1, step))
    if not values:
        return (low,)
    if values[-1] != high:
        return (*values, high)
    return values


def stepped_int_contains(low: int, high: int, step: int, value: int) -> bool:
    """True when ``value`` is on the integer grid, including the high endpoint.

    Same membership as :func:`stepped_int_values` without materializing it.
    """

    if step <= 0 or low > high or not low <= value <= high:
        return False
    return value == high or (value - low) % step == 0


def stepped_int_cardinality(low: int, high: int, step: int) -> int:
    """Count integer grid points without materializing them."""

    if step <= 0 or low > high:
        return 0

    count = ((high - low) // step) + 1
    last_value = low + ((count - 1) * step)
    if last_value != high:
        count += 1
    return count


def stepped_float_values(low: float, high: float, step: float) -> tuple[float, ...]:
    """Build the stepped float grid, preserving the inclusive high endpoint."""

    if step <= 0 or low > high:
        return ()

    values: list[float] = []
    epsilon = _FLOAT_STEP_TOLERANCE * max(1.0, abs(high - low), abs(step))
    index = 0
    while True:
        value = low + (index * step)
        if value > high + epsilon:
            break
        value = min(max(value, low), high)
        rounded = round(value, FLOAT_GRID_ROUND_DIGITS)
        if not values or values[-1] != rounded:
            values.append(rounded)
        index += 1

    high_value = round(high, FLOAT_GRID_ROUND_DIGITS)
    if not values or values[-1] != high_value:
        values.append(high_value)
    return tuple(values)


def stepped_float_cardinality(low: float, high: float, step: float) -> int:
    """Count stepped float grid points without materializing them."""

    if step <= 0 or low > high:
        return 0

    epsilon = _FLOAT_STEP_TOLERANCE * max(1.0, abs(high - low), abs(step))
    interval_count = (high - low + epsilon) / step
    if not math.isfinite(interval_count):
        return sys.maxsize

    count = max(math.floor(interval_count) + 1, 1)
    value = low + ((count - 1) * step)
    rounded = round(min(max(value, low), high), FLOAT_GRID_ROUND_DIGITS)
    high_value = round(high, FLOAT_GRID_ROUND_DIGITS)
    if rounded != high_value:
        count += 1
    return count


_CATEGORICAL_TYPES = {"categorical", "choice"}
_FIXED_TYPES = {"fixed", "constant"}
_INT_TYPES = {"int", "integer"}


def _dict_param_shape(definition: dict[Any, Any]) -> tuple[str, bool, bool]:
    """Return ``(param_type, has_range, has_categorical_values)`` for a dict."""

    param_type = str(definition.get("type") or "").lower()
    has_range = "low" in definition and "high" in definition
    has_categorical_values = "choices" in definition or "values" in definition
    return param_type, has_range, has_categorical_values


def _is_categorical_shape(
    param_type: str, has_range: bool, has_categorical_values: bool
) -> bool:
    return param_type in _CATEGORICAL_TYPES or (
        not param_type and not has_range and has_categorical_values
    )


def _categorical_choices(definition: dict[Any, Any]) -> Any:
    return (
        definition["choices"]
        if "choices" in definition
        else definition.get("values", ())
    )


def _int_range_args(
    definition: dict[Any, Any], low: Any, high: Any
) -> tuple[int, int, int] | None:
    """Coerce an integer range, or None when a bound or the step is invalid."""

    step = definition.get("step", 1)
    if step is None:
        step = 1
    try:
        return int(low), int(high), int(step)
    except (TypeError, ValueError):
        return None


def _float_range_args(
    definition: dict[Any, Any], low: Any, high: Any
) -> tuple[float, float, float] | None:
    """Coerce a float range, or None when the step is missing or invalid."""

    step = definition.get("step")
    if step is None:
        return None
    try:
        return float(low), float(high), float(step)
    except (TypeError, ValueError):
        return None


def _range_cardinality(definition: dict[Any, Any], param_type: str) -> int | None:
    low = definition.get("low")
    high = definition.get("high")
    if low is None or high is None:
        return None

    if param_type in _INT_TYPES:
        int_args = _int_range_args(definition, low, high)
        if int_args is None:
            return None
        low_i, high_i, step_i = int_args
        if step_i <= 0 or low_i > high_i:
            return None
        return stepped_int_cardinality(low_i, high_i, step_i)

    float_args = _float_range_args(definition, low, high)
    if float_args is None:
        return None
    low_f, high_f, step_f = float_args
    if step_f <= 0 or low_f > high_f:
        return None
    return stepped_float_cardinality(low_f, high_f, step_f)


def _range_values(
    definition: dict[Any, Any], param_type: str
) -> tuple[Any, ...] | None:
    low = definition.get("low")
    high = definition.get("high")
    if low is None or high is None:
        return None

    if param_type in _INT_TYPES:
        int_args = _int_range_args(definition, low, high)
        if int_args is None:
            return None
        low_i, high_i, step_i = int_args
        if step_i <= 0 or low_i > high_i:
            return ()
        return stepped_int_values(low_i, high_i, step_i)

    float_args = _float_range_args(definition, low, high)
    if float_args is None:
        return None
    low_f, high_f, step_f = float_args
    if step_f <= 0 or low_f > high_f:
        return ()
    return stepped_float_values(low_f, high_f, step_f)


def _fixed_values(definition: dict[Any, Any]) -> tuple[Any, ...]:
    if "value" in definition:
        return (definition["value"],)
    if "default" in definition:
        return (definition["default"],)
    return (definition,)


def _dict_cardinality(definition: dict[Any, Any]) -> int | None:
    param_type, has_range, has_categorical_values = _dict_param_shape(definition)

    if _is_categorical_shape(param_type, has_range, has_categorical_values):
        choices = _categorical_choices(definition)
        try:
            return len(choices)
        except TypeError:
            return None

    if param_type in _FIXED_TYPES:
        return 1

    if has_range:
        return _range_cardinality(definition, param_type)

    return 1


def _dict_values(definition: dict[Any, Any]) -> tuple[Any, ...] | None:
    param_type, has_range, has_categorical_values = _dict_param_shape(definition)

    if _is_categorical_shape(param_type, has_range, has_categorical_values):
        choices = _categorical_choices(definition)
        return tuple(choices)

    if param_type in _FIXED_TYPES:
        return _fixed_values(definition)

    if has_range:
        return _range_values(definition, param_type)

    return (definition,)


def discrete_cardinality_for_config_param(definition: Any) -> int | None:
    """Return finite cardinality for a config-space parameter, or None if continuous."""

    if isinstance(definition, list):
        return len(definition)

    if isinstance(definition, tuple) and len(definition) == 2:
        return None

    if isinstance(definition, dict):
        return _dict_cardinality(definition)

    return 1


def discrete_values_for_config_param(definition: Any) -> tuple[Any, ...] | None:
    """Return finite values for a config-space parameter, or None if continuous."""

    if isinstance(definition, list):
        return tuple(definition)

    if isinstance(definition, tuple) and len(definition) == 2:
        return None

    if isinstance(definition, dict):
        return _dict_values(definition)

    return (definition,)
