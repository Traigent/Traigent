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


def discrete_cardinality_for_config_param(definition: Any) -> int | None:
    """Return finite cardinality for a config-space parameter, or None if continuous."""

    if isinstance(definition, list):
        return len(definition)

    if isinstance(definition, tuple) and len(definition) == 2:
        return None

    if isinstance(definition, dict):
        param_type = str(definition.get("type") or "").lower()
        has_range = "low" in definition and "high" in definition
        has_categorical_values = "choices" in definition or "values" in definition

        if param_type in {"categorical", "choice"} or (
            not param_type and not has_range and has_categorical_values
        ):
            choices = (
                definition["choices"]
                if "choices" in definition
                else definition.get("values", ())
            )
            try:
                return len(choices)
            except TypeError:
                return None

        if param_type in {"fixed", "constant"}:
            return 1

        if has_range:
            low = definition.get("low")
            high = definition.get("high")
            if low is None or high is None:
                return None

            if param_type in {"int", "integer"}:
                step = definition.get("step", 1)
                if step is None:
                    step = 1
                try:
                    low_i = int(low)
                    high_i = int(high)
                    step_i = int(step)
                except (TypeError, ValueError):
                    return None
                if step_i <= 0 or low_i > high_i:
                    return None
                return stepped_int_cardinality(low_i, high_i, step_i)

            step = definition.get("step")
            if step is None:
                return None
            try:
                low_f = float(low)
                high_f = float(high)
                step_f = float(step)
            except (TypeError, ValueError):
                return None
            if step_f <= 0 or low_f > high_f:
                return None
            return stepped_float_cardinality(low_f, high_f, step_f)

        return 1

    return 1


def discrete_values_for_config_param(definition: Any) -> tuple[Any, ...] | None:
    """Return finite values for a config-space parameter, or None if continuous."""

    if isinstance(definition, list):
        return tuple(definition)

    if isinstance(definition, tuple) and len(definition) == 2:
        return None

    if isinstance(definition, dict):
        param_type = str(definition.get("type") or "").lower()
        has_range = "low" in definition and "high" in definition
        has_categorical_values = "choices" in definition or "values" in definition

        if param_type in {"categorical", "choice"} or (
            not param_type and not has_range and has_categorical_values
        ):
            choices = (
                definition["choices"]
                if "choices" in definition
                else definition.get("values", ())
            )
            return tuple(choices)

        if param_type in {"fixed", "constant"}:
            if "value" in definition:
                return (definition["value"],)
            if "default" in definition:
                return (definition["default"],)
            return (definition,)

        if has_range:
            low = definition.get("low")
            high = definition.get("high")
            if low is None or high is None:
                return None

            if param_type in {"int", "integer"}:
                step = definition.get("step", 1)
                if step is None:
                    step = 1
                try:
                    low_i = int(low)
                    high_i = int(high)
                    step_i = int(step)
                except (TypeError, ValueError):
                    return None
                if step_i <= 0 or low_i > high_i:
                    return ()
                return stepped_int_values(low_i, high_i, step_i)

            step = definition.get("step")
            if step is None:
                return None
            try:
                low_f = float(low)
                high_f = float(high)
                step_f = float(step)
            except (TypeError, ValueError):
                return None
            if step_f <= 0 or low_f > high_f:
                return ()
            return stepped_float_values(low_f, high_f, step_f)

        return (definition,)

    return (definition,)
