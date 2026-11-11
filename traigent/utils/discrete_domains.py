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


def _finite_grid(low: float, high: float, step: float) -> bool:
    return math.isfinite(low) and math.isfinite(high) and math.isfinite(step)


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


def _grid_round_digits(step: float) -> int:
    """Decimal digits that keep distinct grid points distinct for ``step``.

    The default of 6 digits is kept for ordinary steps; a step below 1e-3 needs
    more, otherwise rounding collapses neighbouring points or rounds the first
    point below ``low`` to ``0.0`` (#2822).
    """

    return max(FLOAT_GRID_ROUND_DIGITS, math.ceil(-math.log10(step)) + 3)


def stepped_float_values(low: float, high: float, step: float) -> tuple[float, ...]:
    """Build the stepped float grid, preserving the inclusive high endpoint."""

    if not _finite_grid(low, high, step) or step <= 0 or low > high:
        return ()
    digits = _grid_round_digits(step)

    values: list[float] = []
    epsilon = _FLOAT_STEP_TOLERANCE * max(1.0, abs(high - low), abs(step))
    index = 0
    while True:
        value = low + (index * step)
        if value > high + epsilon:
            break
        value = min(max(value, low), high)
        rounded = round(value, digits)
        if not values or values[-1] != rounded:
            values.append(rounded)
        index += 1

    high_value = round(high, digits)
    if not values or values[-1] != high_value:
        values.append(high_value)
    return tuple(values)


def stepped_float_cardinality(low: float, high: float, step: float) -> int:
    """Count stepped float grid points without materializing them."""

    if step <= 0 or low > high:
        return 0
    if not (math.isfinite(low) and math.isfinite(high) and math.isfinite(step)):
        return 0
    digits = _grid_round_digits(step)

    epsilon = _FLOAT_STEP_TOLERANCE * max(1.0, abs(high - low), abs(step))
    interval_count = (high - low + epsilon) / step
    if not math.isfinite(interval_count):
        return sys.maxsize

    count = max(math.floor(interval_count) + 1, 1)
    value = low + ((count - 1) * step)
    rounded = round(min(max(value, low), high), digits)
    high_value = round(high, digits)
    if rounded != high_value:
        count += 1
    return count


# Grids up to this many points keep the materialized-tuple draw, so a seeded
# run over a small grid produces the same sequence as before (#3193).
STEPPED_FLOAT_LIST_SAMPLE_CAP = 10_000


def sample_stepped_float(rng: Any, low: float, high: float, step: float) -> float:
    """Draw from the stepped float grid without materializing a large grid.

    Small grids (at most ``STEPPED_FLOAT_LIST_SAMPLE_CAP`` points) keep the
    original ``rng.choice`` over :func:`stepped_float_values`, so seeded
    sequences are unchanged. Larger grids draw one integer index and map it to
    ``low + index * step`` with the same clamping and rounding that
    :func:`stepped_float_values` applies, so the value is a grid member.

    Known limit: on the indexed path, neighbouring indices that rounding
    collapses into one grid value (only when ``|low|`` is so large that
    ``index * step`` loses precision) are drawn in proportion to their index
    count, not uniformly.

    Raises:
        ValueError: if the inputs are non-finite, ``step <= 0``, the range is
            empty, or the grid is too large to index.
    """

    count = stepped_float_cardinality(low, high, step)
    if count <= 0:
        raise ValueError("float range has no values")
    if count <= STEPPED_FLOAT_LIST_SAMPLE_CAP:
        values = stepped_float_values(low, high, step)
        if not values:
            raise ValueError("float range has no values")
        return float(rng.choice(values))
    if count == sys.maxsize:  # non-finite interval count
        raise ValueError("float range is too large to sample")

    digits = _grid_round_digits(step)
    # Index only the points at or below ``high`` (no tolerance), so grid points
    # that merely clamp to ``high`` are not counted as extra draws; ``high``
    # itself is one more entry when the last regular point rounds elsewhere.
    regular = math.floor((high - low) / step) + 1
    last = round(min(max(low + (regular - 1) * step, low), high), digits)
    high_value = round(high, digits)
    total = regular + (1 if last != high_value else 0)
    index = rng.randrange(total)
    if index >= regular:
        return float(high_value)
    value = min(max(low + index * step, low), high)
    return float(round(value, digits))


def sample_stepped_int(rng: Any, low: int, high: int, step: int) -> int:
    """Draw from the integer grid without materializing it.

    The grid is ``low, low+step, ...`` plus the inclusive ``high`` endpoint when
    it is off-grid. Uses one ``randrange`` call, so seeded sequences match
    ``rng.choice`` over the materialized grid.
    """

    count = stepped_int_cardinality(low, high, step)
    if count <= 0:
        raise ValueError("integer range has no values")
    index = rng.randrange(count)
    regular = (high - low) // step + 1
    return low + index * step if index < regular else high


def uniform_within(rng: Any, low: float, high: float, digits: int = 6) -> float:
    """Uniform draw rounded to ``digits``, never leaving ``[low, high]``.

    The raw draw is kept when rounding would push it outside the bounds
    (e.g. a range narrower than the rounding precision).
    """

    raw = rng.uniform(low, high)
    rounded = round(raw, digits)
    return float(rounded if low <= rounded <= high else raw)


def sample_range_dict(rng: Any, definition: dict[Any, Any]) -> int | float:
    """Sample one value from a typed ``{"low", "high", ...}`` range dict."""

    low, high = definition["low"], definition["high"]
    step = definition.get("step")
    if str(definition.get("type", "")).lower() in _INT_TYPES:
        return sample_stepped_int(rng, int(low), int(high), int(step or 1))
    low_f, high_f = float(low), float(high)
    if step is not None:
        return sample_stepped_float(rng, low_f, high_f, float(step))
    return uniform_within(rng, low_f, high_f)


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
