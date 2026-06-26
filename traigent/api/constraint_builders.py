"""Constraint builder mixins for ParameterRange classes.

This module provides mixin classes that add constraint builder methods
to ParameterRange subclasses. Using mixins eliminates code duplication
across Range, IntRange, LogRange, and Choices classes.

Example:
    >>> from traigent.api.parameter_ranges import Range, Choices
    >>> temp = Range(0.0, 1.0)
    >>> cond = temp.lte(0.7)  # Uses NumericConstraintBuilderMixin.lte()
    >>> model = Choices(["gpt-4", "gpt-3.5"])
    >>> cond = model.is_in(["gpt-4"])  # Uses CategoricalConstraintBuilderMixin.is_in()
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from traigent.api.constraints import Condition

T = TypeVar("T")


__all__ = [
    "NumericConstraintBuilderMixin",
    "CategoricalConstraintBuilderMixin",
]


def _parameter_label(tvar: Any) -> str:
    return str(getattr(tvar, "name", None) or "parameter")


def _numeric_bounds(tvar: Any) -> tuple[float, float] | None:
    low = getattr(tvar, "low", None)
    high = getattr(tvar, "high", None)
    if low is None or high is None:
        return None
    return float(low), float(high)


def _validate_numeric_value(tvar: Any, value: float, operator: str) -> None:
    bounds = _numeric_bounds(tvar)
    if bounds is None:
        return
    low, high = bounds
    numeric_value = float(value)
    if not low <= numeric_value <= high:
        raise ValueError(
            f"{_parameter_label(tvar)}.{operator}({value!r}) is outside "
            f"the parameter domain [{low}, {high}]"
        )


def _validate_numeric_values(
    tvar: Any, values: Sequence[float], operator: str
) -> tuple[float, ...]:
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{_parameter_label(tvar)}.{operator}() cannot be empty")
    for value in normalized:
        _validate_numeric_value(tvar, value, operator)
    return normalized


def _validate_numeric_range(tvar: Any, low_value: float, high_value: float) -> None:
    if low_value > high_value:
        raise ValueError(
            f"{_parameter_label(tvar)}.in_range() lower bound must be <= upper bound"
        )
    bounds = _numeric_bounds(tvar)
    if bounds is None:
        return
    low, high = bounds
    if high_value < low or low_value > high:
        raise ValueError(
            f"{_parameter_label(tvar)}.in_range({low_value!r}, {high_value!r}) "
            f"does not overlap the parameter domain [{low}, {high}]"
        )


def _choices(tvar: Any) -> tuple[Any, ...] | None:
    values = getattr(tvar, "values", None)
    if values is None:
        return None
    return tuple(values)


def _validate_choice_value(tvar: Any, value: Any, operator: str) -> None:
    values = _choices(tvar)
    if values is None:
        return
    if value not in values:
        raise ValueError(
            f"{_parameter_label(tvar)}.{operator}({value!r}) is outside "
            f"the parameter choices {list(values)!r}"
        )


def _validate_choice_values(
    tvar: Any, values: Sequence[Any], operator: str
) -> tuple[Any, ...]:
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{_parameter_label(tvar)}.{operator}() cannot be empty")
    for value in normalized:
        _validate_choice_value(tvar, value, operator)
    return normalized


class NumericConstraintBuilderMixin:
    """Mixin providing constraint builder methods for numeric ParameterRanges.

    This mixin is used by Range, IntRange, and LogRange to provide
    comparison-based constraint methods without code duplication.

    Methods:
        equals: Create equality condition
        not_equals: Create inequality condition
        gt: Create greater-than condition
        gte: Create greater-than-or-equal condition
        lt: Create less-than condition
        lte: Create less-than-or-equal condition
        in_range: Create between condition (inclusive)
        is_in: Create membership condition for discrete values
        not_in: Create exclusion condition for discrete values
    """

    def equals(self, value: float) -> Condition:
        """Create condition: this parameter equals value.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.equals(0.5)  # temp == 0.5
        """
        from traigent.api.constraints import Condition

        _validate_numeric_value(self, value, "equals")
        return Condition(_tvar=self, operator="==", value=value)  # type: ignore[arg-type]

    def not_equals(self, value: float) -> Condition:
        """Create condition: this parameter does not equal value.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.not_equals(0.5)  # temp != 0.5
        """
        from traigent.api.constraints import Condition

        _validate_numeric_value(self, value, "not_equals")
        return Condition(_tvar=self, operator="!=", value=value)  # type: ignore[arg-type]

    def gt(self, value: float) -> Condition:
        """Create condition: this parameter > value.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.gt(0.5)  # temp > 0.5
        """
        from traigent.api.constraints import Condition

        _validate_numeric_value(self, value, "gt")
        return Condition(_tvar=self, operator=">", value=value)  # type: ignore[arg-type]

    def gte(self, value: float) -> Condition:
        """Create condition: this parameter >= value.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.gte(0.5)  # temp >= 0.5
        """
        from traigent.api.constraints import Condition

        _validate_numeric_value(self, value, "gte")
        return Condition(_tvar=self, operator=">=", value=value)  # type: ignore[arg-type]

    def lt(self, value: float) -> Condition:
        """Create condition: this parameter < value.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.lt(0.5)  # temp < 0.5
        """
        from traigent.api.constraints import Condition

        _validate_numeric_value(self, value, "lt")
        return Condition(_tvar=self, operator="<", value=value)  # type: ignore[arg-type]

    def lte(self, value: float) -> Condition:
        """Create condition: this parameter <= value.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.lte(0.5)  # temp <= 0.5
        """
        from traigent.api.constraints import Condition

        _validate_numeric_value(self, value, "lte")
        return Condition(_tvar=self, operator="<=", value=value)  # type: ignore[arg-type]

    def in_range(self, low: float, high: float) -> Condition:
        """Create condition: low <= this parameter <= high.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.in_range(0.3, 0.7)  # 0.3 <= temp <= 0.7
        """
        from traigent.api.constraints import Condition

        _validate_numeric_range(self, low, high)
        return Condition(_tvar=self, operator="in_range", value=(low, high))  # type: ignore[arg-type]

    def is_in(self, values: Sequence[float]) -> Condition:
        """Create condition: this parameter is in the given discrete values.

        Useful for constraining numeric parameters to specific values.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.is_in([0.0, 0.5, 1.0])  # temp in [0.0, 0.5, 1.0]
        """
        from traigent.api.constraints import Condition

        normalized = _validate_numeric_values(self, values, "is_in")
        return Condition(_tvar=self, operator="in", value=normalized)  # type: ignore[arg-type]

    def not_in(self, values: Sequence[float]) -> Condition:
        """Create condition: this parameter is not in the given values.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.not_in([0.0, 1.0])  # temp not in [0.0, 1.0]
        """
        from traigent.api.constraints import Condition

        normalized = _validate_numeric_values(self, values, "not_in")
        return Condition(_tvar=self, operator="not_in", value=normalized)  # type: ignore[arg-type]


class CategoricalConstraintBuilderMixin:
    """Mixin providing constraint builder methods for categorical ParameterRanges.

    This mixin is used by Choices to provide equality and membership
    constraint methods without code duplication.

    Methods:
        equals: Create equality condition
        not_equals: Create inequality condition
        is_in: Create membership condition
        not_in: Create exclusion condition
    """

    def equals(self, value: Any) -> Condition:
        """Create condition: this parameter equals value.

        Example:
            >>> model = Choices(["gpt-4", "gpt-3.5"])
            >>> cond = model.equals("gpt-4")  # model == "gpt-4"
        """
        from traigent.api.constraints import Condition

        _validate_choice_value(self, value, "equals")
        return Condition(_tvar=self, operator="==", value=value)  # type: ignore[arg-type]

    def not_equals(self, value: Any) -> Condition:
        """Create condition: this parameter does not equal value.

        Example:
            >>> model = Choices(["gpt-4", "gpt-3.5"])
            >>> cond = model.not_equals("gpt-4")  # model != "gpt-4"
        """
        from traigent.api.constraints import Condition

        _validate_choice_value(self, value, "not_equals")
        return Condition(_tvar=self, operator="!=", value=value)  # type: ignore[arg-type]

    def is_in(self, values: Sequence[Any]) -> Condition:
        """Create condition: this parameter is in the given values.

        Example:
            >>> model = Choices(["gpt-4", "gpt-3.5", "claude"])
            >>> cond = model.is_in(["gpt-4", "gpt-3.5"])  # model in ["gpt-4", "gpt-3.5"]
        """
        from traigent.api.constraints import Condition

        normalized = _validate_choice_values(self, values, "is_in")
        return Condition(_tvar=self, operator="in", value=normalized)  # type: ignore[arg-type]

    def not_in(self, values: Sequence[Any]) -> Condition:
        """Create condition: this parameter is not in the given values.

        Example:
            >>> model = Choices(["gpt-4", "gpt-3.5", "claude"])
            >>> cond = model.not_in(["claude"])  # model not in ["claude"]
        """
        from traigent.api.constraints import Condition

        normalized = _validate_choice_values(self, values, "not_in")
        return Condition(_tvar=self, operator="not_in", value=normalized)  # type: ignore[arg-type]
