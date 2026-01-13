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

        return Condition(_tvar=self, operator="==", value=value)  # type: ignore[arg-type]

    def not_equals(self, value: float) -> Condition:
        """Create condition: this parameter does not equal value.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.not_equals(0.5)  # temp != 0.5
        """
        from traigent.api.constraints import Condition

        return Condition(_tvar=self, operator="!=", value=value)  # type: ignore[arg-type]

    def gt(self, value: float) -> Condition:
        """Create condition: this parameter > value.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.gt(0.5)  # temp > 0.5
        """
        from traigent.api.constraints import Condition

        return Condition(_tvar=self, operator=">", value=value)  # type: ignore[arg-type]

    def gte(self, value: float) -> Condition:
        """Create condition: this parameter >= value.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.gte(0.5)  # temp >= 0.5
        """
        from traigent.api.constraints import Condition

        return Condition(_tvar=self, operator=">=", value=value)  # type: ignore[arg-type]

    def lt(self, value: float) -> Condition:
        """Create condition: this parameter < value.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.lt(0.5)  # temp < 0.5
        """
        from traigent.api.constraints import Condition

        return Condition(_tvar=self, operator="<", value=value)  # type: ignore[arg-type]

    def lte(self, value: float) -> Condition:
        """Create condition: this parameter <= value.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.lte(0.5)  # temp <= 0.5
        """
        from traigent.api.constraints import Condition

        return Condition(_tvar=self, operator="<=", value=value)  # type: ignore[arg-type]

    def in_range(self, low: float, high: float) -> Condition:
        """Create condition: low <= this parameter <= high.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.in_range(0.3, 0.7)  # 0.3 <= temp <= 0.7
        """
        from traigent.api.constraints import Condition

        return Condition(_tvar=self, operator="in_range", value=(low, high))  # type: ignore[arg-type]

    def is_in(self, values: Sequence[float]) -> Condition:
        """Create condition: this parameter is in the given discrete values.

        Useful for constraining numeric parameters to specific values.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.is_in([0.0, 0.5, 1.0])  # temp in [0.0, 0.5, 1.0]
        """
        from traigent.api.constraints import Condition

        return Condition(_tvar=self, operator="in", value=tuple(values))  # type: ignore[arg-type]

    def not_in(self, values: Sequence[float]) -> Condition:
        """Create condition: this parameter is not in the given values.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.not_in([0.0, 1.0])  # temp not in [0.0, 1.0]
        """
        from traigent.api.constraints import Condition

        return Condition(_tvar=self, operator="not_in", value=tuple(values))  # type: ignore[arg-type]


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

        return Condition(_tvar=self, operator="==", value=value)  # type: ignore[arg-type]

    def not_equals(self, value: Any) -> Condition:
        """Create condition: this parameter does not equal value.

        Example:
            >>> model = Choices(["gpt-4", "gpt-3.5"])
            >>> cond = model.not_equals("gpt-4")  # model != "gpt-4"
        """
        from traigent.api.constraints import Condition

        return Condition(_tvar=self, operator="!=", value=value)  # type: ignore[arg-type]

    def is_in(self, values: Sequence[Any]) -> Condition:
        """Create condition: this parameter is in the given values.

        Example:
            >>> model = Choices(["gpt-4", "gpt-3.5", "claude"])
            >>> cond = model.is_in(["gpt-4", "gpt-3.5"])  # model in ["gpt-4", "gpt-3.5"]
        """
        from traigent.api.constraints import Condition

        return Condition(_tvar=self, operator="in", value=tuple(values))  # type: ignore[arg-type]

    def not_in(self, values: Sequence[Any]) -> Condition:
        """Create condition: this parameter is not in the given values.

        Example:
            >>> model = Choices(["gpt-4", "gpt-3.5", "claude"])
            >>> cond = model.not_in(["claude"])  # model not in ["claude"]
        """
        from traigent.api.constraints import Condition

        return Condition(_tvar=self, operator="not_in", value=tuple(values))  # type: ignore[arg-type]
