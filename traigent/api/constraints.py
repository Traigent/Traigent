"""Constraint builder classes for TVL structural constraints.

This module provides SE-friendly classes for expressing structural constraints
on configuration spaces. Constraints define valid parameter combinations and
are enforced during optimization.

The constraint system uses a builder pattern where ParameterRange objects
(Range, IntRange, Choices) create Condition objects via builder methods,
which are then combined into Constraint objects.

Example:
    >>> from traigent import Range, IntRange, Choices, implies
    >>>
    >>> # Define parameters
    >>> temperature = Range(0.0, 2.0)
    >>> max_tokens = IntRange(100, 4096)
    >>> model = Choices(["gpt-4", "gpt-3.5-turbo"])
    >>>
    >>> # Define constraints using builder pattern
    >>> constraints = [
    ...     # When using gpt-4, require high token limit
    ...     implies(model.equals("gpt-4"), max_tokens.gte(1000)),
    ...     # Temperature must be low for turbo models
    ...     implies(model.equals("gpt-3.5-turbo"), temperature.lte(0.7)),
    ... ]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from traigent.api.parameter_ranges import ParameterRange
    from traigent.tvl.models import StructuralConstraint

# Supported comparison operators
OperatorType = Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "in_range"]


@dataclass(frozen=True, slots=True)
class Condition:
    """A single condition in a constraint expression.

    Conditions are created via builder methods on ParameterRange objects
    (Range, IntRange, Choices) and represent atomic predicates like
    "temperature <= 0.7" or "model == 'gpt-4'".

    Attributes:
        tvar: The ParameterRange this condition applies to
        operator: Comparison operator ("==", ">=", "<=", etc.)
        value: The value to compare against

    Example:
        >>> temp = Range(0.0, 2.0)
        >>> cond = temp.lte(0.7)  # Creates Condition(temp, "<=", 0.7)
    """

    tvar: ParameterRange
    operator: OperatorType
    value: Any

    def to_expression(self, var_name: str) -> str:
        """Convert to TVL expression string.

        Args:
            var_name: The variable name to use in the expression

        Returns:
            A string expression like "params.temperature <= 0.7"
        """
        if self.operator == "in_range":
            low, high = self.value
            return f"({var_name} >= {low}) and ({var_name} <= {high})"
        elif self.operator == "in":
            return f"{var_name} in {self.value!r}"
        elif self.operator == "not_in":
            return f"{var_name} not in {self.value!r}"
        elif self.operator == "==":
            return f"{var_name} == {self.value!r}"
        elif self.operator == "!=":
            return f"{var_name} != {self.value!r}"
        else:
            # Numeric comparisons: >, >=, <, <=
            return f"{var_name} {self.operator} {self.value}"

    def evaluate(self, value: Any) -> bool:
        """Evaluate this condition against a concrete value.

        Args:
            value: The actual parameter value to check

        Returns:
            True if the condition is satisfied
        """
        if self.operator == "==":
            return value == self.value
        elif self.operator == "!=":
            return value != self.value
        elif self.operator == ">":
            return value > self.value
        elif self.operator == ">=":
            return value >= self.value
        elif self.operator == "<":
            return value < self.value
        elif self.operator == "<=":
            return value <= self.value
        elif self.operator == "in":
            return value in self.value
        elif self.operator == "not_in":
            return value not in self.value
        elif self.operator == "in_range":
            low, high = self.value
            return low <= value <= high
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


@dataclass(frozen=True)
class Constraint:
    """Structural constraint with when/then (implication) semantics.

    A constraint can be either:
    - An implication: when A is true, then B must be true (A -> B)
    - A standalone expression: condition must always be true

    The implication "when A then B" is equivalent to "not(A) or B",
    meaning the constraint is satisfied if either:
    - The 'when' condition is false (constraint doesn't apply), or
    - The 'then' condition is true (constraint is satisfied)

    Attributes:
        when: Guard/antecedent condition (for implications)
        then: Consequent condition (for implications)
        expr: Standalone condition (mutually exclusive with when/then)
        description: Optional human-readable description
        id: Optional identifier for the constraint

    Example:
        >>> # Implication: if model is gpt-4, temperature must be <= 0.7
        >>> c = Constraint(
        ...     when=model.equals("gpt-4"),
        ...     then=temperature.lte(0.7),
        ...     description="GPT-4 requires low temperature"
        ... )
        >>>
        >>> # Standalone: temperature must always be positive
        >>> c = Constraint(expr=temperature.gt(0))
    """

    when: Condition | None = None
    then: Condition | None = None
    expr: Condition | None = None
    description: str | None = None
    id: str | None = None

    def __post_init__(self) -> None:
        """Validate constraint structure."""
        has_implication = self.when is not None and self.then is not None
        has_expr = self.expr is not None

        if not has_implication and not has_expr:
            raise ValueError(
                "Constraint requires either (when, then) or expr. "
                "Use implies(when, then) or Constraint(expr=condition)."
            )

        if has_implication and has_expr:
            raise ValueError(
                "Constraint cannot have both (when, then) and expr. "
                "Use either implication or standalone expression, not both."
            )

    @property
    def is_implication(self) -> bool:
        """Return True if this is an implication constraint."""
        return self.when is not None and self.then is not None

    def to_expression(self, var_names: dict[int, str]) -> str:
        """Convert to TVL expression string.

        Args:
            var_names: Mapping from ParameterRange id() to variable name.
                Uses id() for identity-based lookup to avoid collision when
                two ParameterRange instances have identical values.

        Returns:
            A string expression representing the constraint
        """
        if self.expr is not None:
            tvar = self.expr.tvar
            var_name = var_names.get(id(tvar), f"params.{tvar.name or 'unknown'}")
            return self.expr.to_expression(var_name)

        # Implication: not(when) or then
        assert self.when is not None and self.then is not None
        when_var = var_names.get(
            id(self.when.tvar), f"params.{self.when.tvar.name or 'unknown'}"
        )
        then_var = var_names.get(
            id(self.then.tvar), f"params.{self.then.tvar.name or 'unknown'}"
        )
        when_expr = self.when.to_expression(when_var)
        then_expr = self.then.to_expression(then_var)
        return f"not ({when_expr}) or ({then_expr})"

    def evaluate(self, config: dict[str, Any], var_names: dict[int, str]) -> bool:
        """Evaluate this constraint against a configuration.

        Args:
            config: The configuration dict with parameter values
            var_names: Mapping from ParameterRange id() to config key.
                Use id(tvar) as key to avoid value-based collision.

        Returns:
            True if the constraint is satisfied
        """
        if self.expr is not None:
            var_name = var_names.get(id(self.expr.tvar))
            if var_name is None or var_name not in config:
                return True  # Missing value - constraint doesn't apply
            return self.expr.evaluate(config[var_name])

        # Implication: not(when) or then
        assert self.when is not None and self.then is not None

        when_var = var_names.get(id(self.when.tvar))
        then_var = var_names.get(id(self.then.tvar))

        # If variables are missing, constraint doesn't apply
        if when_var is None or when_var not in config:
            return True
        if then_var is None or then_var not in config:
            return True

        when_result = self.when.evaluate(config[when_var])
        if not when_result:
            return True  # When is false, implication is satisfied

        return self.then.evaluate(config[then_var])

    def to_structural_constraint(
        self, var_names: dict[int, str]
    ) -> StructuralConstraint:
        """Convert to TVL StructuralConstraint.

        Args:
            var_names: Mapping from ParameterRange id() to variable name.
                Uses id() for identity-based lookup to avoid collision when
                two ParameterRange instances have identical values.

        Returns:
            A TVL StructuralConstraint instance
        """
        from traigent.tvl.models import StructuralConstraint

        if self.expr is not None:
            var_name = var_names.get(
                id(self.expr.tvar), f"params.{self.expr.tvar.name or 'unknown'}"
            )
            return StructuralConstraint(expr=self.expr.to_expression(var_name))

        assert self.when is not None and self.then is not None
        when_var = var_names.get(
            id(self.when.tvar), f"params.{self.when.tvar.name or 'unknown'}"
        )
        then_var = var_names.get(
            id(self.then.tvar), f"params.{self.then.tvar.name or 'unknown'}"
        )
        return StructuralConstraint(
            when=self.when.to_expression(when_var),
            then=self.then.to_expression(then_var),
        )

    def to_callable(
        self, var_names: dict[int, str] | None = None
    ) -> Callable[[dict[str, Any]], bool]:
        """Convert to a callable for use with the optimize decorator.

        This method creates a function compatible with Traigent's constraint
        system, which accepts a config dict and returns True if the constraint
        is satisfied.

        Args:
            var_names: Optional mapping from ParameterRange id() to config key.
                If not provided, uses the name attribute of each ParameterRange.
                Uses id() for identity-based lookup to avoid collision when
                two ParameterRange instances have identical values.

        Returns:
            A callable that takes a config dict and returns bool.

        Raises:
            ValueError: If a ParameterRange in the constraint has no name and
                var_names is not provided.

        Example:
            >>> from traigent import Range, Choices, implies
            >>>
            >>> model = Choices(["gpt-4", "gpt-3.5"], name="model")
            >>> temp = Range(0.0, 2.0, name="temperature")
            >>> constraint = implies(model.equals("gpt-4"), temp.lte(0.7))
            >>>
            >>> # Convert to callable for decorator
            >>> constraint_fn = constraint.to_callable()
            >>> constraint_fn({"model": "gpt-4", "temperature": 0.5})  # True
        """
        import warnings

        # Build var_names from tvar.name if not provided
        if var_names is None:
            var_names = {}
            missing_names: list[str] = []

            if self.expr is not None:
                if self.expr.tvar.name:
                    var_names[id(self.expr.tvar)] = self.expr.tvar.name
                else:
                    missing_names.append("expr.tvar")
            else:
                if self.when is not None:
                    if self.when.tvar.name:
                        var_names[id(self.when.tvar)] = self.when.tvar.name
                    else:
                        missing_names.append("when.tvar")
                if self.then is not None:
                    if self.then.tvar.name:
                        var_names[id(self.then.tvar)] = self.then.tvar.name
                    else:
                        missing_names.append("then.tvar")

            if missing_names:
                warnings.warn(
                    f"Constraint has ParameterRange(s) without names: {missing_names}. "
                    "This may cause constraint evaluation to fail. "
                    "Set the 'name' attribute on your Range/Choices/IntRange objects, "
                    "or provide explicit var_names mapping.",
                    UserWarning,
                    stacklevel=2,
                )

        # Capture var_names in closure
        captured_var_names = dict(var_names)

        def constraint_fn(config: dict[str, Any]) -> bool:
            """Evaluate the structural constraint against the given configuration."""
            return self.evaluate(config, captured_var_names)

        # Add metadata for debugging
        constraint_fn.__doc__ = self.description or "Structural constraint"
        constraint_fn.__name__ = f"constraint_{self.id or 'unnamed'}"

        return constraint_fn


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def implies(
    when: Condition,
    then: Condition,
    description: str | None = None,
    id: str | None = None,
) -> Constraint:
    """Create an implication constraint: when -> then.

    This is the most common way to express structural constraints.
    The constraint is satisfied if either:
    - The 'when' condition is false (constraint doesn't apply), or
    - The 'then' condition is true (constraint is satisfied)

    Args:
        when: The guard/antecedent condition
        then: The consequent condition that must hold when 'when' is true
        description: Optional human-readable description
        id: Optional identifier for the constraint

    Returns:
        A Constraint with implication semantics

    Example:
        >>> from traigent import Range, Choices, implies
        >>>
        >>> model = Choices(["gpt-4", "gpt-3.5"])
        >>> temp = Range(0.0, 2.0)
        >>>
        >>> # If model is gpt-4, temperature must be <= 0.7
        >>> c = implies(model.equals("gpt-4"), temp.lte(0.7))
    """
    return Constraint(when=when, then=then, description=description, id=id)


def require(
    condition: Condition,
    description: str | None = None,
    id: str | None = None,
) -> Constraint:
    """Create a standalone constraint that must always hold.

    Unlike implications, this constraint always applies regardless
    of other parameter values.

    Args:
        condition: The condition that must always be true
        description: Optional human-readable description
        id: Optional identifier for the constraint

    Returns:
        A Constraint with the given condition as a standalone requirement

    Example:
        >>> from traigent import Range, require
        >>>
        >>> temp = Range(0.0, 2.0)
        >>> # Temperature must always be <= 1.5
        >>> c = require(temp.lte(1.5))
    """
    return Constraint(expr=condition, description=description, id=id)


# =============================================================================
# Compound Conditions (for future extension)
# =============================================================================


@dataclass(frozen=True)
class AndCondition:
    """Conjunction of multiple conditions (all must be true).

    Note: This is a placeholder for future extension. Currently,
    compound conditions should be expressed as multiple constraints.
    """

    conditions: tuple[Condition, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.conditions) < 2:
            raise ValueError("AndCondition requires at least 2 conditions")


@dataclass(frozen=True)
class OrCondition:
    """Disjunction of multiple conditions (at least one must be true).

    Note: This is a placeholder for future extension. Currently,
    compound conditions should be expressed as multiple constraints.
    """

    conditions: tuple[Condition, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.conditions) < 2:
            raise ValueError("OrCondition requires at least 2 conditions")


def constraints_to_callables(
    constraints: list[Constraint],
    var_names: dict[int, str] | None = None,
) -> list[Callable[[dict[str, Any]], bool]]:
    """Convert a list of Constraint objects to callable functions.

    This utility function converts structural constraints to the callable
    format expected by the @optimize decorator.

    Args:
        constraints: List of Constraint objects to convert
        var_names: Optional mapping from ParameterRange id() to config key.
            If not provided, uses the name attribute of each ParameterRange.
            Uses id() for identity-based lookup to avoid collision when
            two ParameterRange instances have identical values.

    Returns:
        List of callable constraint functions

    Example:
        >>> from traigent import Range, Choices, implies, constraints_to_callables
        >>>
        >>> model = Choices(["gpt-4", "gpt-3.5"], name="model")
        >>> temp = Range(0.0, 2.0, name="temperature")
        >>>
        >>> constraints = [
        ...     implies(model.equals("gpt-4"), temp.lte(0.7)),
        ... ]
        >>>
        >>> # Convert for use with decorator
        >>> constraint_fns = constraints_to_callables(constraints)
    """
    return [c.to_callable(var_names) for c in constraints]


__all__ = [
    "AndCondition",
    "Condition",
    "Constraint",
    "OperatorType",
    "OrCondition",
    "constraints_to_callables",
    "implies",
    "require",
]
