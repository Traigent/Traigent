"""Constraint builder classes for TVL structural constraints.

This module provides SE-friendly classes for expressing structural constraints
on configuration spaces. Constraints define valid parameter combinations and
are enforced during optimization.

The constraint system uses a builder pattern where ParameterRange objects
(Range, IntRange, Choices) create Condition objects via builder methods,
which are then combined into Constraint objects.

Supports three syntax styles for constraints:

1. Functional (canonical, explicit):
    >>> implies(model.equals("gpt-4"), temp.lte(0.7))

2. Operator-based (concise, formula-like):
    >>> model.equals("gpt-4") >> temp.lte(0.7)

3. Fluent (readable):
    >>> when(model.equals("gpt-4")).then(temp.lte(0.7))

OPERATOR PRECEDENCE WARNING:
    Python precedence: ~ > << >> > & > ^ > |
    This means: a & b >> c  evaluates as  a & (b >> c), NOT (a & b) >> c
    Always use parentheses for clarity:
        (model.equals("gpt-4") & temp.lte(0.7)) >> max_tokens.gte(1000)

Example:
    >>> from traigent import Range, IntRange, Choices, implies
    >>>
    >>> # Define parameters
    >>> temperature = Range(0.0, 2.0)
    >>> max_tokens = IntRange(100, 4096)
    >>> model = Choices(["gpt-4", "gpt-3.5-turbo"])
    >>>
    >>> # Define constraints using any syntax style
    >>> constraints = [
    ...     # Functional style
    ...     implies(model.equals("gpt-4"), max_tokens.gte(1000)),
    ...     # Operator style
    ...     model.equals("gpt-3.5-turbo") >> temperature.lte(0.7),
    ... ]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from traigent.api.parameter_ranges import ParameterRange
    from traigent.tvl.models import StructuralConstraint

# Supported comparison operators
OperatorType = Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "in_range"]


# =============================================================================
# Boolean Expression Base Class
# =============================================================================


class BoolExpr(ABC):
    """Abstract base class for boolean expressions in constraints.

    Provides operator overloading for intuitive constraint syntax:
        - >> for implication (A >> B means "A implies B")
        - & for conjunction (A & B means "A and B")
        - | for disjunction (A | B means "A or B")
        - ~ for negation (~A means "not A")

    OPERATOR PRECEDENCE WARNING:
        Python precedence: ~ > << >> > & > ^ > |
        This means: a & b >> c  evaluates as  a & (b >> c), NOT (a & b) >> c
        Always use parentheses for clarity.
    """

    @property
    def tvar(self) -> ParameterRange | None:
        """Return the parameter range this expression refers to.

        For atomic Condition expressions, returns the ParameterRange.
        For composite expressions (And, Or, Not), returns None.
        """
        return None

    @abstractmethod
    def to_expression(self, var_names: dict[int, str] | str) -> str:
        """Convert to TVL expression string.

        Args:
            var_names: Mapping from ParameterRange id() to variable name.

        Returns:
            A string expression like "params.temperature <= 0.7"
        """
        ...

    @abstractmethod
    def evaluate_config(
        self, config: dict[str, Any], var_names: dict[int, str]
    ) -> bool:
        """Evaluate this expression against a configuration.

        Args:
            config: The configuration dict with parameter values
            var_names: Mapping from ParameterRange id() to config key.

        Returns:
            True if the expression is satisfied
        """
        ...

    def __rshift__(self, other: BoolExpr) -> Constraint:
        """Implication operator: self >> other means 'self implies other'.

        Example:
            >>> model.equals("gpt-4") >> temp.lte(0.7)
        """
        if not isinstance(other, BoolExpr):
            return NotImplemented
        return Constraint(when=self, then=other)

    def __and__(self, other: BoolExpr) -> AndCondition:
        """Conjunction operator: self & other means 'self and other'.

        Example:
            >>> model.equals("gpt-4") & temp.lte(0.7)
        """
        if not isinstance(other, BoolExpr):
            return NotImplemented
        return AndCondition((self, other))

    def __or__(self, other: BoolExpr) -> OrCondition:
        """Disjunction operator: self | other means 'self or other'.

        Example:
            >>> model.equals("gpt-4") | model.equals("gpt-3.5")
        """
        if not isinstance(other, BoolExpr):
            return NotImplemented
        return OrCondition((self, other))

    def __invert__(self) -> NotCondition:
        """Negation operator: ~self means 'not self'.

        Example:
            >>> ~model.equals("gpt-4")
        """
        return NotCondition(self)

    def __bool__(self) -> bool:
        """Prevent accidental boolean evaluation.

        Raises TypeError to catch mistakes like:
            if model.equals("gpt-4"):  # Wrong! Use operators instead.

        Use & | ~ operators instead of 'and', 'or', 'not'.
        """
        raise TypeError(
            "BoolExpr cannot be used in boolean context. "
            "Use & | ~ operators instead of 'and', 'or', 'not'. "
            "Use >> for implication, or wrap in Constraint/implies()."
        )

    def implies(self, other: BoolExpr) -> Constraint:
        """Fluent method for implication: self.implies(other).

        Example:
            >>> model.equals("gpt-4").implies(temp.lte(0.7))
        """
        return Constraint(when=self, then=other)

    @abstractmethod
    def explain(self, var_names: dict[int, str] | None = None) -> str:
        """Return a plain English explanation of this expression.

        Args:
            var_names: Optional mapping from ParameterRange id() to variable name.
                If not provided, uses the name attribute of each ParameterRange.

        Returns:
            Human-readable explanation like "model equals 'gpt-4'"

        Example:
            >>> temp = Range(0.0, 2.0, name="temperature")
            >>> cond = temp.lte(0.7)
            >>> cond.explain()
            "temperature is at most 0.7"
        """
        ...


# =============================================================================
# Atomic Condition
# =============================================================================


@dataclass(frozen=True, slots=True)
class Condition(BoolExpr):
    """A single condition in a constraint expression.

    Conditions are created via builder methods on ParameterRange objects
    (Range, IntRange, Choices) and represent atomic predicates like
    "temperature <= 0.7" or "model == 'gpt-4'".

    Attributes:
        _tvar: The ParameterRange this condition applies to
        operator: Comparison operator ("==", ">=", "<=", etc.)
        value: The value to compare against

    Example:
        >>> temp = Range(0.0, 2.0)
        >>> cond = temp.lte(0.7)  # Creates Condition(temp, "<=", 0.7)
    """

    _tvar: ParameterRange
    operator: OperatorType
    value: Any

    @property
    def tvar(self) -> ParameterRange:
        """Return the parameter range this condition applies to."""
        return self._tvar

    def to_expression(self, var_names: dict[int, str] | str) -> str:
        """Convert to TVL expression string.

        Args:
            var_names: Either a mapping from ParameterRange id() to variable name,
                or a single variable name string (for backward compatibility).

        Returns:
            A string expression like "params.temperature <= 0.7"
        """
        # Backward compatibility: accept single string
        if isinstance(var_names, str):
            var_name = var_names
        else:
            var_name = var_names.get(
                id(self.tvar), f"params.{self.tvar.name or 'unknown'}"
            )
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

    def evaluate_config(
        self, config: dict[str, Any], var_names: dict[int, str]
    ) -> bool:
        """Evaluate against a configuration.

        Args:
            config: The configuration dict with parameter values
            var_names: Mapping from ParameterRange id() to config key.

        Returns:
            True if the condition is satisfied
        """
        var_name = var_names.get(id(self.tvar))
        if var_name is None or var_name not in config:
            return True  # Missing value - constraint doesn't apply
        return self.evaluate(config[var_name])

    def evaluate(self, value: Any) -> bool:
        """Evaluate this condition against a concrete value.

        Args:
            value: The actual parameter value to check

        Returns:
            True if the condition is satisfied
        """
        if self.operator == "==":
            return bool(value == self.value)
        elif self.operator == "!=":
            return bool(value != self.value)
        elif self.operator == ">":
            return bool(value > self.value)
        elif self.operator == ">=":
            return bool(value >= self.value)
        elif self.operator == "<":
            return bool(value < self.value)
        elif self.operator == "<=":
            return bool(value <= self.value)
        elif self.operator == "in":
            return bool(value in self.value)
        elif self.operator == "not_in":
            return bool(value not in self.value)
        elif self.operator == "in_range":
            low, high = self.value
            return bool(low <= value <= high)
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

    def explain(self, var_names: dict[int, str] | None = None) -> str:
        """Return plain English explanation of this condition."""
        # Get variable name
        if var_names and id(self.tvar) in var_names:
            name = var_names[id(self.tvar)]
        elif self.tvar.name:
            name = self.tvar.name
        else:
            name = "parameter"

        # Map operators to plain English (computed on demand)
        if self.operator == "==":
            return f"{name} equals {self.value!r}"
        elif self.operator == "!=":
            return f"{name} is not {self.value!r}"
        elif self.operator == ">":
            return f"{name} is greater than {self.value}"
        elif self.operator == ">=":
            return f"{name} is at least {self.value}"
        elif self.operator == "<":
            return f"{name} is less than {self.value}"
        elif self.operator == "<=":
            return f"{name} is at most {self.value}"
        elif self.operator == "in":
            return f"{name} is one of {list(self.value)}"
        elif self.operator == "not_in":
            return f"{name} is not one of {list(self.value)}"
        elif self.operator == "in_range":
            low, high = self.value
            return f"{name} is between {low} and {high}"
        else:
            return f"{name} {self.operator} {self.value}"


# =============================================================================
# Compound Conditions
# =============================================================================


@dataclass(frozen=True)
class AndCondition(BoolExpr):
    """Conjunction of multiple conditions (all must be true).

    Example:
        >>> model.equals("gpt-4") & temp.lte(0.7)
    """

    conditions: tuple[BoolExpr, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.conditions) < 2:
            raise ValueError("AndCondition requires at least 2 conditions")

    def to_expression(self, var_names: dict[int, str] | str) -> str:
        """Convert this conjunction to a TVL expression string.

        Args:
            var_names: Mapping from ParameterRange id() to variable name,
                or a single variable name string.

        Returns:
            A string expression with all conditions joined by 'and'.
        """
        parts = [c.to_expression(var_names) for c in self.conditions]
        return f"({' and '.join(parts)})"

    def evaluate_config(
        self, config: dict[str, Any], var_names: dict[int, str]
    ) -> bool:
        """Evaluate this conjunction against a configuration.

        Args:
            config: The configuration dict with parameter values.
            var_names: Mapping from ParameterRange id() to config key.

        Returns:
            True if all conditions are satisfied.
        """
        return all(c.evaluate_config(config, var_names) for c in self.conditions)

    def explain(self, var_names: dict[int, str] | None = None) -> str:
        """Return plain English explanation of this conjunction."""
        parts = [c.explain(var_names) for c in self.conditions]
        return " AND ".join(parts)


@dataclass(frozen=True)
class OrCondition(BoolExpr):
    """Disjunction of multiple conditions (at least one must be true).

    Example:
        >>> model.equals("gpt-4") | model.equals("gpt-3.5")
    """

    conditions: tuple[BoolExpr, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.conditions) < 2:
            raise ValueError("OrCondition requires at least 2 conditions")

    def to_expression(self, var_names: dict[int, str] | str) -> str:
        """Convert this disjunction to a TVL expression string.

        Args:
            var_names: Mapping from ParameterRange id() to variable name,
                or a single variable name string.

        Returns:
            A string expression with all conditions joined by 'or'.
        """
        parts = [c.to_expression(var_names) for c in self.conditions]
        return f"({' or '.join(parts)})"

    def evaluate_config(
        self, config: dict[str, Any], var_names: dict[int, str]
    ) -> bool:
        """Evaluate this disjunction against a configuration.

        Args:
            config: The configuration dict with parameter values.
            var_names: Mapping from ParameterRange id() to config key.

        Returns:
            True if any condition is satisfied.
        """
        return any(c.evaluate_config(config, var_names) for c in self.conditions)

    def explain(self, var_names: dict[int, str] | None = None) -> str:
        """Return plain English explanation of this disjunction."""
        parts = [c.explain(var_names) for c in self.conditions]
        return "(" + " OR ".join(parts) + ")"


@dataclass(frozen=True)
class NotCondition(BoolExpr):
    """Negation of a condition.

    Example:
        >>> ~model.equals("gpt-4")
    """

    condition: BoolExpr

    def to_expression(self, var_names: dict[int, str] | str) -> str:
        """Convert this negation to a TVL expression string.

        Args:
            var_names: Mapping from ParameterRange id() to variable name,
                or a single variable name string.

        Returns:
            A string expression with 'not' prefix.
        """
        return f"not ({self.condition.to_expression(var_names)})"

    def evaluate_config(
        self, config: dict[str, Any], var_names: dict[int, str]
    ) -> bool:
        """Evaluate this negation against a configuration.

        Args:
            config: The configuration dict with parameter values.
            var_names: Mapping from ParameterRange id() to config key.

        Returns:
            True if the inner condition is NOT satisfied.
        """
        return not self.condition.evaluate_config(config, var_names)

    def explain(self, var_names: dict[int, str] | None = None) -> str:
        """Return plain English explanation of this negation."""
        return f"NOT ({self.condition.explain(var_names)})"


# =============================================================================
# Constraint Class
# =============================================================================


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

    when: BoolExpr | None = None
    then: BoolExpr | None = None
    expr: BoolExpr | None = None
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

    def to_expression(self, var_names: dict[int, str] | str) -> str:
        """Convert to TVL expression string.

        Args:
            var_names: Mapping from ParameterRange id() to variable name,
                or a single variable name string.
                Uses id() for identity-based lookup to avoid collision when
                two ParameterRange instances have identical values.

        Returns:
            A string expression representing the constraint
        """
        if self.expr is not None:
            return self.expr.to_expression(var_names)

        # Implication: not(when) or then
        assert self.when is not None and self.then is not None
        when_expr = self.when.to_expression(var_names)
        then_expr = self.then.to_expression(var_names)
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
            return self.expr.evaluate_config(config, var_names)

        # Implication: not(when) or then
        assert self.when is not None and self.then is not None

        when_result = self.when.evaluate_config(config, var_names)
        if not when_result:
            return True  # When is false, implication is satisfied

        return self.then.evaluate_config(config, var_names)

    def explain(self, var_names: dict[int, str] | None = None) -> str:
        """Return a plain English explanation of this constraint.

        Args:
            var_names: Optional mapping from ParameterRange id() to name.

        Returns:
            Human-readable explanation of the constraint.

        Example:
            >>> c = implies(model.equals("gpt-4"), temp.lte(0.7))
            >>> c.explain()
            "IF model equals 'gpt-4' THEN temperature is at most 0.7"
        """
        if self.expr is not None:
            return f"REQUIRE: {self.expr.explain(var_names)}"

        assert self.when is not None and self.then is not None
        when_text = self.when.explain(var_names)
        then_text = self.then.explain(var_names)
        return f"IF {when_text} THEN {then_text}"

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
            return StructuralConstraint(expr=self.expr.to_expression(var_names))

        assert self.when is not None and self.then is not None
        return StructuralConstraint(
            when=self.when.to_expression(var_names),
            then=self.then.to_expression(var_names),
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
            self._collect_tvars(var_names, missing_names)

            if missing_names:
                warnings.warn(
                    f"Constraint has ParameterRange(s) without names: "
                    f"{missing_names}. "
                    "This may cause constraint evaluation to fail. "
                    "Set the 'name' attribute on your Range/Choices/IntRange "
                    "objects, or provide explicit var_names mapping.",
                    UserWarning,
                    stacklevel=2,
                )

        # Capture var_names in closure
        captured_var_names = dict(var_names)

        def constraint_fn(config: dict[str, Any]) -> bool:
            """Evaluate the structural constraint against the config."""
            return self.evaluate(config, captured_var_names)

        # Add metadata for debugging
        constraint_fn.__doc__ = self.description or "Structural constraint"
        constraint_fn.__name__ = f"constraint_{self.id or 'unnamed'}"

        return constraint_fn

    def _collect_tvars(
        self,
        var_names: dict[int, str],
        missing_names: list[str],
    ) -> None:
        """Recursively collect tvar references from expression tree."""
        # Collect from main expression or when/then pair
        expressions_to_collect = (
            [("expr", self.expr)]
            if self.expr is not None
            else [("when", self.when), ("then", self.then)]
        )
        for path, expr in expressions_to_collect:
            if expr is not None:
                self._collect_from_expr(expr, path, var_names, missing_names)

    def _collect_from_expr(
        self,
        expr: BoolExpr,
        path: str,
        var_names: dict[int, str],
        missing_names: list[str],
    ) -> None:
        """Recursively extract tvar names from a boolean expression tree."""
        if isinstance(expr, Condition):
            if expr.tvar.name:
                var_names[id(expr.tvar)] = expr.tvar.name
            else:
                missing_names.append(path)
        elif isinstance(expr, (AndCondition, OrCondition)):
            for i, sub in enumerate(expr.conditions):
                self._collect_from_expr(sub, f"{path}[{i}]", var_names, missing_names)
        elif isinstance(expr, NotCondition):
            self._collect_from_expr(
                expr.condition, f"~{path}", var_names, missing_names
            )


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def implies(
    when: BoolExpr,
    then: BoolExpr,
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
    condition: BoolExpr,
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
# Fluent Builder
# =============================================================================


class WhenBuilder:
    """Builder for when(condition).then(consequence) fluent syntax.

    Example:
        >>> when(model.equals("gpt-4")).then(temp.lte(0.7))
    """

    def __init__(self, condition: BoolExpr) -> None:
        """Initialize the WhenBuilder with a guard condition.

        Args:
            condition: The boolean expression that serves as the implication guard.
        """
        self._condition = condition

    def then(
        self,
        consequence: BoolExpr,
        description: str | None = None,
        id: str | None = None,
    ) -> Constraint:
        """Complete the implication constraint.

        Args:
            consequence: The condition that must hold when the guard is true
            description: Optional human-readable description
            id: Optional identifier for the constraint

        Returns:
            A Constraint with implication semantics
        """
        return Constraint(
            when=self._condition,
            then=consequence,
            description=description,
            id=id,
        )


def when(condition: BoolExpr) -> WhenBuilder:
    """Start a when().then() fluent implication chain.

    Provides readable syntax for constraints:
        when(model.equals("gpt-4")).then(temp.lte(0.7))

    Args:
        condition: The guard/antecedent condition

    Returns:
        A WhenBuilder to complete with .then()

    Example:
        >>> from traigent import Range, Choices, when
        >>>
        >>> model = Choices(["gpt-4", "gpt-3.5"])
        >>> temp = Range(0.0, 2.0)
        >>>
        >>> # If model is gpt-4, temperature must be <= 0.7
        >>> c = when(model.equals("gpt-4")).then(temp.lte(0.7))
    """
    return WhenBuilder(condition)


# =============================================================================
# Utility Functions
# =============================================================================


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


def normalize_constraints(
    constraints: list[Constraint | BoolExpr | Callable[..., Any]] | None,
    var_names: dict[int, str] | None = None,
) -> list[Callable[[dict[str, Any]], bool]]:
    """Normalize mixed Constraint/BoolExpr/Callable list to pure callables.

    This function converts a list containing Constraint objects, bare BoolExpr
    objects, and raw callable functions into a uniform list of callables
    compatible with the optimizer.

    - Constraint objects: converted via to_callable()
    - BoolExpr objects: wrapped in Constraint(expr=...) then converted
    - Raw callables: passed through unchanged

    This enables users to mix all syntax styles:
    - Functional: implies(model.equals("gpt-4"), temp.lte(0.7))
    - Operator: model.equals("gpt-4") >> temp.lte(0.7)
    - Bare condition: temp.lte(1.5)  (auto-wrapped with require())
    - Legacy callable: lambda cfg: cfg["max_tokens"] < 4096

    Args:
        constraints: List of Constraint/BoolExpr/callable objects, or None
        var_names: Optional mapping from ParameterRange id() to config key.
            If not provided, uses the name attribute of each ParameterRange.
            Only used for Constraint/BoolExpr objects.

    Returns:
        List of callable constraint functions. Empty list if input is None.

    Raises:
        TypeError: If an element is not a Constraint, BoolExpr, or callable.

    Example:
        >>> from traigent import Range, Choices, implies, normalize_constraints
        >>>
        >>> model = Choices(["gpt-4", "gpt-3.5"], name="model")
        >>> temp = Range(0.0, 2.0, name="temperature")
        >>>
        >>> # Mixed list of different constraint types
        >>> constraints = [
        ...     implies(model.equals("gpt-4"), temp.lte(0.7)),
        ...     model.equals("gpt-3.5") >> temp.lte(0.9),
        ...     temp.lte(1.5),  # bare BoolExpr, auto-wrapped
        ...     lambda cfg: cfg["max_tokens"] < 4096,
        ... ]
        >>>
        >>> # Normalize for use with decorator
        >>> normalized = normalize_constraints(constraints)
    """
    if not constraints:
        return []

    result: list[Callable[[dict[str, Any]], bool]] = []
    for i, constraint in enumerate(constraints):
        if isinstance(constraint, Constraint):
            result.append(constraint.to_callable(var_names))
        elif isinstance(constraint, BoolExpr):
            # Wrap bare BoolExpr in a require() constraint
            wrapped = Constraint(expr=constraint)
            result.append(wrapped.to_callable(var_names))
        elif callable(constraint):
            result.append(constraint)
        else:
            raise TypeError(
                f"constraints[{i}]: Expected Constraint, BoolExpr, or callable, "
                f"got {type(constraint).__name__}"
            )
    return result


# =============================================================================
# Constraint Conflict Detection
# =============================================================================


@dataclass
class ConstraintConflict:
    """Describes a conflict between constraints.

    Attributes:
        constraints: The constraints that conflict.
        config: A sample configuration that violates the constraints.
        messages: Human-readable explanations of each violation.
    """

    constraints: list[Constraint]
    config: dict[str, Any]
    messages: list[str]

    def __str__(self) -> str:
        lines = ["Constraint conflict detected:"]
        for i, (c, msg) in enumerate(zip(self.constraints, self.messages, strict=True)):
            lines.append(f"  [{i + 1}] {c.explain()}")
            lines.append(f"      Violated: {msg}")
        lines.append(f"  Sample config: {self.config}")
        return "\n".join(lines)


def check_constraints_conflict(
    constraints: list[Constraint],
    sample_configs: list[dict[str, Any]] | None = None,
    var_names: dict[int, str] | None = None,
) -> ConstraintConflict | None:
    """Check if constraints conflict with each other.

    This performs a simple heuristic check by testing sample configurations
    against the constraint set. For comprehensive SAT-based checking,
    consider using z3-solver (MIT license).

    Args:
        constraints: List of Constraint objects to check.
        sample_configs: Optional list of configs to test. If not provided,
            the function returns None (no conflict detected by default).
        var_names: Optional mapping from ParameterRange id() to config key.

    Returns:
        ConstraintConflict if a conflict is detected, None otherwise.

    Example:
        >>> temp = Range(0.0, 2.0, name="temperature")
        >>> c1 = require(temp.lte(0.5))
        >>> c2 = require(temp.gte(0.8))
        >>> # These cannot both be satisfied
        >>> conflict = check_constraints_conflict(
        ...     [c1, c2],
        ...     sample_configs=[{"temperature": 0.3}, {"temperature": 0.9}]
        ... )
        >>> if conflict:
        ...     print(conflict)
    """
    if not constraints or not sample_configs:
        return None

    # Build var_names from constraints if not provided
    if var_names is None:
        var_names = {}
        for c in constraints:
            c._collect_tvars(var_names, [])

    # Test each sample config against all constraints
    for config in sample_configs:
        violations: list[tuple[Constraint, str]] = []

        for constraint in constraints:
            if not constraint.evaluate(config, var_names):
                # Constraint violated - record it
                explanation = constraint.explain(var_names)
                violations.append((constraint, explanation))

        # If all constraints violated, we found a conflict scenario
        if len(violations) == len(constraints) and len(violations) > 1:
            return ConstraintConflict(
                constraints=[v[0] for v in violations],
                config=config,
                messages=[v[1] for v in violations],
            )

    return None


def explain_constraint_violation(
    constraint: Constraint,
    config: dict[str, Any],
    var_names: dict[int, str] | None = None,
) -> str | None:
    """Explain why a constraint is violated by a configuration.

    Args:
        constraint: The constraint to check.
        config: The configuration to test.
        var_names: Optional mapping from ParameterRange id() to config key.

    Returns:
        Human-readable explanation if violated, None if satisfied.

    Example:
        >>> temp = Range(0.0, 2.0, name="temperature")
        >>> c = require(temp.lte(0.5))
        >>> msg = explain_constraint_violation(c, {"temperature": 0.9})
        >>> print(msg)
        "Constraint violated: REQUIRE: temperature is at most 0.5
         Config has temperature=0.9"
    """
    if var_names is None:
        var_names = {}
        constraint._collect_tvars(var_names, [])

    if constraint.evaluate(config, var_names):
        return None  # Not violated

    explanation = constraint.explain(var_names)

    # Find which values caused the violation
    relevant_values = []
    for _tvar_id, name in var_names.items():
        if name in config:
            relevant_values.append(f"{name}={config[name]!r}")

    config_str = ", ".join(relevant_values) if relevant_values else str(config)

    return f"Constraint violated: {explanation}\n  Config has: {config_str}"


__all__ = [
    "AndCondition",
    "BoolExpr",
    "Condition",
    "Constraint",
    "ConstraintConflict",
    "NotCondition",
    "OperatorType",
    "OrCondition",
    "WhenBuilder",
    "check_constraints_conflict",
    "constraints_to_callables",
    "explain_constraint_violation",
    "implies",
    "normalize_constraints",
    "require",
    "when",
]
