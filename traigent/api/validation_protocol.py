"""Validation protocol for TVL structural constraints.

This module defines the ConstraintValidator protocol that allows swapping
between different validation implementations (Python evaluation, SAT/SMT
solvers, etc.) without changing the constraint interface.

The protocol-based design enables:
- Python-based evaluation for simple cases (default)
- Future integration with SAT/SMT solvers for complex constraint satisfaction
- Custom domain-specific validators

Example:
    >>> from traigent import Range, Choices, implies
    >>> from traigent.api.validation_protocol import PythonConstraintValidator
    >>>
    >>> # Define parameters and constraints
    >>> temp = Range(0.0, 2.0, name="temperature")
    >>> model = Choices(["gpt-4", "gpt-3.5"], name="model")
    >>> constraints = [implies(model.equals("gpt-4"), temp.lte(0.7))]
    >>>
    >>> # Validate a config
    >>> validator = PythonConstraintValidator()
    >>> result = validator.validate_config(
    ...     config={"temperature": 0.5, "model": "gpt-4"},
    ...     constraints=constraints,
    ...     var_names={id(temp): "temperature", id(model): "model"}
    ... )
    >>> result.is_valid
    True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from traigent.api.constraints import Constraint
    from traigent.api.parameter_ranges import ParameterRange


class SatStatus(Enum):
    """Satisfiability status for constraint systems.

    Represents the result of checking whether a configuration space
    has any valid configurations given the constraints.

    Attributes:
        SAT: The constraint system is satisfiable (valid configs exist)
        UNSAT: The constraint system is unsatisfiable (no valid configs)
        UNKNOWN: Satisfiability could not be determined
    """

    SAT = "satisfiable"
    UNSAT = "unsatisfiable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ConstraintViolation:
    """Details about a single constraint violation.

    Provides information about which constraint was violated and
    what values caused the violation.

    Attributes:
        constraint_index: Index of the violated constraint in the list
        constraint_id: Optional identifier for the constraint
        constraint_description: Human-readable description if available
        violating_values: The parameter values that caused the violation
        message: Human-readable explanation of the violation

    Example:
        >>> violation = ConstraintViolation(
        ...     constraint_index=0,
        ...     constraint_description="GPT-4 requires low temperature",
        ...     violating_values={"temperature": 1.5, "model": "gpt-4"},
        ...     message="temperature=1.5 violates condition: temperature <= 0.7"
        ... )
    """

    constraint_index: int
    constraint_id: str | None = None
    constraint_description: str | None = None
    violating_values: dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of validating a configuration against constraints.

    Contains both the overall validity status and details about
    any violations that occurred.

    Attributes:
        is_valid: True if all constraints are satisfied
        violations: List of constraint violations (empty if valid)
        checked_count: Number of constraints that were checked

    Example:
        >>> result = ValidationResult(is_valid=True, violations=[], checked_count=3)
        >>> if result.is_valid:
        ...     print("Configuration is valid!")
        Configuration is valid!
    """

    is_valid: bool
    violations: list[ConstraintViolation] = field(default_factory=list)
    checked_count: int = 0


@dataclass(frozen=True, slots=True)
class SatResult:
    """Result of checking constraint system satisfiability.

    Provides information about whether the configuration space
    defined by parameters and constraints has any valid configurations.

    Attributes:
        status: The satisfiability status (SAT, UNSAT, or UNKNOWN)
        unsat_core: Indices of constraints forming the minimal unsatisfiable
            core (only present if UNSAT and solver supports it)
        example_config: An example satisfying configuration (only present
            if SAT and solver found one)
        message: Optional explanatory message

    Example:
        >>> # A satisfiable result with example
        >>> result = SatResult(
        ...     status=SatStatus.SAT,
        ...     example_config={"temperature": 0.5, "model": "gpt-4"}
        ... )
        >>>
        >>> # An unsatisfiable result with core
        >>> result = SatResult(
        ...     status=SatStatus.UNSAT,
        ...     unsat_core=[0, 2],  # Constraints 0 and 2 conflict
        ...     message="Constraints are mutually exclusive"
        ... )
    """

    status: SatStatus
    unsat_core: list[int] | None = None
    example_config: dict[str, Any] | None = None
    message: str | None = None


@runtime_checkable
class ConstraintValidator(Protocol):
    """Protocol for constraint validation implementations.

    This protocol allows swapping between different validation backends:
    - Python evaluation (default, simple cases)
    - SAT/SMT solvers (complex constraint satisfaction)
    - Custom domain-specific validators

    Implementations must provide two methods:
    1. validate_config: Check if a specific configuration satisfies constraints
    2. check_satisfiability: Check if any valid configuration exists

    Example Implementation:
        >>> class CustomValidator:
        ...     def validate_config(
        ...         self,
        ...         config: dict[str, Any],
        ...         constraints: list[Constraint],
        ...         var_names: dict[ParameterRange, str]
        ...     ) -> ValidationResult:
        ...         # Custom validation logic
        ...         pass
        ...
        ...     def check_satisfiability(
        ...         self,
        ...         tvars: dict[str, ParameterRange],
        ...         constraints: list[Constraint]
        ...     ) -> SatResult:
        ...         # Custom SAT checking logic
        ...         pass
    """

    def validate_config(
        self,
        config: dict[str, Any],
        constraints: list[Constraint],
        var_names: dict[int, str],
    ) -> ValidationResult:
        """Check if a specific configuration satisfies all constraints.

        Args:
            config: The configuration dict with parameter values
            constraints: List of constraints to check
            var_names: Mapping from ParameterRange id() to config key names.
                Uses id() for identity-based lookup to avoid collision when
                two ParameterRange instances have identical values.

        Returns:
            ValidationResult with is_valid flag and any violations
        """
        ...

    def check_satisfiability(
        self,
        tvars: dict[str, ParameterRange],
        constraints: list[Constraint],
    ) -> SatResult:
        """Check if the configuration space has any valid configurations.

        This is useful for detecting impossible constraint combinations
        before starting optimization.

        Args:
            tvars: Dict mapping parameter names to their ParameterRange
            constraints: List of constraints to satisfy

        Returns:
            SatResult with status and optional example/unsat_core
        """
        ...


class PythonConstraintValidator:
    """Default Python-based constraint validator.

    Evaluates constraints using Python expressions. This is suitable
    for simple constraint systems but cannot prove satisfiability
    without enumeration.

    For complex constraint systems where satisfiability proofs are
    needed, consider using a SAT/SMT-based validator.

    Example:
        >>> from traigent import Range, Choices, implies
        >>>
        >>> temp = Range(0.0, 2.0, name="temperature")
        >>> model = Choices(["gpt-4", "gpt-3.5"], name="model")
        >>> constraints = [implies(model.equals("gpt-4"), temp.lte(0.7))]
        >>>
        >>> validator = PythonConstraintValidator()
        >>> result = validator.validate_config(
        ...     config={"temperature": 0.5, "model": "gpt-4"},
        ...     constraints=constraints,
        ...     var_names={id(temp): "temperature", id(model): "model"}
        ... )
        >>> result.is_valid
        True
    """

    def validate_config(
        self,
        config: dict[str, Any],
        constraints: list[Constraint],
        var_names: dict[int, str],
    ) -> ValidationResult:
        """Validate a configuration against constraints using Python evaluation.

        Args:
            config: The configuration dict with parameter values
            constraints: List of constraints to check
            var_names: Mapping from ParameterRange id() to config key names.
                Uses id() for identity-based lookup to avoid collision when
                two ParameterRange instances have identical values.

        Returns:
            ValidationResult with is_valid flag and any violations
        """
        violations: list[ConstraintViolation] = []

        for i, constraint in enumerate(constraints):
            try:
                is_satisfied = constraint.evaluate(config, var_names)
                if not is_satisfied:
                    # Build violation details
                    relevant_values = self._get_relevant_values(
                        config, constraint, var_names
                    )
                    message = self._build_violation_message(
                        constraint, config, var_names
                    )

                    violations.append(
                        ConstraintViolation(
                            constraint_index=i,
                            constraint_id=constraint.id,
                            constraint_description=constraint.description,
                            violating_values=relevant_values,
                            message=message,
                        )
                    )
            except Exception as e:
                # Constraint evaluation error - treat as violation
                violations.append(
                    ConstraintViolation(
                        constraint_index=i,
                        constraint_id=constraint.id,
                        constraint_description=constraint.description,
                        violating_values=config.copy(),
                        message=f"Constraint evaluation error: {e}",
                    )
                )

        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            checked_count=len(constraints),
        )

    def check_satisfiability(
        self,
        tvars: dict[str, ParameterRange],
        constraints: list[Constraint],
    ) -> SatResult:
        """Check satisfiability (returns UNKNOWN for Python validator).

        The Python validator cannot prove satisfiability without
        enumerating all possible configurations. For constraint
        satisfiability proofs, use a SAT/SMT-based validator.

        Args:
            tvars: Dict mapping parameter names to their ParameterRange
            constraints: List of constraints to satisfy

        Returns:
            SatResult with UNKNOWN status (unless trivially SAT/UNSAT)
        """
        # If no constraints, always satisfiable
        if not constraints:
            return SatResult(
                status=SatStatus.SAT,
                message="No constraints to satisfy",
            )

        # If no tvars, no valid configs possible
        if not tvars:
            return SatResult(
                status=SatStatus.UNSAT,
                message="No parameters defined",
            )

        # Python evaluation cannot prove satisfiability without enumeration
        return SatResult(
            status=SatStatus.UNKNOWN,
            message=(
                "Python validator cannot prove satisfiability. "
                "Consider using a SAT/SMT-based validator for complex constraints."
            ),
        )

    def _add_tvar_value(
        self,
        relevant: dict[str, Any],
        config: dict[str, Any],
        tvar: Any,
        var_names: dict[int, str],
    ) -> None:
        """Add a tvar's value to the relevant dict if present in config."""
        if tvar is None:
            return
        var_name = var_names.get(id(tvar))
        if var_name and var_name in config:
            relevant[var_name] = config[var_name]

    def _get_relevant_values(
        self,
        config: dict[str, Any],
        constraint: Constraint,
        var_names: dict[int, str],
    ) -> dict[str, Any]:
        """Extract values relevant to a constraint from the config."""
        relevant: dict[str, Any] = {}

        if constraint.expr is not None:
            self._add_tvar_value(relevant, config, constraint.expr.tvar, var_names)
        else:
            # Implication constraint
            self._add_tvar_value(
                relevant,
                config,
                constraint.when.tvar if constraint.when else None,
                var_names,
            )
            self._add_tvar_value(
                relevant,
                config,
                constraint.then.tvar if constraint.then else None,
                var_names,
            )

        return relevant

    def _build_violation_message(
        self,
        constraint: Constraint,
        config: dict[str, Any],
        var_names: dict[int, str],
    ) -> str:
        """Build a human-readable violation message."""
        if constraint.description:
            return f"Violated: {constraint.description}"

        if constraint.expr is not None:
            var_name = var_names.get(id(constraint.expr.tvar), "?")
            value = config.get(var_name, "?")
            expr_str = constraint.expr.to_expression(var_name)
            return f"{var_name}={value} violates condition: {expr_str}"

        # Implication constraint
        if constraint.when is not None and constraint.then is not None:
            when_var = var_names.get(id(constraint.when.tvar), "?")
            then_var = var_names.get(id(constraint.then.tvar), "?")
            when_value = config.get(when_var, "?")
            then_value = config.get(then_var, "?")

            when_expr = constraint.when.to_expression(when_var)
            then_expr = constraint.then.to_expression(then_var)

            return (
                f"When {when_expr} (value: {when_value}), "
                f"expected {then_expr} but got {then_var}={then_value}"
            )

        return "Constraint violated"


# =============================================================================
# Future Extension: SAT/SMT Validator Interface
# =============================================================================


class SATConstraintValidator:
    """Placeholder for SAT/SMT-based constraint validator.

    This class provides a template for implementing a validator
    using SAT/SMT solvers like Z3, PySAT, or similar.

    Note:
        This is a placeholder for future implementation.
        The interface is defined to ensure compatibility with
        the ConstraintValidator protocol.

    Future Implementation Notes:
        - Convert ParameterRange domains to SMT variables
        - Convert Condition expressions to SMT constraints
        - Use solver to check satisfiability
        - Extract models (example configs) when SAT
        - Extract unsat cores when UNSAT
    """

    def validate_config(
        self,
        config: dict[str, Any],
        constraints: list[Constraint],
        var_names: dict[int, str],
    ) -> ValidationResult:
        """Validate using SAT solver (delegates to Python for config check)."""
        # For single config validation, Python evaluation is efficient
        return PythonConstraintValidator().validate_config(
            config, constraints, var_names
        )

    def check_satisfiability(
        self,
        tvars: dict[str, ParameterRange],
        constraints: list[Constraint],
    ) -> SatResult:
        """Check satisfiability using SAT/SMT solver.

        Raises:
            NotImplementedError: SAT solver integration not yet implemented
        """
        raise NotImplementedError(
            "SAT/SMT solver integration not yet implemented. "
            "Use PythonConstraintValidator for Python-based evaluation, "
            "or contribute a solver integration."
        )


__all__ = [
    "ConstraintValidator",
    "ConstraintViolation",
    "PythonConstraintValidator",
    "SATConstraintValidator",
    "SatResult",
    "SatStatus",
    "ValidationResult",
]
