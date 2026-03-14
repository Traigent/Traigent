"""Built-in validators for Traigent constraint systems."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from traigent_validation.base import (
    ConstraintViolation,
    SatResult,
    SatStatus,
    ValidationResult,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from traigent.api.constraints import Constraint
    from traigent.api.parameter_ranges import ParameterRange
    from traigent.tvl.models import StructuralConstraint

_PARAM_REF_PATTERN = re.compile(r"\bparams\.([A-Za-z_][A-Za-z0-9_]*)\b")


def _extract_param_refs(expression: str) -> set[str]:
    return {match.group(1) for match in _PARAM_REF_PATTERN.finditer(expression)}


class PythonConstraintValidator:
    """Default Python-based validator for runtime and compile-time checks."""

    def validate_config(
        self,
        config: dict[str, Any],
        constraints: Sequence[Constraint],
        var_names: Mapping[int, str],
    ) -> ValidationResult:
        violations: list[ConstraintViolation] = []
        resolved_var_names = dict(var_names)

        for i, constraint in enumerate(constraints):
            try:
                is_satisfied = constraint.evaluate(config, resolved_var_names)
                if not is_satisfied:
                    relevant_values = self._get_relevant_values(
                        config, constraint, resolved_var_names
                    )
                    message = self._build_violation_message(
                        constraint, config, resolved_var_names
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
            except Exception as exc:
                violations.append(
                    ConstraintViolation(
                        constraint_index=i,
                        constraint_id=constraint.id,
                        constraint_description=constraint.description,
                        violating_values=config.copy(),
                        message=f"Constraint evaluation error: {exc}",
                    )
                )

        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            checked_count=len(constraints),
        )

    def check_satisfiability(
        self,
        tvars: Mapping[str, ParameterRange],
        constraints: Sequence[Constraint],
    ) -> SatResult:
        if not constraints:
            return SatResult(status=SatStatus.SAT, message="No constraints to satisfy")

        if not tvars:
            return SatResult(status=SatStatus.UNSAT, message="No parameters defined")

        return SatResult(
            status=SatStatus.UNKNOWN,
            message=(
                "Python validator cannot prove satisfiability. "
                "Consider a SAT/SMT validator for complex constraints."
            ),
        )

    def validate_structural_constraints(
        self,
        constraints: Sequence[StructuralConstraint],
        available_parameters: set[str],
    ) -> list[str]:
        """Validate TVL structural constraint parameter references."""
        issues: list[str] = []

        for constraint in constraints:
            index = getattr(constraint, "index", 0)
            expressions = []
            if getattr(constraint, "expr", None):
                expressions.append(("expr", constraint.expr))
            if getattr(constraint, "when", None):
                expressions.append(("when", constraint.when))
            if getattr(constraint, "then", None):
                expressions.append(("then", constraint.then))

            for clause_name, expression in expressions:
                if not isinstance(expression, str):
                    continue
                refs = _extract_param_refs(expression)
                if not refs:
                    continue

                # Some TVL specs use `params.*` for runtime observability values
                # (e.g., latency/cost metrics) that are not part of the tunable
                # configuration space. To avoid false positives, only enforce
                # unknown-reference checks for clauses that also reference at
                # least one known tunable parameter.
                if not (refs & available_parameters):
                    continue

                unknown = sorted(refs - available_parameters)
                if unknown:
                    issues.append(
                        "Constraint index "
                        f"{index} {clause_name} references unknown parameter(s): "
                        f"{', '.join(unknown)}"
                    )

        return issues

    def _add_tvar_value(
        self,
        relevant: dict[str, Any],
        config: dict[str, Any],
        tvar: Any,
        var_names: dict[int, str],
    ) -> None:
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
        relevant: dict[str, Any] = {}

        if constraint.expr is not None:
            self._add_tvar_value(relevant, config, constraint.expr.tvar, var_names)
        else:
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
        if constraint.description:
            return f"Violated: {constraint.description}"

        if constraint.expr is not None:
            var_name = var_names.get(id(constraint.expr.tvar), "?")
            value = config.get(var_name, "?")
            expr_str = constraint.expr.to_expression(var_name)
            return f"{var_name}={value} violates condition: {expr_str}"

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


class SATConstraintValidator:
    """Compatibility validator that delegates to PythonConstraintValidator.

    TODO: Replace delegation with a real SAT/SMT backend implementation.
    """

    def __init__(self) -> None:
        self._delegate = PythonConstraintValidator()

    def validate_config(
        self,
        config: dict[str, Any],
        constraints: Sequence[Constraint],
        var_names: Mapping[int, str],
    ) -> ValidationResult:
        return self._delegate.validate_config(config, constraints, var_names)

    def check_satisfiability(
        self,
        tvars: Mapping[str, ParameterRange],
        constraints: Sequence[Constraint],
    ) -> SatResult:
        return self._delegate.check_satisfiability(tvars, constraints)

    def validate_structural_constraints(
        self,
        constraints: Sequence[StructuralConstraint],
        available_parameters: set[str],
    ) -> list[str]:
        return self._delegate.validate_structural_constraints(
            constraints, available_parameters
        )


__all__ = [
    "PythonConstraintValidator",
    "SATConstraintValidator",
]
