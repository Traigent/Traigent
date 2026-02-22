"""Core validation protocols and result models for Traigent constraint validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from traigent.api.constraints import Constraint
    from traigent.api.parameter_ranges import ParameterRange
    from traigent.tvl.models import StructuralConstraint


class SatStatus(Enum):
    """Satisfiability status for constraint systems."""

    SAT = "satisfiable"
    UNSAT = "unsatisfiable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ConstraintViolation:
    """Details about a single constraint violation."""

    constraint_index: int
    constraint_id: str | None = None
    constraint_description: str | None = None
    violating_values: dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of validating a configuration against constraints."""

    is_valid: bool
    violations: list[ConstraintViolation] = field(default_factory=list)
    checked_count: int = 0


@dataclass(frozen=True, slots=True)
class SatResult:
    """Result of checking constraint system satisfiability."""

    status: SatStatus
    unsat_core: list[int] | None = None
    example_config: dict[str, Any] | None = None
    message: str | None = None


@runtime_checkable
class ConstraintValidator(Protocol):
    """Protocol for runtime config validation backends."""

    def validate_config(
        self,
        config: dict[str, Any],
        constraints: list[Constraint],
        var_names: dict[int, str],
    ) -> ValidationResult:
        """Validate one configuration against a set of constraints."""
        ...

    def check_satisfiability(
        self,
        tvars: dict[str, ParameterRange],
        constraints: list[Constraint],
    ) -> SatResult:
        """Check if any valid configuration can satisfy the constraints."""
        ...


@runtime_checkable
class StructuralConstraintValidator(Protocol):
    """Protocol for compile-time TVL structural-constraint validation."""

    def validate_structural_constraints(
        self,
        constraints: Sequence[StructuralConstraint],
        available_parameters: set[str],
    ) -> list[str]:
        """Return human-readable issues for invalid structural constraints."""
        ...


__all__ = [
    "ConstraintValidator",
    "ConstraintViolation",
    "SatResult",
    "SatStatus",
    "StructuralConstraintValidator",
    "ValidationResult",
]
