"""ConfigSpace: A complete configuration space with TVARs and constraints.

This module provides the ConfigSpace class, which is the SE-friendly equivalent
of a TVL spec's tvars + constraints. It combines parameter definitions with
structural constraints and provides validation capabilities.

Example:
    >>> from traigent import Range, IntRange, Choices, implies
    >>> from traigent.api.config_space import ConfigSpace
    >>>
    >>> # Define parameters
    >>> temperature = Range(0.0, 2.0, name="temperature", unit="ratio")
    >>> max_tokens = IntRange(100, 4096, name="max_tokens", unit="count")
    >>> model = Choices(["gpt-4", "gpt-3.5-turbo"], name="model")
    >>>
    >>> # Define constraints
    >>> constraints = [
    ...     implies(model.equals("gpt-4"), max_tokens.gte(1000)),
    ... ]
    >>>
    >>> # Create config space
    >>> space = ConfigSpace(
    ...     tvars={"temperature": temperature, "max_tokens": max_tokens, "model": model},
    ...     constraints=constraints,
    ... )
    >>>
    >>> # Validate a configuration
    >>> result = space.validate({"temperature": 0.5, "max_tokens": 2000, "model": "gpt-4"})
    >>> print(result.is_valid)  # True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from traigent.api.constraints import Constraint
from traigent.api.parameter_ranges import (
    Choices,
    IntRange,
    LogRange,
    ParameterRange,
    Range,
)
from traigent.api.validation_protocol import (
    ConstraintValidator,
    PythonConstraintValidator,
    SatResult,
    ValidationResult,
)

if TYPE_CHECKING:
    from traigent.tvl.models import StructuralConstraint, TVarDecl


@dataclass
class ConfigSpace:
    """A complete configuration space with TVARs and constraints.

    This is the SE-friendly equivalent of a TVL spec's tvars + constraints.
    It combines parameter definitions (Range, IntRange, Choices, etc.) with
    structural constraints and provides validation capabilities.

    ConfigSpace can be created directly or built from decorator arguments.
    It supports exporting to TVL-compatible format for integration with
    the TVL ecosystem.

    Attributes:
        tvars: Dict mapping parameter names to their ParameterRange definitions
        constraints: List of structural constraints on the configuration space
        description: Optional description of the configuration space

    Example:
        >>> from traigent import Range, Choices, implies
        >>> from traigent.api.config_space import ConfigSpace
        >>>
        >>> temp = Range(0.0, 2.0, name="temperature")
        >>> model = Choices(["gpt-4", "gpt-3.5"], name="model")
        >>>
        >>> space = ConfigSpace(
        ...     tvars={"temperature": temp, "model": model},
        ...     constraints=[implies(model.equals("gpt-4"), temp.lte(0.7))],
        ... )
    """

    tvars: dict[str, ParameterRange]
    constraints: list[Constraint] = field(default_factory=list)
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate the configuration space after initialization."""
        # Ensure all tvars have unique names
        seen_names: set[str] = set()
        for name, tvar in self.tvars.items():
            if name in seen_names:
                raise ValueError(f"Duplicate parameter name: {name}")
            seen_names.add(name)

            # Set the tvar's name if not already set
            if hasattr(tvar, "name") and tvar.name is None:
                # Note: We can't modify frozen dataclasses, so we just validate
                pass

    @classmethod
    def from_decorator_args(
        cls,
        configuration_space: dict[str, Any] | None = None,
        inline_params: dict[str, Any] | None = None,
        constraints: list[Constraint] | None = None,
        description: str | None = None,
    ) -> ConfigSpace:
        """Build ConfigSpace from decorator arguments.

        This factory method handles the various ways parameters can be
        specified to the @optimize decorator:
        1. Via configuration_space dict
        2. Via inline **kwargs
        3. Or a combination of both

        Args:
            configuration_space: Dict of parameter definitions
            inline_params: Inline parameter definitions from **kwargs
            constraints: List of structural constraints
            description: Optional description

        Returns:
            A ConfigSpace instance

        Example:
            >>> # From decorator usage:
            >>> # @optimize(temperature=Range(0.0, 2.0), model=Choices([...]))
            >>> space = ConfigSpace.from_decorator_args(
            ...     inline_params={"temperature": Range(0.0, 2.0)}
            ... )
        """
        tvars: dict[str, ParameterRange] = {}

        # Process configuration_space first, then inline params (may override)
        for params in (configuration_space, inline_params):
            if params:
                cls._process_param_dict(params, tvars)

        return cls(
            tvars=tvars,
            constraints=constraints or [],
            description=description,
        )

    @classmethod
    def _process_param_dict(
        cls, params: dict[str, Any], tvars: dict[str, ParameterRange]
    ) -> None:
        """Process a parameter dictionary and add ranges to tvars.

        Args:
            params: Dictionary of parameter definitions
            tvars: Target dictionary to populate with ParameterRange instances
        """
        for name, value in params.items():
            if isinstance(value, ParameterRange):
                tvars[name] = value
            elif isinstance(value, (list, tuple)):
                tvars[name] = cls._sequence_to_range(name, value)
            elif isinstance(value, dict):
                tvars[name] = cls._dict_to_range(name, value)
            # Skip non-range values (they might be other decorator args)

    @classmethod
    def _sequence_to_range(
        cls, name: str, value: list[Any] | tuple[Any, ...]
    ) -> ParameterRange:
        """Convert a list or tuple to appropriate ParameterRange.

        Heuristics:
        - A 2-element tuple of numeric values (int/float) is treated as Range(low, high)
        - All other sequences are treated as Choices

        Args:
            name: Parameter name
            value: List or tuple to convert

        Returns:
            Appropriate ParameterRange instance
        """
        # Check for range-like tuple: (low, high) with exactly 2 numeric values
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
            )
        ):
            low, high = value
            # Use IntRange if both are integers, otherwise Range
            if isinstance(low, int) and isinstance(high, int):
                return IntRange(low=low, high=high, name=name)
            return Range(low=float(low), high=float(high), name=name)

        # Everything else is Choices
        return Choices(values=list(value), name=name)

    @staticmethod
    def _dict_to_range(name: str, spec: dict[str, Any]) -> ParameterRange:
        """Convert a dict specification to appropriate ParameterRange.

        Supports various dict formats:
        - {"choices": [...]} -> Choices
        - {"type": "categorical", "values": [...]} -> Choices
        - {"low": x, "high": y} -> Range/IntRange based on types
        - {"low": x, "high": y, "log": True} -> LogRange

        Args:
            name: Parameter name
            spec: Dict with range specification

        Returns:
            Appropriate ParameterRange instance

        Raises:
            ValueError: If dict format is invalid
        """
        # Check for explicit categorical type
        if spec.get("type") == "categorical" or "values" in spec:
            values = spec.get("values") or spec.get("choices")
            if values is not None:
                return Choices(
                    values=list(values),
                    name=name,
                    default=spec.get("default"),
                )

        # Check for choices format (legacy)
        if "choices" in spec:
            return Choices(
                values=spec["choices"],
                name=name,
                default=spec.get("default"),
            )

        # Check for range format
        if "low" in spec and "high" in spec:
            low = spec["low"]
            high = spec["high"]

            # LogRange takes precedence if log=True
            if spec.get("log", False):
                return LogRange(
                    low=float(low),
                    high=float(high),
                    default=spec.get("default"),
                    name=name,
                    unit=spec.get("unit"),
                )

            # Use IntRange if both bounds are integers (with or without step)
            if isinstance(low, int) and isinstance(high, int):
                return IntRange(
                    low=low,
                    high=high,
                    step=spec.get("step"),
                    log=False,
                    default=spec.get("default"),
                    name=name,
                    unit=spec.get("unit"),
                )

            # Default to Range for floats
            return Range(
                low=float(low),
                high=float(high),
                step=spec.get("step"),
                log=False,
                default=spec.get("default"),
                name=name,
                unit=spec.get("unit"),
            )

        raise ValueError(
            f"Invalid range specification for '{name}': {spec}. "
            "Expected dict with 'low'/'high' or 'choices'."
        )

    @property
    def var_names(self) -> dict[int, str]:
        """Get mapping from ParameterRange object IDs to their names.

        Uses id() for identity-based lookup to avoid collision when
        two ParameterRange instances have identical values but different
        purposes (e.g., Range(0.0, 1.0) used for two different params).

        Returns:
            Dict mapping each ParameterRange's id() to its parameter name
        """
        return {id(tvar): name for name, tvar in self.tvars.items()}

    def get_var_name(self, tvar: ParameterRange) -> str | None:
        """Get the parameter name for a ParameterRange using identity lookup.

        Args:
            tvar: The ParameterRange to look up

        Returns:
            The parameter name, or None if not found
        """
        return self.var_names.get(id(tvar))

    def validate(
        self,
        config: dict[str, Any],
        validator: ConstraintValidator | None = None,
    ) -> ValidationResult:
        """Validate a configuration against this space's constraints.

        Args:
            config: The configuration dict to validate
            validator: Optional custom validator (defaults to PythonConstraintValidator)

        Returns:
            ValidationResult with is_valid flag and any violations
        """
        validator = validator or PythonConstraintValidator()
        return validator.validate_config(config, self.constraints, self.var_names)

    def check_satisfiability(
        self,
        validator: ConstraintValidator | None = None,
    ) -> SatResult:
        """Check if this configuration space has any valid configurations.

        This is useful for detecting impossible constraint combinations
        before starting optimization.

        Args:
            validator: Optional custom validator (defaults to PythonConstraintValidator)

        Returns:
            SatResult with satisfiability status
        """
        validator = validator or PythonConstraintValidator()
        return validator.check_satisfiability(self.tvars, self.constraints)

    def to_tvl_tvars(self) -> list[TVarDecl]:
        """Export TVARs as TVL TVarDecl objects.

        Returns:
            List of TVarDecl instances for each parameter
        """
        from traigent.tvl.models import TVarDecl

        tvardecls: list[TVarDecl] = []

        for name, tvar in self.tvars.items():
            tvar_type = self._infer_tvar_type(tvar)
            domain = self._tvar_to_domain(tvar)

            tvardecls.append(
                TVarDecl(  # type: ignore[call-arg]
                    name=name,
                    type=tvar_type,  # type: ignore[arg-type]
                    raw_type=tvar_type,
                    domain=domain,  # type: ignore[arg-type]
                    unit=getattr(tvar, "unit", None),
                    default=getattr(tvar, "default", None),
                )
            )

        return tvardecls

    def to_tvl_constraints(self) -> list[StructuralConstraint]:
        """Export constraints as TVL StructuralConstraint objects.

        Returns:
            List of StructuralConstraint instances
        """
        return [c.to_structural_constraint(self.var_names) for c in self.constraints]

    def to_tvl_spec(self) -> dict[str, Any]:
        """Export as TVL-compatible dict.

        This format can be serialized to YAML/JSON for TVL spec files.

        Returns:
            Dict in TVL spec format
        """
        tvars_list = [
            self._tvar_to_spec_dict(name, tvar) for name, tvar in self.tvars.items()
        ]
        constraints_list = self._build_constraints_list()

        spec: dict[str, Any] = {"tvars": tvars_list}
        if constraints_list:
            spec["constraints"] = {"structural": constraints_list}
        if self.description:
            spec["description"] = self.description

        return spec

    def _tvar_to_spec_dict(self, name: str, tvar: ParameterRange) -> dict[str, Any]:
        """Convert a single tvar to TVL spec dict format."""
        tvar_dict: dict[str, Any] = {
            "name": name,
            "type": self._infer_tvar_type(tvar),
            "domain": self._tvar_to_domain(tvar),
        }
        if hasattr(tvar, "unit") and tvar.unit:
            tvar_dict["unit"] = tvar.unit
        if hasattr(tvar, "default") and tvar.default is not None:
            tvar_dict["default"] = tvar.default
        return tvar_dict

    def _build_constraints_list(self) -> list[dict[str, Any]]:
        """Build the constraints list for TVL spec."""
        constraints_list = []
        for idx, c in enumerate(self.constraints):
            constraint_dict = self._constraint_to_spec_dict(c, idx)
            if constraint_dict:
                constraints_list.append(constraint_dict)
        return constraints_list

    def _constraint_to_spec_dict(
        self, c: Constraint, idx: int
    ) -> dict[str, Any] | None:
        """Convert a single constraint to TVL spec dict format."""
        if c.expr is not None:
            return self._expr_constraint_to_dict(c, idx)
        elif c.when is not None and c.then is not None:
            return self._implication_constraint_to_dict(c, idx)
        return None

    def _expr_constraint_to_dict(self, c: Constraint, idx: int) -> dict[str, Any]:
        """Convert an expression constraint to dict."""
        expr_tvar = c.expr.tvar  # type: ignore[union-attr]
        base_name = self.var_names.get(
            id(expr_tvar),
            (expr_tvar.name if expr_tvar else None) or "unknown",
        )
        var_name = f"params.{base_name}"
        constraint_dict: dict[str, Any] = {
            "expr": c.expr.to_expression(var_name),  # type: ignore[union-attr]
            "index": idx,
        }
        self._add_constraint_metadata(constraint_dict, c)
        return constraint_dict

    def _implication_constraint_to_dict(
        self, c: Constraint, idx: int
    ) -> dict[str, Any]:
        """Convert an implication (when/then) constraint to dict."""
        when_tvar = c.when.tvar  # type: ignore[union-attr]
        then_tvar = c.then.tvar  # type: ignore[union-attr]
        when_base = self.var_names.get(
            id(when_tvar),
            (when_tvar.name if when_tvar else None) or "unknown",
        )
        then_base = self.var_names.get(
            id(then_tvar),
            (then_tvar.name if then_tvar else None) or "unknown",
        )
        when_var = f"params.{when_base}"
        then_var = f"params.{then_base}"
        constraint_dict: dict[str, Any] = {
            "when": c.when.to_expression(when_var),  # type: ignore[union-attr]
            "then": c.then.to_expression(then_var),  # type: ignore[union-attr]
            "index": idx,
        }
        self._add_constraint_metadata(constraint_dict, c)
        return constraint_dict

    @staticmethod
    def _add_constraint_metadata(
        constraint_dict: dict[str, Any], constraint: Constraint
    ) -> None:
        """Add optional metadata fields to a constraint dict.

        Adds id, description, and error_message fields if present on the constraint.

        Args:
            constraint_dict: The constraint dict to modify in-place
            constraint: The Constraint object with metadata
        """
        if constraint.id:
            constraint_dict["id"] = constraint.id
        if constraint.description:
            constraint_dict["description"] = constraint.description
            # Also add as error_message for user-friendly constraint violations
            constraint_dict["error_message"] = constraint.description

    @staticmethod
    def _infer_tvar_type(tvar: ParameterRange) -> str:
        """Infer TVL type from ParameterRange.

        Args:
            tvar: The ParameterRange to get type for

        Returns:
            TVL type string ("float", "int", "enum", etc.)
        """
        if isinstance(tvar, IntRange):
            return "int"
        elif isinstance(tvar, (Range, LogRange)):
            return "float"
        elif isinstance(tvar, Choices):
            # Check if all choices are strings
            if all(isinstance(c, str) for c in tvar.values):
                return "enum"
            elif all(isinstance(c, bool) for c in tvar.values):
                return "bool"
            else:
                return "enum"  # Fallback to enum for mixed types
        else:
            return "float"  # Default

    @staticmethod
    def _tvar_to_domain(tvar: ParameterRange) -> dict[str, Any]:
        """Convert ParameterRange to TVL 0.9 domain specification.

        TVL 0.9 domains have explicit 'kind' field:
        - enum: {"kind": "enum", "values": [...]}
        - range: {"kind": "range", "range": [low, high], "resolution": step}

        Args:
            tvar: The ParameterRange to convert

        Returns:
            Domain specification dict in TVL 0.9 format
        """
        if isinstance(tvar, Choices):
            return {"kind": "enum", "values": list(tvar.values)}

        if isinstance(tvar, LogRange):
            return {"kind": "range", "range": [tvar.low, tvar.high], "log": True}

        if isinstance(tvar, (IntRange, Range)):
            return ConfigSpace._range_to_domain(tvar)

        # Fallback for other ParameterRange types
        return ConfigSpace._fallback_domain(tvar)

    @staticmethod
    def _range_to_domain(tvar: IntRange | Range) -> dict[str, Any]:
        """Convert IntRange or Range to domain dict."""
        domain: dict[str, Any] = {"kind": "range", "range": [tvar.low, tvar.high]}
        if tvar.step is not None:
            domain["resolution"] = tvar.step
        if tvar.log:
            domain["log"] = True
        return domain

    @staticmethod
    def _fallback_domain(tvar: ParameterRange) -> dict[str, Any]:
        """Convert unknown ParameterRange types to domain dict."""
        config_value = tvar.to_config_value()
        if isinstance(config_value, list):
            return {"kind": "enum", "values": config_value}
        if isinstance(config_value, tuple) and len(config_value) == 2:
            return {"kind": "range", "range": list(config_value)}
        return {"kind": "enum", "values": [config_value]}

    def __repr__(self) -> str:
        """Return string representation."""
        tvar_names = list(self.tvars.keys())
        return f"ConfigSpace(tvars={tvar_names}, constraints={len(self.constraints)})"


__all__ = ["ConfigSpace"]
