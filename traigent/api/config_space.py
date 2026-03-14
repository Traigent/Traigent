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
    >>> result.is_valid
    True
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from traigent.api.constraints import BoolExpr, Constraint
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


@dataclass(frozen=True)
class _ImportedConstraintExpression(BoolExpr):
    """Opaque TVL expression imported from a serialized spec."""

    expression: str
    _evaluator: Callable[[dict[str, Any], dict[str, Any] | None], bool] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def to_expression(self, _var_names: dict[int, str] | str) -> str:
        """Return the original expression unchanged."""
        return self.expression

    def evaluate_config(
        self, config: dict[str, Any], _var_names: dict[int, str]
    ) -> bool:
        """Evaluate the stored TVL expression against a config."""
        evaluator = self._evaluator
        if evaluator is None:
            from traigent.tvl.spec_loader import compile_constraint_expression

            evaluator = compile_constraint_expression(
                self.expression,
                label=f"imported_constraint:{self.expression}",
            )
            object.__setattr__(self, "_evaluator", evaluator)
        return bool(evaluator(config, None))

    def explain(self, _var_names: dict[int, str] | None = None) -> str:
        """Surface the raw expression for human-readable diagnostics."""
        return self.expression


@dataclass(frozen=True)
class ConfigSpace:
    """A complete configuration space with TVARs and constraints.

    This is the SE-friendly equivalent of a TVL spec's tvars + constraints.
    It combines parameter definitions (Range, IntRange, Choices, etc.) with
    structural constraints and provides validation capabilities.

    ConfigSpace can be created directly or built from decorator arguments.
    It supports exporting to TVL-compatible format for integration with
    the TVL ecosystem.

    Attributes:
        tvars: Read-only mapping of parameter names to ParameterRange definitions
        constraints: Immutable tuple of structural constraints on the space
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

    tvars: Mapping[str, ParameterRange]
    constraints: tuple[Constraint, ...] = field(default_factory=tuple)
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate the configuration space after initialization."""
        object.__setattr__(self, "tvars", MappingProxyType(dict(self.tvars)))
        object.__setattr__(self, "constraints", tuple(self.constraints))

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
        configuration_space: Mapping[str, Any] | None = None,
        inline_params: Mapping[str, Any] | None = None,
        constraints: Sequence[Constraint] | None = None,
        description: str | None = None,
    ) -> ConfigSpace:
        """Build ConfigSpace from decorator arguments.

        This factory method handles the various ways parameters can be
        specified to the @optimize decorator:
        1. Via configuration_space dict
        2. Via inline **kwargs
        3. Or a combination of both

        Args:
            configuration_space: Mapping of parameter definitions
            inline_params: Mapping of inline parameter definitions
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
            constraints=tuple(constraints or ()),
            description=description,
        )

    @classmethod
    def _process_param_dict(
        cls, params: Mapping[str, Any], tvars: dict[str, ParameterRange]
    ) -> None:
        """Process a parameter dictionary and add ranges to tvars.

        Args:
            params: Mapping of parameter definitions
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
    def from_tvl_spec(cls, spec: Mapping[str, Any]) -> ConfigSpace:
        """Build ConfigSpace from a TVL spec dictionary."""
        if "tvars" in spec:
            tvars = cls._tvars_from_spec(spec["tvars"])
        elif "configuration_space" in spec:
            configuration_space = spec["configuration_space"]
            if not isinstance(configuration_space, Mapping):
                raise ValueError("TVL configuration_space must be a mapping")
            tvars = {}
            cls._process_param_dict(configuration_space, tvars)
        else:
            raise ValueError(
                "TVL spec must define either 'tvars' or 'configuration_space'"
            )

        constraints = cls._constraints_from_spec(spec.get("constraints"))
        description = spec.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError("TVL description must be a string when provided")

        return cls(tvars=tvars, constraints=constraints, description=description)

    @classmethod
    def _tvars_from_spec(cls, tvars_section: Any) -> dict[str, ParameterRange]:
        """Reconstruct ParameterRange instances from a TVL tvars section."""
        if not isinstance(tvars_section, list) or not tvars_section:
            raise ValueError("TVL 'tvars' must be a non-empty list")

        tvars: dict[str, ParameterRange] = {}
        for idx, entry in enumerate(tvars_section):
            if not isinstance(entry, Mapping):
                raise ValueError(f"TVAR at index {idx} must be a mapping")
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"TVAR at index {idx} requires a 'name' string")
            tvars[name] = cls._spec_entry_to_range(name, entry)
        return tvars

    @classmethod
    def _spec_entry_to_range(
        cls, name: str, entry: Mapping[str, Any]
    ) -> ParameterRange:
        """Convert a TVL tvar entry back into a ParameterRange."""
        domain = entry.get("domain")
        if isinstance(domain, list):
            spec_dict: dict[str, Any] = {
                "values": list(domain),
                "default": entry.get("default"),
            }
        elif isinstance(domain, Mapping):
            kind = domain.get("kind")
            if kind == "enum":
                spec_dict = {
                    "values": list(domain.get("values", [])),
                    "default": entry.get("default"),
                }
            elif kind == "range":
                range_values = domain.get("range")
                if not isinstance(range_values, list) or len(range_values) != 2:
                    raise ValueError(
                        f"TVAR '{name}' range domain must contain a 2-item list"
                    )
                spec_dict = {
                    "low": range_values[0],
                    "high": range_values[1],
                    "step": domain.get("resolution"),
                    "log": bool(domain.get("log", False)),
                    "default": entry.get("default"),
                }
            else:
                raise ValueError(f"TVAR '{name}' has unsupported domain kind '{kind}'")
        else:
            raise ValueError(f"TVAR '{name}' domain must be a list or mapping")

        if isinstance(entry.get("unit"), str):
            spec_dict["unit"] = entry["unit"]
        if isinstance(entry.get("agent"), str):
            spec_dict["agent"] = entry["agent"]

        range_type = entry.get("x_traigent_parameter_range")
        return cls._build_parameter_range(name, spec_dict, range_type)

    @classmethod
    def _build_parameter_range(
        cls,
        name: str,
        spec_dict: dict[str, Any],
        range_type: Any,
    ) -> ParameterRange:
        """Reconstruct the exact ParameterRange subclass from a TVL entry."""
        if range_type == "Choices":
            values = spec_dict.get("values")
            if not isinstance(values, list):
                raise ValueError(f"TVAR '{name}' Choices spec requires 'values'")
            return Choices(
                values=values,
                name=name,
                default=spec_dict.get("default"),
                unit=spec_dict.get("unit"),
                agent=spec_dict.get("agent"),
            )

        if range_type == "LogRange":
            return LogRange(
                low=float(spec_dict["low"]),
                high=float(spec_dict["high"]),
                default=spec_dict.get("default"),
                name=name,
                unit=spec_dict.get("unit"),
                agent=spec_dict.get("agent"),
            )

        if range_type == "IntRange":
            low = cls._coerce_integral_spec_value(name, "low", spec_dict["low"])
            high = cls._coerce_integral_spec_value(name, "high", spec_dict["high"])
            if low is None or high is None:  # pragma: no cover - defensive guard
                raise ValueError(f"TVAR '{name}' IntRange bounds are required")
            return IntRange(
                low=low,
                high=high,
                step=cls._coerce_integral_spec_value(
                    name,
                    "step",
                    spec_dict.get("step"),
                    allow_none=True,
                ),
                log=bool(spec_dict.get("log", False)),
                default=cls._coerce_integral_spec_value(
                    name,
                    "default",
                    spec_dict.get("default"),
                    allow_none=True,
                ),
                name=name,
                unit=spec_dict.get("unit"),
                agent=spec_dict.get("agent"),
            )

        if range_type == "Range":
            return Range(
                low=float(spec_dict["low"]),
                high=float(spec_dict["high"]),
                step=spec_dict.get("step"),
                log=bool(spec_dict.get("log", False)),
                default=spec_dict.get("default"),
                name=name,
                unit=spec_dict.get("unit"),
                agent=spec_dict.get("agent"),
            )

        return cls._dict_to_range(name, spec_dict)

    @staticmethod
    def _coerce_integral_spec_value(
        name: str,
        field_name: str,
        value: Any,
        *,
        allow_none: bool = False,
    ) -> int | None:
        """Coerce an imported TVL field to int without silent truncation."""
        if value is None:
            if allow_none:
                return None
            raise ValueError(f"TVAR '{name}' field '{field_name}' is required")

        if isinstance(value, bool):
            raise ValueError(
                f"TVAR '{name}' field '{field_name}' must be an integer, got bool"
            )

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ValueError(
                f"TVAR '{name}' field '{field_name}' must be integral, got {value!r}"
            )

        raise ValueError(
            f"TVAR '{name}' field '{field_name}' must be an integer, "
            f"got {type(value).__name__}"
        )

    @classmethod
    def _constraints_from_spec(cls, constraints_section: Any) -> tuple[Constraint, ...]:
        """Rebuild structural constraints from a TVL constraints section."""
        if constraints_section is None:
            return ()

        entries: list[Any]
        if isinstance(constraints_section, list):
            entries = constraints_section
        elif isinstance(constraints_section, Mapping):
            structural = constraints_section.get("structural", [])
            if not isinstance(structural, list):
                raise ValueError("TVL 'constraints.structural' must be a list")
            entries = structural
        else:
            raise ValueError("TVL constraints must be a list or mapping")

        constraints: list[Constraint] = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"Constraint at index {idx} must be a mapping")

            expr = entry.get("expr")
            when = entry.get("when")
            then = entry.get("then")
            description = entry.get("description")
            constraint_id = entry.get("id")

            if description is not None and not isinstance(description, str):
                raise ValueError(
                    f"Constraint at index {idx} description must be a string"
                )
            if constraint_id is not None and not isinstance(constraint_id, str):
                raise ValueError(f"Constraint at index {idx} id must be a string")

            if isinstance(expr, str):
                constraints.append(
                    Constraint(
                        expr=_ImportedConstraintExpression(expr),
                        description=description,
                        id=constraint_id,
                    )
                )
                continue

            if isinstance(when, str) and isinstance(then, str):
                constraints.append(
                    Constraint(
                        when=_ImportedConstraintExpression(when),
                        then=_ImportedConstraintExpression(then),
                        description=description,
                        id=constraint_id,
                    )
                )
                continue

            raise ValueError(
                f"Constraint at index {idx} must define 'expr' or both 'when' and 'then'"
            )

        return tuple(constraints)

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
            "x_traigent_parameter_range": type(tvar).__name__,
        }
        if hasattr(tvar, "unit") and tvar.unit:
            tvar_dict["unit"] = tvar.unit
        if hasattr(tvar, "default") and tvar.default is not None:
            tvar_dict["default"] = tvar.default
        if hasattr(tvar, "agent") and tvar.agent:
            tvar_dict["agent"] = tvar.agent
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
