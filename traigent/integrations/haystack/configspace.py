"""Configuration space data models for optimization.

This module defines the ExplorationSpace and TVAR types used for optimization
search space definition. These types are separate from the discovery types
(PipelineSpec, DiscoveredTVAR) to allow:

1. Discovery to capture raw pipeline state without optimization constraints
2. Optimization to add ranges, conditionals, and sampling logic
3. Users to modify the search space after discovery but before optimization

Terminology (aligned with TVL Glossary v2.0):
- TVAR (tᵢ): A Tuned Variable - a single controllable knob influencing behavior
- TVARConstraint: Domain constraint (Dᵢ) defining valid values for a single TVAR
  - CategoricalConstraint: Domain with discrete choices
  - NumericalConstraint: Domain with numerical range [min, max]
- ExplorationSpace: Currently the parameter space Θ = D₁ × D₂ × ... × Dₙ
  - Will become full 𝒳 when structural constraints (C^str) are added (Story 2.4)
  - Will incorporate operational constraints (C^op) in Epic 4

TVL Glossary Mapping:
    TVL Term                 | Implementation
    -------------------------|------------------
    TVAR (tᵢ)                | TVAR class
    Domain (Dᵢ)              | TVARConstraint
    Configuration (θ)        | dict[str, Any]
    Structural Constraint    | Categorical/NumericalConstraint
    ExplorationSpace (𝒳)     | ExplorationSpace class
    TSpec (future)           | Epic 7
    GSpec (future)           | Epic 5/6

Relationship to Discovery Types:
    PipelineSpec (discovery) -> ExplorationSpace (optimization)
    DiscoveredTVAR (discovery) -> TVAR (optimization)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .models import PipelineSpec


@dataclass
class CategoricalConstraint:
    """Domain constraint (Dᵢ) for categorical/discrete choice parameters.

    Defines valid values for a single TVAR with discrete choices.
    Used for Literal types, enums, booleans, and model selection.

    Attributes:
        choices: List of valid values for this parameter (the domain Dᵢ).

    Example:
        >>> constraint = CategoricalConstraint(choices=["gpt-4o", "gpt-4o-mini"])
        >>> constraint.validate("gpt-4o")
        True
        >>> constraint.validate("invalid-model")
        False
    """

    choices: list[Any]

    def validate(self, value: Any) -> bool:
        """Check if a value is in the allowed choices.

        Args:
            value: The value to validate.

        Returns:
            True if value is in choices, False otherwise.
        """
        return value in self.choices

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        if len(self.choices) <= 5:
            return f"CategoricalConstraint(choices={self.choices})"
        return f"CategoricalConstraint(choices=[{len(self.choices)} items])"


@dataclass
class NumericalConstraint:
    """Domain constraint (Dᵢ) for numerical parameters with range bounds.

    Defines valid values for a single TVAR as a continuous or discrete range.
    The domain is [min, max] with optional log scale for sampling.

    Attributes:
        min: Minimum value (inclusive) - lower bound of domain Dᵢ.
        max: Maximum value (inclusive) - upper bound of domain Dᵢ.
        log_scale: If True, values are sampled uniformly in log space.
        step: Step size for discrete numerical values (None for continuous).

    Example:
        >>> constraint = NumericalConstraint(min=0.0, max=2.0)
        >>> constraint.validate(1.0)
        True
        >>> constraint.validate(3.0)
        False
    """

    min: float | int
    max: float | int
    log_scale: bool = False
    step: float | int | None = None

    def validate(self, value: float | int | None) -> bool:
        """Check if a value is within the allowed range and on step grid.

        Args:
            value: The value to validate.

        Returns:
            True if min <= value <= max and value is on step grid (if step set).
            Returns False for None values (use TVAR.is_optional to check if None is valid).
        """
        # None is not valid for numerical constraints
        # (optional handling is done at the TVAR/ExplorationSpace level)
        if value is None:
            return False

        if not (self.min <= value <= self.max):
            return False

        # Check step alignment if step is set
        if self.step is not None and self.step > 0:
            # Check if (value - min) is a multiple of step (with floating point tolerance)
            offset = value - self.min
            remainder = offset % self.step
            # Allow for floating point tolerance
            tolerance = 1e-9 * max(abs(self.step), 1)
            if not (remainder < tolerance or abs(remainder - self.step) < tolerance):
                return False

        return True

    @property
    def is_discrete(self) -> bool:
        """Check if this constraint represents discrete values."""
        return self.step is not None

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        parts = [f"min={self.min}", f"max={self.max}"]
        if self.log_scale:
            parts.append("log_scale=True")
        if self.step is not None:
            parts.append(f"step={self.step}")
        return f"NumericalConstraint({', '.join(parts)})"


# Forward reference for ConditionalConstraint (defined below)
# Type alias will be updated after ConditionalConstraint is defined


@dataclass
class ConditionalConstraint:
    """Constraint dependent on another TVAR's value (structural constraint C^str).

    Implements structural constraints where the domain of one TVAR depends on
    the value of another TVAR. This is the most common form of inter-TVAR
    dependency in optimization search spaces.

    Attributes:
        parent_qualified_name: The TVAR this constraint depends on (e.g., "generator.model")
        conditions: Mapping of parent values to child constraints
        default_constraint: Fallback constraint if parent value not in conditions

    Example:
        >>> constraint = ConditionalConstraint(
        ...     parent_qualified_name="generator.model",
        ...     conditions={
        ...         "gpt-4o": NumericalConstraint(min=100, max=8192),
        ...         "gpt-4o-mini": NumericalConstraint(min=100, max=4096),
        ...     }
        ... )
        >>> constraint.get_constraint_for("gpt-4o")
        NumericalConstraint(min=100, max=8192)
    """

    parent_qualified_name: str
    conditions: dict[Any, CategoricalConstraint | NumericalConstraint]
    default_constraint: CategoricalConstraint | NumericalConstraint | None = None

    def get_constraint_for(
        self, parent_value: Any
    ) -> CategoricalConstraint | NumericalConstraint | None:
        """Get the constraint for a specific parent value.

        Args:
            parent_value: The value of the parent TVAR.

        Returns:
            The constraint for this parent value, or default_constraint if not found.
        """
        return self.conditions.get(parent_value, self.default_constraint)

    def validate(self, value: Any, parent_value: Any) -> bool:
        """Validate value given the parent's value.

        Args:
            value: The value to validate.
            parent_value: The value of the parent TVAR.

        Returns:
            True if valid, False otherwise. Returns True if no constraint
            is defined for this parent value.
        """
        constraint = self.get_constraint_for(parent_value)
        if constraint is None:
            return True  # No constraint for this parent value
        return constraint.validate(value)

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        num_conditions = len(self.conditions)
        return (
            f"ConditionalConstraint(parent='{self.parent_qualified_name}', "
            f"conditions={num_conditions})"
        )


# Type alias for constraint union (structural constraints C^str)
# Now includes ConditionalConstraint for inter-TVAR dependencies
TVARConstraint = CategoricalConstraint | NumericalConstraint | ConditionalConstraint

# Type alias for Configuration (θ) - assignment to all TVARs in scope
# Maps qualified TVAR names (e.g., "generator.temperature") to their values
Configuration = dict[str, Any]


@dataclass
class TVAR:
    """A Tuned Variable (tᵢ) for optimization search space.

    In TVL Glossary v2.0: A TVAR is a single controllable knob that influences
    the behavior, cost, or quality of a Tunable. Each TVAR has a domain (Dᵢ)
    defining its valid values, represented by a TVARConstraint.

    TVARs are created from DiscoveredTVARs during conversion from PipelineSpec
    to ExplorationSpace.

    Attributes:
        name: The parameter name (e.g., "temperature").
        scope: The component scope (e.g., "generator").
        python_type: Python type as string ("int", "float", "str", "bool").
        default_value: The current/default value of the parameter.
        constraint: Domain constraint (Dᵢ) - categorical or numerical.
        is_tunable: Whether this TVAR is included in optimization.
        is_optional: Whether None is a valid value (for Optional[T] types).
            For numerical constraints, sample() uses a probability to return
            None instead of sampling from the range.

    Example:
        >>> tvar = TVAR(
        ...     name="temperature",
        ...     scope="generator",
        ...     python_type="float",
        ...     default_value=0.7,
        ...     constraint=NumericalConstraint(min=0.0, max=2.0)
        ... )
        >>> tvar.qualified_name
        'generator.temperature'
    """

    name: str
    scope: str
    python_type: str
    default_value: Any
    constraint: TVARConstraint | None = None
    is_tunable: bool = True
    is_optional: bool = False  # True if None is a valid value (for Optional[T] types)
    # Private fields for fix/unfix - store original state when fixed
    _original_constraint: TVARConstraint | None = field(default=None, repr=False)
    _original_default: Any = field(default=None, repr=False)
    _is_fixed: bool = field(default=False, repr=False)

    @property
    def qualified_name(self) -> str:
        """Get the fully qualified name (scope.name).

        Returns:
            Full path like 'generator.temperature'.
        """
        return f"{self.scope}.{self.name}"

    def validate(self, value: Any, parent_value: Any = None) -> bool:
        """Validate a value against this TVAR's constraint.

        Args:
            value: The value to validate.
            parent_value: For conditional constraints, the parent TVAR's value.
                         Ignored for non-conditional constraints.

        Returns:
            True if the value is valid, False otherwise.
            Returns True if no constraint is defined.
        """
        if self.constraint is None:
            return True
        if isinstance(self.constraint, ConditionalConstraint):
            return self.constraint.validate(value, parent_value)
        return self.constraint.validate(value)

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        tunable_str = "tunable" if self.is_tunable else "fixed"
        constraint_str = ""
        if self.constraint is not None:
            if isinstance(self.constraint, CategoricalConstraint):
                if len(self.constraint.choices) <= 3:
                    constraint_str = f", choices={self.constraint.choices}"
                else:
                    constraint_str = f", choices=[{len(self.constraint.choices)} items]"
            elif isinstance(self.constraint, NumericalConstraint):
                constraint_str = (
                    f", range=[{self.constraint.min}, {self.constraint.max}]"
                )
            elif isinstance(self.constraint, ConditionalConstraint):
                constraint_str = (
                    f", conditional(parent='{self.constraint.parent_qualified_name}')"
                )
        return (
            f"TVAR('{self.qualified_name}', type='{self.python_type}', "
            f"default={self.default_value!r}{constraint_str}, {tunable_str})"
        )


@dataclass
class ExplorationSpace:
    """The optimization search space (𝒳) with TVAR domains and constraints.

    Represents the feasible configuration space 𝒳 = Θ ∩ C^str where:
    - Θ = D₁ × D₂ × ... × Dₙ is the Cartesian product of per-TVAR domains
    - C^str are structural constraints (conditionals between TVARs)

    Supports:
    - Categorical constraints: Discrete choices for parameters
    - Numerical constraints: Ranges with optional log-scale and step
    - Conditional constraints: Inter-TVAR dependencies (child domain depends on parent value)
    - Optional parameters: None is a valid value for Optional[T] types
    - Fix/unfix: Explicitly exclude or include TVARs from optimization

    Roadmap:
    - Epic 4: Operational constraints (C^op) for cost/latency budgets
    - Epic 7: TSpec integration for full tunable specification

    Attributes:
        tvars: Dictionary mapping qualified names to TVAR objects.

    Class Constants:
        OPTIONAL_NONE_PROBABILITY: Probability of sampling None for optional
            numerical parameters (default: 0.1 = 10%).

    Example:
        >>> from traigent.integrations.haystack import from_pipeline
        >>> pipeline_spec = from_pipeline(pipeline)
        >>> space = ExplorationSpace.from_pipeline_spec(pipeline_spec)
        >>> space.set_conditional("generator.max_tokens", "generator.model", {
        ...     "gpt-4o": {"min": 100, "max": 8192},
        ...     "gpt-4o-mini": {"min": 100, "max": 4096},
        ... })
        >>> config = space.sample()
    """

    tvars: dict[str, TVAR] = field(default_factory=dict)

    # Class constant: Probability of sampling None for optional numerical parameters
    # Cannot be a field since it's a class-level configuration
    OPTIONAL_NONE_PROBABILITY: ClassVar[float] = 0.1  # 10% chance of None

    def get_tvar(self, scope: str, name: str) -> TVAR | None:
        """Get a TVAR by scope and name.

        Args:
            scope: The scope name (e.g., "generator").
            name: The TVAR name (e.g., "temperature").

        Returns:
            The TVAR if found, None otherwise.
        """
        return self.tvars.get(f"{scope}.{name}")

    def get_tvar_by_qualified_name(self, qualified_name: str) -> TVAR | None:
        """Get a TVAR by its fully qualified name.

        Args:
            qualified_name: The full path (e.g., "generator.temperature").

        Returns:
            The TVAR if found, None otherwise.
        """
        return self.tvars.get(qualified_name)

    @property
    def tunable_tvars(self) -> dict[str, TVAR]:
        """Return only tunable TVARs.

        Returns:
            Dictionary of TVARs where is_tunable=True.
        """
        return {k: v for k, v in self.tvars.items() if v.is_tunable}

    @property
    def fixed_tvars(self) -> dict[str, TVAR]:
        """Return only fixed (non-tunable) TVARs.

        Returns:
            Dictionary of TVARs where is_tunable=False.
        """
        return {k: v for k, v in self.tvars.items() if not v.is_tunable}

    @property
    def scope_names(self) -> list[str]:
        """Get unique scope names in this exploration space.

        Returns:
            List of unique scope names.
        """
        return list({tvar.scope for tvar in self.tvars.values()})

    def get_tvars_by_scope(self, scope: str) -> dict[str, TVAR]:
        """Get all TVARs in a specific scope.

        Args:
            scope: The scope name to filter by.

        Returns:
            Dictionary of TVARs in the specified scope.
        """
        return {k: v for k, v in self.tvars.items() if v.scope == scope}

    @classmethod
    def from_pipeline_spec(cls, spec: PipelineSpec) -> ExplorationSpace:
        """Create an ExplorationSpace from a discovered PipelineSpec.

        Converts DiscoveredTVAR objects to optimization TVAR objects,
        mapping discovery attributes to optimization constraints.

        Args:
            spec: A PipelineSpec from pipeline introspection.

        Returns:
            An ExplorationSpace with all tunable and fixed TVARs.

        Mapping:
            - literal_choices -> CategoricalConstraint
            - default_range -> NumericalConstraint
            - range_type="log" -> NumericalConstraint(log_scale=True)
            - range_type="discrete" -> NumericalConstraint(step=1)
            - python_type="bool" -> CategoricalConstraint([True, False])
            - is_optional=True for categorical -> Include None in choices
            - is_optional=True for numerical -> TVAR.is_optional=True (samples None with 10% probability)
        """
        tvars: dict[str, TVAR] = {}

        for scope in spec.scopes:
            for _tvar_name, discovered in scope.tvars.items():
                constraint = cls._convert_to_constraint(discovered)

                # Determine if this TVAR should be marked as optional
                # For categorical constraints, None is already in choices if optional
                # For numerical constraints (or no constraint), we track is_optional separately
                # This allows users to later set a range and have optional semantics work
                is_optional = discovered.is_optional and not isinstance(
                    constraint, CategoricalConstraint
                )

                tvar = TVAR(
                    name=discovered.name,
                    scope=scope.name,
                    python_type=discovered.python_type,
                    default_value=discovered.value,
                    constraint=constraint,
                    is_tunable=discovered.is_tunable,
                    is_optional=is_optional,
                )
                tvars[tvar.qualified_name] = tvar

        return cls(tvars=tvars)

    @staticmethod
    def _convert_to_constraint(
        discovered: DiscoveredTVAR,  # noqa: F821
    ) -> TVARConstraint | None:
        """Convert a DiscoveredTVAR to a TVARConstraint.

        Args:
            discovered: A DiscoveredTVAR from introspection.

        Returns:
            A TVARConstraint (categorical or numerical) or None.
        """
        # Import here to avoid circular imports

        # Handle boolean types as categorical
        if discovered.python_type == "bool":
            choices: list[Any] = [True, False]
            if discovered.is_optional:
                choices.append(None)
            return CategoricalConstraint(choices=choices)

        # Handle Literal types with choices
        if discovered.literal_choices is not None:
            choices = list(discovered.literal_choices)
            if discovered.is_optional and None not in choices:
                choices.append(None)
            return CategoricalConstraint(choices=choices)

        # Handle numerical types with ranges
        if discovered.default_range is not None:
            min_val, max_val = discovered.default_range
            log_scale = discovered.range_type == "log"
            step = 1 if discovered.range_type == "discrete" else None
            return NumericalConstraint(
                min=min_val,
                max=max_val,
                log_scale=log_scale,
                step=step,
            )

        # No constraint for parameters without range or choices
        return None

    @classmethod
    def from_tvl_spec(
        cls,
        path: str | Path,
        environment: str | None = None,
    ) -> ExplorationSpace:
        """Create an ExplorationSpace from a TVL specification file.

        Loads a TVL spec using the existing traigent.tvl.spec_loader infrastructure
        and converts the configuration_space section to an ExplorationSpace with
        appropriate TVAR constraints.

        Args:
            path: Path to the TVL specification file (YAML format).
            environment: Optional environment name for environment-specific overrides.

        Returns:
            An ExplorationSpace configured per the TVL definitions.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            TVLValidationError: If the TVL syntax is invalid.

        Example:
            >>> space = ExplorationSpace.from_tvl_spec("search_space.tvl")
            >>> space = ExplorationSpace.from_tvl_spec("config.yaml", environment="dev")

        TVL Format:
            The TVL file should have a configuration_space section:

            ```yaml
            configuration_space:
              generator.model:
                type: categorical
                values: ["gpt-4o", "gpt-4o-mini"]
                default: "gpt-4o"
              generator.temperature:
                type: continuous
                range: [0.0, 2.0]
                default: 0.7
              retriever.top_k:
                type: integer
                range: [1, 50]
                default: 10
            ```
        """
        from traigent.tvl.spec_loader import load_tvl_spec

        path = Path(path)
        spec = load_tvl_spec(spec_path=path, environment=environment)

        tvars: dict[str, TVAR] = {}
        for qualified_name, domain in spec.configuration_space.items():
            tvar = cls._tvar_from_tvl_domain(
                qualified_name, domain, spec.default_config.get(qualified_name)
            )
            tvars[qualified_name] = tvar

        return cls(tvars=tvars)

    @classmethod
    def _tvar_from_tvl_domain(
        cls,
        qualified_name: str,
        domain: Any,
        default_value: Any,
    ) -> TVAR:
        """Convert a TVL domain specification to a TVAR.

        Args:
            qualified_name: Full parameter path like 'generator.model'.
            domain: TVL domain - list for categorical or tuple for numerical.
            default_value: Default value from the TVL spec.

        Returns:
            A TVAR with appropriate constraint.
        """
        # Extract scope and name from qualified name
        parts = qualified_name.rsplit(".", 1)
        if len(parts) == 2:
            scope, name = parts
        else:
            scope, name = "default", qualified_name

        # Determine constraint and python_type based on domain format
        if isinstance(domain, list):
            # Categorical constraint
            constraint = CategoricalConstraint(choices=domain)
            # Infer python_type from first non-None choice
            python_type = _infer_type_from_choices(domain)
            is_optional = None in domain
        elif isinstance(domain, tuple) and len(domain) == 2:
            # Numerical constraint (min, max)
            min_val, max_val = domain
            constraint = NumericalConstraint(min=min_val, max=max_val)
            # Determine type based on values
            if isinstance(min_val, int) and isinstance(max_val, int):
                python_type = "int"
            else:
                python_type = "float"
            is_optional = False
        else:
            # Unknown format - treat as fixed
            constraint = None
            python_type = type(domain).__name__ if domain is not None else "Any"
            is_optional = domain is None

        return TVAR(
            name=name,
            scope=scope,
            python_type=python_type,
            default_value=default_value if default_value is not None else domain,
            constraint=constraint,
            is_tunable=constraint is not None,
            is_optional=is_optional,
        )

    def to_tvl(
        self,
        path: str | Path,
        *,
        include_metadata: bool = True,
        description: str | None = None,
    ) -> None:
        """Export this ExplorationSpace to a TVL specification file.

        Creates a TVL file with a configuration_space section that can be
        reloaded with from_tvl_spec() for round-trip preservation.

        Args:
            path: Path to write the TVL file.
            include_metadata: If True, include a metadata section with export info.
            description: Optional description to include in metadata.

        Example:
            >>> space.to_tvl("exported_space.tvl")
            >>> reloaded = ExplorationSpace.from_tvl_spec("exported_space.tvl")
        """
        import datetime

        import yaml

        path = Path(path)

        data: dict[str, Any] = {}

        # Add metadata if requested
        if include_metadata:
            data["metadata"] = {
                "description": description or "Exported ExplorationSpace",
                "exported_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "num_tvars": len(self.tvars),
                "num_tunable": len(self.tunable_tvars),
            }

        # Build configuration_space section
        config_space: dict[str, dict[str, Any]] = {}
        for qualified_name, tvar in self.tvars.items():
            param_spec = self._tvar_to_tvl_spec(tvar)
            config_space[qualified_name] = param_spec

        data["configuration_space"] = config_space

        # Write YAML file
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _tvar_to_tvl_spec(self, tvar: TVAR) -> dict[str, Any]:
        """Convert a TVAR to a TVL parameter specification.

        Args:
            tvar: The TVAR to convert.

        Returns:
            A dictionary suitable for YAML output.
        """
        spec = self._constraint_to_tvl_spec(tvar)

        # Add default value
        if tvar.default_value is not None:
            spec["default"] = tvar.default_value

        return spec

    def _constraint_to_tvl_spec(self, tvar: TVAR) -> dict[str, Any]:
        """Convert a TVAR's constraint to TVL spec format."""
        if isinstance(tvar.constraint, CategoricalConstraint):
            return self._categorical_to_tvl_spec(tvar.constraint)
        elif isinstance(tvar.constraint, NumericalConstraint):
            return self._numerical_to_tvl_spec(tvar.constraint, tvar.python_type)
        elif isinstance(tvar.constraint, ConditionalConstraint):
            return self._conditional_to_tvl_spec(tvar.constraint)
        else:
            return {"type": "fixed", "value": tvar.default_value}

    def _categorical_to_tvl_spec(
        self, constraint: CategoricalConstraint
    ) -> dict[str, Any]:
        """Convert a categorical constraint to TVL spec."""
        return {"type": "categorical", "values": constraint.choices}

    def _numerical_to_tvl_spec(
        self, constraint: NumericalConstraint, python_type: str
    ) -> dict[str, Any]:
        """Convert a numerical constraint to TVL spec."""
        spec: dict[str, Any] = {
            "type": "integer" if python_type == "int" else "continuous",
            "range": [constraint.min, constraint.max],
        }
        if constraint.log_scale:
            spec["log_scale"] = True
        if constraint.step is not None:
            spec["step"] = constraint.step
        return spec

    def _conditional_to_tvl_spec(
        self, constraint: ConditionalConstraint
    ) -> dict[str, Any]:
        """Convert a conditional constraint to TVL spec."""
        spec: dict[str, Any] = {
            "type": "conditional",
            "parent": constraint.parent_qualified_name,
            "conditions": {
                parent_val: self._inner_constraint_to_spec(inner)
                for parent_val, inner in constraint.conditions.items()
            },
        }
        if constraint.default_constraint is not None:
            spec["default"] = self._inner_constraint_to_spec(
                constraint.default_constraint
            )
        return spec

    def _inner_constraint_to_spec(
        self, constraint: TVARConstraint | None
    ) -> dict[str, Any]:
        """Convert an inner constraint to a TVL spec fragment."""
        if isinstance(constraint, CategoricalConstraint):
            return {"type": "categorical", "values": constraint.choices}
        elif isinstance(constraint, NumericalConstraint):
            result: dict[str, Any] = {"range": [constraint.min, constraint.max]}
            if constraint.log_scale:
                result["log_scale"] = True
            if constraint.step is not None:
                result["step"] = constraint.step
            return result
        return {}

    def set_choices(self, qualified_name: str, choices: list[Any]) -> None:
        """Set or override choices for a categorical TVAR.

        Modifies the domain (Dᵢ) of a TVAR to be a categorical constraint
        with the specified choices. This can convert a numerical TVAR to
        categorical or update an existing categorical TVAR's choices.

        Args:
            qualified_name: Full path like 'generator.model'.
            choices: List of valid values for this parameter. Order is
                preserved (affects sampling). Duplicates are allowed but
                will bias sampling toward repeated values.

        Raises:
            KeyError: If the TVAR is not found.
            ValueError: If the choices list is empty.

        Note:
            If the TVAR was previously optional (with None as a valid value),
            you must include None in the new choices list to preserve that.

        Example:
            >>> space.set_choices("generator.model", ["gpt-4o", "gpt-4o-mini"])
        """
        if not choices:
            raise ValueError(f"Choices list cannot be empty for {qualified_name}")

        tvar = self.tvars.get(qualified_name)
        if tvar is None:
            raise KeyError(f"TVAR not found: {qualified_name}")

        # Replace constraint with CategoricalConstraint
        tvar.constraint = CategoricalConstraint(choices=list(choices))

    def set_range(
        self,
        qualified_name: str,
        min_val: float | int,
        max_val: float | int,
        log_scale: bool = False,
        step: float | int | None = None,
    ) -> None:
        """Set or override range for a numerical TVAR.

        Modifies the domain (Dᵢ) of a TVAR to be a numerical constraint
        with the specified range. This can convert a categorical TVAR to
        numerical or update an existing numerical TVAR's range.

        Args:
            qualified_name: Full path like 'generator.temperature'.
            min_val: Minimum value (inclusive).
            max_val: Maximum value (inclusive).
            log_scale: If True, sample uniformly in log space.
            step: Step size for discrete values (None for continuous).

        Raises:
            KeyError: If the TVAR is not found.
            ValueError: If min_val >= max_val, or step <= 0.

        Example:
            >>> space.set_range("generator.temperature", 0.0, 1.5)
        """
        if min_val >= max_val:
            raise ValueError(
                f"min_val ({min_val}) must be less than max_val ({max_val})"
            )
        if step is not None and step <= 0:
            raise ValueError(f"step must be positive, got {step}")

        tvar = self.tvars.get(qualified_name)
        if tvar is None:
            raise KeyError(f"TVAR not found: {qualified_name}")

        tvar.constraint = NumericalConstraint(
            min=min_val, max=max_val, log_scale=log_scale, step=step
        )

    def fix(self, qualified_name: str, value: Any) -> None:
        """Fix a TVAR to a specific value, excluding it from optimization.

        The fixed value is used in all sampled configurations. The original
        constraint is preserved so the TVAR can be unfixed later.

        Args:
            qualified_name: Full path like 'generator.model'.
            value: The value to fix the parameter to.

        Raises:
            KeyError: If the TVAR is not found.
            ValueError: If the value is not valid per the TVAR's constraint.

        Example:
            >>> space.fix("generator.model", "gpt-4o")
            >>> config = space.sample()  # model is always "gpt-4o"
        """
        tvar = self.tvars.get(qualified_name)
        if tvar is None:
            raise KeyError(f"TVAR not found: {qualified_name}")

        # Allow None for optional TVARs
        if value is None and tvar.is_optional:
            pass  # None is valid for optional TVARs
        elif tvar.constraint is not None and not tvar.validate(value):
            # Validate value against constraint (if constraint exists)
            if isinstance(tvar.constraint, CategoricalConstraint):
                raise ValueError(
                    f"Cannot fix {qualified_name} to {value!r}: "
                    f"not in choices {tvar.constraint.choices}"
                )
            elif isinstance(tvar.constraint, NumericalConstraint):
                raise ValueError(
                    f"Cannot fix {qualified_name} to {value!r}: "
                    f"not in range [{tvar.constraint.min}, {tvar.constraint.max}]"
                )
            else:
                raise ValueError(
                    f"Cannot fix {qualified_name} to {value!r}: invalid value"
                )

        # Store original state for unfix (only if not already fixed)
        if not tvar._is_fixed:
            tvar._original_constraint = tvar.constraint
            tvar._original_default = tvar.default_value
            tvar._is_fixed = True

        # Fix the TVAR
        tvar.is_tunable = False
        tvar.default_value = value

    def unfix(self, qualified_name: str) -> None:
        """Restore a fixed TVAR to the optimization search space.

        Restores the original constraint and default value that were saved
        when the TVAR was fixed. If the TVAR was never fixed, this is a no-op.

        Args:
            qualified_name: Full path like 'generator.model'.

        Raises:
            KeyError: If the TVAR is not found.

        Example:
            >>> space.fix("generator.model", "gpt-4o")
            >>> space.unfix("generator.model")
            >>> # model is now back in the search space
        """
        tvar = self.tvars.get(qualified_name)
        if tvar is None:
            raise KeyError(f"TVAR not found: {qualified_name}")

        # Restore original state if it was fixed
        if tvar._is_fixed:
            tvar.constraint = tvar._original_constraint
            tvar.default_value = tvar._original_default
            tvar._original_constraint = None
            tvar._original_default = None
            tvar._is_fixed = False
            tvar.is_tunable = True

    def set_conditional(
        self,
        child: str,
        parent: str,
        conditions: dict[Any, dict[str, Any]],
        default: dict[str, Any] | None = None,
    ) -> None:
        """Set a conditional constraint on a child TVAR based on parent value.

        Creates a ConditionalConstraint that makes the child TVAR's domain
        depend on the parent TVAR's value. This is useful for parameters like
        max_tokens whose valid range depends on which model is selected.

        Args:
            child: Qualified name of the child TVAR (e.g., "generator.max_tokens")
            parent: Qualified name of the parent TVAR (e.g., "generator.model")
            conditions: Mapping of parent values to child constraints.
                Each value should be a dict with either:
                - {"choices": [...]} for categorical constraint
                - {"min": N, "max": M, ...} for numerical constraint
            default: Optional default constraint if parent value not in conditions

        Raises:
            KeyError: If child or parent TVAR is not found
            ValueError: If parent is not categorical, or would create circular dependency

        Example:
            >>> space.set_conditional(
            ...     child="generator.max_tokens",
            ...     parent="generator.model",
            ...     conditions={
            ...         "gpt-4o": {"min": 100, "max": 8192},
            ...         "gpt-4o-mini": {"min": 100, "max": 4096},
            ...     }
            ... )
        """
        # Validate child exists
        child_tvar = self.tvars.get(child)
        if child_tvar is None:
            raise KeyError(f"Child TVAR not found: {child}")

        # Validate parent exists
        parent_tvar = self.tvars.get(parent)
        if parent_tvar is None:
            raise KeyError(f"Parent TVAR not found: {parent}")

        # Check for circular dependencies FIRST (before type checking)
        if self._would_create_cycle(child, parent):
            raise ValueError(
                f"Setting conditional from '{parent}' to '{child}' "
                f"would create a circular dependency"
            )

        # Validate parent has discrete choices (conditionals require discrete parent)
        # This can be CategoricalConstraint directly, or a ConditionalConstraint
        # where all conditions produce categorical choices
        if not self._has_discrete_choices(parent_tvar):
            raise ValueError(
                f"Parent TVAR '{parent}' must have discrete choices "
                f"(CategoricalConstraint or ConditionalConstraint with categorical outputs), "
                f"got {type(parent_tvar.constraint).__name__}"
            )

        # Get parent choices and validate condition keys
        parent_choices = self._get_all_parent_choices(parent_tvar)
        invalid_keys = [k for k in conditions.keys() if k not in parent_choices]
        if invalid_keys:
            raise ValueError(
                f"Condition keys {invalid_keys} are not valid choices for parent '{parent}'. "
                f"Valid choices are: {parent_choices}"
            )

        # Warn if not all parent choices are covered and no default is provided
        uncovered = [c for c in parent_choices if c not in conditions]
        if uncovered and default is None:
            import warnings

            warnings.warn(
                f"Conditional '{child}' does not cover all parent choices: {uncovered}. "
                f"Sampling will use default_value when parent has these values. "
                f"Consider providing a 'default' constraint.",
                UserWarning,
                stacklevel=2,
            )

        # Convert condition specs to actual constraints (with validation)
        converted_conditions: dict[Any, CategoricalConstraint | NumericalConstraint] = (
            {}
        )
        for parent_value, spec in conditions.items():
            constraint = self._spec_to_constraint(spec)
            # Validate the constraint is well-formed
            self._validate_constraint_spec(
                child, constraint, f"when {parent}={parent_value!r}"
            )
            converted_conditions[parent_value] = constraint

        # Convert default if provided (with validation)
        default_constraint = None
        if default is not None:
            default_constraint = self._spec_to_constraint(default)
            self._validate_constraint_spec(
                child, default_constraint, "default constraint"
            )

        # Create and assign the conditional constraint
        child_tvar.constraint = ConditionalConstraint(
            parent_qualified_name=parent,
            conditions=converted_conditions,
            default_constraint=default_constraint,
        )

    def _spec_to_constraint(
        self, spec: dict[str, Any]
    ) -> CategoricalConstraint | NumericalConstraint:
        """Convert a constraint specification dict to a constraint object.

        Args:
            spec: Dict with either {"choices": [...]} or {"min": N, "max": M, ...}

        Returns:
            CategoricalConstraint or NumericalConstraint
        """
        if "choices" in spec:
            return CategoricalConstraint(choices=list(spec["choices"]))
        elif "min" in spec and "max" in spec:
            return NumericalConstraint(
                min=spec["min"],
                max=spec["max"],
                log_scale=spec.get("log_scale", False),
                step=spec.get("step"),
            )
        else:
            raise ValueError(
                f"Invalid constraint spec: must have 'choices' or 'min'/'max', got {spec}"
            )

    def _validate_constraint_spec(
        self,
        tvar_name: str,
        constraint: CategoricalConstraint | NumericalConstraint,
        context: str = "",
    ) -> None:
        """Validate a constraint specification is well-formed.

        Args:
            tvar_name: The TVAR name for error messages.
            constraint: The constraint to validate.
            context: Optional context for error messages (e.g., "when parent=value").

        Raises:
            ValueError: If the constraint is invalid (min >= max, empty choices, etc.)
        """
        ctx = f" ({context})" if context else ""

        if isinstance(constraint, NumericalConstraint):
            if constraint.min >= constraint.max:
                raise ValueError(
                    f"Invalid range for {tvar_name}{ctx}: "
                    f"min ({constraint.min}) >= max ({constraint.max})"
                )
            if constraint.step is not None and constraint.step <= 0:
                raise ValueError(
                    f"Invalid step for {tvar_name}{ctx}: "
                    f"step must be positive, got {constraint.step}"
                )
        elif isinstance(constraint, CategoricalConstraint):
            if len(constraint.choices) == 0:
                raise ValueError(f"Empty choices for {tvar_name}{ctx}")

    def _would_create_cycle(self, child: str, parent: str) -> bool:
        """Check if adding child->parent dependency would create a cycle.

        Args:
            child: The child TVAR name
            parent: The parent TVAR name

        Returns:
            True if adding this dependency would create a cycle
        """
        # Build current dependency graph: child -> parent
        deps: dict[str, str] = {}
        for name, tvar in self.tvars.items():
            if isinstance(tvar.constraint, ConditionalConstraint):
                deps[name] = tvar.constraint.parent_qualified_name

        # Add the proposed dependency
        deps[child] = parent

        # Check for cycle starting from child
        visited: set[str] = set()
        current = child
        while current in deps:
            if current in visited:
                return True
            visited.add(current)
            current = deps[current]

        return False

    def _has_discrete_choices(self, tvar: TVAR) -> bool:
        """Check if a TVAR has discrete/categorical choices.

        A TVAR has discrete choices if it has:
        - CategoricalConstraint directly
        - ConditionalConstraint where ALL conditions produce CategoricalConstraint

        Args:
            tvar: The TVAR to check

        Returns:
            True if the TVAR has discrete choices, False otherwise
        """
        if tvar.constraint is None:
            return False

        if isinstance(tvar.constraint, CategoricalConstraint):
            return True

        if isinstance(tvar.constraint, ConditionalConstraint):
            # Check that all condition constraints are categorical
            for condition_constraint in tvar.constraint.conditions.values():
                if not isinstance(condition_constraint, CategoricalConstraint):
                    return False
            # Check default too if present
            if tvar.constraint.default_constraint is not None and not isinstance(
                tvar.constraint.default_constraint, CategoricalConstraint
            ):
                return False
            # All conditions are categorical
            return True

        return False

    def _get_dependency_order(self) -> list[str]:
        """Get TVAR names in topological order (parents before children).

        Returns:
            List of qualified names where parents appear before their dependents.
        """
        # Build dependency graph: child -> parent
        deps: dict[str, str] = {}
        for name, tvar in self.tvars.items():
            if isinstance(tvar.constraint, ConditionalConstraint):
                deps[name] = tvar.constraint.parent_qualified_name

        # Topological sort using Kahn's algorithm
        # First, identify nodes with no dependencies
        all_names = set(self.tvars.keys())
        children = set(deps.keys())
        roots = all_names - children

        result: list[str] = []
        remaining = set(deps.keys())

        # Add all root nodes first
        result.extend(sorted(roots))

        # Process remaining nodes
        while remaining:
            # Find nodes whose parent is already in result
            ready = [n for n in remaining if deps[n] in result]
            if not ready:
                # This shouldn't happen if _would_create_cycle works correctly
                raise ValueError("Circular dependency detected in TVAR graph")
            for name in sorted(ready):
                result.append(name)
                remaining.remove(name)

        return result

    def _sample_numerical(
        self,
        constraint: NumericalConstraint,
        python_type: str,
        rng: random.Random,
    ) -> float | int:
        """Sample a value from a numerical constraint.

        Args:
            constraint: The numerical constraint defining the domain.
            python_type: The Python type of the TVAR ("int" or "float").
            rng: Random number generator instance.

        Returns:
            Sampled value respecting constraint bounds, step, and log_scale.
        """
        import math

        # Discrete sampling: uniform over grid points
        if constraint.step is not None and constraint.step > 0:
            num_steps = int((constraint.max - constraint.min) / constraint.step)
            step_idx = rng.randint(0, num_steps)
            value = constraint.min + step_idx * constraint.step
            # Keep as int if step and min are integral
            if isinstance(constraint.step, int) and isinstance(constraint.min, int):
                return int(value)
            return value

        # Integer type without step: use randint for uniform discrete
        if python_type == "int":
            return rng.randint(int(constraint.min), int(constraint.max))

        # Log-uniform sampling (only valid for positive ranges)
        if constraint.log_scale and constraint.min > 0:
            log_min = math.log(constraint.min)
            log_max = math.log(constraint.max)
            return math.exp(rng.uniform(log_min, log_max))

        # Uniform sampling (also fallback for log_scale with min <= 0)
        return rng.uniform(constraint.min, constraint.max)

    def sample(self, seed: int | None = None) -> Configuration:
        """Sample a random configuration from the exploration space.

        Generates a Configuration (θ) by sampling from each TVAR's domain (Dᵢ):
        - Categorical TVARs: uniform random selection from choices
        - Numerical TVARs: uniform random within range (log-uniform if log_scale)
        - Conditional TVARs: sampled using constraint for parent's value
        - Fixed TVARs: use default value (not sampled)
        - TVARs without constraints: use default value

        Args:
            seed: Random seed for reproducibility. Uses a local RNG instance
                  to avoid mutating global random state.

        Returns:
            Configuration dict mapping qualified names to sampled values.

        Note:
            For log_scale sampling, min must be > 0. If min <= 0, falls back
            to linear uniform sampling.

            Conditional TVARs are sampled after their parent TVARs to ensure
            the parent's value is available.

        Example:
            >>> config = space.sample(seed=42)
            >>> config
            {'generator.temperature': 0.7, 'generator.model': 'gpt-4o'}
        """
        import random

        # Use local RNG to avoid mutating global state
        rng = random.Random(seed)

        config: Configuration = {}

        # Process TVARs in dependency order (parents before children)
        ordered_names = self._get_dependency_order()

        for qualified_name in ordered_names:
            tvar = self.tvars[qualified_name]
            config[qualified_name] = self._sample_tvar(tvar, config, rng)

        return config

    def _sample_tvar(
        self, tvar: TVAR, config: Configuration, rng: random.Random
    ) -> Any:
        """Sample a single TVAR value.

        Args:
            tvar: The TVAR to sample.
            config: Current configuration (for conditional parent lookups).
            rng: Random number generator.

        Returns:
            Sampled value for this TVAR.
        """
        # Fixed TVARs or no constraint: use default value
        if not tvar.is_tunable or tvar.constraint is None:
            return tvar.default_value

        if isinstance(tvar.constraint, ConditionalConstraint):
            return self._sample_conditional_tvar(tvar, config, rng)

        if isinstance(tvar.constraint, CategoricalConstraint):
            return rng.choice(tvar.constraint.choices)

        if isinstance(tvar.constraint, NumericalConstraint):
            return self._sample_optional_numerical(tvar, rng)

        # Fallback for unknown constraint types
        return tvar.default_value

    def _sample_conditional_tvar(
        self, tvar: TVAR, config: Configuration, rng: random.Random
    ) -> Any:
        """Sample a TVAR with a conditional constraint.

        Args:
            tvar: The TVAR with conditional constraint.
            config: Current configuration for parent value lookup.
            rng: Random number generator.

        Returns:
            Sampled value based on parent's value.
        """
        constraint = tvar.constraint
        if not isinstance(constraint, ConditionalConstraint):
            return tvar.default_value

        parent_value = config.get(constraint.parent_qualified_name)
        effective_constraint = constraint.get_constraint_for(parent_value)

        if effective_constraint is None:
            return tvar.default_value

        return self._sample_with_constraint(
            effective_constraint, tvar.python_type, rng, is_optional=tvar.is_optional
        )

    def _sample_optional_numerical(self, tvar: TVAR, rng: random.Random) -> Any:
        """Sample a numerical TVAR, handling optional (None) values.

        Args:
            tvar: The TVAR with numerical constraint.
            rng: Random number generator.

        Returns:
            Sampled numerical value, or None for optional TVARs.
        """
        if not isinstance(tvar.constraint, NumericalConstraint):
            return tvar.default_value

        # For optional numerical TVARs, sometimes sample None
        if tvar.is_optional and rng.random() < self.OPTIONAL_NONE_PROBABILITY:
            return None

        return self._sample_numerical(tvar.constraint, tvar.python_type, rng)

    def _sample_with_constraint(
        self,
        constraint: CategoricalConstraint | NumericalConstraint,
        python_type: str,
        rng: random.Random,
        *,
        is_optional: bool = False,
    ) -> Any:
        """Sample a value from a constraint (categorical or numerical).

        Args:
            constraint: The constraint to sample from
            python_type: The Python type of the TVAR
            rng: Random number generator
            is_optional: If True, may sample None for numerical constraints

        Returns:
            Sampled value
        """
        if isinstance(constraint, CategoricalConstraint):
            return rng.choice(constraint.choices)
        elif isinstance(constraint, NumericalConstraint):
            # For optional numerical TVARs, sometimes sample None
            if is_optional and rng.random() < self.OPTIONAL_NONE_PROBABILITY:
                return None
            return self._sample_numerical(constraint, python_type, rng)
        else:
            raise ValueError(f"Unknown constraint type: {type(constraint)}")

    def _format_constraint_error(
        self,
        qualified_name: str,
        value: Any,
        constraint: CategoricalConstraint | NumericalConstraint | None,
        parent_context: str = "",
    ) -> str:
        """Format an error message for a constraint violation.

        Args:
            qualified_name: The TVAR name
            value: The invalid value
            constraint: The constraint that was violated
            parent_context: Optional context string like "when parent=value"

        Returns:
            Formatted error message
        """
        suffix = f" {parent_context}" if parent_context else ""
        if isinstance(constraint, CategoricalConstraint):
            return (
                f"Invalid value for {qualified_name}: {value!r} "
                f"not in choices {constraint.choices}{suffix}"
            )
        elif isinstance(constraint, NumericalConstraint):
            return (
                f"Invalid value for {qualified_name}: {value!r} "
                f"not in range [{constraint.min}, {constraint.max}]{suffix}"
            )
        else:
            return f"Invalid value for {qualified_name}: {value!r}{suffix}"

    def validate_config(self, config: Configuration) -> tuple[bool, list[str]]:
        """Validate a configuration (θ) against this exploration space.

        In TVL terms, checks if configuration θ is in ExplorationSpace 𝒳.
        Handles conditional constraints by validating against the constraint
        appropriate for the parent's value.

        Args:
            config: Configuration (θ) mapping qualified names to values.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors: list[str] = []

        for qualified_name, value in config.items():
            tvar = self.tvars.get(qualified_name)
            if tvar is None:
                errors.append(f"Unknown parameter: {qualified_name}")
                continue

            # Accept None for optional TVARs
            if value is None and tvar.is_optional:
                continue

            # Handle conditional constraints
            if isinstance(tvar.constraint, ConditionalConstraint):
                parent_name = tvar.constraint.parent_qualified_name
                parent_value = config.get(parent_name)
                if not tvar.validate(value, parent_value):
                    effective_constraint = tvar.constraint.get_constraint_for(
                        parent_value
                    )
                    parent_context = f"when {parent_name}={parent_value!r}"
                    errors.append(
                        self._format_constraint_error(
                            qualified_name, value, effective_constraint, parent_context
                        )
                    )
            elif not tvar.validate(value):
                errors.append(
                    self._format_constraint_error(
                        qualified_name, value, tvar.constraint
                    )
                )

        return len(errors) == 0, errors

    def validate(self) -> bool:
        """Validate the exploration space for consistency.

        Checks that the ExplorationSpace is well-formed before optimization:
        - At least one TVAR is tunable (otherwise nothing to optimize)
        - Numerical constraints have valid ranges (min < max)
        - Numerical constraints have positive step (if specified)
        - Categorical constraints have non-empty choices
        - Conditional constraints reference valid parent TVARs
        - Conditional constraints have valid parent values
        - No circular conditional dependencies

        Returns:
            True if validation passes.

        Raises:
            ValueError: If no tunable TVARs exist (nothing to optimize).
            ConfigurationSpaceError: If any constraint is invalid.

        Example:
            >>> space = ExplorationSpace.from_pipeline_spec(spec)
            >>> space.set_range("generator.temperature", 2.0, 0.5)  # Mistake!
            >>> space.validate()  # Catches error before expensive optimization
            ConfigurationSpaceError: Invalid range for generator.temperature: min (2.0) >= max (0.5)
        """

        # Check at least one tunable TVAR
        if len(self.tunable_tvars) == 0:
            raise ValueError("No tunable parameters found in ExplorationSpace")

        # Validate each TVAR's constraint
        for qualified_name, tvar in self.tvars.items():
            self._validate_tvar_constraint(qualified_name, tvar)

        # Validate conditional dependencies
        self._validate_conditionals()

        return True

    def _validate_tvar_constraint(self, name: str, tvar: TVAR) -> None:
        """Validate a single TVAR's constraint.

        Args:
            name: The qualified name of the TVAR.
            tvar: The TVAR to validate.

        Raises:
            ConfigurationSpaceError: If the constraint is invalid.
        """
        import warnings

        # Warn if tunable TVAR has no constraint (treated as fixed during sampling)
        if tvar.constraint is None:
            if tvar.is_tunable:
                warnings.warn(
                    f"TVAR '{name}' is marked tunable but has no constraint. "
                    f"It will use default_value={tvar.default_value!r} during sampling. "
                    f"Use set_choices() or set_range() to define its search space.",
                    UserWarning,
                    stacklevel=4,  # Point to the validate() call site
                )
            return

        if isinstance(tvar.constraint, NumericalConstraint):
            self._validate_numerical_constraint(name, tvar.constraint)
        elif isinstance(tvar.constraint, CategoricalConstraint):
            self._validate_categorical_constraint(name, tvar.constraint)
        elif isinstance(tvar.constraint, ConditionalConstraint):
            self._validate_conditional_inner_constraints(name, tvar.constraint)

    def _validate_numerical_constraint(
        self, name: str, constraint: NumericalConstraint, context: str = ""
    ) -> None:
        """Validate a numerical constraint.

        Args:
            name: The TVAR name.
            constraint: The numerical constraint to validate.
            context: Optional context string for error messages.

        Raises:
            ConfigurationSpaceError: If the constraint is invalid.
        """
        from traigent.utils.exceptions import ConfigurationSpaceError

        ctx = f" ({context})" if context else ""
        if constraint.min >= constraint.max:
            raise ConfigurationSpaceError(
                f"Invalid range for {name}{ctx}: min ({constraint.min}) "
                f">= max ({constraint.max})"
            )
        if constraint.step is not None and constraint.step <= 0:
            raise ConfigurationSpaceError(
                f"Invalid step for {name}{ctx}: step must be positive, "
                f"got {constraint.step}"
            )

    def _validate_categorical_constraint(
        self, name: str, constraint: CategoricalConstraint, context: str = ""
    ) -> None:
        """Validate a categorical constraint.

        Args:
            name: The TVAR name.
            constraint: The categorical constraint to validate.
            context: Optional context string for error messages.

        Raises:
            ConfigurationSpaceError: If the constraint is invalid.
        """
        from traigent.utils.exceptions import ConfigurationSpaceError

        ctx = f" ({context})" if context else ""
        if len(constraint.choices) == 0:
            raise ConfigurationSpaceError(f"Empty choices for {name}{ctx}")

    def _validate_conditional_inner_constraints(
        self, name: str, constraint: ConditionalConstraint
    ) -> None:
        """Validate the inner constraints within a conditional constraint.

        Args:
            name: The TVAR name.
            constraint: The conditional constraint to validate.

        Raises:
            ConfigurationSpaceError: If any inner constraint is invalid.
        """
        from traigent.utils.exceptions import ConfigurationSpaceError

        # Check for empty conditions with no default (effectively unconstrained)
        if len(constraint.conditions) == 0 and constraint.default_constraint is None:
            raise ConfigurationSpaceError(
                f"Conditional constraint for {name} has no conditions and no default. "
                f"This leaves the parameter unconstrained for all parent values."
            )

        # Validate each condition's constraint
        for parent_value, inner in constraint.conditions.items():
            context = f"when parent={parent_value!r}"
            if isinstance(inner, NumericalConstraint):
                self._validate_numerical_constraint(name, inner, context)
            elif isinstance(inner, CategoricalConstraint):
                self._validate_categorical_constraint(name, inner, context)
            else:
                # Unknown constraint type - should not happen with proper API usage
                raise ConfigurationSpaceError(
                    f"Invalid inner constraint type for {name} ({context}): "
                    f"expected NumericalConstraint or CategoricalConstraint, "
                    f"got {type(inner).__name__}"
                )

        # Validate default constraint if present
        if constraint.default_constraint is not None:
            default = constraint.default_constraint
            if isinstance(default, NumericalConstraint):
                self._validate_numerical_constraint(name, default, "default constraint")
            elif isinstance(default, CategoricalConstraint):
                self._validate_categorical_constraint(
                    name, default, "default constraint"
                )
            else:
                # Unknown default constraint type
                raise ConfigurationSpaceError(
                    f"Invalid default constraint type for {name}: "
                    f"expected NumericalConstraint or CategoricalConstraint, "
                    f"got {type(default).__name__}"
                )

    def _validate_conditionals(self) -> None:
        """Validate conditional constraint dependencies.

        Checks:
        - Parent TVAR exists
        - Parent has discrete choices
        - All condition values are valid parent choices
        - No circular dependencies

        Raises:
            ConfigurationSpaceError: If any conditional constraint is invalid.
        """
        # Build dependency graph: child -> parent
        deps: dict[str, str] = {}

        for name, tvar in self.tvars.items():
            if isinstance(tvar.constraint, ConditionalConstraint):
                parent_name = tvar.constraint.parent_qualified_name
                self._validate_conditional_parent(name, parent_name, tvar.constraint)
                deps[name] = parent_name

        # Check for cycles
        self._check_circular_dependencies_validate(deps)

    def _validate_conditional_parent(
        self, child_name: str, parent_name: str, constraint: ConditionalConstraint
    ) -> None:
        """Validate that a conditional's parent is valid.

        Args:
            child_name: The child TVAR name.
            parent_name: The parent TVAR name.
            constraint: The conditional constraint.

        Raises:
            ConfigurationSpaceError: If the parent is invalid.
        """
        from traigent.utils.exceptions import ConfigurationSpaceError

        # Check parent exists
        if parent_name not in self.tvars:
            raise ConfigurationSpaceError(
                f"Conditional {child_name} references non-existent parent {parent_name}"
            )

        # Check parent has discrete choices
        parent = self.tvars[parent_name]
        if not self._has_discrete_choices(parent):
            raise ConfigurationSpaceError(
                f"Conditional {child_name} requires parent with discrete choices, "
                f"but {parent_name} has {type(parent.constraint).__name__}"
            )

        # Check all condition values are valid parent choices
        parent_choices = self._get_all_parent_choices(parent)
        for parent_value in constraint.conditions.keys():
            if parent_value not in parent_choices:
                raise ConfigurationSpaceError(
                    f"Conditional {child_name} references invalid parent value "
                    f"{parent_value!r} (not in {parent_name} choices: {parent_choices})"
                )

    def _get_all_parent_choices(self, parent: TVAR) -> list[Any]:
        """Get all valid choices for a parent TVAR.

        Args:
            parent: The parent TVAR.

        Returns:
            List of valid choices.
        """
        if isinstance(parent.constraint, CategoricalConstraint):
            return parent.constraint.choices
        elif isinstance(parent.constraint, ConditionalConstraint):
            # For conditional parents, collect all choices from all conditions
            all_choices: list[Any] = []
            for constraint in parent.constraint.conditions.values():
                if isinstance(constraint, CategoricalConstraint):
                    all_choices.extend(constraint.choices)
            if parent.constraint.default_constraint is not None:
                if isinstance(
                    parent.constraint.default_constraint, CategoricalConstraint
                ):
                    all_choices.extend(parent.constraint.default_constraint.choices)
            return all_choices
        return []

    def _check_circular_dependencies_validate(self, deps: dict[str, str]) -> None:
        """Detect circular dependencies in conditional graph.

        Args:
            deps: Dictionary mapping child TVAR names to parent TVAR names.

        Raises:
            ConfigurationSpaceError: If a circular dependency is detected.
        """
        from traigent.utils.exceptions import ConfigurationSpaceError

        for start in deps:
            visited: set[str] = set()
            current = start
            while current in deps:
                if current in visited:
                    raise ConfigurationSpaceError(
                        f"Circular dependency detected involving {current}"
                    )
                visited.add(current)
                current = deps[current]

    def __len__(self) -> int:
        """Return the number of TVARs."""
        return len(self.tvars)

    def __iter__(self):
        """Iterate over TVAR objects."""
        return iter(self.tvars.values())

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        tunable_count = len(self.tunable_tvars)
        fixed_count = len(self.fixed_tvars)
        return f"ExplorationSpace(tunable={tunable_count}, fixed={fixed_count})"


# Module-level helper functions


def _infer_type_from_choices(choices: list[Any]) -> str:
    """Infer Python type from a list of choices.

    Args:
        choices: List of choice values.

    Returns:
        String representing the Python type (e.g., "str", "int", "bool").
    """
    for choice in choices:
        if choice is not None:
            if isinstance(choice, bool):
                return "bool"
            elif isinstance(choice, int):
                return "int"
            elif isinstance(choice, float):
                return "float"
            elif isinstance(choice, str):
                return "str"
            else:
                return type(choice).__name__
    return "Any"
