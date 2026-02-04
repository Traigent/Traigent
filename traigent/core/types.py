"""Core type definitions for Traigent optimization system.

This module provides essential data structures for optimization configuration,
parameter definition, and trial management.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from traigent.api.types import OptimizationStatus, Trial, TrialResult, TrialStatus
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Re-export Trial from api.types for backward compatibility

__all__ = [
    "Trial",
    "TrialResult",
    "TrialStatus",
    "OptimizationStatus",
    "Parameter",
    "ParameterType",
    "ConfigurationSpace",
]


class ParameterType(StrEnum):
    """Types of parameters supported in optimization configuration spaces."""

    FLOAT = "float"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class Parameter:
    """Definition of a single optimization parameter.

    Represents a parameter that can be optimized, including its type,
    bounds, and optional default value.
    """

    name: str
    type: ParameterType
    bounds: tuple[float, float] | tuple[int, int] | list[Any]
    default: Any | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate parameter configuration after initialization."""
        if self.type == ParameterType.FLOAT:
            if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
                raise ValueError(
                    f"Float parameter {self.name} must have tuple bounds (min, max)"
                )
            if not all(isinstance(x, (int, float)) for x in self.bounds):
                raise ValueError(f"Float parameter {self.name} bounds must be numeric")

        elif self.type == ParameterType.INTEGER:
            if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
                raise ValueError(
                    f"Integer parameter {self.name} must have tuple bounds (min, max)"
                )
            if not all(isinstance(x, int) for x in self.bounds):
                raise ValueError(
                    f"Integer parameter {self.name} bounds must be integers"
                )

        elif self.type == ParameterType.CATEGORICAL:
            if not isinstance(self.bounds, (list, tuple)) or len(self.bounds) == 0:
                raise ValueError(
                    f"Categorical parameter {self.name} must have non-empty list of choices"
                )

        elif self.type == ParameterType.BOOLEAN:
            # Boolean parameters don't need bounds - always [True, False]
            if self.bounds is None:
                self.bounds: list[Any] = [True, False]

    def validate_value(self, value: Any) -> bool:
        """Check if a value is valid for this parameter.

        Args:
            value: Value to validate

        Returns:
            True if value is valid for this parameter
        """
        if self.type == ParameterType.FLOAT:
            return (
                isinstance(value, (int, float))
                and self.bounds[0] <= value <= self.bounds[1]
            )

        elif self.type == ParameterType.INTEGER:
            return isinstance(value, int) and self.bounds[0] <= value <= self.bounds[1]

        elif self.type == ParameterType.CATEGORICAL:
            return value in self.bounds

        elif self.type == ParameterType.BOOLEAN:
            return isinstance(value, bool)

        return False


@dataclass
class ConfigurationSpace:
    """Defines the complete parameter space for optimization.

    Contains all parameters that can be optimized along with any
    constraints between them. Provides dict-like interface for
    backward compatibility with existing optimizers.
    """

    parameters: list[Parameter] = field(default_factory=list)
    constraints: list[Callable[..., bool]] | None = None
    name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration space after initialization."""
        # Check for duplicate parameter names
        param_names = [p.name for p in self.parameters]
        if len(param_names) != len(set(param_names)):
            duplicates = [name for name in param_names if param_names.count(name) > 1]
            raise ValueError(f"Duplicate parameter names found: {duplicates}")

    # Dict-like interface for backward compatibility
    def items(self):
        """Iterate over parameter name-definition pairs like a dictionary."""
        for param in self.parameters:
            if param.type in (ParameterType.FLOAT, ParameterType.INTEGER):
                yield param.name, param.bounds
            elif param.type == ParameterType.CATEGORICAL:
                yield param.name, param.bounds
            elif param.type == ParameterType.BOOLEAN:
                yield param.name, [True, False]

    def keys(self):
        """Get parameter names like a dictionary."""
        return [param.name for param in self.parameters]

    def values(self):
        """Get parameter definitions like a dictionary."""
        definitions = []
        for param in self.parameters:
            if param.type in (ParameterType.FLOAT, ParameterType.INTEGER):
                definitions.append(param.bounds)
            elif param.type == ParameterType.CATEGORICAL:
                definitions.append(param.bounds)
            elif param.type == ParameterType.BOOLEAN:
                definitions.append([True, False])
        return definitions

    def __getitem__(self, key: str):
        """Get parameter definition by name like a dictionary."""
        param = self.get_parameter(key)
        if param is None:
            raise KeyError(f"Parameter '{key}' not found")

        if param.type in (ParameterType.FLOAT, ParameterType.INTEGER):
            return param.bounds
        elif param.type == ParameterType.CATEGORICAL:
            return param.bounds
        elif param.type == ParameterType.BOOLEAN:
            return [True, False]

    def __contains__(self, key: str) -> bool:
        """Check if parameter exists like a dictionary."""
        return any(param.name == key for param in self.parameters)

    def __len__(self) -> int:
        """Get number of parameters like a dictionary."""
        return len(self.parameters)

    def add_parameter(self, parameter: Parameter) -> None:
        """Add a parameter to the configuration space.

        Args:
            parameter: Parameter to add

        Raises:
            ValueError: If parameter name already exists
        """
        if parameter.name in [p.name for p in self.parameters]:
            raise ValueError(f"Parameter '{parameter.name}' already exists")
        self.parameters.append(parameter)

    def get_parameter(self, name: str) -> Parameter | None:
        """Get a parameter by name.

        Args:
            name: Parameter name to find

        Returns:
            Parameter if found, None otherwise
        """
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate a configuration against this space.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid
        """
        # Check all required parameters are present
        param_names = {p.name for p in self.parameters}
        config_keys = set(config.keys())

        missing = param_names - config_keys
        if missing:
            return False

        # Check each parameter value is valid
        for param in self.parameters:
            if param.name in config:
                if not param.validate_value(config[param.name]):
                    return False

        # Check constraints if any
        if self.constraints:
            for constraint in self.constraints:
                try:
                    if not constraint(config):
                        return False
                except Exception as e:
                    logger.debug(
                        f"Constraint check raised exception (treating as invalid): {e}"
                    )
                    return False

        return True

    def sample_config(self) -> dict[str, Any]:
        """Generate a random valid configuration.

        Returns:
            Dictionary with random values for all parameters
        """
        import random

        config = {}
        for param in self.parameters:
            if param.type == ParameterType.FLOAT:
                config[param.name] = random.uniform(*param.bounds)
            elif param.type == ParameterType.INTEGER:
                config[param.name] = random.randint(*param.bounds)
            elif param.type == ParameterType.CATEGORICAL:
                config[param.name] = random.choice(param.bounds)
            elif param.type == ParameterType.BOOLEAN:
                config[param.name] = random.choice([True, False])

        return config

    @classmethod
    def from_dict(cls, space_dict: dict[str, Any]) -> ConfigurationSpace:
        """Create configuration space from dictionary.

        Args:
            space_dict: Dictionary defining parameter space

        Returns:
            ConfigurationSpace instance
        """
        parameters = []

        for name, definition in space_dict.items():
            if isinstance(definition, tuple) and len(definition) == 2:
                # Assume numeric bounds (float, int)
                if all(isinstance(x, float) for x in definition):
                    param = Parameter(name, ParameterType.FLOAT, definition)
                elif all(isinstance(x, int) for x in definition):
                    param = Parameter(name, ParameterType.INTEGER, definition)
                else:
                    # Mixed types - default to float
                    param = Parameter(name, ParameterType.FLOAT, definition)
            elif isinstance(definition, list):
                # Categorical parameter
                param = Parameter(name, ParameterType.CATEGORICAL, definition)
            elif definition is bool or definition == "bool":
                # Boolean parameter
                param = Parameter(name, ParameterType.BOOLEAN, [True, False])
            else:
                raise ValueError(
                    f"Unsupported parameter definition for '{name}': {definition}"
                )

            parameters.append(param)

        return cls(parameters=parameters)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration space to dictionary representation.

        Returns:
            Dictionary representation of the configuration space
        """
        space_dict: dict[str, Any] = {}

        for param in self.parameters:
            if param.type in (ParameterType.FLOAT, ParameterType.INTEGER):
                space_dict[param.name] = param.bounds
            elif param.type == ParameterType.CATEGORICAL:
                space_dict[param.name] = param.bounds
            elif param.type == ParameterType.BOOLEAN:
                space_dict[param.name] = "bool"

        return space_dict
