"""SE-friendly parameter range definitions for Traigent configuration spaces.

This module provides intuitive constructors for defining parameter search spaces
that "speak the language" of software engineers while maintaining full backward
compatibility with tuple/list syntax.

These classes are TVL (Tuned Variable Language) first-class citizens, supporting:
- Domain specification (range, choices, log-scale)
- Optional naming and units for TVL integration
- Constraint builder methods for expressing structural constraints

Example:
    >>> from traigent import Range, IntRange, Choices, LogRange, implies
    >>>
    >>> # Basic usage
    >>> @traigent.optimize(
    ...     temperature=Range(0.0, 2.0),
    ...     max_tokens=IntRange(100, 4096),
    ...     model=Choices(["gpt-4", "gpt-3.5-turbo"]),
    ...     learning_rate=LogRange(1e-5, 1e-1),
    ... )
    ... def my_func(...): ...
    >>>
    >>> # With TVL features (constraints, units)
    >>> temp = Range(0.0, 2.0, unit="ratio")
    >>> model = Choices(["gpt-4", "gpt-3.5"])
    >>> constraints = [
    ...     implies(model.equals("gpt-4"), temp.lte(0.7)),
    ... ]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from traigent.api.constraints import Condition

T = TypeVar("T")


class ParameterRange(ABC):
    """Base class for all parameter range types.

    All range classes inherit from this to enable isinstance() checks
    and provide a common interface for normalization.
    """

    @abstractmethod
    def to_config_value(self) -> tuple[Any, ...] | list[Any] | dict[str, Any]:
        """Convert to the internal configuration space format.

        Returns:
            - tuple (low, high) for simple numeric ranges
            - list of values for categorical choices
            - dict with type info for ranges with log/step options
        """
        ...

    @abstractmethod
    def get_default(self) -> Any | None:
        """Return the default value if set, otherwise None."""
        ...


@dataclass(frozen=True, slots=True)
class Range(ParameterRange):
    """Continuous float range for optimization.

    Args:
        low: Lower bound (inclusive)
        high: Upper bound (inclusive)
        step: Optional step size for discretization
        log: Whether to use log-scale sampling (default: False).
             Note: Cannot be combined with step (Optuna limitation).
        default: Optional default value (populates default_config)
        name: Optional TVAR name (auto-assigned from decorator kwarg if not set)
        unit: Optional unit of measurement (e.g., "ratio", "seconds", "USD")

    Example:
        >>> temperature = Range(0.0, 2.0)
        >>> temperature_fine = Range(0.0, 1.0, step=0.1)
        >>> learning_rate = Range(1e-5, 1e-1, log=True)
        >>> # With TVL features
        >>> temp = Range(0.0, 2.0, unit="ratio")

    Raises:
        ValueError: If low >= high, step <= 0, log with non-positive low,
                    or log combined with step
    """

    low: float
    high: float
    step: float | None = None
    log: bool = False
    default: float | None = None
    # TVL fields
    name: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(
                f"Range low ({self.low}) must be less than high ({self.high})"
            )
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Range step must be positive, got {self.step}")
        if self.log and self.low <= 0:
            raise ValueError(f"log=True requires positive bounds, got low={self.low}")
        if self.log and self.step is not None:
            raise ValueError(
                "Cannot use log=True with step (Optuna limitation). "
                "Use either log-scale OR discrete steps, not both."
            )
        if self.default is not None and not (self.low <= self.default <= self.high):
            raise ValueError(
                f"default {self.default} is outside range [{self.low}, {self.high}]"
            )

    def to_config_value(self) -> tuple[float, float] | dict[str, Any]:
        """Convert to internal format.

        Returns tuple for simple ranges, dict when log/step is set.
        """
        if self.step is None and not self.log:
            return (self.low, self.high)
        # Use dict format for advanced options
        result: dict[str, Any] = {
            "type": "float",
            "low": self.low,
            "high": self.high,
        }
        if self.step is not None:
            result["step"] = self.step
        if self.log:
            result["log"] = True
        return result

    def get_default(self) -> float | None:
        """Return the default value if set."""
        return self.default

    def to_tuple(self) -> tuple[float, float]:
        """Return as simple (low, high) tuple for backward compatibility."""
        return (self.low, self.high)

    # =========================================================================
    # Constraint Builder Methods
    # =========================================================================

    def equals(self, value: float) -> Condition:
        """Create condition: this parameter equals value.

        Example:
            >>> temp = Range(0.0, 2.0)
            >>> cond = temp.equals(0.5)  # temp == 0.5
        """
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="==", value=value)

    def not_equals(self, value: float) -> Condition:
        """Create condition: this parameter does not equal value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="!=", value=value)

    def gt(self, value: float) -> Condition:
        """Create condition: this parameter > value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator=">", value=value)

    def gte(self, value: float) -> Condition:
        """Create condition: this parameter >= value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator=">=", value=value)

    def lt(self, value: float) -> Condition:
        """Create condition: this parameter < value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="<", value=value)

    def lte(self, value: float) -> Condition:
        """Create condition: this parameter <= value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="<=", value=value)

    def in_range(self, low: float, high: float) -> Condition:
        """Create condition: low <= this parameter <= high."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="in_range", value=(low, high))


@dataclass(frozen=True, slots=True)
class IntRange(ParameterRange):
    """Integer range for optimization.

    Args:
        low: Lower bound (inclusive)
        high: Upper bound (inclusive)
        step: Optional step size (default: 1)
        log: Whether to use log-scale sampling (default: False).
             Note: Cannot be combined with step (Optuna limitation).
        default: Optional default value (populates default_config)
        name: Optional TVAR name (auto-assigned from decorator kwarg if not set)
        unit: Optional unit of measurement (e.g., "count", "tokens")

    Example:
        >>> max_tokens = IntRange(100, 4096)
        >>> batch_size = IntRange(16, 256, step=16)
        >>> # With TVL features
        >>> tokens = IntRange(100, 4096, unit="tokens")

    Raises:
        TypeError: If low/high are not integers
        ValueError: If low >= high, step <= 0, or invalid log/step combination
    """

    low: int
    high: int
    step: int | None = None
    log: bool = False
    default: int | None = None
    # TVL fields
    name: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.low, int) or not isinstance(self.high, int):
            raise TypeError(
                f"IntRange bounds must be integers, got low={type(self.low).__name__}, "
                f"high={type(self.high).__name__}"
            )
        if self.low >= self.high:
            raise ValueError(
                f"IntRange low ({self.low}) must be less than high ({self.high})"
            )
        if self.step is not None and self.step <= 0:
            raise ValueError(f"IntRange step must be positive, got {self.step}")
        if self.log and self.low <= 0:
            raise ValueError(f"log=True requires positive bounds, got low={self.low}")
        if self.log and self.step is not None:
            raise ValueError(
                "Cannot use log=True with step (Optuna limitation). "
                "Use either log-scale OR discrete steps, not both."
            )
        if self.default is not None and not (self.low <= self.default <= self.high):
            raise ValueError(
                f"default {self.default} is outside range [{self.low}, {self.high}]"
            )

    def to_config_value(self) -> tuple[int, int] | dict[str, Any]:
        """Convert to internal format.

        Returns tuple for simple ranges, dict when log/step is set.
        """
        if self.step is None and not self.log:
            return (self.low, self.high)
        result: dict[str, Any] = {
            "type": "int",
            "low": self.low,
            "high": self.high,
        }
        if self.step is not None:
            result["step"] = self.step
        if self.log:
            result["log"] = True
        return result

    def get_default(self) -> int | None:
        """Return the default value if set."""
        return self.default

    def to_tuple(self) -> tuple[int, int]:
        """Return as simple (low, high) tuple."""
        return (self.low, self.high)

    # =========================================================================
    # Constraint Builder Methods
    # =========================================================================

    def equals(self, value: int) -> Condition:
        """Create condition: this parameter equals value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="==", value=value)

    def not_equals(self, value: int) -> Condition:
        """Create condition: this parameter does not equal value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="!=", value=value)

    def gt(self, value: int) -> Condition:
        """Create condition: this parameter > value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator=">", value=value)

    def gte(self, value: int) -> Condition:
        """Create condition: this parameter >= value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator=">=", value=value)

    def lt(self, value: int) -> Condition:
        """Create condition: this parameter < value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="<", value=value)

    def lte(self, value: int) -> Condition:
        """Create condition: this parameter <= value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="<=", value=value)

    def in_range(self, low: int, high: int) -> Condition:
        """Create condition: low <= this parameter <= high."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="in_range", value=(low, high))


@dataclass(frozen=True, slots=True)
class LogRange(ParameterRange):
    """Log-scale float range for optimization.

    Convenience class for Range(low, high, log=True). Useful for parameters
    that vary over orders of magnitude like learning rates and regularization.

    Args:
        low: Lower bound (must be positive)
        high: Upper bound (must be positive)
        default: Optional default value (populates default_config)
        name: Optional TVAR name (auto-assigned from decorator kwarg if not set)
        unit: Optional unit of measurement

    Example:
        >>> learning_rate = LogRange(1e-5, 1e-1)
        >>> regularization = LogRange(0.001, 10.0)

    Raises:
        ValueError: If bounds are not positive or low >= high
    """

    low: float
    high: float
    default: float | None = None
    # TVL fields
    name: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.low <= 0 or self.high <= 0:
            raise ValueError(
                f"LogRange requires positive bounds, got ({self.low}, {self.high})"
            )
        if self.low >= self.high:
            raise ValueError(
                f"LogRange low ({self.low}) must be less than high ({self.high})"
            )
        if self.default is not None and not (self.low <= self.default <= self.high):
            raise ValueError(
                f"default {self.default} is outside range [{self.low}, {self.high}]"
            )

    def to_config_value(self) -> dict[str, Any]:
        """Convert to internal format with log=True."""
        return {
            "type": "float",
            "low": self.low,
            "high": self.high,
            "log": True,
        }

    def get_default(self) -> float | None:
        """Return the default value if set."""
        return self.default

    def to_tuple(self) -> tuple[float, float]:
        """Return as simple (low, high) tuple (loses log information)."""
        return (self.low, self.high)

    # =========================================================================
    # Constraint Builder Methods (same as Range)
    # =========================================================================

    def equals(self, value: float) -> Condition:
        """Create condition: this parameter equals value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="==", value=value)

    def not_equals(self, value: float) -> Condition:
        """Create condition: this parameter does not equal value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="!=", value=value)

    def gt(self, value: float) -> Condition:
        """Create condition: this parameter > value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator=">", value=value)

    def gte(self, value: float) -> Condition:
        """Create condition: this parameter >= value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator=">=", value=value)

    def lt(self, value: float) -> Condition:
        """Create condition: this parameter < value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="<", value=value)

    def lte(self, value: float) -> Condition:
        """Create condition: this parameter <= value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="<=", value=value)

    def in_range(self, low: float, high: float) -> Condition:
        """Create condition: low <= this parameter <= high."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="in_range", value=(low, high))


@dataclass(frozen=True, slots=True)
class Choices(ParameterRange, Generic[T]):
    """Categorical choices for optimization.

    Args:
        values: Sequence of allowed values (list or tuple, NOT str/bytes)
        default: Optional default value (must be in values, populates default_config)
        name: Optional TVAR name (auto-assigned from decorator kwarg if not set)
        unit: Optional unit of measurement (rarely needed for categorical)

    Example:
        >>> model = Choices(["gpt-4", "gpt-3.5-turbo", "claude-2"])
        >>> use_cache = Choices([True, False], default=True)
        >>> temperature = Choices([0.0, 0.3, 0.7, 1.0])

    Raises:
        TypeError: If values is a string or bytes
        ValueError: If values is empty or default is not in values
    """

    values: Sequence[T]
    default: T | None = None
    # TVL fields
    name: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        # Reject str/bytes which are technically sequences but not valid choices
        if isinstance(self.values, (str, bytes)):
            raise TypeError(
                "Choices values must be a list or tuple, not str/bytes. "
                f"Did you mean Choices([{self.values!r}])?"
            )
        # Convert to tuple for immutability (frozen dataclass)
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise ValueError("Choices must have at least one value")
        if self.default is not None and self.default not in self.values:
            raise ValueError(
                f"default {self.default!r} is not in choices {list(self.values)}"
            )

    def to_config_value(self) -> list[T]:
        """Convert to internal list format."""
        return list(self.values)

    def get_default(self) -> T | None:
        """Return the default value if set."""
        return self.default

    def to_list(self) -> list[T]:
        """Return as list for backward compatibility."""
        return list(self.values)

    def __iter__(self):
        """Allow iteration over choices."""
        return iter(self.values)

    def __len__(self) -> int:
        """Return number of choices."""
        return len(self.values)

    def __contains__(self, item: T) -> bool:
        """Check if item is in choices."""
        return item in self.values

    # =========================================================================
    # Constraint Builder Methods
    # =========================================================================

    def equals(self, value: T) -> Condition:
        """Create condition: this parameter equals value.

        Example:
            >>> model = Choices(["gpt-4", "gpt-3.5"])
            >>> cond = model.equals("gpt-4")  # model == "gpt-4"
        """
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="==", value=value)

    def not_equals(self, value: T) -> Condition:
        """Create condition: this parameter does not equal value."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="!=", value=value)

    def is_in(self, values: Sequence[T]) -> Condition:
        """Create condition: this parameter is in the given values.

        Example:
            >>> model = Choices(["gpt-4", "gpt-3.5", "claude"])
            >>> cond = model.is_in(["gpt-4", "gpt-3.5"])  # model in ["gpt-4", "gpt-3.5"]
        """
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="in", value=tuple(values))

    def not_in(self, values: Sequence[T]) -> Condition:
        """Create condition: this parameter is not in the given values."""
        from traigent.api.constraints import Condition

        return Condition(tvar=self, operator="not_in", value=tuple(values))


# =============================================================================
# Utility Functions
# =============================================================================


def is_parameter_range(value: Any) -> bool:
    """Check if a value is a ParameterRange instance.

    Useful for validation code that needs to detect the new range types.
    """
    return isinstance(value, ParameterRange)


def is_inline_param_definition(value: Any) -> bool:
    """Check if a value looks like an inline parameter definition.

    Recognizes:
    - ParameterRange instances (Range, IntRange, LogRange, Choices)
    - Tuples with exactly 2 numeric elements (legacy range syntax)

    Note: Lists are NOT recognized as inline param definitions to catch typos.
    Users who want inline list syntax should use Choices() instead.

    Args:
        value: The value to check

    Returns:
        True if value appears to be a parameter definition
    """
    if is_parameter_range(value):
        return True
    if isinstance(value, tuple) and len(value) == 2:
        return all(isinstance(v, (int, float)) for v in value)
    # Lists are NOT treated as inline params to catch typos like `objectivs=[...]`
    # Users should use Choices([...]) for categorical parameters
    return False


def normalize_config_value(value: Any) -> tuple[Any, ...] | list[Any] | dict[str, Any]:
    """Convert a ParameterRange to its primitive format.

    If value is already a primitive (tuple/list/dict), returns it unchanged.

    Args:
        value: A ParameterRange instance or primitive value

    Returns:
        The primitive configuration format
    """
    if isinstance(value, ParameterRange):
        return value.to_config_value()
    return value


def normalize_parameter_value(
    value: Any,
) -> tuple[Any, ...] | list[Any] | dict[str, Any]:
    """Normalize a ParameterRange or primitive configuration value.

    Alias for normalize_config_value to match API naming expectations.
    """
    return normalize_config_value(value)


def _process_param_entry(
    key: str,
    value: Any,
    result: dict[str, Any],
    defaults: dict[str, Any],
) -> None:
    """Process a single parameter entry, updating result and defaults dicts."""
    if isinstance(value, ParameterRange):
        result[key] = value.to_config_value()
        default_val = value.get_default()
        if default_val is not None:
            defaults[key] = default_val
    else:
        result[key] = normalize_config_value(value)


def normalize_configuration_space(
    config_space: dict[str, Any] | None,
    inline_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize a configuration space and extract defaults.

    Merges inline decorator kwargs with an explicit configuration_space dict,
    normalizing all ParameterRange instances to their primitive formats.

    Precedence: inline_params override config_space entries.

    Args:
        config_space: Explicit configuration_space dict (may contain Range/Choices)
        inline_params: Inline kwargs from decorator that are param definitions

    Returns:
        Tuple of (normalized_config_space, defaults_dict)
        - normalized_config_space: All values converted to primitives
        - defaults_dict: Extracted default values from Range/Choices
    """
    result: dict[str, Any] = {}
    defaults: dict[str, Any] = {}

    # Start with explicit config_space (lower precedence)
    if config_space:
        if not isinstance(config_space, dict):
            from traigent.utils.exceptions import ValidationError

            raise ValidationError(
                f"Expected dictionary for configuration_space, got {type(config_space).__name__}"
            )
        for key, value in config_space.items():
            _process_param_entry(key, value, result, defaults)

    # Add/override with inline parameters (higher precedence)
    if inline_params:
        for key, value in inline_params.items():
            _process_param_entry(key, value, result, defaults)

    return result, defaults


__all__ = [
    "ParameterRange",
    "Range",
    "IntRange",
    "LogRange",
    "Choices",
    "is_parameter_range",
    "is_inline_param_definition",
    "normalize_config_value",
    "normalize_parameter_value",
    "normalize_configuration_space",
]
