"""Consolidated validation utilities for Traigent SDK.

This module combines the best features from the three existing validation systems:
- validation.py: Simple validation functions
- enhanced_validation.py: User-friendly error messages
- common_validators.py: Structured validation with error codes

The goal is to provide a single, comprehensive validation system that is both
developer-friendly and user-friendly.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Maintainability CONC-Quality-Reliability FUNC-INVOKERS REQ-INJ-002 SYNC-OptimizationFlow

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import lru_cache
from pathlib import Path
from typing import Any

from traigent.utils.exceptions import ConfigurationError
from traigent.utils.exceptions import ValidationError as ValidationException
from traigent.utils.logging import get_logger
from traigent.utils.secure_path import safe_open

logger = get_logger(__name__)


@dataclass
class ValidationError:
    """Structured validation error with helpful context."""

    field: str
    message: str
    error_code: str = "VALIDATION_ERROR"
    severity: str = "error"  # error, warning, info
    suggestions: list[str] = dataclass_field(default_factory=list)
    context: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_exception(self) -> ValidationException:
        """Convert to exception for backward compatibility."""
        full_message = f"{self.field}: {self.message}"
        if self.suggestions:
            full_message += "\nSuggestions:\n" + "\n".join(
                f"  - {s}" for s in self.suggestions
            )
        return ValidationException(full_message)


@dataclass
class ValidationResult:
    """Result of validation containing errors, warnings, and suggestions."""

    errors: list[ValidationError] = dataclass_field(default_factory=list)
    warnings: list[ValidationError] = dataclass_field(default_factory=list)
    suggestions: list[str] = dataclass_field(default_factory=list)
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def add_error(self, field: str, message: str, **kwargs) -> None:
        """Add an error to the result."""
        self.errors.append(
            ValidationError(field=field, message=message, severity="error", **kwargs)
        )

    def add_warning(self, field: str, message: str, **kwargs) -> None:
        """Add a warning to the result."""
        self.warnings.append(
            ValidationError(field=field, message=message, severity="warning", **kwargs)
        )

    def raise_if_invalid(self) -> None:
        """Raise exception if validation failed."""
        if not self.is_valid:
            # Combine all error messages
            error_messages = [f"{e.field}: {e.message}" for e in self.errors]
            full_message = "Validation failed:\n" + "\n".join(error_messages)

            # Add suggestions if any
            all_suggestions = []
            for error in self.errors:
                all_suggestions.extend(error.suggestions)
            if all_suggestions:
                full_message += "\n\nSuggestions:\n" + "\n".join(
                    f"  - {s}" for s in all_suggestions
                )

            raise ValidationException(full_message)

    def get_feedback(self, include_warnings: bool = True) -> str:
        """Get user-friendly feedback message."""
        lines = []

        # Add errors
        if self.errors:
            lines.append("❌ Validation Errors:")
            for error in self.errors:
                lines.append(f"  • {error.field}: {error.message}")
                if error.suggestions:
                    for suggestion in error.suggestions:
                        lines.append(f"    💡 {suggestion}")

        # Add warnings
        if include_warnings and self.warnings:
            if lines:
                lines.append("")
            lines.append("⚠️  Warnings:")
            for warning in self.warnings:
                lines.append(f"  • {warning.field}: {warning.message}")
                if warning.suggestions:
                    for suggestion in warning.suggestions:
                        lines.append(f"    💡 {suggestion}")

        # Add general suggestions
        if self.suggestions:
            if lines:
                lines.append("")
            lines.append("💡 Suggestions:")
            for suggestion in self.suggestions:
                lines.append(f"  • {suggestion}")

        # Add success message if valid
        if self.is_valid and not self.warnings:
            lines.append("✅ Validation passed!")

        return "\n".join(lines)


# ===== Canonical Knob Ranges =====

# Preset range type for a knob whose values are whole numbers. The finest gap
# such a knob can express is 1, whatever its range.
_INTEGER_PRESET_RANGE_TYPE = "IntRange"

# Preset range types that describe a numeric interval. Categorical presets
# (Choices) have no span to compare a declared sweep against.
_NUMERIC_PRESET_RANGE_TYPES: frozenset[str] = frozenset(
    {"Range", _INTEGER_PRESET_RANGE_TYPE}
)


@dataclass(frozen=True)
class _KnobRange:
    """The span a knob is normally tuned over, and whether it is whole-numbered."""

    low: float
    high: float
    is_integer: bool

    @property
    def bounds_whole_domain(self) -> bool:
        """Whether the range is everything the knob can be, not a suggested sweep.

        A continuous preset bounds a knob that has nowhere else to go:
        ``temperature`` is 0-1, ``top_p`` is a probability. An integer preset is
        a suggested sweep instead - ``max_tokens`` is 256-4096 for long-form
        generation and 80-120 for one-line answers, both entirely legitimate -
        so measuring a declared sweep against it would report the preset rather
        than the configuration space.
        """
        return not self.is_integer


@lru_cache(maxsize=1)
def _canonical_knob_ranges() -> dict[str, _KnobRange]:
    """Canonical range per knob, read from the config-generator presets.

    Reused rather than re-derived so the degenerate-space diagnostics cannot
    drift from the ranges the SDK already publishes. Imported lazily because
    the presets package is only needed when a configuration space is diagnosed.
    """
    from traigent.config_generator.presets import range_presets

    ranges: dict[str, _KnobRange] = {}
    for canonical_name in range_presets.all_canonical_names():
        preset = range_presets.get_preset_range(canonical_name)
        if preset is None:
            continue
        range_type = preset.get("range_type")
        if range_type not in _NUMERIC_PRESET_RANGE_TYPES:
            continue
        kwargs = preset.get("kwargs", {})
        low = kwargs.get("low")
        high = kwargs.get("high")
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            continue
        if high <= low:
            continue
        ranges[canonical_name] = _KnobRange(
            low=float(low),
            high=float(high),
            is_integer=range_type == _INTEGER_PRESET_RANGE_TYPE,
        )
    return ranges


class Validators:
    """Core validation functions with consistent interface."""

    # ===== Type Validators =====

    @staticmethod
    def validate_type(
        value: Any, expected_type: type, field_name: str
    ) -> ValidationResult:
        """Validate that value is of expected type."""
        result = ValidationResult()

        if not isinstance(value, expected_type):
            result.add_error(
                field_name,
                f"Expected {expected_type.__name__}, got {type(value).__name__}",
                error_code="TYPE_ERROR",
                suggestions=[
                    f"Ensure {field_name} is of type {expected_type.__name__}"
                ],
            )

        return result

    @staticmethod
    def validate_string(
        value: Any,
        field_name: str,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
    ) -> ValidationResult:
        """Validate string value with optional constraints."""
        result = ValidationResult()

        # Type check
        if not isinstance(value, str):
            result.add_error(
                field_name,
                f"Expected string, got {type(value).__name__}",
                error_code="TYPE_ERROR",
            )
            return result

        # Length checks
        if min_length is not None and len(value) < min_length:
            result.add_error(
                field_name,
                f"String too short (minimum {min_length} characters)",
                error_code="LENGTH_ERROR",
                suggestions=[f"Provide at least {min_length} characters"],
            )

        if max_length is not None and len(value) > max_length:
            result.add_error(
                field_name,
                f"String too long (maximum {max_length} characters)",
                error_code="LENGTH_ERROR",
                suggestions=[f"Limit to {max_length} characters"],
            )

        # Pattern check
        if pattern and not re.match(pattern, value):
            result.add_error(
                field_name,
                "String does not match required pattern",
                error_code="PATTERN_ERROR",
                suggestions=[f"String should match pattern: {pattern}"],
            )

        return result

    @staticmethod
    def validate_string_non_empty(value: Any, field_name: str) -> ValidationResult:
        """Validate that a string is not empty."""
        return Validators.validate_string(value, field_name, min_length=1)

    @staticmethod
    def validate_choices(
        value: Any, field_name: str, choices: list[Any]
    ) -> ValidationResult:
        """Validate value is in allowed choices."""
        result = ValidationResult()

        if value not in choices:
            result.add_error(
                field_name,
                f"Invalid choice: {value}",
                error_code="INVALID_CHOICE",
                suggestions=[f"Choose from: {', '.join(map(str, choices))}"],
            )

        return result

    # ===== Numeric Validators =====

    @staticmethod
    def validate_number(
        value: Any,
        field_name: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> ValidationResult:
        """Validate numeric value with optional range."""
        result = ValidationResult()

        if not isinstance(value, (int, float)):
            result.add_error(
                field_name,
                f"Expected number, got {type(value).__name__}",
                error_code="TYPE_ERROR",
            )
            return result

        if min_value is not None and value < min_value:
            result.add_error(
                field_name,
                f"Value {value} is below minimum {min_value}",
                error_code="RANGE_ERROR",
                suggestions=[f"Use a value >= {min_value}"],
            )

        if max_value is not None and value > max_value:
            result.add_error(
                field_name,
                f"Value {value} exceeds maximum {max_value}",
                error_code="RANGE_ERROR",
                suggestions=[f"Use a value <= {max_value}"],
            )

        return result

    @staticmethod
    def validate_positive_int(value: Any, field_name: str) -> ValidationResult:
        """Validate positive integer."""
        result = ValidationResult()

        if not isinstance(value, int):
            result.add_error(
                field_name,
                f"Expected integer, got {type(value).__name__}",
                error_code="TYPE_ERROR",
            )
            return result

        if value <= 0:
            result.add_error(
                field_name,
                f"Value must be positive, got {value}",
                error_code="RANGE_ERROR",
                suggestions=["Use a positive integer value"],
            )

        return result

    @staticmethod
    def validate_probability(value: Any, field_name: str) -> ValidationResult:
        """Validate probability value (0.0 to 1.0)."""
        result = Validators.validate_number(value, field_name, 0.0, 1.0)
        if result.is_valid:
            result.metadata["is_probability"] = True
        return result

    # ===== Collection Validators =====

    @staticmethod
    def validate_list(
        value: Any,
        field_name: str,
        min_length: int | None = None,
        max_length: int | None = None,
        item_validator: Callable[..., Any] | None = None,
    ) -> ValidationResult:
        """Validate list with optional constraints."""
        result = ValidationResult()

        if not isinstance(value, list):
            result.add_error(
                field_name,
                f"Expected list, got {type(value).__name__}",
                error_code="TYPE_ERROR",
            )
            return result

        # Length checks
        if min_length is not None and len(value) < min_length:
            result.add_error(
                field_name,
                f"List too short (minimum {min_length} items)",
                error_code="LENGTH_ERROR",
            )

        if max_length is not None and len(value) > max_length:
            result.add_error(
                field_name,
                f"List too long (maximum {max_length} items)",
                error_code="LENGTH_ERROR",
            )

        # Validate individual items
        if item_validator:
            for i, item in enumerate(value):
                item_result = item_validator(item, f"{field_name}[{i}]")
                result.errors.extend(item_result.errors)
                result.warnings.extend(item_result.warnings)

        return result

    @staticmethod
    def validate_dict(
        value: Any,
        field_name: str,
        required_keys: set[str] | None = None,
        allowed_keys: set[str] | None = None,
    ) -> ValidationResult:
        """Validate dictionary with key constraints."""
        result = ValidationResult()

        if not isinstance(value, dict):
            result.add_error(
                field_name,
                f"Expected dictionary, got {type(value).__name__}",
                error_code="TYPE_ERROR",
            )
            return result

        # Check required keys
        if required_keys:
            missing = required_keys - set(value.keys())
            if missing:
                result.add_error(
                    field_name,
                    f"Missing required keys: {', '.join(missing)}",
                    error_code="MISSING_KEY",
                    suggestions=[f"Add the missing keys: {', '.join(missing)}"],
                )

        # Check allowed keys
        if allowed_keys:
            extra = set(value.keys()) - allowed_keys
            if extra:
                result.add_warning(
                    field_name,
                    f"Unknown keys will be ignored: {', '.join(extra)}",
                    error_code="UNKNOWN_KEY",
                )

        return result

    # ===== Path and File Validators =====

    @staticmethod
    def validate_path(
        value: Any,
        field_name: str,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        allowed_extensions: list[str] | None = None,
        allowed_base_dirs: list[str | Path] | None = None,
    ) -> ValidationResult:
        """Validate file system path with security checks."""
        result = ValidationResult()

        if not isinstance(value, (str, Path)):
            result.add_error(
                field_name,
                f"Expected path string, got {type(value).__name__}",
                error_code="TYPE_ERROR",
            )
            return result

        try:
            path = Path(value)
            resolved_path = path.resolve()

            allowed_bases = Validators._resolve_allowed_bases(allowed_base_dirs)
            if not any(
                Validators._is_relative_to(resolved_path, base)
                for base in allowed_bases
            ):
                result.add_error(
                    field_name,
                    f"Path {resolved_path} escapes the allowed directories",
                    error_code="SECURITY_ERROR",
                    suggestions=[
                        "Provide a path located within the project workspace "
                        f"({', '.join(str(b) for b in allowed_bases)})"
                    ],
                )
                return result

            # Security: Check for system directories
            path_str = str(resolved_path)
            if path_str.startswith("/etc") or path_str.startswith("/sys"):
                result.add_error(
                    field_name,
                    "Path targets restricted system directories",
                    error_code="SECURITY_ERROR",
                    suggestions=["Use a path within the project workspace"],
                )
                return result

            # Existence check
            if must_exist and not resolved_path.exists():
                result.add_error(
                    field_name,
                    f"Path does not exist: {value}",
                    error_code="NOT_FOUND",
                    suggestions=["Check the file path and ensure the file exists"],
                )

            # Type checks
            if must_be_file and resolved_path.exists() and not resolved_path.is_file():
                result.add_error(
                    field_name, f"Path is not a file: {value}", error_code="WRONG_TYPE"
                )

            if must_be_dir and resolved_path.exists() and not resolved_path.is_dir():
                result.add_error(
                    field_name,
                    f"Path is not a directory: {value}",
                    error_code="WRONG_TYPE",
                )

            # Extension check
            if (
                allowed_extensions
                and resolved_path.exists()
                and resolved_path.is_file()
                and resolved_path.suffix not in allowed_extensions
            ):
                result.add_error(
                    field_name,
                    f"File extension {resolved_path.suffix} not allowed",
                    error_code="INVALID_EXTENSION",
                    suggestions=[f"Use one of: {', '.join(allowed_extensions)}"],
                )

        except Exception as e:
            result.add_error(
                field_name, f"Invalid path: {str(e)}", error_code="INVALID_PATH"
            )

        return result

    @staticmethod
    def _resolve_allowed_bases(
        allowed_base_dirs: list[str | Path] | None,
    ) -> list[Path]:
        bases: list[Path] = []
        candidates: list[str | Path] = list(allowed_base_dirs or [])

        if not candidates:
            candidates.append(Path.cwd())

        for candidate in candidates:
            try:
                bases.append(Path(candidate).resolve())
            except Exception:
                logger.debug("Failed to resolve allowed base directory %s", candidate)

        if not bases:
            bases.append(Path.cwd().resolve())

        return bases

    @staticmethod
    def _is_relative_to(path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False

    # ===== Traigent-specific Validators =====

    _VALID_PARAM_TYPES: set[str] = {
        "float",
        "double",
        "loguniform",
        "qloguniform",
        "int",
        "integer",
        "categorical",
        "choice",
        "fixed",
        "constant",
    }

    _NUMERIC_TYPES: set[str] = {
        "float",
        "double",
        "loguniform",
        "qloguniform",
        "int",
        "integer",
    }

    _TYPED_KEYS: set[str] = {
        "type",
        "low",
        "high",
        "min",
        "max",
        "values",
        "choices",
        "value",
        "step",
        "log",
    }

    # Two values of a continuous knob closer than this fraction of its canonical
    # range are the same configuration in practice. Deliberately not applied to
    # whole-numbered knobs, where 1 is the finest gap that exists at all.
    _RESOLUTION_FLOOR_FRACTION: float = 0.02

    # Per-knob resolution floors in the knob's own units, for continuous knobs
    # whose canonical fraction is too fine to be meaningful.
    _RESOLUTION_FLOOR_OVERRIDES: dict[str, float] = {
        "temperature": 0.05,
        "top_p": 0.05,
    }

    # Knobs whose values are labels rather than settings: sweeping them measures
    # run-to-run variance, so the distance between two of them carries no
    # information and no closeness check applies.
    _VARIANCE_ONLY_KNOBS: frozenset[str] = frozenset({"seed"})

    # A sweep covering less than this fraction of a knob's canonical range
    # cannot tell the user whether the knob matters.
    _NARROW_SPAN_FRACTION: float = 0.2

    # Declared combinations per allowed trial above which a completed run is a
    # sample of the space rather than a search of it.
    _OVERSIZED_SPACE_RATIO: float = 4.0

    @staticmethod
    def _validate_list_param(
        result: ValidationResult, param_name: str, param_values: list[Any]
    ) -> None:
        """Validate a list parameter value."""
        if not param_values:
            result.add_error(
                f"configuration_space.{param_name}",
                "Parameter list cannot be empty",
                error_code="EMPTY_LIST",
            )
        elif len(param_values) == 1:
            result.add_warning(
                f"configuration_space.{param_name}",
                "Single value in list - no optimization possible",
                suggestions=["Add more values or remove this parameter"],
            )

    @staticmethod
    def _validate_tuple_param(
        result: ValidationResult, param_name: str, param_values: tuple[Any, ...]
    ) -> None:
        """Validate a tuple (range) parameter value."""
        if len(param_values) != 2:
            return
        min_val, max_val = param_values
        if not all(isinstance(v, (int, float)) for v in param_values):
            result.add_error(
                f"configuration_space.{param_name}",
                "Range values must be numeric",
                error_code="INVALID_RANGE",
            )
        elif min_val >= max_val:
            result.add_error(
                f"configuration_space.{param_name}",
                f"Invalid range: min ({min_val}) >= max ({max_val})",
                error_code="INVALID_RANGE",
            )

    @staticmethod
    def _validate_numeric_type_param(
        result: ValidationResult,
        param_name: str,
        param_type: str,
        param_values: dict[str, Any],
    ) -> None:
        """Validate a numeric type parameter (float, int, etc.)."""
        low = param_values.get("low", param_values.get("min"))
        high = param_values.get("high", param_values.get("max"))
        if low is None or high is None:
            result.add_error(
                f"configuration_space.{param_name}",
                f"{param_type} parameter requires bounds (low/high or min/max)",
                error_code="MISSING_BOUNDS",
            )
            return
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            result.add_error(
                f"configuration_space.{param_name}",
                "Range bounds must be numeric",
                error_code="INVALID_RANGE",
            )
            return
        if low >= high:
            result.add_error(
                f"configuration_space.{param_name}",
                f"Invalid range: low ({low}) >= high ({high})",
                error_code="INVALID_RANGE",
            )
            return

        step = param_values.get("step")
        if step is not None:
            if not isinstance(step, (int, float)) or step <= 0:
                result.add_error(
                    f"configuration_space.{param_name}",
                    f"Step must be a positive number, got {step!r}",
                    error_code="INVALID_RANGE",
                )
            if param_values.get("log"):
                result.add_error(
                    f"configuration_space.{param_name}",
                    "log=True cannot be combined with step",
                    error_code="INVALID_RANGE",
                )
        if param_values.get("log") and low <= 0:
            result.add_error(
                f"configuration_space.{param_name}",
                "log=True requires positive bounds",
                error_code="INVALID_RANGE",
            )

    @staticmethod
    def _validate_categorical_type_param(
        result: ValidationResult, param_name: str, param_values: dict[str, Any]
    ) -> None:
        """Validate a categorical/choice type parameter."""
        choices = param_values.get("choices") or param_values.get("values")
        if not choices:
            result.add_error(
                f"configuration_space.{param_name}",
                "Categorical parameter requires 'choices' or 'values'",
                error_code="INVALID_PARAM_TYPE",
            )
        elif isinstance(choices, (str, bytes)):
            result.add_error(
                f"configuration_space.{param_name}",
                "Categorical choices must be a list or tuple",
                error_code="INVALID_PARAM_TYPE",
            )

    @staticmethod
    def _validate_fixed_type_param(
        result: ValidationResult, param_name: str, param_values: dict[str, Any]
    ) -> None:
        """Validate a fixed/constant type parameter."""
        if "value" not in param_values:
            result.add_error(
                f"configuration_space.{param_name}",
                "Fixed parameters require a 'value'",
                error_code="INVALID_PARAM_TYPE",
            )

    @classmethod
    def _infer_param_type(cls, param_values: dict[str, Any]) -> str:
        """Infer parameter type from param_values dict."""
        param_type = (param_values.get("type") or "").lower()
        if param_type:
            return param_type
        if "choices" in param_values or "values" in param_values:
            return "categorical"
        if {"low", "high", "min", "max"} & param_values.keys():
            return "float"
        return "categorical"

    @classmethod
    def _validate_typed_dict_param(
        cls, result: ValidationResult, param_name: str, param_values: dict[str, Any]
    ) -> None:
        """Validate a typed dictionary parameter."""
        param_type = cls._infer_param_type(param_values)

        if param_type not in cls._VALID_PARAM_TYPES:
            result.add_error(
                f"configuration_space.{param_name}",
                f"Unknown parameter type: {param_type!r}",
                error_code="INVALID_PARAM_TYPE",
                suggestions=[
                    f"Valid types: {', '.join(sorted(cls._VALID_PARAM_TYPES))}"
                ],
            )
            return

        if param_type in cls._NUMERIC_TYPES:
            cls._validate_numeric_type_param(
                result, param_name, param_type, param_values
            )
        elif param_type in {"categorical", "choice"}:
            cls._validate_categorical_type_param(result, param_name, param_values)
        elif param_type in {"fixed", "constant"}:
            cls._validate_fixed_type_param(result, param_name, param_values)

    @classmethod
    def validate_configuration_space(
        cls, config_space: Any, *, max_trials: int | None = None
    ) -> ValidationResult:
        """Validate Traigent configuration space.

        ``max_trials`` is optional and only used to report a space whose size
        the trial budget cannot cover.
        """
        result = ValidationResult()

        # Type check
        if not isinstance(config_space, dict):
            result.add_error(
                "configuration_space",
                f"Expected dictionary, got {type(config_space).__name__}",
                error_code="TYPE_ERROR",
            )
            return result

        if not config_space:
            result.add_error(
                "configuration_space",
                "Configuration space cannot be empty",
                error_code="EMPTY_CONFIG",
                suggestions=["Add at least one parameter to optimize"],
            )
            return result

        from traigent.api.parameter_ranges import ParameterRange

        # Validate each parameter
        for param_name, param_values in config_space.items():
            if not isinstance(param_name, str):
                result.add_error(
                    f"configuration_space.{param_name}",
                    "Parameter name must be a string",
                    error_code="INVALID_PARAM_NAME",
                )
                continue

            # Convert ParameterRange to config value
            if isinstance(param_values, ParameterRange):
                param_values = param_values.to_config_value()

            cls._validate_single_param(result, param_name, param_values)

        # Non-fatal diagnostics: a structurally legal space can still be unable
        # to produce a meaningful comparison (issue #2025). Only worth running
        # once the space is known to be well-formed.
        if result.is_valid:
            cls._add_degenerate_variation_diagnostics(result, config_space, max_trials)

        # Add suggestions for common parameters
        if "model" not in config_space:
            result.suggestions.append(
                "Consider adding 'model' parameter for LLM selection"
            )
        if "temperature" not in config_space:
            result.suggestions.append(
                "Consider adding 'temperature' parameter for output randomness"
            )

        return result

    @classmethod
    def _validate_single_param(
        cls, result: ValidationResult, param_name: str, param_values: Any
    ) -> None:
        """Validate a single configuration space parameter."""
        if isinstance(param_values, list):
            cls._validate_list_param(result, param_name, param_values)
        elif isinstance(param_values, tuple) and len(param_values) == 2:
            cls._validate_tuple_param(result, param_name, param_values)
        elif isinstance(param_values, dict):
            cls._validate_dict_param(result, param_name, param_values)
        else:
            result.add_error(
                f"configuration_space.{param_name}",
                "Parameter must be a list of values or a (min, max) tuple",
                error_code="INVALID_PARAM_TYPE",
                suggestions=[
                    "Use a list for categorical values: ['option1', 'option2']",
                    "Use a tuple for numeric ranges: (0.0, 1.0)",
                ],
            )

    @classmethod
    def _validate_dict_param(
        cls, result: ValidationResult, param_name: str, param_values: dict[str, Any]
    ) -> None:
        """Validate a dictionary parameter value."""
        if len(param_values) == 0:
            result.add_error(
                f"configuration_space.{param_name}",
                "Parameter dict cannot be empty",
                error_code="EMPTY_CONFIG",
            )
            return

        # Check if this is a typed parameter definition
        if cls._TYPED_KEYS.intersection(param_values):
            cls._validate_typed_dict_param(result, param_name, param_values)
        # Otherwise accept as nested configuration (no validation needed)

    # ----- Degenerate variation diagnostics (non-fatal) -----

    @staticmethod
    def _declared_values(param_values: Any) -> list[Any] | None:
        """Return the explicit value list a parameter declares, if it has one."""
        if isinstance(param_values, list):
            return param_values
        if isinstance(param_values, dict):
            choices = param_values.get("choices")
            if choices is None:
                choices = param_values.get("values")
            if isinstance(choices, (list, tuple)):
                return list(choices)
        return None

    @staticmethod
    def _as_number(value: Any) -> float | None:
        """Return *value* as a float, or None when it is not a real number."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @classmethod
    def _numeric_values(cls, declared: list[Any]) -> list[float] | None:
        """Return *declared* as floats, or None unless every entry is a number."""
        numbers: list[float] = []
        for value in declared:
            number = cls._as_number(value)
            if number is None:
                return None
            numbers.append(number)
        return numbers or None

    @classmethod
    def _swept_span(cls, param_values: Any) -> tuple[float, float] | None:
        """Return the (low, high) a numeric parameter actually sweeps."""
        declared = cls._declared_values(param_values)
        if declared is not None:
            numbers = cls._numeric_values(declared)
            if numbers is None:
                return None
            return min(numbers), max(numbers)

        if isinstance(param_values, tuple) and len(param_values) == 2:
            low, high = (cls._as_number(bound) for bound in param_values)
        elif isinstance(param_values, dict):
            low = cls._as_number(param_values.get("low", param_values.get("min")))
            high = cls._as_number(param_values.get("high", param_values.get("max")))
        else:
            return None

        if low is None or high is None:
            return None
        return low, high

    @classmethod
    def _resolution_floor(cls, param_name: str) -> float | None:
        """Smallest difference that can matter for *param_name*, if known."""
        canonical = _canonical_knob_ranges().get(param_name)
        if canonical is None:
            return None
        if canonical.is_integer:
            # 1 is the finest gap a whole-numbered knob can express, so any two
            # values that far apart are genuinely two configurations - top_k=1
            # is greedy decoding and top_k=2 is not. Scaling the floor to the
            # canonical range here would make the check unfalsifiable.
            return 1.0
        override = cls._RESOLUTION_FLOOR_OVERRIDES.get(param_name)
        if override is not None:
            return override
        return (canonical.high - canonical.low) * cls._RESOLUTION_FLOOR_FRACTION

    @classmethod
    def _report_duplicate_values(
        cls, result: ValidationResult, param_name: str, param_values: Any
    ) -> int:
        """Warn about repeated values in a knob; return how many repeat."""
        declared = cls._declared_values(param_values)
        if not declared:
            return 0

        distinct: list[Any] = []
        repeated: list[Any] = []
        for value in declared:
            if _strict_choice_match(value, distinct):
                if not _strict_choice_match(value, repeated):
                    repeated.append(value)
            else:
                distinct.append(value)

        if not repeated:
            return 0

        result.add_warning(
            f"configuration_space.{param_name}",
            f"{len(declared)} values declared but only {len(distinct)} are distinct "
            f"({', '.join(repr(value) for value in repeated)} repeated) - a repeat is "
            "the same configuration again, so those trials re-measure a value already "
            "tried instead of exploring a new one",
            suggestions=[
                f"Remove the repeated values: {sorted(distinct, key=repr)!r}",
            ],
        )
        return len(declared) - len(distinct)

    @classmethod
    def _closest_pair(cls, param_values: Any) -> tuple[float, float, float] | None:
        """Return (gap, lower, upper) for the two closest values a knob offers."""
        declared = cls._declared_values(param_values)
        if declared is not None:
            numbers = cls._numeric_values(declared)
            if numbers is None:
                return None
            ordered = sorted(set(numbers))
            if len(ordered) < 2:
                return None
            return min(
                (
                    ordered[index + 1] - ordered[index],
                    ordered[index],
                    ordered[index + 1],
                )
                for index in range(len(ordered) - 1)
            )

        # A stepped range is never materialized here: the smallest gap between
        # two of its grid points is the step itself.
        if not isinstance(param_values, dict):
            return None
        step = cls._as_number(param_values.get("step"))
        span = cls._swept_span(param_values)
        if step is None or step <= 0 or span is None or span[1] - span[0] < step:
            return None
        return step, span[0], span[0] + step

    @staticmethod
    def _spaced_example(canonical: _KnobRange) -> str:
        """A three-point sweep of *canonical*, kept whole for integer knobs."""
        midpoint = (canonical.low + canonical.high) / 2
        if canonical.is_integer:
            points = sorted({int(canonical.low), int(midpoint), int(canonical.high)})
            return repr(points)
        return f"[{canonical.low:g}, {midpoint:g}, {canonical.high:g}]"

    @classmethod
    def _report_indistinguishable_values(
        cls, result: ValidationResult, param_name: str, param_values: Any
    ) -> None:
        """Warn when two declared values are too close to behave differently."""
        canonical = _canonical_knob_ranges().get(param_name)
        floor = cls._resolution_floor(param_name)
        if canonical is None or floor is None or floor <= 0:
            return

        closest = cls._closest_pair(param_values)
        if closest is None:
            return

        gap, lower, upper = closest
        if gap >= floor:
            return

        if canonical.is_integer:
            # Only reachable for fractional values: two whole numbers are always
            # at least 1 apart, which is exactly the floor.
            reason = (
                f"below the 1 that separates two whole values of {param_name}, "
                "a whole-number setting"
            )
            fix = (
                "Use whole numbers, at least 1 apart, "
                f"e.g. {cls._spaced_example(canonical)}"
            )
        else:
            reason = (
                f"far below the ~{floor:g} that changes behaviour across "
                f"{param_name}'s usual {canonical.low:g}-{canonical.high:g} range"
            )
            fix = (
                f"Space the values at least {floor:g} apart, "
                f"e.g. {cls._spaced_example(canonical)}"
            )

        result.add_warning(
            f"configuration_space.{param_name}",
            f"{lower:g} and {upper:g} differ by {gap:g}, {reason} - these two values "
            "are the same configuration in practice, so the trial comparing them "
            "measures noise rather than the parameter",
            suggestions=[fix],
        )

    @classmethod
    def _report_narrow_span(
        cls, result: ValidationResult, param_name: str, param_values: Any
    ) -> None:
        """Warn when a knob only sweeps a sliver of its canonical range."""
        canonical = _canonical_knob_ranges().get(param_name)
        if canonical is None or not canonical.bounds_whole_domain:
            return
        span = cls._swept_span(param_values)
        if span is None:
            return

        swept_low, swept_high = span
        low, high = canonical.low, canonical.high
        swept = swept_high - swept_low
        if swept <= 0:
            return  # A single point is already reported as "no optimization possible"

        covered = swept / (high - low)
        if covered >= cls._NARROW_SPAN_FRACTION:
            return

        result.add_warning(
            f"configuration_space.{param_name}",
            f"sweeps {swept_low:g}-{swept_high:g}, only {covered:.0%} of "
            f"{param_name}'s usual {low:g}-{high:g} range - the run can tell you which "
            f"of these settings won, not whether {param_name} matters for your task",
            suggestions=[
                f"Widen the sweep toward {low:g}-{high:g} to learn whether "
                f"{param_name} is worth tuning at all",
            ],
        )

    @classmethod
    def _add_degenerate_variation_diagnostics(
        cls,
        result: ValidationResult,
        config_space: dict[str, Any],
        max_trials: int | None,
    ) -> None:
        """Report structurally legal spaces that cannot produce a real comparison.

        Every finding is a warning, never an error: the space is runnable, it
        just cannot answer the question the user thinks it asks.
        """
        from traigent.api.parameter_ranges import ParameterRange
        from traigent.utils.discrete_domains import (
            discrete_cardinality_for_config_param,
        )

        total_combinations: int | None = 1
        varying_params = 0

        for param_name, raw_values in config_space.items():
            param_values = (
                raw_values.to_config_value()
                if isinstance(raw_values, ParameterRange)
                else raw_values
            )

            # A repeat is the same configuration again whatever the knob means,
            # so duplicates are reported for every parameter. The other two ask
            # what the distance between values is worth, which only some knobs
            # can answer.
            duplicates = cls._report_duplicate_values(result, param_name, param_values)
            if param_name not in cls._VARIANCE_ONLY_KNOBS:
                cls._report_indistinguishable_values(result, param_name, param_values)
                cls._report_narrow_span(result, param_name, param_values)

            cardinality = discrete_cardinality_for_config_param(param_values)
            if cardinality is None:
                # Continuous range: always varies, but has no finite cardinality.
                varying_params += 1
                total_combinations = None
                continue

            cardinality -= duplicates
            if cardinality >= 2:
                varying_params += 1
            if total_combinations is not None:
                total_combinations *= max(cardinality, 1)

        cls._report_single_varying_param(result, config_space, varying_params)
        cls._report_undersized_budget(result, total_combinations, max_trials)

    @classmethod
    def _report_single_varying_param(
        cls, result: ValidationResult, config_space: dict[str, Any], varying: int
    ) -> None:
        """Warn when a multi-parameter space only varies along one axis."""
        # A deliberately single-parameter space is a normal thing to declare;
        # the surprise is declaring several and pinning all but one.
        if len(config_space) < 2 or varying > 1:
            return

        if varying == 0:
            result.add_warning(
                "configuration_space",
                f"none of the {len(config_space)} parameters has two or more distinct "
                "values - every trial runs the identical configuration, so the "
                "reported best configuration is the only one that ever existed",
                suggestions=["Give at least one parameter two or more distinct values"],
            )
            return

        result.add_warning(
            "configuration_space",
            f"{len(config_space)} parameters declared but only one of them varies - "
            "every difference between trials comes from that one parameter, so the "
            "run cannot say anything about the others",
            suggestions=[
                "Add values to the pinned parameters, or drop them from "
                "configuration_space so the results are not read as covering them",
            ],
        )

    @classmethod
    def _report_undersized_budget(
        cls,
        result: ValidationResult,
        total_combinations: int | None,
        max_trials: int | None,
    ) -> None:
        """Warn when max_trials can only reach a fraction of the declared space."""
        if total_combinations is None or max_trials is None or max_trials <= 0:
            return
        if total_combinations <= max_trials * cls._OVERSIZED_SPACE_RATIO:
            return

        result.add_warning(
            "configuration_space",
            f"{total_combinations} distinct configurations declared but max_trials="
            f"{max_trials} - the run can reach at most "
            f"{max_trials / total_combinations:.0%} of them, so a completed run is a "
            "sample of the space, not a search of what you declared",
            suggestions=[
                f"Raise max_trials, or shrink the space so it fits the budget "
                f"(currently {total_combinations} combinations)",
            ],
        )

    @staticmethod
    def validate_objectives(objectives: Any) -> ValidationResult:
        """Validate optimization objectives."""
        result = ValidationResult()

        # Type check
        if not isinstance(objectives, list):
            result.add_error(
                "objectives",
                f"Expected list, got {type(objectives).__name__}",
                error_code="TYPE_ERROR",
            )
            return result

        if not objectives:
            result.add_error(
                "objectives",
                "Objectives list cannot be empty",
                error_code="EMPTY_LIST",
                suggestions=[
                    "Add at least one objective like 'accuracy', 'cost', or 'latency'"
                ],
            )
            return result

        # Validate each objective
        valid_objectives = {
            "accuracy",
            "cost",
            "latency",
            "throughput",
            "quality",
            "relevance",
            "coherence",
            "safety",
            "helpfulness",
        }

        for i, obj in enumerate(objectives):
            if not isinstance(obj, str):
                result.add_error(
                    f"objectives[{i}]",
                    f"Objective must be string, got {type(obj).__name__}",
                    error_code="TYPE_ERROR",
                )
            elif obj not in valid_objectives:
                result.add_warning(
                    f"objectives[{i}]",
                    f"Unknown objective '{obj}'",
                    suggestions=[
                        f"Common objectives: {', '.join(sorted(valid_objectives))}"
                    ],
                )

        # Check for conflicting objectives
        if "cost" in objectives and "quality" in objectives:
            result.add_warning(
                "objectives",
                "Optimizing for both 'cost' and 'quality' may lead to trade-offs",
                suggestions=["Consider using multi-objective optimization or weights"],
            )

        return result

    @staticmethod
    def validate_dataset(
        dataset_path: Any,
        *,
        base_dir: str | Path | None = None,
    ) -> ValidationResult:
        """Validate a dataset file or path.

        Relative paths are resolved within ``base_dir`` (or the invoking
        process's current working directory when omitted). Absolute paths keep
        their existing compatibility behavior and are not constrained by that
        relative-path boundary.
        """
        result = ValidationResult()

        from traigent.evaluators.base import (
            Dataset,
            EvaluationExample,
            load_inline_dataset,
        )

        def resolve_dataset_path(
            raw_path: Path, dataset_base: Path | None
        ) -> Path | None:
            if raw_path.is_absolute():
                candidate = raw_path
            else:
                if dataset_base is None:
                    result.add_error(
                        "dataset",
                        "Dataset base directory is required for a relative path",
                        error_code="SECURITY_ERROR",
                    )
                    return None
                candidate = dataset_base / raw_path

            try:
                return candidate.resolve(strict=True)
            except FileNotFoundError:
                # Keep the public validator's established NOT_FOUND result for a
                # missing in-base target while strictly resolving existing paths.
                return candidate.resolve()
            except (OSError, RuntimeError) as exc:
                result.add_error(
                    "dataset",
                    f"Dataset path cannot be resolved: {exc}",
                    error_code="SECURITY_ERROR",
                )
                return None

        if isinstance(dataset_path, Dataset):
            return result

        dataset_base: Path | None = None
        if isinstance(dataset_path, list):
            if all(
                isinstance(item, (dict, EvaluationExample)) for item in dataset_path
            ):
                try:
                    load_inline_dataset(dataset_path)
                except ValidationException as exc:
                    result.add_error(
                        "dataset",
                        str(exc),
                        error_code="INVALID_FORMAT",
                    )
                return result

            if not all(isinstance(item, (str, Path)) for item in dataset_path):
                result.add_error(
                    "dataset",
                    "Dataset list entries must all be file paths or inline example objects",
                    error_code="TYPE_ERROR",
                )
                return result

            relative_paths = [
                path for path in dataset_path if not Path(path).is_absolute()
            ]
            if relative_paths:
                try:
                    dataset_base = (
                        Path(Path.cwd() if base_dir is None else base_dir)
                        .expanduser()
                        .resolve(strict=True)
                    )
                except (OSError, RuntimeError) as exc:
                    result.add_error(
                        "dataset",
                        f"Dataset base directory cannot be resolved: {exc}",
                        error_code="SECURITY_ERROR",
                    )
                    return result

                if not dataset_base.is_dir():
                    result.add_error(
                        "dataset",
                        f"Dataset base path is not a directory: {dataset_base}",
                        error_code="SECURITY_ERROR",
                    )
                    return result

            # Multiple datasets
            for i, path in enumerate(dataset_path):
                raw_path = Path(path)
                if not raw_path.is_absolute():
                    if dataset_base is None:
                        result.add_error(
                            "dataset",
                            "Dataset base directory is required for a relative path",
                            error_code="SECURITY_ERROR",
                        )
                        return result
                    allowed_base = dataset_base
                resolved_dataset_path = resolve_dataset_path(raw_path, dataset_base)
                if resolved_dataset_path is None:
                    return result
                if raw_path.is_absolute():
                    allowed_base = resolved_dataset_path.parent
                path_result = Validators.validate_path(
                    resolved_dataset_path,
                    f"dataset[{i}]",
                    must_exist=True,
                    must_be_file=True,
                    allowed_extensions=[".json", ".jsonl"],
                    allowed_base_dirs=[allowed_base],
                )
                result.errors.extend(path_result.errors)
                result.warnings.extend(path_result.warnings)
        elif isinstance(dataset_path, (str, Path)):
            # Single dataset
            raw_dataset_path = Path(dataset_path)
            if not raw_dataset_path.is_absolute():
                try:
                    dataset_base = (
                        Path(Path.cwd() if base_dir is None else base_dir)
                        .expanduser()
                        .resolve(strict=True)
                    )
                except (OSError, RuntimeError) as exc:
                    result.add_error(
                        "dataset",
                        f"Dataset base directory cannot be resolved: {exc}",
                        error_code="SECURITY_ERROR",
                    )
                    return result

                if not dataset_base.is_dir():
                    result.add_error(
                        "dataset",
                        f"Dataset base path is not a directory: {dataset_base}",
                        error_code="SECURITY_ERROR",
                    )
                    return result

            if not raw_dataset_path.is_absolute():
                if dataset_base is None:
                    result.add_error(
                        "dataset",
                        "Dataset base directory is required for a relative path",
                        error_code="SECURITY_ERROR",
                    )
                    return result
                allowed_base = dataset_base
            resolved_dataset_path = resolve_dataset_path(raw_dataset_path, dataset_base)
            if resolved_dataset_path is None:
                return result
            if raw_dataset_path.is_absolute():
                allowed_base = resolved_dataset_path.parent
            path_result = Validators.validate_path(
                resolved_dataset_path,
                "dataset",
                must_exist=True,
                must_be_file=True,
                allowed_extensions=[".json", ".jsonl"],
                allowed_base_dirs=[allowed_base],
            )
            result.errors.extend(path_result.errors)
            result.warnings.extend(path_result.warnings)

            # Try to validate content if path is valid
            if path_result.is_valid:
                try:
                    # Use the RESOLVED absolute path (and its parent as base_dir)
                    # so safe_open's containment guard does not re-join a relative
                    # path onto its own parent (which doubled nested segments and
                    # produced a spurious READ_ERROR for relative dataset paths).
                    with safe_open(
                        resolved_dataset_path,
                        resolved_dataset_path.parent,
                        mode="r",
                        encoding="utf-8",
                    ) as f:
                        line_count = 0
                        for line_num, line in enumerate(f, 1):
                            line_count += 1
                            if line_count > 5:  # Only check first 5 lines
                                break

                            try:
                                data = json.loads(line.strip())
                                if "input" not in data and "input_data" not in data:
                                    result.add_error(
                                        f"dataset:line{line_num}",
                                        "Missing 'input' or 'input_data' field",
                                        error_code="INVALID_FORMAT",
                                    )
                            except json.JSONDecodeError:
                                result.add_error(
                                    f"dataset:line{line_num}",
                                    "Invalid JSON",
                                    error_code="JSON_ERROR",
                                )

                        if line_count == 0:
                            result.add_error(
                                "dataset",
                                "Dataset file is empty",
                                error_code="EMPTY_FILE",
                            )

                except Exception as e:
                    result.add_error(
                        "dataset",
                        f"Could not read dataset: {str(e)}",
                        error_code="READ_ERROR",
                    )
        else:
            result.add_error(
                "dataset",
                f"Expected dataset path, inline example list, or Dataset object, got {type(dataset_path).__name__}",
                error_code="TYPE_ERROR",
            )

        return result


# ===== Metric Validation Functions =====


def validate_numeric_metric(
    value: Any,
    field_name: str,
    trial_id: str | None = None,
    example_id: str | None = None,
) -> float:
    """Validate and convert metric to float with strict checks.

    This function NEVER returns a default value (like 0.0) on failure.
    Instead, it raises MetricExtractionError to prevent silent data corruption
    that could invalidate optimization results.

    Args:
        value: The value to validate and convert
        field_name: Name of the metric field
        trial_id: Optional trial identifier for error context
        example_id: Optional example identifier for error context

    Returns:
        The validated value as a float

    Raises:
        MetricExtractionError: If value is None, NaN, Inf, or cannot be converted

    Example:
        >>> validate_numeric_metric(0.95, "accuracy")
        0.95
        >>> validate_numeric_metric("0.95", "accuracy")
        0.95
        >>> validate_numeric_metric(None, "accuracy")
        MetricExtractionError: Metric 'accuracy' is None
        >>> validate_numeric_metric(float('nan'), "accuracy")
        MetricExtractionError: Metric 'accuracy' is NaN or Inf
        >>> validate_numeric_metric("invalid", "accuracy")
        MetricExtractionError: Cannot convert metric 'accuracy' to numeric
    """
    import math

    from traigent.utils.exceptions import MetricExtractionError

    if value is None:
        raise MetricExtractionError(
            f"Metric '{field_name}' is None",
            field=field_name,
            value=value,
            trial_id=trial_id,
            example_id=example_id,
        )

    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            raise MetricExtractionError(
                f"Metric '{field_name}' is NaN or Inf",
                field=field_name,
                value=value,
                trial_id=trial_id,
                example_id=example_id,
            )
        return float(value)

    try:
        converted = float(value)
        if math.isnan(converted) or math.isinf(converted):
            raise ValueError("Converted to NaN or Inf")
        return converted
    except (TypeError, ValueError) as e:
        raise MetricExtractionError(
            f"Cannot convert metric '{field_name}' to numeric: {value!r}",
            field=field_name,
            value=value,
            trial_id=trial_id,
            example_id=example_id,
            original_error=e,
        ) from e


# ===== Convenience Functions (Backward Compatibility) =====


def _type_name(value: Any) -> str:
    return type(value).__name__


def _raise_default_config_domain_error(
    param_name: str,
    value: Any,
    detail: str,
) -> None:
    raise ConfigurationError(
        f"default_config[{param_name!r}] value {value!r} "
        f"(type {_type_name(value)}) {detail}"
    )


def _strict_choice_match(value: Any, choices: list[Any] | tuple[Any, ...]) -> bool:
    return any(type(value) is type(choice) and value == choice for choice in choices)


def _validate_default_choice(
    param_name: str,
    value: Any,
    choices: list[Any] | tuple[Any, ...],
) -> None:
    if _strict_choice_match(value, choices):
        return

    for choice in choices:
        if value == choice:
            _raise_default_config_domain_error(
                param_name,
                value,
                f"matches configuration_space[{param_name!r}] choice {choice!r} "
                f"but does not match declared type {_type_name(choice)}",
            )


def _validate_default_config_value(
    param_name: str,
    value: Any,
    param_values: Any,
) -> None:
    from traigent.api.parameter_ranges import ParameterRange

    if isinstance(param_values, ParameterRange):
        param_values = param_values.to_config_value()

    if isinstance(param_values, list):
        _validate_default_choice(param_name, value, param_values)
        return

    if isinstance(param_values, tuple) and len(param_values) == 2:
        return

    if isinstance(param_values, dict):
        param_type = Validators._infer_param_type(param_values)
        if param_type in {"categorical", "choice"}:
            choices = param_values.get("choices") or param_values.get("values")
            if isinstance(choices, (list, tuple)):
                _validate_default_choice(param_name, value, choices)
            return

        if param_type in Validators._NUMERIC_TYPES:
            return


def _validate_default_config_against_config_space(
    config_space: dict[str, Any],
    default_config: dict[str, Any] | None,
) -> None:
    if not isinstance(default_config, dict):
        return

    for param_name, value in default_config.items():
        if param_name not in config_space:
            continue
        _validate_default_config_value(param_name, value, config_space[param_name])


def validate_config_space(
    config_space: dict[str, Any],
    default_config: dict[str, Any] | None = None,
) -> None:
    """Validate configuration space (raises exception on error)."""
    result = Validators.validate_configuration_space(config_space)
    result.raise_if_invalid()
    _validate_default_config_against_config_space(config_space, default_config)


def validate_objectives(objectives: list[str]) -> None:
    """Validate objectives list (raises exception on error)."""
    result = Validators.validate_objectives(objectives)
    result.raise_if_invalid()


def validate_dataset_path(dataset_path: str | list[str]) -> None:
    """Validate dataset path (raises exception on error)."""
    result = Validators.validate_dataset(dataset_path)
    result.raise_if_invalid()


def validate_positive_int(value: int, name: str) -> None:
    """Validate positive integer (raises exception on error)."""
    result = Validators.validate_positive_int(value, name)
    result.raise_if_invalid()


def validate_probability(value: float, name: str) -> None:
    """Validate probability value (raises exception on error)."""
    result = Validators.validate_probability(value, name)
    result.raise_if_invalid()


def validate_or_raise(result: ValidationResult) -> None:
    """Raise exception if validation failed (common_validators compatibility)."""
    if not result.is_valid:
        # Check if we need ValueError for backward compatibility (API functions)
        import inspect

        frame = inspect.currentframe()
        if (
            frame
            and frame.f_back
            and frame.f_back.f_code.co_filename.endswith("api/functions.py")
        ):
            # API functions expect ValueError for backward compatibility
            if result.errors:
                first_error = result.errors[0]
                msg = first_error.message
                if "positive" in msg.lower():
                    raise ValueError("Must be positive") from None
                elif "invalid choice" in msg.lower():
                    raise ValueError("Must be one of")
                elif "expected dict" in msg.lower():
                    raise ValueError(f"Expected dict, got {msg.split('got ')[-1]}")
                else:
                    raise ValueError(msg)
            else:
                raise ValueError("Validation failed")
        else:
            # Normal case - use ValidationError
            result.raise_if_invalid()


# ===== High-Level Validators =====


class OptimizationValidator:
    """Comprehensive validator for optimization configurations."""

    @classmethod
    def validate_optimization_config(
        cls,
        config_space: dict[str, Any],
        objectives: list[str],
        dataset: str | list[str] | None = None,
        strategy: str | None = None,
        max_trials: int | None = None,
    ) -> ValidationResult:
        """Validate optimization configuration (class method for backward compatibility)."""
        validator = cls()
        return validator.validate(
            config_space=config_space,
            objectives=objectives,
            dataset=dataset,
            max_trials=max_trials,
        )

    def validate(
        self,
        config_space: dict[str, Any] | None = None,
        objectives: list[str] | None = None,
        dataset: str | list[str] | None = None,
        constraints: list[Callable[..., Any]] | None = None,
        max_trials: int | None = None,
    ) -> ValidationResult:
        """Validate complete optimization setup."""
        result = ValidationResult()

        # Validate configuration space
        if config_space is not None:
            config_result = Validators.validate_configuration_space(
                config_space, max_trials=max_trials
            )
            result.errors.extend(config_result.errors)
            result.warnings.extend(config_result.warnings)
            result.suggestions.extend(config_result.suggestions)

        # Validate objectives
        if objectives is not None:
            obj_result = Validators.validate_objectives(objectives)
            result.errors.extend(obj_result.errors)
            result.warnings.extend(obj_result.warnings)
            result.suggestions.extend(obj_result.suggestions)

        # Validate dataset
        if dataset is not None:
            dataset_result = Validators.validate_dataset(dataset)
            result.errors.extend(dataset_result.errors)
            result.warnings.extend(dataset_result.warnings)

        # Validate constraints
        if constraints is not None:
            if not isinstance(constraints, list):
                result.add_error(
                    "constraints",
                    "Constraints must be a list of functions",
                    error_code="TYPE_ERROR",
                )
            else:
                for i, constraint in enumerate(constraints):
                    if not callable(constraint):
                        result.add_error(
                            f"constraints[{i}]",
                            "Constraint must be callable",
                            error_code="TYPE_ERROR",
                        )

        return result


def validate_and_suggest(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to validate optimization parameters with helpful suggestions."""

    def wrapper(*args, **kwargs):
        # Extract parameters
        config_space = kwargs.get("configuration_space")
        objectives = kwargs.get("objectives")
        dataset = kwargs.get("eval_dataset")

        # Validate
        validator = OptimizationValidator()
        result = validator.validate(
            config_space=config_space, objectives=objectives, dataset=dataset
        )

        # Show feedback if invalid
        if not result.is_valid:
            logger.error(result.get_feedback())
            result.raise_if_invalid()

        # Show warnings if any
        if result.has_warnings:
            logger.warning(result.get_feedback(include_warnings=True))

        return func(*args, **kwargs)

    return wrapper


# Export commonly used names for backward compatibility
CoreValidators = Validators
ConfigurationValidator = OptimizationValidator
DatasetValidator = OptimizationValidator  # They share the same validation logic
