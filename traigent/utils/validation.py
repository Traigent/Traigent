"""Consolidated validation utilities for TraiGent SDK.

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
from pathlib import Path
from typing import Any

from traigent.utils.exceptions import ValidationError as ValidationException
from traigent.utils.logging import get_logger

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

    # ===== TraiGent-specific Validators =====

    @staticmethod
    def validate_configuration_space(config_space: Any) -> ValidationResult:
        """Validate TraiGent configuration space."""
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

        # Validate each parameter
        for param_name, param_values in config_space.items():
            # Check parameter name
            if not isinstance(param_name, str):
                result.add_error(
                    f"configuration_space.{param_name}",
                    "Parameter name must be a string",
                    error_code="INVALID_PARAM_NAME",
                )
                continue

            # Check parameter values
            if isinstance(param_values, list):
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
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # Range validation
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
    def validate_dataset(dataset_path: Any) -> ValidationResult:
        """Validate dataset file or path."""
        result = ValidationResult()

        if isinstance(dataset_path, list):
            # Multiple datasets
            for i, path in enumerate(dataset_path):
                path_result = Validators.validate_path(
                    path,
                    f"dataset[{i}]",
                    must_exist=True,
                    must_be_file=True,
                    allowed_extensions=[".json", ".jsonl"],
                    allowed_base_dirs=[Path(path).resolve().parent],
                )
                result.errors.extend(path_result.errors)
                result.warnings.extend(path_result.warnings)
        else:
            # Single dataset
            resolved_dataset_path = Path(dataset_path)
            path_result = Validators.validate_path(
                resolved_dataset_path,
                "dataset",
                must_exist=True,
                must_be_file=True,
                allowed_extensions=[".json", ".jsonl"],
                allowed_base_dirs=[resolved_dataset_path.resolve().parent],
            )
            result.errors.extend(path_result.errors)
            result.warnings.extend(path_result.warnings)

            # Try to validate content if path is valid
            if path_result.is_valid:
                try:
                    path = Path(dataset_path)
                    with open(path) as f:
                        line_count = 0
                        for line_num, line in enumerate(f, 1):
                            line_count += 1
                            if line_count > 5:  # Only check first 5 lines
                                break

                            try:
                                data = json.loads(line.strip())
                                if "input" not in data:
                                    result.add_error(
                                        f"dataset:line{line_num}",
                                        "Missing 'input' field",
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

        return result


# ===== Convenience Functions (Backward Compatibility) =====


def validate_config_space(config_space: dict[str, Any]) -> None:
    """Validate configuration space (raises exception on error)."""
    result = Validators.validate_configuration_space(config_space)
    result.raise_if_invalid()


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
    ) -> ValidationResult:
        """Validate optimization configuration (class method for backward compatibility)."""
        validator = cls()
        return validator.validate(
            config_space=config_space, objectives=objectives, dataset=dataset
        )

    def validate(
        self,
        config_space: dict[str, Any] | None = None,
        objectives: list[str] | None = None,
        dataset: str | list[str] | None = None,
        constraints: list[Callable[..., Any]] | None = None,
    ) -> ValidationResult:
        """Validate complete optimization setup."""
        result = ValidationResult()

        # Validate configuration space
        if config_space is not None:
            config_result = Validators.validate_configuration_space(config_space)
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
