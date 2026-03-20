"""
Parameter validation for the @traigent.optimize decorator.

This module extracts parameter validation logic from the main decorator
to reduce complexity and improve maintainability.
Traceability: CONC-Layer-API CONC-Quality-Reliability CONC-Quality-Usability CONC-Quality-Maintainability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

from traigent.config.types import ExecutionMode, InjectionMode, resolve_execution_mode
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class OptimizeParameters:
    """Structured representation of optimize decorator parameters."""

    eval_dataset: (
        str | list[str | EvaluationExample | dict[str, Any]] | Dataset | None
    ) = None
    objectives: list[str] | None = None
    configuration_space: dict[str, Any] | None = None
    default_config: dict[str, Any] | None = None
    constraints: list[Callable[..., Any]] | None = None
    injection_mode: str | InjectionMode = InjectionMode.CONTEXT
    config_param: str | None = None
    auto_override_frameworks: bool = False  # Requires traigent-integrations plugin
    framework_targets: list[str] | None = None
    execution_mode: str | ExecutionMode = ExecutionMode.EDGE_ANALYTICS
    local_storage_path: str | None = None
    minimal_logging: bool = True
    privacy_enabled: bool | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class ParameterValidator:
    """Validates and normalizes parameters for the optimize decorator."""

    VALID_EXECUTION_MODES = {mode.value for mode in ExecutionMode}
    VALID_INJECTION_MODES = {
        InjectionMode.CONTEXT,
        InjectionMode.PARAMETER,
        InjectionMode.SEAMLESS,
    }

    def validate_parameters(self, params: OptimizeParameters) -> OptimizeParameters:
        """
        Validate and normalize all decorator parameters.

        Args:
            params: Parameter object to validate

        Returns:
            Validated and normalized parameters

        Raises:
            ValidationError: If any parameter is invalid
        """
        # Validate individual parameter groups
        self._validate_execution_mode(params.execution_mode)
        self._validate_injection_mode(params.injection_mode)
        self._validate_dataset(params.eval_dataset)
        self._validate_objectives(params.objectives)
        self._validate_configuration_space(params.configuration_space)
        self._validate_constraints(params.constraints)

        # Normalize parameters
        params.injection_mode = self._normalize_injection_mode(params.injection_mode)
        params.execution_mode = self._normalize_execution_mode(params.execution_mode)

        return params

    def _validate_execution_mode(self, execution_mode: str | ExecutionMode) -> None:
        """Validate execution mode parameter."""
        from traigent.utils.exceptions import ConfigurationError

        try:
            normalized = resolve_execution_mode(execution_mode)
        except (TypeError, ValueError, ConfigurationError) as exc:
            raise ValidationError(
                f"Invalid execution_mode '{execution_mode}'. {exc}"
            ) from exc

        if normalized.value not in self.VALID_EXECUTION_MODES:
            raise ValidationError(
                f"Invalid execution_mode '{execution_mode}'. "
                f"Must be one of: {', '.join(self.VALID_EXECUTION_MODES)}"
            )

    def _normalize_execution_mode(self, execution_mode: str | ExecutionMode) -> str:
        """Normalize execution mode string for internal consistency."""
        return resolve_execution_mode(execution_mode).value

    def _validate_injection_mode(self, injection_mode: str | InjectionMode) -> None:
        """Validate injection mode parameter."""
        if isinstance(injection_mode, str):
            try:
                injection_mode = InjectionMode(injection_mode)
            except ValueError as e:
                raise ValidationError(
                    f"Invalid injection_mode '{injection_mode}'. "
                    f"Must be one of: {[mode.value for mode in InjectionMode]}"
                ) from e

        if injection_mode not in self.VALID_INJECTION_MODES:
            raise ValidationError(
                f"Invalid injection_mode '{injection_mode}'. "
                f"Must be one of: {[mode.value for mode in self.VALID_INJECTION_MODES]}"
            )

    def _validate_dataset(
        self,
        eval_dataset: (
            str | list[str | EvaluationExample | dict[str, Any]] | Dataset | None
        ),
    ) -> None:
        """Validate evaluation dataset parameter."""
        if eval_dataset is None:
            return

        if isinstance(eval_dataset, (str, Dataset)):
            return

        if isinstance(eval_dataset, Sequence) and not isinstance(
            eval_dataset, (str, bytes)
        ):
            invalid_entries = [
                entry
                for entry in eval_dataset
                if not isinstance(entry, (str, EvaluationExample, Mapping))
            ]
            if invalid_entries:
                invalid_types = ", ".join(
                    sorted({type(entry).__name__ for entry in invalid_entries})
                )
                raise ValidationError(
                    "eval_dataset iterable must contain only string paths, inline example dicts, "
                    "or EvaluationExample objects. "
                    f"Invalid entries of types: {invalid_types}"
                )

            has_paths = any(isinstance(entry, str) for entry in eval_dataset)
            has_inline_examples = any(
                isinstance(entry, (EvaluationExample, Mapping))
                for entry in eval_dataset
            )
            if has_paths and has_inline_examples:
                raise ValidationError(
                    "eval_dataset list cannot mix file paths with inline example objects"
                )
            return

        raise ValidationError(
            "eval_dataset must be a string path, list of paths, inline example list, or Dataset object"
        )

    def _validate_objectives(self, objectives: list[str] | None) -> None:
        """Validate objectives parameter."""
        if objectives is None:
            return

        if not isinstance(objectives, list):
            raise ValidationError("objectives must be a list of strings")

        if not all(isinstance(obj, str) for obj in objectives):
            raise ValidationError("All objectives must be strings")

        # Check for common typos or invalid objective names
        valid_objectives = {
            "accuracy",
            "cost",
            "latency",
            "f1_score",
            "precision",
            "recall",
            "bleu",
            "rouge",
            "meteor",
            "bertscore",
            "semantic_similarity",
        }

        for obj in objectives:
            if (
                obj is not None
                and obj not in valid_objectives
                and not obj.startswith("custom_")
            ):
                logger.warning(
                    f"Objective '{obj}' not in common objectives list. "
                    f"Make sure it's implemented in your evaluator. "
                    f"Common objectives: {', '.join(sorted(valid_objectives))}"
                )

    def _validate_configuration_space(
        self, config_space: dict[str, Any] | None
    ) -> None:
        """Validate configuration space parameter."""
        if config_space is None:
            return

        if not isinstance(config_space, dict):
            raise ValidationError("configuration_space must be a dictionary")

        # Reject empty configuration space
        if len(config_space) == 0:
            raise ValidationError(
                "configuration_space cannot be empty. "
                "You must define at least one parameter to optimize. "
                "Example: {'model': ['gpt-3.5-turbo', 'gpt-4'], 'temperature': [0.3, 0.7]}"
            )

        for key, value in config_space.items():
            if not isinstance(key, str):
                raise ValidationError(
                    f"Configuration space keys must be strings, got {type(key)}"
                )

            # Validate value format with specific error messages
            error_msg = self._get_config_value_error(key, value)
            if error_msg:
                raise ValidationError(error_msg)

    def _validate_list_value(self, key: str, value: list[Any]) -> str | None:
        """Validate a list configuration value."""
        if len(value) == 0:
            return (
                f"Parameter '{key}' has an empty list. "
                f"Provide at least one value to try. "
                f"Example: ['{key}_value1', '{key}_value2']"
            )
        return None

    def _validate_tuple_value(self, key: str, value: tuple[Any, ...]) -> str | None:
        """Validate a tuple (range) configuration value."""
        if len(value) != 2:
            return (
                f"Parameter '{key}' has an invalid range tuple with {len(value)} elements. "
                f"Continuous ranges must have exactly 2 elements: (min, max). "
                f"Example: (0.0, 1.0)"
            )
        if not all(isinstance(v, (int, float)) for v in value):
            return (
                f"Parameter '{key}' has non-numeric range values: {value}. "
                f"Range tuples must contain numbers. "
                f"For string options, use a list instead: ['{value[0]}', '{value[1]}']"
            )
        low, high = value
        if low > high:
            return (
                f"Parameter '{key}' has invalid range ({low}, {high}) where min > max. "
                f"The first value must be less than the second. "
                f"Did you mean ({high}, {low})?"
            )
        if low == high:
            return (
                f"Parameter '{key}' has a point range ({low}, {high}) where min equals max. "
                f"A range with equal bounds has no values to search. "
                f"Use a single-element list [{low}] for a fixed value, "
                f"or provide a proper range like ({low}, {high + 0.5})"
            )
        return None

    def _validate_typed_dict_bounds(
        self, key: str, value: dict[str, Any], param_type: str
    ) -> str | None:
        """Validate bounds for typed numeric parameter definitions."""
        if "low" not in value or "high" not in value:
            return (
                f"Parameter '{key}' with type '{param_type}' must have "
                f"'low' and 'high' bounds."
            )
        low, high = value["low"], value["high"]
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            return (
                f"Parameter '{key}' has non-numeric bounds: low={low!r}, high={high!r}"
            )
        if low >= high:
            return (
                f"Parameter '{key}' has invalid bounds: "
                f"low ({low}) must be less than high ({high})"
            )
        return None

    def _validate_dict_value(self, key: str, value: dict[str, Any]) -> str | None:
        """Validate a dict configuration value (nested config or typed param)."""
        if len(value) == 0:
            return (
                f"Parameter '{key}' has an empty nested configuration. "
                f"Provide at least one sub-parameter."
            )
        # Accept typed parameter definitions (from Range/IntRange/LogRange)
        # Format: {"type": "float"|"int", "low": x, "high": y, ...}
        if "type" in value:
            param_type = value.get("type")
            valid_types = ("float", "int", "integer", "categorical", "fixed")
            if param_type not in valid_types:
                return (
                    f"Parameter '{key}' has unknown type: {param_type!r}. "
                    f"Valid types are: {', '.join(valid_types)}"
                )
            # Validate bounds for numeric types
            if param_type in ("float", "int", "integer"):
                return self._validate_typed_dict_bounds(key, value, param_type)
        return None

    def _get_config_value_error(self, key: str, value: Any) -> str | None:
        """Get specific error message for invalid config value, or None if valid."""
        if isinstance(value, list):
            return self._validate_list_value(key, value)

        if isinstance(value, tuple):
            return self._validate_tuple_value(key, value)

        if isinstance(value, dict):
            return self._validate_dict_value(key, value)

        # Not a valid type
        return (
            f"Parameter '{key}' has invalid type {type(value).__name__}: {value!r}. "
            f"Configuration values must be: "
            f"list (discrete values), tuple (continuous range), or dict (nested config). "
            f"Example: ['value1', 'value2'] or (0.0, 1.0)"
        )

    def _is_valid_config_value(self, value: Any) -> bool:
        """Check if a configuration space value is valid.

        Returns True if valid, or a string error message if invalid.
        For backward compatibility, also returns False for invalid values.
        """
        if isinstance(value, list):
            return len(value) > 0
        elif isinstance(value, tuple):
            if len(value) != 2:
                return False
            if not all(isinstance(v, (int, float)) for v in value):
                return False
            # Check that min < max (not min == max or min > max)
            low, high = value
            if low >= high:
                return False
            return True
        elif isinstance(value, dict):
            return len(value) > 0
        else:
            return False

    def _validate_constraints(
        self, constraints: list[Callable[..., Any]] | None
    ) -> None:
        """Validate constraints parameter."""
        if constraints is None:
            return

        if not isinstance(constraints, list):
            raise ValidationError("constraints must be a list of callable functions")

        for i, constraint in enumerate(constraints):
            if not callable(constraint):
                raise ValidationError(f"constraint at index {i} is not callable")

    def _normalize_injection_mode(
        self, injection_mode: str | InjectionMode
    ) -> InjectionMode:
        """Normalize injection mode to enum.

        Args:
            injection_mode: Either a string name or InjectionMode enum value.

        Returns:
            The normalized InjectionMode enum value.

        Note:
            Type signature guarantees input is str | InjectionMode,
            so after str check, it must be InjectionMode.
        """
        if isinstance(injection_mode, str):
            return InjectionMode(injection_mode)
        # Type narrowing: after str check, must be InjectionMode
        return injection_mode


def validate_optimize_parameters(**kwargs: Any) -> OptimizeParameters:
    """
    Convenience function to validate optimize decorator parameters.

    Args:
        **kwargs: All parameters passed to the optimize decorator

    Returns:
        Validated OptimizeParameters object

    Raises:
        ValidationError: If any parameter is invalid
    """
    unsupported = {
        key
        for key in ("auto_optimize", "trigger", "batch_size", "parallel_trials")
        if key in kwargs
    }
    if unsupported:
        joined = ", ".join(sorted(unsupported))
        raise ValidationError(
            f"The following parameters have been removed from @traigent.optimize: {joined}. "
            "Use parallel_config or ExecutionOptions instead."
        )

    param_fields = {f.name: f for f in fields(OptimizeParameters)}
    init_kwargs: dict[str, Any] = {}

    for field_name, field_info in param_fields.items():
        if field_name == "kwargs":
            continue

        if field_info.default is not MISSING:
            default = field_info.default
        elif field_info.default_factory is not MISSING:
            default = field_info.default_factory()
        else:
            default = None

        init_kwargs[field_name] = kwargs.pop(field_name, default)

    init_kwargs["kwargs"] = kwargs  # Remaining parameters

    return _PARAMETER_VALIDATOR.validate_parameters(OptimizeParameters(**init_kwargs))


_PARAMETER_VALIDATOR = ParameterValidator()
