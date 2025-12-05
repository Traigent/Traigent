"""Parameter validation for framework integrations.

This module provides utilities to validate and sanitize override parameters,
ensuring type safety and compatibility.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping, Sequence
from types import UnionType
from typing import Any, Literal, Union, cast, get_args, get_origin

from ...utils.logging import get_logger

logger = get_logger(__name__)


class ParameterValidator:
    """Validate and sanitize override parameters."""

    @staticmethod
    def validate_parameter_types(
        params: dict[str, Any], signature: inspect.Signature
    ) -> tuple[dict[str, Any], list[str]]:
        """Validate parameter types match expected signature.

        Args:
            params: Parameters to validate
            signature: Expected function signature

        Returns:
            Tuple of (validated parameters, list of issues)
        """
        validated = {}
        issues = []

        for param_name, param_value in params.items():
            if param_name not in signature.parameters:
                continue

            param_spec = signature.parameters[param_name]
            expected_type = param_spec.annotation

            if expected_type != inspect.Parameter.empty:
                if not ParameterValidator._check_type_compatibility(
                    param_value, expected_type
                ):
                    issues.append(
                        f"Parameter '{param_name}' expected type {expected_type}, "
                        f"got {type(param_value)}"
                    )
                    # Try to convert if possible
                    converted = ParameterValidator._try_convert_type(
                        param_value, expected_type
                    )
                    if converted is not None:
                        validated[param_name] = converted
                        issues[-1] += " (converted)"
                    else:
                        continue
                else:
                    validated[param_name] = param_value
            else:
                validated[param_name] = param_value

        return validated, issues

    @staticmethod
    def _check_union_compatibility(value: Any, args: tuple[Any, ...]) -> bool:
        """Check if value matches any type in a Union."""
        return any(
            ParameterValidator._check_type_compatibility(value, arg) for arg in args
        )

    @staticmethod
    def _check_collection_compatibility(
        value: Any, origin: type, args: tuple[Any, ...]
    ) -> bool:
        """Check compatibility for list, set, frozenset."""
        if not isinstance(value, origin):
            return False
        if not args:
            return True
        # Cast to Iterable since we've verified it's a list/set/frozenset
        items = cast(Iterable[Any], value)
        return all(
            ParameterValidator._check_type_compatibility(item, args[0])
            for item in items
        )

    @staticmethod
    def _check_tuple_compatibility(value: Any, args: tuple[Any, ...]) -> bool:
        """Check compatibility for tuple types."""
        if not isinstance(value, tuple):
            return False
        if not args:
            return True
        if len(args) == 2 and args[1] is Ellipsis:
            return all(
                ParameterValidator._check_type_compatibility(item, args[0])
                for item in value
            )
        if len(args) != len(value):
            return False
        return all(
            ParameterValidator._check_type_compatibility(item, subtype)
            for item, subtype in zip(value, args)
        )

    @staticmethod
    def _check_mapping_compatibility(value: Any, args: tuple[Any, ...]) -> bool:
        """Check compatibility for dict/Mapping types."""
        if not isinstance(value, Mapping):
            return False
        if len(args) != 2:
            return True
        key_type, val_type = args
        return all(
            ParameterValidator._check_type_compatibility(key, key_type)
            and ParameterValidator._check_type_compatibility(val, val_type)
            for key, val in value.items()
        )

    @staticmethod
    def _check_sequence_compatibility(value: Any, args: tuple[Any, ...]) -> bool:
        """Check compatibility for Sequence types."""
        if not isinstance(value, Sequence):
            return False
        if not args:
            return True
        return all(
            ParameterValidator._check_type_compatibility(item, args[0])
            for item in value
        )

    @staticmethod
    def _check_type_compatibility(value: Any, expected_type: type | Any) -> bool:
        """Check if a value is compatible with expected type."""
        if expected_type is Any:
            return True

        origin = get_origin(expected_type)
        args = get_args(expected_type)

        if origin in {Union, UnionType}:
            return ParameterValidator._check_union_compatibility(value, args)

        if origin in {list, set, frozenset}:
            return ParameterValidator._check_collection_compatibility(
                value, origin, args
            )

        if origin is tuple:
            return ParameterValidator._check_tuple_compatibility(value, args)

        if origin in {dict, Mapping}:
            return ParameterValidator._check_mapping_compatibility(value, args)

        if origin in {Sequence}:
            return ParameterValidator._check_sequence_compatibility(value, args)

        if origin is Literal:
            return value in args

        if origin is not None:
            try:
                return isinstance(value, origin)
            except TypeError:
                return False

        if isinstance(expected_type, tuple):
            return isinstance(value, expected_type)

        try:
            return isinstance(value, expected_type)
        except TypeError:
            return False

    @staticmethod
    def _try_convert_type(value: Any, expected_type: type) -> Any | None:
        """Try to convert value to expected type."""
        try:
            # Handle common conversions
            if expected_type is str:
                return str(value)
            elif expected_type is int:
                return int(value)
            elif expected_type is float:
                return float(value)
            elif expected_type is bool:
                return bool(value)
            else:
                return None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def validate_parameter_values(
        params: dict[str, Any], constraints: dict[str, dict[str, Any]]
    ) -> list[str]:
        """Validate parameter values against known constraints.

        Args:
            params: Parameters to validate
            constraints: Dictionary of parameter constraints

        Returns:
            List of validation issues
        """
        issues = []

        for param_name, param_value in params.items():
            if param_name not in constraints:
                continue

            constraint = constraints[param_name]

            # Check min/max values
            if "min" in constraint and param_value < constraint["min"]:
                issues.append(
                    f"Parameter '{param_name}' value {param_value} "
                    f"is below minimum {constraint['min']}"
                )

            if "max" in constraint and param_value > constraint["max"]:
                issues.append(
                    f"Parameter '{param_name}' value {param_value} "
                    f"is above maximum {constraint['max']}"
                )

            # Check allowed values
            if "allowed" in constraint and param_value not in constraint["allowed"]:
                issues.append(
                    f"Parameter '{param_name}' value {param_value} "
                    f"not in allowed values: {constraint['allowed']}"
                )

        return issues

    @staticmethod
    def sanitize_parameters(
        params: dict[str, Any], target_class: type
    ) -> dict[str, Any]:
        """Remove or transform incompatible parameters.

        Args:
            params: Parameters to sanitize
            target_class: Target class to validate against

        Returns:
            Sanitized parameters
        """
        try:
            # Get signature directly from the class constructor
            sig = inspect.signature(target_class)
        except Exception:
            # If we can't get signature, return params as-is
            return params

        sanitized = {}

        for param_name, param_value in params.items():
            # Skip if parameter doesn't exist in target
            if param_name not in sig.parameters:
                logger.debug(
                    f"Skipping unknown parameter '{param_name}' for {target_class}"
                )
                continue

            # Skip None values if parameter has a default
            param_spec = sig.parameters[param_name]
            if param_value is None and param_spec.default != inspect.Parameter.empty:
                continue

            sanitized[param_name] = param_value

        return sanitized

    @staticmethod
    def get_common_constraints() -> dict[str, dict[str, Any]]:
        """Get common parameter constraints for LLM parameters."""
        return {
            "temperature": {
                "min": 0.0,
                "max": 2.0,
                "type": float,
            },
            "top_p": {
                "min": 0.0,
                "max": 1.0,
                "type": float,
            },
            "top_k": {
                "min": 1,
                "type": int,
            },
            "max_tokens": {
                "min": 1,
                "type": int,
            },
            "frequency_penalty": {
                "min": -2.0,
                "max": 2.0,
                "type": float,
            },
            "presence_penalty": {
                "min": -2.0,
                "max": 2.0,
                "type": float,
            },
            "n": {
                "min": 1,
                "type": int,
            },
        }
