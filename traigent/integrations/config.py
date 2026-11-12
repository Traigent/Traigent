"""Configuration for the integration system.

This module defines configuration options and constraints for framework integrations.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility CONC-Quality-Maintainability FUNC-INTEGRATIONS FUNC-INVOKERS REQ-INT-008 REQ-INJ-002 SYNC-IntegrationHook

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from ..utils.exceptions import ValidationError as ValidationException
from ..utils.logging import get_logger
from ..utils.validation import CoreValidators, validate_or_raise

logger = get_logger(__name__)

__all__ = (
    "IntegrationConfig",
    "ParameterConstraints",
    "FrameworkConstraints",
    "configure_integrations",
    "integration_config",
    "get_integration_config",
)


@dataclass
class IntegrationConfig:
    """Configuration for integration behavior."""

    # Discovery settings
    auto_discover: bool = True
    discovery_cache_ttl: int = 3600  # seconds
    cache_discovered_classes: bool = True

    # Override behavior
    strict_mode: bool = False  # Fail on unknown parameters
    fuzzy_matching_enabled: bool = True
    fuzzy_matching_threshold: float = 0.8

    # Validation
    validate_types: bool = True
    validate_values: bool = True
    auto_convert_types: bool = True

    # Compatibility
    version_check: bool = True
    warn_on_deprecated: bool = True
    auto_migrate_parameters: bool = True

    # Performance
    max_fallback_attempts: int = 4
    log_override_details: bool = False

    # Safety
    allow_unknown_parameters: bool = False
    sanitize_parameters: bool = True


@dataclass
class ParameterConstraints:
    """Constraints for a specific parameter."""

    type: type | None = None
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: set[Any] | None = None
    required: bool = False
    deprecated: bool = False
    deprecated_message: str | None = None
    aliases: list[str] = field(default_factory=list)


class FrameworkConstraints:
    """Known constraints for popular frameworks."""

    # Common LLM parameter constraints
    COMMON_CONSTRAINTS = {
        "temperature": ParameterConstraints(
            type=float,
            min_value=0.0,
            max_value=2.0,
        ),
        "top_p": ParameterConstraints(
            type=float,
            min_value=0.0,
            max_value=1.0,
        ),
        "top_k": ParameterConstraints(
            type=int,
            min_value=1,
        ),
        "max_tokens": ParameterConstraints(
            type=int,
            min_value=1,
            aliases=["max_length", "max_new_tokens", "max_tokens_to_sample"],
        ),
        "frequency_penalty": ParameterConstraints(
            type=float,
            min_value=-2.0,
            max_value=2.0,
        ),
        "presence_penalty": ParameterConstraints(
            type=float,
            min_value=-2.0,
            max_value=2.0,
        ),
        "n": ParameterConstraints(
            type=int,
            min_value=1,
            aliases=["num_completions", "best_of"],
        ),
        "seed": ParameterConstraints(
            type=int,
            aliases=["random_seed"],
        ),
    }

    # OpenAI-specific constraints
    # Note: Model validation uses dynamic discovery instead of hardcoded allowed_values.
    # The model_discovery module handles SDK-based discovery + config fallback + pattern.
    OPENAI_CONSTRAINTS = {
        **COMMON_CONSTRAINTS,
        "model": ParameterConstraints(
            type=str,
            # No allowed_values - validation delegated to model_discovery module
        ),
        "logit_bias": ParameterConstraints(
            type=dict,
        ),
        "response_format": ParameterConstraints(
            type=dict,
        ),
    }

    # Anthropic-specific constraints
    # Note: Model validation uses dynamic discovery instead of hardcoded allowed_values.
    ANTHROPIC_CONSTRAINTS = {
        **COMMON_CONSTRAINTS,
        "model": ParameterConstraints(
            type=str,
            # No allowed_values - validation delegated to model_discovery module
        ),
        "max_tokens": ParameterConstraints(
            type=int,
            min_value=1,
            max_value=200000,  # Claude 3 max tokens
            aliases=["max_tokens_to_sample"],
        ),
    }

    @classmethod
    def get_constraints_for_framework(
        cls, framework: str
    ) -> dict[str, ParameterConstraints]:
        """Get parameter constraints for a specific framework.

        Args:
            framework: Framework name (e.g., "openai", "anthropic")

        Returns:
            Dictionary of parameter constraints
        """
        framework_lower = framework.lower()

        if framework_lower == "openai":
            return cls.OPENAI_CONSTRAINTS
        elif framework_lower == "anthropic":
            return cls.ANTHROPIC_CONSTRAINTS
        else:
            # Return common constraints for unknown frameworks
            return cls.COMMON_CONSTRAINTS

    @staticmethod
    def _append_value_issues(
        constraint: ParameterConstraints,
        param_name: str,
        param_value: Any,
        issues: list[str],
    ) -> None:
        """Append range and allowed-value issues in their validation order."""
        if constraint.min_value is not None and param_value < constraint.min_value:
            issues.append(
                f"Parameter '{param_name}' value {param_value} "
                f"is below minimum {constraint.min_value}"
            )

        if constraint.max_value is not None and param_value > constraint.max_value:
            issues.append(
                f"Parameter '{param_name}' value {param_value} "
                f"is above maximum {constraint.max_value}"
            )

        if constraint.allowed_values and param_value not in constraint.allowed_values:
            issues.append(
                f"Parameter '{param_name}' value {param_value} not in allowed values"
            )

    @classmethod
    def validate_parameter(
        cls, framework: str, param_name: str, param_value: Any
    ) -> list[str]:
        """Validate a parameter value against known constraints.

        Args:
            framework: Framework name
            param_name: Parameter name
            param_value: Parameter value

        Returns:
            List of validation issues (empty if valid)
        """
        constraints = cls.get_constraints_for_framework(framework)
        if param_name not in constraints:
            # Check aliases
            for name, constraint in constraints.items():
                if param_name in constraint.aliases:
                    param_name = name
                    break
            else:
                return []  # Unknown parameter, no constraints

        constraint = constraints[param_name]
        issues = []

        # Type check
        if constraint.type and not isinstance(param_value, constraint.type):
            issues.append(
                f"Parameter '{param_name}' expected type {constraint.type.__name__}, "
                f"got {type(param_value).__name__}"
            )
            # Return early if wrong type to avoid comparison errors
            return issues

        # Value range and allowed-value checks (only if type is correct)
        cls._append_value_issues(constraint, param_name, param_value, issues)

        # Deprecation warning
        if constraint.deprecated:
            message = (
                constraint.deprecated_message
                or f"Parameter '{param_name}' is deprecated"
            )
            logger.warning(message)

        return issues


# Global configuration instance
integration_config = IntegrationConfig()


def get_integration_config() -> IntegrationConfig:
    """Return the active integration configuration."""
    return integration_config


_BOOL_OPTIONS = {
    "auto_discover",
    "cache_discovered_classes",
    "strict_mode",
    "fuzzy_matching_enabled",
    "validate_types",
    "validate_values",
    "auto_convert_types",
    "version_check",
    "warn_on_deprecated",
    "auto_migrate_parameters",
    "log_override_details",
    "allow_unknown_parameters",
    "sanitize_parameters",
}


def _validate_config_option(key: str, value: Any) -> None:
    if key in _BOOL_OPTIONS:
        validate_or_raise(CoreValidators.validate_type(value, bool, key))
        return

    if key == "discovery_cache_ttl":
        validate_or_raise(CoreValidators.validate_positive_int(value, key))
        return

    if key == "fuzzy_matching_threshold":
        validate_or_raise(CoreValidators.validate_number(value, key, 0.0, 1.0))
        return

    if key == "max_fallback_attempts":
        validate_or_raise(CoreValidators.validate_positive_int(value, key))
        return

    # For any other field, ensure types align with dataclass annotations if available
    expected_type = IntegrationConfig.__annotations__.get(key)
    if expected_type and expected_type is not Any:
        validate_or_raise(CoreValidators.validate_type(value, expected_type, key))


_WARNED_UNUSED_OPTIONS: set[str] = set()


def _warn_if_option_unused(key: str) -> None:
    """Warn once per process that an explicitly supplied option is not read.

    No integration module consults any IntegrationConfig field yet, so the
    value is stored but has no effect. The warning fires whatever the value,
    so setting a safety option to its default does not look enforced.
    """
    if key in _WARNED_UNUSED_OPTIONS:
        return
    _WARNED_UNUSED_OPTIONS.add(key)
    warnings.warn(
        f"configure_integrations: option '{key}' is stored but not yet read by "
        "any integration, so setting it has no effect.",
        UserWarning,
        stacklevel=3,
    )


def configure_integrations(**kwargs) -> IntegrationConfig:
    """Configure integration behavior.

    Args:
        **kwargs: Configuration options to set

    Returns:
        Updated IntegrationConfig instance.
    """
    for key, value in kwargs.items():
        if not hasattr(integration_config, key):
            logger.warning(f"Unknown configuration option: {key}")
            continue

        try:
            _validate_config_option(key, value)
        except ValidationException as exc:
            raise ValidationException(f"Invalid value for '{key}': {exc}") from exc

        _warn_if_option_unused(key)
        setattr(integration_config, key, value)

    return integration_config
