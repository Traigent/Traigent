"""
Configuration builder for the @traigent.optimize decorator.

This module extracts configuration building logic from the main decorator
to reduce complexity and improve maintainability.
Traceability: CONC-Layer-API CONC-Quality-Maintainability CONC-Quality-Usability FUNC-API-ENTRY REQ-API-001 SYNC-OptimizationFlow
"""

import logging
from typing import Any

from traigent.api.parameter_validator import OptimizeParameters
from traigent.config.parallel import coerce_parallel_config
from traigent.config.types import InjectionMode, resolve_execution_policy
from traigent.core.objectives import normalize_objectives
from traigent.utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Global configuration store
_GLOBAL_CONFIG: dict[str, Any] = {}


__all__ = [
    "ConfigurationBuilder",
    "build_optimize_configuration",
    "update_global_config",
    "get_global_config",
    "clear_global_config",
]


class ConfigurationBuilder:
    """Builds configuration for OptimizedFunction from validated parameters."""

    def __init__(self, global_config: dict[str, Any] | None = None) -> None:
        """Initialize with optional global configuration."""
        self.global_config = global_config or _GLOBAL_CONFIG

    def build_configuration(
        self, params: OptimizeParameters, original_execution_mode: str | None = None
    ) -> dict[str, Any]:
        """
        Build final configuration for OptimizedFunction.

        Args:
            params: Validated parameters
            original_execution_mode: Original execution_mode parameter value

        Returns:
            Configuration dictionary for OptimizedFunction
        """
        execution_policy = params.execution_policy or self._resolve_execution_policy(
            params, original_execution_mode
        )
        actual_execution_mode = execution_policy.legacy_execution_mode.value

        # Resolve injection mode with validation
        # At this point, injection_mode should be normalized to InjectionMode enum in validation
        # But we need to handle the Union type properly for mypy
        injection_mode_enum: InjectionMode
        if isinstance(params.injection_mode, str):
            injection_mode_enum = InjectionMode(params.injection_mode)
        elif isinstance(params.injection_mode, InjectionMode):
            injection_mode_enum = params.injection_mode
        else:
            raise ConfigurationError(
                "injection_mode must be a str or InjectionMode enumeration. "
                "Make sure ParameterValidator.normalize_injection_mode() "
                "has been applied before building the configuration."
            )

        actual_injection_mode = self._resolve_injection_mode(
            injection_mode_enum, params.config_param
        )

        resolved_schema = normalize_objectives(params.objectives)

        extra_kwargs = params.kwargs.copy()
        if "parallel_config" in extra_kwargs:
            extra_kwargs["parallel_config"] = coerce_parallel_config(
                extra_kwargs["parallel_config"]
            )

        config = {
            "func": None,  # Will be set by decorator
            "eval_dataset": params.eval_dataset,
            "objectives": resolved_schema,
            "configuration_space": params.configuration_space,
            "default_config": params.default_config,
            "constraints": params.constraints,
            "injection_mode": actual_injection_mode,
            "config_param": params.config_param,
            "auto_override_frameworks": params.auto_override_frameworks,
            "framework_targets": params.framework_targets,
            "algorithm": execution_policy.algorithm,
            "offline": execution_policy.offline,
            "execution_policy": execution_policy,
            "execution_mode": actual_execution_mode,
            "local_storage_path": params.local_storage_path,
            "minimal_logging": params.minimal_logging,
            **extra_kwargs,
        }

        return config

    def _resolve_execution_policy(
        self, params: OptimizeParameters, original_execution_mode: str | None
    ):
        """Resolve execution policy with global legacy-mode fallback."""
        legacy_execution_mode = params.execution_mode
        if legacy_execution_mode is None and original_execution_mode:
            legacy_execution_mode = original_execution_mode
        if legacy_execution_mode is None and "execution_mode" in self.global_config:
            legacy_execution_mode = self.global_config["execution_mode"]

        return resolve_execution_policy(
            algorithm=params.algorithm,
            offline=params.offline,
            execution_mode=legacy_execution_mode,
            privacy_enabled=params.privacy_enabled,
            source_hint="config_builder",
        )

    def _resolve_injection_mode(
        self, injection_mode: InjectionMode, config_param: str | None
    ) -> InjectionMode:
        """
        Resolve and validate injection mode configuration.

        Args:
            injection_mode: Injection mode enum
            config_param: Config parameter name for parameter injection

        Returns:
            Resolved injection mode

        Raises:
            ConfigurationError: If injection mode configuration is invalid
        """
        if injection_mode == InjectionMode.PARAMETER and config_param is None:
            raise ConfigurationError(
                "config_param must be specified when injection_mode='parameter'"
            )

        if injection_mode != InjectionMode.PARAMETER and config_param is not None:
            logger.warning(
                f"config_param '{config_param}' specified but injection_mode is "
                f"'{injection_mode.value}'. config_param will be ignored."
            )

        return injection_mode

    def update_global_config(self, **config_updates: Any) -> None:
        """Update global configuration."""
        self.global_config.update(config_updates)

    def get_global_config(self) -> dict[str, Any]:
        """Get current global configuration."""
        return self.global_config.copy()

    def clear_global_config(self) -> None:
        """Clear global configuration."""
        self.global_config.clear()


# Module-level functions for backward compatibility
def build_optimize_configuration(
    params: OptimizeParameters, original_execution_mode: str | None = None
) -> dict[str, Any]:
    """
    Build configuration for OptimizedFunction.

    Args:
        params: Validated parameters
        original_execution_mode: Original execution_mode parameter value

    Returns:
        Configuration dictionary
    """
    builder = ConfigurationBuilder()
    return builder.build_configuration(params, original_execution_mode)


def update_global_config(**config_updates: Any) -> None:
    """Update global configuration."""
    _GLOBAL_CONFIG.update(config_updates)


def get_global_config() -> dict[str, Any]:
    """Get current global configuration."""
    return _GLOBAL_CONFIG.copy()


def clear_global_config() -> None:
    """Clear global configuration."""
    _GLOBAL_CONFIG.clear()
