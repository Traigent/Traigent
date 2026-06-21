"""Configuration management for Traigent SDK.

This module provides flexible configuration injection strategies for optimized functions.
"""

# Traceability: CONC-ConfigInjection CONC-Invocation FUNC-API-ENTRY FUNC-INVOKERS REQ-INJ-002 REQ-API-001 SYNC-OptimizationFlow CONC-Layer-Interface

from __future__ import annotations

from importlib import import_module
from typing import Any
import warnings

from traigent.config.context import (
    TrialContext,
    config_context,
    get_config,
    get_trial_context,
    set_config,
)
from traigent.config.providers import (
    ConfigurationProvider,
    ContextBasedProvider,
    ParameterBasedProvider,
    SeamlessParameterProvider,
    get_provider,
)
from traigent.config.types import TraigentConfig

_DEPRECATED_EXPORT_MAP = {
    "ExecutionIntent": ("traigent.config.types", "ExecutionIntent"),
    "ExecutionMode": ("traigent.config.types", "ExecutionMode"),
    "ResolvedExecutionPolicy": ("traigent.config.types", "ResolvedExecutionPolicy"),
}

__all__ = [
    # Context management
    "config_context",
    "get_config",
    "set_config",
    "get_trial_context",
    "TrialContext",
    # Providers
    "ConfigurationProvider",
    "ContextBasedProvider",
    "ParameterBasedProvider",
    "SeamlessParameterProvider",
    "get_provider",
    # Types
    "TraigentConfig",
]


def __getattr__(name: str) -> Any:
    if name not in _DEPRECATED_EXPORT_MAP:
        raise AttributeError(f"module 'traigent.config' has no attribute {name!r}")
    module_name, attr_name = _DEPRECATED_EXPORT_MAP[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    warnings.warn(
        f"traigent.config.{name} is a deprecated compatibility alias and is "
        "no longer part of the public config export surface.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
