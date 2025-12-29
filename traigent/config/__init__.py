"""Configuration management for Traigent SDK.

This module provides flexible configuration injection strategies for optimized functions.
"""

# Traceability: CONC-ConfigInjection CONC-Invocation FUNC-API-ENTRY FUNC-INVOKERS REQ-INJ-002 REQ-API-001 SYNC-OptimizationFlow CONC-Layer-Interface

from __future__ import annotations

from traigent.config.context import (
    TrialContext,
    config_context,
    get_config,
    get_trial_context,
    set_config,
)
from traigent.config.providers import (
    AttributeBasedProvider,
    ConfigurationProvider,
    ContextBasedProvider,
    ParameterBasedProvider,
    SeamlessParameterProvider,
    get_provider,
)
from traigent.config.types import TraigentConfig

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
    "AttributeBasedProvider",
    "SeamlessParameterProvider",
    "get_provider",
    # Types
    "TraigentConfig",
]
