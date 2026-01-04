"""Plugin system for Traigent SDK.

This module provides the unified plugin discovery and registration system.

Feature Flags:
    Use FEATURE_* constants to check for optional plugin capabilities:

    >>> from traigent.plugins import get_plugin_registry, FEATURE_PARALLEL
    >>> registry = get_plugin_registry()
    >>> if registry.has_feature(FEATURE_PARALLEL):
    ...     # Use parallel features
    ...     pass

Plugin Types:
    - TraigentPlugin: Base class for all plugins
    - OptimizerPlugin: Adds new optimization algorithms
    - EvaluatorPlugin: Adds new evaluation methods
    - MetricPlugin: Adds custom metrics
    - IntegrationPlugin: Adds framework integrations
    - FeaturePlugin: Adds optional feature capabilities
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

from .registry import (  # Plugin base classes; Feature flag constants; Registry functions
    FEATURE_ADVANCED_ALGORITHMS,
    FEATURE_ANALYTICS,
    FEATURE_CLOUD,
    FEATURE_EVALUATION,
    FEATURE_EXPERIMENT_TRACKING,
    FEATURE_HOOKS,
    FEATURE_INTEGRATIONS,
    FEATURE_MULTI_OBJECTIVE,
    FEATURE_PARALLEL,
    FEATURE_SEAMLESS,
    FEATURE_SECURITY,
    FEATURE_TRACING,
    FEATURE_TVL,
    FEATURE_UI,
    EvaluatorPlugin,
    FeaturePlugin,
    IntegrationPlugin,
    MetricPlugin,
    OptimizerPlugin,
    PluginRegistry,
    TraigentPlugin,
    discover_plugins,
    get_available_evaluators,
    get_available_integrations,
    get_available_metrics,
    get_available_optimizers,
    get_plugin_registry,
    has_feature,
    list_available_plugins,
    load_plugin,
    register_plugin,
)

__all__ = [
    # Plugin base classes
    "TraigentPlugin",
    "OptimizerPlugin",
    "EvaluatorPlugin",
    "MetricPlugin",
    "IntegrationPlugin",
    "FeaturePlugin",
    "PluginRegistry",
    # Feature flag constants
    "FEATURE_PARALLEL",
    "FEATURE_MULTI_OBJECTIVE",
    "FEATURE_SEAMLESS",
    "FEATURE_CLOUD",
    "FEATURE_ADVANCED_ALGORITHMS",
    "FEATURE_TVL",
    "FEATURE_TRACING",
    "FEATURE_ANALYTICS",
    "FEATURE_INTEGRATIONS",
    "FEATURE_SECURITY",
    "FEATURE_HOOKS",
    "FEATURE_EVALUATION",
    "FEATURE_UI",
    "FEATURE_EXPERIMENT_TRACKING",
    # Registry functions
    "get_plugin_registry",
    "register_plugin",
    "load_plugin",
    "discover_plugins",
    "list_available_plugins",
    "get_available_optimizers",
    "get_available_evaluators",
    "get_available_metrics",
    "get_available_integrations",
    "has_feature",
]
