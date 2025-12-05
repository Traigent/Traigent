"""Plugin system for TraiGent SDK."""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

from .registry import (
    EvaluatorPlugin,
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
    list_available_plugins,
    load_plugin,
    register_plugin,
)

__all__ = [
    "TraigentPlugin",
    "OptimizerPlugin",
    "EvaluatorPlugin",
    "MetricPlugin",
    "IntegrationPlugin",
    "PluginRegistry",
    "get_plugin_registry",
    "register_plugin",
    "load_plugin",
    "discover_plugins",
    "list_available_plugins",
    "get_available_optimizers",
    "get_available_evaluators",
    "get_available_metrics",
    "get_available_integrations",
]
