"""Core orchestration components for Traigent SDK.

This package keeps exports lazy so importing ``traigent.core.constants`` does
not eagerly import the full optimization stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "OptimizationOrchestrator",
    "OptimizedFunction",
    "NAMESPACE_DELIMITER",
    "ParsedNamespace",
    "is_namespaced",
    "parse_namespaced_param",
    "parse_namespace",
    "create_namespaced_param",
    "extract_agents_from_config",
    "group_params_by_agent",
    "flatten_agent_config",
    "sanitize_metric_name",
    "extract_agent_metrics",
    "build_per_agent_objectives",
]

_EXPORT_MAP = {
    "OptimizationOrchestrator": (
        "traigent.core.orchestrator",
        "OptimizationOrchestrator",
    ),
    "OptimizedFunction": ("traigent.core.optimized_function", "OptimizedFunction"),
    "NAMESPACE_DELIMITER": ("traigent.core.namespace", "NAMESPACE_DELIMITER"),
    "ParsedNamespace": ("traigent.core.namespace", "ParsedNamespace"),
    "is_namespaced": ("traigent.core.namespace", "is_namespaced"),
    "parse_namespaced_param": ("traigent.core.namespace", "parse_namespaced_param"),
    "parse_namespace": ("traigent.core.namespace", "parse_namespace"),
    "create_namespaced_param": ("traigent.core.namespace", "create_namespaced_param"),
    "extract_agents_from_config": (
        "traigent.core.namespace",
        "extract_agents_from_config",
    ),
    "group_params_by_agent": ("traigent.core.namespace", "group_params_by_agent"),
    "flatten_agent_config": ("traigent.core.namespace", "flatten_agent_config"),
    "sanitize_metric_name": ("traigent.core.namespace", "sanitize_metric_name"),
    "extract_agent_metrics": ("traigent.core.namespace", "extract_agent_metrics"),
    "build_per_agent_objectives": (
        "traigent.core.namespace",
        "build_per_agent_objectives",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'traigent.core' has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
