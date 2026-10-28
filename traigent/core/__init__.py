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

_TRAIGENT_CORE_NAMESPACE = "traigent.core.namespace"

_EXPORT_MAP = {
    "OptimizationOrchestrator": (
        "traigent.core.orchestrator",
        "OptimizationOrchestrator",
    ),
    "OptimizedFunction": ("traigent.core.optimized_function", "OptimizedFunction"),
    "NAMESPACE_DELIMITER": (_TRAIGENT_CORE_NAMESPACE, "NAMESPACE_DELIMITER"),
    "ParsedNamespace": (_TRAIGENT_CORE_NAMESPACE, "ParsedNamespace"),
    "is_namespaced": (_TRAIGENT_CORE_NAMESPACE, "is_namespaced"),
    "parse_namespaced_param": (_TRAIGENT_CORE_NAMESPACE, "parse_namespaced_param"),
    "parse_namespace": (_TRAIGENT_CORE_NAMESPACE, "parse_namespace"),
    "create_namespaced_param": (_TRAIGENT_CORE_NAMESPACE, "create_namespaced_param"),
    "extract_agents_from_config": (
        _TRAIGENT_CORE_NAMESPACE,
        "extract_agents_from_config",
    ),
    "group_params_by_agent": (_TRAIGENT_CORE_NAMESPACE, "group_params_by_agent"),
    "flatten_agent_config": (_TRAIGENT_CORE_NAMESPACE, "flatten_agent_config"),
    "sanitize_metric_name": (_TRAIGENT_CORE_NAMESPACE, "sanitize_metric_name"),
    "extract_agent_metrics": (_TRAIGENT_CORE_NAMESPACE, "extract_agent_metrics"),
    "build_per_agent_objectives": (
        _TRAIGENT_CORE_NAMESPACE,
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
