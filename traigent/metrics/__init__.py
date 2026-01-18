"""Metric extension helpers."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Observability FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

# ALL imports are lazy to avoid circular dependency with traigent.core
# The circular import chain is: metrics/__init__ -> registry -> core/metric_registry
# -> core/__init__ -> optimized_function -> api.functions (circular)
_REGISTRY_MODULE = "traigent.metrics.registry"
_METRIC_REGISTRY_MODULE = "traigent.core.metric_registry"
_RAGAS_METRICS_MODULE = "traigent.metrics.ragas_metrics"

_AGENT_METRICS_MODULE = "traigent.metrics.agent_metrics"

_LAZY_IMPORTS = {
    # From registry module
    "clone_registry": _REGISTRY_MODULE,
    "get_registry": _REGISTRY_MODULE,
    "register_metric": _REGISTRY_MODULE,
    "register_metrics": _REGISTRY_MODULE,
    "reset_registry": _REGISTRY_MODULE,
    # From core.metric_registry
    "AggregatorType": _METRIC_REGISTRY_MODULE,
    "MetricSpec": _METRIC_REGISTRY_MODULE,
    # From ragas_metrics
    "POPULAR_RAGAS_METRICS": _RAGAS_METRICS_MODULE,
    "RagasConfig": _RAGAS_METRICS_MODULE,
    "RagasConfigurationError": _RAGAS_METRICS_MODULE,
    "compute_ragas_metrics": _RAGAS_METRICS_MODULE,
    "configure_ragas_defaults": _RAGAS_METRICS_MODULE,
    # From agent_metrics
    "REFERENCE_FREE_RAGAS_METRICS": _AGENT_METRICS_MODULE,
    "REFERENCE_REQUIRED_RAGAS_METRICS": _AGENT_METRICS_MODULE,
    "ALL_RAGAS_METRICS": _AGENT_METRICS_MODULE,
    "AGENT_QUALITY_METRICS": _AGENT_METRICS_MODULE,
    "AGENT_PERFORMANCE_METRICS": _AGENT_METRICS_MODULE,
    "AgentMetricsSummary": _AGENT_METRICS_MODULE,
    "MultiAgentMetricsSummary": _AGENT_METRICS_MODULE,
    "compute_per_agent_metrics": _AGENT_METRICS_MODULE,
    "aggregate_agent_metrics": _AGENT_METRICS_MODULE,
    "get_reference_free_metrics": _AGENT_METRICS_MODULE,
    "get_metrics_for_available_data": _AGENT_METRICS_MODULE,
    "build_agent_objectives": _AGENT_METRICS_MODULE,
    "extract_namespaced_config_for_agent": _AGENT_METRICS_MODULE,
    "validate_agent_metrics": _AGENT_METRICS_MODULE,
}


def __getattr__(name: str):
    """Lazy import to avoid circular imports with traigent.core."""
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_name)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Registry
    "AggregatorType",
    "clone_registry",
    "get_registry",
    "MetricSpec",
    "register_metric",
    "register_metrics",
    "reset_registry",
    # RAGAS
    "compute_ragas_metrics",
    "configure_ragas_defaults",
    "POPULAR_RAGAS_METRICS",
    "RagasConfig",
    "RagasConfigurationError",
    # Agent metrics (Phase 4)
    "REFERENCE_FREE_RAGAS_METRICS",
    "REFERENCE_REQUIRED_RAGAS_METRICS",
    "ALL_RAGAS_METRICS",
    "AGENT_QUALITY_METRICS",
    "AGENT_PERFORMANCE_METRICS",
    "AgentMetricsSummary",
    "MultiAgentMetricsSummary",
    "compute_per_agent_metrics",
    "aggregate_agent_metrics",
    "get_reference_free_metrics",
    "get_metrics_for_available_data",
    "build_agent_objectives",
    "extract_namespaced_config_for_agent",
    "validate_agent_metrics",
]
