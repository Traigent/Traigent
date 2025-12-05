"""Metric extension helpers."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Observability FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

# ALL imports are lazy to avoid circular dependency with traigent.core
# The circular import chain is: metrics/__init__ -> registry -> core/metric_registry
# -> core/__init__ -> optimized_function -> api.functions (circular)
_REGISTRY_MODULE = "traigent.metrics.registry"
_METRIC_REGISTRY_MODULE = "traigent.core.metric_registry"
_RAGAS_METRICS_MODULE = "traigent.metrics.ragas_metrics"

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
    "AggregatorType",
    "clone_registry",
    "compute_ragas_metrics",
    "configure_ragas_defaults",
    "get_registry",
    "MetricSpec",
    "POPULAR_RAGAS_METRICS",
    "RagasConfig",
    "RagasConfigurationError",
    "register_metric",
    "register_metrics",
    "reset_registry",
]
