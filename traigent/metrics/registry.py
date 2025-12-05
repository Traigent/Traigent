"""Global metric registry helpers for orchestrator aggregation."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Observability FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

import threading
from typing import Iterable

from traigent.core.metric_registry import MetricRegistry, MetricSpec

_GLOBAL_REGISTRY: MetricRegistry = MetricRegistry.default()
_registry_lock = threading.Lock()


def get_registry() -> MetricRegistry:
    """Return the mutable global metric registry."""
    return _GLOBAL_REGISTRY


def clone_registry() -> MetricRegistry:
    """Return a cloned registry for use in isolated contexts."""
    with _registry_lock:
        return _GLOBAL_REGISTRY.clone()


def register_metric(spec: MetricSpec) -> None:
    """Register or override a metric specification on the global registry."""
    with _registry_lock:
        _GLOBAL_REGISTRY.register(spec)


def register_metrics(*specs: MetricSpec) -> None:
    """Register multiple metric specifications on the global registry."""
    with _registry_lock:
        _GLOBAL_REGISTRY.register_many(*specs)


def reset_registry(*, specs: Iterable[MetricSpec] | None = None) -> None:
    """Reset the global registry to defaults, optionally seeding custom specs.

    Intended primarily for tests to ensure isolated expectations.
    """
    global _GLOBAL_REGISTRY
    with _registry_lock:
        _GLOBAL_REGISTRY = MetricRegistry.default()
        if specs:
            _GLOBAL_REGISTRY.register_many(*specs)


__all__ = [
    "clone_registry",
    "get_registry",
    "register_metric",
    "register_metrics",
    "reset_registry",
]
