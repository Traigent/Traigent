"""Core orchestration components for Traigent SDK."""

# Traceability: CONC-Layer-Core CONC-Quality-Usability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from traigent.core.namespace import (
    NAMESPACE_DELIMITER,
    ParsedNamespace,
    build_per_agent_objectives,
    create_namespaced_param,
    extract_agent_metrics,
    extract_agents_from_config,
    flatten_agent_config,
    group_params_by_agent,
    is_namespaced,
    parse_namespace,
    parse_namespaced_param,
    sanitize_metric_name,
)
from traigent.core.optimized_function import OptimizedFunction
from traigent.core.orchestrator import OptimizationOrchestrator

__all__ = [
    "OptimizationOrchestrator",
    "OptimizedFunction",
    # Namespace utilities for multi-agent optimization
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
