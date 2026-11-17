"""Observability and monitoring integrations for Traigent."""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability CONC-Quality-Compatibility FUNC-INTEGRATIONS FUNC-ANALYTICS REQ-INT-008 REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import importlib
import threading
from typing import TYPE_CHECKING, Any

# Workflow traces integration
from traigent.integrations.observability.workflow_traces import (
    OTEL_AVAILABLE,
    SpanPayload,
    SpanStatus,
    SpanType,
    TraceIngestionRequest,
    TraceIngestionResponse,
    TraigentSpanExporter,
    WorkflowEdge,
    WorkflowGraphPayload,
    WorkflowLoop,
    WorkflowNode,
    WorkflowTracesClient,
    WorkflowTracesTracker,
    create_workflow_tracker,
    detect_loops_in_graph,
    extract_edges_from_langgraph,
    extract_nodes_from_langgraph,
    setup_workflow_tracing,
)

# MLflow and W&B adapters load lazily (PEP 562) on first attribute access so that
# importing this package -- which ``traigent.optimize`` does transitively for the
# workflow-traces helpers -- never imports the optional ``wandb`` package.
# MLFLOW_AVAILABLE comes FROM the mlflow submodule (computed there via
# importlib.util.find_spec): that submodule never eagerly imports the real
# `mlflow` package, so the flag must not be derived from import success.
_LAZY_MLFLOW = (
    "MLFLOW_AVAILABLE",
    "MLflowOptimizationCallback",
    "TraigentMLflowTracker",
    "compare_traigent_runs",
    "create_mlflow_tracker",
    "enable_mlflow_autolog",
    "get_best_traigent_run",
    "log_traigent_optimization",
)
_LAZY_WANDB = {
    "TraigentWandBTracker": "TraigentWandBTracker",
    "WandBOptimizationCallback": "WandBOptimizationCallback",
    "create_wandb_tracker": "create_wandb_tracker",
    "enable_wandb_autolog": "enable_wandb_autolog",
    "log_traigent_to_wandb": "log_traigent_optimization",
}
_LAZY_NAMES = (*_LAZY_MLFLOW, *_LAZY_WANDB, "WANDB_AVAILABLE")
_lazy_lock = threading.RLock()

if TYPE_CHECKING:
    from traigent.integrations.observability.mlflow import (
        MLFLOW_AVAILABLE,
        MLflowOptimizationCallback,
        TraigentMLflowTracker,
        compare_traigent_runs,
        create_mlflow_tracker,
        enable_mlflow_autolog,
        get_best_traigent_run,
        log_traigent_optimization,
    )
    from traigent.integrations.observability.wandb import (
        TraigentWandBTracker,
        WandBOptimizationCallback,
        create_wandb_tracker,
        enable_wandb_autolog,
    )
    from traigent.integrations.observability.wandb import (
        log_traigent_optimization as log_traigent_to_wandb,
    )

    WANDB_AVAILABLE: bool


def _load_mlflow() -> None:
    namespace = globals()
    try:
        module = importlib.import_module("traigent.integrations.observability.mlflow")
        for name in _LAZY_MLFLOW:
            namespace[name] = getattr(module, name)
    except ImportError:
        namespace["MLFLOW_AVAILABLE"] = False


def _load_wandb() -> None:
    namespace = globals()
    try:
        module = importlib.import_module("traigent.integrations.observability.wandb")
        for name, source in _LAZY_WANDB.items():
            namespace[name] = getattr(module, source)
        namespace["WANDB_AVAILABLE"] = True
    except ImportError:
        namespace["WANDB_AVAILABLE"] = False


def __getattr__(name: str) -> Any:
    if name not in _LAZY_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    with _lazy_lock:
        if name not in globals():
            if name in _LAZY_MLFLOW:
                _load_mlflow()
            else:
                _load_wandb()
    try:
        return globals()[name]
    except KeyError:
        # The optional integration is unavailable and exports no such name.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_NAMES))


__all__ = [
    # MLflow
    "MLFLOW_AVAILABLE",
    "TraigentMLflowTracker",
    "MLflowOptimizationCallback",
    "create_mlflow_tracker",
    "enable_mlflow_autolog",
    "log_traigent_optimization",
    "compare_traigent_runs",
    "get_best_traigent_run",
    # W&B
    "WANDB_AVAILABLE",
    "TraigentWandBTracker",
    "WandBOptimizationCallback",
    "create_wandb_tracker",
    "enable_wandb_autolog",
    "log_traigent_to_wandb",
    # Workflow Traces
    "OTEL_AVAILABLE",
    "SpanStatus",
    "SpanType",
    "SpanPayload",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowLoop",
    "WorkflowGraphPayload",
    "TraceIngestionRequest",
    "TraceIngestionResponse",
    "WorkflowTracesClient",
    "TraigentSpanExporter",
    "WorkflowTracesTracker",
    "create_workflow_tracker",
    "setup_workflow_tracing",
    "extract_nodes_from_langgraph",
    "extract_edges_from_langgraph",
    "detect_loops_in_graph",
]
