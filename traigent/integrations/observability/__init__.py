"""Observability and monitoring integrations for Traigent."""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability CONC-Quality-Compatibility FUNC-INTEGRATIONS FUNC-ANALYTICS REQ-INT-008 REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

# MLflow integration
try:
    from traigent.integrations.observability.mlflow import (
        MLflowOptimizationCallback,
        TraigentMLflowTracker,
        compare_traigent_runs,
        create_mlflow_tracker,
        enable_mlflow_autolog,
        get_best_traigent_run,
        log_traigent_optimization,
    )

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# W&B integration
try:
    from traigent.integrations.observability.wandb import (
        TraigentWandBTracker,
        WandBOptimizationCallback,
        create_wandb_tracker,
        enable_wandb_autolog,
    )
    from traigent.integrations.observability.wandb import (
        log_traigent_optimization as log_traigent_to_wandb,
    )

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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
