"""Langfuse integration for reading traces and extracting metrics.

This module provides integration with Langfuse for observability bridging.
Traigent reads traces from Langfuse to extract metrics for optimization,
enabling automatic cost/latency tracking for multi-agent workflows.

Quick Start:
    from traigent.integrations.langfuse import create_langfuse_tracker

    # Create tracker with trace_id resolver
    tracker = create_langfuse_tracker(
        trace_id_resolver=lambda trial: trial.metadata.get("langfuse_trace_id")
    )

    # Use callback in optimization
    @optimize(..., callbacks=[tracker.get_callback()])
    def my_workflow(input_data, **config):
        # Your LangGraph workflow here
        # Langfuse traces are automatically fetched and metrics merged
        ...

Direct Client Usage:
    from traigent.integrations.langfuse import LangfuseClient

    client = LangfuseClient(
        public_key="your_public_key",
        secret_key="your_secret_key",
    )

    # Get metrics from a trace
    metrics = client.get_trace_metrics(trace_id="trace_123")
    print(metrics.total_cost)
    print(metrics.per_agent_costs)  # {"grader": 0.001, "generator": 0.005}

    # Convert to MeasuresDict format
    measures = metrics.to_measures_dict(prefix="langfuse_")
    # {"langfuse_total_cost": 0.006, "langfuse_grader_cost": 0.001, ...}
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

# Import callback
from traigent.integrations.langfuse.callback import (
    LangfuseOptimizationCallback,
    TraceIdResolver,
)

# Import client (always available, handles optional deps internally)
from traigent.integrations.langfuse.client import (
    LangfuseClient,
    LangfuseObservation,
    LangfuseTraceMetrics,
)

# Import tracker
from traigent.integrations.langfuse.tracker import (
    LangfuseTracker,
    create_langfuse_tracker,
)
from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Check for optional langfuse SDK dependency
try:
    from langfuse import Langfuse  # noqa: F401 - imported for availability check

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

__all__ = [
    # Availability flag
    "LANGFUSE_AVAILABLE",
    # Core client
    "LangfuseClient",
    "LangfuseTraceMetrics",
    "LangfuseObservation",
    # Callback integration
    "LangfuseOptimizationCallback",
    "TraceIdResolver",
    # High-level tracker (recommended API)
    "LangfuseTracker",
    "create_langfuse_tracker",
]
