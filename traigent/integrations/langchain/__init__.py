"""LangChain/LangGraph callback handler integration for Traigent.

This module provides callback handlers for capturing metrics from LangChain
and LangGraph workflows, enabling optimization based on actual execution data.

Quick Start:
    from traigent.integrations.langchain import TraigentHandler

    # Create handler
    handler = TraigentHandler(metric_prefix="myapp_")

    # Pass to LangGraph/LangChain invocation
    result = app.invoke(
        {"question": "What is AI?"},
        config={"callbacks": [handler]}
    )

    # Get captured metrics
    metrics = handler.get_measures_dict()
    # {"myapp_total_cost": 0.001, "myapp_total_latency_ms": 150.0, ...}

For async-safe trial context:
    from traigent.integrations.langchain import trial_context, TraigentHandler

    async def run_trial(config):
        handler = TraigentHandler()
        with trial_context(config):
            result = await app.ainvoke(input, config={"callbacks": [handler]})
        return result, handler.get_measures_dict()
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

from traigent.integrations.langchain.handler import (
    LANGCHAIN_AVAILABLE,
    LLMCallMetrics,
    ToolCallMetrics,
    TraigentHandler,
    TraigentHandlerMetrics,
    create_traigent_handler,
    get_current_node_name,
    get_current_trial_config,
    node_context,
    trial_context,
)

__all__ = [
    # Availability flag
    "LANGCHAIN_AVAILABLE",
    # Main handler
    "TraigentHandler",
    "create_traigent_handler",
    # Context managers
    "trial_context",
    "node_context",
    "get_current_trial_config",
    "get_current_node_name",
    # Data models
    "TraigentHandlerMetrics",
    "LLMCallMetrics",
    "ToolCallMetrics",
]
