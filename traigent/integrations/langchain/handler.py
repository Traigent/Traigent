"""LangGraph/LangChain callback handler for Traigent instrumentation.

This module provides a callback handler that can be passed to LangGraph/LangChain
invocations to capture metrics for Traigent optimization. It follows the Langfuse
callback handler pattern for consistency.

Usage:
    from traigent.integrations.langchain import TraigentHandler

    handler = TraigentHandler(
        trace_id_generator=lambda: str(uuid.uuid4()),
        metric_prefix="langchain_",
    )

    # Pass to LangGraph/LangChain invocation
    result = app.invoke(
        {"question": "What is AI?"},
        config={"callbacks": [handler]}
    )

    # Get captured metrics
    metrics = handler.get_metrics()

For optimization integration:
    @optimize(callbacks=[tracker.get_callback()])
    def my_workflow(input_data, **config):
        handler = TraigentHandler()
        result = app.invoke(input_data, config={"callbacks": [handler]})
        return result, handler.get_metrics()
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Check for LangChain availability
try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.messages import BaseMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore[assignment, misc]
    BaseMessage = None  # type: ignore[assignment, misc]

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult


# =============================================================================
# Context Variables for Async-Safe State
# =============================================================================

# Trial context for async-safe node identification (use contextvars, not thread-local state)
_current_trial_context: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("traigent_trial_context", default=None)
)

_current_node_name: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "traigent_current_node", default=None
)


@contextmanager
def trial_context(config: dict[str, Any]):
    """Context manager for trial-scoped configuration.

    Use this to set trial configuration that propagates through async LangGraph calls.

    Args:
        config: Trial configuration dict (temperature, model, etc.)

    Example:
        with trial_context({"temperature": 0.7, "model": "gpt-4"}):
            result = await app.ainvoke(inputs)
    """
    token = _current_trial_context.set(config)
    try:
        yield
    finally:
        _current_trial_context.reset(token)


@contextmanager
def node_context(node_name: str):
    """Context manager for node-scoped identification.

    Args:
        node_name: Name of the current LangGraph node
    """
    token = _current_node_name.set(node_name)
    try:
        yield
    finally:
        _current_node_name.reset(token)


def get_current_trial_config() -> dict[str, Any]:
    """Get current trial configuration from context."""
    return _current_trial_context.get() or {}


def get_current_node_name() -> str | None:
    """Get current node name from context."""
    return _current_node_name.get()


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class LLMCallMetrics:
    """Metrics captured from a single LLM call."""

    call_id: str
    node_name: str | None
    model: str | None
    start_time: float
    end_time: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        """Calculate latency in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


@dataclass
class ToolCallMetrics:
    """Metrics captured from a single tool call."""

    call_id: str
    node_name: str | None
    tool_name: str
    start_time: float
    end_time: float | None = None
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        """Calculate latency in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


@dataclass
class TraigentHandlerMetrics:
    """Aggregated metrics from a TraigentHandler session."""

    trace_id: str
    llm_calls: list[LLMCallMetrics] = field(default_factory=list)
    tool_calls: list[ToolCallMetrics] = field(default_factory=list)
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    def to_measures_dict(
        self, *, prefix: str = "", include_per_node: bool = True
    ) -> dict[str, float | int]:
        """Convert to MeasuresDict-compatible format.

        Uses underscore naming per MeasuresDict constraints.

        Args:
            prefix: Prefix for all metric keys
            include_per_node: Include per-node breakdown

        Returns:
            Dict with underscore-separated keys and numeric values
        """
        import re

        measures: dict[str, float | int] = {
            f"{prefix}total_cost": self.total_cost,
            f"{prefix}total_latency_ms": self.total_latency_ms,
            f"{prefix}total_input_tokens": self.total_input_tokens,
            f"{prefix}total_output_tokens": self.total_output_tokens,
            f"{prefix}total_tokens": self.total_tokens,
            f"{prefix}llm_call_count": len(self.llm_calls),
            f"{prefix}tool_call_count": len(self.tool_calls),
        }

        if include_per_node:
            # Aggregate by node
            node_costs: dict[str, float] = {}
            node_latencies: dict[str, float] = {}
            node_tokens: dict[str, int] = {}

            for call in self.llm_calls:
                node = call.node_name or "unknown"
                node_costs[node] = node_costs.get(node, 0.0) + call.cost
                node_latencies[node] = node_latencies.get(node, 0.0) + call.latency_ms
                node_tokens[node] = node_tokens.get(node, 0) + call.total_tokens

            # Sanitize node names for MeasuresDict
            def sanitize(name: str) -> str:
                return re.sub(r"\W", "_", name, flags=re.ASCII)

            for node, cost in node_costs.items():
                safe_node = sanitize(node)
                measures[f"{prefix}{safe_node}_cost"] = cost
            for node, latency in node_latencies.items():
                safe_node = sanitize(node)
                measures[f"{prefix}{safe_node}_latency_ms"] = latency
            for node, tokens in node_tokens.items():
                safe_node = sanitize(node)
                measures[f"{prefix}{safe_node}_tokens"] = tokens

        return measures


# =============================================================================
# TraigentHandler - LangChain/LangGraph Callback Handler
# =============================================================================


class TraigentHandler(BaseCallbackHandler):
    """LangChain/LangGraph callback handler for Traigent instrumentation.

    Captures LLM calls, tool calls, and chain events for optimization.
    Thread-safe and async-compatible using contextvars.

    Args:
        trace_id: Optional trace ID (generated if not provided)
        trace_id_generator: Optional function to generate trace IDs
        metric_prefix: Prefix for metric keys (default: "")
        include_per_node: Include per-node metrics (default: True)

    Example:
        handler = TraigentHandler(metric_prefix="langchain_")
        result = chain.invoke(input, config={"callbacks": [handler]})
        metrics = handler.get_metrics()
    """

    def __init__(
        self,
        *,
        trace_id: str | None = None,
        trace_id_generator: Callable[[], str] | None = None,
        metric_prefix: str = "",
        include_per_node: bool = True,
    ) -> None:
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for TraigentHandler. "
                "Install with: pip install langchain-core"
            )

        super().__init__()
        from traigent.utils.env_config import is_strict_cost_accounting

        self._trace_id = trace_id or (
            trace_id_generator() if trace_id_generator else str(uuid4())
        )
        self._metric_prefix = metric_prefix
        self._include_per_node = include_per_node
        # Latch strict mode at handler creation for consistent per-run behavior.
        self._strict_cost_accounting = is_strict_cost_accounting()

        # Thread-safe state
        self._lock = threading.Lock()
        self._llm_calls: dict[str, LLMCallMetrics] = {}
        self._tool_calls: dict[str, ToolCallMetrics] = {}
        self._completed_llm_calls: list[LLMCallMetrics] = []
        self._completed_tool_calls: list[ToolCallMetrics] = []
        # Track contextvar tokens for proper cleanup on chain end (supports nested chains)
        self._chain_node_tokens: dict[str, contextvars.Token[str | None]] = {}

    @property
    def trace_id(self) -> str:
        """Get the trace ID for this handler session."""
        return self._trace_id

    # =========================================================================
    # LLM Callbacks
    # =========================================================================

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running."""
        call_id = str(run_id)
        node_name = get_current_node_name()

        # Try to extract node name from metadata (LangGraph pattern)
        if node_name is None and metadata:
            node_name = metadata.get("langgraph_node") or metadata.get("node_name")

        # Try to get model from serialized or kwargs
        model = None
        if serialized:
            model = serialized.get("kwargs", {}).get("model") or serialized.get(
                "kwargs", {}
            ).get("model_name")
        if model is None:
            model = kwargs.get("invocation_params", {}).get("model")

        with self._lock:
            self._llm_calls[call_id] = LLMCallMetrics(
                call_id=call_id,
                node_name=node_name,
                model=model,
                start_time=time.time(),
            )

        logger.debug(f"LLM start: {call_id} node={node_name} model={model}")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chat model starts running."""
        call_id = str(run_id)
        node_name = get_current_node_name()

        if node_name is None and metadata:
            node_name = metadata.get("langgraph_node") or metadata.get("node_name")

        model = None
        if serialized:
            model = serialized.get("kwargs", {}).get("model") or serialized.get(
                "kwargs", {}
            ).get("model_name")
        if model is None:
            model = kwargs.get("invocation_params", {}).get("model")

        with self._lock:
            self._llm_calls[call_id] = LLMCallMetrics(
                call_id=call_id,
                node_name=node_name,
                model=model,
                start_time=time.time(),
            )

        logger.debug(f"Chat model start: {call_id} node={node_name} model={model}")

    def on_chat_model_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chat model ends running.

        Chat models (like ChatOpenAI) emit end events via this callback,
        not on_llm_end. Without this, chat model calls would never finalize.
        """
        call_id = str(run_id)

        with self._lock:
            call = self._llm_calls.pop(call_id, None)
            if call is None:
                logger.warning(f"Chat model end for unknown call: {call_id}")
                return

            call.end_time = time.time()

            # Extract token usage from response
            if response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                call.input_tokens = token_usage.get("prompt_tokens", 0)
                call.output_tokens = token_usage.get("completion_tokens", 0)
                call.total_tokens = token_usage.get("total_tokens", 0)

                # Try to get model info if not already set
                if call.model is None:
                    call.model = response.llm_output.get("model_name")

            # Calculate cost (uses litellm if available)
            if call.total_tokens > 0 and call.model:
                call.cost = self._estimate_cost(
                    call.model, call.input_tokens, call.output_tokens
                )

            self._completed_llm_calls.append(call)

        logger.debug(
            f"Chat model end: {call_id} tokens={call.total_tokens} "
            f"latency={call.latency_ms:.1f}ms cost=${call.cost:.6f}"
        )

    def on_chat_model_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chat model errors.

        Chat models (like ChatOpenAI) emit error events via this callback,
        not on_llm_error. Without this, errored chat calls would leak state.
        """
        call_id = str(run_id)

        with self._lock:
            call = self._llm_calls.pop(call_id, None)
            if call is None:
                return

            call.end_time = time.time()
            call.error = str(error)
            self._completed_llm_calls.append(call)

        logger.warning(f"Chat model error: {call_id} error={error}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends running."""
        call_id = str(run_id)

        with self._lock:
            call = self._llm_calls.pop(call_id, None)
            if call is None:
                logger.warning(f"LLM end for unknown call: {call_id}")
                return

            call.end_time = time.time()

            # Extract token usage
            if response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                call.input_tokens = token_usage.get("prompt_tokens", 0)
                call.output_tokens = token_usage.get("completion_tokens", 0)
                call.total_tokens = token_usage.get("total_tokens", 0)

                # Try to get model info if not already set
                if call.model is None:
                    call.model = response.llm_output.get("model_name")

            # Calculate cost (simplified - uses litellm if available)
            if call.total_tokens > 0 and call.model:
                call.cost = self._estimate_cost(
                    call.model, call.input_tokens, call.output_tokens
                )

            self._completed_llm_calls.append(call)

        logger.debug(
            f"LLM end: {call_id} tokens={call.total_tokens} "
            f"latency={call.latency_ms:.1f}ms cost=${call.cost:.6f}"
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        call_id = str(run_id)

        with self._lock:
            call = self._llm_calls.pop(call_id, None)
            if call is None:
                return

            call.end_time = time.time()
            call.error = str(error)
            self._completed_llm_calls.append(call)

        logger.warning(f"LLM error: {call_id} error={error}")

    # =========================================================================
    # Tool Callbacks
    # =========================================================================

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts running."""
        call_id = str(run_id)
        node_name = get_current_node_name()

        if node_name is None and metadata:
            node_name = metadata.get("langgraph_node") or metadata.get("node_name")

        tool_name = serialized.get("name", "unknown") if serialized else "unknown"

        with self._lock:
            self._tool_calls[call_id] = ToolCallMetrics(
                call_id=call_id,
                node_name=node_name,
                tool_name=tool_name,
                start_time=time.time(),
            )

        logger.debug(f"Tool start: {call_id} tool={tool_name} node={node_name}")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool ends running."""
        call_id = str(run_id)

        with self._lock:
            call = self._tool_calls.pop(call_id, None)
            if call is None:
                return

            call.end_time = time.time()
            self._completed_tool_calls.append(call)

        logger.debug(f"Tool end: {call_id} latency={call.latency_ms:.1f}ms")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        call_id = str(run_id)

        with self._lock:
            call = self._tool_calls.pop(call_id, None)
            if call is None:
                return

            call.end_time = time.time()
            call.error = str(error)
            self._completed_tool_calls.append(call)

        logger.warning(f"Tool error: {call_id} error={error}")

    # =========================================================================
    # Chain/Agent Callbacks (for context propagation)
    # =========================================================================

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts. Used for node context propagation."""
        chain_id = str(run_id)

        # Check for LangGraph node in metadata
        if metadata:
            node_name = metadata.get("langgraph_node")
            if node_name:
                # Store token for proper cleanup on chain end (supports nested chains)
                token = _current_node_name.set(node_name)
                with self._lock:
                    self._chain_node_tokens[chain_id] = token
                logger.debug(f"Chain start: node={node_name}")

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain ends. Restores previous node context."""
        self._restore_chain_node_context(run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors. Restores previous node context."""
        self._restore_chain_node_context(run_id)

    def _restore_chain_node_context(self, run_id: Any) -> None:
        """Restore node context to previous value after chain completes."""
        chain_id = str(run_id)
        with self._lock:
            token = self._chain_node_tokens.pop(chain_id, None)
        if token is not None:
            _current_node_name.reset(token)
            logger.debug("Chain end: restored previous node context")

    # =========================================================================
    # Metrics Aggregation
    # =========================================================================

    def _estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Estimate cost for LLM call via the canonical cost_from_tokens().

        In default mode, unknown models return 0.0 with a warning.
        When TRAIGENT_STRICT_COST_ACCOUNTING=true, unknown models raise.
        """
        strict_cost_accounting = self._strict_cost_accounting
        total_tokens = input_tokens + output_tokens
        try:
            from traigent.utils.cost_calculator import (
                _estimation_cost_from_tokens,
                cost_from_tokens,
            )

            input_cost, output_cost = cost_from_tokens(
                input_tokens,
                output_tokens,
                model,
                strict=strict_cost_accounting,
            )
            total_cost = float(input_cost + output_cost)

            # Non-strict mode may return zero for unknown/unmapped aliases.
            # Preserve backward-compatible estimator behavior by attempting
            # the local estimation table before returning zero.
            if not strict_cost_accounting and total_cost <= 0.0 and total_tokens > 0:
                est_input_cost, est_output_cost = _estimation_cost_from_tokens(
                    model,
                    input_tokens,
                    output_tokens,
                    _quiet=True,
                )
                est_total = float(est_input_cost + est_output_cost)
                if est_total > 0.0:
                    logger.debug(
                        "Using estimation pricing fallback for model %r in TraigentHandler",
                        model,
                    )
                    return est_total

            return total_cost
        except Exception:
            if strict_cost_accounting:
                raise
            logger.warning(
                "Cost calculation failed for model %r with %d tokens",
                model,
                total_tokens,
            )
            return 0.0

    def get_metrics(self) -> TraigentHandlerMetrics:
        """Get aggregated metrics from this handler session.

        Returns:
            TraigentHandlerMetrics with all captured data
        """
        with self._lock:
            llm_calls = list(self._completed_llm_calls)
            tool_calls = list(self._completed_tool_calls)

        total_cost = sum(call.cost for call in llm_calls)
        total_latency = sum(call.latency_ms for call in llm_calls) + sum(
            call.latency_ms for call in tool_calls
        )
        total_input = sum(call.input_tokens for call in llm_calls)
        total_output = sum(call.output_tokens for call in llm_calls)
        total_tokens = sum(call.total_tokens for call in llm_calls)

        return TraigentHandlerMetrics(
            trace_id=self._trace_id,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            total_cost=total_cost,
            total_latency_ms=total_latency,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
        )

    def get_measures_dict(self) -> dict[str, float | int]:
        """Get metrics in MeasuresDict-compatible format.

        Convenience method that calls get_metrics().to_measures_dict().
        """
        return self.get_metrics().to_measures_dict(
            prefix=self._metric_prefix,
            include_per_node=self._include_per_node,
        )

    def reset(self) -> None:
        """Reset handler state for reuse."""
        with self._lock:
            self._llm_calls.clear()
            self._tool_calls.clear()
            self._completed_llm_calls.clear()
            self._completed_tool_calls.clear()
            self._chain_node_tokens.clear()
        self._trace_id = str(uuid4())


# =============================================================================
# Factory Functions
# =============================================================================


def create_traigent_handler(
    *,
    trace_id: str | None = None,
    metric_prefix: str = "langchain_",
    include_per_node: bool = True,
) -> TraigentHandler:
    """Factory function to create a TraigentHandler.

    Args:
        trace_id: Optional trace ID (auto-generated if not provided)
        metric_prefix: Prefix for metric keys (default: "langchain_")
        include_per_node: Include per-node metrics breakdown (default: True)

    Returns:
        Configured TraigentHandler instance

    Example:
        handler = create_traigent_handler(metric_prefix="myapp_")
        result = chain.invoke(input, config={"callbacks": [handler]})
        metrics = handler.get_measures_dict()
    """
    return TraigentHandler(
        trace_id=trace_id,
        metric_prefix=metric_prefix,
        include_per_node=include_per_node,
    )
