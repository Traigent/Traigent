"""Workflow traces integration for LangGraph visualization.

This module provides integration with the Traigent backend for multi-agent
workflow visualization with trace graphs, attribution analysis, and
performance recommendations.

Usage:
    # Initialize tracker
    tracker = WorkflowTracesTracker(
        backend_url="http://backend:5000",
        auth_token="your_token"
    )

    # Send workflow graph
    graph_id = tracker.send_workflow_graph(
        experiment_id="exp_123",
        experiment_run_id="run_456",
        graph=langgraph_app
    )

    # Start trial tracing
    with tracker.trace_trial(configuration_run_id="config_run_789"):
        # Execute LangGraph workflow
        result = app.invoke(inputs)

For OpenTelemetry integration, use OptiGenSpanExporter:
    exporter = OptiGenSpanExporter(backend_url, auth_token)
    provider.add_span_processor(BatchSpanProcessor(exporter))
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability FUNC-INTEGRATIONS FUNC-ANALYTICS REQ-INT-008 REQ-ANLY-011

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Check for optional dependencies
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore[assignment]
    ReadableSpan = None  # type: ignore[assignment,misc]
    SpanExporter = object  # type: ignore[assignment,misc]
    SpanExportResult = None  # type: ignore[assignment]

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan as ReadableSpanType


# =============================================================================
# Enums and Constants
# =============================================================================


class SpanStatus(StrEnum):
    """OpenTelemetry-compatible span status values."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


class SpanType(StrEnum):
    """Types of spans in a workflow trace."""

    NODE = "node"
    LLM_CALL = "llm_call"
    TOOL = "tool"
    EDGE = "edge"


_SENSITIVE_KEY_PATTERN = re.compile(
    r"(password|secret|token|api[_-]?key|authorization|credential|private[_-]?key)",
    re.IGNORECASE,
)


def _is_sensitive_key(key: str) -> bool:
    return bool(_SENSITIVE_KEY_PATTERN.search(key))


def _to_observability_object(value: Any) -> Any:
    """Normalize supported object types into JSON-serializable containers."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _to_observability_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_observability_object(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _to_observability_object(asdict(value))
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _to_observability_object(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _to_observability_object(value.dict())
        except Exception:
            return str(value)
    return str(value)


def _redact_observability_object(value: Any, parent_key: str | None = None) -> Any:
    """Redact values for sensitive keys in normalized observability payloads."""
    if isinstance(value, dict):
        return {
            str(key): _redact_observability_object(item, str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_observability_object(item, parent_key) for item in value]
    if parent_key and _is_sensitive_key(parent_key):
        return "[REDACTED]"
    return value


def _normalize_and_redact(value: Any) -> Any:
    return _redact_observability_object(_to_observability_object(value))


# =============================================================================
# Data Models / DTOs
# =============================================================================


@dataclass
class SpanPayload:
    """Span data payload for backend ingestion.

    Matches the backend TraceSpan model schema.
    """

    # Required fields
    span_id: str
    trace_id: str
    configuration_run_id: str
    span_name: str
    span_type: str
    start_time: str
    idempotency_key: str | None = None

    # Optional fields
    parent_span_id: str | None = None
    node_id: str | None = None
    end_time: str | None = None
    status: str = SpanStatus.RUNNING.value
    error_message: str | None = None
    decision_reason: str | None = None

    # Metrics
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    # Data
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce deterministic idempotency and redaction at capture time."""
        if not self.idempotency_key:
            self.idempotency_key = (
                f"{self.trace_id}:{self.configuration_run_id}:{self.span_id}"
            )

        normalized_metadata = _normalize_and_redact(self.metadata or {})
        self.metadata = (
            normalized_metadata
            if isinstance(normalized_metadata, dict)
            else {"value": normalized_metadata}
        )

        if self.input_data is not None:
            normalized_input = _normalize_and_redact(self.input_data)
            self.input_data = (
                normalized_input
                if isinstance(normalized_input, dict)
                else {"value": normalized_input}
            )

        if self.output_data is not None:
            normalized_output = _normalize_and_redact(self.output_data)
            self.output_data = (
                normalized_output
                if isinstance(normalized_output, dict)
                else {"value": normalized_output}
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        result: dict[str, Any] = {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "configuration_run_id": self.configuration_run_id,
            "span_name": self.span_name,
            "span_type": self.span_type,
            "start_time": self.start_time,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }

        # Add optional fields only if present
        if self.parent_span_id:
            result["parent_span_id"] = self.parent_span_id
        if self.node_id:
            result["node_id"] = self.node_id
        if self.end_time:
            result["end_time"] = self.end_time
        if self.error_message:
            result["error_message"] = self.error_message
        if self.decision_reason:
            result["decision_reason"] = self.decision_reason
        if self.input_data:
            result["input_data"] = self.input_data
        if self.output_data:
            result["output_data"] = self.output_data
        if self.metadata:
            result["metadata"] = self.metadata

        return result


@dataclass
class WorkflowNode:
    """Node in a workflow graph."""

    id: str
    type: str
    display_name: str
    tunable_params: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        return {
            "id": self.id,
            "type": self.type,
            "display_name": self.display_name,
            "tunable_params": self.tunable_params,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowEdge:
    """Edge in a workflow graph."""

    from_node: str
    to_node: str
    edge_type: str = "default"
    condition: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        result: dict[str, Any] = {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "edge_type": self.edge_type,
        }
        if self.condition:
            result["condition"] = self.condition
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class WorkflowLoop:
    """Loop in a workflow graph."""

    loop_id: str
    entry_node: str
    exit_condition: str
    max_iterations: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        result: dict[str, Any] = {
            "loop_id": self.loop_id,
            "entry_node": self.entry_node,
            "exit_condition": self.exit_condition,
        }
        if self.max_iterations is not None:
            result["max_iterations"] = self.max_iterations
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class WorkflowGraphPayload:
    """Workflow graph topology payload for backend ingestion."""

    experiment_id: str
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge]
    loops: list[WorkflowLoop] = field(default_factory=list)
    experiment_run_id: str | None = None
    sdk_version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        result: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "loops": [loop.to_dict() for loop in self.loops],
            "sdk_version": self.sdk_version,
        }
        if self.experiment_run_id:
            result["experiment_run_id"] = self.experiment_run_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def compute_hash(self) -> str:
        """Compute SHA256 hash of the graph structure for deduplication."""
        # Create a normalized representation
        nodes_data = sorted([n.to_dict() for n in self.nodes], key=lambda x: x["id"])
        edges_data = sorted(
            [e.to_dict() for e in self.edges],
            key=lambda x: (x["from_node"], x["to_node"]),
        )
        loops_data = sorted(
            [loop.to_dict() for loop in self.loops], key=lambda x: x["loop_id"]
        )

        hash_content = json.dumps(
            {"nodes": nodes_data, "edges": edges_data, "loops": loops_data},
            sort_keys=True,
        )
        return hashlib.sha256(hash_content.encode()).hexdigest()


@dataclass
class TraceIngestionRequest:
    """Request payload for trace ingestion endpoint."""

    graph: WorkflowGraphPayload | None = None
    spans: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        result: dict[str, Any] = {}
        if self.graph:
            result["graph"] = self.graph.to_dict()
        if self.spans:
            result["spans"] = self.spans
        return result


@dataclass
class TraceIngestionResponse:
    """Response from trace ingestion endpoint."""

    success: bool
    graph_id: str | None = None
    spans_ingested: int = 0
    trace_id: str | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceIngestionResponse:
        """Create from API response dictionary."""
        response_data = data.get("data", {})
        return cls(
            success=data.get("success", False),
            graph_id=response_data.get("graph_id"),
            spans_ingested=response_data.get("spans_ingested", 0),
            trace_id=response_data.get("trace_id"),
            error=data.get("error"),
        )


# =============================================================================
# LangGraph Extraction Utilities
# =============================================================================


def extract_nodes_from_langgraph(graph: Any) -> list[WorkflowNode]:
    """Extract nodes from a LangGraph StateGraph or CompiledStateGraph.

    Args:
        graph: A LangGraph StateGraph or CompiledStateGraph instance

    Returns:
        List of WorkflowNode objects
    """
    nodes: list[WorkflowNode] = []

    try:
        # Handle CompiledStateGraph
        if hasattr(graph, "nodes"):
            node_dict = graph.nodes
            if isinstance(node_dict, dict):
                for node_id, node_func in node_dict.items():
                    # Determine node type from function or class
                    node_type = _infer_node_type(node_func)
                    display_name = _get_display_name(node_id, node_func)
                    tunable_params = _extract_tunable_params(node_func)

                    nodes.append(
                        WorkflowNode(
                            id=str(node_id),
                            type=node_type,
                            display_name=display_name,
                            tunable_params=tunable_params,
                            metadata={
                                "function_name": getattr(
                                    node_func, "__name__", str(node_func)
                                )
                            },
                        )
                    )

        # Handle StateGraph (uncompiled)
        elif hasattr(graph, "_nodes"):
            node_dict = graph._nodes
            if isinstance(node_dict, dict):
                for node_id, node_func in node_dict.items():
                    node_type = _infer_node_type(node_func)
                    display_name = _get_display_name(node_id, node_func)
                    tunable_params = _extract_tunable_params(node_func)

                    nodes.append(
                        WorkflowNode(
                            id=str(node_id),
                            type=node_type,
                            display_name=display_name,
                            tunable_params=tunable_params,
                        )
                    )

        # Add special entry and end nodes if present
        if hasattr(graph, "entry_point") and graph.entry_point:
            nodes.insert(
                0,
                WorkflowNode(
                    id="__start__",
                    type="entry",
                    display_name="Start",
                ),
            )

        # Check if END is used
        if hasattr(graph, "_edges"):
            for _, to_node in graph._edges.items():
                if to_node == "__end__" or str(to_node) == "END":
                    if not any(n.id == "__end__" for n in nodes):
                        nodes.append(
                            WorkflowNode(
                                id="__end__",
                                type="exit",
                                display_name="End",
                            )
                        )
                    break

    except Exception as e:
        logger.warning(f"Failed to extract nodes from LangGraph: {e}")

    return nodes


def extract_edges_from_langgraph(graph: Any) -> list[WorkflowEdge]:
    """Extract edges from a LangGraph StateGraph or CompiledStateGraph.

    Args:
        graph: A LangGraph StateGraph or CompiledStateGraph instance

    Returns:
        List of WorkflowEdge objects
    """
    edges: list[WorkflowEdge] = []

    try:
        # Handle compiled graph edges
        if hasattr(graph, "edges"):
            edge_data = graph.edges
            if isinstance(edge_data, dict):
                for from_node, to_nodes in edge_data.items():
                    if isinstance(to_nodes, dict):
                        # Conditional edges
                        for condition, to_node in to_nodes.items():
                            edges.append(
                                WorkflowEdge(
                                    from_node=str(from_node),
                                    to_node=str(to_node),
                                    edge_type="conditional",
                                    condition=str(condition),
                                )
                            )
                    elif isinstance(to_nodes, (list, tuple)):
                        for to_node in to_nodes:
                            edges.append(
                                WorkflowEdge(
                                    from_node=str(from_node),
                                    to_node=str(to_node),
                                    edge_type="default",
                                )
                            )
                    else:
                        edges.append(
                            WorkflowEdge(
                                from_node=str(from_node),
                                to_node=str(to_nodes),
                                edge_type="default",
                            )
                        )

        # Handle uncompiled graph edges
        elif hasattr(graph, "_edges"):
            edge_data = graph._edges
            if isinstance(edge_data, dict):
                for from_node, to_node in edge_data.items():
                    edges.append(
                        WorkflowEdge(
                            from_node=str(from_node),
                            to_node=str(to_node),
                            edge_type="default",
                        )
                    )

        # Handle conditional edges
        if hasattr(graph, "_conditional_edges"):
            for from_node, conditionals in graph._conditional_edges.items():
                if isinstance(conditionals, dict):
                    for _condition_func, mappings in conditionals.items():
                        if isinstance(mappings, dict):
                            for condition_value, to_node in mappings.items():
                                edges.append(
                                    WorkflowEdge(
                                        from_node=str(from_node),
                                        to_node=str(to_node),
                                        edge_type="conditional",
                                        condition=str(condition_value),
                                    )
                                )

        # Add entry edge if entry_point is set
        if hasattr(graph, "entry_point") and graph.entry_point:
            edges.insert(
                0,
                WorkflowEdge(
                    from_node="__start__",
                    to_node=str(graph.entry_point),
                    edge_type="entry",
                ),
            )

    except Exception as e:
        logger.warning(f"Failed to extract edges from LangGraph: {e}")

    return edges


def detect_loops_in_graph(graph: Any) -> list[WorkflowLoop]:
    """Detect loops in a LangGraph by analyzing conditional edges.

    Args:
        graph: A LangGraph StateGraph or CompiledStateGraph instance

    Returns:
        List of WorkflowLoop objects
    """
    loops: list[WorkflowLoop] = []

    try:
        # Build adjacency list
        adjacency: dict[str, set[str]] = {}

        if hasattr(graph, "_edges"):
            for from_node, to_node in graph._edges.items():
                from_str = str(from_node)
                to_str = str(to_node)
                if from_str not in adjacency:
                    adjacency[from_str] = set()
                adjacency[from_str].add(to_str)

        if hasattr(graph, "_conditional_edges"):
            for from_node, conditionals in graph._conditional_edges.items():
                from_str = str(from_node)
                if from_str not in adjacency:
                    adjacency[from_str] = set()

                if isinstance(conditionals, dict):
                    for _, mappings in conditionals.items():
                        if isinstance(mappings, dict):
                            for _, to_node in mappings.items():
                                adjacency[from_str].add(str(to_node))

        # Detect back edges (simple cycle detection)
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def find_cycles(node: str, path: list[str]) -> None:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle_nodes = path[cycle_start:]
                if len(cycle_nodes) >= 2:
                    loop_id = f"loop_{'_'.join(cycle_nodes[:3])}"
                    entry_node = cycle_nodes[0]
                    loops.append(
                        WorkflowLoop(
                            loop_id=loop_id,
                            entry_node=entry_node,
                            exit_condition="cycle_complete",
                            metadata={"cycle_nodes": cycle_nodes},
                        )
                    )
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adjacency.get(node, []):
                if neighbor not in ["__end__", "END"]:
                    find_cycles(neighbor, path.copy())

            rec_stack.discard(node)

        # Start DFS from all nodes
        for start_node in adjacency:
            if start_node not in visited:
                find_cycles(start_node, [])

    except Exception as e:
        logger.warning(f"Failed to detect loops in LangGraph: {e}")

    return loops


def _infer_node_type(node_func: Any) -> str:
    """Infer the node type from the function or object."""
    func_name = getattr(node_func, "__name__", str(node_func)).lower()

    if any(kw in func_name for kw in ["llm", "model", "chat", "generate"]):
        return "llm"
    elif any(kw in func_name for kw in ["tool", "search", "retrieve", "fetch"]):
        return "tool"
    elif any(kw in func_name for kw in ["agent"]):
        return "agent"
    elif any(kw in func_name for kw in ["router", "branch", "switch"]):
        return "router"
    else:
        return "agent"


def _get_display_name(node_id: str, node_func: Any) -> str:
    """Get a display-friendly name for the node."""
    # Try to get a descriptive name
    if hasattr(node_func, "__name__"):
        name = node_func.__name__
        # Convert snake_case to Title Case
        return " ".join(word.capitalize() for word in name.split("_"))
    return str(node_id).replace("_", " ").title()


def _extract_tunable_params(node_func: Any) -> list[str]:
    """Extract tunable parameters from a node function."""
    params: list[str] = []

    # Common LLM parameters
    common_params = [
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "model",
    ]

    try:
        # Check function signature
        import inspect

        if callable(node_func):
            sig = inspect.signature(node_func)
            for param_name in sig.parameters:
                if param_name in common_params:
                    params.append(param_name)

        # Check for config or llm attribute
        if hasattr(node_func, "config") or hasattr(node_func, "llm"):
            # Add common LLM params as potentially tunable
            params.extend(["temperature", "max_tokens"])

    except Exception:
        pass

    return list(set(params))


# =============================================================================
# Workflow Traces Client
# =============================================================================


class WorkflowTracesClient:
    """Client for sending workflow traces to the backend.

    Supports both synchronous and asynchronous operations.
    """

    def __init__(
        self,
        backend_url: str,
        auth_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the workflow traces client.

        Args:
            backend_url: Base URL of the backend (e.g., "http://backend:5000")
            auth_token: Bearer token for authentication
            timeout: Request timeout in seconds
        """
        self.backend_url = backend_url.rstrip("/")
        self.auth_token = (
            auth_token
            or os.environ.get("TRAIGENT_API_KEY")
            or os.environ.get("TRAIGENT_API_TOKEN")
        )
        self.timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            # Backend expects API key in X-API-Key header
            headers["X-API-Key"] = self.auth_token
        return headers

    def ingest_traces(
        self,
        graph: WorkflowGraphPayload | None = None,
        spans: list[SpanPayload] | None = None,
        trace_id: str | None = None,
        configuration_run_id: str | None = None,
    ) -> TraceIngestionResponse:
        """Send workflow graph and/or spans to the backend.

        Args:
            graph: Workflow graph topology (send once per experiment run)
            spans: List of span payloads
            trace_id: Trace ID grouping all spans for one trial
            configuration_run_id: Links spans to a specific trial

        Returns:
            TraceIngestionResponse with ingestion results
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is required for sync operations. "
                "Install with: pip install requests"
            )

        request_payload: dict[str, Any] = {}

        if graph:
            request_payload["graph"] = graph.to_dict()

        if spans and trace_id and configuration_run_id:
            request_payload["spans"] = {
                "trace_id": trace_id,
                "configuration_run_id": configuration_run_id,
                "spans": [s.to_dict() for s in spans],
            }

        if not request_payload:
            return TraceIngestionResponse(
                success=False, error="Either graph or spans must be provided"
            )

        try:
            response = requests.post(
                f"{self.backend_url}/api/v1/traces/ingest",
                json=request_payload,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return TraceIngestionResponse.from_dict(response.json())
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to ingest traces: {e}")
            return TraceIngestionResponse(success=False, error=str(e))

    async def ingest_traces_async(
        self,
        graph: WorkflowGraphPayload | None = None,
        spans: list[SpanPayload] | None = None,
        trace_id: str | None = None,
        configuration_run_id: str | None = None,
    ) -> TraceIngestionResponse:
        """Async version of ingest_traces.

        Args:
            graph: Workflow graph topology
            spans: List of span payloads
            trace_id: Trace ID grouping all spans
            configuration_run_id: Links spans to a specific trial

        Returns:
            TraceIngestionResponse with ingestion results
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for async operations. "
                "Install with: pip install aiohttp"
            )

        request_payload: dict[str, Any] = {}

        if graph:
            request_payload["graph"] = graph.to_dict()

        if spans and trace_id and configuration_run_id:
            request_payload["spans"] = {
                "trace_id": trace_id,
                "configuration_run_id": configuration_run_id,
                "spans": [s.to_dict() for s in spans],
            }

        if not request_payload:
            return TraceIngestionResponse(
                success=False, error="Either graph or spans must be provided"
            )

        session: aiohttp.ClientSession | None = None
        try:
            url = f"{self.backend_url}/api/v1/traces/ingest"
            headers = self._get_headers()
            session = aiohttp.ClientSession()
            async with session:
                async with session.post(
                    url,
                    json=request_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return TraceIngestionResponse.from_dict(data)
        except aiohttp.ClientError as e:
            logger.error(f"Failed to ingest traces: {e}")
            return TraceIngestionResponse(success=False, error=str(e))
        finally:
            if session and not session.closed:
                await session.close()


# =============================================================================
# OpenTelemetry Span Exporter
# =============================================================================

if OTEL_AVAILABLE:

    class OptiGenSpanExporter(SpanExporter):
        """OpenTelemetry span exporter that sends traces to the Traigent backend.

        This exporter integrates with the OpenTelemetry SDK to automatically
        capture and export spans during LangGraph execution.

        Usage:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider()
            exporter = OptiGenSpanExporter(
                backend_url="http://backend:5000",
                auth_token="your_token"
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
        """

        def __init__(
            self,
            backend_url: str,
            auth_token: str | None = None,
            configuration_run_id_getter: Any | None = None,
        ) -> None:
            """Initialize the OptiGen span exporter.

            Args:
                backend_url: Base URL of the backend
                auth_token: Bearer token for authentication
                configuration_run_id_getter: Callable that returns the current
                    configuration_run_id for span association
            """
            self.client = WorkflowTracesClient(backend_url, auth_token)
            self._config_run_getter = configuration_run_id_getter
            self._current_config_run_id: str | None = None
            self._lock = threading.Lock()

        def set_configuration_run_id(self, config_run_id: str) -> None:
            """Set the current configuration run ID for span association."""
            with self._lock:
                self._current_config_run_id = config_run_id

        def clear_configuration_run_id(self) -> None:
            """Clear the current configuration run ID."""
            with self._lock:
                self._current_config_run_id = None

        def _get_config_run_id(self) -> str | None:
            """Get the current configuration run ID."""
            if self._config_run_getter:
                result = self._config_run_getter()
                return str(result) if result else None
            with self._lock:
                return self._current_config_run_id

        def export(self, spans: list[ReadableSpanType]) -> SpanExportResult:
            """Export spans to the Traigent backend.

            Args:
                spans: List of OpenTelemetry spans to export

            Returns:
                SpanExportResult indicating success or failure
            """
            if not spans:
                return SpanExportResult.SUCCESS

            config_run_id = self._get_config_run_id()
            if not config_run_id:
                logger.debug("No configuration_run_id set, skipping span export")
                return SpanExportResult.SUCCESS

            # Group spans by trace_id
            trace_groups: dict[str, list[SpanPayload]] = {}

            for otel_span in spans:
                try:
                    span_payload = self._convert_span(otel_span, config_run_id)
                    trace_id = span_payload.trace_id
                    if trace_id not in trace_groups:
                        trace_groups[trace_id] = []
                    trace_groups[trace_id].append(span_payload)
                except Exception as e:
                    logger.warning(f"Failed to convert span: {e}")

            # Export each trace group
            all_success = True
            for trace_id, span_payloads in trace_groups.items():
                try:
                    response = self.client.ingest_traces(
                        spans=span_payloads,
                        trace_id=trace_id,
                        configuration_run_id=config_run_id,
                    )
                    if not response.success:
                        logger.warning(f"Failed to export spans: {response.error}")
                        all_success = False
                except Exception as e:
                    logger.error(f"Error exporting spans: {e}")
                    all_success = False

            return SpanExportResult.SUCCESS if all_success else SpanExportResult.FAILURE

        def _convert_span(
            self, otel_span: ReadableSpanType, config_run_id: str
        ) -> SpanPayload:
            """Convert an OpenTelemetry span to SpanPayload format.

            Args:
                otel_span: OpenTelemetry span to convert
                config_run_id: Configuration run ID to associate with the span

            Returns:
                SpanPayload ready for backend ingestion
            """
            context = otel_span.get_span_context()
            parent = otel_span.parent

            # Convert timestamps (nanoseconds to ISO 8601)
            start_time = datetime.fromtimestamp(
                otel_span.start_time / 1e9, tz=UTC
            ).isoformat()
            end_time = None
            if otel_span.end_time:
                end_time = datetime.fromtimestamp(
                    otel_span.end_time / 1e9, tz=UTC
                ).isoformat()

            # Extract attributes
            attributes = dict(otel_span.attributes) if otel_span.attributes else {}

            # Map OpenTelemetry status to SpanStatus
            status = SpanStatus.COMPLETED.value
            error_message = None
            if otel_span.status:
                if otel_span.status.is_ok:
                    status = SpanStatus.COMPLETED.value
                elif otel_span.status.status_code.name == "ERROR":
                    status = SpanStatus.FAILED.value
                    error_message = otel_span.status.description

            return SpanPayload(
                span_id=format(context.span_id, "016x"),
                trace_id=format(context.trace_id, "032x"),
                configuration_run_id=config_run_id,
                span_name=otel_span.name,
                span_type=attributes.get("span.type", SpanType.NODE.value),
                start_time=start_time,
                parent_span_id=format(parent.span_id, "016x") if parent else None,
                node_id=attributes.get("node.id"),
                end_time=end_time,
                status=status,
                error_message=error_message,
                decision_reason=attributes.get("decision.reason"),
                input_tokens=int(attributes.get("llm.input_tokens", 0)),
                output_tokens=int(attributes.get("llm.output_tokens", 0)),
                cost_usd=float(attributes.get("llm.cost_usd", 0.0)),
                metadata={
                    k: v
                    for k, v in attributes.items()
                    if not k.startswith(("span.", "node.", "llm.", "decision."))
                },
            )

        def shutdown(self) -> None:
            """Shutdown the exporter."""
            pass

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            """Force flush any pending spans.

            Args:
                timeout_millis: Timeout in milliseconds

            Returns:
                True if flush was successful
            """
            return True

else:
    # Stub class when OpenTelemetry is not available
    class OptiGenSpanExporter:  # type: ignore[no-redef]
        """Stub for OptiGenSpanExporter when OpenTelemetry is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "OpenTelemetry is required for OptiGenSpanExporter. "
                "Install with: pip install opentelemetry-sdk"
            )


# =============================================================================
# Workflow Traces Tracker (High-Level API)
# =============================================================================


class WorkflowTracesTracker:
    """High-level tracker for workflow traces with automatic span collection.

    This class provides a convenient interface for:
    - Extracting and sending workflow graphs from LangGraph
    - Collecting spans during trial execution
    - Automatic attribution triggering

    Example:
        tracker = WorkflowTracesTracker(
            backend_url="http://backend:5000",
            auth_token="your_token"
        )

        # Send graph once
        tracker.send_workflow_graph(
            experiment_id="exp_123",
            experiment_run_id="run_456",
            graph=langgraph_app
        )

        # Trace each trial
        with tracker.trace_trial("config_run_789") as trace_ctx:
            result = app.invoke(inputs)
            # Spans are collected automatically if OTEL is configured
    """

    def __init__(
        self,
        backend_url: str | None = None,
        auth_token: str | None = None,
        auto_send: bool = True,
        batch_size: int = 100,
    ) -> None:
        """Initialize the workflow traces tracker.

        Args:
            backend_url: Base URL of the backend (defaults to env var TRAIGENT_BACKEND_URL)
            auth_token: Bearer token for authentication (defaults to env var TRAIGENT_API_TOKEN)
            auto_send: Automatically send spans at end of trial context
            batch_size: Number of spans to batch before sending
        """
        self.backend_url = backend_url or os.environ.get(
            "TRAIGENT_BACKEND_URL", "http://localhost:5000"
        )
        self.auth_token = (
            auth_token
            or os.environ.get("TRAIGENT_API_KEY")
            or os.environ.get("TRAIGENT_API_TOKEN")
        )
        self.auto_send = auto_send
        self.batch_size = batch_size

        self.client = WorkflowTracesClient(
            self.backend_url, self.auth_token  # type: ignore[arg-type]
        )

        # Thread-local storage for trial context
        self._local = threading.local()
        self._graph_id: str | None = None
        self._lock = threading.Lock()

    @property
    def _spans(self) -> list[SpanPayload]:
        """Get thread-local span buffer."""
        if not hasattr(self._local, "spans"):
            self._local.spans = []  # type: ignore[attr-defined]
        spans: list[SpanPayload] = self._local.spans  # type: ignore[attr-defined]
        return spans

    @property
    def _current_trace_id(self) -> str | None:
        """Get thread-local current trace ID."""
        return getattr(self._local, "trace_id", None)

    @property
    def _current_config_run_id(self) -> str | None:
        """Get thread-local current configuration run ID."""
        return getattr(self._local, "config_run_id", None)

    def send_workflow_graph(
        self,
        experiment_id: str,
        experiment_run_id: str,
        graph: Any,
        sdk_version: str = "1.0.0",
    ) -> str | None:
        """Extract and send a workflow graph from a LangGraph instance.

        Args:
            experiment_id: ID of the experiment
            experiment_run_id: ID of the experiment run
            graph: LangGraph StateGraph or CompiledStateGraph instance
            sdk_version: SDK version string

        Returns:
            Graph ID if successful, None otherwise
        """
        # Extract graph components
        nodes = extract_nodes_from_langgraph(graph)
        edges = extract_edges_from_langgraph(graph)
        loops = detect_loops_in_graph(graph)

        graph_payload = WorkflowGraphPayload(
            experiment_id=experiment_id,
            experiment_run_id=experiment_run_id,
            nodes=nodes,
            edges=edges,
            loops=loops,
            sdk_version=sdk_version,
        )

        response = self.client.ingest_traces(graph=graph_payload)

        if response.success and response.graph_id:
            with self._lock:
                self._graph_id = response.graph_id
            logger.info(f"Workflow graph sent successfully: {response.graph_id}")
            return response.graph_id
        else:
            logger.error(f"Failed to send workflow graph: {response.error}")
            return None

    def send_workflow_graph_raw(
        self,
        experiment_id: str,
        experiment_run_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        loops: list[dict[str, Any]] | None = None,
        sdk_version: str = "1.0.0",
    ) -> str | None:
        """Send a raw workflow graph without LangGraph extraction.

        Args:
            experiment_id: ID of the experiment
            experiment_run_id: ID of the experiment run
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            loops: Optional list of loop dictionaries
            sdk_version: SDK version string

        Returns:
            Graph ID if successful, None otherwise
        """
        graph_payload = WorkflowGraphPayload(
            experiment_id=experiment_id,
            experiment_run_id=experiment_run_id,
            nodes=[
                WorkflowNode(
                    id=n["id"],
                    type=n.get("type", "agent"),
                    display_name=n.get("display_name", n["id"]),
                    tunable_params=n.get("tunable_params", []),
                    metadata=n.get("metadata", {}),
                )
                for n in nodes
            ],
            edges=[
                WorkflowEdge(
                    from_node=e["from_node"],
                    to_node=e["to_node"],
                    edge_type=e.get("edge_type", "default"),
                    condition=e.get("condition"),
                    metadata=e.get("metadata", {}),
                )
                for e in edges
            ],
            loops=[
                WorkflowLoop(
                    loop_id=loop["loop_id"],
                    entry_node=loop["entry_node"],
                    exit_condition=loop["exit_condition"],
                    max_iterations=loop.get("max_iterations"),
                    metadata=loop.get("metadata", {}),
                )
                for loop in (loops or [])
            ],
            sdk_version=sdk_version,
        )

        response = self.client.ingest_traces(graph=graph_payload)

        if response.success and response.graph_id:
            with self._lock:
                self._graph_id = response.graph_id
            logger.info(f"Workflow graph sent successfully: {response.graph_id}")
            return response.graph_id
        else:
            logger.error(f"Failed to send workflow graph: {response.error}")
            return None

    @contextmanager
    def trace_trial(
        self,
        configuration_run_id: str,
        trace_id: str | None = None,
    ) -> Generator[dict[str, str | None], None, None]:
        """Context manager for tracing a trial execution.

        Args:
            configuration_run_id: ID of the configuration run (trial)
            trace_id: Optional trace ID (auto-generated if not provided)

        Yields:
            Dictionary with trace context information

        Example:
            with tracker.trace_trial("config_run_123") as ctx:
                result = app.invoke(inputs)
                print(f"Trace ID: {ctx['trace_id']}")
        """
        import uuid

        # Generate trace ID if not provided
        _trace_id = trace_id or str(uuid.uuid4()).replace("-", "")

        # Set thread-local context
        self._local.trace_id = _trace_id
        self._local.config_run_id = configuration_run_id
        self._local.spans = []

        context = {
            "trace_id": _trace_id,
            "configuration_run_id": configuration_run_id,
            "graph_id": self._graph_id,
        }

        try:
            yield context
        finally:
            # Send collected spans if auto_send is enabled
            if self.auto_send and self._spans:
                self._flush_spans()

            # Clear thread-local context
            self._local.trace_id = None
            self._local.config_run_id = None
            self._local.spans = []

    def add_span(
        self,
        span_id: str,
        span_name: str,
        span_type: str | SpanType,
        start_time: datetime | str,
        end_time: datetime | str | None = None,
        parent_span_id: str | None = None,
        node_id: str | None = None,
        status: str | SpanStatus = SpanStatus.COMPLETED,
        error_message: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Manually add a span to the current trial trace.

        Args:
            span_id: Unique span ID (from OTEL or custom)
            span_name: Display name for the span
            span_type: Type of span (node, llm_call, tool, edge)
            start_time: Span start time
            end_time: Span end time (None if still running)
            parent_span_id: ID of parent span
            node_id: ID of the associated workflow node
            status: Span status
            error_message: Error message if failed
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Cost in USD
            metadata: Additional metadata
        """
        if not self._current_trace_id or not self._current_config_run_id:
            logger.warning("No active trial context. Call trace_trial() first.")
            return

        # Convert datetime to ISO string if needed
        if isinstance(start_time, datetime):
            start_time_str = start_time.isoformat()
        else:
            start_time_str = start_time

        end_time_str = None
        if end_time:
            if isinstance(end_time, datetime):
                end_time_str = end_time.isoformat()
            else:
                end_time_str = end_time

        # Convert enums to strings
        if isinstance(span_type, SpanType):
            span_type = span_type.value
        if isinstance(status, SpanStatus):
            status = status.value

        span = SpanPayload(
            span_id=span_id,
            trace_id=self._current_trace_id,
            configuration_run_id=self._current_config_run_id,
            span_name=span_name,
            span_type=span_type,
            start_time=start_time_str,
            end_time=end_time_str,
            parent_span_id=parent_span_id,
            node_id=node_id,
            status=status,
            error_message=error_message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )

        self._spans.append(span)

        # Flush if batch size reached
        if len(self._spans) >= self.batch_size:
            self._flush_spans()

    def _flush_spans(self) -> None:
        """Flush collected spans to the backend."""
        if not self._spans:
            return

        if not self._current_trace_id or not self._current_config_run_id:
            logger.warning("No trace context for flushing spans")
            return

        spans_to_send = self._spans.copy()
        self._local.spans = []

        response = self.client.ingest_traces(
            spans=spans_to_send,
            trace_id=self._current_trace_id,
            configuration_run_id=self._current_config_run_id,
        )

        if response.success:
            logger.debug(f"Flushed {response.spans_ingested} spans")
        else:
            logger.error(f"Failed to flush spans: {response.error}")

    def flush(self) -> None:
        """Manually flush any pending spans."""
        self._flush_spans()

    async def ingest_traces_async(
        self,
        spans: list[SpanPayload],
        trace_id: str,
        configuration_run_id: str,
        graph: WorkflowGraphPayload | None = None,
    ) -> TraceIngestionResponse:
        """Async method to ingest traces directly.

        This is a convenience wrapper around the client's async ingestion
        for use by the orchestrator's workflow trace submission.

        Args:
            spans: List of span payloads to ingest
            trace_id: Trace ID grouping all spans
            configuration_run_id: Links spans to a specific trial
            graph: Optional workflow graph topology

        Returns:
            TraceIngestionResponse with ingestion results
        """
        return await self.client.ingest_traces_async(
            graph=graph,
            spans=spans,
            trace_id=trace_id,
            configuration_run_id=configuration_run_id,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_workflow_tracker(
    backend_url: str | None = None,
    auth_token: str | None = None,
) -> WorkflowTracesTracker:
    """Create a workflow traces tracker instance.

    Args:
        backend_url: Backend URL (defaults to env var)
        auth_token: Auth token (defaults to env var)

    Returns:
        Configured WorkflowTracesTracker instance
    """
    return WorkflowTracesTracker(backend_url=backend_url, auth_token=auth_token)


def setup_workflow_tracing(
    backend_url: str | None = None,
    auth_token: str | None = None,
) -> OptiGenSpanExporter | None:
    """Set up OpenTelemetry tracing with the OptiGen exporter.

    This function configures OpenTelemetry to automatically capture
    and export spans to the Traigent backend.

    Args:
        backend_url: Backend URL (defaults to env var)
        auth_token: Auth token (defaults to env var)

    Returns:
        OptiGenSpanExporter instance if OTEL is available, None otherwise
    """
    if not OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry not available. Install with: pip install opentelemetry-sdk"
        )
        return None

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resolved_url = backend_url or os.environ.get(
        "TRAIGENT_BACKEND_URL", "http://localhost:5000"
    )
    exporter = OptiGenSpanExporter(
        backend_url=resolved_url,  # type: ignore[arg-type]
        auth_token=auth_token,
    )

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    logger.info("OpenTelemetry workflow tracing configured")
    return exporter


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "SpanStatus",
    "SpanType",
    # Data Models
    "SpanPayload",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowLoop",
    "WorkflowGraphPayload",
    "TraceIngestionRequest",
    "TraceIngestionResponse",
    # LangGraph Extraction
    "extract_nodes_from_langgraph",
    "extract_edges_from_langgraph",
    "detect_loops_in_graph",
    # Client
    "WorkflowTracesClient",
    # OpenTelemetry
    "OptiGenSpanExporter",
    "OTEL_AVAILABLE",
    # High-Level API
    "WorkflowTracesTracker",
    # Convenience Functions
    "create_workflow_tracker",
    "setup_workflow_tracing",
]
