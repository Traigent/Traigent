"""Unit tests for workflow traces integration.

Tests for Traigent workflow traces with LangGraph visualization support.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Observability
# CONC-Quality-Compatibility FUNC-INTEGRATIONS FUNC-ANALYTICS
# REQ-INT-008 REQ-ANLY-011 SYNC-Observability

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.config.backend_config import DEFAULT_CLOUD_URL
from traigent.integrations.observability import workflow_traces as wt_module
from traigent.integrations.observability.workflow_traces import (
    SpanPayload,
    SpanStatus,
    SpanType,
    TraceIngestionResponse,
    WorkflowEdge,
    WorkflowGraphPayload,
    WorkflowLoop,
    WorkflowNode,
    WorkflowTracesClient,
    WorkflowTracesTracker,
    detect_loops_in_graph,
    extract_edges_from_langgraph,
    extract_nodes_from_langgraph,
)


class MockLangGraphNode:
    """Mock LangGraph node function."""

    def __init__(self, name: str = "test_node") -> None:
        self.__name__ = name

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        return state


class MockStateGraph:
    """Mock LangGraph StateGraph (uncompiled)."""

    def __init__(self) -> None:
        self._nodes: dict[str, Any] = {}
        self._edges: dict[str, str] = {}
        self._conditional_edges: dict[str, Any] = {}
        self.entry_point: str | None = None

    def add_node(self, name: str, func: Any) -> None:
        """Add a node to the graph."""
        self._nodes[name] = func

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge to the graph."""
        self._edges[from_node] = to_node

    def set_entry_point(self, name: str) -> None:
        """Set entry point."""
        self.entry_point = name


class MockCompiledStateGraph:
    """Mock LangGraph CompiledStateGraph."""

    def __init__(self) -> None:
        self.nodes: dict[str, Any] = {}
        self.edges: dict[str, Any] = {}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_requests() -> MagicMock:
    """Create mock requests module."""
    mock = MagicMock()
    mock.post.return_value.status_code = 201
    mock.post.return_value.json.return_value = {
        "success": True,
        "data": {
            "graph_id": "graph_abc123",
            "spans_ingested": 5,
            "trace_id": "trace_xyz",
        },
    }
    return mock


@pytest.fixture
def patched_requests(
    monkeypatch: pytest.MonkeyPatch, mock_requests: MagicMock
) -> MagicMock:
    """Patch requests module in workflow_traces."""
    monkeypatch.setattr(wt_module, "requests", mock_requests)
    monkeypatch.setattr(wt_module, "REQUESTS_AVAILABLE", True)
    return mock_requests


@pytest.fixture
def sample_nodes() -> list[WorkflowNode]:
    """Create sample workflow nodes."""
    return [
        WorkflowNode(
            id="generator",
            type="llm",
            display_name="Content Generator",
            tunable_params=["temperature", "max_tokens"],
        ),
        WorkflowNode(
            id="critic",
            type="llm",
            display_name="Quality Critic",
            tunable_params=["temperature"],
        ),
        WorkflowNode(
            id="router",
            type="router",
            display_name="Decision Router",
        ),
    ]


@pytest.fixture
def sample_edges() -> list[WorkflowEdge]:
    """Create sample workflow edges."""
    return [
        WorkflowEdge(from_node="generator", to_node="critic", edge_type="default"),
        WorkflowEdge(
            from_node="critic",
            to_node="router",
            edge_type="conditional",
            condition="quality_check",
        ),
        WorkflowEdge(from_node="router", to_node="generator", edge_type="loop"),
    ]


@pytest.fixture
def sample_span() -> SpanPayload:
    """Create sample span payload."""
    return SpanPayload(
        span_id="span_001",
        trace_id="trace_abc123",
        configuration_run_id="config_run_456",
        span_name="Content Generator",
        span_type=SpanType.NODE.value,
        start_time=datetime.now(UTC).isoformat(),
        end_time=datetime.now(UTC).isoformat(),
        node_id="generator",
        status=SpanStatus.COMPLETED.value,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.05,
    )


@pytest.fixture
def mock_state_graph() -> MockStateGraph:
    """Create mock LangGraph StateGraph."""
    graph = MockStateGraph()
    graph.add_node("generator", MockLangGraphNode("generate_content"))
    graph.add_node("critic", MockLangGraphNode("evaluate_quality"))
    graph.add_node("router", MockLangGraphNode("route_decision"))
    graph.add_edge("generator", "critic")
    graph.add_edge("critic", "router")
    graph.add_edge("router", "__end__")
    graph.set_entry_point("generator")
    return graph


# =============================================================================
# Tests for Data Models
# =============================================================================


class TestSpanPayload:
    """Tests for SpanPayload dataclass."""

    def test_to_dict_required_fields(self) -> None:
        """Test to_dict includes all required fields."""
        span = SpanPayload(
            span_id="span_001",
            trace_id="trace_abc",
            configuration_run_id="config_001",
            span_name="Test Span",
            span_type="node",
            start_time="2026-01-13T10:00:00Z",
        )

        result = span.to_dict()

        assert result["span_id"] == "span_001"
        assert result["trace_id"] == "trace_abc"
        assert result["configuration_run_id"] == "config_001"
        assert result["span_name"] == "Test Span"
        assert result["span_type"] == "node"
        assert result["start_time"] == "2026-01-13T10:00:00Z"
        assert result["status"] == SpanStatus.RUNNING.value
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["cost_usd"] == 0.0

    def test_to_dict_optional_fields(self) -> None:
        """Test to_dict includes optional fields when present."""
        span = SpanPayload(
            span_id="span_001",
            trace_id="trace_abc",
            configuration_run_id="config_001",
            span_name="Test Span",
            span_type="node",
            start_time="2026-01-13T10:00:00Z",
            end_time="2026-01-13T10:00:05Z",
            parent_span_id="span_root",
            node_id="node_1",
            error_message="Test error",
            metadata={"key": "value"},
        )

        result = span.to_dict()

        assert result["end_time"] == "2026-01-13T10:00:05Z"
        assert result["parent_span_id"] == "span_root"
        assert result["node_id"] == "node_1"
        assert result["error_message"] == "Test error"
        assert result["metadata"] == {"key": "value"}

    def test_to_dict_excludes_none_optionals(self) -> None:
        """Test to_dict excludes optional fields when None."""
        span = SpanPayload(
            span_id="span_001",
            trace_id="trace_abc",
            configuration_run_id="config_001",
            span_name="Test Span",
            span_type="node",
            start_time="2026-01-13T10:00:00Z",
        )

        result = span.to_dict()

        assert "end_time" not in result
        assert "parent_span_id" not in result
        assert "node_id" not in result
        assert "error_message" not in result


class TestWorkflowNode:
    """Tests for WorkflowNode dataclass."""

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        node = WorkflowNode(
            id="node_1",
            type="llm",
            display_name="Test Node",
            tunable_params=["temperature"],
            metadata={"model": "gpt-4o"},
        )

        result = node.to_dict()

        assert result["id"] == "node_1"
        assert result["type"] == "llm"
        assert result["display_name"] == "Test Node"
        assert result["tunable_params"] == ["temperature"]
        assert result["metadata"] == {"model": "gpt-4o"}


class TestWorkflowEdge:
    """Tests for WorkflowEdge dataclass."""

    def test_to_dict_basic(self) -> None:
        """Test to_dict for basic edge."""
        edge = WorkflowEdge(from_node="a", to_node="b", edge_type="default")

        result = edge.to_dict()

        assert result["from_node"] == "a"
        assert result["to_node"] == "b"
        assert result["edge_type"] == "default"
        assert "condition" not in result

    def test_to_dict_with_condition(self) -> None:
        """Test to_dict for conditional edge."""
        edge = WorkflowEdge(
            from_node="a",
            to_node="b",
            edge_type="conditional",
            condition="score >= 0.8",
        )

        result = edge.to_dict()

        assert result["condition"] == "score >= 0.8"


class TestWorkflowLoop:
    """Tests for WorkflowLoop dataclass."""

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        loop = WorkflowLoop(
            loop_id="retry_loop",
            entry_node="generator",
            exit_condition="quality >= 0.9",
            max_iterations=5,
        )

        result = loop.to_dict()

        assert result["loop_id"] == "retry_loop"
        assert result["entry_node"] == "generator"
        assert result["exit_condition"] == "quality >= 0.9"
        assert result["max_iterations"] == 5


class TestWorkflowGraphPayload:
    """Tests for WorkflowGraphPayload dataclass."""

    def test_to_dict(
        self, sample_nodes: list[WorkflowNode], sample_edges: list[WorkflowEdge]
    ) -> None:
        """Test to_dict conversion."""
        graph = WorkflowGraphPayload(
            experiment_id="exp_123",
            experiment_run_id="run_456",
            nodes=sample_nodes,
            edges=sample_edges,
            sdk_version="1.0.0",
        )

        result = graph.to_dict()

        assert result["experiment_id"] == "exp_123"
        assert result["experiment_run_id"] == "run_456"
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 3
        assert result["sdk_version"] == "1.0.0"

    def test_compute_hash_deterministic(
        self, sample_nodes: list[WorkflowNode], sample_edges: list[WorkflowEdge]
    ) -> None:
        """Test compute_hash returns deterministic hash."""
        graph1 = WorkflowGraphPayload(
            experiment_id="exp_123",
            nodes=sample_nodes,
            edges=sample_edges,
        )
        graph2 = WorkflowGraphPayload(
            experiment_id="exp_456",  # Different experiment
            nodes=sample_nodes,
            edges=sample_edges,
        )

        # Same structure should produce same hash
        assert graph1.compute_hash() == graph2.compute_hash()

    def test_compute_hash_different_for_different_graphs(
        self, sample_nodes: list[WorkflowNode]
    ) -> None:
        """Test compute_hash returns different hash for different graphs."""
        graph1 = WorkflowGraphPayload(
            experiment_id="exp_123",
            nodes=sample_nodes,
            edges=[WorkflowEdge(from_node="a", to_node="b")],
        )
        graph2 = WorkflowGraphPayload(
            experiment_id="exp_123",
            nodes=sample_nodes,
            edges=[WorkflowEdge(from_node="x", to_node="y")],
        )

        assert graph1.compute_hash() != graph2.compute_hash()


class TestTraceIngestionResponse:
    """Tests for TraceIngestionResponse dataclass."""

    def test_from_dict_success(self) -> None:
        """Test from_dict for successful response."""
        data = {
            "success": True,
            "data": {
                "graph_id": "graph_123",
                "spans_ingested": 10,
                "trace_id": "trace_abc",
            },
        }

        response = TraceIngestionResponse.from_dict(data)

        assert response.success is True
        assert response.graph_id == "graph_123"
        assert response.spans_ingested == 10
        assert response.trace_id == "trace_abc"
        assert response.error is None

    def test_from_dict_error(self) -> None:
        """Test from_dict for error response."""
        data = {
            "success": False,
            "error": "Configuration run not found",
        }

        response = TraceIngestionResponse.from_dict(data)

        assert response.success is False
        assert response.error == "Configuration run not found"


# =============================================================================
# Tests for LangGraph Extraction
# =============================================================================


class TestExtractNodesFromLangGraph:
    """Tests for extract_nodes_from_langgraph function."""

    def test_extract_from_state_graph(self, mock_state_graph: MockStateGraph) -> None:
        """Test extracting nodes from uncompiled StateGraph."""
        nodes = extract_nodes_from_langgraph(mock_state_graph)

        assert len(nodes) >= 3  # At least the 3 added nodes
        node_ids = [n.id for n in nodes]
        assert "generator" in node_ids
        assert "critic" in node_ids
        assert "router" in node_ids

    def test_extract_from_compiled_graph(self) -> None:
        """Test extracting nodes from CompiledStateGraph."""
        graph = MockCompiledStateGraph()
        graph.nodes = {
            "agent": MockLangGraphNode("agent_func"),
            "tool": MockLangGraphNode("search_tool"),
        }

        nodes = extract_nodes_from_langgraph(graph)

        assert len(nodes) == 2
        node_ids = [n.id for n in nodes]
        assert "agent" in node_ids
        assert "tool" in node_ids

    def test_extract_adds_entry_node(self, mock_state_graph: MockStateGraph) -> None:
        """Test that entry node is added when entry_point is set."""
        nodes = extract_nodes_from_langgraph(mock_state_graph)

        # Should have __start__ node
        node_ids = [n.id for n in nodes]
        assert "__start__" in node_ids

    def test_extract_adds_end_node(self, mock_state_graph: MockStateGraph) -> None:
        """Test that end node is added when __end__ edge exists."""
        nodes = extract_nodes_from_langgraph(mock_state_graph)

        # Should have __end__ node
        node_ids = [n.id for n in nodes]
        assert "__end__" in node_ids

    def test_extract_handles_empty_graph(self) -> None:
        """Test extracting from empty graph."""
        graph = MockStateGraph()

        nodes = extract_nodes_from_langgraph(graph)

        assert len(nodes) == 0

    def test_extract_handles_invalid_graph(self) -> None:
        """Test extracting from invalid object."""
        nodes = extract_nodes_from_langgraph("not a graph")

        assert len(nodes) == 0


class TestExtractEdgesFromLangGraph:
    """Tests for extract_edges_from_langgraph function."""

    def test_extract_from_state_graph(self, mock_state_graph: MockStateGraph) -> None:
        """Test extracting edges from uncompiled StateGraph."""
        edges = extract_edges_from_langgraph(mock_state_graph)

        assert len(edges) >= 3  # At least the 3 added edges

    def test_extract_includes_entry_edge(
        self, mock_state_graph: MockStateGraph
    ) -> None:
        """Test that entry edge is added when entry_point is set."""
        edges = extract_edges_from_langgraph(mock_state_graph)

        # Should have entry edge from __start__
        entry_edges = [e for e in edges if e.from_node == "__start__"]
        assert len(entry_edges) == 1
        assert entry_edges[0].to_node == "generator"

    def test_extract_handles_conditional_edges(self) -> None:
        """Test extracting conditional edges."""
        graph = MockStateGraph()
        graph.add_node("a", MockLangGraphNode("a"))
        graph.add_node("b", MockLangGraphNode("b"))
        graph.add_node("c", MockLangGraphNode("c"))
        graph._conditional_edges = {"a": {"router_func": {"pass": "b", "fail": "c"}}}

        edges = extract_edges_from_langgraph(graph)

        conditional_edges = [e for e in edges if e.edge_type == "conditional"]
        assert len(conditional_edges) == 2

    def test_extract_handles_empty_graph(self) -> None:
        """Test extracting from empty graph."""
        graph = MockStateGraph()

        edges = extract_edges_from_langgraph(graph)

        assert len(edges) == 0


class TestDetectLoopsInGraph:
    """Tests for detect_loops_in_graph function."""

    def test_detect_simple_loop(self) -> None:
        """Test detecting a simple loop."""
        graph = MockStateGraph()
        graph.add_node("a", MockLangGraphNode("a"))
        graph.add_node("b", MockLangGraphNode("b"))
        graph._edges = {"a": "b", "b": "a"}  # Loop: a -> b -> a

        loops = detect_loops_in_graph(graph)

        assert len(loops) >= 1

    def test_no_loops_in_linear_graph(self) -> None:
        """Test no loops detected in linear graph."""
        graph = MockStateGraph()
        graph.add_node("a", MockLangGraphNode("a"))
        graph.add_node("b", MockLangGraphNode("b"))
        graph.add_node("c", MockLangGraphNode("c"))
        graph._edges = {"a": "b", "b": "c", "c": "__end__"}

        loops = detect_loops_in_graph(graph)

        assert len(loops) == 0

    def test_handles_empty_graph(self) -> None:
        """Test handling empty graph."""
        graph = MockStateGraph()

        loops = detect_loops_in_graph(graph)

        assert len(loops) == 0


# =============================================================================
# Tests for WorkflowTracesClient
# =============================================================================


class TestWorkflowTracesClient:
    """Tests for WorkflowTracesClient."""

    def test_init_defaults(self) -> None:
        """Test initialization with defaults."""
        client = WorkflowTracesClient(backend_url="http://localhost:5000")

        assert client.backend_url == "http://localhost:5000"
        assert client.timeout == 30.0

    def test_init_with_auth_token(self) -> None:
        """Test initialization with auth token."""
        client = WorkflowTracesClient(
            backend_url="http://localhost:5000",
            auth_token="test_token",
        )

        headers = client._get_headers()

        # Backend expects API key in X-API-Key header (not Authorization)
        assert headers["X-API-Key"] == "test_token"

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from URL."""
        client = WorkflowTracesClient(backend_url="http://localhost:5000/")

        assert client.backend_url == "http://localhost:5000"

    def test_ingest_traces_with_graph(
        self,
        patched_requests: MagicMock,
        sample_nodes: list[WorkflowNode],
        sample_edges: list[WorkflowEdge],
    ) -> None:
        """Test ingesting workflow graph."""
        client = WorkflowTracesClient(
            backend_url="http://localhost:5000",
            auth_token="token",
        )

        graph = WorkflowGraphPayload(
            experiment_id="exp_123",
            experiment_run_id="run_456",
            nodes=sample_nodes,
            edges=sample_edges,
        )

        response = client.ingest_traces(graph=graph)

        assert response.success is True
        assert response.graph_id == "graph_abc123"
        patched_requests.post.assert_called_once()

    def test_ingest_traces_with_spans(
        self, patched_requests: MagicMock, sample_span: SpanPayload
    ) -> None:
        """Test ingesting spans."""
        client = WorkflowTracesClient(
            backend_url="http://localhost:5000",
            auth_token="token",
        )

        response = client.ingest_traces(
            spans=[sample_span],
            trace_id="trace_abc",
            configuration_run_id="config_001",
        )

        assert response.success is True
        assert response.spans_ingested == 5
        patched_requests.post.assert_called_once()

    def test_ingest_traces_no_data_returns_error(self) -> None:
        """Test ingesting with no data returns error."""
        client = WorkflowTracesClient(backend_url="http://localhost:5000")

        response = client.ingest_traces()

        assert response.success is False
        assert response.error is not None

    def test_ingest_traces_handles_request_error(
        self, patched_requests: MagicMock
    ) -> None:
        """Test handling request errors."""
        patched_requests.post.side_effect = Exception("Connection failed")

        client = WorkflowTracesClient(backend_url="http://localhost:5000")

        response = client.ingest_traces(
            spans=[],
            trace_id="trace",
            configuration_run_id="config",
        )

        # Should return error response rather than failing with empty data
        assert response.success is False


# =============================================================================
# Tests for WorkflowTracesTracker
# =============================================================================


class TestWorkflowTracesTracker:
    """Tests for WorkflowTracesTracker."""

    def test_init_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with defaults."""
        monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://test:5000")

        tracker = WorkflowTracesTracker()

        assert tracker.backend_url == "http://test:5000"
        assert tracker.auto_send is True
        assert tracker.batch_size == 100

    def test_init_defaults_to_cloud_when_env_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default tracker backend should be cloud-first, not localhost."""
        monkeypatch.delenv("TRAIGENT_BACKEND_URL", raising=False)

        tracker = WorkflowTracesTracker()

        assert tracker.backend_url == DEFAULT_CLOUD_URL

    def test_init_with_params(self) -> None:
        """Test initialization with custom parameters."""
        tracker = WorkflowTracesTracker(
            backend_url="http://custom:8000",
            auth_token="my_token",
            auto_send=False,
            batch_size=50,
        )

        assert tracker.backend_url == "http://custom:8000"
        assert tracker.auth_token == "my_token"
        assert tracker.auto_send is False
        assert tracker.batch_size == 50

    def test_send_workflow_graph(
        self, patched_requests: MagicMock, mock_state_graph: MockStateGraph
    ) -> None:
        """Test sending workflow graph from LangGraph."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auth_token="token",
        )

        graph_id = tracker.send_workflow_graph(
            experiment_id="exp_123",
            experiment_run_id="run_456",
            graph=mock_state_graph,
        )

        assert graph_id == "graph_abc123"
        patched_requests.post.assert_called_once()

    def test_send_workflow_graph_raw(self, patched_requests: MagicMock) -> None:
        """Test sending raw workflow graph data."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auth_token="token",
        )

        nodes = [
            {"id": "n1", "type": "llm", "display_name": "Generator"},
            {"id": "n2", "type": "llm", "display_name": "Critic"},
        ]
        edges = [{"from_node": "n1", "to_node": "n2"}]

        graph_id = tracker.send_workflow_graph_raw(
            experiment_id="exp_123",
            experiment_run_id="run_456",
            nodes=nodes,
            edges=edges,
        )

        assert graph_id == "graph_abc123"

    def test_trace_trial_context(self, patched_requests: MagicMock) -> None:
        """Test trace_trial context manager."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auth_token="token",
            auto_send=False,
        )

        with tracker.trace_trial("config_run_123") as ctx:
            assert "trace_id" in ctx
            assert ctx["configuration_run_id"] == "config_run_123"

    def test_trace_trial_with_custom_trace_id(self) -> None:
        """Test trace_trial with custom trace ID."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auto_send=False,
        )

        with tracker.trace_trial("config_123", trace_id="custom_trace") as ctx:
            assert ctx["trace_id"] == "custom_trace"

    def test_add_span_within_context(self, patched_requests: MagicMock) -> None:
        """Test adding span within trial context."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auth_token="token",
            auto_send=False,
        )

        with tracker.trace_trial("config_123"):
            tracker.add_span(
                span_id="span_001",
                span_name="Test Node",
                span_type=SpanType.NODE,
                start_time=datetime.now(UTC),
                end_time=datetime.now(UTC),
                status=SpanStatus.COMPLETED,
            )

            assert len(tracker._spans) == 1

    def test_add_span_outside_context_warns(
        self, patched_requests: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test adding span outside context logs warning."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auto_send=False,
        )

        with caplog.at_level(logging.WARNING):
            tracker.add_span(
                span_id="span_001",
                span_name="Test Node",
                span_type="node",
                start_time=datetime.now(UTC),
            )

        assert "No active trial context" in caplog.text

    def test_auto_send_on_context_exit(self, patched_requests: MagicMock) -> None:
        """Test spans are automatically sent on context exit."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auth_token="token",
            auto_send=True,
        )

        with tracker.trace_trial("config_123"):
            tracker.add_span(
                span_id="span_001",
                span_name="Test Node",
                span_type="node",
                start_time=datetime.now(UTC),
                end_time=datetime.now(UTC),
            )

        # Should have made API call
        patched_requests.post.assert_called_once()

    def test_batch_flush_on_batch_size(self, patched_requests: MagicMock) -> None:
        """Test spans are flushed when batch size is reached."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auth_token="token",
            auto_send=True,
            batch_size=3,
        )

        with tracker.trace_trial("config_123"):
            for i in range(5):
                tracker.add_span(
                    span_id=f"span_{i}",
                    span_name=f"Node {i}",
                    span_type="node",
                    start_time=datetime.now(UTC),
                )

        # Should have flushed at batch size + final flush
        assert patched_requests.post.call_count >= 1

    def test_manual_flush(self, patched_requests: MagicMock) -> None:
        """Test manual flush of spans."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auth_token="token",
            auto_send=False,
        )

        with tracker.trace_trial("config_123"):
            tracker.add_span(
                span_id="span_001",
                span_name="Test Node",
                span_type="node",
                start_time=datetime.now(UTC),
            )
            tracker.flush()

        patched_requests.post.assert_called_once()


# =============================================================================
# Tests for Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_workflow_tracker(self) -> None:
        """Test create_workflow_tracker function."""
        tracker = wt_module.create_workflow_tracker(
            backend_url="http://localhost:5000",
            auth_token="token",
        )

        assert isinstance(tracker, WorkflowTracesTracker)
        assert tracker.backend_url == "http://localhost:5000"

    def test_setup_workflow_tracing_without_otel(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test setup_workflow_tracing when OTEL not available."""
        monkeypatch.setattr(wt_module, "OTEL_AVAILABLE", False)

        with caplog.at_level(logging.WARNING):
            result = wt_module.setup_workflow_tracing()

        assert result is None
        assert "OpenTelemetry not available" in caplog.text


# =============================================================================
# Tests for SpanStatus and SpanType Enums
# =============================================================================


class TestEnums:
    """Tests for enum values."""

    def test_span_status_values(self) -> None:
        """Test SpanStatus enum values."""
        assert SpanStatus.RUNNING.value == "RUNNING"
        assert SpanStatus.COMPLETED.value == "COMPLETED"
        assert SpanStatus.FAILED.value == "FAILED"
        assert SpanStatus.REJECTED.value == "REJECTED"
        assert SpanStatus.TIMEOUT.value == "TIMEOUT"
        assert SpanStatus.CANCELLED.value == "CANCELLED"

    def test_span_type_values(self) -> None:
        """Test SpanType enum values."""
        assert SpanType.NODE.value == "node"
        assert SpanType.LLM_CALL.value == "llm_call"
        assert SpanType.TOOL.value == "tool"
        assert SpanType.EDGE.value == "edge"


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tracker_thread_isolation(self) -> None:
        """Test that tracker maintains thread-local state."""
        import threading

        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auto_send=False,
        )

        results = {}

        def thread_func(thread_id: int) -> None:
            with tracker.trace_trial(f"config_{thread_id}") as ctx:
                results[thread_id] = ctx["configuration_run_id"]

        threads = [threading.Thread(target=thread_func, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results[0] == "config_0"
        assert results[1] == "config_1"
        assert results[2] == "config_2"

    def test_span_with_datetime_start_time(self) -> None:
        """Test adding span with datetime object for start_time."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auto_send=False,
        )

        with tracker.trace_trial("config_123"):
            now = datetime.now(UTC)
            tracker.add_span(
                span_id="span_001",
                span_name="Test Node",
                span_type=SpanType.NODE,
                start_time=now,
            )

            assert len(tracker._spans) == 1
            assert isinstance(tracker._spans[0].start_time, str)

    def test_span_with_string_start_time(self) -> None:
        """Test adding span with string for start_time."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auto_send=False,
        )

        with tracker.trace_trial("config_123"):
            tracker.add_span(
                span_id="span_001",
                span_name="Test Node",
                span_type="node",
                start_time="2026-01-13T10:00:00Z",
            )

            assert len(tracker._spans) == 1
            assert tracker._spans[0].start_time == "2026-01-13T10:00:00Z"

    def test_graph_extraction_infers_node_type_llm(self) -> None:
        """Test that node type is inferred correctly for LLM nodes."""
        graph = MockStateGraph()
        graph.add_node("llm_node", MockLangGraphNode("generate_with_llm"))

        nodes = extract_nodes_from_langgraph(graph)

        llm_nodes = [n for n in nodes if n.id == "llm_node"]
        assert len(llm_nodes) == 1
        assert llm_nodes[0].type == "llm"

    def test_graph_extraction_infers_node_type_tool(self) -> None:
        """Test that node type is inferred correctly for tool nodes."""
        graph = MockStateGraph()
        graph.add_node("tool_node", MockLangGraphNode("search_tool"))

        nodes = extract_nodes_from_langgraph(graph)

        tool_nodes = [n for n in nodes if n.id == "tool_node"]
        assert len(tool_nodes) == 1
        assert tool_nodes[0].type == "tool"

    def test_graph_extraction_infers_node_type_agent(self) -> None:
        """Test that node type is inferred correctly for agent nodes."""
        graph = MockStateGraph()
        # Use "task_agent" - "research_agent" contains "search" which triggers "tool" type
        graph.add_node("agent_node", MockLangGraphNode("task_agent"))

        nodes = extract_nodes_from_langgraph(graph)

        agent_nodes = [n for n in nodes if n.id == "agent_node"]
        assert len(agent_nodes) == 1
        assert agent_nodes[0].type == "agent"

    def test_graph_extraction_infers_node_type_router(self) -> None:
        """Test that node type is inferred correctly for router nodes."""
        graph = MockStateGraph()
        graph.add_node("router_node", MockLangGraphNode("branch_router"))

        nodes = extract_nodes_from_langgraph(graph)

        router_nodes = [n for n in nodes if n.id == "router_node"]
        assert len(router_nodes) == 1
        assert router_nodes[0].type == "router"

    def test_get_display_name_from_node_function(self) -> None:
        """Test _get_display_name extracts name from function."""
        from traigent.integrations.observability.workflow_traces import (
            _get_display_name,
        )

        node_func = MockLangGraphNode("process_input_data")
        name = _get_display_name("node_1", node_func)
        assert name == "Process Input Data"

    def test_get_display_name_fallback(self) -> None:
        """Test _get_display_name falls back to node_id."""
        from traigent.integrations.observability.workflow_traces import (
            _get_display_name,
        )

        # Object without __name__
        node_obj = object()
        name = _get_display_name("my_node_func", node_obj)
        assert name == "My Node Func"

    def test_infer_node_type_chat(self) -> None:
        """Test _infer_node_type for chat nodes."""
        from traigent.integrations.observability.workflow_traces import _infer_node_type

        node_func = MockLangGraphNode("chat_completion")
        assert _infer_node_type(node_func) == "llm"

    def test_infer_node_type_model(self) -> None:
        """Test _infer_node_type for model nodes."""
        from traigent.integrations.observability.workflow_traces import _infer_node_type

        node_func = MockLangGraphNode("run_model")
        assert _infer_node_type(node_func) == "llm"

    def test_infer_node_type_retrieve(self) -> None:
        """Test _infer_node_type for retrieve nodes."""
        from traigent.integrations.observability.workflow_traces import _infer_node_type

        node_func = MockLangGraphNode("retrieve_documents")
        assert _infer_node_type(node_func) == "tool"

    def test_infer_node_type_fetch(self) -> None:
        """Test _infer_node_type for fetch nodes."""
        from traigent.integrations.observability.workflow_traces import _infer_node_type

        node_func = MockLangGraphNode("fetch_data")
        assert _infer_node_type(node_func) == "tool"

    def test_infer_node_type_switch(self) -> None:
        """Test _infer_node_type for switch/branch nodes."""
        from traigent.integrations.observability.workflow_traces import _infer_node_type

        node_func = MockLangGraphNode("switch_handler")
        assert _infer_node_type(node_func) == "router"

    def test_infer_node_type_default(self) -> None:
        """Test _infer_node_type returns agent for unknown types."""
        from traigent.integrations.observability.workflow_traces import _infer_node_type

        node_func = MockLangGraphNode("process_something")
        assert _infer_node_type(node_func) == "agent"

    def test_extract_tunable_params_from_signature(self) -> None:
        """Test _extract_tunable_params extracts common params."""
        from traigent.integrations.observability.workflow_traces import (
            _extract_tunable_params,
        )

        def node_func(state, temperature=0.7, max_tokens=100):
            pass

        params = _extract_tunable_params(node_func)
        assert "temperature" in params
        assert "max_tokens" in params

    def test_add_span_with_datetime_end_time(self) -> None:
        """Test adding span with datetime end_time."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auto_send=False,
        )

        with tracker.trace_trial("config_123"):
            now = datetime.now(UTC)
            tracker.add_span(
                span_id="span_001",
                span_name="Test Node",
                span_type=SpanType.NODE,
                start_time=now,
                end_time=now,
            )

            assert len(tracker._spans) == 1
            assert isinstance(tracker._spans[0].end_time, str)

    def test_add_span_with_string_end_time(self) -> None:
        """Test adding span with string end_time."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auto_send=False,
        )

        with tracker.trace_trial("config_123"):
            tracker.add_span(
                span_id="span_001",
                span_name="Test Node",
                span_type="node",
                start_time="2026-01-13T10:00:00Z",
                end_time="2026-01-13T10:00:05Z",
            )

            assert len(tracker._spans) == 1
            assert tracker._spans[0].end_time == "2026-01-13T10:00:05Z"

    def test_add_span_with_span_status_enum(self) -> None:
        """Test adding span with SpanStatus enum."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auto_send=False,
        )

        with tracker.trace_trial("config_123"):
            tracker.add_span(
                span_id="span_001",
                span_name="Test Node",
                span_type=SpanType.LLM_CALL,
                start_time=datetime.now(UTC),
                status=SpanStatus.FAILED,
                error_message="Test error",
            )

            assert len(tracker._spans) == 1
            assert tracker._spans[0].status == "FAILED"


# =============================================================================
# Tests for Async Methods
# =============================================================================


class TestAsyncMethods:
    """Tests for async methods in workflow traces."""

    @pytest.fixture
    def mock_aiohttp(self) -> MagicMock:
        """Create mock aiohttp module."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def patched_aiohttp(
        self, monkeypatch: pytest.MonkeyPatch, mock_aiohttp: MagicMock
    ) -> MagicMock:
        """Patch aiohttp module in workflow_traces."""
        monkeypatch.setattr(wt_module, "aiohttp", mock_aiohttp)
        monkeypatch.setattr(wt_module, "AIOHTTP_AVAILABLE", True)
        return mock_aiohttp

    @pytest.mark.asyncio
    async def test_ingest_traces_async_requires_aiohttp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that async ingest raises ImportError without aiohttp."""
        monkeypatch.setattr(wt_module, "AIOHTTP_AVAILABLE", False)

        client = WorkflowTracesClient(backend_url="http://localhost:5000")

        with pytest.raises(ImportError, match="aiohttp is required"):
            await client.ingest_traces_async()

    @pytest.mark.asyncio
    async def test_ingest_traces_async_empty_returns_error(
        self, patched_aiohttp: MagicMock
    ) -> None:
        """Test async ingest with no data returns error."""
        client = WorkflowTracesClient(backend_url="http://localhost:5000")

        response = await client.ingest_traces_async()
        assert response.success is False
        assert response.error is not None


# =============================================================================
# Tests for TraceIngestionRequest
# =============================================================================


class TestTraceIngestionRequest:
    """Tests for TraceIngestionRequest dataclass."""

    def test_to_dict_empty(self) -> None:
        """Test to_dict with no data."""
        from traigent.integrations.observability.workflow_traces import (
            TraceIngestionRequest,
        )

        request = TraceIngestionRequest()
        result = request.to_dict()
        assert result == {}

    def test_to_dict_with_graph(
        self, sample_nodes: list[WorkflowNode], sample_edges: list[WorkflowEdge]
    ) -> None:
        """Test to_dict with graph."""
        from traigent.integrations.observability.workflow_traces import (
            TraceIngestionRequest,
        )

        graph = WorkflowGraphPayload(
            experiment_id="exp_123",
            nodes=sample_nodes,
            edges=sample_edges,
        )
        request = TraceIngestionRequest(graph=graph)
        result = request.to_dict()

        assert "graph" in result
        assert result["graph"]["experiment_id"] == "exp_123"

    def test_to_dict_with_spans(self) -> None:
        """Test to_dict with spans."""
        from traigent.integrations.observability.workflow_traces import (
            TraceIngestionRequest,
        )

        spans_data = {
            "trace_id": "trace_123",
            "configuration_run_id": "config_456",
            "spans": [],
        }
        request = TraceIngestionRequest(spans=spans_data)
        result = request.to_dict()

        assert "spans" in result
        assert result["spans"]["trace_id"] == "trace_123"


# =============================================================================
# Tests for Compiled Graph Edge Extraction
# =============================================================================


class TestCompiledGraphEdgeExtraction:
    """Tests for edge extraction from compiled graphs."""

    def test_extract_edges_from_compiled_graph_dict_mapping(self) -> None:
        """Test extracting edges from compiled graph with dict mapping."""
        graph = MockCompiledStateGraph()
        graph.edges = {
            "node_a": {"pass": "node_b", "fail": "node_c"},  # Conditional
        }

        edges = extract_edges_from_langgraph(graph)

        # Should have 2 conditional edges
        assert len(edges) == 2
        edge_targets = {e.to_node for e in edges}
        assert "node_b" in edge_targets
        assert "node_c" in edge_targets

    def test_extract_edges_from_compiled_graph_list_mapping(self) -> None:
        """Test extracting edges from compiled graph with list mapping."""
        graph = MockCompiledStateGraph()
        graph.edges = {
            "node_a": ["node_b", "node_c"],  # Multiple targets
        }

        edges = extract_edges_from_langgraph(graph)

        assert len(edges) == 2
        edge_targets = {e.to_node for e in edges}
        assert "node_b" in edge_targets
        assert "node_c" in edge_targets

    def test_extract_edges_from_compiled_graph_simple_mapping(self) -> None:
        """Test extracting edges from compiled graph with simple string mapping."""
        graph = MockCompiledStateGraph()
        graph.edges = {
            "node_a": "node_b",  # Simple edge
            "node_b": "node_c",
        }

        edges = extract_edges_from_langgraph(graph)

        assert len(edges) == 2


# =============================================================================
# Tests for Raw Graph Submission with Loops
# =============================================================================


class TestRawGraphWithLoops:
    """Tests for raw graph submission with loops."""

    def test_send_workflow_graph_raw_with_loops(
        self, patched_requests: MagicMock
    ) -> None:
        """Test sending raw workflow graph with loops."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auth_token="token",
        )

        nodes = [
            {"id": "n1", "type": "llm", "display_name": "Generator"},
            {"id": "n2", "type": "llm", "display_name": "Critic"},
        ]
        edges = [
            {"from_node": "n1", "to_node": "n2"},
            {"from_node": "n2", "to_node": "n1"},
        ]
        loops = [
            {
                "loop_id": "retry_loop",
                "entry_node": "n1",
                "exit_condition": "quality >= 0.9",
                "max_iterations": 3,
            }
        ]

        graph_id = tracker.send_workflow_graph_raw(
            experiment_id="exp_123",
            experiment_run_id="run_456",
            nodes=nodes,
            edges=edges,
            loops=loops,
        )

        assert graph_id == "graph_abc123"

    def test_send_workflow_graph_raw_with_metadata(
        self, patched_requests: MagicMock
    ) -> None:
        """Test sending raw workflow graph with node/edge metadata."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auth_token="token",
        )

        nodes = [
            {
                "id": "n1",
                "type": "llm",
                "display_name": "Generator",
                "tunable_params": ["temperature", "max_tokens"],
                "metadata": {"model": "gpt-4o"},
            },
        ]
        edges = [
            {
                "from_node": "n1",
                "to_node": "n2",
                "edge_type": "conditional",
                "condition": "score >= 0.8",
                "metadata": {"priority": "high"},
            },
        ]

        graph_id = tracker.send_workflow_graph_raw(
            experiment_id="exp_123",
            experiment_run_id="run_456",
            nodes=nodes,
            edges=edges,
        )

        assert graph_id == "graph_abc123"


# =============================================================================
# Tests for Graph Payload without experiment_run_id
# =============================================================================


class TestGraphPayloadOptionalFields:
    """Tests for WorkflowGraphPayload optional fields."""

    def test_to_dict_without_experiment_run_id(
        self, sample_nodes: list[WorkflowNode], sample_edges: list[WorkflowEdge]
    ) -> None:
        """Test to_dict excludes experiment_run_id when None."""
        graph = WorkflowGraphPayload(
            experiment_id="exp_123",
            nodes=sample_nodes,
            edges=sample_edges,
        )

        result = graph.to_dict()

        assert "experiment_run_id" not in result

    def test_to_dict_without_metadata(
        self, sample_nodes: list[WorkflowNode], sample_edges: list[WorkflowEdge]
    ) -> None:
        """Test to_dict excludes metadata when empty."""
        graph = WorkflowGraphPayload(
            experiment_id="exp_123",
            nodes=sample_nodes,
            edges=sample_edges,
            metadata={},  # Empty
        )

        result = graph.to_dict()

        assert "metadata" not in result

    def test_to_dict_with_metadata(
        self, sample_nodes: list[WorkflowNode], sample_edges: list[WorkflowEdge]
    ) -> None:
        """Test to_dict includes metadata when present."""
        graph = WorkflowGraphPayload(
            experiment_id="exp_123",
            nodes=sample_nodes,
            edges=sample_edges,
            metadata={"version": "1.0"},
        )

        result = graph.to_dict()

        assert "metadata" in result
        assert result["metadata"]["version"] == "1.0"


# =============================================================================
# Tests for Loop to_dict with optional fields
# =============================================================================


class TestLoopOptionalFields:
    """Tests for WorkflowLoop optional fields in to_dict."""

    def test_to_dict_without_max_iterations(self) -> None:
        """Test to_dict excludes max_iterations when None."""
        loop = WorkflowLoop(
            loop_id="loop_1",
            entry_node="node_a",
            exit_condition="complete",
        )

        result = loop.to_dict()

        assert "max_iterations" not in result

    def test_to_dict_without_metadata(self) -> None:
        """Test to_dict excludes metadata when empty."""
        loop = WorkflowLoop(
            loop_id="loop_1",
            entry_node="node_a",
            exit_condition="complete",
            metadata={},
        )

        result = loop.to_dict()

        assert "metadata" not in result

    def test_to_dict_with_all_fields(self) -> None:
        """Test to_dict includes all fields when present."""
        loop = WorkflowLoop(
            loop_id="loop_1",
            entry_node="node_a",
            exit_condition="complete",
            max_iterations=5,
            metadata={"reason": "retry"},
        )

        result = loop.to_dict()

        assert result["max_iterations"] == 5
        assert result["metadata"]["reason"] == "retry"


# =============================================================================
# Tests for SpanPayload with decision_reason
# =============================================================================


class TestSpanPayloadDecisionReason:
    """Tests for SpanPayload decision_reason field."""

    def test_to_dict_includes_decision_reason(self) -> None:
        """Test to_dict includes decision_reason when present."""
        span = SpanPayload(
            span_id="span_001",
            trace_id="trace_abc",
            configuration_run_id="config_001",
            span_name="Router Decision",
            span_type="node",
            start_time="2026-01-13T10:00:00Z",
            decision_reason="Score exceeded threshold (0.85 > 0.8)",
        )

        result = span.to_dict()

        assert "decision_reason" in result
        assert result["decision_reason"] == "Score exceeded threshold (0.85 > 0.8)"

    def test_to_dict_with_input_output_data(self) -> None:
        """Test to_dict includes input_data and output_data when present."""
        span = SpanPayload(
            span_id="span_001",
            trace_id="trace_abc",
            configuration_run_id="config_001",
            span_name="LLM Call",
            span_type="llm_call",
            start_time="2026-01-13T10:00:00Z",
            input_data={"prompt": "Hello world"},
            output_data={"response": "Hi there!"},
        )

        result = span.to_dict()

        assert "input_data" in result
        assert result["input_data"]["prompt"] == "Hello world"
        assert "output_data" in result
        assert result["output_data"]["response"] == "Hi there!"


# =============================================================================
# Tests for Edge to_dict with metadata
# =============================================================================


class TestEdgeMetadata:
    """Tests for WorkflowEdge metadata in to_dict."""

    def test_to_dict_without_metadata(self) -> None:
        """Test to_dict excludes metadata when empty."""
        edge = WorkflowEdge(from_node="a", to_node="b", metadata={})

        result = edge.to_dict()

        assert "metadata" not in result

    def test_to_dict_with_metadata(self) -> None:
        """Test to_dict includes metadata when present."""
        edge = WorkflowEdge(
            from_node="a", to_node="b", metadata={"priority": 1, "label": "high"}
        )

        result = edge.to_dict()

        assert "metadata" in result
        assert result["metadata"]["priority"] == 1


# =============================================================================
# Tests for Flush Spans Without Context
# =============================================================================


class TestFlushSpansWithoutContext:
    """Tests for flushing spans without active context."""

    def test_flush_without_context_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test _flush_spans logs warning when no context."""
        tracker = WorkflowTracesTracker(
            backend_url="http://localhost:5000",
            auto_send=False,
        )

        # Manually inject spans without context (use ContextVar directly)
        tracker._cv_spans.set(
            [
                SpanPayload(
                    span_id="span_001",
                    trace_id="trace_abc",
                    configuration_run_id="config_001",
                    span_name="Test",
                    span_type="node",
                    start_time="2026-01-13T10:00:00Z",
                )
            ]
        )

        with caplog.at_level(logging.WARNING):
            tracker._flush_spans()

        assert "No trace context" in caplog.text
