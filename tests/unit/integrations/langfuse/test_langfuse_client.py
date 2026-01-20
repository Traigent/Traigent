"""Unit tests for Langfuse client.

Tests the LangfuseClient class for trace retrieval and metric extraction.
Run with: TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/langfuse/ -v
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from traigent.integrations.langfuse.client import (
    AIOHTTP_AVAILABLE,
    LANGFUSE_SDK_AVAILABLE,
    REQUESTS_AVAILABLE,
    LangfuseClient,
    LangfuseObservation,
    LangfuseTraceMetrics,
)


class TestLangfuseObservation:
    """Test LangfuseObservation data class."""

    def test_basic_creation(self):
        """Test creating a basic observation."""
        obs = LangfuseObservation(
            id="obs-123",
            name="test-observation",
            observation_type="generation",
        )
        assert obs.id == "obs-123"
        assert obs.name == "test-observation"
        assert obs.observation_type == "generation"
        assert obs.input_tokens == 0
        assert obs.output_tokens == 0
        assert obs.cost == 0.0

    def test_full_observation(self):
        """Test observation with all fields populated."""
        now = datetime.now(timezone.utc)
        obs = LangfuseObservation(
            id="obs-456",
            name="llm-call",
            observation_type="generation",
            start_time=now,
            end_time=now,
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.002,
            latency_ms=500.0,
            status="success",
            langgraph_node="grader",
            langgraph_step=2,
            metadata={"custom": "value"},
        )
        assert obs.model == "gpt-4o-mini"
        assert obs.total_tokens == 150
        assert obs.langgraph_node == "grader"

    def test_get_agent_identifier_openinference_priority(self):
        """Test that OpenInference node_id takes priority."""
        obs = LangfuseObservation(
            id="obs-1",
            name="observation-name",
            observation_type="span",
            langgraph_node="grader",
            openinference_node_id="oi-agent-1",
        )
        assert obs.get_agent_identifier() == "oi-agent-1"

    def test_get_agent_identifier_langgraph_fallback(self):
        """Test fallback to langgraph_node."""
        obs = LangfuseObservation(
            id="obs-1",
            name="observation-name",
            observation_type="span",
            langgraph_node="generator",
        )
        assert obs.get_agent_identifier() == "generator"

    def test_get_agent_identifier_name_fallback(self):
        """Test fallback to observation name."""
        obs = LangfuseObservation(
            id="obs-1",
            name="my_agent_name",
            observation_type="span",
        )
        assert obs.get_agent_identifier() == "my_agent_name"

    def test_get_agent_identifier_excludes_generic_names(self):
        """Test that generic names like LLMChain are not used as agent IDs."""
        obs = LangfuseObservation(
            id="obs-1",
            name="LLMChain",
            observation_type="span",
        )
        # Should return None for generic names
        assert obs.get_agent_identifier() is None


class TestLangfuseTraceMetrics:
    """Test LangfuseTraceMetrics data class."""

    @pytest.fixture
    def sample_metrics(self) -> LangfuseTraceMetrics:
        """Create sample metrics for testing."""
        return LangfuseTraceMetrics(
            trace_id="trace-abc",
            total_cost=0.01,
            total_latency_ms=2000.0,
            total_input_tokens=200,
            total_output_tokens=100,
            total_tokens=300,
            per_agent_costs={"grader": 0.003, "generator": 0.007},
            per_agent_latencies={"grader": 800.0, "generator": 1200.0},
            per_agent_tokens={"grader": 100, "generator": 200},
            observations=[],
        )

    def test_to_measures_dict_basic(self, sample_metrics: LangfuseTraceMetrics):
        """Test basic measures dict conversion."""
        measures = sample_metrics.to_measures_dict()

        assert "total_cost" in measures
        assert "total_latency_ms" in measures
        assert "total_tokens" in measures
        assert measures["total_cost"] == 0.01
        assert measures["total_latency_ms"] == 2000.0

    def test_to_measures_dict_per_agent_metrics(
        self, sample_metrics: LangfuseTraceMetrics
    ):
        """Test per-agent metrics in measures dict."""
        measures = sample_metrics.to_measures_dict()

        # Per-agent costs
        assert "grader_cost" in measures
        assert "generator_cost" in measures
        assert measures["grader_cost"] == 0.003
        assert measures["generator_cost"] == 0.007

        # Per-agent latencies
        assert "grader_latency_ms" in measures
        assert "generator_latency_ms" in measures

        # Per-agent tokens
        assert "grader_tokens" in measures
        assert "generator_tokens" in measures

    def test_to_measures_dict_with_prefix(self, sample_metrics: LangfuseTraceMetrics):
        """Test measures dict with prefix."""
        measures = sample_metrics.to_measures_dict(prefix="langfuse_")

        assert "langfuse_total_cost" in measures
        assert "langfuse_grader_cost" in measures
        assert measures["langfuse_total_cost"] == 0.01

    def test_to_measures_dict_without_per_agent(
        self, sample_metrics: LangfuseTraceMetrics
    ):
        """Test measures dict without per-agent metrics."""
        measures = sample_metrics.to_measures_dict(include_per_agent=False)

        assert "total_cost" in measures
        assert "grader_cost" not in measures
        assert "generator_cost" not in measures

    def test_to_measures_dict_no_dots_in_keys(
        self, sample_metrics: LangfuseTraceMetrics
    ):
        """Verify MeasuresDict constraint - no dots in keys."""
        measures = sample_metrics.to_measures_dict()
        for key in measures:
            assert "." not in key, f"Key '{key}' contains dot"

    def test_to_measures_dict_valid_identifiers(
        self, sample_metrics: LangfuseTraceMetrics
    ):
        """Verify keys are valid Python identifiers."""
        measures = sample_metrics.to_measures_dict()
        import re

        pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        for key in measures:
            assert re.match(
                pattern, key
            ), f"Key '{key}' is not a valid Python identifier"

    def test_empty_metrics(self):
        """Test metrics with no per-agent data."""
        metrics = LangfuseTraceMetrics(
            trace_id="trace-empty",
            total_cost=0.0,
            total_latency_ms=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            per_agent_costs={},
            per_agent_latencies={},
            per_agent_tokens={},
            observations=[],
        )
        measures = metrics.to_measures_dict()
        assert measures["total_cost"] == 0.0
        assert "grader_cost" not in measures


class TestLangfuseClientInit:
    """Test LangfuseClient initialization."""

    def test_init_with_explicit_keys(self):
        """Test initialization with explicit keys."""
        client = LangfuseClient(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://custom.langfuse.com",
        )
        assert client.public_key == "pk-test"
        assert client.secret_key == "sk-test"
        assert client.host == "https://custom.langfuse.com"

    def test_init_from_environment(self, monkeypatch):
        """Test initialization from environment variables."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-env")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-env")
        monkeypatch.setenv("LANGFUSE_HOST", "https://env.langfuse.com")

        client = LangfuseClient()
        assert client.public_key == "pk-env"
        assert client.secret_key == "sk-env"
        assert client.host == "https://env.langfuse.com"

    def test_init_allows_missing_keys(self, monkeypatch):
        """Test that init allows missing keys (validation happens later)."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        # Should not raise - keys validated when used
        client = LangfuseClient()
        assert client.public_key is None
        assert client.secret_key is None

    def test_auth_header_requires_keys(self, monkeypatch):
        """Test that _get_auth_header raises when keys are missing."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        client = LangfuseClient()
        with pytest.raises(ValueError, match="public_key and secret_key are required"):
            client._get_auth_header()

    def test_default_host(self, monkeypatch):
        """Test default host when not specified."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)

        client = LangfuseClient()
        assert client.host == "https://cloud.langfuse.com"


class TestLangfuseClientTraceRetrieval:
    """Test trace retrieval methods."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_get_trace_returns_none_for_404(self, client):
        """Test that get_trace returns None for 404."""
        with patch.object(client, "_get_trace_http") as mock_http:
            mock_http.return_value = None
            result = client.get_trace("nonexistent-trace")
            assert result is None

    @pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not installed")
    def test_get_trace_http_success(self, client):
        """Test successful HTTP trace retrieval."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "trace-123",
            "name": "test-trace",
            "observations": [],
        }

        with patch("requests.get", return_value=mock_response):
            result = client._get_trace_http("trace-123")
            assert result is not None
            assert result["id"] == "trace-123"

    @pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not installed")
    def test_get_trace_http_404(self, client):
        """Test HTTP 404 handling."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("requests.get", return_value=mock_response):
            result = client._get_trace_http("nonexistent")
            assert result is None


class TestLangfuseClientMetricExtraction:
    """Test metric extraction from traces."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_dict_to_observation_empty(self, client):
        """Test conversion of minimal observation dict."""
        obs_data = {"id": "obs-empty", "name": "empty", "type": "span"}
        obs = client._dict_to_observation(obs_data)
        assert obs.id == "obs-empty"
        assert obs.name == "empty"
        assert obs.observation_type == "span"
        assert obs.input_tokens == 0
        assert obs.output_tokens == 0

    def test_dict_to_observation_with_usage(self, client):
        """Test conversion with usage data."""
        obs_data = {
            "id": "obs-1",
            "name": "llm-call",
            "type": "GENERATION",
            "model": "gpt-4o-mini",
            "usage": {"input": 100, "output": 50, "total": 150},
            "calculatedTotalCost": 0.002,
            "latency": 0.5,  # 500ms in seconds
        }
        obs = client._dict_to_observation(obs_data)
        assert obs.name == "llm-call"
        assert obs.input_tokens == 100
        assert obs.output_tokens == 50
        assert obs.total_tokens == 150
        assert obs.cost == 0.002
        assert obs.latency_ms == 500.0  # Converted to ms

    def test_dict_to_observation_with_metadata(self, client):
        """Test conversion with langgraph metadata."""
        obs_data = {
            "id": "obs-1",
            "name": "agent-span",
            "type": "SPAN",
            "metadata": {
                "langgraph_node": "grader",
                "langgraph_step": 2,
            },
        }
        obs = client._dict_to_observation(obs_data)
        assert obs.langgraph_node == "grader"
        assert obs.langgraph_step == 2

    def test_dict_to_observation_with_openinference(self, client):
        """Test conversion with OpenInference metadata."""
        obs_data = {
            "id": "obs-1",
            "name": "agent-span",
            "type": "SPAN",
            "metadata": {
                "graph.node.id": "oi-agent-1",
            },
        }
        obs = client._dict_to_observation(obs_data)
        assert obs.openinference_node_id == "oi-agent-1"

    def test_extract_metrics_from_trace_empty(self, client):
        """Test extraction from trace with no observations."""
        trace_data = {"id": "trace-empty", "observations": []}
        metrics = client._extract_metrics_from_trace(trace_data)
        assert metrics.trace_id == "trace-empty"
        assert metrics.total_cost == 0.0
        assert metrics.total_tokens == 0
        assert metrics.per_agent_costs == {}

    def test_extract_metrics_from_trace_single_observation(self, client):
        """Test extraction with single observation."""
        trace_data = {
            "id": "trace-123",
            "observations": [
                {
                    "id": "obs-1",
                    "name": "llm-call",
                    "type": "GENERATION",
                    "model": "gpt-4o-mini",
                    "usage": {"input": 100, "output": 50, "total": 150},
                    "calculatedTotalCost": 0.002,
                    "latency": 0.5,
                }
            ],
        }
        metrics = client._extract_metrics_from_trace(trace_data)
        assert metrics.trace_id == "trace-123"
        assert metrics.total_cost == 0.002
        assert metrics.total_tokens == 150
        assert metrics.total_latency_ms == 500.0

    def test_extract_metrics_from_trace_with_agents(self, client):
        """Test extraction with per-agent metrics."""
        trace_data = {
            "id": "trace-123",
            "observations": [
                {
                    "id": "obs-1",
                    "name": "grader",
                    "type": "GENERATION",
                    "usage": {"total": 100},
                    "calculatedTotalCost": 0.002,
                    "latency": 0.5,
                    "metadata": {"langgraph_node": "grader"},
                },
                {
                    "id": "obs-2",
                    "name": "generator",
                    "type": "GENERATION",
                    "usage": {"total": 150},
                    "calculatedTotalCost": 0.003,
                    "latency": 0.7,
                    "metadata": {"langgraph_node": "generator"},
                },
            ],
        }
        metrics = client._extract_metrics_from_trace(trace_data)

        assert metrics.total_cost == 0.005
        assert metrics.per_agent_costs["grader"] == 0.002
        assert metrics.per_agent_costs["generator"] == 0.003
        assert metrics.per_agent_latencies["grader"] == 500.0
        assert metrics.per_agent_latencies["generator"] == 700.0
        assert metrics.per_agent_tokens["grader"] == 100
        assert metrics.per_agent_tokens["generator"] == 150


class TestLangfuseClientThreadSafety:
    """Test thread safety of LangfuseClient."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_client_has_lock(self, client):
        """Test that client has thread lock."""
        assert hasattr(client, "_lock")
        assert isinstance(client._lock, type(threading.Lock()))


class TestLangfuseClientAuthHeader:
    """Test authentication header generation."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_get_auth_header(self, client):
        """Test auth header is properly formatted."""
        header = client._get_auth_header()
        assert "Authorization" in header
        assert header["Authorization"].startswith("Basic ")

    def test_get_auth_header_encoding(self, client):
        """Test auth header contains properly encoded credentials."""
        import base64

        header = client._get_auth_header()
        # Extract and decode the Base64 portion
        auth_value = header["Authorization"].replace("Basic ", "")
        decoded = base64.b64decode(auth_value).decode()
        assert decoded == "pk-test:sk-test"


class TestLangfuseClientGetTraceMetrics:
    """Test get_trace_metrics method."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_get_trace_metrics_not_found(self, client):
        """Test get_trace_metrics returns None for missing trace."""
        with patch.object(client, "get_trace", return_value=None):
            result = client.get_trace_metrics("nonexistent")
            assert result is None

    def test_get_trace_metrics_success(self, client):
        """Test get_trace_metrics returns metrics for valid trace."""
        trace_data = {
            "id": "trace-123",
            "name": "test-trace",
            "observations": [
                {
                    "id": "obs-1",
                    "name": "llm-call",
                    "type": "GENERATION",
                    "usage": {"total": 100},
                    "calculatedTotalCost": 0.002,
                }
            ],
        }
        with patch.object(client, "get_trace", return_value=trace_data):
            result = client.get_trace_metrics("trace-123")
            assert result is not None
            assert result.trace_id == "trace-123"
            assert result.total_cost == 0.002
            assert result.total_tokens == 100
