"""Unit tests for Langfuse client.

Tests the LangfuseClient class for trace retrieval and metric extraction.
Run with: TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/langfuse/ -v
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.integrations.langfuse.client import (
    AIOHTTP_AVAILABLE,
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
        now = datetime.now(UTC)
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
            secret_key="sk-test",  # pragma: allowlist secret
            host="https://custom.langfuse.com",
        )
        assert client.public_key == "pk-test"
        assert client.secret_key == "sk-test"  # pragma: allowlist secret
        assert client.host == "https://custom.langfuse.com"

    def test_init_from_environment(self, monkeypatch):
        """Test initialization from environment variables."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-env")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-env")
        monkeypatch.setenv("LANGFUSE_HOST", "https://env.langfuse.com")

        client = LangfuseClient()
        assert client.public_key == "pk-env"
        assert client.secret_key == "sk-env"  # pragma: allowlist secret
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
    """Test lightweight concurrency assumptions of LangfuseClient."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_client_does_not_carry_unused_lock(self, client):
        """The client should not expose unused synchronization primitives."""
        assert not hasattr(client, "_lock")


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


class TestLangfuseClientTimestampParsing:
    """Test timestamp parsing utilities."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_parse_timestamp_none(self, client):
        """Test parsing None timestamp."""
        result = client._parse_timestamp(None)
        assert result is None

    def test_parse_timestamp_datetime_object(self, client):
        """Test parsing datetime object."""
        now = datetime.now(UTC)
        result = client._parse_timestamp(now)
        assert result == now

    def test_parse_timestamp_iso_string(self, client):
        """Test parsing ISO format string."""
        ts_str = "2024-01-15T10:30:00Z"
        result = client._parse_timestamp(ts_str)
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_timestamp_invalid_string(self, client):
        """Test parsing invalid string returns None."""
        result = client._parse_timestamp("not-a-timestamp")
        assert result is None


class TestLangfuseClientTraceConversion:
    """Test trace object to dict conversion."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_trace_to_dict_with_dict_input(self, client):
        """Test _trace_to_dict with dict input passes through."""
        trace_dict = {"id": "trace-123", "name": "test"}
        result = client._trace_to_dict(trace_dict)
        assert result == trace_dict

    def test_trace_to_dict_with_object(self, client):
        """Test _trace_to_dict with SDK-like object."""
        mock_trace = MagicMock()
        mock_trace.id = "trace-123"
        mock_trace.name = "test-trace"
        mock_trace.metadata = {"key": "value"}
        mock_trace.session_id = "session-1"
        mock_trace.user_id = "user-1"
        mock_trace.input = "input text"
        mock_trace.output = "output text"
        mock_trace.observations = []

        result = client._trace_to_dict(mock_trace)
        assert result["id"] == "trace-123"
        assert result["name"] == "test-trace"
        assert result["sessionId"] == "session-1"

    def test_observation_to_dict_with_dict_input(self, client):
        """Test _observation_to_dict with dict input passes through."""
        obs_dict = {"id": "obs-123", "name": "test"}
        result = client._observation_to_dict(obs_dict)
        assert result == obs_dict


class TestLangfuseClientObservationsRetrieval:
    """Test observations retrieval."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    @pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not installed")
    def test_get_observations_http_single_page(self, client):
        """Test getting observations via HTTP with single page."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "obs-1", "name": "llm-call", "type": "GENERATION"},
                {"id": "obs-2", "name": "embed", "type": "SPAN"},
            ],
            "meta": {"totalItems": 2},
        }

        with patch("requests.get", return_value=mock_response):
            result = client._get_observations_http("trace-123")
            assert len(result) == 2
            assert result[0].id == "obs-1"
            assert result[1].id == "obs-2"

    @pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not installed")
    def test_get_observations_http_empty(self, client):
        """Test getting observations when none exist."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [], "meta": {"totalItems": 0}}

        with patch("requests.get", return_value=mock_response):
            result = client._get_observations_http("trace-123")
            assert result == []


class TestLangfuseClientWaitForTrace:
    """Test wait_for_trace functionality."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_wait_for_trace_found_immediately(self, client):
        """Test wait_for_trace when trace is immediately available."""
        with (
            patch.object(client, "get_trace", return_value={"id": "trace-123"}),
            patch.object(
                client, "get_observations_for_trace", return_value=[MagicMock()]
            ),
        ):
            result = client.wait_for_trace("trace-123", timeout_seconds=5.0)
            assert result is True

    def test_wait_for_trace_timeout(self, client):
        """Test wait_for_trace when trace is never found."""
        with patch.object(client, "get_trace", return_value=None):
            result = client.wait_for_trace(
                "trace-123", timeout_seconds=0.1, poll_interval=0.05
            )
            assert result is False


class TestLangfuseClientErrorHandling:
    """Test error handling in client methods."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    @pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not installed")
    def test_get_trace_http_handles_request_exception(self, client):
        """Test that HTTP errors are handled gracefully."""
        import requests

        with patch(
            "requests.get", side_effect=requests.exceptions.ConnectionError("Failed")
        ):
            result = client._get_trace_http("trace-123")
            assert result is None

    @pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not installed")
    def test_get_observations_http_handles_request_exception(self, client):
        """Test that observations HTTP errors return partial results."""
        import requests

        with patch("requests.get", side_effect=requests.exceptions.Timeout("Timeout")):
            result = client._get_observations_http("trace-123")
            assert result == []


class TestLangfuseClientGetTrace:
    """Test get_trace method with SDK fallback."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_get_trace_falls_back_to_http(self, client):
        """Test get_trace falls back to HTTP when SDK unavailable."""
        client._sdk_client = None
        with patch.object(client, "_get_trace_http") as mock_http:
            mock_http.return_value = {"id": "trace-123"}
            result = client.get_trace("trace-123")
            assert result is not None
            mock_http.assert_called_once_with("trace-123")

    def test_get_trace_sdk_exception_falls_back(self, client):
        """Test get_trace falls back to HTTP on SDK exception."""
        mock_sdk = MagicMock()
        mock_sdk.get_trace.side_effect = Exception("SDK error")
        client._sdk_client = mock_sdk

        with patch.object(client, "_get_trace_http") as mock_http:
            mock_http.return_value = {"id": "trace-123"}
            result = client.get_trace("trace-123")
            assert result is not None
            mock_http.assert_called_once()


class TestLangfuseClientMetricsAggregation:
    """Test metrics aggregation edge cases."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_extract_metrics_with_trace_metadata(self, client):
        """Test extraction preserves trace-level metadata."""
        trace_data = {
            "id": "trace-123",
            "name": "my-workflow",
            "metadata": {"environment": "production"},
            "sessionId": "session-abc",
            "userId": "user-xyz",
            "observations": [],
        }
        metrics = client._extract_metrics_from_trace(trace_data)
        assert metrics.trace_name == "my-workflow"
        assert metrics.session_id == "session-abc"
        assert metrics.user_id == "user-xyz"
        assert metrics.trace_metadata == {"environment": "production"}

    def test_extract_metrics_fetches_observations_when_missing(self, client):
        """Test that observations are fetched if not in trace data."""
        trace_data = {"id": "trace-123", "name": "test"}  # No observations key
        with patch.object(client, "get_observations_for_trace") as mock_get_obs:
            mock_get_obs.return_value = []
            client._extract_metrics_from_trace(trace_data)
            mock_get_obs.assert_called_once_with("trace-123")

    def test_aggregate_multiple_agents_accumulates(self, client):
        """Test that metrics accumulate correctly for same agent."""
        trace_data = {
            "id": "trace-123",
            "observations": [
                {
                    "id": "obs-1",
                    "name": "grader-1",
                    "type": "GENERATION",
                    "usage": {"total": 50},
                    "calculatedTotalCost": 0.001,
                    "metadata": {"langgraph_node": "grader"},
                },
                {
                    "id": "obs-2",
                    "name": "grader-2",
                    "type": "GENERATION",
                    "usage": {"total": 75},
                    "calculatedTotalCost": 0.002,
                    "metadata": {"langgraph_node": "grader"},
                },
            ],
        }
        metrics = client._extract_metrics_from_trace(trace_data)
        # Same agent should accumulate
        assert metrics.per_agent_costs["grader"] == 0.003
        assert metrics.per_agent_tokens["grader"] == 125


class TestLangfuseClientUsageFormats:
    """Test handling of different usage data formats."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_dict_to_observation_promptTokens_format(self, client):
        """Test parsing promptTokens/completionTokens format."""
        obs_data = {
            "id": "obs-1",
            "name": "llm",
            "type": "GENERATION",
            "usage": {"promptTokens": 80, "completionTokens": 40, "totalTokens": 120},
        }
        obs = client._dict_to_observation(obs_data)
        assert obs.input_tokens == 80
        assert obs.output_tokens == 40
        assert obs.total_tokens == 120

    def test_dict_to_observation_calculates_total_from_parts(self, client):
        """Test that total is calculated if not provided."""
        obs_data = {
            "id": "obs-1",
            "name": "llm",
            "type": "GENERATION",
            "usage": {"input": 60, "output": 30},  # No total
        }
        obs = client._dict_to_observation(obs_data)
        assert obs.input_tokens == 60
        assert obs.output_tokens == 30
        assert obs.total_tokens == 90  # Calculated

    def test_dict_to_observation_latency_from_timestamps(self, client):
        """Test latency calculation from start/end times."""
        obs_data = {
            "id": "obs-1",
            "name": "llm",
            "type": "GENERATION",
            "startTime": "2024-01-15T10:00:00Z",
            "endTime": "2024-01-15T10:00:01.500Z",  # 1.5 seconds later
        }
        obs = client._dict_to_observation(obs_data)
        assert obs.latency_ms == 1500.0

    def test_dict_to_observation_error_status(self, client):
        """Test that ERROR level is mapped to error status."""
        obs_data = {
            "id": "obs-1",
            "name": "llm",
            "type": "GENERATION",
            "level": "ERROR",
        }
        obs = client._dict_to_observation(obs_data)
        assert obs.status == "error"


class TestLangfuseClientAsync:
    """Test async methods of LangfuseClient."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_get_trace_async_method_exists(self, client):
        """Test that async trace retrieval method exists."""
        # Just verify the method exists and has correct signature
        assert hasattr(client, "get_trace_async")
        assert callable(client.get_trace_async)

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
    @pytest.mark.asyncio
    async def test_get_trace_metrics_async(self, client):
        """Test async metrics retrieval."""
        # Mock get_trace_async
        trace_data = {
            "id": "trace-123",
            "observations": [
                {
                    "id": "obs-1",
                    "name": "llm",
                    "type": "GENERATION",
                    "usage": {"total": 100},
                }
            ],
        }
        with patch.object(client, "get_trace_async", return_value=trace_data):
            result = await client.get_trace_metrics_async("trace-123")
            assert result is not None
            assert result.trace_id == "trace-123"

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
    @pytest.mark.asyncio
    async def test_get_trace_metrics_async_not_found(self, client):
        """Test async metrics when trace not found."""
        with patch.object(client, "get_trace_async", return_value=None):
            result = await client.get_trace_metrics_async("nonexistent")
            assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_trace_async_avoids_deprecated_get_event_loop(self, client):
        """wait_for_trace_async should use the running loop, not get_event_loop."""
        with (
            patch(
                "asyncio.get_event_loop",
                side_effect=AssertionError("deprecated API should not be used"),
            ),
            patch.object(
                client, "get_trace_async", new=AsyncMock(return_value={"id": "trace-1"})
            ),
            patch.object(
                client,
                "get_observations_for_trace_async",
                new=AsyncMock(return_value=[MagicMock()]),
            ),
        ):
            assert (
                await client.wait_for_trace_async(
                    "trace-1", timeout_seconds=0.1, poll_interval=0.0
                )
                is True
            )


class TestLangfuseClientSDK:
    """Test SDK integration paths."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_get_observations_for_trace_sdk_fallback(self, client):
        """Test get_observations_for_trace with SDK fallback."""
        client._sdk_client = None
        with patch.object(client, "_get_observations_http") as mock_http:
            mock_http.return_value = []
            client.get_observations_for_trace("trace-123")
            mock_http.assert_called_once()

    def test_get_trace_with_sdk_success(self, client):
        """Test get_trace using SDK."""
        mock_sdk = MagicMock()
        mock_trace = MagicMock()
        mock_trace.id = "trace-123"
        mock_trace.name = "test"
        mock_trace.observations = []
        mock_sdk.get_trace.return_value = mock_trace
        client._sdk_client = mock_sdk

        result = client.get_trace("trace-123")
        assert result is not None
        assert result["id"] == "trace-123"
        mock_sdk.get_trace.assert_called_once_with("trace-123")

    def test_get_observations_for_trace_sdk_success(self, client):
        """Test get_observations_for_trace using SDK."""
        mock_sdk = MagicMock()
        mock_obs = MagicMock()
        mock_obs.id = "obs-123"
        mock_obs.name = "test"
        mock_obs.type = "GENERATION"
        mock_sdk.get_observations.return_value = MagicMock(data=[mock_obs])
        client._sdk_client = mock_sdk

        result = client.get_observations_for_trace("trace-123")
        assert len(result) == 1
        mock_sdk.get_observations.assert_called_once()

    def test_get_observations_for_trace_sdk_exception(self, client):
        """Test get_observations_for_trace falls back on SDK exception."""
        mock_sdk = MagicMock()
        mock_sdk.get_observations.side_effect = Exception("SDK error")
        client._sdk_client = mock_sdk

        with patch.object(client, "_get_observations_http") as mock_http:
            mock_http.return_value = []
            client.get_observations_for_trace("trace-123")
            mock_http.assert_called_once()


class TestLangfuseAgentSanitization:
    """Test agent name sanitization in metrics."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a client for testing."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        return LangfuseClient()

    def test_agent_name_with_dots_sanitized(self, client):
        """Test that dots in agent names are replaced with underscores."""
        trace_data = {
            "id": "trace-123",
            "observations": [
                {
                    "id": "obs-1",
                    "name": "agent.name.with.dots",
                    "type": "GENERATION",
                    "usage": {"total": 100},
                    "calculatedTotalCost": 0.001,
                    "metadata": {"langgraph_node": "agent.name.with.dots"},
                }
            ],
        }
        metrics = client._extract_metrics_from_trace(trace_data)
        # Dots should be converted to underscores
        assert "agent_name_with_dots" in metrics.per_agent_costs

    def test_agent_name_with_dashes_sanitized(self, client):
        """Test that dashes in agent names are replaced with underscores."""
        trace_data = {
            "id": "trace-123",
            "observations": [
                {
                    "id": "obs-1",
                    "name": "agent-with-dashes",
                    "type": "GENERATION",
                    "usage": {"total": 100},
                    "calculatedTotalCost": 0.001,
                    "metadata": {"langgraph_node": "agent-with-dashes"},
                }
            ],
        }
        metrics = client._extract_metrics_from_trace(trace_data)
        # Dashes should be converted to underscores
        assert "agent_with_dashes" in metrics.per_agent_costs

    def test_to_measures_dict_sanitizes_agent_names(self):
        """Test that to_measures_dict sanitizes agent names."""
        metrics = LangfuseTraceMetrics(
            trace_id="trace-123",
            per_agent_costs={"agent.with.dots": 0.001, "agent-with-dashes": 0.002},
            per_agent_latencies={},
            per_agent_tokens={},
        )
        measures = metrics.to_measures_dict()
        assert "agent_with_dots_cost" in measures
        assert "agent_with_dashes_cost" in measures
