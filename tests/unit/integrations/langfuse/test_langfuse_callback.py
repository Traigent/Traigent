"""Unit tests for Langfuse optimization callback.

Tests the LangfuseOptimizationCallback class for optimization lifecycle integration.
Run with: TRAIGENT_MOCK_LLM=true pytest tests/unit/integrations/langfuse/ -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from traigent.api.types import OptimizationResult
from traigent.core.types import TrialResult, TrialStatus
from traigent.integrations.langfuse.callback import LangfuseOptimizationCallback
from traigent.integrations.langfuse.client import LangfuseClient, LangfuseTraceMetrics
from traigent.utils.callbacks import ProgressInfo


class TestLangfuseOptimizationCallbackInit:
    """Test callback initialization."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Langfuse client."""
        return MagicMock(spec=LangfuseClient)

    @pytest.fixture
    def mock_resolver(self):
        """Create mock trace resolver."""
        return MagicMock(return_value="trace-123")

    def test_init_with_client(self, mock_client, mock_resolver):
        """Test initialization with provided client."""
        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=mock_resolver,
        )
        assert callback._client is mock_client

    def test_init_with_trace_resolver(self, mock_client, mock_resolver):
        """Test initialization with trace resolver."""
        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=mock_resolver,
        )
        assert callback._trace_id_resolver is mock_resolver

    def test_init_with_options(self, mock_client, mock_resolver):
        """Test initialization with custom options."""
        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=mock_resolver,
            metric_prefix="custom_",
            include_per_agent=False,
        )
        assert callback._metric_prefix == "custom_"
        assert callback._include_per_agent is False


class TestLangfuseCallbackTrialLifecycle:
    """Test callback trial lifecycle methods."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Langfuse client."""
        return MagicMock(spec=LangfuseClient)

    @pytest.fixture
    def mock_resolver(self):
        """Create mock trace resolver that returns None."""
        return MagicMock(return_value=None)

    @pytest.fixture
    def callback(self, mock_client, mock_resolver):
        """Create callback for testing."""
        return LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=mock_resolver,
        )

    def test_on_trial_start(self, callback):
        """Test trial start callback."""
        # Should not raise - just a no-op
        callback.on_trial_start(
            trial_number=1,
            config={"model": "gpt-4o-mini", "temperature": 0.7},
        )

    def test_on_trial_complete_without_trace(self, callback):
        """Test trial complete when resolver returns None."""
        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-1"
        progress = MagicMock(spec=ProgressInfo)

        # Should not raise - skips enrichment when no trace_id
        callback.on_trial_complete(trial, progress)


class TestLangfuseCallbackOptimizationLifecycle:
    """Test callback optimization lifecycle methods."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Langfuse client."""
        return MagicMock(spec=LangfuseClient)

    @pytest.fixture
    def mock_resolver(self):
        """Create mock trace resolver."""
        return MagicMock(return_value="trace-123")

    @pytest.fixture
    def callback(self, mock_client, mock_resolver):
        """Create callback for testing."""
        return LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=mock_resolver,
        )

    def test_on_optimization_start(self, callback):
        """Test optimization start callback."""
        # Should not raise - just logs
        callback.on_optimization_start(
            config_space={"model": ["gpt-4o-mini", "gpt-4"]},
            objectives=["accuracy", "cost"],
            algorithm="optuna",
        )

    def test_on_optimization_complete(self, callback):
        """Test optimization complete callback."""
        result = MagicMock(spec=OptimizationResult)
        result.trials = [MagicMock(), MagicMock()]

        # Should not raise - just logs
        callback.on_optimization_complete(result)


class TestLangfuseCallbackMetricsEnrichment:
    """Test enrichment of trial metrics with Langfuse data."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Langfuse client."""
        return MagicMock(spec=LangfuseClient)

    def test_enriches_trial_metrics(self, mock_client):
        """Test that Langfuse metrics are added to trial."""
        mock_metrics = LangfuseTraceMetrics(
            trace_id="trace-123",
            total_cost=0.01,
            total_latency_ms=2000.0,
            total_input_tokens=200,
            total_output_tokens=100,
            total_tokens=300,
            per_agent_costs={"grader": 0.004, "generator": 0.006},
            per_agent_latencies={"grader": 800.0, "generator": 1200.0},
            per_agent_tokens={"grader": 120, "generator": 180},
            observations=[],
        )
        mock_metrics.to_measures_dict = MagicMock(
            return_value={
                "langfuse_total_cost": 0.01,
                "langfuse_total_latency_ms": 2000.0,
            }
        )
        mock_client.get_trace_metrics.return_value = mock_metrics

        def resolver(trial):
            return "trace-123"

        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=resolver,
        )

        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-1"
        trial.metrics = {"accuracy": 0.85}
        progress = MagicMock(spec=ProgressInfo)

        callback.on_trial_complete(trial, progress)

        mock_client.get_trace_metrics.assert_called_once_with("trace-123")

    def test_handles_none_trial_metrics(self, mock_client):
        """Test enrichment when trial.metrics is None."""
        mock_metrics = LangfuseTraceMetrics(
            trace_id="trace-123",
            total_cost=0.01,
            total_latency_ms=1000.0,
            total_input_tokens=100,
            total_output_tokens=50,
            total_tokens=150,
            per_agent_costs={},
            per_agent_latencies={},
            per_agent_tokens={},
            observations=[],
        )
        mock_metrics.to_measures_dict = MagicMock(
            return_value={"langfuse_total_cost": 0.01}
        )
        mock_client.get_trace_metrics.return_value = mock_metrics

        def resolver(trial):
            return "trace-123"

        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=resolver,
        )

        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-1"
        trial.metrics = None
        progress = MagicMock(spec=ProgressInfo)

        # Should not raise
        callback.on_trial_complete(trial, progress)


class TestLangfuseCallbackErrorHandling:
    """Test error handling in callback."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Langfuse client that raises errors."""
        client = MagicMock(spec=LangfuseClient)
        client.get_trace_metrics.side_effect = Exception("Network error")
        return client

    def test_handles_client_error_gracefully(self, mock_client):
        """Test callback handles client errors without crashing."""

        def resolver(trial):
            return "trace-123"

        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=resolver,
        )

        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-1"
        progress = MagicMock(spec=ProgressInfo)

        # Should not raise despite client error
        callback.on_trial_complete(trial, progress)

    def test_handles_missing_metrics_gracefully(self, mock_client):
        """Test callback handles missing metrics gracefully."""
        mock_client.get_trace_metrics.return_value = None
        mock_client.get_trace_metrics.side_effect = None  # Clear previous side effect

        def resolver(trial):
            return "trace-123"

        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=resolver,
        )

        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-1"
        progress = MagicMock(spec=ProgressInfo)

        # Should not raise
        callback.on_trial_complete(trial, progress)


class TestLangfuseCallbackMetricsPrefixOptions:
    """Test callback with different prefix and per_agent options."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Langfuse client."""
        return MagicMock(spec=LangfuseClient)

    def test_callback_with_custom_prefix(self, mock_client):
        """Test callback uses custom metric prefix."""
        # Use MagicMock for metrics so we can track calls
        mock_metrics = MagicMock()
        mock_metrics.to_measures_dict.return_value = {"lf_total_cost": 0.01}
        mock_client.get_trace_metrics.return_value = mock_metrics

        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=lambda t: "trace-123",
            metric_prefix="lf_",
        )

        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-1"
        trial.metrics = {}
        progress = MagicMock(spec=ProgressInfo)

        callback.on_trial_complete(trial, progress)

        # Verify to_measures_dict was called with custom prefix
        mock_metrics.to_measures_dict.assert_called_once()
        call_args = mock_metrics.to_measures_dict.call_args
        assert call_args[1]["prefix"] == "lf_"

    def test_callback_without_per_agent(self, mock_client):
        """Test callback with per_agent disabled."""
        # Use MagicMock for metrics so we can track calls
        mock_metrics = MagicMock()
        mock_metrics.to_measures_dict.return_value = {"langfuse_total_cost": 0.01}
        mock_client.get_trace_metrics.return_value = mock_metrics

        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=lambda t: "trace-123",
            include_per_agent=False,
        )

        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-1"
        trial.metrics = {}
        progress = MagicMock(spec=ProgressInfo)

        callback.on_trial_complete(trial, progress)

        # Verify to_measures_dict was called with include_per_agent=False
        mock_metrics.to_measures_dict.assert_called_once()
        call_args = mock_metrics.to_measures_dict.call_args
        assert call_args[1]["include_per_agent"] is False

    def test_callback_default_prefix(self, mock_client):
        """Test callback uses default langfuse_ prefix."""
        # Use MagicMock for metrics so we can track calls
        mock_metrics = MagicMock()
        mock_metrics.to_measures_dict.return_value = {"langfuse_total_cost": 0.01}
        mock_client.get_trace_metrics.return_value = mock_metrics

        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=lambda t: "trace-123",
        )

        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-1"
        trial.metrics = {}
        progress = MagicMock(spec=ProgressInfo)

        callback.on_trial_complete(trial, progress)

        # Verify default prefix
        call_args = mock_metrics.to_measures_dict.call_args
        assert call_args[1]["prefix"] == "langfuse_"


class TestLangfuseCallbackMetricsMerging:
    """Test that metrics are properly merged into trial."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Langfuse client."""
        client = MagicMock(spec=LangfuseClient)
        mock_metrics = MagicMock()
        mock_metrics.to_measures_dict.return_value = {
            "langfuse_total_cost": 0.01,
            "langfuse_total_tokens": 100,
        }
        client.get_trace_metrics.return_value = mock_metrics
        return client

    def test_metrics_merged_into_existing(self, mock_client):
        """Test Langfuse metrics are merged with existing trial metrics."""
        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=lambda t: "trace-123",
        )

        # Create real dict for metrics that supports |=
        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-1"
        trial.metrics = {"accuracy": 0.9, "existing": 42}
        progress = MagicMock(spec=ProgressInfo)

        callback.on_trial_complete(trial, progress)

        # Verify metrics were merged (|= operation)
        assert trial.metrics is not None

    def test_metrics_created_when_none(self, mock_client):
        """Test metrics dict is created when trial.metrics is None."""
        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=lambda t: "trace-123",
        )

        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-1"
        trial.metrics = None
        progress = MagicMock(spec=ProgressInfo)

        callback.on_trial_complete(trial, progress)

        # Verify metrics dict was created
        assert trial.metrics is not None
