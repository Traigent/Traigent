"""Integration tests for Langfuse bridge.

Run with: TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true pytest tests/integration/langfuse/ -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from traigent.cloud.dtos import MeasuresDict
from traigent.core.types import TrialResult, TrialStatus
from traigent.integrations.langfuse import (
    LangfuseClient,
    LangfuseOptimizationCallback,
    LangfuseTraceMetrics,
    LangfuseTracker,
    create_langfuse_tracker,
)
from traigent.utils.callbacks import ProgressInfo


class TestLangfuseTraceMetrics:
    """Test LangfuseTraceMetrics data model."""

    @pytest.fixture
    def sample_metrics(self) -> LangfuseTraceMetrics:
        """Create sample metrics for testing."""
        return LangfuseTraceMetrics(
            trace_id="test-trace-123",
            total_cost=0.006,
            total_latency_ms=1200.0,
            total_input_tokens=100,
            total_output_tokens=50,
            total_tokens=150,
            per_agent_costs={"grader": 0.002, "generator": 0.004},
            per_agent_latencies={"grader": 400.0, "generator": 800.0},
            per_agent_tokens={"grader": 50, "generator": 100},
            observations=[],
        )

    def test_to_measures_dict_underscore_naming(
        self, sample_metrics: LangfuseTraceMetrics
    ):
        """Verify MeasuresDict-compliant output with underscore naming."""
        measures = sample_metrics.to_measures_dict()

        # Verify no dots in keys (MeasuresDict constraint)
        for key in measures:
            assert (
                "." not in key
            ), f"Key '{key}' contains dot (invalid for MeasuresDict)"
            assert (
                "-" not in key
            ), f"Key '{key}' contains hyphen (invalid for MeasuresDict)"

        # Verify expected keys exist
        assert "total_cost" in measures
        assert "grader_cost" in measures
        assert "generator_latency_ms" in measures

        # Verify numeric values
        for value in measures.values():
            assert isinstance(value, (int, float)), f"Value {value} is not numeric"

        # Verify can construct MeasuresDict without error
        validated = MeasuresDict(measures)
        assert len(validated) > 0

    def test_to_measures_dict_with_prefix(self, sample_metrics: LangfuseTraceMetrics):
        """Verify prefix is applied to all metric keys."""
        measures = sample_metrics.to_measures_dict(prefix="langfuse_")

        # All keys should start with prefix
        for key in measures:
            assert key.startswith("langfuse_"), f"Key '{key}' missing prefix"

        # Verify specific keys
        assert "langfuse_total_cost" in measures
        assert "langfuse_grader_cost" in measures
        assert measures["langfuse_total_cost"] == 0.006

    def test_to_measures_dict_without_per_agent(
        self, sample_metrics: LangfuseTraceMetrics
    ):
        """Verify include_per_agent=False excludes per-agent metrics."""
        measures = sample_metrics.to_measures_dict(include_per_agent=False)

        # Should have total metrics
        assert "total_cost" in measures
        assert "total_latency_ms" in measures

        # Should NOT have per-agent metrics
        assert "grader_cost" not in measures
        assert "generator_cost" not in measures
        assert "grader_latency_ms" not in measures

    def test_to_measures_dict_sanitizes_agent_names(self):
        """Verify agent names with special chars are sanitized."""
        metrics = LangfuseTraceMetrics(
            trace_id="test",
            total_cost=0.01,
            total_latency_ms=100.0,
            total_input_tokens=10,
            total_output_tokens=5,
            total_tokens=15,
            per_agent_costs={"my-agent": 0.005, "other.agent": 0.005},
            per_agent_latencies={},
            per_agent_tokens={},
            observations=[],
        )

        measures = metrics.to_measures_dict()

        # Dashes and dots should be replaced with underscores
        assert "my_agent_cost" in measures
        assert "other_agent_cost" in measures
        assert "my-agent_cost" not in measures
        assert "other.agent_cost" not in measures


class TestLangfuseOptimizationCallback:
    """Test callback integration with orchestrator."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create mock LangfuseClient."""
        client = MagicMock(spec=LangfuseClient)
        client.get_trace_metrics = MagicMock(
            return_value=LangfuseTraceMetrics(
                trace_id="trace-123",
                total_cost=0.005,
                total_latency_ms=500.0,
                total_input_tokens=100,
                total_output_tokens=50,
                total_tokens=150,
                per_agent_costs={"agent1": 0.003, "agent2": 0.002},
                per_agent_latencies={"agent1": 200.0, "agent2": 300.0},
                per_agent_tokens={"agent1": 80, "agent2": 70},
                observations=[],
            )
        )
        return client

    @pytest.fixture
    def mock_trial(self) -> MagicMock:
        """Create mock TrialResult."""
        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-1"
        trial.metadata = {"langfuse_trace_id": "trace-123"}
        trial.metrics = {"accuracy": 0.95}
        trial.config = {"temperature": 0.7}
        trial.status = TrialStatus.COMPLETED
        return trial

    @pytest.fixture
    def mock_progress(self) -> MagicMock:
        """Create mock ProgressInfo."""
        progress = MagicMock(spec=ProgressInfo)
        progress.completed_trials = 1
        progress.total_trials = 10
        return progress

    def test_on_trial_complete_enriches_metrics(
        self,
        mock_client: MagicMock,
        mock_trial: MagicMock,
        mock_progress: MagicMock,
    ):
        """Verify callback enriches trial metrics with Langfuse data."""
        # Setup callback with trace_id resolver
        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=lambda t: t.metadata.get("langfuse_trace_id"),
            metric_prefix="langfuse_",
        )

        # Execute callback
        callback.on_trial_complete(mock_trial, mock_progress)

        # Verify client was called with correct trace_id
        mock_client.get_trace_metrics.assert_called_once_with("trace-123")

        # Verify metrics were enriched
        assert "langfuse_total_cost" in mock_trial.metrics
        assert mock_trial.metrics["langfuse_total_cost"] == 0.005
        assert mock_trial.metrics["accuracy"] == 0.95  # Original preserved

    def test_on_trial_complete_handles_missing_trace_id(
        self,
        mock_client: MagicMock,
        mock_progress: MagicMock,
    ):
        """Verify callback handles missing trace_id gracefully."""
        # Trial with no trace_id in metadata
        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-2"
        trial.metadata = {}  # No trace_id
        trial.metrics = {"accuracy": 0.9}

        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=lambda t: t.metadata.get("langfuse_trace_id"),
        )

        # Should not raise, should not call client
        callback.on_trial_complete(trial, mock_progress)
        mock_client.get_trace_metrics.assert_not_called()

    def test_on_trial_complete_handles_client_error(
        self,
        mock_client: MagicMock,
        mock_trial: MagicMock,
        mock_progress: MagicMock,
    ):
        """Verify callback handles client errors gracefully."""
        # Setup client to raise an error
        mock_client.get_trace_metrics.side_effect = Exception("API error")

        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=lambda t: t.metadata.get("langfuse_trace_id"),
        )

        # Should not raise, should log warning
        callback.on_trial_complete(mock_trial, mock_progress)

        # Original metrics should be preserved
        assert mock_trial.metrics["accuracy"] == 0.95

    def test_on_trial_complete_handles_null_metrics(
        self,
        mock_client: MagicMock,
        mock_progress: MagicMock,
    ):
        """Verify callback handles trial with None metrics."""
        trial = MagicMock(spec=TrialResult)
        trial.trial_id = "trial-3"
        trial.metadata = {"langfuse_trace_id": "trace-456"}
        trial.metrics = None  # None metrics

        callback = LangfuseOptimizationCallback(
            client=mock_client,
            trace_id_resolver=lambda t: t.metadata.get("langfuse_trace_id"),
            metric_prefix="lf_",
        )

        callback.on_trial_complete(trial, mock_progress)

        # Metrics should be initialized and populated
        assert trial.metrics is not None
        assert "lf_total_cost" in trial.metrics


class TestLangfuseTracker:
    """Test high-level tracker interface."""

    def test_create_langfuse_tracker_returns_valid_instance(self):
        """Verify factory creates properly configured tracker."""
        tracker = create_langfuse_tracker(
            trace_id_resolver=lambda trial: "trace-id",
            public_key="test-key",
            secret_key="test-secret",
            metric_prefix="lf_",
            include_per_agent=False,
        )

        assert tracker.client is not None
        assert isinstance(tracker, LangfuseTracker)

    def test_tracker_get_callback_returns_callback(self):
        """Verify get_callback returns LangfuseOptimizationCallback."""
        tracker = create_langfuse_tracker(
            trace_id_resolver=lambda trial: trial.metadata.get("trace_id"),
            public_key="test",
            secret_key="test",
        )

        callback = tracker.get_callback()
        assert isinstance(callback, LangfuseOptimizationCallback)

    def test_tracker_get_callback_returns_same_instance(self):
        """Verify get_callback returns the same instance on multiple calls."""
        tracker = create_langfuse_tracker(
            trace_id_resolver=lambda trial: "id",
            public_key="test",
            secret_key="test",
        )

        callback1 = tracker.get_callback()
        callback2 = tracker.get_callback()

        assert callback1 is callback2

    def test_tracker_client_access(self):
        """Verify client property provides access to underlying client."""
        tracker = create_langfuse_tracker(
            trace_id_resolver=lambda trial: "id",
            public_key="pk-test",
            secret_key="sk-test",
        )

        client = tracker.client
        assert isinstance(client, LangfuseClient)


class TestTraceIdResolver:
    """Test TraceIdResolver protocol."""

    def test_lambda_resolver(self):
        """Verify lambda functions work as resolvers."""
        resolver = lambda trial: trial.metadata.get("trace_id")  # noqa: E731

        trial = MagicMock()
        trial.metadata = {"trace_id": "my-trace"}

        assert resolver(trial) == "my-trace"

    def test_callable_class_resolver(self):
        """Verify callable classes work as resolvers."""

        class MyResolver:
            def __call__(self, trial) -> str | None:
                return f"trace_{trial.trial_id}"

        resolver = MyResolver()
        trial = MagicMock()
        trial.trial_id = "123"

        assert resolver(trial) == "trace_123"

    def test_resolver_returning_none(self):
        """Verify resolvers can return None."""
        resolver = lambda trial: None  # noqa: E731

        trial = MagicMock()
        result = resolver(trial)

        assert result is None
