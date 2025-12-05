"""Comprehensive tests for traigent.core.metadata_helpers module.

Tests cover metadata construction, metric merging, and privacy handling
for trial and session results.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import pytest

from traigent.api.types import OptimizationResult, TrialResult
from traigent.config.types import ExecutionMode, TraigentConfig
from traigent.core.metadata_helpers import (
    _build_measures_full,
    _build_measures_privacy,
    build_backend_metadata,
    merge_run_metrics_into_session_summary,
)


@pytest.fixture
def mock_trial_result():
    """Create mock trial result."""
    trial = Mock(spec=TrialResult)
    trial.trial_id = "trial_123"
    trial.duration = 1.5
    trial.timestamp = datetime(2024, 1, 1, 12, 0, 0)
    trial.metrics = {"accuracy": 0.85, "cost": 0.05}
    trial.metadata = {}
    trial.summary_stats = None
    return trial


@pytest.fixture
def mock_config():
    """Create mock TraiGent config."""
    config = Mock(spec=TraigentConfig)
    config.execution_mode = "edge_analytics"
    config.minimal_logging = False
    config.privacy_enabled = False
    config.execution_mode_enum = ExecutionMode.EDGE_ANALYTICS
    return config


@pytest.fixture
def example_result():
    """Create mock example result."""
    result = Mock()
    result.metrics = {"accuracy": 0.9, "cost": 0.02}
    result.execution_time = 0.5
    result.expected_output = "expected"
    result.actual_output = "expected"
    return result


class TestMergeRunMetricsIntoSessionSummary:
    """Test merge_run_metrics_into_session_summary function."""

    def test_merge_with_existing_session_summary(self):
        """Test merging metrics into existing session summary."""
        result = Mock(spec=OptimizationResult)
        result.metadata = {"session_summary": {"metrics": {"existing": 100}}}
        result.metrics = {"accuracy": 0.85, "cost": 0.05}

        merge_run_metrics_into_session_summary(result)

        metrics = result.metadata["session_summary"]["metrics"]
        assert "run_accuracy" in metrics
        assert metrics["run_accuracy"] == 0.85
        assert "run_cost" in metrics
        assert metrics["run_cost"] == 0.05
        assert "existing" in metrics

    def test_merge_creates_metrics_dict(self):
        """Test merging creates metrics dict if missing."""
        result = Mock(spec=OptimizationResult)
        result.metadata = {"session_summary": {}}
        result.metrics = {"accuracy": 0.9}

        merge_run_metrics_into_session_summary(result)

        assert "metrics" in result.metadata["session_summary"]
        assert "run_accuracy" in result.metadata["session_summary"]["metrics"]

    def test_merge_preserves_run_prefix(self):
        """Test metrics with run_ prefix are not double-prefixed."""
        result = Mock(spec=OptimizationResult)
        result.metadata = {"session_summary": {"metrics": {}}}
        result.metrics = {"run_accuracy": 0.85}

        merge_run_metrics_into_session_summary(result)

        metrics = result.metadata["session_summary"]["metrics"]
        assert "run_accuracy" in metrics
        assert "run_run_accuracy" not in metrics

    def test_merge_filters_non_numeric_metrics(self):
        """Test non-numeric metrics are filtered out."""
        result = Mock(spec=OptimizationResult)
        result.metadata = {"session_summary": {"metrics": {}}}
        result.metrics = {
            "accuracy": 0.85,
            "model": "gpt-4",
            "cost": 0.05,
        }

        merge_run_metrics_into_session_summary(result)

        metrics = result.metadata["session_summary"]["metrics"]
        assert "run_accuracy" in metrics
        assert "run_cost" in metrics
        assert "run_model" not in metrics

    def test_merge_no_metadata(self):
        """Test merge handles missing metadata gracefully."""
        result = Mock(spec=OptimizationResult)
        result.metadata = None

        merge_run_metrics_into_session_summary(result)
        # Should not raise exception

    def test_merge_no_session_summary(self):
        """Test merge handles missing session_summary."""
        result = Mock(spec=OptimizationResult)
        result.metadata = {}

        merge_run_metrics_into_session_summary(result)
        # Should not raise exception

    def test_merge_non_dict_session_summary(self):
        """Test merge handles non-dict session_summary."""
        result = Mock(spec=OptimizationResult)
        result.metadata = {"session_summary": "not a dict"}

        merge_run_metrics_into_session_summary(result)
        # Should not raise exception

    def test_merge_non_dict_metrics(self):
        """Test merge handles non-dict existing metrics."""
        result = Mock(spec=OptimizationResult)
        result.metadata = {"session_summary": {"metrics": "not a dict"}}
        result.metrics = {"accuracy": 0.85}

        merge_run_metrics_into_session_summary(result)

        metrics = result.metadata["session_summary"]["metrics"]
        assert isinstance(metrics, dict)
        assert "run_accuracy" in metrics


class TestBuildBackendMetadataBasic:
    """Test basic build_backend_metadata functionality."""

    def test_basic_metadata_structure(self, mock_trial_result, mock_config):
        """Test basic metadata structure is created."""
        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        assert "duration" in metadata
        assert "trial_id" in metadata
        assert "execution_mode" in metadata
        assert metadata["trial_id"] == "trial_123"
        assert metadata["duration"] == 1.5

    def test_minimal_logging_mode(self, mock_trial_result, mock_config):
        """Test minimal logging excludes timestamp and all_metrics."""
        mock_config.minimal_logging = True

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        assert "timestamp" not in metadata
        assert "all_metrics" not in metadata

    def test_full_logging_mode(self, mock_trial_result, mock_config):
        """Test full logging includes timestamp and all_metrics."""
        mock_config.minimal_logging = False

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        assert "timestamp" in metadata
        assert "all_metrics" in metadata
        assert metadata["timestamp"] == "2024-01-01T12:00:00"

    def test_additional_metrics_added(self, mock_trial_result, mock_config):
        """Test additional metrics beyond primary objective are added."""
        mock_trial_result.metrics = {
            "accuracy": 0.85,
            "cost": 0.05,
            "latency": 0.2,
        }

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        assert "cost" in metadata
        assert "latency" in metadata
        assert metadata["cost"] == 0.05
        assert metadata["latency"] == 0.2


class TestBuildBackendMetadataSummaryStats:
    """Test summary_stats handling in build_backend_metadata."""

    def test_summary_stats_added_non_cloud(self, mock_trial_result, mock_config):
        """Test summary_stats are added in non-cloud modes."""
        mock_trial_result.summary_stats = {"mean": 0.85, "std": 0.1}
        mock_config.execution_mode = "edge_analytics"

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        assert "summary_stats" in metadata
        assert metadata["summary_stats"]["mean"] == 0.85
        assert "metadata" in metadata["summary_stats"]
        assert metadata["summary_stats"]["metadata"]["aggregation_level"] == "trial"

    def test_summary_stats_not_added_cloud(self, mock_trial_result, mock_config):
        """Test summary_stats are not added in cloud mode."""
        mock_trial_result.summary_stats = {"mean": 0.85}
        mock_config.execution_mode = "cloud"
        mock_config.execution_mode_enum = ExecutionMode.CLOUD

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        assert "summary_stats" not in metadata

    def test_summary_stats_enhancement(self, mock_trial_result, mock_config):
        """Test summary_stats are enhanced with metadata."""
        mock_trial_result.summary_stats = {"mean": 0.85}
        mock_config.execution_mode = "edge_analytics"

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        summary_stats = metadata["summary_stats"]
        assert "metadata" in summary_stats
        assert summary_stats["metadata"]["sdk_version"] == "2.0.0"
        assert summary_stats["metadata"]["aggregation_level"] == "trial"

    def test_summary_stats_with_existing_metadata(self, mock_trial_result, mock_config):
        """Test summary_stats with existing metadata field."""
        mock_trial_result.summary_stats = {
            "mean": 0.85,
            "metadata": {"existing": "value"},
        }
        mock_config.execution_mode = "edge_analytics"

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        summary_meta = metadata["summary_stats"]["metadata"]
        assert "existing" in summary_meta
        assert summary_meta["aggregation_level"] == "trial"


class TestBuildBackendMetadataPrivacy:
    """Test privacy handling in build_backend_metadata."""

    def test_privacy_enabled_flag(self, mock_trial_result, mock_config, example_result):
        """Test privacy_enabled flag triggers privacy mode."""
        mock_config.privacy_enabled = True
        mock_config.execution_mode = "edge_analytics"
        mock_trial_result.metadata = {"example_results": [example_result]}

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        assert "measures" in metadata
        # Privacy mode should sanitize measures
        assert len(metadata["measures"]) == 1

    def test_privacy_execution_mode(
        self, mock_trial_result, mock_config, example_result
    ):
        """Test privacy execution mode triggers privacy measures."""
        mock_config.execution_mode = "privacy"
        mock_trial_result.metadata = {"example_results": [example_result]}

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        assert "measures" in metadata

    def test_example_results_removed_privacy(
        self, mock_trial_result, mock_config, example_result
    ):
        """Test example_results are removed in privacy mode."""
        mock_config.privacy_enabled = True
        mock_trial_result.metadata = {"example_results": [example_result]}

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        assert "example_results" not in metadata

    def test_full_measures_non_privacy(
        self, mock_trial_result, mock_config, example_result
    ):
        """Test full measures are built in non-privacy mode."""
        mock_config.execution_mode = "edge_analytics"
        mock_trial_result.metadata = {"example_results": [example_result]}

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        assert "measures" in metadata
        assert len(metadata["measures"]) == 1


class TestBuildMeasuresFull:
    """Test _build_measures_full function."""

    def test_extract_score_from_primary_objective(self, example_result):
        """Test score extraction from primary objective."""
        example_result.metrics = {"accuracy": 0.9}

        measures = _build_measures_full([example_result], "accuracy")

        assert len(measures) == 1
        assert measures[0]["score"] == 0.9

    def test_fallback_to_score_field(self, example_result):
        """Test fallback to 'score' field if primary not found."""
        example_result.metrics = {"score": 0.85, "cost": 0.02}

        measures = _build_measures_full([example_result], "nonexistent")

        assert measures[0]["score"] == 0.85

    def test_fallback_to_accuracy_field(self, example_result):
        """Test fallback to 'accuracy' field."""
        example_result.metrics = {"accuracy": 0.8}

        measures = _build_measures_full([example_result], "nonexistent")

        assert measures[0]["score"] == 0.8

    def test_fallback_to_expected_actual_comparison(self, example_result):
        """Test fallback to expected/actual comparison."""
        example_result.metrics = {}
        example_result.expected_output = "test"
        example_result.actual_output = "test"

        measures = _build_measures_full([example_result], "accuracy")

        assert measures[0]["score"] == 1.0

    def test_expected_actual_mismatch(self, example_result):
        """Test expected/actual mismatch gives score 0."""
        example_result.metrics = {}
        example_result.expected_output = "test"
        example_result.actual_output = "wrong"

        measures = _build_measures_full([example_result], "accuracy")

        assert measures[0]["score"] == 0.0

    def test_include_all_scalar_metrics(self, example_result):
        """Test all scalar metrics are included."""
        example_result.metrics = {
            "accuracy": 0.9,
            "cost": 0.02,
            "latency": 0.5,
            "model": "gpt-4",
        }

        measures = _build_measures_full([example_result], "accuracy")

        measure = measures[0]
        assert "accuracy" in measure
        assert "cost" in measure
        assert "latency" in measure
        assert "model" in measure

    def test_include_execution_time(self, example_result):
        """Test execution_time is included as response_time."""
        example_result.execution_time = 1.5

        measures = _build_measures_full([example_result], "accuracy")

        assert "response_time" in measures[0]
        assert measures[0]["response_time"] == 1.5

    def test_multiple_examples(self, example_result):
        """Test building measures for multiple examples."""
        example2 = Mock()
        example2.metrics = {"accuracy": 0.8}
        example2.execution_time = 0.6

        measures = _build_measures_full([example_result, example2], "accuracy")

        assert len(measures) == 2
        assert measures[0]["score"] == 0.9
        assert measures[1]["score"] == 0.8


class TestBuildMeasuresPrivacy:
    """Test _build_measures_privacy function."""

    def test_extract_score_privacy(self, example_result):
        """Test score extraction in privacy mode."""
        example_result.metrics = {"accuracy": 0.9}

        measures = _build_measures_privacy([example_result], "accuracy")

        assert len(measures) == 1
        assert measures[0]["score"] == 0.9

    def test_include_response_time(self, example_result):
        """Test response_time is included in privacy mode."""
        example_result.execution_time = 1.5

        measures = _build_measures_privacy([example_result], "accuracy")

        assert "response_time" in measures[0]
        assert measures[0]["response_time"] == 1.5

    def test_include_token_metrics(self, example_result):
        """Test token metrics are included in privacy mode."""
        example_result.metrics = {
            "accuracy": 0.9,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

        measures = _build_measures_privacy([example_result], "accuracy")

        measure = measures[0]
        assert "input_tokens" in measure
        assert "output_tokens" in measure
        assert "total_tokens" in measure
        assert measure["input_tokens"] == 100

    def test_include_cost_metrics(self, example_result):
        """Test cost metrics are included in privacy mode."""
        example_result.metrics = {
            "accuracy": 0.9,
            "input_cost": 0.01,
            "output_cost": 0.02,
            "total_cost": 0.03,
        }

        measures = _build_measures_privacy([example_result], "accuracy")

        measure = measures[0]
        assert "input_cost" in measure
        assert "output_cost" in measure
        assert "total_cost" in measure
        assert measure["total_cost"] == 0.03

    def test_exclude_sensitive_metrics(self, example_result):
        """Test sensitive metrics are excluded in privacy mode."""
        example_result.metrics = {
            "accuracy": 0.9,
            "user_input": "sensitive data",
            "model_output": "private response",
        }

        measures = _build_measures_privacy([example_result], "accuracy")

        measure = measures[0]
        assert "score" in measure
        assert "user_input" not in measure
        assert "model_output" not in measure

    def test_fallback_score_calculation_privacy(self, example_result):
        """Test fallback score calculation in privacy mode."""
        example_result.metrics = {}
        example_result.expected_output = "test"
        example_result.actual_output = "test"

        measures = _build_measures_privacy([example_result], "accuracy")

        assert measures[0]["score"] == 1.0


class TestBuildBackendMetadataIntegration:
    """Test integration scenarios for build_backend_metadata."""

    def test_complete_metadata_pipeline(
        self, mock_trial_result, mock_config, example_result
    ):
        """Test complete metadata construction pipeline."""
        mock_config.execution_mode = "edge_analytics"
        mock_config.minimal_logging = False
        mock_trial_result.summary_stats = {"mean": 0.85, "std": 0.1}
        mock_trial_result.metadata = {"example_results": [example_result]}
        mock_trial_result.metrics = {
            "accuracy": 0.9,
            "cost": 0.05,
            "latency": 0.2,
        }

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        # Check all components present
        assert "duration" in metadata
        assert "trial_id" in metadata
        assert "timestamp" in metadata
        assert "all_metrics" in metadata
        assert "summary_stats" in metadata
        assert "measures" in metadata
        assert "cost" in metadata
        assert "latency" in metadata

    def test_cloud_mode_restrictions(self, mock_trial_result, mock_config):
        """Test cloud mode applies appropriate restrictions."""
        mock_config.execution_mode = "cloud"
        mock_config.execution_mode_enum = ExecutionMode.CLOUD
        mock_trial_result.summary_stats = {"mean": 0.85}

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        # Summary stats should not be added in cloud mode
        assert "summary_stats" not in metadata

    def test_empty_example_results(self, mock_trial_result, mock_config):
        """Test handling of empty example_results."""
        mock_config.execution_mode = "edge_analytics"
        mock_trial_result.metadata = {"example_results": []}

        metadata = build_backend_metadata(mock_trial_result, "accuracy", mock_config)

        # Empty example_results should create empty measures list
        if "measures" in metadata:
            assert metadata["measures"] == []
        # Or measures may not be added at all for empty list
