"""Comprehensive tests for traigent.core.mandatory_metrics module.

Tests cover MandatoryMetricsTotals and MandatoryMetricsCollector for
metric aggregation across trials.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.mandatory_metrics import (
    MandatoryMetricsCollector,
    MandatoryMetricsTotals,
)


@pytest.fixture
def completed_trial():
    """Create a completed trial result."""
    trial = Mock(spec=TrialResult)
    trial.trial_id = "trial_123"
    trial.status = TrialStatus.COMPLETED
    trial.duration = 1.5
    trial.metrics = {
        "total_cost": 0.05,
        "total_tokens": 100,
        "examples_attempted": 10,
    }
    trial.metadata = {}
    return trial


@pytest.fixture
def failed_trial():
    """Create a failed trial result."""
    trial = Mock(spec=TrialResult)
    trial.trial_id = "trial_456"
    trial.status = TrialStatus.FAILED
    trial.duration = 0.5
    trial.metrics = {}
    trial.metadata = {}
    return trial


class TestMandatoryMetricsTotals:
    """Test MandatoryMetricsTotals dataclass."""

    def test_default_initialization(self):
        """Test default initialization with zero values."""
        totals = MandatoryMetricsTotals()

        assert totals.total_cost == 0.0
        assert totals.total_tokens == 0
        assert totals.total_duration == 0.0
        assert totals.total_examples_attempted == 0

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        totals = MandatoryMetricsTotals(
            total_cost=1.5,
            total_tokens=500,
            total_duration=10.0,
            total_examples_attempted=50,
        )

        assert totals.total_cost == 1.5
        assert totals.total_tokens == 500
        assert totals.total_duration == 10.0
        assert totals.total_examples_attempted == 50

    def test_as_metrics_dict_all_positive(self):
        """Test as_metrics_dict with all positive values."""
        totals = MandatoryMetricsTotals(
            total_cost=1.5,
            total_tokens=500,
            total_duration=10.0,
            total_examples_attempted=50,
        )

        metrics = totals.as_metrics_dict()

        assert "total_cost" in metrics
        assert "total_tokens" in metrics
        assert "total_duration" in metrics
        assert "examples_attempted_total" in metrics
        assert metrics["total_cost"] == 1.5
        assert metrics["total_tokens"] == 500.0
        assert metrics["total_duration"] == 10.0
        assert metrics["examples_attempted_total"] == 50.0

    def test_as_metrics_dict_excludes_zeros(self):
        """Test as_metrics_dict excludes zero values."""
        totals = MandatoryMetricsTotals(
            total_cost=0.0,
            total_tokens=0,
            total_duration=0.0,
            total_examples_attempted=0,
        )

        metrics = totals.as_metrics_dict()

        assert metrics == {}

    def test_as_metrics_dict_partial_values(self):
        """Test as_metrics_dict with partial values."""
        totals = MandatoryMetricsTotals(
            total_cost=1.5,
            total_tokens=0,
            total_duration=10.0,
            total_examples_attempted=0,
        )

        metrics = totals.as_metrics_dict()

        assert "total_cost" in metrics
        assert "total_duration" in metrics
        assert "total_tokens" not in metrics
        assert "examples_attempted_total" not in metrics

    def test_tokens_converted_to_float(self):
        """Test that tokens are converted to float in metrics dict."""
        totals = MandatoryMetricsTotals(total_tokens=100)

        metrics = totals.as_metrics_dict()

        assert isinstance(metrics["total_tokens"], float)
        assert metrics["total_tokens"] == 100.0

    def test_slots_optimization(self):
        """Test that dataclass uses slots."""
        totals = MandatoryMetricsTotals()

        with pytest.raises(AttributeError):
            totals.new_attribute = "value"  # type: ignore


class TestMandatoryMetricsCollectorBasics:
    """Test basic MandatoryMetricsCollector functionality."""

    def test_initialization(self):
        """Test collector initialization."""
        collector = MandatoryMetricsCollector()

        totals = collector.totals()
        assert totals.total_cost == 0.0
        assert totals.total_tokens == 0
        assert totals.total_duration == 0.0
        assert totals.total_examples_attempted == 0

    def test_accumulate_completed_trial(self, completed_trial):
        """Test accumulating a completed trial."""
        collector = MandatoryMetricsCollector()

        collector.accumulate(completed_trial)

        totals = collector.totals()
        assert totals.total_cost == 0.05
        assert totals.total_tokens == 100
        assert totals.total_duration == 1.5
        assert totals.total_examples_attempted == 10

    def test_accumulate_failed_trial_ignored(self, failed_trial):
        """Test that failed trials are ignored."""
        collector = MandatoryMetricsCollector()

        collector.accumulate(failed_trial)

        totals = collector.totals()
        assert totals.total_cost == 0.0
        assert totals.total_tokens == 0
        assert totals.total_duration == 0.0

    def test_accumulate_multiple_trials(self, completed_trial):
        """Test accumulating multiple trials."""
        collector = MandatoryMetricsCollector()

        trial1 = completed_trial
        trial2 = Mock(spec=TrialResult)
        trial2.trial_id = "trial_789"
        trial2.status = TrialStatus.COMPLETED
        trial2.duration = 2.0
        trial2.metrics = {
            "total_cost": 0.10,
            "total_tokens": 200,
            "examples_attempted": 15,
        }
        trial2.metadata = {}

        collector.accumulate(trial1)
        collector.accumulate(trial2)

        totals = collector.totals()
        assert (
            abs(totals.total_cost - 0.15) < 0.001
        )  # 0.05 + 0.10 (floating point tolerance)
        assert totals.total_tokens == 300  # 100 + 200
        assert totals.total_duration == 3.5  # 1.5 + 2.0
        assert totals.total_examples_attempted == 25  # 10 + 15


class TestMandatoryMetricsCollectorDuration:
    """Test duration extraction and accumulation."""

    def test_valid_duration(self):
        """Test accumulating valid duration."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 2.5
        trial.metrics = {}
        trial.metadata = {}

        collector.accumulate(trial)

        assert collector.totals().total_duration == 2.5

    def test_none_duration(self):
        """Test handling None duration."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = None
        trial.metrics = {}
        trial.metadata = {}

        collector.accumulate(trial)

        assert collector.totals().total_duration == 0.0

    def test_invalid_duration_type(self):
        """Test handling invalid duration type."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = "invalid"
        trial.metrics = {}
        trial.metadata = {}

        # Should handle gracefully
        collector.accumulate(trial)

        assert collector.totals().total_duration == 0.0


class TestMandatoryMetricsCollectorExamplesAttempted:
    """Test examples_attempted extraction."""

    def test_from_metrics(self):
        """Test extracting examples_attempted from metrics."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {"examples_attempted": 20}
        trial.metadata = {}

        collector.accumulate(trial)

        assert collector.totals().total_examples_attempted == 20

    def test_from_metadata_fallback(self):
        """Test extracting examples_attempted from metadata fallback."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {}
        trial.metadata = {"examples_attempted": 15}

        collector.accumulate(trial)

        assert collector.totals().total_examples_attempted == 15

    def test_missing_examples_attempted(self):
        """Test handling missing examples_attempted."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {}
        trial.metadata = {}

        collector.accumulate(trial)

        assert collector.totals().total_examples_attempted == 0

    def test_invalid_examples_attempted_type(self):
        """Test handling invalid examples_attempted type."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {"examples_attempted": "invalid"}
        trial.metadata = {}

        collector.accumulate(trial)

        assert collector.totals().total_examples_attempted == 0


class TestMandatoryMetricsCollectorCostAndTokens:
    """Test cost and tokens extraction."""

    def test_basic_cost_and_tokens(self):
        """Test basic cost and tokens extraction."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {"total_cost": 0.25, "total_tokens": 500}
        trial.metadata = {}

        collector.accumulate(trial)

        totals = collector.totals()
        assert totals.total_cost == 0.25
        assert totals.total_tokens == 500

    def test_missing_cost(self):
        """Test handling missing cost."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {"total_tokens": 100}
        trial.metadata = {}

        collector.accumulate(trial)

        assert collector.totals().total_cost == 0.0

    def test_missing_tokens(self):
        """Test handling missing tokens."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {"total_cost": 0.1}
        trial.metadata = {}

        collector.accumulate(trial)

        assert collector.totals().total_tokens == 0

    def test_none_cost(self):
        """Test handling None cost."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {"total_cost": None}
        trial.metadata = {}

        collector.accumulate(trial)

        assert collector.totals().total_cost == 0.0

    def test_negative_cost_ignored(self):
        """Test negative cost is treated as zero."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {"total_cost": -0.5}
        trial.metadata = {}

        collector.accumulate(trial)

        assert collector.totals().total_cost == 0.0


class TestMandatoryMetricsCollectorAggregatedMetrics:
    """Test extraction from aggregated metrics."""

    def test_aggregated_cost_fallback(self):
        """Test falling back to aggregated cost metrics."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {}

        eval_result = Mock()
        eval_result.total_examples = 10
        eval_result.aggregated_metrics = {"total_cost": {"mean": 0.05}}

        trial.metadata = {"evaluation_result": eval_result}

        collector.accumulate(trial)

        # 0.05 * 10 examples = 0.5
        assert collector.totals().total_cost == 0.5

    def test_aggregated_tokens_fallback(self):
        """Test falling back to aggregated tokens metrics."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {}

        eval_result = Mock()
        eval_result.total_examples = 5
        eval_result.aggregated_metrics = {"total_tokens": {"mean": 100.0}}

        trial.metadata = {"evaluation_result": eval_result}

        collector.accumulate(trial)

        # 100.0 * 5 examples = 500
        assert collector.totals().total_tokens == 500

    def test_aggregated_invalid_structure(self):
        """Test handling invalid aggregated metrics structure."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {}

        eval_result = Mock()
        eval_result.aggregated_metrics = {
            "total_cost": "not_a_dict",  # Invalid structure
        }

        trial.metadata = {"evaluation_result": eval_result}

        collector.accumulate(trial)

        assert collector.totals().total_cost == 0.0

    def test_aggregated_missing_mean(self):
        """Test handling aggregated metrics without mean."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {}

        eval_result = Mock()
        eval_result.total_examples = 10
        eval_result.aggregated_metrics = {"total_cost": {}}  # No mean field

        trial.metadata = {"evaluation_result": eval_result}

        collector.accumulate(trial)

        assert collector.totals().total_cost == 0.0


class TestMandatoryMetricsCollectorEdgeCases:
    """Test edge cases and error handling."""

    def test_none_metrics(self):
        """Test handling None metrics."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = None
        trial.metadata = {}

        collector.accumulate(trial)

        totals = collector.totals()
        assert totals.total_cost == 0.0
        assert totals.total_tokens == 0

    def test_none_metadata(self):
        """Test handling None metadata."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.COMPLETED
        trial.duration = 1.0
        trial.metrics = {}
        trial.metadata = None

        collector.accumulate(trial)

        assert collector.totals().total_examples_attempted == 0

    def test_pruned_trial_ignored(self):
        """Test pruned trials are ignored."""
        collector = MandatoryMetricsCollector()
        trial = Mock(spec=TrialResult)
        trial.trial_id = "trial_123"
        trial.status = TrialStatus.PRUNED
        trial.duration = 1.0
        trial.metrics = {"total_cost": 0.5}
        trial.metadata = {}

        collector.accumulate(trial)

        assert collector.totals().total_cost == 0.0

    def test_accumulate_preserves_previous_totals(self, completed_trial):
        """Test that accumulate adds to existing totals."""
        collector = MandatoryMetricsCollector()

        collector.accumulate(completed_trial)
        first_cost = collector.totals().total_cost
        first_tokens = collector.totals().total_tokens

        collector.accumulate(completed_trial)
        second_cost = collector.totals().total_cost
        second_tokens = collector.totals().total_tokens

        # Values should double
        assert second_cost == first_cost * 2
        assert second_tokens == first_tokens * 2


class TestMandatoryMetricsCollectorIntegration:
    """Test integration scenarios."""

    def test_complete_workflow(self):
        """Test complete metrics collection workflow."""
        collector = MandatoryMetricsCollector()

        # Add multiple trials with varying completeness
        trial1 = Mock(spec=TrialResult)
        trial1.trial_id = "trial_1"
        trial1.status = TrialStatus.COMPLETED
        trial1.duration = 1.0
        trial1.metrics = {
            "total_cost": 0.10,
            "total_tokens": 100,
            "examples_attempted": 10,
        }
        trial1.metadata = {}

        trial2 = Mock(spec=TrialResult)
        trial2.trial_id = "trial_2"
        trial2.status = TrialStatus.FAILED
        trial2.duration = 0.5
        trial2.metrics = {"total_cost": 0.20}  # Should be ignored
        trial2.metadata = {}

        trial3 = Mock(spec=TrialResult)
        trial3.trial_id = "trial_3"
        trial3.status = TrialStatus.COMPLETED
        trial3.duration = 2.0
        trial3.metrics = {}
        trial3.metadata = {"examples_attempted": 5}

        collector.accumulate(trial1)
        collector.accumulate(trial2)
        collector.accumulate(trial3)

        totals = collector.totals()
        metrics = totals.as_metrics_dict()

        assert totals.total_cost == 0.10
        assert totals.total_tokens == 100
        assert totals.total_duration == 3.0  # 1.0 + 2.0 (failed ignored)
        assert totals.total_examples_attempted == 15  # 10 + 5

        assert "total_cost" in metrics
        assert "examples_attempted_total" in metrics
