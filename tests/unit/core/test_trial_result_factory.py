"""Comprehensive tests for traigent.core.trial_result_factory module.

Tests cover trial result construction for success, pruned, and failed states.
"""

from __future__ import annotations

from datetime import UTC
from unittest.mock import Mock

import pytest

from traigent.api.types import TrialStatus
from traigent.core.trial_result_factory import (
    build_failed_result,
    build_pruned_result,
    build_success_result,
)
from traigent.utils.exceptions import TrialPrunedError


@pytest.fixture
def eval_config():
    """Create evaluation configuration."""
    return {"model": "gpt-4", "temperature": 0.7}


@pytest.fixture
def eval_result():
    """Create mock evaluation result."""
    result = Mock()
    result.metrics = {"accuracy": 0.85, "cost": 0.05}
    result.success_rate = 0.9
    result.has_errors = False
    result.outputs = ["output1", "output2", "output3"]
    result.example_results = [Mock(), Mock()]
    result.summary_stats = {"mean": 0.85, "std": 0.1}
    return result


@pytest.fixture
def prune_error():
    """Create mock pruned error."""
    return TrialPrunedError(step=5, message="Pruned at step 5")


@pytest.fixture
def progress_state():
    """Create progress state dict."""
    return {
        "evaluated": 10,
        "total_examples": 20,
        "correct_sum": 8.5,
        "total_cost": 0.15,
    }


class TestBuildSuccessResult:
    """Test build_success_result function."""

    def test_basic_success_result(self, eval_config, eval_result):
        """Test basic successful result construction."""
        result = build_success_result(
            trial_id="trial_123",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.5,
            examples_attempted=10,
            total_cost=0.05,
            optuna_trial_id=None,
        )

        assert result.trial_id == "trial_123"
        assert result.config == eval_config
        assert result.status == TrialStatus.COMPLETED
        assert result.duration == 1.5
        assert result.metrics == {
            "accuracy": 0.85,
            "cost": 0.05,
            "examples_attempted": 10,
            "total_cost": 0.05,
        }

    def test_metadata_fields(self, eval_config, eval_result):
        """Test metadata fields are populated correctly."""
        result = build_success_result(
            trial_id="trial_123",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.5,
            examples_attempted=10,
            total_cost=0.05,
            optuna_trial_id=None,
        )

        metadata = result.metadata
        assert "success_rate" in metadata
        assert "has_errors" in metadata
        assert "output_count" in metadata
        assert metadata["success_rate"] == 0.9
        assert metadata["has_errors"] is False
        assert metadata["output_count"] == 3

    def test_example_results_included(self, eval_config, eval_result):
        """Test example_results are included in metadata."""
        result = build_success_result(
            trial_id="trial_123",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.5,
            examples_attempted=None,
            total_cost=None,
            optuna_trial_id=None,
        )

        assert "example_results" in result.metadata
        assert len(result.metadata["example_results"]) == 2

    def test_no_example_results(self, eval_config):
        """Test handling when example_results is None."""
        eval_result = Mock()
        eval_result.metrics = {}
        eval_result.example_results = None
        eval_result.outputs = None  # Fix: add outputs attribute

        result = build_success_result(
            trial_id="trial_123",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.5,
            examples_attempted=None,
            total_cost=None,
            optuna_trial_id=None,
        )

        assert "example_results" not in result.metadata

    def test_examples_attempted_included(self, eval_config, eval_result):
        """Test examples_attempted is included in metadata and metrics."""
        result = build_success_result(
            trial_id="trial_123",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.5,
            examples_attempted=15,
            total_cost=None,
            optuna_trial_id=None,
        )

        assert result.metadata["examples_attempted"] == 15
        assert result.metrics["examples_attempted"] == 15

    def test_total_cost_included(self, eval_config, eval_result):
        """Test total_cost is included in metadata and metrics."""
        result = build_success_result(
            trial_id="trial_123",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.5,
            examples_attempted=None,
            total_cost=0.25,
            optuna_trial_id=None,
        )

        assert result.metadata["total_example_cost"] == 0.25
        assert result.metrics["total_cost"] == 0.25

    def test_optuna_trial_id_included(self, eval_config, eval_result):
        """Test optuna_trial_id is included in metadata."""
        result = build_success_result(
            trial_id="trial_123",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.5,
            examples_attempted=None,
            total_cost=None,
            optuna_trial_id=42,
        )

        assert result.metadata["optuna_trial_id"] == 42

    def test_summary_stats_included(self, eval_config, eval_result):
        """Test summary_stats are included if present."""
        result = build_success_result(
            trial_id="trial_123",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.5,
            examples_attempted=None,
            total_cost=None,
            optuna_trial_id=None,
        )

        assert hasattr(result, "summary_stats")
        assert result.summary_stats == {"mean": 0.85, "std": 0.1}

    def test_no_summary_stats(self, eval_config):
        """Test handling when summary_stats is None."""
        eval_result = Mock()
        eval_result.metrics = {}
        eval_result.summary_stats = None
        eval_result.outputs = None  # Fix: add outputs attribute

        result = build_success_result(
            trial_id="trial_123",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.5,
            examples_attempted=None,
            total_cost=None,
            optuna_trial_id=None,
        )

        # Should not have summary_stats or should be None
        assert not hasattr(result, "summary_stats") or result.summary_stats is None

    def test_timestamp_is_utc(self, eval_config, eval_result):
        """Test timestamp is in UTC timezone."""
        result = build_success_result(
            trial_id="trial_123",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.5,
            examples_attempted=None,
            total_cost=None,
            optuna_trial_id=None,
        )

        assert result.timestamp.tzinfo == UTC

    def test_invalid_examples_attempted(self, eval_config, eval_result):
        """Test handling of invalid examples_attempted value."""
        # The code converts to int without try-except at line 41, so it will raise ValueError
        # Test that it raises ValueError for invalid input
        with pytest.raises(ValueError):
            build_success_result(
                trial_id="trial_123",
                evaluation_config=eval_config,
                eval_result=eval_result,
                duration=1.5,
                examples_attempted="invalid",  # type: ignore
                total_cost=None,
                optuna_trial_id=None,
            )

    def test_invalid_total_cost(self, eval_config, eval_result):
        """Test handling of invalid total_cost value."""
        # Should handle gracefully without crashing
        result = build_success_result(
            trial_id="trial_123",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.5,
            examples_attempted=None,
            total_cost="invalid",  # type: ignore
            optuna_trial_id=None,
        )

        # Should still create result, may not have total_cost in metrics
        assert result.trial_id == "trial_123"


class TestBuildPrunedResult:
    """Test build_pruned_result function."""

    def test_basic_pruned_result(self, eval_config, prune_error):
        """Test basic pruned result construction."""
        result = build_pruned_result(
            trial_id="trial_456",
            evaluation_config=eval_config,
            duration=0.5,
            prune_error=prune_error,
            progress_state=None,
            optuna_trial_id=None,
        )

        assert result.trial_id == "trial_456"
        assert result.config == eval_config
        assert result.status == TrialStatus.PRUNED
        assert result.duration == 0.5
        assert result.metadata["pruned"] is True
        assert result.metadata["pruned_step"] == 5

    def test_pruned_with_progress_state(self, eval_config, prune_error, progress_state):
        """Test pruned result with progress state."""
        result = build_pruned_result(
            trial_id="trial_456",
            evaluation_config=eval_config,
            duration=0.5,
            prune_error=prune_error,
            progress_state=progress_state,
            optuna_trial_id=None,
        )

        assert "accuracy" in result.metrics
        assert result.metrics["accuracy"] == 0.85  # 8.5 / 10
        assert result.metadata["examples_attempted"] == 10

    def test_pruned_with_cost(self, eval_config, prune_error, progress_state):
        """Test pruned result includes cost metrics."""
        result = build_pruned_result(
            trial_id="trial_456",
            evaluation_config=eval_config,
            duration=0.5,
            prune_error=prune_error,
            progress_state=progress_state,
            optuna_trial_id=None,
        )

        assert "total_cost" in result.metrics
        assert result.metrics["total_cost"] == 0.15
        assert "cost" in result.metrics
        assert result.metrics["cost"] == 0.15
        assert result.metadata["total_example_cost"] == 0.15

    def test_pruned_with_optuna_id(self, eval_config, prune_error):
        """Test pruned result includes optuna trial ID."""
        result = build_pruned_result(
            trial_id="trial_456",
            evaluation_config=eval_config,
            duration=0.5,
            prune_error=prune_error,
            progress_state=None,
            optuna_trial_id=99,
        )

        assert result.metadata["optuna_trial_id"] == 99

    def test_pruned_clamped_evaluated(self, eval_config, prune_error):
        """Test evaluated is clamped to total_examples."""
        progress = {
            "evaluated": 25,  # More than total
            "total_examples": 20,
            "correct_sum": 18.0,
        }

        result = build_pruned_result(
            trial_id="trial_456",
            evaluation_config=eval_config,
            duration=0.5,
            prune_error=prune_error,
            progress_state=progress,
            optuna_trial_id=None,
        )

        # Should clamp to 20
        assert result.metadata["examples_attempted"] == 20
        assert result.metrics["accuracy"] == 0.9  # 18.0 / 20

    def test_pruned_no_examples_attempted(self, eval_config, prune_error):
        """Test pruned result when no examples were attempted."""
        progress = {"evaluated": 0, "total_examples": 20, "correct_sum": 0.0}

        result = build_pruned_result(
            trial_id="trial_456",
            evaluation_config=eval_config,
            duration=0.5,
            prune_error=prune_error,
            progress_state=progress,
            optuna_trial_id=None,
        )

        # No accuracy metric if evaluated is 0
        assert "accuracy" not in result.metrics
        assert "examples_attempted" not in result.metadata

    def test_pruned_invalid_cost(self, eval_config, prune_error):
        """Test pruned result with invalid cost value."""
        progress = {
            "evaluated": 10,
            "total_examples": 20,
            "correct_sum": 8.0,
            "total_cost": "invalid",
        }

        # Should handle gracefully
        result = build_pruned_result(
            trial_id="trial_456",
            evaluation_config=eval_config,
            duration=0.5,
            prune_error=prune_error,
            progress_state=progress,
            optuna_trial_id=None,
        )

        assert result.trial_id == "trial_456"
        # May not have cost in metrics


class TestBuildFailedResult:
    """Test build_failed_result function."""

    def test_basic_failed_result(self, eval_config):
        """Test basic failed result construction."""
        error = ValueError("Test error")

        result = build_failed_result(
            trial_id="trial_789",
            evaluation_config=eval_config,
            duration=0.3,
            error=error,
            progress_state=None,
            optuna_trial_id=None,
        )

        assert result.trial_id == "trial_789"
        assert result.config == eval_config
        assert result.status == TrialStatus.FAILED
        assert result.duration == 0.3
        assert result.error_message == "Test error"

    def test_failed_with_progress_state(self, eval_config, progress_state):
        """Test failed result with progress state."""
        error = RuntimeError("Execution failed")

        result = build_failed_result(
            trial_id="trial_789",
            evaluation_config=eval_config,
            duration=0.3,
            error=error,
            progress_state=progress_state,
            optuna_trial_id=None,
        )

        assert result.metadata["examples_attempted"] == 10

    def test_failed_with_cost(self, eval_config, progress_state):
        """Test failed result includes cost metrics."""
        error = Exception("Cost tracking test")

        result = build_failed_result(
            trial_id="trial_789",
            evaluation_config=eval_config,
            duration=0.3,
            error=error,
            progress_state=progress_state,
            optuna_trial_id=None,
        )

        assert "total_cost" in result.metrics
        assert result.metrics["total_cost"] == 0.15
        assert result.metadata["total_example_cost"] == 0.15

    def test_failed_with_optuna_id(self, eval_config):
        """Test failed result includes optuna trial ID."""
        error = Exception("Test failure")

        result = build_failed_result(
            trial_id="trial_789",
            evaluation_config=eval_config,
            duration=0.3,
            error=error,
            progress_state=None,
            optuna_trial_id=77,
        )

        assert result.metadata["optuna_trial_id"] == 77

    def test_failed_empty_metadata_no_optuna(self, eval_config):
        """Test failed result has empty metadata when no optuna ID."""
        error = Exception("Test")

        result = build_failed_result(
            trial_id="trial_789",
            evaluation_config=eval_config,
            duration=0.3,
            error=error,
            progress_state=None,
            optuna_trial_id=None,
        )

        # Metadata should be empty dict when no optuna_trial_id
        assert result.metadata == {}

    def test_failed_invalid_cost(self, eval_config):
        """Test failed result with invalid cost value."""
        error = Exception("Test")
        progress = {"evaluated": 5, "total_cost": "invalid"}

        # Should handle gracefully
        result = build_failed_result(
            trial_id="trial_789",
            evaluation_config=eval_config,
            duration=0.3,
            error=error,
            progress_state=progress,
            optuna_trial_id=None,
        )

        assert result.trial_id == "trial_789"
        # May not have cost in metrics

    def test_failed_timestamp_is_utc(self, eval_config):
        """Test failed result timestamp is in UTC."""
        error = Exception("Test")

        result = build_failed_result(
            trial_id="trial_789",
            evaluation_config=eval_config,
            duration=0.3,
            error=error,
            progress_state=None,
            optuna_trial_id=None,
        )

        assert result.timestamp.tzinfo == UTC


class TestTrialResultFactoryIntegration:
    """Test integration scenarios across all factory functions."""

    def test_all_results_have_required_fields(
        self, eval_config, eval_result, prune_error
    ):
        """Test all result types have required fields."""
        success = build_success_result(
            trial_id="s1",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.0,
            examples_attempted=None,
            total_cost=None,
            optuna_trial_id=None,
        )

        pruned = build_pruned_result(
            trial_id="p1",
            evaluation_config=eval_config,
            duration=0.5,
            prune_error=prune_error,
            progress_state=None,
            optuna_trial_id=None,
        )

        failed = build_failed_result(
            trial_id="f1",
            evaluation_config=eval_config,
            duration=0.3,
            error=Exception("test"),
            progress_state=None,
            optuna_trial_id=None,
        )

        for result in [success, pruned, failed]:
            assert hasattr(result, "trial_id")
            assert hasattr(result, "config")
            assert hasattr(result, "metrics")
            assert hasattr(result, "status")
            assert hasattr(result, "duration")
            assert hasattr(result, "timestamp")
            assert hasattr(result, "metadata")

    def test_status_differentiation(self, eval_config, eval_result, prune_error):
        """Test each factory creates different status."""
        success = build_success_result(
            trial_id="s1",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.0,
            examples_attempted=None,
            total_cost=None,
            optuna_trial_id=None,
        )

        pruned = build_pruned_result(
            trial_id="p1",
            evaluation_config=eval_config,
            duration=0.5,
            prune_error=prune_error,
            progress_state=None,
            optuna_trial_id=None,
        )

        failed = build_failed_result(
            trial_id="f1",
            evaluation_config=eval_config,
            duration=0.3,
            error=Exception("test"),
            progress_state=None,
            optuna_trial_id=None,
        )

        assert success.status == TrialStatus.COMPLETED
        assert pruned.status == TrialStatus.PRUNED
        assert failed.status == TrialStatus.FAILED

    def test_error_message_only_in_failed(self, eval_config, eval_result, prune_error):
        """Test error_message only present in failed results."""
        success = build_success_result(
            trial_id="s1",
            evaluation_config=eval_config,
            eval_result=eval_result,
            duration=1.0,
            examples_attempted=None,
            total_cost=None,
            optuna_trial_id=None,
        )

        pruned = build_pruned_result(
            trial_id="p1",
            evaluation_config=eval_config,
            duration=0.5,
            prune_error=prune_error,
            progress_state=None,
            optuna_trial_id=None,
        )

        failed = build_failed_result(
            trial_id="f1",
            evaluation_config=eval_config,
            duration=0.3,
            error=Exception("test"),
            progress_state=None,
            optuna_trial_id=None,
        )

        assert not hasattr(success, "error_message") or success.error_message is None
        assert not hasattr(pruned, "error_message") or pruned.error_message is None
        assert hasattr(failed, "error_message") and failed.error_message is not None
