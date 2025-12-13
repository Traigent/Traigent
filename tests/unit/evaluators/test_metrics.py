"""Unit tests for traigent.evaluators.metrics.

Tests for metrics computation and evaluation result handling.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Performance FUNC-EVAL-METRICS REQ-EVAL-005 SYNC-OptimizationFlow

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from traigent.evaluators.metrics import MetricsComputer, MetricsEvaluationResult
from traigent.invokers.base import InvocationResult


class TestMetricsEvaluationResult:
    """Tests for MetricsEvaluationResult class."""

    @pytest.fixture
    def default_result(self) -> MetricsEvaluationResult:
        """Create result with default values."""
        return MetricsEvaluationResult()

    @pytest.fixture
    def populated_result(self) -> MetricsEvaluationResult:
        """Create result with populated values."""
        return MetricsEvaluationResult(
            metrics={"accuracy": 0.85, "success_rate": 0.9},
            total_invocations=10,
            successful_invocations=9,
            duration=1.5,
            metadata={"test": "value"},
        )

    def test_default_initialization(
        self, default_result: MetricsEvaluationResult
    ) -> None:
        """Test result initializes with default values."""
        assert default_result.metrics == {}
        assert default_result.total_invocations == 0
        assert default_result.successful_invocations == 0
        assert default_result.duration == 0.0
        assert default_result.metadata == {}

    def test_populated_initialization(
        self, populated_result: MetricsEvaluationResult
    ) -> None:
        """Test result initializes with provided values."""
        assert populated_result.metrics == {"accuracy": 0.85, "success_rate": 0.9}
        assert populated_result.total_invocations == 10
        assert populated_result.successful_invocations == 9
        assert populated_result.duration == 1.5
        assert populated_result.metadata == {"test": "value"}

    def test_success_rate_with_zero_invocations(
        self, default_result: MetricsEvaluationResult
    ) -> None:
        """Test success_rate returns 0.0 when total_invocations is 0."""
        assert default_result.success_rate == 0.0

    def test_success_rate_with_all_successful(self) -> None:
        """Test success_rate returns 1.0 when all invocations successful."""
        result = MetricsEvaluationResult(
            total_invocations=10, successful_invocations=10
        )
        assert result.success_rate == 1.0

    def test_success_rate_with_partial_success(self) -> None:
        """Test success_rate calculates correctly with partial success."""
        result = MetricsEvaluationResult(total_invocations=10, successful_invocations=7)
        assert result.success_rate == 0.7

    def test_success_rate_with_no_success(self) -> None:
        """Test success_rate returns 0.0 when no invocations successful."""
        result = MetricsEvaluationResult(total_invocations=10, successful_invocations=0)
        assert result.success_rate == 0.0

    def test_error_rate_with_zero_invocations(
        self, default_result: MetricsEvaluationResult
    ) -> None:
        """Test error_rate returns 1.0 when total_invocations is 0."""
        assert default_result.error_rate == 1.0

    def test_error_rate_with_all_successful(self) -> None:
        """Test error_rate returns 0.0 when all invocations successful."""
        result = MetricsEvaluationResult(
            total_invocations=10, successful_invocations=10
        )
        assert result.error_rate == 0.0

    def test_error_rate_with_partial_success(self) -> None:
        """Test error_rate calculates correctly with partial success."""
        result = MetricsEvaluationResult(total_invocations=10, successful_invocations=7)
        assert abs(result.error_rate - 0.3) < 1e-9

    def test_error_rate_with_no_success(self) -> None:
        """Test error_rate returns 1.0 when no invocations successful."""
        result = MetricsEvaluationResult(total_invocations=10, successful_invocations=0)
        assert result.error_rate == 1.0


class TestMetricsComputer:
    """Tests for MetricsComputer class."""

    @pytest.fixture
    def default_computer(self) -> MetricsComputer:
        """Create computer with default metrics."""
        return MetricsComputer()

    @pytest.fixture
    def custom_computer(self) -> MetricsComputer:
        """Create computer with custom metrics list."""
        return MetricsComputer(
            metrics=["accuracy", "success_rate", "avg_execution_time"]
        )

    @pytest.fixture
    def successful_results(self) -> list[InvocationResult]:
        """Create list of successful invocation results."""
        return [
            InvocationResult(result="output1", execution_time=1.0, is_successful=True),
            InvocationResult(result="output2", execution_time=1.5, is_successful=True),
            InvocationResult(result="output3", execution_time=2.0, is_successful=True),
        ]

    @pytest.fixture
    def mixed_results(self) -> list[InvocationResult]:
        """Create list of mixed successful and failed results."""
        return [
            InvocationResult(result="output1", execution_time=1.0, is_successful=True),
            InvocationResult(
                result=None, execution_time=0.0, error="Error", is_successful=False
            ),
            InvocationResult(result="output3", execution_time=2.0, is_successful=True),
        ]

    def test_initialization_with_default_metrics(
        self, default_computer: MetricsComputer
    ) -> None:
        """Test computer initializes with default metrics."""
        assert default_computer.metrics == ["accuracy", "success_rate"]

    def test_initialization_with_custom_metrics(
        self, custom_computer: MetricsComputer
    ) -> None:
        """Test computer initializes with custom metrics list."""
        assert custom_computer.metrics == [
            "accuracy",
            "success_rate",
            "avg_execution_time",
        ]

    def test_initialization_with_none_metrics(self) -> None:
        """Test computer initializes with default metrics when None provided."""
        computer = MetricsComputer(metrics=None)
        assert computer.metrics == ["accuracy", "success_rate"]

    def test_compute_metrics_raises_on_length_mismatch(
        self,
        default_computer: MetricsComputer,
        successful_results: list[InvocationResult],
    ) -> None:
        """Test compute_metrics raises ValueError when lengths don't match."""
        expected_outputs = ["output1", "output2"]  # Only 2 outputs for 3 results

        with pytest.raises(
            ValueError,
            match="Number of invocation results \\(3\\) must match number of expected outputs \\(2\\)",
        ):
            default_computer.compute_metrics(successful_results, expected_outputs)

    def test_compute_metrics_with_all_successful_exact_match(
        self,
        default_computer: MetricsComputer,
        successful_results: list[InvocationResult],
    ) -> None:
        """Test compute_metrics with all successful results matching expected."""
        expected_outputs = ["output1", "output2", "output3"]

        result = default_computer.compute_metrics(successful_results, expected_outputs)

        assert result.total_invocations == 3
        assert result.successful_invocations == 3
        assert result.metrics["success_rate"] == 1.0
        assert result.metrics["accuracy"] == 1.0
        assert result.duration > 0
        assert result.metadata["metrics_requested"] == ["accuracy", "success_rate"]
        assert result.metadata["successful_pairs"] == 3
        assert result.metadata["failed_invocations"] == 0

    def test_compute_metrics_with_partial_accuracy(
        self,
        default_computer: MetricsComputer,
        successful_results: list[InvocationResult],
    ) -> None:
        """Test compute_metrics with some outputs not matching expected."""
        expected_outputs = ["output1", "wrong", "output3"]

        result = default_computer.compute_metrics(successful_results, expected_outputs)

        assert result.total_invocations == 3
        assert result.successful_invocations == 3
        assert result.metrics["success_rate"] == 1.0
        assert result.metrics["accuracy"] == 2 / 3  # 2 out of 3 match

    def test_compute_metrics_with_mixed_success(
        self, default_computer: MetricsComputer, mixed_results: list[InvocationResult]
    ) -> None:
        """Test compute_metrics with mixed successful and failed results."""
        expected_outputs = ["output1", "output2", "output3"]

        result = default_computer.compute_metrics(mixed_results, expected_outputs)

        assert result.total_invocations == 3
        assert result.successful_invocations == 2
        assert result.metrics["success_rate"] == 2 / 3
        assert result.metrics["accuracy"] == 1.0  # Both successful ones match

    def test_compute_metrics_with_none_expected_outputs(
        self, default_computer: MetricsComputer
    ) -> None:
        """Test compute_metrics with None values in expected outputs."""
        invocation_results = [
            InvocationResult(result="output1", is_successful=True),
            InvocationResult(result="output2", is_successful=True),
        ]
        expected_outputs = ["output1", None]

        result = default_computer.compute_metrics(invocation_results, expected_outputs)

        assert result.total_invocations == 2
        assert result.successful_invocations == 2
        assert result.metadata["successful_pairs"] == 1  # Only 1 valid pair

    def test_compute_metrics_with_error_rate(self) -> None:
        """Test compute_metrics computes error_rate when requested."""
        computer = MetricsComputer(metrics=["success_rate", "error_rate"])
        invocation_results = [
            InvocationResult(result="output1", is_successful=True),
            InvocationResult(error="Error", is_successful=False),
        ]
        expected_outputs = ["output1", "output2"]

        result = computer.compute_metrics(invocation_results, expected_outputs)

        assert result.metrics["success_rate"] == 0.5
        assert result.metrics["error_rate"] == 0.5

    def test_compute_metrics_with_avg_execution_time(self) -> None:
        """Test compute_metrics computes avg_execution_time when requested."""
        computer = MetricsComputer(metrics=["avg_execution_time"])
        invocation_results = [
            InvocationResult(result="output1", execution_time=1.0, is_successful=True),
            InvocationResult(result="output2", execution_time=2.0, is_successful=True),
            InvocationResult(result="output3", execution_time=3.0, is_successful=True),
        ]
        expected_outputs = ["output1", "output2", "output3"]

        result = computer.compute_metrics(invocation_results, expected_outputs)

        assert result.metrics["avg_execution_time"] == 2.0

    def test_compute_metrics_with_zero_execution_times(self) -> None:
        """Test compute_metrics handles zero execution times correctly."""
        computer = MetricsComputer(metrics=["avg_execution_time"])
        invocation_results = [
            InvocationResult(result="output1", execution_time=0.0, is_successful=True),
            InvocationResult(result="output2", execution_time=0.0, is_successful=True),
        ]
        expected_outputs = ["output1", "output2"]

        result = computer.compute_metrics(invocation_results, expected_outputs)

        assert result.metrics["avg_execution_time"] == 0.0

    def test_compute_metrics_with_avg_output_length_string(self) -> None:
        """Test compute_metrics computes avg_output_length for string outputs."""
        computer = MetricsComputer(metrics=["avg_output_length"])
        invocation_results = [
            InvocationResult(result="abc", is_successful=True),
            InvocationResult(result="abcde", is_successful=True),
            InvocationResult(result="abcdefg", is_successful=True),
        ]
        expected_outputs = ["abc", "abcde", "abcdefg"]

        result = computer.compute_metrics(invocation_results, expected_outputs)

        assert result.metrics["avg_output_length"] == 5.0  # (3+5+7)/3

    def test_compute_metrics_with_avg_output_length_list(self) -> None:
        """Test compute_metrics computes avg_output_length for list outputs."""
        computer = MetricsComputer(metrics=["avg_output_length"])
        invocation_results = [
            InvocationResult(result=[1, 2], is_successful=True),
            InvocationResult(result=[1, 2, 3, 4], is_successful=True),
        ]
        expected_outputs = [[1, 2], [1, 2, 3, 4]]

        result = computer.compute_metrics(invocation_results, expected_outputs)

        assert result.metrics["avg_output_length"] == 3.0  # (2+4)/2

    def test_compute_metrics_with_avg_output_length_no_valid_outputs(self) -> None:
        """Test compute_metrics handles no valid outputs for avg_output_length."""
        computer = MetricsComputer(metrics=["avg_output_length"])
        invocation_results = [
            InvocationResult(result=123, is_successful=True),  # Not a string or len()
            InvocationResult(result=456, is_successful=True),
        ]
        expected_outputs = [123, 456]

        result = computer.compute_metrics(invocation_results, expected_outputs)

        assert result.metrics["avg_output_length"] == 0.0

    def test_compute_metrics_with_avg_output_length_failed_invocations(self) -> None:
        """Test compute_metrics handles failed invocations for avg_output_length."""
        computer = MetricsComputer(metrics=["avg_output_length"])
        invocation_results = [
            InvocationResult(error="Error", is_successful=False),
            InvocationResult(error="Error", is_successful=False),
        ]
        expected_outputs = ["output1", "output2"]

        result = computer.compute_metrics(invocation_results, expected_outputs)

        assert result.metrics["avg_output_length"] == 0.0

    def test_compute_metrics_with_accuracy_no_successful_pairs(self) -> None:
        """Test compute_metrics sets accuracy to 0.0 when no successful pairs."""
        computer = MetricsComputer(metrics=["accuracy"])
        invocation_results = [
            InvocationResult(error="Error", is_successful=False),
            InvocationResult(error="Error", is_successful=False),
        ]
        expected_outputs = ["output1", "output2"]

        result = computer.compute_metrics(invocation_results, expected_outputs)

        assert result.metrics["accuracy"] == 0.0

    def test_compute_metrics_with_empty_inputs(
        self, default_computer: MetricsComputer
    ) -> None:
        """Test compute_metrics handles empty input lists."""
        result = default_computer.compute_metrics([], [])

        assert result.total_invocations == 0
        assert result.successful_invocations == 0
        assert result.metrics["success_rate"] == 0.0
        assert result.metrics["accuracy"] == 0.0

    @patch("traigent.evaluators.metrics.time.time")
    def test_compute_metrics_tracks_duration(
        self,
        mock_time: MagicMock,
        default_computer: MetricsComputer,
        successful_results: list[InvocationResult],
    ) -> None:
        """Test compute_metrics tracks evaluation duration."""
        mock_time.side_effect = [100.0, 102.5]  # Start and end times
        expected_outputs = ["output1", "output2", "output3"]

        result = default_computer.compute_metrics(successful_results, expected_outputs)

        assert result.duration == 2.5
        assert result.metadata["evaluation_start_time"] == 100.0
        assert result.metadata["evaluation_end_time"] == 102.5

    def test_add_custom_metric_new_metric(
        self, default_computer: MetricsComputer
    ) -> None:
        """Test add_custom_metric adds new metric to list."""

        def custom_func(outputs: list, expected: list) -> float:
            return 0.5

        default_computer.add_custom_metric("custom_metric", custom_func)

        assert "custom_metric" in default_computer.metrics
        assert hasattr(default_computer, "_custom_metrics")
        assert "custom_metric" in default_computer._custom_metrics

    def test_add_custom_metric_existing_metric(
        self, default_computer: MetricsComputer
    ) -> None:
        """Test add_custom_metric doesn't duplicate existing metric."""
        default_computer.metrics = ["accuracy", "custom_metric"]

        def custom_func(outputs: list, expected: list) -> float:
            return 0.5

        default_computer.add_custom_metric("custom_metric", custom_func)

        assert default_computer.metrics.count("custom_metric") == 1
        assert "custom_metric" in default_computer._custom_metrics

    def test_compute_custom_metrics_no_custom_metrics(
        self, default_computer: MetricsComputer
    ) -> None:
        """Test compute_custom_metrics returns empty dict when no custom metrics."""
        invocation_results = [InvocationResult(result="output1", is_successful=True)]
        expected_outputs = ["output1"]

        result = default_computer.compute_custom_metrics(
            invocation_results, expected_outputs
        )

        assert result == {}

    def test_compute_custom_metrics_with_successful_computation(
        self, default_computer: MetricsComputer
    ) -> None:
        """Test compute_custom_metrics computes custom metrics successfully."""

        def custom_func(outputs: list, expected: list) -> float:
            return len(outputs) / (len(expected) + 1)

        default_computer.add_custom_metric("custom_metric", custom_func)

        invocation_results = [
            InvocationResult(result="output1", is_successful=True),
            InvocationResult(result="output2", is_successful=True),
        ]
        expected_outputs = ["output1", "output2"]

        result = default_computer.compute_custom_metrics(
            invocation_results, expected_outputs
        )

        assert "custom_metric" in result
        assert result["custom_metric"] == 2 / 3

    def test_compute_custom_metrics_filters_failed_invocations(
        self, default_computer: MetricsComputer
    ) -> None:
        """Test compute_custom_metrics filters out failed invocations."""

        def custom_func(outputs: list, expected: list) -> float:
            return len(outputs)

        default_computer.add_custom_metric("count_metric", custom_func)

        invocation_results = [
            InvocationResult(result="output1", is_successful=True),
            InvocationResult(error="Error", is_successful=False),
            InvocationResult(result="output3", is_successful=True),
        ]
        expected_outputs = ["output1", "output2", "output3"]

        result = default_computer.compute_custom_metrics(
            invocation_results, expected_outputs
        )

        assert result["count_metric"] == 2.0  # Only 2 successful

    def test_compute_custom_metrics_filters_none_expected(
        self, default_computer: MetricsComputer
    ) -> None:
        """Test compute_custom_metrics filters out None expected outputs."""

        def custom_func(outputs: list, expected: list) -> float:
            return len(outputs)

        default_computer.add_custom_metric("count_metric", custom_func)

        invocation_results = [
            InvocationResult(result="output1", is_successful=True),
            InvocationResult(result="output2", is_successful=True),
        ]
        expected_outputs = ["output1", None]

        result = default_computer.compute_custom_metrics(
            invocation_results, expected_outputs
        )

        assert result["count_metric"] == 1.0  # Only 1 valid pair

    def test_compute_custom_metrics_handles_exception(
        self, default_computer: MetricsComputer
    ) -> None:
        """Test compute_custom_metrics handles exceptions gracefully."""

        def failing_func(outputs: list, expected: list) -> float:
            raise ValueError("Custom metric failed")

        default_computer.add_custom_metric("failing_metric", failing_func)

        invocation_results = [InvocationResult(result="output1", is_successful=True)]
        expected_outputs = ["output1"]

        result = default_computer.compute_custom_metrics(
            invocation_results, expected_outputs
        )

        assert result["failing_metric"] == 0.0

    def test_compute_custom_metrics_converts_to_float(
        self, default_computer: MetricsComputer
    ) -> None:
        """Test compute_custom_metrics converts result to float."""

        def int_returning_func(outputs: list, expected: list) -> int:
            return 5

        default_computer.add_custom_metric("int_metric", int_returning_func)

        invocation_results = [InvocationResult(result="output1", is_successful=True)]
        expected_outputs = ["output1"]

        result = default_computer.compute_custom_metrics(
            invocation_results, expected_outputs
        )

        assert result["int_metric"] == 5.0
        assert isinstance(result["int_metric"], float)

    def test_compute_custom_metrics_aligns_mismatched_arrays(
        self, default_computer: MetricsComputer
    ) -> None:
        """Test compute_custom_metrics aligns outputs and expected when lengths differ."""

        def custom_func(outputs: list, expected: list) -> float:
            # Should only receive aligned successful outputs with valid expected
            return len(outputs)

        default_computer.add_custom_metric("count_metric", custom_func)

        invocation_results = [
            InvocationResult(result="output1", is_successful=True),
            InvocationResult(error="Error", is_successful=False),
            InvocationResult(result="output3", is_successful=True),
            InvocationResult(result="output4", is_successful=True),
        ]
        expected_outputs = ["output1", "output2", None, "output4"]

        result = default_computer.compute_custom_metrics(
            invocation_results, expected_outputs
        )

        # Should only count successful invocations with non-None expected
        assert result["count_metric"] == 2.0
