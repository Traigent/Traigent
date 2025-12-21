"""Tests for Haystack latency tracking module.

Coverage: Epic 4, Story 4.2 (Track Latency Metrics Per Run)
"""

from __future__ import annotations

import pytest

from traigent.integrations.haystack.latency_tracking import (
    LatencyStats,
    compute_latency_stats,
    extract_latencies_from_results,
    get_latency_metrics,
)


class TestLatencyStats:
    """Tests for LatencyStats dataclass."""

    def test_default_values(self):
        """Test default values are zeros."""
        stats = LatencyStats()
        assert stats.p50_ms == 0.0
        assert stats.p95_ms == 0.0
        assert stats.p99_ms == 0.0
        assert stats.mean_ms == 0.0
        assert stats.min_ms == 0.0
        assert stats.max_ms == 0.0
        assert stats.total_ms == 0.0
        assert stats.count == 0


class TestComputeLatencyStats:
    """Tests for compute_latency_stats function."""

    def test_empty_list(self):
        """Test handling of empty list."""
        stats = compute_latency_stats([])
        assert stats.count == 0
        assert stats.p50_ms == 0.0

    def test_single_value(self):
        """Test handling of single value."""
        stats = compute_latency_stats([0.1])  # 100ms
        assert stats.count == 1
        assert stats.p50_ms == 100.0
        assert stats.p95_ms == 100.0
        assert stats.p99_ms == 100.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 100.0

    def test_known_percentiles(self):
        """Test percentile computation with known values."""
        # 10 values: 0.1 to 1.0 seconds
        latencies = [i / 10.0 for i in range(1, 11)]  # 0.1, 0.2, ..., 1.0
        stats = compute_latency_stats(latencies)

        assert stats.count == 10
        assert stats.min_ms == 100.0  # 0.1s = 100ms
        assert stats.max_ms == 1000.0  # 1.0s = 1000ms

        # Median of [100, 200, ..., 1000] is between 500 and 600
        assert 500 <= stats.p50_ms <= 600

        # P95 should be close to 950
        assert stats.p95_ms > 800

    def test_zeros_excluded_by_default(self):
        """Test that zero values are excluded by default."""
        latencies = [0.0, 0.1, 0.2, 0.0]  # Two zeros
        stats = compute_latency_stats(latencies)

        assert stats.count == 2  # Only non-zero values
        assert stats.min_ms == 100.0
        assert stats.max_ms == 200.0

    def test_zeros_included_when_requested(self):
        """Test that zeros can be included."""
        latencies = [0.0, 0.1, 0.2]
        stats = compute_latency_stats(latencies, include_zeros=True)

        assert stats.count == 3
        assert stats.min_ms == 0.0

    def test_negative_values_excluded(self):
        """Test that negative values are excluded."""
        latencies = [-0.1, 0.1, 0.2]
        stats = compute_latency_stats(latencies)

        assert stats.count == 2
        assert stats.min_ms == 100.0

    def test_mean_calculation(self):
        """Test mean is correctly calculated."""
        latencies = [0.1, 0.2, 0.3]  # 100, 200, 300ms
        stats = compute_latency_stats(latencies)

        assert stats.mean_ms == 200.0  # (100+200+300)/3

    def test_total_calculation(self):
        """Test total is correctly calculated."""
        latencies = [0.1, 0.2, 0.3]  # 100, 200, 300ms
        stats = compute_latency_stats(latencies)

        assert stats.total_ms == 600.0  # 100+200+300


class TestExtractLatenciesFromResults:
    """Tests for extract_latencies_from_results function."""

    def test_extract_from_mock_results(self):
        """Test extraction from mock ExampleResult objects."""

        class MockResult:
            def __init__(self, execution_time: float, success: bool = True):
                self.execution_time = execution_time
                self.success = success

        results = [
            MockResult(0.1),
            MockResult(0.2),
            MockResult(0.3),
        ]

        latencies = extract_latencies_from_results(results)
        assert latencies == [0.1, 0.2, 0.3]

    def test_include_failed_by_default(self):
        """Test that failed results are included by default."""

        class MockResult:
            def __init__(self, execution_time: float, success: bool = True):
                self.execution_time = execution_time
                self.success = success

        results = [
            MockResult(0.1, success=True),
            MockResult(0.2, success=False),
            MockResult(0.3, success=True),
        ]

        latencies = extract_latencies_from_results(results, include_failed=True)
        assert len(latencies) == 3

    def test_exclude_failed_when_requested(self):
        """Test that failed results can be excluded."""

        class MockResult:
            def __init__(self, execution_time: float, success: bool = True):
                self.execution_time = execution_time
                self.success = success

        results = [
            MockResult(0.1, success=True),
            MockResult(0.2, success=False),
            MockResult(0.3, success=True),
        ]

        latencies = extract_latencies_from_results(results, include_failed=False)
        assert latencies == [0.1, 0.3]

    def test_empty_results(self):
        """Test handling of empty results list."""
        latencies = extract_latencies_from_results([])
        assert latencies == []


class TestGetLatencyMetrics:
    """Tests for get_latency_metrics function."""

    def test_all_fields_included(self):
        """Test all required fields are in metrics dict."""
        stats = LatencyStats(
            p50_ms=100.0,
            p95_ms=200.0,
            p99_ms=300.0,
            mean_ms=150.0,
            min_ms=50.0,
            max_ms=400.0,
            total_ms=1500.0,
            count=10,
        )

        metrics = get_latency_metrics(stats)

        assert "latency_p50_ms" in metrics
        assert "latency_p95_ms" in metrics
        assert "latency_p99_ms" in metrics
        assert "latency_mean_ms" in metrics
        assert "latency_min_ms" in metrics
        assert "latency_max_ms" in metrics
        assert "total_latency_ms" in metrics
        assert "latency_count" in metrics

        assert metrics["latency_p50_ms"] == 100.0
        assert metrics["latency_p95_ms"] == 200.0
        assert metrics["latency_p99_ms"] == 300.0
        assert metrics["latency_mean_ms"] == 150.0

    def test_empty_stats(self):
        """Test metrics from empty stats."""
        stats = LatencyStats()
        metrics = get_latency_metrics(stats)

        assert metrics["latency_p50_ms"] == 0.0
        assert metrics["latency_count"] == 0


class TestIntegration:
    """Integration tests for latency tracking."""

    @pytest.mark.asyncio
    async def test_evaluator_includes_latency_metrics(self):
        """Test HaystackEvaluator includes latency metrics in results."""
        from unittest.mock import MagicMock

        from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

        # Create mock pipeline
        pipeline = MagicMock()
        pipeline.run.return_value = {
            "llm": {
                "replies": ["Response"],
            }
        }

        dataset = EvaluationDataset.from_dicts(
            [
                {"input": {"query": "test1"}, "expected": "result1"},
                {"input": {"query": "test2"}, "expected": "result2"},
                {"input": {"query": "test3"}, "expected": "result3"},
            ]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
            track_latency=True,
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # Check latency metrics are included
        assert "latency_p50_ms" in result.aggregated_metrics
        assert "latency_p95_ms" in result.aggregated_metrics
        assert "latency_p99_ms" in result.aggregated_metrics
        assert "latency_mean_ms" in result.aggregated_metrics
        assert "latency_count" in result.aggregated_metrics

        # Should have counted all examples
        assert result.aggregated_metrics["latency_count"] == 3

    @pytest.mark.asyncio
    async def test_evaluator_latency_tracking_disabled(self):
        """Test latency metrics are not included when tracking disabled."""
        from unittest.mock import MagicMock

        from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

        pipeline = MagicMock()
        pipeline.run.return_value = {"llm": {"replies": ["Response"]}}

        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
            track_latency=False,  # Disabled
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # Check latency metrics are NOT included
        assert "latency_p50_ms" not in result.aggregated_metrics
        assert "latency_p95_ms" not in result.aggregated_metrics

    @pytest.mark.asyncio
    async def test_evaluator_includes_both_cost_and_latency(self):
        """Test both cost and latency metrics are tracked together."""
        from unittest.mock import MagicMock

        from traigent.integrations.haystack import EvaluationDataset, HaystackEvaluator

        pipeline = MagicMock()
        pipeline.run.return_value = {
            "llm": {
                "replies": ["Response"],
                "meta": [
                    {
                        "model": "gpt-4o",
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                    }
                ],
            }
        }

        dataset = EvaluationDataset.from_dicts(
            [{"input": {"query": "test"}, "expected": "result"}]
        )

        evaluator = HaystackEvaluator(
            pipeline=pipeline,
            haystack_dataset=dataset,
            output_key="llm.replies",
            track_costs=True,
            track_latency=True,
        )

        result = await evaluator.evaluate(
            func=pipeline.run,
            config={},
            dataset=dataset.to_core_dataset(),
        )

        # Check both cost and latency metrics are present
        assert "total_cost" in result.aggregated_metrics
        assert "latency_p50_ms" in result.aggregated_metrics


class TestPercentileAlgorithm:
    """Tests for the percentile algorithm accuracy."""

    def test_percentile_matches_expected(self):
        """Test that our percentile matches expected values."""
        # Using 100 values for easier percentile calculation
        latencies = [i / 1000.0 for i in range(1, 101)]  # 0.001 to 0.1 seconds
        stats = compute_latency_stats(latencies)

        # p50 should be around the 50th value (0.050s = 50ms)
        assert 49 <= stats.p50_ms <= 51

        # p95 should be around the 95th value (0.095s = 95ms)
        assert 94 <= stats.p95_ms <= 96

        # p99 should be around the 99th value (0.099s = 99ms)
        assert 98 <= stats.p99_ms <= 100

    def test_small_sample_percentiles(self):
        """Test percentiles with small samples."""
        # Only 2 values
        latencies = [0.1, 0.2]  # 100ms, 200ms
        stats = compute_latency_stats(latencies)

        # With 2 values, interpolation should give reasonable results
        assert stats.p50_ms == 150.0  # Midpoint
        assert 100 <= stats.p99_ms <= 200
