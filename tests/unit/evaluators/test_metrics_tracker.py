"""Unit tests for MetricsTracker and summary_stats functionality."""

from unittest.mock import patch

import pytest

from traigent.evaluators.metrics_tracker import (
    CostMetrics,
    ExampleMetrics,
    MetricsTracker,
    ResponseMetrics,
    TokenMetrics,
    extract_llm_metrics,
)


class TestMetricsTracker:
    """Test MetricsTracker functionality."""

    def test_metrics_tracker_initialization(self):
        """Test MetricsTracker initialization."""
        tracker = MetricsTracker()
        assert tracker.example_metrics == []
        assert tracker.start_time is None
        assert tracker.end_time is None

    def test_add_example_metrics(self):
        """Test adding example metrics."""
        tracker = MetricsTracker()
        example = ExampleMetrics(
            tokens=TokenMetrics(input_tokens=100, output_tokens=50),
            response=ResponseMetrics(response_time_ms=1000),
            cost=CostMetrics(input_cost=0.001, output_cost=0.002),
        )
        tracker.add_example_metrics(example)
        assert len(tracker.example_metrics) == 1
        assert tracker.example_metrics[0] == example

    def test_calculate_statistics(self):
        """Test statistical calculations."""
        tracker = MetricsTracker()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = tracker.calculate_statistics(values)

        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["std"] > 1.5 and stats["std"] < 1.6  # ~1.58

    def test_calculate_statistics_empty(self):
        """Test statistics with empty values."""
        tracker = MetricsTracker()
        stats = tracker.calculate_statistics([])

        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["std"] == 0.0

    def test_format_for_backend(self):
        """Test formatting metrics for backend submission."""
        tracker = MetricsTracker()
        tracker.start_tracking()

        # Add some example metrics
        for i in range(3):
            example = ExampleMetrics(
                tokens=TokenMetrics(
                    input_tokens=100 + i * 10, output_tokens=50 + i * 5
                ),
                response=ResponseMetrics(response_time_ms=1000 + i * 100),
                cost=CostMetrics(input_cost=0.001, output_cost=0.002),
                success=True,
            )
            tracker.add_example_metrics(example)

        tracker.end_tracking()

        formatted = tracker.format_for_backend()

        # Check that all expected keys are present
        # The backend format now uses cleaner keys without _mean/_median/_std suffixes
        assert "score" in formatted
        assert "accuracy" in formatted
        assert "duration" in formatted
        assert "input_tokens" in formatted  # Returns mean value directly
        assert "output_tokens" in formatted
        assert "total_tokens" in formatted
        assert "response_time_ms" in formatted  # Returns mean value directly
        assert "cost" in formatted  # Returns mean total cost directly
        assert "total_examples" in formatted
        assert "successful_examples" in formatted

    def test_format_as_summary_stats(self):
        """Test generating pandas.describe()-compatible summary stats."""
        tracker = MetricsTracker()
        tracker.start_tracking()

        # Add some example metrics with one failure
        for i in range(5):
            example = ExampleMetrics(
                tokens=TokenMetrics(
                    input_tokens=100 + i * 10, output_tokens=50 + i * 5
                ),
                response=ResponseMetrics(response_time_ms=1000 + i * 100),
                cost=CostMetrics(
                    input_cost=0.001 + i * 0.0001, output_cost=0.002 + i * 0.0002
                ),
                success=True if i != 2 else False,  # One failure
                error="Test error" if i == 2 else None,
            )
            tracker.add_example_metrics(example)

        tracker.end_tracking()

        summary_stats = tracker.format_as_summary_stats()

        # Check structure
        assert "metrics" in summary_stats
        assert "execution_time" in summary_stats
        assert "total_examples" in summary_stats
        assert "metadata" in summary_stats

        # Check metadata
        assert summary_stats["metadata"]["aggregation_method"] == "pandas.describe"
        assert (
            "sdk_version" in summary_stats["metadata"]
        )  # Check key exists, don't hardcode version
        assert isinstance(
            summary_stats["metadata"]["sdk_version"], str
        )  # Verify it's a string
        assert "timestamp" in summary_stats["metadata"]

        # Check that metrics have pandas.describe() format
        for metric_name, stats in summary_stats["metrics"].items():
            assert "count" in stats, f"Missing 'count' in {metric_name}"
            assert "mean" in stats, f"Missing 'mean' in {metric_name}"
            assert "std" in stats, f"Missing 'std' in {metric_name}"
            assert "min" in stats, f"Missing 'min' in {metric_name}"
            assert "25%" in stats, f"Missing '25%' in {metric_name}"
            assert "50%" in stats, f"Missing '50%' in {metric_name}"
            assert "75%" in stats, f"Missing '75%' in {metric_name}"
            assert "max" in stats, f"Missing 'max' in {metric_name}"

        # Check accuracy metric specifically (4 success out of 5)
        assert summary_stats["metrics"]["accuracy"]["mean"] == 0.8
        assert summary_stats["metrics"]["accuracy"]["count"] == 5

    def test_calculate_describe_stats(self):
        """Test pandas.describe()-compatible statistics calculation."""
        tracker = MetricsTracker()

        # Test with simple values
        values = [1, 2, 3, 4, 5]
        stats = tracker._calculate_describe_stats(values)

        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["min"] == 1
        assert stats["max"] == 5
        assert stats["50%"] == 3.0  # median

        # Check percentiles
        assert stats["25%"] == 2.0
        assert stats["75%"] == 4.0

    def test_calculate_describe_stats_empty(self):
        """Test describe stats with empty values."""
        tracker = MetricsTracker()
        stats = tracker._calculate_describe_stats([])

        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["std"] == 0.0
        assert stats["min"] == 0.0
        assert stats["25%"] == 0.0
        assert stats["50%"] == 0.0
        assert stats["75%"] == 0.0
        assert stats["max"] == 0.0

    def test_empty_summary_stats(self):
        """Test empty summary stats structure."""
        tracker = MetricsTracker()
        empty_stats = tracker._empty_summary_stats()

        # Check that metrics contains expected keys with empty describe stats
        assert "metrics" in empty_stats
        expected_metrics = [
            "accuracy",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "response_time_ms",
            "total_cost",
        ]
        for metric in expected_metrics:
            assert metric in empty_stats["metrics"]
            # Check each metric has pandas.describe() format with 0 values
            stats = empty_stats["metrics"][metric]
            assert stats["count"] == 0
            assert stats["mean"] == 0.0
            assert stats["std"] == 0.0
            assert stats["min"] == 0.0
            assert stats["max"] == 0.0

        assert empty_stats["execution_time"] == 0.0
        assert empty_stats["total_examples"] == 0
        assert empty_stats["metadata"]["aggregation_method"] == "pandas.describe"

    def test_extract_llm_metrics_unknown_model_raises_in_strict_mode(self):
        """Strict cost accounting should fail on unknown priced model."""
        from traigent.utils.cost_calculator import UnknownModelError

        response = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        with patch.dict(
            "os.environ",
            {
                "TRAIGENT_STRICT_COST_ACCOUNTING": "true",
                "TRAIGENT_MOCK_LLM": "false",
                "TRAIGENT_GENERATE_MOCKS": "false",
            },
            clear=False,
        ):
            with pytest.raises(UnknownModelError):
                extract_llm_metrics(
                    response=response,
                    model_name="unknown-model-xyz-123",
                )


class TestTokenMetrics:
    """Test TokenMetrics dataclass."""

    def test_token_metrics_initialization(self):
        """Test TokenMetrics initialization and total calculation."""
        metrics = TokenMetrics(input_tokens=100, output_tokens=50)
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.total_tokens == 150

    def test_token_metrics_defaults(self):
        """Test TokenMetrics default values."""
        metrics = TokenMetrics()
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.total_tokens == 0


class TestResponseMetrics:
    """Test ResponseMetrics dataclass."""

    def test_response_metrics_initialization(self):
        """Test ResponseMetrics initialization."""
        metrics = ResponseMetrics(
            response_time_ms=1000, first_token_ms=100, tokens_per_second=10.5
        )
        assert metrics.response_time_ms == 1000
        assert metrics.first_token_ms == 100
        assert metrics.tokens_per_second == 10.5

    def test_response_metrics_defaults(self):
        """Test ResponseMetrics default values."""
        metrics = ResponseMetrics()
        assert metrics.response_time_ms == 0.0
        assert metrics.first_token_ms is None
        assert metrics.tokens_per_second is None


class TestCostMetrics:
    """Test CostMetrics dataclass."""

    def test_cost_metrics_initialization(self):
        """Test CostMetrics initialization and total calculation."""
        metrics = CostMetrics(input_cost=0.001, output_cost=0.002)
        assert metrics.input_cost == 0.001
        assert metrics.output_cost == 0.002
        assert metrics.total_cost == 0.003

    def test_cost_metrics_defaults(self):
        """Test CostMetrics default values."""
        metrics = CostMetrics()
        assert metrics.input_cost == 0.0
        assert metrics.output_cost == 0.0
        assert metrics.total_cost == 0.0


class TestExampleMetrics:
    """Test ExampleMetrics dataclass."""

    def test_example_metrics_initialization(self):
        """Test ExampleMetrics initialization."""
        metrics = ExampleMetrics(
            tokens=TokenMetrics(input_tokens=100, output_tokens=50),
            response=ResponseMetrics(response_time_ms=1000),
            cost=CostMetrics(input_cost=0.001, output_cost=0.002),
            success=True,
            error=None,
            custom_metrics={"custom": 1.0},
        )
        assert metrics.tokens.total_tokens == 150
        assert metrics.response.response_time_ms == 1000
        assert metrics.cost.total_cost == 0.003
        assert metrics.success is True
        assert metrics.error is None
        assert metrics.custom_metrics["custom"] == 1.0

    def test_example_metrics_defaults(self):
        """Test ExampleMetrics default values."""
        metrics = ExampleMetrics()
        assert metrics.tokens.total_tokens == 0
        assert metrics.response.response_time_ms == 0.0
        assert metrics.cost.total_cost == 0.0
        assert metrics.success is True
        assert metrics.error is None
        assert metrics.custom_metrics == {}


class TestExtractLLMMetrics:
    """Test extract_llm_metrics function."""

    def test_extract_from_openai_format(self):
        """Test extracting metrics from OpenAI-style response."""

        # Mock OpenAI response
        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class MockResponse:
            usage = MockUsage()
            response_time_ms = 1234.5

        metrics = extract_llm_metrics(MockResponse())

        assert metrics.tokens.input_tokens == 100
        assert metrics.tokens.output_tokens == 50
        assert metrics.tokens.total_tokens == 150
        assert metrics.response.response_time_ms == 1234.5

    def test_extract_from_metadata(self, monkeypatch):
        """Test extracting metrics from metadata."""

        # Disable mock mode for this test
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "")
        monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")

        class MockResponse:
            metadata = {
                "tokens": {"input": 200, "output": 100},
                "cost": {"input": 0.002, "output": 0.004, "total": 0.006},
                "response_time_ms": 2000,
            }

        metrics = extract_llm_metrics(MockResponse())

        assert metrics.tokens.input_tokens == 200
        assert metrics.tokens.output_tokens == 100
        assert metrics.cost.input_cost == 0.002
        assert metrics.cost.output_cost == 0.004
        assert metrics.cost.total_cost == 0.006
        assert metrics.response.response_time_ms == 2000

    def test_extract_tokens_per_second_calculation(self):
        """Test that tokens_per_second is calculated correctly."""

        class MockResponse:
            metadata = {
                "tokens": {"input": 100, "output": 100},
                "response_time_ms": 2000,  # 2 seconds
            }

        metrics = extract_llm_metrics(MockResponse())

        # 200 total tokens / 2 seconds = 100 tokens per second
        assert metrics.response.tokens_per_second == 100.0

    def test_extract_from_unknown_format(self):
        """Test extracting metrics from unknown format returns defaults."""

        class MockResponse:
            pass

        metrics = extract_llm_metrics(MockResponse())

        assert metrics.tokens.input_tokens == 0
        assert metrics.tokens.output_tokens == 0
        assert metrics.tokens.total_tokens == 0
        assert metrics.response.response_time_ms == 0.0
        assert metrics.cost.total_cost == 0.0
