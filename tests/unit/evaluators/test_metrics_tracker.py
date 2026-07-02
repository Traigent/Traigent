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
        assert "execution_time_ms" in formatted
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

    def test_extract_llm_metrics_uses_response_model_when_model_name_missing(
        self, monkeypatch
    ):
        """Cost is computed from response.model when config has no 'model' key.

        Multi-model/multi-step agents have no single config 'model' key, so the
        central extractor must fall back to the model carried on the response
        itself instead of skipping cost calculation entirely (#1599).
        """
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "")
        monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")

        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class MockResponse:
            model = "gpt-4o-mini"
            usage = MockUsage()

        metrics = extract_llm_metrics(MockResponse(), model_name=None)

        assert metrics.tokens.input_tokens == 100
        assert metrics.tokens.output_tokens == 50
        assert metrics.cost.total_cost > 0

    def test_extract_llm_metrics_uses_response_metadata_model(self, monkeypatch):
        """Cost falls back to response_metadata['model'] (LangChain shape)."""
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "")
        monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")

        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class MockResponse:
            usage = MockUsage()
            response_metadata = {"model": "gpt-4o-mini"}

        metrics = extract_llm_metrics(MockResponse(), model_name=None)

        assert metrics.tokens.input_tokens == 100
        assert metrics.tokens.output_tokens == 50
        assert metrics.cost.total_cost > 0

    def test_extract_llm_metrics_uses_llm_output_model(self, monkeypatch):
        """Cost falls back to llm_output['model_name'] (LangChain shape)."""
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "")
        monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")

        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class MockResponse:
            usage = MockUsage()
            llm_output = {"model_name": "gpt-4o-mini"}

        metrics = extract_llm_metrics(MockResponse(), model_name=None)

        assert metrics.tokens.input_tokens == 100
        assert metrics.tokens.output_tokens == 50
        assert metrics.cost.total_cost > 0

    def test_extract_llm_metrics_skips_cost_when_no_model_anywhere(
        self, monkeypatch, caplog
    ):
        """Cost stays $0 and the skip warning still fires when neither the
        config nor the response yields a model name (unchanged behavior)."""
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "")
        monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")

        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class MockResponse:
            usage = MockUsage()

        with caplog.at_level("WARNING"):
            metrics = extract_llm_metrics(MockResponse(), model_name=None)

        assert metrics.tokens.input_tokens == 100
        assert metrics.tokens.output_tokens == 50
        assert metrics.cost.total_cost == 0.0
        assert any(
            "Cost calculation skipped: model_name is None/empty" in record.message
            for record in caplog.records
        )

    def test_extract_llm_metrics_survives_raising_model_property(
        self, monkeypatch, caplog
    ):
        """A response whose ``model`` accessor raises must not blow up
        ``extract_llm_metrics``; it should degrade to skipped-cost, matching
        the "no model anywhere" behavior (#1599 follow-up)."""
        monkeypatch.setenv("TRAIGENT_MOCK_LLM", "")
        monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")

        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class MockResponse:
            usage = MockUsage()

            @property
            def model(self):
                raise RuntimeError("boom")

            @property
            def response_metadata(self):
                raise RuntimeError("boom")

            @property
            def llm_output(self):
                raise RuntimeError("boom")

        with caplog.at_level("WARNING"):
            metrics = extract_llm_metrics(MockResponse(), model_name=None)

        assert metrics.tokens.input_tokens == 100
        assert metrics.tokens.output_tokens == 50
        assert metrics.cost.total_cost == 0.0
        assert any(
            "Cost calculation skipped: model_name is None/empty" in record.message
            for record in caplog.records
        )


class TestOpenRouterCostExtraction:
    """Tests for OpenRouter provider-reported cost extraction (issue #1317).

    OpenRouter passes the actual call cost in LiteLLM's
    ``response._hidden_params['response_cost']`` and/or ``response.usage.cost``.
    Models served via OpenRouter are often absent from LiteLLM's static pricing
    table, so ``cost_from_tokens`` returns $0.00 for them.  The SDK must fall
    back to the provider-reported cost from these fields.
    """

    def test_extract_cost_from_hidden_params_response_cost(self):
        """Cost in _hidden_params['response_cost'] is captured (OpenRouter via LiteLLM)."""
        from traigent.evaluators.metrics_tracker import (
            GenericResponseHandler,
        )

        class MockUsage:
            prompt_tokens = 200
            completion_tokens = 100
            total_tokens = 300

        class MockHiddenParams(dict):
            """Minimal LiteLLM HiddenParams stand-in (dict + .get())."""

        hidden = MockHiddenParams(response_cost=0.00456)

        class MockLiteLLMResponse:
            model = "openrouter/meta-llama/llama-3.1-8b-instruct"
            usage = MockUsage()
            _hidden_params = hidden
            response_time_ms = 0.0

        # Use GenericResponseHandler directly so we exercise extract_metadata_cost
        # without depending on which handler wins the chain.
        handler = GenericResponseHandler()
        cost = handler.extract_metadata_cost(MockLiteLLMResponse())

        assert cost.total_cost == pytest.approx(0.00456), (
            "extract_metadata_cost must return the provider-reported cost from "
            "_hidden_params['response_cost'] when the standard cost field is absent"
        )

    def test_extract_cost_from_usage_cost_field(self):
        """Cost in usage.cost is captured as a fallback (LiteLLM Usage object)."""
        from traigent.evaluators.metrics_tracker import GenericResponseHandler

        class MockUsage:
            prompt_tokens = 150
            completion_tokens = 75
            total_tokens = 225
            cost = 0.00789  # LiteLLM Usage.cost field

        class MockLiteLLMResponse:
            model = "openrouter/google/gemma-7b-it"
            usage = MockUsage()
            response_time_ms = 0.0

        handler = GenericResponseHandler()
        cost = handler.extract_metadata_cost(MockLiteLLMResponse())

        assert cost.total_cost == pytest.approx(0.00789), (
            "extract_metadata_cost must return the provider-reported cost from "
            "usage.cost when _hidden_params is absent"
        )

    def test_hidden_params_takes_precedence_over_usage_cost(self):
        """_hidden_params.response_cost takes precedence over usage.cost."""
        from traigent.evaluators.metrics_tracker import GenericResponseHandler

        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150
            cost = 0.0001  # Should be ignored — _hidden_params wins

        class MockHiddenParams(dict):
            pass

        class MockLiteLLMResponse:
            model = "openrouter/mistralai/mistral-7b-instruct"
            usage = MockUsage()
            _hidden_params = MockHiddenParams(response_cost=0.00222)
            response_time_ms = 0.0

        handler = GenericResponseHandler()
        cost = handler.extract_metadata_cost(MockLiteLLMResponse())

        assert cost.total_cost == pytest.approx(0.00222), (
            "_hidden_params.response_cost must take precedence over usage.cost"
        )

    def test_zero_response_cost_falls_through_to_usage_cost(self):
        """A zero response_cost in _hidden_params does not block usage.cost."""
        from traigent.evaluators.metrics_tracker import GenericResponseHandler

        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150
            cost = 0.00333

        class MockHiddenParams(dict):
            pass

        class MockLiteLLMResponse:
            model = "openrouter/openai/gpt-3.5-turbo"
            usage = MockUsage()
            _hidden_params = MockHiddenParams(response_cost=0.0)
            response_time_ms = 0.0

        handler = GenericResponseHandler()
        cost = handler.extract_metadata_cost(MockLiteLLMResponse())

        assert cost.total_cost == pytest.approx(0.00333), (
            "A zero response_cost must be treated as absent so usage.cost is used"
        )

    def test_no_provider_cost_returns_zero(self):
        """When neither _hidden_params nor usage.cost is set, cost stays 0."""
        from traigent.evaluators.metrics_tracker import GenericResponseHandler

        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50
            total_tokens = 150

        class MockLiteLLMResponse:
            model = "openrouter/some-provider/some-model"
            usage = MockUsage()
            response_time_ms = 0.0

        handler = GenericResponseHandler()
        cost = handler.extract_metadata_cost(MockLiteLLMResponse())

        assert cost.total_cost == 0.0

    def test_extract_llm_metrics_picks_up_openrouter_cost(self, monkeypatch):
        """End-to-end: extract_llm_metrics returns the provider-reported cost.

        Simulates a model absent from LiteLLM's pricing table (strict=False
        returns 0.0 from cost_from_tokens) but whose actual cost is in
        _hidden_params['response_cost'].
        """
        monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")

        class MockUsage:
            prompt_tokens = 200
            completion_tokens = 80
            total_tokens = 280

        class MockHiddenParams(dict):
            pass

        class MockLiteLLMOpenRouterResponse:
            model = "openrouter/meta-llama/llama-3.1-8b-instruct"
            usage = MockUsage()
            _hidden_params = MockHiddenParams(response_cost=0.00099)
            response_time_ms = 50.0

        # Stub cost_from_tokens to simulate an unpriced model returning (0, 0).
        # The function is imported lazily inside _compute_cost, so patch it at
        # its definition site in cost_calculator.
        monkeypatch.setattr(
            "traigent.utils.cost_calculator.cost_from_tokens",
            lambda *a, **kw: (0.0, 0.0),
        )

        metrics = extract_llm_metrics(
            MockLiteLLMOpenRouterResponse(),
            model_name="openrouter/meta-llama/llama-3.1-8b-instruct",
        )

        assert metrics.tokens.input_tokens == 200
        assert metrics.tokens.output_tokens == 80
        assert metrics.cost.total_cost == pytest.approx(0.00099), (
            "extract_llm_metrics must report the OpenRouter provider-reported cost "
            "from _hidden_params['response_cost'] even when the model is not in "
            "LiteLLM's pricing table (issue #1317)"
        )

    def test_reported_cost_below_token_floor_is_clamped(self, monkeypatch, caplog):
        """Known model + real tokens prevent implausibly low reported cost."""
        from traigent.utils.cost_calculator import cost_from_tokens

        monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")

        class MockUsage:
            prompt_tokens = 1000
            completion_tokens = 500
            total_tokens = 1500

        class MockHiddenParams(dict):
            pass

        class MockLiteLLMResponse:
            model = "gpt-4o-mini"
            usage = MockUsage()
            _hidden_params = MockHiddenParams(response_cost=0.00000001)

        expected = sum(cost_from_tokens(1000, 500, "gpt-4o-mini", strict=False))

        with caplog.at_level("WARNING"):
            metrics = extract_llm_metrics(
                MockLiteLLMResponse(), model_name="gpt-4o-mini"
            )

        assert metrics.cost.total_cost == pytest.approx(expected)
        assert "implausibly below token-derived estimate" in caplog.text

    def test_plausible_reported_cost_is_unchanged(self, monkeypatch, caplog):
        """Honest provider/user costs remain authoritative when plausible."""
        from traigent.utils.cost_calculator import cost_from_tokens

        monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")

        class MockUsage:
            prompt_tokens = 1000
            completion_tokens = 500
            total_tokens = 1500

        class MockHiddenParams(dict):
            pass

        token_cost = sum(cost_from_tokens(1000, 500, "gpt-4o-mini", strict=False))
        reported_cost = token_cost * 0.75

        class MockLiteLLMResponse:
            model = "gpt-4o-mini"
            usage = MockUsage()
            _hidden_params = MockHiddenParams(response_cost=reported_cost)

        with caplog.at_level("WARNING"):
            metrics = extract_llm_metrics(
                MockLiteLLMResponse(), model_name="gpt-4o-mini"
            )

        assert metrics.cost.total_cost == pytest.approx(reported_cost)
        assert "implausibly below token-derived estimate" not in caplog.text

    def test_unknown_model_accepts_reported_cost_without_clamp(
        self, monkeypatch, caplog
    ):
        """Unknown pricing is the residual BYOK trust boundary."""
        monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")

        class MockUsage:
            prompt_tokens = 1000
            completion_tokens = 500
            total_tokens = 1500

        class MockHiddenParams(dict):
            pass

        class MockLiteLLMResponse:
            model = "unknown-model-xyz-123"
            usage = MockUsage()
            _hidden_params = MockHiddenParams(response_cost=0.00000001)

        with caplog.at_level("WARNING"):
            metrics = extract_llm_metrics(
                MockLiteLLMResponse(), model_name="unknown-model-xyz-123"
            )

        assert metrics.cost.total_cost == pytest.approx(0.00000001)
        assert "implausibly below token-derived estimate" not in caplog.text

    def test_no_tokens_accepts_reported_cost_without_clamp(self, monkeypatch, caplog):
        """No token data means there is no token-derived floor to enforce."""
        monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", "")

        class MockHiddenParams(dict):
            pass

        class MockLiteLLMResponse:
            model = "gpt-4o-mini"
            _hidden_params = MockHiddenParams(response_cost=0.00000001)

        with caplog.at_level("WARNING"):
            metrics = extract_llm_metrics(
                MockLiteLLMResponse(), model_name="gpt-4o-mini"
            )

        assert metrics.tokens.total_tokens == 0
        assert metrics.cost.total_cost == pytest.approx(0.00000001)
        assert "implausibly below token-derived estimate" not in caplog.text
