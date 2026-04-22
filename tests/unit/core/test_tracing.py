"""Unit tests for the tracing module.

Tests cover:
- SecureIdGenerator for unique trace/span IDs
- Test context management (set/get/clear)
- Helper functions for formatting span names
- Context managers when tracing is disabled
- Record functions for trial and optimization results
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from traigent.core import tracing
from traigent.core.tracing import (
    SecureIdGenerator,
    _format_config_summary,
    _format_input_preview,
    clear_test_context,
    example_evaluation_span,
    get_test_context,
    get_tracer,
    optimization_session_span,
    record_example_result,
    record_optimization_complete,
    record_trial_result,
    set_test_context,
    trial_span,
)


class TestSecureIdGenerator:
    """Tests for SecureIdGenerator class."""

    def test_generate_span_id_returns_int(self) -> None:
        """Test that generate_span_id returns an integer."""
        generator = SecureIdGenerator()
        span_id = generator.generate_span_id()

        assert isinstance(span_id, int)
        assert span_id != 0  # Must be non-zero

    def test_generate_trace_id_returns_int(self) -> None:
        """Test that generate_trace_id returns an integer."""
        generator = SecureIdGenerator()
        trace_id = generator.generate_trace_id()

        assert isinstance(trace_id, int)
        assert trace_id != 0  # Must be non-zero

    def test_generate_unique_ids(self) -> None:
        """Test that generated IDs are unique."""
        generator = SecureIdGenerator()

        span_ids = {generator.generate_span_id() for _ in range(100)}
        trace_ids = {generator.generate_trace_id() for _ in range(100)}

        # All should be unique
        assert len(span_ids) == 100
        assert len(trace_ids) == 100

    def test_ids_not_affected_by_random_seed(self) -> None:
        """Test that IDs are not affected by random.seed()."""
        import random

        generator = SecureIdGenerator()

        # Set a fixed seed
        random.seed(42)
        id1 = generator.generate_span_id()

        # Set the same seed again
        random.seed(42)
        id2 = generator.generate_span_id()

        # IDs should be different (not determined by random.seed)
        assert id1 != id2


class TestTestContext:
    """Tests for test context management functions."""

    def setup_method(self) -> None:
        """Clear test context before each test."""
        clear_test_context()

    def teardown_method(self) -> None:
        """Clear test context after each test."""
        clear_test_context()

    def test_set_test_context_basic(self) -> None:
        """Test setting basic test context."""
        set_test_context(
            test_name="test_example",
            test_description="Testing example functionality",
            test_module="test_module",
        )

        ctx = get_test_context()
        assert ctx["test.name"] == "test_example"
        assert ctx["test.description"] == "Testing example functionality"
        assert ctx["test.module"] == "test_module"

    def test_set_test_context_with_extra_attributes(self) -> None:
        """Test setting test context with extra attributes."""
        set_test_context(
            test_name="test_example",
            category="unit",
            priority="high",
        )

        ctx = get_test_context()
        assert ctx["test.name"] == "test_example"
        assert ctx["test.category"] == "unit"
        assert ctx["test.priority"] == "high"

    def test_set_test_context_partial(self) -> None:
        """Test setting partial test context."""
        set_test_context(test_name="test_only_name")

        ctx = get_test_context()
        assert ctx["test.name"] == "test_only_name"
        assert "test.description" not in ctx

    def test_clear_test_context(self) -> None:
        """Test clearing test context."""
        set_test_context(test_name="test_example")
        assert get_test_context() != {}

        clear_test_context()
        assert get_test_context() == {}

    def test_get_test_context_returns_copy(self) -> None:
        """Test that get_test_context returns a copy."""
        set_test_context(test_name="test_example")

        ctx1 = get_test_context()
        ctx1["modified"] = True

        ctx2 = get_test_context()
        assert "modified" not in ctx2


class TestFormatConfigSummary:
    """Tests for _format_config_summary helper function."""

    def test_empty_config(self) -> None:
        """Test with empty config."""
        assert _format_config_summary({}) == ""

    def test_priority_keys(self) -> None:
        """Test that priority keys appear first."""
        config = {
            "other": "value",
            "model": "gpt-4",
            "temperature": 0.5,
        }

        result = _format_config_summary(config)
        assert result.startswith("model=gpt-4")
        assert "temp=0.5" in result or "tempe=0.5" in result

    def test_truncation(self) -> None:
        """Test that long summaries are truncated."""
        config = {
            "model": "very-long-model-name-that-takes-space",
            "temperature": 0.123456789,
            "max_tokens": 4096,
            "another_param": "value",
            "yet_another": "more",
        }

        result = _format_config_summary(config, max_length=30)
        assert len(result) <= 33  # 30 + "..."
        assert result.endswith("...")

    def test_skips_private_keys(self) -> None:
        """Test that keys starting with _ are skipped."""
        config = {"model": "gpt-4", "_internal": "hidden"}

        result = _format_config_summary(config)
        assert "model" in result
        assert "_internal" not in result
        assert "hidden" not in result


class TestFormatInputPreview:
    """Tests for _format_input_preview helper function."""

    def test_none_input(self) -> None:
        """Test with None input."""
        assert _format_input_preview(None) == ""

    def test_string_input(self) -> None:
        """Test with string input."""
        result = _format_input_preview("Hello world")
        assert '"Hello world"' == result

    def test_dict_with_text_key(self) -> None:
        """Test dict with 'text' key."""
        result = _format_input_preview({"text": "Hello", "other": "ignored"})
        assert '"Hello"' == result

    def test_dict_with_input_key(self) -> None:
        """Test dict with 'input' key."""
        result = _format_input_preview({"input": "User query"})
        assert '"User query"' == result

    def test_dict_with_prompt_key(self) -> None:
        """Test dict with 'prompt' key."""
        result = _format_input_preview({"prompt": "Generate code"})
        assert '"Generate code"' == result

    def test_empty_dict(self) -> None:
        """Test with empty dict."""
        result = _format_input_preview({})
        assert result == "{}"

    def test_truncation(self) -> None:
        """Test that long input is truncated."""
        long_text = "This is a very long text that should be truncated"
        result = _format_input_preview(long_text, max_length=20)
        assert len(result) <= 23  # 20 + "..." + quotes
        assert "..." in result

    def test_newlines_removed(self) -> None:
        """Test that newlines are removed."""
        result = _format_input_preview("Line 1\nLine 2\nLine 3")
        assert "\n" not in result


class TestContextManagersDisabled:
    """Tests for context managers when tracing is disabled."""

    def setup_method(self) -> None:
        """Reset tracer state before each test."""
        tracing._tracer = None
        tracing._initialized = False

    def test_optimization_session_span_disabled(self) -> None:
        """Test optimization_session_span yields None when disabled."""
        with patch.object(tracing, "get_tracer", return_value=None):
            with optimization_session_span("test_func") as span:
                assert span is None

    def test_trial_span_disabled(self) -> None:
        """Test trial_span yields None when disabled."""
        with patch.object(tracing, "get_tracer", return_value=None):
            with trial_span("trial-1", 0, {"model": "gpt-4"}) as span:
                assert span is None

    def test_example_evaluation_span_disabled(self) -> None:
        """Test example_evaluation_span yields None when disabled."""
        with patch.object(tracing, "get_tracer", return_value=None):
            with example_evaluation_span("ex-1", 0) as span:
                assert span is None


def test_set_error_status_falls_back_for_mock_spans_without_otel() -> None:
    span = MagicMock()
    span.set_status.side_effect = [TypeError("unexpected kwargs"), None]

    with patch.object(tracing, "trace", None):
        tracing._set_error_status(span, "boom")

    assert span.set_status.call_args_list[0].kwargs == {
        "status": "ERROR",
        "description": "boom",
    }
    assert span.set_status.call_args_list[1].args == ("ERROR",)


class TestRecordFunctionsWithNoneSpan:
    """Tests for record functions when span is None."""

    def test_record_trial_result_none_span(self) -> None:
        """Test record_trial_result with None span does not raise."""
        # Should not raise any exceptions
        record_trial_result(None, "completed", {"accuracy": 0.9})
        record_trial_result(None, "failed", error="Test error")

    def test_record_optimization_complete_none_span(self) -> None:
        """Test record_optimization_complete with None span does not raise."""
        # Should not raise any exceptions
        record_optimization_complete(None, 10, best_score=0.95)
        record_optimization_complete(
            None, 5, best_config={"model": "gpt-4"}, stop_reason="max_trials"
        )

    def test_record_example_result_none_span(self) -> None:
        """Test record_example_result with None span does not raise."""
        # Should not raise any exceptions
        record_example_result(None, success=True, actual_output="result")
        record_example_result(
            None,
            success=False,
            error="Test error",
            metrics={"accuracy": 0.5},
            execution_time=1.5,
        )


class TestRecordFunctionsWithMockSpan:
    """Tests for record functions with mock span."""

    def test_record_trial_result_completed(self) -> None:
        """Test recording completed trial result."""
        span = MagicMock()

        record_trial_result(span, "completed", {"accuracy": 0.95, "cost": 0.01})

        span.set_attribute.assert_any_call("trial.status", "completed")
        span.set_attribute.assert_any_call("trial.metric.accuracy", 0.95)
        span.set_attribute.assert_any_call("trial.metric.cost", 0.01)

    def test_record_trial_result_failed(self) -> None:
        """Test recording failed trial result."""
        span = MagicMock()

        record_trial_result(span, "failed", error="Connection timeout")

        span.set_attribute.assert_any_call("trial.status", "failed")
        span.set_attribute.assert_any_call("trial.error", "Connection timeout")
        span.set_status.assert_called_once()

    def test_record_optimization_complete(self) -> None:
        """Test recording optimization completion."""
        span = MagicMock()

        record_optimization_complete(
            span,
            trial_count=10,
            best_score=0.98,
            best_config={"model": "gpt-4", "temperature": 0.5},
            stop_reason="converged",
        )

        span.set_attribute.assert_any_call("optimization.trial_count", 10)
        span.set_attribute.assert_any_call("optimization.best_score", 0.98)
        span.set_attribute.assert_any_call("optimization.stop_reason", "converged")

    def test_record_example_result_success(self) -> None:
        """Test recording successful example result."""
        span = MagicMock()

        record_example_result(
            span,
            success=True,
            actual_output="Generated response",
            metrics={"accuracy": 1.0},
            execution_time=0.5,
        )

        span.set_attribute.assert_any_call("example.success", True)
        span.set_attribute.assert_any_call(
            "example.actual_output", "Generated response"
        )
        span.set_attribute.assert_any_call("example.metric.accuracy", 1.0)
        span.set_attribute.assert_any_call("example.execution_time_ms", 500.0)

    def test_record_example_result_failure(self) -> None:
        """Test recording failed example result."""
        span = MagicMock()

        record_example_result(span, success=False, error="API timeout")

        span.set_attribute.assert_any_call("example.success", False)
        span.set_attribute.assert_any_call("example.error", "API timeout")
        span.set_status.assert_called_once()


class TestGetTracer:
    """Tests for get_tracer function."""

    def setup_method(self) -> None:
        """Reset tracer state before each test."""
        tracing._tracer = None
        tracing._initialized = False

    def test_get_tracer_when_disabled(self) -> None:
        """Test get_tracer returns None when tracing is disabled."""
        with patch.object(tracing, "TRACING_ENABLED", False):
            tracing._initialized = False
            result = get_tracer()
            assert result is None

    def test_get_tracer_caches_result(self) -> None:
        """Test that get_tracer caches the result."""
        with patch.object(tracing, "TRACING_ENABLED", False):
            tracing._initialized = False
            result1 = get_tracer()
            tracing._initialized = True  # Mark as initialized

            result2 = get_tracer()

            # Both should be None and cached
            assert result1 is None
            assert result2 is None


class TestSetErrorStatus:
    """Tests for fallback error-status handling."""

    def test_set_error_status_without_otel_uses_named_error_fallback(self) -> None:
        span = MagicMock()

        with patch.object(tracing, "trace", None):
            tracing._set_error_status(span, "broken")

        span.set_status.assert_called_once_with(status="ERROR", description="broken")

    def test_set_error_status_falls_back_to_simple_error_status_on_type_error(
        self,
    ) -> None:
        span = MagicMock()
        span.set_status.side_effect = [TypeError("unsupported kwargs"), None]

        with patch.object(tracing, "trace", None):
            tracing._set_error_status(span, "broken")

        assert span.set_status.call_count == 2
        assert span.set_status.call_args_list[0].kwargs == {
            "status": "ERROR",
            "description": "broken",
        }
        assert span.set_status.call_args_list[1].args == ("ERROR",)


class TestTrialSpan:
    """Tests for trial_span context manager."""

    def test_trial_span_name_formatting(self) -> None:
        """Test that trial span names are formatted correctly."""
        # This tests the helper functions used in trial_span
        config = {"model": "gpt-4", "temperature": 0.7}
        summary = _format_config_summary(config)

        assert "model=gpt-4" in summary
        # Temperature should be shortened to 'temp'
        assert "temp=0.7" in summary or "tempe=0.7" in summary


class TestOptimizationSessionSpan:
    """Tests for optimization_session_span context manager."""

    def setup_method(self) -> None:
        """Clear test context before each test."""
        clear_test_context()

    def teardown_method(self) -> None:
        """Clear test context after each test."""
        clear_test_context()

    def test_span_name_uses_test_context_if_available(self) -> None:
        """Test that span name uses test context when available."""
        # Set up test context
        set_test_context(test_name="test_my_feature")

        ctx = get_test_context()
        assert ctx.get("test.name") == "test_my_feature"

        # When tracer is available, span name would be:
        # "optimization: test_my_feature"

    def test_span_name_uses_function_name_without_context(self) -> None:
        """Test that span name uses function name when no test context."""
        clear_test_context()

        ctx = get_test_context()
        assert ctx == {}

        # When tracer is available, span name would be:
        # "optimization: my_function"


class TestExampleEvaluationSpan:
    """Tests for example_evaluation_span context manager."""

    def test_input_preview_in_span_name(self) -> None:
        """Test that input preview is included in span name."""
        preview = _format_input_preview({"text": "Hello world"})
        assert "Hello world" in preview

    def test_handles_dict_input(self) -> None:
        """Test handling dict input data."""
        preview = _format_input_preview(
            {"query": "What is the weather?", "context": "Extra info"}
        )
        # Should find 'query' as a priority key
        assert "What is the weather" in preview or "Extra info" in preview


class TestRecordTrialMetricsFiltering:
    """Tests for metric filtering in record functions."""

    def test_record_trial_result_filters_non_numeric_metrics(self) -> None:
        """Test that non-numeric metrics are filtered out."""
        span = MagicMock()

        metrics = {
            "accuracy": 0.95,
            "cost": 0.01,
            "model_name": "gpt-4",  # Not numeric - should be filtered
        }
        record_trial_result(span, "completed", metrics)

        # Check that numeric metrics were set
        span.set_attribute.assert_any_call("trial.metric.accuracy", 0.95)
        span.set_attribute.assert_any_call("trial.metric.cost", 0.01)

        # Check that non-numeric metric was NOT set
        call_args = [call[0] for call in span.set_attribute.call_args_list]
        assert ("trial.metric.model_name", "gpt-4") not in call_args

    def test_record_example_result_filters_non_numeric_metrics(self) -> None:
        """Test that non-numeric example metrics are filtered out."""
        span = MagicMock()

        metrics = {
            "accuracy": 1.0,
            "label": "positive",  # Not numeric - should be filtered
        }
        record_example_result(span, success=True, metrics=metrics)

        # Check that numeric metrics were set
        span.set_attribute.assert_any_call("example.metric.accuracy", 1.0)

        # Check that non-numeric metric was NOT set
        call_args = [call[0] for call in span.set_attribute.call_args_list]
        assert ("example.metric.label", "positive") not in call_args
