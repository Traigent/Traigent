"""OpenTelemetry tracing for TraiGent optimization.

Provides automatic span creation for optimization sessions and trials,
making it easy to trace and analyze optimization runs in Jaeger or other
OTLP-compatible backends.

Usage:
    # Traces are automatically created when TRAIGENT_TRACE_ENABLED=true
    # and OTEL_EXPORTER_OTLP_ENDPOINT is configured

    # To view traces, start Jaeger:
    # docker run -d -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/all-in-one

    # Then visit http://localhost:16686
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

# Check if tracing is enabled and OpenTelemetry is available
TRACING_ENABLED = os.environ.get("TRAIGENT_TRACE_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as OTLPHttpSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    TracerProvider = None  # type: ignore
    OTLPHttpSpanExporter = None  # type: ignore

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

# Global tracer instance
_tracer: Tracer | None = None
_initialized = False

# Test context for adding metadata to spans
_test_context: dict[str, Any] = {}


def set_test_context(
    test_name: str | None = None,
    test_description: str | None = None,
    test_module: str | None = None,
    **extra_attributes: Any,
) -> None:
    """Set test context that will be added to all subsequent spans.

    This allows test frameworks to add informative metadata to traces.

    Args:
        test_name: Name of the test (e.g., "test_injection_mode_basic[context]")
        test_description: Human-readable description of what the test validates
        test_module: Module path (e.g., "test_injection_modes")
        **extra_attributes: Additional attributes to include
    """
    global _test_context
    _test_context = {}
    if test_name:
        _test_context["test.name"] = test_name
    if test_description:
        _test_context["test.description"] = test_description
    if test_module:
        _test_context["test.module"] = test_module
    _test_context.update({f"test.{k}": v for k, v in extra_attributes.items()})


def clear_test_context() -> None:
    """Clear the test context."""
    global _test_context
    _test_context = {}


def get_test_context() -> dict[str, Any]:
    """Get the current test context."""
    return _test_context.copy()


def _initialize_tracer() -> Tracer | None:
    """Initialize the OpenTelemetry tracer if not already done."""
    global _tracer, _initialized

    if _initialized:
        return _tracer

    _initialized = True

    if not OTEL_AVAILABLE or not TRACING_ENABLED:
        return None

    # Get OTLP endpoint from environment
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otlp_endpoint:
        return None

    try:
        # Create resource with service name
        resource = Resource.create(
            {
                "service.name": "traigent-optimizer",
                "service.version": "1.0.0",
            }
        )

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add OTLP exporter
        otlp_exporter = OTLPHttpSpanExporter(
            endpoint=f"{otlp_endpoint}/v1/traces",
        )
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        # Set as global provider
        trace.set_tracer_provider(provider)

        # Get tracer
        _tracer = trace.get_tracer("traigent.core", "1.0.0")
        return _tracer

    except Exception as e:
        # Log but don't fail if tracing setup fails
        import logging

        logging.getLogger(__name__).warning(f"Failed to initialize tracing: {e}")
        return None


def get_tracer() -> Tracer | None:
    """Get the TraiGent tracer, initializing if needed."""
    return _initialize_tracer()


@contextmanager
def optimization_session_span(
    function_name: str,
    max_trials: int | None = None,
    timeout: float | None = None,
    algorithm: str | None = None,
    objectives: list[str] | None = None,
    config_space: dict[str, Any] | None = None,
) -> Generator[Span | None, None, None]:
    """Create a root span for an optimization session.

    Args:
        function_name: Name of the function being optimized
        max_trials: Maximum number of trials
        timeout: Optimization timeout
        algorithm: Optimization algorithm
        objectives: List of objective names
        config_space: Configuration space

    Yields:
        The span object, or None if tracing is disabled
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    # Build a descriptive span name
    test_ctx = get_test_context()
    if test_ctx.get("test.name"):
        span_name = f"optimization: {test_ctx['test.name']}"
    else:
        span_name = f"optimization: {function_name}"

    with tracer.start_as_current_span(span_name) as span:
        # Add test context attributes first (so they appear at top in Jaeger)
        for key, value in test_ctx.items():
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(key, value)

        span.set_attribute("traigent.function_name", function_name)
        if max_trials is not None:
            span.set_attribute("traigent.max_trials", max_trials)
        if timeout is not None:
            span.set_attribute("traigent.timeout", timeout)
        if algorithm:
            span.set_attribute("traigent.algorithm", algorithm)
        if objectives:
            span.set_attribute("traigent.objectives", ",".join(objectives))
        if config_space:
            try:
                # Serialize config space (truncate if too large)
                config_str = json.dumps(config_space)
                if len(config_str) > 1000:
                    config_str = config_str[:1000] + "..."
                span.set_attribute("traigent.config_space", config_str)
            except (TypeError, ValueError):
                pass
        yield span


def _format_config_summary(config: dict[str, Any], max_length: int = 50) -> str:
    """Format config dict into a readable summary for span names.

    Args:
        config: Trial configuration dictionary
        max_length: Maximum length of the summary

    Returns:
        Formatted string like "model=gpt-4, temp=0.3"
    """
    if not config:
        return ""

    # Priority keys to show first (common LLM config params)
    priority_keys = ["model", "temperature", "temp", "max_tokens", "provider"]

    parts = []
    seen_keys = set()

    # Add priority keys first
    for key in priority_keys:
        if key in config:
            value = config[key]
            # Shorten key names for readability
            short_key = key[:5] if len(key) > 5 else key
            if short_key == "tempe":
                short_key = "temp"
            parts.append(f"{short_key}={value}")
            seen_keys.add(key)

    # Add remaining keys if there's room
    for key, value in config.items():
        if key in seen_keys or key.startswith("_"):
            continue
        short_key = key[:5] if len(key) > 5 else key
        parts.append(f"{short_key}={value}")
        if len(", ".join(parts)) > max_length:
            break

    result = ", ".join(parts)
    if len(result) > max_length:
        result = result[: max_length - 3] + "..."
    return result


def _format_input_preview(input_data: Any, max_length: int = 35) -> str:
    """Format input data into a preview for span names.

    Args:
        input_data: Input data (dict, string, or other)
        max_length: Maximum length of the preview

    Returns:
        Formatted string preview of the input
    """
    if input_data is None:
        return ""

    try:
        if isinstance(input_data, str):
            preview = input_data
        elif isinstance(input_data, dict):
            # Try to get a meaningful value from common keys
            for key in ["text", "input", "query", "prompt", "message", "content"]:
                if key in input_data:
                    preview = str(input_data[key])
                    break
            else:
                # Just show first value or the dict itself
                if input_data:
                    first_value = next(iter(input_data.values()))
                    preview = str(first_value)
                else:
                    preview = "{}"
        else:
            preview = str(input_data)

        # Clean up and truncate
        preview = preview.replace("\n", " ").strip()
        if len(preview) > max_length:
            preview = preview[: max_length - 3] + "..."

        # Add quotes if it looks like text
        if preview and not preview.startswith("{") and not preview.startswith("["):
            preview = f'"{preview}"'

        return preview
    except Exception:
        return ""


@contextmanager
def trial_span(
    trial_id: str,
    trial_number: int,
    config: dict[str, Any],
) -> Generator[Span | None, None, None]:
    """Create a span for a trial execution.

    Args:
        trial_id: Unique trial identifier
        trial_number: Trial number (1-based)
        config: Trial configuration

    Yields:
        The span object, or None if tracing is disabled
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    # Build descriptive span name
    config_summary = _format_config_summary(config)
    if config_summary:
        span_name = f"trial {trial_number}: {config_summary}"
    else:
        span_name = f"trial {trial_number}"

    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("trial.id", trial_id)
        span.set_attribute("trial.number", trial_number)
        try:
            span.set_attribute("trial.config", json.dumps(config))
        except (TypeError, ValueError):
            span.set_attribute("trial.config", str(config))
        yield span


def record_trial_result(
    span: Span | None,
    status: str,
    metrics: dict[str, float] | None = None,
    error: str | None = None,
) -> None:
    """Record trial result on a span.

    Args:
        span: The trial span (or None if tracing disabled)
        status: Trial status ("completed", "failed", etc.)
        metrics: Trial metrics
        error: Error message if failed
    """
    if span is None:
        return

    span.set_attribute("trial.status", status)

    if metrics:
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                span.set_attribute(f"trial.metric.{name}", value)

    if error:
        span.set_attribute("trial.error", error)
        span.set_status(trace.Status(trace.StatusCode.ERROR, error))


def record_optimization_complete(
    span: Span | None,
    trial_count: int,
    best_score: float | None = None,
    best_config: dict[str, Any] | None = None,
    stop_reason: str | None = None,
) -> None:
    """Record optimization completion on the session span.

    Args:
        span: The optimization session span
        trial_count: Number of trials completed
        best_score: Best score achieved
        best_config: Best configuration found
        stop_reason: Reason for stopping
    """
    if span is None:
        return

    span.set_attribute("optimization.trial_count", trial_count)
    if best_score is not None:
        span.set_attribute("optimization.best_score", best_score)
    if best_config:
        try:
            span.set_attribute("optimization.best_config", json.dumps(best_config))
        except (TypeError, ValueError):
            pass
    if stop_reason:
        span.set_attribute("optimization.stop_reason", stop_reason)


@contextmanager
def example_evaluation_span(
    example_id: str,
    example_index: int,
    input_data: dict[str, Any] | None = None,
    expected_output: Any | None = None,
) -> Generator[Span | None, None, None]:
    """Create a span for a single example evaluation within a trial.

    Args:
        example_id: Unique example identifier
        example_index: Index of example in dataset (0-based)
        input_data: Input data for the example
        expected_output: Expected output for the example

    Yields:
        The span object, or None if tracing is disabled
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    # Build descriptive span name with input preview
    input_preview = _format_input_preview(input_data)
    if input_preview:
        span_name = f"example {example_index}: {input_preview}"
    else:
        span_name = f"example {example_index}"

    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("example.id", example_id)
        span.set_attribute("example.index", example_index)
        if input_data:
            try:
                input_str = json.dumps(input_data)
                # Truncate if too large
                if len(input_str) > 500:
                    input_str = input_str[:500] + "..."
                span.set_attribute("example.input", input_str)
            except (TypeError, ValueError):
                span.set_attribute("example.input", str(input_data)[:500])
        if expected_output is not None:
            try:
                expected_str = (
                    json.dumps(expected_output)
                    if not isinstance(expected_output, str)
                    else expected_output
                )
                if len(expected_str) > 200:
                    expected_str = expected_str[:200] + "..."
                span.set_attribute("example.expected_output", expected_str)
            except (TypeError, ValueError):
                span.set_attribute(
                    "example.expected_output", str(expected_output)[:200]
                )
        yield span


def record_example_result(
    span: Span | None,
    success: bool,
    actual_output: Any | None = None,
    metrics: dict[str, float] | None = None,
    error: str | None = None,
    execution_time: float | None = None,
) -> None:
    """Record example evaluation result on a span.

    Args:
        span: The example span (or None if tracing disabled)
        success: Whether evaluation succeeded
        actual_output: The actual output from the function
        metrics: Example-level metrics (accuracy, cost, etc.)
        error: Error message if failed
        execution_time: Execution time in seconds
    """
    if span is None:
        return

    span.set_attribute("example.success", success)

    if actual_output is not None:
        try:
            output_str = (
                json.dumps(actual_output)
                if not isinstance(actual_output, str)
                else actual_output
            )
            # Truncate if too large
            if len(output_str) > 500:
                output_str = output_str[:500] + "..."
            span.set_attribute("example.actual_output", output_str)
        except (TypeError, ValueError):
            span.set_attribute("example.actual_output", str(actual_output)[:500])

    if metrics:
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                span.set_attribute(f"example.metric.{name}", value)

    if execution_time is not None:
        span.set_attribute("example.execution_time_ms", execution_time * 1000)

    if error:
        span.set_attribute("example.error", error)
        span.set_status(trace.Status(trace.StatusCode.ERROR, error))
