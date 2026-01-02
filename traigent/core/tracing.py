"""OpenTelemetry tracing for Traigent optimization.

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
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

# Check if tracing is enabled and OpenTelemetry is available
TRACING_ENABLED = os.environ.get("TRAIGENT_TRACE_ENABLED", "false").lower() in (
    "true",
    "1",
    "yes",
)

try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as OTLPHttpSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.id_generator import IdGenerator

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore
    otel_context = None  # type: ignore
    TracerProvider = None  # type: ignore
    OTLPHttpSpanExporter = None  # type: ignore
    IdGenerator = None  # type: ignore

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


class SecureIdGenerator(IdGenerator if IdGenerator else object):  # type: ignore[misc]
    """ID generator using os.urandom for cryptographically secure random IDs.

    This generator is immune to random.seed() calls that could cause
    deterministic trace IDs, ensuring each trace gets a unique ID.
    """

    def generate_span_id(self) -> int:
        """Generate a cryptographically secure 64-bit span ID."""
        import os

        span_id = int.from_bytes(os.urandom(8), byteorder="big")
        # Ensure non-zero (INVALID_SPAN_ID is 0)
        while span_id == 0:
            span_id = int.from_bytes(os.urandom(8), byteorder="big")
        return span_id

    def generate_trace_id(self) -> int:
        """Generate a cryptographically secure 128-bit trace ID."""
        import os

        trace_id = int.from_bytes(os.urandom(16), byteorder="big")
        # Ensure non-zero (INVALID_TRACE_ID is 0)
        while trace_id == 0:
            trace_id = int.from_bytes(os.urandom(16), byteorder="big")
        return trace_id


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

        # Create tracer provider with secure ID generator
        # This ensures trace IDs are not affected by random.seed() calls
        provider = TracerProvider(
            resource=resource,
            id_generator=SecureIdGenerator(),
        )

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
    """Get the Traigent tracer, initializing if needed."""
    return _initialize_tracer()


def _set_session_span_attributes(
    span: Span,
    function_name: str,
    max_trials: int | None,
    timeout: float | None,
    algorithm: str | None,
    objectives: list[str] | None,
    config_space: dict[str, Any] | None,
) -> None:
    """Set attributes on an optimization session span."""
    # Add test context attributes first (so they appear at top in Jaeger)
    for key, value in get_test_context().items():
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
            config_str = json.dumps(config_space)
            if len(config_str) > 1000:
                config_str = config_str[:1000] + "..."
            span.set_attribute("traigent.config_space", config_str)
        except (TypeError, ValueError):
            pass


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

    This creates a NEW trace for each optimization session, ensuring
    that each test/optimization run appears as a separate trace in Jaeger.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    # Build span name from test context or function name
    test_ctx = get_test_context()
    span_name = (
        f"optimization: {test_ctx['test.name']}"
        if test_ctx.get("test.name")
        else f"optimization: {function_name}"
    )

    # Create fresh context to start a NEW trace (not nested under parent)
    fresh_context = otel_context.Context() if otel_context else None
    token = (
        otel_context.attach(fresh_context) if otel_context and fresh_context else None
    )

    try:
        with tracer.start_as_current_span(span_name) as span:
            _set_session_span_attributes(
                span,
                function_name,
                max_trials,
                timeout,
                algorithm,
                objectives,
                config_space,
            )
            yield span
    finally:
        if token is not None and otel_context:
            otel_context.detach(token)


def _shorten_key(key: str) -> str:
    """Shorten a key name for readability."""
    short_key = key[:5] if len(key) > 5 else key
    return "temp" if short_key == "tempe" else short_key


def _add_priority_keys(
    config: dict[str, Any], priority_keys: list[str]
) -> tuple[list[str], set[str]]:
    """Add priority keys to parts list."""
    parts = []
    seen_keys: set[str] = set()
    for key in priority_keys:
        if key in config:
            short_key = _shorten_key(key)
            parts.append(f"{short_key}={config[key]}")
            seen_keys.add(key)
    return parts, seen_keys


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
    parts, seen_keys = _add_priority_keys(config, priority_keys)

    # Add remaining keys if there's room
    for key, value in config.items():
        if key in seen_keys or key.startswith("_"):
            continue
        parts.append(f"{_shorten_key(key)}={value}")
        if len(", ".join(parts)) > max_length:
            break

    result = ", ".join(parts)
    if len(result) > max_length:
        result = result[: max_length - 3] + "..."
    return result


def _extract_dict_preview(input_data: dict[str, Any]) -> str:
    """Extract a preview string from a dictionary."""
    common_keys = ["text", "input", "query", "prompt", "message", "content"]
    for key in common_keys:
        if key in input_data:
            return str(input_data[key])
    # Fallback to first value or empty dict representation
    if input_data:
        return str(next(iter(input_data.values())))
    return "{}"


def _truncate_and_quote(preview: str, max_length: int) -> str:
    """Clean up, truncate, and optionally quote a preview string."""
    preview = preview.replace("\n", " ").strip()
    if len(preview) > max_length:
        preview = preview[: max_length - 3] + "..."
    # Add quotes if it looks like text (not JSON-like)
    if preview and not preview.startswith("{") and not preview.startswith("["):
        preview = f'"{preview}"'
    return preview


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
            preview = _extract_dict_preview(input_data)
        else:
            preview = str(input_data)

        return _truncate_and_quote(preview, max_length)
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
        trial_number: Trial number (0-based internally, displayed as 1-based)
        config: Trial configuration

    Yields:
        The span object, or None if tracing is disabled
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    # Build descriptive span name (display as 1-based for readability)
    display_number = trial_number + 1
    config_summary = _format_config_summary(config)
    if config_summary:
        span_name = f"trial {display_number}: {config_summary}"
    else:
        span_name = f"trial {display_number}"

    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("trial.id", trial_id)
        span.set_attribute(
            "trial.number", trial_number
        )  # Keep 0-based for programmatic access
        span.set_attribute(
            "trial.display_number", display_number
        )  # 1-based for display
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


def _serialize_output(output: Any, max_length: int) -> str:
    """Serialize output to string with truncation."""
    try:
        output_str = json.dumps(output) if not isinstance(output, str) else output
    except (TypeError, ValueError):
        output_str = str(output)
    if len(output_str) > max_length:
        output_str = output_str[:max_length] + "..."
    return output_str


def _set_span_metrics(span: Span, metrics: dict[str, float]) -> None:
    """Set numeric metrics on a span."""
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            span.set_attribute(f"example.metric.{name}", value)


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
        span.set_attribute(
            "example.actual_output", _serialize_output(actual_output, 500)
        )

    if metrics:
        _set_span_metrics(span, metrics)

    if execution_time is not None:
        span.set_attribute("example.execution_time_ms", execution_time * 1000)

    if error:
        span.set_attribute("example.error", error)
        span.set_status(trace.Status(trace.StatusCode.ERROR, error))
