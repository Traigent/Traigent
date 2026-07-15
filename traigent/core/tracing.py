"""OpenTelemetry tracing for Traigent optimization.

This module provides backward-compatible tracing functionality. The full
implementation has been moved to the traigent-tracing plugin.

When traigent-tracing is installed, all functionality is imported from the plugin.
When not installed, a fallback embedded implementation is used.

Usage:
    # Traces are automatically created when TRAIGENT_TRACE_ENABLED=true
    # and OTEL_EXPORTER_OTLP_ENDPOINT is configured

    # To view traces, start Jaeger:
    # docker run -d -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/all-in-one

    # Then visit http://localhost:16686

For best tracing support, install the plugin:
    pip install traigent-tracing
"""

from __future__ import annotations

# Try to import from the plugin first
try:
    from traigent_tracing import (
        OTEL_AVAILABLE,
        TRACING_ENABLED,
        SecureIdGenerator,
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

    _PLUGIN_AVAILABLE = True

except ImportError:
    # Plugin not installed, use embedded implementation
    _PLUGIN_AVAILABLE = False

    import json
    import os
    import re
    from collections.abc import Generator
    from contextlib import contextmanager
    from typing import TYPE_CHECKING, Any

    from traigent.core.trace_env import is_trace_enabled

    # Check if tracing is enabled and OpenTelemetry is available
    TRACING_ENABLED = is_trace_enabled()

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

    # Substrings that mark a config/payload key as secret-bearing.
    # Span attributes that ship to OTLP exporters MUST scrub these.
    _SECRET_KEY_SUBSTRINGS = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "passwd",
        "token",
        "authorization",
        "credential",
        "private_key",
        "bearer",
    )

    # Patterns for PII-bearing values that may appear in user prompts,
    # expected outputs, or actual outputs. These get scrubbed BEFORE
    # being attached to a span — secret-bearing dict keys aren't enough
    # because raw prompt text, expected text, and actual text are
    # privacy-sensitive even when no key name says so.
    _PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (
            re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
            "***EMAIL***",
        ),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "***SSN***"),
        (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "***AWS_ACCESS_KEY***"),
        (
            re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
            "***JWT***",
        ),
        # 13-19 digit numbers (credit cards), possibly separated by - or space.
        (re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), "***CC***"),
        # NANP-format phone numbers (NNN-NNN-NNNN, NNN.NNN.NNNN, NNN NNN NNNN).
        # Narrower than a generic "digits-with-separators" pattern so decimal
        # floats and ISO dates aren't false-positives.
        (re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"), "***PHONE***"),
        # International phone numbers MUST start with an explicit +.
        (
            re.compile(r"\+\d{1,3}[\s\-]\d{1,4}[\s\-]\d{2,4}[\s\-]?\d{3,4}\b"),
            "***PHONE***",
        ),
    )

    def _scrub_pii_text(text: Any) -> Any:
        """Scrub PII patterns from a free-form string.

        Non-string inputs are returned unchanged so callers can fan out
        through ``_redact_payload`` without repeated type checks.
        """
        if not isinstance(text, str):
            return text
        result = text
        for pattern, replacement in _PII_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    def _redact_payload(value: Any) -> Any:
        """Recursively scrub secrets *and* PII from trace payloads.

        Two complementary protections run against every span attribute
        before it ships to an OTLP exporter:

        - **Secret-bearing keys** (``api_key``, ``authorization`` …) in
          dict-shaped payloads are replaced with ``"***REDACTED***"``
          regardless of the value's type.
        - **PII-bearing values** (emails, SSNs, credit cards, JWTs …)
          are scrubbed even when the surrounding key name looks
          innocuous. This is what closes the leak in raw prompt /
          message / content text that ends up in span attributes or
          span names.
        """
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, sub in value.items():
                key_str = str(key).lower()
                if any(needle in key_str for needle in _SECRET_KEY_SUBSTRINGS):
                    redacted[key] = "***REDACTED***"
                else:
                    redacted[key] = _redact_payload(sub)
            return redacted
        if isinstance(value, list):
            return [_redact_payload(item) for item in value]
        if isinstance(value, tuple):
            return tuple(_redact_payload(item) for item in value)
        if isinstance(value, str):
            return _scrub_pii_text(value)
        return value

    def _scrub_for_export(value: Any) -> Any:
        """Convenience wrapper that mirrors ``_redact_payload``.

        Provided as an explicit name for call sites that scrub a single
        scalar (e.g. ``expected_output``) so the intent is obvious.
        """
        return _redact_payload(value)

    class SecureIdGenerator(IdGenerator if IdGenerator else object):  # type: ignore[misc,no-redef]
        """ID generator using os.urandom for cryptographically secure random IDs."""

        def generate_span_id(self) -> int:
            """Generate a cryptographically secure 64-bit span ID."""
            span_id = int.from_bytes(os.urandom(8), byteorder="big")
            while span_id == 0:
                span_id = int.from_bytes(os.urandom(8), byteorder="big")
            return span_id

        def generate_trace_id(self) -> int:
            """Generate a cryptographically secure 128-bit trace ID."""
            trace_id = int.from_bytes(os.urandom(16), byteorder="big")
            while trace_id == 0:
                trace_id = int.from_bytes(os.urandom(16), byteorder="big")
            return trace_id

    # Global tracer instance
    _tracer: Tracer | None = None
    _initialized = False
    _test_context: dict[str, Any] = {}

    def set_test_context(
        test_name: str | None = None,
        test_description: str | None = None,
        test_module: str | None = None,
        **extra_attributes: Any,
    ) -> None:
        """Set test context that will be added to all subsequent spans."""
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

        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not otlp_endpoint:
            return None

        try:
            resource = Resource.create(
                {"service.name": "traigent-optimizer", "service.version": "1.0.0"}
            )
            provider = TracerProvider(
                resource=resource, id_generator=SecureIdGenerator()
            )
            otlp_exporter = OTLPHttpSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            trace.set_tracer_provider(provider)
            _tracer = trace.get_tracer("traigent.core", "1.0.0")
            return _tracer
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Failed to initialize tracing: {e}")
            return None

    def get_tracer() -> Tracer | None:
        """Get the Traigent tracer, initializing if needed."""
        return _initialize_tracer()

    def _scrub_error_text(error: object) -> str:
        """Coerce an error of any runtime shape to a scrubbed *string*.

        Callers are typed ``error: str | None``, but Exception instances
        and dict payloads leak through in practice. Deterministic order:
        dict/list/tuple payloads get key-aware secret redaction first
        (``_redact_payload``), then everything is stringified, then the
        text is PII-scrubbed — so the scrubber always sees a string and
        the result is always a scrubbed ``str`` (valid as an OTLP status
        description, never a redaction bypass for non-string shapes).
        """
        if isinstance(error, (dict, list, tuple)):
            error = _redact_payload(error)
        return str(_scrub_pii_text(str(error)))

    def _set_error_status(
        span: Span | None, error: object, attribute_name: str | None = None
    ) -> None:
        """Mark a span as failed without requiring OpenTelemetry imports.

        ``error`` is scrubbed through the same PII/secret scrubber used for
        ``example.input`` / ``example.expected_output`` / ``example.actual_output``
        before it is attached to the span (see ``_scrub_error_text`` for
        the non-string coercion rules). Without this, error strings
        (which routinely echo the input, expected/actual output, or a
        provider's raw response body) ship unredacted to the OTLP status
        description and any ``attribute_name`` this is called with — see
        local validation gap for traigent/core/tracing.py#issues/0.

        If ``attribute_name`` is given, the scrubbed error is also set as
        a span attribute under that name, so callers no longer need a
        separate raw ``set_attribute`` call before this one.
        """
        if span is None:
            return
        scrubbed_error = _scrub_error_text(error)
        if attribute_name:
            span.set_attribute(attribute_name, scrubbed_error)
        if trace is not None:
            span.set_status(trace.Status(trace.StatusCode.ERROR, scrubbed_error))
        else:
            # Best-effort fallback for no-OTEL environments and mock spans.
            try:
                span.set_status(status="ERROR", description=scrubbed_error)
            except TypeError:
                span.set_status("ERROR")

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
                config_str = json.dumps(_redact_payload(config_space))
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
        """Create a root span for an optimization session."""
        tracer = get_tracer()
        if tracer is None:
            yield None
            return

        test_ctx = get_test_context()
        span_name = (
            f"optimization: {test_ctx['test.name']}"
            if test_ctx.get("test.name")
            else f"optimization: {function_name}"
        )
        fresh_context = otel_context.Context() if otel_context else None
        token = (
            otel_context.attach(fresh_context)
            if otel_context and fresh_context
            else None
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

    def _format_config_summary(config: dict[str, Any], max_length: int = 50) -> str:
        """Format config dict into a readable summary for span names.

        Span names ship to OTLP backends in plaintext, so any value that
        leaks into the summary is leaked off-host. Values are scrubbed
        through ``_scrub_pii_text`` and secret-keyed entries are
        replaced wholesale before the summary is assembled.
        """
        if not config:
            return ""
        priority_keys = ["model", "temperature", "temp", "max_tokens", "provider"]
        parts = []
        seen_keys: set[str] = set()

        def _safe_value(key: str, value: Any) -> str:
            if any(needle in key.lower() for needle in _SECRET_KEY_SUBSTRINGS):
                return "***REDACTED***"
            return str(_scrub_pii_text(str(value)))

        for key in priority_keys:
            if key in config:
                short_key = key[:5] if len(key) > 5 else key
                if short_key == "tempe":
                    short_key = "temp"
                parts.append(f"{short_key}={_safe_value(key, config[key])}")
                seen_keys.add(key)
        for key, value in config.items():
            if key in seen_keys or key.startswith("_"):
                continue
            short_key = key[:5] if len(key) > 5 else key
            parts.append(f"{short_key}={_safe_value(key, value)}")
            if len(", ".join(parts)) > max_length:
                break
        result = ", ".join(parts)
        if len(result) > max_length:
            result = result[: max_length - 3] + "..."
        return result

    def _format_input_preview(input_data: Any, max_length: int = 35) -> str:
        """Format input data into a preview for span names.

        Span names ship to OTLP backends, so PII/secrets in raw prompts
        / messages / content fields must be scrubbed BEFORE the preview
        becomes part of the span name.
        """
        if input_data is None:
            return ""
        try:
            if isinstance(input_data, str):
                preview = input_data
            elif isinstance(input_data, dict):
                for key in ["text", "input", "query", "prompt", "message", "content"]:
                    if key in input_data:
                        preview = str(input_data[key])
                        break
                else:
                    preview = (
                        str(next(iter(input_data.values()))) if input_data else "{}"
                    )
            else:
                preview = str(input_data)
            preview = _scrub_pii_text(preview)
            preview = preview.replace("\n", " ").strip()
            if len(preview) > max_length:
                preview = preview[: max_length - 3] + "..."
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
        """Create a span for a trial execution."""
        tracer = get_tracer()
        if tracer is None:
            yield None
            return
        display_number = trial_number + 1
        config_summary = _format_config_summary(config)
        span_name = (
            f"trial {display_number}: {config_summary}"
            if config_summary
            else f"trial {display_number}"
        )
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("trial.id", trial_id)
            span.set_attribute("trial.number", trial_number)
            span.set_attribute("trial.display_number", display_number)
            redacted_config = _redact_payload(config)
            try:
                span.set_attribute("trial.config", json.dumps(redacted_config))
            except (TypeError, ValueError):
                span.set_attribute("trial.config", str(redacted_config))
            yield span

    def record_trial_result(
        span: Span | None,
        status: str,
        metrics: dict[str, float] | None = None,
        error: str | None = None,
    ) -> None:
        """Record trial result on a span."""
        if span is None:
            return
        span.set_attribute("trial.status", status)
        if metrics:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    span.set_attribute(f"trial.metric.{name}", value)
        if error:
            _set_error_status(span, error, attribute_name="trial.error")

    def record_optimization_complete(
        span: Span | None,
        trial_count: int,
        best_score: float | None = None,
        best_config: dict[str, Any] | None = None,
        stop_reason: str | None = None,
    ) -> None:
        """Record optimization completion on the session span."""
        if span is None:
            return
        span.set_attribute("optimization.trial_count", trial_count)
        if best_score is not None:
            span.set_attribute("optimization.best_score", best_score)
        if best_config:
            try:
                span.set_attribute(
                    "optimization.best_config",
                    json.dumps(_redact_payload(best_config)),
                )
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
        """Create a span for a single example evaluation within a trial."""
        tracer = get_tracer()
        if tracer is None:
            yield None
            return
        input_preview = _format_input_preview(input_data)
        span_name = (
            f"example {example_index}: {input_preview}"
            if input_preview
            else f"example {example_index}"
        )
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("example.id", example_id)
            span.set_attribute("example.index", example_index)
            if input_data:
                redacted_input = _redact_payload(input_data)
                try:
                    input_str = json.dumps(redacted_input)
                    if len(input_str) > 500:
                        input_str = input_str[:500] + "..."
                    span.set_attribute("example.input", input_str)
                except (TypeError, ValueError):
                    span.set_attribute("example.input", str(redacted_input)[:500])
            if expected_output is not None:
                # Scrub PII/secrets before serialising. Without this, raw
                # ``example.expected_output`` text ships unredacted to
                # OTLP exporters (see local validation gap for
                # traigent/core/tracing.py#issues/0).
                scrubbed_expected = _scrub_for_export(expected_output)
                try:
                    expected_str = (
                        json.dumps(scrubbed_expected)
                        if not isinstance(scrubbed_expected, str)
                        else scrubbed_expected
                    )
                    if len(expected_str) > 200:
                        expected_str = expected_str[:200] + "..."
                    span.set_attribute("example.expected_output", expected_str)
                except (TypeError, ValueError):
                    span.set_attribute(
                        "example.expected_output",
                        _scrub_pii_text(str(scrubbed_expected))[:200],
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
        """Record example evaluation result on a span."""
        if span is None:
            return
        span.set_attribute("example.success", success)
        if actual_output is not None:
            # Scrub PII/secrets before exporting. Raw model output ships
            # unredacted otherwise (see local validation gap for
            # traigent/core/tracing.py#issues/0).
            scrubbed_actual = _scrub_for_export(actual_output)
            try:
                output_str = (
                    json.dumps(scrubbed_actual)
                    if not isinstance(scrubbed_actual, str)
                    else scrubbed_actual
                )
            except (TypeError, ValueError):
                output_str = _scrub_pii_text(str(scrubbed_actual))
            if len(output_str) > 500:
                output_str = output_str[:500] + "..."
            span.set_attribute("example.actual_output", output_str)
        if metrics:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    span.set_attribute(f"example.metric.{name}", value)
        if execution_time is not None:
            span.set_attribute("example.execution_time_ms", execution_time * 1000)
        if error:
            _set_error_status(span, error, attribute_name="example.error")


__all__ = [
    "TRACING_ENABLED",
    "OTEL_AVAILABLE",
    "SecureIdGenerator",
    "get_tracer",
    "set_test_context",
    "clear_test_context",
    "get_test_context",
    "optimization_session_span",
    "trial_span",
    "record_trial_result",
    "record_optimization_complete",
    "example_evaluation_span",
    "record_example_result",
]
