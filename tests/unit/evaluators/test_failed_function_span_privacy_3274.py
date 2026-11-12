"""#3274: a raising user function never leaks its message into span data."""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry import trace as otel_trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

import traigent.core.tracing as core_tracing  # noqa: E402
from traigent.evaluators.base import Dataset, EvaluationExample  # noqa: E402
from traigent.evaluators.local import LocalEvaluator  # noqa: E402
from traigent.utils.wire_text import span_safe_error  # noqa: E402

MESSAGE = "zq-patient-has-diabetes"


def _raise(row):
    raise ValueError(MESSAGE)


def _span_text(spans) -> str:
    parts = []
    for span in spans:
        parts.append(repr(dict(span.attributes or {})))
        parts.append(repr(span.status.description))
        parts.extend(repr(dict(e.attributes or {})) for e in span.events)
    return "\n".join(parts)


@pytest.mark.asyncio
@pytest.mark.parametrize("detailed", [True, False])
async def test_example_span_carries_class_and_code_not_message(monkeypatch, detailed):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(core_tracing, "get_tracer", lambda: provider.get_tracer("t"))
    monkeypatch.setattr(core_tracing, "TRACING_ENABLED", True, raising=False)

    dataset = Dataset(
        examples=[EvaluationExample(input_data={"row": 1}, expected_output=1)],
        name="d",
    )
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=detailed)
    await evaluator.evaluate(_raise, {"m": "a"}, dataset)

    spans = exporter.get_finished_spans()
    example_spans = [s for s in spans if "example.error" in (s.attributes or {})]
    assert example_spans, "tracing produced no failing example span"
    assert "ValueError" in example_spans[0].attributes["example.error"]
    assert MESSAGE not in _span_text(spans)


def test_span_safe_error_drops_message_for_exceptions_and_text():
    for value in (
        ValueError(MESSAGE),
        f"Function call failed: ValueError: {MESSAGE}",
    ):
        text = span_safe_error(value)
        assert MESSAGE not in text
        assert text.startswith("ValueError [code=")
    assert span_safe_error(None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("span_kind", ["session", "trial", "example"])
async def test_escaping_configuration_error_message_absent_from_every_span(
    monkeypatch, span_kind
):
    """A ConfigurationError raised by user code escapes the evaluator (#3274)."""
    from traigent.utils.exceptions import ConfigurationError

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(core_tracing, "get_tracer", lambda: provider.get_tracer("t"))
    monkeypatch.setattr(core_tracing, "TRACING_ENABLED", True, raising=False)
    monkeypatch.setattr(core_tracing, "trace", otel_trace)

    def func(row):
        raise ConfigurationError("zq-secret-text")

    spans_cm = {
        "session": lambda: core_tracing.optimization_session_span("fn", max_trials=1),
        "trial": lambda: core_tracing.trial_span("t1", 0, {"m": "a"}),
        "example": lambda: core_tracing.example_evaluation_span("e1", 0, {"q": "x"}),
    }[span_kind]
    dataset = Dataset(
        examples=[EvaluationExample(input_data={"row": 1}, expected_output=1)],
        name="d",
    )
    evaluator = LocalEvaluator(metrics=["accuracy"], detailed=False)
    with pytest.raises(ConfigurationError):
        with spans_cm():
            await evaluator.evaluate(func, {"m": "a"}, dataset)

    spans = exporter.get_finished_spans()
    assert spans
    assert "zq-secret-text" not in _span_text(spans)
    escaping = [s for s in spans if any(e.name == "exception" for e in s.events)]
    assert escaping
    assert escaping[0].status.description.startswith("ConfigurationError [code=")
