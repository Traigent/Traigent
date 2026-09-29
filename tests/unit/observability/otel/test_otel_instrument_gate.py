"""Instrumentor exposure gate and the mixed-provider canary."""

from __future__ import annotations

import sys
import types

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import traigent.observability.otel as otel


class FakeInstrumentor:
    """Stands in for a third-party instrumentor forced to capture content."""

    def __init__(self):
        self.provider = None
        self.kwargs = None
        self.uninstrumented = False

    def instrument(self, tracer_provider=None, config=None, **kw):
        self.provider = tracer_provider
        self.kwargs = {"config": config, **kw}

    def uninstrument(self):
        self.uninstrumented = True

    def call_llm(self, prompt):
        tracer = self.provider.get_tracer("fake.llm.lib")
        with tracer.start_as_current_span("chat") as span:
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", "m")
            span.set_attribute("gen_ai.input.messages", prompt)
            span.set_attribute("input.value", prompt)


def _init(collector, **kw):
    return otel.init(
        api_key="k", endpoint=collector.base_url, exit_flush=False,
        schedule_delay_s=0.05, **kw
    )


def test_created_provider_is_verified_and_instrumentors_get_it_explicitly(collector, canary):
    h = _init(collector)
    fake = FakeInstrumentor()
    h.instrument(fake)
    assert fake.provider is h.provider  # explicit, never the global provider
    fake.call_llm(canary)
    assert otel.flush(5).flushed
    assert collector.spans()
    assert canary.encode() not in b"".join(collector.raw_bodies)


def test_unknown_exporter_blocks_instrumentation_without_consent(collector, canary):
    mem = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(mem))
    h = _init(collector, tracer_provider=provider)
    fake = FakeInstrumentor()
    with pytest.raises(otel.UnverifiedExporterError, match="InMemorySpanExporter"):
        h.instrument(fake)
    assert fake.provider is None  # nothing was installed


def test_mixed_provider_canary_consent_exposes_only_the_foreign_exporter(collector, canary):
    """With explicit consent the OTHER exporter sees content; ours never does."""
    mem = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(mem))
    h = _init(collector, tracer_provider=provider, allow_unverified_exporters=True)
    fake = FakeInstrumentor()
    h.instrument(fake)
    fake.call_llm(canary)
    assert otel.flush(5).flushed
    foreign = repr([dict(s.attributes) for s in mem.get_finished_spans()])
    assert canary in foreign  # documented exposure the caller consented to
    assert canary.encode() not in b"".join(collector.raw_bodies)  # ours is clean


def test_per_call_consent_flag(collector):
    mem = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(mem))
    h = _init(collector, tracer_provider=provider)
    fake = FakeInstrumentor()
    h.instrument(fake, allow_unverified_exporters=True)
    assert fake.provider is provider


def test_exporter_added_after_init_is_caught_at_instrument_time(collector):
    h = _init(collector)
    h.provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    with pytest.raises(otel.UnverifiedExporterError):
        h.instrument(FakeInstrumentor())


def test_uninspectable_provider_counts_as_unknown(collector):
    class Opaque:
        def add_span_processor(self, p):
            pass

        def get_tracer(self, *a, **k):
            raise AssertionError("not used")

    h = _init(collector, tracer_provider=Opaque())
    with pytest.raises(otel.UnverifiedExporterError, match="cannot be inspected"):
        h.instrument(FakeInstrumentor())


def test_init_with_instrument_on_unverified_provider_fails_and_cleans_up(collector):
    mem = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(mem))
    with pytest.raises(otel.UnverifiedExporterError):
        _init(collector, tracer_provider=provider, instrument=[FakeInstrumentor()])
    assert otel.get_handle() is None  # a failed init does not leave state behind
    _init(collector)  # and can be retried


def test_unknown_name_and_missing_package_messages(collector):
    h = _init(collector)
    with pytest.raises(ValueError, match="unknown instrumentor"):
        h.instrument("nope")
    with pytest.raises(ImportError, match=r"traigent\[observability-openai\]"):
        h.instrument("openai")


def test_source_masking_config_only_outside_record_mode(collector, monkeypatch):
    seen: list[dict] = []

    class TraceConfig:  # stand-in for the optional package's documented config type
        def __init__(self, **kw):
            seen.append(kw)

    module = types.ModuleType("openinference.instrumentation")
    module.TraceConfig = TraceConfig
    monkeypatch.setitem(sys.modules, "openinference", types.ModuleType("openinference"))
    monkeypatch.setitem(sys.modules, "openinference.instrumentation", module)

    h = _init(collector)  # metadata
    fake = FakeInstrumentor()
    h.instrument(fake)
    assert isinstance(fake.kwargs["config"], TraceConfig)
    assert seen[0]["hide_inputs"] is True and seen[0]["hide_outputs"] is True
    otel.shutdown()

    h = _init(collector, content_mode="record")
    fake2 = FakeInstrumentor()
    h.instrument(fake2)
    assert fake2.kwargs["config"] is None  # record mode: nothing hidden at source


def test_shutdown_uninstruments(collector):
    h = _init(collector)
    fake = FakeInstrumentor()
    h.instrument(fake)
    otel.shutdown()
    assert fake.uninstrumented
