"""init / flush / shutdown / stats, provider handling and env isolation."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import traigent.observability.otel as otel
from traigent.observability.otel import contract as C


def _init(collector, **kw):
    kw.setdefault("api_key", "tg_test_key")
    kw.setdefault("endpoint", collector.base_url)
    kw.setdefault("schedule_delay_s", 0.05)
    kw.setdefault("exit_flush", False)
    return otel.init(**kw)


def test_init_exports_over_http_with_headers_and_mode_declaration(collector):
    h = _init(
        collector,
        service_name="svc-a",
        environment="prod",
        release="1.2.3",
        project="proj-1",
    )
    with h.tracer.start_as_current_span("x") as span:
        span.set_attribute("gen_ai.request.model", "m1")
    assert otel.flush(5).flushed
    (rs, ss, span) = collector.spans()[0]
    entry = collector.requests[0]
    assert entry["path"] == "/v1/traces"
    assert entry["headers"]["x-api-key"] == "tg_test_key"
    assert entry["headers"]["content-encoding"] == "gzip"
    assert entry["headers"]["content-type"] == "application/x-protobuf"
    assert entry["headers"]["x-project-id"] == "proj-1"
    res = {a.key: a.value.string_value for a in rs.resource.attributes}
    assert res[C.CONTENT_MODE_ATTRIBUTE] == "metadata"  # default
    assert res["service.name"] == "svc-a"
    assert res["deployment.environment.name"] == "prod"
    assert res["service.version"] == "1.2.3"
    assert otel.stats()["exported"] == 1


def test_declared_mode_follows_content_mode(collector):
    h = _init(collector, content_mode="record")
    with h.tracer.start_as_current_span("x"):
        pass
    otel.flush(5)
    (rs, _, _) = collector.spans()[0]
    res = {a.key: a.value.string_value for a in rs.resource.attributes}
    assert res[C.CONTENT_MODE_ATTRIBUTE] == "record"


def test_env_content_mode_wins_when_more_restrictive(collector, monkeypatch):
    monkeypatch.setenv("TRAIGENT_OBSERVABILITY_CONTENT", "metadata")
    h = _init(collector, content_mode="record")
    assert h.content_mode == "metadata"


def test_otel_exporter_env_cannot_redirect_the_api_key(collector, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:1/x")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=stolen")
    h = _init(collector)
    with h.tracer.start_as_current_span("x"):
        pass
    assert otel.flush(5).flushed
    assert len(collector.requests) == 1  # went to the configured endpoint only
    assert "authorization" not in {k.lower() for k in collector.requests[0]["headers"]}


def test_redirect_is_not_followed_and_key_not_replayed(collector):
    other = type(collector)()
    try:
        collector.redirect_to = other.base_url + "/v1/traces"
        h = _init(collector)
        with h.tracer.start_as_current_span("x"):
            pass
        otel.flush(5)
        assert other.requests == []  # never replayed with credentials
        assert otel.stats()["dropped_non_retryable"] == 1
    finally:
        other.close()


def test_missing_api_key_is_an_error_unless_offline(collector, monkeypatch):
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.setattr(
        "traigent.observability.otel.api.BackendConfig.get_api_key",
        staticmethod(lambda: None),
    )
    with pytest.raises(ValueError, match="api_key"):
        otel.init(endpoint=collector.base_url, exit_flush=False)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    h = otel.init(endpoint=collector.base_url, exit_flush=False)
    assert h.enabled is False
    with h.tracer.start_as_current_span("x"):
        pass
    assert collector.requests == []


def test_unsafe_endpoint_is_rejected(monkeypatch):
    monkeypatch.setenv("TRAIGENT_ENV", "production")
    with pytest.raises(ValueError):
        otel.init(
            api_key="k", endpoint="http://169.254.169.254/latest", exit_flush=False
        )
    with pytest.raises(ValueError):
        otel.init(api_key="k", endpoint="http://example.com", exit_flush=False)


def test_double_init_is_refused_and_shutdown_allows_reinit(collector):
    _init(collector)
    with pytest.raises(RuntimeError, match="already"):
        _init(collector)
    otel.shutdown()
    _init(collector)


def test_attach_to_existing_provider_keeps_its_processors_and_sampler(collector):
    mem = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(mem))
    h = _init(collector, tracer_provider=provider, sample_rate=0.0)
    assert h.provider is provider and not h.created_provider
    with provider.get_tracer("t").start_as_current_span("x"):
        pass
    otel.flush(5)
    # sample_rate applies only to providers we create: the attached provider's
    # own sampler (default: always on) stays authoritative.
    assert len(mem.get_finished_spans()) == 1
    assert len(collector.spans()) == 1


def test_non_sdk_provider_is_rejected(collector):
    with pytest.raises(TypeError):
        _init(collector, tracer_provider=object())


def test_created_provider_applies_sample_rate(collector):
    h = _init(collector, sample_rate=0.0)
    for _ in range(10):
        with h.tracer.start_as_current_span("x"):
            pass
    assert otel.flush(2).flushed
    assert collector.spans() == []


def test_sample_rate_env_and_validation(collector, monkeypatch):
    monkeypatch.setenv("TRAIGENT_OBSERVABILITY_SAMPLE_RATE", "0")
    h = _init(collector)
    with h.tracer.start_as_current_span("x"):
        pass
    otel.flush(2)
    assert collector.spans() == []
    otel.shutdown()
    monkeypatch.setenv("TRAIGENT_OBSERVABILITY_SAMPLE_RATE", "banana")
    with pytest.raises(ValueError):
        _init(collector)
    monkeypatch.delenv("TRAIGENT_OBSERVABILITY_SAMPLE_RATE")
    with pytest.raises(ValueError):
        _init(collector, sample_rate=2)


def test_flush_and_stats_without_init_are_safe():
    assert otel.flush().flushed
    assert otel.stats() == {"enabled": False}
    otel.shutdown()
