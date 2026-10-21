"""Offline is rechecked at record and send time for an online-initialised pipeline (#2240)."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider

from traigent.observability.otel.exporter import TraigentOTLPExporter
from traigent.observability.otel.processor import TraigentSpanProcessor
from traigent.observability.otel.transport import TransportResponse
from traigent.utils.env_config import offline_run_scope

CANARY = "CANARY_OTEL_OFFLINE_2240_q7"


class SpyTransport:
    def __init__(self) -> None:
        self.posts: list[bytes] = []

    def post(self, body: bytes, *, timeout: float) -> TransportResponse:
        self.posts.append(body)
        return TransportResponse(200)


@pytest.fixture(autouse=True)
def _clear_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TRAIGENT_OFFLINE_MODE", "TRAIGENT_OFFLINE"):
        monkeypatch.delenv(name, raising=False)


def _pipeline() -> tuple[TracerProvider, TraigentSpanProcessor, SpyTransport]:
    transport = SpyTransport()
    exporter = TraigentOTLPExporter(transport, export_timeout=2.0)
    proc = TraigentSpanProcessor(exporter, exit_flush=False)
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(proc)
    return provider, proc, transport


def test_spans_recorded_offline_are_never_sent_during_or_after_scope() -> None:
    provider, proc, transport = _pipeline()
    tracer = provider.get_tracer("t")
    try:
        with offline_run_scope(True):
            with tracer.start_as_current_span("op") as span:
                span.set_attribute("traigent.canary", CANARY)
            proc.flush(2.0)
            assert transport.posts == []
        proc.flush(2.0)
        assert transport.posts == []
        stats = proc.stats()
        assert stats["dropped_offline"] == 1
        assert stats["queued"] == 0
    finally:
        provider.shutdown()


def test_span_queued_online_is_not_sent_if_offline_at_send_time() -> None:
    provider, proc, transport = _pipeline()
    tracer = provider.get_tracer("t")
    try:
        with tracer.start_as_current_span("op"):
            pass
        with offline_run_scope(True):
            proc.flush(2.0)
        assert transport.posts == []
    finally:
        provider.shutdown()


def test_online_spans_still_export() -> None:
    provider, proc, transport = _pipeline()
    tracer = provider.get_tracer("t")
    try:
        with tracer.start_as_current_span("op"):
            pass
        proc.flush(2.0)
        assert len(transport.posts) == 1
    finally:
        provider.shutdown()


def test_span_started_offline_but_ended_online_is_dropped() -> None:
    provider, proc, transport = _pipeline()
    tracer = provider.get_tracer("t")
    try:
        with offline_run_scope(True):
            span = tracer.start_span("op")
            span.set_attribute("traigent.canary", CANARY)
        # Scope exited: the span ends while online.
        span.end()
        proc.flush(2.0)
        assert transport.posts == []
        stats = proc.stats()
        assert stats["dropped_offline"] == 1
        assert stats["queued"] == 0
    finally:
        provider.shutdown()


def test_offline_tracking_overflow_latches_drop_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.observability.otel import processor as processor_mod

    monkeypatch.setattr(processor_mod, "_MAX_OFFLINE_TRACKED", 2)
    provider, proc, transport = _pipeline()
    tracer = provider.get_tracer("t")
    try:
        with offline_run_scope(True):
            spans = [tracer.start_span(f"op{i}") for i in range(3)]
        # Scope exited; end the two tracked spans, then the untracked one online.
        spans[0].end()
        spans[1].end()
        spans[2].end()
        proc.flush(2.0)
        assert transport.posts == []
        assert proc.stats()["dropped_offline"] == 3
        # Latched: a fresh online span is also dropped.
        with tracer.start_as_current_span("fresh"):
            pass
        proc.flush(2.0)
        assert transport.posts == []
        assert proc.stats()["dropped_offline"] == 4
    finally:
        provider.shutdown()
