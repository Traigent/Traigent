"""Exporter retry matrix with an injected clock, sleeper and transport."""

from __future__ import annotations

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

from traigent.observability.otel.exporter import TraigentOTLPExporter
from traigent.observability.otel.transport import TransportError, TransportResponse


class FakeTransport:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[bytes] = []

    def post(self, body, *, timeout):
        self.calls.append(body)
        item = self.script.pop(0) if self.script else TransportResponse(200)
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _spans(provider_pair, n=3):
    provider, exporter = provider_pair
    tracer = provider.get_tracer("t")
    for i in range(n):
        with tracer.start_as_current_span(f"s{i}") as span:
            span.set_attribute("gen_ai.request.model", "m")
    return exporter.get_finished_spans()


def _exp(script, **kw):
    clock = Clock()
    transport = FakeTransport(script)
    exp = TraigentOTLPExporter(
        transport,
        clock=clock,
        sleep=clock.sleep,
        rng=lambda: 1.0,  # deterministic: full ceiling
        **kw,
    )
    return exp, transport, clock


def R(status, **kw):
    return TransportResponse(status, **kw)


def test_success_single_attempt(memory_provider):
    exp, tr, _ = _exp([R(200)])
    out = exp.export_batch(_spans(memory_provider))
    assert (out.exported, out.retries, len(tr.calls)) == (3, 0, 1)


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_retryable_statuses_are_retried_then_succeed(memory_provider, status):
    exp, tr, clock = _exp([R(status), R(status), R(200)])
    out = exp.export_batch(_spans(memory_provider))
    assert out.exported == 3 and out.retries == 2 and len(tr.calls) == 3
    assert clock.sleeps == [0.5, 1.0]  # full-jitter ceiling with rng=1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 422, 500, 501])
def test_other_statuses_are_not_retried(memory_provider, status):
    """Negative control: a blanket 408/5xx policy would make 2+ calls here."""
    exp, tr, _ = _exp([R(status), R(200)])
    out = exp.export_batch(_spans(memory_provider))
    assert len(tr.calls) == 1
    assert out.dropped_non_retryable == 3 and out.exported == 0


def test_network_error_is_retried(memory_provider):
    exp, tr, _ = _exp([TransportError("URLError"), R(200)])
    out = exp.export_batch(_spans(memory_provider))
    assert out.exported == 3 and len(tr.calls) == 2


def test_retry_after_is_honoured_and_capped(memory_provider):
    exp, _, clock = _exp([R(429, retry_after=7.0), R(503, retry_after=9999.0), R(200)])
    exp.export_batch(_spans(memory_provider))
    assert clock.sleeps == [7.0, 60.0]


def test_attempts_exhausted_counts_drop(memory_provider):
    exp, tr, _ = _exp([R(503)] * 10, max_attempts=5)
    out = exp.export_batch(_spans(memory_provider))
    assert len(tr.calls) == 5
    assert out.dropped_retry_exhausted == 3 and out.exported == 0


def test_batch_older_than_max_age_is_dropped(memory_provider):
    exp, tr, clock = _exp([R(429, retry_after=60.0)] * 10, max_batch_age=100.0)
    out = exp.export_batch(_spans(memory_provider))
    assert out.dropped_retry_exhausted == 3
    assert clock.now - 1000.0 <= 100.0 + 1e-9  # never slept past the age limit
    assert len(tr.calls) == 2


def test_413_splits_once_and_delivers_both_halves(memory_provider):
    exp, tr, _ = _exp([R(413), R(200), R(200)])
    out = exp.export_batch(_spans(memory_provider, 4))
    assert out.exported == 4 and len(tr.calls) == 3
    sizes = []
    for body in tr.calls[1:]:
        req = ExportTraceServiceRequest()
        req.ParseFromString(body)
        sizes.append(
            sum(len(ss.spans) for rs in req.resource_spans for ss in rs.scope_spans)
        )
    assert sizes == [2, 2]


def test_413_second_level_is_dropped_not_resplit(memory_provider):
    exp, tr, _ = _exp([R(413), R(413), R(200)])
    out = exp.export_batch(_spans(memory_provider, 4))
    assert len(tr.calls) == 3  # original + two halves, no third-level split
    assert out.dropped_non_retryable == 2 and out.exported == 2


def test_413_on_single_span_is_dropped(memory_provider):
    exp, tr, _ = _exp([R(413)])
    out = exp.export_batch(_spans(memory_provider, 1))
    assert out.dropped_non_retryable == 1 and len(tr.calls) == 1


def test_partial_success_is_counted_and_never_retried(memory_provider):
    body = ExportTraceServiceResponse()
    body.partial_success.rejected_spans = 2
    body.partial_success.error_message = "quota"
    exp, tr, _ = _exp([R(200, body=body.SerializeToString())])
    out = exp.export_batch(_spans(memory_provider))
    assert len(tr.calls) == 1
    assert (out.rejected_by_server, out.exported) == (2, 1)


def test_hard_deadline_bounds_total_wait_regardless_of_retry_budget(memory_provider):
    exp, tr, clock = _exp([R(503)] * 20, backoff_base=10.0, max_attempts=50)
    deadline = clock.now + 3.0
    out = exp.export_batch(_spans(memory_provider), deadline=deadline)
    assert clock.now <= deadline
    assert out.dropped_deadline == 3
    assert len(tr.calls) < 50


def test_expired_deadline_sends_nothing(memory_provider):
    exp, tr, clock = _exp([R(200)])
    out = exp.export_batch(_spans(memory_provider), deadline=clock.now - 1)
    assert tr.calls == [] and out.dropped_deadline == 3


def test_oversize_batch_is_split_before_sending(memory_provider):
    exp, tr, _ = _exp([], max_batch_bytes=600)
    out = exp.export_batch(_spans(memory_provider, 6))
    assert out.exported == 6 and len(tr.calls) > 1


def test_wire_body_is_metadata_only(memory_provider, canary):
    provider, _ = memory_provider
    with provider.get_tracer("t").start_as_current_span("x") as s:
        s.set_attribute("gen_ai.input.messages", canary)
    exp, tr, _ = _exp([R(200)])
    exp.export_batch(
        provider._active_span_processor._span_processors[
            0
        ].span_exporter.get_finished_spans()
    )
    assert canary.encode() not in b"".join(tr.calls)
