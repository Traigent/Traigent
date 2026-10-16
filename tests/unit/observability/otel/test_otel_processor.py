"""Span processor: bounds, triggers, single in-flight, flush deadline, fork."""

from __future__ import annotations

import os
import threading
import time

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult

from traigent.observability.otel.exporter import TraigentOTLPExporter
from traigent.observability.otel.processor import TraigentSpanProcessor
from traigent.observability.otel.transport import TransportError, TransportResponse


def _count(body: bytes) -> int:
    req = ExportTraceServiceRequest()
    req.ParseFromString(body)
    return sum(len(ss.spans) for rs in req.resource_spans for ss in rs.scope_spans)


class GateTransport:
    """Records calls; optionally blocks until released; tracks concurrency."""

    def __init__(self, block: bool = False):
        self.spans = 0
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.gate = threading.Event()
        if not block:
            self.gate.set()
        self._lock = threading.Lock()

    def post(self, body, *, timeout):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        released = self.gate.wait(timeout)
        with self._lock:
            self.active -= 1
            if released:
                self.spans += _count(body)
        if not released:  # a real socket would time out here
            raise TransportError("timeout")
        return TransportResponse(200)


def _make(transport, **kw):
    exporter = TraigentOTLPExporter(
        transport, export_timeout=kw.pop("export_timeout", 2.0)
    )
    proc = TraigentSpanProcessor(exporter, exit_flush=False, **kw)
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(proc)
    return provider, proc


def _emit(provider, n):
    tracer = provider.get_tracer("t")
    for _ in range(n):
        with tracer.start_as_current_span("s"):
            pass


def test_timer_triggers_export_without_flush():
    tr = GateTransport()
    provider, proc = _make(tr, schedule_delay_s=0.05)
    _emit(provider, 2)
    deadline = time.time() + 3
    while tr.spans < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert tr.spans == 2
    proc.shutdown()


def test_size_trigger_exports_before_the_timer():
    tr = GateTransport()
    provider, proc = _make(tr, schedule_delay_s=60, max_batch_spans=3)
    _emit(provider, 3)
    deadline = time.time() + 3
    while tr.spans < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert tr.spans == 3
    proc.shutdown()


def test_negative_control_no_trigger_no_export():
    """With a long timer and a large batch nothing leaves until flushed."""
    tr = GateTransport()
    provider, proc = _make(tr, schedule_delay_s=60, max_batch_spans=100)
    _emit(provider, 3)
    time.sleep(0.3)
    assert tr.spans == 0
    assert proc.flush(2).flushed and tr.spans == 3
    proc.shutdown()


def test_queue_is_bounded_and_drops_new_spans_with_count():
    tr = GateTransport(block=True)
    drops: list[tuple[str, int]] = []
    provider, proc = _make(
        tr,
        max_queue_spans=5,
        max_batch_spans=1,
        schedule_delay_s=0.01,
        on_drop=lambda r, n: drops.append((r, n)),
    )
    _emit(provider, 1)  # picked up and blocked in flight
    deadline = time.time() + 3
    while tr.active < 1 and time.time() < deadline:
        time.sleep(0.005)
    _emit(provider, 20)
    stats = proc.stats()
    assert stats["queue_depth"] == 5
    assert stats["dropped_queue_full"] == 15
    assert ("queue_full", 1) in drops
    tr.gate.set()
    proc.shutdown()


def test_only_one_export_in_flight():
    tr = GateTransport()
    provider, proc = _make(tr, max_batch_spans=2, schedule_delay_s=0.01)
    _emit(provider, 50)
    assert proc.flush(5).flushed
    assert tr.spans == 50 and tr.max_active == 1
    proc.shutdown()


def test_flush_returns_at_deadline_even_when_export_is_stuck():
    tr = GateTransport(block=True)
    provider, proc = _make(tr, schedule_delay_s=0.01, export_timeout=30.0)
    _emit(provider, 3)
    start = time.time()
    outcome = proc.flush(0.3)
    elapsed = time.time() - start
    assert outcome.timed_out and not outcome.flushed
    assert elapsed < 1.5
    tr.gate.set()
    proc.shutdown()


def test_on_drop_callback_errors_do_not_break_delivery():
    tr = GateTransport(block=True)

    def boom(reason, n):
        raise RuntimeError("callback bug")

    provider, proc = _make(
        tr, max_queue_spans=1, max_batch_spans=1, schedule_delay_s=0.01, on_drop=boom
    )
    _emit(provider, 1)
    time.sleep(0.1)
    _emit(provider, 5)  # overflow -> callback raises -> swallowed
    tr.gate.set()
    assert proc.flush(3).flushed
    proc.shutdown()


class RecordOnly(Sampler):
    def should_sample(self, *a, **k):
        return SamplingResult(Decision.RECORD_ONLY)

    def get_description(self):
        return "record-only"


def test_unsampled_record_only_spans_are_never_queued():
    tr = GateTransport()
    exporter = TraigentOTLPExporter(tr)
    proc = TraigentSpanProcessor(exporter, exit_flush=False)
    provider = TracerProvider(sampler=RecordOnly(), shutdown_on_exit=False)
    provider.add_span_processor(proc)
    _emit(provider, 4)
    assert proc.flush(1).flushed
    assert tr.spans == 0 and proc.stats()["sampled_out"] == 4
    proc.shutdown()


def test_shutdown_flushes_pending_and_ignores_later_spans():
    tr = GateTransport()
    provider, proc = _make(tr, schedule_delay_s=60, max_batch_spans=100)
    _emit(provider, 3)
    proc.shutdown()
    assert tr.spans == 3
    _emit(provider, 2)
    assert tr.spans == 3


def test_stats_counters_after_success_and_server_rejection():
    class Reject:
        def post(self, body, *, timeout):
            return TransportResponse(403)

    exporter = TraigentOTLPExporter(Reject())
    proc = TraigentSpanProcessor(exporter, exit_flush=False)
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(proc)
    _emit(provider, 3)
    proc.flush(2)
    s = proc.stats()
    assert s["queued"] == 3 and s["dropped_non_retryable"] == 3 and s["exported"] == 0
    proc.shutdown()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_fork_child_discards_parent_queue_and_exports_its_own_spans():
    tr = GateTransport()
    provider, proc = _make(tr, schedule_delay_s=60, max_batch_spans=100)
    _emit(provider, 3)  # parent-queued, not yet exported
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            _emit(provider, 1)
            ok = proc.flush(3).flushed
            os.write(w, f"{tr.spans},{int(ok)}".encode())
        finally:
            os._exit(0)
    os.close(w)
    data = os.read(r, 64).decode()
    os.waitpid(pid, 0)
    child_spans, child_ok = data.split(",")
    assert (child_spans, child_ok) == ("1", "1")  # only its own span, worker alive
    assert proc.flush(3).flushed and tr.spans == 3  # parent still delivers its own
    proc.shutdown()
