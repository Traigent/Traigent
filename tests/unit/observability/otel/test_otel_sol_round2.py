"""Regression tests for the Sol round-2 review of the Python OTel layer.

Privacy items are WIRE canaries (real provider, processor, exporter, collector
stub); each has a record-mode control proving the channel can carry the value.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from opentelemetry.sdk.trace import TracerProvider

import traigent.observability.otel as otel
from traigent.observability.otel import contract as C
from traigent.observability.otel.exporter import TraigentOTLPExporter
from traigent.observability.otel.processor import TraigentSpanProcessor
from traigent.observability.otel.transport import TransportResponse

pytestmark = pytest.mark.backend_online

CONTENT = "CANARY-9d41c7e0-do-not-egress"  # not scrubbable: only the mode hides it


def _init(collector, **kw):
    return otel.init(
        api_key="k",
        endpoint=collector.base_url,
        exit_flush=False,
        schedule_delay_s=0.05,
        **kw,
    )


def _wire(collector) -> dict[str, tuple[object, dict]]:
    return {s.name: (s, collector.attrs(s)) for _r, _s, s in collector.spans()}


def _leaked(collector) -> bool:
    return CONTENT.encode() in b"".join(collector.raw_bodies)


def _content_span(tracer, name):
    with tracer.start_as_current_span(name, attributes={"input.value": CONTENT}):
        pass


# ---------------------------------------------------------------------------
# 1  attributes() merges ambient state at ENTRY, not construction
# ---------------------------------------------------------------------------


def test_attributes_prepared_early_do_not_undo_a_later_metadata_override(collector):
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")
    prepared = otel.attributes(user_id="u-1")  # constructed under plain record mode
    with otel.observe("outer", content_mode="metadata", session_id="sess-B"):
        with prepared:  # entered under the metadata override and session B
            _content_span(tracer, "child")
    assert otel.flush(5).flushed
    span, attrs = _wire(collector)["child"]
    assert not _leaked(collector), "record mode came back through a prepared scope"
    assert attrs[C.ATTR_SESSION_ID] == "sess-B"
    assert attrs[C.ATTR_USER_ID] == "u-1"


def test_attributes_prepared_under_session_a_entered_under_b_uses_b(collector):
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")
    with otel.attributes(session_id="sess-A"):
        prepared = otel.attributes(user_id="u-1")
    with otel.attributes(session_id="sess-B"):
        with prepared:
            _content_span(tracer, "child")
    assert otel.flush(5).flushed
    _span, attrs = _wire(collector)["child"]
    assert attrs[C.ATTR_SESSION_ID] == "sess-B"


def test_control_record_mode_without_override_exports_content(collector):
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")
    with otel.attributes(user_id="u-1"):
        _content_span(tracer, "child")
    assert otel.flush(5).flushed
    assert _leaked(collector)  # control: the channel can carry the canary


# ---------------------------------------------------------------------------
# 2  generator close()/aclose() cleanup runs in the stream's scope
# ---------------------------------------------------------------------------


def _sync_stream(tracer, **opts):
    @otel.observe("stream", **opts)
    def stream():
        try:
            yield 1
            yield 2
        finally:
            _content_span(tracer, "cleanup")

    return stream()


def test_sync_close_cleanup_is_stamped_with_the_streams_mode_and_identity(collector):
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")
    gen = _sync_stream(tracer, content_mode="metadata", session_id="stream-sess")
    next(gen)
    with otel.attributes(session_id="closer-sess"):  # the CLOSING caller's scope
        gen.close()
    assert otel.flush(5).flushed
    assert not _leaked(collector)
    _s, attrs = _wire(collector)["cleanup"]
    assert attrs[C.ATTR_SESSION_ID] == "stream-sess"  # control: not the closer's


def test_sync_close_control_record_stream_exports_cleanup_content(collector):
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")
    gen = _sync_stream(tracer, session_id="stream-sess")
    next(gen)
    gen.close()
    assert otel.flush(5).flushed
    assert _leaked(collector)


def _async_stream(tracer, **opts):
    @otel.observe("astream", **opts)
    async def stream():
        try:
            yield 1
            yield 2
        finally:
            _content_span(tracer, "cleanup")

    return stream()


def test_async_aclose_cleanup_is_stamped_with_the_streams_mode_and_identity(
    collector,
):
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")

    async def main():
        agen = _async_stream(tracer, content_mode="metadata", session_id="stream-sess")
        await agen.__anext__()
        with otel.attributes(session_id="closer-sess"):
            await agen.aclose()

    asyncio.run(main())
    assert otel.flush(5).flushed
    assert not _leaked(collector)
    _s, attrs = _wire(collector)["cleanup"]
    assert attrs[C.ATTR_SESSION_ID] == "stream-sess"


def test_async_aclose_control_record_stream_exports_cleanup_content(collector):
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")

    async def main():
        agen = _async_stream(tracer, session_id="stream-sess")
        await agen.__anext__()
        await agen.aclose()

    asyncio.run(main())
    assert otel.flush(5).flushed
    assert _leaked(collector)


# ---------------------------------------------------------------------------
# 3  scopes opened INSIDE a generator survive its yields
# ---------------------------------------------------------------------------


def test_sync_inner_scope_spans_several_yields(collector):
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")

    @otel.observe("outer")
    def stream():
        with otel.observe("inner", content_mode="metadata", session_id="in-sess"):
            yield 1
            _content_span(tracer, "after-resume")
            yield 2
            _content_span(tracer, "after-second")

    gen = stream()
    assert next(gen) == 1
    with otel.attributes(session_id="consumer-sess"):  # unrelated consumer scope
        assert next(gen) == 2
    with pytest.raises(StopIteration):
        next(gen)
    assert otel.flush(5).flushed
    assert not _leaked(collector)
    wire = _wire(collector)
    inner = wire["inner"][0]
    for name in ("after-resume", "after-second"):
        span, attrs = wire[name]
        assert span.parent_span_id == inner.span_id, name
        assert attrs[C.ATTR_SESSION_ID] == "in-sess", name
    assert wire["inner"][0].parent_span_id == wire["outer"][0].span_id


def test_async_inner_scope_spans_yields_and_cross_task_close(collector):
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")

    @otel.observe("outer")
    async def stream():
        with otel.observe("inner", content_mode="metadata", session_id="in-sess"):
            try:
                yield 1
                _content_span(tracer, "after-resume")
                yield 2
            finally:
                _content_span(tracer, "cleanup")

    async def main():
        agen = stream()
        assert await asyncio.create_task(agen.__anext__()) == 1
        assert await asyncio.create_task(agen.__anext__()) == 2  # another task
        await asyncio.create_task(agen.aclose())  # and cleanup in a third

    asyncio.run(main())
    assert otel.flush(5).flushed
    assert not _leaked(collector)
    wire = _wire(collector)
    inner = wire["inner"][0]
    for name in ("after-resume", "cleanup"):
        span, attrs = wire[name]
        assert span.parent_span_id == inner.span_id, name
        assert attrs[C.ATTR_SESSION_ID] == "in-sess", name


def test_stream_state_does_not_leak_into_the_caller_between_resumes(collector):
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")

    @otel.observe("outer", session_id="stream-sess")
    def stream():
        with otel.observe("inner", session_id="in-sess"):
            yield 1
            yield 2

    gen = stream()
    next(gen)
    with tracer.start_as_current_span("ambient"):
        pass
    next(gen)
    gen.close()
    assert otel.flush(5).flushed
    assert C.ATTR_SESSION_ID not in _wire(collector)["ambient"][1]


# ---------------------------------------------------------------------------
# 4/5  instrumentor install: partial patches undone; unverifiable not owned
# ---------------------------------------------------------------------------


class _Target:
    @staticmethod
    def call():
        return "original"


def _replica_base():
    """BaseInstrumentor's documented semantics: the flag flips AFTER _instrument()
    returns, and uninstrument() is a no-op while the flag is False."""

    class Base:
        _is_instrumented_by_opentelemetry = False

        def instrument(self, **kwargs):
            if self._is_instrumented_by_opentelemetry:
                return None
            kwargs.pop("tracer_provider", None)
            result = self._instrument(**kwargs)
            self._is_instrumented_by_opentelemetry = True
            return result

        def uninstrument(self, **kwargs):
            if self._is_instrumented_by_opentelemetry:
                self._uninstrument(**kwargs)
                self._is_instrumented_by_opentelemetry = False

    return Base


def _real_base():
    mod = pytest.importorskip("opentelemetry.instrumentation.instrumentor")
    return mod.BaseInstrumentor


@pytest.fixture(params=["replica", "real"])
def base_cls(request):
    return _replica_base() if request.param == "replica" else _real_base()


def _patching_instrumentor(base_cls, *, fail: bool):
    class Patching(base_cls):  # type: ignore[misc, valid-type]
        _instance = None  # a fresh singleton per test class

        def instrumentation_dependencies(self):
            return []

        def _instrument(self, **kwargs):
            self._orig = _Target.call
            _Target.call = staticmethod(lambda: "patched")  # type: ignore[assignment]
            if fail:
                raise RuntimeError("boom after patching")

        def _uninstrument(self, **kwargs):
            _Target.call = staticmethod(self._orig)  # type: ignore[assignment]

    return Patching()


def test_failing_instrumentor_partial_patches_are_undone(collector, base_cls):
    handle = _init(collector)
    inst = _patching_instrumentor(base_cls, fail=True)
    with pytest.raises(RuntimeError, match="boom after patching"):
        handle.instrument(inst)
    assert _Target.call() == "original", "partial patch survived a failed install"
    assert handle._instrumentors == []


def test_control_successful_patching_instrumentor_is_patched_then_restored(
    collector, base_cls
):
    handle = _init(collector)
    inst = _patching_instrumentor(base_cls, fail=False)
    handle.instrument(inst)
    assert _Target.call() == "patched"  # control: the patch is observable
    handle.shutdown()
    assert _Target.call() == "original"


class _NoState:
    """A custom instrumentor that exposes no install-state flag at all."""

    def __init__(self):
        self.calls = 0
        self.uninstrumented = False

    def instrument(self, tracer_provider=None, **kw):
        self.calls += 1

    def uninstrument(self, **kw):
        self.uninstrumented = True


def test_unverifiable_instrumentor_is_refused_by_default(collector):
    handle = _init(collector)
    inst = _NoState()
    with pytest.raises(otel.InstrumentationStateError):
        handle.instrument(inst)
    assert inst.calls == 0  # never even asked to install
    assert handle._instrumentors == []
    handle.shutdown()
    assert inst.uninstrumented is False  # not ours


def test_unverifiable_instrumentor_installs_only_on_explicit_adoption(collector):
    handle = _init(collector)
    inst = _NoState()
    handle.instrument(inst, adopt_unverifiable=True)
    assert inst.calls == 1
    handle.shutdown()
    assert inst.uninstrumented is True


# ---------------------------------------------------------------------------
# 6  a flush arriving during backoff wakes the wait
# ---------------------------------------------------------------------------


class _Always503:
    def __init__(self):
        self.posts: list[float] = []

    def post(self, body, *, timeout):
        self.posts.append(time.monotonic())
        return TransportResponse(503, retry_after=60.0)


def test_flush_during_a_long_retry_after_returns_within_its_deadline():
    transport = _Always503()
    exporter = TraigentOTLPExporter(transport, max_attempts=5, retry_after_cap=60.0)
    proc = TraigentSpanProcessor(exporter, exit_flush=False, schedule_delay_s=0.01)
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(proc)
    with provider.get_tracer("t").start_as_current_span("s"):
        pass
    deadline = time.monotonic() + 5
    while not transport.posts and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(transport.posts) == 1  # first POST done; now backing off for 60 s
    started = time.monotonic()
    outcome = proc.flush(1.0)
    elapsed = time.monotonic() - started
    assert elapsed < 2.5, f"flush blocked {elapsed:.1f}s behind a 60 s backoff"
    assert outcome.timed_out or not outcome.flushed
    # the abandoned export is accounted and no POST follows
    end = time.monotonic() + 3
    while proc.stats()["in_flight"] and time.monotonic() < end:
        time.sleep(0.01)
    assert not proc.stats()["in_flight"], "backoff wait outlived the flush deadline"
    assert len(transport.posts) == 1
    assert proc.stats()["dropped_deadline"] == 1
    proc.shutdown()


def test_control_without_flush_the_backoff_still_waits():
    """Negative control: with no flush/shutdown the wait is NOT cut short."""
    transport = _Always503()
    exporter = TraigentOTLPExporter(
        transport, max_attempts=3, retry_after_cap=0.3, backoff_base=0.0
    )
    proc = TraigentSpanProcessor(exporter, exit_flush=False, schedule_delay_s=0.01)
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(proc)
    with provider.get_tracer("t").start_as_current_span("s"):
        pass
    deadline = time.monotonic() + 5
    while len(transport.posts) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(transport.posts) >= 2
    assert transport.posts[1] - transport.posts[0] >= 0.25  # waited ~retry_after
    proc.shutdown()


def _spans():
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    mem = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(mem))
    with provider.get_tracer("t").start_as_current_span("s"):
        pass
    return list(mem.get_finished_spans())


def test_shutdown_wakes_the_backoff_wait():
    transport = _Always503()
    exporter = TraigentOTLPExporter(transport, max_attempts=5, retry_after_cap=60.0)
    spans = _spans()
    thread = threading.Thread(target=lambda: exporter.export_batch(spans), daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not transport.posts and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(transport.posts) == 1
    started = time.monotonic()
    exporter.shutdown()
    thread.join(3)
    assert not thread.is_alive()
    assert time.monotonic() - started < 2.5
    assert len(transport.posts) == 1
