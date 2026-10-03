"""observe(): spans for sync/async/generators/streams, content modes at source."""

from __future__ import annotations

import asyncio
import gc

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

import traigent.observability.otel as otel
from traigent.observability.otel import contract as C


# Connected-mode semantics (mocked local collector), declared per file (#2033).
pytestmark = pytest.mark.backend_online


@pytest.fixture
def h(collector):
    """Initialised handle attached to a provider that also has an in-memory exporter."""
    mem = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(mem))
    handle = otel.init(
        api_key="k",
        endpoint=collector.base_url,
        tracer_provider=provider,
        exit_flush=False,
        schedule_delay_s=0.05,
    )
    handle.mem = mem
    handle.collector = collector
    return handle


def _finished(h):
    return {s.name: s for s in h.mem.get_finished_spans()}


def test_sync_decorator_nesting_and_type(h):
    @otel.observe("inner", as_type="tool", tool_name="calc")
    def inner(x):
        return x + 1

    @otel.observe
    def outer(x):
        return inner(x)

    assert outer(1) == 2
    spans = _finished(h)
    assert spans["inner"].parent.span_id == spans["outer"].context.span_id
    assert spans["inner"].attributes[C.ATTR_OBSERVATION_TYPE] == "tool"
    assert spans["inner"].attributes["gen_ai.tool.name"] == "calc"
    assert spans["outer"].attributes[C.ATTR_OBSERVATION_TYPE] == "span"


def test_invalid_type_and_mode_fail_at_creation():
    with pytest.raises(ValueError):
        otel.observe("x", as_type="nope")
    with pytest.raises(ValueError):
        otel.observe("x", content_mode="banana")


def test_error_sets_status_and_type_but_no_message_in_metadata_mode(h, canary):
    @otel.observe("boom")
    def boom():
        raise ValueError(canary)

    with pytest.raises(ValueError):
        boom()
    span = _finished(h)["boom"]
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description is None
    assert span.attributes["error.type"] == "ValueError"
    assert canary not in repr(span.events) + repr(span.attributes)


def test_metadata_mode_puts_no_content_on_the_span_at_all(h, canary):
    """Source minimisation: other exporters on a shared provider see nothing."""

    @otel.observe("f", metadata={"k": canary})
    def f(secret):
        return {"out": secret}

    f(canary)
    span = _finished(h)["f"]
    assert canary not in repr(dict(span.attributes))
    assert C.ATTR_INPUT not in span.attributes and C.ATTR_OUTPUT not in span.attributes


@pytest.mark.parametrize("mode", ["redacted", "record"])
def test_redacted_and_record_modes_at_source(collector, canary, mode):
    mem = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(mem))
    otel.init(
        api_key="k",
        endpoint=collector.base_url,
        tracer_provider=provider,
        content_mode=mode,
        exit_flush=False,
    )

    @otel.observe("f", metadata={"k": canary})
    def f(secret):
        return {"out": secret}

    f(canary)
    (span,) = mem.get_finished_spans()
    if mode == "redacted":
        assert span.attributes[C.ATTR_INPUT] == C.REDACTED_PLACEHOLDER
        assert span.attributes[C.ATTR_OUTPUT] == C.REDACTED_PLACEHOLDER
        assert canary not in repr(dict(span.attributes))
    else:
        assert canary in span.attributes[C.ATTR_INPUT]
        assert canary in span.attributes[C.ATTR_OUTPUT]
        assert span.attributes[C.CONTENT_METADATA_PREFIX + "k"] == canary


def test_per_call_override_only_tightens(collector, canary):
    mem = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(mem))
    otel.init(
        api_key="k",
        endpoint=collector.base_url,
        tracer_provider=provider,
        content_mode="record",
        exit_flush=False,
    )
    otel.observe("tight", content_mode="metadata")(lambda x: x)(canary)
    otel.observe("loose", redact_input=True)(lambda x: x)(canary)
    spans = {s.name: s for s in mem.get_finished_spans()}
    assert C.ATTR_INPUT not in spans["tight"].attributes
    assert spans["tight"].attributes[C.CONTENT_MODE_ATTRIBUTE] == "metadata"
    assert spans["loose"].attributes[C.ATTR_INPUT] == C.REDACTED_PLACEHOLDER
    otel.shutdown()
    # loosening beyond the client's mode is ignored
    otel.init(
        api_key="k",
        endpoint=collector.base_url,
        tracer_provider=provider,
        content_mode="metadata",
        exit_flush=False,
    )
    otel.observe("loosen", content_mode="record")(lambda x: x)(canary)
    assert (
        C.ATTR_INPUT
        not in {s.name: s for s in mem.get_finished_spans()}["loosen"].attributes
    )


def test_context_manager_and_async_context_manager(h):
    with otel.observe("cm", as_type="chain") as span:
        assert span.is_recording()

    async def go():
        async with otel.observe("acm"):
            pass

    asyncio.run(go())
    assert {"cm", "acm"} <= set(_finished(h))
    assert _finished(h)["cm"].attributes[C.ATTR_OBSERVATION_TYPE] == "chain"


def test_async_function_and_concurrent_attribute_isolation(h):
    @otel.observe("work")
    async def work(i):
        await asyncio.sleep(0.01 * (3 - i))
        with otel.observe(f"child{i}"):
            pass
        return i

    async def run(i):
        with otel.attributes(session_id=f"s{i}", user_id=f"u{i}", tags=[f"t{i}"]):
            return await work(i)

    async def main():
        return await asyncio.gather(*(run(i) for i in range(3)))

    assert asyncio.run(main()) == [0, 1, 2]
    for s in h.mem.get_finished_spans():
        if s.name.startswith("child"):
            i = s.name[-1]
            assert s.attributes[C.ATTR_SESSION_ID] == f"s{i}"
            assert s.attributes[C.ATTR_USER_ID] == f"u{i}"
            assert C.ATTR_TAGS not in s.attributes  # tags are content
    # nothing leaked into the ambient context afterwards
    with otel.observe("after"):
        pass
    assert C.ATTR_SESSION_ID not in _finished(h)["after"].attributes


def test_attributes_restored_after_exception_and_nested_merge(h):
    with pytest.raises(RuntimeError):
        with otel.attributes(session_id="outer"):
            with otel.attributes(user_id="inner"):
                with otel.observe("in"):
                    pass
            raise RuntimeError
    with otel.observe("out"):
        pass
    spans = _finished(h)
    assert spans["in"].attributes[C.ATTR_SESSION_ID] == "outer"
    assert spans["in"].attributes[C.ATTR_USER_ID] == "inner"
    assert C.ATTR_SESSION_ID not in spans["out"].attributes


def test_generator_span_ends_only_when_consumed(h):
    @otel.observe("stream")
    def stream():
        yield 1
        yield 2

    it = stream()
    assert "stream" not in _finished(h)
    assert next(it) == 1
    assert "stream" not in _finished(h)  # still open mid-stream
    assert list(it) == [2]
    assert "stream" in _finished(h)


def test_generator_child_spans_parent_to_the_stream_span(h):
    @otel.observe("stream")
    def stream():
        with otel.observe("step"):
            yield 1

    list(stream())
    spans = _finished(h)
    assert spans["step"].parent.span_id == spans["stream"].context.span_id


def test_abandoned_generator_ends_its_span(h):
    @otel.observe("stream")
    def stream():
        yield 1
        yield 2

    it = stream()
    next(it)
    it.close()
    assert "stream" in _finished(h)
    it2 = stream()
    next(it2)
    del it2
    gc.collect()
    assert len([s for s in h.mem.get_finished_spans() if s.name == "stream"]) == 2


def test_generator_error_marks_span(h):
    @otel.observe("stream")
    def stream():
        yield 1
        raise KeyError("x")

    with pytest.raises(KeyError):
        list(stream())
    assert _finished(h)["stream"].status.status_code is StatusCode.ERROR


def test_generator_send_and_throw_are_forwarded(h):
    @otel.observe("co")
    def co():
        got = yield 1
        try:
            yield got * 2
        except ValueError:
            yield "handled"

    g = co()
    assert next(g) == 1
    assert g.send(21) == 42
    assert g.throw(ValueError()) == "handled"
    g.close()
    assert "co" in _finished(h)


def test_async_generator_stream_and_abandonment(h):
    @otel.observe("astream")
    async def astream():
        yield "a"
        yield "b"

    async def consume():
        return [x async for x in astream()]

    assert asyncio.run(consume()) == ["a", "b"]
    assert "astream" in _finished(h)

    async def abandon():
        agen = astream()
        await agen.__anext__()
        await agen.aclose()

    asyncio.run(abandon())
    assert len([s for s in h.mem.get_finished_spans() if s.name == "astream"]) == 2


def test_observe_without_init_is_a_harmless_noop():
    otel.shutdown()

    @otel.observe
    def f():
        return 5

    assert f() == 5


def test_end_to_end_metadata_only_on_the_wire(h, canary):
    @otel.observe("chatty")
    def chatty(prompt):
        return prompt

    chatty(canary)
    assert otel.flush(5).flushed
    assert h.collector.spans()
    assert canary.encode() not in b"".join(h.collector.raw_bodies)


def test_observe_arguments_stamp_the_span_itself_and_tags_follow_mode(collector):
    mem = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(mem))
    otel.init(
        api_key="k",
        endpoint=collector.base_url,
        tracer_provider=provider,
        content_mode="record",
        exit_flush=False,
    )
    with otel.observe("root", session_id="s1", user_id="u1", tags=["a", "b"]):
        with otel.observe("kid"):
            pass
    spans = {s.name: s for s in mem.get_finished_spans()}
    for name in ("root", "kid"):
        assert spans[name].attributes[C.ATTR_SESSION_ID] == "s1"
        assert tuple(spans[name].attributes[C.ATTR_TAGS]) == ("a", "b")
    otel.shutdown()
    otel.init(
        api_key="k",
        endpoint=collector.base_url,
        tracer_provider=provider,
        content_mode="redacted",
        exit_flush=False,
    )
    with otel.observe("red", tags=["secret-tag"]):
        pass
    red = {s.name: s for s in mem.get_finished_spans()}["red"]
    assert tuple(red.attributes[C.ATTR_TAGS]) == (C.REDACTED_PLACEHOLDER,)


def test_wire_has_no_name_or_tracestate_canary_in_metadata_mode(h, canary):
    """observe(name) under the default scope + a vendor tracestate: raw and
    gunzipped wire bytes never contain the canary."""
    import opentelemetry.trace as ot
    from opentelemetry.trace import SpanContext, TraceFlags, TraceState

    ts = TraceState.from_header([f"vendor={canary}"])
    remote = SpanContext(0xAB, 0xCD, True, TraceFlags(1), trace_state=ts)
    ctx = ot.set_span_in_context(ot.NonRecordingSpan(remote))
    link = ot.Link(SpanContext(1, 2, False, TraceFlags(1), trace_state=ts))
    name = f"summarise {canary} for bob@example.com"

    def run():
        with otel.observe(name):
            pass
        tracer = h.provider.get_tracer("traigent.observability")
        tracer.start_span("linked", context=ctx, links=[link]).end()

    from opentelemetry import context as otctx

    token = otctx.attach(ctx)
    try:
        run()
    finally:
        otctx.detach(token)
    assert otel.flush(5).flushed
    assert h.collector.spans()
    joined = b"".join(h.collector.raw_bodies)
    assert canary.encode() not in joined
    assert canary.encode() not in b"".join(h.collector.raw_wire)
    assert b"user@example.com" not in joined
