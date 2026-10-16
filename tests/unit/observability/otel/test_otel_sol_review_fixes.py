"""Regression tests for the Sol 6.1 review of the Python OTel layer.

Every privacy test here is a WIRE canary: a planted secret goes through the
real provider, processor and exporter, the collector stub decodes the actual
OTLP request, and absence is asserted in every channel (attributes, status
description, span names, event names, links, tracestate, resource, scope).
A record-mode control keeps each canary honest: the same channel must show the
planted value when the content mode allows it.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from opentelemetry import trace as ot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import Decision
from opentelemetry.trace import Link, SpanContext, SpanKind, TraceFlags, TraceState

import traigent.observability.otel as otel
from traigent.observability.otel import contract as C
from traigent.observability.otel.exporter import TraigentOTLPExporter
from traigent.observability.otel.processor import TraigentSpanProcessor
from traigent.observability.otel.sampling import TraceIdRatioRoot
from traigent.observability.otel.transport import TransportResponse

# Connected-mode semantics (mocked local collector), declared per file (#2033).
pytestmark = pytest.mark.backend_online

SECRET = "sk-CANARYabcdefghijklmnop123456"  # matches the SDK's API-key scrubber
VISIBLE = "visible-record-text"
CONTENT = "CANARY-7f3a91c2-do-not-egress"  # not scrubbable: only the mode hides it


def _init(collector, **kw):
    return otel.init(
        api_key="k",
        endpoint=collector.base_url,
        exit_flush=False,
        schedule_delay_s=0.05,
        **kw,
    )


def _with_memory(handle):
    mem = InMemorySpanExporter()
    handle.provider.add_span_processor(SimpleSpanProcessor(mem))
    return mem


def _strings(val):
    kind = val.WhichOneof("value")
    if kind == "string_value":
        yield val.string_value
    elif kind == "array_value":
        for item in val.array_value.values:
            yield from _strings(item)
    elif kind == "kvlist_value":
        for kv in val.kvlist_value.values:
            yield kv.key
            yield from _strings(kv.value)


def channels(collector) -> dict[str, list[str]]:
    """Every textual channel of every decoded span, by channel name."""
    out: dict[str, list[str]] = {
        "attributes": [],
        "status": [],
        "span_name": [],
        "event_name": [],
        "event_attributes": [],
        "link_attributes": [],
        "trace_state": [],
        "resource": [],
        "scope": [],
    }
    for entry in collector.requests:
        for rs in entry["req"].resource_spans:
            for kv in rs.resource.attributes:
                out["resource"] += [kv.key, *_strings(kv.value)]
            for ss in rs.scope_spans:
                out["scope"] += [ss.scope.name, ss.scope.version]
                for span in ss.spans:
                    out["span_name"].append(span.name)
                    out["status"].append(span.status.message)
                    out["trace_state"].append(span.trace_state)
                    for kv in span.attributes:
                        out["attributes"] += [kv.key, *_strings(kv.value)]
                    for ev in span.events:
                        out["event_name"].append(ev.name)
                        for kv in ev.attributes:
                            out["event_attributes"] += [kv.key, *_strings(kv.value)]
                    for link in span.links:
                        out["trace_state"].append(link.trace_state)
                        for kv in link.attributes:
                            out["link_attributes"] += [kv.key, *_strings(kv.value)]
    return out


def assert_absent_everywhere(collector, needle: str) -> None:
    for name, values in channels(collector).items():
        assert not any(needle in v for v in values), f"{needle!r} leaked in {name}"
    assert needle.encode() not in b"".join(collector.raw_bodies)
    assert needle.encode() not in b"".join(collector.raw_wire)


def _wire_attrs(collector, name: str) -> dict:
    for _rs, _ss, span in collector.spans():
        if span.name == name:
            return collector.attrs(span)
    raise AssertionError(f"span {name!r} not on the wire")


# ---------------------------------------------------------------------------
# P1-1  sampled root spans keep the attributes passed to start_span()
# ---------------------------------------------------------------------------


def test_sampler_returns_start_attributes_when_recording():
    result = TraceIdRatioRoot(1.0).should_sample(
        None, 1, "s", SpanKind.INTERNAL, {"gen_ai.request.model": "m"}
    )
    assert result.decision is Decision.RECORD_AND_SAMPLE
    assert dict(result.attributes) == {"gen_ai.request.model": "m"}


def test_unsampled_root_span_records_nothing():
    result = TraceIdRatioRoot(0.0).should_sample(None, 1, "s", None, {"a": "b"})
    assert result.decision is Decision.DROP


def test_root_and_child_start_attributes_reach_the_wire(collector):
    handle = _init(collector)
    tracer = handle.provider.get_tracer("lib")
    with tracer.start_as_current_span(
        "root",
        attributes={
            "gen_ai.request.model": "m-root",
            "gen_ai.usage.input_tokens": 7,
            "openinference.span.kind": "LLM",
        },
    ):
        with tracer.start_as_current_span(
            "child", attributes={"gen_ai.request.model": "m-child"}
        ):
            pass
    assert otel.flush(5).flushed
    root = _wire_attrs(collector, "root")
    assert root["gen_ai.request.model"] == "m-root"
    assert root["gen_ai.usage.input_tokens"] == 7
    assert root["openinference.span.kind"] == "LLM"
    assert _wire_attrs(collector, "child")["gen_ai.request.model"] == "m-child"


# ---------------------------------------------------------------------------
# P1-3  content-mode declaration vectors come from the shared contract
# ---------------------------------------------------------------------------


def _vector_attrs(vector) -> dict:
    declared = vector["declared"]
    return {C.CONTENT_MODE_ATTRIBUTE: declared["value"]} if declared["present"] else {}


def test_effective_mode_follows_every_contract_vector():
    from traigent.observability.otel.policy import ContentPolicy

    vectors = C.CONTRACT["content_mode"]["resolution_vectors"]
    assert len(vectors) >= 14
    for vector in vectors:
        got = ContentPolicy(vector["configured"]).effective_mode(_vector_attrs(vector))
        assert got == vector["expected"], vector["id"]


def test_sanitize_follows_every_contract_vector_with_a_record_control():
    """Drive the real rebuild: expected ``record`` keeps content, all else drops it."""
    from opentelemetry.sdk.trace import ReadableSpan

    from traigent.observability.otel.policy import ContentPolicy

    ctx = SpanContext(0xAB, 0xCD, False, TraceFlags(1))
    for vector in C.CONTRACT["content_mode"]["resolution_vectors"]:
        attrs = {"input.value": CONTENT, **_vector_attrs(vector)}
        span = ReadableSpan(name="s", context=ctx, attributes=attrs)
        out = ContentPolicy(vector["configured"]).sanitize(span)
        expected = vector["expected"]
        value = out.attributes.get("input.value")
        if expected == "record":
            assert value == CONTENT, vector["id"]  # control: content can egress
        elif expected == "redacted":
            assert value == C.REDACTED_PLACEHOLDER, vector["id"]
        else:
            assert value is None, vector["id"]
        assert CONTENT not in repr(dict(out.attributes)) or expected == "record"


@pytest.mark.parametrize("bad", ["full", "Record", "", " record", "RECORD"])
def test_invalid_declaration_on_a_record_client_is_metadata_on_the_wire(collector, bad):
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")
    with tracer.start_as_current_span("bad") as span:
        span.set_attribute(C.CONTENT_MODE_ATTRIBUTE, bad)
        span.set_attribute("input.value", CONTENT)
        span.set_attribute("tag.tags", [CONTENT])
    with tracer.start_as_current_span("control") as span:
        span.set_attribute("input.value", VISIBLE)
    assert otel.flush(5).flushed
    assert_absent_everywhere(collector, CONTENT)
    assert _wire_attrs(collector, "control")["input.value"] == VISIBLE  # control


# ---------------------------------------------------------------------------
# P1-2  per-call override governs stamping and every descendant
# ---------------------------------------------------------------------------


def test_metadata_override_on_a_record_client_covers_stamping_and_children(collector):
    handle = _init(collector, content_mode="record")
    mem = _with_memory(handle)  # a foreign exporter sees the raw span
    tracer = handle.provider.get_tracer("lib")
    with otel.observe(
        "parent", content_mode="metadata", session_id="s1", tags=[CONTENT]
    ):
        with otel.attributes(metadata={"k": CONTENT}):
            with tracer.start_as_current_span("child") as child:
                child.set_attribute("input.value", CONTENT)
                child.set_attribute("gen_ai.request.model", "m")
    assert otel.flush(5).flushed
    assert_absent_everywhere(collector, CONTENT)
    raw = {s.name: s for s in mem.get_finished_spans()}
    for name in ("parent", "child"):
        attrs = raw[name].attributes
        # stamping itself respects the effective mode: nothing raw is ever set
        assert C.ATTR_TAGS not in attrs, name
        assert C.CONTENT_METADATA_PREFIX + "k" not in attrs, name
    assert raw["child"].attributes[C.CONTENT_MODE_ATTRIBUTE] == "metadata"
    assert _wire_attrs(collector, "child")["gen_ai.request.model"] == "m"


def test_record_client_without_override_still_records(collector):
    """Control for the test above: the same flow is visible in plain record mode."""
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")
    with otel.observe("parent", tags=["tag-visible"]):
        with tracer.start_as_current_span("child") as child:
            child.set_attribute("input.value", VISIBLE)
    assert otel.flush(5).flushed
    assert _wire_attrs(collector, "child")["input.value"] == VISIBLE
    assert "tag-visible" in channels(collector)["attributes"]


def test_nested_overrides_only_tighten(collector):
    handle = _init(collector, content_mode="record")
    mem = _with_memory(handle)
    with otel.observe("outer", content_mode="redacted"):
        with otel.observe("inner", content_mode="record"):  # cannot loosen
            with otel.observe("leaf"):
                pass
    modes = {
        s.name: s.attributes.get(C.CONTENT_MODE_ATTRIBUTE)
        for s in mem.get_finished_spans()
    }
    assert modes == {"outer": "redacted", "inner": "redacted", "leaf": "redacted"}


# ---------------------------------------------------------------------------
# P1-4  every exported text channel is scrubbed in record mode
# ---------------------------------------------------------------------------


def test_record_mode_scrubs_secrets_in_status_names_events_links(collector):
    handle = _init(collector, content_mode="record")
    tracer = handle.provider.get_tracer("lib")
    ts = TraceState.from_header([f"vendor={SECRET}"])
    linked = SpanContext(1, 2, False, TraceFlags(1), trace_state=ts)
    with pytest.raises(RuntimeError):
        with otel.observe(f"call {SECRET} {VISIBLE}"):
            raise RuntimeError(f"upstream rejected {SECRET}")
    with tracer.start_as_current_span(
        "evt", links=[Link(linked, {"link.note": f"n {SECRET}"})]
    ) as span:
        span.add_event(f"event {SECRET}", {"detail": f"d {SECRET}"})
        span.set_status(ot.Status(ot.StatusCode.ERROR, f"status {SECRET}"))
    assert otel.flush(5).flushed
    assert_absent_everywhere(collector, SECRET)
    chans = channels(collector)
    # controls: record mode still ships the surrounding text, scrubbed
    assert any(VISIBLE in n for n in chans["span_name"])
    assert any("REDACTED" in n for n in chans["span_name"])
    assert any("REDACTED" in m for m in chans["status"])
    assert any(n.startswith("event ") for n in chans["event_name"])


def test_metadata_mode_blocks_the_same_secrets_in_every_channel(collector):
    handle = _init(collector)
    tracer = handle.provider.get_tracer("lib")
    ts = TraceState.from_header([f"vendor={SECRET}"])
    linked = SpanContext(1, 2, False, TraceFlags(1), trace_state=ts)
    with pytest.raises(RuntimeError):
        with otel.observe(f"call {SECRET}"):
            raise RuntimeError(f"upstream rejected {SECRET}")
    with tracer.start_as_current_span(
        SECRET, links=[Link(linked, {"link.note": SECRET})]
    ) as span:
        span.add_event(SECRET, {"detail": SECRET})
        span.set_status(ot.Status(ot.StatusCode.ERROR, SECRET))
    assert otel.flush(5).flushed
    assert channels(collector)["span_name"]  # something was exported
    assert_absent_everywhere(collector, SECRET)


# ---------------------------------------------------------------------------
# P1-6  usage aliases cross the wire (metadata mode), untouched
# ---------------------------------------------------------------------------


def _usage_aliases() -> list[str]:
    classes = C.CONTRACT["usage_classes"]
    aliases = [a for names in classes["attributes"].values() for a in names]
    return [*aliases, *classes["total_tokens"]]


def test_every_usage_alias_survives_metadata_mode_with_its_value(collector):
    handle = _init(collector)
    tracer = handle.provider.get_tracer("lib")
    aliases = _usage_aliases()
    assert len(aliases) >= 13
    expected = {alias: 100 + i for i, alias in enumerate(aliases)}
    with tracer.start_as_current_span("usage") as span:
        for alias, value in expected.items():
            span.set_attribute(alias, value)
    assert otel.flush(5).flushed
    wire = _wire_attrs(collector, "usage")
    for alias, value in expected.items():
        assert wire.get(alias) == value, alias


def test_usage_values_out_of_bounds_are_dropped(collector):
    handle = _init(collector)
    tracer = handle.provider.get_tracer("lib")
    top = C.CONTRACT["metadata_allowlist"]["attributes"][
        "gen_ai.usage.cache_read.input_tokens"
    ]["maximum"]
    with tracer.start_as_current_span("bounds") as span:
        span.set_attribute("gen_ai.usage.cache_read.input_tokens", top)  # ok
        span.set_attribute("gen_ai.usage.cache_creation.input_tokens", top + 1)
        span.set_attribute("llm.token_count.prompt", -1)
        span.set_attribute("llm.token_count.completion", True)
    assert otel.flush(5).flushed
    wire = _wire_attrs(collector, "bounds")
    assert wire["gen_ai.usage.cache_read.input_tokens"] == top
    assert "gen_ai.usage.cache_creation.input_tokens" not in wire
    assert "llm.token_count.prompt" not in wire
    assert "llm.token_count.completion" not in wire


def test_sdk_never_pre_subtracts_usage_for_any_normalisation_vector(collector):
    """The SDK forwards raw buckets; normalisation happens once, in the receiver."""
    handle = _init(collector)
    tracer = handle.provider.get_tracer("lib")
    classes = C.CONTRACT["usage_classes"]["attributes"]
    vectors = C.CONTRACT["usage_classes"]["normalisation_vectors"]
    marker = C.CONTRACT["usage_classes"]["semantics_marker"]["attribute"]
    marker_values = C.CONTRACT["usage_classes"]["semantics_marker"]["values"]
    for vector in vectors:
        with tracer.start_as_current_span(vector["id"]) as span:
            for bucket in ("input", "output", "cache_read", "cache_write", "reasoning"):
                if bucket in vector:  # absent counters stay absent (contract 0ce328a0)
                    span.set_attribute(classes[bucket][0], vector[bucket])
            if vector["semantics"] is not None:
                span.set_attribute(marker, vector["semantics"])
    assert otel.flush(5).flushed
    for vector in vectors:
        wire = _wire_attrs(collector, vector["id"])
        for bucket in ("input", "output", "cache_read", "cache_write", "reasoning"):
            if bucket in vector:
                assert wire[classes[bucket][0]] == vector[bucket], vector["id"]
            else:
                assert classes[bucket][0] not in wire, vector["id"]
        declared = vector["semantics"] if vector["semantics"] in marker_values else None
        assert wire.get(marker) == declared, vector["id"]


def test_invalid_usage_semantics_marker_is_dropped(collector):
    handle = _init(collector)
    tracer = handle.provider.get_tracer("lib")
    with tracer.start_as_current_span("m") as span:
        span.set_attribute("traigent.usage.semantics", "sideways")
    assert otel.flush(5).flushed
    assert "traigent.usage.semantics" not in _wire_attrs(collector, "m")


# ---------------------------------------------------------------------------
# P1-5  generator scopes are isolated per stream
# ---------------------------------------------------------------------------


def _session_of(span) -> str | None:
    return span.attributes.get(C.ATTR_SESSION_ID)


def test_interleaved_sync_streams_do_not_share_or_leak_scope(collector):
    handle = _init(collector)
    mem = _with_memory(handle)
    tracer = handle.provider.get_tracer("lib")

    def make(session):
        @otel.observe(f"stream-{session}", session_id=session)
        def stream():
            for i in range(3):
                with tracer.start_as_current_span(f"{session}-child-{i}"):
                    pass
                yield i

        return stream()

    g1, g2 = make("user-a"), make("user-b")
    next(g1)
    next(g2)
    # an unrelated ambient span started BETWEEN resumes carries no stream scope
    with tracer.start_as_current_span("ambient"):
        pass
    next(g1)
    next(g2)
    next(g2)
    next(g1)
    g1.close()
    g2.close()
    # after every stream ends, the caller scope is fully restored
    with tracer.start_as_current_span("after"):
        pass
    by_name = {s.name: s for s in mem.get_finished_spans()}
    assert _session_of(by_name["ambient"]) is None
    assert _session_of(by_name["after"]) is None
    for i in range(3):
        assert _session_of(by_name[f"user-a-child-{i}"]) == "user-a"
        assert _session_of(by_name[f"user-b-child-{i}"]) == "user-b"
    assert _session_of(by_name["stream-user-a"]) == "user-a"
    assert _session_of(by_name["stream-user-b"]) == "user-b"


def test_sync_stream_can_be_resumed_from_different_threads(collector):
    handle = _init(collector)
    mem = _with_memory(handle)

    @otel.observe("threaded", session_id="sess")
    def stream():
        yield 1
        yield 2

    gen = stream()
    errors: list[BaseException] = []

    def step():
        try:
            next(gen)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    for _ in range(2):
        t = threading.Thread(target=step)
        t.start()
        t.join()
    gen.close()
    assert not errors
    assert [s.name for s in mem.get_finished_spans()] == ["threaded"]


def test_async_stream_closed_from_another_task_ends_span_without_error(collector):
    handle = _init(collector)
    mem = _with_memory(handle)

    @otel.observe("astream", session_id="sess")
    async def stream():
        yield 1
        yield 2

    async def main():
        agen = stream()
        first = asyncio.create_task(agen.__anext__())
        assert await first == 1
        closer = asyncio.create_task(agen.aclose())  # a DIFFERENT task/context
        await closer  # must not raise "created in a different Context"

    asyncio.run(main())
    (span,) = [s for s in mem.get_finished_spans() if s.name == "astream"]
    assert span.end_time is not None
    # the ambient scope of the test thread is untouched
    tracer = handle.provider.get_tracer("lib")
    with tracer.start_as_current_span("after"):
        pass
    after = [s for s in mem.get_finished_spans() if s.name == "after"][0]
    assert _session_of(after) is None


def test_interleaved_async_streams_do_not_share_scope(collector):
    handle = _init(collector)
    mem = _with_memory(handle)
    tracer = handle.provider.get_tracer("lib")

    def make(session):
        @otel.observe(f"as-{session}", session_id=session)
        async def stream():
            for i in range(2):
                with tracer.start_as_current_span(f"{session}-c{i}"):
                    pass
                yield i

        return stream()

    async def main():
        a, b = make("a"), make("b")
        await a.__anext__()
        await b.__anext__()
        with tracer.start_as_current_span("amb"):
            pass
        await a.__anext__()
        await b.__anext__()
        await a.aclose()
        await b.aclose()

    asyncio.run(main())
    by_name = {s.name: s for s in mem.get_finished_spans()}
    assert _session_of(by_name["amb"]) is None
    assert _session_of(by_name["a-c0"]) == "a" and _session_of(by_name["a-c1"]) == "a"
    assert _session_of(by_name["b-c0"]) == "b" and _session_of(by_name["b-c1"]) == "b"


# ---------------------------------------------------------------------------
# P2  overlapping exits of one reused observe() object
# ---------------------------------------------------------------------------


def test_overlapping_async_exits_end_their_own_span(collector):
    handle = _init(collector)
    mem = _with_memory(handle)
    obs = otel.observe("shared")
    spans: dict[str, object] = {}
    entered = {"a": asyncio.Event(), "b": asyncio.Event()}
    a_may_exit = asyncio.Event()

    async def worker(tag):
        async with obs as span:
            spans[tag] = span
            entered[tag].set()
            if tag == "a":
                await entered["b"].wait()
                a_may_exit.set()
            else:
                await a_may_exit.wait()
                await asyncio.sleep(0)  # let A exit first
                await asyncio.sleep(0)

    async def main():
        await asyncio.gather(worker("a"), worker("b"))

    asyncio.run(main())
    finished = mem.get_finished_spans()
    assert len(finished) == 2
    ids = {tag: span.get_span_context().span_id for tag, span in spans.items()}
    assert ids["a"] != ids["b"]
    assert {s.context.span_id for s in finished} == set(ids.values())
    # A exited first: its own span must be the first to end
    assert finished[0].context.span_id == ids["a"]


def test_overlapping_thread_exits_end_their_own_span(collector):
    handle = _init(collector)
    mem = _with_memory(handle)
    obs = otel.observe("shared-threads")
    ids: dict[str, int] = {}
    entered = threading.Barrier(2)
    a_done = threading.Event()

    def worker(tag):
        with obs as span:
            ids[tag] = span.get_span_context().span_id
            entered.wait(5)
            if tag == "b":
                assert a_done.wait(5)
        if tag == "a":
            a_done.set()

    threads = [threading.Thread(target=worker, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    finished = mem.get_finished_spans()
    assert {s.context.span_id for s in finished} == set(ids.values())
    assert finished[0].context.span_id == ids["a"]


def test_exit_without_matching_enter_in_this_context_is_an_explicit_error():
    obs = otel.observe("lonely")
    with pytest.raises(RuntimeError, match="not entered"):
        obs.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# P1-7 / P2  instrumentor installation is verified, owned and rolled back
# ---------------------------------------------------------------------------


class _Base:
    """Mimics the public behaviour of OpenTelemetry's ``BaseInstrumentor``.

    ``instrument()`` silently returns when already instrumented or when it
    cannot install; state is visible through ``_is_instrumented_by_opentelemetry``.
    """

    installs = 0
    silent_conflict = False

    def __init__(self):
        self._flag = False
        self.uninstrumented = False
        self.provider = None

    @property
    def _is_instrumented_by_opentelemetry(self) -> bool:
        return self._flag

    def instrument(self, tracer_provider=None, **kw):
        if self._flag or self.silent_conflict:
            return None
        self.provider = tracer_provider
        self._flag = True
        type(self).installs += 1
        return None

    def uninstrument(self, **kw):
        self._flag = False
        self.uninstrumented = True


class _Conflicting(_Base):
    silent_conflict = True


class _Exploding(_Base):
    def instrument(self, tracer_provider=None, **kw):
        raise RuntimeError("boom during install")


def test_pre_instrumented_instrumentor_is_rejected_and_never_uninstrumented(
    collector,
):
    pre = _Base()
    pre.instrument(tracer_provider=object())  # someone else's unaudited provider
    handle = _init(collector)
    with pytest.raises(otel.InstrumentationStateError):
        handle.instrument(pre)
    handle.shutdown()
    assert pre.uninstrumented is False  # not ours to uninstrument
    assert pre.provider is not handle.provider


def test_silent_dependency_conflict_is_raised_not_assumed(collector):
    handle = _init(collector)
    with pytest.raises(otel.InstrumentationStateError):
        handle.instrument(_Conflicting())
    assert handle._instrumentors == []


def test_failed_later_instrumentor_rolls_back_earlier_installs(collector):
    handle = _init(collector)
    good = _Base()
    with pytest.raises(RuntimeError, match="boom"):
        handle.instrument(good, _Exploding())
    assert good.uninstrumented is True
    assert good._is_instrumented_by_opentelemetry is False
    assert handle._instrumentors == []


def test_successful_installs_are_owned_and_uninstrumented_on_shutdown(collector):
    handle = _init(collector)
    good = _Base()
    handle.instrument(good)
    assert good.provider is handle.provider
    handle.shutdown()
    assert good.uninstrumented is True


# ---------------------------------------------------------------------------
# P2  exporter / processor
# ---------------------------------------------------------------------------


class _ScriptedTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies: list[bytes] = []
        self.first_entered = threading.Event()
        self.release_first = threading.Event()
        self.block_first = False

    def post(self, body, *, timeout):
        self.bodies.append(body)
        if len(self.bodies) == 1 and self.block_first:
            self.first_entered.set()
            self.release_first.wait(5)
        return self.responses.pop(0) if self.responses else TransportResponse(200)


def _span_batch(n=1, attr_bytes=0):
    provider = TracerProvider(shutdown_on_exit=False)
    mem = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(mem))
    tracer = provider.get_tracer("t")
    for i in range(n):
        with tracer.start_as_current_span(f"s{i}") as span:
            span.set_attribute("gen_ai.request.model", "m")
            if attr_bytes:
                span.set_attribute("tool.name", "x" * attr_bytes)
    return list(mem.get_finished_spans())


def test_flush_reaches_an_export_that_was_already_in_flight():
    transport = _ScriptedTransport([TransportResponse(503), TransportResponse(503)])
    transport.block_first = True
    exporter = TraigentOTLPExporter(
        transport, backoff_base=0.0, sleep=lambda s: None, max_attempts=5
    )
    proc = TraigentSpanProcessor(exporter, exit_flush=False, schedule_delay_s=0.01)
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(proc)
    with provider.get_tracer("t").start_as_current_span("s"):
        pass
    assert transport.first_entered.wait(5)  # the export started with NO deadline
    outcome = proc.flush(0.2)  # flush arrives while it is in flight
    assert outcome.timed_out
    time.sleep(0.05)
    transport.release_first.set()  # first POST now returns a retryable 503
    deadline = time.time() + 3
    while proc.stats()["in_flight"] and time.time() < deadline:
        time.sleep(0.01)
    assert len(transport.bodies) == 1, "retry continued past the flush deadline"
    assert proc.stats()["dropped_deadline"] == 1
    proc.shutdown()


def test_no_post_happens_after_exporter_shutdown():
    transport = _ScriptedTransport([TransportResponse(503)] * 5)
    exporter = TraigentOTLPExporter(
        transport, backoff_base=0.0, max_attempts=5, sleep=lambda s: None
    )
    exporter.shutdown()
    outcome = exporter.export_batch(_span_batch(1))
    assert transport.bodies == []
    assert outcome.exported == 0
    assert outcome.dropped_deadline == 1


def test_shutdown_during_retry_backoff_stops_further_posts():
    transport = _ScriptedTransport([TransportResponse(503)] * 5)
    holder: dict[str, TraigentOTLPExporter] = {}

    def sleep(_s):
        holder["e"].shutdown()  # shutdown lands while the loop is backing off

    exporter = TraigentOTLPExporter(
        transport, backoff_base=0.5, max_attempts=5, sleep=sleep, rng=lambda: 1.0
    )
    holder["e"] = exporter
    outcome = exporter.export_batch(_span_batch(1))
    assert len(transport.bodies) == 1
    assert outcome.dropped_deadline == 1


def test_oversized_single_span_is_dropped_before_transport():
    transport = _ScriptedTransport([])
    fits = TraigentOTLPExporter(transport, max_batch_bytes=4_000)
    assert fits.export_batch(_span_batch(1)).exported == 1  # control: it can send
    assert transport.bodies and len(transport.bodies[0]) <= 4_000
    transport.bodies.clear()
    tiny = TraigentOTLPExporter(transport, max_batch_bytes=10)
    outcome = tiny.export_batch(_span_batch(1))
    assert transport.bodies == []
    assert outcome.exported == 0
    assert outcome.dropped_non_retryable == 1


def test_no_request_exceeds_the_cap_for_a_mixed_batch():
    transport = _ScriptedTransport([])
    cap = 600
    exporter = TraigentOTLPExporter(transport, max_batch_bytes=cap)
    exporter.export_batch(_span_batch(6))
    assert transport.bodies
    assert all(len(b) <= cap for b in transport.bodies)


def test_malformed_nonempty_ack_is_a_protocol_failure_not_success():
    transport = _ScriptedTransport([TransportResponse(200, body=b"\xff\xff\xff\xff")])
    exporter = TraigentOTLPExporter(transport)
    outcome = exporter.export_batch(_span_batch(2))
    assert outcome.exported == 0
    assert outcome.protocol_failures == 2


def test_empty_ack_body_is_still_success():
    transport = _ScriptedTransport([TransportResponse(200, body=b"")])
    exporter = TraigentOTLPExporter(transport)
    outcome = exporter.export_batch(_span_batch(2))
    assert outcome.exported == 2
    assert outcome.protocol_failures == 0
