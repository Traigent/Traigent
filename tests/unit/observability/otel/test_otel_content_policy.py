"""Content policy: the egress-authoritative span rebuild."""

from __future__ import annotations

import pytest
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.trace import Link, SpanContext, TraceFlags
from opentelemetry.trace.status import Status, StatusCode

from traigent.observability.otel import contract as C
from traigent.observability.otel.policy import ContentPolicy

CANARY = "CANARY-7f3a91c2-do-not-egress"  # keep in sync with conftest


def _emit_hostile_span(provider, canary: str):
    """A span an instrumentor forced into 'capture everything' would produce."""
    tracer = provider.get_tracer(f"lib-{canary}", "1.0.0")
    linked = SpanContext(1, 2, False, TraceFlags(1))
    with tracer.start_as_current_span(
        f"chat {canary}", links=[Link(linked, {"link.note": canary})]
    ) as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", "model-x")
        span.set_attribute("gen_ai.usage.input_tokens", 11)
        span.set_attribute("gen_ai.usage.output_tokens", 7)
        span.set_attribute("gen_ai.input.messages", canary)
        span.set_attribute("gen_ai.system_instructions", canary)
        span.set_attribute("gen_ai.tool.call.arguments", canary)
        span.set_attribute("input.value", canary)
        span.set_attribute("some.unknown.attr", canary)
        span.set_attribute("server.evil", canary)  # wildcard-ish prefix probe
        span.set_attribute("traigent.sneaky", canary)  # prefix probe
        span.set_attribute("traigent.input", canary)
        span.set_attribute("traigent.metadata.k", canary)
        span.set_attribute("gen_ai.request.model_typo", canary)
        span.set_attribute("gen_ai.request.temperature", canary)  # wrong type
        span.add_event("gen_ai.user.message", {"content": canary})
        span.add_event(canary, {"x": canary})
        span.add_event(
            "exception",
            {
                "exception.type": "ValueError",
                "exception.message": canary,
                "exception.stacktrace": canary,
            },
        )
        span.set_status(Status(StatusCode.ERROR, canary))
    return provider


def _wire(provider, exporter, policy: ContentPolicy) -> bytes:
    spans = exporter.get_finished_spans()
    return encode_spans([policy.sanitize(s) for s in spans]).SerializeToString()


def test_metadata_mode_drops_canary_from_every_field(memory_provider):
    provider, exporter = memory_provider
    _emit_hostile_span(provider, CANARY)
    body = _wire(provider, exporter, ContentPolicy("metadata"))
    assert CANARY.encode() not in body
    # structural facts the product needs survive
    assert b"model-x" in body
    assert b"gen_ai.usage.input_tokens" in body
    assert b"ValueError" in body


def test_metadata_mode_keeps_only_allowlisted_and_counts_drops(memory_provider):
    provider, exporter = memory_provider
    _emit_hostile_span(provider, CANARY)
    (span,) = exporter.get_finished_spans()
    out = ContentPolicy("metadata").sanitize(span)
    assert set(out.attributes) <= set(C.ATTRIBUTE_ALLOWLIST) | {"error.type"}
    assert out.attributes[C.ATTR_DROPPED_ATTRS] >= 10
    assert out.attributes["gen_ai.request.model"] == "model-x"
    assert "gen_ai.request.temperature" not in out.attributes  # wrong type dropped
    assert out.status.status_code is StatusCode.ERROR
    assert out.status.description == "ValueError"
    assert [e.name for e in out.events] == ["exception"]
    assert dict(out.events[0].attributes) == {"exception.type": "ValueError"}
    assert all(not link.attributes for link in out.links)
    assert out.name == "chat model-x"  # unsafe third-party name replaced


def test_resource_and_scope_are_rebuilt_from_allowlist(memory_provider):
    provider, exporter = memory_provider
    _emit_hostile_span(provider, CANARY)
    (span,) = exporter.get_finished_spans()
    out = ContentPolicy("metadata").sanitize(span)
    assert set(out.resource.attributes) <= set(C.RESOURCE_ALLOWLIST)
    assert out.resource.attributes["service.name"] == "svc"
    assert out.resource.attributes[C.CONTENT_MODE_ATTRIBUTE] == "metadata"
    assert out.instrumentation_scope.name == "unknown"  # canary in scope name


def test_redacted_mode_keeps_content_keys_with_placeholder(memory_provider):
    provider, exporter = memory_provider
    _emit_hostile_span(provider, CANARY)
    body = _wire(provider, exporter, ContentPolicy("redacted"))
    assert CANARY.encode() not in body
    (span,) = exporter.get_finished_spans()
    out = ContentPolicy("redacted").sanitize(span)
    assert out.attributes["gen_ai.input.messages"] == C.REDACTED_PLACEHOLDER
    assert out.attributes["traigent.metadata.k"] == C.REDACTED_PLACEHOLDER
    assert "some.unknown.attr" not in out.attributes


def test_record_mode_sends_content(memory_provider):
    """Control: the canary check is capable of failing when content is allowed."""
    provider, exporter = memory_provider
    _emit_hostile_span(provider, CANARY)
    body = _wire(provider, exporter, ContentPolicy("record"))
    assert CANARY.encode() in body


def test_record_mode_scrubs_secrets_and_caps_size(memory_provider):
    provider, exporter = memory_provider
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("s") as span:
        span.set_attribute("input.value", "mail me at alice@example.com")
        span.set_attribute("output.value", "x" * (C.MAX_RECORD_ATTR_BYTES * 2))
        span.set_attribute("api_key", "sk-abcdefghijklmnopqrstuvwx")
    (s,) = exporter.get_finished_spans()
    out = ContentPolicy("record").sanitize(s)
    assert "alice@example.com" not in out.attributes["input.value"]
    assert len(out.attributes["output.value"].encode()) <= C.MAX_RECORD_ATTR_BYTES
    assert "api_key" not in out.attributes


def test_per_span_override_can_only_tighten(memory_provider):
    provider, exporter = memory_provider
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("a") as span:
        span.set_attribute(C.CONTENT_MODE_ATTRIBUTE, "metadata")
        span.set_attribute("input.value", CANARY)
    with tracer.start_as_current_span("b") as span:
        span.set_attribute(C.CONTENT_MODE_ATTRIBUTE, "record")  # loosen attempt
        span.set_attribute("input.value", CANARY)
    a, b = exporter.get_finished_spans()
    pa = ContentPolicy("record").sanitize(a)
    assert "input.value" not in pa.attributes
    assert pa.attributes[C.CONTENT_MODE_ATTRIBUTE] == "metadata"
    pb = ContentPolicy("metadata").sanitize(b)
    assert "input.value" not in pb.attributes
    assert C.CONTENT_MODE_ATTRIBUTE not in pb.attributes


def test_allowlist_has_no_wildcards_and_only_bounded_specs():
    for key, spec in {**C.ATTRIBUTE_ALLOWLIST, **C.RESOURCE_ALLOWLIST}.items():
        assert "*" not in key and not key.endswith("."), key
        assert spec.kind in {"str", "int", "float", "bool", "str_seq", "enum"}
        if spec.kind in {"str", "str_seq"}:
            assert 0 < spec.max_len <= 255, key
        if spec.kind == "enum":
            assert spec.choices, key
    assert "traigent.input" not in C.ATTRIBUTE_ALLOWLIST
    assert "traigent.output" not in C.ATTRIBUTE_ALLOWLIST


@pytest.mark.parametrize(
    "value",
    ["x" * 129, "line\nbreak", "", 5, True, ["a"]],
)
def test_string_bounds_are_enforced(memory_provider, value):
    provider, exporter = memory_provider
    with provider.get_tracer("t").start_as_current_span("s") as span:
        span.set_attribute("gen_ai.request.model", value)
    (s,) = exporter.get_finished_spans()
    out = ContentPolicy("metadata").sanitize(s)
    assert "gen_ai.request.model" not in out.attributes


def test_negative_control_denylist_policy_leaks(memory_provider, monkeypatch):
    """Mutation: a denylist-style policy leaks the unknown-key canary.

    Proves the canary assertion above would catch an allowlist regression.
    """
    provider, exporter = memory_provider
    _emit_hostile_span(provider, CANARY)

    class Denylist(ContentPolicy):
        def _attrs(self, attrs, mode):  # deny only the well-known keys
            deny = C.CONTENT_ATTRIBUTE_KEYS
            return {k: v for k, v in (attrs or {}).items() if k not in deny}, 0

    body = _wire(provider, exporter, Denylist("metadata"))
    assert CANARY.encode() in body


# -- span names are not a content channel; tracestate is not forwarded ----------
NAME_CANARY = f"what is {CANARY} for user@example.com"
VENDOR_TS = f"vendor={CANARY},other=v"


def _name_and_tracestate_span(provider, name: str, scope: str):
    from opentelemetry.trace import TraceState

    ts = TraceState.from_header([VENDOR_TS])
    linked = SpanContext(1, 2, False, TraceFlags(1), trace_state=ts)
    with provider.get_tracer(scope).start_as_current_span(
        name, links=[Link(linked)]
    ) as span:
        span.set_attribute(C.ATTR_OBSERVATION_TYPE, "tool")


def _sanitized_wire(exporter, policy) -> bytes:
    spans = exporter.get_finished_spans()
    return encode_spans([policy.sanitize(s) for s in spans]).SerializeToString()


@pytest.mark.parametrize("mode", ["metadata", "redacted"])
def test_default_scope_name_is_not_a_content_channel(memory_provider, mode):
    provider, exporter = memory_provider
    _name_and_tracestate_span(provider, NAME_CANARY, C.TRAIGENT_SCOPE_NAME)
    (s,) = exporter.get_finished_spans()
    out = ContentPolicy(mode).sanitize(s)
    assert out.name == "tool"  # replaced by the observation type
    assert out.attributes[C.ATTR_DROPPED_ATTRS] >= 1  # replacement is counted
    assert CANARY.encode() not in _sanitized_wire(exporter, ContentPolicy(mode))


def test_identifier_names_still_pass_in_default_scope(memory_provider):
    provider, exporter = memory_provider
    _name_and_tracestate_span(provider, "my_tool.step-1", C.TRAIGENT_SCOPE_NAME)
    (s,) = exporter.get_finished_spans()
    assert ContentPolicy("metadata").sanitize(s).name == "my_tool.step-1"


@pytest.mark.parametrize("mode", ["metadata", "redacted", "record"])
def test_tracestate_is_dropped_on_span_link_and_parent(memory_provider, mode):
    from opentelemetry.trace import TraceState

    provider, exporter = memory_provider
    _name_and_tracestate_span(provider, "ok", C.TRAIGENT_SCOPE_NAME)
    (s,) = exporter.get_finished_spans()
    out = ContentPolicy(mode).sanitize(s)
    assert len(out.links[0].context.trace_state) == 0
    assert len(out.context.trace_state) == 0
    body = _sanitized_wire(exporter, ContentPolicy(mode))
    assert CANARY.encode() not in body
    # link identity survives (only tracestate is rebuilt)
    assert out.links[0].context.trace_id == 1 and out.links[0].context.span_id == 2

    # the span's OWN tracestate and a parent's, via a remote parent context
    ts = TraceState.from_header([VENDOR_TS])
    remote = SpanContext(0xAB, 0xCD, True, TraceFlags(1), trace_state=ts)
    import opentelemetry.trace as ot

    ctx = ot.set_span_in_context(ot.NonRecordingSpan(remote))
    exporter.clear()
    with provider.get_tracer("t").start_as_current_span("child", context=ctx):
        pass
    (child,) = exporter.get_finished_spans()
    assert CANARY in str(child.context.trace_state) + str(child.parent.trace_state)
    out2 = ContentPolicy(mode).sanitize(child)
    assert len(out2.parent.trace_state) == 0 and len(out2.context.trace_state) == 0
    assert CANARY.encode() not in _sanitized_wire(exporter, ContentPolicy(mode))
    assert out2.context.trace_id == child.context.trace_id  # ids preserved


def test_negative_control_name_scope_exemption_leaks(memory_provider):
    provider, exporter = memory_provider
    _name_and_tracestate_span(provider, NAME_CANARY, C.TRAIGENT_SCOPE_NAME)

    class Exempt(ContentPolicy):
        @staticmethod
        def _name(raw, attrs, scope_name, mode):
            return raw  # restored scope exemption

    assert CANARY.encode() in _sanitized_wire(exporter, Exempt("metadata"))


def test_negative_control_keeping_tracestate_leaks(memory_provider, monkeypatch):
    """The installed Python encoder serialises only the span's OWN tracestate
    (links' tracestate never reaches the wire), so drive it via a remote parent."""
    import opentelemetry.trace as ot
    from opentelemetry.trace import TraceState

    import traigent.observability.otel.policy as pol

    provider, exporter = memory_provider
    remote = SpanContext(
        0xAB, 0xCD, True, TraceFlags(1), trace_state=TraceState.from_header([VENDOR_TS])
    )
    ctx = ot.set_span_in_context(ot.NonRecordingSpan(remote))
    with provider.get_tracer("t").start_as_current_span("ok", context=ctx):
        pass
    assert CANARY.encode() not in _sanitized_wire(exporter, ContentPolicy("metadata"))
    monkeypatch.setattr(pol, "_strip_trace_state", lambda c: c)
    assert CANARY.encode() in _sanitized_wire(exporter, ContentPolicy("metadata"))
