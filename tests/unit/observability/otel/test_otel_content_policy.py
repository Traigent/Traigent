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
