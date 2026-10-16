"""Wire canaries: credentials glued to identifiers (``sess_AKIA...``) never leave.

The shared redactor's lookbehind ignores a credential preceded by ``_``; the
OTel credential predicate must not.  It must also be credential-only: a bare
numeric id is not a credential.  One channel per test so a missing check in
one channel cannot be masked by another.
"""

from __future__ import annotations

import pytest

import traigent.observability.otel as otel
from traigent.observability.otel.policy import looks_sensitive

pytestmark = pytest.mark.backend_online

AWS = "AKIAZQ4XW7PLM2NB8VC3"
WRAPPED = f"sess_{AWS}"
MODES = ("metadata", "redacted", "record")
CHANNELS = (
    "attribute_value",
    "service_name",
    "scope_name",
    "scope_version",
    "exception_type",
    "string_array",
)


def _raw(collector) -> bytes:
    return b"".join(collector.raw_bodies) + b"".join(collector.raw_wire)


def _init(collector, mode, **kw):
    return otel.init(
        api_key="k",
        endpoint=collector.base_url,
        content_mode=mode,
        exit_flush=False,
        schedule_delay_s=0.05,
        **kw,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("channel", CHANNELS)
def test_wrapped_credential_never_reaches_the_wire(collector, channel, mode):
    service = WRAPPED if channel == "service_name" else "checkout-service"
    handle = _init(collector, mode, service_name=service)
    if channel == "scope_name":
        handle.provider.get_tracer(WRAPPED, "1.0.0").start_span("s").end()
    elif channel == "scope_version":
        handle.provider.get_tracer("canary_scope", f"v_{AWS}").start_span("s").end()
    elif channel == "attribute_value":
        with otel.observe("c", session_id=WRAPPED, user_id=WRAPPED):
            pass
    elif channel == "exception_type":
        with otel.observe("c") as span:
            span.add_event("exception", {"exception.type": WRAPPED})
    elif channel == "string_array":
        with otel.observe("c") as span:
            span.set_attribute("gen_ai.response.finish_reasons", ["stop", WRAPPED])
    else:
        with otel.observe("c"):
            pass
    assert otel.flush(5).flushed
    assert collector.spans(), "nothing was exported"
    raw = _raw(collector)
    assert AWS.encode() not in raw, (channel, mode)


@pytest.mark.parametrize("mode", MODES)
def test_numeric_ids_and_benign_scope_survive(collector, mode):
    handle = _init(collector, mode, service_name="checkout-service")
    handle.provider.get_tracer("benign_scope.sub", "2.3.4-rc1").start_span("s").end()
    with otel.observe("control", session_id="sess-1234", user_id="123456789"):
        pass
    assert otel.flush(5).flushed
    attrs = [collector.attrs(span) for _, _, span in collector.spans()]
    assert any(a.get("user.id") == "123456789" for a in attrs), attrs
    raw = _raw(collector)
    assert b"benign_scope.sub" in raw
    assert b"2.3.4-rc1" in raw


@pytest.mark.parametrize("mode", MODES)
def test_numeric_service_version_survives(collector, mode):
    _init(collector, mode, service_name="checkout-service", release="123456789")
    with otel.observe("control"):
        pass
    assert otel.flush(5).flushed
    assert b"123456789" in _raw(collector)


@pytest.mark.parametrize(
    "text,expected",
    [
        (WRAPPED, True),
        (f"x-{AWS}", True),
        (AWS, True),
        ("123456789", False),
        ("123-45-6789", False),
        ("user@example.com", False),
        ("sess-1234", False),
    ],
)
def test_looks_sensitive_is_credential_only(text, expected):
    assert looks_sensitive(text) is expected
