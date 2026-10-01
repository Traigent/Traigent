"""Wire canaries: credential shapes in ALLOWLISTED channels never leave.

Allowlisted string attributes (``session.id``, ``user.id``), resource values
and the instrumentation scope name are bounded but were copied verbatim; a
credential-shaped value there reached the OTLP request in every mode.  Each
shared-fixture credential goes through the real provider, processor and
exporter and must be absent from the decoded request and the raw bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import traigent.observability.otel as otel

pytestmark = pytest.mark.backend_online

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "security"
    / "redactor_credential_shapes.json"
)
POSITIVES = json.loads(FIXTURE.read_text(encoding="utf-8"))["positives"]
MODES = ("metadata", "redacted", "record")


def _raw(collector) -> bytes:
    return b"".join(collector.raw_bodies) + b"".join(collector.raw_wire)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", POSITIVES, ids=lambda c: c["id"])
def test_credential_in_allowlisted_channels_never_reaches_the_wire(
    collector, case, mode
):
    secret = "".join(case["parts"])
    leak = case["parts"][case["leak_part"]]
    handle = otel.init(
        api_key="k",
        endpoint=collector.base_url,
        content_mode=mode,
        service_name=secret,  # resource attribute (service.name)
        exit_flush=False,
        schedule_delay_s=0.05,
    )
    span = handle.provider.get_tracer(secret, secret[:20]).start_span("scope-canary")
    span.end()
    with otel.observe("credential-canary", session_id=secret, user_id=secret):
        pass
    assert otel.flush(5).flushed
    assert collector.spans(), "nothing was exported"
    raw = _raw(collector)
    for needle in (secret, leak):
        assert needle.encode() not in raw, (case["id"], mode)


@pytest.mark.parametrize("mode", MODES)
def test_ordinary_session_and_user_ids_survive(collector, mode):
    otel.init(
        api_key="k",
        endpoint=collector.base_url,
        content_mode=mode,
        service_name="checkout-service",
        exit_flush=False,
        schedule_delay_s=0.05,
    )
    with otel.observe("control", session_id="sess-1234", user_id="user-42"):
        pass
    assert otel.flush(5).flushed
    attrs = [collector.attrs(span) for _, _, span in collector.spans()]
    assert any(
        a.get("session.id") == "sess-1234" and a.get("user.id") == "user-42"
        for a in attrs
    ), attrs
    raw = _raw(collector)
    assert b"checkout-service" in raw
