"""Record-mode wire canaries for well-known credential shapes.

An exception text carrying each credential shape goes through the real
provider, processor and exporter; the collector stub decodes the OTLP request
and the credential must be absent from every channel and from the raw bytes.
The same shared fixture drives the JS SDK.
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


def _init(collector):
    return otel.init(
        api_key="k",
        endpoint=collector.base_url,
        content_mode="record",
        exit_flush=False,
        schedule_delay_s=0.05,
    )


@pytest.mark.parametrize("case", POSITIVES, ids=lambda c: c["id"])
def test_exception_text_with_credential_never_reaches_the_wire(collector, case):
    secret = "".join(case["parts"])
    leak = case["parts"][case["leak_part"]]
    text = case["context"].replace("{SECRET}", secret)
    _init(collector)
    with pytest.raises(RuntimeError):
        with otel.observe("credential-canary") as span:
            span.add_event("note", {"detail": text})
            raise RuntimeError(text)
    assert otel.flush(5).flushed
    raw = b"".join(collector.raw_bodies) + b"".join(collector.raw_wire)
    assert collector.spans(), "nothing was exported"
    for needle in (secret, leak):
        assert needle.encode() not in raw, case["id"]
    # control: the surrounding text still flows in record mode
    assert b"REDACTED" in raw, case["id"]
