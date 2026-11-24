"""The telemetry opt-out is honoured after credential validation and at the HTTP send (SDK-129 final r4)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from traigent.observability.otel import exporter as mod
from traigent.observability.otel.transport import (
    TransportResponse,
    TransportSuppressed,
    UrllibTransport,
)


def _spans() -> list[ReadableSpan]:
    return [ReadableSpan(name="s")]


def test_opt_out_switched_on_by_credential_validation_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(mod, "is_backend_offline", lambda: False)
    monkeypatch.setattr(mod.oo_egress, "is_active", lambda scope=None: False)

    def validate(*_a, **_k) -> bool:
        monkeypatch.setenv("TRAIGENT_DISABLE_TELEMETRY", "true")
        return True

    monkeypatch.setattr(mod.oo_egress, "ensure_trace_egress_allowed", validate)
    transport = MagicMock()
    transport.post.return_value = TransportResponse(status=200)
    exporter = mod.TraigentOTLPExporter(transport)
    exporter.egress_credential = ("key", "https://example.invalid")  # type: ignore[attr-defined]

    outcome = mod.ExportOutcome()
    exporter._send_with_retry(  # noqa: SLF001
        _spans(), b"x", None, outcome, may_split=False
    )

    assert transport.post.call_count == 0
    assert outcome.dropped_offline == 1


def test_stock_http_transport_has_a_live_opt_out_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = UrllibTransport("https://example.invalid/v1/traces", {})
    transport._opener = MagicMock()  # noqa: SLF001
    monkeypatch.setenv("TRAIGENT_DISABLE_TELEMETRY", "true")
    with pytest.raises(TransportSuppressed):
        transport.post(b"x", timeout=1.0)
    assert transport._opener.open.call_count == 0  # noqa: SLF001


def test_exporter_counts_a_transport_suppressed_batch_as_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAIGENT_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(mod, "is_backend_offline", lambda: False)
    monkeypatch.setattr(mod.oo_egress, "is_active", lambda scope=None: False)
    transport = MagicMock()
    transport.post.side_effect = TransportSuppressed("telemetry_disabled")
    exporter = mod.TraigentOTLPExporter(transport)
    outcome = mod.ExportOutcome()
    exporter._send_with_retry(  # noqa: SLF001
        _spans(), b"x", None, outcome, may_split=False
    )
    assert transport.post.call_count == 1
    assert outcome.dropped_offline == 1
