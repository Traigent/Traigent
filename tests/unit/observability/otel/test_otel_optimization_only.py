"""optimization_only: no OTLP span ever reaches the collector (init, record, flush, exit)."""

from __future__ import annotations

import pytest

import traigent.observability.otel as otel
from traigent.utils import optimization_only_egress as oo

pytestmark = pytest.mark.backend_online


#: Hermetic credential: never rely on the ambient ``TRAIGENT_API_KEY``.
FAKE_API_KEY = "uk_" + "o" * 43  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _mode(monkeypatch):
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    oo.reset_latch()
    yield
    oo.reset_latch()
    otel.shutdown()


def _init(collector):
    return otel.init(
        endpoint=collector.base_url,
        api_key=FAKE_API_KEY,
        schedule_delay_s=0.05,
        exit_flush=False,
    )


def test_mode_off_exports_to_collector(collector) -> None:
    handle = _init(collector)
    with handle.tracer.start_as_current_span("x"):
        pass
    assert otel.flush(5).flushed
    assert collector.requests


def test_mode_on_at_init_builds_no_pipeline_and_sends_nothing(
    collector, monkeypatch
) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    handle = _init(collector)
    with handle.tracer.start_as_current_span("x"):
        pass
    otel.flush(1)
    otel.shutdown()
    assert collector.requests == []


def test_mode_latched_after_init_drops_at_record_and_send_time(collector) -> None:
    handle = _init(collector)
    # e.g. a closed 403 arrived from the session API for this credential
    oo.latch_on(handle.exporter.oo_scope)
    with handle.tracer.start_as_current_span("x"):
        pass
    otel.flush(1)
    otel.shutdown()
    assert collector.requests == []
