"""Transport gate: classification counts, report vs refuse, and the hook points."""

from __future__ import annotations

import json
import logging

import pytest

from traigent.labels import transport_gate as gate
from traigent.labels.content_classes import get_store
from traigent.labels.transport_gate import TransportGateRefusal, inspect_outgoing

CANARY = "Alice_has_diabetes"
URL = "https://backend.example.test/api/v1/sessions"


def _session_body() -> bytes:
    return json.dumps(
        {
            "function_name": CANARY,
            "configuration_space": {CANARY: [CANARY]},
            "objectives": ["accuracy"],
            "max_trials": 3,
        }
    ).encode()


@pytest.fixture(autouse=True)
def _reset():
    gate.set_mode(None)
    gate.clear_refusals()
    yield
    gate.uninstall_transport_gate()
    gate.set_mode(None)
    gate.clear_refusals()


def test_route_table_covers_the_body_carrying_routes() -> None:
    store = get_store()
    assert store.find_route("POST", "/api/v1/sessions")
    assert store.find_route("POST", "/api/v1/sessions/abc-1/results")
    assert store.find_route("POST", "/api/v1beta/projects/p1/sessions/s1/finalize")
    assert store.find_route("GET", "/api/v1/sessions") is None
    assert store.find_route("POST", "/other") is None


def test_report_mode_counts_without_blocking_or_leaking(caplog, monkeypatch) -> None:
    gate.set_mode("report")
    with caplog.at_level(logging.DEBUG):
        counts = inspect_outgoing("POST", URL, _session_body())
    assert counts and counts.get("legacy_content", 0) > 0
    assert gate.report_totals()["legacy_content"] >= counts["legacy_content"]
    assert CANARY not in caplog.text
    assert gate.refusal_log() == []


def test_report_mode_never_raises_on_garbage() -> None:
    gate.set_mode("report")
    assert inspect_outgoing("POST", URL, b"not json") is None
    assert inspect_outgoing("GET", URL, b"{}") is None
    assert inspect_outgoing("POST", URL, None) is None
    assert inspect_outgoing("POST", "https://x.test/other", b"{}") is None


def test_refuse_mode_raises_with_counts_only() -> None:
    gate.set_mode("refuse")
    with pytest.raises(TransportGateRefusal) as err:
        inspect_outgoing("POST", URL, _session_body())
    assert err.value.counts["legacy_content"] > 0
    assert CANARY not in str(err.value)
    assert gate.refusal_log()[0][0] == "/api/v1/sessions"
    with pytest.raises(
        TransportGateRefusal
    ):  # unparseable body on a gated route fails closed
        inspect_outgoing("POST", URL, b"not json")


def test_off_mode_does_nothing() -> None:
    gate.set_mode("off")
    assert inspect_outgoing("POST", URL, _session_body()) is None
    assert gate.install_transport_gate() is False


def test_aiohttp_hook_refuses_before_send() -> None:
    aiohttp = pytest.importorskip("aiohttp")
    sent: list[str] = []
    session_cls = aiohttp.ClientSession
    original = session_cls._request

    async def fake(self, method, url, **kw):  # noqa: ANN001
        sent.append(str(url))
        raise RuntimeError("would send")

    session_cls._request = fake
    try:
        gate.set_mode("refuse")
        assert gate.install_transport_gate()
        s = session_cls.__new__(session_cls)
        with pytest.raises(TransportGateRefusal):
            s.post(URL, json={"function_name": CANARY})
        assert sent == []
    finally:
        session_cls._request = original


def test_requests_hook_refuses_before_send(monkeypatch) -> None:
    requests = pytest.importorskip("requests")
    sent: list[str] = []

    def fake_send(self, request, **kw):  # noqa: ANN001
        sent.append(request.url)
        resp = requests.Response()
        resp.status_code = 200
        resp.request = request
        return resp

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
    gate.set_mode("refuse")
    gate.install_transport_gate()
    with pytest.raises(TransportGateRefusal):
        requests.post(URL, json={"function_name": CANARY}, timeout=1)
    assert sent == []
    gate.set_mode("report")
    requests.post(URL, json={"function_name": CANARY}, timeout=1)
    assert sent == [URL]  # report mode lets the request through unchanged
