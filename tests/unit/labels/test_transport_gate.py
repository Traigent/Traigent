"""Transport gate: scoping, origin match, report/refuse semantics, every send path."""

from __future__ import annotations

import json
import logging
import sys

import pytest

from traigent.labels import transport_gate as gate
from traigent.labels.content_classes import get_table
from traigent.labels.transport_gate import TransportGateRefusal, inspect_outgoing

CANARY = "Alice_has_diabetes"
BACKEND = "https://backend.example.test"
URL = f"{BACKEND}/api/v1/sessions"
CUSTOMER_URL = "https://customer.example/api/v1/sessions"


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
    gate.set_allowed_origins([BACKEND])
    gate.clear_refusals()
    yield
    gate.set_mode(None)
    gate.set_allowed_origins(None)
    gate.clear_refusals()


# --- table and classification --------------------------------------------------------


def test_route_table_covers_routes_with_and_without_bodies() -> None:
    t = get_table()
    assert t.find("POST", "/api/v1/sessions")
    assert t.find("POST", "/api/v1/sessions/abc-1/results")
    assert t.find("POST", "/api/v1beta/projects/p1/sessions/s1/finalize")
    assert t.find("GET", "/api/v1/sessions/s1/identity-binding")  # no body route
    assert t.find("GET", "/api/v1/sessions") is None
    assert t.find("POST", "/other") is None


def test_query_and_path_parameters_are_classified() -> None:
    gate.set_mode("report")
    c = inspect_outgoing(
        "GET",
        f"{BACKEND}/api/v1/optimization/recommendations?agent_type=x&bogus=1",
        None,
    )
    assert c and c.get("legacy_content", 0) >= 1 and c.get("unclassified", 0) >= 1
    c = inspect_outgoing(
        "GET", f"{BACKEND}/api/v1/sessions/{CANARY}/identity-binding", None
    )
    assert (
        c and c.get("legacy_content", 0) + c.get("unclassified", 0) >= 1
    )  # declared legacy


def test_refuse_mode_refuses_a_content_path_param() -> None:
    gate.set_mode("refuse")
    with pytest.raises(TransportGateRefusal):
        inspect_outgoing(
            "GET", f"{BACKEND}/api/v1/optimization/recommendations?metric=m", None
        )


# --- report mode ---------------------------------------------------------------------


def test_report_counts_without_blocking_or_leaking(caplog) -> None:
    gate.set_mode("report")
    with caplog.at_level(logging.DEBUG):
        counts = inspect_outgoing("POST", URL, _session_body())
    assert counts and counts.get("legacy_content", 0) > 0
    assert gate.report_totals()["legacy_content"] >= counts["legacy_content"]
    assert CANARY not in caplog.text and gate.refusal_log() == []


def test_report_swallows_gate_failures_into_a_counter(monkeypatch) -> None:
    gate.set_mode("report")

    def boom(*a, **k):
        raise RuntimeError("gate bug")

    monkeypatch.setattr(gate, "classify_request", boom)
    before = gate.gate_errors().get("RuntimeError", 0)
    assert inspect_outgoing("POST", URL, _session_body()) is None
    assert gate.gate_errors()["RuntimeError"] == before + 1
    assert inspect_outgoing("POST", URL, b"not json") is None  # never raises in report
    gate.set_mode("refuse")
    with pytest.raises(TransportGateRefusal):
        inspect_outgoing("POST", URL, b"not json")


def test_off_mode_does_nothing() -> None:
    gate.set_mode("off")
    assert inspect_outgoing("POST", URL, _session_body()) is None

    class Plain:
        pass

    p = Plain()
    assert gate.gate_session(p) is p


# --- scoping: other origins are never inspected ---------------------------------------


def test_other_hosts_and_origins_are_untouched() -> None:
    gate.set_mode("refuse")
    assert inspect_outgoing("POST", CUSTOMER_URL, _session_body()) is None
    assert (
        inspect_outgoing(
            "POST", "http://backend.example.test/api/v1/sessions", _session_body()
        )
        is None
    )
    assert (
        inspect_outgoing("POST", f"{BACKEND}:8443/api/v1/sessions", _session_body())
        is None
    )
    with pytest.raises(TransportGateRefusal):
        inspect_outgoing("POST", URL, _session_body())


def test_importing_the_sdk_patches_nothing_global() -> None:
    import aiohttp
    import httpx
    import requests

    import traigent.cloud.backend_client  # noqa: F401

    assert not getattr(aiohttp.ClientSession, "_traigent_gated", False)
    assert not getattr(requests.Session, "_traigent_gated", False)
    assert not getattr(httpx.Client, "_traigent_gated", False)
    assert not getattr(httpx.AsyncClient, "_traigent_gated", False)


def test_unrelated_requests_are_untouched_in_refuse_mode(monkeypatch) -> None:
    requests = pytest.importorskip("requests")
    sent: list[str] = []

    def fake_send(self, request, **kw):
        sent.append(request.url)
        resp = requests.Response()
        resp.status_code = 200
        resp.request = request
        return resp

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
    gate.set_mode("refuse")
    import traigent.cloud.backend_client  # noqa: F401  (SDK imported, nothing patched)

    requests.post(CUSTOMER_URL, json={"function_name": CANARY}, timeout=1)
    assert sent == [CUSTOMER_URL]


# --- every send path refuses before bytes leave ----------------------------------------


@pytest.mark.parametrize("verb", ["post", "request"])
async def test_aiohttp_every_entry_point_refuses_before_send(verb) -> None:
    aiohttp = pytest.importorskip("aiohttp")
    sent: list[str] = []
    cls = aiohttp.ClientSession
    original = cls._request

    async def fake(self, method, url, **kw):
        sent.append(str(url))
        raise RuntimeError("would send")

    cls._request = fake
    try:
        gate.set_mode("refuse")
        async with gate.gate_session(aiohttp.ClientSession()) as s:
            args = ("POST", URL) if verb == "request" else (URL,)
            with pytest.raises(TransportGateRefusal):
                async with getattr(s, verb)(*args, json={"function_name": CANARY}):
                    pass
            with pytest.raises(TransportGateRefusal):  # data= bytes path
                async with getattr(s, verb)(*args, data=_session_body()):
                    pass
            assert sent == []
            with pytest.raises(
                RuntimeError, match="would send"
            ):  # other host: untouched
                async with s.post(CUSTOMER_URL, json={"function_name": CANARY}):
                    pass
            assert sent == [CUSTOMER_URL]
    finally:
        cls._request = original


async def test_aiohttp_report_serializes_once_and_never_fails_the_request() -> None:
    aiohttp = pytest.importorskip("aiohttp")
    calls = {"n": 0}

    def once_only(obj):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("single-use serializer reused")
        return json.dumps(obj)

    seen: list[object] = []
    cls = aiohttp.ClientSession
    original = cls._request

    async def fake(self, method, url, **kw):
        seen.append(kw.get("data"))
        return "ok"

    cls._request = fake
    try:
        gate.set_mode("report")
        async with gate.gate_session(
            aiohttp.ClientSession(json_serialize=once_only)
        ) as s:
            assert await s._request("POST", URL, json={"function_name": CANARY}) == "ok"
        assert calls["n"] == 1 and seen and getattr(seen[0], "_value", b"")
        assert gate.report_totals().get("legacy_content", 0) > 0
    finally:
        cls._request = original


def test_requests_session_send_refuses_before_bytes_leave(monkeypatch) -> None:
    requests = pytest.importorskip("requests")
    sent: list[str] = []

    def fake_send(self, request, **kw):
        sent.append(request.url)
        resp = requests.Response()
        resp.status_code = 200
        resp.request = request
        return resp

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
    s = gate.gate_session(requests.Session())
    gate.set_mode("refuse")
    for call in (
        lambda: s.post(URL, json={"function_name": CANARY}, timeout=1),
        lambda: s.send(requests.Request("POST", URL, data=_session_body()).prepare()),
    ):
        with pytest.raises(TransportGateRefusal):
            call()
    assert sent == []
    gate.set_mode("report")
    s.post(URL, json={"function_name": CANARY}, timeout=1)
    assert sent == [URL]  # report mode passes the request through unchanged


def test_requests_module_level_preflight(monkeypatch) -> None:
    pytest.importorskip("requests")
    gate.set_mode("refuse")
    with pytest.raises(TransportGateRefusal):
        gate.preflight_requests("POST", URL, json={"function_name": CANARY})
    gate.preflight_requests(
        "POST", CUSTOMER_URL, json={"function_name": CANARY}
    )  # untouched
    gate.set_mode("report")
    gate.preflight_requests("POST", URL, json={"function_name": CANARY})


def test_httpx_send_request_and_stream_refuse_before_send() -> None:
    httpx = pytest.importorskip("httpx")
    sent: list[str] = []

    def handler(request):
        sent.append(str(request.url))
        return httpx.Response(200, json={})

    client = gate.gate_session(httpx.Client(transport=httpx.MockTransport(handler)))
    gate.set_mode("refuse")
    with pytest.raises(TransportGateRefusal):
        client.send(httpx.Request("POST", URL, json={"function_name": CANARY}))
    with pytest.raises(TransportGateRefusal):
        client.post(URL, json={"function_name": CANARY})
    with pytest.raises(TransportGateRefusal):
        with client.stream("POST", URL, json={"function_name": CANARY}):
            pass
    assert sent == []
    client.post(CUSTOMER_URL, json={"function_name": CANARY})
    assert sent == [CUSTOMER_URL]


async def test_httpx_async_send_and_stream_refuse_before_send() -> None:
    httpx = pytest.importorskip("httpx")
    sent: list[str] = []

    def handler(request):
        sent.append(str(request.url))
        return httpx.Response(200, json={})

    client = gate.gate_session(
        httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    gate.set_mode("refuse")
    with pytest.raises(TransportGateRefusal):
        await client.send(httpx.Request("POST", URL, json={"function_name": CANARY}))
    with pytest.raises(TransportGateRefusal):
        async with client.stream("POST", URL, json={"function_name": CANARY}):
            pass
    assert sent == []
    await client.aclose()


def test_gate_session_ignores_mocks_and_is_idempotent() -> None:
    from unittest.mock import MagicMock

    m = MagicMock()
    assert gate.gate_session(m) is m
    requests = pytest.importorskip("requests")
    s = gate.gate_session(requests.Session())
    cls = type(s)
    assert gate.gate_session(s) is s and type(s) is cls
    assert (
        requests.Session is not cls
        and "traigent" not in sys.modules.get("requests").__name__
    )


@pytest.mark.parametrize("verb", ["put", "patch", "delete"])
async def test_aiohttp_other_verbs_reach_the_gate_but_have_no_gated_route(verb) -> None:
    aiohttp = pytest.importorskip("aiohttp")
    sent: list[str] = []
    cls = aiohttp.ClientSession
    original = cls._request

    async def fake(self, method, url, **kw):
        sent.append(str(method))
        return "ok"

    cls._request = fake
    try:
        gate.set_mode("refuse")
        async with gate.gate_session(aiohttp.ClientSession()) as s:
            assert await getattr(s, verb)(URL, json={"a": 1}) == "ok"
        assert sent == [
            verb.upper()
        ]  # table has no such route: not inspected, not blocked
    finally:
        cls._request = original


def test_undeclared_path_parameters_are_refused_not_guessed() -> None:
    gate.set_mode("refuse")
    ok_body = json.dumps({"x": 1}).encode()
    for sid in (
        "prompt_hello-world",
        "sess_0123456789ab",
        "00000000-0000-4000-8000-000000000001",
    ):
        with pytest.raises(TransportGateRefusal) as err:
            inspect_outgoing(
                "POST", f"{BACKEND}/api/v1/sessions/{sid}/results", ok_body
            )
        assert err.value.counts.get("legacy_content", 0) >= 1
        assert sid not in str(err.value)
    gate.set_mode("report")
    c = inspect_outgoing(
        "POST", f"{BACKEND}/api/v1/sessions/prompt_hello-world/results", ok_body
    )
    assert c and c["legacy_content"] >= 1


def test_every_client_construction_site_is_gated_or_allowlisted() -> None:
    """AST census of HTTP client construction and module-level requests calls."""
    import ast
    from pathlib import Path as P

    root = P(__file__).resolve().parents[3] / "traigent"
    # Sites that can never talk to the configured Traigent backend (third parties,
    # customer-defined endpoints, MCP stdio transport). Each needs a reason.
    allow = {
        "integrations/langfuse/client.py": "Langfuse API, third party",
        "integrations/observability/workflow_traces.py": "separate observability product, own endpoint",
        "integrations/llms/nous_auth.py": "Nous provider auth, third party",
        "connectors/http.py": "customer-defined connector endpoint",
        "utils/retry.py": "generic retry helper, caller-supplied URL docs example",
        "cloud/production_mcp_client.py": "MCP transport session, not HTTP to the backend",
    }

    def ctor(name: str) -> bool:
        return name.endswith(("ClientSession", "AsyncClient")) or name in (
            "requests.Session",
            "httpx.Client",
        )

    ungated: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("labels/") or rel in allow:
            continue
        src = path.read_text("utf-8")
        tree = ast.parse(src)
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            name = ast.unparse(n.func)
            parent = parents.get(n)
            if ctor(name) and not (
                isinstance(parent, ast.Call)
                and ast.unparse(parent.func) == "gate_session"
            ):
                ungated.append(f"{rel}:{n.lineno} {name}")
            if name in (
                "requests.get",
                "requests.post",
                "requests.put",
                "requests.patch",
                "requests.delete",
                "requests.request",
            ):
                fn = n
                while fn in parents and not isinstance(
                    fn, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    fn = parents[fn]
                body = ast.get_source_segment(src, fn) or ""
                if "preflight_requests(" not in body:
                    ungated.append(
                        f"{rel}:{n.lineno} {name} without preflight_requests"
                    )
    assert not ungated, "ungated HTTP client construction:\n" + "\n".join(ungated)
