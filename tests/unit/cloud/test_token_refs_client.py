"""Token issuance over the real BackendIntegratedClient HTTP layer (content-less 1b-2a).

``POST /api/v1/optimization/token-refs`` goes through the client's strict gated session, its
auth headers and its egress switches. The adapter's send is replaced, so the request that
would leave the machine is inspected byte for byte.
"""

from __future__ import annotations

import json
import socket
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import requests

from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudServiceError
from traigent.labels import token_store as ts
from traigent.labels.token_store import TokenNamespace, TokenStoreError, ensure_tokens

pytestmark = pytest.mark.backend_online

AGENT = "agentref_" + "ab" * 16
_KEY = "tg_" + "x" * 61  # pragma: allowlist secret
HEBREW = "שלום_עולם_טמפרטורה"
EMAIL = "alice.smith+secret@example.com"
_PUBLIC_ADDRINFO = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


class _Wire:
    def __init__(self, *responses: tuple[int, Any]) -> None:
        self.responses = list(responses)
        self.requests: list[requests.PreparedRequest] = []

    def send(self, request):
        self.requests.append(request)
        status, body = self.responses.pop(0) if self.responses else (500, {})
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(body).encode()
        response.headers["Content-Type"] = "application/json"
        response.url = request.url
        response.request = request
        return response


def _wired(*responses):
    wire = _Wire(*responses)

    def adapter_send(adapter, request, **kwargs):
        return wire.send(request)

    return wire, patch.object(requests.adapters.HTTPAdapter, "send", adapter_send)


def _client() -> BackendIntegratedClient:
    with patch("socket.getaddrinfo", return_value=_PUBLIC_ADDRINFO):
        client = BackendIntegratedClient(api_key=_KEY, base_url="https://api.test")
    client.auth_manager.auth.get_headers = AsyncMock(
        return_value={"Authorization": "Bearer test-token"}
    )
    return client


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg-fake-portal-key-token-store")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "state"))
    monkeypatch.setenv(ts.ENV_FLAG, "true")
    for name in ("TRAIGENT_ENVIRONMENT", "ENVIRONMENT", "TRAIGENT_ENV"):
        monkeypatch.delenv(name, raising=False)


def _closed(code: str, message: str) -> dict[str, Any]:
    return {"success": False, "error": code, "error_code": code, "message": message}


def test_issuance_uses_the_contract_route_auth_and_a_names_free_body():
    wire, patcher = _wired(
        (200, {"refs": ["knobref_" + "1" * 32, "metricref_" + "2" * 32]})
    )
    client = _client()
    namespace = TokenNamespace.for_backend("https://api.test/api/v1", AGENT)
    with patcher:
        got = ensure_tokens(
            namespace,
            [("knob", None, HEBREW), ("metric", None, EMAIL)],
            transport=ts.backend_transport(client),
        )
    assert got == {
        ("knob", None, HEBREW): "knobref_" + "1" * 32,
        ("metric", None, EMAIL): "metricref_" + "2" * 32,
    }
    (request,) = wire.requests
    assert request.method == "POST"
    assert request.url == "https://api.test/api/v1/optimization/token-refs"
    assert request.headers["Authorization"] == "Bearer test-token"
    body = json.loads(request.body)
    assert set(body) == {"schema_version", "agent_ref", "operation_key", "items"}
    assert body["items"] == [{"kind": "knob"}, {"kind": "metric"}]
    # nothing of any name anywhere on the request
    blob = json.dumps(
        [request.url, dict(request.headers), request.body.decode()], ensure_ascii=False
    ) + repr(request.body)
    for canary in (HEBREW, EMAIL):
        assert canary not in blob
        assert canary.encode() not in request.body


def test_flag_off_server_404_is_token_refs_unavailable():
    wire, patcher = _wired((404, {"error": "Not Found"}))
    with patcher, pytest.raises(TokenStoreError) as err:
        ensure_tokens(
            TokenNamespace.for_backend("https://api.test", AGENT),
            [("knob", None, "k")],
            transport=ts.backend_transport(_client()),
        )
    assert err.value.code == "token_refs_unavailable"


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (401, {"success": False}, "token_refs_unauthenticated"),
        (403, {"success": False}, "token_refs_forbidden"),
        (
            404,
            _closed("reference_not_found", "Reference not found."),
            "token_refs_reference_not_found",
        ),
        (
            400,
            _closed("invalid_request", "Request is invalid."),
            "token_refs_invalid_request",
        ),
        (500, _closed("internal_error", "Internal error."), "token_refs_server_error"),
    ],
)
def test_closed_answers_map_without_echoing_the_body(status, body, code):
    body = dict(body, leak=HEBREW) if code in {"token_refs_unauthenticated"} else body
    wire, patcher = _wired((status, body))
    with patcher, pytest.raises(TokenStoreError) as err:
        ensure_tokens(
            TokenNamespace.for_backend("https://api.test", AGENT),
            [("knob", None, "k")],
            transport=ts.backend_transport(_client()),
        )
    assert err.value.code == code
    assert HEBREW not in str(err.value) + repr(err.value)


def test_replay_after_a_transport_failure_sends_the_same_key_and_batch():
    ok = {"refs": ["knobref_" + "3" * 32]}
    wire, patcher = _wired((200, ok), (200, ok))
    client = _client()
    namespace = TokenNamespace.for_backend("https://api.test", AGENT)
    real = client._protected_request
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise CloudServiceError("Token-refs issuance transport failed")
        return real(*a, **k)

    with patcher:
        client._protected_request = flaky  # type: ignore[method-assign]
        with pytest.raises(TokenStoreError) as err:
            ensure_tokens(
                namespace, [("knob", None, "k")], transport=ts.backend_transport(client)
            )
        assert err.value.code == "token_refs_transport_failed"
        got = ensure_tokens(
            namespace, [("knob", None, "k")], transport=ts.backend_transport(client)
        )
    assert got == {("knob", None, "k"): "knobref_" + "3" * 32}
    assert len(wire.requests) == 1  # the first attempt never reached the wire


def test_no_egress_client_sends_nothing():
    wire, patcher = _wired((200, {"refs": ["knobref_" + "1" * 32]}))
    client = _client()
    client.no_egress = True
    with patcher, pytest.raises(TokenStoreError):
        ensure_tokens(
            TokenNamespace.for_backend("https://api.test", AGENT),
            [("knob", None, "k")],
            transport=ts.backend_transport(client),
        )
    assert wire.requests == []


# ---- K3: the send method itself enforces the closed request contract ------------------


def _good_body() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "agent_ref": AGENT,
        "operation_key": "0" * 32,
        "items": [
            {"kind": "knob"},
            {"kind": "choice", "parent_ref": "knobref_" + "b" * 32},
        ],
    }


def _bad_bodies() -> list[dict[str, Any]]:
    def mutate(fn):
        body = _good_body()
        fn(body)
        return body

    return [
        mutate(lambda b: b["items"][0].update(name=EMAIL)),  # an extra name key
        mutate(lambda b: b["items"][0].update(kind=EMAIL)),  # email as kind
        mutate(
            lambda b: b["items"][1].update(parent_ref=HEBREW)
        ),  # Hebrew as parent_ref
        mutate(lambda b: b.update(label=HEBREW)),  # extra top-level key
        mutate(lambda b: b.update(operation_key=HEBREW)),
        mutate(lambda b: b.update(agent_ref=EMAIL)),
        mutate(lambda b: b.update(schema_version=True)),
        mutate(lambda b: b.update(schema_version=2)),
        mutate(lambda b: b.update(items=[])),
        mutate(lambda b: b.update(items=[{"kind": "knob"}] * 201)),
        mutate(lambda b: b["items"][0].update(parent_ref="knobref_" + "c" * 32)),
        mutate(lambda b: b["items"][1].pop("parent_ref")),
        mutate(lambda b: b["items"].append("not-an-object")),
        mutate(lambda b: b.update(operation_key="0" * 32 + "\n")),
        {"items": [{"kind": "knob", "name": HEBREW}]},
        ["not", "a", "dict"],
    ]


@pytest.mark.parametrize("index", range(16))
def test_malformed_or_content_bearing_bodies_never_reach_the_http_adapter(index):
    body = _bad_bodies()[index]
    wire, patcher = _wired((200, {"refs": ["knobref_" + "1" * 32]}))
    with patcher, pytest.raises(CloudServiceError) as err:
        _client().issue_token_refs_sync(body)
    assert wire.requests == []  # nothing reached the adapter
    blob = repr(err.value) + str(err.value)
    assert HEBREW not in blob and EMAIL not in blob
    assert str(err.value) == "Token-refs request is not the closed contract shape"
    assert err.value.__cause__ is None


def test_the_closed_body_is_still_sent():
    wire, patcher = _wired(
        (200, {"refs": ["knobref_" + "1" * 32, "choiceref_" + "2" * 32]})
    )
    with patcher:
        status, payload = _client().issue_token_refs_sync(_good_body())
    assert status == 200 and "refs" in payload
    assert len(wire.requests) == 1


def test_transport_gate_in_refusal_mode_classifies_the_route():
    """The generated table now knows the route: a content body is refused, a closed one passes."""
    from traigent.labels import transport_gate as gate

    url = "https://api.test/api/v1/optimization/token-refs"
    gate.set_mode("refuse")
    gate.set_allowed_origins(["https://api.test"])
    try:
        counts = gate.inspect_outgoing("POST", url, json.dumps(_good_body()))
        assert counts is not None and not any(counts.get(c, 0) for c in gate.BLOCKING)
        for body in _bad_bodies()[:6]:
            with pytest.raises(gate.TransportGateRefusal):
                gate.inspect_outgoing("POST", url, json.dumps(body))
    finally:
        gate.set_mode(None)
        gate.set_allowed_origins(None)
