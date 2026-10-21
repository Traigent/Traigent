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

from traigent.cloud.auth import AuthenticationError
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudEgressBlockedError, CloudServiceError
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
        status, body, *extra = self.responses.pop(0) if self.responses else (500, {})
        response = requests.Response()
        response.status_code = status
        for name, value in (extra[0] if extra else {}).items():
            response.headers[name] = value
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
    # the state root is created 0700 up front: the client may create it under a 0002 umask
    # (0775), which the strict store refuses (non-sticky group-writable component)
    state = tmp_path / "state"
    state.mkdir(mode=0o700, exist_ok=True)
    state.chmod(0o700)
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(state))
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


@pytest.mark.parametrize(
    "operation_key",
    ["customer_name", "Alice_has_diabetes", "a" * 31, "A" * 32, "0" * 31 + "G", ""],
)
def test_gate_refuses_an_operation_key_that_is_not_32_lowercase_hex(operation_key):
    """L3: the generated table keeps ``^[0-9a-f]{32}$`` (and minLength) for operation_key."""
    from traigent.labels import transport_gate as gate

    url = "https://api.test/api/v1/optimization/token-refs"
    body = _good_body()
    body["operation_key"] = operation_key
    gate.set_mode("refuse")
    gate.set_allowed_origins(["https://api.test"])
    try:
        with pytest.raises(gate.TransportGateRefusal):
            gate.inspect_outgoing("POST", url, json.dumps(body))
        good = gate.inspect_outgoing("POST", url, json.dumps(_good_body()))
        assert good is not None and not any(good.get(c, 0) for c in gate.BLOCKING)
        valid = _good_body()
        valid["operation_key"] = "0123456789abcdef" * 2
        assert gate.inspect_outgoing("POST", url, json.dumps(valid)) is not None
    finally:
        gate.set_mode(None)
        gate.set_allowed_origins(None)


def _own_raises(case, client, monkeypatch):
    """Arrange one of the method's OWN raise paths; return the body to send."""
    body = _good_body()
    if case == "validation":
        body["items"][0]["name"] = EMAIL
    elif case == "egress_disabled":
        client.no_egress = True  # the real egress switch: CloudEgressBlockedError
    elif case == "url_invalid":
        monkeypatch.setattr(client, "_build_token_refs_url", lambda: None)
    elif case == "auth_rejected":
        from traigent.cloud.auth import InvalidCredentialsError

        client.auth_manager.auth.get_headers = AsyncMock(
            side_effect=InvalidCredentialsError(f"rejected {HEBREW} {EMAIL}")
        )
    elif case == "transport":
        monkeypatch.setattr(
            client,
            "_protected_request",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"down {HEBREW}")),
        )
    return body


_EXPECTED_TYPE = {
    "validation": CloudServiceError,
    "egress_disabled": CloudEgressBlockedError,
    "url_invalid": CloudServiceError,
    "auth_rejected": AuthenticationError,
    "transport": CloudServiceError,
}


@pytest.mark.parametrize("case", list(_EXPECTED_TYPE))
def test_direct_issuance_clears_exception_context_inside_a_callers_except(
    case, monkeypatch
):
    """L4: the K5 boundary on the public client method, for each of its own raises."""
    import traceback

    client = _client()
    body = _own_raises(case, client, monkeypatch)
    caught: BaseException | None = None
    try:
        raise ValueError(f"customer {HEBREW} {EMAIL}")
    except ValueError:
        try:
            client.issue_token_refs_sync(body)
        except (
            CloudServiceError,
            CloudEgressBlockedError,
            AuthenticationError,
        ) as exc:  # INSIDE the handler
            caught = exc
    assert caught is not None
    assert type(caught) is _EXPECTED_TYPE[case]  # exact type, not "one of the families"
    assert caught.__cause__ is None and caught.__context__ is None
    rendered = "".join(traceback.format_exception(caught))
    assert HEBREW not in rendered and EMAIL not in rendered
    assert HEBREW not in str(caught) and EMAIL not in repr(caught)


def test_direct_issuance_sends_the_validated_snapshot_not_the_callers_mutated_body():
    """L5: a name inserted into the caller's body while auth blocks never reaches the wire."""
    wire, patcher = _wired((200, {"refs": ["knobref_" + "1" * 32] * 2}))
    client = _client()
    body = _good_body()
    original = json.loads(json.dumps(body))
    real_headers = client._get_sync_auth_headers

    def mutating_auth(*args, **kwargs):
        body["operation_key"] = HEBREW  # another thread edits the shared dict mid-auth
        body["items"][0]["name"] = EMAIL
        return real_headers(*args, **kwargs)

    client._get_sync_auth_headers = mutating_auth  # type: ignore[method-assign]
    with patcher:
        status, _ = client.issue_token_refs_sync(body)
    assert status == 200
    sent = wire.requests[0].body
    sent_text = sent.decode() if isinstance(sent, bytes) else str(sent)
    assert json.loads(sent_text) == original
    assert HEBREW not in sent_text and EMAIL not in sent_text


def test_direct_issuance_refuses_non_plain_data():
    client = _client()
    body = _good_body()
    body["schema_version"] = 1.0  # a float is not plain contract data
    with pytest.raises(CloudServiceError):
        client.issue_token_refs_sync(body)


_UNAVAILABLE = {
    "success": False,
    "error": "service_unavailable",
    "error_code": "service_unavailable",
    "message": "Service is temporarily unavailable.",
}


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("7", 7.0),
        ("0", 0.0),
        ("999999", 3600.0),
        ("nan", None),
        ("inf", None),
        ("-5", None),
        ("abc", None),
        ("Wed, 21 Oct 2026 07:28:00 GMT", None),
        ("1e3", None),
        (" 2.5 ", 2.5),
        (None, None),
    ],
)
def test_503_retry_after_header_is_read_as_a_bounded_number_of_seconds(
    header, expected
):
    """1b-2b: the client returns the Retry-After hint (closed number), never header text."""
    headers = {} if header is None else {"Retry-After": header}
    _wire, patcher = _wired((503, _UNAVAILABLE, headers))
    with patcher:
        status, payload, retry_after = _client().issue_token_refs_with_retry_after_sync(
            _good_body()
        )
    assert status == 503 and payload == _UNAVAILABLE
    assert retry_after == expected


def test_503_end_to_end_retries_with_retry_after_then_issues(monkeypatch):
    """1b-2b: ensure_tokens over the real client: 503 + Retry-After, then the same key succeeds."""
    sleeps: list[float] = []
    monkeypatch.setattr(ts, "_sleep", sleeps.append)
    refs = ["knobref_" + "7" * 32]
    wire, patcher = _wired(
        (503, _UNAVAILABLE, {"Retry-After": "3"}), (200, {"refs": refs})
    )
    client = _client()
    with patcher:
        got = ensure_tokens(
            TokenNamespace.for_backend("https://api.test", AGENT),
            [("knob", None, HEBREW)],
            transport=ts.backend_transport(client),
        )
    assert list(got.values()) == refs
    assert sleeps == [3.0]
    keys = {json.loads(r.body)["operation_key"] for r in wire.requests}
    assert len(wire.requests) == 2 and len(keys) == 1
    assert all(HEBREW not in str(r.body) for r in wire.requests)


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Retry-After": True}, None),
        ({"Retry-After": False}, None),
        ({"Retry-After": [3]}, None),
        ({"Retry-After": b"3"}, None),
        ({"Retry-After": -1}, None),
        ({"Retry-After": float("nan")}, None),
        ({"Retry-After": 4}, 4.0),
        ({"Retry-After": 2.5}, 2.5),
        ({"Retry-After": 99999}, 3600.0),
        (object(), None),
        ("Retry-After: 3", None),
        (None, None),
        ({}, None),
    ],
)
def test_retry_after_seconds_validates_types_without_swallowing(headers, expected):
    """1b-2b: wrong types are rejected by explicit checks, never by catching errors."""
    from traigent.cloud.backend_client import _retry_after_seconds

    assert _retry_after_seconds(headers) == expected
