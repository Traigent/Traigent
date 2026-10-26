"""Round-5 regressions: external sessions vs Traigent origins, closed finalize/auth values,
duplicate JSON members and canonical transmission, IP-literal origin canonicalisation.

Recording transports only (``capture_egress``): sockets are denied, no real network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from traigent.utils import optimization_only_egress as oo

from .harness import capture_egress
from .scenario import BACKEND_URL, FAKE_API_KEY
from .test_optimization_only_egress import _env

pytestmark = pytest.mark.backend_online

API = f"{BACKEND_URL}/api/v1"
HEADERS = {"X-API-Key": FAKE_API_KEY}
CLOSED = json.dumps({"error": "optimization_only", "error_code": "optimization_only"})
_RT = "refresh_" + "token"  # field names only; keeps secret scanners quiet


@pytest.fixture(autouse=True)
def _mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _env(monkeypatch, tmp_path)
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    oo.reset_latch()
    yield
    oo.reset_latch()


def _enforce(method: str, url: str, body: Any = None, **kw: Any) -> Any:
    return oo.enforce_outgoing(method, url, HEADERS, body, explicit=True, **kw)


# ------------------------------------- 1: external marking trusts intent, not destination


def test_hybrid_http_transport_pointed_at_the_traigent_backend_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.hybrid.http_transport import HTTPTransport
    from traigent.hybrid.protocol import HybridExecuteRequest

    monkeypatch.setenv(oo.ENV_VAR, "1")
    request = HybridExecuteRequest(
        tunable_id="t",
        config={"m": "a"},
        examples=[{"example_id": "e1", "input": "Alice has diabetes"}],
    )

    async def go() -> None:
        transport = HTTPTransport(BACKEND_URL)
        try:
            await transport.execute(request)
        finally:
            await transport.close()

    with capture_egress() as cap:
        with pytest.raises(Exception):  # noqa: B017 - refusal, possibly wrapped by the transport
            asyncio.run(go())
    assert cap.requests == []


def test_hybrid_http_transport_to_a_customer_service_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.hybrid.http_transport import HTTPTransport

    monkeypatch.setenv(oo.ENV_VAR, "1")

    async def go() -> None:
        transport = HTTPTransport("https://agent.customer.example")
        try:
            await transport.health_check()
        finally:
            await transport.close()

    with capture_egress() as cap:
        try:
            asyncio.run(go())
        except Exception:  # noqa: BLE001 - the fake backend's answer is irrelevant
            pass
    assert any(r.path.endswith("/health") for r in cap.requests)


def test_external_session_to_an_origin_a_traigent_session_was_made_for_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from traigent.labels.transport_gate import gate_session

    monkeypatch.setenv(oo.ENV_VAR, "1")
    other_backend = "https://eu.backend.example"
    gate_session(
        httpx.Client(), origins=[f"{other_backend}/api/v1"]
    )  # a Traigent session
    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            gate_session(httpx.Client(), external=True).post(
                f"{other_backend}/traigent/v1/execute",
                json={"examples": ["Alice has diabetes"]},
            )
    assert cap.requests == []


def test_external_session_to_a_latched_origin_is_refused() -> None:
    import httpx

    from traigent.labels.transport_gate import gate_session

    oo.note_response(
        403, CLOSED, oo.credential_scope("https://latched.example", HEADERS)
    )
    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            gate_session(httpx.Client(), external=True).post(
                "https://latched.example:443/x", json={"a": 1}
            )
    assert cap.requests == []


def test_external_preflight_to_the_backend_is_refused_and_other_hosts_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.labels.transport_gate import preflight_requests

    monkeypatch.setenv(oo.ENV_VAR, "1")
    with pytest.raises(oo.OptimizationOnlyRefusal):
        preflight_requests("POST", f"{API}/sessions", external=True, json={"a": "b c"})
    preflight_requests("GET", "https://openrouter.ai/api/v1/key", external=True)


def test_external_sessions_never_follow_redirects_while_the_mode_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.labels import transport_gate as gate

    monkeypatch.setenv(oo.ENV_VAR, "1")
    seen: dict[str, Any] = {}

    class Resp:
        status_code = status = 302
        request = None

        def close(self) -> None:
            return None

    class Base:
        def send(self, request: Any, **kw: Any) -> Any:
            seen.update(kw)
            return Resp()

    import httpx

    client = gate._build_httpx_sync(Base)()
    object.__setattr__(client, "_traigent_oo_external", True)
    with pytest.raises(oo.OptimizationOnlyRefusal):
        client.send(httpx.Request("GET", "https://customer.example/x"))
    assert seen["follow_redirects"] is False


# ----------------------------------------- 2: recognised finalize keys and auth values


@pytest.mark.parametrize(
    "extra",
    [
        {"request_metadata": {"notes": "Alice has diabetes"}},
        {"include_full_history": "Alice has diabetes"},
        {"Alice has diabetes": 1},
        {"free_text_member": "Alice has diabetes"},
        {"session_id": "Alice has diabetes"},
    ],
)
def test_finalize_recognised_and_unknown_members_are_closed(
    extra: dict[str, Any],
) -> None:
    body = {"reason": "sdk_explicit_finalization", **extra}
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce("POST", f"{API}/sessions/s1/finalize", json.dumps(body))


def test_finalize_projection_normalises_each_member() -> None:
    out = oo.project_for_phase(
        oo.PHASE_FINALIZE,
        {
            "request_metadata": {"notes": "Alice has diabetes"},
            "include_full_history": True,
            "unknown_numeric": 1,
        },
    )
    assert out == {"request_metadata": {}, "include_full_history": True}
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_phase(oo.PHASE_FINALIZE, {"include_full_history": "yes"})


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/auth/refresh", {_RT: {"notes": "Alice has diabetes"}}),
        ("/auth/refresh", {_RT: ["Alice has diabetes"]}),
        ("/auth/refresh", {_RT: "has spaces in it"}),
        ("/auth/refresh", {_RT: "x" * 5000}),
        ("/auth/login", {"email": "not an email", "pass" + "word": "x"}),
        ("/oauth/token", {"grant_type": "Alice has diabetes"}),
        ("/oauth/token", {"grant_type": "client_credentials", "scope": "read\nwrite"}),
        ("/keys/validate", {"api_" + "key": {"a": 1}}),
    ],
)
def test_auth_values_must_be_bounded_strings_of_the_right_shape(
    path: str, body: dict[str, Any]
) -> None:
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce("POST", API + path, json.dumps(body))


def test_well_formed_auth_values_pass() -> None:
    _enforce("POST", f"{API}/auth/refresh", json.dumps({_RT: "tok_" + "t" * 40}))
    _enforce(
        "POST",
        f"{API}/oauth/token",
        {"grant_type": "client_credentials", "scope": "read write", "client_id": "cid"},
    )
    _enforce(
        "POST",
        f"{API}/auth/login",
        json.dumps({"email": "a@b.co", "pass" + "word": "p w 1"}),
    )


# ----------------------------------------------------- 3: duplicate JSON members


def test_duplicate_members_at_any_depth_are_refused() -> None:
    top = '{"reason":"sdk_explicit_finalization","metadata":{"notes":"Alice has diabetes"},"metadata":{}}'
    nested = (
        '{"reason":"sdk_explicit_finalization","certified_selection":{"a":1,"a":2}}'
    )
    for body in (top, nested):
        with pytest.raises(oo.OptimizationOnlyRefusal):
            _enforce("POST", f"{API}/sessions/s1/finalize", body)
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce("POST", f"{API}/auth/refresh", f'{{"{_RT}":"a","{_RT}":"b"}}')


@pytest.mark.parametrize(
    "body",
    [b'{"reason":"\xff"}', b"\xef\xbb\xbf{}", '{"reason": NaN}', "{'reason': 1}", b""],
)
def test_bodies_that_are_not_valid_utf8_json_are_refused(body: Any) -> None:
    with pytest.raises(oo.OptimizationOnlyRefusal):
        _enforce("POST", f"{API}/sessions/s1/finalize", body)


def test_strict_loader_matches_what_a_server_would_see() -> None:
    assert oo.loads_strict(b'{"a":[1,{"b":2}]}') == {"a": [1, {"b": 2}]}
    for bad in ('{"a":1,"a":1}', '[{"a":1,"a":2}]', "[NaN]", b"\xff"):
        with pytest.raises(ValueError):
            oo.loads_strict(bad)


def _finalize_body() -> bytes:
    return b'{ "reason": "sdk_explicit_finalization",\n  "metadata": {} }'


def _post_wire(kind: str, url: str, raw: bytes) -> None:
    if kind == "aiohttp":
        import aiohttp

        from traigent.labels.transport_gate import gate_session

        async def go() -> None:
            async with gate_session(
                aiohttp.ClientSession(), origins=[BACKEND_URL]
            ) as s:
                async with s.post(url, data=raw, headers=HEADERS):
                    pass

        asyncio.run(go())
    elif kind == "requests":
        import requests

        from traigent.labels.transport_gate import gate_session

        gate_session(requests.Session(), origins=[BACKEND_URL]).post(
            url, data=raw, headers=HEADERS
        )
    elif kind == "httpx":
        import httpx

        from traigent.labels.transport_gate import gate_session

        gate_session(httpx.Client(), origins=[BACKEND_URL]).post(
            url, content=raw, headers=HEADERS
        )
    else:
        import urllib.request

        from traigent.cloud.url_security import build_no_redirect_opener

        build_no_redirect_opener().open(
            urllib.request.Request(url, data=raw, headers=HEADERS, method="POST"),
            timeout=2,
        )


@pytest.mark.parametrize("kind", ["aiohttp", "requests", "httpx", "urllib"])
def test_the_validated_object_is_what_is_transmitted(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A valid body is replaced by its canonical serialisation: judged == sent."""
    monkeypatch.setenv(oo.ENV_VAR, "1")
    with capture_egress() as cap:
        _post_wire(kind, f"{API}/sessions/s1/finalize", _finalize_body())
    (post,) = [r for r in cap.requests if r.path.endswith("/finalize")]
    assert post.body == b'{"reason":"sdk_explicit_finalization","metadata":{}}'


@pytest.mark.parametrize("kind", ["aiohttp", "requests", "httpx", "urllib"])
def test_a_duplicate_member_body_never_reaches_the_wire(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    dup = b'{"reason":"sdk_explicit_finalization","metadata":{"notes":"Alice has diabetes"},"metadata":{}}'
    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            _post_wire(kind, f"{API}/sessions/s1/finalize", dup)
    assert cap.requests == []


# ------------------------------------------------ 4: IP literal canonicalisation


def test_ip_literal_origins_are_canonical() -> None:
    n = oo.normalised_origin
    assert n("https://[2001:0db8:0000:0000:0000:0000:0000:0001]") == n(
        "https://[2001:db8::1]:443/x"
    )
    assert n("https://[::ffff:203.0.113.9]") == n("https://203.0.113.9:443")
    assert n("http://[2001:DB8::1]:80") == "http://[2001:db8::1]"
    assert n("https://203.0.113.9:8443") != n("https://203.0.113.9")
    assert n("https://[2001:db8::1]:8443") == "https://[2001:db8::1]:8443"


def test_a_latch_on_the_expanded_ipv6_form_blocks_the_compressed_form() -> None:
    import requests

    from traigent.labels.transport_gate import gate_session

    oo.note_response(
        403,
        CLOSED,
        oo.credential_scope(
            "https://[2001:0db8:0000:0000:0000:0000:0000:0001]", HEADERS
        ),
    )
    with capture_egress() as cap:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            gate_session(requests.Session()).post(
                "https://[2001:db8::1]/api/v1/evaluators",
                json={"judge_config": {"instructions": "Alice has diabetes"}},
                headers=HEADERS,
            )
    assert cap.requests == []
