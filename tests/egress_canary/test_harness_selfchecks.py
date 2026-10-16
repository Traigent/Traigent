"""Checks that the egress harness and detector themselves are trustworthy."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import socket
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

from .canaries import CANARIES, find_leaks, forms
from .harness import (
    EgressCapture,
    EgressGuardError,
    normalize_path,
    redact_body_credentials,
)
from .report import render_report
from .scenario import FAKE_API_KEY, ScenarioResult, run_scenario

pytestmark = pytest.mark.backend_online

SECRET = "Bearer s3cr3t-token-value-9f8e7d"
SECRET_HEADERS = {
    "Authorization": SECRET,
    "X-API-Key": "uk_" + "z" * 43,
    "X-Service-Key": "svc-key-123",
    "Cookie": "session=abc",
    "Content-Type": "application/json",
}


def _send_all_transports(url: str) -> None:
    import aiohttp
    import httpx
    import requests

    requests.post(url, json={"a": 1}, headers=SECRET_HEADERS, timeout=5)

    async def _aio() -> None:
        async with aiohttp.ClientSession(headers={"Authorization": SECRET}) as s:
            async with s.post(url, json={"a": 2}, headers=SECRET_HEADERS) as r:
                await r.read()

    asyncio.run(_aio())

    async def _httpx() -> None:
        async with httpx.AsyncClient() as c:
            await c.post(url, json={"a": 3}, headers=SECRET_HEADERS)

    asyncio.run(_httpx())
    httpx.Client().post(url, json={"a": 4}, headers=SECRET_HEADERS)
    req = urllib.request.Request(
        url, data=b'{"a":5}', headers=SECRET_HEADERS, method="POST"
    )
    urllib.request.urlopen(req, timeout=5).read()  # noqa: S310


def test_every_transport_is_captured_and_authorization_is_never_stored() -> None:
    url = "https://backend.example.test/api/v1/sessions"
    with EgressCapture() as cap:
        _send_all_transports(url)
    assert {r.transport for r in cap.requests} == {
        "requests",
        "aiohttp",
        "httpx",
        "urllib",
    }
    assert len(cap.requests) == 5
    for req in cap.requests:
        names = {k.lower() for k, _ in req.headers}
        assert not names & {"authorization", "x-api-key", "x-service-key", "cookie"}
        blob = repr(req)
        for secret in (SECRET, "s3cr3t", "svc-key-123", "session=abc", "z" * 43):
            assert secret not in blob
        assert ("Content-Type", "application/json") in req.headers or any(
            k.lower() == "content-type" for k, _ in req.headers
        )


def test_credentials_in_json_bodies_are_redacted_at_capture() -> None:
    body = json.dumps({"api_key": "uk_" + "q" * 43, "nested": [{"token": "t0k3n"}]})
    out = redact_body_credentials(body.encode()).decode()
    assert "qqqq" not in out and "t0k3n" not in out
    assert json.loads(out)["api_key"] == "<redacted-credential>"
    assert redact_body_credentials(b"not json") == b"not json"


def test_authorization_never_reaches_scenario_capture_or_report(tmp_path: Path) -> None:
    result = run_scenario(tmp_path, direct_paths=False)
    report = render_report(result)
    assert result.requests
    for req in result.requests:
        assert FAKE_API_KEY not in repr(req)
        assert "authorization" not in {k.lower() for k, _ in req.headers}
        assert "x-api-key" not in {k.lower() for k, _ in req.headers}
    assert FAKE_API_KEY not in report
    assert "Bearer" not in report


def test_socket_guard_blocks_and_records_unmocked_connects() -> None:
    original_connect = socket.socket.connect
    original_getaddrinfo = socket.getaddrinfo
    with EgressCapture() as cap:
        with pytest.raises(EgressGuardError):
            socket.create_connection(("203.0.113.9", 80), timeout=1)
        probe = socket.socket()
        try:
            with pytest.raises(EgressGuardError):
                probe.connect(("127.0.0.1", 9))
        finally:
            probe.close()
        with pytest.raises(EgressGuardError):
            socket.getaddrinfo("not-the-fake-host.invalid", 443)
    assert len(cap.guard_violations) == 3
    # Patches are removed again on exit.
    assert socket.socket.connect is original_connect
    assert socket.getaddrinfo is original_getaddrinfo


def test_dns_for_the_fake_backend_host_is_answered_locally() -> None:
    with EgressCapture() as cap:
        infos = socket.getaddrinfo("backend.example.test", 443)
    assert infos and cap.dns_lookups == ["backend.example.test"]
    assert cap.guard_violations == []


@pytest.mark.parametrize(
    "form",
    [
        "raw",
        "lowercase",
        "slug",
        "url",
        "hex",
        "sha256",
        "md5",
        "sha256_lower",
        "md5_lower",
        "base64",
    ],
)
def test_detector_finds_each_encoded_form(form: str) -> None:
    from .harness import CapturedRequest

    kind = "run_title"
    value = "Weird Title/with spaces&Mixed_Case-123 xYz9Q"
    needles = forms(value)
    if form == "base64":
        inner = base64.b64encode(value.encode()).decode()
        payload = f"prefix-{inner}-suffix"
    elif form == "url":
        payload = quote(value, safe="")
    elif form == "slug":
        payload = needles["slug"]
    elif form == "lowercase":
        payload = value.lower()
    elif form == "hex":
        payload = value.encode().hex()
    elif form == "sha256":
        payload = hashlib.sha256(value.encode()).hexdigest()
    elif form == "md5":
        payload = hashlib.md5(value.encode(), usedforsecurity=False).hexdigest()
    elif form == "sha256_lower":
        payload = hashlib.sha256(value.lower().encode()).hexdigest()
    elif form == "md5_lower":
        payload = hashlib.md5(value.lower().encode(), usedforsecurity=False).hexdigest()
    else:
        payload = value
    for location, req in {
        "body.value": CapturedRequest(
            "t",
            "POST",
            "https://x/y",
            "/y",
            "",
            (),
            json.dumps({"k": payload}).encode(),
        ),
        "query": CapturedRequest(
            "t", "GET", "https://x/y", "/y", f"q={payload}", (), b""
        ),
        "path": CapturedRequest(
            "t", "GET", "https://x/" + payload, "/" + payload, "", (), b""
        ),
        "header": CapturedRequest(
            "t", "GET", "https://x/y", "/y", "", (("X-Foo", payload),), b""
        ),
    }.items():
        leaks = find_leaks([req], {kind: value})
        assert leaks, f"{form} not found in {location}"
        assert {leak.location for leak in leaks} == {location}


def test_detector_has_no_false_positive_on_clean_traffic() -> None:
    from .harness import CapturedRequest

    req = CapturedRequest(
        "t",
        "POST",
        "https://x/api/v1/sessions",
        "/api/v1/sessions",
        "a=1",
        (("Content-Type", "application/json"),),
        json.dumps({"agent_key": "ordinary", "n": 3}).encode(),
    )
    assert find_leaks([req]) == []


def test_every_canary_is_distinct_and_high_entropy() -> None:
    values = list(CANARIES.values())
    assert len(set(values)) == len(values)
    for value in values:
        assert len(value) >= 18
        assert value != value.lower()  # mixed case: lowercase form differs


def test_route_normalization_folds_server_ids_only() -> None:
    assert (
        normalize_path("/api/v1/sessions/sess-canary-1/results")
        == "/api/v1/sessions/{id}/results"
    )
    assert normalize_path("/api/v1/sessions") == "/api/v1/sessions"
    assert normalize_path("/api/v1/best-configs/abc") == "/api/v1/best-configs/{id}"


def test_scenario_result_type_is_report_renderable(tmp_path: Path) -> None:
    result: ScenarioResult = run_scenario(tmp_path, trials=2, direct_paths=False)
    text = render_report(result, 0)
    assert "## Leaks by route" in text and "## Routes captured" in text
