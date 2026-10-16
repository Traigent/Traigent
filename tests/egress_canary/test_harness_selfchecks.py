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


def test_socket_guard_is_in_process_only_subprocess_sockets_are_out_of_scope() -> None:
    """Documented limitation: the guard patches Python objects in this process.

    A child process (or native code opening its own socket) does not inherit the
    patches, so the guard neither blocks nor records it.  The harness is an
    in-process guard, not a network sandbox; this test pins that boundary so a
    future reader does not mistake a clean run for proof about subprocesses.
    """

    import subprocess
    import sys

    code = (
        "import socket;"
        "orig=socket.socket.connect;"
        "print(orig.__module__ if hasattr(orig,'__module__') else 'native')"
    )
    with EgressCapture() as cap:
        out = subprocess.run(  # noqa: S603
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        # In this process connect is the guard; in the child it is the stock one.
        assert socket.socket.connect.__module__ != "socket"
    assert "tests.egress_canary" not in out.stdout
    assert cap.guard_violations == []  # the child's sockets are invisible to us


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


@pytest.mark.parametrize("algo", ["sha256", "md5"])
@pytest.mark.parametrize("length", [8, 12, 16, 24, 32, 63])
def test_detector_finds_digest_prefixes_from_8_characters(
    algo: str, length: int
) -> None:
    from .harness import CapturedRequest

    value = "cnryPrefix_Qa1B-2cDe3Fg4"
    digest = getattr(hashlib, algo)(value.encode()).hexdigest()[:length]
    req = CapturedRequest(
        "t",
        "POST",
        "https://x/y",
        "/y",
        "",
        (),
        json.dumps({"checksum": f"v1:{digest}:end"}).encode(),
    )
    assert find_leaks([req], {"k": value}), f"{algo} prefix of {length} not found"
    # 7 characters is below the documented floor (too short to be distinctive).
    short = getattr(hashlib, algo)(value.encode()).hexdigest()[:7]
    below = CapturedRequest(
        "t", "POST", "https://x/y", "/y", "", (), json.dumps({"c": short}).encode()
    )
    assert find_leaks([below], {"k": value}) == []


@pytest.mark.parametrize("algo", ["sha256", "md5"])
def test_detector_finds_base64_of_the_digest(algo: str) -> None:
    from .harness import CapturedRequest

    value = "cnryPrefix_Qa1B-2cDe3Fg4"
    raw = getattr(hashlib, algo)(value.encode()).digest()
    for enc in (
        base64.b64encode(raw).decode(),
        base64.urlsafe_b64encode(raw).decode().rstrip("="),
    ):
        req = CapturedRequest(
            "t", "POST", "https://x/y", "/y", "", (), json.dumps({"d": enc}).encode()
        )
        assert find_leaks([req], {"k": value}), f"{algo} base64 digest not found"


def test_report_never_renders_raw_exception_text(tmp_path: Path) -> None:
    secret = "Bearer abcdef0123456789SECRETTOKEN uk_" + "b" * 43
    result = run_scenario(tmp_path, trials=2, direct_paths=False)
    result.optimization_error = f"RuntimeError: auth failed with {secret}"
    result.offline_error = f"ValueError: api_key=hunter2hunter2 {secret}"
    result.exerciser_outcomes = {"x": f"RuntimeError: {secret}"}
    text = render_report(result, 0)
    for needle in ("SECRETTOKEN", "hunter2", "b" * 43, "auth failed"):
        assert needle not in text
    assert "RuntimeError" in text


def test_scrubber_redacts_credential_shapes() -> None:
    from .report import scrub

    for raw in (
        "Authorization: Bearer abcdef0123456789",
        "token=abcd1234efgh",
        "key uk_" + "c" * 40,
    ):
        assert "abcd" not in scrub(raw) and "c" * 40 not in scrub(raw)


def test_fault_injection_serves_each_outcome_once_then_recovers() -> None:
    import requests

    with EgressCapture() as cap:
        cap.backend.inject("POST", r"/api/v1/sessions$", [503, "connection"])
        url = "https://backend.example.test/api/v1/sessions"
        assert requests.post(url, json={"a": 1}, timeout=5).status_code == 503
        with pytest.raises(requests.ConnectionError):
            requests.post(url, json={"a": 1}, timeout=5)
        assert requests.post(url, json={"a": 1}, timeout=5).status_code == 201
    assert len(cap.requests) == 3
    assert cap.backend.faults[0].consumed == 2


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


_A = ("POST /api/v1/sessions", "body.value", "$.a", "agent_name")
_B = ("POST /api/v1/sessions", "body.value", "$.b", "run_title")
_NEW = ("POST /api/v1/sessions", "body.value", "$.new", "metadata_value")


def _disp(identity, reference):
    r, loc, fp, kind = identity
    return {
        "route": r,
        "location": loc,
        "field_path": fp,
        "kind": kind,
        "reason": "fixed",
        "reference": reference,
    }


def test_edited_manifest_hiding_an_addition_still_fails() -> None:
    from .pins import ratchet_problems

    anchor = {_A, _B}
    live = {_A, _B, _NEW}
    # The PR edits the working-tree manifest to include the new leak...
    edited_register = {_A, _B, _NEW}
    problems = ratchet_problems(live, anchor, edited_register, [])
    assert any("new leak not in" in p for p in problems)
    # ...and it passes only against a (wrongly) self-referential anchor.
    assert ratchet_problems(live, live, edited_register, []) == []


def test_bogus_disposition_reference_fails() -> None:
    from .pins import ratchet_problems

    ids = {"S1"}
    for bad in (
        "because",
        "INV:S999",
        "no/such/file.py:3",
        "tests/egress_canary/pins.py:999999",
        "",
    ):
        problems = ratchet_problems({_A}, {_A, _B}, {_A}, [_disp(_B, bad)], ids=ids)
        assert any("invalid disposition" in p for p in problems), bad
    for good in ("INV:S1", "tests/egress_canary/pins.py:1"):
        assert ratchet_problems({_A}, {_A, _B}, {_A}, [_disp(_B, good)], ids=ids) == []
    assert any(
        "without disposition" in p for p in ratchet_problems({_A}, {_A, _B}, {_A}, [])
    )


def test_first_introduction_requires_exact_manifest_match() -> None:
    from .pins import ratchet_problems

    assert ratchet_problems({_A}, None, {_A}, []) == []
    assert ratchet_problems({_A, _NEW}, None, {_A}, [])
    assert ratchet_problems({_A}, None, {_A, _B}, [])


def test_unreadable_authority_fails_not_skips(monkeypatch) -> None:
    from . import pins

    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(pins.subprocess, "run", boom)
    with pytest.raises(pins.AnchorUnavailable):
        pins.load_anchor()
