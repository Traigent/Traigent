"""A 302 from the relay origin is never followed by any gated SDK HTTP client.

Production posture: no environment marker, ``TRAIGENT_EGRESS_RELAY=1``, the SDK
pointed at a loopback origin that answers 302 to a second loopback server.
"""

from __future__ import annotations

import asyncio
import http.server
import threading
from collections.abc import Iterator

import pytest

from traigent.labels import transport_gate as gate


class _Target:
    hits = 0


def _server(handler: type[http.server.BaseHTTPRequestHandler]):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture
def redirecting_origin(monkeypatch) -> Iterator[tuple[str, _Target]]:
    target = _Target()

    class Final(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            return

        def do_GET(self):
            target.hits += 1
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    final, t1 = _server(Final)
    final_url = f"http://127.0.0.1:{final.server_address[1]}/final"

    class Redirector(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            return

        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", final_url)
            self.send_header("Content-Length", "0")
            self.end_headers()

    red, t2 = _server(Redirector)
    origin = f"http://127.0.0.1:{red.server_address[1]}"
    for name in ("TRAIGENT_ENVIRONMENT", "ENVIRONMENT", "TRAIGENT_ENV"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIGENT_EGRESS_RELAY", "1")
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", origin)
    monkeypatch.delenv("TRAIGENT_TRANSPORT_GATE", raising=False)
    yield origin, target
    for srv, th in ((red, t2), (final, t1)):
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)


def _outcome(call):
    try:
        response = call()
    except Exception as exc:  # the gate's local refusal or an HTTP error
        return type(exc).__name__
    return getattr(response, "status_code", None) or getattr(response, "status", None)


def test_gated_httpx_does_not_follow(redirecting_origin) -> None:
    import httpx

    origin, target = redirecting_origin
    client = gate._build_httpx_sync(httpx.Client)(follow_redirects=True)
    try:
        result = _outcome(lambda: client.get(f"{origin}/api/v1/x"))
    finally:
        client.close()
    assert target.hits == 0
    assert result != 200


def test_gated_requests_does_not_follow(redirecting_origin) -> None:
    import requests

    origin, target = redirecting_origin
    session = gate._build_requests(requests.Session)()
    try:
        result = _outcome(lambda: session.get(f"{origin}/api/v1/x"))
    finally:
        session.close()
    assert target.hits == 0
    assert result != 200


def test_sdk_urllib_helper_does_not_follow(redirecting_origin) -> None:
    import urllib.request

    from traigent.cloud.url_security import open_no_redirect

    origin, target = redirecting_origin
    req = urllib.request.Request(f"{origin}/api/v1/x")  # noqa: S310
    result = _outcome(lambda: open_no_redirect(req, timeout=5))
    assert target.hits == 0
    assert result != 200


def test_gated_aiohttp_does_not_follow(redirecting_origin) -> None:
    aiohttp = pytest.importorskip("aiohttp")
    origin, target = redirecting_origin

    async def run():
        session = gate._build_aiohttp(aiohttp.ClientSession)()
        try:
            return await session.get(f"{origin}/api/v1/x")
        finally:
            await session.close()

    result = _outcome(lambda: asyncio.run(run()))
    assert target.hits == 0
    assert result != 200
