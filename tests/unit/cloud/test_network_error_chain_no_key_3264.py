"""#3264: a wrapped transport error must not expose the API key via its chain.

A backend (or proxy) answering HTTP 200 with a non-JSON content type makes
``response.json()`` raise aiohttp ``ContentTypeError`` whose ``repr`` prints
``RequestInfo(headers=...)`` including the live ``X-API-Key``. ``raise X from None``
inside the handler left that exception reachable through ``__context__``.
Only a synthetic canary key is used.
"""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.client import CloudServiceError, TraigentCloudClient

CANARY = "tg_CANARY3264" + "x" * 51  # 64 chars, synthetic


class _HtmlHandler(BaseHTTPRequestHandler):
    def _respond(self) -> None:
        body = b"<html>not json</html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_DELETE = do_PUT = _respond

    def log_message(self, *args, **kwargs) -> None:  # silence
        pass


@pytest.fixture
def stub_url(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _HtmlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("TRAIGENT_ALLOW_PRIVATE_BACKEND", "1")
    monkeypatch.delenv("TRAIGENT_OFFLINE_MODE", raising=False)

    async def _no_remote_validation(self, api_key):  # never leave the loopback
        return None

    monkeypatch.setattr(
        "traigent.cloud.auth.AuthManager._validate_api_key_with_backend",
        _no_remote_validation,
    )
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _inject_content_type_error(monkeypatch) -> None:
    import aiohttp
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    async def _boom(self, method, url, **kwargs):
        info = aiohttp.RequestInfo(
            url=URL(str(url)),
            method=method,
            headers=CIMultiDictProxy(CIMultiDict({"X-API-Key": CANARY})),
            real_url=URL(str(url)),
        )
        raise aiohttp.ContentTypeError(info, (), message="Attempt to decode JSON")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", _boom)


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    stack: list[BaseException | None] = [exc]
    while stack:
        cur = stack.pop()
        if cur is None or any(cur is s for s in seen):
            continue
        seen.append(cur)
        stack.extend([cur.__cause__, cur.__context__])
    return seen


def _assert_clean(exc: BaseException) -> None:
    assert isinstance(exc, CloudServiceError)
    for item in _chain(exc):
        assert CANARY not in repr(item), type(item).__name__
        assert CANARY not in str(item), type(item).__name__
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert str(exc).startswith("Network error")


def _backend(url: str) -> BackendIntegratedClient:
    return BackendIntegratedClient(api_key=CANARY, base_url=url)


async def _capture(coro):
    try:
        await coro
    except Exception as exc:  # noqa: BLE001
        return exc
    pytest.fail("expected CloudServiceError")


@pytest.mark.parametrize(
    "method",
    ["get_status", "finalize", "delete", "create"],
)
def test_session_operations_network_error_chain_has_no_key(
    stub_url, method, monkeypatch
):
    if method in {"delete", "create"}:
        # The loopback 200/html stub is handled without an exception on these two
        # endpoints, so inject the same aiohttp error the other paths see.
        _inject_content_type_error(monkeypatch)

    async def run():
        client = _backend(stub_url)
        try:
            ops = client
            if method == "get_status":
                coro = ops.get_hybrid_session_status("s1")
            elif method == "finalize":
                coro = ops.finalize_hybrid_session("s1")
            elif method == "delete":
                coro = ops.delete_session("s1")
            else:
                coro = ops.create_hybrid_session(
                    "problem", {"a": [1, 2]}, {"max_trials": 1}
                )
            return await _capture(coro)
        finally:
            await client.close() if hasattr(client, "close") else None

    exc = asyncio.run(run())
    if method == "get_status":
        # control: the message itself must also be clean
        assert CANARY not in str(exc)
    _assert_clean(exc)


@pytest.mark.parametrize("method", ["next_trial", "finalize_optimization"])
def test_cloud_client_network_error_chain_has_no_key(stub_url, method):
    async def run():
        client = TraigentCloudClient(api_key=CANARY, base_url=stub_url)

        async def _headers(*_a, **_k):
            return {"X-API-Key": CANARY, "Content-Type": "application/json"}

        client._get_headers = _headers  # type: ignore[method-assign]
        try:
            if method == "next_trial":
                coro = client.get_next_trial("s1")
            else:
                coro = client.finalize_optimization("s1")
            return await _capture(coro)
        finally:
            await client.__aexit__(None, None, None)

    exc = asyncio.run(run())
    _assert_clean(exc)
