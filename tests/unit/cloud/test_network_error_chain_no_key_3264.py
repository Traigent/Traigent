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


def _traceback_locals_exceptions(exc: BaseException) -> list[BaseException]:
    """Exceptions held in the locals of frames the raised error's traceback keeps."""
    found: list[BaseException] = []
    tb = exc.__traceback__
    while tb is not None:
        for value in tb.tb_frame.f_locals.values():
            if isinstance(value, BaseException) and value is not exc:
                found.append(value)
        tb = tb.tb_next
    return found


def _assert_clean(exc: BaseException) -> None:
    for held in _traceback_locals_exceptions(exc):
        assert CANARY not in repr(held), type(held).__name__
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


def _response_error(status: int):
    import aiohttp
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    info = aiohttp.RequestInfo(
        url=URL("http://127.0.0.1/x"),
        method="GET",
        headers=CIMultiDictProxy(CIMultiDict({"X-API-Key": CANARY})),
        real_url=URL("http://127.0.0.1/x"),
    )
    return aiohttp.ClientResponseError(info, (), status=status, message="boom")


@pytest.mark.parametrize("method", ["list_sessions", "delete_session"])
def test_standardized_session_errors_hold_no_key(stub_url, monkeypatch, method):
    from traigent.cloud.client import StandardizedClientError

    async def run():
        client = TraigentCloudClient(api_key=CANARY, base_url=stub_url)

        async def _headers(*_a, **_k):
            return {"X-API-Key": CANARY}

        client._get_headers = _headers  # type: ignore[method-assign]
        _inject_content_type_error(monkeypatch)
        try:
            if method == "list_sessions":
                return await _capture(client.list_sessions())
            return await _capture(client.delete_session("s1"))
        finally:
            await client.__aexit__(None, None, None)

    exc = asyncio.run(run())
    assert isinstance(exc, StandardizedClientError)
    assert exc.original_error is None
    assert str(exc).startswith("Failed to")
    for item in _chain(exc):
        assert CANARY not in repr(item)
    for held in _traceback_locals_exceptions(exc):
        assert CANARY not in repr(held), type(held).__name__
    assert exc.__cause__ is None and exc.__context__ is None


@pytest.mark.parametrize(
    "status,text", [(401, "Authentication failed"), (403, "Session ID mismatch")]
)
@pytest.mark.parametrize("call", ["single", "batch"])
def test_optimizer_client_response_error_holds_no_key(stub_url, status, text, call):
    from traigent.cloud.optimizer_client import OptimizerDirectClient

    class _Ctx:
        async def __aenter__(self):
            raise _response_error(status)

        async def __aexit__(self, *a):
            return False

    class _Session:
        def post(self, *a, **k):
            return _Ctx()

    async def run():
        client = OptimizerDirectClient("https://api.example.com", "tok")
        client.session = _Session()  # type: ignore[assignment]
        if call == "single":
            return await _capture(client._submit_single("s1", {"trial_id": "t"}))
        return await _capture(client._submit_batch("s1", [{"trial_id": "t"}]))

    exc = asyncio.run(run())
    assert isinstance(exc, ValueError)
    assert text in str(exc)
    for item in _chain(exc):
        assert CANARY not in repr(item)
    for held in _traceback_locals_exceptions(exc):
        assert CANARY not in repr(held), type(held).__name__
    assert exc.__cause__ is None and exc.__context__ is None


@pytest.mark.parametrize("handler", ["_handle_client_error", "_handle_connector_error"])
def test_api_operations_handlers_hold_no_key(handler):
    from traigent.cloud.api_operations import ApiOperations

    err = _response_error(500)
    with pytest.raises(CloudServiceError) as info:
        try:
            raise err
        except Exception as caught:
            getattr(ApiOperations, handler)(object(), caught)
    exc = info.value
    del err
    assert exc.__cause__ is None and exc.__context__ is None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_name == handler:
            assert not any(
                isinstance(v, BaseException) and v is not exc
                for v in tb.tb_frame.f_locals.values()
            )
        tb = tb.tb_next
