import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import format_datetime
import threading
import time
import errno
import socket

import pytest

from traigent.connectors.http import (
    ConnectionCredentials,
    CleanupDeadlineExceeded,
    DeadlineExceeded,
    HttpKernel,
    HttpResponse,
    iter_pages,
)


@dataclass
class Transport:
    responses: list
    requests: list

    def send(self, method, url, *, headers, timeout):
        self.requests.append((method, url, dict(headers), timeout))
        result = self.responses.pop(0)
        return result


@dataclass
class Clock:
    now: float = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def _test_transport_factory(transport):
    """Private kernel test seam for deterministic, non-network responses."""
    httpx = pytest.importorskip("httpx")

    class ResponseTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            timeout = request.extensions["timeout"]["connect"]
            response = transport.send(
                request.method,
                str(request.url),
                headers=dict(request.headers),
                timeout=timeout,
            )
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=response.content,
                request=request,
            )

    return ResponseTransport


def test_http_kernel_isolates_connection_credentials():
    transport = Transport(
        [HttpResponse(200, {}, b"ok"), HttpResponse(200, {}, b"ok")], []
    )
    one = HttpKernel(
        "https://one.example",
        ConnectionCredentials("Bearer one"),
        _test_transport_factory=_test_transport_factory(transport),
    )
    two = HttpKernel(
        "https://two.example",
        ConnectionCredentials("Bearer two"),
        _test_transport_factory=_test_transport_factory(transport),
    )
    one.request("GET", "/x")
    two.request("GET", "/x")
    assert transport.requests[0][2]["authorization"] == "Bearer one"
    assert transport.requests[1][2]["authorization"] == "Bearer two"


def test_http_kernel_rejects_host_header_override():
    transport = Transport([], [])
    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("Bearer secret"),
        _test_transport_factory=_test_transport_factory(transport),
    )
    for name in ("Host", ":authority", "Forwarded", "X-Forwarded-Host"):
        with pytest.raises(ValueError, match="authority"):
            kernel.request("GET", "/", headers={name: "evil.example"})
    assert transport.requests == []


def test_http_kernel_rejects_userinfo():
    transport = Transport([], [])
    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("Bearer secret"),
        _test_transport_factory=_test_transport_factory(transport),
    )
    with pytest.raises(ValueError, match="credentials"):
        kernel.request("GET", "https://safe.example:443@safe.example/steal")
    assert transport.requests == []


def test_http_kernel_rejects_port_mismatch():
    transport = Transport([], [])
    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("Bearer secret"),
        _test_transport_factory=_test_transport_factory(transport),
    )
    for url in ("https://safe.example:0/x", "https://safe.example:444/x"):
        with pytest.raises(ValueError, match="configured host"):
            kernel.request("GET", url)
    assert transport.requests == []


def test_http_kernel_rejects_non_tls_base_url():
    with pytest.raises(ValueError):
        HttpKernel(
            "http://connector.example",
            ConnectionCredentials("secret"),
            _test_transport_factory=_test_transport_factory(Transport([], [])),
        )


def test_http_kernel_drops_credentials_on_cross_host_redirect():
    transport = Transport(
        [
            HttpResponse(302, {"location": "https://evil.example/steal"}, b""),
            HttpResponse(200, {}, b"forwarded"),
        ],
        [],
    )
    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("Bearer secret"),
        _test_transport_factory=_test_transport_factory(transport),
    )
    with pytest.raises(ValueError, match="cross-host"):
        kernel.request("GET", "/start")
    assert len(transport.requests) == 1
    assert transport.requests[0][1] == "https://safe.example/start"
    assert all("evil.example" not in request[1] for request in transport.requests)


def test_http_kernel_timeout_is_bounded():
    clock = Clock()

    @dataclass
    class DeadlineTransport:
        requests: list

        def send(self, method, url, *, headers, timeout):
            self.requests.append((method, url, dict(headers), timeout))
            clock.now += timeout
            return HttpResponse(503, {}, b"")

    transport = DeadlineTransport([])
    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        _test_transport_factory=_test_transport_factory(transport),
        deadline=2,
        timeout=10,
        backoff_base=0,
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(TimeoutError, match="deadline"):
        kernel.request("GET", "/", timeout=50)
    assert [request[3] for request in transport.requests] == [2]


def test_http_kernel_total_deadline_includes_fragmented_headers():
    httpx = pytest.importorskip("httpx")

    class FragmentedHeadersTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
        """Models a peer which takes several fragments to finish response headers."""

        def handle_request(self, request):
            for _ in range(10):
                time.sleep(0.05)
            return httpx.Response(200, request=request, content=b"ok")

        async def handle_async_request(self, request):
            for _ in range(10):
                await asyncio.sleep(0.05)
            return httpx.Response(200, request=request, content=b"ok")

    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        _test_transport_factory=FragmentedHeadersTransport,
        deadline=0.1,
        timeout=10,
    )
    started = time.monotonic()
    with pytest.raises(DeadlineExceeded):
        kernel.request("GET", "/")
    assert time.monotonic() - started <= 0.35


def test_http_kernel_total_deadline_includes_body_read():
    httpx = pytest.importorskip("httpx")

    class DrippingBodyTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
        def handle_request(self, request):
            return httpx.Response(200, request=request, stream=SyncDripStream())

        async def handle_async_request(self, request):
            return httpx.Response(200, request=request, stream=AsyncDripStream())

    class SyncDripStream(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(10):
                time.sleep(0.5)
                yield b"x"

    class AsyncDripStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(10):
                await asyncio.sleep(0.5)
                yield b"x"

    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        _test_transport_factory=DrippingBodyTransport,
        deadline=0.1,
        timeout=10,
    )
    started = time.monotonic()
    with pytest.raises(DeadlineExceeded):
        kernel.request("GET", "/")
    assert time.monotonic() - started <= 0.35


def test_http_kernel_retries_are_bounded(monkeypatch):
    clock = Clock()
    transport = Transport(
        [
            HttpResponse(503, {}, b""),
            HttpResponse(503, {}, b""),
            HttpResponse(503, {}, b""),
        ],
        [],
    )
    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        _test_transport_factory=_test_transport_factory(transport),
        max_retries=1,
        backoff_base=0.5,
        clock=clock,
        sleep=clock.sleep,
    )
    monkeypatch.setattr("traigent.connectors.http.random.uniform", lambda a, b: b)
    with pytest.raises(RuntimeError, match="503"):
        kernel.request("GET", "/")
    assert len(transport.requests) == 2
    assert clock.now == 0.5


def test_http_kernel_honours_retry_after_forms(monkeypatch):
    clock = Clock()
    wall_now = 1_700_000_000.0
    http_date = format_datetime(datetime.fromtimestamp(wall_now + 30, UTC), usegmt=True)
    monkeypatch.setattr("traigent.connectors.http.random.uniform", lambda a, b: 0.0)
    retry_at = datetime.fromtimestamp(wall_now + 30, UTC)
    rfc850_date = retry_at.strftime("%A, %d-%b-%y %H:%M:%S GMT")
    asctime_date = retry_at.strftime("%a %b %e %H:%M:%S %Y")
    for retry_after in ("30", http_date, rfc850_date, asctime_date):
        transport = Transport(
            [
                HttpResponse(503, {"Retry-After": retry_after}, b""),
                HttpResponse(200, {}, b""),
            ],
            [],
        )
        kernel = HttpKernel(
            "https://safe.example",
            ConnectionCredentials("x"),
            _test_transport_factory=_test_transport_factory(transport),
            max_retries=1,
            max_backoff=3,
            clock=clock,
            wall_clock=lambda: wall_now,
            sleep=clock.sleep,
        )
        before = clock.now
        assert kernel.request("GET", "/").status_code == 200
        assert clock.now - before == 3


def test_http_kernel_never_forwards_cookies_same_host():
    httpx = pytest.importorskip("httpx")
    received_cookies = []

    def handler(request):
        received_cookies.append(request.headers.get("cookie"))
        if len(received_cookies) == 1:
            return httpx.Response(200, headers={"set-cookie": "session=one"})
        return httpx.Response(200)

    shared_transport = httpx.MockTransport(handler)
    one = HttpKernel(
        "https://same.example",
        ConnectionCredentials("Bearer one"),
        _test_transport_factory=lambda: shared_transport,
    )
    two = HttpKernel(
        "https://same.example",
        ConnectionCredentials("Bearer two"),
        _test_transport_factory=lambda: shared_transport,
    )
    one.request("GET", "/", headers={"Cookie": "caller=one"})
    one.request("GET", "/")
    two.request("GET", "/", headers={"Cookie": "caller=two"})
    assert received_cookies == [None, None, None]


def test_transport_boundary_excludes_summary_content_canaries():
    canary = "private-summary-canary"
    transport = Transport([HttpResponse(500, {}, canary.encode())], [])
    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("token"),
        _test_transport_factory=_test_transport_factory(transport),
        max_retries=0,
    )
    with pytest.raises(RuntimeError) as error:
        kernel.request("GET", "/health")
    assert canary not in str(error.value)
    assert canary not in repr(kernel)


def test_http_kernel_repeated_sync_calls_across_loops():
    httpx = pytest.importorskip("httpx")

    class LoopBoundTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.loop = None

        async def handle_async_request(self, request):
            loop = asyncio.get_running_loop()
            if self.loop is None:
                self.loop = loop
            elif self.loop is not loop:
                raise RuntimeError("transport reused across event loops")
            return httpx.Response(200, request=request)

    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        _test_transport_factory=LoopBoundTransport,
    )
    assert kernel.request("GET", "/").status_code == 200
    assert kernel.request("GET", "/").status_code == 200


def test_http_kernel_sync_call_inside_running_loop():
    httpx = pytest.importorskip("httpx")

    class LoopBoundTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.loop = None

        async def handle_async_request(self, request):
            loop = asyncio.get_running_loop()
            if self.loop is None:
                self.loop = loop
            elif self.loop is not loop:
                raise RuntimeError("transport reused across event loops")
            return httpx.Response(200, request=request)

    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        _test_transport_factory=LoopBoundTransport,
    )
    assert kernel.request("GET", "/").status_code == 200

    async def call_sync_api():
        return kernel.request("GET", "/").status_code

    assert asyncio.run(call_sync_api()) == 200


def test_http_kernel_never_forwards_cookies_shared_protocol_transport():
    httpx = pytest.importorskip("httpx")
    received_cookies = []

    class StatefulTransport(httpx.AsyncBaseTransport):
        cookie = None

        async def handle_async_request(self, request):
            if self.cookie is not None:
                request.headers["Cookie"] = self.cookie
            received_cookies.append(request.headers.get("cookie"))
            self.cookie = "session=one"
            return httpx.Response(
                200, headers={"set-cookie": self.cookie}, request=request
            )

    one = HttpKernel(
        "https://same.example",
        ConnectionCredentials("Bearer one"),
        _test_transport_factory=StatefulTransport,
    )
    two = HttpKernel(
        "https://same.example",
        ConnectionCredentials("Bearer two"),
        _test_transport_factory=StatefulTransport,
    )
    one.request("GET", "/")
    two.request("GET", "/")
    assert received_cookies == [None, None]


def test_http_kernel_concurrent_calls_have_distinct_transports():
    httpx = pytest.importorskip("httpx")
    created = []
    responses = []

    class DelayedTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            created.append(self)

        async def handle_async_request(self, request):
            await asyncio.sleep(0.02)
            return httpx.Response(200, request=request)

    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        _test_transport_factory=DelayedTransport,
    )

    def request():
        responses.append(kernel.request("GET", "/").status_code)

    workers = [threading.Thread(target=request) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert responses == [200, 200]
    assert len(created) == 2
    assert created[0] is not created[1]


def test_http_kernel_sync_total_deadline_fragmented_headers():
    httpx = pytest.importorskip("httpx")
    closed = []

    class BlockingHeadersTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            for _ in range(10):
                await asyncio.sleep(0.05)
            return httpx.Response(200, request=request)

        async def aclose(self):
            closed.append(True)

    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        _test_transport_factory=BlockingHeadersTransport,
        deadline=0.1,
        timeout=10,
    )
    started = time.monotonic()
    with pytest.raises(DeadlineExceeded):
        kernel.request("GET", "/")
    assert time.monotonic() - started <= 0.35
    assert closed == [True]


def test_http_kernel_cancellation_closes_streams():
    httpx = pytest.importorskip("httpx")
    streams = []

    class DrippingStream(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = False

        async def __aiter__(self):
            while True:
                await asyncio.sleep(1)
                yield b"x"

        async def aclose(self):
            self.closed = True

    class DrippingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            stream = DrippingStream()
            streams.append(stream)
            return httpx.Response(200, request=request, stream=stream)

    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        deadline=0.05,
        _test_transport_factory=DrippingTransport,
    )
    with pytest.raises(DeadlineExceeded):
        kernel.request("GET", "/")
    assert streams and streams[0].closed


def test_http_kernel_timeout_cleanup_completes_within_budget():
    httpx = pytest.importorskip("httpx")
    workers = []
    closed = []

    class SlowClosingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(10)
            yield b"x"

        async def aclose(self):
            await asyncio.sleep(0.06)
            closed.append("stream")

    class SlowClosingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            workers.append(threading.current_thread())
            return httpx.Response(200, request=request, stream=SlowClosingStream())

        async def aclose(self):
            await asyncio.sleep(0.06)
            closed.append("client")

    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        deadline=0.5,
        _test_transport_factory=SlowClosingTransport,
    )
    started = time.monotonic()
    with pytest.raises(DeadlineExceeded):
        kernel.request("GET", "/")
    elapsed = time.monotonic() - started
    assert elapsed <= 0.5
    assert closed == ["stream", "client"]
    assert workers and not workers[0].is_alive()


@pytest.mark.parametrize("successful_response", [False, True])
def test_http_kernel_reports_unfinished_cleanup(successful_response):
    httpx = pytest.importorskip("httpx")
    release_cleanup = threading.Event()
    cleanup_started = threading.Event()
    cleanup_finished = threading.Event()
    workers = []

    class UnfinishedCleanupTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            workers.append(threading.current_thread())
            if not successful_response:
                await asyncio.sleep(10)
            return httpx.Response(200, request=request)

        async def aclose(self):
            cleanup_started.set()
            while not release_cleanup.is_set():
                await asyncio.sleep(0.001)
            cleanup_finished.set()

    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        deadline=0.1,
        _test_transport_factory=UnfinishedCleanupTransport,
    )
    started = time.monotonic()
    try:
        with pytest.raises(CleanupDeadlineExceeded, match="cleanup deadline"):
            kernel.request("GET", "/")
        assert time.monotonic() - started <= 0.4
        assert cleanup_started.is_set()
        assert not cleanup_finished.is_set()
        assert workers and workers[0].is_alive()
    finally:
        release_cleanup.set()
        for worker in workers:
            worker.join(timeout=1)
    assert cleanup_finished.is_set()
    assert all(not worker.is_alive() for worker in workers)


def test_http_kernel_timeout_closes_real_socket_127_0_0_1():
    httpx = pytest.importorskip("httpx")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
    except OSError as error:
        listener.close()
        if error.errno in (errno.EPERM, errno.EACCES):
            pytest.skip("unverified: sandbox disallows real 127.0.0.1 sockets")
        raise
    listener.settimeout(2)
    port = listener.getsockname()[1]
    disconnected = threading.Event()
    server_errors = []
    active_connections = []
    workers = []

    def serve():
        try:
            connection, _ = listener.accept()
            with connection:
                active_connections.append(connection)
                connection.settimeout(2)
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = connection.recv(4096)
                    assert chunk, "client closed before sending complete headers"
                    request += chunk
                connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
                assert connection.recv(1) == b""
                active_connections.remove(connection)
                disconnected.set()
        except BaseException as error:
            server_errors.append(error)

    class LocalNetworkTransport(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request):
            workers.append(threading.current_thread())
            # Keep the public HTTPS contract; only this private test transport
            # maps the request to a real plaintext loopback HTTP server.
            request.url = httpx.URL(f"http://127.0.0.1:{port}/")
            return await super().handle_async_request(request)

    server = threading.Thread(target=serve)
    server.start()
    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("synthetic"),
        deadline=0.6,
        _test_transport_factory=LocalNetworkTransport,
    )
    try:
        started = time.monotonic()
        with pytest.raises(DeadlineExceeded):
            kernel.request("GET", "/")
        assert time.monotonic() - started <= 0.55
        assert disconnected.wait(0.1), "server did not observe client socket EOF"
        assert active_connections == []
        assert workers and not workers[0].is_alive()
    finally:
        server.join(timeout=2.5)
        listener.close()
    assert not server.is_alive()
    assert server_errors == []


def test_http_kernel_cancel_races_loop_close(monkeypatch):
    httpx = pytest.importorskip("httpx")
    release_cleanup = threading.Event()
    loop_closed = threading.Event()
    workers = []
    new_event_loop = asyncio.new_event_loop
    caller = threading.current_thread()

    def racing_event_loop():
        loop = new_event_loop()
        is_closed = loop.is_closed
        close = loop.close

        def racing_is_closed():
            result = is_closed()
            if threading.current_thread() is caller and not result:
                release_cleanup.set()
                # Force the worker to close between the check and scheduling
                # unless the same lock protects both that check and close.
                loop_closed.wait(0.05)
            return result

        def tracked_close():
            close()
            loop_closed.set()

        monkeypatch.setattr(loop, "is_closed", racing_is_closed)
        monkeypatch.setattr(loop, "close", tracked_close)
        return loop

    class RacingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            workers.append(threading.current_thread())
            await asyncio.sleep(10)
            return httpx.Response(200, request=request)

        async def aclose(self):
            while not release_cleanup.is_set():
                await asyncio.sleep(0.001)

    monkeypatch.setattr(asyncio, "new_event_loop", racing_event_loop)
    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        deadline=0.1,
        _test_transport_factory=RacingTransport,
    )
    try:
        with pytest.raises(DeadlineExceeded):
            kernel.request("GET", "/")
        assert loop_closed.is_set()
        assert workers and not workers[0].is_alive()
    finally:
        release_cleanup.set()


def test_http_kernel_aclose_releases_pool():
    httpx = pytest.importorskip("httpx")
    closed = []

    class ClosableTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, request=request)

        async def aclose(self):
            closed.append(True)

    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("x"),
        _test_transport_factory=ClosableTransport,
    )
    assert kernel.request("GET", "/").status_code == 200
    asyncio.run(kernel.aclose())
    asyncio.run(kernel.aclose())
    assert closed == [True]


def test_cursor_paging_yields_items():
    pages = iter_pages(lambda cursor: {"items": ["first"], "next_cursor": None}, None)
    assert list(pages) == ["first"]
