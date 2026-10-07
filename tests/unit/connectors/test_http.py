import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import format_datetime
import time

import pytest

from traigent.connectors.http import (
    ConnectionCredentials,
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


def test_http_kernel_isolates_connection_credentials():
    transport = Transport(
        [HttpResponse(200, {}, b"ok"), HttpResponse(200, {}, b"ok")], []
    )
    one = HttpKernel(
        "https://one.example", ConnectionCredentials("Bearer one"), transport=transport
    )
    two = HttpKernel(
        "https://two.example", ConnectionCredentials("Bearer two"), transport=transport
    )
    one.request("GET", "/x")
    two.request("GET", "/x")
    assert transport.requests[0][2]["Authorization"] == "Bearer one"
    assert transport.requests[1][2]["Authorization"] == "Bearer two"


def test_http_kernel_rejects_host_header_override():
    transport = Transport([], [])
    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("Bearer secret"),
        transport=transport,
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
        transport=transport,
    )
    with pytest.raises(ValueError, match="credentials"):
        kernel.request("GET", "https://safe.example:443@safe.example/steal")
    assert transport.requests == []


def test_http_kernel_rejects_port_mismatch():
    transport = Transport([], [])
    kernel = HttpKernel(
        "https://safe.example",
        ConnectionCredentials("Bearer secret"),
        transport=transport,
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
            transport=Transport([], []),
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
        transport=transport,
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
        transport=transport,
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
        transport=FragmentedHeadersTransport(),
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
        transport=DrippingBodyTransport(),
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
        transport=transport,
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
            transport=transport,
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
        transport=shared_transport,
    )
    two = HttpKernel(
        "https://same.example",
        ConnectionCredentials("Bearer two"),
        transport=shared_transport,
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
        transport=transport,
        max_retries=0,
    )
    with pytest.raises(RuntimeError) as error:
        kernel.request("GET", "/health")
    assert canary not in str(error.value)
    assert canary not in repr(kernel)


def test_cursor_paging_yields_items():
    pages = iter_pages(lambda cursor: {"items": ["first"], "next_cursor": None}, None)
    assert list(pages) == ["first"]
