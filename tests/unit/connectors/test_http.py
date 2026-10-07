from dataclasses import dataclass

import pytest

from traigent.connectors.http import (
    ConnectionCredentials,
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
