"""Small customer-side HTTPS kernel with bounded retry and paging behavior."""

from __future__ import annotations

import asyncio
from http.cookiejar import CookieJar, DefaultCookiePolicy
from dataclasses import dataclass, field
from datetime import UTC
from email.utils import parsedate_to_datetime
import math
import random
import threading
import time
from typing import Any, Protocol
from collections.abc import Callable, Coroutine, Mapping
from urllib.parse import urljoin, urlsplit


@dataclass(frozen=True, slots=True, repr=False)
class ConnectionCredentials:
    authorization: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.authorization) is not str or not self.authorization:
            raise ValueError("authorization credential must be nonempty")


@dataclass(frozen=True, slots=True, repr=False)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str] = field(repr=False)
    content: bytes = field(repr=False)


class DeadlineExceeded(TimeoutError):
    """Closed failure used when a connector call exceeds its total budget."""


class Transport(Protocol):
    def send(
        self, method: str, url: str, *, headers: Mapping[str, str], timeout: float
    ) -> HttpResponse: ...


class _RejectCookies(DefaultCookiePolicy):
    """Prevent bearer-authenticated connector sessions from accepting cookies."""

    def set_ok(self, cookie: Any, request: Any) -> bool:
        return False


class _HttpxTransport:
    def __init__(self, transport: Any = None) -> None:
        import httpx

        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            transport = _AsyncTransportAdapter(transport)
        self._client = httpx.AsyncClient(
            cookies=CookieJar(policy=_RejectCookies()),
            follow_redirects=False,
            transport=transport,
        )

    async def send(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        async with self._client.stream(
            method, url, headers=headers, timeout=timeout
        ) as response:
            chunks = [chunk async for chunk in response.aiter_bytes()]
            return HttpResponse(
                response.status_code, dict(response.headers), b"".join(chunks)
            )


class _AsyncTransportAdapter:
    """Adapt legacy test doubles without putting production I/O on a sync client."""

    def __init__(self, transport: Any) -> None:
        self._transport = transport

    async def handle_async_request(self, request: Any) -> Any:
        return await asyncio.to_thread(self._transport.handle_request, request)


class HttpKernel:
    def __init__(
        self,
        base_url: str,
        credentials: ConnectionCredentials,
        *,
        transport: Transport | None = None,
        deadline: float = 30.0,
        timeout: float = 10.0,
        max_retries: int = 2,
        backoff_base: float = 0.1,
        max_backoff: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme.lower() != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "base URL must be an https URL without embedded credentials"
            )
        try:
            base_port = parsed.port if parsed.port is not None else 443
        except ValueError as error:
            raise ValueError("base URL port is invalid") from error
        if (
            any(
                not math.isfinite(value)
                for value in (deadline, timeout, backoff_base, max_backoff)
            )
            or deadline <= 0
            or timeout <= 0
            or max_retries < 0
            or backoff_base < 0
            or max_backoff < 0
        ):
            raise ValueError("deadline, timeout, retry, and backoff bounds are invalid")
        self._base_url = base_url.rstrip("/") + "/"
        self._origin = (
            parsed.scheme.lower(),
            parsed.hostname.lower(),
            base_port,
        )
        self._credentials = credentials
        self._transport = self._new_transport(transport)
        self._deadline = deadline
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._max_backoff = max_backoff
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleep = sleep

    @staticmethod
    def _new_transport(
        transport: Transport | Any | None,
    ) -> Transport | _HttpxTransport:
        if transport is None:
            return _HttpxTransport()
        try:
            import httpx
        except ImportError:
            return transport
        if isinstance(transport, (httpx.BaseTransport, httpx.AsyncBaseTransport)):
            # Each kernel owns a client and a rejecting cookie jar, even when
            # its low-level transport is shared by several connections.
            return _HttpxTransport(transport)
        return transport

    @staticmethod
    def _origin_for(parsed: Any) -> tuple[str, str, int]:
        if parsed.username or parsed.password:
            raise ValueError("request URL must not contain embedded credentials")
        try:
            port = parsed.port if parsed.port is not None else 443
        except ValueError as error:
            raise ValueError("request URL port is invalid") from error
        return (parsed.scheme.lower(), (parsed.hostname or "").lower(), port)

    def _retry_after_delay(self, value: str) -> float | None:
        try:
            delay = float(value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, IndexError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                # RFC 9110 permits asctime-date, whose historical spelling
                # has no explicit zone and is defined as GMT.
                retry_at = retry_at.replace(tzinfo=UTC)
            delay = retry_at.timestamp() - self._wall_clock()
        if not math.isfinite(delay):
            return None
        return min(self._max_backoff, max(0.0, delay))

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        return self._run_async(self._request_async(method, path, headers, timeout))

    @staticmethod
    def _run_async(coroutine: Coroutine[Any, Any, HttpResponse]) -> HttpResponse:
        """Run the async kernel without ever nesting an event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        result: list[HttpResponse] = []
        error: list[BaseException] = []

        def run_in_worker() -> None:
            try:
                result.append(asyncio.run(coroutine))
            except BaseException as caught:
                error.append(caught)

        worker = threading.Thread(target=run_in_worker, daemon=True)
        worker.start()
        worker.join()
        if error:
            raise error[0]
        return result[0]

    async def _request_async(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str] | None,
        timeout: float | None,
    ) -> HttpResponse:
        try:
            return await asyncio.wait_for(
                self._request_with_retries(method, path, headers, timeout),
                timeout=self._deadline,
            )
        except TimeoutError as error:
            raise DeadlineExceeded("connector request deadline exceeded") from error

    async def _request_with_retries(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str] | None,
        timeout: float | None,
    ) -> HttpResponse:
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("request timeout must be finite and positive")
        url = urljoin(self._base_url, path.lstrip("/"))
        parsed = urlsplit(url)
        if self._origin_for(parsed) != self._origin:
            raise ValueError("request URL must remain on the configured host")
        authority_headers = {
            "host",
            ":authority",
            "forwarded",
            "x-forwarded-host",
            "x-original-host",
        }
        if any(name.lower() in authority_headers for name in (headers or {})):
            raise ValueError("request headers must not override authority")
        end = self._clock() + self._deadline
        request_headers = {
            name: value
            for name, value in (headers or {}).items()
            if name.lower() not in {"authorization", "cookie"}
        }
        request_headers["Authorization"] = self._credentials.authorization
        for attempt in range(self._max_retries + 1):
            remaining = end - self._clock()
            if remaining <= 0:
                raise DeadlineExceeded("connector request deadline exceeded")
            attempt_timeout = min(self._timeout, timeout or self._timeout, remaining)
            if isinstance(self._transport, _HttpxTransport):
                response = await self._transport.send(
                    method,
                    url,
                    headers=request_headers,
                    timeout=attempt_timeout,
                )
            else:
                response = await asyncio.to_thread(
                    self._transport.send,
                    method,
                    url,
                    headers=request_headers,
                    timeout=attempt_timeout,
                )
            if self._clock() > end:
                raise DeadlineExceeded("connector request deadline exceeded")
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location") or response.headers.get(
                    "Location"
                )
                target = urlsplit(urljoin(url, location or ""))
                target_origin = self._origin_for(target)
                if target_origin != self._origin:
                    raise ValueError(
                        "refusing cross-host redirect; credentials were not forwarded"
                    )
                raise ValueError(
                    "redirects are not followed by the connector transport"
                )
            if response.status_code not in {429, 500, 502, 503, 504}:
                if response.status_code >= 400:
                    raise RuntimeError(f"connector HTTP status {response.status_code}")
                return response
            if attempt >= self._max_retries:
                raise RuntimeError(
                    f"connector HTTP status {response.status_code} after bounded retries"
                )
            delay = min(self._max_backoff, self._backoff_base * (2**attempt))
            retry_after = response.headers.get("Retry-After") or response.headers.get(
                "retry-after"
            )
            server_delay = self._retry_after_delay(retry_after) if retry_after else None
            if server_delay is None:
                delay = random.uniform(0.0, delay)
            else:
                delay = server_delay
            if self._clock() + delay >= end:
                raise DeadlineExceeded("connector request deadline exceeded")
            if self._sleep is time.sleep:
                await asyncio.sleep(delay)
            else:
                self._sleep(delay)
        raise AssertionError("unreachable")


def iter_pages(
    fetch: Callable[[str | None], Mapping[str, Any]],
    cursor: str | None,
    *,
    max_pages: int = 1000,
):
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    seen: set[str] = set()
    for _ in range(max_pages):
        page = fetch(cursor)
        items = page.get("items")
        if type(items) is not list:
            raise ValueError("page items must be a list")
        yield from items
        cursor = page.get("next_cursor")
        if cursor is None:
            return
        if type(cursor) is not str or cursor in seen:
            raise ValueError("invalid or repeated page cursor")
        seen.add(cursor)
    raise RuntimeError("connector page limit exceeded")
