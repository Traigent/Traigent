"""Small customer-side HTTPS kernel with bounded retry and paging behavior."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, TimeoutError as FutureTimeout
from http.cookiejar import CookieJar, DefaultCookiePolicy
from dataclasses import dataclass, field
from datetime import UTC
from email.utils import parsedate_to_datetime
import math
import random
import threading
import time
from typing import Any
from collections.abc import Callable, Mapping
from urllib.parse import urljoin, urlsplit

import httpx


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


class CleanupDeadlineExceeded(DeadlineExceeded):
    """The worker did not finish cleanup within the bounded shutdown grace."""


class _RejectCookies(DefaultCookiePolicy):
    """Prevent bearer-authenticated connector sessions from accepting cookies."""

    def set_ok(self, cookie: Any, request: Any) -> bool:
        return False


class _HeaderGuardTransport(httpx.AsyncBaseTransport):
    """Final request boundary for connection authority and cookie isolation."""

    _STRIPPED_HEADERS = {
        "cookie",
        "host",
        ":authority",
        "forwarded",
        "x-forwarded-host",
        "x-original-host",
    }

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport

    @staticmethod
    def _host_for(url: httpx.URL) -> str:
        default_port = 443 if url.scheme == "https" else 80
        return (
            url.host if url.port in (None, default_port) else f"{url.host}:{url.port}"
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        for name in tuple(request.headers):
            if name.lower() in self._STRIPPED_HEADERS:
                del request.headers[name]
        request.headers["Host"] = self._host_for(request.url)
        response = await self._transport.handle_async_request(request)
        for name in tuple(response.headers):
            if name.lower() == "set-cookie":
                del response.headers[name]
        return response

    async def aclose(self) -> None:
        await self._transport.aclose()


class HttpKernel:
    def __init__(
        self,
        base_url: str,
        credentials: ConnectionCredentials,
        *,
        deadline: float = 30.0,
        timeout: float = 10.0,
        max_retries: int = 2,
        backoff_base: float = 0.1,
        max_backoff: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        _test_transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
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
        self._test_transport_factory = _test_transport_factory
        self._deadline = deadline
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._max_backoff = max_backoff
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleep = sleep

    def _new_transport(self) -> _HeaderGuardTransport:
        """Create a fresh low-level transport inside the call's worker loop.

        Production calls have no injection seam.  The private factory exists only
        for socket-free unit tests and is deliberately invoked for every call.
        """
        transport = (
            self._test_transport_factory()
            if self._test_transport_factory is not None
            else httpx.AsyncHTTPTransport(retries=0)
        )
        if not isinstance(transport, httpx.AsyncBaseTransport):
            raise TypeError("test transport factory must return an async transport")
        return _HeaderGuardTransport(transport)

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
        """Execute one call in its own loop and bound caller-side shutdown.

        Reserve time inside the deadline for stream and client cleanup. The
        caller also establishes loop closure and worker termination before
        returning; unfinished cleanup raises a distinct bounded failure.
        """
        started_at = time.monotonic()
        deadline_at = started_at + self._deadline
        io_deadline_at = deadline_at - min(0.2, self._deadline / 2)
        shutdown_at = deadline_at + 0.25
        completed: Future[HttpResponse] = Future()
        finished = threading.Event()
        cancel_requested = threading.Event()
        closing_client = threading.Event()
        state_lock = threading.Lock()
        state: dict[str, Any] = {"loop": None, "task": None}

        def cancel_task() -> None:
            task = state["task"]
            if (
                task is not None
                and not task.done()
                and not task.cancelling()
                and not closing_client.is_set()
            ):
                task.cancel()

        def run_in_worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def run_call() -> HttpResponse:
                transport = self._new_transport()
                async with httpx.AsyncClient(
                    cookies=CookieJar(policy=_RejectCookies()),
                    follow_redirects=False,
                    transport=transport,
                ) as client:
                    try:
                        return await self._request_async(
                            client, method, path, headers, timeout, io_deadline_at
                        )
                    finally:
                        closing_client.set()

            task = loop.create_task(run_call())
            with state_lock:
                state["loop"] = loop
                state["task"] = task
                if cancel_requested.is_set():
                    loop.call_soon(task.cancel)
            try:
                response = loop.run_until_complete(task)
            except BaseException as error:
                completed.set_exception(error)
            else:
                completed.set_result(response)
            finally:
                with state_lock:
                    try:
                        loop.close()
                    finally:
                        finished.set()

        worker = threading.Thread(target=run_in_worker, daemon=True)
        worker.start()
        try:
            return completed.result(timeout=max(0.0, deadline_at - time.monotonic()))
        except FutureTimeout as error:
            cancel_requested.set()
            with state_lock:
                loop = state["loop"]
                if loop is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(cancel_task)
            raise DeadlineExceeded("connector request deadline exceeded") from error
        finally:
            if not finished.wait(timeout=max(0.0, shutdown_at - time.monotonic())):
                raise CleanupDeadlineExceeded(
                    "connector cleanup deadline exceeded; worker is unfinished"
                )
            worker.join(timeout=max(0.0, shutdown_at - time.monotonic()))
            if worker.is_alive():
                raise CleanupDeadlineExceeded(
                    "connector cleanup deadline exceeded; worker is unfinished"
                )

    async def aclose(self) -> None:
        """Idempotent lifecycle hook; calls do not retain clients or pools."""

    async def _request_async(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        headers: Mapping[str, str] | None,
        timeout: float | None,
        deadline_at: float,
    ) -> HttpResponse:
        try:
            remaining = deadline_at - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                return await self._request_with_retries(
                    client, method, path, headers, timeout
                )
        except TimeoutError as error:
            raise DeadlineExceeded("connector request deadline exceeded") from error

    async def _request_with_retries(
        self,
        client: httpx.AsyncClient,
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
            async with client.stream(
                method, url, headers=request_headers, timeout=attempt_timeout
            ) as raw_response:
                chunks = [chunk async for chunk in raw_response.aiter_bytes()]
                response = HttpResponse(
                    raw_response.status_code,
                    dict(raw_response.headers),
                    b"".join(chunks),
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
