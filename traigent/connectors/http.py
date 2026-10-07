"""Small customer-side HTTPS kernel with bounded retry and paging behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
import random
import time
import math
from typing import Any, Protocol
from collections.abc import Callable, Mapping
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


class _HttpxTransport:
    def __init__(self, transport: Any = None, *, clock: Callable[[], float]) -> None:
        import httpx

        self._client = httpx.Client(follow_redirects=False, transport=transport)
        self._clock = clock

    def send(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
        deadline: float | None = None,
    ) -> HttpResponse:
        with self._client.stream(
            method, url, headers=headers, timeout=timeout
        ) as response:
            chunks: list[bytes] = []
            for chunk in response.iter_bytes():
                if deadline is not None and self._clock() > deadline:
                    raise DeadlineExceeded("connector request deadline exceeded")
                chunks.append(chunk)
            if deadline is not None and self._clock() > deadline:
                raise DeadlineExceeded("connector request deadline exceeded")
            return HttpResponse(
                response.status_code, dict(response.headers), b"".join(chunks)
            )

    def close(self) -> None:
        self._client.close()


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
        self._transport = self._new_transport(transport, clock)
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
        transport: Transport | Any | None, clock: Callable[[], float]
    ) -> Transport:
        if transport is None:
            return _HttpxTransport(clock=clock)
        try:
            import httpx
        except ImportError:
            return transport
        if isinstance(transport, httpx.BaseTransport):
            # A shared HTTPX transport is safe to reuse only underneath a new
            # client for each connection, which gives each its own cookie jar.
            return _HttpxTransport(transport, clock=clock)
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
                return None
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
            if name.lower() != "authorization"
        }
        request_headers["Authorization"] = self._credentials.authorization
        for attempt in range(self._max_retries + 1):
            remaining = end - self._clock()
            if remaining <= 0:
                raise DeadlineExceeded("connector request deadline exceeded")
            attempt_timeout = min(self._timeout, timeout or self._timeout, remaining)
            if isinstance(self._transport, _HttpxTransport):
                response = self._transport.send(
                    method,
                    url,
                    headers=request_headers,
                    timeout=attempt_timeout,
                    deadline=end,
                )
            else:
                response = self._transport.send(
                    method, url, headers=request_headers, timeout=attempt_timeout
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
