"""Small customer-side HTTPS kernel with bounded retry and paging behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
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


class Transport(Protocol):
    def send(
        self, method: str, url: str, *, headers: Mapping[str, str], timeout: float
    ) -> HttpResponse: ...


class _HttpxTransport:
    def __init__(self) -> None:
        import httpx

        self._client = httpx.Client(follow_redirects=False)

    def send(
        self, method: str, url: str, *, headers: Mapping[str, str], timeout: float
    ) -> HttpResponse:
        response = self._client.request(method, url, headers=headers, timeout=timeout)
        return HttpResponse(
            response.status_code, dict(response.headers), response.content
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
            parsed.port or 443,
        )
        self._credentials = credentials
        self._transport = transport or _HttpxTransport()
        self._deadline = deadline
        self._timeout = timeout
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._max_backoff = max_backoff
        self._clock = clock
        self._sleep = sleep

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
        if (
            parsed.scheme.lower(),
            (parsed.hostname or "").lower(),
            parsed.port or 443,
        ) != self._origin:
            raise ValueError("request URL must remain on the configured host")
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
                raise TimeoutError("connector request deadline exceeded")
            response = self._transport.send(
                method,
                url,
                headers=request_headers,
                timeout=min(self._timeout, timeout or self._timeout, remaining),
            )
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location") or response.headers.get(
                    "Location"
                )
                target = urlsplit(urljoin(url, location or ""))
                target_origin = (
                    target.scheme.lower(),
                    (target.hostname or "").lower(),
                    target.port or 443,
                )
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
            if retry_after:
                try:
                    delay = min(self._max_backoff, max(0.0, float(retry_after)))
                except ValueError:
                    pass
            delay = random.uniform(0.0, delay)
            if self._clock() + delay >= end:
                raise TimeoutError("connector request deadline exceeded")
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
