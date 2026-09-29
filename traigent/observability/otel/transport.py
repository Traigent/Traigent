"""HTTP transport for OTLP/HTTP protobuf, reusing the SDK's no-redirect opener."""

from __future__ import annotations

import gzip
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from traigent.observability.client import _NoRedirectHandler, _parse_retry_after


class TransportError(Exception):
    """Network-level failure (connect/reset/timeout); retryable."""


@dataclass
class TransportResponse:
    status: int
    body: bytes = b""
    retry_after: float | None = None
    headers: dict[str, str] = field(default_factory=dict)


class Transport(Protocol):
    def post(self, body: bytes, *, timeout: float) -> TransportResponse: ...


class UrllibTransport:
    """POST ``application/x-protobuf`` (gzip) without following redirects."""

    def __init__(self, url: str, headers: dict[str, str], *, gzip_body: bool = True):
        self._url = url
        self._headers = dict(headers)
        self._gzip = gzip_body
        self._opener = urllib.request.build_opener(_NoRedirectHandler)

    def post(self, body: bytes, *, timeout: float) -> TransportResponse:
        headers = {"Content-Type": "application/x-protobuf", **self._headers}
        payload = body
        if self._gzip:
            payload = gzip.compress(body, compresslevel=5)
            headers["Content-Encoding"] = "gzip"
        req = urllib.request.Request(
            self._url, data=payload, headers=headers, method="POST"
        )
        try:
            with self._opener.open(req, timeout=max(0.05, timeout)) as resp:  # nosec B310
                return TransportResponse(
                    status=resp.status,
                    body=resp.read(1 << 20),
                    retry_after=_parse_retry_after(resp.headers),
                )
        except urllib.error.HTTPError as exc:
            try:
                data = exc.read(1 << 16)
            except Exception:  # pragma: no cover - body is best effort
                data = b""
            return TransportResponse(
                status=exc.code,
                body=data,
                retry_after=_parse_retry_after(exc.headers),
            )
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise TransportError(type(exc).__name__) from None
        except Exception as exc:  # http.client errors etc.
            raise TransportError(type(exc).__name__) from None


