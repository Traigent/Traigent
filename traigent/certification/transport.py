"""Transport seam for the G1 certificate endpoints.

The bind flow talks to the Backend only through :class:`G1Transport`, so tests
drive it with a recording fake and the default :class:`BackendG1Transport`
reuses the existing authenticated, egress-gated Backend client. Failures are
closed: only an HTTP status and a plain error token survive, never a body.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol, cast
from urllib.parse import urlparse, urlunparse

__all__ = ["BackendG1Transport", "G1RequestError", "G1Response", "G1Transport"]

_SAFE_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")


class G1RequestError(Exception):
    """A G1 request failed. ``status`` is None for transport faults; ``code`` is a plain token."""

    def __init__(self, status: int | None, code: str | None) -> None:
        self.status = status
        self.code = code
        super().__init__(f"g1 request failed (status={status}, code={code})")


@dataclass(frozen=True)
class G1Response:
    status: int
    body: Any


class G1Transport(Protocol):
    def request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> G1Response: ...


_FORBIDDEN_PATH_SUBSTRINGS: Final = ("..", "//", "?", "#", "%", "\\")


def _path_is_safe(path: str) -> bool:
    """A closed request-path shape: absolute, no traversal, query, fragment or escapes."""
    if not isinstance(path, str) or not path.startswith("/"):
        return False
    if any(token in path for token in _FORBIDDEN_PATH_SUBSTRINGS):
        return False
    return all(ch.isprintable() and not ch.isspace() for ch in path)


class BackendG1Transport:
    """Default transport over ``BackendIntegratedClient`` auth and the transport gate."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def _url(self, path: str) -> str | None:
        from traigent.cloud.url_security import is_plain_http_origin_parts

        if not _path_is_safe(path):
            return None
        parsed = urlparse(str(self._client.api_base_url))
        if not is_plain_http_origin_parts(parsed):
            return None
        return cast(
            str,
            urlunparse(parsed._replace(path=path, params="", query="", fragment="")),
        )

    def request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> G1Response:
        from traigent.labels.transport_gate import (
            TransportGateRefusal,
            freeze_json_body,
            observe_requests_response,
            preflight_requests,
        )

        self._client._raise_if_backend_egress_disabled("call the certificate service")
        import requests

        url = self._url(path)
        if not url:
            raise G1RequestError(None, "invalid_backend_url")
        headers = self._client._get_sync_auth_headers(target="backend")
        headers["Cache-Control"] = "no-store"
        payload = None
        if body is not None:
            payload = freeze_json_body(dict(body))
        failed = False
        response: Any = None
        parsed: Any = None
        status = 0
        code: str | None = None
        try:
            preflight_requests(method, url, json=payload)
            response = requests.request(  # nosec B113 - timeout is provided
                method,
                url,
                headers=headers,
                json=payload,
                timeout=(5.0, float(getattr(self._client, "timeout", 30.0))),
                allow_redirects=False,
            )
            observe_requests_response(response)
            status = int(response.status_code)
            try:
                parsed = response.json()
            except ValueError:
                parsed = None
        except TransportGateRefusal:
            raise
        except Exception:
            failed = True
        if failed:
            raise G1RequestError(None, "transport_error")
        if status >= 400 or status == 0:
            if isinstance(parsed, dict):
                candidate = parsed.get("error_code")
                if isinstance(candidate, str) and _SAFE_CODE_RE.fullmatch(candidate):
                    code = candidate
            raise G1RequestError(status, code)
        return G1Response(status, parsed)
