"""GitHub repository snapshot store using the Contents API on an existing branch."""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

import requests

from traigent.snapshot.errors import SnapshotErrorCode, SnapshotStoreError
from traigent.snapshot.store import CreateResult, require_secure_endpoint, validate_key

__all__ = ["GitHubStore"]

_BACKEND = "github"
_REPO_RE = re.compile(r"[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}")
_BRANCH_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*")
_ENV_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_TIMEOUT_SECONDS = 30
_MAX_PART_BYTES = 50 * 1024**2  # SDK policy cap; GitHub warns >50 MiB, rejects >100 MiB
_MAX_READ_BYTES = 100 * 1024**2
_MAX_CONFLICT_RETRIES = 3
_MAX_RATE_ATTEMPTS = 4
_MAX_TOTAL_WAIT_SECONDS = 60.0
_MAX_REDIRECTS = 3
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
_RAW_ACCEPT = "application/vnd.github.raw+json"
_JSON_ACCEPT = "application/vnd.github+json"
_API_VERSION = "2022-11-28"
_COMMIT_MESSAGE = "traigent execution snapshot object"


def _error(code: SnapshotErrorCode) -> SnapshotStoreError:
    return SnapshotStoreError(code, backend=_BACKEND)


def _header_float(response: Any, name: str) -> float | None:
    value = response.headers.get(name)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


class GitHubStore:
    """Snapshot objects under ``github://<owner>/<repo>@<branch>/<prefix>``.

    The branch must already exist (an empty repository or missing branch is
    ``NOT_FOUND``). The repository must be private unless ``allow_public=True``
    (``UNSAFE_DESTINATION`` otherwise); this is checked once, before the first
    request that touches content. Objects are created with a Contents API PUT
    that carries no ``sha``, so an existing file is never overwritten: GitHub
    refuses, and the store reads and compares the bytes.

    The token is read from ``os.environ[token_env]`` at call time and only
    ever placed in the request header; it is not stored, shown in ``repr`` or
    carried by an error. Redirects are followed only within the API host.
    """

    max_object_bytes: int | None = _MAX_PART_BYTES

    def __init__(
        self,
        repo: str,
        branch: str,
        prefix: str = "",
        *,
        token_env: str = "GITHUB_TOKEN",
        api_url: str = "https://api.github.com",
        allow_public: bool = False,
        allow_insecure_local: bool = False,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if (
            not isinstance(repo, str)
            or _REPO_RE.fullmatch(repo) is None
            or any(part in (".", "..") for part in repo.split("/"))
        ):
            raise ValueError("repo must be 'owner/name'")
        if (
            not isinstance(branch, str)
            or _BRANCH_RE.fullmatch(branch) is None
            or ".." in branch
            or branch.endswith((".lock", "."))
        ):
            raise ValueError("invalid branch name")
        if not isinstance(token_env, str) or _ENV_RE.fullmatch(token_env) is None:
            raise ValueError("token_env must be an environment variable name")
        clean = prefix.strip("/") if isinstance(prefix, str) else ""
        if clean:
            validate_key(clean)
        require_secure_endpoint(
            api_url, allow_insecure_local=allow_insecure_local, backend=_BACKEND
        )
        parts = urlsplit(api_url)
        if parts.username or parts.password or parts.query or parts.fragment:
            raise _error(SnapshotErrorCode.UNSAFE_DESTINATION) from None
        self._repo = repo
        self._branch = branch
        self._prefix = clean
        self._token_env = token_env
        self._api_url = api_url.rstrip("/")
        self._origin = (parts.scheme, parts.hostname, parts.port)
        self._allow_public = bool(allow_public)
        self._session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._ready = False
        self._write_lock = threading.Lock()

    def __repr__(self) -> str:
        return self.locator()

    def locator(self) -> str:
        base = f"github://{self._repo}@{self._branch}"
        return f"{base}/{self._prefix}" if self._prefix else base

    def _path(self, key: str) -> str:
        validate_key(key)
        return f"{self._prefix}/{key}" if self._prefix else key

    # ------------------------------------------------------------ transport

    def _headers(self, accept: str) -> dict[str, str]:
        token = os.environ.get(self._token_env)
        if not token:
            raise _error(SnapshotErrorCode.AUTH) from None
        return {
            "Authorization": f"Bearer {token}",
            "Accept": accept,
            "X-GitHub-Api-Version": _API_VERSION,
        }

    def _same_origin(self, url: str) -> bool:
        try:
            parts = urlsplit(url)
            return (parts.scheme, parts.hostname, parts.port) == self._origin
        except ValueError:
            return False

    def _wait_seconds(self, response: Any) -> float | None:
        """Seconds to wait if ``response`` is a rate limit, else ``None``."""
        if response.status_code not in (403, 429):
            return None
        retry_after = _header_float(response, "Retry-After")
        if retry_after is not None:
            return retry_after
        remaining = response.headers.get("x-ratelimit-remaining")
        reset = _header_float(response, "x-ratelimit-reset")
        if reset is not None and (remaining == "0" or response.status_code == 429):
            return max(0.0, reset - self._clock()) + 1.0
        if response.status_code == 429:
            return 1.0
        return None

    def _send_once(
        self, method: str, url: str, accept: str, **kwargs: Any
    ) -> tuple[Any, bytes]:
        """One request, following same-origin redirects only."""
        for _ in range(_MAX_REDIRECTS + 1):
            try:
                response = self._session.request(
                    method,
                    url,
                    headers=self._headers(accept),
                    timeout=_TIMEOUT_SECONDS,
                    allow_redirects=False,
                    stream=True,
                    **kwargs,
                )
            except requests.RequestException:
                raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
            if response.status_code not in _REDIRECT_STATUS:
                return response, self._body(response)
            location = response.headers.get("Location")
            response.close()
            if not isinstance(location, str) or not location:
                raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
            target = urljoin(url, location)
            if not self._same_origin(target):
                raise _error(SnapshotErrorCode.UNSAFE_DESTINATION) from None
            url = target
        raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None

    @staticmethod
    def _body(response: Any) -> bytes:
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in response.iter_content(65536):
                total += len(chunk)
                if total > _MAX_READ_BYTES:
                    raise _error(SnapshotErrorCode.TOO_LARGE) from None
                chunks.append(chunk)
        except requests.RequestException:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
        finally:
            response.close()
        return b"".join(chunks)

    def _request(
        self, method: str, path: str, accept: str = _JSON_ACCEPT, **kwargs: Any
    ) -> tuple[int, bytes]:
        """Send with rate-limit backoff; return ``(status, body)`` for 2xx/404/409/422."""
        url = f"{self._api_url}{path}"
        waited = 0.0
        for attempt in range(_MAX_RATE_ATTEMPTS):
            response, body = self._send_once(method, url, accept, **kwargs)
            status = response.status_code
            wait = self._wait_seconds(response)
            if wait is not None:
                if attempt + 1 >= _MAX_RATE_ATTEMPTS or waited + wait > (
                    _MAX_TOTAL_WAIT_SECONDS
                ):
                    raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
                self._sleep(wait)
                waited += wait
                continue
            if status in (401, 403):
                raise _error(SnapshotErrorCode.AUTH) from None
            if status == 413:
                raise _error(SnapshotErrorCode.TOO_LARGE) from None
            if 200 <= status < 300 or status in (404, 409, 422):
                return status, body
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
        raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None

    # ------------------------------------------------------------- preflight

    def _ensure_ready(self) -> None:
        """Check visibility and that the branch exists, once."""
        if self._ready:
            return
        repo_path = f"/repos/{self._repo}"
        status, body = self._request("GET", repo_path)
        if status == 404:
            raise _error(SnapshotErrorCode.NOT_FOUND) from None
        if status != 200:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
        try:
            info = json.loads(body)
        except ValueError:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
        if not isinstance(info, dict):
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
        if info.get("private") is not True and not self._allow_public:
            raise _error(SnapshotErrorCode.UNSAFE_DESTINATION) from None
        status, _ = self._request(
            "GET", f"{repo_path}/branches/{quote(self._branch, safe='/')}"
        )
        if status == 404:
            raise _error(SnapshotErrorCode.NOT_FOUND) from None
        if status != 200:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
        self._ready = True

    # ------------------------------------------------------------------ API

    def _contents_path(self, key: str) -> str:
        return f"/repos/{self._repo}/contents/{quote(self._path(key), safe='/')}"

    def read(self, key: str) -> bytes | None:
        path = self._contents_path(key)
        self._ensure_ready()
        status, body = self._request(
            "GET", path, accept=_RAW_ACCEPT, params={"ref": self._branch}
        )
        if status == 404:
            return None
        if status != 200:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
        return body

    def create_if_absent(self, key: str, data: bytes) -> CreateResult:
        path = self._contents_path(key)
        data = bytes(data)
        if len(data) > _MAX_PART_BYTES:
            raise _error(SnapshotErrorCode.TOO_LARGE) from None
        payload = {
            "message": _COMMIT_MESSAGE,
            "content": base64.b64encode(data).decode("ascii"),
            "branch": self._branch,
        }
        with self._write_lock:
            self._ensure_ready()
            for attempt in range(_MAX_CONFLICT_RETRIES + 1):
                status, _ = self._request("PUT", path, json=payload)
                if status in (200, 201):
                    return CreateResult.CREATED
                if status == 404:
                    raise _error(SnapshotErrorCode.NOT_FOUND) from None
                existing = self.read(key)
                if existing is not None:
                    if existing == data:
                        return CreateResult.EXISTED_IDENTICAL
                    raise _error(SnapshotErrorCode.CONFLICT) from None
                if status != 409 or attempt >= _MAX_CONFLICT_RETRIES:
                    break
                self._sleep(float(2**attempt))  # branch advanced: back off, retry
        raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
