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
from urllib3.util import parse_url

from traigent.labels.transport_gate import gate_session
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
_MAX_TOTAL_WAIT_SECONDS = 60.0  # per public operation, shared by every request in it
_SECONDARY_LIMIT_WAIT_SECONDS = 60.0  # GitHub: wait at least a minute
_SECONDARY_LIMIT_MARKERS = (b"secondary rate limit", b"abuse detection")
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


class _WaitBudget:
    """Seconds one public operation may spend sleeping, across all its requests."""

    __slots__ = ("remaining",)

    def __init__(self) -> None:
        self.remaining = _MAX_TOTAL_WAIT_SECONDS

    def spend(self, seconds: float) -> None:
        if seconds > self.remaining:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
        self.remaining -= seconds


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
    carried by an error. Redirects are followed only within the API host:
    every request is prepared by Requests itself and the prepared URL, the one
    actually sent, is checked against the API origin (no userinfo, no
    backslashes). One operation (``read`` / ``create_if_absent``) may sleep at
    most 60 seconds in total across all of its requests.
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
        # SDK-created sessions are gated as the customer's own endpoint; a caller-supplied
        # session is left untouched.
        self._session: Any = (
            session
            if session is not None
            else gate_session(requests.Session(), external=True)
        )
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

    def _is_safe_url(self, url: object) -> bool:
        """True only if every parser that will see ``url`` agrees on the origin.

        ``urlsplit`` and urllib3 (what Requests sends with) disagree on inputs
        such as ``https://evil\\@api.github.com``; both must name the API
        origin, and the URL must carry no userinfo, backslash or control
        character.
        """
        if not isinstance(url, str) or any(
            char == "\\" or ord(char) <= 0x20 or ord(char) == 0x7F for char in url
        ):
            return False
        try:
            split = urlsplit(url)
            wire = parse_url(url)
        except ValueError:
            return False
        if split.username or split.password or "@" in split.netloc or wire.auth:
            return False
        # urllib3 brackets IPv6 literals ("[::1]"); urlsplit does not ("::1").
        host = (wire.host or "").lower().removeprefix("[").removesuffix("]")
        return (split.scheme, split.hostname, split.port) == self._origin and (
            (wire.scheme, host, wire.port) == self._origin
        )

    def _wait_seconds(self, response: Any, body: bytes) -> float | None:
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
        lowered = body.lower()
        if response.status_code == 429 or any(
            marker in lowered for marker in _SECONDARY_LIMIT_MARKERS
        ):
            return _SECONDARY_LIMIT_WAIT_SECONDS  # headerless secondary limit
        return None

    def _prepare(
        self, method: str, url: str, accept: str, kwargs: dict[str, Any]
    ) -> requests.PreparedRequest:
        """Prepare with Requests and refuse any URL that is not the API origin."""
        request = requests.Request(
            method,
            url,
            headers=self._headers(accept),
            params=kwargs.get("params"),
            json=kwargs.get("json"),
        )
        try:
            prepared: requests.PreparedRequest = self._session.prepare_request(request)
        except (requests.RequestException, ValueError):
            raise _error(SnapshotErrorCode.UNSAFE_DESTINATION) from None
        if not self._is_safe_url(prepared.url):
            raise _error(SnapshotErrorCode.UNSAFE_DESTINATION) from None
        return prepared

    def _send_once(
        self, method: str, url: str, accept: str, **kwargs: Any
    ) -> tuple[Any, bytes]:
        """One request, following same-origin redirects only."""
        for _ in range(_MAX_REDIRECTS + 1):
            prepared = self._prepare(method, url, accept, kwargs)
            # Send through the transport adapter directly: Session.send would also
            # run response hooks and parse Location for us, and an unrelated
            # ValueError from either is not a destination problem.
            try:
                settings = self._session.merge_environment_settings(
                    prepared.url, {}, True, None, None
                )
                # _prepare has already refused any URL that is not a string.
                response = self._session.get_adapter(prepared.url or "").send(
                    prepared, timeout=_TIMEOUT_SECONDS, **settings
                )
            except (requests.RequestException, ValueError):
                raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
            if response.status_code not in _REDIRECT_STATUS:
                return response, self._body(response)
            location = response.headers.get("Location")
            response.close()
            if not isinstance(location, str) or not location:
                raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
            try:
                target = urljoin(url, location)
                refused = (
                    "\\" in location
                    or "@" in urlsplit(target).netloc
                    or not self._same_origin(target)
                )
            except ValueError:
                # A Location this store cannot parse is refused here, where it is
                # parsed; the original request was sent, the target is never requested.
                refused = True
            if refused:
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
        self,
        method: str,
        path: str,
        budget: _WaitBudget,
        accept: str = _JSON_ACCEPT,
        **kwargs: Any,
    ) -> tuple[int, bytes]:
        """Send with rate-limit backoff; return ``(status, body)`` for 2xx/404/409/422."""
        url = f"{self._api_url}{path}"
        for attempt in range(_MAX_RATE_ATTEMPTS):
            response, body = self._send_once(method, url, accept, **kwargs)
            status = response.status_code
            wait = self._wait_seconds(response, body)
            if wait is not None:
                if attempt + 1 >= _MAX_RATE_ATTEMPTS:
                    raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
                budget.spend(wait)
                self._sleep(wait)
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

    def _branch_path(self) -> str:
        return f"/repos/{self._repo}/branches/{quote(self._branch, safe='/')}"

    def _ensure_ready(self, budget: _WaitBudget) -> None:
        """Check visibility and that the branch exists, once."""
        if self._ready:
            return
        repo_path = f"/repos/{self._repo}"
        status, body = self._request("GET", repo_path, budget)
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
        self._require_branch(budget)
        self._ready = True

    def _require_branch(self, budget: _WaitBudget) -> None:
        status, _ = self._request("GET", self._branch_path(), budget)
        if status == 404:
            self._ready = False
            raise _error(SnapshotErrorCode.NOT_FOUND) from None
        if status != 200:
            raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None

    # ------------------------------------------------------------------ API

    def _contents_path(self, key: str) -> str:
        return f"/repos/{self._repo}/contents/{quote(self._path(key), safe='/')}"

    def read(self, key: str) -> bytes | None:
        return self._read(key, _WaitBudget())

    def _read(self, key: str, budget: _WaitBudget) -> bytes | None:
        path = self._contents_path(key)
        self._ensure_ready(budget)
        status, body = self._request(
            "GET", path, budget, accept=_RAW_ACCEPT, params={"ref": self._branch}
        )
        if status == 404:
            # GitHub also answers 404 for a deleted branch or lost access:
            # only a live branch makes this "the object is absent".
            self._require_branch(budget)
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
        budget = _WaitBudget()
        with self._write_lock:
            self._ensure_ready(budget)
            for attempt in range(_MAX_CONFLICT_RETRIES + 1):
                status, _ = self._request("PUT", path, budget, json=payload)
                if status in (200, 201):
                    return CreateResult.CREATED
                if status == 404:
                    raise _error(SnapshotErrorCode.NOT_FOUND) from None
                existing = self._read(key, budget)
                if existing is not None:
                    if existing == data:
                        return CreateResult.EXISTED_IDENTICAL
                    raise _error(SnapshotErrorCode.CONFLICT) from None
                if status != 409 or attempt >= _MAX_CONFLICT_RETRIES:
                    break
                pause = float(2**attempt)  # branch advanced: back off, retry
                budget.spend(pause)
                self._sleep(pause)
        raise _error(SnapshotErrorCode.BACKEND_UNAVAILABLE) from None
