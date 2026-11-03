"""Hand-written GitHub Contents API fake behind a ``requests.Session``-shaped object."""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.parse import parse_qs, urlsplit

import requests

OWNER_REPO = "acme/snaps"


class FakeResponse:
    def __init__(
        self,
        status: int,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size: int) -> Any:
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    def close(self) -> None:
        self.closed = True


def json_response(status: int, payload: Any) -> FakeResponse:
    return FakeResponse(status, json.dumps(payload).encode("utf-8"))


class FakeGitHubSession:
    """One repository: files on branches, with injectable failures."""

    def __init__(
        self,
        *,
        private: bool = True,
        branches: tuple[str, ...] = ("main",),
        repo_exists: bool = True,
    ) -> None:
        self.private = private
        self.branches = set(branches)
        self.repo_exists = repo_exists
        self.files: dict[tuple[str, str], bytes] = {}
        self.calls: list[dict[str, Any]] = []
        self.queue: list[FakeResponse | Exception] = []  # served before routing
        self.always: FakeResponse | Exception | None = None
        self.put_conflicts = 0  # next N PUTs answer 409 without writing
        self.redirects: dict[str, str] = {}  # url -> Location on GET

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        parts = urlsplit(url)
        self.calls.append(
            {
                "method": method,
                "url": url,
                "path": parts.path,
                "headers": dict(kwargs.get("headers", {})),
                "params": kwargs.get("params") or parse_qs(parts.query),
                "json": kwargs.get("json"),
                "timeout": kwargs.get("timeout"),
                # A transport adapter never follows redirects; the store sends none.
                "allow_redirects": kwargs.get("allow_redirects", False),
            }
        )
        for source in (self.queue.pop(0) if self.queue else None, self.always):
            if isinstance(source, Exception):
                raise source
            if source is not None:
                return source
        if url in self.redirects and method == "GET":
            return FakeResponse(302, headers={"Location": self.redirects[url]})
        return self._route(method, parts.path, kwargs)

    def prepare_request(self, request: requests.Request) -> requests.PreparedRequest:
        """Real Requests preparation, as the store must use it."""
        return requests.Session().prepare_request(request)

    def get_adapter(self, url: str) -> FakeGitHubSession:
        """The transport seam the store sends through: this fake, as its own adapter."""
        return self

    def merge_environment_settings(self, *_args: Any) -> dict[str, Any]:
        return {"proxies": {}, "stream": True, "verify": True, "cert": None}

    def send(self, prepared: requests.PreparedRequest, **kwargs: Any) -> FakeResponse:
        """Serve a prepared request through the same routing as ``request``."""
        url = prepared.url or ""
        parts = urlsplit(url)
        params = {key: values[0] for key, values in parse_qs(parts.query).items()}
        if set(params) == {"ref"}:
            url = url.split("?", 1)[0]  # redirects are keyed on the bare URL
        body = json.loads(prepared.body) if prepared.body else None
        return self.request(
            prepared.method or "GET",
            url,
            headers=dict(prepared.headers),
            params=params,
            json=body,
            **kwargs,
        )

    def _route(self, method: str, path: str, kwargs: dict[str, Any]) -> FakeResponse:
        prefix = f"/repos/{OWNER_REPO}"
        if not self.repo_exists or not path.startswith(prefix):
            return FakeResponse(404)
        rest = path[len(prefix) :]
        if rest == "":
            return json_response(200, {"private": self.private, "size": 1})
        if rest.startswith("/branches/"):
            branch = rest[len("/branches/") :]
            return FakeResponse(200 if branch in self.branches else 404)
        if rest.startswith("/contents/"):
            file_path = rest[len("/contents/") :]
            if method == "GET":
                ref = kwargs["params"]["ref"]
                data = self.files.get((ref, file_path))
                if data is None:
                    return FakeResponse(404)
                accept = kwargs["headers"]["Accept"]
                if accept.endswith("raw+json"):
                    return FakeResponse(200, data)
                if len(data) > 1024 * 1024:  # the JSON form omits big content
                    return FakeResponse(403, b'{"message":"too large"}')
                return json_response(200, {"content": base64.b64encode(data).decode()})
            body = kwargs["json"]
            branch = body["branch"]
            if branch not in self.branches:
                return FakeResponse(404)
            if (branch, file_path) in self.files:
                return FakeResponse(422, b'{"message":"\\"sha\\" wasn\'t supplied."}')
            if self.put_conflicts > 0:
                self.put_conflicts -= 1
                return FakeResponse(409)
            self.files[(branch, file_path)] = base64.b64decode(body["content"])
            return FakeResponse(201)
        return FakeResponse(404)

    def puts(self) -> list[dict[str, Any]]:
        return [call for call in self.calls if call["method"] == "PUT"]


def connection_error(text: str) -> Exception:
    return requests.ConnectionError(text)
