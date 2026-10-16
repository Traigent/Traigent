"""Egress-capture harness: records every outbound HTTP request, never networks.

Patched layers (all lowest-level send points, so every call site, retry,
buffer flush and alternate serializer above them is captured):

* aiohttp: ``ClientSession._request`` (``post``/``put``/``get``/``request``)
* requests: ``HTTPAdapter.send`` (``requests.post``, ``Session.request``, ...)
* httpx: ``Client.send`` and ``AsyncClient.send``
* urllib: ``OpenerDirector.open`` (``urlopen``)
* socket: ``connect``/``connect_ex``/``getaddrinfo`` raise and are recorded,
  so any transport the patches above do not cover fails loudly.

Authorization material (Authorization, Proxy-Authorization, X-API-Key,
X-Service-Key, Cookie) is dropped at capture time and never stored.
"""

from __future__ import annotations

import io
import json
import re
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest import mock
from urllib.parse import urlencode, urlsplit

#: Header names whose values are credentials. Dropped, never captured.
SECRET_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "x-service-key", "cookie"}
)


class EgressGuardError(RuntimeError):
    """Raised when code reaches a real socket instead of a captured transport."""


def _connection_error(transport: str) -> Exception:
    """The connection-level exception each client library raises."""

    if transport == "aiohttp":
        import aiohttp

        return aiohttp.ClientConnectionError("injected connection error")
    if transport == "requests":
        import requests

        return requests.ConnectionError("injected connection error")
    if transport == "httpx":
        import httpx

        return httpx.ConnectError("injected connection error")
    return urllib.error.URLError("injected connection error")


@dataclass(frozen=True)
class CapturedRequest:
    transport: str
    method: str
    url: str
    path: str
    query: str
    headers: tuple[tuple[str, str], ...]
    body: bytes

    @property
    def route(self) -> str:
        """Method plus path with opaque ids folded, e.g. ``POST /api/v1/sessions/{id}/results``."""

        return f"{self.method} {normalize_path(self.path)}"


_ID_SEGMENT = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|(?:sess|trial|cfg|run|exp|hybrid|best|opt|backend)[-_][\w-]+|\d+)$",
    re.IGNORECASE,
)
_ID_PARENTS = frozenset(
    {
        "sessions",
        "next-trial",
        "configuration-runs",
        "runs",
        "best-configs",
        "optimizer",
        "example-sets",
        "agents",
        "agent-heads",
    }
)


def normalize_path(path: str) -> str:
    """Fold server-issued ids to ``{id}`` so a route is stable across runs.

    A segment is folded when it follows a known collection segment, and the
    segment is not itself a fixed verb.  Customer-controlled path segments are
    *not* folded when they do not match: that is exactly what a leak in a path
    looks like in the report.
    """

    fixed_verbs = {
        "next-trial",
        "results",
        "finalize",
        "intermediate-report",
        "status",
        "measures",
        "batch",
        "next",
        "turn",
        "state",
        "validate",
        "summary-stats",
        "workflow-metadata",
        "cancel",
        "hybrid",
        "sessions",
        "optimizer",
        "plan",
        "recommendations",
        "upload",
        "identity-binding",
    }
    out: list[str] = []
    prev = ""
    for seg in path.split("/"):
        if seg and prev in _ID_PARENTS and seg not in fixed_verbs and prev != "":
            out.append("{id}")
        else:
            out.append(seg)
        prev = seg
    return "/".join(out)


def _clean_headers(headers: Any) -> tuple[tuple[str, str], ...]:
    items: list[tuple[str, str]] = []
    if headers is None:
        return ()
    pairs = headers.items() if hasattr(headers, "items") else headers
    for key, value in pairs:
        name = key.decode() if isinstance(key, bytes) else str(key)
        if name.lower() in SECRET_HEADERS:
            continue
        val = value.decode("latin-1") if isinstance(value, bytes) else str(value)
        items.append((name, val))
    return tuple(items)


#: JSON body keys whose string values are credentials; redacted at capture.
#: (``POST /keys/validate`` carries ``api_key`` in its body, not a header.)
SECRET_BODY_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "token",
        "password",
        "secret",
        "authorization",
        "access_token",
    }
)
REDACTED = "<redacted-credential>"


def _redact_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: (
                REDACTED
                if str(k).lower() in SECRET_BODY_KEYS and isinstance(v, str)
                else _redact_json(v)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_json(v) for v in obj]
    return obj


def redact_body_credentials(body: bytes) -> bytes:
    """Replace credential values in a JSON body; non-JSON bodies are unchanged."""

    if not body or b"{" not in body:
        return body
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    redacted = _redact_json(parsed)
    if redacted == parsed:
        return body
    return json.dumps(redacted).encode()


def _body_bytes(data: Any, json_obj: Any = None, serializer: Any = None) -> bytes:
    if json_obj is not None:
        dumps = serializer or json.dumps
        out = dumps(json_obj)
        return out.encode() if isinstance(out, str) else bytes(out)
    if data is None:
        return b""
    if isinstance(data, bytes | bytearray | memoryview):
        return bytes(data)
    if isinstance(data, str):
        return data.encode()
    if isinstance(data, dict):
        return urlencode(data).encode()
    if isinstance(data, list | tuple):
        try:
            return urlencode(list(data)).encode()
        except (TypeError, ValueError):
            pass
    fields = getattr(data, "_fields", None)  # aiohttp.FormData
    if fields:
        chunks: list[bytes] = []
        for type_options, _headers, value in fields:
            chunks.append(repr(dict(type_options)).encode())
            if isinstance(value, str):
                chunks.append(value.encode())
            elif isinstance(value, bytes | bytearray):
                chunks.append(bytes(value))
            elif hasattr(value, "getvalue"):
                got = value.getvalue()
                chunks.append(got if isinstance(got, bytes) else str(got).encode())
            else:
                chunks.append(repr(value).encode())
        return b"\n".join(chunks)
    if hasattr(data, "read"):
        try:
            return bytes(data.read())
        except Exception:  # noqa: BLE001 - capture must never break the run
            return repr(data).encode()
    return repr(data).encode()


@dataclass
class Fault:
    """Controlled failure for requests matching ``method`` + ``path_regex``.

    ``outcomes`` are consumed one per matching request, in order; an ``int`` is
    an HTTP status returned instead of the normal handler, ``"connection"``
    raises the transport's connection error.  After the list is exhausted the
    normal handler answers, so a retry eventually succeeds.
    """

    method: str
    path_regex: str
    outcomes: list[int | str]
    consumed: int = 0
    #: Requests that matched this fault's route while it was active.
    matches: int = 0
    #: Exerciser that armed it (filled when retired).
    owner: str = ""
    #: True when the SDK path under test is expected to resend after a failure.
    expects_retry: bool = True


class FakeBackend:
    """Stateful, schema-plausible fake of the Traigent optimization backend.

    Responses follow ``session_create_response_schema.json`` /
    ``session_next_trial_response_schema.json`` closely enough for an
    ``optimize()`` run to complete offline.  Server-side behaviour is not under
    test; only what the SDK *sends*.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.sessions: dict[str, dict[str, Any]] = {}
        self.unhandled: list[str] = []
        self._session_counter = 0
        self.faults: list[Fault] = []
        self.fault_log: list[Fault] = []

    @staticmethod
    def _pick(spec: Any, trial_number: int) -> Any:
        if isinstance(spec, dict):
            if "choices" in spec:
                return FakeBackend._pick(spec["choices"], trial_number)
            if "values" in spec:
                return FakeBackend._pick(spec["values"], trial_number)
            low, high = spec.get("low"), spec.get("high")
            if low is not None and high is not None:
                mid = (float(low) + float(high)) / 2 if trial_number % 2 == 0 else low
                return int(mid) if spec.get("type") == "int" else mid
            return spec.get("default")
        if isinstance(spec, list | tuple) and spec:
            if len(spec) == 2 and all(isinstance(x, int | float) for x in spec):
                return spec[(trial_number - 1) % 2]
            return spec[(trial_number - 1) % len(spec)]
        return spec

    def inject(
        self,
        method: str,
        path_regex: str,
        outcomes: list[int | str],
        *,
        expects_retry: bool = True,
    ) -> Fault:
        fault = Fault(
            method.upper(), path_regex, list(outcomes), expects_retry=expects_retry
        )
        with self._lock:
            self.faults.append(fault)
        return fault

    def retire_faults(self, owner: str) -> None:
        """Move armed faults to ``fault_log`` so they cannot bleed into the next run."""

        with self._lock:
            for fault in self.faults:
                fault.owner = owner
            self.fault_log.extend(self.faults)
            self.faults = []

    def next_fault(self, method: str, path: str) -> int | str | None:
        with self._lock:
            for fault in self.faults:
                if fault.method == method and re.search(fault.path_regex, path):
                    fault.matches += 1
                    if fault.consumed < len(fault.outcomes):
                        out = fault.outcomes[fault.consumed]
                        fault.consumed += 1
                        return out
        return None

    def handle(
        self, method: str, path: str, query: str, body: bytes
    ) -> tuple[int, Any]:
        try:
            payload = json.loads(body) if body else {}
        except (ValueError, UnicodeDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        p = path.rstrip("/")
        m = re.search(
            r"/sessions/([^/]+)/(next-trial|results|finalize|intermediate-report)$", p
        )
        with self._lock:
            if p.endswith("/keys/validate"):
                return 200, {"valid": True}
            if method == "POST" and p.endswith("/hybrid/sessions"):
                return 201, {
                    "session_id": "hybrid-canary",
                    "token": "optimizer-token",
                    "optimizer_endpoint": "https://backend.example.test/api/v1/hybrid/optimizer",
                }
            if method == "POST" and re.search(r"/sessions$", p) and "/hybrid/" not in p:
                self._session_counter += 1
                sid = f"sess-canary-{self._session_counter}"
                self.sessions[sid] = {
                    "space": payload.get("configuration_space") or {},
                    "max_trials": int(payload.get("max_trials") or 3),
                    "slots": 0,
                    "suggested": 0,
                    "results": 0,
                }
                return 201, {
                    "session_id": sid,
                    "status": "active",
                    "metadata": {
                        "experiment_id": "exp-canary",
                        "experiment_run_id": "run-canary",
                        "dataset_size": 2,
                    },
                    "optimization_strategy": {"source": "cloud_brain"},
                    "project_id": "proj-canary",
                    "tenant_id": "ten-canary",
                    "agent_id": "agent-canary",
                }
            if m:
                sid, verb = m.group(1), m.group(2)
                state = self.sessions.setdefault(
                    sid,
                    {
                        "space": {},
                        "max_trials": 3,
                        "slots": 0,
                        "suggested": 0,
                        "results": 0,
                    },
                )
                if verb == "next-trial":
                    state["slots"] += 1
                    n = state["slots"]
                    is_slot = not (
                        "request_metadata" in payload or payload.get("previous_results")
                    )
                    if is_slot:
                        config: dict[str, Any] = {}
                    else:
                        state["suggested"] += 1
                        n = state["suggested"]
                        config = {
                            k: self._pick(v, n) for k, v in state["space"].items()
                        }
                    return 200, {
                        "should_continue": n <= state["max_trials"],
                        "has_next": n <= state["max_trials"],
                        "suggestion": {
                            "trial_id": f"trial-canary-{sid}-{state['slots']}",
                            "session_id": sid,
                            "trial_number": n,
                            "config": config,
                            "dataset_subset": {
                                "indices": [0, 1],
                                "selection_strategy": "full",
                                "confidence_level": 1.0,
                                "estimated_representativeness": 1.0,
                                "metadata": {"dataset_size": 2},
                            },
                            "exploration_type": "exploration",
                            "metadata": {"config_id": f"cfg-{n}"},
                        },
                        "session_status": "active",
                        "metadata": {},
                    }
                if verb == "results":
                    state["results"] += 1
                    return 201, {
                        "success": True,
                        "continue_optimization": state["results"] < state["max_trials"],
                        "message": "Result recorded",
                    }
                if verb == "intermediate-report":
                    return 200, {"prune": False}
                if verb == "finalize":
                    return 200, {
                        "session_id": sid,
                        "best_config": {},
                        "best_metrics": {},
                        "total_trials": state["results"],
                        "successful_trials": state["results"],
                        "total_duration": 0.0,
                        "cost_savings": 0.0,
                        "metadata": {
                            "experiment_id": "exp-canary",
                            "experiment_run_id": "run-canary",
                        },
                    }
            if method == "PUT" and re.search(
                r"/(status|measures)$|/experiment-runs/runs/", p
            ):
                return 200, {"success": True}
            if method == "POST" and p.endswith("/best-configs"):
                spec = payload.get("spec", {})
                return 200, {"data": {"config_id": "best-canary", "spec": spec}}
            if method == "GET" and "/best-configs/" in p:
                return 200, {"data": {"config_id": "best-canary", "spec": {}}}
            if method == "GET" and p.endswith("/status"):
                return 200, {"status": "active", "session_id": "hybrid-canary"}
            if method == "POST" and p.endswith("/example-sets"):
                return 201, {"example_set_id": "exs-canary"}
            if method == "POST" and p.endswith("/upload"):
                return 200, {"added": 1, "skipped": 0, "invalid": 0}
            if method == "GET" and "/example-sets/" in p:
                return 200, {"id": "exs-canary", "name": "exs", "description": ""}
            if method == "POST" and p.endswith("/optimize"):
                return 200, {"status": "ok"}
            if p.endswith("/features"):
                return 200, {"ok": True}
            if "/hybrid/optimizer/" in p:
                if p.endswith("/next"):
                    return 200, {"configuration": {}, "done": True}
                return 200, {"status": "ok", "accepted": True}
            if p.endswith("/optimization/plan") or "/optimization/recommendations" in p:
                return 200, {"recommendations": []}
            self.unhandled.append(f"{method} {path}")
            return 200, {}


@dataclass
class EgressCapture:
    """Records requests and installs/removes all transport patches."""

    backend: FakeBackend = field(default_factory=FakeBackend)
    requests: list[CapturedRequest] = field(default_factory=list)
    guard_violations: list[str] = field(default_factory=list)
    dns_lookups: list[str] = field(default_factory=list)
    fake_hosts: frozenset[str] = frozenset({"backend.example.test"})
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _stack: Any = None

    # -- recording ---------------------------------------------------------
    def _record(
        self, transport: str, method: str, url: str, headers: Any, body: bytes
    ) -> tuple[int, Any]:
        parts = urlsplit(url)
        cap = CapturedRequest(
            transport=transport,
            method=method.upper(),
            url=f"{parts.scheme}://{parts.netloc}{parts.path}",
            path=parts.path,
            query=parts.query,
            headers=_clean_headers(headers),
            body=redact_body_credentials(body),
        )
        with self._lock:
            self.requests.append(cap)
        fault = self.backend.next_fault(cap.method, cap.path)
        if fault == "connection":
            raise _connection_error(transport)
        if isinstance(fault, int):
            return fault, {"error": "injected fault"}
        return self.backend.handle(cap.method, cap.path, cap.query, body)

    # -- guards ------------------------------------------------------------
    def _deny(self, what: str) -> None:
        import traceback

        frames = [
            f"{f.filename.split('Traigent/')[-1]}:{f.lineno}"
            for f in traceback.extract_stack()[-12:-2]
        ]
        with self._lock:
            self.guard_violations.append(f"{what} via {' <- '.join(reversed(frames))}")
        raise EgressGuardError(f"non-captured network access: {what}")

    # -- install -----------------------------------------------------------
    def install(self) -> EgressCapture:
        from contextlib import ExitStack

        stack = ExitStack()
        self._stack = stack
        self._patch_aiohttp(stack)
        self._patch_requests(stack)
        self._patch_httpx(stack)
        self._patch_urllib(stack)
        self._patch_socket(stack)
        return self

    def uninstall(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def __enter__(self) -> EgressCapture:
        return self.install()

    def __exit__(self, *exc: Any) -> None:
        self.uninstall()

    # -- aiohttp -----------------------------------------------------------
    def _patch_aiohttp(self, stack: Any) -> None:
        try:
            import aiohttp
            from yarl import URL
        except ImportError:  # pragma: no cover
            return
        capture = self

        class _Resp:
            def __init__(self, status: int, payload: Any, url: Any) -> None:
                self.status = status
                self.reason = "OK"
                self.ok = status < 400
                self.url = url
                self.headers = {"Content-Type": "application/json"}
                self.content_type = "application/json"
                self._body = json.dumps(payload, default=str).encode()
                self.closed = False

            async def json(self, *a: Any, **k: Any) -> Any:
                return json.loads(self._body)

            async def text(self, *a: Any, **k: Any) -> str:
                return self._body.decode()

            async def read(self) -> bytes:
                return self._body

            def raise_for_status(self) -> None:
                if self.status >= 400:
                    raise aiohttp.ClientResponseError(
                        request_info=mock.MagicMock(), history=(), status=self.status
                    )

            def release(self) -> None:
                self.closed = True

            async def wait_for_close(self) -> None:
                return None

            def close(self) -> None:
                self.closed = True

            async def __aenter__(self) -> _Resp:
                return self

            async def __aexit__(self, *exc: Any) -> bool:
                return False

        async def _request(
            session: Any, method: str, str_or_url: Any, **kwargs: Any
        ) -> _Resp:
            try:
                url = session._build_url(str_or_url)
            except Exception:  # noqa: BLE001
                url = URL(str(str_or_url))
            params = kwargs.get("params")
            if params:
                url = url.update_query(params)
            hdrs: list[tuple[str, str]] = []
            default_headers = getattr(session, "_default_headers", None)
            if default_headers:
                hdrs.extend(default_headers.items())
            if kwargs.get("headers"):
                hdrs.extend(dict(kwargs["headers"]).items())
            body = _body_bytes(
                kwargs.get("data"),
                kwargs.get("json"),
                getattr(session, "_json_serialize", None),
            )
            status, payload = capture._record("aiohttp", method, str(url), hdrs, body)
            return _Resp(status, payload, url)

        stack.enter_context(
            mock.patch.object(aiohttp.ClientSession, "_request", _request)
        )

    # -- requests ----------------------------------------------------------
    def _patch_requests(self, stack: Any) -> None:
        try:
            import requests
            from requests.adapters import HTTPAdapter
        except ImportError:  # pragma: no cover
            return
        capture = self

        def send(adapter: Any, request: Any, **kwargs: Any) -> Any:
            body = _body_bytes(request.body)
            status, payload = capture._record(
                "requests", request.method, request.url, request.headers, body
            )
            resp = requests.Response()
            resp.status_code = status
            resp._content = json.dumps(payload, default=str).encode()
            resp.headers["Content-Type"] = "application/json"
            resp.url = request.url
            resp.request = request
            resp.encoding = "utf-8"
            resp.raw = io.BytesIO(resp._content)
            return resp

        stack.enter_context(mock.patch.object(HTTPAdapter, "send", send))

    # -- httpx -------------------------------------------------------------
    def _patch_httpx(self, stack: Any) -> None:
        try:
            import httpx
        except ImportError:  # pragma: no cover
            return
        capture = self

        def _respond(request: Any, content: bytes) -> Any:
            status, payload = capture._record(
                "httpx", request.method, str(request.url), request.headers, content
            )
            return httpx.Response(status, json=payload, request=request)

        def send(client: Any, request: Any, **kwargs: Any) -> Any:
            return _respond(request, request.read())

        async def asend(client: Any, request: Any, **kwargs: Any) -> Any:
            return _respond(request, await request.aread())

        stack.enter_context(mock.patch.object(httpx.Client, "send", send))
        stack.enter_context(mock.patch.object(httpx.AsyncClient, "send", asend))

    # -- urllib ------------------------------------------------------------
    def _patch_urllib(self, stack: Any) -> None:
        capture = self

        class _UResp(io.BytesIO):
            def __init__(self, status: int, payload: Any, url: str) -> None:
                super().__init__(json.dumps(payload, default=str).encode())
                self.status = status
                self.code = status
                self.url = url
                self.headers = {"Content-Type": "application/json"}

            def getcode(self) -> int:
                return self.status

            def geturl(self) -> str:
                return self.url

            def info(self) -> dict[str, str]:
                return self.headers

        def open_(
            opener: Any, fullurl: Any, data: Any = None, timeout: Any = None
        ) -> Any:
            if isinstance(fullurl, str):
                req = urllib.request.Request(fullurl, data=data)
            else:
                req = fullurl
                if data is not None:
                    req.data = data
            body = _body_bytes(req.data)
            method = req.get_method()
            status, payload = capture._record(
                "urllib", method, req.full_url, dict(req.header_items()), body
            )
            resp = _UResp(status, payload, req.full_url)
            if status >= 400:
                raise urllib.error.HTTPError(
                    req.full_url,
                    status,
                    "error",
                    resp.headers,
                    resp,  # type: ignore[arg-type]
                )
            return resp

        stack.enter_context(
            mock.patch.object(urllib.request.OpenerDirector, "open", open_)
        )

    # -- sockets -----------------------------------------------------------
    def _patch_socket(self, stack: Any) -> None:
        capture = self

        def connect(sock: Any, address: Any) -> None:
            capture._deny(f"socket.connect {address!r}")

        def connect_ex(sock: Any, address: Any) -> int:
            capture._deny(f"socket.connect_ex {address!r}")
            return 1

        def getaddrinfo(host: Any, *a: Any, **k: Any) -> Any:
            # The SDK resolves the backend host to vet it (url_security.py).
            # Answer for the fake host with a fixed public address; nothing
            # real is resolved.  Any other name is a guard violation.
            if host in self.fake_hosts:
                with capture._lock:
                    capture.dns_lookups.append(str(host))
                return [
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
                ]
            capture._deny(f"socket.getaddrinfo {host!r}")

        stack.enter_context(mock.patch.object(socket.socket, "connect", connect))
        stack.enter_context(mock.patch.object(socket.socket, "connect_ex", connect_ex))
        stack.enter_context(mock.patch.object(socket, "getaddrinfo", getaddrinfo))


@contextmanager
def capture_egress() -> Iterator[EgressCapture]:
    cap = EgressCapture()
    cap.install()
    try:
        yield cap
    finally:
        cap.uninstall()


Predicate = Callable[[CapturedRequest], bool]
