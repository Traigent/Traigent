"""Transport gate for the SDK's own cloud clients (phase 2: "report" mode by default).

The gate classifies the FINAL request (path parameters, query string, serialized body)
against the Schema content classes (``content_classes``). Modes (env
``TRAIGENT_TRANSPORT_GATE``):

* ``off``    - sessions are not gated.
* ``report`` - DEFAULT. Counts values per class into process-local totals and a debug
  log (class -> count, route template; never a value, key or path). Never blocks, never
  changes the request, and any gate failure is swallowed into :func:`gate_errors`.
* ``refuse`` - raises :class:`TransportGateRefusal` BEFORE any bytes leave when a value
  is ``legacy_content``/``unclassified``/``violation``. For tests now, phase-4
  enforcement later.

Scope: nothing is monkeypatched globally. :func:`gate_session` re-classes ONE
SDK-created client instance (aiohttp ``ClientSession``, ``requests.Session``,
``httpx.Client``/``AsyncClient``) so that its single send funnel inspects the request,
and :func:`preflight_requests` covers the SDK's few module-level ``requests.post``
calls. Only requests to the configured Traigent backend origin (scheme, host, port) are
inspected; every other request, including a customer's own call to any other host, is
untouched.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlencode, urlsplit

from traigent.utils.logging import get_logger

from .content_classes import BLOCKING, classify_request, get_table

logger = get_logger(__name__)

MODE_ENV = "TRAIGENT_TRANSPORT_GATE"
MODES = ("off", "report", "refuse")

_mode_override: str | None = None
_origins_override: frozenset[tuple[str, str, int]] | None = None
_totals: Counter[str] = Counter()
_errors: Counter[str] = Counter()
_refusals: list[tuple[str, dict[str, int]]] = []
_lock = threading.Lock()
_CLASS_CACHE: dict[tuple[type, str], type] = {}


class TransportGateRefusal(RuntimeError):
    """A request was refused before send. Carries counts only, never values."""

    def __init__(self, route: str, counts: dict[str, int]) -> None:
        self.route = route
        self.counts = dict(counts)
        blocked = {k: v for k, v in counts.items() if k in BLOCKING}
        super().__init__(f"transport gate refused {route}: {blocked}")


# --- mode, origin, counters -------------------------------------------------------


def current_mode() -> str:
    mode = _mode_override or os.environ.get(MODE_ENV, "report").strip().lower()
    return mode if mode in MODES else "report"


def set_mode(mode: str | None) -> None:
    """Override the mode in-process (tests). ``None`` returns to the environment value."""
    global _mode_override
    if mode is not None and mode not in MODES:
        raise ValueError("unknown transport gate mode")
    _mode_override = mode


def _origin(url: str) -> tuple[str, str, int] | None:
    """(scheme, host, port) of ``url``; ``None`` for anything unparseable.

    An invalid URL (bad port, bad netloc) is not ours to judge: it returns ``None`` so the
    HTTP library raises its own native error.
    """
    try:
        parts = urlsplit(url)
        if not parts.scheme or not parts.hostname:
            return None
        default = 443 if parts.scheme == "https" else 80
        return parts.scheme.lower(), parts.hostname.lower(), parts.port or default
    except ValueError:
        return None


def applies_to(
    method: str, url: str, extra_origins: frozenset[tuple[str, str, int]] = frozenset()
) -> bool:
    """True only for a backend origin AND a known optimization route.

    The origin is a configured backend or one of ``extra_origins`` (the per-client API
    base a gated session was created for). Evaluated before anything is parsed or
    serialized, so other hosts and other routes (auth, observability, multipart uploads,
    ...) are never touched.
    """
    if current_mode() == "off":
        return False
    try:
        origin = _origin(str(url))
        if origin is None or (
            origin not in extra_origins and origin not in backend_origins()
        ):
            return False
    except Exception:  # noqa: BLE001 - cannot establish the backend origin: not ours
        return False
    try:
        return get_table().find(method, urlsplit(str(url)).path) is not None
    except Exception:  # noqa: BLE001 - backend origin but no table: let the caller decide
        return True


def set_allowed_origins(urls: Iterable[str] | None) -> None:
    """Pin the inspected origins (tests). ``None`` returns to the configured backend."""
    global _origins_override
    _origins_override = (
        None if urls is None else frozenset(o for u in urls if (o := _origin(u)))
    )


def backend_origins() -> frozenset[tuple[str, str, int]]:
    if _origins_override is not None:
        return _origins_override
    from traigent.config.backend_config import BackendConfig

    urls = [BackendConfig.get_backend_url(), BackendConfig.get_cloud_backend_url()]
    configured = BackendConfig.get_configured_backend_url()
    if configured:
        urls.append(configured)
    return frozenset(o for u in urls if (o := _origin(u)))


def report_totals() -> dict[str, int]:
    """Process-lifetime class -> count totals of everything inspected."""
    return dict(_totals)


def gate_errors() -> dict[str, int]:
    """Closed counters of gate-internal failures swallowed in report mode."""
    return dict(_errors)


def refusal_log() -> list[tuple[str, dict[str, int]]]:
    """(route template, class counts) of every refusal so far. Counts only."""
    return list(_refusals)


def clear_refusals() -> None:
    _refusals.clear()


# --- core -------------------------------------------------------------------------


def inspect_outgoing(
    method: str,
    url: str,
    body: bytes | str | None,
    *,
    body_present: bool | None = None,
    extra_origins: frozenset[tuple[str, str, int]] = frozenset(),
) -> dict[str, int] | None:
    """Classify one outgoing request from its FINAL bytes; counts, or ``None`` if skipped.

    ``body_present`` lets a caller say "there is a body I cannot read" (``body=None``):
    refuse mode then refuses (cannot classify); report mode skips it.
    """
    mode = current_mode()
    if mode == "off":
        return None
    refuse = mode == "refuse"
    if not applies_to(method, str(url), extra_origins):
        return None
    try:
        parts = urlsplit(str(url))
        table = get_table()
        has_body = bool(body) or bool(body_present)
        parsed: Any = None
        if body:
            parsed = json.loads(body)
        elif body_present:
            raise ValueError("unreadable body")
        result = classify_request(
            table, method, parts.path, parts.query, parsed, has_body=has_body
        )
    except TransportGateRefusal:
        raise
    except Exception as exc:  # noqa: BLE001 - report mode must never fail a request
        if refuse:
            raise TransportGateRefusal("unclassifiable", {"violation": 1}) from exc
        with _lock:
            _errors[type(exc).__name__] += 1
        return None
    if result is None:
        return None
    template, counter = result
    counts = dict(counter)
    with _lock:
        _totals.update(counts)
    logger.debug("transport gate [%s] %s %s", mode, template, counts)
    if refuse and any(counts.get(c, 0) for c in BLOCKING):
        _refusals.append((template, counts))
        raise TransportGateRefusal(template, counts)
    return counts


def _with_params(url: str, params: Any) -> str:
    if not params:
        return url
    try:
        extra = urlencode(params, doseq=True)
    except (TypeError, ValueError):
        return url
    return url + ("&" if "?" in url else "?") + extra


# --- per-instance scoping -----------------------------------------------------------


def _derived(base: type, tag: str, build: Any) -> type:
    key = (base, tag)
    with _lock:
        if key not in _CLASS_CACHE:
            _CLASS_CACHE[key] = build(base)
        return _CLASS_CACHE[key]


def client_origins(client: Any) -> list[str]:
    """API-base URLs a backend client object is configured with (duck-typed, str only).

    Used to gate a session created for ``client`` against ITS base, not just the global one.
    """
    found: list[str] = []
    config = getattr(client, "backend_config", None)
    for owner, names in (
        (config, ("api_base_url", "backend_base_url")),
        (client, ("api_base_url", "base_url")),
    ):
        for name in names:
            value = getattr(owner, name, None) if owner is not None else None
            if isinstance(value, str) and value:
                found.append(value)
    return found


def _instance_origins(client: Any) -> frozenset[tuple[str, str, int]]:
    """Origins registered for THIS gated instance (its own API base); never global."""
    return getattr(client, "_traigent_gate_origins", frozenset())


def _check_strict(client: Any, url: str) -> None:
    """A strict instance (content-less) fails closed on any origin it was not built for."""
    if not getattr(client, "_traigent_gate_strict", False) or current_mode() == "off":
        return
    origin = _origin(str(url))
    if origin is None or origin not in (_instance_origins(client) | backend_origins()):
        raise TransportGateRefusal("unrecognised_origin", {"violation": 1})


def _build_aiohttp(base: type) -> type:
    from aiohttp import payload

    class GatedClientSession(base):  # type: ignore[valid-type, misc]
        _traigent_gated = True

        async def _request(self, method: Any, str_or_url: Any, **kwargs: Any) -> Any:
            url = str(str_or_url)
            base_url = getattr(self, "_base_url", None)
            if base_url is not None and not urlsplit(url).scheme:
                url = str(base_url).rstrip("/") + "/" + url.lstrip("/")
            url = _with_params(url, kwargs.get("params"))
            _check_strict(self, url)
            extra = _instance_origins(self)
            body: bytes | None = None
            unreadable = False
            if applies_to(str(method), url, extra):
                try:
                    data = kwargs.get("data")
                    if kwargs.get("json") is not None and data is None:
                        # Serialize ONCE, exactly as aiohttp would, and hand the same
                        # payload on: no second serialization.
                        sb = getattr(self, "_json_serialize_bytes", None)
                        if sb is not None:
                            pl = payload.JsonBytesPayload(kwargs["json"], dumps=sb)
                        else:
                            pl = payload.JsonPayload(
                                kwargs["json"], dumps=self._json_serialize
                            )
                        kwargs = {k: v for k, v in kwargs.items() if k != "json"}
                        kwargs["data"] = pl
                        data = pl
                    if isinstance(data, (bytes, bytearray)):
                        body = bytes(data)
                    elif isinstance(data, str):
                        body = data.encode("utf-8")
                    elif data is not None:
                        value = getattr(data, "_value", None)
                        if isinstance(value, (bytes, bytearray)):
                            body = bytes(value)
                        else:
                            unreadable = True
                except Exception as exc:  # noqa: BLE001
                    with _lock:
                        _errors[type(exc).__name__] += 1
                    unreadable = True
                inspect_outgoing(
                    str(method),
                    url,
                    body,
                    body_present=unreadable or None,
                    extra_origins=extra,
                )
            return await super()._request(method, str_or_url, **kwargs)

    GatedClientSession.__name__ = f"Gated{base.__name__}"
    return GatedClientSession


def _build_requests(base: type) -> type:
    class GatedRequestsSession(base):  # type: ignore[valid-type, misc]
        _traigent_gated = True

        def send(self, request: Any, **kwargs: Any) -> Any:
            body = request.body
            if isinstance(body, str):
                body = body.encode("utf-8")
            readable = body is None or isinstance(body, (bytes, bytearray))
            _check_strict(self, str(request.url))
            inspect_outgoing(
                str(request.method),
                str(request.url),
                bytes(body) if isinstance(body, (bytes, bytearray)) else None,
                body_present=None if readable else True,
                extra_origins=_instance_origins(self),
            )
            return super().send(request, **kwargs)

    GatedRequestsSession.__name__ = f"Gated{base.__name__}"
    return GatedRequestsSession


def _httpx_body(request: Any) -> tuple[bytes | None, bool]:
    try:
        return bytes(request.content), False
    except Exception:  # noqa: BLE001 - streaming body not yet read
        return None, True


def _build_httpx_sync(base: type) -> type:
    class GatedHttpxClient(base):  # type: ignore[valid-type, misc]
        _traigent_gated = True

        def send(self, request: Any, **kwargs: Any) -> Any:
            body, unreadable = _httpx_body(request)
            _check_strict(self, str(request.url))
            inspect_outgoing(
                str(request.method),
                str(request.url),
                body,
                body_present=True if unreadable else None,
                extra_origins=_instance_origins(self),
            )
            return super().send(request, **kwargs)

    GatedHttpxClient.__name__ = f"Gated{base.__name__}"
    return GatedHttpxClient


def _build_httpx_async(base: type) -> type:
    class GatedHttpxAsyncClient(base):  # type: ignore[valid-type, misc]
        _traigent_gated = True

        async def send(self, request: Any, **kwargs: Any) -> Any:
            body, unreadable = _httpx_body(request)
            _check_strict(self, str(request.url))
            inspect_outgoing(
                str(request.method),
                str(request.url),
                body,
                body_present=True if unreadable else None,
                extra_origins=_instance_origins(self),
            )
            return await super().send(request, **kwargs)

    GatedHttpxAsyncClient.__name__ = f"Gated{base.__name__}"
    return GatedHttpxAsyncClient


def _is_instance(obj: Any, module: str, name: str) -> bool:
    """isinstance against a library class; False if the library or class is unusable
    (not installed, or replaced by a test double that is not a type)."""
    try:
        mod = __import__(module)
        return isinstance(obj, getattr(mod, name))
    except (ImportError, AttributeError, TypeError):
        return False


def gate_session(
    client: Any,
    *,
    origins: Iterable[str] | None = None,
    strict: bool = False,
) -> Any:
    """Gate ONE SDK-created client instance in place and return it.

    Re-classes the instance so its send funnel inspects requests. Mocks and unknown
    objects are returned untouched; ``TRAIGENT_TRANSPORT_GATE=off`` disables gating.

    ``origins`` are the URLs (API bases) this instance sends to; they are inspected in
    addition to the configured backend, for THIS instance only (no global state), so a
    per-client API base is gated like the default one. ``strict`` makes the instance fail
    closed on any other origin (content-less sessions).
    """
    if current_mode() == "off":
        return client
    if not getattr(type(client), "_traigent_gated", False):
        if _is_instance(client, "aiohttp", "ClientSession"):
            object.__setattr__(
                client, "__class__", _derived(type(client), "aiohttp", _build_aiohttp)
            )
        elif _is_instance(client, "requests", "Session"):
            client.__class__ = _derived(type(client), "requests", _build_requests)
        elif _is_instance(client, "httpx", "AsyncClient"):
            client.__class__ = _derived(type(client), "httpx-async", _build_httpx_async)
        elif _is_instance(client, "httpx", "Client"):
            client.__class__ = _derived(type(client), "httpx", _build_httpx_sync)
    if getattr(type(client), "_traigent_gated", False):
        parsed = frozenset(o for u in (origins or ()) if (o := _origin(str(u))))
        if parsed:
            object.__setattr__(
                client, "_traigent_gate_origins", _instance_origins(client) | parsed
            )
        if strict:
            object.__setattr__(client, "_traigent_gate_strict", True)
    return client


def preflight_requests(method: str, url: str, **kwargs: Any) -> None:
    """Inspect a module-level ``requests.<method>(url, ...)`` call before it is made.

    The body is prepared exactly as ``requests`` will prepare it (deterministic JSON
    encoding); the caller then makes its normal call.
    """
    if current_mode() == "off":
        return
    try:
        import requests

        if not applies_to(method, str(url)):
            return
        prepared = requests.Request(
            method.upper(),
            url,
            params=kwargs.get("params"),
            data=kwargs.get("data"),
            json=kwargs.get("json"),
        ).prepare()
        body = prepared.body
        readable = body is None or isinstance(body, (bytes, str, bytearray))
        inspect_outgoing(
            method,
            str(prepared.url),
            body.encode() if isinstance(body, str) else (bytes(body) if body else None),
            body_present=None if readable else True,
        )
    except TransportGateRefusal:
        raise
    except Exception as exc:  # noqa: BLE001
        if current_mode() == "refuse":
            raise TransportGateRefusal("unclassifiable", {"violation": 1}) from exc
        with _lock:
            _errors[type(exc).__name__] += 1
