"""Single transport gate for the SDK's cloud send paths (phase 2: "report" mode).

The gate classifies the FINAL serialized request body against the Schema content-class
annotations (``content_classes``). Modes (env ``TRAIGENT_TRANSPORT_GATE``):

* ``off``    - not installed.
* ``report`` - DEFAULT. Logs a closed count summary locally (class -> count, route
  template; never a value, key or path) and never blocks or changes the request.
* ``refuse`` - raises :class:`TransportGateRefusal` BEFORE the request is handed to the
  transport when any value is ``legacy_content``/``unclassified``/``violation``.
  Intended for tests now and for phase 4 enforcement.

Hook points (one wrapper per client library the SDK sends with; all funnel into
:func:`inspect_outgoing`): ``aiohttp.ClientSession.request/post/put/patch/delete``,
``requests.Session.send`` (final prepared body) and ``httpx.Client/AsyncClient.request``.
They sit ABOVE the lowest send layer, so a test that replaces the low-level send still
exercises the gate. Only paths that match a body-carrying route of the optimization
endpoint inventory are classified; anything else passes through untouched.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import functools
import json
import os
import threading
from collections import Counter
from typing import Any
from urllib.parse import urlsplit

from traigent.utils.logging import get_logger

from .content_classes import BLOCKING, SchemaStoreError, classify_body, get_store

logger = get_logger(__name__)

MODE_ENV = "TRAIGENT_TRANSPORT_GATE"
MODES = ("off", "report", "refuse")
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_lock = threading.Lock()
_installed = False
_originals: list[tuple[Any, str, Any]] = []
_mode_override: str | None = None
_totals: Counter[str] = Counter()
_refusals: list[tuple[str, dict[str, int]]] = []


class TransportGateRefusal(RuntimeError):
    """A request was refused before send. Carries counts only, never values."""

    def __init__(self, route: str, counts: dict[str, int]) -> None:
        self.route = route
        self.counts = dict(counts)
        blocked = {k: v for k, v in counts.items() if k in BLOCKING}
        super().__init__(f"transport gate refused {route}: {blocked}")


def current_mode() -> str:
    mode = _mode_override or os.environ.get(MODE_ENV, "report").strip().lower()
    return mode if mode in MODES else "report"


def set_mode(mode: str | None) -> None:
    """Override the mode in-process (tests). ``None`` returns to the environment value."""
    global _mode_override
    if mode is not None and mode not in MODES:
        raise ValueError("unknown transport gate mode")
    _mode_override = mode


def report_totals() -> dict[str, int]:
    """Process-lifetime class -> count totals of everything the gate has seen (report mode)."""
    return dict(_totals)


def refusal_log() -> list[tuple[str, dict[str, int]]]:
    """(route template, class counts) of every refusal so far. Counts only."""
    return list(_refusals)


def clear_refusals() -> None:
    _refusals.clear()


def inspect_outgoing(
    method: str, url: str, body: bytes | str | None
) -> dict[str, int] | None:
    """Classify one outgoing request; returns the counts, or ``None`` if not classifiable.

    Raises :class:`TransportGateRefusal` in refuse mode. In report mode never raises.
    """
    mode = current_mode()
    if mode == "off" or method.upper() not in _BODY_METHODS or not body:
        return None
    refuse = mode == "refuse"
    try:
        route = get_store().find_route(method, urlsplit(str(url)).path)
        if route is None:
            return None
        template, schema = route
        parsed = json.loads(body)
        counts = dict(classify_body(parsed, schema, get_store()))
    except (SchemaStoreError, ValueError, KeyError, RecursionError) as exc:
        if refuse:
            raise TransportGateRefusal("unclassifiable", {"violation": 1}) from exc
        logger.debug(
            "transport gate could not classify a request (%s)", type(exc).__name__
        )
        return None
    _totals.update(counts)
    logger.debug("transport gate [%s] %s %s", mode, template, counts)
    if refuse and any(counts.get(c, 0) for c in BLOCKING):
        _refusals.append((template, counts))
        raise TransportGateRefusal(template, counts)
    return counts


def _json_bytes(obj: Any, dumps: Any = None) -> bytes:
    return (dumps or json.dumps)(obj).encode("utf-8")


def _body_from_kwargs(kwargs: dict[str, Any], dumps: Any = None) -> bytes | str | None:
    if kwargs.get("json") is not None:
        return _json_bytes(kwargs["json"], dumps)
    data = kwargs.get("data", kwargs.get("content"))
    if isinstance(data, (bytes, bytearray, str)):
        return bytes(data) if isinstance(data, bytearray) else data
    return None


def _wrap(owner: Any, name: str, make: Any) -> None:
    original = getattr(owner, name)
    _originals.append((owner, name, original))
    setattr(owner, name, make(original))


def install_transport_gate() -> bool:
    """Idempotently wrap the SDK's send libraries. Returns False when mode is ``off``."""
    global _installed
    if current_mode() == "off":
        return False
    with _lock:
        if _installed:
            return True
        _install_aiohttp()
        _install_requests()
        _install_httpx()
        _installed = True
    return True


def uninstall_transport_gate() -> None:
    global _installed
    with _lock:
        for owner, name, original in reversed(_originals):
            setattr(owner, name, original)
        _originals.clear()
        _installed = False


def _install_aiohttp() -> None:
    try:
        import aiohttp
    except ImportError:  # pragma: no cover
        return
    cls = aiohttp.ClientSession

    def verb(original: Any, fixed_method: str | None) -> Any:
        @functools.wraps(original)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if fixed_method:
                method, url = fixed_method, args[0] if args else kwargs.get("url", "")
            else:
                method = args[0] if args else kwargs.get("method", "")
                url = args[1] if len(args) > 1 else kwargs.get("url", "")
            body = _body_from_kwargs(kwargs, getattr(self, "_json_serialize", None))
            inspect_outgoing(str(method), str(url), body)
            return original(self, *args, **kwargs)

        return wrapper

    _wrap(cls, "request", lambda o: verb(o, None))
    for name in ("post", "put", "patch", "delete"):
        _wrap(cls, name, lambda o, n=name: verb(o, n.upper()))


def _install_requests() -> None:
    try:
        import requests
    except ImportError:  # pragma: no cover
        return

    def make(original: Any) -> Any:
        @functools.wraps(original)
        def send(self: Any, request: Any, **kwargs: Any) -> Any:
            body = request.body
            if isinstance(body, (bytes, str)):
                inspect_outgoing(str(request.method), str(request.url), body)
            return original(self, request, **kwargs)

        return send

    _wrap(requests.Session, "send", make)


def _install_httpx() -> None:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        return

    def make(original: Any) -> Any:
        @functools.wraps(original)
        def request(self: Any, method: str, url: Any, *args: Any, **kwargs: Any) -> Any:
            inspect_outgoing(str(method), str(url), _body_from_kwargs(kwargs))
            return original(self, method, url, *args, **kwargs)

        return request

    _wrap(httpx.Client, "request", make)
    _wrap(httpx.AsyncClient, "request", make)
