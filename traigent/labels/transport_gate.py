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

import ipaddress
import json
import os
import threading
import urllib.request
from collections import Counter
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlencode, urlsplit

from traigent.utils import optimization_only_egress as oo_egress
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

    def __init__(
        self, route: str, counts: dict[str, int], *, cause_type: str | None = None
    ) -> None:
        self.route = route
        self.cause_type = cause_type  # type name only, never the cause's text
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
        host = parts.hostname.lower().rstrip(".")
        try:  # one spelling per IP address (compressed IPv6, no brackets)
            host = str(ipaddress.ip_address(host))
        except ValueError:
            pass
        return parts.scheme.lower(), host, parts.port or default
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
            # No printable cause: it can quote a path or request content.
            raise TransportGateRefusal(
                "unclassifiable", {"violation": 1}, cause_type=type(exc).__name__
            ) from None
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
    # Strict means THIS instance's registered origins only: a process-wide configured
    # backend that this session was not built for is not accepted either.
    if origin is None or origin not in _instance_origins(client):
        raise TransportGateRefusal("unrecognised_origin", {"violation": 1})


def _is_redirect_status(status: int) -> bool:
    """Every 3xx (including 304/305/306): a strict session never returns one to a caller."""
    return 300 <= int(status) < 400


def _is_strict(client: Any) -> bool:
    return bool(getattr(client, "_traigent_gate_strict", False)) and (
        current_mode() != "off"
    )


def _refuses_redirects(client: Any, url: str) -> bool:
    """True when this request must not follow redirects.

    Strict sessions always refuse them. Any request to the origin admitted by the
    self-hosted opt-in (``TRAIGENT_ALLOW_PRIVATE_BACKEND``) refuses them too, so the
    credential-bearing private backend cannot be redirected to an address the
    validator never checked.
    """
    if _is_strict(client) or _oo_redirects_refused(client):
        return True
    if current_mode() == "off":
        return False
    from traigent.cloud.url_security import private_backend_opt_in_covers

    return private_backend_opt_in_covers(url)


def _refuse_redirect(client: Any = None) -> Exception:
    """A strict session never follows a redirect: it would carry the body and credentials
    to an origin the gate has not inspected. While optimization_only is active no redirect is
    followed either, and a 3xx is a local refusal."""
    if client is not None and _oo_redirects_refused(client):
        return oo_egress.OptimizationOnlyRefusal("redirect_refused")
    return TransportGateRefusal("redirect_refused", {"violation": 1})


async def _latch_aiohttp(response: Any, url: Any = "", headers: Any = None) -> None:
    """Latch optimization_only mode, for the request's credential, on the server's closed
    403 (best effort, never raises)."""
    try:
        if response.status == 403:
            oo_egress.note_response(
                403,
                await response.read(),
                oo_egress.credential_scope(url, headers),
            )
    except Exception:  # noqa: BLE001 - observation must never break the request
        return


def _latch_sync(response: Any, status_attr: str, body_attr: str) -> None:
    """requests / httpx response: scope comes from the response's own request."""
    try:
        if getattr(response, status_attr) == 403:
            request = getattr(response, "request", None)
            oo_egress.note_response(
                403,
                getattr(response, body_attr),
                oo_egress.credential_scope(
                    getattr(request, "url", "") or getattr(response, "url", ""),
                    getattr(request, "headers", None),
                ),
            )
    except Exception:  # noqa: BLE001 - observation must never break the request
        return


def observe_requests_response(response: Any) -> None:
    """Observe a module-level ``requests`` response (``requests.post(...)`` etc.), which
    ``preflight_requests`` cannot see. Safe to call on any object."""
    _latch_sync(response, "status_code", "content")


def _configured_api_prefixes() -> frozenset[str]:
    """Path part of the configured backend API base (``/proxy/api/v1`` style bases)."""
    try:
        from traigent.config.backend_config import BackendConfig

        return frozenset(
            {urlsplit(BackendConfig.get_backend_api_url()).path.rstrip("/")}
        )
    except Exception:  # noqa: BLE001 - silent-ok: default prefix handling still applies
        return frozenset()


#: Normalised origins of every NON-external gated session (the Traigent backends the SDK talks
#: to), so an external-marked session can never be pointed back at one of them.
_traigent_origin_registry: set[str] = set()


def _known_traigent_origins() -> set[str]:
    known = set(_traigent_origin_registry) | oo_egress.latched_origins()
    for scheme, host, port in backend_origins():
        shown = f"[{host}]" if ":" in host else host
        known.add(oo_egress.normalised_origin(f"{scheme}://{shown}:{port}"))
    return known


def _refuse_external_to_traigent(url: Any) -> None:
    if oo_egress.normalised_origin(url) in _known_traigent_origins():
        raise oo_egress.OptimizationOnlyRefusal("external_session_to_traigent_origin")


def enforce_optimization_only(
    method: str,
    url: Any,
    *,
    headers: Any = None,
    body: Any = None,
    explicit: bool | None = None,
    api_prefixes: Iterable[str] = (),
) -> bytes | None:
    """Transport backstop (default-deny while optimization_only is active).

    Called by every wrapper with the FINAL method, URL, headers and body, immediately before
    bytes leave, for requests to ANY origin. Sessions the SDK creates for customer or
    third-party endpoints are marked ``external`` at creation and judged only for "is this
    actually a Traigent backend origin" (refused if so); provider SDKs do not use the gate.
    Mode off and nothing latched: one cheap check. Returns the canonical body bytes to
    transmit in place of the original, or ``None``.
    """
    if not oo_egress.outgoing_may_be_active(explicit):
        return None
    return oo_egress.enforce_outgoing(
        method,
        url,
        headers,
        body,
        explicit=explicit,
        api_prefixes={*api_prefixes, *_configured_api_prefixes()},
    )


def _enforce_for_session(
    session: Any, method: str, url: Any, headers_fn: Any, body_fn: Any
) -> bytes | None:
    """The wrapper hook: judge one request. External sessions are refused if their target is
    a known Traigent origin; otherwise skipped. Returns replacement body bytes or ``None``."""
    explicit = getattr(session, "_traigent_oo_explicit", None)
    if not oo_egress.outgoing_may_be_active(explicit):
        return None
    if getattr(session, "_traigent_oo_external", False):
        _refuse_external_to_traigent(url)
        return None
    return enforce_optimization_only(
        method,
        url,
        headers=headers_fn(),
        body=body_fn(),
        explicit=explicit,
        api_prefixes=getattr(session, "_traigent_oo_prefixes", ()),
    )


def _oo_redirects_refused(session: Any) -> bool:
    """While the mode is active NO gated session (external ones included) follows a redirect."""
    return oo_egress.outgoing_may_be_active(
        getattr(session, "_traigent_oo_explicit", None)
    )


def _aiohttp_body(kwargs: dict[str, Any]) -> Any:
    if kwargs.get("json") is not None:
        return kwargs["json"]
    data = kwargs.get("data")
    if data is None:
        return None
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        return dict(data)  # form fields (auth/oauth routes judge the field names)
    return oo_egress.UNREADABLE_BODY


def _bytes_body(body: Any) -> Any:
    if body is None or isinstance(body, (bytes, str)):
        return body
    if isinstance(body, bytearray):
        return bytes(body)
    return oo_egress.UNREADABLE_BODY


class GatedOpenerDirector(urllib.request.OpenerDirector):
    """``urllib`` opener whose every ``open`` passes the optimization_only backstop first."""

    def open(self, fullurl: Any, data: Any = None, *args: Any, **kwargs: Any) -> Any:
        if isinstance(fullurl, urllib.request.Request):
            replacement = enforce_optimization_only(
                fullurl.get_method(),
                fullurl.full_url,
                headers=dict(fullurl.header_items()),
                body=_bytes_body(data if data is not None else fullurl.data),
            )
            if replacement is not None:
                fullurl.data = replacement
                fullurl.add_unredirected_header("Content-length", str(len(replacement)))
                data = None
        else:
            replacement = enforce_optimization_only(
                "POST" if data is not None else "GET",
                str(fullurl),
                body=_bytes_body(data),
            )
            if replacement is not None:
                data = replacement
        return super().open(fullurl, data, *args, **kwargs)


def gate_opener(opener: urllib.request.OpenerDirector) -> urllib.request.OpenerDirector:
    """Re-class ONE urllib opener so its sends pass the transport gate; returns it."""
    if not isinstance(opener, GatedOpenerDirector) and isinstance(
        opener, urllib.request.OpenerDirector
    ):
        opener.__class__ = _derived(
            type(opener),
            "urllib",
            lambda base: type(f"Gated{base.__name__}", (GatedOpenerDirector, base), {}),
        )
    return opener


def _effective_headers(session: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Session default headers overlaid with the per-request ones (for credential scope)."""
    merged: dict[str, Any] = {}
    for source in (getattr(session, "headers", None), kwargs.get("headers")):
        try:
            merged.update(dict(source or {}))
        except (TypeError, ValueError):
            continue
    return merged


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
            replacement = _enforce_for_session(
                self,
                str(method),
                url,
                lambda: _effective_headers(self, kwargs),
                lambda: _aiohttp_body(kwargs),
            )
            if replacement is not None and kwargs.get("json") is None:
                kwargs["data"] = replacement  # send exactly what was judged
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
            if not _refuses_redirects(self, url):
                response = await super()._request(method, str_or_url, **kwargs)
                await _latch_aiohttp(response, url, _effective_headers(self, kwargs))
                return response
            kwargs["allow_redirects"] = False
            response = await super()._request(method, str_or_url, **kwargs)
            if _is_redirect_status(response.status):
                response.release()
                raise _refuse_redirect(self)
            await _latch_aiohttp(response, url, _effective_headers(self, kwargs))
            return response

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
            replacement = _enforce_for_session(
                self,
                str(request.method),
                str(request.url),
                lambda: request.headers,
                lambda: _bytes_body(request.body),
            )
            if replacement is not None:
                request.body = replacement
                request.headers["Content-Length"] = str(len(replacement))
            inspect_outgoing(
                str(request.method),
                str(request.url),
                bytes(body) if isinstance(body, (bytes, bytearray)) else None,
                body_present=None if readable else True,
                extra_origins=_instance_origins(self),
            )
            if not _refuses_redirects(self, str(request.url)):
                response = super().send(request, **kwargs)
                _latch_sync(response, "status_code", "content")
                return response
            kwargs["allow_redirects"] = False
            response = super().send(request, **kwargs)
            if _is_redirect_status(response.status_code):
                response.close()
                raise _refuse_redirect(self)
            _latch_sync(response, "status_code", "content")
            return response

    GatedRequestsSession.__name__ = f"Gated{base.__name__}"
    return GatedRequestsSession


def _replace_httpx_body(request: Any, content: bytes) -> None:
    """Swap an httpx request's body for the canonical, validated bytes."""
    import httpx

    request.stream = httpx.ByteStream(content)
    request._content = content  # noqa: SLF001 - httpx has no public setter
    request.headers["Content-Length"] = str(len(content))


def _httpx_body(request: Any) -> tuple[bytes | None, bool]:
    try:
        return bytes(request.content), False
    except Exception:  # noqa: BLE001 - streaming body not yet read
        return None, True


def _latch_httpx_sync(response: Any) -> None:
    try:
        if response.status_code == 403:
            response.read()  # a 403 body is small; makes ``.content`` available
    except Exception:  # noqa: BLE001 - observation must never break the request
        return
    _latch_sync(response, "status_code", "content")


async def _latch_httpx_async(response: Any) -> None:
    try:
        if response.status_code == 403:
            await response.aread()
    except Exception:  # noqa: BLE001 - observation must never break the request
        return
    _latch_sync(response, "status_code", "content")


def _build_httpx_sync(base: type) -> type:
    class GatedHttpxClient(base):  # type: ignore[valid-type, misc]
        _traigent_gated = True

        def send(self, request: Any, **kwargs: Any) -> Any:
            body, unreadable = _httpx_body(request)
            _check_strict(self, str(request.url))
            replacement = _enforce_for_session(
                self,
                str(request.method),
                str(request.url),
                lambda: request.headers,
                lambda: oo_egress.UNREADABLE_BODY if unreadable else body,
            )
            if replacement is not None:
                _replace_httpx_body(request, replacement)
            inspect_outgoing(
                str(request.method),
                str(request.url),
                body,
                body_present=True if unreadable else None,
                extra_origins=_instance_origins(self),
            )
            if not _refuses_redirects(self, str(request.url)):
                response = super().send(request, **kwargs)
                _latch_httpx_sync(response)
                return response
            kwargs["follow_redirects"] = False
            response = super().send(request, **kwargs)
            if _is_redirect_status(response.status_code):
                response.close()
                raise _refuse_redirect(self)
            _latch_httpx_sync(response)
            return response

    GatedHttpxClient.__name__ = f"Gated{base.__name__}"
    return GatedHttpxClient


def _build_httpx_async(base: type) -> type:
    class GatedHttpxAsyncClient(base):  # type: ignore[valid-type, misc]
        _traigent_gated = True

        async def send(self, request: Any, **kwargs: Any) -> Any:
            body, unreadable = _httpx_body(request)
            _check_strict(self, str(request.url))
            replacement = _enforce_for_session(
                self,
                str(request.method),
                str(request.url),
                lambda: request.headers,
                lambda: oo_egress.UNREADABLE_BODY if unreadable else body,
            )
            if replacement is not None:
                _replace_httpx_body(request, replacement)
            inspect_outgoing(
                str(request.method),
                str(request.url),
                body,
                body_present=True if unreadable else None,
                extra_origins=_instance_origins(self),
            )
            if not _refuses_redirects(self, str(request.url)):
                response = await super().send(request, **kwargs)
                await _latch_httpx_async(response)
                return response
            kwargs["follow_redirects"] = False
            response = await super().send(request, **kwargs)
            if _is_redirect_status(response.status_code):
                await response.aclose()
                raise _refuse_redirect(self)
            await _latch_httpx_async(response)
            return response

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
    optimization_only: bool | None = None,
    external: bool = False,
) -> Any:
    """Gate ONE SDK-created client instance in place and return it.

    Re-classes the instance so its send funnel inspects requests. Mocks and unknown
    objects are returned untouched; ``TRAIGENT_TRANSPORT_GATE=off`` disables gating.

    ``origins`` are the URLs (API bases) this instance sends to; they are inspected in
    addition to the configured backend, for THIS instance only (no global state), so a
    per-client API base is gated like the default one. ``strict`` makes the instance fail
    closed on any other origin (content-less sessions).
    """
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
        if optimization_only is not None:
            object.__setattr__(client, "_traigent_oo_explicit", optimization_only)
        if external:
            object.__setattr__(client, "_traigent_oo_external", True)
        else:
            _traigent_origin_registry.update(
                n for u in (origins or ()) if (n := oo_egress.normalised_origin(u))
            )
        prefixes = frozenset(
            urlsplit(str(u)).path.rstrip("/") for u in (origins or ()) if str(u)
        )
        if prefixes:
            object.__setattr__(
                client,
                "_traigent_oo_prefixes",
                getattr(client, "_traigent_oo_prefixes", frozenset()) | prefixes,
            )
    return client


def mark_explicit(session: Any, explicit: bool | None) -> Any:
    """Record a client's explicit optimization_only setting on its gated session (no-op for
    ungated objects and mocks); returns the session."""
    if explicit is not None and getattr(type(session), "_traigent_gated", False):
        object.__setattr__(session, "_traigent_oo_explicit", explicit)
    return session


def preflight_requests(
    method: str, url: str, *, external: bool = False, **kwargs: Any
) -> None:
    """Inspect a module-level ``requests.<method>(url, ...)`` call before it is made.

    The body is prepared exactly as ``requests`` will prepare it (deterministic JSON
    encoding); the caller then makes its normal call.
    """
    if external:
        if oo_egress.outgoing_may_be_active(None):
            _refuse_external_to_traigent(_with_params(str(url), kwargs.get("params")))
    else:
        enforce_optimization_only(
            method,
            _with_params(str(url), kwargs.get("params")),
            headers=kwargs.get("headers"),
            body=(
                kwargs["json"]
                if kwargs.get("json") is not None
                else _aiohttp_body({"data": kwargs.get("data")})
            ),
        )
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
        outgoing_body: bytes | None
        if isinstance(body, str):
            outgoing_body = body.encode()
        elif body:
            outgoing_body = bytes(body)
        else:
            outgoing_body = None
        inspect_outgoing(
            method,
            str(prepared.url),
            outgoing_body,
            body_present=None if readable else True,
        )
    except TransportGateRefusal:
        raise
    except Exception as exc:  # noqa: BLE001
        if current_mode() == "refuse":
            # No printable cause: it can quote a path or request content.
            raise TransportGateRefusal(
                "unclassifiable", {"violation": 1}, cause_type=type(exc).__name__
            ) from None
        with _lock:
            _errors[type(exc).__name__] += 1
