"""Transport-level guard: block known LLM provider hosts while mock mode is on.

Mock mode (``TRAIGENT_MOCK_LLM=true`` or
:func:`traigent.testing.enable_mock_mode_for_quickstart`) promises that no real
LLM call is made. Interceptors cover LiteLLM and LangChain entry points, but a
raw ``openai`` / ``anthropic`` client, or a LiteLLM call that supplies its own
``litellm_logging_obj``, never passes through them (#2418). This guard sits at
the HTTP send boundary instead:

* ``httpx.Client.send`` / ``httpx.AsyncClient.send`` (and the same on
  ``httpx2``, which the installed ``anthropic`` uses) -- the call made by
  ``openai`` (``_base_client.py`` ``_send_request``), ``anthropic``
  (``_base_client.py`` ``self._client.send``) and LiteLLM's ``HTTPHandler`` /
  ``AsyncHTTPHandler`` (``self.client.send``), including its aiohttp transport,
  which is an httpx transport reached through ``AsyncClient.send``.
* ``aiohttp.connector.BaseConnector.connect`` -- LiteLLM's
  ``BaseLLMAIOHTTPHandler`` calls ``ClientSession.post`` directly, bypassing
  httpx. ``connect`` receives the ``ClientRequest`` whose ``url`` is already
  resolved (``base_url`` + relative path, yarl-canonicalized) and is entered
  once per redirect hop and per pooled reuse. ``ClientSession._request`` is
  also checked (early, unresolved) for a clearer failure point.
* ``requests.Session.send`` -- ``resolve_redirects`` calls it for every hop.

httpx/httpx2 are guarded at ``_send_single_request`` (called once per redirect
hop by ``_send_handling_redirects``) as well as at ``send``, so a redirect from
an allowed host to a provider host is blocked before the second hop is sent.

The mock flag is read at call time, so the guard is inert when mock mode is
off. Errors are closed: only the (fixed-list) host name is named, never the
path, query, headers, body or credentials.
"""

from __future__ import annotations

import functools
import re
import threading
from typing import Any
from urllib.parse import urlsplit

__all__ = [
    "KNOWN_LLM_PROVIDER_HOSTS",
    "MockModeTransportBlockedError",
    "install_mock_transport_guard",
    "is_known_llm_provider_host",
]

KNOWN_LLM_PROVIDER_HOSTS: frozenset[str] = frozenset(
    {
        "api.openai.com",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
        "api.mistral.ai",
        "api.cohere.ai",
        "api.cohere.com",
        "api.groq.com",
        "api.together.xyz",
        "api.deepseek.com",
        "openrouter.ai",
        "api.x.ai",
        "api.fireworks.ai",
    }
)

# Documented suffix patterns: Bedrock runtime endpoints and Azure OpenAI
# resources. Anchored on both ends; one DNS label for the wildcard part.
_PROVIDER_HOST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^bedrock-runtime(-fips)?\.[a-z0-9-]+\.amazonaws\.com(\.cn)?$"),
    re.compile(r"^[a-z0-9-]+\.openai\.azure\.com$"),
)


class MockModeTransportBlockedError(RuntimeError):
    """A request to a known LLM provider host was blocked under mock mode."""


def _canonical_host(host: str) -> str:
    """Lowercase, strip a trailing dot, and IDNA-normalize ``host``.

    Unicode label separators (e.g. U+3002) and non-ASCII look-alikes are folded
    the way the URL libraries fold them before connecting, so the match sees the
    same name the socket layer will resolve. A name IDNA rejects is matched
    as-is (lowercased).
    """
    cleaned = host.strip().rstrip(".\u3002\uff0e\uff61").lower()
    try:
        return cleaned.encode("idna").decode("ascii").rstrip(".")
    except UnicodeError:
        return cleaned


def is_known_llm_provider_host(host: str | None) -> bool:
    """True iff ``host`` (parsed, case-insensitive) is a known provider host."""
    if not host:
        return False
    normalized = _canonical_host(host)
    if normalized in KNOWN_LLM_PROVIDER_HOSTS:
        return True
    return any(p.match(normalized) for p in _PROVIDER_HOST_PATTERNS)


def _host_of(url: Any) -> str | None:
    host = getattr(url, "host", None)
    if isinstance(host, str):
        return host
    try:
        return urlsplit(str(url)).hostname
    except ValueError:
        return None


def _raise_if_blocked(url: Any) -> None:
    host = _host_of(url)
    if not is_known_llm_provider_host(host):
        return
    from traigent.utils.env_config import is_mock_llm

    if is_mock_llm():
        raise MockModeTransportBlockedError(
            f"Traigent mock mode blocked a request to a known LLM provider "
            f"host ({str(host).lower()}). Mock mode promises no real LLM "
            "calls; call traigent.testing.disable_mock_mode() (and unset "
            "TRAIGENT_MOCK_LLM) to allow real requests."
        )


_install_lock = threading.RLock()


def _wrap_sync(original: Any, url_of: Any) -> Any:
    @functools.wraps(original)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        _raise_if_blocked(url_of(self, *args, **kwargs))
        return original(self, *args, **kwargs)

    guarded._traigent_mock_transport_guard = True  # type: ignore[attr-defined]
    guarded._traigent_original = original  # type: ignore[attr-defined]
    return guarded


def _wrap_async(original: Any, url_of: Any) -> Any:
    @functools.wraps(original)
    async def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        _raise_if_blocked(url_of(self, *args, **kwargs))
        return await original(self, *args, **kwargs)

    guarded._traigent_mock_transport_guard = True  # type: ignore[attr-defined]
    guarded._traigent_original = original  # type: ignore[attr-defined]
    return guarded


def _patch(owner: Any, name: str, wrapper_factory: Any, url_of: Any) -> None:
    current = getattr(owner, name)
    if getattr(current, "_traigent_mock_transport_guard", False):
        return
    setattr(owner, name, wrapper_factory(current, url_of))


def install_mock_transport_guard() -> None:
    """Install the guard on httpx (and aiohttp / requests when importable).

    Idempotent and thread-safe. ``httpx`` is a hard SDK dependency, so a
    failure to patch it raises (callers fail closed). ``aiohttp`` and
    ``requests`` are patched only when the package is importable; an import
    failure means the library is absent, but a patch failure raises.
    """
    with _install_lock:
        import httpx

        # ``httpx2`` is the separate package newer ``anthropic`` releases use
        # for their HTTP client (httpx objects are rejected there); guard it
        # exactly like httpx when it is installed.
        http_modules = [httpx]
        try:
            import httpx2
        except ImportError:
            httpx2 = None  # type: ignore[assignment]
        if httpx2 is not None:
            http_modules.append(httpx2)
        for module in http_modules:
            for name in ("send", "_send_single_request"):
                _patch(
                    module.Client,
                    name,
                    _wrap_sync,
                    lambda self, request, *a, **k: request.url,
                )
                _patch(
                    module.AsyncClient,
                    name,
                    _wrap_async,
                    lambda self, request, *a, **k: request.url,
                )
        try:
            import aiohttp
            import aiohttp.connector
        except ImportError:
            aiohttp = None  # type: ignore[assignment]
        if aiohttp is not None:
            _patch(
                aiohttp.ClientSession,
                "_request",
                _wrap_async,
                lambda self, method, str_or_url, *a, **k: str_or_url,
            )
            _patch(
                aiohttp.connector.BaseConnector,
                "connect",
                _wrap_async,
                lambda self, req, *a, **k: req.url,
            )
        try:
            import requests
        except ImportError:
            requests = None  # type: ignore[assignment]
        if requests is not None:
            _patch(
                requests.Session,
                "send",
                _wrap_sync,
                lambda self, request, *a, **k: request.url,
            )
