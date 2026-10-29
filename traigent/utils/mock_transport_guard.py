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
* ``botocore.httpsession.URLLib3Session.send`` -- the send boundary of every
  ``boto3`` / ``botocore`` client, including ``bedrock-runtime`` (#2811).
  ``aiobotocore`` / ``aioboto3`` send through aiohttp and are covered by the
  aiohttp patches above.
* ``grpc.secure_channel`` / ``grpc.insecure_channel`` (and the ``grpc.aio``
  twins) -- gRPC clients such as ``google-generativeai`` and Vertex AI open a
  channel to the provider host (#2811). The target is checked at channel
  creation, before any connection.

Direct ``urllib3`` use that bypasses ``requests`` and ``botocore`` is NOT
covered.

httpx/httpx2 are guarded at ``_send_single_request`` (called once per redirect
hop by ``_send_handling_redirects``) as well as at ``send``, so a redirect from
an allowed host to a provider host is blocked before the second hop is sent.

The mock flag is read at call time, so the guard is inert when mock mode is
off. Errors are closed: only the (fixed-list) host name is named, never the
path, query, headers, body or credentials.

The covered hosts are per SDK provider (``PROVIDER_HOSTS``); a test pins that
every provider in ``traigent.config.provider_support.PROVIDER_SPECS`` has at
least one covered host (#2811).
"""

from __future__ import annotations

import functools
import re
import threading
from typing import Any
from urllib.parse import urlsplit

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "GUARDED_TRANSPORTS",
    "KNOWN_LLM_PROVIDER_HOSTS",
    "MOCK_SCOPE_SUMMARY",
    "MockModeTransportBlockedError",
    "PROVIDER_HOSTS",
    "install_mock_transport_guard",
    "is_known_llm_provider_host",
    "mock_blocked_calls",
]

# Hosts per provider the SDK registers (keys are the canonical names in
# ``traigent.config.provider_support.PROVIDER_SPECS``; a test fails when a
# registered provider is missing here). Each value is (literal hosts, anchored
# patterns, one DNS label per wildcard).
PROVIDER_HOSTS: dict[str, tuple[frozenset[str], tuple[re.Pattern[str], ...]]] = {
    "openai": (frozenset({"api.openai.com"}), ()),
    "anthropic": (frozenset({"api.anthropic.com"}), ()),
    "google": (
        frozenset({"generativelanguage.googleapis.com", "aiplatform.googleapis.com"}),
        # Vertex AI regional endpoints, e.g. us-central1-aiplatform.googleapis.com.
        (re.compile(r"^[a-z0-9-]+-aiplatform\.googleapis\.com$"),),
    ),
    "mistral": (frozenset({"api.mistral.ai"}), ()),
    "cohere": (frozenset({"api.cohere.ai", "api.cohere.com"}), ()),
    "huggingface": (
        frozenset({"api-inference.huggingface.co", "router.huggingface.co"}),
        (re.compile(r"^[a-z0-9-]+\.endpoints\.huggingface\.cloud$"),),
    ),
    "azure_openai": (
        frozenset(),
        (
            re.compile(r"^[a-z0-9-]+\.openai\.azure\.com$"),
            # Deliberately broad: Azure OpenAI is served from here too, and the
            # host cannot tell it from Speech/Vision resources, which are then
            # blocked in a mock run as well.
            re.compile(r"^[a-z0-9-]+\.cognitiveservices\.azure\.com$"),
            re.compile(r"^[a-z0-9-]+\.services\.ai\.azure\.com$"),
        ),
    ),
    "bedrock": (
        frozenset(),
        (
            # bedrock, bedrock-runtime, bedrock-agent(-runtime), -fips variants.
            re.compile(r"^bedrock[a-z-]*\.[a-z0-9-]+\.amazonaws\.com(\.cn)?$"),
            # Interface VPC endpoints.
            re.compile(
                r"^([a-z0-9-]+\.)?bedrock[a-z-]*\.[a-z0-9-]+\.vpce\.amazonaws\.com$"
            ),
        ),
    ),
    "nous": (frozenset({"inference-api.nousresearch.com"}), ()),
}

# Hosts of providers LiteLLM reaches that are not in the SDK provider registry.
_OTHER_PROVIDER_HOSTS: frozenset[str] = frozenset(
    {
        "api.groq.com",
        "api.together.xyz",
        "api.deepseek.com",
        "openrouter.ai",
        "api.x.ai",
        "api.fireworks.ai",
        "api.perplexity.ai",
    }
)

KNOWN_LLM_PROVIDER_HOSTS: frozenset[str] = _OTHER_PROVIDER_HOSTS.union(
    *(hosts for hosts, _patterns in PROVIDER_HOSTS.values())
)

_PROVIDER_HOST_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    pattern for _hosts, patterns in PROVIDER_HOSTS.values() for pattern in patterns
)

#: What mock mode simulates, shared by the activation banner and the guard
#: error so the two cannot disagree (#2811, #2813).
MOCK_SCOPE_SUMMARY = (
    "Mock mode simulates LiteLLM completion/acompletion, the SDK's Bedrock chat "
    "client and the LangChain ChatOpenAI, ChatAnthropic and langchain_aws "
    "Bedrock chat classes. "
    "Embeddings (litellm.embedding, LangChain Embeddings) and other chat "
    "classes (for example ChatGoogleGenerativeAI) are not mocked."
)

#: Transports the guard covers, for the banner and docs (#2811).
GUARDED_TRANSPORTS = "httpx, aiohttp, requests, botocore (boto3) and gRPC channels"


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


_blocked_lock = threading.Lock()
_blocked_hosts: dict[str, int] = {}


def mock_blocked_calls() -> dict[str, int]:
    """Snapshot of blocked-call counts per host since process start (#2812).

    Only fixed-list provider host names appear, never paths or credentials.
    """
    with _blocked_lock:
        return dict(_blocked_hosts)


def _reset_blocked_calls_for_tests() -> None:
    with _blocked_lock:
        _blocked_hosts.clear()


def _record_block(host: str) -> bool:
    """Count a block; return True the first time ``host`` is blocked."""
    with _blocked_lock:
        first = host not in _blocked_hosts
        _blocked_hosts[host] = _blocked_hosts.get(host, 0) + 1
    return first


def _raise_if_blocked(url: Any) -> None:
    host = _host_of(url)
    if not is_known_llm_provider_host(host):
        return
    from traigent.utils.env_config import is_mock_llm

    if is_mock_llm():
        name = str(host).lower()
        message = (
            f"Traigent mock mode blocked a request to a known LLM provider "
            f"host ({name}). Mock mode promises no real LLM calls. "
            f"{MOCK_SCOPE_SUMMARY} To allow real requests, call "
            "traigent.testing.disable_mock_mode() (and unset the legacy mock "
            "environment flag)."
        )
        # Vendor SDKs wrap this error into a bare "Connection error." that
        # reads like an outage (#2812), so say it once where it cannot be
        # swallowed. One warning per host keeps a many-trial run readable.
        if _record_block(name):
            logger.warning("%s", message)
        from traigent.utils.mock_provenance import note_mock_blocked

        note_mock_blocked(name)
        raise MockModeTransportBlockedError(message)


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
            import aiohttp.client
            import aiohttp.connector
        except ImportError:
            aiohttp = None  # type: ignore[assignment]
        if aiohttp is not None:
            # Patch the real class from ``aiohttp.client``: the public
            # ``aiohttp.ClientSession`` name may be replaced by a stand-in
            # (tests) that has no transport of its own.
            _patch(
                aiohttp.client.ClientSession,
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
        _install_botocore_guard()
        _install_grpc_guard()


def _install_botocore_guard() -> None:
    """Guard ``botocore``'s send boundary (every boto3 client, #2811)."""
    try:
        import botocore.httpsession
    except ImportError:
        return
    _patch(
        botocore.httpsession.URLLib3Session,
        "send",
        _wrap_sync,
        lambda self, request, *a, **k: request.url,
    )


class _GrpcTarget:
    """Adapter exposing the host of a gRPC target string as ``.host``."""

    def __init__(self, target: Any) -> None:
        self.host = _grpc_target_host(target)


def _grpc_target_host(target: Any) -> str | None:
    """Host of a gRPC target (``host:port``, ``dns:///host:443``, ``[::1]:443``)."""
    if not isinstance(target, str):
        return None
    text = target.strip()
    # A resolver scheme: ``dns:///h:443``, ``passthrough:///h:443``,
    # ``xds:///h``, ``dns://authority/h:443`` or ``ipv4:h``. A bare ``host:443``
    # has a numeric "scheme" and is left alone.
    scheme, sep, rest = text.partition(":")
    if sep and scheme.isalpha() and not scheme.isdigit() and not rest[:1].isdigit():
        text = rest
        if text.startswith("//"):
            # ``//authority/host:port`` (empty authority for ``///``).
            text = text[2:].split("/", 1)[1] if "/" in text[2:] else text[2:]
    text = text.rsplit("@", 1)[-1]
    text = text.split("/", 1)[0]
    if text.startswith("["):
        return text[1:].split("]", 1)[0] or None
    host, _sep, port = text.rpartition(":")
    return (host if port.isdigit() and host else text) or None


def _install_grpc_guard() -> None:
    """Guard gRPC channel creation to known provider hosts (#2811)."""
    try:
        import grpc
    except ImportError:
        return

    def target_of(target: Any, *a: Any, **k: Any) -> _GrpcTarget:
        return _GrpcTarget(target)

    def wrap_factory(original: Any, url_of: Any) -> Any:
        @functools.wraps(original)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            target = args[0] if args else kwargs.get("target")
            _raise_if_blocked(url_of(target))
            return original(*args, **kwargs)

        guarded._traigent_mock_transport_guard = True  # type: ignore[attr-defined]
        guarded._traigent_original = original  # type: ignore[attr-defined]
        return guarded

    owners: list[Any] = [grpc]
    try:
        import grpc.aio

        owners.append(grpc.aio)
    except ImportError:
        pass
    for owner in owners:
        for name in ("secure_channel", "insecure_channel"):
            if hasattr(owner, name):
                _patch(owner, name, wrap_factory, target_of)
