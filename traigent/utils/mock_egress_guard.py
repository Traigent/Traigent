"""Fail-closed egress guard for mock mode.

Mock mode intercepts LiteLLM and LangChain calls, but a call that bypasses
those interceptors (a raw ``openai``/``anthropic`` client, or a function bound
with ``from litellm import completion`` before the patch) would otherwise reach
the real provider and be billed. While mock mode is on, this guard blocks HTTP
requests from the common Python clients (httpx, requests) to known model
providers, or to a custom ``*_BASE_URL`` / ``*_API_BASE`` host the user
configured for one. Every redirect hop is checked, not only the first URL.

Limits: it does not block all egress. Other clients (aiohttp, urllib, raw
sockets, non-Python subprocesses) are not covered.

The guard is installed at ``import traigent`` and in
``enable_mock_mode_for_quickstart()``. It is installed once per process and
checks mock mode on every request, so it is inert when mock mode is off.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Security FUNC-INTEGRATIONS SYNC-IntegrationHook

from __future__ import annotations

import importlib
import importlib.util
import ipaddress
import os
import threading
from typing import Any
from urllib.parse import urlparse

from traigent.utils.logging import get_logger

logger = get_logger(__name__)

# Host suffixes of hosted model providers. Matching is on the host or any
# parent domain (``eu.api.openai.com`` matches ``openai.com``-style entries
# only where listed here).
_PROVIDER_HOST_SUFFIXES: tuple[str, ...] = (
    "api.openai.com",
    "openai.azure.com",
    "api.anthropic.com",
    "openrouter.ai",
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
    "api.cohere.com",
    "api.cohere.ai",
    "api.mistral.ai",
    "api.groq.com",
    "api.together.xyz",
    "api.together.ai",
    "api.fireworks.ai",
    "api.deepseek.com",
    "api.x.ai",
    "api.perplexity.ai",
    "api.ai21.com",
    "api.replicate.com",
    "api-inference.huggingface.co",
    "router.huggingface.co",
    "api.cerebras.ai",
    "api.sambanova.ai",
    "api.voyageai.com",
    "api.jina.ai",
    "bedrock-runtime.amazonaws.com",
)
_PROVIDER_HOST_SUBSTRINGS: tuple[str, ...] = ("bedrock-runtime.",)
_BASE_URL_ENV_VARS: tuple[str, ...] = (
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_BASE",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_API_BASE",
    "OPENROUTER_API_BASE",
    "LITELLM_API_BASE",
)

_install_lock = threading.Lock()
_installed = False
_warned = False


class MockEgressBlockedError(RuntimeError):
    """Raised when mock mode refuses a request that would reach a model provider."""


def _is_local(host: str) -> bool:
    if host in ("localhost", "") or host.endswith(".localhost"):
        return True
    try:
        addr = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private or addr.is_unspecified


def _configured_provider_hosts() -> set[str]:
    hosts: set[str] = set()
    for var in _BASE_URL_ENV_VARS:
        value = os.environ.get(var)
        if not value:
            continue
        host = (
            urlparse(value if "//" in value else f"//{value}").hostname or ""
        ).lower()
        if host and not _is_local(host):
            hosts.add(host)
    return hosts


def is_provider_host(host: str | None) -> bool:
    """Whether ``host`` is a model-provider endpoint mock mode must not reach."""
    if not host:
        return False
    host = host.lower().rstrip(".")
    if _is_local(host):
        return False
    if host in _configured_provider_hosts():
        return True
    if any(host == s or host.endswith("." + s) for s in _PROVIDER_HOST_SUFFIXES):
        return True
    return any(s in host for s in _PROVIDER_HOST_SUBSTRINGS)


def _check_url(url: Any) -> None:
    from traigent.utils.env_config import is_mock_llm

    if not is_mock_llm():
        return
    host = getattr(url, "host", None)
    if host is None:
        host = urlparse(str(url)).hostname
    if not is_provider_host(host):
        return
    global _warned
    message = (
        f"Traigent MOCK MODE blocked an outgoing request to model provider "
        f"'{host}'. Mock mode never reaches a provider, but this call is not "
        f"intercepted (raw openai/anthropic clients and LLM calls bound "
        f"before Traigent patched them are not mocked). Route the call through "
        f"`import litellm; litellm.completion(...)` or a LangChain chat "
        f"model, or turn mock mode off to make a real, billed call."
    )
    if not _warned:
        _warned = True
        logger.error(message)
    raise MockEgressBlockedError(message)


def _wrap_httpx(httpx: Any) -> None:
    original_send = httpx.Client.send
    original_asend = httpx.AsyncClient.send

    def send(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        _check_url(request.url)
        return original_send(self, request, *args, **kwargs)

    async def asend(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        _check_url(request.url)
        return await original_asend(self, request, *args, **kwargs)

    httpx.Client.send = send
    httpx.AsyncClient.send = asend

    # ``send`` only sees the first URL; the redirect loop issues each hop
    # through ``_send_single_request``, so check there as well.
    if hasattr(httpx.Client, "_send_single_request"):
        original_single = httpx.Client._send_single_request

        def send_single(self: Any, request: Any) -> Any:
            _check_url(request.url)
            return original_single(self, request)

        httpx.Client._send_single_request = send_single
    if hasattr(httpx.AsyncClient, "_send_single_request"):
        original_asingle = httpx.AsyncClient._send_single_request

        async def asend_single(self: Any, request: Any) -> Any:
            _check_url(request.url)
            return await original_asingle(self, request)

        httpx.AsyncClient._send_single_request = asend_single


def install_mock_egress_guard() -> bool:
    """Install the guard on ``httpx`` (and vendored ``httpx2``) and ``requests``. Idempotent."""
    global _installed
    with _install_lock:
        if _installed:
            return False
        # Only patch clients that are importable; nothing optional is forced in.
        # ``httpx2`` is the vendored copy some provider SDKs (anthropic) ship.
        patched_http = False
        for module_name in ("httpx", "httpx2"):
            try:
                module = importlib.import_module(module_name)
                _wrap_httpx(module)
                patched_http = True
            except (ImportError, AttributeError):
                continue
        if not patched_http:  # pragma: no cover - httpx is a core dependency
            return False

        try:
            if importlib.util.find_spec("requests") is None:
                raise ImportError("requests not installed")
            import requests

            # ``Session.send`` is re-entered for every redirect hop, so each
            # hop is checked.
            original_rsend = requests.Session.send

            def rsend(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
                _check_url(request.url)
                return original_rsend(self, request, *args, **kwargs)

            requests.Session.send = rsend  # type: ignore[method-assign]
        except ImportError:
            pass

        _installed = True
        logger.debug("Installed mock-mode egress guard")
        return True
