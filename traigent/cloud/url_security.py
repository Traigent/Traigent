"""URL validation helpers for cloud-facing SDK HTTP calls."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from urllib.parse import unquote, urlparse, urlunparse

_DEVELOPMENT_ENV_NAMES = {"dev", "development", "local", "test", "testing"}
_PRODUCTION_ENV_NAMES = {"prod", "production", "stage", "staging"}
_ENV_KEYS = ("TRAIGENT_ENV", "ENVIRONMENT", "APP_ENV", "FLASK_ENV")


def _is_development_environment() -> bool:
    """Return True for explicit/local SDK development environments."""
    for key in _ENV_KEYS:
        value = os.getenv(key)
        if value and value.strip().lower() in _PRODUCTION_ENV_NAMES:
            return False

    for key in _ENV_KEYS:
        value = os.getenv(key)
        if value and value.strip().lower() in _DEVELOPMENT_ENV_NAMES:
            return True
    return False


def _reject_unsafe_hostname(hostname: str, *, allow_local: bool) -> None:
    normalized = hostname.strip("[]").rstrip(".").lower()
    if not normalized:
        raise ValueError("Cloud base URL must include a hostname") from None

    if normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(
        (".localhost", ".local")
    ):
        if allow_local:
            return
        raise ValueError("Cloud base URL host is not allowed in production") from None

    try:
        host_ip = ipaddress.ip_address(normalized)
    except ValueError:
        host_ip = None

    if host_ip is not None:
        if not allow_local and not host_ip.is_global:
            raise ValueError(
                "Cloud base URL must not target private or loopback IPs"
            ) from None
        return

    if allow_local:
        return

    try:
        addr_infos = socket.getaddrinfo(normalized, None)
    except socket.gaierror:
        raise ValueError("Cloud base URL host could not be resolved") from None

    for _family, _socktype, _proto, _canonname, sockaddr in addr_infos:
        try:
            resolved_ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if not resolved_ip.is_global:
            raise ValueError(
                "Cloud base URL must not resolve to private or loopback IPs"
            ) from None


def validate_cloud_base_url(base_url: str, *, purpose: str = "cloud request") -> str:
    """Validate and normalize a cloud base URL before sending credentials.

    Production calls must use a public HTTP(S) host. Localhost and private
    networks are allowed only in explicit development/test environments.
    """
    candidate = (base_url or "").strip().rstrip("/")
    if not candidate:
        raise ValueError(f"{purpose} base URL cannot be empty") from None

    if any(ord(ch) < 32 or ch.isspace() for ch in candidate):
        raise ValueError(f"{purpose} base URL contains unsafe whitespace") from None

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{purpose} base URL must be http(s) with a host") from None
    if parsed.username or parsed.password:
        raise ValueError(f"{purpose} base URL must not include credentials") from None
    if parsed.params or parsed.query or parsed.fragment:
        raise ValueError(
            f"{purpose} base URL must not include params, query, or fragment"
        ) from None

    allow_local = _is_development_environment()
    if not allow_local and parsed.scheme != "https":
        raise ValueError(f"{purpose} base URL must use https in production") from None

    decoded_path = parsed.path
    for _ in range(2):
        next_path = unquote(decoded_path)
        if next_path == decoded_path:
            break
        decoded_path = next_path

    path_segments = [part for part in decoded_path.split("/") if part]
    if any(part in {".", ".."} for part in path_segments):
        raise ValueError(
            f"{purpose} base URL must not contain path traversal"
        ) from None

    hostname = parsed.hostname
    if hostname is None:
        raise ValueError(f"{purpose} base URL must include a hostname") from None
    _reject_unsafe_hostname(hostname, allow_local=allow_local)

    clean_path = parsed.path.rstrip("/")
    return urlunparse(
        parsed._replace(path=clean_path, params="", query="", fragment="")
    )


async def validate_cloud_base_url_async(
    base_url: str, *, purpose: str = "cloud request"
) -> str:
    """Validate a cloud URL without blocking the event loop on DNS resolution."""
    return await asyncio.to_thread(
        validate_cloud_base_url,
        base_url,
        purpose=purpose,
    )
