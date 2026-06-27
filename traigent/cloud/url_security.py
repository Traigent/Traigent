"""URL validation helpers for cloud-facing SDK HTTP calls."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from urllib.parse import unquote, urlparse, urlunparse

_DEVELOPMENT_ENV_NAMES = {"dev", "development", "local", "test", "testing"}
_PRODUCTION_ENV_NAMES = {"prod", "production", "stage", "staging"}
_ENV_KEYS = (
    "TRAIGENT_ENV",
    "ENVIRONMENT",
    "TRAIGENT_ENVIRONMENT",
    "APP_ENV",
    "FLASK_ENV",
)
_METADATA_HOSTNAMES = {
    "metadata",
    "metadata.google.internal",
}


def _is_development_environment() -> bool:
    """Return True only for explicit/local SDK development environments.

    Fail closed: an unset / unrecognized environment is treated as production
    (strict) so that a deployment which never set an env marker does not
    silently allow credential egress to localhost / private / metadata hosts.
    """
    for key in _ENV_KEYS:
        value = os.getenv(key)
        if value and value.strip().lower() in _PRODUCTION_ENV_NAMES:
            return False

    for key in _ENV_KEYS:
        value = os.getenv(key)
        if value and value.strip().lower() in _DEVELOPMENT_ENV_NAMES:
            return True
    return False


def _normalize_hostname(hostname: str) -> str:
    return hostname.strip("[]").rstrip(".").lower()


def _is_numeric_hostname_label(label: str) -> bool:
    lowered = label.lower()
    if lowered.startswith("0x"):
        return len(lowered) > 2 and all(ch in "0123456789abcdef" for ch in lowered[2:])
    return lowered.isdigit()


def _is_nonstandard_ip_notation(hostname: str) -> bool:
    labels = hostname.split(".")
    return bool(labels) and all(_is_numeric_hostname_label(label) for label in labels)


def _parse_ip_literal(
    hostname: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return None


def _reject_always_blocked_hostname(hostname: str, *, purpose: str) -> str:
    normalized = _normalize_hostname(hostname)
    if not normalized:
        raise ValueError("Cloud base URL must include a hostname") from None

    if normalized in _METADATA_HOSTNAMES:
        raise ValueError(f"{purpose} base URL points at a metadata service") from None

    if _parse_ip_literal(normalized) is None and _is_nonstandard_ip_notation(
        normalized
    ):
        raise ValueError(
            f"{purpose} base URL uses non-standard IP address notation"
        ) from None

    return normalized


def _reject_unsafe_hostname(hostname: str, *, allow_local: bool) -> None:
    normalized = _reject_always_blocked_hostname(hostname, purpose="Cloud")

    if normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(
        (".localhost", ".local")
    ):
        if allow_local:
            return
        raise ValueError("Cloud base URL host is not allowed in production") from None

    host_ip = _parse_ip_literal(normalized)

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

    hostname = parsed.hostname
    if hostname is None:
        raise ValueError(f"{purpose} base URL must include a hostname") from None
    _reject_always_blocked_hostname(hostname, purpose=purpose)

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

    _reject_unsafe_hostname(hostname, allow_local=allow_local)

    clean_path = parsed.path.rstrip("/")
    return urlunparse(
        parsed._replace(path=clean_path, params="", query="", fragment="")
    )


def _validation_needs_dns_resolution(base_url: str) -> bool:
    """Return True when URL validation may perform blocking DNS lookup."""
    candidate = (base_url or "").strip().rstrip("/")
    parsed = urlparse(candidate)
    hostname = parsed.hostname
    if hostname is None:
        return False

    normalized = _normalize_hostname(hostname)
    if normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(
        (".localhost", ".local")
    ):
        return False

    if normalized in _METADATA_HOSTNAMES:
        return False
    if _parse_ip_literal(normalized) is not None:
        return False
    return not _is_nonstandard_ip_notation(normalized)


async def validate_cloud_base_url_async(
    base_url: str, *, purpose: str = "cloud request"
) -> str:
    """Validate a cloud URL without blocking the event loop on DNS resolution."""
    if not _validation_needs_dns_resolution(base_url):
        return validate_cloud_base_url(base_url, purpose=purpose)

    return await asyncio.to_thread(
        validate_cloud_base_url,
        base_url,
        purpose=purpose,
    )
