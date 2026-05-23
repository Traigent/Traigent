"""Outbound URL validation helpers."""

from __future__ import annotations

import ipaddress
from urllib.parse import ParseResult, unquote, urlparse, urlunparse


class UnsafeUrlError(ValueError):
    """Raised when a configured outbound URL is unsafe."""


_LOCALHOST_NAMES = {"localhost", "localhost.localdomain"}
_METADATA_HOSTNAMES = {
    "metadata",
    "metadata.google.internal",
    "metadata.google.internal.",
}


def _is_numeric_hostname_label(label: str) -> bool:
    lowered = label.lower()
    if lowered.startswith("0x"):
        return len(lowered) > 2 and all(ch in "0123456789abcdef" for ch in lowered[2:])
    return lowered.isdigit()


def _reject_nonstandard_ip_notation(hostname: str, purpose: str) -> None:
    labels = hostname.split(".")
    if labels and all(_is_numeric_hostname_label(label) for label in labels):
        raise UnsafeUrlError(f"{purpose} uses non-standard IP address notation")


def _parse_ip_literal(
    hostname: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return None


def _validate_parsed_url(parsed: ParseResult, purpose: str) -> None:
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError(f"{purpose} must use http or https")
    if not parsed.netloc or not parsed.hostname:
        raise UnsafeUrlError(f"{purpose} must include a hostname")
    if parsed.username or parsed.password:
        raise UnsafeUrlError(f"{purpose} must not include embedded credentials")
    if parsed.query or parsed.fragment:
        raise UnsafeUrlError(f"{purpose} must not include query or fragment data")

    try:
        _ = parsed.port
    except ValueError as exc:
        raise UnsafeUrlError(f"{purpose} has an invalid port") from exc


def _validate_hostname(
    hostname: str,
    *,
    purpose: str,
    allow_private_hosts: bool,
) -> None:
    normalized_hostname = hostname.rstrip(".").lower()
    if normalized_hostname in _METADATA_HOSTNAMES:
        raise UnsafeUrlError(f"{purpose} points at a metadata service")

    ip = _parse_ip_literal(normalized_hostname)
    if ip is not None:
        _validate_ip_address(
            ip, purpose=purpose, allow_private_hosts=allow_private_hosts
        )
        return

    _reject_nonstandard_ip_notation(normalized_hostname, purpose)
    if not allow_private_hosts and normalized_hostname in _LOCALHOST_NAMES:
        raise UnsafeUrlError(f"{purpose} points at localhost")


def _normalize_path(path: str, purpose: str) -> str:
    decoded_path = unquote(path)
    if any(segment in {".", ".."} for segment in decoded_path.split("/")):
        raise UnsafeUrlError(f"{purpose} must not include dot-segment paths")
    return path.rstrip("/")


def _validate_ip_address(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    purpose: str,
    allow_private_hosts: bool,
) -> None:
    if ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
        raise UnsafeUrlError(f"{purpose} points at a non-routable address")
    if not allow_private_hosts and (ip.is_private or ip.is_loopback):
        raise UnsafeUrlError(f"{purpose} points at a private address")


def validate_outbound_url(
    url: str,
    *,
    purpose: str = "outbound URL",
    allow_private_hosts: bool = False,
) -> str:
    """Validate and normalize an outbound HTTP(S) base URL.

    The helper blocks common SSRF footguns: unsupported schemes, embedded
    credentials, query/fragment smuggling, localhost/private IPs unless
    explicitly allowed, and link-local metadata endpoints.
    """
    candidate = (url or "").strip()
    if not candidate:
        raise UnsafeUrlError(f"{purpose} must not be empty")

    parsed = urlparse(candidate)
    _validate_parsed_url(parsed, purpose)
    _validate_hostname(
        parsed.hostname or "",
        purpose=purpose,
        allow_private_hosts=allow_private_hosts,
    )

    normalized_path = _normalize_path(parsed.path, purpose)
    return urlunparse((parsed.scheme, parsed.netloc, normalized_path, "", "", ""))
