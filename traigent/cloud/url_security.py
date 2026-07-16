"""URL validation helpers for cloud-facing SDK HTTP calls."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from urllib.parse import unquote, urlparse, urlunparse

_METADATA_HOSTNAMES = {
    "metadata",
    "metadata.google.internal",
}
# Cloud instance-metadata service (IMDS) IP literals that must never receive a
# credential-bearing request, kept in parity with
# ``traigent/hybrid/transport.py::_METADATA_SERVICE_IPS``. These are matched by
# resolved IP value (not hostname string) and blocked ALWAYS — including in
# development — alongside the generic link-local range. 169.254.169.254 is also
# link-local, but 100.100.100.200 (Alibaba, CGNAT range) and fd00:ec2::254
# (AWS IMDSv2 IPv6, unique-local) are not, so they need explicit listing.
_METADATA_SERVICE_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),
        ipaddress.ip_address("100.100.100.200"),
        ipaddress.ip_address("fd00:ec2::254"),
    }
)

# IPv6 prefixes that carry an IPv4 address in their low 32 bits, beyond the
# IPv4-mapped range that ``IPv6Address.ipv4_mapped`` already reports. See
# ``_unwrap_embedded_ipv4``.
_IPV4_COMPATIBLE_PREFIX = ipaddress.ip_network("::/96")
_NAT64_WELL_KNOWN_PREFIX = ipaddress.ip_network("64:ff9b::/96")


class CloudUrlUnreachableError(ValueError):
    """A cloud base URL was structurally valid but its host could not be resolved.

    Distinct from the other ``ValueError``s raised by :func:`validate_cloud_base_url`,
    which signal an *unsafe* URL (private/loopback/metadata IP, bad scheme, embedded
    credentials, path traversal) and MUST always fail loud. This subclass marks the
    narrow "backend is simply unreachable" case, so best-effort callers (e.g. the
    interaction-policy read) may fall back to a static default instead of crashing,
    WITHOUT relaxing any SSRF/unsafe-origin protection. Subclasses ``ValueError`` so
    existing ``except ValueError`` callers keep catching it.
    """


# Deployment tiers that ``env_config`` classifies as non-production but that
# this gate must still treat as strict. Staging runs on real cloud hosts with
# real Traigent API keys, so it must not inherit the local-development
# allowance for plaintext HTTP or private/loopback destinations — the
# classification that is right for feature/billing gates is wrong for
# credential egress. Kept as an explicit subtraction from the canonical value
# set (below) rather than a private value universe, so a new canonical env name
# lands here automatically and can only ever be *more* strict than
# ``treat_as_production_policy``.
_STRICT_EGRESS_ENV_NAMES = frozenset({"stage", "staging"})


def _is_development_environment() -> bool:
    """Return True only for explicit/local SDK development environments.

    Reads the canonical env-var keys and value set from
    :mod:`traigent.utils.env_config` — the same signal every other SDK security
    gate uses (auth, credential-fallback, mock bypass). Before issue #1905 this
    helper had a private, divergent key set (adding ``APP_ENV``/``FLASK_ENV``),
    so a deployment could be 'dev' here and 'prod' to every other gate.

    Two deliberate ways this gate is STRICTER than
    :func:`~traigent.utils.env_config.treat_as_production_policy`, because it
    guards credential egress rather than a feature flag:

    * ``stage``/``staging`` are strict (see ``_STRICT_EGRESS_ENV_NAMES``).
    * A dev marker on one canonical key cannot override a strict marker on
      another. ``resolve_environment_name`` returns the first key that is set,
      so ``ENVIRONMENT=dev TRAIGENT_ENV=production`` would resolve to ``dev``
      purely from key order; here *every* declared key must be a development
      name, so any strict signal wins regardless of order.

    Fail closed: an unset or unrecognized environment is treated as production
    (strict), so a deployment that never set an env marker — or set a typo —
    does not silently allow credential egress to localhost / private / metadata
    hosts.
    """
    # Imported lazily so this module stays import-cycle-free relative to
    # ``env_config`` (which is a foundational module).
    from traigent.utils.env_config import (
        _ENVIRONMENT_KEYS,
        _NON_PRODUCTION_ENV_NAMES,
        _normalize_str,
    )

    development_env_names = _NON_PRODUCTION_ENV_NAMES - _STRICT_EGRESS_ENV_NAMES

    declared = [
        value.lower()
        for value in (_normalize_str(os.getenv(key)) for key in _ENVIRONMENT_KEYS)
        if value is not None
    ]
    if not declared:
        return False
    return all(name in development_env_names for name in declared)


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


def _unwrap_embedded_ipv4(
    host_ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Return the IPv4 address an IPv6 address actually reaches, if any.

    ``::ffff:169.254.169.254`` connects to 169.254.169.254 on any dual-stack
    host (Linux default ``bindv6only=0``), but the guards downstream do not see
    through the mapping on their own. ``_METADATA_SERVICE_IPS`` matches by exact
    address value, so no mapped form ever equals an entry in it, on any
    interpreter. The link-local check cannot cover the gap either:
    ``::ffff:100.100.100.200`` (Alibaba IMDS) is never link-local on any
    interpreter, because 100.64.0.0/10 is CGNAT. Python releases further differ
    in whether an IPv4-mapped address is classified by its embedded IPv4 or by
    the IPv6 value, so ``is_global``/``is_link_local`` answers are not stable
    across builds. Normalizing here means every downstream classification sees
    the address the request will really hit, independent of interpreter version.

    The IPv4-mapped range is the one that is verified routable; the IPv4-
    compatible (RFC 4291, deprecated), 6to4, and NAT64 well-known prefixes also
    embed an IPv4 address in their low 32 bits and are unwrapped as defense in
    depth — they reach IPv4 only through a relay/gateway, but a credential-
    bearing request has no legitimate reason to name IMDS in any of them.

    Addresses that embed nothing (``::1``, ``fd00::1``, ordinary global IPv6)
    are returned unchanged.
    """
    if not isinstance(host_ip, ipaddress.IPv6Address):
        return host_ip
    if host_ip.ipv4_mapped is not None:
        return host_ip.ipv4_mapped
    if host_ip.sixtofour is not None:
        return host_ip.sixtofour
    if host_ip in _NAT64_WELL_KNOWN_PREFIX:
        return ipaddress.IPv4Address(int(host_ip) & 0xFFFFFFFF)
    # ``::`` (unspecified) and ``::1`` (loopback) live in ::/96 but embed no
    # IPv4 address; unwrapping them would turn ::1 into 0.0.0.1.
    if host_ip in _IPV4_COMPATIBLE_PREFIX and int(host_ip) > 1:
        return ipaddress.IPv4Address(int(host_ip))
    return host_ip


def _parse_ip_literal(
    hostname: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        host_ip = ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return None
    # Normalized at the single parse point every caller shares, so the IMDS set,
    # the link-local range check, and the ``is_global`` production gate all
    # classify the address that is actually reached — for URL literals and for
    # resolved answers alike (``_parse_resolved_ip`` delegates here).
    return _unwrap_embedded_ipv4(host_ip)


def _reject_always_blocked_ip(
    host_ip: ipaddress.IPv4Address | ipaddress.IPv6Address, *, purpose: str
) -> None:
    """Reject IMDS / link-local destinations in EVERY environment.

    Applied both to IP literals in the URL and to the addresses a hostname
    resolves to, because a credential-bearing request must never reach a cloud
    metadata endpoint or the link-local range — development included.

    Callers must pass an address from :func:`_parse_ip_literal` /
    :func:`_parse_resolved_ip` rather than ``ipaddress.ip_address`` directly:
    both classifications below are IPv4-vs-IPv6 exact, so an IPv6 form
    embedding an IPv4 IMDS address evades them unless it has been through
    :func:`_unwrap_embedded_ipv4`.
    """
    if host_ip in _METADATA_SERVICE_IPS:
        raise ValueError(f"{purpose} base URL points at a metadata service") from None
    if host_ip.is_link_local:
        raise ValueError(f"{purpose} base URL points at a link-local address") from None


def _reject_always_blocked_hostname(hostname: str, *, purpose: str) -> str:
    normalized = _normalize_hostname(hostname)
    if not normalized:
        raise ValueError("Cloud base URL must include a hostname") from None

    if normalized in _METADATA_HOSTNAMES:
        raise ValueError(f"{purpose} base URL points at a metadata service") from None

    host_ip = _parse_ip_literal(normalized)
    if host_ip is not None:
        # Matches by IP value, not by hostname string, so IP literals such as
        # http://169.254.169.254/ are rejected even though they are absent from
        # ``_METADATA_HOSTNAMES`` (issue #1908). Mirrors ``hybrid/transport.py``'s
        # unconditional guard.
        _reject_always_blocked_ip(host_ip, purpose=purpose)
    elif _is_nonstandard_ip_notation(normalized):
        raise ValueError(
            f"{purpose} base URL uses non-standard IP address notation"
        ) from None

    return normalized


def _is_loopback_name(normalized: str) -> bool:
    """Return True for names RFC 6761 reserves for the loopback interface."""
    return normalized in {"localhost", "localhost.localdomain"} or normalized.endswith(
        ".localhost"
    )


def _is_local_name(normalized: str) -> bool:
    return _is_loopback_name(normalized) or normalized.endswith(".local")


def _parse_resolved_ip(
    sockaddr_host: str | int,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    # ``getaddrinfo``'s sockaddr is a union whose first element is typed
    # ``str | int``; a non-str answer is not an address this guard can classify,
    # so skip it rather than crash.
    if not isinstance(sockaddr_host, str):
        return None
    # IPv6 results carry a zone id (``fe80::1%eth0``) that ``ip_address``
    # rejects; strip it so scoped link-local answers are still checked.
    return _parse_ip_literal(sockaddr_host.split("%", 1)[0])


def _reject_unsafe_hostname(hostname: str, *, allow_local: bool) -> None:
    normalized = _reject_always_blocked_hostname(hostname, purpose="Cloud")

    if _is_local_name(normalized):
        if not allow_local:
            raise ValueError(
                "Cloud base URL host is not allowed in production"
            ) from None
        # ``localhost``/``.localhost`` are reserved for loopback and cannot
        # reach a metadata endpoint. ``.local`` (mDNS) can, so it falls through
        # to the resolved-IP guard below.
        if _is_loopback_name(normalized):
            return

    host_ip = _parse_ip_literal(normalized)

    if host_ip is not None:
        if not allow_local and not host_ip.is_global:
            raise ValueError(
                "Cloud base URL must not target private or loopback IPs"
            ) from None
        return

    try:
        addr_infos = socket.getaddrinfo(normalized, None)
    except socket.gaierror:
        if allow_local:
            # Development only: let offline/local-only workflows validate rather
            # than hard-failing on a name that does not resolve right now.
            # KNOWN RESIDUE: resolution failing HERE does not mean the request's
            # own later lookup fails — a hostile resolver can answer SERVFAIL now
            # and a metadata address at connect time. This is the same
            # validate-then-request window documented on
            # ``validate_cloud_base_url``; closing it needs connect-time peer-IP
            # enforcement, not a stricter check here.
            return
        raise CloudUrlUnreachableError(
            "Cloud base URL host could not be resolved"
        ) from None

    for _family, _socktype, _proto, _canonname, sockaddr in addr_infos:
        resolved_ip = _parse_resolved_ip(sockaddr[0])
        if resolved_ip is None:
            continue
        # Checked in EVERY environment: the development allowance covers
        # legitimate local/private endpoints, never a hostname that resolves to
        # a metadata service or the link-local range (issue #1908).
        _reject_always_blocked_ip(resolved_ip, purpose="Cloud")
        if not allow_local and not resolved_ip.is_global:
            raise ValueError(
                "Cloud base URL must not resolve to private or loopback IPs"
            ) from None


def validate_cloud_base_url(base_url: str, *, purpose: str = "cloud request") -> str:
    """Validate and normalize a cloud base URL before sending credentials.

    Production calls must use a public HTTP(S) host. Localhost and private
    networks are allowed only in explicit development/test environments;
    ``staging`` is strict here even though other policy gates treat it as
    non-production (see :func:`_is_development_environment`).

    Not a hard guarantee against a hostile resolver. This validates a URL
    *string*: any hostname is resolved here, but the returned string is what
    callers hand to their HTTP client, which resolves the name again at connect
    time. Nothing pins the address that was checked, so an attacker who controls
    DNS answers/TTLs can still steer the request elsewhere between the two
    lookups (classic DNS rebinding). Closing that window requires connect-time
    peer-IP enforcement in the transport, which no SDK HTTP path implements
    today. What this function does guarantee is that
    metadata/link-local destinations are rejected in EVERY environment when they
    are visible at validation time — as an IP literal or as a resolved answer,
    including IPv6 forms that embed the IPv4 address they reach
    (``[::ffff:169.254.169.254]``), which are normalized before classification
    (see :func:`_unwrap_embedded_ipv4`).
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

    # Decode to a fixed point (bounded) so multiply-encoded traversal
    # (e.g. %25252e) cannot survive a fixed two-pass decode. Mirrors the
    # explicit-api_base_url guard in backend_client.py.
    decoded_path = parsed.path
    for _ in range(8):
        next_path = unquote(decoded_path)
        if next_path == decoded_path:
            break
        decoded_path = next_path
    else:
        # Still changing after 8 decodes — pathologically encoded; reject.
        raise ValueError(
            f"{purpose} base URL must not contain path traversal"
        ) from None

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
    # ``.local`` is deliberately absent: it is resolved (and its answers guarded)
    # in development, so it must be offloaded like any other resolvable name.
    if _is_loopback_name(normalized):
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
