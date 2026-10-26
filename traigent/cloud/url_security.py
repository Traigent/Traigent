"""URL validation helpers for cloud-facing SDK HTTP calls."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import urllib.request
from typing import Any
from urllib.parse import ParseResult, unquote, urlparse, urlunparse

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


def _production_assumed_hint() -> str:
    """Explain why plain-http/local hosts were refused and how to change it.

    The gate fails closed when no environment is declared, so the message must
    say production was *assumed* and name the variable that changes it (#2503).
    """
    from traigent.utils.env_config import _ENVIRONMENT_KEYS, _normalize_str

    declared = [
        f"{key}={value}"
        for key in _ENVIRONMENT_KEYS
        if (value := _normalize_str(os.getenv(key))) is not None
    ]
    keys = " / ".join(_ENVIRONMENT_KEYS)
    if declared:
        return (
            f"the declared environment ({', '.join(declared)}) is not a "
            "development environment (staging counts as production here). Use "
            "https, or for a local backend set every declared key to a "
            "development value such as ENVIRONMENT=development"
        )
    return (
        f"no {keys} is set, so production was assumed. Use https, or set "
        "ENVIRONMENT=development for a local backend"
    )


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


_PRIVATE_BACKEND_OPT_IN_ENV = "TRAIGENT_ALLOW_PRIVATE_BACKEND"
_PRIVATE_BACKEND_OPT_IN_VALUES = frozenset({"1", "true"})
_PRIVATE_BACKEND_HINT = (
    f"set {_PRIVATE_BACKEND_OPT_IN_ENV}=1 if this is your self-hosted Traigent backend"
)
_DEFAULT_PORTS = {"http": 80, "https": 443}


def is_plain_http_origin_parts(parsed: ParseResult) -> bool:
    """Structural check: http(s) scheme, a host, no userinfo/params/query/fragment.

    Pure shape validation (no host policy, no DNS); callers layer their own
    host policy on top.
    """
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    return not (
        parsed.username
        or parsed.password
        or parsed.params
        or parsed.query
        or parsed.fragment
    )


def _has_invalid_port(parsed: ParseResult) -> bool:
    """Return True when an explicit port is out of range or is ``0``.

    ``parsed.port`` accepts ``0`` (and raises for values above 65535). An explicit
    port ``0`` is never a valid destination, so it is treated as invalid rather
    than as "no port".
    """
    try:
        port = parsed.port
    except ValueError:
        return True
    return port == 0


def _origin_key(url: str) -> tuple[str, str, int] | None:
    """Return ``(scheme, host, port)`` for a URL, or None when it cannot be parsed.

    Hostnames are normalized (lower-cased, IPv6 brackets stripped) and a missing
    port (``None``) is replaced by the scheme default, so ``https://host`` and
    ``https://host:443`` compare equal while ``https://host:0`` never does.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _DEFAULT_PORTS or not parsed.hostname:
        return None
    if _has_invalid_port(parsed):
        return None
    port = parsed.port
    return (
        scheme,
        _normalize_hostname(parsed.hostname),
        _DEFAULT_PORTS[scheme] if port is None else port,
    )


_GATE_OFF_OPT_IN_MESSAGE = (
    f"{_PRIVATE_BACKEND_OPT_IN_ENV} requires the transport gate; "
    "unset TRAIGENT_TRANSPORT_GATE=off"
)


def _opt_in_origin_matches(candidate: str) -> bool:
    """True when the opt-in env is on and ``candidate`` has the configured origin.

    Independent of the transport-gate mode; see :func:`_private_backend_opt_in`.
    """
    raw = os.getenv(_PRIVATE_BACKEND_OPT_IN_ENV, "")
    if raw.lower() not in _PRIVATE_BACKEND_OPT_IN_VALUES:
        return False
    # Lazy import: backend_config is a foundational module and may import cloud code.
    from traigent.config.backend_config import BackendConfig

    configured = BackendConfig._normalize_origin(os.environ.get("TRAIGENT_BACKEND_URL"))
    if not configured:
        return False
    if _has_invalid_port(urlparse(configured)):
        raise ValueError(
            "TRAIGENT_BACKEND_URL has an invalid port (must be 1-65535); "
            f"{_PRIVATE_BACKEND_OPT_IN_ENV} cannot be honoured"
        ) from None
    configured_key = _origin_key(configured)
    return configured_key is not None and configured_key == _origin_key(candidate)


def _transport_gate_is_off() -> bool:
    # Lazy import: transport_gate imports this module's public opt-in helper.
    from traigent.labels.transport_gate import current_mode

    return bool(current_mode() == "off")


def _private_backend_opt_in(candidate: str) -> bool:
    """Return True when the explicit self-hosted opt-in covers ``candidate``.

    ``TRAIGENT_ALLOW_PRIVATE_BACKEND`` must be exactly ``1`` or ``true`` (case-
    insensitive); any other value is off. The allowance is pinned to the origin
    (scheme + host + port) of ``TRAIGENT_BACKEND_URL``. It does not fall back to
    ``TRAIGENT_API_URL`` or stored credentials, so an unset backend URL means the
    opt-in does nothing. It never relaxes metadata/link-local blocking, the https
    requirement, redirect handling or DNS-failure handling.

    It is also refused while ``TRAIGENT_TRANSPORT_GATE=off``: the redirect refusal
    lives in the transport gate, so an opt-in without the gate would send
    credentials to the private origin with redirects followed.
    """
    if not _opt_in_origin_matches(candidate):
        return False
    return not _transport_gate_is_off()


def _opt_in_blocked_by_gate_off(candidate: str) -> bool:
    return _opt_in_origin_matches(candidate) and _transport_gate_is_off()


class _RefuseRedirectHandler(urllib.request.HTTPRedirectHandler):
    """urllib redirect handler that never follows a 3xx.

    Returning None makes urllib raise ``HTTPError`` with the 3xx status; the
    redirect target is never requested and the credential headers are not replayed.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


def build_no_redirect_opener() -> urllib.request.OpenerDirector:
    """An opener for credential-carrying urllib requests that refuses every redirect."""
    return urllib.request.build_opener(_RefuseRedirectHandler)


def open_no_redirect(req: urllib.request.Request, *, timeout: float) -> Any:
    """``urlopen`` for credential-carrying requests: a 3xx raises ``HTTPError``.

    urllib follows redirects by default and replays the credential headers to the
    target. The redirect target is never requested here.
    """
    return build_no_redirect_opener().open(req, timeout=timeout)


def private_backend_opt_in_covers(url: str) -> bool:
    """Public form of the opt-in check, used by the transport gate.

    True when ``TRAIGENT_ALLOW_PRIVATE_BACKEND`` is on and ``url`` has exactly the
    origin of ``TRAIGENT_BACKEND_URL``. Callers that send credentials to that origin
    refuse redirects while this is True.
    """
    return _private_backend_opt_in(url)


def _reject_unsafe_hostname(
    hostname: str,
    *,
    allow_local: bool,
    allow_private: bool | None = None,
) -> None:
    """Reject unsafe hosts.

    ``allow_local`` is the development allowance (private/loopback hosts and
    unresolvable names). ``allow_private`` controls only private/loopback
    destinations, so the self-hosted opt-in can admit them without also admitting
    unresolvable names. It defaults to ``allow_local``.
    """
    if allow_private is None:
        allow_private = allow_local
    normalized = _reject_always_blocked_hostname(hostname, purpose="Cloud")

    if _is_local_name(normalized):
        if not allow_private:
            raise ValueError(
                "Cloud base URL host is not allowed in production: "
                f"{_production_assumed_hint()}"
            ) from None
        # Development only: ``localhost``/``.localhost`` are reserved for loopback
        # and skip the lookup. Under the self-hosted opt-in the name is resolved and
        # every answer is checked, so a loopback name cannot stand in for an
        # unchecked destination. ``.local`` (mDNS) always falls through to the
        # resolved-IP guard below.
        if allow_local and _is_loopback_name(normalized):
            return

    host_ip = _parse_ip_literal(normalized)

    if host_ip is not None:
        if not allow_private and not host_ip.is_global:
            raise ValueError(
                "Cloud base URL must not target private or loopback IPs; "
                f"{_PRIVATE_BACKEND_HINT}"
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
        if not allow_private and not resolved_ip.is_global:
            raise ValueError(
                "Cloud base URL must not resolve to private or loopback IPs; "
                f"{_PRIVATE_BACKEND_HINT}"
            ) from None


_CLOUD_REQUEST = "cloud request"


def validate_cloud_base_url(base_url: str, *, purpose: str = _CLOUD_REQUEST) -> str:
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
    if _has_invalid_port(parsed):
        raise ValueError(f"{purpose} base URL has an invalid port") from None
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
        raise ValueError(
            f"{purpose} base URL must use https in production: "
            f"{_production_assumed_hint()}"
        ) from None

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

    allow_private = allow_local or _private_backend_opt_in(candidate)
    try:
        _reject_unsafe_hostname(
            hostname, allow_local=allow_local, allow_private=allow_private
        )
    except CloudUrlUnreachableError:
        raise
    except ValueError:
        if not allow_private and _opt_in_blocked_by_gate_off(candidate):
            raise ValueError(_GATE_OFF_OPT_IN_MESSAGE) from None
        raise

    clean_path = parsed.path.rstrip("/")
    return urlunparse(
        parsed._replace(path=clean_path, params="", query="", fragment="")
    )


def validate_always_blocked_cloud_url(
    base_url: str, *, purpose: str = _CLOUD_REQUEST, resolve: bool = True
) -> None:
    """Refuse metadata/link-local destinations and, outside development, plaintext HTTP.

    For callers that keep their own private-range policy and so cannot use
    :func:`validate_cloud_base_url`. The checks are: the scheme is http(s) with a
    host, plain HTTP is refused outside development, the metadata hostnames and
    IP literals are refused, and every resolved answer is checked against the
    always-blocked set. Private and loopback answers are allowed. A name that does
    not resolve is refused (:class:`CloudUrlUnreachableError`) outside development,
    and tolerated only in development. The shared validator's stricter rules do not
    apply here.

    ``resolve=False`` skips the DNS step (no network I/O): construction-time callers
    use it so importing the SDK never resolves a host. Egress callers keep the default
    and resolve immediately before the credential-bearing request.
    """
    parsed = urlparse((base_url or "").strip().rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{purpose} base URL must be http(s) with a host") from None
    if parsed.scheme != "https" and not _is_development_environment():
        raise ValueError(
            f"{purpose} base URL must use https in production: "
            f"{_production_assumed_hint()}"
        ) from None

    normalized = _normalize_hostname(parsed.hostname)
    if normalized in _METADATA_HOSTNAMES:
        raise ValueError(f"{purpose} base URL points at a metadata service") from None
    host_ip = _parse_ip_literal(normalized)
    if host_ip is not None:
        _reject_always_blocked_ip(host_ip, purpose=purpose)
        return
    if not resolve:
        return
    try:
        addr_infos = socket.getaddrinfo(normalized, None)
    except socket.gaierror:
        # Fail closed in production: an unresolvable name cannot be checked against
        # the always-blocked set, and "not blocked" must not be the default. The
        # development allowance mirrors _reject_unsafe_hostname's allow_local branch.
        if _is_development_environment():
            return
        raise CloudUrlUnreachableError(
            f"{purpose} base URL host could not be resolved"
        ) from None
    for _family, _socktype, _proto, _canonname, sockaddr in addr_infos:
        resolved_ip = _parse_resolved_ip(sockaddr[0])
        if resolved_ip is not None:
            _reject_always_blocked_ip(resolved_ip, purpose=purpose)


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
    if _is_loopback_name(normalized) and _is_development_environment():
        return False

    if normalized in _METADATA_HOSTNAMES:
        return False
    if _parse_ip_literal(normalized) is not None:
        return False
    return not _is_nonstandard_ip_notation(normalized)


async def validate_cloud_base_url_async(
    base_url: str, *, purpose: str = _CLOUD_REQUEST
) -> str:
    """Validate a cloud URL without blocking the event loop on DNS resolution."""
    if not _validation_needs_dns_resolution(base_url):
        return validate_cloud_base_url(base_url, purpose=purpose)

    return await asyncio.to_thread(
        validate_cloud_base_url,
        base_url,
        purpose=purpose,
    )
