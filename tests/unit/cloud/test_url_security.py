"""Tests for cloud URL safety validation."""

from __future__ import annotations

import ipaddress
import socket
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.url_security import (
    _unwrap_embedded_ipv4,
    _validation_needs_dns_resolution,
    validate_cloud_base_url,
    validate_cloud_base_url_async,
)


def _addr_info(ip_address: str) -> list[tuple[int, int, int, str, tuple[str, int]]]:
    return [
        (
            socket.AF_INET,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            (ip_address, 443),
        )
    ]


def test_validate_cloud_base_url_normalizes_safe_public_url() -> None:
    with patch(
        "socket.getaddrinfo",
        return_value=_addr_info("93.184.216.34"),
    ):
        assert (
            validate_cloud_base_url("https://auth.example.com/base/")
            == "https://auth.example.com/base"
        )


@pytest.mark.parametrize(
    "url",
    [
        "ftp://auth.example.com",
        "https://user:pass@auth.example.com",  # pragma: allowlist secret
        "https://auth.example.com?token=secret",  # pragma: allowlist secret
        "https://auth.example.com/../admin",
        "https://auth.example.com/%2e%2e/admin",
        "https://auth.example.com/safe%2f..%2fadmin",
    ],
)
def test_validate_cloud_base_url_rejects_malformed_urls(url: str) -> None:
    with pytest.raises(ValueError):
        validate_cloud_base_url(url)


def test_validate_cloud_base_url_fails_closed_when_env_is_unset() -> None:
    # Unset / unrecognized environment must be treated as production (strict):
    # a deployment that never set an env marker must not silently allow
    # credential egress to localhost / private / metadata hosts.
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="private or loopback"):
            validate_cloud_base_url("https://127.0.0.1:5000")


def test_validate_cloud_base_url_rejects_private_ip_in_production() -> None:
    with patch.dict("os.environ", {"ENVIRONMENT": "production"}, clear=True):
        with pytest.raises(ValueError, match="private or loopback"):
            validate_cloud_base_url("https://127.0.0.1:5000")


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://metadata.google.internal", "metadata service"),
        ("http://metadata.google.internal.", "metadata service"),
        ("http://0x7f.0.0.1", "non-standard IP address notation"),
        ("http://2130706433", "non-standard IP address notation"),
    ],
)
def test_validate_cloud_base_url_rejects_metadata_and_nonstandard_ip_hosts(
    url: str,
    message: str,
) -> None:
    with patch.dict("os.environ", {"ENVIRONMENT": "production"}, clear=True):
        with pytest.raises(ValueError, match=message):
            validate_cloud_base_url(url)


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://metadata", "metadata service"),
        ("https://metadata.", "metadata service"),
        ("https://metadata.google.internal", "metadata service"),
        ("https://metadata.google.internal.", "metadata service"),
        ("https://0x7f.0.0.1", "non-standard IP address notation"),
        ("https://2130706433", "non-standard IP address notation"),
        ("https://0177.0.0.1", "non-standard IP address notation"),
    ],
)
def test_validate_cloud_base_url_rejects_metadata_and_numeric_hosts_before_dns(
    url: str,
    message: str,
) -> None:
    mock_getaddrinfo = Mock(return_value=_addr_info("93.184.216.34"))

    with (
        patch.dict("os.environ", {"ENVIRONMENT": "production"}, clear=True),
        patch("traigent.cloud.url_security.socket.getaddrinfo", mock_getaddrinfo),
    ):
        with pytest.raises(ValueError, match=message):
            validate_cloud_base_url(url)

    mock_getaddrinfo.assert_not_called()


def test_validate_cloud_base_url_fails_closed_on_dns_failure() -> None:
    with (
        patch.dict("os.environ", {"ENVIRONMENT": "production"}, clear=True),
        patch("socket.getaddrinfo", side_effect=socket.gaierror),
    ):
        with pytest.raises(ValueError, match="could not be resolved"):
            validate_cloud_base_url("https://unresolvable.example")


@pytest.mark.asyncio
async def test_validate_cloud_base_url_async_uses_thread_for_dns_work() -> None:
    with patch(
        "traigent.cloud.url_security.asyncio.to_thread",
        new=AsyncMock(return_value="https://auth.example.com"),
    ) as mock_to_thread:
        assert (
            await validate_cloud_base_url_async(
                "https://auth.example.com", purpose="async test"
            )
            == "https://auth.example.com"
        )

    mock_to_thread.assert_called_once()


def test_validate_cloud_base_url_allows_localhost_in_development() -> None:
    with patch.dict("os.environ", {"TRAIGENT_ENV": "development"}, clear=True):
        assert (
            validate_cloud_base_url("http://localhost:5000/") == "http://localhost:5000"
        )


def test_validate_cloud_base_url_honors_traigent_environment_development() -> None:
    with patch.dict("os.environ", {"TRAIGENT_ENVIRONMENT": "development"}, clear=True):
        assert (
            validate_cloud_base_url("http://localhost:5000/") == "http://localhost:5000"
        )


@pytest.mark.parametrize(
    "env",
    [
        # Canonical-first key order: the strict marker is the one that resolves.
        {"ENVIRONMENT": "production", "TRAIGENT_ENV": "development"},
        # Reversed: the dev marker resolves FIRST via the canonical key order,
        # so a gate that simply asked resolve_environment_name() would hand a
        # production deployment the development allowance. Any strict signal on
        # any canonical key must win regardless of order.
        {"ENVIRONMENT": "development", "TRAIGENT_ENV": "production"},
        {"ENVIRONMENT": "development", "TRAIGENT_ENVIRONMENT": "production"},
        # A staging marker is strict here too, so it must not be overridden by
        # a dev marker on an earlier-resolving key either.
        {"ENVIRONMENT": "development", "TRAIGENT_ENV": "staging"},
    ],
)
def test_strict_env_signal_wins_over_dev_signal_in_any_key_order(
    env: dict[str, str],
) -> None:
    # Conflicting canonical markers must fail closed: a box whose shell exports
    # ENVIRONMENT=development must not relax credential egress just because it
    # sorts ahead of an explicit TRAIGENT_ENV=production deployment marker.
    with patch.dict("os.environ", env, clear=True):
        with pytest.raises(ValueError, match="must use https in production"):
            validate_cloud_base_url("http://localhost:5000")
        with pytest.raises(ValueError, match="private or loopback"):
            validate_cloud_base_url("https://127.0.0.1:5000")


@pytest.mark.parametrize("env_value", ["staging", "stage"])
@pytest.mark.parametrize(
    "env_key", ["ENVIRONMENT", "TRAIGENT_ENV", "TRAIGENT_ENVIRONMENT"]
)
def test_staging_is_strict_like_production(env_key: str, env_value: str) -> None:
    # Staging is a real deployment tier: it runs on real cloud hosts carrying
    # real Traigent API keys, so it must not inherit the local-development
    # allowance for plaintext HTTP or private/loopback egress — even though the
    # canonical policy helper classifies it non-production for feature gates.
    with patch.dict("os.environ", {env_key: env_value}, clear=True):
        with pytest.raises(ValueError, match="must use https in production"):
            validate_cloud_base_url("http://localhost:5000")
        with pytest.raises(ValueError, match="private or loopback"):
            validate_cloud_base_url("https://127.0.0.1:5000")


def test_staging_rejects_plaintext_egress_to_private_hostname() -> None:
    # The concrete regression: ENVIRONMENT=staging with an internal backend URL
    # resolving to a private address must not send the API key in cleartext.
    with patch.dict("os.environ", {"ENVIRONMENT": "staging"}, clear=True):
        with patch("socket.getaddrinfo", return_value=_addr_info("10.0.0.5")):
            with pytest.raises(ValueError, match="must use https in production"):
                validate_cloud_base_url("http://internal-db.corp")


# --- Regression: unified env detection (issue #1905) ---------------------------


@pytest.mark.parametrize("noncanonical_key", ["APP_ENV", "FLASK_ENV"])
def test_env_detection_ignores_noncanonical_keys(noncanonical_key: str) -> None:
    # Before #1905 url_security read a private, divergent key set that included
    # APP_ENV/FLASK_ENV, so one of these alone could relax the credential-egress
    # gate even though the canonical policy detector (ENVIRONMENT/TRAIGENT_ENV/
    # TRAIGENT_ENVIRONMENT) still treated the run as production. The unified gate
    # must ignore these keys and fail closed (strict) for localhost.
    # https:// so the strict-TLS check is satisfied and the assertion targets
    # the localhost host-rejection specifically (the SSRF-relevant relaxation).
    with patch.dict("os.environ", {noncanonical_key: "development"}, clear=True):
        with pytest.raises(ValueError, match="not allowed in production"):
            validate_cloud_base_url("https://localhost:5000")


def test_env_detection_matches_canonical_policy_helper() -> None:
    # ENVIRONMENT=test is a canonical non-production marker, so the unified gate
    # must classify it from the shared value set (allow_local) rather than via a
    # private value-set that could disagree with other gates. The gate is allowed
    # to be STRICTER than the helper, never looser — see
    # test_staging_is_strict_like_production for the one documented divergence.
    from traigent.utils.env_config import treat_as_production_policy

    with patch.dict("os.environ", {"ENVIRONMENT": "test"}, clear=True):
        assert treat_as_production_policy() is False
        assert (
            validate_cloud_base_url("http://localhost:5000/") == "http://localhost:5000"
        )


def test_env_detection_reuses_canonical_key_and_value_sets() -> None:
    # Pin the reuse structurally, not just behaviorally: this gate's development
    # set must be the canonical non-production set minus the strict-egress tiers,
    # so a new canonical env name cannot silently miss this gate and cannot be
    # looser here than treat_as_production_policy.
    from traigent.cloud.url_security import (
        _STRICT_EGRESS_ENV_NAMES,
        _is_development_environment,
    )
    from traigent.utils.env_config import (
        _ENVIRONMENT_KEYS,
        _NON_PRODUCTION_ENV_NAMES,
        treat_as_production_policy,
    )

    assert _STRICT_EGRESS_ENV_NAMES <= _NON_PRODUCTION_ENV_NAMES

    for key in _ENVIRONMENT_KEYS:
        for name in _NON_PRODUCTION_ENV_NAMES:
            with patch.dict("os.environ", {key: name}, clear=True):
                assert treat_as_production_policy() is False
                assert _is_development_environment() is (
                    name not in _STRICT_EGRESS_ENV_NAMES
                )


# --- Regression: metadata/link-local egress is always blocked (issue #1908) ----


@pytest.mark.parametrize(
    ("url", "message"),
    [
        # IMDS IP literals absent from the hostname set must be blocked by IP.
        ("http://169.254.169.254", "metadata service"),
        ("http://169.254.169.254/latest/meta-data/", "metadata service"),
        # Alibaba IMDS (not link-local).
        ("http://100.100.100.200", "metadata service"),
        # AWS IMDSv2 IPv6 (unique-local, not link-local).
        ("http://[fd00:ec2::254]", "metadata service"),
        # Any other link-local literal is blocked by range, not just IMDS.
        ("http://169.254.1.1", "link-local address"),
        # IPv6 literals embedding an IPv4 IMDS address. Linux maps ::ffff:0:0/96
        # onto the IPv4 peer at connect time (bindv6only=0), so these reach the
        # metadata service exactly like the bare literal above, while Python
        # reports them as neither link-local nor equal to the IPv4 address.
        ("http://[::ffff:169.254.169.254]", "metadata service"),
        # Same address, hex spelling of the embedded IPv4 octets.
        ("http://[::ffff:a9fe:a9fe]", "metadata service"),
        # Same address, fully expanded.
        ("http://[0:0:0:0:0:ffff:169.254.169.254]", "metadata service"),
        # Alibaba IMDS mapped: is_global=True on IPv6 (100.64.0.0/10 is not
        # classified private), so it also cleared the production gate.
        ("http://[::ffff:100.100.100.200]", "metadata service"),
        # Mapped link-local generally, not just the IMDS addresses.
        ("http://[::ffff:169.254.1.1]", "link-local address"),
        # Deprecated IPv4-compatible / NAT64 / 6to4 forms (defense in depth).
        ("http://[::169.254.169.254]", "metadata service"),
        ("http://[64:ff9b::169.254.169.254]", "metadata service"),
        ("http://[2002:a9fe:a9fe::1]", "metadata service"),
    ],
)
def test_metadata_and_link_local_ips_blocked_even_in_development(
    url: str,
    message: str,
) -> None:
    # allow_local=True (development) must NOT open a path to the cloud metadata
    # service or the link-local range for a credential-bearing request.
    with patch.dict("os.environ", {"TRAIGENT_ENV": "development"}, clear=True):
        with pytest.raises(ValueError, match=message):
            validate_cloud_base_url(url)


@pytest.mark.parametrize(
    ("resolved_ip", "message"),
    [
        ("169.254.169.254", "metadata service"),  # AWS/GCP/Azure IMDS.
        ("100.100.100.200", "metadata service"),  # Alibaba IMDS.
        ("fd00:ec2::254", "metadata service"),  # AWS IMDSv2 IPv6.
        ("169.254.1.1", "link-local address"),  # Link-local range generally.
        # A resolver can answer with the IPv4-mapped form (an AAAA record, or
        # getaddrinfo normalizing a mapped literal), which reaches the same
        # IPv4 peer and must be classified the same way.
        ("::ffff:169.254.169.254", "metadata service"),
        ("::ffff:100.100.100.200", "metadata service"),
        ("::ffff:169.254.1.1", "link-local address"),
    ],
)
def test_hostname_resolving_to_metadata_blocked_in_development(
    resolved_ip: str,
    message: str,
) -> None:
    # The residue behind the IP-literal guard: development returned before DNS
    # resolution, so an attacker-chosen hostname resolving to IMDS still received
    # credentials. Resolution must happen in every environment and its answers
    # must go through the always-blocked guard.
    with patch.dict("os.environ", {"TRAIGENT_ENV": "development"}, clear=True):
        with patch("socket.getaddrinfo", return_value=_addr_info(resolved_ip)):
            with pytest.raises(ValueError, match=message):
                validate_cloud_base_url("http://imds.rebind.example.com")


def test_hostname_resolving_to_scoped_link_local_blocked_in_development() -> None:
    # getaddrinfo returns IPv6 link-local answers with a zone id; the zone must
    # be stripped rather than making the answer unparseable and skipped.
    scoped = [
        (
            socket.AF_INET6,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
            "",
            ("fe80::1%eth0", 443),
        )
    ]
    with patch.dict("os.environ", {"TRAIGENT_ENV": "development"}, clear=True):
        with patch("socket.getaddrinfo", return_value=scoped):
            with pytest.raises(ValueError, match="link-local address"):
                validate_cloud_base_url("http://rebind.example.com")


def test_local_mdns_hostname_resolving_to_metadata_blocked_in_development() -> None:
    # ``.local`` is allowed in development, but the allowance covers legitimate
    # local endpoints — not an mDNS name pointing at the metadata service.
    with patch.dict("os.environ", {"TRAIGENT_ENV": "development"}, clear=True):
        with patch(
            "socket.getaddrinfo",
            return_value=_addr_info("169.254.169.254"),
        ):
            with pytest.raises(ValueError, match="metadata service"):
                validate_cloud_base_url("http://imds.local")


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        # IPv4-mapped: the routable case (Linux connects these to the IPv4 peer).
        ("::ffff:169.254.169.254", "169.254.169.254"),
        ("::ffff:a9fe:a9fe", "169.254.169.254"),
        ("0:0:0:0:0:ffff:169.254.169.254", "169.254.169.254"),
        ("::ffff:93.184.216.34", "93.184.216.34"),
        # Deprecated/relayed forms unwrapped as defense in depth.
        ("::169.254.169.254", "169.254.169.254"),
        ("64:ff9b::169.254.169.254", "169.254.169.254"),
        ("2002:a9fe:a9fe::1", "169.254.169.254"),
        # Embeds nothing -> returned unchanged. ::1 and :: sit inside ::/96 but
        # are the loopback/unspecified addresses, not IPv4-compatible addresses;
        # unwrapping them would report ::1 as 0.0.0.1. Both classify the same way
        # today, so only this assertion pins the distinction.
        ("::1", "::1"),
        ("::", "::"),
        ("fd00:ec2::254", "fd00:ec2::254"),
        ("fe80::1", "fe80::1"),
        ("2606:2800:220:1:248:1893:25c8:1946", "2606:2800:220:1:248:1893:25c8:1946"),
        # IPv4 input is passed through untouched.
        ("169.254.169.254", "169.254.169.254"),
    ],
)
def test_unwrap_embedded_ipv4(address: str, expected: str) -> None:
    assert _unwrap_embedded_ipv4(ipaddress.ip_address(address)) == ipaddress.ip_address(
        expected
    )


def test_mapped_alibaba_imds_blocked_in_production() -> None:
    # ::ffff:100.100.100.200 cleared BOTH always-blocked guards (exact-match IMDS
    # set, is_link_local) and the production is_global gate, because
    # IPv6Address.is_global is `not is_private` and 100.64.0.0/10 is not private.
    # It was therefore reachable from production, not just development.
    with patch.dict("os.environ", {"TRAIGENT_ENV": "production"}, clear=True):
        with pytest.raises(ValueError, match="metadata service"):
            validate_cloud_base_url("https://[::ffff:100.100.100.200]")


@pytest.mark.parametrize(
    "url",
    [
        # NAT64 64:ff9b::/96 embedding 127.0.0.1 -> loopback via a NAT64 gateway.
        # Rejected on the production is_global gate (not the always-blocked set),
        # so this pins the private/loopback path through a transition prefix,
        # which the metadata-IP cases above do not exercise (issue #1914).
        "https://[64:ff9b::7f00:1]/api",
        # NAT64 embedding 169.254.169.254 -> cloud IMDS via a NAT64 gateway.
        "https://[64:ff9b::a9fe:a9fe]/api",
        # IPv4-mapped Alibaba metadata IP; is_global reports True on the outer
        # IPv6 literal, so the embedded IPv4 must be re-classified.
        "https://[::ffff:100.100.100.200]/api",
        # 6to4 2002::/16 embedding a link-local metadata IPv4 (169.254.169.254).
        "https://[2002:a9fe:a9fe::]/api",
    ],
)
def test_validate_cloud_base_url_rejects_ipv6_transition_metadata_ips(
    url: str,
) -> None:
    # Regression for issue #1914: ``is_global`` reports True on these IPv6
    # transition / embedded-IPv4 literals, so without unwrapping the embedded
    # IPv4 the SSRF / credential-egress gate could be bypassed to loopback or a
    # metadata (IMDS) endpoint in production. The unwrap classifies the address
    # the request actually reaches, so every vector is rejected — some as a
    # metadata service (blocked in every environment), the pure-loopback NAT64
    # form on the production private/loopback gate.
    with patch.dict("os.environ", {"ENVIRONMENT": "production"}, clear=True):
        with pytest.raises(
            ValueError, match="metadata service|private or loopback|link-local"
        ):
            validate_cloud_base_url(url)


@pytest.mark.parametrize(
    "url",
    [
        # Loopback: ::1 lives in ::/96 but embeds no IPv4 address — unwrapping it
        # would yield 0.0.0.1 and change how it is classified.
        "http://[::1]:8000",
        # Unique-local IPv6 dev endpoint.
        "http://[fd12:3456::1]:8000",
        # Mapped IPv4 private endpoint stays usable in development.
        "http://[::ffff:192.168.1.10]:8000",
    ],
)
def test_development_still_allows_local_ipv6_literals(url: str) -> None:
    # The unwrap must not over-block: legitimate loopback/private IPv6 literals
    # keep the development allowance they had before it.
    with patch.dict("os.environ", {"TRAIGENT_ENV": "development"}, clear=True):
        assert validate_cloud_base_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        # Ordinary global IPv6 is untouched by the unwrap.
        "https://[2606:2800:220:1:248:1893:25c8:1946]",
        # A mapped *public* IPv4 address is global and must stay allowed.
        "https://[::ffff:93.184.216.34]",
    ],
)
def test_production_still_allows_global_ipv6_literals(url: str) -> None:
    with patch.dict("os.environ", {"TRAIGENT_ENV": "production"}, clear=True):
        assert validate_cloud_base_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://[::1]:8000",
        "http://[::ffff:169.254.169.254]",
    ],
)
def test_ipv6_literals_never_trigger_dns_resolution(url: str) -> None:
    # The unwrap runs inside _parse_ip_literal, which _validation_needs_dns_resolution
    # also consults. An unwrap that made a literal unparseable would silently turn
    # it into a name lookup and break sync/async parity.
    assert _validation_needs_dns_resolution(url) is False


# --- The development allowance itself is preserved ----------------------------


@pytest.mark.parametrize(
    ("url", "resolved_ip", "expected"),
    [
        # A private/LAN dev backend stays reachable in development.
        (
            "http://dev-backend.example.com:8000/",
            "10.0.0.5",
            "http://dev-backend.example.com:8000",
        ),
        ("http://box.local:8000/", "192.168.1.10", "http://box.local:8000"),
    ],
)
def test_development_still_allows_private_endpoints(
    url: str,
    resolved_ip: str,
    expected: str,
) -> None:
    with patch.dict("os.environ", {"TRAIGENT_ENV": "development"}, clear=True):
        with patch("socket.getaddrinfo", return_value=_addr_info(resolved_ip)):
            assert validate_cloud_base_url(url) == expected


def test_development_allows_unresolvable_host() -> None:
    # Adding resolution to the development path must not turn an offline/local-only
    # hostname into a hard failure. This pins a deliberate development-only
    # allowance, NOT a safety property: a name that fails to resolve at validation
    # time may still resolve at connect time (see the residue note in
    # _reject_unsafe_hostname). It is strictly narrower than the base behavior,
    # which skipped resolution in development entirely.
    with patch.dict("os.environ", {"TRAIGENT_ENV": "development"}, clear=True):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror):
            assert (
                validate_cloud_base_url("http://offline-dev.example.com/")
                == "http://offline-dev.example.com"
            )


def test_development_loopback_name_skips_resolution() -> None:
    # localhost is reserved for loopback, so it must not pay for a DNS lookup.
    with patch.dict("os.environ", {"TRAIGENT_ENV": "development"}, clear=True):
        with patch("socket.getaddrinfo", side_effect=AssertionError("no DNS")) as dns:
            assert (
                validate_cloud_base_url("http://localhost:5000/")
                == "http://localhost:5000"
            )
        dns.assert_not_called()
