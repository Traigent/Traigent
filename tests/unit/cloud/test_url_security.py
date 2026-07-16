"""Tests for cloud URL safety validation."""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, Mock, patch

import pytest

from traigent.cloud.url_security import (
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


def test_validate_cloud_base_url_production_signal_wins_over_dev_signal() -> None:
    with patch.dict(
        "os.environ",
        {"ENVIRONMENT": "production", "TRAIGENT_ENV": "development"},
        clear=True,
    ):
        with pytest.raises(ValueError, match="must use https in production"):
            validate_cloud_base_url("http://localhost:5000")


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
    # must classify it exactly as the shared policy helper does (allow_local),
    # rather than via a private value-set that could disagree with other gates.
    from traigent.utils.env_config import treat_as_production_policy

    with patch.dict("os.environ", {"ENVIRONMENT": "test"}, clear=True):
        assert treat_as_production_policy() is False
        assert (
            validate_cloud_base_url("http://localhost:5000/") == "http://localhost:5000"
        )


# --- Regression: metadata/link-local egress is always blocked (issue #1908) ----


@pytest.mark.parametrize(
    ("url", "message"),
    [
        # IMDS IP literals absent from the hostname set must be blocked by IP.
        ("http://169.254.169.254", "metadata service"),
        ("http://169.254.169.254/latest/meta-data/", "metadata service"),
        ("http://100.100.100.200", "metadata service"),  # Alibaba IMDS (not link-local)
        ("http://[fd00:ec2::254]", "metadata service"),  # AWS IMDSv2 IPv6 (unique-local)
        # Any other link-local literal is blocked by range, not just IMDS.
        ("http://169.254.1.1", "link-local address"),
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
