"""Tests for cloud URL safety validation."""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, patch

import pytest

from traigent.cloud.url_security import (
    validate_cloud_base_url,
    validate_cloud_base_url_async,
)


def test_validate_cloud_base_url_normalizes_safe_public_url() -> None:
    with patch(
        "socket.getaddrinfo",
        return_value=[
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 443),
            )
        ],
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
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="private or loopback"):
            validate_cloud_base_url("https://127.0.0.1:5000")


def test_validate_cloud_base_url_rejects_private_ip_in_production() -> None:
    with patch.dict("os.environ", {"ENVIRONMENT": "production"}, clear=True):
        with pytest.raises(ValueError, match="private or loopback"):
            validate_cloud_base_url("https://127.0.0.1:5000")


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


def test_validate_cloud_base_url_production_signal_wins_over_dev_signal() -> None:
    with patch.dict(
        "os.environ",
        {"ENVIRONMENT": "production", "TRAIGENT_ENV": "development"},
        clear=True,
    ):
        with pytest.raises(ValueError, match="must use https in production"):
            validate_cloud_base_url("http://localhost:5000")
