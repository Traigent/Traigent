"""Tests for outbound URL safety validation."""

import socket
from unittest.mock import Mock

import pytest

from traigent.utils.url_security import UnsafeUrlError, validate_outbound_url


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


def test_validate_outbound_url_normalizes_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, _port: _addr_info("93.184.216.34"),
    )

    assert (
        validate_outbound_url("https://api.example.com/") == "https://api.example.com"
    )


@pytest.mark.parametrize(
    "url, message",
    [
        ("", "must not be empty"),
        ("file:///tmp/socket", "http or https"),
        ("https://example.com?token=abc", "query or fragment"),
        (
            "https://user:pass@example.com",
            "embedded credentials",
        ),  # pragma: allowlist secret
        ("https://example.com:99999", "invalid port"),
        ("https://api.example.com/v1/..", "dot-segment"),
        ("https://api.example.com/v1/%2e%2e", "dot-segment"),
        ("https://api.example.com/v1%2f..", "dot-segment"),
        ("http://metadata.google.internal", "metadata service"),
        ("http://169.254.169.254", "non-routable"),
        ("http://2130706433", "non-standard IP"),
        ("http://0x7f.0.0.1", "non-standard IP"),
        ("http://127.0.0.1", "private address"),
        ("http://localhost", "localhost"),
    ],
)
def test_validate_outbound_url_rejects_unsafe_urls(url: str, message: str) -> None:
    with pytest.raises(UnsafeUrlError, match=message):
        validate_outbound_url(url)


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://metadata", "metadata service"),
        ("https://metadata.", "metadata service"),
        ("https://metadata.google.internal", "metadata service"),
        ("https://metadata.google.internal.", "metadata service"),
        ("https://0x7f.0.0.1", "non-standard IP"),
        ("https://2130706433", "non-standard IP"),
        ("https://0177.0.0.1", "non-standard IP"),
    ],
)
def test_validate_outbound_url_rejects_metadata_and_numeric_hosts_before_dns(
    url: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_getaddrinfo = Mock(return_value=_addr_info("93.184.216.34"))
    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo)

    with pytest.raises(UnsafeUrlError, match=message):
        validate_outbound_url(url)

    mock_getaddrinfo.assert_not_called()


def test_validate_outbound_url_allows_private_hosts_when_requested() -> None:
    assert (
        validate_outbound_url("http://127.0.0.1:8080/", allow_private_hosts=True)
        == "http://127.0.0.1:8080"
    )


def test_validate_outbound_url_rejects_hostname_resolving_to_private_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, _port: _addr_info("169.254.169.254"),
    )

    with pytest.raises(UnsafeUrlError, match="must not resolve"):
        validate_outbound_url("https://rebind.example")


def test_validate_outbound_url_accepts_hostname_resolving_to_global_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, _port: _addr_info("93.184.216.34"),
    )

    assert (
        validate_outbound_url("https://api.example.com/") == "https://api.example.com"
    )


def test_validate_outbound_url_allows_localhost_when_private_hosts_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_resolved(_host: str, _port: int | None) -> list[object]:
        raise AssertionError("allow_private_hosts should skip DNS resolution")

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fail_if_resolved,
    )

    assert (
        validate_outbound_url("http://localhost:8080/", allow_private_hosts=True)
        == "http://localhost:8080"
    )


def test_validate_outbound_url_blocks_metadata_when_private_hosts_allowed() -> None:
    with pytest.raises(UnsafeUrlError, match="non-routable"):
        validate_outbound_url("http://169.254.169.254", allow_private_hosts=True)
