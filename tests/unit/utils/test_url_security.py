"""Tests for outbound URL safety validation."""

import pytest

from traigent.utils.url_security import UnsafeUrlError, validate_outbound_url


def test_validate_outbound_url_normalizes_trailing_slash() -> None:
    assert validate_outbound_url("https://api.example.com/") == "https://api.example.com"


@pytest.mark.parametrize(
    "url, message",
    [
        ("", "must not be empty"),
        ("file:///tmp/socket", "http or https"),
        ("https://example.com?token=abc", "query or fragment"),
        ("https://user:pass@example.com", "embedded credentials"),  # pragma: allowlist secret
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


def test_validate_outbound_url_allows_private_hosts_when_requested() -> None:
    assert (
        validate_outbound_url("http://127.0.0.1:8080/", allow_private_hosts=True)
        == "http://127.0.0.1:8080"
    )


def test_validate_outbound_url_still_blocks_metadata_when_private_hosts_allowed() -> None:
    with pytest.raises(UnsafeUrlError, match="non-routable"):
        validate_outbound_url("http://169.254.169.254", allow_private_hosts=True)
