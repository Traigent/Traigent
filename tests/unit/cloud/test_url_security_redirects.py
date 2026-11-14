"""Cross-host redirect guard for credential-bearing aiohttp sessions."""

from __future__ import annotations

import pytest

from traigent.cloud.url_security import (
    CrossHostRedirectError,
    check_redirect_stays_on_host,
    same_host_redirect_trace_configs,
)


def test_same_host_redirects_allowed():
    url = "https://api.example.com/api/v1/x"
    assert (
        check_redirect_stays_on_host(url, "/api/v1/y")
        == "https://api.example.com/api/v1/y"
    )
    assert check_redirect_stays_on_host(
        "http://api.example.com/a", "https://api.example.com/a"
    ).startswith("https://")


@pytest.mark.parametrize(
    "location",
    [
        "https://evil.example.net/steal",
        "//evil.example.net/steal",
        "http://api.example.com/downgrade",
        "http://169.254.169.254/latest/meta-data",
    ],
)
def test_cross_host_or_downgrade_redirects_refused(location):
    with pytest.raises(CrossHostRedirectError):
        check_redirect_stays_on_host("https://api.example.com/api/v1/x", location)


def test_trace_configs_registered():
    configs = same_host_redirect_trace_configs()
    assert len(configs) == 1
    assert configs[0].on_request_redirect
