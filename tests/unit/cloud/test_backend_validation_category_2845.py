"""#2845: HTTP status decides the key-validation category; body words only refine."""

from __future__ import annotations

import pytest

from traigent.cloud.auth import AuthManager

_category = AuthManager._backend_validation_category


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (
            503,
            "service unavailable: unable to verify authentication, upstream timeout",
            "backend temporarily unavailable",
        ),
        (
            500,
            "internal error: invalid state in handler",
            "backend temporarily unavailable",
        ),
        (
            502,
            "<html>bad gateway permission denied by upstream</html>",
            "backend temporarily unavailable",
        ),
        (
            504,
            "gateway timeout waiting for scope service",
            "backend temporarily unavailable",
        ),
        (408, "request timeout", "backend temporarily unavailable"),
        (429, "invalid request rate for scope tenant", "rate limited"),
    ],
)
def test_transient_status_beats_body_words(
    status: int, body: str, expected: str
) -> None:
    assert _category(status, body.lower(), {}) == expected


def test_request_id_containing_waf_is_not_edge_blocked() -> None:
    body = '{"request_id":"zx9wafq1kp"}'
    assert _category(403, body, {}) == "insufficient scope"


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        ("blocked by waf", {}),
        ("", {"cf-ray": "abc"}),
        ("", {"x-amzn-waf-action": "block"}),
        ("attention required | cloudflare", {}),
    ],
)
def test_real_edge_signals_still_read_as_edge_blocked(body: str, headers: dict) -> None:
    assert _category(403, body, headers) == "edge blocked"


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, "invalid api key", "invalid API key"),
        (401, "key expired on 2026-01-01", "expired API key"),
        (403, "missing required scope", "insufficient scope"),
        (403, "", "insufficient scope"),
        (401, "", "invalid API key"),
        (400, "invalid payload", "backend validation failed"),
    ],
)
def test_auth_statuses_keep_their_categories(
    status: int, body: str, expected: str
) -> None:
    assert _category(status, body, {}) == expected
