"""Issue #2877: OpenRouter funds preflight must consider the account balance.

The per-key limit (``/api/v1/key``) and the account balance
(``/api/v1/credits``) are independent; the lower one binds. All HTTP is mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from traigent.core.exception_handler import VendorErrorCategory, classify_vendor_error
from traigent.providers.validation import ProviderValidator

_SECRET = "sk-or-SECRETVALUE123"  # pragma: allowlist secret
_KEY = "https://openrouter.ai/api/v1/key"
_CREDITS = "https://openrouter.ai/api/v1/credits"


def _resp(body, status=200):
    return SimpleNamespace(status_code=status, json=lambda: body)


def _check(monkeypatch, key_resp, credits_resp=None, credits_exc=None):
    monkeypatch.setenv("OPENROUTER_API_KEY", _SECRET)
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if url == _KEY:
            return key_resp
        if url == _CREDITS:
            if credits_exc is not None:
                raise credits_exc
            return credits_resp
        raise AssertionError(f"unexpected url {url}")

    with patch("requests.get", side_effect=fake_get):
        status = ProviderValidator()._check_openrouter()
    return status, calls


def test_low_account_balance_beats_roomy_key_limit(monkeypatch):
    # Issue evidence: key has 16.69 left, account has 0.53 left.
    status, calls = _check(
        monkeypatch,
        _resp({"data": {"limit": 20, "limit_remaining": 16.69}}),
        _resp({"data": {"total_credits": 20, "total_usage": 19.47}}),
    )
    assert _CREDITS in calls
    assert status.funds == "available"
    assert "0.53" in status.message
    assert status.valid is True


def test_exhausted_account_balance_is_depleted(monkeypatch):
    status, _ = _check(
        monkeypatch,
        _resp({"data": {"limit": 20, "limit_remaining": 16.69}}),
        _resp({"data": {"total_credits": 20, "total_usage": 20}}),
    )
    assert status.valid is False
    assert status.funds == "depleted"
    assert status.error_type == "InsufficientFunds"


def test_key_limit_lower_than_balance_still_binds(monkeypatch):
    status, _ = _check(
        monkeypatch,
        _resp({"data": {"limit": 5, "limit_remaining": 0}}),
        _resp({"data": {"total_credits": 100, "total_usage": 1}}),
    )
    assert status.funds == "depleted"


def test_uncapped_key_uses_account_balance(monkeypatch):
    ok, _ = _check(
        monkeypatch,
        _resp({"data": {"limit": None, "limit_remaining": None}}),
        _resp({"data": {"total_credits": 10, "total_usage": 1}}),
    )
    assert ok.funds == "available"
    empty, _ = _check(
        monkeypatch,
        _resp({"data": {"limit": None, "limit_remaining": None}}),
        _resp({"data": {"total_credits": 10, "total_usage": 10}}),
    )
    assert empty.funds == "depleted"


@pytest.mark.parametrize(
    "credits_resp,credits_exc",
    [
        (_resp({}, status=500), None),
        (_resp({}, status=403), None),
        (_resp({"data": {}}), None),
        (_resp({"data": {"total_credits": "x", "total_usage": 1}}), None),
        (_resp({"data": {"total_credits": float("nan"), "total_usage": 1}}), None),
        (None, ConnectionError(f"boom {_SECRET}")),
    ],
)
def test_credits_failure_keeps_unknown(monkeypatch, credits_resp, credits_exc):
    status, _ = _check(
        monkeypatch,
        _resp({"data": {"limit": 20, "limit_remaining": 16.69}}),
        credits_resp,
        credits_exc,
    )
    assert status.valid is True
    assert status.funds == "unknown"
    assert _SECRET not in status.message


def test_unreadable_key_body_is_unknown_even_if_credits_ok(monkeypatch):
    status, _ = _check(
        monkeypatch,
        _resp({"data": {}}),
        _resp({"data": {"total_credits": 10, "total_usage": 1}}),
    )
    assert status.funds == "unknown"


def test_depleted_key_limit_skips_credits_call(monkeypatch):
    status, calls = _check(
        monkeypatch, _resp({"data": {"limit": 1, "limit_remaining": 0}})
    )
    assert status.funds == "depleted"
    assert calls == [_KEY]


@pytest.mark.parametrize(
    "message",
    [
        "APIError: This request would exceed your available credits given "
        "your current in-flight requests.",
        "This request would exceed your available credits",
    ],
)
def test_in_flight_credits_message_is_a_funds_error(message):
    assert classify_vendor_error(Exception(message)) == (
        VendorErrorCategory.INSUFFICIENT_FUNDS
    )


def test_non_finite_balance_is_unknown_not_available(monkeypatch):
    # inf - inf is NaN: a hostile/garbled body must not read as "available".
    status, _ = _check(
        monkeypatch,
        _resp({"data": {"limit": 20, "limit_remaining": 16.69}}),
        _resp({"data": {"total_credits": float("inf"), "total_usage": float("inf")}}),
    )
    assert status.funds == "unknown"
