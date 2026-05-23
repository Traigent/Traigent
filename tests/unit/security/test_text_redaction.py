"""Tests for traigent.security.redaction."""

from __future__ import annotations

import pytest

from traigent.security.redaction import redact_sensitive_data, redact_sensitive_text


class TestCreditCardRedaction:
    @pytest.mark.parametrize(
        "pan",
        [
            "4111-1111-1111-1111",  # Visa test PAN, valid Luhn
            "4111111111111111",  # same without separators
            "4111 1111 1111 1111",  # space separators
            "5555555555554444",  # Mastercard test PAN, valid Luhn
            "378282246310005",  # Amex test PAN, valid Luhn
        ],
    )
    def test_valid_pan_is_redacted(self, pan: str) -> None:
        out = redact_sensitive_text(f"card: {pan} end")
        assert "[REDACTED:credit_card]" in out
        assert pan not in out

    @pytest.mark.parametrize(
        "timestamp",
        [
            "20260523-010956",  # YYYYMMDD-HHMMSS — was the false positive
            "walkthrough-mock-tiny-20260523-010956-abc123",
            "logs-2026-05-23-01-09-56",
        ],
    )
    def test_timestamp_is_not_redacted(self, timestamp: str) -> None:
        out = redact_sensitive_text(timestamp)
        assert "REDACTED" not in out, f"timestamp false-positive: {out!r}"

    def test_random_luhn_failing_digit_run_not_redacted(self) -> None:
        # Sequential digits — does NOT pass Luhn
        out = redact_sensitive_text("order id 1234567890123 created")
        assert "REDACTED" not in out


class TestOtherPatternsStillWork:
    def test_email_redacted(self) -> None:
        assert "[REDACTED:email]" in redact_sensitive_text("contact: alice@example.com")

    def test_ssn_redacted(self) -> None:
        assert "[REDACTED:ssn]" in redact_sensitive_text("ssn 123-45-6789")

    def test_bearer_token_redacted(self) -> None:
        assert "[REDACTED:bearer_token]" in redact_sensitive_text("Bearer eyJhbGciOiJIUzI1NiIs")

    def test_api_key_redacted(self) -> None:
        assert "[REDACTED:api_key]" in redact_sensitive_text("X-Api-Key: sk-abcd1234abcd1234")


class TestNestedRedaction:
    def test_dict_recursive(self) -> None:
        out = redact_sensitive_data({"k": "alice@example.com", "n": 1})
        assert out["k"] == "[REDACTED:email]"
        assert out["n"] == 1

    def test_list_recursive(self) -> None:
        out = redact_sensitive_data(["alice@example.com", "plain"])
        assert out[0] == "[REDACTED:email]"
        assert out[1] == "plain"

    def test_none_passthrough(self) -> None:
        assert redact_sensitive_text(None) is None
