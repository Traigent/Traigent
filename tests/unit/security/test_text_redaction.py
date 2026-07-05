"""Tests for traigent.security.redaction."""

from __future__ import annotations

import pytest

from traigent.security.redaction import (
    CONTENT_KEY_FRAGMENTS,
    CREDENTIAL_KEY_FRAGMENTS,
    is_content_key_name,
    is_credential_key_name,
    redact_sensitive_data,
    redact_sensitive_text,
)


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
            "20260524-210743",  # valid Luhn if separators are stripped
            "walkthrough-mock-tiny-20260523-010956-abc123",
            "walkthrough-mock-tiny-20260524-210743-abc123",
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
        assert "[REDACTED:bearer_token]" in redact_sensitive_text(
            "Bearer eyJhbGciOiJIUzI1NiIs"
        )

    def test_api_key_redacted(self) -> None:
        assert "[REDACTED:api_key]" in redact_sensitive_text(
            "X-Api-Key: sk-abcd1234abcd1234"
        )


class TestCanonicalKeyNameSets:
    """The two canonical key-name sets consumed by the three
    formerly-divergent SDK sanitizers (dataset_converter, observability
    decorators, agent_spans). See issue #1649.

    Deliberately two tiers: the credential set applies on every sanitizer
    path; the content-marker set only where content-shaped fields must be
    dropped entirely (agent_spans) — NOT on tuned-config surfaces, where a
    key literally named "prompt" is a variant label the portal must show.
    """

    @pytest.mark.parametrize(
        "key",
        [
            # Union member from traigent.cloud.dataset_converter's old regex
            "authorization",
            "credit_card",
            "creditcard",
            # Union member from traigent.observability.decorators' old list
            "credential",
            "private_key",
            "apikey",
            "auth_header",
            "password",
            "secret",
            "token",
        ],
    )
    def test_credential_union_member_flagged(self, key: str) -> None:
        assert is_credential_key_name(key) is True

    @pytest.mark.parametrize(
        "key",
        ["prompt", "response", "actual_output", "expected", "completion", "output"],
    )
    def test_content_marker_flagged_as_content_not_credential(self, key: str) -> None:
        assert is_content_key_name(key) is True
        # Content markers must NOT be treated as credentials — tuned config
        # variables are routinely named "prompt"/"response_format" etc.
        assert is_credential_key_name(key) is False

    def test_ordinary_key_not_flagged(self) -> None:
        assert is_credential_key_name("model_name") is False
        assert is_content_key_name("model_name") is False
        assert is_credential_key_name("safe_score") is False
        assert is_content_key_name("safe_score") is False

    def test_fragment_sets_are_frozen_nonempty_and_disjoint(self) -> None:
        assert isinstance(CREDENTIAL_KEY_FRAGMENTS, frozenset)
        assert isinstance(CONTENT_KEY_FRAGMENTS, frozenset)
        assert CREDENTIAL_KEY_FRAGMENTS
        assert CONTENT_KEY_FRAGMENTS
        assert not (CREDENTIAL_KEY_FRAGMENTS & CONTENT_KEY_FRAGMENTS)


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
