"""Unit tests for traigent/security/auth/sms.py.

Tests for SMS-based authentication provider using Twilio.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from traigent.security.auth.sms import SMSAuthProvider
from traigent.utils.exceptions import AuthenticationError


class TestSMSAuthProviderInit:
    """Tests for SMSAuthProvider initialization."""

    @patch("traigent.security.auth.sms.TWILIO_AVAILABLE", True)
    @patch("traigent.security.auth.sms.TwilioClient")
    def test_init_with_valid_settings(self, mock_client: MagicMock) -> None:
        """Test initialization with valid settings."""

        settings = {
            "account_sid": "AC123456789",
            "auth_token": "test_token_12345",
            "from_number": "+15551234567",
        }

        provider = SMSAuthProvider(settings)

        assert provider.account_sid == "AC123456789"
        assert provider.auth_token == "test_token_12345"
        assert provider.from_number == "+15551234567"
        assert provider.max_attempts == 3  # Default
        assert provider.code_expiry_minutes == 5  # Default
        assert provider.rate_limit_minutes == 1  # Default
        assert provider.max_codes_per_hour == 5  # Default
        mock_client.assert_called_once_with("AC123456789", "test_token_12345")

    @patch("traigent.security.auth.sms.TWILIO_AVAILABLE", True)
    @patch("traigent.security.auth.sms.TwilioClient")
    def test_init_with_custom_settings(self, mock_client: MagicMock) -> None:
        """Test initialization with custom configuration."""

        settings = {
            "account_sid": "AC123456789",
            "auth_token": "test_token_12345",
            "from_number": "+15551234567",
            "max_attempts": 5,
            "code_expiry_minutes": 10,
            "rate_limit_minutes": 2,
            "max_codes_per_hour": 10,
        }

        provider = SMSAuthProvider(settings)

        assert provider.max_attempts == 5
        assert provider.code_expiry_minutes == 10
        assert provider.rate_limit_minutes == 2
        assert provider.max_codes_per_hour == 10

    @patch("traigent.security.auth.sms.TWILIO_AVAILABLE", False)
    def test_init_raises_import_error_when_twilio_unavailable(self) -> None:
        """Test initialization raises ImportError when Twilio is not available."""

        settings = {
            "account_sid": "AC123456789",
            "auth_token": "test_token_12345",
            "from_number": "+15551234567",
        }

        with pytest.raises(
            ImportError, match="twilio is required for SMS authentication"
        ):
            SMSAuthProvider(settings)

    @patch("traigent.security.auth.sms.TWILIO_AVAILABLE", True)
    @patch("traigent.security.auth.sms.TwilioClient")
    def test_init_raises_value_error_missing_account_sid(
        self, mock_client: MagicMock
    ) -> None:
        """Test initialization raises ValueError when account_sid is missing."""

        settings = {
            "auth_token": "test_token_12345",
            "from_number": "+15551234567",
        }

        with pytest.raises(
            ValueError, match="SMS settings missing required Twilio configuration"
        ):
            SMSAuthProvider(settings)

    @patch("traigent.security.auth.sms.TWILIO_AVAILABLE", True)
    @patch("traigent.security.auth.sms.TwilioClient")
    def test_init_raises_value_error_missing_auth_token(
        self, mock_client: MagicMock
    ) -> None:
        """Test initialization raises ValueError when auth_token is missing."""

        settings = {
            "account_sid": "AC123456789",
            "from_number": "+15551234567",
        }

        with pytest.raises(
            ValueError, match="SMS settings missing required Twilio configuration"
        ):
            SMSAuthProvider(settings)

    @patch("traigent.security.auth.sms.TWILIO_AVAILABLE", True)
    @patch("traigent.security.auth.sms.TwilioClient")
    def test_init_raises_value_error_missing_from_number(
        self, mock_client: MagicMock
    ) -> None:
        """Test initialization raises ValueError when from_number is missing."""

        settings = {
            "account_sid": "AC123456789",
            "auth_token": "test_token_12345",
        }

        with pytest.raises(
            ValueError, match="SMS settings missing required Twilio configuration"
        ):
            SMSAuthProvider(settings)

    @patch("traigent.security.auth.sms.TWILIO_AVAILABLE", True)
    @patch("traigent.security.auth.sms.TwilioClient")
    def test_init_raises_value_error_invalid_from_number_format(
        self, mock_client: MagicMock
    ) -> None:
        """Test initialization raises ValueError for invalid from_number format."""

        settings = {
            "account_sid": "AC123456789",
            "auth_token": "test_token_12345",
            "from_number": "5551234567",  # Missing +
        }

        with pytest.raises(
            ValueError, match="Invalid from_number format. Must be E.164 format"
        ):
            SMSAuthProvider(settings)

    @patch("traigent.security.auth.sms.TWILIO_AVAILABLE", True)
    @patch("traigent.security.auth.sms.TwilioClient")
    @pytest.mark.parametrize(
        "invalid_number",
        [
            "invalid",
            "+0123456789",  # Starts with 0
            "15551234567",  # Missing +
            "+1 555 123 4567",  # Contains spaces
            "+1-555-123-4567",  # Contains dashes
        ],
    )
    def test_init_invalid_phone_formats(
        self, mock_client: MagicMock, invalid_number: str
    ) -> None:
        """Test initialization rejects various invalid phone number formats."""

        settings = {
            "account_sid": "AC123456789",
            "auth_token": "test_token_12345",
            "from_number": invalid_number,
        }

        with pytest.raises(ValueError, match="Invalid from_number format"):
            SMSAuthProvider(settings)


class TestSendVerificationCode:
    """Tests for send_verification_code method."""

    @pytest.fixture
    def provider(self) -> SMSAuthProvider:
        """Create test provider instance."""

        with patch("traigent.security.auth.sms.TWILIO_AVAILABLE", True):
            with patch("traigent.security.auth.sms.TwilioClient") as mock_client:
                settings = {
                    "account_sid": "AC123456789",
                    "auth_token": "test_token_12345",
                    "from_number": "+15551234567",
                }
                provider = SMSAuthProvider(settings)
                provider.client = mock_client.return_value
                return provider

    def test_send_code_success(self, provider: SMSAuthProvider) -> None:
        """Test successful verification code sending."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        sid = provider.send_verification_code("+15559876543", "user123")

        assert sid == "SM123456789"
        assert "user123" in provider._verification_codes
        assert provider._verification_codes["user123"]["phone_number"] == "+15559876543"
        assert provider._verification_codes["user123"]["attempts"] == 0
        provider.client.messages.create.assert_called_once()

    def test_send_code_generates_six_digit_code(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test that generated code is exactly 6 digits."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        provider.send_verification_code("+15559876543", "user123")

        call_args = provider.client.messages.create.call_args
        message_body = call_args.kwargs["body"]
        # Extract code from message
        assert "verification code is:" in message_body
        assert "expires in" in message_body

    def test_send_code_stores_hash_not_plaintext(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test that code is stored as hash, not plaintext."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        provider.send_verification_code("+15559876543", "user123")

        stored = provider._verification_codes["user123"]
        assert "code_hash" in stored
        assert len(stored["code_hash"]) == 64  # SHA-256 hex digest

    def test_send_code_sets_expiration(self, provider: SMSAuthProvider) -> None:
        """Test that verification code has proper expiration time."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        before = datetime.now(UTC)
        provider.send_verification_code("+15559876543", "user123")
        after = datetime.now(UTC)

        stored = provider._verification_codes["user123"]
        expected_expiry = before + timedelta(minutes=provider.code_expiry_minutes)
        assert stored["expires_at"] >= expected_expiry
        assert stored["expires_at"] <= after + timedelta(
            minutes=provider.code_expiry_minutes
        )

    def test_send_code_raises_for_invalid_phone_number(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test sending code raises AuthenticationError for invalid phone number."""
        with pytest.raises(AuthenticationError, match="Invalid phone number format"):
            provider.send_verification_code("5551234567", "user123")

    @pytest.mark.parametrize(
        "invalid_phone",
        [
            "invalid",
            "+0123456789",
            "15551234567",
            "+1 555 123 4567",
            "",
        ],
    )
    def test_send_code_invalid_phone_formats(
        self, provider: SMSAuthProvider, invalid_phone: str
    ) -> None:
        """Test various invalid phone number formats are rejected."""
        with pytest.raises(AuthenticationError, match="Invalid phone number format"):
            provider.send_verification_code(invalid_phone, "user123")

    @pytest.mark.parametrize(
        "invalid_user_id",
        [
            "",
            None,
            "a" * 256,  # Too long
        ],
    )
    def test_send_code_raises_for_invalid_user_id(
        self, provider: SMSAuthProvider, invalid_user_id: str | None
    ) -> None:
        """Test sending code raises AuthenticationError for invalid user_id."""
        with pytest.raises(AuthenticationError, match="Invalid user_id"):
            provider.send_verification_code("+15559876543", invalid_user_id)  # type: ignore

    def test_send_code_rate_limit_per_minute(self, provider: SMSAuthProvider) -> None:
        """Test rate limiting prevents sending codes too frequently."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        # First send should succeed
        provider.send_verification_code("+15559876543", "user123")

        # Second send immediately should fail
        with pytest.raises(
            AuthenticationError,
            match=r"Please wait \d+ minute\(s\) before requesting a new code",
        ):
            provider.send_verification_code("+15559876543", "user123")

    def test_send_code_rate_limit_per_hour(self, provider: SMSAuthProvider) -> None:
        """Test hourly rate limiting prevents excessive code requests."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        # Send max_codes_per_hour codes with time manipulation
        with patch("traigent.security.auth.sms.datetime") as mock_datetime:
            base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = base_time

            for i in range(provider.max_codes_per_hour):
                # Advance time to bypass per-minute rate limit
                mock_datetime.now.return_value = base_time + timedelta(minutes=i * 2)
                provider.send_verification_code("+15559876543", "user123")

            # Next attempt should fail
            mock_datetime.now.return_value = base_time + timedelta(
                minutes=provider.max_codes_per_hour * 2
            )
            with pytest.raises(
                AuthenticationError, match="Too many verification codes requested"
            ):
                provider.send_verification_code("+15559876543", "user123")

    def test_send_code_cleans_old_rate_limit_entries(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test that old rate limit entries are cleaned up."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        with patch("traigent.security.auth.sms.datetime") as mock_datetime:
            base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = base_time

            # Send first code
            provider.send_verification_code("+15559876543", "user123")

            # Advance time by over an hour
            mock_datetime.now.return_value = base_time + timedelta(hours=2)

            # Should succeed because old entry is cleaned
            provider.send_verification_code("+15559876543", "user123")

    def test_send_code_twilio_exception_wrapped(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test that Twilio exceptions are wrapped as AuthenticationError."""
        provider.client.messages.create.side_effect = Exception("Twilio API error")

        with pytest.raises(AuthenticationError, match="SMS sending failed"):
            provider.send_verification_code("+15559876543", "user123")

    def test_send_code_sanitizes_auth_token_in_error(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test that auth token is redacted from error messages."""
        provider.client.messages.create.side_effect = Exception(
            f"Auth failed: {provider.auth_token}"
        )

        with pytest.raises(AuthenticationError) as exc_info:
            provider.send_verification_code("+15559876543", "user123")

        assert provider.auth_token not in str(exc_info.value)
        assert "***" in str(exc_info.value)

    def test_send_code_updates_rate_limit_tracking(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test that rate limit tracking is updated correctly."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        assert "user123" not in provider._rate_limit

        provider.send_verification_code("+15559876543", "user123")

        assert "user123" in provider._rate_limit
        assert len(provider._rate_limit["user123"]) == 1


class TestVerifySMSCode:
    """Tests for verify_sms_code method."""

    @pytest.fixture
    def provider(self) -> SMSAuthProvider:
        """Create test provider instance."""

        with patch("traigent.security.auth.sms.TWILIO_AVAILABLE", True):
            with patch("traigent.security.auth.sms.TwilioClient") as mock_client:
                settings = {
                    "account_sid": "AC123456789",
                    "auth_token": "test_token_12345",
                    "from_number": "+15551234567",
                }
                provider = SMSAuthProvider(settings)
                provider.client = mock_client.return_value
                return provider

    def test_verify_code_success(self, provider: SMSAuthProvider) -> None:
        """Test successful code verification."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        # Send code first
        provider.send_verification_code("+15559876543", "user123")

        # Extract the actual code from the message sent
        call_args = provider.client.messages.create.call_args
        message_body = call_args.kwargs["body"]
        # Parse code from "Your Traigent verification code is: 123456\n\n..."
        code = message_body.split("code is: ")[1].split("\n")[0]

        result = provider.verify_sms_code("user123", code)

        assert result is True
        assert (
            "user123" not in provider._verification_codes
        )  # Should be deleted on success
        assert "user123" not in provider._rate_limit  # Should be cleared on success

    def test_verify_code_invalid_user_id(self, provider: SMSAuthProvider) -> None:
        """Test verification fails for invalid user_id."""
        assert provider.verify_sms_code("", "123456") is False
        assert provider.verify_sms_code(None, "123456") is False  # type: ignore

    def test_verify_code_invalid_format(self, provider: SMSAuthProvider) -> None:
        """Test verification fails for invalid code format."""
        assert provider.verify_sms_code("user123", "") is False
        assert provider.verify_sms_code("user123", "12345") is False  # Too short
        assert provider.verify_sms_code("user123", "1234567") is False  # Too long
        assert provider.verify_sms_code("user123", "abcdef") is False  # Not digits
        assert provider.verify_sms_code("user123", "12345a") is False  # Mixed

    def test_verify_code_no_pending_verification(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test verification fails when no code exists for user."""
        with patch("traigent.security.auth.sms.time.sleep"):
            result = provider.verify_sms_code("user123", "123456")

        assert result is False

    def test_verify_code_expired(self, provider: SMSAuthProvider) -> None:
        """Test verification fails for expired code."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        with patch("traigent.security.auth.sms.datetime") as mock_datetime:
            base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = base_time

            provider.send_verification_code("+15559876543", "user123")

            # Advance time past expiry
            mock_datetime.now.return_value = base_time + timedelta(
                minutes=provider.code_expiry_minutes + 1
            )

            with patch("traigent.security.auth.sms.time.sleep"):
                result = provider.verify_sms_code("user123", "123456")

            assert result is False
            assert "user123" not in provider._verification_codes  # Should be deleted

    def test_verify_code_wrong_code(self, provider: SMSAuthProvider) -> None:
        """Test verification fails for incorrect code."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        provider.send_verification_code("+15559876543", "user123")

        with patch("traigent.security.auth.sms.time.sleep"):
            result = provider.verify_sms_code("user123", "999999")

        assert result is False
        assert "user123" in provider._verification_codes  # Should still exist for retry

    def test_verify_code_max_attempts_exceeded(self, provider: SMSAuthProvider) -> None:
        """Test verification fails after max attempts exceeded."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        provider.send_verification_code("+15559876543", "user123")

        with patch("traigent.security.auth.sms.time.sleep"):
            # Exceed max attempts
            for _ in range(provider.max_attempts + 1):
                result = provider.verify_sms_code("user123", "999999")

            assert result is False
            assert "user123" not in provider._verification_codes  # Should be deleted

    def test_verify_code_increments_attempts(self, provider: SMSAuthProvider) -> None:
        """Test that verification increments attempt counter."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        provider.send_verification_code("+15559876543", "user123")

        with patch("traigent.security.auth.sms.time.sleep"):
            provider.verify_sms_code("user123", "999999")

        assert provider._verification_codes["user123"]["attempts"] == 1

        with patch("traigent.security.auth.sms.time.sleep"):
            provider.verify_sms_code("user123", "999999")

        assert provider._verification_codes["user123"]["attempts"] == 2

    def test_verify_code_timing_attack_delay_on_invalid(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test that timing attack delay is applied on invalid verification."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        provider.send_verification_code("+15559876543", "user123")

        with patch("traigent.security.auth.sms.time.sleep") as mock_sleep:
            provider.verify_sms_code("user123", "999999")
            mock_sleep.assert_called()

    def test_verify_code_timing_attack_delay_increases_with_attempts(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test timing attack delay increases with attempt count."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        provider.send_verification_code("+15559876543", "user123")

        with patch("traigent.security.auth.sms.time.sleep") as mock_sleep:
            from traigent.security.auth.helpers import TIMING_ATTACK_DELAY_SECONDS

            # First failed attempt
            provider.verify_sms_code("user123", "999999")
            first_delay = mock_sleep.call_args[0][0]

            # Second failed attempt
            provider.verify_sms_code("user123", "999999")
            second_delay = mock_sleep.call_args[0][0]

            assert second_delay > first_delay
            assert second_delay == TIMING_ATTACK_DELAY_SECONDS * 2

    def test_verify_code_no_delay_on_success(self, provider: SMSAuthProvider) -> None:
        """Test no timing delay on successful verification."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        provider.send_verification_code("+15559876543", "user123")

        # Extract the actual code
        call_args = provider.client.messages.create.call_args
        message_body = call_args.kwargs["body"]
        code = message_body.split("code is: ")[1].split("\n")[0]

        with patch("traigent.security.auth.sms.time.sleep") as mock_sleep:
            result = provider.verify_sms_code("user123", code)

        assert result is True
        mock_sleep.assert_not_called()


class TestSendNotification:
    """Tests for send_notification method."""

    @pytest.fixture
    def provider(self) -> SMSAuthProvider:
        """Create test provider instance."""

        with patch("traigent.security.auth.sms.TWILIO_AVAILABLE", True):
            with patch("traigent.security.auth.sms.TwilioClient") as mock_client:
                settings = {
                    "account_sid": "AC123456789",
                    "auth_token": "test_token_12345",
                    "from_number": "+15551234567",
                }
                provider = SMSAuthProvider(settings)
                provider.client = mock_client.return_value
                return provider

    def test_send_notification_success(self, provider: SMSAuthProvider) -> None:
        """Test successful notification sending."""
        mock_message = MagicMock()
        mock_message.sid = "SM987654321"
        provider.client.messages.create.return_value = mock_message

        sid = provider.send_notification("+15559876543", "Test notification message")

        assert sid == "SM987654321"
        provider.client.messages.create.assert_called_once_with(
            body="Test notification message",
            from_="+15551234567",
            to="+15559876543",
        )

    def test_send_notification_twilio_exception_wrapped(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test that Twilio exceptions are wrapped as AuthenticationError."""
        provider.client.messages.create.side_effect = Exception("Twilio API error")

        with pytest.raises(AuthenticationError, match="SMS notification failed"):
            provider.send_notification("+15559876543", "Test message")

    def test_send_notification_uses_correct_from_number(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test notification uses configured from_number."""
        mock_message = MagicMock()
        mock_message.sid = "SM987654321"
        provider.client.messages.create.return_value = mock_message

        provider.send_notification("+15559876543", "Test message")

        call_args = provider.client.messages.create.call_args
        assert call_args.kwargs["from_"] == "+15551234567"

    def test_send_notification_preserves_message_content(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test notification preserves exact message content."""
        mock_message = MagicMock()
        mock_message.sid = "SM987654321"
        provider.client.messages.create.return_value = mock_message

        test_message = "This is a test notification with special chars: !@#$%"
        provider.send_notification("+15559876543", test_message)

        call_args = provider.client.messages.create.call_args
        assert call_args.kwargs["body"] == test_message


class TestEdgeCasesAndIntegration:
    """Tests for edge cases and integration scenarios."""

    @pytest.fixture
    def provider(self) -> SMSAuthProvider:
        """Create test provider instance."""

        with patch("traigent.security.auth.sms.TWILIO_AVAILABLE", True):
            with patch("traigent.security.auth.sms.TwilioClient") as mock_client:
                settings = {
                    "account_sid": "AC123456789",
                    "auth_token": "test_token_12345",
                    "from_number": "+15551234567",
                    "max_attempts": 3,
                }
                provider = SMSAuthProvider(settings)
                provider.client = mock_client.return_value
                return provider

    def test_multiple_users_independent_codes(self, provider: SMSAuthProvider) -> None:
        """Test that multiple users can have independent verification codes."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        provider.send_verification_code("+15551111111", "user1")
        provider.send_verification_code("+15552222222", "user2")

        assert "user1" in provider._verification_codes
        assert "user2" in provider._verification_codes
        assert (
            provider._verification_codes["user1"]["phone_number"]
            != provider._verification_codes["user2"]["phone_number"]
        )

    def test_code_regeneration_replaces_old_code(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test that sending a new code replaces the old one for the same user."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        with patch("traigent.security.auth.sms.datetime") as mock_datetime:
            base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = base_time

            provider.send_verification_code("+15559876543", "user123")
            first_hash = provider._verification_codes["user123"]["code_hash"]

            # Advance time to bypass rate limit
            mock_datetime.now.return_value = base_time + timedelta(minutes=2)

            provider.send_verification_code("+15559876543", "user123")
            second_hash = provider._verification_codes["user123"]["code_hash"]

            # Codes should be different (very high probability)
            assert first_hash != second_hash

    def test_verification_clears_rate_limit_on_success(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test successful verification clears rate limit for user."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        provider.send_verification_code("+15559876543", "user123")
        assert "user123" in provider._rate_limit

        # Extract and verify code
        call_args = provider.client.messages.create.call_args
        message_body = call_args.kwargs["body"]
        code = message_body.split("code is: ")[1].split("\n")[0]

        provider.verify_sms_code("user123", code)

        assert "user123" not in provider._rate_limit

    def test_hmac_compare_digest_used_for_security(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test that HMAC comparison is timing-safe."""
        # This test verifies the code uses hmac.compare_digest
        # We can't directly test timing, but we verify it's called
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        provider.send_verification_code("+15559876543", "user123")

        with patch("traigent.security.auth.sms.hmac.compare_digest") as mock_compare:
            mock_compare.return_value = True
            provider.verify_sms_code("user123", "123456")
            mock_compare.assert_called_once()

    def test_code_format_always_six_digits(self, provider: SMSAuthProvider) -> None:
        """Test that generated codes are always exactly 6 digits."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        with patch("traigent.security.auth.sms.secrets.randbelow") as mock_random:
            # Test minimum value (100000)
            mock_random.return_value = 0
            provider.send_verification_code("+15559876543", "user1")
            call_args = provider.client.messages.create.call_args
            message_body = call_args.kwargs["body"]
            code = message_body.split("code is: ")[1].split("\n")[0]
            assert len(code) == 6
            assert code == "100000"

            # Test maximum value (999999)
            mock_random.return_value = 899999
            provider.send_verification_code("+15559876543", "user2")
            call_args = provider.client.messages.create.call_args
            message_body = call_args.kwargs["body"]
            code = message_body.split("code is: ")[1].split("\n")[0]
            assert len(code) == 6
            assert code == "999999"

    def test_phone_number_validation_e164_compliance(
        self, provider: SMSAuthProvider
    ) -> None:
        """Test phone number validation enforces E.164 format."""
        mock_message = MagicMock()
        mock_message.sid = "SM123456789"
        provider.client.messages.create.return_value = mock_message

        # Valid E.164 numbers
        valid_numbers = [
            "+15551234567",  # US
            "+442071234567",  # UK
            "+81312345678",  # Japan
            "+861012345678",  # China
        ]

        for number in valid_numbers:
            provider.send_verification_code(number, f"user_{number}")

        # Verify all succeeded
        assert len(provider._verification_codes) == len(valid_numbers)
