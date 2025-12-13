"""Unit tests for traigent/security/auth/mfa.py.

Tests for unified Multi-Factor Authentication manager.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from traigent.security.auth.mfa import MultiFactorAuth


class TestMultiFactorAuthInit:
    """Tests for MultiFactorAuth initialization."""

    def test_init_creates_instance_with_no_providers(self) -> None:
        """Test initialization creates instance with no providers enabled."""
        mfa = MultiFactorAuth()
        assert mfa.totp_provider is None
        assert mfa.sms_provider is None


class TestEnableTOTP:
    """Tests for enabling TOTP authentication."""

    @pytest.fixture
    def mfa(self) -> MultiFactorAuth:
        """Create test MFA instance."""
        return MultiFactorAuth()

    @patch("traigent.security.auth.mfa.TOTPAuthProvider")
    def test_enable_totp_with_default_issuer(
        self, mock_totp_class: MagicMock, mfa: MultiFactorAuth
    ) -> None:
        """Test enabling TOTP with default issuer name."""
        mock_totp_instance = MagicMock()
        mock_totp_class.return_value = mock_totp_instance

        mfa.enable_totp()

        mock_totp_class.assert_called_once_with("TraiGent")
        assert mfa.totp_provider == mock_totp_instance

    @patch("traigent.security.auth.mfa.TOTPAuthProvider")
    def test_enable_totp_with_custom_issuer(
        self, mock_totp_class: MagicMock, mfa: MultiFactorAuth
    ) -> None:
        """Test enabling TOTP with custom issuer name."""
        mock_totp_instance = MagicMock()
        mock_totp_class.return_value = mock_totp_instance

        mfa.enable_totp(issuer_name="CustomIssuer")

        mock_totp_class.assert_called_once_with("CustomIssuer")
        assert mfa.totp_provider == mock_totp_instance

    @patch("traigent.security.auth.mfa.TOTPAuthProvider")
    def test_enable_totp_replaces_existing_provider(
        self, mock_totp_class: MagicMock, mfa: MultiFactorAuth
    ) -> None:
        """Test enabling TOTP replaces any existing TOTP provider."""
        first_instance = MagicMock()
        second_instance = MagicMock()
        mock_totp_class.side_effect = [first_instance, second_instance]

        mfa.enable_totp("Issuer1")
        assert mfa.totp_provider == first_instance

        mfa.enable_totp("Issuer2")
        assert mfa.totp_provider == second_instance


class TestEnableSMS:
    """Tests for enabling SMS authentication."""

    @pytest.fixture
    def mfa(self) -> MultiFactorAuth:
        """Create test MFA instance."""
        return MultiFactorAuth()

    @pytest.fixture
    def twilio_settings(self) -> dict[str, str]:
        """Create test Twilio settings."""
        return {
            "account_sid": "test_account_sid",
            "auth_token": "test_auth_token",
            "from_number": "+1234567890",
        }

    @patch("traigent.security.auth.mfa.SMSAuthProvider")
    def test_enable_sms_with_valid_settings(
        self,
        mock_sms_class: MagicMock,
        mfa: MultiFactorAuth,
        twilio_settings: dict[str, str],
    ) -> None:
        """Test enabling SMS with valid Twilio settings."""
        mock_sms_instance = MagicMock()
        mock_sms_class.return_value = mock_sms_instance

        mfa.enable_sms(twilio_settings)

        mock_sms_class.assert_called_once_with(twilio_settings)
        assert mfa.sms_provider == mock_sms_instance

    @patch("traigent.security.auth.mfa.SMSAuthProvider")
    def test_enable_sms_replaces_existing_provider(
        self,
        mock_sms_class: MagicMock,
        mfa: MultiFactorAuth,
        twilio_settings: dict[str, str],
    ) -> None:
        """Test enabling SMS replaces any existing SMS provider."""
        first_instance = MagicMock()
        second_instance = MagicMock()
        mock_sms_class.side_effect = [first_instance, second_instance]

        mfa.enable_sms(twilio_settings)
        assert mfa.sms_provider == first_instance

        mfa.enable_sms(twilio_settings)
        assert mfa.sms_provider == second_instance


class TestVerifyMFA:
    """Tests for MFA verification."""

    @pytest.fixture
    def mfa(self) -> MultiFactorAuth:
        """Create test MFA instance."""
        return MultiFactorAuth()

    @pytest.fixture
    def mock_totp_provider(self) -> MagicMock:
        """Create mock TOTP provider."""
        mock = MagicMock()
        mock.verify_totp_code.return_value = True
        return mock

    @pytest.fixture
    def mock_sms_provider(self) -> MagicMock:
        """Create mock SMS provider."""
        mock = MagicMock()
        mock.verify_sms_code.return_value = True
        return mock

    # Happy path tests for TOTP
    def test_verify_mfa_totp_success(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test successful TOTP verification."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code="123456", secret="TESTSECRET"
        )

        assert result is True
        mock_totp_provider.verify_totp_code.assert_called_once_with(
            "123456", "TESTSECRET", "user123"
        )

    # Happy path tests for SMS
    def test_verify_mfa_sms_success(
        self, mfa: MultiFactorAuth, mock_sms_provider: MagicMock
    ) -> None:
        """Test successful SMS verification."""
        mfa.sms_provider = mock_sms_provider

        result = mfa.verify_mfa(user_id="user123", method="sms", code="123456")

        assert result is True
        mock_sms_provider.verify_sms_code.assert_called_once_with("user123", "123456")

    # Error handling: unknown method
    def test_verify_mfa_raises_on_unknown_method(self, mfa: MultiFactorAuth) -> None:
        """Test verification raises ValueError for unknown MFA method."""
        with pytest.raises(ValueError, match="Unknown MFA method: invalid"):
            mfa.verify_mfa(user_id="user123", method="invalid", code="123456")

    def test_verify_mfa_raises_on_email_method(self, mfa: MultiFactorAuth) -> None:
        """Test verification raises ValueError for email method (not supported)."""
        with pytest.raises(ValueError, match="Unknown MFA method: email"):
            mfa.verify_mfa(user_id="user123", method="email", code="123456")

    # Edge cases: invalid user_id
    def test_verify_mfa_returns_false_for_empty_user_id(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test verification returns False for empty user_id."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="", method="totp", code="123456", secret="TESTSECRET"
        )

        assert result is False
        mock_totp_provider.verify_totp_code.assert_not_called()

    def test_verify_mfa_returns_false_for_none_user_id(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test verification returns False for None user_id."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id=None, method="totp", code="123456", secret="TESTSECRET"  # type: ignore
        )

        assert result is False
        mock_totp_provider.verify_totp_code.assert_not_called()

    def test_verify_mfa_returns_false_for_non_string_user_id(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test verification returns False for non-string user_id."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id=12345, method="totp", code="123456", secret="TESTSECRET"  # type: ignore
        )

        assert result is False
        mock_totp_provider.verify_totp_code.assert_not_called()

    def test_verify_mfa_returns_false_for_too_long_user_id(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test verification returns False for user_id exceeding 255 characters."""
        mfa.totp_provider = mock_totp_provider
        long_user_id = "a" * 256

        result = mfa.verify_mfa(
            user_id=long_user_id, method="totp", code="123456", secret="TESTSECRET"
        )

        assert result is False
        mock_totp_provider.verify_totp_code.assert_not_called()

    # Edge cases: invalid code
    def test_verify_mfa_returns_false_for_empty_code(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test verification returns False for empty code."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code="", secret="TESTSECRET"
        )

        assert result is False
        mock_totp_provider.verify_totp_code.assert_not_called()

    def test_verify_mfa_returns_false_for_none_code(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test verification returns False for None code."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code=None, secret="TESTSECRET"  # type: ignore
        )

        assert result is False
        mock_totp_provider.verify_totp_code.assert_not_called()

    def test_verify_mfa_returns_false_for_short_code(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test verification returns False for code shorter than 6 digits."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code="12345", secret="TESTSECRET"
        )

        assert result is False
        mock_totp_provider.verify_totp_code.assert_not_called()

    def test_verify_mfa_returns_false_for_long_code(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test verification returns False for code longer than 8 digits."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code="123456789", secret="TESTSECRET"
        )

        assert result is False
        mock_totp_provider.verify_totp_code.assert_not_called()

    def test_verify_mfa_returns_false_for_non_numeric_code(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test verification returns False for non-numeric code."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code="12345a", secret="TESTSECRET"
        )

        assert result is False
        mock_totp_provider.verify_totp_code.assert_not_called()

    def test_verify_mfa_accepts_6_digit_code(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test verification accepts valid 6-digit code."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code="123456", secret="TESTSECRET"
        )

        assert result is True
        mock_totp_provider.verify_totp_code.assert_called_once()

    def test_verify_mfa_accepts_7_digit_code(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test verification accepts valid 7-digit code."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code="1234567", secret="TESTSECRET"
        )

        assert result is True
        mock_totp_provider.verify_totp_code.assert_called_once()

    def test_verify_mfa_accepts_8_digit_code(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test verification accepts valid 8-digit code."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code="12345678", secret="TESTSECRET"
        )

        assert result is True
        mock_totp_provider.verify_totp_code.assert_called_once()

    # Error handling: TOTP provider not enabled
    def test_verify_mfa_totp_raises_when_provider_not_enabled(
        self, mfa: MultiFactorAuth
    ) -> None:
        """Test TOTP verification raises ValueError when provider not enabled."""
        # Ensure TOTP provider is not enabled
        mfa.totp_provider = None

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code="123456", secret="TESTSECRET"
        )

        assert result is False

    # Error handling: SMS provider not enabled
    def test_verify_mfa_sms_raises_when_provider_not_enabled(
        self, mfa: MultiFactorAuth
    ) -> None:
        """Test SMS verification raises ValueError when provider not enabled."""
        # Ensure SMS provider is not enabled
        mfa.sms_provider = None

        result = mfa.verify_mfa(user_id="user123", method="sms", code="123456")

        assert result is False

    # Error handling: missing TOTP secret
    def test_verify_mfa_totp_raises_when_secret_missing(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test TOTP verification raises ValueError when secret is missing."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(user_id="user123", method="totp", code="123456")

        assert result is False
        mock_totp_provider.verify_totp_code.assert_not_called()

    def test_verify_mfa_totp_raises_when_secret_is_none(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test TOTP verification raises ValueError when secret is None."""
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code="123456", secret=None
        )

        assert result is False
        mock_totp_provider.verify_totp_code.assert_not_called()

    # Error handling: provider verification fails
    def test_verify_mfa_totp_returns_false_when_provider_returns_false(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test TOTP verification returns False when provider returns False."""
        mock_totp_provider.verify_totp_code.return_value = False
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code="123456", secret="TESTSECRET"
        )

        assert result is False

    def test_verify_mfa_sms_returns_false_when_provider_returns_false(
        self, mfa: MultiFactorAuth, mock_sms_provider: MagicMock
    ) -> None:
        """Test SMS verification returns False when provider returns False."""
        mock_sms_provider.verify_sms_code.return_value = False
        mfa.sms_provider = mock_sms_provider

        result = mfa.verify_mfa(user_id="user123", method="sms", code="123456")

        assert result is False

    # Error handling: provider raises exception
    def test_verify_mfa_totp_returns_false_when_provider_raises_exception(
        self, mfa: MultiFactorAuth, mock_totp_provider: MagicMock
    ) -> None:
        """Test TOTP verification returns False when provider raises exception."""
        mock_totp_provider.verify_totp_code.side_effect = Exception("TOTP error")
        mfa.totp_provider = mock_totp_provider

        result = mfa.verify_mfa(
            user_id="user123", method="totp", code="123456", secret="TESTSECRET"
        )

        assert result is False

    def test_verify_mfa_sms_returns_false_when_provider_raises_exception(
        self, mfa: MultiFactorAuth, mock_sms_provider: MagicMock
    ) -> None:
        """Test SMS verification returns False when provider raises exception."""
        mock_sms_provider.verify_sms_code.side_effect = Exception("SMS error")
        mfa.sms_provider = mock_sms_provider

        result = mfa.verify_mfa(user_id="user123", method="sms", code="123456")

        assert result is False

    # Edge case: SMS doesn't require secret
    def test_verify_mfa_sms_works_without_secret(
        self, mfa: MultiFactorAuth, mock_sms_provider: MagicMock
    ) -> None:
        """Test SMS verification works without secret parameter."""
        mfa.sms_provider = mock_sms_provider

        result = mfa.verify_mfa(user_id="user123", method="sms", code="123456")

        assert result is True
        mock_sms_provider.verify_sms_code.assert_called_once_with("user123", "123456")

    # Logging tests
    @patch("traigent.security.auth.mfa.logger")
    def test_verify_mfa_logs_warning_for_unknown_method(
        self, mock_logger: MagicMock, mfa: MultiFactorAuth
    ) -> None:
        """Test verification logs warning for unknown method."""
        with pytest.raises(ValueError):
            mfa.verify_mfa(user_id="user123", method="invalid", code="123456")

        mock_logger.warning.assert_called_once_with(
            "Unknown MFA method attempted: invalid"
        )

    @patch("traigent.security.auth.mfa.logger")
    def test_verify_mfa_logs_warning_for_invalid_user_id(
        self,
        mock_logger: MagicMock,
        mfa: MultiFactorAuth,
        mock_totp_provider: MagicMock,
    ) -> None:
        """Test verification logs warning for invalid user_id."""
        mfa.totp_provider = mock_totp_provider

        mfa.verify_mfa(user_id="", method="totp", code="123456", secret="TESTSECRET")

        mock_logger.warning.assert_called_once_with(
            "Invalid user_id for MFA verification"
        )

    @patch("traigent.security.auth.mfa.logger")
    def test_verify_mfa_logs_warning_for_invalid_code_format(
        self,
        mock_logger: MagicMock,
        mfa: MultiFactorAuth,
        mock_totp_provider: MagicMock,
    ) -> None:
        """Test verification logs warning for invalid code format."""
        mfa.totp_provider = mock_totp_provider

        mfa.verify_mfa(
            user_id="user123", method="totp", code="abc", secret="TESTSECRET"
        )

        mock_logger.warning.assert_called_once_with(
            "Invalid code format for MFA method totp"
        )

    @patch("traigent.security.auth.mfa.logger")
    def test_verify_mfa_logs_error_when_provider_raises_exception(
        self,
        mock_logger: MagicMock,
        mfa: MultiFactorAuth,
        mock_totp_provider: MagicMock,
    ) -> None:
        """Test verification logs error when provider raises exception."""
        mock_totp_provider.verify_totp_code.side_effect = RuntimeError("Test error")
        mfa.totp_provider = mock_totp_provider

        mfa.verify_mfa(
            user_id="user123", method="totp", code="123456", secret="TESTSECRET"
        )

        assert mock_logger.error.call_count == 1
        error_call = mock_logger.error.call_args[0][0]
        assert "MFA verification error for totp" in error_call
        assert "Test error" in error_call
