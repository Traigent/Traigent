"""Unit tests for traigent/security/auth/totp.py.

Tests for Time-based One-Time Password (TOTP) authentication provider.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import base64
import urllib.parse
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from traigent.security.auth.totp import PYOTP_AVAILABLE, TOTPAuthProvider


class TestTOTPAuthProviderInit:
    """Tests for TOTPAuthProvider initialization."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        if not PYOTP_AVAILABLE:
            pytest.skip("pyotp not available")

        provider = TOTPAuthProvider()
        assert provider.issuer_name == "Traigent"
        assert provider.secret_length == 32
        assert provider._used_codes == {}
        assert provider._code_expiry == {}

    def test_init_custom_issuer(self) -> None:
        """Test initialization with custom issuer name."""
        if not PYOTP_AVAILABLE:
            pytest.skip("pyotp not available")

        provider = TOTPAuthProvider(issuer_name="CustomIssuer")
        assert provider.issuer_name == "CustomIssuer"

    def test_init_custom_secret_length(self) -> None:
        """Test initialization with custom secret length."""
        if not PYOTP_AVAILABLE:
            pytest.skip("pyotp not available")

        provider = TOTPAuthProvider(secret_length=16)
        assert provider.secret_length == 16

    def test_init_secret_length_clamped_minimum(self) -> None:
        """Test secret length is clamped to minimum of 16."""
        if not PYOTP_AVAILABLE:
            pytest.skip("pyotp not available")

        provider = TOTPAuthProvider(secret_length=8)
        assert provider.secret_length == 16

    def test_init_secret_length_clamped_maximum(self) -> None:
        """Test secret length is clamped to maximum of 64."""
        if not PYOTP_AVAILABLE:
            pytest.skip("pyotp not available")

        provider = TOTPAuthProvider(secret_length=128)
        assert provider.secret_length == 64

    def test_init_issuer_name_sanitized(self) -> None:
        """Test issuer name is sanitized with max length 50."""
        if not PYOTP_AVAILABLE:
            pytest.skip("pyotp not available")

        long_name = "A" * 100
        provider = TOTPAuthProvider(issuer_name=long_name)
        assert len(provider.issuer_name) == 50

    @patch("traigent.security.auth.totp.PYOTP_AVAILABLE", False)
    def test_init_raises_import_error_when_pyotp_unavailable(self) -> None:
        """Test initialization raises ImportError when pyotp is not available."""
        with pytest.raises(
            ImportError, match="pyotp is required for TOTP authentication"
        ):
            TOTPAuthProvider()


class TestGenerateTOTPSecret:
    """Tests for TOTP secret generation."""

    @pytest.fixture
    def provider(self) -> TOTPAuthProvider:
        """Create test provider instance."""
        if not PYOTP_AVAILABLE:
            pytest.skip("pyotp not available")
        return TOTPAuthProvider()

    def test_generate_secret_returns_base32_string(
        self, provider: TOTPAuthProvider
    ) -> None:
        """Test generated secret is a valid base32 string."""
        secret = provider.generate_totp_secret()
        assert isinstance(secret, str)
        assert len(secret) > 0
        # Base32 alphabet: A-Z and 2-7
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567=" for c in secret)

    def test_generate_secret_correct_length(self, provider: TOTPAuthProvider) -> None:
        """Test generated secret has correct length based on secret_length."""
        secret = provider.generate_totp_secret()
        # Base32 encoding: 8 chars for every 5 bytes
        expected_min_length = (provider.secret_length * 8) // 5
        assert len(secret) >= expected_min_length

    def test_generate_secret_unique_values(self, provider: TOTPAuthProvider) -> None:
        """Test each generated secret is unique."""
        secrets = [provider.generate_totp_secret() for _ in range(10)]
        assert len(set(secrets)) == 10

    def test_generate_secret_with_custom_length(self) -> None:
        """Test secret generation with custom length."""
        if not PYOTP_AVAILABLE:
            pytest.skip("pyotp not available")

        provider = TOTPAuthProvider(secret_length=16)
        secret = provider.generate_totp_secret()
        # For 16 bytes, we expect at least (16 * 8) // 5 = 25 base32 chars
        assert len(secret) >= 25


class TestGenerateProvisioningURI:
    """Tests for provisioning URI generation."""

    @pytest.fixture
    def provider(self) -> TOTPAuthProvider:
        """Create test provider instance."""
        if not PYOTP_AVAILABLE:
            pytest.skip("pyotp not available")
        return TOTPAuthProvider(issuer_name="TestIssuer")

    @pytest.fixture
    def valid_secret(self) -> str:
        """Create a valid base32 secret for testing."""
        return base64.b32encode(b"test_secret_12345").decode("utf-8")

    def test_generate_provisioning_uri_format(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test provisioning URI has correct format."""
        uri = provider.generate_provisioning_uri("testuser", valid_secret)
        assert uri.startswith("otpauth://totp/")
        assert "TestIssuer" in uri
        assert "testuser" in uri
        # Secret may be URL-encoded in the URI
        assert valid_secret in uri or urllib.parse.quote(valid_secret, safe="") in uri

    def test_generate_provisioning_uri_contains_issuer(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test provisioning URI includes issuer name."""
        uri = provider.generate_provisioning_uri("user@example.com", valid_secret)
        assert "issuer=TestIssuer" in uri

    def test_generate_provisioning_uri_contains_username(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test provisioning URI includes username."""
        uri = provider.generate_provisioning_uri("alice", valid_secret)
        assert "alice" in uri


class TestVerifyTOTPCode:
    """Tests for TOTP code verification."""

    @pytest.fixture
    def provider(self) -> TOTPAuthProvider:
        """Create test provider instance."""
        if not PYOTP_AVAILABLE:
            pytest.skip("pyotp not available")
        return TOTPAuthProvider()

    @pytest.fixture
    def valid_secret(self) -> str:
        """Create a valid base32 secret for testing."""
        return base64.b32encode(b"test_secret_12345").decode("utf-8")

    @pytest.fixture
    def mock_totp(self) -> MagicMock:
        """Create mock TOTP instance."""
        mock = MagicMock()
        mock.verify.return_value = True
        return mock

    def test_verify_valid_code(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test verification succeeds with valid TOTP code."""
        with patch("pyotp.TOTP") as mock_totp_class:
            mock_totp = MagicMock()
            mock_totp.verify.return_value = True
            mock_totp_class.return_value = mock_totp

            result = provider.verify_totp_code("123456", valid_secret, "user1")
            assert result is True
            mock_totp.verify.assert_called_once_with("123456", valid_window=1)

    def test_verify_invalid_code_format_too_short(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test verification fails with code too short."""
        result = provider.verify_totp_code("12345", valid_secret, "user1")
        assert result is False

    def test_verify_invalid_code_format_non_numeric(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test verification fails with non-numeric code."""
        result = provider.verify_totp_code("12345a", valid_secret, "user1")
        assert result is False

    def test_verify_empty_code(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test verification fails with empty code."""
        result = provider.verify_totp_code("", valid_secret, "user1")
        assert result is False

    def test_verify_invalid_secret_format(self, provider: TOTPAuthProvider) -> None:
        """Test verification fails with invalid secret format."""
        result = provider.verify_totp_code("123456", "invalid_secret!", "user1")
        assert result is False

    def test_verify_empty_secret(self, provider: TOTPAuthProvider) -> None:
        """Test verification fails with empty secret."""
        result = provider.verify_totp_code("123456", "", "user1")
        assert result is False

    def test_verify_code_replay_prevention(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test code cannot be reused (replay prevention)."""
        with patch("pyotp.TOTP") as mock_totp_class:
            mock_totp = MagicMock()
            mock_totp.verify.return_value = True
            mock_totp_class.return_value = mock_totp

            # First use should succeed
            result1 = provider.verify_totp_code("123456", valid_secret, "user1")
            assert result1 is True

            # Second use of same code should fail
            result2 = provider.verify_totp_code("123456", valid_secret, "user1")
            assert result2 is False

    def test_verify_expired_codes_cleared(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test expired codes are cleared from used codes cache."""
        with patch("pyotp.TOTP") as mock_totp_class:
            mock_totp = MagicMock()
            mock_totp.verify.return_value = True
            mock_totp_class.return_value = mock_totp

            # Use first code
            provider.verify_totp_code("123456", valid_secret, "user1")

            # Set expiry to past
            provider._code_expiry["user1"] = datetime.now(UTC) - timedelta(seconds=1)

            # Mock time to be after expiry
            with patch("traigent.security.auth.totp.datetime") as mock_dt:
                mock_dt.now.return_value = datetime.now(UTC)
                # This should clear the used codes
                result = provider.verify_totp_code("789012", valid_secret, "user1")
                assert result is True
                # Old code should now be cleared
                assert "123456" not in provider._used_codes.get("user1", set())

    def test_verify_window_parameter(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test verification uses custom window parameter."""
        with patch("pyotp.TOTP") as mock_totp_class:
            mock_totp = MagicMock()
            mock_totp.verify.return_value = True
            mock_totp_class.return_value = mock_totp

            provider.verify_totp_code("123456", valid_secret, "user1", window=2)
            # Window should be clamped to max of 2
            mock_totp.verify.assert_called_once_with("123456", valid_window=2)

    def test_verify_window_clamped_to_maximum(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test window parameter is clamped to maximum of 2."""
        with patch("pyotp.TOTP") as mock_totp_class:
            mock_totp = MagicMock()
            mock_totp.verify.return_value = True
            mock_totp_class.return_value = mock_totp

            provider.verify_totp_code("123456", valid_secret, "user1", window=10)
            # Window should be clamped to 2
            mock_totp.verify.assert_called_once_with("123456", valid_window=2)

    def test_verify_failed_code(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test verification fails with incorrect code."""
        with patch("pyotp.TOTP") as mock_totp_class:
            mock_totp = MagicMock()
            mock_totp.verify.return_value = False
            mock_totp_class.return_value = mock_totp

            result = provider.verify_totp_code("999999", valid_secret, "user1")
            assert result is False

    def test_verify_exception_handling(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test verification handles exceptions gracefully."""
        with patch("pyotp.TOTP") as mock_totp_class:
            mock_totp_class.side_effect = Exception("TOTP error")

            result = provider.verify_totp_code("123456", valid_secret, "user1")
            assert result is False

    def test_verify_used_codes_limit(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test used codes are limited to prevent memory bloat."""
        with patch("pyotp.TOTP") as mock_totp_class:
            mock_totp = MagicMock()
            mock_totp.verify.return_value = True
            mock_totp_class.return_value = mock_totp

            # Add 11 codes (limit is 10, should keep last 5)
            for i in range(11):
                provider.verify_totp_code(f"{100000 + i}", valid_secret, "user1")

            # Should only have 5 codes
            assert len(provider._used_codes["user1"]) == 5

    def test_verify_code_expiry_set(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test code expiry is set after successful verification."""
        with patch("pyotp.TOTP") as mock_totp_class:
            mock_totp = MagicMock()
            mock_totp.verify.return_value = True
            mock_totp_class.return_value = mock_totp

            now = datetime.now(UTC)
            with patch("traigent.security.auth.totp.datetime") as mock_dt:
                mock_dt.now.return_value = now

                provider.verify_totp_code("123456", valid_secret, "user1")

                assert "user1" in provider._code_expiry
                # Expiry should be set to 90 seconds in future
                expected_expiry = now + timedelta(seconds=90)
                # Allow small time difference
                assert (
                    abs(
                        (
                            provider._code_expiry["user1"] - expected_expiry
                        ).total_seconds()
                    )
                    < 1
                )

    def test_verify_six_digit_code(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test verification accepts 6-digit codes."""
        with patch("pyotp.TOTP") as mock_totp_class:
            mock_totp = MagicMock()
            mock_totp.verify.return_value = True
            mock_totp_class.return_value = mock_totp

            result = provider.verify_totp_code("123456", valid_secret, "user1")
            assert result is True

    def test_verify_eight_digit_code(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test verification accepts 8-digit codes."""
        with patch("pyotp.TOTP") as mock_totp_class:
            mock_totp = MagicMock()
            mock_totp.verify.return_value = True
            mock_totp_class.return_value = mock_totp

            result = provider.verify_totp_code("12345678", valid_secret, "user1")
            assert result is True

    def test_verify_seven_digit_code(
        self, provider: TOTPAuthProvider, valid_secret: str
    ) -> None:
        """Test verification accepts 7-digit codes."""
        with patch("pyotp.TOTP") as mock_totp_class:
            mock_totp = MagicMock()
            mock_totp.verify.return_value = True
            mock_totp_class.return_value = mock_totp

            result = provider.verify_totp_code("1234567", valid_secret, "user1")
            assert result is True


class TestGenerateBackupCodes:
    """Tests for backup code generation."""

    @pytest.fixture
    def provider(self) -> TOTPAuthProvider:
        """Create test provider instance."""
        if not PYOTP_AVAILABLE:
            pytest.skip("pyotp not available")
        return TOTPAuthProvider()

    def test_generate_default_count(self, provider: TOTPAuthProvider) -> None:
        """Test backup codes generation with default count."""
        codes = provider.generate_backup_codes()
        assert len(codes) == 10

    def test_generate_custom_count(self, provider: TOTPAuthProvider) -> None:
        """Test backup codes generation with custom count."""
        codes = provider.generate_backup_codes(count=5)
        assert len(codes) == 5

    def test_generate_count_clamped_minimum(self, provider: TOTPAuthProvider) -> None:
        """Test count is clamped to minimum of 1."""
        codes = provider.generate_backup_codes(count=0)
        assert len(codes) >= 1

    def test_generate_count_clamped_maximum(self, provider: TOTPAuthProvider) -> None:
        """Test count is clamped to maximum of 20."""
        codes = provider.generate_backup_codes(count=100)
        assert len(codes) <= 20

    def test_generate_default_code_length(self, provider: TOTPAuthProvider) -> None:
        """Test backup codes have default length of 8."""
        codes = provider.generate_backup_codes()
        for code in codes:
            assert len(code) == 8

    def test_generate_custom_code_length(self, provider: TOTPAuthProvider) -> None:
        """Test backup codes generation with custom length."""
        codes = provider.generate_backup_codes(code_length=12)
        for code in codes:
            assert len(code) == 12

    def test_generate_code_length_clamped_minimum(
        self, provider: TOTPAuthProvider
    ) -> None:
        """Test code length is clamped to minimum of 8."""
        codes = provider.generate_backup_codes(code_length=4)
        for code in codes:
            assert len(code) == 8

    def test_generate_code_length_clamped_maximum(
        self, provider: TOTPAuthProvider
    ) -> None:
        """Test code length is clamped to maximum of 16."""
        codes = provider.generate_backup_codes(code_length=32)
        for code in codes:
            assert len(code) == 16

    def test_generate_codes_unique(self, provider: TOTPAuthProvider) -> None:
        """Test all generated backup codes are unique."""
        codes = provider.generate_backup_codes(count=20)
        assert len(codes) == len(set(codes))

    def test_generate_codes_uppercase(self, provider: TOTPAuthProvider) -> None:
        """Test backup codes are uppercase."""
        codes = provider.generate_backup_codes()
        for code in codes:
            assert code.isupper()

    def test_generate_codes_alphanumeric(self, provider: TOTPAuthProvider) -> None:
        """Test backup codes contain only alphanumeric characters."""
        codes = provider.generate_backup_codes()
        for code in codes:
            assert code.isalnum()

    def test_generate_codes_no_special_chars(self, provider: TOTPAuthProvider) -> None:
        """Test backup codes do not contain dash or underscore."""
        codes = provider.generate_backup_codes(count=20)
        for code in codes:
            assert "-" not in code
            assert "_" not in code

    def test_generate_codes_deterministic_randomness(
        self, provider: TOTPAuthProvider
    ) -> None:
        """Test backup codes are different on each generation."""
        codes1 = provider.generate_backup_codes()
        codes2 = provider.generate_backup_codes()
        # Should be different sets
        assert codes1 != codes2

    def test_generate_handles_collision_attempts(
        self, provider: TOTPAuthProvider
    ) -> None:
        """Test generation handles potential collisions gracefully."""
        # This should succeed even if there are internal collisions
        codes = provider.generate_backup_codes(count=20)
        assert len(codes) > 0
        assert len(codes) <= 20
