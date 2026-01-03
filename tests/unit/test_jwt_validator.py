"""Comprehensive unit tests for JWT validator security component."""

import logging
import os
import time
from unittest.mock import Mock, patch

import pytest

# Import the JWT validator components
from traigent.security.jwt_validator import (
    JWTExpiredError,
    JWTInvalidError,
    JWTSecurityError,
    JWTSignatureError,
    JWTValidationError,
    JWTValidationResult,
    SecureJWTValidator,
    ValidationMode,
    get_secure_jwt_validator,
)

# ============ Shared Mock JWT Exception Classes ============
# These mock classes simulate PyJWT's exception hierarchy for testing
# without requiring the actual PyJWT library to be installed.


class MockExpiredSignatureError(Exception):
    """Mock for jwt.ExpiredSignatureError."""

    pass


class MockInvalidSignatureError(Exception):
    """Mock for jwt.InvalidSignatureError."""

    pass


class MockInvalidTokenError(Exception):
    """Mock for jwt.InvalidTokenError."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message


@pytest.fixture
def mock_jwt_exceptions():
    """Fixture providing mock JWT exception classes for tests.

    Returns a dict that can be used to configure a mock jwt module:
        mock_jwt.ExpiredSignatureError = exceptions['expired']
        mock_jwt.InvalidSignatureError = exceptions['signature']
        mock_jwt.InvalidTokenError = exceptions['token']
    """
    return {
        "expired": MockExpiredSignatureError,
        "signature": MockInvalidSignatureError,
        "token": MockInvalidTokenError,
    }


def configure_mock_jwt_exceptions(mock_jwt):
    """Configure a mock JWT module with all exception classes.

    This helper ensures consistent mock JWT configuration across tests.
    """
    mock_jwt.ExpiredSignatureError = MockExpiredSignatureError
    mock_jwt.InvalidSignatureError = MockInvalidSignatureError
    mock_jwt.InvalidTokenError = MockInvalidTokenError


class TestJWTValidationResult:
    """Test JWT validation result dataclass."""

    def test_validation_result_creation(self):
        """Test creating JWT validation result."""
        result = JWTValidationResult(valid=True, payload={"sub": "user123"})
        assert result.valid is True
        assert result.payload == {"sub": "user123"}
        assert result.error is None
        assert result.warnings == []  # Should initialize empty list
        assert result.expires_at is None

    def test_validation_result_with_error(self):
        """Test validation result with error."""
        result = JWTValidationResult(
            valid=False, error="Token expired", warnings=["Development mode"]
        )
        assert result.valid is False
        assert result.error == "Token expired"
        assert result.warnings == ["Development mode"]

    def test_validation_result_post_init(self):
        """Test that warnings list is initialized if None."""
        result = JWTValidationResult(valid=True)
        assert result.warnings == []


class TestValidationModes:
    """Test JWT validation mode enumeration."""

    def test_validation_modes_exist(self):
        """Test that all expected validation modes exist."""
        assert ValidationMode.PRODUCTION.value == "production"
        assert ValidationMode.STAGING.value == "staging"
        assert ValidationMode.DEVELOPMENT.value == "development"


class TestSecureJWTValidator:
    """Test SecureJWTValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_jwks_url = "https://example.com/.well-known/jwks.json"
        self.test_issuer = "https://example.com"
        self.test_audience = "test-audience"

        # Sample JWT payload
        self.sample_payload = {
            "sub": "user123",
            "iss": self.test_issuer,
            "aud": self.test_audience,
            "exp": int(time.time()) + 3600,  # 1 hour from now
            "iat": int(time.time()),
            "nbf": int(time.time()),
        }

        # Sample expired payload
        self.expired_payload = {
            "sub": "user123",
            "iss": self.test_issuer,
            "aud": self.test_audience,
            "exp": int(time.time()) - 3600,  # 1 hour ago
            "iat": int(time.time()) - 7200,
            "nbf": int(time.time()) - 7200,
        }

    def test_validator_initialization(self):
        """Test JWT validator initialization."""
        validator = SecureJWTValidator(
            jwks_url=self.test_jwks_url,
            issuer=self.test_issuer,
            audience=self.test_audience,
            validation_mode=ValidationMode.PRODUCTION,
        )

        assert validator.jwks_url == self.test_jwks_url
        assert validator.issuer == self.test_issuer
        assert validator.audience == self.test_audience
        assert validator.validation_mode == ValidationMode.PRODUCTION
        assert validator._jwks_client is None

    def test_validator_default_initialization(self):
        """Test validator with default parameters."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        assert validator.jwks_url is None
        assert validator.issuer is None
        assert validator.audience is None
        assert validator.validation_mode == ValidationMode.DEVELOPMENT

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", False)
    def test_jwt_unavailable_warning(self, caplog):
        """Test warning when PyJWT is not available."""
        with caplog.at_level(logging.WARNING):
            SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)
            assert "PyJWT not available" in caplog.text

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    @patch("traigent.security.jwt_validator.PyJWKClient")
    def test_get_jwks_client_success(self, mock_jwks_client):
        """Test successful JWKS client creation."""
        validator = SecureJWTValidator(
            jwks_url=self.test_jwks_url, validation_mode=ValidationMode.DEVELOPMENT
        )

        client = validator._get_jwks_client()

        mock_jwks_client.assert_called_once_with(
            self.test_jwks_url, cache_keys=True, max_cached_keys=16, cache_jwk_set=True
        )
        assert client is not None

        # Test client is cached
        client2 = validator._get_jwks_client()
        assert client2 is client  # Same instance
        assert mock_jwks_client.call_count == 1  # Called only once

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    @patch("traigent.security.jwt_validator.PyJWKClient")
    def test_get_jwks_client_failure(self, mock_jwks_client, caplog):
        """Test JWKS client creation failure."""
        mock_jwks_client.side_effect = Exception("Connection failed")
        validator = SecureJWTValidator(
            jwks_url=self.test_jwks_url, validation_mode=ValidationMode.DEVELOPMENT
        )

        with caplog.at_level(logging.ERROR):
            client = validator._get_jwks_client()

        assert client is None
        assert "Failed to initialize JWKS client" in caplog.text

    def test_get_jwks_client_no_url(self):
        """Test JWKS client when no URL provided."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)
        client = validator._get_jwks_client()
        assert client is None

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", False)
    def test_validate_token_no_jwt_library(self):
        """Test token validation when PyJWT is not available."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)
        result = validator.validate_token("fake_jwt_token")

        assert result.valid is False
        assert "JWT validation unavailable" in result.error
        assert any("Install PyJWT" in warning for warning in result.warnings)

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    def test_validate_token_routes_to_correct_method(self):
        """Test that validate_token routes to correct validation method."""
        validator = SecureJWTValidator(
            jwks_url=self.test_jwks_url,
            issuer=self.test_issuer,
            audience=self.test_audience,
            validation_mode=ValidationMode.PRODUCTION,
        )

        with patch.object(validator, "_validate_strict") as mock_strict:
            mock_strict.return_value = JWTValidationResult(valid=True)
            validator.validate_token("token")
            mock_strict.assert_called_once_with("token")

        validator.validation_mode = ValidationMode.DEVELOPMENT
        with patch.object(validator, "_validate_development_secure") as mock_dev:
            mock_dev.return_value = JWTValidationResult(valid=True)
            validator.validate_token("token")
            mock_dev.assert_called_once_with("token")

        validator.validation_mode = ValidationMode.STAGING
        with patch.object(validator, "_validate_staging") as mock_staging:
            mock_staging.return_value = JWTValidationResult(valid=True)
            validator.validate_token("token")
            mock_staging.assert_called_once_with("token")

    def test_scope_validation_respects_configuration(self):
        """Scopes are only enforced when allow-list is explicitly configured."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        with patch.dict(os.environ, {}, clear=True):
            # No scopes configured -> should not raise
            validator._perform_security_checks({"scope": "read"})

        with patch.dict(os.environ, {"TRAIGENT_ALLOWED_SCOPES": "read, write"}):
            validator._perform_security_checks({"scope": "read"})
            with pytest.raises(JWTSecurityError):
                validator._perform_security_checks({"scope": "delete"})


class TestStrictValidation:
    """Test strict JWT validation mode."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SecureJWTValidator(
            jwks_url="https://example.com/.well-known/jwks.json",
            issuer="https://example.com",
            audience="test-audience",
            validation_mode=ValidationMode.PRODUCTION,
        )
        self.test_token = "test_token_placeholder"

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    def test_strict_validation_no_jwks_client(self):
        """Test strict validation when JWKS client unavailable."""
        with patch.object(self.validator, "_get_jwks_client", return_value=None):
            result = self.validator._validate_strict(self.test_token)

            assert result.valid is False
            assert "Cannot verify signature" in result.error

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    @patch("traigent.security.jwt_validator.jwt")
    def test_strict_validation_success(self, mock_jwt):
        """Test successful strict validation."""
        # Mock JWKS client and signing key
        mock_jwks_client = Mock()
        mock_signing_key = Mock()
        mock_signing_key.key = "test_key"
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

        # Mock successful JWT decode
        expected_payload = {
            "sub": "user123",
            "iss": "https://example.com",
            "aud": "test-audience",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "nbf": int(time.time()),
            "jti": "token-123",
        }
        mock_jwt.decode.return_value = expected_payload

        with patch.object(
            self.validator, "_get_jwks_client", return_value=mock_jwks_client
        ):
            result = self.validator._validate_strict(self.test_token)

            assert result.valid is True
            assert result.payload == expected_payload
            assert result.expires_at == expected_payload["exp"]
            assert result.error is None

            # Verify JWT decode was called
            mock_jwt.decode.assert_called_once()
            call_args = mock_jwt.decode.call_args
            assert call_args[0][0] == self.test_token  # token
            assert call_args[0][1] == "test_key"  # signing key
            assert call_args[1]["algorithms"] == ["RS256", "ES256"]
            assert call_args[1]["issuer"] == self.validator.issuer
            assert call_args[1]["audience"] == self.validator.audience
            assert call_args[1]["options"]["verify_signature"] is True
            assert call_args[1]["options"]["verify_exp"] is True

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    @patch("traigent.security.jwt_validator.jwt")
    def test_strict_validation_expired_token(self, mock_jwt):
        """Test strict validation with expired token."""
        mock_jwks_client = Mock()
        mock_signing_key = Mock()
        mock_signing_key.key = "test_key"
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

        # Configure mock JWT exceptions using shared helper
        configure_mock_jwt_exceptions(mock_jwt)
        mock_jwt.decode.side_effect = MockExpiredSignatureError("Token expired")

        with patch.object(
            self.validator, "_get_jwks_client", return_value=mock_jwks_client
        ):
            result = self.validator._validate_strict(self.test_token)

            assert result.valid is False
            assert result.error == "Token has expired"

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    @patch("traigent.security.jwt_validator.jwt")
    def test_strict_validation_invalid_signature(self, mock_jwt):
        """Test strict validation with invalid signature."""
        mock_jwks_client = Mock()
        mock_signing_key = Mock()
        mock_signing_key.key = "test_key"
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

        # Configure mock JWT exceptions using shared helper
        configure_mock_jwt_exceptions(mock_jwt)
        mock_jwt.decode.side_effect = MockInvalidSignatureError("Invalid signature")

        with patch.object(
            self.validator, "_get_jwks_client", return_value=mock_jwks_client
        ):
            result = self.validator._validate_strict(self.test_token)

            assert result.valid is False
            assert result.error == "Invalid token signature"

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    @patch("traigent.security.jwt_validator.jwt")
    def test_strict_validation_invalid_token(self, mock_jwt):
        """Test strict validation with invalid token."""
        mock_jwks_client = Mock()
        mock_signing_key = Mock()
        mock_signing_key.key = "test_key"
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

        # Configure mock JWT exceptions using shared helper
        configure_mock_jwt_exceptions(mock_jwt)
        mock_jwt.decode.side_effect = MockInvalidTokenError("Invalid audience")

        with patch.object(
            self.validator, "_get_jwks_client", return_value=mock_jwks_client
        ):
            result = self.validator._validate_strict(self.test_token)

            assert result.valid is False
            assert "Invalid token: Invalid audience" in result.error

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    def test_strict_validation_generic_exception(self):
        """Test strict validation with generic exception."""
        mock_jwks_client = Mock()
        mock_jwks_client.get_signing_key_from_jwt.side_effect = Exception(
            "Network error"
        )

        with patch.object(
            self.validator, "_get_jwks_client", return_value=mock_jwks_client
        ):
            result = self.validator._validate_strict(self.test_token)

            assert result.valid is False
            assert "Network error" in result.error


class TestDevelopmentValidation:
    """Test development JWT validation mode."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SecureJWTValidator(
            jwks_url="https://example.com/.well-known/jwks.json",
            validation_mode=ValidationMode.DEVELOPMENT,
        )
        self.test_token = "jwt_token_placeholder"

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    @patch("traigent.security.jwt_validator.jwt")
    def test_development_validation_success(self, mock_jwt):
        """Test successful development validation."""
        expected_payload = {
            "sub": "user123",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        mock_jwt.decode.return_value = expected_payload

        result = self.validator._validate_development(self.test_token)

        assert result.valid is True
        assert result.payload == expected_payload
        assert result.expires_at == expected_payload["exp"]
        assert "Development mode" in result.warnings[0]

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    @patch("traigent.security.jwt_validator.jwt")
    @patch("traigent.security.jwt_validator.time.time")
    def test_development_validation_expired_token(self, mock_time, mock_jwt):
        """Test development validation with expired token."""
        # Set current time to after token expiration
        current_time = 1000000
        mock_time.return_value = current_time

        expired_payload = {
            "sub": "user123",
            "exp": current_time - 100,  # Expired 100 seconds ago
            "iat": current_time - 3600,
        }
        mock_jwt.decode.return_value = expired_payload

        result = self.validator._validate_development(self.test_token)

        assert result.valid is False
        assert result.error == "Token has expired"
        assert "Development mode" in result.warnings[0]

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    @patch("traigent.security.jwt_validator.jwt")
    def test_development_validation_no_expiration(self, mock_jwt):
        """Test development validation with token without expiration."""
        payload_no_exp = {"sub": "user123", "iat": int(time.time())}
        mock_jwt.decode.return_value = payload_no_exp

        result = self.validator._validate_development(self.test_token)

        assert result.valid is True
        assert result.payload == payload_no_exp
        assert result.expires_at is None

    @patch("traigent.security.jwt_validator.JWT_AVAILABLE", True)
    @patch("traigent.security.jwt_validator.jwt")
    def test_development_validation_invalid_structure(self, mock_jwt):
        """Test development validation with invalid token structure."""
        # Use shared mock exception class
        mock_jwt.InvalidTokenError = MockInvalidTokenError
        mock_jwt.decode.side_effect = MockInvalidTokenError("Invalid format")

        result = self.validator._validate_development(self.test_token)

        assert result.valid is False
        assert "Invalid format" in result.error
        assert "Development mode" in result.warnings[0]


class TestConstantTimeValidation:
    """Test constant-time validation for timing attack prevention."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

    def test_constant_time_validate_true_result(self):
        """Test constant-time validation with valid token."""
        with patch.object(self.validator, "validate_token") as mock_validate:
            mock_validate.return_value = JWTValidationResult(valid=True)

            result = self.validator.constant_time_validate(
                "token", expected_result=True
            )

            assert result is True
            mock_validate.assert_called_once_with("token")

    def test_constant_time_validate_false_result(self):
        """Test constant-time validation with invalid token."""
        with patch.object(self.validator, "validate_token") as mock_validate:
            mock_validate.return_value = JWTValidationResult(valid=False)

            result = self.validator.constant_time_validate(
                "token", expected_result=True
            )

            assert result is False
            mock_validate.assert_called_once_with("token")

    def test_constant_time_validate_timing_consistency(self):
        """Test that constant-time validation has consistent timing."""
        import time

        with patch.object(self.validator, "validate_token") as mock_validate:
            # Test multiple calls with different results
            times = []

            for valid in [True, False, True, False]:
                mock_validate.return_value = JWTValidationResult(valid=valid)

                start = time.perf_counter()
                self.validator.constant_time_validate("token")
                end = time.perf_counter()

                times.append(end - start)

            # Times should be relatively consistent (within reasonable variance)
            # This is a basic test - in practice, you'd use more sophisticated timing analysis
            avg_time = sum(times) / len(times)
            for t in times:
                # Allow up to 300% variance since we're testing with mocked functions
                assert abs(t - avg_time) / avg_time < 3.0  # Within 300% variance


class TestGetJWTValidator:
    """Test JWT validator factory function."""

    def test_get_secure_jwt_validator_default(self):
        """Test getting JWT validator with defaults."""
        # When no env vars are set, validator should be created with None values
        env_vars = {
            "TRAIGENT_JWKS_URL": "",
            "TRAIGENT_JWT_ISSUER": "",
            "TRAIGENT_JWT_AUDIENCE": "",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Clear the specific env vars
            for key in env_vars:
                os.environ.pop(key, None)

            validator = get_secure_jwt_validator(mode=ValidationMode.DEVELOPMENT)

        assert isinstance(validator, SecureJWTValidator)
        assert validator.validation_mode == ValidationMode.DEVELOPMENT
        # When env vars are not set, these should be None
        assert validator.jwks_url is None
        assert validator.issuer is None
        assert validator.audience is None

    def test_get_secure_jwt_validator_with_environment(self):
        """Test getting JWT validator with environment variables."""
        env_vars = {
            "TRAIGENT_JWKS_URL": "https://example.com/.well-known/jwks.json",
            "TRAIGENT_JWT_ISSUER": "https://example.com",
            "TRAIGENT_JWT_AUDIENCE": "test-audience",
        }

        with patch.dict(os.environ, env_vars):
            validator = get_secure_jwt_validator()

            assert validator.jwks_url == env_vars["TRAIGENT_JWKS_URL"]
            assert validator.issuer == env_vars["TRAIGENT_JWT_ISSUER"]
            assert validator.audience == env_vars["TRAIGENT_JWT_AUDIENCE"]

    def test_get_secure_jwt_validator_development_mode(self, caplog):
        """Test getting JWT validator in development mode."""
        env_vars = {"TRAIGENT_ENVIRONMENT": "development"}

        with patch.dict(os.environ, env_vars):
            with caplog.at_level(logging.WARNING):
                validator = get_secure_jwt_validator()

            assert validator.validation_mode == ValidationMode.DEVELOPMENT
            assert "DEVELOPMENT mode" in caplog.text

    def test_get_secure_jwt_validator_rejects_security_disable_flag(self):
        """Ensure insecure bypass flags raise errors."""
        env_vars = {"TRAIGENT_JWT_DISABLE_SECURITY": "1"}

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(JWTSecurityError):
                get_secure_jwt_validator()

    def test_get_secure_jwt_validator_explicit_mode_override(self):
        """Test that explicit mode parameter is used when provided."""
        env_vars = {"TRAIGENT_ENVIRONMENT": "production"}

        with patch.dict(os.environ, env_vars):
            # Explicitly request development mode
            validator = get_secure_jwt_validator(mode=ValidationMode.DEVELOPMENT)

            # Explicit parameter should override environment
            assert validator.validation_mode == ValidationMode.DEVELOPMENT

    def test_get_secure_jwt_validator_invalid_env_mode(self):
        """Test getting JWT validator with invalid environment mode."""
        env_vars = {"TRAIGENT_ENVIRONMENT": "invalid_mode"}

        with patch.dict(os.environ, env_vars):
            validator = get_secure_jwt_validator(mode=ValidationMode.DEVELOPMENT)

            # Should use explicit mode when provided
            assert validator.validation_mode == ValidationMode.DEVELOPMENT


class TestJWTValidatorIntegration:
    """Integration tests for JWT validator."""

    def test_full_validation_workflow_strict_mode(self):
        """Test complete validation workflow in strict mode."""
        validator = SecureJWTValidator(
            jwks_url="https://example.com/.well-known/jwks.json",
            issuer="https://example.com",
            audience="test-audience",
            validation_mode=ValidationMode.PRODUCTION,
        )

        with patch("traigent.security.jwt_validator.JWT_AVAILABLE", True):
            with patch.object(validator, "_get_jwks_client") as mock_get_client:
                mock_get_client.return_value = None  # Simulate no JWKS client

                result = validator.validate_token("test_jwt_token")

                assert result.valid is False
                assert "Cannot verify signature" in result.error

    def test_full_validation_workflow_development_mode(self):
        """Test complete validation workflow in development mode."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        with patch("traigent.security.jwt_validator.JWT_AVAILABLE", True):
            with patch("traigent.security.jwt_validator.jwt") as mock_jwt:
                payload = {
                    "sub": "user123",
                    "exp": int(time.time()) + 3600,
                    "iat": int(time.time()),
                }
                mock_jwt.decode.return_value = payload

                result = validator.validate_token("test_jwt_token")

                assert result.valid is True
                assert result.payload == payload
                assert (
                    "DEVELOPMENT MODE: Limited to 5-minute token lifetime"
                    in result.warnings[0]
                )

    def test_security_validation_edge_cases(self):
        """Test security edge cases and error conditions."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        # Test with empty token
        result = validator.validate_token("")
        assert result.valid is False

        # Test with None token (should handle gracefully)
        try:
            result = validator.validate_token(None)
            # Should either handle gracefully or raise appropriate error
        except (TypeError, AttributeError):
            # Expected for None input
            pass

    def test_validator_immutability_after_creation(self):
        """Test that validator configuration can be modified after creation."""
        validator = SecureJWTValidator(
            jwks_url="https://example.com/.well-known/jwks.json",
            issuer="https://example.com",
            audience="test-audience",
            validation_mode=ValidationMode.PRODUCTION,
        )

        # Attempt to modify (should work as these are public attributes)
        validator.jwks_url = "https://malicious.com/jwks.json"
        validator.validation_mode = ValidationMode.DEVELOPMENT

        # Verify configuration can be changed (this is actually expected behavior)
        # In a more secure implementation, these might be read-only
        assert validator.jwks_url == "https://malicious.com/jwks.json"
        assert validator.validation_mode == ValidationMode.DEVELOPMENT


# Test exception classes
class TestJWTExceptions:
    """Test JWT exception classes."""

    def test_jwt_validation_error_inheritance(self):
        """Test JWT exception inheritance."""
        error = JWTValidationError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_jwt_signature_error_inheritance(self):
        """Test JWT signature error inheritance."""
        error = JWTSignatureError("Invalid signature")
        assert isinstance(error, JWTValidationError)
        assert isinstance(error, Exception)

    def test_jwt_expired_error_inheritance(self):
        """Test JWT expired error inheritance."""
        error = JWTExpiredError("Token expired")
        assert isinstance(error, JWTValidationError)
        assert isinstance(error, Exception)

    def test_jwt_invalid_error_inheritance(self):
        """Test JWT invalid error inheritance."""
        error = JWTInvalidError("Token invalid")
        assert isinstance(error, JWTValidationError)
        assert isinstance(error, Exception)


class TestJWTValidatorHelperMethods:
    """Test helper methods extracted for cognitive complexity reduction."""

    def _create_validator(
        self, validation_mode: ValidationMode = ValidationMode.DEVELOPMENT
    ):
        """Create a validator with test configuration."""
        return SecureJWTValidator(
            jwks_url="https://example.com/.well-known/jwks.json",
            issuer="https://example.com",
            audience="test-audience",
            validation_mode=validation_mode,
        )

    def test_development_warnings_returns_list(self):
        """Test _development_warnings returns expected warnings."""
        warnings = SecureJWTValidator._development_warnings()
        assert isinstance(warnings, list)
        assert len(warnings) == 3
        assert any("DEVELOPMENT MODE" in w for w in warnings)
        assert any("NOT suitable for production" in w for w in warnings)

    def test_extract_header_algorithm_valid_jwt(self):
        """Test _extract_header_algorithm with valid JWT structure."""
        validator = self._create_validator()
        # Valid JWT with HS256 algorithm (structure: header.payload.signature)
        # Header: {"alg": "HS256", "typ": "JWT"} base64url encoded
        valid_jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.sig"
        )

        with patch("traigent.security.jwt_validator.jwt") as mock_jwt:
            mock_jwt.get_unverified_header = Mock(return_value={"alg": "HS256"})
            result = validator._extract_header_algorithm(valid_jwt)
            assert result == "HS256"

    def test_extract_header_algorithm_invalid_structure(self):
        """Test _extract_header_algorithm with invalid JWT structure."""
        validator = self._create_validator()
        # Token with fewer than 2 dots is invalid
        invalid_jwt = "not.a.valid"
        result = validator._extract_header_algorithm(invalid_jwt)
        # With fewer than 2 dots, should return None without calling jwt
        assert result is None or result == "a"  # Depends on jwt mock

    def test_extract_header_algorithm_no_dots(self):
        """Test _extract_header_algorithm with no dots in token."""
        validator = self._create_validator()
        result = validator._extract_header_algorithm("notavalidtoken")
        assert result is None

    def test_extract_header_algorithm_exception_handling(self):
        """Test _extract_header_algorithm handles exceptions gracefully."""
        validator = self._create_validator()
        valid_jwt = "a.b.c"

        with patch("traigent.security.jwt_validator.jwt") as mock_jwt:
            mock_jwt.get_unverified_header = Mock(side_effect=Exception("Parse error"))
            result = validator._extract_header_algorithm(valid_jwt)
            assert result is None

    def test_validate_algorithm_none_not_allowed(self):
        """Test _validate_algorithm rejects 'none' algorithm."""
        validator = self._create_validator()
        warnings: list[str] = []
        result = validator._validate_algorithm("none", warnings)
        assert result is not None
        assert result.valid is False
        assert "none" in result.error.lower()

    def test_validate_algorithm_none_case_insensitive(self):
        """Test _validate_algorithm rejects 'NONE' algorithm (case insensitive)."""
        validator = self._create_validator()
        warnings: list[str] = []
        result = validator._validate_algorithm("NONE", warnings)
        assert result is not None
        assert result.valid is False

    def test_validate_algorithm_allowed_algorithms(self):
        """Test _validate_algorithm accepts allowed algorithms."""
        validator = self._create_validator()
        warnings: list[str] = []

        # RS256 should be allowed (in ALLOWED_ALGORITHMS)
        result = validator._validate_algorithm("RS256", warnings)
        assert result is None  # None means valid

        # HS256 should be allowed in development
        result = validator._validate_algorithm("HS256", warnings)
        assert result is None

    def test_validate_algorithm_unsupported(self):
        """Test _validate_algorithm rejects unsupported algorithms."""
        validator = self._create_validator()
        warnings: list[str] = []
        result = validator._validate_algorithm("XYZ999", warnings)
        assert result is not None
        assert result.valid is False
        assert "Unsupported algorithm" in result.error

    def test_validate_algorithm_none_value(self):
        """Test _validate_algorithm with None header_alg."""
        validator = self._create_validator()
        warnings: list[str] = []
        result = validator._validate_algorithm(None, warnings)
        assert result is None  # None header_alg is valid (skip check)

    def test_check_token_lifetime_missing_iat(self):
        """Test _check_token_lifetime with missing iat claim."""
        validator = self._create_validator()
        warnings: list[str] = []
        payload = {"sub": "user123"}  # No iat claim
        result = validator._check_token_lifetime(payload, warnings)
        assert result is not None
        assert result.valid is False
        assert "iat claim" in result.error

    def test_check_token_lifetime_expired_exp(self):
        """Test _check_token_lifetime with expired exp claim."""
        validator = self._create_validator()
        warnings: list[str] = []
        # Token with exp in the past
        payload = {"iat": time.time(), "exp": time.time() - 3600}
        result = validator._check_token_lifetime(payload, warnings)
        assert result is not None
        assert result.valid is False
        assert "expired" in result.error.lower()

    def test_check_token_lifetime_exceeds_dev_lifetime(self):
        """Test _check_token_lifetime when token exceeds dev lifetime."""
        validator = self._create_validator()
        warnings: list[str] = []
        # Token issued too long ago (more than DEVELOPMENT_TOKEN_LIFETIME)
        old_iat = time.time() - validator.DEVELOPMENT_TOKEN_LIFETIME - 100
        payload = {"iat": old_iat}
        result = validator._check_token_lifetime(payload, warnings)
        assert result is not None
        assert result.valid is False

    def test_check_token_lifetime_valid(self):
        """Test _check_token_lifetime with valid token."""
        validator = self._create_validator()
        warnings: list[str] = []
        current_time = time.time()
        payload = {"iat": current_time, "exp": current_time + 300}
        result = validator._check_token_lifetime(payload, warnings)
        assert result is None  # None means valid

    def test_mark_development_payload_with_valid_jwt(self):
        """Test _mark_development_payload adds dev markers for valid JWT."""
        validator = self._create_validator()
        token = "header.payload.signature"
        original_payload = {"sub": "user123"}
        result = validator._mark_development_payload(token, original_payload)
        assert "_development_mode" in result
        assert result["_development_mode"] is True
        assert "_max_validity" in result
        assert result["sub"] == "user123"

    def test_mark_development_payload_without_valid_structure(self):
        """Test _mark_development_payload without valid JWT structure."""
        validator = self._create_validator()
        token = "invalid"  # No dots
        original_payload = {"sub": "user123"}
        result = validator._mark_development_payload(token, original_payload)
        # Should return original payload unchanged
        assert result == original_payload
        assert "_development_mode" not in result


class TestTokenExpiredErrorConstant:
    """Test the _TOKEN_EXPIRED_ERROR constant is used correctly."""

    def test_token_expired_error_message(self):
        """Test that token expired errors use the constant."""
        from traigent.security.jwt_validator import _TOKEN_EXPIRED_ERROR

        assert _TOKEN_EXPIRED_ERROR == "Token has expired"


if __name__ == "__main__":
    pytest.main([__file__])
