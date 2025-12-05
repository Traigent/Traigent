"""Integration tests for JWT validation across all modes."""

import os
import time
from unittest.mock import MagicMock, patch

import jwt
import pytest

from traigent.security.jwt_validator import (
    JWTSecurityError,
    SecureJWTValidator,
    ValidationMode,
    get_secure_jwt_validator,
)


class TestJWTProductionMode:
    """Test JWT validation in production mode."""

    def test_production_mode_strict_requirements(self, clean_environment):
        """Test that production mode enforces strict requirements."""
        # Should fail without proper configuration
        with pytest.raises(JWTSecurityError) as exc:
            SecureJWTValidator(validation_mode=ValidationMode.PRODUCTION)
        assert "JWKS URL required" in str(exc.value)

        # Should fail without issuer
        with pytest.raises(JWTSecurityError) as exc:
            SecureJWTValidator(
                jwks_url="https://example.com/jwks",
                validation_mode=ValidationMode.PRODUCTION,
            )
        assert "Issuer required" in str(exc.value)

        # Should fail without audience
        with pytest.raises(JWTSecurityError) as exc:
            SecureJWTValidator(
                jwks_url="https://example.com/jwks",
                issuer="test_issuer",
                validation_mode=ValidationMode.PRODUCTION,
            )
        assert "Audience required" in str(exc.value)

    def test_production_mode_with_valid_config(
        self, test_environments, clean_environment
    ):
        """Test production mode with valid configuration."""
        # Set production environment
        os.environ.update(test_environments["production"])

        # Should create validator successfully
        validator = get_secure_jwt_validator()
        assert validator.validation_mode == ValidationMode.PRODUCTION
        assert (
            validator.jwks_url == test_environments["production"]["TRAIGENT_JWKS_URL"]
        )
        assert (
            validator.issuer == test_environments["production"]["TRAIGENT_JWT_ISSUER"]
        )

    @patch("traigent.security.jwt_validator.PyJWKClient")
    def test_production_mode_signature_verification(
        self, mock_jwks_client, sample_jwt_payload
    ):
        """Test that production mode properly verifies signatures."""
        # Setup mock JWKS client
        mock_key = MagicMock()
        mock_key.key = "test_key"
        mock_key.algorithm_name = "RS256"
        mock_key.key_id = "key123"
        mock_key.key_size = 2048

        mock_client_instance = MagicMock()
        mock_client_instance.get_signing_key_from_jwt.return_value = mock_key
        mock_jwks_client.return_value = mock_client_instance

        # Create validator
        validator = SecureJWTValidator(
            jwks_url="https://example.com/jwks",
            issuer="test_issuer",
            audience="test_audience",
            validation_mode=ValidationMode.PRODUCTION,
        )

        # Create test token
        token = jwt.encode(sample_jwt_payload, "secret", algorithm="HS256")

        # Mock jwt.decode to simulate successful validation
        with patch("traigent.security.jwt_validator.jwt.decode") as mock_decode:
            mock_decode.return_value = sample_jwt_payload

            validator.validate_token(token)

            # Verify proper validation was performed
            mock_decode.assert_called_once()
            call_args = mock_decode.call_args
            assert call_args[1]["algorithms"] == ["RS256", "ES256"]
            assert call_args[1]["issuer"] == "test_issuer"
            assert call_args[1]["audience"] == "test_audience"
            assert call_args[1]["options"]["verify_signature"] is True

    def test_production_mode_replay_protection(self, clean_environment):
        """Test JWT replay protection in production mode."""
        validator = SecureJWTValidator(
            jwks_url="https://example.com/jwks",
            issuer="test_issuer",
            audience="test_audience",
            validation_mode=ValidationMode.PRODUCTION,
            require_jti=True,
        )

        # Add a JTI to the seen list
        with validator._jti_lock:
            validator._seen_jti.add("already_used_jti")

        # Create token with the same JTI
        payload = {
            "jti": "already_used_jti",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
        token = jwt.encode(payload, "secret", algorithm="HS256")

        # Should detect replay
        with patch.object(validator, "_get_jwks_client") as mock_client:
            mock_key = MagicMock()
            mock_key.key = "test_key"
            mock_client.return_value.get_signing_key_from_jwt.return_value = mock_key

            with patch("traigent.security.jwt_validator.jwt.decode") as mock_decode:
                mock_decode.return_value = payload

                result = validator.validate_token(token)
                assert not result.valid
                assert "replay" in result.error.lower()


class TestJWTStagingMode:
    """Test JWT validation in staging mode."""

    def test_staging_mode_configuration(self, test_environments, clean_environment):
        """Test staging mode configuration."""
        os.environ.update(test_environments["staging"])

        validator = get_secure_jwt_validator()
        assert validator.validation_mode == ValidationMode.STAGING

    @patch("traigent.security.jwt_validator.logger")
    def test_staging_mode_enhanced_logging(self, mock_logger, clean_environment):
        """Test that staging mode provides enhanced logging."""
        validator = SecureJWTValidator(
            jwks_url="https://example.com/jwks",
            issuer="test_issuer",
            audience="test_audience",
            validation_mode=ValidationMode.STAGING,
        )

        # Create test token
        token = jwt.encode({"test": "data"}, "secret", algorithm="HS256")

        # Trigger validation (will fail, but we're checking logging)
        with patch.object(validator, "_validate_production") as mock_prod:
            mock_prod.return_value.valid = False
            mock_prod.return_value.error = "Test error"

            validator.validate_token(token)

            # Check that logging occurred
            assert mock_logger.info.called
            assert mock_logger.warning.called or mock_logger.info.called


class TestJWTDevelopmentMode:
    """Test JWT validation in development mode."""

    def test_development_mode_time_limit(self, clean_environment):
        """Test that development mode enforces time limits."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        # Old token (older than 5 minutes)
        old_payload = {
            "iat": int(time.time()) - 400,  # 400 seconds ago
            "exp": int(time.time()) + 3600,
        }
        old_token = jwt.encode(old_payload, "secret", algorithm="HS256")

        result = validator.validate_token(old_token)
        assert not result.valid
        assert "expired" in result.error.lower()

        # Recent token (within 5 minutes)
        recent_payload = {
            "iat": int(time.time()) - 100,  # 100 seconds ago
            "exp": int(time.time()) + 3600,
        }
        recent_token = jwt.encode(recent_payload, "secret", algorithm="HS256")

        result = validator.validate_token(recent_token)
        assert result.valid
        assert "DEVELOPMENT MODE" in result.warnings[0]
        assert result.payload["_development_mode"] is True

    def test_development_mode_blocks_none_algorithm(self, clean_environment):
        """Test that 'none' algorithm is blocked even in development."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        # Create token with 'none' algorithm
        import base64
        import json

        header = {"alg": "none", "typ": "JWT"}
        payload = {"test": "data", "iat": int(time.time())}

        header_b64 = (
            base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
        )
        payload_b64 = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        )

        none_token = f"{header_b64}.{payload_b64}."

        result = validator.validate_token(none_token)
        assert not result.valid
        assert "none" in result.error.lower()

    def test_development_mode_markers(self, clean_environment):
        """Test that development tokens are properly marked."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        payload = {
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "sub": "test_user",
        }
        token = jwt.encode(payload, "secret", algorithm="HS256")

        result = validator.validate_token(token)
        assert result.valid
        assert result.payload["_development_mode"] is True
        assert result.payload["_max_validity"] == validator.DEVELOPMENT_TOKEN_LIFETIME
        assert "DEVELOPMENT MODE" in result.warnings[0]


class TestJWTEnvironmentDetection:
    """Test automatic environment detection."""

    def test_environment_detection(self, test_environments, clean_environment):
        """Test that environment is properly detected."""
        # Test production
        os.environ.update(test_environments["production"])
        validator = get_secure_jwt_validator()
        assert validator.validation_mode == ValidationMode.PRODUCTION

        # Clean and test staging
        for key in test_environments["production"]:
            os.environ.pop(key, None)
        os.environ.update(test_environments["staging"])
        validator = get_secure_jwt_validator()
        assert validator.validation_mode == ValidationMode.STAGING

        # Clean and test development
        for key in test_environments["staging"]:
            os.environ.pop(key, None)
        os.environ.update(test_environments["development"])
        validator = get_secure_jwt_validator()
        assert validator.validation_mode == ValidationMode.DEVELOPMENT

    def test_unknown_environment_defaults_to_production(self, clean_environment):
        """Test that unknown environment defaults to production for safety."""
        os.environ["TRAIGENT_ENVIRONMENT"] = "unknown"
        os.environ["TRAIGENT_JWKS_URL"] = "https://example.com/jwks"
        os.environ["TRAIGENT_JWT_ISSUER"] = "issuer"
        os.environ["TRAIGENT_JWT_AUDIENCE"] = "audience"

        with patch("traigent.security.jwt_validator.logger") as mock_logger:
            validator = get_secure_jwt_validator()
            assert validator.validation_mode == ValidationMode.PRODUCTION
            mock_logger.warning.assert_called()


class TestJWTSecurityFeatures:
    """Test security features across all modes."""

    def test_bypass_prevention(self, clean_environment):
        """Test that bypass attempts are prevented."""
        # Test bypass environment variable
        os.environ["TRAIGENT_JWT_BYPASS"] = "true"
        with pytest.raises(JWTSecurityError) as exc:
            SecureJWTValidator(
                jwks_url="https://example.com/jwks",
                issuer="issuer",
                audience="audience",
                validation_mode=ValidationMode.PRODUCTION,
            )
        assert "bypass" in str(exc.value).lower()

        # Test disable security environment variable
        os.environ.pop("TRAIGENT_JWT_BYPASS")
        os.environ["TRAIGENT_JWT_DISABLE_SECURITY"] = "true"
        with pytest.raises(JWTSecurityError) as exc:
            get_secure_jwt_validator()
        assert "bypass" in str(exc.value).lower()

    def test_token_size_limits(self, clean_environment):
        """Test that oversized tokens are rejected."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        # Create oversized token
        large_payload = {
            "data": "x" * 10000,
            "iat": int(time.time()),
        }
        large_token = jwt.encode(large_payload, "secret", algorithm="HS256")

        with pytest.raises(JWTSecurityError) as exc:
            validator.validate_token(large_token)
        assert "exceeds maximum allowed size" in str(exc.value)

    def test_constant_time_validation(self, clean_environment):
        """Test constant-time validation to prevent timing attacks."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        valid_token = jwt.encode({"iat": int(time.time())}, "secret", algorithm="HS256")

        # Test that constant_time_validate works
        result = validator.constant_time_validate(valid_token, True)
        assert isinstance(result, bool)

    def test_validation_metrics(self, clean_environment):
        """Test that validation metrics are tracked."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        # Initial metrics
        metrics = validator.get_validation_metrics()
        assert metrics["total_validations"] == 0

        # Valid token
        token = jwt.encode({"iat": int(time.time())}, "secret", algorithm="HS256")
        validator.validate_token(token)

        # Invalid token
        validator.validate_token("invalid")

        # Check updated metrics
        metrics = validator.get_validation_metrics()
        assert metrics["total_validations"] == 2
        assert metrics["successful_validations"] == 1
        assert metrics["failed_validations"] == 1

    @patch("traigent.security.jwt_validator.logger")
    def test_suspicious_claims_detection(self, mock_logger, clean_environment):
        """Test detection of suspicious claims."""
        validator = SecureJWTValidator(
            jwks_url="https://example.com/jwks",
            issuer="issuer",
            audience="audience",
            validation_mode=ValidationMode.PRODUCTION,
        )

        suspicious_payload = {
            "admin": True,
            "superuser": True,
            "root": True,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }

        with patch.object(validator, "_get_jwks_client"):
            with patch("traigent.security.jwt_validator.jwt.decode") as mock_decode:
                mock_decode.return_value = suspicious_payload

                validator._perform_security_checks(suspicious_payload)

                # Check that warnings were logged
                assert (
                    mock_logger.warning.call_count >= 3
                )  # One for each suspicious claim


class TestJWTPerformance:
    """Test JWT validation performance."""

    @pytest.mark.integration
    def test_validation_performance(self, clean_environment):
        """Test that validation performs well under load."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        # Create test token
        token = jwt.encode(
            {"iat": int(time.time()), "sub": "test"},
            "secret",
            algorithm="HS256",
        )

        # Validate many tokens
        start_time = time.time()
        for _ in range(100):
            validator.validate_token(token)
        elapsed = time.time() - start_time

        # Should handle 100 validations quickly
        assert elapsed < 1.0  # Less than 1 second for 100 tokens

        # Check metrics
        metrics = validator.get_validation_metrics()
        assert metrics["total_validations"] == 100

    def test_jti_cache_management(self, clean_environment):
        """Test JWT ID cache management."""
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            require_jti=True,
        )

        # Add JTIs
        with validator._jti_lock:
            validator._seen_jti.add("jti1")
            validator._seen_jti.add("jti2")
            validator._seen_jti.add("jti3")

        assert len(validator._seen_jti) == 3

        # Clear cache
        validator.clear_jti_cache()
        assert len(validator._seen_jti) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
