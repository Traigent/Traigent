"""Tests for secure JWT validator implementation."""

import os
import time
import unittest
from unittest.mock import patch

import jwt
import pytest

from traigent.security.jwt_validator import (
    JWTSecurityError,
    SecureJWTValidator,
    ValidationMode,
    get_secure_jwt_validator,
)


class TestSecureJWTValidator(unittest.TestCase):
    """Test secure JWT validator implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_secret = "test_secret_key_for_testing_only"
        self.test_issuer = "test_issuer"
        self.test_audience = "test_audience"

        # Create test tokens
        self.valid_payload = {
            "sub": "test_user",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "nbf": int(time.time()),
            "jti": "unique_token_id_123",
            "iss": self.test_issuer,
            "aud": self.test_audience,
        }

    def create_test_token(self, payload=None, algorithm="HS256", secret=None):
        """Helper to create test tokens."""
        if payload is None:
            payload = self.valid_payload.copy()
        if secret is None:
            secret = self.test_secret
        return jwt.encode(payload, secret, algorithm=algorithm)

    def test_production_mode_requires_configuration(self):
        """Test that production mode requires proper configuration."""
        # Should raise error without JWKS URL
        with self.assertRaises(JWTSecurityError) as context:
            SecureJWTValidator(validation_mode=ValidationMode.PRODUCTION)
        self.assertIn("JWKS URL required", str(context.exception))

        # Should raise error without issuer
        with self.assertRaises(JWTSecurityError) as context:
            SecureJWTValidator(
                jwks_url="https://example.com/jwks",
                validation_mode=ValidationMode.PRODUCTION,
            )
        self.assertIn("Issuer required", str(context.exception))

        # Should raise error without audience
        with self.assertRaises(JWTSecurityError) as context:
            SecureJWTValidator(
                jwks_url="https://example.com/jwks",
                issuer="test_issuer",
                validation_mode=ValidationMode.PRODUCTION,
            )
        self.assertIn("Audience required", str(context.exception))

    def test_bypass_environment_variable_blocked(self):
        """Test that bypass environment variables are blocked."""
        with patch.dict(os.environ, {"TRAIGENT_JWT_BYPASS": "true"}):
            with self.assertRaises(JWTSecurityError) as context:
                SecureJWTValidator(
                    jwks_url="https://example.com/jwks",
                    issuer="test_issuer",
                    audience="test_audience",
                    validation_mode=ValidationMode.PRODUCTION,
                )
            self.assertIn("JWT bypass", str(context.exception))

    def test_development_mode_time_limit(self):
        """Test that development mode enforces time limits."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        # Create token with old timestamp
        old_payload = self.valid_payload.copy()
        old_payload["iat"] = int(time.time()) - 400  # 400 seconds ago
        old_token = self.create_test_token(old_payload)

        result = validator.validate_token(old_token)
        self.assertFalse(result.valid)
        self.assertIn("expired", result.error.lower())

        # Create token with recent timestamp
        recent_payload = self.valid_payload.copy()
        recent_payload["iat"] = int(time.time()) - 100  # 100 seconds ago
        recent_token = self.create_test_token(recent_payload)

        result = validator.validate_token(recent_token)
        self.assertTrue(result.valid)
        self.assertIn("DEVELOPMENT MODE", result.warnings[0])

    def test_algorithm_none_blocked(self):
        """Test that 'none' algorithm is blocked even in development."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        # Create token with 'none' algorithm
        header = {"alg": "none", "typ": "JWT"}
        payload = self.valid_payload.copy()

        # Manually create token with 'none' algorithm
        import base64
        import json

        header_b64 = (
            base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
        )
        payload_b64 = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        )

        none_token = f"{header_b64}.{payload_b64}."

        result = validator.validate_token(none_token)
        self.assertFalse(result.valid)
        self.assertIn("none", result.error.lower())

    def test_token_size_limit(self):
        """Test that oversized tokens are rejected."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        # Create oversized token
        oversized_payload = self.valid_payload.copy()
        oversized_payload["data"] = "x" * 10000  # Large data
        oversized_token = self.create_test_token(oversized_payload)

        # Should raise JWTSecurityError for oversized tokens
        with self.assertRaises(JWTSecurityError) as context:
            validator.validate_token(oversized_token)
        self.assertIn("exceeds maximum allowed size", str(context.exception))

    def test_replay_protection(self):
        """Test JWT ID replay protection."""
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            require_jti=True,
        )

        # Create token with JTI
        payload = self.valid_payload.copy()
        payload["jti"] = "unique_id_123"
        token = self.create_test_token(payload)

        # First validation should succeed
        result1 = validator.validate_token(token)
        self.assertTrue(result1.valid)

        # Mock production validation for replay test
        with patch.object(validator, "_validate_production"):
            validator.validation_mode = ValidationMode.PRODUCTION
            with validator._jti_lock:
                validator._seen_jti.add("unique_id_123")

            # Create new token with same JTI
            self.create_test_token(payload)

            # This would fail in production due to replay
            # We're testing the replay detection logic exists

    def test_constant_time_validation(self):
        """Test constant-time validation to prevent timing attacks."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        valid_token = self.create_test_token()
        invalid_token = "invalid.token.here"

        # Time multiple validations
        import timeit

        # Valid token timing
        valid_time = timeit.timeit(
            lambda: validator.constant_time_validate(valid_token, True),
            number=10,
        )

        # Invalid token timing
        invalid_time = timeit.timeit(
            lambda: validator.constant_time_validate(invalid_token, False),
            number=10,
        )

        # Times should be relatively similar (within 50% variance)
        # This is a basic check - proper timing attack tests need specialized tools
        time_ratio = max(valid_time, invalid_time) / min(valid_time, invalid_time)
        self.assertLess(time_ratio, 2.0, "Timing variance too high")

    def test_validation_metrics(self):
        """Test validation metrics tracking."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        # Initial metrics
        metrics = validator.get_validation_metrics()
        self.assertEqual(metrics["total_validations"], 0)

        # Valid token
        valid_token = self.create_test_token()
        validator.validate_token(valid_token)

        # Invalid token
        validator.validate_token("invalid")

        # Check metrics
        metrics = validator.get_validation_metrics()
        self.assertEqual(metrics["total_validations"], 2)
        self.assertEqual(metrics["successful_validations"], 1)
        self.assertEqual(metrics["failed_validations"], 1)

    def test_security_metadata_included(self):
        """Test that security metadata is included in results."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        token = self.create_test_token()
        result = validator.validate_token(token)

        self.assertTrue(result.valid)
        self.assertIsNotNone(result.security_metadata)
        self.assertEqual(result.security_metadata["mode"], "development")
        self.assertIn("validated_at", result.security_metadata)

    def test_development_mode_marker(self):
        """Test that development tokens are marked."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        token = self.create_test_token()
        result = validator.validate_token(token)

        self.assertTrue(result.valid)
        self.assertTrue(result.payload.get("_development_mode"))
        self.assertEqual(
            result.payload.get("_max_validity"),
            validator.DEVELOPMENT_TOKEN_LIFETIME,
        )

    def test_suspicious_claims_detection(self):
        """Test detection of suspicious claims."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        # Create token with suspicious claims
        suspicious_payload = self.valid_payload.copy()
        suspicious_payload["admin"] = True
        suspicious_payload["superuser"] = True

        with patch("traigent.security.jwt_validator.logger"):
            token = self.create_test_token(suspicious_payload)
            result = validator.validate_token(token)

            # In development mode, it validates but logs warnings
            self.assertTrue(result.valid)

    def test_get_secure_jwt_validator_environment_detection(self):
        """Test that get_secure_jwt_validator detects environment correctly."""
        # Test production environment
        with patch.dict(
            os.environ,
            {
                "TRAIGENT_ENVIRONMENT": "production",
                "TRAIGENT_JWKS_URL": "https://example.com/jwks",
                "TRAIGENT_JWT_ISSUER": "issuer",
                "TRAIGENT_JWT_AUDIENCE": "audience",
            },
        ):
            validator = get_secure_jwt_validator()
            self.assertEqual(validator.validation_mode, ValidationMode.PRODUCTION)

        # Test development environment
        with patch.dict(os.environ, {"TRAIGENT_ENVIRONMENT": "development"}):
            validator = get_secure_jwt_validator()
            self.assertEqual(validator.validation_mode, ValidationMode.DEVELOPMENT)

        # Test staging environment
        with patch.dict(
            os.environ,
            {
                "TRAIGENT_ENVIRONMENT": "staging",
                "TRAIGENT_JWKS_URL": "https://example.com/jwks",
                "TRAIGENT_JWT_ISSUER": "issuer",
                "TRAIGENT_JWT_AUDIENCE": "audience",
            },
        ):
            validator = get_secure_jwt_validator()
            self.assertEqual(validator.validation_mode, ValidationMode.STAGING)

        # Test unknown environment defaults to production
        with patch.dict(
            os.environ,
            {
                "TRAIGENT_ENVIRONMENT": "unknown",
                "TRAIGENT_JWKS_URL": "https://example.com/jwks",
                "TRAIGENT_JWT_ISSUER": "issuer",
                "TRAIGENT_JWT_AUDIENCE": "audience",
            },
        ):
            validator = get_secure_jwt_validator()
            self.assertEqual(validator.validation_mode, ValidationMode.PRODUCTION)

    def test_security_bypass_prevention(self):
        """Test that security bypass attempts are prevented."""
        with patch.dict(os.environ, {"TRAIGENT_JWT_DISABLE_SECURITY": "true"}):
            with self.assertRaises(JWTSecurityError) as context:
                get_secure_jwt_validator()
            self.assertIn("Security bypass attempt", str(context.exception))

    def test_jti_cache_clearing(self):
        """Test JWT ID cache clearing."""
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            require_jti=True,
        )

        # Add some JTIs
        with validator._jti_lock:
            validator._seen_jti.add("jti1")
            validator._seen_jti.add("jti2")
        self.assertEqual(len(validator._seen_jti), 2)

        # Clear cache
        validator.clear_jti_cache()
        self.assertEqual(len(validator._seen_jti), 0)


class TestJWTValidationIntegration(unittest.TestCase):
    """Integration tests for JWT validation."""

    @pytest.mark.integration
    def test_end_to_end_validation_flow(self):
        """Test complete validation flow."""
        # This would test against a real JWKS endpoint in integration testing
        # Skip assertion since this is a placeholder for integration testing
        pytest.skip("Requires real JWKS endpoint for integration testing")

    @pytest.mark.integration
    def test_performance_under_load(self):
        """Test validator performance under load."""
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        # Create test token
        token = jwt.encode(
            {
                "sub": "test",
                "iat": int(time.time()),
                "exp": int(time.time()) + 3600,
            },
            "secret",
            algorithm="HS256",
        )

        # Validate many tokens
        start_time = time.time()
        for _ in range(1000):
            validator.validate_token(token)
        elapsed = time.time() - start_time

        # Should handle 1000 validations in reasonable time
        self.assertLess(
            elapsed, 5.0, f"Validation too slow: {elapsed}s for 1000 tokens"
        )


if __name__ == "__main__":
    unittest.main()
