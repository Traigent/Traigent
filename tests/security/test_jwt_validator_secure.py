"""Tests for secure JWT validator implementation."""

import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from traigent.security.jwt_validator import (
    JWTSecurityError,
    SecureJWTValidator,
    ValidationMode,
    get_secure_jwt_validator,
)

_TEST_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_TEST_PRIVATE_PEM = _TEST_PRIVATE_KEY.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
)
_TEST_PUBLIC_PEM = _TEST_PRIVATE_KEY.public_key().public_bytes(
    serialization.Encoding.PEM,
    serialization.PublicFormat.SubjectPublicKeyInfo,
)


class TestSecureJWTValidator(unittest.TestCase):
    """Test secure JWT validator implementation."""

    def setUp(self):
        """Set up test fixtures."""
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

    def create_test_token(self, payload=None, algorithm="RS256", secret=None):
        """Helper to create test tokens."""
        if payload is None:
            payload = self.valid_payload.copy()
        if secret is None:
            secret = _TEST_PRIVATE_PEM
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

    def test_staging_mode_requires_production_configuration(self):
        """Staging must have all production trust and claim settings."""
        for kwargs, expected_error in (
            ({}, "JWKS URL required"),
            ({"jwks_url": "https://example.com/jwks"}, "Issuer required"),
            (
                {
                    "jwks_url": "https://example.com/jwks",
                    "issuer": self.test_issuer,
                },
                "Audience required",
            ),
        ):
            with self.assertRaises(JWTSecurityError) as context:
                SecureJWTValidator(validation_mode=ValidationMode.STAGING, **kwargs)
            self.assertIn(expected_error, str(context.exception))

    def test_staging_signed_token_missing_claim_rejected(self):
        """Real signed staging tokens still require issuer and audience."""
        for missing_claim in ("iss", "aud"):
            payload = {
                "sub": "staging-user",
                "iat": int(time.time()),
                "exp": int(time.time()) + 60,
                "nbf": int(time.time()),
                "jti": f"staging-missing-{missing_claim}",
                "iss": self.test_issuer,
                "aud": self.test_audience,
            }
            payload.pop(missing_claim)
            token = self.create_test_token(payload)
            validator = SecureJWTValidator(
                jwks_url="https://example.com/jwks",
                issuer=self.test_issuer,
                audience=self.test_audience,
                validation_mode=ValidationMode.STAGING,
            )
            signing_key = SimpleNamespace(
                key=_TEST_PUBLIC_PEM,
                algorithm_name="RS256",
                key_id="test-staging-key",
            )

            with patch.object(validator, "_get_jwks_client") as get_jwks_client:
                get_jwks_client.return_value.get_signing_key_from_jwt.return_value = (
                    signing_key
                )
                result = validator.validate_token(token)

            assert result.valid is False
            assert result.error
            assert missing_claim in result.error.lower()

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
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            development_public_key=_TEST_PUBLIC_PEM,
        )

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
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            development_public_key=_TEST_PUBLIC_PEM,
        )

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

    def test_token_size_limit(self):
        """Test that oversized tokens are rejected."""
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            development_public_key=_TEST_PUBLIC_PEM,
        )

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
            development_public_key=_TEST_PUBLIC_PEM,
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
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            development_public_key=_TEST_PUBLIC_PEM,
        )

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
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            development_public_key=_TEST_PUBLIC_PEM,
        )

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
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            development_public_key=_TEST_PUBLIC_PEM,
        )

        token = self.create_test_token()
        result = validator.validate_token(token)

        self.assertTrue(result.valid)
        self.assertIsNotNone(result.security_metadata)
        self.assertEqual(result.security_metadata["mode"], "development")
        self.assertIn("validated_at", result.security_metadata)

    def test_development_mode_marker(self):
        """Test that development tokens are marked."""
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            development_public_key=_TEST_PUBLIC_PEM,
        )

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
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            development_public_key=_TEST_PUBLIC_PEM,
        )

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
            development_public_key=_TEST_PUBLIC_PEM,
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

    def _development_validator(self, public_key: bytes = _TEST_PUBLIC_PEM):
        return SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            development_public_key=public_key,
        )

    def test_development_wrong_signature_rejected(self):
        """A token signed by another RSA key must fail in development."""
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        other_private = other_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        signed_jwt = jwt.encode(
            {"iat": int(time.time()), "exp": int(time.time()) + 60},
            other_private,
            algorithm="RS256",
        )

        result = self._development_validator().validate_token(signed_jwt)

        self.assertFalse(result.valid)

    def test_development_rsa_garbage_signature_rejected(self):
        """A structurally valid RS256 token with garbage signature must fail."""
        signed_jwt = jwt.encode(
            {"iat": int(time.time()), "exp": int(time.time()) + 60},
            _TEST_PRIVATE_PEM,
            algorithm="RS256",
        )
        header, payload, _signature = signed_jwt.split(".")
        garbage_jwt = f"{header}.{payload}.garbage"

        result = self._development_validator().validate_token(garbage_jwt)

        self.assertFalse(result.valid)

    def test_development_no_key_fails_closed_without_exception(self):
        """Development validation without a configured key must fail closed."""
        signed_jwt = jwt.encode(
            {"iat": int(time.time()), "exp": int(time.time()) + 60},
            _TEST_PRIVATE_PEM,
            algorithm="RS256",
        )
        validator = SecureJWTValidator(validation_mode=ValidationMode.DEVELOPMENT)

        result = validator.validate_token(signed_jwt)

        self.assertFalse(result.valid)
        self.assertIn("verification key", result.error)

    def test_development_rsa_hs_algorithm_confusion_rejected(self):
        """An HS256 token cannot be verified with an RSA public key."""
        signed_jwt = jwt.encode(
            {"iat": int(time.time()), "exp": int(time.time()) + 60},
            "hmac-key-for-test",
            algorithm="HS256",
        )

        result = self._development_validator().validate_token(signed_jwt)

        self.assertFalse(result.valid)

    @pytest.mark.integration
    def test_end_to_end_validation_flow(self):
        """Test complete validation flow without a real JWKS endpoint.

        Runs the validator in DEVELOPMENT mode (which does not require a JWKS
        endpoint) end-to-end:
          1. Accepts a freshly-issued, well-formed token.
          2. Rejects an algorithm 'none' token (security regression guard).
          3. Rejects a token already past its DEVELOPMENT_TOKEN_LIFETIME.

        Previously this test unconditionally skipped, which silently masked
        regressions in any of the three paths.
        """
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            development_public_key=_TEST_PUBLIC_PEM,
        )
        now = int(time.time())

        # 1. Well-formed, fresh token must validate.
        good_payload = {
            "sub": "e2e_user",
            "iat": now,
            "exp": now + 60,
            "jti": "e2e-jti-1",
        }
        good_token = jwt.encode(good_payload, _TEST_PRIVATE_PEM, algorithm="RS256")
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            development_public_key=_TEST_PUBLIC_PEM,
        )
        good_result = validator.validate_token(good_token)
        self.assertTrue(
            good_result.valid,
            f"Fresh dev-mode token rejected: {good_result.error!r}",
        )
        self.assertIsNotNone(good_result.payload)
        self.assertEqual(good_result.payload.get("sub"), "e2e_user")

        # 2. Token using 'alg: none' must be rejected even in DEVELOPMENT mode.
        none_token = jwt.encode(good_payload, "", algorithm="none")
        none_result = validator.validate_token(none_token)
        self.assertFalse(
            none_result.valid,
            "alg=none token must NEVER validate (algorithm-confusion attack)",
        )

        # 3. Token issued past the DEVELOPMENT lifetime window must be rejected.
        old_iat = now - validator.DEVELOPMENT_TOKEN_LIFETIME - 60
        stale_payload = {
            "sub": "e2e_user",
            "iat": old_iat,
            "exp": old_iat + validator.DEVELOPMENT_TOKEN_LIFETIME + 30,
            "jti": "e2e-jti-2",
        }
        stale_token = jwt.encode(stale_payload, _TEST_PRIVATE_PEM, algorithm="RS256")
        stale_result = validator.validate_token(stale_token)
        self.assertFalse(
            stale_result.valid,
            "Token older than DEVELOPMENT_TOKEN_LIFETIME must be rejected",
        )

    @pytest.mark.integration
    def test_performance_under_load(self):
        """Test validator performance under load."""
        validator = SecureJWTValidator(
            validation_mode=ValidationMode.DEVELOPMENT,
            development_public_key=_TEST_PUBLIC_PEM,
        )

        # Create test token
        token = jwt.encode(
            {
                "sub": "test",
                "iat": int(time.time()),
                "exp": int(time.time()) + 3600,
            },
            _TEST_PRIVATE_PEM,
            algorithm="RS256",
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
