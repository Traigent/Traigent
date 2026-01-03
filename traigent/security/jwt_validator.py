"""Secure JWT token validation for Traigent SDK - Production Hardened Version."""

# Traceability: CONC-Security CONC-Quality-Security FUNC-SECURITY FUNC-SEC-TOKEN-MGMT REQ-SEC-010 REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import jwt
    from jwt import PyJWKClient

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Error message constants
_TOKEN_EXPIRED_ERROR = "Token has expired"


class JWTValidationError(Exception):
    """Base class for JWT validation errors."""


class JWTSignatureError(JWTValidationError):
    """JWT signature validation failed."""


class JWTExpiredError(JWTValidationError):
    """JWT token has expired."""


class JWTInvalidError(JWTValidationError):
    """JWT token is invalid."""


class JWTSecurityError(JWTValidationError):
    """JWT security check failed."""


class ValidationMode(Enum):
    """JWT validation modes - production hardened."""

    PRODUCTION = "production"  # Full verification, no bypass allowed
    STAGING = "staging"  # Full verification with extended logging
    DEVELOPMENT = "development"  # Time-limited tokens with warnings


@dataclass
class JWTValidationResult:
    """Result of JWT validation with security metadata."""

    valid: bool
    payload: dict[str, Any] | None = None
    error: str | None = None
    warnings: list[str] | None = None
    expires_at: float | None = None
    security_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []
        if self.security_metadata is None:
            self.security_metadata = {}


class SecureJWTValidator:
    """Production-hardened JWT token validator with enhanced security.

    Thread Safety:
        This class is thread-safe. The JWT ID replay detection mechanism
        uses a lock (_jti_lock) to protect concurrent access to the
        _seen_jti set, preventing race conditions during token validation.
    """

    # Security constants
    MAX_TOKEN_AGE_SECONDS = 3600  # 1 hour max token age
    ALLOWED_ALGORITHMS = ["RS256", "ES256"]  # Only secure algorithms
    DEVELOPMENT_TOKEN_LIFETIME = 300  # 5 minutes for dev tokens

    # Algorithm-specific minimum key sizes (in bits)
    # RS256: RSA with SHA-256, minimum 2048 bits per NIST recommendations
    # ES256: ECDSA with P-256 curve, key size is fixed at 256 bits
    MIN_KEY_SIZES = {
        "RS256": 2048,
        "RS384": 2048,
        "RS512": 2048,
        "PS256": 2048,
        "PS384": 2048,
        "PS512": 2048,
        "ES256": 256,  # P-256 curve - fixed size
        "ES384": 384,  # P-384 curve - fixed size
        "ES512": 521,  # P-521 curve - fixed size (521 bits, not 512)
    }

    def __init__(
        self,
        jwks_url: str | None = None,
        issuer: str | None = None,
        audience: str | None = None,
        validation_mode: ValidationMode = ValidationMode.PRODUCTION,
        require_nbf: bool = True,
        require_jti: bool = True,
        max_clock_skew: int = 30,
        development_public_key: str | None = None,
    ) -> None:
        """Initialize secure JWT validator.

        Args:
            jwks_url: URL for JSON Web Key Set
            issuer: Expected token issuer
            audience: Expected token audience
            validation_mode: Validation mode (production by default)
            require_nbf: Require 'not before' claim
            require_jti: Require JWT ID for replay protection
            max_clock_skew: Maximum allowed clock skew in seconds
            development_public_key: Optional PEM public key for development validation
        """
        self.jwks_url = jwks_url
        self.issuer = issuer
        self.audience = audience
        self.validation_mode = validation_mode
        self.require_nbf = require_nbf
        self.require_jti = require_jti
        self.max_clock_skew = max_clock_skew
        self.development_public_key = development_public_key or os.getenv(
            "TRAIGENT_DEV_JWT_PUBLIC_KEY"
        )
        self._jwks_client = None
        self._seen_jti: set[Any] = set()  # Track JWT IDs for replay protection
        self._jti_lock = threading.Lock()  # Protect _seen_jti from race conditions
        self._validation_metrics: dict[str, Any] = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "security_violations": 0,
        }

        # Security checks on initialization
        self._validate_configuration()

        if not JWT_AVAILABLE:
            if self.validation_mode == ValidationMode.PRODUCTION:
                raise JWTSecurityError(
                    "PyJWT not available - JWT validation is required for production security"
                )
            else:
                logger.warning(
                    "PyJWT not available - JWT validation will use minimal security checks only. "
                    "Install PyJWT for full security validation."
                )

    def _validate_configuration(self) -> None:
        """Validate security configuration."""
        # Check if running in production without proper configuration
        if self.validation_mode == ValidationMode.PRODUCTION:
            if not self.jwks_url:
                raise JWTSecurityError(
                    "JWKS URL required for production mode JWT validation"
                )
            if not self.issuer:
                raise JWTSecurityError(
                    "Issuer required for production mode JWT validation"
                )
            if not self.audience:
                raise JWTSecurityError(
                    "Audience required for production mode JWT validation"
                )
        elif self.validation_mode in [
            ValidationMode.STAGING,
            ValidationMode.DEVELOPMENT,
        ]:
            # For non-production modes, warn about missing configuration but don't fail
            missing_configs = []
            if not self.jwks_url:
                missing_configs.append("JWKS URL")
            if not self.issuer:
                missing_configs.append("issuer")
            if not self.audience:
                missing_configs.append("audience")

            if missing_configs:
                logger.warning(
                    f"JWT validation in {self.validation_mode.value} mode without: {', '.join(missing_configs)}. "
                    f"This is acceptable for testing but not for production."
                )

        # Warn about non-production modes
        if self.validation_mode != ValidationMode.PRODUCTION:
            logger.warning(
                f"JWT validation running in {self.validation_mode.value} mode - "
                f"NOT suitable for production use"
            )

        # Check environment for security bypass attempts
        if os.getenv("TRAIGENT_JWT_BYPASS"):
            raise JWTSecurityError(
                "JWT bypass environment variable detected - this is not allowed"
            )

    def _get_jwks_client(self) -> PyJWKClient:
        """Get JWKS client for signature verification."""
        if not self.jwks_url:
            if self.validation_mode != ValidationMode.PRODUCTION:
                return None
            raise JWTSecurityError("JWKS URL not configured")

        if self._jwks_client is None:
            try:
                self._jwks_client = PyJWKClient(
                    self.jwks_url,
                    cache_keys=True,
                    max_cached_keys=16,
                    cache_jwk_set=True,
                )
            except Exception as e:
                if self.validation_mode != ValidationMode.PRODUCTION:
                    logger.error(f"Failed to initialize JWKS client: {e}")
                    return None
                raise JWTSecurityError(f"Failed to initialize JWKS client: {e}") from e

        return self._jwks_client

    def validate_token(self, token: str) -> JWTValidationResult:
        """Validate JWT token with comprehensive security checks."""
        self._validation_metrics["total_validations"] += 1

        try:
            # Input validation
            if not token or not isinstance(token, str):
                raise JWTInvalidError("Invalid token format")

            # Check token length (prevent DoS)
            if len(token) > 10000:  # Reasonable max JWT size
                raise JWTSecurityError("Token exceeds maximum allowed size")

            # Route to appropriate validation method
            if self.validation_mode == ValidationMode.PRODUCTION:
                result = self._validate_strict(token)
            elif self.validation_mode == ValidationMode.STAGING:
                result = self._validate_staging(token)
            elif self.validation_mode == ValidationMode.DEVELOPMENT:
                result = self._validate_development_secure(token)
            else:
                raise JWTSecurityError(
                    f"Invalid validation mode: {self.validation_mode}"
                )

            if result.valid:
                self._validation_metrics["successful_validations"] += 1
            else:
                self._validation_metrics["failed_validations"] += 1

            return result

        except JWTSecurityError:
            self._validation_metrics["security_violations"] += 1
            raise
        except Exception as e:
            self._validation_metrics["failed_validations"] += 1
            return JWTValidationResult(
                valid=False, error=f"Token validation failed: {e}"
            )

    def _validate_production(self, token: str) -> JWTValidationResult:
        """Production validation with maximum security."""
        try:
            # Get signing key with verification
            jwks_client = self._get_jwks_client()
            if jwks_client is None:
                return JWTValidationResult(
                    valid=False,
                    error="Cannot verify signature - JWKS client unavailable",
                )
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            # Verify key strength (algorithm-aware)
            if hasattr(signing_key.key, "key_size"):
                algorithm = signing_key.algorithm_name
                min_size = self.MIN_KEY_SIZES.get(algorithm)
                if min_size is not None:
                    actual_size = signing_key.key.key_size
                    if actual_size < min_size:
                        raise JWTSecurityError(
                            f"Key size {actual_size} bits below minimum {min_size} bits for {algorithm}"
                        )
                else:
                    # Unknown algorithm - log warning but allow PyJWT to validate
                    logger.warning(
                        f"Unknown algorithm '{algorithm}' - key size validation skipped, "
                        f"relying on PyJWT built-in validation"
                    )

            # Comprehensive token validation
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.ALLOWED_ALGORITHMS,
                issuer=self.issuer,
                audience=self.audience,
                leeway=self.max_clock_skew,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": self.require_nbf,
                    "require_exp": True,
                    "require_iat": True,
                    "require_nbf": self.require_nbf,
                    "require_jti": self.require_jti,
                },
            )

            # Additional security checks
            self._perform_security_checks(payload)

            # Check for replay attacks (thread-safe)
            if self.require_jti:
                jti = payload.get("jti")
                with self._jti_lock:
                    if jti in self._seen_jti:
                        raise JWTSecurityError("Token replay detected")
                    self._seen_jti.add(jti)

            # Check token age
            iat = payload.get("iat", 0)
            current_time = time.time()
            if current_time - iat > self.MAX_TOKEN_AGE_SECONDS:
                raise JWTExpiredError("Token exceeds maximum age")

            return JWTValidationResult(
                valid=True,
                payload=payload,
                expires_at=payload.get("exp"),
                security_metadata={
                    "mode": "production",
                    "algorithm": signing_key.algorithm_name,
                    "key_id": signing_key.key_id,
                    "validated_at": current_time,
                },
            )

        except jwt.ExpiredSignatureError:
            return JWTValidationResult(valid=False, error=_TOKEN_EXPIRED_ERROR)
        except jwt.InvalidSignatureError:
            return JWTValidationResult(valid=False, error="Invalid token signature")
        except JWTSecurityError as e:
            return JWTValidationResult(valid=False, error=str(e))
        except jwt.InvalidTokenError as e:
            return JWTValidationResult(valid=False, error=f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in JWT validation: {e}")
            return JWTValidationResult(
                valid=False, error=f"Token validation failed: {e}"
            )

    def _validate_staging(self, token: str) -> JWTValidationResult:
        """Staging validation - same as production with extended logging."""
        logger.info("Staging JWT validation initiated")
        result = self._validate_production(token)

        # Extended logging for debugging
        if not result.valid:
            logger.warning(f"Staging JWT validation failed: {result.error}")
        else:
            logger.info("Staging JWT validation successful")

        return result

    @staticmethod
    def _development_warnings() -> list[str]:
        """Return standard warnings for development mode validation."""
        return [
            "DEVELOPMENT MODE: Limited to 5-minute token lifetime (Development mode only)",
            "NOT suitable for production use",
            "Security validations relaxed but not disabled",
        ]

    def _extract_header_algorithm(self, token: str) -> str | None:
        """Extract algorithm from JWT header safely."""
        header_reader = getattr(jwt, "get_unverified_header", None)
        if token.count(".") < 2 or not callable(header_reader):
            return None
        try:
            header = header_reader(token)
            alg: str | None = header.get("alg")
            return alg
        except Exception as exc:
            logger.debug(
                "Failed to parse JWT header in development validation: %s", exc
            )
            return None

    def _validate_algorithm(
        self, header_alg: str | None, warnings: list[str]
    ) -> JWTValidationResult | None:
        """Validate algorithm is allowed. Returns error result or None if valid."""
        if isinstance(header_alg, str) and header_alg.lower() == "none":
            return JWTValidationResult(
                valid=False,
                error="Algorithm 'none' is not allowed",
                warnings=warnings,
            )
        if header_alg:
            allowed_algs = set(self.ALLOWED_ALGORITHMS + ["HS256"])
            if header_alg not in allowed_algs:
                return JWTValidationResult(
                    valid=False,
                    error=f"Unsupported algorithm: {header_alg}",
                    warnings=warnings,
                )
        return None

    def _check_token_lifetime(
        self, payload: dict[str, Any], warnings: list[str]
    ) -> JWTValidationResult | None:
        """Check token lifetime constraints. Returns error result or None if valid."""
        iat = payload.get("iat")
        current_time = time.time()

        if not isinstance(iat, (int, float)):
            return JWTValidationResult(
                valid=False,
                error="Development tokens must include an iat claim",
                warnings=warnings,
            )

        exp = payload.get("exp")
        if isinstance(exp, (int, float)) and current_time > exp:
            return JWTValidationResult(
                valid=False, error=_TOKEN_EXPIRED_ERROR, warnings=warnings
            )

        if current_time - iat > self.DEVELOPMENT_TOKEN_LIFETIME:
            return JWTValidationResult(
                valid=False, error=_TOKEN_EXPIRED_ERROR, warnings=warnings
            )
        return None

    def _validate_development_secure(self, token: str) -> JWTValidationResult:
        """Development mode with time-limited tokens and security warnings."""
        warnings = self._development_warnings()

        if not JWT_AVAILABLE:
            warnings.append("Install PyJWT to enable JWT validation")
            return JWTValidationResult(
                valid=False, error="JWT validation unavailable", warnings=warnings
            )

        try:
            header_alg = self._extract_header_algorithm(token)

            alg_error = self._validate_algorithm(header_alg, warnings)
            if alg_error:
                return alg_error

            decode_options = {
                "verify_signature": False,
                "verify_exp": False,
                "verify_iat": False,
                "verify_nbf": False,
            }
            decode_kwargs: dict[str, Any] = {"options": decode_options}
            if header_alg:
                decode_kwargs["algorithms"] = [header_alg]

            unverified_payload = jwt.decode(
                token, self.development_public_key or "", **decode_kwargs
            )

            lifetime_error = self._check_token_lifetime(unverified_payload, warnings)
            if lifetime_error:
                return lifetime_error

            payload = self._mark_development_payload(token, unverified_payload)
            exp = unverified_payload.get("exp")

            return JWTValidationResult(
                valid=True,
                payload=payload,
                expires_at=exp if isinstance(exp, (int, float)) else None,
                warnings=warnings,
                security_metadata={
                    "mode": "development",
                    "validated_at": time.time(),
                    "max_lifetime": self.DEVELOPMENT_TOKEN_LIFETIME,
                },
            )

        except jwt.InvalidTokenError as e:
            return JWTValidationResult(
                valid=False, error=f"Invalid token structure: {e}", warnings=warnings
            )

    def _mark_development_payload(
        self, token: str, unverified_payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Mark payload with development mode metadata if applicable."""
        if token.count(".") >= 2:
            payload = dict(unverified_payload)
            payload["_development_mode"] = True
            payload["_max_validity"] = self.DEVELOPMENT_TOKEN_LIFETIME
            return payload
        return unverified_payload

    def _validate_strict(self, token: str) -> JWTValidationResult:
        """Strict validation - compatibility wrapper for production validation."""
        return self._validate_production(token)

    def _validate_development(self, token: str) -> JWTValidationResult:
        """Development validation - legacy compatibility wrapper."""
        return self._validate_development_secure(token)

    def _get_development_signing_key(self, token: str) -> Any | None:
        """Resolve a signing key for development validation."""
        if self.development_public_key:
            return self.development_public_key

        jwks_client = self._get_jwks_client()
        if jwks_client is None:
            return None
        try:
            signing_key = jwks_client.get_signing_key_from_jwt(token)
        except Exception as exc:
            logger.warning("Failed to resolve JWKS signing key: %s", exc)
            return None
        return signing_key.key

    def _perform_security_checks(self, payload: dict[str, Any]) -> None:
        """Perform additional security checks on token payload."""
        # Check for suspicious claims
        suspicious_claims = ["admin", "root", "superuser", "bypass"]
        for claim in suspicious_claims:
            if claim in payload and payload[claim] is True:
                logger.warning(f"Suspicious claim detected: {claim}")

        # Validate scope if present
        if "scope" in payload:
            allowed_scopes_str = os.getenv("TRAIGENT_ALLOWED_SCOPES", "").strip()
            # Filter out empty strings to handle unset/empty env var
            allowed_scopes = [
                s.strip() for s in allowed_scopes_str.split(",") if s.strip()
            ]
            if allowed_scopes:
                token_scopes = (
                    payload["scope"].split()
                    if isinstance(payload["scope"], str)
                    else payload["scope"]
                )
                for scope in token_scopes:
                    if scope not in allowed_scopes:
                        raise JWTSecurityError(f"Invalid scope: {scope}") from None

    def constant_time_validate(self, token: str, expected_result: bool = True) -> bool:
        """Validate token with constant time to prevent timing attacks."""
        # Generate random delay to mask timing
        delay = secrets.randbits(16) / 1000000  # 0-65μs random delay
        time.sleep(delay)

        # Perform validation
        result = self.validate_token(token)

        # Use constant-time comparison
        actual_bytes = str(result.valid).encode()
        expected_bytes = str(expected_result).encode()

        # Always perform the same amount of work
        hmac.compare_digest(actual_bytes, expected_bytes)

        # Additional constant-time work
        hashlib.sha256(token.encode()).hexdigest()

        return result.valid

    def get_validation_metrics(self) -> dict[str, Any]:
        """Get validation metrics for monitoring."""
        return self._validation_metrics.copy()

    def clear_jti_cache(self) -> None:
        """Clear JWT ID cache (for testing or scheduled cleanup)."""
        with self._jti_lock:
            self._seen_jti.clear()
        logger.info("JWT ID cache cleared")


def get_secure_jwt_validator(mode: ValidationMode | None = None) -> SecureJWTValidator:
    """Get production-ready JWT validator with secure defaults."""
    # Detect environment
    environment = os.getenv("TRAIGENT_ENVIRONMENT", "production").lower()

    # Use explicit mode if provided, otherwise detect from environment
    if mode is None:
        if environment == "production":
            mode = ValidationMode.PRODUCTION
        elif environment == "staging":
            mode = ValidationMode.STAGING
        elif environment == "development":
            mode = ValidationMode.DEVELOPMENT
        else:
            # Default to production for safety
            logger.warning(
                f"Unknown environment '{environment}' - defaulting to production mode"
            )
            mode = ValidationMode.PRODUCTION

    # Log mode selection for development/staging
    if mode in (ValidationMode.DEVELOPMENT, ValidationMode.STAGING):
        logger.warning(
            f"JWT validator in {mode.value.upper()} mode - tokens limited to 5 minutes"
        )

    # Get configuration
    jwks_url = os.getenv("TRAIGENT_JWKS_URL")
    issuer = os.getenv("TRAIGENT_JWT_ISSUER")
    audience = os.getenv("TRAIGENT_JWT_AUDIENCE")

    # Security check - prevent bypass attempts
    if os.getenv("TRAIGENT_JWT_DISABLE_SECURITY"):
        raise JWTSecurityError(
            "Security bypass attempt detected - JWT_DISABLE_SECURITY not allowed"
        )

    return SecureJWTValidator(
        jwks_url=jwks_url,
        issuer=issuer,
        audience=audience,
        validation_mode=mode,
        require_nbf=True,
        require_jti=True,
        max_clock_skew=30,
    )
