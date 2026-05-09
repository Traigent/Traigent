"""License validation for Traigent SDK freemium features.

This module provides license validation for paid features via:
1. Cloud API key validation (primary)
2. Offline license files (signed JWT for air-gapped environments)
3. Graceful degradation with caching and grace periods

Enterprise Configuration:
    TRAIGENT_API_KEY - API key for cloud validation
    TRAIGENT_OFFLINE_MODE - Set to "true" to disable outbound calls
    TRAIGENT_LICENSE_FILE - Path to offline license file (signed JWT)
    TRAIGENT_LICENSE_CACHE_TTL - Cache TTL in seconds (default: 3600)
    TRAIGENT_LICENSE_GRACE_PERIOD - Grace period in seconds (default: 86400)

Offline-license signature configuration (phased rollout):
    TRAIGENT_LICENSE_PUBLIC_KEY - PEM-encoded RSA/EC public key inline.
    TRAIGENT_LICENSE_PUBLIC_KEY_FILE - Path to a PEM public-key file.
    TRAIGENT_ALLOW_UNSIGNED_LICENSE - "true" temporarily accepts the
        legacy unsigned offline-license format for migration only.
    TRAIGENT_REQUIRE_SIGNED_LICENSE - "true" rejects unsigned tokens
        even when TRAIGENT_ALLOW_UNSIGNED_LICENSE is also set.

Behavior:
    - If a public key is configured (env or file) the offline token's
      signature is verified and rejected if invalid.
    - If no public key is configured, offline tokens are rejected by
      default because their signature cannot be verified.
    - The legacy unsigned format is accepted only when
      TRAIGENT_ALLOW_UNSIGNED_LICENSE=true, and the resulting
      LicenseInfo is tagged ``validation_source="offline_unsigned_legacy"``
      so callers and observability can distinguish trusted from legacy
      validations.
"""

# Traceability: CONC-Layer-Core FUNC-LICENSING REQ-LICENSE-001

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LicenseFeature(StrEnum):
    """Features that require a license for full functionality."""

    PARALLEL_EXECUTION = "parallel_execution"
    MULTI_OBJECTIVE = "multi_objective"
    ADVANCED_ALGORITHMS = "advanced_algorithms"
    CLOUD_EXECUTION = "cloud_execution"


class LicenseTier(StrEnum):
    """License tiers with different feature access."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


# Features available in each tier
TIER_FEATURES: dict[LicenseTier, set[LicenseFeature]] = {
    LicenseTier.FREE: set(),  # No paid features
    LicenseTier.PRO: {
        LicenseFeature.PARALLEL_EXECUTION,
        LicenseFeature.MULTI_OBJECTIVE,
        LicenseFeature.ADVANCED_ALGORITHMS,
    },
    LicenseTier.ENTERPRISE: {
        LicenseFeature.PARALLEL_EXECUTION,
        LicenseFeature.MULTI_OBJECTIVE,
        LicenseFeature.ADVANCED_ALGORITHMS,
        LicenseFeature.CLOUD_EXECUTION,
    },
}


@dataclass
class LicenseInfo:
    """Information about the current license."""

    tier: LicenseTier
    features: set[LicenseFeature]
    expires_at: float | None = None  # Unix timestamp, None = never expires
    organization: str | None = None
    validated_at: float = field(default_factory=time.time)
    validation_source: str = "none"  # "cloud", "offline", "cache", "grace"

    @property
    def is_expired(self) -> bool:
        """Check if the license has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if the license is currently valid."""
        return not self.is_expired

    def has_feature(self, feature: LicenseFeature) -> bool:
        """Check if the license includes a specific feature."""
        return feature in self.features and self.is_valid


class FeatureRequiresLicenseError(Exception):
    """Raised when a feature requires a license that is not present."""

    def __init__(self, feature: str, required_tier: str = "Pro"):
        self.feature = feature
        self.required_tier = required_tier
        super().__init__(
            f"'{feature}' requires a Traigent {required_tier} license. "
            "Visit https://traigent.ai/pricing for details."
        )


class LicenseValidator:
    """Validates license for paid features via cloud API or offline file.

    This validator:
    - Caches validation results to minimize latency impact
    - Supports offline license files for air-gapped environments
    - Provides grace periods for temporary network issues
    - Never blocks critical paths synchronously

    Usage:
        validator = LicenseValidator()
        if validator.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION):
            # Use parallel features
            pass
        else:
            raise FeatureRequiresLicenseError("parallel_execution")
    """

    # Default configuration
    DEFAULT_CACHE_TTL = 3600  # 1 hour
    DEFAULT_GRACE_PERIOD = 86400  # 24 hours
    CLOUD_VALIDATION_TIMEOUT = 5.0  # seconds

    def __init__(
        self,
        api_key: str | None = None,
        offline_mode: bool | None = None,
        license_file: str | None = None,
        cache_ttl: int | None = None,
        grace_period: int | None = None,
    ):
        """Initialize the license validator.

        Args:
            api_key: API key for cloud validation. Falls back to TRAIGENT_API_KEY env var.
            offline_mode: If True, disables all outbound calls. Falls back to TRAIGENT_OFFLINE_MODE env var.
            license_file: Path to offline license file. Falls back to TRAIGENT_LICENSE_FILE env var.
            cache_ttl: Cache TTL in seconds. Falls back to TRAIGENT_LICENSE_CACHE_TTL env var.
            grace_period: Grace period in seconds when validation fails. Falls back to TRAIGENT_LICENSE_GRACE_PERIOD env var.
        """
        self._api_key = api_key or os.environ.get("TRAIGENT_API_KEY")
        self._offline_mode = (
            offline_mode
            if offline_mode is not None
            else os.environ.get("TRAIGENT_OFFLINE_MODE", "").lower() == "true"
        )
        self._license_file = license_file or os.environ.get("TRAIGENT_LICENSE_FILE")
        self._cache_ttl = cache_ttl or int(
            os.environ.get("TRAIGENT_LICENSE_CACHE_TTL", str(self.DEFAULT_CACHE_TTL))
        )
        self._grace_period = grace_period or int(
            os.environ.get(
                "TRAIGENT_LICENSE_GRACE_PERIOD", str(self.DEFAULT_GRACE_PERIOD)
            )
        )

        # Thread-safe cache
        self._lock = threading.Lock()
        self._cached_license: LicenseInfo | None = None
        self._last_validation_attempt: float = 0
        self._validation_in_progress = False

        # Background validation event loop
        self._background_loop: asyncio.AbstractEventLoop | None = None

    def _get_free_license(self) -> LicenseInfo:
        """Get a free tier license with no paid features."""
        return LicenseInfo(
            tier=LicenseTier.FREE,
            features=set(),
            expires_at=None,
            validated_at=time.time(),
            validation_source="none",
        )

    def _is_cache_valid(self) -> bool:
        """Check if the cached license is still valid."""
        if self._cached_license is None:
            return False
        age = time.time() - self._cached_license.validated_at
        return age < self._cache_ttl

    def _is_in_grace_period(self) -> bool:
        """Check if we're within the grace period for a previously valid license."""
        if self._cached_license is None:
            return False
        age = time.time() - self._cached_license.validated_at
        return age < self._grace_period

    def _validate_offline_license(self) -> LicenseInfo | None:
        """Validate an offline license file (signed JWT).

        Returns:
            LicenseInfo if valid, None if invalid or not found.
        """
        if not self._license_file:
            return None

        license_path = Path(self._license_file)
        if not license_path.exists():
            logger.warning(f"Offline license file not found: {license_path}")
            return None

        try:
            token = license_path.read_text().strip()
            return self._decode_license_token(token)
        except Exception as e:
            logger.warning(f"Failed to validate offline license: {e}")
            return None

    # Algorithms accepted for offline-license signature verification.
    # The "none" algorithm is explicitly excluded to defeat the classic
    # JWT alg=none downgrade attack.
    _SUPPORTED_LICENSE_ALGORITHMS = ("RS256", "ES256")

    @staticmethod
    def _truthy(value: str | None) -> bool:
        return (value or "").strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _resolve_license_public_key(cls) -> tuple[Any, bool]:
        """Resolve the offline-license public key.

        Returns ``(public_key, was_configured)``. Distinguishing
        "configured but failed to load" from "not configured at all"
        is critical: the fail-open from a broken-key configuration is
        exactly the bypass an attacker would aim for if they could
        influence the env var.

        - was_configured=False, public_key=None  -> nothing was set
        - was_configured=True,  public_key=<key> -> ready to verify
        - was_configured=True,  public_key=None  -> configured but
          unreadable / invalid / empty; the caller MUST fail closed.

        "Configured" is determined by env var *presence* (``in os.environ``),
        not truthiness. ``TRAIGENT_LICENSE_PUBLIC_KEY=""`` is treated as a
        broken configuration, not absence — otherwise an attacker who could
        clear the env var would silently re-enable the legacy bypass.
        """
        inline_configured = "TRAIGENT_LICENSE_PUBLIC_KEY" in os.environ
        file_configured = "TRAIGENT_LICENSE_PUBLIC_KEY_FILE" in os.environ

        if not inline_configured and not file_configured:
            return None, False

        pem_inline = os.environ.get("TRAIGENT_LICENSE_PUBLIC_KEY", "")
        pem_path = os.environ.get("TRAIGENT_LICENSE_PUBLIC_KEY_FILE", "")

        pem_bytes: bytes | None = None
        if pem_inline:
            pem_bytes = pem_inline.encode("utf-8")
        elif pem_path:
            try:
                pem_bytes = Path(pem_path).expanduser().read_bytes()
            except OSError as exc:
                logger.error(
                    "Failed to read TRAIGENT_LICENSE_PUBLIC_KEY_FILE=%s: %s",
                    pem_path,
                    exc,
                )
                return None, True  # configured but unreadable -> fail closed

        if not pem_bytes:
            # Env var present but empty (or whitespace-only inline) is
            # broken configuration, not absence.
            logger.error(
                "License public key env var is set but empty; refusing to "
                "verify offline licenses until a non-empty key is provided."
            )
            return None, True

        try:
            from cryptography.hazmat.primitives import serialization

            return serialization.load_pem_public_key(pem_bytes), True
        except Exception as exc:
            logger.error("Configured license public key is invalid: %s", exc)
            return None, True  # configured but invalid -> fail closed

    def _decode_license_token(self, token: str) -> LicenseInfo | None:
        """Decode and validate an offline license token.

        A public key (inline or file) verifies the signature when
        configured. Without a public key, the validator fails closed by
        default because accepting the payload without verifying the
        signature makes license forgery trivial. A short-term legacy
        escape hatch is available via ``TRAIGENT_ALLOW_UNSIGNED_LICENSE``
        and is tagged ``validation_source="offline_unsigned_legacy"``.

        If a public key is configured but cannot be loaded (unreadable
        file, malformed PEM, empty inline value), the validator MUST
        fail closed: silently treating a broken key configuration as
        "no key configured" would let an attacker who can influence the
        env var trigger the legacy bypass.

        Args:
            token: The license token to decode.

        Returns:
            LicenseInfo if valid, None if invalid.
        """
        public_key, key_was_configured = self._resolve_license_public_key()
        require_signed = self._truthy(os.environ.get("TRAIGENT_REQUIRE_SIGNED_LICENSE"))
        allow_unsigned = self._truthy(os.environ.get("TRAIGENT_ALLOW_UNSIGNED_LICENSE"))

        if public_key is not None:
            return self._decode_signed_license_token(token, public_key)

        if key_was_configured:
            # Env points at a key that did not load. Refuse rather than
            # falling through to legacy — the loader has already logged
            # the underlying error.
            logger.error(
                "License public key is configured but could not be "
                "loaded; refusing to verify the offline license. Fix "
                "TRAIGENT_LICENSE_PUBLIC_KEY / TRAIGENT_LICENSE_PUBLIC_KEY_FILE."
            )
            return None

        if require_signed:
            logger.error(
                "TRAIGENT_REQUIRE_SIGNED_LICENSE is set but no license "
                "public key is configured (TRAIGENT_LICENSE_PUBLIC_KEY or "
                "TRAIGENT_LICENSE_PUBLIC_KEY_FILE). Refusing to accept "
                "the offline license."
            )
            return None

        if not allow_unsigned:
            logger.error(
                "Offline license token cannot be verified because no "
                "license public key is configured. Set "
                "TRAIGENT_LICENSE_PUBLIC_KEY or "
                "TRAIGENT_LICENSE_PUBLIC_KEY_FILE to verify signed "
                "offline licenses. For a temporary legacy migration only, "
                "set TRAIGENT_ALLOW_UNSIGNED_LICENSE=true."
            )
            return None

        logger.warning(
            "Accepting an UNSIGNED offline license token because "
            "TRAIGENT_ALLOW_UNSIGNED_LICENSE=true. This format is "
            "deprecated and must be used only for short-term migration. "
            "Provision a signing keypair and set TRAIGENT_LICENSE_PUBLIC_KEY "
            "(or _FILE) to verify offline licenses."
        )
        return self._decode_unsigned_license_token(token)

    @staticmethod
    def _b64url_decode(segment: str) -> bytes:
        """Decode a JWT base64url segment strictly."""
        padded = segment + "=" * ((-len(segment)) % 4)
        return base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)

    @classmethod
    def _decode_jwt_json_segment(cls, segment: str, name: str) -> dict | None:
        """Decode a JWT JSON segment into an object dictionary."""
        try:
            value = json.loads(cls._b64url_decode(segment).decode("utf-8"))
        except Exception as exc:
            logger.warning("Failed to decode license token %s: %s", name, exc)
            return None
        if not isinstance(value, dict):
            logger.warning("Invalid license token %s: expected object", name)
            return None
        return value

    def _decode_signed_license_token(
        self, token: str, public_key
    ) -> LicenseInfo | None:
        """Verify the JWT signature on the offline license and decode it."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                logger.warning("Invalid license token format")
                return None

            header = self._decode_jwt_json_segment(parts[0], "header")
            if header is None:
                return None

            alg = header.get("alg")
            if alg not in self._SUPPORTED_LICENSE_ALGORITHMS:
                logger.warning("Unsupported offline license signing algorithm: %s", alg)
                return None

            signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
            signature = self._b64url_decode(parts[2])
            self._verify_license_signature(alg, public_key, signature, signing_input)

            payload = self._decode_jwt_json_segment(parts[1], "payload")
        except Exception as exc:
            logger.warning("Signed license verification failed: %s", exc)
            return None

        if payload is None:
            return None

        info = self._build_license_info(payload, validation_source="offline")
        if info is None:
            return None
        if info.is_expired:
            logger.warning("Offline license has expired")
            return None
        return info

    @staticmethod
    def _verify_license_signature(
        alg: str, public_key, signature: bytes, signing_input: bytes
    ) -> None:
        """Verify a JWT signature with cryptography's public-key APIs."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec, padding, utils

        if alg == "RS256":
            public_key.verify(
                signature,
                signing_input,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return

        if alg == "ES256":
            if len(signature) != 64:
                raise ValueError("ES256 signature must be 64 raw bytes")
            r = int.from_bytes(signature[:32], "big")
            s = int.from_bytes(signature[32:], "big")
            der_signature = utils.encode_dss_signature(r, s)
            public_key.verify(der_signature, signing_input, ec.ECDSA(hashes.SHA256()))
            return

        raise ValueError(f"Unsupported offline license signing algorithm: {alg}")

    def _decode_unsigned_license_token(self, token: str) -> LicenseInfo | None:
        """Legacy path: base64-decode the middle segment without verifying."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                logger.warning("Invalid license token format")
                return None

            header = self._decode_jwt_json_segment(parts[0], "header")
            if header is None:
                return None
            if str(header.get("alg", "none")).lower() != "none":
                logger.warning(
                    "Refusing to parse a signed-looking license token without "
                    "signature verification."
                )
                return None

            payload = self._decode_jwt_json_segment(parts[1], "payload")
        except Exception as exc:
            logger.warning("Failed to decode license token: %s", exc)
            return None

        if payload is None:
            return None

        info = self._build_license_info(
            payload, validation_source="offline_unsigned_legacy"
        )
        if info is None:
            return None
        if info.is_expired:
            logger.warning("Offline license has expired")
            return None
        return info

    @staticmethod
    def _build_license_info(
        payload: dict, *, validation_source: str
    ) -> LicenseInfo | None:
        """Common post-decode payload -> LicenseInfo conversion."""
        try:
            tier = LicenseTier(payload.get("tier", "free"))
            features = {LicenseFeature(f) for f in payload.get("features", [])}
        except ValueError as exc:
            logger.warning("License payload contained unknown enum value: %s", exc)
            return None

        return LicenseInfo(
            tier=tier,
            features=features,
            expires_at=payload.get("exp"),
            organization=payload.get("org"),
            validated_at=time.time(),
            validation_source=validation_source,
        )

    async def _validate_cloud_license(self) -> LicenseInfo | None:
        """Validate license via cloud API.

        Returns:
            LicenseInfo if valid, None if validation failed.
        """
        if self._offline_mode:
            logger.debug("Offline mode enabled, skipping cloud validation")
            return None

        if not self._api_key:
            logger.debug("No API key configured, cannot validate cloud license")
            return None

        try:
            # Import here to avoid circular dependency
            from traigent.cloud.client import TraigentCloudClient

            async with TraigentCloudClient() as client:
                # Call the license features endpoint
                response = await asyncio.wait_for(
                    client.get_license_features(),
                    timeout=self.CLOUD_VALIDATION_TIMEOUT,
                )

                if response is None:
                    return None

                # Parse response
                tier_str = response.get("tier", "free")
                tier = LicenseTier(tier_str)
                features = {LicenseFeature(f) for f in response.get("features", [])}
                expires_at = response.get("expires_at")
                organization = response.get("organization")

                return LicenseInfo(
                    tier=tier,
                    features=features,
                    expires_at=expires_at,
                    organization=organization,
                    validated_at=time.time(),
                    validation_source="cloud",
                )

        except TimeoutError:
            logger.warning("Cloud license validation timed out")
            return None
        except ImportError:
            logger.debug("Cloud client not available")
            return None
        except Exception as e:
            logger.warning(f"Cloud license validation failed: {e}")
            return None

    async def validate_async(self) -> LicenseInfo:
        """Validate the license asynchronously.

        This method:
        1. Returns cached license if still valid
        2. Tries offline license file if configured
        3. Validates via cloud API if not in offline mode
        4. Falls back to grace period if validation fails
        5. Returns free tier if all else fails

        Returns:
            LicenseInfo with the validated license information.
        """
        # Check cache first
        with self._lock:
            if self._is_cache_valid():
                return self._cached_license  # type: ignore

        # Try offline license file
        if self._license_file:
            offline_license = self._validate_offline_license()
            if offline_license:
                with self._lock:
                    self._cached_license = offline_license
                return offline_license

        # Try cloud validation
        cloud_license = await self._validate_cloud_license()
        if cloud_license:
            with self._lock:
                self._cached_license = cloud_license
            return cloud_license

        # Check grace period
        with self._lock:
            if self._is_in_grace_period() and self._cached_license:
                logger.info("Using cached license within grace period")
                return LicenseInfo(
                    tier=self._cached_license.tier,
                    features=self._cached_license.features,
                    expires_at=self._cached_license.expires_at,
                    organization=self._cached_license.organization,
                    validated_at=self._cached_license.validated_at,
                    validation_source="grace",
                )

        # Fall back to free tier
        free_license = self._get_free_license()
        with self._lock:
            self._cached_license = free_license
        return free_license

    def validate_sync(self) -> LicenseInfo:
        """Validate the license synchronously.

        This is a blocking wrapper around validate_async. Use sparingly
        to avoid blocking critical paths.

        Returns:
            LicenseInfo with the validated license information.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.validate_async())

        # We're already in an async context, use cached or free
        with self._lock:
            if self._cached_license:
                return self._cached_license
        return self._get_free_license()

    def has_feature_sync(self, feature: LicenseFeature | str) -> bool:
        """Check if the license includes a specific feature (synchronous).

        This method is optimized for fast checks:
        - Returns cached result if available
        - Triggers background validation if cache is stale
        - Never blocks on network calls

        Args:
            feature: The feature to check.

        Returns:
            True if the feature is available, False otherwise.
        """
        if isinstance(feature, str):
            try:
                feature = LicenseFeature(feature)
            except ValueError:
                # Unknown feature - fail closed for security
                return False

        # Check cache first (fast path)
        with self._lock:
            if self._cached_license and self._is_cache_valid():
                return self._cached_license.has_feature(feature)

        # Try to validate synchronously without blocking
        # First check offline license
        if self._license_file:
            offline_license = self._validate_offline_license()
            if offline_license:
                with self._lock:
                    self._cached_license = offline_license
                return offline_license.has_feature(feature)

        # If we have a cached license in grace period, use it
        with self._lock:
            if self._is_in_grace_period() and self._cached_license:
                return self._cached_license.has_feature(feature)

        # No valid license, feature not available
        return False

    async def has_feature_async(self, feature: LicenseFeature | str) -> bool:
        """Check if the license includes a specific feature (asynchronous).

        Args:
            feature: The feature to check.

        Returns:
            True if the feature is available, False otherwise.
        """
        if isinstance(feature, str):
            try:
                feature = LicenseFeature(feature)
            except ValueError:
                # Unknown feature - fail closed for security
                return False

        license_info = await self.validate_async()
        return license_info.has_feature(feature)

    def require_feature(self, feature: LicenseFeature | str) -> None:
        """Require a feature, raising an error if not available.

        Args:
            feature: The feature to require.

        Raises:
            FeatureRequiresLicenseError: If the feature is not available.
        """
        if isinstance(feature, str):
            feature_name = feature
            try:
                feature = LicenseFeature(feature)
            except ValueError:
                # Unknown feature - fail closed for security
                raise FeatureRequiresLicenseError(feature_name, "Unknown") from None
        else:
            feature_name = feature.value

        if not self.has_feature_sync(feature):
            # Determine required tier
            required_tier = "Pro"
            if feature == LicenseFeature.CLOUD_EXECUTION:
                required_tier = "Enterprise"
            raise FeatureRequiresLicenseError(feature_name, required_tier)

    def get_license_info(self) -> LicenseInfo:
        """Get the current license information.

        Returns:
            LicenseInfo with the current license details.
        """
        with self._lock:
            if self._cached_license:
                return self._cached_license
        return self._get_free_license()


# Global singleton instance
_license_validator: LicenseValidator | None = None
_license_validator_lock = threading.Lock()


def get_license_validator() -> LicenseValidator:
    """Get the global license validator instance.

    Returns:
        The global LicenseValidator instance.
    """
    global _license_validator
    with _license_validator_lock:
        if _license_validator is None:
            _license_validator = LicenseValidator()
        return _license_validator


def has_feature(feature: LicenseFeature | str) -> bool:
    """Check if a feature is available in the current license.

    Convenience function that uses the global validator.

    Args:
        feature: The feature to check.

    Returns:
        True if the feature is available, False otherwise.
    """
    return get_license_validator().has_feature_sync(feature)


def require_feature(feature: LicenseFeature | str) -> None:
    """Require a feature, raising an error if not available.

    Convenience function that uses the global validator.

    Args:
        feature: The feature to require.

    Raises:
        FeatureRequiresLicenseError: If the feature is not available.
    """
    get_license_validator().require_feature(feature)
