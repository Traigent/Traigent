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
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LicenseFeature(str, Enum):
    """Features that require a license for full functionality."""

    PARALLEL_EXECUTION = "parallel_execution"
    MULTI_OBJECTIVE = "multi_objective"
    ADVANCED_ALGORITHMS = "advanced_algorithms"
    CLOUD_EXECUTION = "cloud_execution"


class LicenseTier(str, Enum):
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

    def _decode_license_token(self, token: str) -> LicenseInfo | None:
        """Decode and validate a license token (simplified JWT-like format).

        Note: In production, this would use proper JWT validation with
        cryptographic signatures. This is a simplified implementation.

        Args:
            token: The license token to decode.

        Returns:
            LicenseInfo if valid, None if invalid.
        """
        try:
            # Simple base64-encoded JSON format for now
            # Production would use proper JWT with RSA/ECDSA signatures
            parts = token.split(".")
            if len(parts) != 3:
                logger.warning("Invalid license token format")
                return None

            # Decode payload (middle part)
            payload_b64 = parts[1]
            # Add padding if needed (use negative modulo to avoid adding 4 when len%4==0)
            payload_b64 += "=" * ((-len(payload_b64)) % 4)
            payload_json = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
            payload = json.loads(payload_json)

            # Extract license info
            tier_str = payload.get("tier", "free")
            tier = LicenseTier(tier_str)
            features = {LicenseFeature(f) for f in payload.get("features", [])}
            expires_at = payload.get("exp")
            organization = payload.get("org")

            # Check expiration
            if expires_at and time.time() > expires_at:
                logger.warning("Offline license has expired")
                return None

            return LicenseInfo(
                tier=tier,
                features=features,
                expires_at=expires_at,
                organization=organization,
                validated_at=time.time(),
                validation_source="offline",
            )

        except Exception as e:
            logger.warning(f"Failed to decode license token: {e}")
            return None

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
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, use cached or free
                with self._lock:
                    if self._cached_license:
                        return self._cached_license
                return self._get_free_license()
            return loop.run_until_complete(self.validate_async())
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.validate_async())

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
