"""Comprehensive tests for traigent.core.license module.

Tests cover:
- LicenseFeature and LicenseTier StrEnum values
- TIER_FEATURES mapping
- LicenseInfo dataclass and its properties/methods
- FeatureRequiresLicenseError exception
- LicenseValidator class (init, caching, offline, cloud, sync/async validation)
- Module-level convenience functions (get_license_validator, has_feature, require_feature)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.core.license import (
    TIER_FEATURES,
    FeatureRequiresLicenseError,
    LicenseFeature,
    LicenseInfo,
    LicenseTier,
    LicenseValidator,
    get_license_validator,
    has_feature,
    require_feature,
)

# ---------------------------------------------------------------------------
# LicenseFeature StrEnum
# ---------------------------------------------------------------------------


class TestLicenseFeature:
    """Test LicenseFeature StrEnum."""

    def test_parallel_execution_value(self):
        """Test PARALLEL_EXECUTION enum value."""
        assert LicenseFeature.PARALLEL_EXECUTION == "parallel_execution"
        assert LicenseFeature.PARALLEL_EXECUTION.value == "parallel_execution"

    def test_multi_objective_value(self):
        """Test MULTI_OBJECTIVE enum value."""
        assert LicenseFeature.MULTI_OBJECTIVE == "multi_objective"
        assert LicenseFeature.MULTI_OBJECTIVE.value == "multi_objective"

    def test_advanced_algorithms_value(self):
        """Test ADVANCED_ALGORITHMS enum value."""
        assert LicenseFeature.ADVANCED_ALGORITHMS == "advanced_algorithms"
        assert LicenseFeature.ADVANCED_ALGORITHMS.value == "advanced_algorithms"

    def test_cloud_execution_value(self):
        """Test CLOUD_EXECUTION enum value."""
        assert LicenseFeature.CLOUD_EXECUTION == "cloud_execution"
        assert LicenseFeature.CLOUD_EXECUTION.value == "cloud_execution"

    def test_all_features_count(self):
        """Test total number of features."""
        assert len(LicenseFeature) == 4

    def test_feature_is_str(self):
        """Test that LicenseFeature values are strings (StrEnum)."""
        for feature in LicenseFeature:
            assert isinstance(feature, str)

    def test_feature_from_string(self):
        """Test creating LicenseFeature from a string value."""
        feature = LicenseFeature("parallel_execution")
        assert feature is LicenseFeature.PARALLEL_EXECUTION

    def test_feature_from_invalid_string(self):
        """Test creating LicenseFeature from an invalid string raises ValueError."""
        with pytest.raises(ValueError):
            LicenseFeature("nonexistent_feature")

    def test_feature_membership(self):
        """Test membership in LicenseFeature."""
        values = [f.value for f in LicenseFeature]
        assert "parallel_execution" in values
        assert "multi_objective" in values
        assert "advanced_algorithms" in values
        assert "cloud_execution" in values


# ---------------------------------------------------------------------------
# LicenseTier StrEnum
# ---------------------------------------------------------------------------


class TestLicenseTier:
    """Test LicenseTier StrEnum."""

    def test_free_value(self):
        """Test FREE tier value."""
        assert LicenseTier.FREE == "free"
        assert LicenseTier.FREE.value == "free"

    def test_pro_value(self):
        """Test PRO tier value."""
        assert LicenseTier.PRO == "pro"
        assert LicenseTier.PRO.value == "pro"

    def test_enterprise_value(self):
        """Test ENTERPRISE tier value."""
        assert LicenseTier.ENTERPRISE == "enterprise"
        assert LicenseTier.ENTERPRISE.value == "enterprise"

    def test_all_tiers_count(self):
        """Test total number of tiers."""
        assert len(LicenseTier) == 3

    def test_tier_is_str(self):
        """Test that LicenseTier values are strings (StrEnum)."""
        for tier in LicenseTier:
            assert isinstance(tier, str)

    def test_tier_from_string(self):
        """Test creating LicenseTier from a string value."""
        tier = LicenseTier("pro")
        assert tier is LicenseTier.PRO

    def test_tier_from_invalid_string(self):
        """Test creating LicenseTier from an invalid string raises ValueError."""
        with pytest.raises(ValueError):
            LicenseTier("platinum")


# ---------------------------------------------------------------------------
# TIER_FEATURES mapping
# ---------------------------------------------------------------------------


class TestTierFeatures:
    """Test the TIER_FEATURES mapping."""

    def test_free_tier_has_no_features(self):
        """Test FREE tier has no paid features."""
        assert TIER_FEATURES[LicenseTier.FREE] == set()

    def test_pro_tier_features(self):
        """Test PRO tier has expected features."""
        pro_features = TIER_FEATURES[LicenseTier.PRO]
        assert LicenseFeature.PARALLEL_EXECUTION in pro_features
        assert LicenseFeature.MULTI_OBJECTIVE in pro_features
        assert LicenseFeature.ADVANCED_ALGORITHMS in pro_features
        assert LicenseFeature.CLOUD_EXECUTION not in pro_features

    def test_enterprise_tier_has_all_features(self):
        """Test ENTERPRISE tier has all features including cloud execution."""
        enterprise_features = TIER_FEATURES[LicenseTier.ENTERPRISE]
        for feature in LicenseFeature:
            assert feature in enterprise_features

    def test_enterprise_superset_of_pro(self):
        """Test ENTERPRISE features are a superset of PRO features."""
        pro_features = TIER_FEATURES[LicenseTier.PRO]
        enterprise_features = TIER_FEATURES[LicenseTier.ENTERPRISE]
        assert pro_features.issubset(enterprise_features)

    def test_all_tiers_present(self):
        """Test all LicenseTier values have entries in TIER_FEATURES."""
        for tier in LicenseTier:
            assert tier in TIER_FEATURES


# ---------------------------------------------------------------------------
# LicenseInfo dataclass
# ---------------------------------------------------------------------------


class TestLicenseInfo:
    """Test LicenseInfo dataclass."""

    def test_basic_creation(self):
        """Test basic LicenseInfo creation."""
        info = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
        )
        assert info.tier == LicenseTier.PRO
        assert LicenseFeature.PARALLEL_EXECUTION in info.features
        assert info.expires_at is None
        assert info.organization is None
        assert info.validation_source == "none"

    def test_creation_with_all_fields(self):
        """Test LicenseInfo creation with all fields set."""
        now = time.time()
        info = LicenseInfo(
            tier=LicenseTier.ENTERPRISE,
            features={LicenseFeature.CLOUD_EXECUTION, LicenseFeature.MULTI_OBJECTIVE},
            expires_at=now + 86400,
            organization="TestCorp",
            validated_at=now,
            validation_source="cloud",
        )
        assert info.tier == LicenseTier.ENTERPRISE
        assert len(info.features) == 2
        assert info.expires_at == now + 86400
        assert info.organization == "TestCorp"
        assert info.validated_at == now
        assert info.validation_source == "cloud"

    def test_is_expired_no_expiry(self):
        """Test is_expired returns False when expires_at is None."""
        info = LicenseInfo(
            tier=LicenseTier.PRO,
            features=set(),
            expires_at=None,
        )
        assert info.is_expired is False

    def test_is_expired_future_expiry(self):
        """Test is_expired returns False when expiry is in the future."""
        info = LicenseInfo(
            tier=LicenseTier.PRO,
            features=set(),
            expires_at=time.time() + 86400,
        )
        assert info.is_expired is False

    def test_is_expired_past_expiry(self):
        """Test is_expired returns True when expiry is in the past."""
        info = LicenseInfo(
            tier=LicenseTier.PRO,
            features=set(),
            expires_at=time.time() - 1,
        )
        assert info.is_expired is True

    def test_is_valid_when_not_expired(self):
        """Test is_valid returns True when not expired."""
        info = LicenseInfo(
            tier=LicenseTier.PRO,
            features=set(),
            expires_at=time.time() + 86400,
        )
        assert info.is_valid is True

    def test_is_valid_when_expired(self):
        """Test is_valid returns False when expired."""
        info = LicenseInfo(
            tier=LicenseTier.PRO,
            features=set(),
            expires_at=time.time() - 1,
        )
        assert info.is_valid is False

    def test_is_valid_no_expiry(self):
        """Test is_valid returns True when no expiry set."""
        info = LicenseInfo(
            tier=LicenseTier.FREE,
            features=set(),
        )
        assert info.is_valid is True

    def test_has_feature_present_and_valid(self):
        """Test has_feature returns True when feature present and license valid."""
        info = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            expires_at=time.time() + 86400,
        )
        assert info.has_feature(LicenseFeature.PARALLEL_EXECUTION) is True

    def test_has_feature_absent(self):
        """Test has_feature returns False when feature not in features set."""
        info = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            expires_at=time.time() + 86400,
        )
        assert info.has_feature(LicenseFeature.CLOUD_EXECUTION) is False

    def test_has_feature_present_but_expired(self):
        """Test has_feature returns False when feature present but license expired."""
        info = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            expires_at=time.time() - 1,
        )
        assert info.has_feature(LicenseFeature.PARALLEL_EXECUTION) is False

    def test_has_feature_no_expiry(self):
        """Test has_feature returns True when feature present and no expiry."""
        info = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.MULTI_OBJECTIVE},
        )
        assert info.has_feature(LicenseFeature.MULTI_OBJECTIVE) is True

    def test_validated_at_default(self):
        """Test validated_at defaults to current time."""
        before = time.time()
        info = LicenseInfo(tier=LicenseTier.FREE, features=set())
        after = time.time()
        assert before <= info.validated_at <= after

    def test_empty_features_set(self):
        """Test LicenseInfo with empty features set."""
        info = LicenseInfo(tier=LicenseTier.FREE, features=set())
        for feature in LicenseFeature:
            assert info.has_feature(feature) is False


# ---------------------------------------------------------------------------
# FeatureRequiresLicenseError
# ---------------------------------------------------------------------------


class TestFeatureRequiresLicenseError:
    """Test FeatureRequiresLicenseError exception."""

    def test_default_tier(self):
        """Test error with default required tier."""
        err = FeatureRequiresLicenseError("parallel_execution")
        assert err.feature == "parallel_execution"
        assert err.required_tier == "Pro"
        assert "parallel_execution" in str(err)
        assert "Pro" in str(err)
        assert "traigent.ai/pricing" in str(err)

    def test_custom_tier(self):
        """Test error with custom required tier."""
        err = FeatureRequiresLicenseError("cloud_execution", "Enterprise")
        assert err.feature == "cloud_execution"
        assert err.required_tier == "Enterprise"
        assert "Enterprise" in str(err)

    def test_is_exception(self):
        """Test FeatureRequiresLicenseError inherits from Exception."""
        err = FeatureRequiresLicenseError("test_feature")
        assert isinstance(err, Exception)

    def test_can_be_raised_and_caught(self):
        """Test the error can be raised and caught."""
        with pytest.raises(FeatureRequiresLicenseError) as exc_info:
            raise FeatureRequiresLicenseError("test_feature", "Pro")
        assert exc_info.value.feature == "test_feature"


# ---------------------------------------------------------------------------
# LicenseValidator.__init__
# ---------------------------------------------------------------------------


class TestLicenseValidatorInit:
    """Test LicenseValidator initialization."""

    def test_default_init(self):
        """Test default initialization with no arguments."""
        with patch.dict(os.environ, {}, clear=True):
            validator = LicenseValidator()
            assert validator._api_key is None
            assert validator._offline_mode is False
            assert validator._license_file is None
            assert validator._cache_ttl == LicenseValidator.DEFAULT_CACHE_TTL
            assert validator._grace_period == LicenseValidator.DEFAULT_GRACE_PERIOD
            assert validator._cached_license is None

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        validator = LicenseValidator(api_key="test-key-123")  # pragma: allowlist secret
        assert validator._api_key == "test-key-123"  # pragma: allowlist secret

    def test_init_with_api_key_env(self):
        """Test initialization falls back to TRAIGENT_API_KEY env var."""
        with patch.dict(
            os.environ, {"TRAIGENT_API_KEY": "env-key-456"}  # pragma: allowlist secret
        ):
            validator = LicenseValidator()
            assert validator._api_key == "env-key-456"  # pragma: allowlist secret

    def test_init_explicit_api_key_overrides_env(self):
        """Test explicit api_key takes precedence over env var."""
        with patch.dict(
            os.environ, {"TRAIGENT_API_KEY": "env-key"}  # pragma: allowlist secret
        ):
            validator = LicenseValidator(
                api_key="explicit-key"  # pragma: allowlist secret
            )
            assert validator._api_key == "explicit-key"  # pragma: allowlist secret

    def test_init_offline_mode_explicit(self):
        """Test initialization with explicit offline_mode."""
        validator = LicenseValidator(offline_mode=True)
        assert validator._offline_mode is True

    def test_init_offline_mode_env(self):
        """Test initialization reads TRAIGENT_OFFLINE_MODE env var."""
        with patch.dict(os.environ, {"TRAIGENT_OFFLINE_MODE": "true"}):
            validator = LicenseValidator()
            assert validator._offline_mode is True

    def test_init_offline_mode_env_false(self):
        """Test TRAIGENT_OFFLINE_MODE=false results in offline_mode=False."""
        with patch.dict(os.environ, {"TRAIGENT_OFFLINE_MODE": "false"}):
            validator = LicenseValidator()
            assert validator._offline_mode is False

    def test_init_license_file(self):
        """Test initialization with explicit license file."""
        validator = LicenseValidator(license_file="/path/to/license.jwt")
        assert validator._license_file == "/path/to/license.jwt"

    def test_init_license_file_env(self):
        """Test initialization reads TRAIGENT_LICENSE_FILE env var."""
        with patch.dict(os.environ, {"TRAIGENT_LICENSE_FILE": "/env/path.jwt"}):
            validator = LicenseValidator()
            assert validator._license_file == "/env/path.jwt"

    def test_init_cache_ttl_explicit(self):
        """Test initialization with explicit cache TTL."""
        validator = LicenseValidator(cache_ttl=7200)
        assert validator._cache_ttl == 7200

    def test_init_cache_ttl_env(self):
        """Test initialization reads TRAIGENT_LICENSE_CACHE_TTL env var."""
        with patch.dict(os.environ, {"TRAIGENT_LICENSE_CACHE_TTL": "1800"}):
            validator = LicenseValidator()
            assert validator._cache_ttl == 1800

    def test_init_grace_period_explicit(self):
        """Test initialization with explicit grace period."""
        validator = LicenseValidator(grace_period=43200)
        assert validator._grace_period == 43200

    def test_init_grace_period_env(self):
        """Test initialization reads TRAIGENT_LICENSE_GRACE_PERIOD env var."""
        with patch.dict(os.environ, {"TRAIGENT_LICENSE_GRACE_PERIOD": "172800"}):
            validator = LicenseValidator()
            assert validator._grace_period == 172800

    def test_init_has_thread_lock(self):
        """Test validator has a threading lock for cache."""
        validator = LicenseValidator()
        assert isinstance(validator._lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# LicenseValidator._get_free_license
# ---------------------------------------------------------------------------


class TestGetFreeLicense:
    """Test LicenseValidator._get_free_license."""

    def test_returns_free_tier(self):
        """Test free license has FREE tier."""
        validator = LicenseValidator()
        license_info = validator._get_free_license()
        assert license_info.tier == LicenseTier.FREE

    def test_returns_no_features(self):
        """Test free license has empty features set."""
        validator = LicenseValidator()
        license_info = validator._get_free_license()
        assert license_info.features == set()

    def test_never_expires(self):
        """Test free license has no expiration."""
        validator = LicenseValidator()
        license_info = validator._get_free_license()
        assert license_info.expires_at is None

    def test_validation_source_is_none(self):
        """Test free license validation source is 'none'."""
        validator = LicenseValidator()
        license_info = validator._get_free_license()
        assert license_info.validation_source == "none"


# ---------------------------------------------------------------------------
# LicenseValidator cache helpers
# ---------------------------------------------------------------------------


class TestCacheHelpers:
    """Test LicenseValidator._is_cache_valid and _is_in_grace_period."""

    def test_cache_invalid_when_no_cached_license(self):
        """Test cache is invalid when nothing is cached."""
        validator = LicenseValidator()
        assert validator._is_cache_valid() is False

    def test_cache_valid_with_fresh_license(self):
        """Test cache is valid with recently validated license."""
        validator = LicenseValidator(cache_ttl=3600)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features=set(),
            validated_at=time.time(),
        )
        assert validator._is_cache_valid() is True

    def test_cache_invalid_with_stale_license(self):
        """Test cache is invalid with expired TTL."""
        validator = LicenseValidator(cache_ttl=3600)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features=set(),
            validated_at=time.time() - 7200,  # 2 hours ago
        )
        assert validator._is_cache_valid() is False

    def test_grace_period_false_when_no_cached_license(self):
        """Test grace period returns False when nothing cached."""
        validator = LicenseValidator()
        assert validator._is_in_grace_period() is False

    def test_grace_period_true_within_period(self):
        """Test grace period returns True within grace period."""
        validator = LicenseValidator(grace_period=86400)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features=set(),
            validated_at=time.time() - 3600,  # 1 hour ago
        )
        assert validator._is_in_grace_period() is True

    def test_grace_period_false_past_period(self):
        """Test grace period returns False past grace period."""
        validator = LicenseValidator(grace_period=86400)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features=set(),
            validated_at=time.time() - 172800,  # 2 days ago
        )
        assert validator._is_in_grace_period() is False


# ---------------------------------------------------------------------------
# LicenseValidator._validate_offline_license
# ---------------------------------------------------------------------------


class TestValidateOfflineLicense:
    """Test LicenseValidator._validate_offline_license."""

    def test_returns_none_when_no_license_file(self):
        """Test returns None when no license file configured."""
        validator = LicenseValidator()
        assert validator._validate_offline_license() is None

    def test_returns_none_when_file_not_found(self):
        """Test returns None when license file doesn't exist."""
        validator = LicenseValidator(license_file="/nonexistent/path/license.jwt")
        assert validator._validate_offline_license() is None

    def test_valid_license_file(self):
        """Test valid offline license file is parsed correctly."""
        payload = {
            "tier": "pro",
            "features": ["parallel_execution", "multi_objective"],
            "exp": time.time() + 86400,
            "org": "TestCorp",
        }
        token = self._create_token(payload)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jwt", delete=False) as f:
            f.write(token)
            f.flush()
            temp_path = f.name

        try:
            with patch.dict(
                os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
            ):
                validator = LicenseValidator(license_file=temp_path)
                result = validator._validate_offline_license()
            assert result is not None
            assert result.tier == LicenseTier.PRO
            assert LicenseFeature.PARALLEL_EXECUTION in result.features
            assert LicenseFeature.MULTI_OBJECTIVE in result.features
            assert result.organization == "TestCorp"
            # Unsigned tokens are now tagged as legacy until a public key
            # is configured (see C3 phased license signature rollout).
            assert result.validation_source == "offline_unsigned_legacy"
        finally:
            os.unlink(temp_path)

    def test_expired_license_file(self):
        """Test expired offline license file returns None."""
        payload = {
            "tier": "pro",
            "features": ["parallel_execution"],
            "exp": time.time() - 86400,  # expired yesterday
        }
        token = self._create_token(payload)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jwt", delete=False) as f:
            f.write(token)
            f.flush()
            temp_path = f.name

        try:
            with patch.dict(
                os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
            ):
                validator = LicenseValidator(license_file=temp_path)
                result = validator._validate_offline_license()
            assert result is None
        finally:
            os.unlink(temp_path)

    def test_corrupt_license_file(self):
        """Test corrupt license file returns None."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jwt", delete=False) as f:
            f.write("this is not a valid token")
            f.flush()
            temp_path = f.name

        try:
            validator = LicenseValidator(license_file=temp_path)
            result = validator._validate_offline_license()
            assert result is None
        finally:
            os.unlink(temp_path)

    @staticmethod
    def _b64url_encode(data: bytes) -> str:
        """Encode bytes as an unpadded JWT base64url segment."""
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    @classmethod
    def _create_token(cls, payload: dict) -> str:
        """Create a simplified JWT-like token for testing."""
        header = cls._b64url_encode(json.dumps({"alg": "none"}).encode())
        body = cls._b64url_encode(json.dumps(payload).encode())
        signature = cls._b64url_encode(b"test-sig")
        return f"{header}.{body}.{signature}"

    @classmethod
    def _create_rs256_token(cls, payload: dict, private_key) -> str:
        """Create an RS256 JWT-like token for signature verification tests."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        header = cls._b64url_encode(json.dumps({"alg": "RS256"}).encode())
        body = cls._b64url_encode(json.dumps(payload).encode())
        signing_input = f"{header}.{body}".encode("ascii")
        signature = private_key.sign(
            signing_input,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return f"{header}.{body}.{cls._b64url_encode(signature)}"

    @staticmethod
    def _create_es256_token(payload: dict, private_key) -> str:
        """Create an ES256 JWT-like token for signature verification tests."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec, utils

        header = TestValidateOfflineLicense._b64url_encode(
            json.dumps({"alg": "ES256"}).encode()
        )
        body = TestValidateOfflineLicense._b64url_encode(json.dumps(payload).encode())
        signing_input = f"{header}.{body}".encode("ascii")
        der_signature = private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
        r, s = utils.decode_dss_signature(der_signature)
        raw_signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        signature = TestValidateOfflineLicense._b64url_encode(raw_signature)
        return f"{header}.{body}.{signature}"

    @staticmethod
    def _rsa_private_key():
        """Create a small test RSA keypair."""
        from cryptography.hazmat.primitives.asymmetric import rsa

        return rsa.generate_private_key(public_exponent=65537, key_size=2048)

    @staticmethod
    def _ec_private_key():
        """Create a P-256 test keypair."""
        from cryptography.hazmat.primitives.asymmetric import ec

        return ec.generate_private_key(ec.SECP256R1())

    @staticmethod
    def _public_key_pem(private_key) -> str:
        """Serialize a test public key (RSA or EC) to PEM."""
        from cryptography.hazmat.primitives import serialization

        return (
            private_key.public_key()
            .public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode("ascii")
        )


# ---------------------------------------------------------------------------
# LicenseValidator._decode_license_token
# ---------------------------------------------------------------------------


class TestDecodeLicenseToken:
    """Test LicenseValidator._decode_license_token."""

    def test_invalid_format_no_dots(self):
        """Test token without dots returns None."""
        with patch.dict(
            os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
        ):
            validator = LicenseValidator()
            assert validator._decode_license_token("nodots") is None

    def test_invalid_format_wrong_parts(self):
        """Test token with wrong number of parts returns None."""
        with patch.dict(
            os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
        ):
            validator = LicenseValidator()
            assert validator._decode_license_token("one.two") is None
            assert validator._decode_license_token("one.two.three.four") is None

    def test_valid_token_no_expiry(self):
        """Test a signed valid token without expiry is accepted."""
        payload = {"tier": "enterprise", "features": ["cloud_execution"], "org": "Acme"}
        private_key = TestValidateOfflineLicense._rsa_private_key()
        token = TestValidateOfflineLicense._create_rs256_token(payload, private_key)

        with patch.dict(
            os.environ,
            {
                "TRAIGENT_LICENSE_PUBLIC_KEY": TestValidateOfflineLicense._public_key_pem(
                    private_key
                )
            },
            clear=True,
        ):
            validator = LicenseValidator()
            result = validator._decode_license_token(token)
        assert result is not None
        assert result.tier == LicenseTier.ENTERPRISE
        assert LicenseFeature.CLOUD_EXECUTION in result.features
        assert result.organization == "Acme"
        assert result.validation_source == "offline"

    def test_valid_es256_token_is_accepted(self):
        """Test a valid ES256 token exercises the ECDSA verification branch."""
        payload = {"tier": "pro", "features": ["parallel_execution"], "org": "Acme"}
        private_key = TestValidateOfflineLicense._ec_private_key()
        token = TestValidateOfflineLicense._create_es256_token(payload, private_key)

        with patch.dict(
            os.environ,
            {
                "TRAIGENT_LICENSE_PUBLIC_KEY": TestValidateOfflineLicense._public_key_pem(
                    private_key
                )
            },
            clear=True,
        ):
            validator = LicenseValidator()
            result = validator._decode_license_token(token)
        assert result is not None
        assert result.tier == LicenseTier.PRO
        assert LicenseFeature.PARALLEL_EXECUTION in result.features
        assert result.organization == "Acme"
        assert result.validation_source == "offline"

    def test_es256_token_tampering_is_rejected(self):
        """Test ES256 payload tampering fails after raw-signature conversion."""
        private_key = TestValidateOfflineLicense._ec_private_key()
        token = TestValidateOfflineLicense._create_es256_token(
            {"tier": "free", "features": []}, private_key
        )
        header, _body, signature = token.split(".")
        tampered_body = TestValidateOfflineLicense._b64url_encode(
            json.dumps({"tier": "pro", "features": ["parallel_execution"]}).encode()
        )
        tampered_token = f"{header}.{tampered_body}.{signature}"

        with patch.dict(
            os.environ,
            {
                "TRAIGENT_LICENSE_PUBLIC_KEY": TestValidateOfflineLicense._public_key_pem(
                    private_key
                )
            },
            clear=True,
        ):
            validator = LicenseValidator()
            assert validator._decode_license_token(tampered_token) is None

    def test_signed_token_tampering_is_rejected(self):
        """Test a signed token with a modified payload is rejected."""
        private_key = TestValidateOfflineLicense._rsa_private_key()
        token = TestValidateOfflineLicense._create_rs256_token(
            {"tier": "free", "features": []}, private_key
        )
        header, _body, signature = token.split(".")
        tampered_body = TestValidateOfflineLicense._b64url_encode(
            json.dumps({"tier": "enterprise", "features": ["cloud_execution"]}).encode()
        )
        tampered_token = f"{header}.{tampered_body}.{signature}"

        with patch.dict(
            os.environ,
            {
                "TRAIGENT_LICENSE_PUBLIC_KEY": TestValidateOfflineLicense._public_key_pem(
                    private_key
                )
            },
            clear=True,
        ):
            validator = LicenseValidator()
            assert validator._decode_license_token(tampered_token) is None

    def test_signed_token_with_unsupported_algorithm_is_rejected(self):
        """Test configured-key verification rejects unsupported JWT algorithms."""
        private_key = TestValidateOfflineLicense._rsa_private_key()
        header = TestValidateOfflineLicense._b64url_encode(
            json.dumps({"alg": "HS256"}).encode()
        )
        body = TestValidateOfflineLicense._b64url_encode(
            json.dumps({"tier": "pro", "features": ["parallel_execution"]}).encode()
        )
        token = f"{header}.{body}.{TestValidateOfflineLicense._b64url_encode(b'sig')}"

        with patch.dict(
            os.environ,
            {
                "TRAIGENT_LICENSE_PUBLIC_KEY": TestValidateOfflineLicense._public_key_pem(
                    private_key
                )
            },
            clear=True,
        ):
            validator = LicenseValidator()
            assert validator._decode_license_token(token) is None

    def test_malformed_es256_signature_is_rejected(self):
        """Test ES256 signatures must be raw 64-byte P1363 signatures."""
        private_key = TestValidateOfflineLicense._ec_private_key()
        token = TestValidateOfflineLicense._create_es256_token(
            {"tier": "pro", "features": ["parallel_execution"]}, private_key
        )
        header, body, _signature = token.split(".")
        malformed_token = (
            f"{header}.{body}.{TestValidateOfflineLicense._b64url_encode(b'short')}"
        )

        with patch.dict(
            os.environ,
            {
                "TRAIGENT_LICENSE_PUBLIC_KEY": TestValidateOfflineLicense._public_key_pem(
                    private_key
                )
            },
            clear=True,
        ):
            validator = LicenseValidator()
            assert validator._decode_license_token(malformed_token) is None

    def test_unsigned_token_rejected_by_default(self):
        """Test unsigned offline tokens are rejected unless explicitly allowed."""
        payload = {"tier": "enterprise", "features": ["cloud_execution"], "org": "Acme"}
        token = TestValidateOfflineLicense._create_token(payload)

        with patch.dict(os.environ, {}, clear=True):
            validator = LicenseValidator()
            assert validator._decode_license_token(token) is None

    def test_signed_looking_token_rejected_even_when_legacy_is_allowed(self):
        """Test legacy mode refuses signed-looking tokens without verification."""
        private_key = TestValidateOfflineLicense._rsa_private_key()
        token = TestValidateOfflineLicense._create_rs256_token(
            {"tier": "pro", "features": ["parallel_execution"]}, private_key
        )

        with patch.dict(
            os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
        ):
            validator = LicenseValidator()
            assert validator._decode_license_token(token) is None

    def test_null_algorithm_token_rejected_even_when_legacy_is_allowed(self):
        """Test legacy mode requires an explicit alg=none header."""
        header = TestValidateOfflineLicense._b64url_encode(
            json.dumps({"alg": None}).encode()
        )
        body = TestValidateOfflineLicense._b64url_encode(
            json.dumps({"tier": "pro", "features": ["parallel_execution"]}).encode()
        )
        token = f"{header}.{body}.{TestValidateOfflineLicense._b64url_encode(b'sig')}"

        with patch.dict(
            os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
        ):
            validator = LicenseValidator()
            assert validator._decode_license_token(token) is None

    def test_empty_public_key_configuration_fails_closed(self):
        """Test configured-but-empty public key does not fall back to legacy mode."""
        payload = {"tier": "enterprise", "features": ["cloud_execution"], "org": "Acme"}
        token = TestValidateOfflineLicense._create_token(payload)

        with patch.dict(
            os.environ,
            {
                "TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true",
                "TRAIGENT_LICENSE_PUBLIC_KEY": "",
            },
            clear=True,
        ):
            validator = LicenseValidator()
            assert validator._decode_license_token(token) is None

    def test_public_key_file_configuration_is_used(self):
        """Test signed offline licenses can be verified from a public-key file."""
        payload = {"tier": "pro", "features": ["parallel_execution"], "org": "FileCorp"}
        private_key = TestValidateOfflineLicense._rsa_private_key()
        token = TestValidateOfflineLicense._create_rs256_token(payload, private_key)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as f:
            f.write(TestValidateOfflineLicense._public_key_pem(private_key))
            f.flush()
            temp_path = f.name

        try:
            with patch.dict(
                os.environ,
                {"TRAIGENT_LICENSE_PUBLIC_KEY_FILE": temp_path},
                clear=True,
            ):
                validator = LicenseValidator()
                result = validator._decode_license_token(token)
            assert result is not None
            assert result.tier == LicenseTier.PRO
            assert result.organization == "FileCorp"
            assert result.validation_source == "offline"
        finally:
            os.unlink(temp_path)

    def test_require_signed_overrides_unsigned_legacy_escape_hatch(self):
        """Test require-signed mode rejects unsigned tokens even if legacy is allowed."""
        payload = {"tier": "enterprise", "features": ["cloud_execution"], "org": "Acme"}
        token = TestValidateOfflineLicense._create_token(payload)

        with patch.dict(
            os.environ,
            {
                "TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true",
                "TRAIGENT_REQUIRE_SIGNED_LICENSE": "true",
            },
            clear=True,
        ):
            validator = LicenseValidator()
            assert validator._decode_license_token(token) is None

    def test_valid_token_defaults_to_free(self):
        """Test explicitly allowed legacy token with no tier defaults to free."""
        payload = {"features": []}
        token = TestValidateOfflineLicense._create_token(payload)

        with patch.dict(
            os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
        ):
            validator = LicenseValidator()
            result = validator._decode_license_token(token)
        assert result is not None
        assert result.tier == LicenseTier.FREE

    def test_invalid_base64_payload(self):
        """Test token with invalid base64 payload returns None."""
        with patch.dict(
            os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
        ):
            validator = LicenseValidator()
            result = validator._decode_license_token("header.!!!invalid!!!.signature")
        assert result is None

    def test_invalid_json_payload(self):
        """Test token with non-JSON payload returns None."""
        header = TestValidateOfflineLicense._b64url_encode(
            json.dumps({"alg": "none"}).encode()
        )
        bad_payload = base64.urlsafe_b64encode(b"not json").decode().rstrip("=")
        token = f"{header}.{bad_payload}.signature"
        with patch.dict(
            os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
        ):
            validator = LicenseValidator()
            result = validator._decode_license_token(token)
        assert result is None

    def test_invalid_tier_value(self):
        """Test token with invalid tier string returns None."""
        payload = {"tier": "platinum", "features": []}
        token = TestValidateOfflineLicense._create_token(payload)

        with patch.dict(
            os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
        ):
            validator = LicenseValidator()
            result = validator._decode_license_token(token)
        assert result is None

    def test_invalid_feature_value(self):
        """Test token with invalid feature string returns None."""
        payload = {"tier": "pro", "features": ["nonexistent_feature"]}
        token = TestValidateOfflineLicense._create_token(payload)

        with patch.dict(
            os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
        ):
            validator = LicenseValidator()
            result = validator._decode_license_token(token)
        assert result is None

    def test_base64_padding_handling(self):
        """Test that base64 padding is handled correctly for various payload lengths."""
        # Try payloads of different lengths to exercise the padding logic
        with patch.dict(
            os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
        ):
            for extra in ["", "x", "xx", "xxx"]:
                payload = {"tier": "free", "features": [], "extra": extra}
                token = TestValidateOfflineLicense._create_token(payload)
                validator = LicenseValidator()
                result = validator._decode_license_token(token)
                assert result is not None
                assert result.tier == LicenseTier.FREE


# ---------------------------------------------------------------------------
# LicenseValidator._validate_cloud_license (async)
# ---------------------------------------------------------------------------


class TestValidateCloudLicense:
    """Test LicenseValidator._validate_cloud_license."""

    @pytest.mark.asyncio
    async def test_returns_none_in_offline_mode(self):
        """Test cloud validation skipped in offline mode."""
        validator = LicenseValidator(api_key="key", offline_mode=True)
        result = await validator._validate_cloud_license()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_without_api_key(self):
        """Test cloud validation skipped without API key."""
        with patch.dict(os.environ, {}, clear=True):
            validator = LicenseValidator(offline_mode=False)
            result = await validator._validate_cloud_license()
            assert result is None

    @pytest.mark.asyncio
    async def test_successful_cloud_validation(self):
        """Test successful cloud license validation."""
        mock_response = {
            "tier": "pro",
            "features": ["parallel_execution", "multi_objective"],
            "expires_at": time.time() + 86400,
            "organization": "CloudCorp",
        }

        mock_client = AsyncMock()
        mock_client.get_license_features = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "traigent.cloud.client.TraigentCloudClient",
            return_value=mock_client,
        ):
            validator = LicenseValidator(api_key="test-key", offline_mode=False)
            result = await validator._validate_cloud_license()

        assert result is not None
        assert result.tier == LicenseTier.PRO
        assert LicenseFeature.PARALLEL_EXECUTION in result.features
        assert result.organization == "CloudCorp"
        assert result.validation_source == "cloud"

    @pytest.mark.asyncio
    async def test_cloud_returns_none_response(self):
        """Test cloud validation when API returns None."""
        mock_client = AsyncMock()
        mock_client.get_license_features = AsyncMock(return_value=None)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "traigent.cloud.client.TraigentCloudClient",
            return_value=mock_client,
        ):
            validator = LicenseValidator(api_key="test-key", offline_mode=False)
            result = await validator._validate_cloud_license()

        assert result is None

    @pytest.mark.asyncio
    async def test_cloud_timeout(self):
        """Test cloud validation handles timeout."""

        async def slow_call():
            await asyncio.sleep(10)

        mock_client = AsyncMock()
        mock_client.get_license_features = slow_call
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "traigent.cloud.client.TraigentCloudClient",
            return_value=mock_client,
        ):
            validator = LicenseValidator(api_key="test-key", offline_mode=False)
            # Override timeout to make test fast
            validator.CLOUD_VALIDATION_TIMEOUT = 0.01
            result = await validator._validate_cloud_license()

        assert result is None

    @pytest.mark.asyncio
    async def test_cloud_import_error(self):
        """Test cloud validation handles missing cloud client."""
        with patch(
            "traigent.cloud.client.TraigentCloudClient",
            side_effect=ImportError("no cloud client"),
        ):
            # We need to make the import inside the method fail
            pass

        # Simpler approach: patch the import mechanism at the function level
        validator = LicenseValidator(api_key="test-key", offline_mode=False)

        with patch.dict("sys.modules", {"traigent.cloud.client": None}):
            # Force the import to fail by making the module None
            # The actual method does a local import, so we mock it differently
            with patch(
                "traigent.cloud.client.TraigentCloudClient",
                side_effect=ImportError("no cloud"),
            ):
                result = await validator._validate_cloud_license()
                assert result is None

    @pytest.mark.asyncio
    async def test_cloud_general_exception(self):
        """Test cloud validation handles general exceptions."""
        mock_client = AsyncMock()
        mock_client.get_license_features = AsyncMock(
            side_effect=ConnectionError("network down")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "traigent.cloud.client.TraigentCloudClient",
            return_value=mock_client,
        ):
            validator = LicenseValidator(api_key="test-key", offline_mode=False)
            result = await validator._validate_cloud_license()

        assert result is None


# ---------------------------------------------------------------------------
# LicenseValidator.validate_async
# ---------------------------------------------------------------------------


class TestValidateAsync:
    """Test LicenseValidator.validate_async."""

    @pytest.mark.asyncio
    async def test_returns_cached_license_when_valid(self):
        """Test validate_async returns cached license if valid."""
        validator = LicenseValidator(cache_ttl=3600)
        cached = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            validated_at=time.time(),
            validation_source="cloud",
        )
        validator._cached_license = cached

        result = await validator.validate_async()
        assert result is cached

    @pytest.mark.asyncio
    async def test_tries_offline_when_cache_stale(self):
        """Test validate_async tries offline license when cache stale."""
        payload = {
            "tier": "pro",
            "features": ["parallel_execution"],
            "exp": time.time() + 86400,
        }
        token = TestValidateOfflineLicense._create_token(payload)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jwt", delete=False) as f:
            f.write(token)
            f.flush()
            temp_path = f.name

        try:
            with patch.dict(
                os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
            ):
                validator = LicenseValidator(
                    license_file=temp_path,
                    offline_mode=True,
                )
                result = await validator.validate_async()
            assert result.tier == LicenseTier.PRO
            # See C3 phased rollout: unsigned tokens use the legacy tag.
            assert result.validation_source == "offline_unsigned_legacy"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_falls_back_to_free_tier(self):
        """Test validate_async falls back to free tier when all validation fails."""
        with patch.dict(os.environ, {}, clear=True):
            validator = LicenseValidator(offline_mode=True)
            result = await validator.validate_async()
            assert result.tier == LicenseTier.FREE
            assert result.features == set()

    @pytest.mark.asyncio
    async def test_grace_period_used_when_cloud_fails(self):
        """Test validate_async uses grace period when cloud validation fails."""
        validator = LicenseValidator(
            api_key="test-key",  # pragma: allowlist secret
            offline_mode=True,  # skip cloud to simplify
            grace_period=86400,
        )
        # Set a stale but within grace period cached license
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            validated_at=time.time() - 7200,  # 2 hours ago
            validation_source="cloud",
        )
        validator._cache_ttl = 3600  # 1 hour TTL - cache is stale

        result = await validator.validate_async()
        # Should return grace period license
        assert result.tier == LicenseTier.PRO
        assert result.validation_source == "grace"

    @pytest.mark.asyncio
    async def test_cloud_validation_success(self):
        """Test validate_async succeeds via cloud validation."""
        mock_response = {
            "tier": "enterprise",
            "features": ["cloud_execution", "parallel_execution"],
            "expires_at": time.time() + 86400,
            "organization": "CloudOrg",
        }

        mock_client = AsyncMock()
        mock_client.get_license_features = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "traigent.cloud.client.TraigentCloudClient",
            return_value=mock_client,
        ):
            validator = LicenseValidator(api_key="test-key", offline_mode=False)
            result = await validator.validate_async()

        assert result.tier == LicenseTier.ENTERPRISE
        assert result.validation_source == "cloud"
        assert validator._cached_license is result


# ---------------------------------------------------------------------------
# LicenseValidator.validate_sync
# ---------------------------------------------------------------------------


class TestValidateSync:
    """Test LicenseValidator.validate_sync."""

    def test_returns_cached_license_in_running_loop(self):
        """Test validate_sync returns cached license when event loop is running."""
        validator = LicenseValidator()
        cached = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            validated_at=time.time(),
            validation_source="cloud",
        )
        validator._cached_license = cached

        with patch("asyncio.get_running_loop", return_value=MagicMock()):
            result = validator.validate_sync()

        assert result is cached

    def test_returns_free_in_running_loop_no_cache(self):
        """Test validate_sync returns free license in running loop with no cache."""
        validator = LicenseValidator()

        with patch("asyncio.get_running_loop", return_value=MagicMock()):
            result = validator.validate_sync()

        assert result.tier == LicenseTier.FREE

    def test_runs_validate_async_when_no_loop_is_running(self):
        """Test validate_sync runs validate_async via asyncio.run when needed."""
        validator = LicenseValidator(offline_mode=True)
        expected = LicenseInfo(
            tier=LicenseTier.FREE,
            features=set(),
            validation_source="none",
        )

        def _run_stub(coro):
            coro.close()
            return expected

        with patch(
            "asyncio.get_running_loop", side_effect=RuntimeError("no running loop")
        ):
            with patch("asyncio.run", side_effect=_run_stub) as mock_run:
                result = validator.validate_sync()

        assert result is expected
        mock_run.assert_called_once()

    def test_handles_runtime_error_no_loop(self):
        """Test validate_sync handles RuntimeError when no event loop exists."""
        validator = LicenseValidator(offline_mode=True)
        expected = LicenseInfo(
            tier=LicenseTier.FREE,
            features=set(),
            validation_source="none",
        )

        def _run_stub(coro):
            coro.close()
            return expected

        with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
            with patch("asyncio.run", side_effect=_run_stub) as mock_run:
                result = validator.validate_sync()

        assert result is expected

    def test_uses_fresh_runner_when_default_loop_is_closed(self):
        """A closed default loop should not break synchronous validation."""
        validator = LicenseValidator(offline_mode=True)
        expected = LicenseInfo(
            tier=LicenseTier.FREE,
            features=set(),
            validation_source="none",
        )

        closed_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(closed_loop)
        closed_loop.close()

        try:
            with patch.object(
                validator, "validate_async", AsyncMock(return_value=expected)
            ):
                result = validator.validate_sync()
        finally:
            asyncio.set_event_loop(None)

        assert result is expected


# ---------------------------------------------------------------------------
# LicenseValidator.has_feature_sync
# ---------------------------------------------------------------------------


class TestHasFeatureSync:
    """Test LicenseValidator.has_feature_sync."""

    def test_returns_true_for_cached_valid_feature(self):
        """Test has_feature_sync returns True when cached license has feature."""
        validator = LicenseValidator(cache_ttl=3600)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            validated_at=time.time(),
        )
        assert validator.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION) is True

    def test_returns_false_for_cached_missing_feature(self):
        """Test has_feature_sync returns False when cached license lacks feature."""
        validator = LicenseValidator(cache_ttl=3600)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            validated_at=time.time(),
        )
        assert validator.has_feature_sync(LicenseFeature.CLOUD_EXECUTION) is False

    def test_accepts_string_feature(self):
        """Test has_feature_sync accepts string feature names."""
        validator = LicenseValidator(cache_ttl=3600)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            validated_at=time.time(),
        )
        assert validator.has_feature_sync("parallel_execution") is True

    def test_unknown_string_feature_returns_false(self):
        """Test has_feature_sync returns False for unknown feature string."""
        validator = LicenseValidator(cache_ttl=3600)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            validated_at=time.time(),
        )
        assert validator.has_feature_sync("nonexistent_feature") is False

    def test_tries_offline_when_cache_stale(self):
        """Test has_feature_sync tries offline when cache is stale."""
        payload = {
            "tier": "pro",
            "features": ["parallel_execution"],
            "exp": time.time() + 86400,
        }
        token = TestValidateOfflineLicense._create_token(payload)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jwt", delete=False) as f:
            f.write(token)
            f.flush()
            temp_path = f.name

        try:
            with patch.dict(
                os.environ, {"TRAIGENT_ALLOW_UNSIGNED_LICENSE": "true"}, clear=True
            ):
                validator = LicenseValidator(license_file=temp_path)
                result = validator.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION)
            assert result is True
        finally:
            os.unlink(temp_path)

    def test_uses_grace_period(self):
        """Test has_feature_sync uses grace period when no fresh validation."""
        validator = LicenseValidator(grace_period=86400)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.MULTI_OBJECTIVE},
            validated_at=time.time() - 7200,  # 2 hours stale
        )
        validator._cache_ttl = 3600  # 1 hour TTL
        assert validator.has_feature_sync(LicenseFeature.MULTI_OBJECTIVE) is True

    def test_returns_false_no_valid_license(self):
        """Test has_feature_sync returns False when no valid license at all."""
        with patch.dict(os.environ, {}, clear=True):
            validator = LicenseValidator()
            assert (
                validator.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION) is False
            )


# ---------------------------------------------------------------------------
# LicenseValidator.has_feature_async
# ---------------------------------------------------------------------------


class TestHasFeatureAsync:
    """Test LicenseValidator.has_feature_async."""

    @pytest.mark.asyncio
    async def test_returns_true_for_valid_feature(self):
        """Test has_feature_async returns True when feature available."""
        validator = LicenseValidator(cache_ttl=3600)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            validated_at=time.time(),
        )
        result = await validator.has_feature_async(LicenseFeature.PARALLEL_EXECUTION)
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_missing_feature(self):
        """Test has_feature_async returns False when feature not available."""
        validator = LicenseValidator(offline_mode=True)
        result = await validator.has_feature_async(LicenseFeature.CLOUD_EXECUTION)
        assert result is False

    @pytest.mark.asyncio
    async def test_accepts_string_feature(self):
        """Test has_feature_async accepts string feature names."""
        validator = LicenseValidator(cache_ttl=3600)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            validated_at=time.time(),
        )
        result = await validator.has_feature_async("parallel_execution")
        assert result is True

    @pytest.mark.asyncio
    async def test_unknown_string_returns_false(self):
        """Test has_feature_async returns False for unknown string feature."""
        validator = LicenseValidator(cache_ttl=3600)
        result = await validator.has_feature_async("unknown_feature")
        assert result is False


# ---------------------------------------------------------------------------
# LicenseValidator.require_feature
# ---------------------------------------------------------------------------


class TestRequireFeature:
    """Test LicenseValidator.require_feature."""

    def test_raises_for_missing_feature(self):
        """Test require_feature raises when feature not available."""
        with patch.dict(os.environ, {}, clear=True):
            validator = LicenseValidator()
            with pytest.raises(FeatureRequiresLicenseError) as exc_info:
                validator.require_feature(LicenseFeature.PARALLEL_EXECUTION)
            assert exc_info.value.feature == "parallel_execution"
            assert exc_info.value.required_tier == "Pro"

    def test_raises_enterprise_for_cloud_execution(self):
        """Test require_feature raises with Enterprise tier for cloud_execution."""
        with patch.dict(os.environ, {}, clear=True):
            validator = LicenseValidator()
            with pytest.raises(FeatureRequiresLicenseError) as exc_info:
                validator.require_feature(LicenseFeature.CLOUD_EXECUTION)
            assert exc_info.value.required_tier == "Enterprise"

    def test_no_error_when_feature_available(self):
        """Test require_feature does not raise when feature is available."""
        validator = LicenseValidator(cache_ttl=3600)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            validated_at=time.time(),
        )
        # Should not raise
        validator.require_feature(LicenseFeature.PARALLEL_EXECUTION)

    def test_accepts_string_feature(self):
        """Test require_feature accepts string feature names."""
        with patch.dict(os.environ, {}, clear=True):
            validator = LicenseValidator()
            with pytest.raises(FeatureRequiresLicenseError) as exc_info:
                validator.require_feature("parallel_execution")
            assert exc_info.value.feature == "parallel_execution"

    def test_unknown_string_feature_raises(self):
        """Test require_feature raises for unknown string features."""
        validator = LicenseValidator()
        with pytest.raises(FeatureRequiresLicenseError) as exc_info:
            validator.require_feature("nonexistent_feature")
        assert exc_info.value.feature == "nonexistent_feature"
        assert exc_info.value.required_tier == "Unknown"

    def test_string_cloud_execution_raises_enterprise(self):
        """Test require_feature with string 'cloud_execution' raises Enterprise."""
        with patch.dict(os.environ, {}, clear=True):
            validator = LicenseValidator()
            with pytest.raises(FeatureRequiresLicenseError) as exc_info:
                validator.require_feature("cloud_execution")
            assert exc_info.value.required_tier == "Enterprise"


# ---------------------------------------------------------------------------
# LicenseValidator.get_license_info
# ---------------------------------------------------------------------------


class TestGetLicenseInfo:
    """Test LicenseValidator.get_license_info."""

    def test_returns_cached_license(self):
        """Test get_license_info returns cached license when available."""
        validator = LicenseValidator()
        cached = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            validated_at=time.time(),
            validation_source="cloud",
        )
        validator._cached_license = cached
        result = validator.get_license_info()
        assert result is cached

    def test_returns_free_when_no_cache(self):
        """Test get_license_info returns free license when nothing cached."""
        validator = LicenseValidator()
        result = validator.get_license_info()
        assert result.tier == LicenseTier.FREE
        assert result.features == set()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_get_license_validator_returns_singleton(self):
        """Test get_license_validator returns a LicenseValidator."""
        # Reset the global singleton for isolation
        import traigent.core.license as license_mod

        original = license_mod._license_validator
        try:
            license_mod._license_validator = None
            validator = get_license_validator()
            assert isinstance(validator, LicenseValidator)

            # Second call returns same instance
            validator2 = get_license_validator()
            assert validator is validator2
        finally:
            license_mod._license_validator = original

    def test_has_feature_convenience(self):
        """Test has_feature convenience function delegates to validator."""
        import traigent.core.license as license_mod

        original = license_mod._license_validator
        try:
            # Set up a validator with a known cached license
            validator = LicenseValidator(cache_ttl=3600)
            validator._cached_license = LicenseInfo(
                tier=LicenseTier.PRO,
                features={LicenseFeature.PARALLEL_EXECUTION},
                validated_at=time.time(),
            )
            license_mod._license_validator = validator

            assert has_feature(LicenseFeature.PARALLEL_EXECUTION) is True
            assert has_feature(LicenseFeature.CLOUD_EXECUTION) is False
        finally:
            license_mod._license_validator = original

    def test_require_feature_convenience_raises(self):
        """Test require_feature convenience function raises when feature missing."""
        import traigent.core.license as license_mod

        original = license_mod._license_validator
        try:
            license_mod._license_validator = LicenseValidator()
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(FeatureRequiresLicenseError):
                    require_feature(LicenseFeature.PARALLEL_EXECUTION)
        finally:
            license_mod._license_validator = original

    def test_require_feature_convenience_no_error(self):
        """Test require_feature convenience function does not raise when available."""
        import traigent.core.license as license_mod

        original = license_mod._license_validator
        try:
            validator = LicenseValidator(cache_ttl=3600)
            validator._cached_license = LicenseInfo(
                tier=LicenseTier.ENTERPRISE,
                features=set(LicenseFeature),
                validated_at=time.time(),
            )
            license_mod._license_validator = validator

            # Should not raise for any feature
            require_feature(LicenseFeature.PARALLEL_EXECUTION)
            require_feature(LicenseFeature.CLOUD_EXECUTION)
        finally:
            license_mod._license_validator = original


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Test thread safety of LicenseValidator."""

    def test_concurrent_has_feature_sync(self):
        """Test has_feature_sync is thread-safe under concurrent access."""
        validator = LicenseValidator(cache_ttl=3600)
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features={
                LicenseFeature.PARALLEL_EXECUTION,
                LicenseFeature.MULTI_OBJECTIVE,
            },
            validated_at=time.time(),
        )

        results = []
        errors = []

        def check_feature():
            try:
                r = validator.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION)
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_feature) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r is True for r in results)
        assert len(results) == 20

    def test_concurrent_get_license_info(self):
        """Test get_license_info is thread-safe under concurrent access."""
        validator = LicenseValidator()
        validator._cached_license = LicenseInfo(
            tier=LicenseTier.PRO,
            features={LicenseFeature.PARALLEL_EXECUTION},
            validated_at=time.time(),
        )

        results = []

        def get_info():
            info = validator.get_license_info()
            results.append(info.tier)

        threads = [threading.Thread(target=get_info) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(tier == LicenseTier.PRO for tier in results)


# ---------------------------------------------------------------------------
# Constants on LicenseValidator
# ---------------------------------------------------------------------------


class TestValidatorConstants:
    """Test LicenseValidator class constants."""

    def test_default_cache_ttl(self):
        """Test DEFAULT_CACHE_TTL constant value."""
        assert LicenseValidator.DEFAULT_CACHE_TTL == 3600

    def test_default_grace_period(self):
        """Test DEFAULT_GRACE_PERIOD constant value."""
        assert LicenseValidator.DEFAULT_GRACE_PERIOD == 86400

    def test_cloud_validation_timeout(self):
        """Test CLOUD_VALIDATION_TIMEOUT constant value."""
        assert LicenseValidator.CLOUD_VALIDATION_TIMEOUT == 5.0
