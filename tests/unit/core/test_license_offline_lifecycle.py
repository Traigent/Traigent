"""Offline licence lifecycle tests.

These tests pin what the SDK does TODAY with no network: valid, expired,
missing-expiry, grace-window and bad-signature licences, and a missing
file. The clock is frozen by replacing the
``time`` name inside ``traigent.core.license``; there are no sleeps and no
network calls (``offline_mode=True``).

Current behaviour pinned here (not a statement of the desired policy):
- The SDK default grace period is 86400 s (24 h).
  ``test_default_grace_period_is_current_24h_value`` pins that default.
- A licence with a missing or non-numeric ``exp`` is refused
  (``test_missing_expiry_is_refused``).
"""

from __future__ import annotations

import types

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import traigent.core.license as license_mod
from traigent.core.license import (
    LicenseFeature,
    LicenseTier,
    LicenseValidator,
)

NOW = 1_800_000_000.0
GRACE = 86400  # current SDK default, see LicenseValidator.DEFAULT_GRACE_PERIOD


class FrozenClock:
    def __init__(self, now: float = NOW) -> None:
        self.now = now

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = FrozenClock()
    monkeypatch.setattr(license_mod, "time", types.SimpleNamespace(time=c.time))
    return c


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "TRAIGENT_LICENSE_PUBLIC_KEY",
        "TRAIGENT_LICENSE_PUBLIC_KEY_FILE",
        "TRAIGENT_REQUIRE_SIGNED_LICENSE",
        "TRAIGENT_ALLOW_UNSIGNED_LICENSE",
        "TRAIGENT_LICENSE_FILE",
        "TRAIGENT_LICENSE_GRACE_PERIOD",
        "TRAIGENT_LICENSE_CACHE_TTL",
        "TRAIGENT_API_KEY",
        "TRAIGENT_OFFLINE_MODE",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return key, pem.decode()


def _token(key, **claims) -> str:
    payload = {"tier": "pro", "features": ["parallel_execution"], "org": "Acme"}
    payload.update(claims)
    return pyjwt.encode(payload, key, algorithm="RS256")


@pytest.fixture
def make_validator(tmp_path, keypair, monkeypatch):
    _, pem = keypair
    monkeypatch.setenv("TRAIGENT_LICENSE_PUBLIC_KEY", pem)

    def _make(token: str | None, **kwargs) -> LicenseValidator:
        path = tmp_path / "license.jwt"
        if token is not None:
            path.write_text(token)
        return LicenseValidator(offline_mode=True, license_file=str(path), **kwargs)

    return _make


class TestOfflineLicenceStates:
    def test_valid_licence_grants_features(self, clock, keypair, make_validator):
        key, _ = keypair
        v = make_validator(_token(key, exp=NOW + 3600))
        info = v.validate_sync()
        assert info.tier == LicenseTier.PRO
        assert info.validation_source == "offline"
        assert v.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION) is True

    def test_expired_licence_falls_back_to_free(self, clock, keypair, make_validator):
        key, _ = keypair
        v = make_validator(_token(key, exp=NOW - 1))
        info = v.validate_sync()
        assert info.tier == LicenseTier.FREE
        assert info.validation_source == "none"
        assert v.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION) is False

    def test_expiry_boundary_is_strictly_greater_than(
        self, clock, keypair, make_validator
    ):
        """Revalidate through LicenseValidator as the clock advances: at
        now == exp the licence is still valid, one second later it is not."""
        key, _ = keypair
        token = _token(key, exp=NOW + 10)

        clock.advance(10)  # now == exp
        at_exp = make_validator(token)
        assert at_exp.validate_sync().tier == LicenseTier.PRO
        assert at_exp.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION) is True

        clock.advance(1)  # now == exp + 1
        after_exp = make_validator(token)
        assert after_exp.validate_sync().tier == LicenseTier.FREE
        assert after_exp.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION) is False

    def test_missing_expiry_is_refused(self, clock, keypair, make_validator, caplog):
        """A signed licence without ``exp`` fails closed to the free tier."""
        key, _ = keypair
        v = make_validator(_token(key))
        with caplog.at_level("WARNING"):
            info = v.validate_sync()
        assert info.tier == LicenseTier.FREE
        assert info.validation_source == "none"
        assert v.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION) is False
        assert "missing or invalid expiry" in caplog.text
        assert "Acme" not in caplog.text

    @pytest.mark.parametrize(
        "bad_exp",
        ["4102444800", None, True, [1], {"a": 1}, float("inf"), float("nan")],
    )
    def test_non_numeric_expiry_is_refused(
        self, clock, keypair, make_validator, bad_exp
    ):
        key, _ = keypair
        v = make_validator(_token(key, exp=bad_exp))
        assert v.validate_sync().tier == LicenseTier.FREE
        assert v.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION) is False

    def test_bad_signature_is_rejected(self, clock, make_validator):
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        v = make_validator(_token(other, exp=NOW + 3600))
        assert v.validate_sync().tier == LicenseTier.FREE
        assert v.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION) is False

    def test_tampered_payload_is_rejected(self, clock, keypair, make_validator):
        key, _ = keypair
        good = _token(key, exp=NOW + 3600).split(".")
        forged = _token(key, tier="enterprise", exp=NOW + 3600).split(".")
        tampered = ".".join([good[0], forged[1], good[2]])
        assert make_validator(tampered).validate_sync().tier == LicenseTier.FREE

    def test_missing_licence_file_is_free(self, clock, make_validator):
        v = make_validator(None)
        assert v.validate_sync().tier == LicenseTier.FREE
        assert v.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION) is False

    def test_no_licence_configured_is_free(self, clock):
        v = LicenseValidator(offline_mode=True)
        assert v.validate_sync().tier == LicenseTier.FREE


class TestOfflineGraceWindow:
    """Grace applies to a previously validated licence once the cache has
    lapsed and the file can no longer be validated (here: file removed)."""

    def _prime_then_lose_file(self, make_validator, keypair, tmp_path, **kw):
        key, _ = keypair
        v = make_validator(_token(key, exp=NOW + 10 * GRACE), **kw)
        assert v.validate_sync().validation_source == "offline"
        (tmp_path / "license.jwt").unlink()
        return v

    def test_default_grace_period_is_current_24h_value(self, clock):
        """Pins the current default (86400 s). Changing the default must
        update this test."""
        assert LicenseValidator.DEFAULT_GRACE_PERIOD == 86400
        assert LicenseValidator(offline_mode=True)._grace_period == 86400

    def test_inside_grace_keeps_features(
        self, clock, keypair, make_validator, tmp_path
    ):
        v = self._prime_then_lose_file(make_validator, keypair, tmp_path)
        clock.advance(GRACE - 1)
        info = v.validate_sync()
        assert info.validation_source == "grace"
        assert info.tier == LicenseTier.PRO
        assert v.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION) is True

    def test_exactly_at_grace_boundary_is_outside(
        self, clock, keypair, make_validator, tmp_path
    ):
        """age < grace is the rule, so age == 86400 s is already past grace."""
        v = self._prime_then_lose_file(make_validator, keypair, tmp_path)
        clock.advance(GRACE)
        assert v.validate_sync().tier == LicenseTier.FREE

    def test_just_past_grace_is_free_and_features_denied(
        self, clock, keypair, make_validator, tmp_path
    ):
        v = self._prime_then_lose_file(make_validator, keypair, tmp_path)
        clock.advance(GRACE + 1)
        assert v.has_feature_sync(LicenseFeature.PARALLEL_EXECUTION) is False
        assert v.validate_sync().tier == LicenseTier.FREE

    def test_custom_grace_period_is_honoured(
        self, clock, keypair, make_validator, tmp_path
    ):
        v = self._prime_then_lose_file(
            make_validator, keypair, tmp_path, grace_period=100, cache_ttl=10
        )
        clock.advance(99)
        assert v.validate_sync().validation_source == "grace"
        clock.advance(1)
        assert v.validate_sync().tier == LicenseTier.FREE
