"""Regression tests for C3 phased license-signature rollout.

The legacy ``_decode_license_token`` only base64-decoded the middle JWT
segment without verifying the signature, so anyone who could place a
file at ``TRAIGENT_LICENSE_FILE`` could forge an enterprise tier.

The phased fix:

- When ``TRAIGENT_LICENSE_PUBLIC_KEY`` (or ``..._FILE``) is configured,
  the signature must verify; invalid / unsigned tokens are rejected.
- When ``TRAIGENT_REQUIRE_SIGNED_LICENSE`` is truthy, unsigned tokens
  are rejected even if no key is configured (strict / air-gapped mode).
- Otherwise the legacy unsigned format is still accepted with a loud
  deprecation warning, and the resulting LicenseInfo is tagged
  ``validation_source="offline_unsigned_legacy"`` so callers can tell
  trusted from legacy validations apart.
"""

from __future__ import annotations

import base64
import json
import logging
import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from traigent.core.license import LicenseTier, LicenseValidator


@pytest.fixture
def rsa_keypair():
    """Generate a fresh 2048-bit RSA keypair for each test."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_key, public_pem


def _signed_token(private_key, payload: dict, *, algorithm: str = "RS256") -> str:
    return pyjwt.encode(payload, private_key, algorithm=algorithm)


def _unsigned_token(payload: dict) -> str:
    """Build the legacy unsigned token shape: header.payload.signature."""
    header = (
        base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode())
        .decode()
        .rstrip("=")
    )
    body = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    )
    sig = base64.urlsafe_b64encode(b"forged").decode().rstrip("=")
    return f"{header}.{body}.{sig}"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "TRAIGENT_LICENSE_PUBLIC_KEY",
        "TRAIGENT_LICENSE_PUBLIC_KEY_FILE",
        "TRAIGENT_REQUIRE_SIGNED_LICENSE",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def validator():
    return LicenseValidator()


class TestSignedVerificationWhenKeyConfigured:
    def test_valid_signature_decodes_and_tags_offline(
        self, validator, rsa_keypair, monkeypatch
    ):
        private_key, public_pem = rsa_keypair
        monkeypatch.setenv("TRAIGENT_LICENSE_PUBLIC_KEY", public_pem.decode())

        token = _signed_token(
            private_key,
            {"tier": "enterprise", "features": ["cloud_execution"], "org": "Acme"},
        )

        info = validator._decode_license_token(token)

        assert info is not None
        assert info.tier == LicenseTier.ENTERPRISE
        assert info.organization == "Acme"
        assert info.validation_source == "offline", (
            "verified offline tokens must be tagged plain 'offline', "
            "not the legacy tag"
        )

    def test_unsigned_token_is_rejected_when_key_configured(
        self, validator, rsa_keypair, monkeypatch
    ):
        _, public_pem = rsa_keypair
        monkeypatch.setenv("TRAIGENT_LICENSE_PUBLIC_KEY", public_pem.decode())

        # Forged unsigned token claiming enterprise — must NOT be accepted.
        token = _unsigned_token({"tier": "enterprise", "features": []})

        assert validator._decode_license_token(token) is None

    def test_token_signed_by_wrong_key_is_rejected(
        self, validator, monkeypatch
    ):
        attacker_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        legit_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        legit_pub = legit_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        monkeypatch.setenv("TRAIGENT_LICENSE_PUBLIC_KEY", legit_pub.decode())

        token = _signed_token(
            attacker_key,
            {"tier": "enterprise", "features": []},
        )
        assert validator._decode_license_token(token) is None

    def test_alg_none_downgrade_is_rejected(
        self, validator, rsa_keypair, monkeypatch
    ):
        """Defense against the classic JWT alg=none confusion attack:
        even with a public key configured, a token whose header advertises
        alg=none must be rejected because PyJWT is told only RS256/ES256
        are acceptable.
        """
        _, public_pem = rsa_keypair
        monkeypatch.setenv("TRAIGENT_LICENSE_PUBLIC_KEY", public_pem.decode())

        # alg=none token -- PyJWT will refuse it because we restrict
        # algorithms to RS256/ES256.
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "none", "typ": "JWT"}).encode()
        ).decode().rstrip("=")
        payload = base64.urlsafe_b64encode(
            json.dumps({"tier": "enterprise"}).encode()
        ).decode().rstrip("=")
        token = f"{header}.{payload}."

        assert validator._decode_license_token(token) is None

    def test_expired_signed_token_is_rejected(
        self, validator, rsa_keypair, monkeypatch
    ):
        private_key, public_pem = rsa_keypair
        monkeypatch.setenv("TRAIGENT_LICENSE_PUBLIC_KEY", public_pem.decode())

        token = _signed_token(
            private_key,
            {
                "tier": "pro",
                "features": [],
                "exp": int(time.time()) - 60,
            },
        )
        assert validator._decode_license_token(token) is None


class TestPublicKeyFromFile:
    def test_public_key_file_env_var(
        self, validator, rsa_keypair, monkeypatch, tmp_path
    ):
        private_key, public_pem = rsa_keypair
        key_path = tmp_path / "license.pub.pem"
        key_path.write_bytes(public_pem)
        monkeypatch.setenv("TRAIGENT_LICENSE_PUBLIC_KEY_FILE", str(key_path))

        token = _signed_token(private_key, {"tier": "pro", "features": []})

        info = validator._decode_license_token(token)
        assert info is not None
        assert info.validation_source == "offline"

    def test_unreadable_public_key_file_fails_closed(
        self, validator, monkeypatch, tmp_path, caplog
    ):
        """If TRAIGENT_LICENSE_PUBLIC_KEY_FILE points at a missing file,
        the validator MUST refuse rather than silently falling through
        to the legacy unsigned path. Treating a broken key configuration
        as "no key configured" would be the exact attack path: anyone
        who can influence the env var to point at /nonexistent gets the
        legacy bypass for free.
        """
        bogus = tmp_path / "missing.pem"
        monkeypatch.setenv("TRAIGENT_LICENSE_PUBLIC_KEY_FILE", str(bogus))

        # Even an unsigned token claiming the lowest tier must be rejected.
        token = _unsigned_token({"tier": "free", "features": []})
        with caplog.at_level(logging.ERROR):
            info = validator._decode_license_token(token)

        assert info is None, (
            "configured-but-broken public key must fail closed; "
            "a fall-through to legacy would defeat the whole point"
        )
        # The loader logs the underlying read error; the decoder logs the
        # refusal. Both must reference the configured env var so operators
        # can find what to fix.
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "TRAIGENT_LICENSE_PUBLIC_KEY" in joined

    def test_invalid_inline_pem_fails_closed(
        self, validator, monkeypatch, caplog
    ):
        monkeypatch.setenv(
            "TRAIGENT_LICENSE_PUBLIC_KEY",
            "-----BEGIN PUBLIC KEY-----\nnot-a-real-pem-body\n-----END PUBLIC KEY-----\n",
        )

        token = _unsigned_token({"tier": "enterprise", "features": []})
        with caplog.at_level(logging.ERROR):
            info = validator._decode_license_token(token)

        assert info is None
        assert any(
            "invalid" in r.getMessage().lower()
            or "could not be loaded" in r.getMessage().lower()
            for r in caplog.records
        )

    def test_empty_inline_pem_fails_closed(
        self, validator, monkeypatch
    ):
        # An empty/whitespace value is a misconfiguration, not absence.
        # Treat it the same as a broken key.
        monkeypatch.setenv("TRAIGENT_LICENSE_PUBLIC_KEY", "   \n   ")

        token = _unsigned_token({"tier": "enterprise", "features": []})
        # The loader can either reject this as invalid PEM or as empty;
        # either way the result must be None.
        assert validator._decode_license_token(token) is None

    def test_truly_empty_inline_env_fails_closed(
        self, validator, monkeypatch, caplog
    ):
        """``os.environ.get("X")`` returns ``""`` for an env var set to the
        empty string and ``None`` when the var is absent. A truthy check
        cannot distinguish those, which would let an attacker who could
        clear the env var (``export TRAIGENT_LICENSE_PUBLIC_KEY=``) silently
        re-enable the legacy bypass. The resolver checks presence with
        ``in os.environ`` instead, so an empty value is treated as
        broken configuration.
        """
        monkeypatch.setenv("TRAIGENT_LICENSE_PUBLIC_KEY", "")

        token = _unsigned_token({"tier": "enterprise", "features": []})
        with caplog.at_level(logging.ERROR):
            info = validator._decode_license_token(token)

        assert info is None
        # Decoder logs that the configured key could not be loaded.
        assert any(
            "could not be loaded" in r.getMessage().lower()
            or "empty" in r.getMessage().lower()
            for r in caplog.records
        )

    def test_truly_empty_file_env_fails_closed(
        self, validator, monkeypatch
    ):
        """Same threat model as the inline case: an empty
        TRAIGENT_LICENSE_PUBLIC_KEY_FILE must not collapse to "no key
        configured". Path("").read_bytes() raises, which the loader
        treats as a broken key configuration.
        """
        monkeypatch.setenv("TRAIGENT_LICENSE_PUBLIC_KEY_FILE", "")

        token = _unsigned_token({"tier": "enterprise", "features": []})
        assert validator._decode_license_token(token) is None


class TestStrictModeWithoutKey:
    def test_require_signed_rejects_unsigned(
        self, validator, monkeypatch, caplog
    ):
        monkeypatch.setenv("TRAIGENT_REQUIRE_SIGNED_LICENSE", "true")

        token = _unsigned_token({"tier": "enterprise", "features": []})
        with caplog.at_level(logging.ERROR):
            info = validator._decode_license_token(token)

        assert info is None
        assert any(
            "TRAIGENT_REQUIRE_SIGNED_LICENSE" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_strict_mode_values(
        self, validator, monkeypatch, flag
    ):
        monkeypatch.setenv("TRAIGENT_REQUIRE_SIGNED_LICENSE", flag)
        token = _unsigned_token({"tier": "enterprise", "features": []})
        assert validator._decode_license_token(token) is None


class TestLegacyUnsignedFallback:
    def test_unsigned_token_accepted_with_loud_warning(
        self, validator, caplog
    ):
        token = _unsigned_token({"tier": "pro", "features": []})

        with caplog.at_level(logging.WARNING):
            info = validator._decode_license_token(token)

        assert info is not None
        assert info.validation_source == "offline_unsigned_legacy", (
            "the legacy tag is what observability uses to flag deployments "
            "still relying on unsigned offline licenses"
        )
        assert any(
            "deprecated" in r.getMessage().lower() and "UNSIGNED" in r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )
