"""Compatibility tests for fallback credential hashing."""

import pytest

from traigent.security.crypto_utils import FallbackCredentialStorage


def test_verify_password_accepts_legacy_iterated_sha256_hash() -> None:
    """Verify fallback hashes created before the PBKDF2 migration."""
    with pytest.warns(UserWarning):
        storage = FallbackCredentialStorage()

    legacy_hash = "MDEyMzQ1Njc4OWFiY2RlZvFoOzCNroWhuz4AH6fe6ILaS99lBhs8JDdOSArqStos"  # pragma: allowlist secret

    assert (
        storage.verify_password("legacy-login-value", legacy_hash) is True
    )  # pragma: allowlist secret
    assert (
        storage.verify_password("wrong-login-value", legacy_hash) is False
    )  # pragma: allowlist secret
