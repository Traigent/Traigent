"""A failed backend recommendation fetch is not cached for the process (Traigent#3157)."""

from __future__ import annotations

from typing import Any

import pytest

from traigent.cloud.client import RecommendationBundle
from traigent.config_generator.subsystems import tvar_recommendations as mod

REAL = RecommendationBundle(value_recommendations=({"tvar": "model"},))


class _FakeClient:
    bundles: list[RecommendationBundle] = []
    calls = 0

    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def fetch_tvar_recommendations(self, **_kwargs: Any) -> RecommendationBundle:
        type(self).calls += 1
        return type(self).bundles.pop(0)

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _patched(monkeypatch: pytest.MonkeyPatch) -> None:
    mod._RECOMMENDATIONS_CACHE.clear()
    _FakeClient.calls = 0
    monkeypatch.setattr(mod, "_should_fetch_backend_recommendations", lambda: True)
    monkeypatch.setattr("traigent.cloud.client.TraigentCloudClient", _FakeClient)
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "https://one.example")
    monkeypatch.setenv("TRAIGENT_API_KEY", "key-one")
    yield
    mod._RECOMMENDATIONS_CACHE.clear()


def test_failed_fetch_is_retried_not_pinned() -> None:
    _FakeClient.bundles = [RecommendationBundle.empty(), REAL]

    first = mod._fetch_backend_recommendations("rag", ("model",))
    second = mod._fetch_backend_recommendations("rag", ("model",))

    assert first.is_empty
    assert second == REAL
    assert _FakeClient.calls == 2


def test_successful_bundle_is_cached() -> None:
    _FakeClient.bundles = [REAL]

    mod._fetch_backend_recommendations("rag", ("model",))
    again = mod._fetch_backend_recommendations("rag", ("model",))

    assert again == REAL
    assert _FakeClient.calls == 1


def test_cache_key_separates_backend_and_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = RecommendationBundle(value_recommendations=({"tvar": "temperature"},))
    _FakeClient.bundles = [REAL, other, other]

    mod._fetch_backend_recommendations("rag", ("model",))
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "https://two.example")
    assert mod._fetch_backend_recommendations("rag", ("model",)) == other
    monkeypatch.setenv("TRAIGENT_API_KEY", "key-two")
    mod._fetch_backend_recommendations("rag", ("model",))

    assert _FakeClient.calls == 3


def test_cache_is_bounded() -> None:
    _FakeClient.bundles = [REAL] * (mod._RECOMMENDATIONS_CACHE_MAX + 5)
    for i in range(mod._RECOMMENDATIONS_CACHE_MAX + 5):
        mod._fetch_backend_recommendations("rag", (f"v{i}",))

    assert len(mod._RECOMMENDATIONS_CACHE) <= mod._RECOMMENDATIONS_CACHE_MAX


# --- the credential fingerprint in the cache key -------------------------------


def _identity() -> tuple[str, str]:
    return mod._backend_cache_identity()


def test_identity_changes_with_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _identity()
    monkeypatch.setenv("TRAIGENT_API_KEY", "key-two")
    assert _identity() != first


def test_identity_changes_with_the_jwt_when_the_api_key_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_JWT_TOKEN", "jwt-one")
    first = _identity()
    monkeypatch.setenv("TRAIGENT_JWT_TOKEN", "jwt-two")
    assert _identity() != first


def test_identity_is_stable_within_a_process() -> None:
    assert _identity() == _identity()


def test_identity_does_not_expose_a_plain_hash_of_the_credential() -> None:
    import hashlib

    _url, fingerprint = _identity()
    assert "key-one" not in fingerprint
    # A bare digest of the credential is the form the weak-hash scan rejects and
    # an attacker could test guesses against.
    assert fingerprint != hashlib.sha256(b"key-one").hexdigest()[:12]
    assert fingerprint != hashlib.sha256(b"key-one\x00").hexdigest()[:12]


def test_identity_keeps_the_api_key_and_jwt_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """('a\\x00b', '') and ('a', 'b') must not share a fingerprint."""
    monkeypatch.setattr(
        _backend_config(), "get_api_key", staticmethod(lambda: "a\x00b")
    )
    monkeypatch.delenv("TRAIGENT_JWT_TOKEN", raising=False)
    joined = _identity()
    monkeypatch.setattr(_backend_config(), "get_api_key", staticmethod(lambda: "a"))
    monkeypatch.setenv("TRAIGENT_JWT_TOKEN", "b")
    assert _identity() != joined


def test_identity_accepts_a_jwt_that_is_not_valid_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undecodable environment value must not raise out of the recommender."""
    monkeypatch.setenv("TRAIGENT_JWT_TOKEN", "tok-\udcff")
    url, fingerprint = _identity()
    assert fingerprint and fingerprint != _identity_with_jwt(monkeypatch, "other")[1]


def _backend_config():
    from traigent.config.backend_config import BackendConfig

    return BackendConfig


def _identity_with_jwt(monkeypatch: pytest.MonkeyPatch, jwt: str) -> tuple[str, str]:
    monkeypatch.setenv("TRAIGENT_JWT_TOKEN", jwt)
    return _identity()


def test_identity_is_the_keyed_construction_not_a_plain_digest() -> None:
    """Pins HMAC-SHA256(key=credentials, msg=label+per-process nonce)."""
    import hashlib
    import hmac

    key = b"7:key-one0:"  # length-prefixed API key, empty JWT
    message = mod._CACHE_IDENTITY_DOMAIN + mod._CACHE_IDENTITY_NONCE
    expected = hmac.new(key, message, hashlib.sha256).hexdigest()[:12]
    assert _identity()[1] == expected
    # The plain-digest forms a weak-hash scan rejects must not match either.
    assert _identity()[1] != hashlib.sha256(key + message).hexdigest()[:12]


def test_leaked_identity_cannot_be_tested_against_guesses_without_the_nonce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The message carries a per-process nonce, so the same credential gives a
    different fingerprint under another nonce: a fingerprint seen outside this
    process cannot be checked against guessed credentials."""
    first = _identity()[1]
    monkeypatch.setattr(mod, "_CACHE_IDENTITY_NONCE", b"another-process-nonce")
    assert _identity()[1] != first
