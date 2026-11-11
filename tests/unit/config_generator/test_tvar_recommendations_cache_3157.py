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
