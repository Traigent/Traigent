"""get_backend_client keys its shared client on the effective settings (#3015)."""

from __future__ import annotations

import pytest

from traigent.cloud import backend_client


class _FakeClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_client, "BackendIntegratedClient", _FakeClient)
    monkeypatch.setattr(backend_client, "_backend_clients", {}, raising=False)


def test_same_settings_share_one_client() -> None:
    a = backend_client.get_backend_client(enable_fallback=False)
    b = backend_client.get_backend_client(enable_fallback=False)
    assert a is b


def test_different_settings_get_their_own_client() -> None:
    a = backend_client.get_backend_client(
        api_key="key-a",  # pragma: allowlist secret
        base_url="http://127.0.0.1:1",
        timeout=5.0,
    )
    b = backend_client.get_backend_client(
        api_key="key-b",  # pragma: allowlist secret
        base_url="http://127.0.0.1:2",
        timeout=99.0,
    )
    assert a is not b
    assert b.kwargs["timeout"] == 99.0
    assert b.kwargs["base_url"] == "http://127.0.0.1:2"


def test_no_kwargs_and_fallback_off_are_distinct() -> None:
    assert backend_client.get_backend_client() is not (
        backend_client.get_backend_client(enable_fallback=False)
    )


def test_discard_forces_a_fresh_client() -> None:
    a = backend_client.get_backend_client(timeout=5.0)
    backend_client.discard_backend_client(a)
    assert backend_client.get_backend_client(timeout=5.0) is not a


def test_cache_is_bounded() -> None:
    for i in range(backend_client._BACKEND_CLIENT_CACHE_MAX + 10):
        backend_client.get_backend_client(timeout=float(i))
    assert (
        len(backend_client._backend_clients) <= backend_client._BACKEND_CLIENT_CACHE_MAX
    )


@pytest.mark.parametrize(
    "name", ["TRAIGENT_BACKEND_URL", "TRAIGENT_API_URL", "TRAIGENT_API_KEY"]
)
def test_env_change_between_identical_calls_builds_a_new_client(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "first-value")
    a = backend_client.get_backend_client(enable_fallback=False)
    assert backend_client.get_backend_client(enable_fallback=False) is a
    monkeypatch.setenv(name, "second-value")
    assert backend_client.get_backend_client(enable_fallback=False) is not a


def test_same_urls_with_different_explicitness_get_different_clients() -> None:
    from traigent.cloud.backend_components import BackendClientConfig

    implicit = BackendClientConfig(backend_base_url="https://example.invalid")
    explicit = BackendClientConfig(
        backend_base_url="https://example.invalid",
        api_base_url=implicit.api_base_url,
    )
    explicit.api_base_url = implicit.api_base_url
    assert (explicit.api_base_url, explicit.backend_base_url) == (
        implicit.api_base_url,
        implicit.backend_base_url,
    )
    explicit.api_explicitly_set = True
    implicit.api_explicitly_set = False
    a = backend_client.get_backend_client(backend_config=implicit)
    b = backend_client.get_backend_client(backend_config=explicit)
    assert a is not b
