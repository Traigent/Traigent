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
