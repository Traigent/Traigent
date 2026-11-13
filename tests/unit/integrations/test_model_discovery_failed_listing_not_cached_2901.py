"""A failed live model listing must not be cached as a live result (#2901)."""

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.integrations.model_discovery.azure_discovery import AzureOpenAIDiscovery
from traigent.integrations.model_discovery.cache import ModelCache
from traigent.integrations.model_discovery.openai_discovery import OpenAIDiscovery

LIVE = ["live-model-a", "live-model-b"]


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key-not-real")


def _cache(tmp_path: Path) -> ModelCache:
    return ModelCache(cache_dir=tmp_path, default_ttl=3600)


def test_failed_listing_does_not_poison_later_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovery = OpenAIDiscovery(cache=_cache(tmp_path))
    calls = {"n": 0}

    def flaky() -> list[str]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("transient")
        return list(LIVE)

    monkeypatch.setattr(discovery, "_fetch_models_from_sdk", flaky)
    fallback = discovery.list_models()
    assert fallback  # bundled snapshot is still returned for this call
    assert set(LIVE).isdisjoint(fallback)
    # No force_refresh: the next call must try the live listing again.
    assert discovery.list_models() == LIVE


def test_failed_listing_logs_warning_without_error_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    discovery = OpenAIDiscovery(cache=_cache(tmp_path))

    def boom() -> list[str]:
        raise ConnectionError("secret-detail-xyz")

    monkeypatch.setattr(discovery, "_fetch_models_from_sdk", boom)
    with caplog.at_level("WARNING"):
        discovery.list_models()
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("ConnectionError" in m for m in warnings)
    assert not any("secret-detail-xyz" in m for m in warnings)


def test_azure_failed_listing_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    discovery = AzureOpenAIDiscovery(cache=_cache(tmp_path))
    state = {"fail": True}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"id": "my-deployment"}]}

    def fake_get(*_a: object, **_k: object) -> _Resp:
        if state["fail"]:
            raise httpx.ConnectError("down")
        return _Resp()

    monkeypatch.setattr(httpx, "get", fake_get)
    first = discovery.list_models()
    assert "my-deployment" not in first
    state["fail"] = False
    assert "my-deployment" in discovery.list_models()


def test_azure_missing_httpx_is_a_failure_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    discovery = AzureOpenAIDiscovery(cache=_cache(tmp_path))
    real_import = builtins.__import__

    def no_httpx(name: str, *args: object, **kwargs: object):  # noqa: ANN202
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_httpx)
    first = discovery.list_models()
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert "my-deployment" not in first

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": [{"id": "my-deployment"}]}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _Resp())
    assert "my-deployment" in discovery.list_models()
