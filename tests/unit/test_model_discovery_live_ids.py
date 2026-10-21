"""Live model-id listing for providers (mocked; no network) (#1813)."""

from __future__ import annotations

import sys
import types

import pytest

from traigent.integrations.model_discovery.anthropic_discovery import AnthropicDiscovery
from traigent.integrations.model_discovery.azure_discovery import AzureOpenAIDiscovery
from traigent.integrations.model_discovery.cache import ModelCache
from traigent.integrations.model_discovery.openai_discovery import OpenAIDiscovery


def _cache() -> ModelCache:
    return ModelCache(enable_file_cache=False)


def test_openai_live_ids_are_unfiltered(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    class Client:
        def __init__(self, api_key: str) -> None:
            self.models = types.SimpleNamespace(
                list=lambda: types.SimpleNamespace(
                    data=[types.SimpleNamespace(id=i) for i in ("o3-mini", "gpt-4o")]
                )
            )

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=Client))
    assert OpenAIDiscovery(cache=_cache()).live_model_ids() == ["gpt-4o", "o3-mini"]


def test_live_ids_none_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert OpenAIDiscovery(cache=_cache()).live_model_ids() is None


def test_live_ids_none_on_provider_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    def boom(api_key: str):
        raise RuntimeError("network down")

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=boom))
    assert OpenAIDiscovery(cache=_cache()).live_model_ids() is None


def test_anthropic_live_ids_via_sdk(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    class Client:
        def __init__(self, api_key: str) -> None:
            self.models = types.SimpleNamespace(
                list=lambda limit=1000: [
                    types.SimpleNamespace(id="claude-sonnet-4-6"),
                    types.SimpleNamespace(id="claude-opus-4-8"),
                ]
            )

    monkeypatch.setitem(
        sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Client)
    )
    assert AnthropicDiscovery(cache=_cache()).live_model_ids() == [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
    ]


def test_anthropic_live_ids_none_without_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert AnthropicDiscovery(cache=_cache()).live_model_ids() is None


def test_azure_has_no_live_catalog(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.invalid")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    assert AzureOpenAIDiscovery(cache=_cache()).live_model_ids() is None


@pytest.mark.parametrize("empty", [[], None])
def test_base_live_ids_empty_is_none(empty, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    d = OpenAIDiscovery(cache=_cache())
    monkeypatch.setattr(d, "_fetch_live_model_ids", lambda: empty or [])
    assert d.live_model_ids() is None
