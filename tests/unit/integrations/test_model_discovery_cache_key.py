"""Regression tests for credential-scoped model-discovery cache keys (#1971).

The model-discovery response cache used to be keyed solely on the provider
name, so two runs with different API keys / endpoints (which return different
account-specific model sets) collided on one shared cache entry and file. These
tests pin that the cache key now incorporates a non-sensitive credential/
endpoint fingerprint, and that the raw secret never appears in the key.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import tempfile
from pathlib import Path
from unittest.mock import patch

from traigent.integrations.model_discovery.azure_discovery import AzureOpenAIDiscovery
from traigent.integrations.model_discovery.cache import ModelCache
from traigent.integrations.model_discovery.openai_discovery import OpenAIDiscovery


def _isolated_cache() -> ModelCache:
    return ModelCache(cache_dir=Path(tempfile.mkdtemp()))


def test_openai_cache_key_distinguishes_credentials():
    disc = OpenAIDiscovery(cache=_isolated_cache())

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-org-ONE"}, clear=False):
        key_one = disc._get_cache_key()
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-org-TWO"}, clear=False):
        key_two = disc._get_cache_key()

    assert key_one != key_two, "different API keys must not share a cache entry"
    assert key_one.startswith("openai-")
    # The raw secret must never leak into the key (it becomes the cache filename).
    assert "sk-org-ONE" not in key_one
    assert "sk-org-TWO" not in key_two


def test_openai_cache_key_distinguishes_base_url():
    disc = OpenAIDiscovery(cache=_isolated_cache())

    with patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "sk-same", "OPENAI_BASE_URL": "https://a.example"},
        clear=False,
    ):
        key_a = disc._get_cache_key()
    with patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "sk-same", "OPENAI_BASE_URL": "https://b.example"},
        clear=False,
    ):
        key_b = disc._get_cache_key()

    assert key_a != key_b, "different base URLs must not share a cache entry"


def test_azure_cache_key_distinguishes_endpoint_and_key():
    disc = AzureOpenAIDiscovery(cache=_isolated_cache())

    with patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_ENDPOINT": "https://one.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "azkey1",
        },
        clear=False,
    ):
        key_one = disc._get_cache_key()
    with patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_ENDPOINT": "https://two.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "azkey2",
        },
        clear=False,
    ):
        key_two = disc._get_cache_key()

    assert key_one != key_two
    assert key_one.startswith("azure_openai-")
    assert "azkey1" not in key_one


def test_cache_does_not_bleed_between_credentials():
    """End-to-end: a cached list under key-1 is not returned for key-2."""
    cache = _isolated_cache()

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-first"}, clear=False):
        disc1 = OpenAIDiscovery(cache=cache)
        with patch.object(
            disc1, "_fetch_models_from_sdk", return_value=["gpt-first-only"]
        ):
            assert disc1.list_models() == ["gpt-first-only"]

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-second"}, clear=False):
        disc2 = OpenAIDiscovery(cache=cache)
        with patch.object(
            disc2, "_fetch_models_from_sdk", return_value=["gpt-second-only"]
        ) as fetch2:
            models = disc2.list_models()
            # Must re-fetch for the new credential rather than serve sk-first's cache.
            fetch2.assert_called_once()
            assert models == ["gpt-second-only"]


def test_no_credentials_falls_back_to_plain_provider_key():
    """Without any credential env vars the key stays the bare provider name."""
    disc = OpenAIDiscovery(cache=_isolated_cache())
    with patch.dict("os.environ", {}, clear=True):
        assert disc._get_cache_key() == "openai"
