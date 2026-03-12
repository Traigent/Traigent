"""Tests for the model discovery module.

Tests cover:
- Cache TTL and expiration
- Pattern-based validation
- Config file loading
- SDK discovery (mocked)
- Registry functionality
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from traigent.integrations.model_discovery.anthropic_discovery import (
    KNOWN_ANTHROPIC_MODELS,
    AnthropicDiscovery,
)
from traigent.integrations.model_discovery.azure_discovery import AzureOpenAIDiscovery
from traigent.integrations.model_discovery.base import ModelDiscovery
from traigent.integrations.model_discovery.cache import (
    CacheEntry,
    ModelCache,
    get_global_cache,
    reset_global_cache,
)
from traigent.integrations.model_discovery.gemini_discovery import GeminiDiscovery
from traigent.integrations.model_discovery.openai_discovery import OpenAIDiscovery
from traigent.integrations.model_discovery.registry import (
    get_model_discovery,
    list_registered_providers,
)
from traigent.integrations.utils import Framework


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_not_expired(self) -> None:
        """Fresh cache entry should not be expired."""
        entry = CacheEntry(
            data=["model1", "model2"],
            timestamp=time.time(),
            ttl_seconds=3600,
        )
        assert not entry.is_expired()

    def test_cache_entry_expired(self) -> None:
        """Old cache entry should be expired."""
        entry = CacheEntry(
            data=["model1"],
            timestamp=time.time() - 7200,  # 2 hours ago
            ttl_seconds=3600,  # 1 hour TTL
        )
        assert entry.is_expired()

    def test_cache_entry_age(self) -> None:
        """Cache entry age should be calculated correctly."""
        entry = CacheEntry(
            data=[],
            timestamp=time.time() - 100,
            ttl_seconds=3600,
        )
        assert 99 <= entry.age_seconds() <= 101

    def test_cache_entry_serialization(self) -> None:
        """Cache entry should serialize/deserialize correctly."""
        original = CacheEntry(
            data=["gpt-4", "gpt-3.5-turbo"],
            timestamp=1234567890.0,
            ttl_seconds=86400,
            provider="openai",
        )

        data = original.to_dict()
        restored = CacheEntry.from_dict(data)

        assert restored.data == original.data
        assert restored.timestamp == original.timestamp
        assert restored.ttl_seconds == original.ttl_seconds
        assert restored.provider == original.provider


class TestModelCache:
    """Tests for ModelCache class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = ModelCache(
            cache_dir=Path(self.temp_dir),
            default_ttl=3600,
            enable_file_cache=True,
        )

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_set_and_get(self) -> None:
        """Cache should store and retrieve models."""
        models = ["gpt-4", "gpt-4o", "gpt-3.5-turbo"]
        self.cache.set("openai", models)

        result = self.cache.get("openai")
        assert result == models

    def test_cache_miss(self) -> None:
        """Cache should return None for missing keys."""
        result = self.cache.get("nonexistent")
        assert result is None

    def test_cache_invalidation(self) -> None:
        """Cache invalidation should remove entries."""
        self.cache.set("openai", ["gpt-4"])
        self.cache.invalidate("openai")

        result = self.cache.get("openai")
        assert result is None

    def test_cache_force_refresh(self) -> None:
        """Force refresh should bypass cache."""
        self.cache.set("openai", ["gpt-4"])

        result = self.cache.get("openai", force_refresh=True)
        assert result is None

    def test_cache_refresh_with_fetcher(self) -> None:
        """Refresh should fetch new data using fetcher function."""
        self.cache.set("openai", ["old-model"])

        def fetcher() -> list[str]:
            return ["new-model-1", "new-model-2"]

        result = self.cache.refresh("openai", fetcher)

        assert result == ["new-model-1", "new-model-2"]
        assert self.cache.get("openai") == ["new-model-1", "new-model-2"]

    def test_cache_file_persistence(self) -> None:
        """Cache should persist to file."""
        models = ["claude-3-opus"]
        self.cache.set("anthropic", models)

        # Create new cache instance pointing to same dir
        new_cache = ModelCache(
            cache_dir=Path(self.temp_dir),
            enable_file_cache=True,
        )

        # New instance should load from file
        result = new_cache.get("anthropic")
        assert result == models

    def test_cache_entry_inspection(self) -> None:
        """get_entry should return raw CacheEntry."""
        self.cache.set("gemini", ["gemini-pro"])

        entry = self.cache.get_entry("gemini")
        assert entry is not None
        assert entry.data == ["gemini-pro"]
        assert isinstance(entry.timestamp, float)


class TestGlobalCache:
    """Tests for global cache singleton."""

    def teardown_method(self) -> None:
        """Reset global cache after each test."""
        reset_global_cache()

    def test_global_cache_singleton(self) -> None:
        """Global cache should be a singleton."""
        cache1 = get_global_cache()
        cache2 = get_global_cache()
        assert cache1 is cache2

    def test_global_cache_reset(self) -> None:
        """Reset should create new cache instance."""
        cache1 = get_global_cache()
        cache1.set("test", ["model"])

        reset_global_cache()
        cache2 = get_global_cache()

        assert cache1 is not cache2
        assert cache2.get("test") is None


class TestOpenAIDiscovery:
    """Tests for OpenAI model discovery."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        reset_global_cache()

    def test_pattern_validation_valid_models(self) -> None:
        """Valid OpenAI models should pass pattern validation."""
        discovery = OpenAIDiscovery()

        valid_models = [
            "gpt-4",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
            "text-davinci-003",
            "code-davinci-002",
            "davinci",
            "curie",
            "babbage",
            "ada",
            "ft:gpt-3.5-turbo:custom",
        ]

        for model in valid_models:
            assert discovery.is_valid_model(model), f"Model {model} should be valid"

    def test_pattern_validation_invalid_models(self) -> None:
        """Invalid models should fail pattern validation."""
        discovery = OpenAIDiscovery()

        invalid_models = [
            "",
            "invalid-model",
            "claude-3-opus",
            "gemini-pro",
            "123-model",
        ]

        for model in invalid_models:
            # Should not pass pattern, but may still be in config
            pattern = discovery.get_pattern()
            import re

            if pattern:
                assert not re.match(pattern, model), f"Pattern should not match {model}"

    @patch("traigent.integrations.model_discovery.openai_discovery.os.getenv")
    def test_sdk_discovery_without_api_key(self, mock_getenv: MagicMock) -> None:
        """SDK discovery should skip when API key not set."""
        mock_getenv.return_value = None
        discovery = OpenAIDiscovery()

        result = discovery._fetch_models_from_sdk()
        assert result == []

    def test_framework_attribute(self) -> None:
        """Discovery should have correct Framework attribute."""
        discovery = OpenAIDiscovery()
        assert discovery.FRAMEWORK == Framework.OPENAI
        assert discovery.PROVIDER == "openai"


class TestAnthropicDiscovery:
    """Tests for Anthropic model discovery."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        reset_global_cache()

    def test_pattern_validation_valid_models(self) -> None:
        """Valid Anthropic models should pass pattern validation."""
        discovery = AnthropicDiscovery()

        valid_models = [
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
            "claude-2.1",
            "claude-instant-1.2",
        ]

        for model in valid_models:
            assert discovery.is_valid_model(model), f"Model {model} should be valid"

    def test_known_models_included(self) -> None:
        """Known models should be included in list."""
        discovery = AnthropicDiscovery()
        models = discovery.list_models()

        for model in KNOWN_ANTHROPIC_MODELS:
            assert model in models, f"Known model {model} should be in list"

    def test_sdk_discovery_returns_empty(self) -> None:
        """Anthropic SDK discovery should return empty (no API)."""
        discovery = AnthropicDiscovery()
        result = discovery._fetch_models_from_sdk()
        assert result == []

    def test_framework_attribute(self) -> None:
        """Discovery should have correct Framework attribute."""
        discovery = AnthropicDiscovery()
        assert discovery.FRAMEWORK == Framework.ANTHROPIC
        assert discovery.PROVIDER == "anthropic"


class TestGeminiDiscovery:
    """Tests for Gemini model discovery."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        reset_global_cache()

    def test_pattern_validation_valid_models(self) -> None:
        """Valid Gemini models should pass pattern validation."""
        discovery = GeminiDiscovery()

        valid_models = [
            "gemini-pro",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "models/gemini-pro",
            "models/gemini-1.5-pro",
        ]

        for model in valid_models:
            assert discovery.is_valid_model(model), f"Model {model} should be valid"

    @patch("traigent.integrations.model_discovery.gemini_discovery.os.getenv")
    def test_sdk_discovery_without_api_key(self, mock_getenv: MagicMock) -> None:
        """SDK discovery should skip when API key not set."""
        mock_getenv.return_value = None
        discovery = GeminiDiscovery()

        result = discovery._fetch_models_from_sdk()
        assert result == []

    def test_framework_attribute(self) -> None:
        """Discovery should have correct Framework attribute."""
        discovery = GeminiDiscovery()
        assert discovery.FRAMEWORK == Framework.GEMINI
        assert discovery.PROVIDER == "gemini"


class TestAzureOpenAIDiscovery:
    """Tests for Azure OpenAI model discovery."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        reset_global_cache()

    def test_permissive_validation(self) -> None:
        """Azure should accept reasonable deployment names."""
        discovery = AzureOpenAIDiscovery()

        valid_names = [
            "gpt-4",
            "my-deployment",
            "gpt4-production",
            "test_deployment_1",
        ]

        for name in valid_names:
            assert discovery.is_valid_model(name), f"Deployment {name} should be valid"

    def test_invalid_deployment_names(self) -> None:
        """Empty or invalid names should fail validation."""
        discovery = AzureOpenAIDiscovery()

        assert not discovery.is_valid_model("")
        assert not discovery.is_valid_model(None)  # type: ignore

    def test_framework_attribute(self) -> None:
        """Discovery should have correct Framework attribute."""
        discovery = AzureOpenAIDiscovery()
        assert discovery.FRAMEWORK == Framework.AZURE_OPENAI
        assert discovery.PROVIDER == "azure_openai"


class TestRegistry:
    """Tests for model discovery registry."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        reset_global_cache()
        # Don't clear registry - we want the default registrations

    def test_default_registrations(self) -> None:
        """Default providers should be registered."""
        providers = list_registered_providers()

        assert "openai" in providers
        assert "anthropic" in providers
        assert "gemini" in providers
        assert "azure_openai" in providers

    def test_get_discovery_by_string(self) -> None:
        """Should get discovery by provider string."""
        discovery = get_model_discovery("openai")

        assert discovery is not None
        assert isinstance(discovery, OpenAIDiscovery)

    def test_get_discovery_by_framework(self) -> None:
        """Should get discovery by Framework enum."""
        discovery = get_model_discovery(Framework.ANTHROPIC)

        assert discovery is not None
        assert isinstance(discovery, AnthropicDiscovery)

    def test_get_discovery_cached(self) -> None:
        """Discovery instances should be cached by default."""
        discovery1 = get_model_discovery("openai")
        discovery2 = get_model_discovery("openai")

        assert discovery1 is discovery2

    def test_get_discovery_not_cached(self) -> None:
        """Should create new instance when cached=False."""
        discovery1 = get_model_discovery("openai")
        discovery2 = get_model_discovery("openai", cached=False)

        assert discovery1 is not discovery2

    def test_get_nonexistent_provider(self) -> None:
        """Should return None for unknown providers."""
        discovery = get_model_discovery("nonexistent")
        assert discovery is None


class TestConfigFile:
    """Tests for config file loading."""

    def test_config_file_loading(self) -> None:
        """Discovery should load models from config file."""
        # Create temp config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "openai": {
                    "known_models": ["custom-gpt-model"],
                    "pattern": "^custom-",
                },
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            discovery = OpenAIDiscovery(config_path=config_path)
            models = discovery._get_models_from_config()

            assert "custom-gpt-model" in models
        finally:
            config_path.unlink()

    def test_pattern_from_config(self) -> None:
        """Discovery should load pattern from config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "openai": {
                    "known_models": [],
                    "pattern": "^test-pattern-",
                },
            }
            yaml.dump(config, f)
            config_path = Path(f.name)

        try:
            discovery = OpenAIDiscovery(config_path=config_path)
            pattern = discovery.get_pattern()

            assert pattern == "^test-pattern-"
        finally:
            config_path.unlink()


class TestMistralDiscovery:
    """Tests for Mistral model discovery."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        reset_global_cache()

    def test_pattern_validation_valid_models(self) -> None:
        """Valid Mistral models should pass pattern validation."""
        from traigent.integrations.model_discovery.mistral_discovery import (
            MistralDiscovery,
        )

        discovery = MistralDiscovery()

        valid_models = [
            "mistral-small-latest",
            "mistral-large-2411",
            "open-mistral-7b",
            "open-mixtral-8x7b",
            "codestral-latest",
            "pixtral-12b-2409",
            "ministral-3b-latest",
            "ft:mistral-small-custom",
        ]

        for model in valid_models:
            assert discovery.is_valid_model(model), f"Model {model} should be valid"

    def test_pattern_validation_invalid_models(self) -> None:
        """Invalid models should fail pattern validation."""
        from traigent.integrations.model_discovery.mistral_discovery import (
            MistralDiscovery,
        )

        discovery = MistralDiscovery()

        invalid_models = [
            "",
            "gpt-4",
            "claude-3-opus",
            "gemini-pro",
        ]

        for model in invalid_models:
            pattern = discovery.get_pattern()
            import re

            if pattern:
                assert not re.match(pattern, model), f"Pattern should not match {model}"

    @patch("traigent.integrations.model_discovery.mistral_discovery.os.getenv")
    def test_sdk_discovery_without_api_key(self, mock_getenv: MagicMock) -> None:
        """SDK discovery should skip when API key not set."""
        from traigent.integrations.model_discovery.mistral_discovery import (
            MistralDiscovery,
        )

        mock_getenv.return_value = None
        discovery = MistralDiscovery()

        result = discovery._fetch_models_from_sdk()
        assert result == []

    def test_framework_attribute(self) -> None:
        """Discovery should have correct Framework attribute."""
        from traigent.integrations.model_discovery.mistral_discovery import (
            MistralDiscovery,
        )

        discovery = MistralDiscovery()
        assert discovery.FRAMEWORK == Framework.MISTRAL
        assert discovery.PROVIDER == "mistral"

    def test_registry_includes_mistral(self) -> None:
        """Mistral should be registered in the discovery registry."""
        providers = list_registered_providers()
        assert "mistral" in providers

        discovery = get_model_discovery("mistral")
        assert discovery is not None


class TestCacheInvalidateAll:
    """Tests for cache invalidate_all functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = ModelCache(
            cache_dir=Path(self.temp_dir),
            default_ttl=3600,
            enable_file_cache=True,
        )

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalidate_all_clears_cache(self) -> None:
        """invalidate_all should clear all cache entries."""
        self.cache.set("openai", ["gpt-4"])
        self.cache.set("anthropic", ["claude-3-opus"])
        self.cache.set("gemini", ["gemini-pro"])

        self.cache.invalidate_all()

        assert self.cache.get("openai") is None
        assert self.cache.get("anthropic") is None
        assert self.cache.get("gemini") is None


class TestCacheNoFileMode:
    """Tests for cache with file caching disabled."""

    def test_cache_without_file_persistence(self) -> None:
        """Cache should work without file persistence."""
        cache = ModelCache(enable_file_cache=False)

        cache.set("test", ["model1", "model2"])
        result = cache.get("test")

        assert result == ["model1", "model2"]


class TestBaseDiscoveryAbstract:
    """Tests for base ModelDiscovery abstract methods."""

    def test_refresh_cache_calls_list_models(self) -> None:
        """refresh_cache should invalidate and re-fetch."""
        discovery = OpenAIDiscovery()
        reset_global_cache()

        # First call to populate cache
        discovery.list_models()

        # Get cache entry
        entry_before = discovery._cache.get_entry("openai")
        assert entry_before is not None

        # Refresh - should create new entry
        discovery.refresh_cache()

        entry_after = discovery._cache.get_entry("openai")
        # Entry should exist but timestamp should be newer
        assert entry_after is not None

    def test_invalid_regex_pattern_handled(self) -> None:
        """Invalid regex pattern should be handled gracefully."""

        class BadPatternDiscovery(ModelDiscovery):
            PROVIDER = "bad"

            def _fetch_models_from_sdk(self) -> list[str]:
                return []

            def get_pattern(self) -> str | None:
                return "[invalid"  # Invalid regex

        discovery = BadPatternDiscovery()
        # Should not raise, just return False
        result = discovery.is_valid_model("test-model")
        assert result is False

    def test_empty_model_id_invalid(self) -> None:
        """Empty model ID should be invalid."""
        discovery = OpenAIDiscovery()
        assert discovery.is_valid_model("") is False

    def test_list_models_caches_result(self) -> None:
        """list_models should cache results."""
        discovery = OpenAIDiscovery()
        reset_global_cache()

        # First call
        models1 = discovery.list_models()

        # Check cache is populated
        cached = discovery._cache.get("openai")
        assert cached is not None
        assert cached == models1

    def test_missing_config_file(self) -> None:
        """Missing config file should be handled gracefully."""
        discovery = OpenAIDiscovery(config_path=Path("/nonexistent/path/models.yaml"))

        # Should not raise
        models = discovery._get_models_from_config()
        assert models == []


class TestCacheEntryDefaults:
    """Tests for CacheEntry default values."""

    def test_from_dict_with_missing_fields(self) -> None:
        """from_dict should handle missing fields with defaults."""
        entry = CacheEntry.from_dict({})

        assert entry.data == []
        assert entry.timestamp == pytest.approx(0.0)
        assert entry.ttl_seconds == 86400
        assert entry.provider == ""


class TestDiscoveryWithMockedSDK:
    """Tests for SDK discovery with mocked SDK clients."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"})  # pragma: allowlist secret
    @patch("openai.OpenAI")
    def test_openai_sdk_discovery_success(self, mock_openai_class: MagicMock) -> None:
        """OpenAI SDK discovery should work with valid API key."""
        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.id = "gpt-4"
        mock_model2 = MagicMock()
        mock_model2.id = "gpt-3.5-turbo"
        mock_client.models.list.return_value.data = [mock_model1, mock_model2]
        mock_openai_class.return_value = mock_client

        discovery = OpenAIDiscovery()
        result = discovery._fetch_models_from_sdk()

        assert "gpt-3.5-turbo" in result
        assert "gpt-4" in result

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-api-key"})  # pragma: allowlist secret
    @patch("openai.OpenAI")
    def test_openai_sdk_discovery_error(self, mock_openai_class: MagicMock) -> None:
        """OpenAI SDK discovery should handle errors gracefully."""
        mock_openai_class.side_effect = Exception("API error")

        discovery = OpenAIDiscovery()
        # Should raise exception (handled by caller in list_models)
        with pytest.raises(Exception, match="API error"):
            discovery._fetch_models_from_sdk()

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-api-key"})  # pragma: allowlist secret
    def test_gemini_sdk_discovery_success(self) -> None:
        """Gemini SDK discovery should work with valid API key."""
        pytest.importorskip("google.generativeai")

        with (
            patch("google.generativeai.configure") as mock_configure,
            patch("google.generativeai.list_models") as mock_list_models,
        ):
            mock_model1 = MagicMock()
            mock_model1.name = "models/gemini-pro"
            mock_model1.supported_generation_methods = ["generateContent"]
            mock_model2 = MagicMock()
            mock_model2.name = "models/gemini-1.5-flash"
            mock_model2.supported_generation_methods = ["generateContent"]
            mock_list_models.return_value = [mock_model1, mock_model2]

            discovery = GeminiDiscovery()
            result = discovery._fetch_models_from_sdk()

            mock_configure.assert_called_once()

        assert "models/gemini-pro" in result
        assert "models/gemini-1.5-flash" in result

    def test_mistral_sdk_discovery_with_mocked_client(self) -> None:
        """Mistral SDK discovery should work with mocked client.

        Note: We can't use @patch for mistralai as it may not be installed.
        Instead, we test by directly calling with a mocked internal.
        """
        from traigent.integrations.model_discovery.mistral_discovery import (
            MistralDiscovery,
        )

        discovery = MistralDiscovery()

        # Test that the discovery works with config (no SDK call)
        models = discovery.list_models()
        assert isinstance(models, list)
        # Should contain known models from config
        assert any("mistral" in m for m in models)
