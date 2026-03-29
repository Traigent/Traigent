"""Tests for the providers module.

Tests cover:
- Tier-based model selection
- Environment variable override
- API discovery integration
- Fallback model lists
- Provider registration
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from traigent.integrations.providers import (
    _FALLBACK_MODELS,
    _MODEL_TIERS,
    get_all_tiers,
    get_models_for_tier,
    list_available_providers,
    register_provider_tiers,
)


class TestGetModelsForTier:
    """Tests for get_models_for_tier function."""

    def test_returns_list_of_strings(self) -> None:
        """Function should return a list of model strings."""
        models = get_models_for_tier(provider="openai", tier="fast")

        assert isinstance(models, list)
        assert len(models) >= 1
        assert all(isinstance(m, str) for m in models)

    def test_default_tier_is_balanced(self) -> None:
        """Default tier should be 'balanced'."""
        with patch.dict(os.environ, {}, clear=True):
            models = get_models_for_tier(provider="openai")

        # balanced tier typically has more models than fast
        assert len(models) >= 1
        # gpt-4o-mini is in balanced for openai
        assert any("gpt" in m for m in models)

    def test_env_var_override(self) -> None:
        """Environment variable should override default models."""
        env_models = "custom-model-1,custom-model-2,custom-model-3"
        with patch.dict(os.environ, {"TRAIGENT_MODELS_OPENAI_FAST": env_models}):
            models = get_models_for_tier(provider="openai", tier="fast")

        assert models == ["custom-model-1", "custom-model-2", "custom-model-3"]

    def test_env_var_with_whitespace(self) -> None:
        """Environment variable should handle whitespace in values."""
        env_models = " model-1 , model-2 , model-3 "
        with patch.dict(os.environ, {"TRAIGENT_MODELS_ANTHROPIC_QUALITY": env_models}):
            models = get_models_for_tier(provider="anthropic", tier="quality")

        assert models == ["model-1", "model-2", "model-3"]

    def test_env_var_empty_string_falls_back(self) -> None:
        """Empty env var should fall back to discovery/defaults."""
        with patch.dict(os.environ, {"TRAIGENT_MODELS_OPENAI_FAST": ""}):
            models = get_models_for_tier(provider="openai", tier="fast")

        # Should not be empty - should fall back to discovery/defaults
        assert len(models) >= 1

    def test_env_var_case_insensitive_provider(self) -> None:
        """Provider in env var key should be uppercase."""
        with patch.dict(os.environ, {"TRAIGENT_MODELS_GEMINI_BALANCED": "gemini-test"}):
            models = get_models_for_tier(provider="gemini", tier="balanced")

        assert models == ["gemini-test"]

    def test_provider_none_uses_default_key(self) -> None:
        """None provider should use DEFAULT in env var key."""
        with patch.dict(os.environ, {"TRAIGENT_MODELS_DEFAULT_FAST": "default-model"}):
            models = get_models_for_tier(provider=None, tier="fast")

        assert models == ["default-model"]

    def test_fallback_models_for_openai(self) -> None:
        """Fallback should return known OpenAI models."""
        # Clear env and mock discovery to return empty
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery"
            ) as mock_discovery,
        ):
            mock_discovery.return_value = None
            models = get_models_for_tier(provider="openai", tier="quality")

        assert "gpt-4o" in models

    def test_fallback_models_for_anthropic(self) -> None:
        """Fallback should return known Anthropic models."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery"
            ) as mock_discovery,
        ):
            mock_discovery.return_value = None
            models = get_models_for_tier(provider="anthropic", tier="quality")

        assert any("claude" in m for m in models)

    def test_fallback_models_for_google(self) -> None:
        """Fallback should return known Google/Gemini models."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery"
            ) as mock_discovery,
        ):
            mock_discovery.return_value = None
            models = get_models_for_tier(provider="google", tier="fast")

        assert any("gemini" in m for m in models)

    def test_fallback_models_for_groq(self) -> None:
        """Fallback should return known Groq models."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery"
            ) as mock_discovery,
        ):
            mock_discovery.return_value = None
            models = get_models_for_tier(provider="groq", tier="balanced")

        assert any("llama" in m for m in models)

    def test_fallback_models_for_mistral(self) -> None:
        """Fallback should return known Mistral models."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery"
            ) as mock_discovery,
        ):
            mock_discovery.return_value = None
            models = get_models_for_tier(provider="mistral", tier="quality")

        assert any("mistral" in m for m in models)

    def test_unknown_provider_uses_default_fallback(self) -> None:
        """Unknown provider should use default fallback models."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery"
            ) as mock_discovery,
        ):
            mock_discovery.return_value = None
            models = get_models_for_tier(provider="unknown_provider", tier="balanced")

        # Should return something, likely default OpenAI models
        assert len(models) >= 1

    def test_all_tiers_return_models(self) -> None:
        """All tier values should return models."""
        for tier in ["fast", "balanced", "quality"]:
            models = get_models_for_tier(provider="openai", tier=tier)  # type: ignore[arg-type]
            assert len(models) >= 1, f"Tier {tier} should have models"


class TestDiscoveryAndFilter:
    """Tests for API discovery with tier filtering."""

    def test_discovery_filters_by_tier(self) -> None:
        """Discovered models should be filtered by tier classification."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery"
            ) as mock_get_discovery,
        ):
            mock_discovery = MagicMock()
            mock_discovery.list_models.return_value = [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4-turbo",
                "text-embedding-ada-002",
            ]
            mock_get_discovery.return_value = mock_discovery

            models = get_models_for_tier(provider="openai", tier="fast")

            # fast tier for openai includes gpt-4o-mini
            assert "gpt-4o-mini" in models
            # Should not include embedding models
            assert "text-embedding-ada-002" not in models

    def test_discovery_exception_falls_back(self) -> None:
        """Discovery exception should fall back to defaults."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery"
            ) as mock_get_discovery,
        ):
            mock_discovery = MagicMock()
            mock_discovery.list_models.side_effect = Exception("API error")
            mock_get_discovery.return_value = mock_discovery

            # Should not raise, should fall back
            models = get_models_for_tier(provider="openai", tier="balanced")

            assert len(models) >= 1

    def test_discovery_empty_result_falls_back(self) -> None:
        """Empty discovery result should fall back to defaults."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery"
            ) as mock_get_discovery,
        ):
            mock_discovery = MagicMock()
            mock_discovery.list_models.return_value = []
            mock_get_discovery.return_value = mock_discovery

            models = get_models_for_tier(provider="anthropic", tier="fast")

            # Should fall back to defaults
            assert len(models) >= 1


class TestListAvailableProviders:
    """Tests for list_available_providers function."""

    def test_returns_list_of_providers(self) -> None:
        """Function should return list of provider names."""
        providers = list_available_providers()

        assert isinstance(providers, list)
        assert len(providers) >= 4  # At least openai, anthropic, gemini, mistral

    def test_includes_main_providers(self) -> None:
        """Should include main LLM providers."""
        providers = list_available_providers()

        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "groq" in providers
        assert "mistral" in providers


class TestGetAllTiers:
    """Tests for get_all_tiers function."""

    def test_returns_tier_names(self) -> None:
        """Function should return all tier names."""
        tiers = get_all_tiers()

        assert isinstance(tiers, list)
        assert "fast" in tiers
        assert "balanced" in tiers
        assert "quality" in tiers

    def test_returns_exactly_three_tiers(self) -> None:
        """Should have exactly 3 tier levels."""
        tiers = get_all_tiers()
        assert len(tiers) == 3


class TestRegisterProviderTiers:
    """Tests for register_provider_tiers function."""

    def teardown_method(self) -> None:
        """Clean up registered custom providers after each test."""
        # Remove test providers if they exist
        test_providers = [
            "test_provider",
            "custom_llm",
            "uppercase_provider",
            "fallback_test_provider",
        ]
        for provider in test_providers:
            _MODEL_TIERS.pop(provider, None)
            # Also clean up fallback models
            for tier in ["fast", "balanced", "quality"]:
                _FALLBACK_MODELS.pop((provider, tier), None)

    def test_register_new_provider(self) -> None:
        """Should register new provider tier classifications."""
        register_provider_tiers(
            "test_provider",
            {
                "fast": ["test-small"],
                "balanced": ["test-medium"],
                "quality": ["test-large"],
            },
        )

        assert "test_provider" in list_available_providers()

    def test_registered_provider_models_used(self) -> None:
        """Registered provider should be used for model lookup."""
        register_provider_tiers(
            "custom_llm",
            {
                "fast": ["custom-fast-model"],
                "balanced": ["custom-balanced-1", "custom-balanced-2"],
                "quality": ["custom-quality-model"],
            },
        )

        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery"
            ) as mock_discovery,
        ):
            # Mock discovery returns these models
            mock_disc = MagicMock()
            mock_disc.list_models.return_value = [
                "custom-fast-model",
                "custom-balanced-1",
                "custom-balanced-2",
                "custom-quality-model",
            ]
            mock_discovery.return_value = mock_disc

            models = get_models_for_tier(provider="custom_llm", tier="balanced")

            assert "custom-balanced-1" in models
            assert "custom-balanced-2" in models

    def test_register_case_normalization(self) -> None:
        """Provider names should be normalized to lowercase."""
        register_provider_tiers(
            "UPPERCASE_PROVIDER",
            {
                "fast": ["model-1"],
            },
        )

        assert "uppercase_provider" in list_available_providers()

    def test_register_updates_fallback_models(self) -> None:
        """Registered provider should update fallback models for use without discovery."""
        register_provider_tiers(
            "fallback_test_provider",
            {
                "fast": ["fb-fast-model"],
                "balanced": ["fb-balanced-model"],
                "quality": ["fb-quality-model"],
            },
        )

        # Test that fallback models are updated
        assert ("fallback_test_provider", "fast") in _FALLBACK_MODELS
        assert _FALLBACK_MODELS[("fallback_test_provider", "fast")] == ["fb-fast-model"]

        # Verify models can be retrieved (falls back when no discovery)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "traigent.integrations.providers.get_model_discovery"
            ) as mock_discovery,
        ):
            mock_discovery.return_value = None
            models = get_models_for_tier(
                provider="fallback_test_provider", tier="balanced"
            )

            assert models == ["fb-balanced-model"]


class TestFallbackModels:
    """Tests for fallback model dictionaries."""

    # Known built-in providers to test
    BUILTIN_PROVIDERS = {"openai", "anthropic", "google", "groq", "mistral"}

    def test_fallback_models_structure(self) -> None:
        """Fallback models should have correct structure."""
        # Check structure
        for key, models in _FALLBACK_MODELS.items():
            provider, tier = key
            assert provider is None or isinstance(provider, str)
            assert tier in ["fast", "balanced", "quality"]
            assert isinstance(models, list)
            assert len(models) >= 1

    def test_model_tiers_structure(self) -> None:
        """Model tiers should have correct structure."""
        for provider, tiers in _MODEL_TIERS.items():
            assert isinstance(provider, str)
            assert isinstance(tiers, dict)

            for tier, models in tiers.items():
                assert tier in ["fast", "balanced", "quality"]
                assert isinstance(models, list)
                assert len(models) >= 1

    def test_all_builtin_providers_have_all_tiers(self) -> None:
        """All built-in providers should have all three tiers defined."""
        for provider in self.BUILTIN_PROVIDERS:
            if provider not in _MODEL_TIERS:
                continue  # Skip if not registered
            tiers = _MODEL_TIERS[provider]
            assert "fast" in tiers, f"{provider} missing 'fast' tier"
            assert "balanced" in tiers, f"{provider} missing 'balanced' tier"
            assert "quality" in tiers, f"{provider} missing 'quality' tier"


class TestIntegrationWithChoices:
    """Integration tests with Choices parameter range."""

    def test_use_with_choices(self) -> None:
        """Models can be used with Choices parameter range."""
        from traigent.api.parameter_ranges import Choices

        models = get_models_for_tier(provider="openai", tier="balanced")
        choices = Choices(models, name="model")

        assert choices.name == "model"
        assert len(choices.values) >= 1
        assert all(isinstance(v, str) for v in choices.values)

    def test_all_tiers_compatible_with_choices(self) -> None:
        """All tier results should be compatible with Choices."""
        from traigent.api.parameter_ranges import Choices

        for provider in ["openai", "anthropic"]:
            for tier in ["fast", "balanced", "quality"]:
                models = get_models_for_tier(provider=provider, tier=tier)  # type: ignore[arg-type]
                # Should not raise
                choices = Choices(models, name=f"{provider}_model")
                assert len(choices.values) >= 1
