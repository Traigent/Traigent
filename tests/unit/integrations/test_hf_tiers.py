"""Tests for HuggingFace tier-based model selection and provider-aware fallback.

Covers:
- get_models_for_tier("huggingface", ...) returns HF model IDs, not OpenAI models
- list_available_providers() includes "huggingface"
- provider-aware ultimate fallback never returns an OpenAI model for non-OpenAI providers
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

import os
from unittest.mock import patch

from traigent.integrations.providers import (
    get_models_for_tier,
    list_available_providers,
)


class TestHuggingFaceTiers:
    """HuggingFace is a first-class provider with its own tier entries."""

    def _no_discovery(self):
        """Context manager: disable model discovery so fallback/tier data is exercised."""
        return patch(
            "traigent.integrations.providers.get_model_discovery", return_value=None
        )

    def test_balanced_returns_hf_models_not_openai(self) -> None:
        """get_models_for_tier(huggingface, balanced) must NOT return gpt-4o-mini."""
        with patch.dict(os.environ, {}, clear=True), self._no_discovery():
            models = get_models_for_tier(provider="huggingface", tier="balanced")

        assert len(models) >= 1, (
            "HuggingFace balanced tier must return at least one model"
        )
        assert "gpt-4o-mini" not in models, (
            "HF balanced must not fall back to gpt-4o-mini"
        )
        assert "gpt-4o" not in models, "HF balanced must not return any OpenAI model"
        assert all("/" in m for m in models), (
            "HF model IDs should be in 'org/model' format"
        )

    def test_fast_returns_hf_models(self) -> None:
        """Fast tier returns small HF models, not OpenAI models."""
        with patch.dict(os.environ, {}, clear=True), self._no_discovery():
            models = get_models_for_tier(provider="huggingface", tier="fast")

        assert len(models) >= 1
        assert "gpt-4o-mini" not in models
        assert all("/" in m for m in models)

    def test_quality_returns_hf_models(self) -> None:
        """Quality tier returns large HF models."""
        with patch.dict(os.environ, {}, clear=True), self._no_discovery():
            models = get_models_for_tier(provider="huggingface", tier="quality")

        assert len(models) >= 1
        assert "gpt-4o-mini" not in models
        # 70B llama is the flagship quality model
        assert any("70B" in m or "Mixtral" in m for m in models)

    def test_balanced_contains_expected_models(self) -> None:
        """Balanced tier includes representative HF models from models.yaml."""
        with patch.dict(os.environ, {}, clear=True), self._no_discovery():
            models = get_models_for_tier(provider="huggingface", tier="balanced")

        model_ids = set(models)
        assert model_ids & {
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }, f"Expected at least one known HF model in balanced tier, got: {models}"


class TestListAvailableProvidersIncludesHF:
    """list_available_providers() must enumerate huggingface."""

    def test_huggingface_in_provider_list(self) -> None:
        providers = list_available_providers()

        assert "huggingface" in providers

    def test_provider_list_still_has_all_original_providers(self) -> None:
        providers = list_available_providers()

        for expected in ("openai", "anthropic", "gemini", "mistral", "huggingface"):
            assert expected in providers, f"Missing provider: {expected}"


class TestProviderAwareFallback:
    """The ultimate fallback must NOT return an OpenAI model for non-OpenAI providers."""

    def _no_discovery(self):
        return patch(
            "traigent.integrations.providers.get_model_discovery", return_value=None
        )

    def test_truly_unknown_provider_does_not_return_openai_model(self) -> None:
        """An unknown provider should get [] rather than gpt-4o-mini."""
        with patch.dict(os.environ, {}, clear=True), self._no_discovery():
            models = get_models_for_tier(
                provider="unknown_xyz_provider", tier="balanced"
            )

        assert "gpt-4o-mini" not in models, (
            "ultimate fallback returned gpt-4o-mini for an unrelated provider"
        )
        assert "gpt-4o" not in models, (
            "ultimate fallback returned an OpenAI model for an unknown non-OpenAI provider"
        )

    def test_none_provider_may_still_use_openai_default(self) -> None:
        """provider=None (truly unspecified) is allowed to return OpenAI defaults."""
        with patch.dict(os.environ, {}, clear=True), self._no_discovery():
            models = get_models_for_tier(provider=None, tier="balanced")

        assert len(models) >= 1, "None provider should still return something"

    def test_openai_provider_still_returns_openai_models(self) -> None:
        """Regression: openai provider must still get its defaults."""
        with patch.dict(os.environ, {}, clear=True), self._no_discovery():
            models = get_models_for_tier(provider="openai", tier="balanced")

        assert any("gpt" in m for m in models), (
            "openai provider must still receive gpt models via fallback"
        )

    def test_unknown_provider_returns_empty_not_exception(self) -> None:
        """The provider-aware fallback should return [] gracefully, not raise."""
        with patch.dict(os.environ, {}, clear=True), self._no_discovery():
            models = get_models_for_tier(provider="some_future_provider", tier="fast")

        assert isinstance(models, list)
