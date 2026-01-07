"""LLM parameter presets.

Provides pre-configured parameter ranges for LLM optimization including
temperature, top_p, max_tokens, and model selection.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from traigent.api.parameter_ranges import Choices, IntRange, Range

logger = logging.getLogger(__name__)


class LLMPresets:
    """Pre-configured parameter ranges for LLM optimization.

    These presets encode domain knowledge about sensible parameter ranges
    for different LLM optimization scenarios.
    """

    @staticmethod
    def temperature(
        *,
        conservative: bool = False,
        creative: bool = False,
    ) -> Range:
        """Pre-configured temperature range for LLM optimization.

        Args:
            conservative: Use narrow range [0.0, 0.5] for factual tasks
            creative: Use higher range [0.7, 1.5] for creative tasks

        Returns:
            Range instance configured for temperature optimization
        """
        from traigent.api.parameter_ranges import Range

        if conservative:
            return Range(0.0, 0.5, default=0.2, name="temperature")
        elif creative:
            return Range(0.7, 1.5, default=1.0, name="temperature")
        return Range(0.0, 1.0, default=0.7, name="temperature")

    @staticmethod
    def top_p() -> Range:
        """Pre-configured top_p (nucleus sampling) parameter.

        Returns:
            Range instance configured for top_p optimization
        """
        from traigent.api.parameter_ranges import Range

        return Range(0.1, 1.0, default=0.9, name="top_p")

    @staticmethod
    def frequency_penalty() -> Range:
        """Pre-configured frequency penalty parameter.

        Returns:
            Range instance configured for frequency_penalty optimization
        """
        from traigent.api.parameter_ranges import Range

        return Range(0.0, 2.0, default=0.0, name="frequency_penalty")

    @staticmethod
    def presence_penalty() -> Range:
        """Pre-configured presence penalty parameter.

        Returns:
            Range instance configured for presence_penalty optimization
        """
        from traigent.api.parameter_ranges import Range

        return Range(0.0, 2.0, default=0.0, name="presence_penalty")

    @staticmethod
    def max_tokens(
        *,
        task: Literal["short", "medium", "long"] = "medium",
    ) -> IntRange:
        """Pre-configured max_tokens by task type.

        Args:
            task: Task length category affecting token range

        Returns:
            IntRange instance configured for max_tokens optimization
        """
        from traigent.api.parameter_ranges import IntRange

        ranges = {
            "short": (50, 256),
            "medium": (256, 1024),
            "long": (1024, 4096),
        }
        low, high = ranges[task]
        return IntRange(low, high, step=64, name="max_tokens")

    @staticmethod
    def model(
        *,
        provider: str | None = None,
        tier: Literal["fast", "balanced", "quality"] = "balanced",
    ) -> Choices:
        """Pre-configured model selection.

        Uses environment variables or fallback defaults for model lists.
        Models are NOT hard-coded - they come from:
        1. TRAIGENT_MODELS_{PROVIDER}_{TIER} env var
        2. Fallback defaults (warning logged)

        Args:
            provider: LLM provider (openai, anthropic, etc.)
            tier: Performance tier (fast, balanced, quality)

        Returns:
            Choices instance configured for model selection
        """
        from traigent.api.parameter_ranges import Choices

        models = _get_models_for_tier(provider=provider, tier=tier)
        return Choices(models, name="model")


def _get_models_for_tier(
    *,
    provider: str | None = None,
    tier: Literal["fast", "balanced", "quality"] = "balanced",
) -> list[str]:
    """Get current model list for a provider/tier combination.

    Uses environment config or falls back to sensible defaults.

    Args:
        provider: LLM provider name
        tier: Performance tier

    Returns:
        List of model names for the specified provider/tier
    """
    # Check environment variable first
    env_key = f"TRAIGENT_MODELS_{(provider or 'DEFAULT').upper()}_{tier.upper()}"
    if env_models := os.environ.get(env_key):
        return [m.strip() for m in env_models.split(",")]

    # Fallback with warning
    logger.debug(
        f"Using fallback model list for {provider}/{tier}. "
        f"Set {env_key} for explicit control."
    )
    return _get_fallback_models(provider, tier)


def _get_fallback_models(provider: str | None, tier: str) -> list[str]:
    """Fallback model lists - kept minimal and updated rarely.

    Args:
        provider: LLM provider name
        tier: Performance tier

    Returns:
        List of fallback model names
    """
    fallbacks = {
        ("openai", "fast"): ["gpt-4o-mini"],
        ("openai", "balanced"): ["gpt-4o-mini", "gpt-4o"],
        ("openai", "quality"): ["gpt-4o"],
        ("anthropic", "fast"): ["claude-3-haiku-20240307"],
        ("anthropic", "balanced"): ["claude-3-5-sonnet-20241022"],
        ("anthropic", "quality"): ["claude-3-opus-20240229"],
        (None, "fast"): ["gpt-4o-mini"],
        (None, "balanced"): ["gpt-4o-mini", "gpt-4o"],
        (None, "quality"): ["gpt-4o", "claude-3-5-sonnet-20241022"],
    }
    return fallbacks.get((provider, tier), ["gpt-4o-mini"])
