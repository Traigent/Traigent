"""Provider-based model selection with tier support.

This module provides tier-based model selection (fast, balanced, quality)
with dynamic model discovery via environment variables, API-based discovery,
or sensible fallbacks.

Example usage:
    ```python
    from traigent.api.parameter_ranges import Choices
    from traigent.integrations.providers import get_models_for_tier

    # Get models for a specific tier
    models = get_models_for_tier(provider="openai", tier="balanced")
    # Returns: ["gpt-4o-mini", "gpt-4o"]

    # Use with Choices
    model_choices = Choices(
        get_models_for_tier(provider="anthropic", tier="quality"),
        name="model",
    )
    ```

Environment variables:
    TRAIGENT_MODELS_{PROVIDER}_{TIER}: Comma-separated list of models
    Example: TRAIGENT_MODELS_OPENAI_FAST="gpt-4o-mini,gpt-3.5-turbo"

Traceability: CONC-TunedVariable FUNC-TVLSPEC
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from traigent.integrations.model_discovery.registry import get_model_discovery

logger = logging.getLogger(__name__)

# Type alias for tier names
Tier = Literal["fast", "balanced", "quality"]

# Fallback model lists - kept minimal and updated rarely
# These are used only when env vars and API discovery both fail
_FALLBACK_MODELS: dict[tuple[str | None, str], list[str]] = {
    # OpenAI tiers
    ("openai", "fast"): ["gpt-4o-mini"],
    ("openai", "balanced"): ["gpt-4o-mini", "gpt-4o"],
    ("openai", "quality"): ["gpt-4o"],
    # Anthropic tiers
    ("anthropic", "fast"): ["claude-3-haiku-20240307"],
    ("anthropic", "balanced"): ["claude-3-5-sonnet-20241022"],
    ("anthropic", "quality"): ["claude-3-opus-20240229"],
    # Google / Gemini tiers (provider name is "google" per get_provider_for_model)
    ("google", "fast"): ["gemini-1.5-flash"],
    ("google", "balanced"): ["gemini-1.5-flash", "gemini-2.0-flash"],
    ("google", "quality"): ["gemini-1.5-pro", "gemini-2.0-flash"],
    # Groq tiers
    ("groq", "fast"): ["llama-3.1-8b-instant"],
    ("groq", "balanced"): ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    ("groq", "quality"): ["llama-3.3-70b-versatile"],
    # Mistral tiers
    ("mistral", "fast"): ["mistral-small-latest"],
    ("mistral", "balanced"): ["mistral-medium-latest"],
    ("mistral", "quality"): ["mistral-large-latest"],
    # Default (unknown provider)
    (None, "fast"): ["gpt-4o-mini"],
    (None, "balanced"): ["gpt-4o-mini", "gpt-4o"],
    (None, "quality"): ["gpt-4o"],
}

# Model tier classification based on capabilities/cost
_MODEL_TIERS: dict[str, dict[str, list[str]]] = {
    "openai": {
        "fast": ["gpt-4o-mini", "gpt-3.5-turbo"],
        "balanced": ["gpt-4o-mini", "gpt-4o"],
        "quality": ["gpt-4o", "o1-preview", "o1-mini"],
    },
    "anthropic": {
        "fast": ["claude-3-haiku-20240307"],
        "balanced": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
        "quality": ["claude-3-opus-20240229", "claude-3-5-sonnet-20241022"],
    },
    "google": {
        "fast": ["gemini-1.5-flash", "gemini-1.5-flash-8b"],
        "balanced": ["gemini-1.5-flash", "gemini-2.0-flash"],
        "quality": ["gemini-1.5-pro", "gemini-2.0-flash"],
    },
    "groq": {
        "fast": ["llama-3.1-8b-instant"],
        "balanced": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        "quality": ["llama-3.3-70b-versatile"],
    },
    "mistral": {
        "fast": ["mistral-small-latest", "open-mistral-7b"],
        "balanced": ["mistral-medium-latest", "mistral-small-latest"],
        "quality": ["mistral-large-latest"],
    },
}


def get_models_for_tier(
    *,
    provider: str | None = None,
    tier: Tier = "balanced",
) -> list[str]:
    """Get model list for a provider/tier combination.

    Resolution order:
    1. Environment variable TRAIGENT_MODELS_{PROVIDER}_{TIER}
    2. Filter from API-discovered models based on tier classification
    3. Fallback to minimal hard-coded defaults (with warning)

    Args:
        provider: Provider name (e.g., "openai", "anthropic"). If None,
            returns default models.
        tier: Model tier - "fast", "balanced", or "quality".
            - fast: Optimized for speed/cost
            - balanced: Good tradeoff between speed and quality
            - quality: Best capability, higher cost

    Returns:
        List of model identifiers suitable for the provider/tier.

    Example:
        ```python
        # Check env var first
        os.environ["TRAIGENT_MODELS_OPENAI_FAST"] = "gpt-4o-mini,gpt-3.5-turbo"
        models = get_models_for_tier(provider="openai", tier="fast")
        # Returns: ["gpt-4o-mini", "gpt-3.5-turbo"]
        ```
    """
    # 1. Check environment variable
    env_key = f"TRAIGENT_MODELS_{(provider or 'DEFAULT').upper()}_{tier.upper()}"
    env_models = os.environ.get(env_key)
    if env_models:
        models = [m.strip() for m in env_models.split(",") if m.strip()]
        if models:
            logger.debug(f"Using models from env var {env_key}: {models}")
            return models

    # 2. Try API-based discovery + tier filtering
    if provider:
        discovered = _discover_and_filter_by_tier(provider, tier)
        if discovered:
            return discovered

    # 3. Fallback with warning
    provider_key = provider.lower() if provider else None
    fallback = _FALLBACK_MODELS.get((provider_key, tier))

    if fallback:
        logger.warning(
            f"Using fallback model list for {provider}/{tier}. "
            f"Set {env_key} for explicit control."
        )
        return list(fallback)

    # Ultimate fallback
    logger.warning(f"No models found for {provider}/{tier}, using default")
    return _FALLBACK_MODELS.get((None, tier), ["gpt-4o-mini"])


def _discover_and_filter_by_tier(provider: str, tier: Tier) -> list[str]:
    """Discover models from API and filter by tier.

    Args:
        provider: Provider name.
        tier: Desired tier.

    Returns:
        Filtered list of models, or empty list if discovery failed.
    """
    discovery = get_model_discovery(provider)
    if discovery is None:
        return []

    try:
        available_models = discovery.list_models()
        if not available_models:
            return []
    except Exception as e:
        logger.debug(f"Model discovery failed for {provider}: {e}")
        return []

    # Get tier classification for this provider
    provider_tiers = _MODEL_TIERS.get(provider.lower(), {})
    tier_models = provider_tiers.get(tier, [])

    if not tier_models:
        # No tier classification, return all discovered models
        logger.debug(f"No tier classification for {provider}/{tier}")
        return list(available_models)

    # Filter available models by tier preference
    filtered = [m for m in tier_models if m in available_models]

    if filtered:
        logger.debug(f"Discovered {len(filtered)} models for {provider}/{tier}")
        return filtered

    # Tier models not available, return first few available
    return list(available_models[:3])


def list_available_providers() -> list[str]:
    """List all providers with tier-based model selection.

    Returns:
        List of provider names.
    """
    return list(_MODEL_TIERS.keys())


def get_all_tiers() -> list[str]:
    """Get all available tier names.

    Returns:
        List of tier names.
    """
    return ["fast", "balanced", "quality"]


def register_provider_tiers(
    provider: str,
    tiers: dict[str, list[str]],
) -> None:
    """Register custom tier classifications for a provider.

    This updates both the tier classifications and fallback models,
    so registered providers work correctly even without API discovery.

    Args:
        provider: Provider name.
        tiers: Dict mapping tier names to model lists.

    Example:
        ```python
        register_provider_tiers("custom_provider", {
            "fast": ["model-small"],
            "balanced": ["model-medium"],
            "quality": ["model-large"],
        })
        ```
    """
    provider_lower = provider.lower()
    _MODEL_TIERS[provider_lower] = tiers

    # Also update fallback models so registered providers work without discovery
    for tier_name, models in tiers.items():
        if models:
            _FALLBACK_MODELS[(provider_lower, tier_name)] = models

    logger.info(f"Registered tier classifications for {provider}")
