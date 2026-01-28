"""Model capability detection for reasoning support.

This module provides utilities for detecting which models support native
reasoning features (extended thinking, reasoning effort, etc.) and which
specific capabilities are available for each model.

Usage:
    from traigent.integrations.utils.model_capabilities import (
        supports_reasoning,
        get_reasoning_effort_levels,
        is_gemini_3,
    )

    # Check if model supports native reasoning
    if supports_reasoning("o3", "openai"):
        # Apply reasoning parameters
        ...

    # Get available reasoning effort levels for a model
    levels = get_reasoning_effort_levels("gpt-5")
    # Returns: ["minimal", "low", "medium", "high"]
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# =============================================================================
# Reasoning-capable model patterns by provider
# =============================================================================

REASONING_MODELS: dict[str, list[str]] = {
    "openai": [
        r"^o[1-4](-mini|-preview)?$",  # o1, o1-mini, o3, o3-mini, o4-mini, etc.
        r"^gpt-5(\.[0-9]+)?(-.*)?$",  # gpt-5, gpt-5.1, gpt-5.1-codex-max, etc.
    ],
    "anthropic": [
        r"^claude-(sonnet|opus)-[4-9]",  # claude-sonnet-4-5, claude-opus-4-5+
        r"^claude-[4-9]",  # claude-4+ (simplified names)
    ],
    "gemini": [
        r"^gemini-(2\.5|3)",  # gemini-2.5-*, gemini-3-*, gemini-3-flash, etc.
    ],
}

# =============================================================================
# Model-specific reasoning_effort availability (OpenAI only)
# =============================================================================

REASONING_EFFORT_AVAILABILITY: dict[str, list[str]] = {
    "minimal": [r"^gpt-5"],  # GPT-5+ only
    "xhigh": [r"^gpt-5\.1-codex-max$"],  # GPT-5.1-codex-max only
    "low": [r"^o[1-4]", r"^gpt-5"],  # All reasoning models
    "medium": [r"^o[1-4]", r"^gpt-5"],  # All reasoning models
    "high": [r"^o[1-4]", r"^gpt-5"],  # All reasoning models
}


def supports_reasoning(model: str, provider: str) -> bool:
    """Check if a model supports native reasoning features.

    Args:
        model: The model name/ID (e.g., "o3", "claude-sonnet-4-5", "gemini-3-pro")
        provider: The provider name (e.g., "openai", "anthropic", "gemini")

    Returns:
        True if the model supports native reasoning, False otherwise

    Example:
        >>> supports_reasoning("o3", "openai")
        True
        >>> supports_reasoning("gpt-4o", "openai")
        False
        >>> supports_reasoning("claude-sonnet-4-5", "anthropic")
        True
    """
    patterns = REASONING_MODELS.get(provider.lower(), [])
    return any(re.match(p, model, re.IGNORECASE) for p in patterns)


def get_reasoning_effort_levels(model: str) -> list[str]:
    """Get available reasoning_effort levels for an OpenAI model.

    Args:
        model: The OpenAI model name/ID

    Returns:
        List of available reasoning effort levels for this model.
        Returns ["low", "medium", "high"] as default for unknown reasoning models.

    Example:
        >>> get_reasoning_effort_levels("o3")
        ['low', 'medium', 'high']
        >>> get_reasoning_effort_levels("gpt-5")
        ['minimal', 'low', 'medium', 'high']
        >>> get_reasoning_effort_levels("gpt-5.1-codex-max")
        ['minimal', 'low', 'medium', 'high', 'xhigh']
    """
    available = []
    for level, patterns in REASONING_EFFORT_AVAILABILITY.items():
        if any(re.match(p, model, re.IGNORECASE) for p in patterns):
            available.append(level)

    if not available:
        # Default for unknown reasoning models
        return ["low", "medium", "high"]

    # Sort in logical order: minimal < low < medium < high < xhigh
    level_order = ["minimal", "low", "medium", "high", "xhigh"]
    return [lvl for lvl in level_order if lvl in available]


def is_gemini_3(model: str) -> bool:
    """Check if a model is Gemini 3 (uses thinking_level) vs Gemini 2.5 (uses thinking_budget).

    Gemini 3 models use the `thinking_level` parameter with values like
    "MINIMAL", "low", "high". Gemini 2.5 models use `thinking_budget` with
    a token count.

    Args:
        model: The Gemini model name/ID

    Returns:
        True if Gemini 3, False if Gemini 2.5 or other

    Example:
        >>> is_gemini_3("gemini-3-pro")
        True
        >>> is_gemini_3("gemini-3-flash")
        True
        >>> is_gemini_3("gemini-2.5-pro")
        False
    """
    return bool(re.match(r"^gemini-3", model, re.IGNORECASE))


def get_provider_from_model(model: str) -> str | None:
    """Attempt to detect the provider from a model name.

    This is a best-effort detection based on common naming patterns.
    Returns None if the provider cannot be determined.

    Args:
        model: The model name/ID

    Returns:
        Provider name ("openai", "anthropic", "gemini") or None

    Example:
        >>> get_provider_from_model("gpt-4o")
        'openai'
        >>> get_provider_from_model("claude-3-opus")
        'anthropic'
        >>> get_provider_from_model("gemini-2.5-pro")
        'gemini'
    """
    model_lower = model.lower()

    # OpenAI patterns
    if any(
        model_lower.startswith(prefix)
        for prefix in ["gpt-", "o1", "o2", "o3", "o4", "davinci", "text-"]
    ):
        return "openai"

    # Anthropic patterns
    if model_lower.startswith("claude"):
        return "anthropic"

    # Google patterns
    if model_lower.startswith("gemini"):
        return "gemini"

    return None


__all__ = [
    "REASONING_MODELS",
    "REASONING_EFFORT_AVAILABILITY",
    "supports_reasoning",
    "get_reasoning_effort_levels",
    "is_gemini_3",
    "get_provider_from_model",
]
