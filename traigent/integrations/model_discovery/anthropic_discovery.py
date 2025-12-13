"""Anthropic model discovery implementation.

Anthropic SDK does not expose a models.list() API, so this uses
hardcoded known models with pattern-based fallback.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import logging

from traigent.integrations.model_discovery.base import ModelDiscovery
from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)

# Pattern that validates Anthropic model names
# Matches: claude-3-opus-*, claude-3-5-sonnet-*, claude-2.1, etc.
ANTHROPIC_MODEL_PATTERN = r"^claude-[0-9]"

# Known Anthropic models (updated as of Dec 2024)
# Since Anthropic doesn't have a models.list() API, we maintain this list
KNOWN_ANTHROPIC_MODELS = [
    # Claude 3.5 family
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
    # Claude 3 family
    "claude-3-opus-20240229",
    "claude-3-opus-latest",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    # Claude 2 family (legacy)
    "claude-2.1",
    "claude-2.0",
    # Claude Instant (legacy)
    "claude-instant-1.2",
    "claude-instant-1.1",
]


class AnthropicDiscovery(ModelDiscovery):
    """Model discovery for Anthropic.

    Since Anthropic doesn't provide a models.list() SDK method,
    this implementation relies on:
    1. Config file (user can update with new models)
    2. Hardcoded known models
    3. Pattern-based validation (accepts any claude-* model)
    """

    PROVIDER = "anthropic"
    FRAMEWORK = Framework.ANTHROPIC

    def _fetch_models_from_sdk(self) -> list[str]:
        """Anthropic SDK doesn't support listing models.

        Returns:
            Empty list (SDK discovery not available).
        """
        # Anthropic doesn't have a public models.list() API
        # Return empty to fall back to config/hardcoded list
        logger.debug("Anthropic SDK doesn't support model listing")
        return []

    def list_models(self, force_refresh: bool = False) -> list[str]:
        """List available Anthropic models.

        Since SDK discovery isn't available, this combines:
        1. Config file models
        2. Hardcoded known models

        Args:
            force_refresh: Ignored for Anthropic (no SDK to refresh from).

        Returns:
            List of known model names.
        """
        # Try cache first
        if not force_refresh:
            cached = self._cache.get(self.PROVIDER)
            if cached is not None:
                return cached

        # Combine config models and hardcoded models
        config_models = self._get_models_from_config()
        all_models = set(KNOWN_ANTHROPIC_MODELS)
        all_models.update(config_models)

        models = sorted(all_models)
        self._cache.set(self.PROVIDER, models, 604800)  # 7 days (static list)
        return models

    def get_pattern(self) -> str | None:
        """Get pattern for Anthropic models.

        Pattern matches: claude-[0-9]*
        This is permissive to allow new model versions without SDK updates.
        """
        # Try config first
        config_pattern = self._get_pattern_from_config()
        if config_pattern:
            return config_pattern

        return ANTHROPIC_MODEL_PATTERN

    def _get_cache_ttl(self) -> int:
        """Anthropic models are fairly stable, use longer TTL."""
        return 604800  # 7 days
