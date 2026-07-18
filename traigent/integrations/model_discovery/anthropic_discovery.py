"""Anthropic model discovery implementation.

This uses a shipped model snapshot with pattern-based fallback so newly
released Claude IDs are not rejected before the snapshot is refreshed.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import logging

from traigent.integrations.model_discovery.base import ModelDiscovery
from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)

# Pattern that validates Anthropic model names.
# Matches legacy numeric IDs plus current Claude 4 family-first IDs such as
# claude-sonnet-4-6 and claude-opus-4-8.
ANTHROPIC_MODEL_PATTERN = (
    r"^claude-(?:[0-9](?:[-.][a-z0-9]+)*|instant-[0-9](?:\.[0-9]+)?|"
    r"(?:opus|sonnet|haiku)-4(?:-[a-z0-9]+)+)$"
)

# Known Anthropic models (shipped snapshot: 2026-07-05).
# Keep this in sync with traigent/config/models.yaml.
KNOWN_ANTHROPIC_MODELS = [
    # Claude 4 family
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5-20251001",
    # Claude 3.7/3.5 family
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
    # Claude 3 family. claude-3-opus (+ -latest alias) is on the retirement
    # track (#1936/#1937) and swept from this served snapshot; see
    # traigent.config.retired_models.
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

    This implementation relies on:
    1. Config file (user can update with new models)
    2. Hardcoded shipped snapshot
    3. Pattern-based validation for the current Claude 4 ID shape
    """

    PROVIDER = "anthropic"
    FRAMEWORK = Framework.ANTHROPIC

    def _fetch_models_from_sdk(self) -> list[str]:
        """Live Anthropic model listing is not wired into this discovery path.

        Returns:
            Empty list (fall back to config and shipped snapshot).
        """
        logger.debug("Anthropic live model listing is not wired; using snapshot")
        return []

    def list_models(self, force_refresh: bool = False) -> list[str]:
        """List available Anthropic models.

        Since live SDK discovery is not wired here, this combines:
        1. Config file models
        2. Hardcoded shipped snapshot

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

        Pattern matches Claude numeric and current Claude 4 family-first model IDs.
        This is permissive to allow new Claude releases without SDK updates.
        """
        # Try config first
        config_pattern = self._get_pattern_from_config()
        if config_pattern:
            return config_pattern

        return ANTHROPIC_MODEL_PATTERN

    def _get_cache_ttl(self) -> int:
        """Anthropic models are fairly stable, use longer TTL."""
        return 604800  # 7 days
