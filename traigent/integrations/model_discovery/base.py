"""Base class for model discovery implementations.

Provides the abstract interface that all provider-specific
model discovery classes must implement.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from traigent.config.retired_models import is_retired_model
from traigent.integrations.model_discovery.cache import ModelCache, get_global_cache

if TYPE_CHECKING:
    from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)

# Default config file path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "models.yaml"


class ModelDiscovery(ABC):
    """Abstract base class for model discovery.

    Model discovery follows a three-tier validation strategy:
    1. SDK-based discovery (if available and API key is present)
    2. Config file lookup (user-configurable known models)
    3. Pattern-based validation (regex fallback)

    Subclasses must implement provider-specific logic for each tier.
    """

    # Provider name (e.g., "openai", "anthropic")
    PROVIDER: str = ""

    # Framework enum value (for integration with plugins)
    FRAMEWORK: "Framework | None" = None

    def __init__(
        self,
        cache: ModelCache | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Initialize model discovery.

        Args:
            cache: Optional cache instance (uses global if not provided).
            config_path: Optional path to models config file.
        """
        self._cache = cache or get_global_cache()
        self._config_path = config_path or DEFAULT_CONFIG_PATH
        self._config: dict | None = None

    def _get_credential_fingerprint(self) -> str | None:
        """Return a non-sensitive fingerprint of model-list context.

        Providers whose visible model list depends on credentials, account
        context, or an endpoint must override this method. The fingerprint is
        used in the persistent cache key, so it must never contain raw secrets.
        """
        return None

    @staticmethod
    def _fingerprint(*parts: str | None) -> str:
        """Build a short, filesystem-safe fingerprint without retaining secrets."""
        material = "|".join(part or "" for part in parts)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]

    def _get_cache_key(self) -> str:
        """Return the provider cache key scoped to its model-list context."""
        fingerprint = self._get_credential_fingerprint()
        return f"{self.PROVIDER}-{fingerprint}" if fingerprint else self.PROVIDER

    def list_models(self, force_refresh: bool = False) -> list[str]:
        """List available models for this provider.

        Attempts to fetch models from SDK first, falls back to config.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data.

        Returns:
            List of model names.
        """
        cache_key = self._get_cache_key()

        # Try cache first (unless force_refresh)
        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Try SDK-based discovery
        try:
            models = self._fetch_models_from_sdk()
            if models:
                self._cache.set(cache_key, models, self._get_cache_ttl())
                return models
        except Exception as e:
            logger.debug(f"SDK model discovery failed for {self.PROVIDER}: {e}")

        # Fall back to config file
        models = self._get_models_from_config()
        if models:
            # Cache config models with longer TTL (they're static)
            self._cache.set(cache_key, models, 604800)  # 7 days
            return models

        return []

    def is_valid_model(self, model_id: str) -> bool:
        """Check if a model ID is valid for this provider.

        Validation order:
        1. Check against known models (SDK or config)
        2. Check against regex pattern

        Args:
            model_id: The model identifier to validate.

        Returns:
            True if the model is valid, False otherwise.
        """
        if not model_id:
            return False

        # Retired-ID denylist FIRST (#1936/#1937): the pattern fallback exists
        # to accept unknown-but-plausible NEW model IDs, but shape checks also
        # re-admit retired IDs whose form still matches (claude-3-opus-…,
        # dated o1-preview, Gemini 1.5, models/gemini-pro) — silently undoing
        # the catalog sweep. Checked before the known list too, so a stale
        # user-edited config snapshot cannot resurrect a retired ID either.
        if is_retired_model(model_id):
            logger.debug(f"Model {model_id} is retired/delisted for {self.PROVIDER}")
            return False

        # Check known models first
        known_models = self.list_models()
        if model_id in known_models:
            return True

        # Check pattern fallback
        pattern = self.get_pattern()
        if pattern:
            try:
                if re.match(pattern, model_id):
                    logger.debug(
                        f"Model {model_id} matched pattern for {self.PROVIDER}"
                    )
                    return True
            except re.error as e:
                logger.warning(f"Invalid regex pattern for {self.PROVIDER}: {e}")

        return False

    def refresh_cache(self) -> None:
        """Force refresh the model cache."""
        self._cache.invalidate(self._get_cache_key())
        self.list_models(force_refresh=True)

    @abstractmethod
    def _fetch_models_from_sdk(self) -> list[str]:
        """Fetch models from the provider's SDK.

        Returns:
            List of model names, or empty list if unavailable.

        Raises:
            Exception: If SDK call fails (will be caught by caller).
        """
        raise NotImplementedError

    @abstractmethod
    def get_pattern(self) -> str | None:
        """Get the regex pattern for validating model IDs.

        Returns:
            Regex pattern string, or None if no pattern validation.
        """
        raise NotImplementedError

    def _get_cache_ttl(self) -> int:
        """Get the cache TTL for this provider.

        Override in subclasses for provider-specific TTLs.

        Returns:
            TTL in seconds.
        """
        return 86400  # 24 hours default

    def _get_models_from_config(self) -> list[str]:
        """Get known models from config file.

        Returns:
            List of model names from config.
        """
        config = self._load_config()
        if config and self.PROVIDER in config:
            provider_config = config[self.PROVIDER]
            known_models = provider_config.get("known_models", [])
            if isinstance(known_models, list):
                return list(known_models)
            return []
        return []

    def _get_pattern_from_config(self) -> str | None:
        """Get regex pattern from config file.

        Returns:
            Regex pattern or None.
        """
        config = self._load_config()
        if config and self.PROVIDER in config:
            provider_config = config[self.PROVIDER]
            pattern = provider_config.get("pattern")
            if isinstance(pattern, str):
                return pattern
            return None
        return None

    def _load_config(self) -> dict | None:
        """Load the models config file.

        Returns:
            Config dictionary or None if not found.
        """
        if self._config is not None:
            return self._config

        if not self._config_path.exists():
            logger.debug(f"Config file not found: {self._config_path}")
            return None

        try:
            with open(self._config_path) as f:
                self._config = yaml.safe_load(f)
                return self._config
        except Exception as e:
            logger.warning(f"Failed to load models config: {e}")
            return None
