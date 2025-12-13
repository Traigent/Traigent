"""Mistral AI model discovery implementation.

Uses the Mistral SDK's models.list() API for dynamic model discovery.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import logging
import os

from traigent.integrations.model_discovery.base import ModelDiscovery
from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)

# Pattern that validates Mistral model names
# Matches: mistral-small-latest, mistral-large-latest, open-mistral-*, codestral-*, etc.
MISTRAL_MODEL_PATTERN = (
    r"^(mistral-|open-mistral-|open-mixtral-|codestral-|pixtral-|"
    r"ministral-|ft:mistral-)"
)


class MistralDiscovery(ModelDiscovery):
    """Model discovery for Mistral AI."""

    PROVIDER = "mistral"
    FRAMEWORK = Framework.MISTRAL

    def _fetch_models_from_sdk(self) -> list[str]:
        """Fetch models from Mistral SDK.

        Returns:
            List of model IDs available via the API.

        Raises:
            Exception: If SDK is not available or API call fails.
        """
        # Check if API key is available
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logger.debug("MISTRAL_API_KEY not set, skipping SDK discovery")
            return []

        try:
            from mistralai import Mistral

            client = Mistral(api_key=api_key)
            models_response = client.models.list()

            model_ids = []
            for model in models_response.data:
                model_id = model.id
                model_ids.append(model_id)

            logger.info(f"Discovered {len(model_ids)} Mistral models via SDK")
            return sorted(model_ids)

        except ImportError:
            logger.debug("Mistral SDK not installed")
            raise
        except Exception as e:
            logger.debug(f"Mistral SDK model list failed: {e}")
            raise

    def get_pattern(self) -> str | None:
        """Get pattern for Mistral models.

        Pattern matches common Mistral model prefixes:
        - mistral-* (Mistral models like mistral-small, mistral-large)
        - open-mistral-* (Open-source Mistral models)
        - open-mixtral-* (Mixtral MoE models)
        - codestral-* (Code-focused models)
        - pixtral-* (Multimodal models)
        - ministral-* (Smaller models)
        - ft:mistral-* (Fine-tuned models)
        """
        # Try config first
        config_pattern = self._get_pattern_from_config()
        if config_pattern:
            return config_pattern

        return MISTRAL_MODEL_PATTERN

    def _get_cache_ttl(self) -> int:
        """Mistral releases models periodically, use moderate TTL."""
        return 43200  # 12 hours

    def _get_default_models(self) -> list[str]:
        """Get default known Mistral models.

        These are well-known Mistral models that should always be valid.
        Used as fallback when SDK discovery fails.
        """
        return [
            # Latest aliases
            "mistral-small-latest",
            "mistral-medium-latest",
            "mistral-large-latest",
            # Specific versions (as of late 2024)
            "mistral-small-2409",
            "mistral-large-2411",
            # Open models
            "open-mistral-7b",
            "open-mistral-nemo",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b",
            # Specialized models
            "codestral-latest",
            "codestral-2405",
            "pixtral-12b-2409",
            "ministral-3b-latest",
            "ministral-8b-latest",
        ]
