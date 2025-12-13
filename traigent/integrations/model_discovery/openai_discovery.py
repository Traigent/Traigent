"""OpenAI model discovery implementation.

Uses the OpenAI SDK's models.list() API for dynamic model discovery.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import logging
import os

from traigent.integrations.model_discovery.base import ModelDiscovery
from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)

# Pattern that validates OpenAI model names
# Matches: gpt-4, gpt-4o, gpt-4-turbo, o1-preview, text-davinci-003, etc.
OPENAI_MODEL_PATTERN = (
    r"^(gpt-[0-9]|o[0-9]|text-|code-|davinci|curie|babbage|ada|"
    r"chatgpt-|ft:|whisper|tts|dall-e)"
)


class OpenAIDiscovery(ModelDiscovery):
    """Model discovery for OpenAI."""

    PROVIDER = "openai"
    FRAMEWORK = Framework.OPENAI

    def _fetch_models_from_sdk(self) -> list[str]:
        """Fetch models from OpenAI SDK.

        Returns:
            List of model IDs available via the API.

        Raises:
            Exception: If SDK is not available or API call fails.
        """
        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.debug("OPENAI_API_KEY not set, skipping SDK discovery")
            return []

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            models = client.models.list()

            # Filter to chat/completion models (exclude embeddings, etc.)
            model_ids = []
            for model in models.data:
                model_id = model.id
                # Include GPT, O1, text/code completion models
                if any(
                    model_id.startswith(prefix)
                    for prefix in ["gpt-", "o1-", "text-", "code-", "chatgpt-"]
                ) or model_id in ["davinci", "curie", "babbage", "ada"]:
                    model_ids.append(model_id)

            logger.info(f"Discovered {len(model_ids)} OpenAI models via SDK")
            return sorted(model_ids)

        except ImportError:
            logger.debug("OpenAI SDK not installed")
            raise
        except Exception as e:
            logger.debug(f"OpenAI SDK model list failed: {e}")
            raise

    def get_pattern(self) -> str | None:
        """Get pattern for OpenAI models.

        Pattern matches common OpenAI model prefixes:
        - gpt-* (GPT models)
        - o1-* (O1 reasoning models)
        - text-* (legacy completion models)
        - code-* (Codex models)
        - davinci/curie/babbage/ada (base models)
        - ft:* (fine-tuned models)
        """
        # Try config first
        config_pattern = self._get_pattern_from_config()
        if config_pattern:
            return config_pattern

        return OPENAI_MODEL_PATTERN

    def _get_cache_ttl(self) -> int:
        """OpenAI releases models frequently, use shorter TTL."""
        return 43200  # 12 hours
