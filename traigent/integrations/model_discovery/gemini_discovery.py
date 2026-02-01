"""Google Gemini model discovery implementation.

Uses the google-genai SDK's list_models() API for dynamic discovery.
Note: google-generativeai is kept for LangChain compatibility.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import logging
import os

from traigent.integrations.model_discovery.base import ModelDiscovery
from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)

# Pattern that validates Gemini model names
# Matches: gemini-1.5-pro, gemini-pro, models/gemini-1.5-flash, etc.
GEMINI_MODEL_PATTERN = r"^(gemini-[0-9]|models/gemini-)"


class GeminiDiscovery(ModelDiscovery):
    """Model discovery for Google Gemini."""

    PROVIDER = "gemini"
    FRAMEWORK = Framework.GEMINI

    def _fetch_models_from_sdk(self) -> list[str]:
        """Fetch models from Google GenAI SDK.

        Returns:
            List of model names available via the API.

        Raises:
            Exception: If SDK is not available or API call fails.
        """
        # Check if API key is available
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.debug(
                "GOOGLE_API_KEY/GEMINI_API_KEY not set, skipping SDK discovery"
            )
            return []

        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            models_response = client.models.list()

            # Filter to generative models (exclude embedding models)
            model_names = []
            for model in models_response:
                name = model.name
                # Models come as "models/gemini-1.5-pro" format
                # Also accept short names like "gemini-1.5-pro"
                if "gemini" in name.lower():
                    # Store both full and short name
                    model_names.append(name)
                    # Extract short name if in "models/xxx" format
                    if name.startswith("models/"):
                        short_name = name[7:]  # Remove "models/" prefix
                        model_names.append(short_name)

            # Deduplicate and sort
            model_names = sorted(set(model_names))
            logger.info(f"Discovered {len(model_names)} Gemini models via SDK")
            return model_names

        except ImportError:
            logger.debug("google-genai SDK not installed")
            raise
        except Exception as e:
            logger.debug(f"Gemini SDK model list failed: {e}")
            raise

    def get_pattern(self) -> str | None:
        """Get pattern for Gemini models.

        Pattern matches:
        - gemini-* (short format)
        - models/gemini-* (full format from SDK)
        """
        # Try config first
        config_pattern = self._get_pattern_from_config()
        if config_pattern:
            return config_pattern

        return GEMINI_MODEL_PATTERN

    def _get_cache_ttl(self) -> int:
        """Gemini models update moderately, use standard TTL."""
        return 86400  # 24 hours
