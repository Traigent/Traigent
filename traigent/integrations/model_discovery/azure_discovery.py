"""Azure OpenAI model discovery implementation.

Azure OpenAI uses deployments rather than direct model names.
This implementation supports both deployment listing and permissive validation.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import logging
import os

from traigent.integrations.model_discovery.base import ModelDiscovery
from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)

# Pattern that validates Azure OpenAI deployment/model names
# Azure uses deployment names, which can be any user-defined string
# We also accept standard OpenAI model names for convenience
AZURE_MODEL_PATTERN = r"^(gpt-[0-9]|o[0-9]|text-|code-|davinci|curie|babbage|ada|.+)"

# Known Azure OpenAI base models (underlying models, not deployments)
KNOWN_AZURE_BASE_MODELS = [
    # GPT-4 family
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    # GPT-3.5 family
    "gpt-35-turbo",  # Azure uses "35" not "3.5"
    "gpt-35-turbo-16k",
    # O1 family
    "o1-preview",
    "o1-mini",
    # Legacy
    "text-davinci-003",
    "text-davinci-002",
]


class AzureOpenAIDiscovery(ModelDiscovery):
    """Model discovery for Azure OpenAI.

    Azure OpenAI works differently from standard OpenAI:
    - Users create "deployments" with custom names
    - Each deployment is backed by a base model
    - The deployment name is what users pass as "model"

    This discovery:
    1. Attempts to list deployments via Azure Management API (complex auth)
    2. Falls back to known base model names
    3. Uses permissive pattern validation (any string is valid deployment name)
    """

    PROVIDER = "azure_openai"
    FRAMEWORK = Framework.AZURE_OPENAI

    def _fetch_models_from_sdk(self) -> list[str]:
        """Attempt to fetch deployments from Azure.

        Note: This requires Azure credentials and is complex to implement
        properly. For most use cases, pattern-based validation is sufficient.

        Returns:
            List of deployment names, or empty list.

        Raises:
            Exception: If Azure API call fails.
        """
        # Check for Azure credentials
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not endpoint or not api_key:
            logger.debug("Azure OpenAI credentials not set, skipping SDK discovery")
            return []

        try:
            # Attempt to list deployments using REST API
            # This requires the Azure-specific list deployments endpoint
            import httpx

            # Azure API: GET {endpoint}/openai/deployments?api-version=2024-02-01
            url = f"{endpoint.rstrip('/')}/openai/deployments"
            params = {"api-version": "2024-02-01"}
            headers = {"api-key": api_key}

            response = httpx.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            deployments = [d["id"] for d in data.get("data", [])]

            logger.info(f"Discovered {len(deployments)} Azure OpenAI deployments")
            return sorted(deployments)

        except ImportError:
            logger.debug("httpx not installed for Azure deployment listing")
            return []
        except Exception as e:
            logger.debug(f"Azure deployment listing failed: {e}")
            # Don't raise - fall back to config/pattern
            return []

    def list_models(self, force_refresh: bool = False) -> list[str]:
        """List available Azure OpenAI models/deployments.

        Combines:
        1. SDK-discovered deployments (if available)
        2. Config file deployments
        3. Known base model names

        Args:
            force_refresh: If True, bypass cache.

        Returns:
            List of model/deployment names.
        """
        # Try cache first
        if not force_refresh:
            cached = self._cache.get(self.PROVIDER)
            if cached is not None:
                return cached

        all_models = set()

        # Try SDK discovery
        try:
            sdk_models = self._fetch_models_from_sdk()
            all_models.update(sdk_models)
        except Exception:
            pass

        # Add config models
        config_models = self._get_models_from_config()
        all_models.update(config_models)

        # Add known base models
        all_models.update(KNOWN_AZURE_BASE_MODELS)

        models = sorted(all_models)
        self._cache.set(self.PROVIDER, models, 604800)  # 7 days
        return models

    def is_valid_model(self, model_id: str) -> bool:
        """Validate Azure OpenAI model/deployment name.

        Azure is permissive - deployment names are user-defined.
        We accept any non-empty string as valid.

        Args:
            model_id: The model/deployment identifier.

        Returns:
            True if valid (non-empty string).
        """
        if not model_id:
            return False

        # Check known models first
        known_models = self.list_models()
        if model_id in known_models:
            return True

        # Azure deployment names are user-defined, so we're permissive
        # Just check it's a reasonable string (non-empty, alphanumeric with dashes)
        import re

        # Allow alphanumeric, dashes, underscores (typical deployment naming)
        if re.match(r"^[a-zA-Z0-9_-]+$", model_id):
            logger.debug(f"Azure model {model_id} accepted as valid deployment name")
            return True

        # Also accept standard OpenAI model name patterns
        pattern = self.get_pattern()
        if pattern and re.match(pattern, model_id):
            return True

        return False

    def get_pattern(self) -> str | None:
        """Get pattern for Azure OpenAI models.

        Returns a very permissive pattern since deployment names are user-defined.
        """
        # Try config first
        config_pattern = self._get_pattern_from_config()
        if config_pattern:
            return config_pattern

        # Permissive pattern - alphanumeric with dashes/underscores
        return r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$"

    def _get_cache_ttl(self) -> int:
        """Azure deployments are fairly stable, use longer TTL."""
        return 604800  # 7 days
