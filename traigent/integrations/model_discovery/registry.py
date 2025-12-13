"""Registry for model discovery services.

Provides a central registry for looking up model discovery
implementations by provider name or Framework enum.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from traigent.integrations.model_discovery.base import ModelDiscovery

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008


if TYPE_CHECKING:
    from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)

# Registry of discovery implementations
_discovery_registry: dict[str, type[ModelDiscovery]] = {}
_discovery_instances: dict[str, ModelDiscovery] = {}
_registry_lock = threading.Lock()


def register_discovery(
    provider: str,
    discovery_class: type[ModelDiscovery],
) -> None:
    """Register a model discovery implementation.

    Args:
        provider: Provider name (e.g., "openai", "anthropic").
        discovery_class: ModelDiscovery subclass to register.
    """
    with _registry_lock:
        _discovery_registry[provider.lower()] = discovery_class
        logger.debug(f"Registered model discovery for {provider}")


def get_model_discovery(
    provider: str | Framework | Any,
    cached: bool = True,
) -> ModelDiscovery | None:
    """Get a model discovery instance for a provider.

    Args:
        provider: Provider name (e.g., "openai") or Framework enum.
        cached: If True, return cached instance; if False, create new.

    Returns:
        ModelDiscovery instance, or None if not registered.
    """
    # Convert Framework enum to string
    if hasattr(provider, "value"):
        provider_str = provider.value
    else:
        provider_str = str(provider).lower()

    with _registry_lock:
        # Return cached instance if available
        if cached and provider_str in _discovery_instances:
            return _discovery_instances[provider_str]

        # Look up in registry
        discovery_class = _discovery_registry.get(provider_str)
        if discovery_class is None:
            logger.debug(f"No model discovery registered for {provider_str}")
            return None

        # Create instance
        instance = discovery_class()

        if cached:
            _discovery_instances[provider_str] = instance

        return instance


def list_registered_providers() -> list[str]:
    """List all registered provider names.

    Returns:
        List of provider names.
    """
    with _registry_lock:
        return list(_discovery_registry.keys())


def clear_registry() -> None:
    """Clear all registered discoveries (primarily for testing)."""
    with _registry_lock:
        _discovery_registry.clear()
        _discovery_instances.clear()


def _register_default_discoveries() -> None:
    """Register default model discovery implementations."""
    # Import here to avoid circular imports
    from traigent.integrations.model_discovery.anthropic_discovery import (
        AnthropicDiscovery,
    )
    from traigent.integrations.model_discovery.azure_discovery import (
        AzureOpenAIDiscovery,
    )
    from traigent.integrations.model_discovery.gemini_discovery import GeminiDiscovery
    from traigent.integrations.model_discovery.mistral_discovery import MistralDiscovery
    from traigent.integrations.model_discovery.openai_discovery import OpenAIDiscovery

    register_discovery("openai", OpenAIDiscovery)
    register_discovery("anthropic", AnthropicDiscovery)
    register_discovery("gemini", GeminiDiscovery)
    register_discovery("azure_openai", AzureOpenAIDiscovery)
    register_discovery("mistral", MistralDiscovery)


# Auto-register default discoveries on module load
_register_default_discoveries()
