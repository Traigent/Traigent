"""Cohere integration plugin for TraiGent.

This module provides the Cohere-specific plugin implementation for
parameter mappings and framework overrides.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from typing import TYPE_CHECKING

from traigent.integrations.base_plugin import (
    IntegrationPriority,
    PluginMetadata,
    ValidationRule,
)
from traigent.integrations.llms import LLMPlugin
from traigent.integrations.utils import Framework

if TYPE_CHECKING:
    pass


class CoherePlugin(LLMPlugin):
    """Plugin for Cohere SDK integration."""

    FRAMEWORK = Framework.COHERE

    def _get_supported_canonical_params(self) -> set[str]:
        """Cohere supports a limited set of params.

        Params like frequency_penalty, presence_penalty, seed are not
        supported by Cohere's chat/generate endpoints.
        """
        return {
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "stream",
            "system",  # Maps to preamble in Cohere
        }

    def _get_metadata(self) -> PluginMetadata:
        """Return Cohere plugin metadata."""
        return PluginMetadata(
            name="cohere",
            version="1.0.0",
            supported_packages=["cohere"],
            priority=IntegrationPriority.NORMAL,
            description="Cohere integration",
            author="TraiGent Team",
            requires_packages=["cohere>=5.0.0"],
            supports_versions={"cohere": "5."},
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return Cohere-specific mappings not handled by the normalizer."""
        return {
            "messages": "chat_history",
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return validation rules for Cohere parameters."""
        return {
            "model": ValidationRule(required=True),
            "temperature": ValidationRule(min_value=0.0, max_value=1.0),
            "top_p": ValidationRule(min_value=0.0, max_value=1.0),
            "top_k": ValidationRule(min_value=0),
        }

    def get_target_classes(self) -> list[str]:
        """Return list of Cohere classes to override."""
        return [
            "cohere.Client",
            "cohere.ClientV2",
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of classes to methods to override."""
        return {
            "cohere.Client": [
                "chat",
                "chat_stream",
                "generate",
                "generate_stream",
            ],
            "cohere.ClientV2": [
                "chat",
                "chat_stream",
            ],
        }
