"""HuggingFace integration plugin for TraiGent.

This module provides the HuggingFace-specific plugin implementation for
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


class HuggingFacePlugin(LLMPlugin):
    """Plugin for HuggingFace SDK integration."""

    FRAMEWORK = Framework.HUGGINGFACE

    def _get_supported_canonical_params(self) -> set[str]:
        """HuggingFace InferenceClient supports a limited set of params.

        Params like frequency_penalty, presence_penalty, seed are not
        supported by HuggingFace's text_generation/chat_completion.
        """
        return {
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "stream",
        }

    def _get_metadata(self) -> PluginMetadata:
        """Return HuggingFace plugin metadata."""
        return PluginMetadata(
            name="huggingface",
            version="1.0.0",
            supported_packages=["huggingface_hub", "transformers"],
            priority=IntegrationPriority.NORMAL,
            description="HuggingFace integration",
            author="TraiGent Team",
            requires_packages=["huggingface_hub>=0.20.0"],
            supports_versions={"huggingface_hub": "0."},
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return HuggingFace-specific mappings not handled by the normalizer."""
        return {
            # Prefer 'model' parameter name for InferenceClient
            "model": "model",
            "messages": "messages",
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return validation rules for HuggingFace parameters."""
        return {
            "model": ValidationRule(
                required=False
            ),  # Model can be inferred from client
            "temperature": ValidationRule(
                min_value=0.0, max_value=100.0
            ),  # HF allows > 1
            "top_p": ValidationRule(min_value=0.0, max_value=1.0),
            "top_k": ValidationRule(min_value=0),
        }

    def get_target_classes(self) -> list[str]:
        """Return list of HuggingFace classes to override."""
        return [
            "huggingface_hub.InferenceClient",
            "huggingface_hub.AsyncInferenceClient",
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of classes to methods to override."""
        return {
            "huggingface_hub.InferenceClient": [
                "text_generation",
                "chat_completion",
            ],
            "huggingface_hub.AsyncInferenceClient": [
                "text_generation",
                "chat_completion",
            ],
        }
