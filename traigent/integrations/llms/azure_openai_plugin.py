"""Azure OpenAI integration plugin for Traigent.

This module provides the Azure OpenAI-specific plugin implementation for
parameter mappings and framework overrides.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from typing import TYPE_CHECKING, Any

from traigent.integrations.base_plugin import (
    IntegrationPriority,
    PluginMetadata,
    ValidationRule,
)
from traigent.integrations.llms.base_llm_plugin import LLMPlugin
from traigent.integrations.utils import Framework

if TYPE_CHECKING:
    from traigent.config.types import TraigentConfig


class AzureOpenAIPlugin(LLMPlugin):
    """Plugin for Azure OpenAI SDK integration."""

    FRAMEWORK = Framework.AZURE_OPENAI

    def _get_metadata(self) -> PluginMetadata:
        """Return Azure OpenAI plugin metadata."""
        return PluginMetadata(
            name="azure_openai",
            version="1.0.0",
            supported_packages=["openai"],
            priority=IntegrationPriority.HIGH,
            description="Azure OpenAI integration",
            author="Traigent Team",
            requires_packages=["openai>=1.0.0"],
            supports_versions={"openai": "1."},
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return Azure OpenAI-specific parameter mappings not in ParameterNormalizer."""
        return {
            # Azure-specific aliases
            "deployment": "model",  # In v1+, model param is used for deployment
            "azure_deployment": "model",  # Alias
            # Azure endpoint config
            "api_version": "api_version",
            "azure_endpoint": "azure_endpoint",
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return Azure OpenAI-specific validation rules."""
        return {
            "model": ValidationRule(required=True),
            "presence_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "frequency_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "stream": ValidationRule(allowed_values=[True, False]),
        }

    def get_target_classes(self) -> list[str]:
        """Return list of Azure OpenAI classes to override."""
        return [
            "openai.AzureOpenAI",
            "openai.AsyncAzureOpenAI",
            "openai.lib.azure.AzureOpenAI",
            "openai.lib.azure.AsyncAzureOpenAI",
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of classes to methods to override."""
        return {
            "openai.AzureOpenAI": ["chat.completions.create", "embeddings.create"],
            "openai.AsyncAzureOpenAI": ["chat.completions.create", "embeddings.create"],
            "openai.lib.azure.AzureOpenAI": [
                "chat.completions.create",
                "embeddings.create",
            ],
            "openai.lib.azure.AsyncAzureOpenAI": [
                "chat.completions.create",
                "embeddings.create",
            ],
        }

    def apply_overrides(
        self, kwargs: dict[str, Any], config: "TraigentConfig | dict[str, Any]"
    ) -> dict[str, Any]:
        """Apply Azure OpenAI-specific overrides."""
        config_obj = self._normalize_config(config)

        # Apply base overrides
        overridden = super().apply_overrides(kwargs, config_obj)

        custom_params = self._extract_custom_params(config_obj)

        # Handle deployment vs model ambiguity
        # If 'deployment' is provided in custom_params, it should override 'model'
        if "deployment" in custom_params and "model" not in overridden:
            overridden["model"] = custom_params["deployment"]

        return overridden
