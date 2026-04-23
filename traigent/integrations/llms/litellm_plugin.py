"""LiteLLM integration plugin for Traigent.

This module provides the LiteLLM-specific plugin implementation for
parameter mappings and framework overrides. LiteLLM is a multi-provider
proxy that uses OpenAI-compatible parameter names but routes to 100+
providers (OpenAI, Anthropic, Google, Mistral, etc.) via a unified API.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import logging
from typing import Any

from traigent.integrations.base_plugin import (
    IntegrationPriority,
    PluginMetadata,
    ValidationRule,
)
from traigent.integrations.llms.base_llm_plugin import LLMPlugin
from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)


class LiteLLMPlugin(LLMPlugin):
    """Plugin for LiteLLM multi-provider integration.

    LiteLLM uses OpenAI-compatible parameter names for its unified API,
    so most mappings pass through directly. The plugin handles LiteLLM-
    specific parameters like custom_llm_provider, api_base, and the
    provider/model_name model format.
    """

    FRAMEWORK = Framework.LITELLM

    def _get_metadata(self) -> PluginMetadata:
        """Return LiteLLM plugin metadata."""
        return PluginMetadata(
            name="litellm",
            version="1.0.0",
            supported_packages=["litellm"],
            priority=IntegrationPriority.HIGH,
            description="LiteLLM multi-provider integration for 100+ LLM providers",
            author="Traigent Team",
            requires_packages=["litellm>=1.0.0"],
            supports_versions={"litellm": "1."},
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return LiteLLM-specific parameter mappings.

        LiteLLM uses OpenAI-compatible names for most parameters.
        These are the LiteLLM-specific extras.
        """
        return {
            # LiteLLM routing parameters
            "api_base": "api_base",
            "api_key": "api_key",
            "custom_llm_provider": "custom_llm_provider",
            # Response format
            "response_format": "response_format",
            # Tool use (OpenAI-compatible)
            "tools": "tools",
            "tool_choice": "tool_choice",
            # Advanced parameters
            "logit_bias": "logit_bias",
            "n": "n",
            "seed": "seed",
            "user": "user",
            # LiteLLM-specific
            "timeout": "timeout",
            "num_retries": "num_retries",
            "metadata": "metadata",
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return LiteLLM-specific validation rules."""
        return {
            "model": ValidationRule(
                required=True,
                custom_validator="_validate_model",
            ),
            "max_tokens": ValidationRule(min_value=1, max_value=200000),
            "temperature": ValidationRule(min_value=0.0, max_value=2.0),
            "top_p": ValidationRule(min_value=0.0, max_value=1.0),
            "frequency_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "presence_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "n": ValidationRule(min_value=1, max_value=10),
            "seed": ValidationRule(min_value=0),
            "stream": ValidationRule(allowed_values=[True, False]),
            "timeout": ValidationRule(min_value=1, max_value=600),
            "num_retries": ValidationRule(min_value=0, max_value=10),
        }

    def _validate_model(self, param_name: str, value: Any) -> list[str]:
        """Validate LiteLLM model ID.

        LiteLLM supports many model formats:
        - Direct: "gpt-4o", "claude-3-sonnet"
        - Provider-prefixed: "openrouter/openai/gpt-4o"
        - Custom: "custom_llm_provider/model_name"
        """
        errors = []
        if not isinstance(value, str):
            errors.append(f"Parameter '{param_name}' must be a string")
            return errors

        if not value:
            errors.append(f"Parameter '{param_name}' cannot be empty")
            return errors

        # LiteLLM accepts a very wide range of model IDs — don't block
        return errors

    def get_target_classes(self) -> list[str]:
        """Return list of LiteLLM classes/modules to override."""
        return [
            "litellm",
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of LiteLLM modules to functions to override."""
        return {
            "litellm": [
                "completion",
                "acompletion",
                "text_completion",
                "atext_completion",
                "embedding",
                "aembedding",
            ],
        }

    def apply_overrides(self, kwargs: dict[str, Any], config: Any) -> dict[str, Any]:
        """Apply LiteLLM-specific overrides.

        LiteLLM uses OpenAI-compatible parameter names, so the base
        implementation handles most cases. This method adds handling
        for the provider/model format.
        """
        if config is None:
            return kwargs.copy()

        config_obj = self._normalize_config(config)

        # Apply base overrides
        overridden = super().apply_overrides(kwargs, config_obj)

        return overridden
