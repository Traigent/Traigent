"""AWS Bedrock integration plugin for Traigent.

This module provides the Bedrock-specific plugin implementation for
parameter mappings and framework overrides, targeting the Traigent
BedrockChatClient wrapper.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from traigent.integrations.base_plugin import (
    IntegrationPriority,
    PluginMetadata,
    ValidationRule,
)
from traigent.integrations.llms import LLMPlugin
from traigent.integrations.utils import Framework

if TYPE_CHECKING:
    from traigent.config.types import TraigentConfig


class BedrockPlugin(LLMPlugin):
    """Plugin for AWS Bedrock integration via BedrockChatClient."""

    FRAMEWORK = Framework.BEDROCK

    def _get_supported_canonical_params(self) -> set[str]:
        """Bedrock only supports a subset of LLM params.

        Returns only the canonical params that Bedrock's invoke() accepts.
        Other params like frequency_penalty, presence_penalty, etc. are not
        supported and would cause ValidationException/400 errors.
        """
        return {"model", "max_tokens", "temperature", "top_p", "stop"}

    def _get_extra_mappings(self) -> dict[str, str]:
        """Provider-specific mappings not covered by the normalizer."""
        # Explicit stop_sequences key for compatibility with existing configs/tests.
        return {"stop_sequences": "stop_sequences"}

    def _get_metadata(self) -> PluginMetadata:
        """Return Bedrock plugin metadata."""
        return PluginMetadata(
            name="bedrock",
            version="1.0.0",
            supported_packages=["boto3", "botocore"],
            priority=IntegrationPriority.NORMAL,
            description="AWS Bedrock integration for Claude models",
            author="Traigent Team",
            requires_packages=["boto3>=1.34.0"],
            supports_versions={"boto3": "1."},
        )

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return validation rules for Bedrock parameters."""
        return {
            "model": ValidationRule(required=True),
            "max_tokens": ValidationRule(
                min_value=1,
                max_value=200000,  # Claude 3 max
            ),
            "temperature": ValidationRule(min_value=0.0, max_value=1.0),
            "top_p": ValidationRule(min_value=0.0, max_value=1.0),
        }

    def get_target_classes(self) -> list[str]:
        """Return list of classes to override."""
        return [
            "traigent.integrations.bedrock_client.BedrockChatClient",
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of classes to methods to override."""
        return {
            "traigent.integrations.bedrock_client.BedrockChatClient": [
                "invoke",
                "invoke_stream",
            ],
        }

    def apply_overrides(
        self, kwargs: dict[str, Any], config: "TraigentConfig | dict[str, Any]"
    ) -> dict[str, Any]:
        """Apply Bedrock-specific overrides."""
        config_obj = self._normalize_config(config)

        # Apply base overrides
        overridden = super().apply_overrides(kwargs, config_obj)

        custom_params_raw = getattr(config_obj, "custom_params", {}) or {}
        if isinstance(custom_params_raw, Mapping):
            custom_params = dict(custom_params_raw)
        else:
            try:
                custom_params = dict(custom_params_raw)
            except Exception:
                custom_params = {}

        # Handle stop sequences
        if "stop" in custom_params and "stop_sequences" not in overridden:
            overridden["stop_sequences"] = custom_params["stop"]

        # Handle extra_params for BedrockChatClient
        # BedrockChatClient accepts 'extra_params' dict for things not explicitly named
        extra_params = overridden.get("extra_params", {})
        if not isinstance(extra_params, dict):
            extra_params = {}

        # Move unknown params to extra_params if they are not arguments to invoke
        # This is a bit specific to BedrockChatClient's signature
        known_args = {
            "model_id",
            "messages",
            "max_tokens",
            "temperature",
            "top_p",
            "extra_params",
        }

        # Client construction params - these should NOT be passed to invoke()
        # They are only used when creating the BedrockChatClient instance
        client_construction_params = {"region_name", "profile_name"}

        keys_to_move = []
        for key in overridden:
            if key not in known_args and key not in client_construction_params:
                keys_to_move.append(key)

        # Remove client construction params from overridden (they're invalid for invoke)
        for key in client_construction_params:
            overridden.pop(key, None)

        if keys_to_move:
            for key in keys_to_move:
                extra_params[key] = overridden.pop(key)
            overridden["extra_params"] = extra_params

        return overridden
