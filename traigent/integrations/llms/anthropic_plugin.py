"""Anthropic integration plugin for TraiGent.

This module provides the Anthropic-specific plugin implementation for
parameter mappings and framework overrides.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import logging
from collections.abc import Mapping, Sequence
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

logger = logging.getLogger(__name__)


class AnthropicPlugin(LLMPlugin):
    """Plugin for Anthropic SDK integration."""

    FRAMEWORK = Framework.ANTHROPIC

    def _get_metadata(self) -> PluginMetadata:
        """Return Anthropic plugin metadata."""
        return PluginMetadata(
            name="anthropic",
            version="1.0.0",
            supported_packages=["anthropic"],
            priority=IntegrationPriority.HIGH,
            description="Anthropic SDK integration for Claude models",
            author="TraiGent Team",
            requires_packages=["anthropic>=0.3.0"],
            supports_versions={"anthropic": "0."},
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return Anthropic-specific parameter mappings not in ParameterNormalizer."""
        return {
            # Anthropic-specific
            "metadata": "metadata",
            # API configuration
            "anthropic_api_key": "api_key",
            "anthropic_api_url": "base_url",
            "anthropic_version": "anthropic_version",
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return Anthropic-specific validation rules.

        Note: Model validation uses custom_validator with dynamic discovery
        instead of hardcoded allowed_values list. This ensures new Claude
        models are automatically supported via config file updates.
        """
        return {
            "model": ValidationRule(
                required=True,
                custom_validator="_validate_model",
            ),
            "max_tokens": ValidationRule(
                required=True,
                min_value=1,
                max_value=200000,  # Claude 3 max
            ),
            # Anthropic uses 0-1 range (stricter than common 0-2)
            "temperature": ValidationRule(min_value=0.0, max_value=1.0),
            "top_k": ValidationRule(min_value=1, max_value=500),
            "stream": ValidationRule(allowed_values=[True, False]),
            "timeout": ValidationRule(min_value=1, max_value=600),
            "max_retries": ValidationRule(min_value=0, max_value=10),
        }

    def _validate_model(self, param_name: str, value: Any) -> list[str]:
        """Validate model ID using dynamic discovery.

        Uses the model discovery service which:
        1. Falls back to config file known models (Anthropic has no SDK list API)
        2. Falls back to regex pattern validation (claude-*)
        """
        errors = []
        if not isinstance(value, str):
            errors.append(f"Parameter '{param_name}' must be a string")
            return errors

        if not value:
            errors.append(f"Parameter '{param_name}' cannot be empty")
            return errors

        try:
            from traigent.integrations.model_discovery import get_model_discovery

            discovery = get_model_discovery(self.FRAMEWORK)
            if discovery and not discovery.is_valid_model(value):
                errors.append(
                    f"Model '{value}' is not recognized as a valid Anthropic model. "
                    f"If this is a new model, it may still work."
                )
                # Log as warning but don't block - model might be valid
                logger.warning(
                    f"Unrecognized Anthropic model: {value}. "
                    f"Proceeding anyway as it may be a new model."
                )
                # Clear errors - we warn but don't block
                errors.clear()
        except ImportError:
            # Discovery module not available, skip validation
            logger.debug("Model discovery not available, skipping model validation")

        return errors

    def get_target_classes(self) -> list[str]:
        """Return list of Anthropic classes to override."""
        return [
            "anthropic.Anthropic",
            "anthropic.AsyncAnthropic",
            "anthropic.Client",  # Legacy
            "anthropic.AsyncClient",  # Legacy
            "anthropic.resources.messages.Messages",
            "anthropic.resources.completions.Completions",  # If they add this
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of Anthropic classes to methods to override."""
        return {
            "anthropic.Anthropic": ["messages.create", "messages.stream"],
            "anthropic.AsyncAnthropic": ["messages.create", "messages.stream"],
            "anthropic.Client": ["messages.create", "completions.create"],  # Legacy
            "anthropic.AsyncClient": [
                "messages.create",
                "completions.create",
            ],  # Legacy
            "anthropic.resources.messages.Messages": ["create", "stream"],
        }

    def apply_overrides(
        self, kwargs: dict[str, Any], config: "TraigentConfig | dict[str, Any]"
    ) -> dict[str, Any]:
        """Apply Anthropic-specific overrides.

        This method extends the base implementation to handle Anthropic-specific
        logic like message formatting and system prompts.
        """
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

        # Handle system parameter specially for Anthropic
        # Anthropic requires system to be a separate parameter, not in messages
        if "system" in custom_params and "system" not in overridden:
            overridden["system"] = custom_params["system"]

        # Remove leading system message and promote its content when appropriate
        messages = overridden.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            if messages:
                first_message = messages[0]
                if (
                    isinstance(first_message, Mapping)
                    and first_message.get("role") == "system"
                ):
                    overridden["messages"] = list(messages[1:])
                    if "system" not in overridden:
                        system_content = first_message.get("content", "")
                        overridden["system"] = system_content

        # Handle stop_sequences vs stop parameter naming
        if "stop" in custom_params and "stop_sequences" not in overridden:
            overridden["stop_sequences"] = custom_params["stop"]

        # Ensure max_tokens is set (required for Anthropic)
        if "max_tokens" not in overridden and "max_tokens" not in kwargs:
            # Set a reasonable default based on model
            model = overridden.get("model", kwargs.get("model", ""))
            if "claude-3" in model:
                overridden["max_tokens"] = 4096
            elif "claude-2" in model:
                overridden["max_tokens"] = 4096
            else:
                overridden["max_tokens"] = 1024

        # Handle tool use format differences
        if "tools" in overridden:
            # Ensure tools are in Anthropic format
            tools = overridden["tools"]
            if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)):
                anthropic_tools = []
                for tool in tools:
                    if not isinstance(tool, Mapping):
                        continue
                    function_payload = tool.get("function")
                    if isinstance(function_payload, Mapping):
                        function_name = function_payload.get("name")
                        if not function_name:
                            continue
                        anthropic_tools.append(
                            {
                                "name": function_name,
                                "description": function_payload.get("description", ""),
                                "input_schema": function_payload.get("parameters", {}),
                            }
                        )
                    else:
                        anthropic_tools.append(dict(tool))

                if anthropic_tools:
                    overridden["tools"] = anthropic_tools

        return overridden
