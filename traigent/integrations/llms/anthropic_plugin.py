"""Anthropic integration plugin for Traigent.

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

_MESSAGES_CREATE = "messages.create"

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
            author="Traigent Team",
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
                # Log as warning but don't block - model might be valid
                logger.warning(
                    f"Unrecognized Anthropic model: {value}. "
                    f"Proceeding anyway as it may be a new model."
                )
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
            "anthropic.Anthropic": [_MESSAGES_CREATE, "messages.stream"],
            "anthropic.AsyncAnthropic": [_MESSAGES_CREATE, "messages.stream"],
            "anthropic.Client": [_MESSAGES_CREATE, "completions.create"],  # Legacy
            "anthropic.AsyncClient": [
                _MESSAGES_CREATE,
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

        custom_params = self._extract_custom_params(config_obj)

        # Handle system parameter specially for Anthropic
        # Anthropic requires system to be a separate parameter, not in messages
        if "system" in custom_params and "system" not in overridden:
            overridden["system"] = custom_params["system"]

        self._promote_leading_system_message(overridden)

        # Handle stop_sequences vs stop parameter naming
        if "stop" in custom_params and "stop_sequences" not in overridden:
            overridden["stop_sequences"] = custom_params["stop"]

        # Ensure max_tokens is set (required for Anthropic)
        if "max_tokens" not in overridden and "max_tokens" not in kwargs:
            overridden["max_tokens"] = self._default_max_tokens(overridden, kwargs)

        # Handle tool use format differences
        if "tools" in overridden:
            self._convert_tools(overridden)

        return overridden

    @staticmethod
    def _promote_leading_system_message(overridden: dict[str, Any]) -> None:
        """Remove a leading system message and promote its content."""
        messages = overridden.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            return
        if not messages:
            return
        first_message = messages[0]
        if not (
            isinstance(first_message, Mapping) and first_message.get("role") == "system"
        ):
            return
        overridden["messages"] = list(messages[1:])
        if "system" not in overridden:
            overridden["system"] = first_message.get("content", "")

    @staticmethod
    def _default_max_tokens(overridden: dict[str, Any], kwargs: dict[str, Any]) -> int:
        """Pick a reasonable default max_tokens based on the model name."""
        model = overridden.get("model", kwargs.get("model", ""))
        if "claude-3" in model or "claude-2" in model:
            return 4096
        return 1024

    @staticmethod
    def _convert_tools(overridden: dict[str, Any]) -> None:
        """Ensure tools are in Anthropic format."""
        tools = overridden["tools"]
        if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
            return
        anthropic_tools = []
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            function_payload = tool.get("function")
            if not isinstance(function_payload, Mapping):
                anthropic_tools.append(dict(tool))
                continue
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
        if anthropic_tools:
            overridden["tools"] = anthropic_tools
