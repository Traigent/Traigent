"""Mistral AI integration plugin for Traigent.

This module provides the Mistral AI-specific plugin implementation for
parameter mappings and framework overrides.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import logging
from collections.abc import Mapping
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


class MistralPlugin(LLMPlugin):
    """Plugin for Mistral AI SDK integration.

    Supports the official Mistral AI Python SDK (mistralai).
    Handles parameter mapping for chat completions including streaming,
    tool use, and Mistral-specific parameters like safe_prompt.
    """

    FRAMEWORK = Framework.MISTRAL

    def _get_metadata(self) -> PluginMetadata:
        """Return Mistral plugin metadata."""
        return PluginMetadata(
            name="mistral",
            version="1.0.0",
            supported_packages=["mistralai"],
            priority=IntegrationPriority.HIGH,
            description="Mistral AI SDK integration for Mistral models",
            author="Traigent Team",
            requires_packages=["mistralai>=1.0.0"],
            supports_versions={"mistralai": "1."},
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return Mistral-specific parameter mappings not in ParameterNormalizer."""
        return {
            # Mistral-specific parameters
            "random_seed": "random_seed",
            "safe_prompt": "safe_prompt",
            "parallel_tool_calls": "parallel_tool_calls",
            "prompt_mode": "prompt_mode",
            "prediction": "prediction",
            # API configuration
            "mistral_api_key": "api_key",
            # Tool use
            "tool_choice": "tool_choice",
            "tools": "tools",
            # Response format
            "response_format": "response_format",
            # Number of completions
            "n": "n",
            # Common aliases mapped to Mistral-specific names
            "seed": "random_seed",  # OpenAI-style 'seed' → Mistral 'random_seed'
            "stop_sequences": "stop",  # Common alias for stop sequences
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return Mistral-specific validation rules.

        Note: Model validation uses custom_validator with dynamic discovery
        instead of hardcoded allowed_values list. This ensures new Mistral
        models are automatically supported.
        """
        return {
            "model": ValidationRule(
                required=True,
                custom_validator="_validate_model",
            ),
            # Mistral recommends temperature between 0.0 and 0.7
            "temperature": ValidationRule(min_value=0.0, max_value=1.0),
            "top_p": ValidationRule(min_value=0.0, max_value=1.0),
            "max_tokens": ValidationRule(min_value=1, max_value=128000),
            "stream": ValidationRule(allowed_values=[True, False]),
            "safe_prompt": ValidationRule(allowed_values=[True, False]),
            "parallel_tool_calls": ValidationRule(allowed_values=[True, False]),
            "n": ValidationRule(min_value=1, max_value=16),
            # Frequency and presence penalties (like OpenAI)
            "frequency_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "presence_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "random_seed": ValidationRule(min_value=0),
        }

    def _validate_model(self, param_name: str, value: Any) -> list[str]:
        """Validate model ID using dynamic discovery.

        Uses the model discovery service which:
        1. Tries SDK-based discovery (client.models.list())
        2. Falls back to config file known models
        3. Falls back to regex pattern validation (mistral-*)
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
                    f"Model '{value}' is not recognized as a valid Mistral model. "
                    f"If this is a new model, it may still work."
                )
                # Log as warning but don't block - model might be valid
                logger.warning(
                    f"Unrecognized Mistral model: {value}. "
                    f"Proceeding anyway as it may be a new model."
                )
                # Clear errors - we warn but don't block
                errors.clear()
        except ImportError:
            # Discovery module not available, skip validation
            logger.debug("Model discovery not available, skipping model validation")

        return errors

    def get_target_classes(self) -> list[str]:
        """Return list of Mistral classes to override."""
        return [
            "mistralai.Mistral",
            "mistralai.async_client.Mistral",
            # Chat resource classes
            "mistralai.chat.Chat",
            "mistralai.resources.chat.Chat",
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of Mistral classes to methods to override."""
        return {
            "mistralai.Mistral": [
                "chat.complete",
                "chat.complete_async",
                "chat.stream",
                "chat.stream_async",
            ],
            "mistralai.async_client.Mistral": [
                "chat.complete",
                "chat.complete_async",
                "chat.stream",
                "chat.stream_async",
            ],
            "mistralai.chat.Chat": [
                "complete",
                "complete_async",
                "stream",
                "stream_async",
            ],
            "mistralai.resources.chat.Chat": [
                "complete",
                "complete_async",
                "stream",
                "stream_async",
            ],
        }

    def apply_overrides(
        self, kwargs: dict[str, Any], config: "TraigentConfig | dict[str, Any]"
    ) -> dict[str, Any]:
        """Apply Mistral-specific overrides.

        This method extends the base implementation to handle Mistral-specific
        logic like message formatting and safe_prompt handling.
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

        # Handle stop sequences - Mistral uses 'stop' parameter
        if "stop_sequences" in custom_params and "stop" not in overridden:
            overridden["stop"] = custom_params["stop_sequences"]

        # Handle random_seed for reproducibility
        if "seed" in custom_params and "random_seed" not in overridden:
            overridden["random_seed"] = custom_params["seed"]

        # Handle safe_prompt - Mistral-specific safety feature
        if "safe_prompt" in custom_params:
            overridden["safe_prompt"] = custom_params["safe_prompt"]

        # Ensure messages are in Mistral format
        # Mistral expects messages as list of dicts with 'role' and 'content'
        messages = overridden.get("messages")
        if messages is not None:
            formatted_messages = self._format_messages(messages)
            if formatted_messages:
                overridden["messages"] = formatted_messages

        return overridden

    def _format_messages(self, messages: Any) -> list[dict[str, Any]] | None:
        """Format messages to Mistral's expected format.

        Mistral expects messages as:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Args:
            messages: Input messages in various formats.

        Returns:
            Formatted messages list, or None if no formatting needed.
        """
        if messages is None:
            return None

        # Already a list of dicts - likely already formatted
        if isinstance(messages, list):
            if all(isinstance(m, dict) for m in messages):
                return None  # Already in correct format

        # String input - wrap as single user message
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        return None
