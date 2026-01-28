"""OpenAI integration plugin for Traigent.

This module provides the OpenAI-specific plugin implementation for
parameter mappings and framework overrides.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

import logging
from collections.abc import Mapping
from typing import Any

from traigent.integrations.base_plugin import (
    IntegrationPriority,
    PluginMetadata,
    ValidationRule,
)
from traigent.integrations.llms.base_llm_plugin import LLMPlugin
from traigent.integrations.utils import Framework

logger = logging.getLogger(__name__)


class OpenAIPlugin(LLMPlugin):
    """Plugin for OpenAI SDK integration."""

    FRAMEWORK = Framework.OPENAI

    def _get_metadata(self) -> PluginMetadata:
        """Return OpenAI plugin metadata."""
        return PluginMetadata(
            name="openai",
            version="1.0.0",
            supported_packages=["openai"],
            priority=IntegrationPriority.HIGH,
            description="OpenAI SDK integration for GPT models",
            author="Traigent Team",
            requires_packages=["openai>=1.0.0"],
            supports_versions={"openai": "1."},
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return OpenAI-specific parameter mappings not in ParameterNormalizer."""
        return {
            # Function calling (legacy)
            "functions": "functions",
            "function_call": "function_call",
            # Response format
            "response_format": "response_format",
            # Advanced parameters
            "logit_bias": "logit_bias",
            "logprobs": "logprobs",
            "top_logprobs": "top_logprobs",
            "n": "n",
            # System and user messages
            "system": "system",
            "user": "user",
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return OpenAI-specific validation rules.

        Note: Model validation uses custom_validator with dynamic discovery
        instead of hardcoded allowed_values list. This ensures new models
        are automatically supported via SDK discovery or config file updates.
        """
        return {
            "model": ValidationRule(
                required=True,
                custom_validator="_validate_model",
            ),
            "max_tokens": ValidationRule(
                min_value=1, max_value=128000
            ),  # GPT-4 Turbo max
            "frequency_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "presence_penalty": ValidationRule(min_value=-2.0, max_value=2.0),
            "n": ValidationRule(min_value=1, max_value=10),
            "logprobs": ValidationRule(allowed_values=[True, False]),
            "top_logprobs": ValidationRule(min_value=0, max_value=5),
            "seed": ValidationRule(min_value=0, custom_validator="_validate_seed"),
            "stream": ValidationRule(allowed_values=[True, False]),
            "timeout": ValidationRule(min_value=1, max_value=600),
            "max_retries": ValidationRule(min_value=0, max_value=10),
        }

    def _validate_model(self, param_name: str, value: Any) -> list[str]:
        """Validate model ID using dynamic discovery.

        Uses the model discovery service which:
        1. Tries SDK-based discovery (client.models.list())
        2. Falls back to config file known models
        3. Falls back to regex pattern validation
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
                    f"Model '{value}' is not recognized as a valid OpenAI model. "
                    f"If this is a new model or fine-tuned model, it may still work."
                )
                # Log as warning but don't block - model might be valid
                logger.warning(
                    f"Unrecognized OpenAI model: {value}. "
                    f"Proceeding anyway as it may be a new or custom model."
                )
                # Clear errors - we warn but don't block
                errors.clear()
        except ImportError:
            # Discovery module not available, skip validation
            logger.debug("Model discovery not available, skipping model validation")

        return errors

    def get_target_classes(self) -> list[str]:
        """Return list of OpenAI classes to override."""
        return [
            "openai.OpenAI",
            "openai.AsyncOpenAI",
            "openai.ChatCompletion",  # Legacy
            "openai.Completion",  # Legacy
            "openai.resources.chat.completions.Completions",
            "openai.resources.completions.Completions",
            "openai.resources.embeddings.Embeddings",
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of OpenAI classes to methods to override."""
        return {
            "openai.OpenAI": [
                "chat.completions.create",
                "completions.create",
                "embeddings.create",
            ],
            "openai.AsyncOpenAI": [
                "chat.completions.create",
                "completions.create",
                "embeddings.create",
            ],
            "openai.ChatCompletion": ["create", "acreate"],  # Legacy
            "openai.Completion": ["create", "acreate"],  # Legacy
            "openai.resources.chat.completions.Completions": ["create"],
            "openai.resources.completions.Completions": ["create"],
            "openai.resources.embeddings.Embeddings": ["create"],
        }

    def _validate_seed(self, param_name: str, value: Any) -> list[str]:
        """Custom validator for seed parameter."""
        errors = []
        if not isinstance(value, int):
            errors.append(f"Parameter '{param_name}' must be an integer")
        elif value > 2**32 - 1:
            errors.append(
                f"Parameter '{param_name}' value {value} exceeds maximum (2^32 - 1)"
            )
        return errors

    def apply_overrides(self, kwargs: dict[str, Any], config) -> dict[str, Any]:
        """Apply OpenAI-specific overrides.

        This method extends the base implementation to handle OpenAI-specific
        logic like message formatting, function calling, and reasoning parameters.
        """
        if config is None:
            # Nothing to override without configuration; return a shallow copy for safety.
            return kwargs.copy()

        config_obj = self._normalize_config(config)

        # Apply base overrides
        overridden = super().apply_overrides(kwargs, config_obj)

        # Apply reasoning parameter translation
        overridden = self._apply_reasoning_overrides(overridden, config_obj)

        # Handle OpenAI-specific message formatting if needed
        custom_params_raw = getattr(config_obj, "custom_params", {}) or {}
        if isinstance(custom_params_raw, Mapping):
            custom_params = dict(custom_params_raw)
        else:
            custom_params = {}

        system_message = custom_params.get("system")

        if "messages" in overridden and system_message:
            # Ensure system message is at the beginning
            system_msg = {"role": "system", "content": system_message}
            messages = overridden.get("messages", [])
            if not messages or messages[0].get("role") != "system":
                overridden["messages"] = [system_msg] + messages

        # Handle function/tool calling consistency
        if "functions" in overridden and "tools" not in overridden:
            # Convert functions to tools format for newer API
            functions = overridden.pop("functions")
            overridden["tools"] = [
                {"type": "function", "function": f} for f in functions
            ]
            if "function_call" in overridden:
                fc = overridden.pop("function_call")
                if fc == "auto":
                    overridden["tool_choice"] = "auto"
                elif fc == "none":
                    overridden["tool_choice"] = "none"
                elif isinstance(fc, dict) and "name" in fc:
                    overridden["tool_choice"] = {
                        "type": "function",
                        "function": {"name": fc["name"]},
                    }

        return overridden

    def _apply_reasoning_overrides(
        self, params: dict[str, Any], config
    ) -> dict[str, Any]:
        """Apply reasoning parameter translation for OpenAI models.

        Handles:
        - Generic reasoning_mode -> reasoning_effort translation
        - reasoning_budget -> max_completion_tokens translation
        - Model-specific reasoning_effort validation
        - max_tokens vs max_completion_tokens conflict resolution
        """
        from traigent.integrations.utils.model_capabilities import (
            get_reasoning_effort_levels,
            supports_reasoning,
        )

        # Get model from params or config
        model = params.get("model") or getattr(config, "model", None)
        if not model:
            # No model specified, can't determine reasoning support
            self._clean_reasoning_params(params)
            return params

        # Check if model supports native reasoning
        if not supports_reasoning(model, "openai"):
            # Non-reasoning model: remove all reasoning params (generic + provider-specific)
            self._clean_all_reasoning_params(params)
            return params

        # Translate generic reasoning_mode -> reasoning_effort (if not already set)
        if "reasoning_mode" in params and "reasoning_effort" not in params:
            mode_mapping = {"none": "minimal", "standard": "medium", "deep": "high"}
            mode = params.pop("reasoning_mode")
            params["reasoning_effort"] = mode_mapping.get(mode, "medium")

        # Translate reasoning_budget -> max_completion_tokens (if not already set)
        if "reasoning_budget" in params and "max_completion_tokens" not in params:
            params["max_completion_tokens"] = params.pop("reasoning_budget")

        # Validate reasoning_effort level is available for this model
        if "reasoning_effort" in params:
            available = get_reasoning_effort_levels(model)
            if params["reasoning_effort"] not in available:
                original = params["reasoning_effort"]
                # Fall back to closest available level
                if original == "xhigh" and "high" in available:
                    params["reasoning_effort"] = "high"
                elif original == "minimal" and "low" in available:
                    params["reasoning_effort"] = "low"
                else:
                    params["reasoning_effort"] = "medium"
                logger.debug(
                    f"Reasoning effort '{original}' not available for {model}, "
                    f"using '{params['reasoning_effort']}'"
                )

        # Handle max_tokens vs max_completion_tokens conflict
        # For reasoning models, max_completion_tokens takes precedence
        if "max_completion_tokens" in params and "max_tokens" in params:
            params.pop("max_tokens")
            logger.debug(
                f"Removed max_tokens in favor of max_completion_tokens for reasoning model {model}"
            )

        # Clean up remaining generic params
        self._clean_reasoning_params(params)

        return params

    def _clean_reasoning_params(self, params: dict[str, Any]) -> None:
        """Remove generic reasoning params that shouldn't be sent to the API."""
        params.pop("reasoning_mode", None)
        params.pop("reasoning_budget", None)

    def _clean_all_reasoning_params(self, params: dict[str, Any]) -> None:
        """Remove all reasoning params for non-reasoning models.

        This removes both generic params and provider-specific params since
        non-reasoning models don't support any of them.
        """
        params.pop("reasoning_mode", None)
        params.pop("reasoning_budget", None)
        params.pop("reasoning_effort", None)
        params.pop("max_completion_tokens", None)
