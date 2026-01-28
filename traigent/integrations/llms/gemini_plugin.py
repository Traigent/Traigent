"""Google Gemini integration plugin for Traigent.

This module provides the Google Gemini-specific plugin implementation for
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
from traigent.integrations.llms import LLMPlugin
from traigent.integrations.utils import Framework

if TYPE_CHECKING:
    from traigent.config.types import TraigentConfig

logger = logging.getLogger(__name__)


class GeminiPlugin(LLMPlugin):
    """Plugin for Google Gemini SDK integration."""

    FRAMEWORK = Framework.GEMINI

    def _get_supported_canonical_params(self) -> set[str]:
        """Gemini supports a limited set of generation params.

        Params like frequency_penalty, presence_penalty, etc. are not
        supported and would cause TypeError in generate_content().

        Note: 'stream' is included to support streaming responses.
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
        """Return Gemini plugin metadata."""
        return PluginMetadata(
            name="gemini",
            version="1.0.0",
            supported_packages=["google-generativeai"],
            priority=IntegrationPriority.NORMAL,
            description="Google Gemini integration",
            author="Traigent Team",
            requires_packages=["google-generativeai>=0.3.0"],
            supports_versions={"google-generativeai": "0."},
        )

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return provider-specific mappings not handled by the normalizer."""
        return {
            "candidate_count": "candidate_count",
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return validation rules for Gemini parameters.

        Note: Model validation uses custom_validator with dynamic discovery
        for future-proofing as Google releases new Gemini models.
        """
        return {
            "model": ValidationRule(
                required=True,
                custom_validator="_validate_model",
            ),
            "temperature": ValidationRule(min_value=0.0, max_value=1.0),
            "top_p": ValidationRule(min_value=0.0, max_value=1.0),
            "top_k": ValidationRule(min_value=1),
            "candidate_count": ValidationRule(min_value=1, max_value=8),
        }

    def _validate_model(self, param_name: str, value: Any) -> list[str]:
        """Validate model ID using dynamic discovery.

        Uses the model discovery service which:
        1. Tries SDK-based discovery (genai.list_models())
        2. Falls back to config file known models
        3. Falls back to regex pattern validation (gemini-*)
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
                    f"Model '{value}' is not recognized as a valid Gemini model. "
                    f"If this is a new model, it may still work."
                )
                # Log as warning but don't block - model might be valid
                logger.warning(
                    f"Unrecognized Gemini model: {value}. "
                    f"Proceeding anyway as it may be a new model."
                )
                # Clear errors - we warn but don't block
                errors.clear()
        except ImportError:
            # Discovery module not available, skip validation
            logger.debug("Model discovery not available, skipping model validation")

        return errors

    def get_target_classes(self) -> list[str]:
        """Return list of Gemini classes to override."""
        return [
            "google.generativeai.GenerativeModel",
            "google.generativeai.ChatSession",
        ]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of classes to methods to override."""
        return {
            "google.generativeai.GenerativeModel": [
                "generate_content",
                "generate_content_async",
                "start_chat",
            ],
            "google.generativeai.ChatSession": [
                "send_message",
                "send_message_async",
            ],
        }

    def apply_overrides(
        self, kwargs: dict[str, Any], config: "TraigentConfig | dict[str, Any]"
    ) -> dict[str, Any]:
        """Apply Gemini-specific overrides.

        Gemini SDK expects generation parameters inside a 'generation_config' dict,
        not as top-level kwargs. This method wraps them appropriately.
        """
        config_obj = self._normalize_config(config)

        # Apply base overrides
        overridden = super().apply_overrides(kwargs, config_obj)

        # Apply reasoning/thinking parameter translation
        overridden = self._apply_reasoning_overrides(overridden, config_obj)

        custom_params_raw = getattr(config_obj, "custom_params", {}) or {}
        if isinstance(custom_params_raw, Mapping):
            custom_params = dict(custom_params_raw)
        else:
            try:
                custom_params = dict(custom_params_raw)
            except Exception:
                custom_params = {}

        # Handle stop sequences alias
        if "stop" in custom_params and "stop_sequences" not in overridden:
            overridden["stop_sequences"] = custom_params["stop"]

        # Gemini SDK requires generation params inside generation_config dict
        # Extract generation params and wrap them
        generation_params = [
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "candidate_count",
            "stop_sequences",
            "thinking_level",  # Gemini 3 thinking parameter
            "thinking_budget",  # Gemini 2.5 thinking parameter
        ]

        generation_config = overridden.get("generation_config", {})
        if not isinstance(generation_config, dict):
            generation_config = {}

        for param in generation_params:
            if param in overridden:
                generation_config[param] = overridden.pop(param)

        if generation_config:
            overridden["generation_config"] = generation_config

        return overridden

    def _apply_reasoning_overrides(
        self, params: dict[str, Any], config
    ) -> dict[str, Any]:
        """Apply thinking parameter translation for Gemini models.

        Handles the difference between Gemini 3 (thinking_level) and
        Gemini 2.5 (thinking_budget).

        Provider-specific params override generic reasoning_mode/reasoning_budget.
        """
        from traigent.integrations.utils.model_capabilities import (
            is_gemini_3,
            supports_reasoning,
        )

        # Get model from params or config
        model = params.get("model") or getattr(config, "model", None)
        if not model:
            # No model specified, can't determine reasoning support
            self._clean_reasoning_params(params)
            return params

        # Check if model supports native reasoning (thinking)
        if not supports_reasoning(model, "gemini"):
            # Non-reasoning model: remove all reasoning params (generic + provider-specific)
            self._clean_all_reasoning_params(params)
            return params

        if is_gemini_3(model):
            # Gemini 3 uses thinking_level
            if "thinking_level" in params:
                # Provider-specific param already set, use it
                pass
            elif "reasoning_mode" in params:
                mode_mapping = {"none": "MINIMAL", "standard": "low", "deep": "high"}
                mode = params.pop("reasoning_mode")
                params["thinking_level"] = mode_mapping.get(mode, "high")
            # Remove thinking_budget as it's not used for Gemini 3
            params.pop("thinking_budget", None)
        else:
            # Gemini 2.5 uses thinking_budget
            if "thinking_budget" in params:
                # Provider-specific param already set, clamp to max
                params["thinking_budget"] = min(params["thinking_budget"], 32768)
            elif "reasoning_budget" in params:
                # Translate generic param, clamp to Gemini 2.5 max
                params["thinking_budget"] = min(params.pop("reasoning_budget"), 32768)
            elif "reasoning_mode" in params:
                budget_mapping: dict[str, int] = {
                    "none": 0,
                    "standard": 8192,
                    "deep": 32768,
                }
                mode = params.pop("reasoning_mode")
                params["thinking_budget"] = budget_mapping.get(mode, 8192)
            # Remove thinking_level as it's not used for Gemini 2.5
            params.pop("thinking_level", None)

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
        non-reasoning Gemini models (like gemini-1.5-pro) don't support any of them.
        """
        params.pop("reasoning_mode", None)
        params.pop("reasoning_budget", None)
        params.pop("thinking_level", None)
        params.pop("thinking_budget", None)
