"""PydanticAI integration plugin for Traigent.

This module provides the PydanticAI-specific plugin implementation for
parameter mappings and framework overrides.

PydanticAI is an agent-level framework — parameters like temperature and
max_tokens are passed inside a ``model_settings`` dict, not as top-level
kwargs on ``Agent.run()``.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

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


class PydanticAIPlugin(LLMPlugin):
    """Plugin for PydanticAI agent framework integration.

    PydanticAI agents accept model parameters via ``model_settings``
    (a TypedDict), not as top-level kwargs.  The plugin maps Traigent's
    canonical parameters to ModelSettings keys and wraps them accordingly
    in :meth:`apply_overrides`.
    """

    FRAMEWORK = Framework.PYDANTIC_AI

    def _get_metadata(self) -> PluginMetadata:
        """Return PydanticAI plugin metadata."""
        return PluginMetadata(
            name="pydantic_ai",
            version="1.0.0",
            supported_packages=["pydantic_ai"],
            priority=IntegrationPriority.NORMAL,
            description="PydanticAI agent framework integration",
            author="Traigent Team",
            requires_packages=["pydantic-ai>=1,<2"],
            supports_versions={"pydantic_ai": "1."},
        )

    def _get_supported_canonical_params(self) -> set[str]:
        """Return canonical params PydanticAI ModelSettings supports."""
        return {"max_tokens", "temperature", "top_p"}

    def _get_extra_mappings(self) -> dict[str, str]:
        """Return PydanticAI-specific parameter mappings."""
        return {
            "parallel_tool_calls": "parallel_tool_calls",
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Return PydanticAI-specific validation rules."""
        return {
            "temperature": ValidationRule(min_value=0.0, max_value=2.0),
            "max_tokens": ValidationRule(min_value=1, max_value=128000),
            "top_p": ValidationRule(min_value=0.0, max_value=1.0),
        }

    def get_target_classes(self) -> list[str]:
        """Return list of PydanticAI classes to override."""
        return ["pydantic_ai.Agent"]

    def get_target_methods(self) -> dict[str, list[str]]:
        """Return mapping of PydanticAI classes to methods to override."""
        return {
            "pydantic_ai.Agent": [
                "run",
                "run_sync",
                "run_stream",
                "run_stream_sync",
            ],
        }

    def apply_overrides(self, kwargs: dict[str, Any], config: Any) -> dict[str, Any]:
        """Apply PydanticAI-specific overrides.

        Unlike other plugins that inject params as top-level kwargs,
        PydanticAI requires params to be nested inside ``model_settings``.
        """
        if config is None or not self._enabled:
            return kwargs.copy()

        config_obj = self._normalize_config(config)

        # Get the parameter mapping (canonical -> framework name)
        mappings = self.get_parameter_mappings()

        # Extract param values from config
        model_settings_overrides: dict[str, Any] = {}
        for canonical, framework_name in mappings.items():
            value = getattr(config_obj, canonical, None)
            if value is None and hasattr(config_obj, "custom_params"):
                custom = getattr(config_obj, "custom_params", None) or {}
                value = custom.get(canonical)
            if value is not None:
                model_settings_overrides[framework_name] = value

        if not model_settings_overrides:
            return kwargs.copy()

        # Merge into existing model_settings (user values take precedence)
        result = kwargs.copy()
        existing_settings = result.get("model_settings") or {}
        if not isinstance(existing_settings, dict):
            existing_settings = {}

        merged = {**model_settings_overrides, **existing_settings}
        result["model_settings"] = merged

        return result
