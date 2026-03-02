"""Base class for LLM provider plugins.

This module provides the LLMPlugin base class that all LLM provider plugins
should inherit from. It provides automatic parameter mapping via ParameterNormalizer
while allowing provider-specific customization through extension hooks.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from traigent.integrations.base_plugin import IntegrationPlugin, ValidationRule
from traigent.integrations.utils import Framework, get_normalizer

if TYPE_CHECKING:
    from traigent.config.types import TraigentConfig

logger = logging.getLogger(__name__)


class LLMPlugin(IntegrationPlugin):
    """Base class for LLM provider plugins with shared parameter handling.

    This class provides automatic parameter mapping via ParameterNormalizer
    while allowing provider-specific customization through extension hooks.

    Extension Hooks (override in subclasses):
        FRAMEWORK: Set to the Framework enum for this provider
        _should_use_normalizer(): Return False to bypass auto-generated mappings
        _get_supported_canonical_params(): Return set of params this provider supports
        _get_extra_mappings(): Add provider-specific params not in normalizer
        _get_provider_specific_rules(): Add custom validation rules

    Example:
        >>> class OpenAIPlugin(LLMPlugin):
        ...     FRAMEWORK = Framework.OPENAI
        ...
        ...     def _get_extra_mappings(self) -> dict[str, str]:
        ...         return {"logit_bias": "logit_bias"}
        ...
        ...     def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        ...         return {"frequency_penalty": ValidationRule(min_value=-2.0, max_value=2.0)}

    Example with limited params (for providers that don't support all params):
        >>> class BedrockPlugin(LLMPlugin):
        ...     FRAMEWORK = Framework.BEDROCK
        ...
        ...     def _get_supported_canonical_params(self) -> set[str]:
        ...         # Only map params that Bedrock actually supports
        ...         return {"model", "max_tokens", "temperature", "top_p", "stop"}
    """

    # Subclasses must define their framework
    FRAMEWORK: Framework | None = None

    def _should_use_normalizer(self) -> bool:
        """Whether to use ParameterNormalizer for mappings.

        Override and return False for plugins with bespoke params
        that shouldn't use one-size-fits-all mappings.

        Returns:
            True to use normalizer (default), False to bypass
        """
        return True

    def _get_supported_canonical_params(self) -> set[str] | None:
        """Return set of canonical params this provider supports.

        Override to limit which params are auto-mapped from the normalizer.
        Return None to support all canonical params (default behavior).

        This is critical for providers that don't support all LLM params.
        For example, Bedrock only supports model, max_tokens, temperature,
        top_p, and stop_sequences - not frequency_penalty, presence_penalty, etc.

        Returns:
            Set of supported canonical param names, or None for all
        """
        return None

    def _get_extra_mappings(self) -> dict[str, str]:
        """Provider-specific mappings not in ParameterNormalizer.

        Override to add params unique to this provider.
        These are merged AFTER normalizer mappings (higher priority).

        Returns:
            Dict mapping canonical names to provider-specific names
        """
        return {}

    def _get_default_mappings(self) -> dict[str, str]:
        """Get parameter mappings with merge strategy.

        Merge order (later overrides earlier):
        1. Normalizer-generated mappings (if enabled, filtered by supported params)
        2. Provider-specific extra mappings

        Returns:
            Complete parameter mappings for this provider
        """
        mappings: dict[str, str] = {}

        # Step 1: Get normalizer mappings (if enabled)
        if self._should_use_normalizer() and self.FRAMEWORK is not None:
            normalizer = get_normalizer()
            supported = self._get_supported_canonical_params()

            for canonical in normalizer.get_canonical_parameters():
                # Filter to supported params if provider specifies them
                if supported is not None and canonical not in supported:
                    continue

                framework_name = normalizer.get_framework_parameter(
                    canonical, self.FRAMEWORK
                )
                if framework_name:
                    mappings[canonical] = framework_name

        # Step 2: Merge extra mappings (provider-specific overrides)
        mappings.update(self._get_extra_mappings())

        return mappings

    def _get_common_validation_rules(self) -> dict[str, ValidationRule]:
        """Common validation rules for all LLM providers.

        These rules apply to all LLM plugins and provide sensible defaults.

        Returns:
            Dict of common validation rules
        """
        return {
            "temperature": ValidationRule(min_value=0.0, max_value=2.0),
            "top_p": ValidationRule(min_value=0.0, max_value=1.0),
            "top_k": ValidationRule(min_value=1),
            "max_tokens": ValidationRule(min_value=1),
        }

    def _get_provider_specific_rules(self) -> dict[str, ValidationRule]:
        """Provider-specific validation rules.

        Override in subclasses to add custom validation.
        These are merged AFTER common rules (higher priority).

        Returns:
            Dict of provider-specific validation rules
        """
        return {}

    def _get_validation_rules(self) -> dict[str, ValidationRule]:
        """Merge common rules with provider-specific rules.

        Returns:
            Complete validation rules for this provider
        """
        rules = self._get_common_validation_rules()
        rules.update(self._get_provider_specific_rules())
        return rules

    def _extract_custom_params(self, config_obj: Any) -> dict[str, Any]:
        """Return custom parameters as a plain dict with observable fallback behavior."""

        custom_params_raw = getattr(config_obj, "custom_params", {}) or {}
        if isinstance(custom_params_raw, Mapping):
            return dict(custom_params_raw)

        try:
            coerced = dict(custom_params_raw)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Ignoring invalid custom_params in %s: expected mapping-compatible "
                "payload, got %s (%s)",
                self.__class__.__name__,
                type(custom_params_raw).__name__,
                exc,
            )
            return {}

        logger.debug(
            "Coerced non-mapping custom_params in %s from %s",
            self.__class__.__name__,
            type(custom_params_raw).__name__,
        )
        return coerced

    def apply_overrides(
        self, kwargs: dict[str, Any], config: TraigentConfig | dict[str, Any]
    ) -> dict[str, Any]:
        """Apply provider-specific parameter overrides.

        This method extends the base implementation to apply LLM-specific
        parameter transformations after standard mapping.

        Args:
            kwargs: Original keyword arguments
            config: Traigent configuration

        Returns:
            Modified kwargs with overrides applied
        """
        # Apply base overrides (handles standard mapping)
        return super().apply_overrides(kwargs, config)
