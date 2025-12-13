"""Integration utilities for TraiGent framework overrides."""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

from traigent.integrations.utils.message_coercion import (
    coerce_messages,
    coerce_to_anthropic_format,
    coerce_to_gemini_format,
    coerce_to_openai_format,
)
from traigent.integrations.utils.mock_adapter import (
    MockAdapter,
    MockResponse,
    with_mock_support,
)
from traigent.integrations.utils.parameter_normalizer import (
    Framework,
    ParameterAlias,
    ParameterNormalizer,
    get_normalizer,
    normalize_params,
)
from traigent.integrations.utils.response_wrapper import (
    LLMResponse,
    extract_response_metadata,
)

__all__ = [
    # Parameter normalization
    "Framework",
    "ParameterAlias",
    "ParameterNormalizer",
    "get_normalizer",
    "normalize_params",
    # Message coercion
    "coerce_messages",
    "coerce_to_openai_format",
    "coerce_to_anthropic_format",
    "coerce_to_gemini_format",
    # Response wrapper
    "LLMResponse",
    "extract_response_metadata",
    # Mock adapter
    "MockAdapter",
    "MockResponse",
    "with_mock_support",
]
