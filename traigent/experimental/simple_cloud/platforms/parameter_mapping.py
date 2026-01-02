"""Unified parameter mapping system for platform executors.

This module provides a consistent parameter language across all platforms,
with mappings to platform-specific parameter names.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-AGENTS FUNC-INTEGRATIONS REQ-AGNT-013 REQ-INT-008 SYNC-OptimizationFlow

from __future__ import annotations

from typing import Any

# Universal parameter names and descriptions (Traigent's unified language)
UNIFIED_PARAMS = {
    # Core generation parameters
    "model": "The model identifier",
    "temperature": "Randomness control (0.0-2.0, higher = more random)",
    "max_tokens": "Maximum output length in tokens",
    "top_p": "Nucleus sampling threshold (0.0-1.0)",
    "top_k": "Top-k sampling (limits vocabulary)",
    "frequency_penalty": "Repetition penalty for tokens (-2.0 to 2.0)",
    "presence_penalty": "Penalty for new topics (-2.0 to 2.0)",
    "stop_sequences": "List of sequences to stop generation",
    "seed": "Random seed for reproducibility",
    # Advanced parameters
    "system_prompt": "System message to set context",
    "response_format": "Output format (text/json)",
    "timeout": "Request timeout in seconds",
    "stream": "Whether to stream the response",
    # Safety and moderation
    "safety_mode": "Content filtering level",
    "max_retries": "Maximum retry attempts",
}

# Platform-specific parameter mappings
PLATFORM_MAPPINGS: dict[str, dict[str, str | list[str]]] = {
    "anthropic": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens_to_sample",
        "top_p": "top_p",
        "top_k": "top_k",
        # Anthropic doesn't have frequency/presence penalties
        "stop_sequences": "stop_sequences",
        "system_prompt": "system",
        "stream": "stream",
        # Anthropic-specific parameters that don't map
        "_specific": ["metadata", "stop_reason", "user_id"],
    },
    "cohere": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "p",
        "top_k": "k",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "stop_sequences": "stop_sequences",
        "seed": "seed",
        "stream": "stream",
        # Cohere-specific parameters
        "_specific": [
            "connectors",
            "citation_quality",
            "prompt_truncation",
            "return_likelihoods",
        ],
    },
    "huggingface": {
        "model": "model_id",
        "temperature": "temperature",
        "max_tokens": "max_new_tokens",
        "top_p": "top_p",
        "top_k": "top_k",
        "stop_sequences": "stop",
        "seed": "seed",
        # HuggingFace-specific parameters
        "_specific": [
            "do_sample",
            "num_beams",
            "early_stopping",
            "repetition_penalty",
            "length_penalty",
            "no_repeat_ngram_size",
            "use_cache",
        ],
    },
    "openai": {
        "model": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "stop_sequences": "stop",
        "seed": "seed",
        "system_prompt": "system",
        "response_format": "response_format",
        "stream": "stream",
        # OpenAI-specific parameters
        "_specific": ["logit_bias", "user", "tools", "tool_choice", "n"],
    },
    "langchain": {
        "model": "model_name",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "top_p": "top_p",
        "top_k": "top_k",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "stop_sequences": "stop",
        # LangChain-specific
        "_specific": ["callbacks", "tags", "metadata", "verbose"],
    },
}


class ParameterMapper:
    """Handles parameter translation between unified and platform-specific formats."""

    def __init__(self, platform: str) -> None:
        """Initialize mapper for a specific platform.

        Args:
            platform: Platform name (anthropic, cohere, huggingface, etc.)
        """
        self.platform = platform.lower()
        self.mapping = PLATFORM_MAPPINGS.get(self.platform, {})
        self.specific_params: set[str] = set(self.mapping.get("_specific", []))

    def to_platform_params(
        self,
        unified_params: dict[str, Any],
        platform_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert unified parameters to platform-specific format.

        Args:
            unified_params: Parameters in Traigent's unified format
            platform_kwargs: Additional platform-specific parameters

        Returns:
            Platform-specific parameter dictionary
        """
        platform_params = {}

        # Translate unified parameters
        for unified_key, value in unified_params.items():
            if (
                value is not None
                and unified_key in self.mapping
                and unified_key != "_specific"
            ):
                platform_key = self.mapping[unified_key]
                if platform_key and isinstance(
                    platform_key, str
                ):  # Only add if mapping exists and is string
                    platform_params[platform_key] = value

        # Add platform-specific kwargs
        if platform_kwargs:
            # Only include recognized platform-specific parameters
            for key, value in platform_kwargs.items():
                if key in self.specific_params or key in platform_params:
                    platform_params[key] = value
                # Also allow parameters that start with platform name
                elif key.startswith(f"{self.platform}_"):
                    clean_key = key[len(self.platform) + 1 :]
                    platform_params[clean_key] = value

        return platform_params

    def from_platform_params(self, platform_params: dict[str, Any]) -> dict[str, Any]:
        """Convert platform-specific parameters back to unified format.

        Args:
            platform_params: Platform-specific parameters

        Returns:
            Parameters in unified format (with platform-specific params preserved)
        """
        unified_params = {}
        platform_specific = {}

        # Reverse mapping
        reverse_mapping = {v: k for k, v in self.mapping.items() if k != "_specific"}

        for platform_key, value in platform_params.items():
            if platform_key in reverse_mapping:
                unified_key = reverse_mapping[platform_key]
                unified_params[unified_key] = value
            else:
                # Preserve platform-specific parameters
                platform_specific[platform_key] = value

        # Add platform-specific params with prefix
        if platform_specific:
            unified_params[f"{self.platform}_kwargs"] = platform_specific

        return unified_params

    def get_supported_params(self) -> set[str]:
        """Get set of all parameters supported by this platform.

        Returns:
            Set of parameter names (both unified and platform-specific)
        """
        supported = set()

        # Add mapped parameters
        for unified_key, platform_key in self.mapping.items():
            if unified_key != "_specific" and platform_key:
                supported.add(unified_key)

        # Add platform-specific parameters
        supported.update(self.specific_params)

        return supported

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters for this platform.

        Args:
            params: Parameters to validate

        Returns:
            Validated parameters (with unsupported ones removed)
        """
        supported = self.get_supported_params()
        validated = {}

        for key, value in params.items():
            if key in supported:
                validated[key] = value
            elif key.endswith("_kwargs") and key.startswith(self.platform):
                # Platform-specific kwargs bundle
                validated[key] = value

        return validated


def get_mapper(platform: str) -> ParameterMapper:
    """Get a parameter mapper for the specified platform.

    Args:
        platform: Platform name

    Returns:
        ParameterMapper instance
    """
    return ParameterMapper(platform)
