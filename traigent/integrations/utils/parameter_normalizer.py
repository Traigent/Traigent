"""Parameter normalization utilities for cross-framework compatibility.

This module provides a unified ParameterNormalizer class that handles
automatic conversion of parameter names between different LLM frameworks.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Framework(Enum):
    """Supported LLM frameworks."""

    TRAIGENT = "traigent"  # Canonical Traigent parameter names
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"
    GEMINI = "gemini"
    BEDROCK = "bedrock"
    AZURE_OPENAI = "azure_openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    MISTRAL = "mistral"
    LITELLM = "litellm"
    PYDANTIC_AI = "pydantic_ai"


@dataclass
class ParameterAlias:
    """Defines how a parameter maps across frameworks."""

    canonical: str  # Traigent canonical name
    aliases: dict[Framework, str] = field(default_factory=dict)
    description: str = ""

    def get_for_framework(self, framework: Framework) -> str:
        """Get the parameter name for a specific framework."""
        return self.aliases.get(framework, self.canonical)


# Registry of all parameter aliases
# Format: canonical_name -> ParameterAlias with framework-specific names
_PARAMETER_REGISTRY: dict[str, ParameterAlias] = {}


def _init_registry() -> None:
    """Initialize the parameter alias registry."""
    global _PARAMETER_REGISTRY

    _PARAMETER_REGISTRY = {
        # Model identifier
        "model": ParameterAlias(
            canonical="model",
            aliases={
                Framework.TRAIGENT: "model",
                Framework.OPENAI: "model",
                Framework.ANTHROPIC: "model",
                Framework.LANGCHAIN: "model_name",
                Framework.LLAMAINDEX: "model",
                Framework.GEMINI: "model_name",
                Framework.BEDROCK: "model_id",
                Framework.AZURE_OPENAI: "model",
                Framework.COHERE: "model",
                Framework.HUGGINGFACE: "model_id",
            },
            description="The model identifier to use",
        ),
        # Maximum tokens in response
        "max_tokens": ParameterAlias(
            canonical="max_tokens",
            aliases={
                Framework.TRAIGENT: "max_tokens",
                Framework.OPENAI: "max_tokens",
                Framework.ANTHROPIC: "max_tokens",
                Framework.LANGCHAIN: "max_tokens",
                Framework.LLAMAINDEX: "max_tokens",
                Framework.GEMINI: "max_output_tokens",
                Framework.BEDROCK: "max_tokens",
                Framework.AZURE_OPENAI: "max_tokens",
                Framework.COHERE: "max_tokens",
                Framework.HUGGINGFACE: "max_new_tokens",
                Framework.PYDANTIC_AI: "max_tokens",
            },
            description="Maximum number of tokens to generate",
        ),
        # Temperature
        "temperature": ParameterAlias(
            canonical="temperature",
            aliases={
                Framework.TRAIGENT: "temperature",
                Framework.OPENAI: "temperature",
                Framework.ANTHROPIC: "temperature",
                Framework.LANGCHAIN: "temperature",
                Framework.LLAMAINDEX: "temperature",
                Framework.GEMINI: "temperature",
                Framework.BEDROCK: "temperature",
                Framework.AZURE_OPENAI: "temperature",
                Framework.COHERE: "temperature",
                Framework.HUGGINGFACE: "temperature",
                Framework.PYDANTIC_AI: "temperature",
            },
            description="Sampling temperature (0-2)",
        ),
        # Top-p (nucleus sampling)
        "top_p": ParameterAlias(
            canonical="top_p",
            aliases={
                Framework.TRAIGENT: "top_p",
                Framework.OPENAI: "top_p",
                Framework.ANTHROPIC: "top_p",
                Framework.LANGCHAIN: "top_p",
                Framework.LLAMAINDEX: "top_p",
                Framework.GEMINI: "top_p",
                Framework.BEDROCK: "top_p",
                Framework.AZURE_OPENAI: "top_p",
                Framework.COHERE: "p",
                Framework.HUGGINGFACE: "top_p",
                Framework.PYDANTIC_AI: "top_p",
            },
            description="Top-p (nucleus) sampling threshold",
        ),
        # Top-k sampling
        "top_k": ParameterAlias(
            canonical="top_k",
            aliases={
                Framework.TRAIGENT: "top_k",
                Framework.OPENAI: "top_k",  # Not officially supported
                Framework.ANTHROPIC: "top_k",
                Framework.LANGCHAIN: "top_k",
                Framework.LLAMAINDEX: "top_k",
                Framework.GEMINI: "top_k",
                Framework.BEDROCK: "top_k",
                Framework.AZURE_OPENAI: "top_k",
                Framework.COHERE: "k",
                Framework.HUGGINGFACE: "top_k",
            },
            description="Top-k sampling threshold",
        ),
        # Stop sequences
        "stop": ParameterAlias(
            canonical="stop",
            aliases={
                Framework.TRAIGENT: "stop",
                Framework.OPENAI: "stop",
                Framework.ANTHROPIC: "stop_sequences",
                Framework.LANGCHAIN: "stop",
                Framework.LLAMAINDEX: "stop",
                Framework.GEMINI: "stop_sequences",
                Framework.BEDROCK: "stop_sequences",
                Framework.AZURE_OPENAI: "stop",
                Framework.COHERE: "stop_sequences",
                Framework.HUGGINGFACE: "stop_sequences",
            },
            description="Sequences where the model stops generating",
        ),
        # Streaming
        "stream": ParameterAlias(
            canonical="stream",
            aliases={
                Framework.TRAIGENT: "stream",
                Framework.OPENAI: "stream",
                Framework.ANTHROPIC: "stream",
                Framework.LANGCHAIN: "streaming",
                Framework.LLAMAINDEX: "stream",
                Framework.GEMINI: "stream",
                Framework.BEDROCK: "stream",
                Framework.AZURE_OPENAI: "stream",
                Framework.COHERE: "stream",
                Framework.HUGGINGFACE: "stream",
            },
            description="Whether to stream the response",
        ),
        # System message/instruction
        "system": ParameterAlias(
            canonical="system",
            aliases={
                Framework.TRAIGENT: "system",
                Framework.OPENAI: "system",  # Usually in messages array
                Framework.ANTHROPIC: "system",
                Framework.LANGCHAIN: "system_message",
                Framework.LLAMAINDEX: "system_prompt",
                Framework.GEMINI: "system_instruction",
                Framework.BEDROCK: "system",
                Framework.AZURE_OPENAI: "system",
                Framework.COHERE: "preamble",
                Framework.HUGGINGFACE: "system_prompt",
            },
            description="System message/instruction",
        ),
        # Frequency penalty
        "frequency_penalty": ParameterAlias(
            canonical="frequency_penalty",
            aliases={
                Framework.TRAIGENT: "frequency_penalty",
                Framework.OPENAI: "frequency_penalty",
                Framework.ANTHROPIC: "frequency_penalty",
                Framework.LANGCHAIN: "frequency_penalty",
                Framework.LLAMAINDEX: "frequency_penalty",
                Framework.GEMINI: "frequency_penalty",
                Framework.BEDROCK: "frequency_penalty",
                Framework.AZURE_OPENAI: "frequency_penalty",
                Framework.COHERE: "frequency_penalty",
                Framework.HUGGINGFACE: "repetition_penalty",
            },
            description="Penalty for token frequency",
        ),
        # Presence penalty
        "presence_penalty": ParameterAlias(
            canonical="presence_penalty",
            aliases={
                Framework.TRAIGENT: "presence_penalty",
                Framework.OPENAI: "presence_penalty",
                Framework.ANTHROPIC: "presence_penalty",
                Framework.LANGCHAIN: "presence_penalty",
                Framework.LLAMAINDEX: "presence_penalty",
                Framework.GEMINI: "presence_penalty",
                Framework.BEDROCK: "presence_penalty",
                Framework.AZURE_OPENAI: "presence_penalty",
                Framework.COHERE: "presence_penalty",
                Framework.HUGGINGFACE: "presence_penalty",
            },
            description="Penalty for token presence",
        ),
        # Tools/Functions
        "tools": ParameterAlias(
            canonical="tools",
            aliases={
                Framework.TRAIGENT: "tools",
                Framework.OPENAI: "tools",
                Framework.ANTHROPIC: "tools",
                Framework.LANGCHAIN: "tools",
                Framework.LLAMAINDEX: "tools",
                Framework.GEMINI: "tools",
                Framework.BEDROCK: "tools",
                Framework.AZURE_OPENAI: "tools",
                Framework.COHERE: "tools",
                Framework.HUGGINGFACE: "tools",
            },
            description="Tool/function definitions",
        ),
        # Tool choice
        "tool_choice": ParameterAlias(
            canonical="tool_choice",
            aliases={
                Framework.TRAIGENT: "tool_choice",
                Framework.OPENAI: "tool_choice",
                Framework.ANTHROPIC: "tool_choice",
                Framework.LANGCHAIN: "tool_choice",
                Framework.LLAMAINDEX: "tool_choice",
                Framework.GEMINI: "tool_config",
                Framework.BEDROCK: "tool_choice",
                Framework.AZURE_OPENAI: "tool_choice",
                Framework.COHERE: "tool_choice",
                Framework.HUGGINGFACE: "tool_choice",
            },
            description="Control which tool to use",
        ),
        # Response format (JSON mode)
        "response_format": ParameterAlias(
            canonical="response_format",
            aliases={
                Framework.TRAIGENT: "response_format",
                Framework.OPENAI: "response_format",
                Framework.ANTHROPIC: "response_format",
                Framework.LANGCHAIN: "response_format",
                Framework.LLAMAINDEX: "response_format",
                Framework.GEMINI: "generation_config",
                Framework.BEDROCK: "response_format",
                Framework.AZURE_OPENAI: "response_format",
                Framework.COHERE: "response_format",
                Framework.HUGGINGFACE: "response_format",
            },
            description="Response format (e.g., JSON mode)",
        ),
        # Seed for reproducibility
        "seed": ParameterAlias(
            canonical="seed",
            aliases={
                Framework.TRAIGENT: "seed",
                Framework.OPENAI: "seed",
                Framework.ANTHROPIC: "seed",
                Framework.LANGCHAIN: "seed",
                Framework.LLAMAINDEX: "seed",
                Framework.GEMINI: "seed",
                Framework.BEDROCK: "seed",
                Framework.AZURE_OPENAI: "seed",
                Framework.COHERE: "seed",
                Framework.HUGGINGFACE: "seed",
            },
            description="Random seed for reproducibility",
        ),
    }


# Initialize registry on module load
_init_registry()


class ParameterNormalizer:
    """Normalizes parameters across different LLM frameworks.

    This class provides methods to convert parameter names between different
    frameworks, enabling seamless cross-framework compatibility.

    Example usage:
        >>> normalizer = ParameterNormalizer()
        >>> # Convert from LangChain format to OpenAI format
        >>> params = {"model_name": "gpt-4", "streaming": True, "max_tokens": 100}
        >>> normalized = normalizer.convert(params, from_framework=Framework.LANGCHAIN, to_framework=Framework.OPENAI)
        >>> # Result: {"model": "gpt-4", "stream": True, "max_tokens": 100}

        >>> # Or convert to canonical Traigent format first
        >>> canonical = normalizer.to_canonical(params, from_framework=Framework.LANGCHAIN)
        >>> # Result: {"model": "gpt-4", "stream": True, "max_tokens": 100}
    """

    def __init__(self) -> None:
        """Initialize the normalizer with the global parameter registry."""
        self._registry = _PARAMETER_REGISTRY
        self._reverse_mappings: dict[Framework, dict[str, str]] = {}
        self._build_reverse_mappings()

    def _build_reverse_mappings(self) -> None:
        """Build reverse mappings (framework-specific -> canonical) for each framework."""
        for canonical_name, alias in self._registry.items():
            for framework, framework_name in alias.aliases.items():
                if framework not in self._reverse_mappings:
                    self._reverse_mappings[framework] = {}
                # Map framework-specific name back to canonical
                self._reverse_mappings[framework][framework_name] = canonical_name

    def to_canonical(
        self,
        params: Mapping[str, Any],
        from_framework: Framework,
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Convert parameters from a specific framework to canonical Traigent format.

        Args:
            params: Parameters in framework-specific format
            from_framework: The source framework
            strict: If True, raise error for unknown parameters. If False, pass through.

        Returns:
            Parameters with canonical Traigent names

        Raises:
            ValueError: If strict=True and unknown parameters are encountered
        """
        result: dict[str, Any] = {}
        reverse_map = self._reverse_mappings.get(from_framework, {})

        for key, value in params.items():
            if key in reverse_map:
                canonical_key = reverse_map[key]
                result[canonical_key] = value
            elif strict:
                raise ValueError(
                    f"Unknown parameter '{key}' for framework {from_framework.value}"
                )
            else:
                # Pass through unknown parameters
                result[key] = value

        return result

    def from_canonical(
        self,
        params: Mapping[str, Any],
        to_framework: Framework,
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Convert parameters from canonical Traigent format to a specific framework.

        Args:
            params: Parameters in canonical Traigent format
            to_framework: The target framework
            strict: If True, raise error for unknown parameters. If False, pass through.

        Returns:
            Parameters with framework-specific names

        Raises:
            ValueError: If strict=True and unknown parameters are encountered
        """
        result: dict[str, Any] = {}

        for key, value in params.items():
            if key in self._registry:
                framework_key = self._registry[key].get_for_framework(to_framework)
                result[framework_key] = value
            elif strict:
                raise ValueError(f"Unknown canonical parameter '{key}'")
            else:
                # Pass through unknown parameters
                result[key] = value

        return result

    def convert(
        self,
        params: Mapping[str, Any],
        from_framework: Framework,
        to_framework: Framework,
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Convert parameters directly between two frameworks.

        This is a convenience method that chains to_canonical and from_canonical.

        Args:
            params: Parameters in source framework format
            from_framework: The source framework
            to_framework: The target framework
            strict: If True, raise error for unknown parameters

        Returns:
            Parameters with target framework names
        """
        canonical = self.to_canonical(params, from_framework, strict=strict)
        return self.from_canonical(canonical, to_framework, strict=strict)

    def normalize_kwargs(
        self,
        kwargs: dict[str, Any],
        target_framework: Framework,
    ) -> dict[str, Any]:
        """Normalize kwargs by detecting and converting parameter names.

        This method attempts to detect the source framework from parameter names
        and convert them to the target framework format.

        Args:
            kwargs: Keyword arguments with potentially mixed parameter names
            target_framework: The desired target framework

        Returns:
            Kwargs with parameter names normalized for the target framework
        """
        result: dict[str, Any] = {}

        for key, value in kwargs.items():
            # Check if this is a framework-specific name we can normalize
            canonical_key = self._find_canonical_key(key)
            if canonical_key:
                # Convert to target framework format
                target_key = self._registry[canonical_key].get_for_framework(
                    target_framework
                )
                result[target_key] = value
            else:
                # Pass through unknown parameters
                result[key] = value

        return result

    def _find_canonical_key(self, param_name: str) -> str | None:
        """Find the canonical key for any parameter name variant.

        Args:
            param_name: A parameter name from any framework

        Returns:
            The canonical Traigent parameter name, or None if not found
        """
        # Check if it's already a canonical name
        if param_name in self._registry:
            return param_name

        # Search through all framework mappings
        for canonical_name, alias in self._registry.items():
            for framework_name in alias.aliases.values():
                if framework_name == param_name:
                    return canonical_name

        return None

    def get_all_aliases(self, canonical_name: str) -> dict[Framework, str]:
        """Get all framework-specific names for a canonical parameter.

        Args:
            canonical_name: The canonical Traigent parameter name

        Returns:
            Dict mapping frameworks to their specific parameter names
        """
        if canonical_name in self._registry:
            return self._registry[canonical_name].aliases.copy()
        return {}

    def get_canonical_names(self) -> list[str]:
        """Get list of all canonical parameter names.

        Returns:
            List of canonical Traigent parameter names
        """
        return list(self._registry.keys())

    # Alias for consistency with LLMPlugin API
    get_canonical_parameters = get_canonical_names

    def get_framework_parameter(
        self,
        canonical: str,
        framework: Framework,
    ) -> str | None:
        """Get the framework-specific parameter name for a canonical parameter.

        Args:
            canonical: The canonical Traigent parameter name
            framework: The target framework

        Returns:
            The framework-specific parameter name, or None if not found

        Example:
            >>> normalizer = get_normalizer()
            >>> normalizer.get_framework_parameter("max_tokens", Framework.HUGGINGFACE)
            'max_new_tokens'
            >>> normalizer.get_framework_parameter("stream", Framework.LANGCHAIN)
            'streaming'
        """
        alias = self._registry.get(canonical)
        if alias is None:
            return None
        return alias.get_for_framework(framework)

    def is_known_parameter(self, param_name: str) -> bool:
        """Check if a parameter name is known (canonical or alias).

        Args:
            param_name: Parameter name to check

        Returns:
            True if the parameter is known, False otherwise
        """
        return self._find_canonical_key(param_name) is not None

    @staticmethod
    def get_framework_from_string(framework_str: str) -> Framework | None:
        """Convert a framework string to Framework enum.

        Args:
            framework_str: Framework name as string (case-insensitive)

        Returns:
            Framework enum value, or None if not found
        """
        framework_str = framework_str.lower().replace("-", "_").replace(" ", "_")
        for framework in Framework:
            if framework.value == framework_str:
                return framework
        return None


# Singleton instance for convenience
_normalizer: ParameterNormalizer | None = None


def get_normalizer() -> ParameterNormalizer:
    """Get the singleton ParameterNormalizer instance.

    Returns:
        The global ParameterNormalizer instance
    """
    global _normalizer
    if _normalizer is None:
        _normalizer = ParameterNormalizer()
    return _normalizer


def normalize_params(
    params: Mapping[str, Any],
    from_framework: Framework | str,
    to_framework: Framework | str,
) -> dict[str, Any]:
    """Convenience function to normalize parameters between frameworks.

    Args:
        params: Parameters to normalize
        from_framework: Source framework (enum or string)
        to_framework: Target framework (enum or string)

    Returns:
        Normalized parameters for target framework
    """
    normalizer = get_normalizer()

    # Convert string to Framework enum if needed
    if isinstance(from_framework, str):
        from_fw = ParameterNormalizer.get_framework_from_string(from_framework)
        if from_fw is None:
            raise ValueError(f"Unknown framework: {from_framework}")
        from_framework = from_fw

    if isinstance(to_framework, str):
        to_fw = ParameterNormalizer.get_framework_from_string(to_framework)
        if to_fw is None:
            raise ValueError(f"Unknown framework: {to_framework}")
        to_framework = to_fw

    return normalizer.convert(params, from_framework, to_framework)
