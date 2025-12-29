"""LangChain Integration for Traigent.

This module provides seamless integration with LangChain, enabling zero-code-change
optimization of LangChain applications through automatic parameter override.

Key Features:
- Automatic detection and override of LangChain LLM parameters
- Support for ChatOpenAI, ChatAnthropic, and other LangChain models
- Streaming support for LangChain applications
- Tool/function calling integration
- Chain and agent parameter injection
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import warnings
from typing import Any

from ..utils.logging import get_logger
from .framework_override import (
    enable_framework_overrides,
    register_framework_mapping,
)

logger = get_logger(__name__)


class LangChainIntegration:
    """Enhanced LangChain integration with Traigent optimization."""

    def __init__(self) -> None:
        """Initialize LangChain integration."""
        self.supported_llms: dict[str, Any] = {
            "langchain_openai.ChatOpenAI": {
                "model": "model",
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "frequency_penalty": "frequency_penalty",
                "presence_penalty": "presence_penalty",
                "streaming": "streaming",
                "stream": "streaming",
            },
            "langchain_openai.OpenAI": {
                "model": "model",
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "frequency_penalty": "frequency_penalty",
                "presence_penalty": "presence_penalty",
            },
            "langchain_anthropic.ChatAnthropic": {
                "model": "model",
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "top_k": "top_k",
                "stop_sequences": "stop",
                "streaming": "streaming",
                "stream": "streaming",
            },
            "langchain.llms.OpenAI": {
                "model": "model_name",
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "frequency_penalty": "frequency_penalty",
                "presence_penalty": "presence_penalty",
            },
            "langchain.llms.Anthropic": {
                "model": "model",
                "temperature": "temperature",
                "max_tokens": "max_tokens_to_sample",
                "top_p": "top_p",
                "top_k": "top_k",
                "stop_sequences": "stop_sequences",
            },
        }
        self._register_mappings()

    def _register_mappings(self) -> None:
        """Register LangChain parameter mappings."""
        for llm_class, mappings in self.supported_llms.items():
            register_framework_mapping(llm_class, mappings)

    def enable_langchain_overrides(self, llm_types: list[str] | None = None) -> None:
        """Enable LangChain parameter overrides.

        Args:
            llm_types: Optional list of specific LLM types to override.
                      If None, all supported LLMs will be overridden.
        """
        if llm_types is None:
            llm_types = list(self.supported_llms.keys())

        enable_framework_overrides(llm_types)
        logger.info(f"LangChain overrides enabled for: {', '.join(llm_types)}")

    def get_supported_llms(self) -> list[str]:
        """Get list of supported LangChain LLM types."""
        return list(self.supported_llms.keys())

    def add_custom_llm_mapping(
        self, llm_class: str, parameter_mapping: dict[str, str]
    ) -> None:
        """Add custom parameter mapping for a LangChain LLM.

        Args:
            llm_class: Full class name (e.g., "custom_package.CustomLLM")
            parameter_mapping: Mapping from Traigent params to LLM params
        """
        self.supported_llms[llm_class] = parameter_mapping
        register_framework_mapping(llm_class, parameter_mapping)
        logger.info(f"Added custom LangChain LLM mapping: {llm_class}")


# Global LangChain integration instance
_langchain_integration = LangChainIntegration()


def enable_langchain_optimization(llm_types: list[str] | None = None) -> None:
    """Enable Traigent optimization for LangChain applications.

    This function enables automatic parameter override for LangChain LLMs,
    allowing seamless optimization without code changes.

    Args:
        llm_types: Optional list of specific LLM types to override.
                  If None, all supported LLMs will be overridden.

    Example:
        ```python
        import traigent
        from traigent.integrations.langchain import enable_langchain_optimization

        # Enable LangChain optimization
        enable_langchain_optimization()

        # Your existing LangChain code works unchanged
        @traigent.optimize()
        def my_langchain_app():
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            return llm.invoke("Hello world")

        # Traigent will automatically test different models/parameters
        result = my_langchain_app()
        ```
    """
    _langchain_integration.enable_langchain_overrides(llm_types)


def get_supported_langchain_llms() -> list[str]:
    """Get list of supported LangChain LLM types.

    Returns:
        List of supported LangChain LLM class names
    """
    return _langchain_integration.get_supported_llms()


def add_langchain_llm_mapping(
    llm_class: str, parameter_mapping: dict[str, str]
) -> None:
    """Add custom parameter mapping for a LangChain LLM.

    Args:
        llm_class: Full class name (e.g., "custom_package.CustomLLM")
        parameter_mapping: Mapping from Traigent params to LLM params

    Example:
        ```python
        add_langchain_llm_mapping(
            "my_package.CustomLLM",
            {
                "model": "model_name",
                "temperature": "temp",
                "max_tokens": "max_length",
            }
        )
        ```
    """
    _langchain_integration.add_custom_llm_mapping(llm_class, parameter_mapping)


# Convenience functions for specific LLM types
def enable_chatgpt_optimization() -> None:
    """Enable optimization for ChatGPT/OpenAI LangChain models."""
    enable_langchain_optimization(
        [
            "langchain_openai.ChatOpenAI",
            "langchain_openai.OpenAI",
            "langchain.llms.OpenAI",
        ]
    )


def enable_claude_optimization() -> None:
    """Enable optimization for Claude/Anthropic LangChain models."""
    enable_langchain_optimization(
        [
            "langchain_anthropic.ChatAnthropic",
            "langchain.llms.Anthropic",
        ]
    )


def enable_openai_langchain() -> None:
    """Alias for enable_chatgpt_optimization()."""
    enable_chatgpt_optimization()


def enable_anthropic_langchain() -> None:
    """Alias for enable_claude_optimization()."""
    enable_claude_optimization()


# Auto-detection helper
def auto_detect_langchain_llms() -> None:
    """Auto-detect and enable optimization for available LangChain LLMs.

    This function attempts to import and detect which LangChain LLMs are
    available in the current environment, then enables optimization for them.
    """
    available_llms = []

    # Test OpenAI LangChain
    try:
        import langchain_openai  # noqa: F401

        available_llms.extend(
            [
                "langchain_openai.ChatOpenAI",
                "langchain_openai.OpenAI",
            ]
        )
    except ImportError:
        pass

    # Test legacy LangChain OpenAI
    try:
        import langchain.llms

        available_llms.append("langchain.llms.OpenAI")
    except ImportError:
        pass

    # Test Anthropic LangChain
    try:
        import langchain_anthropic  # noqa: F401

        available_llms.append("langchain_anthropic.ChatAnthropic")
    except ImportError:
        pass

    # Test legacy LangChain Anthropic
    try:
        import langchain.llms  # noqa: F401, F811

        available_llms.append("langchain.llms.Anthropic")
    except ImportError:
        pass

    if available_llms:
        enable_langchain_optimization(available_llms)
        logger.info(f"Auto-detected {len(available_llms)} LangChain LLMs")
    else:
        warnings.warn(
            "No LangChain LLMs detected. Install langchain-openai, "
            "langchain-anthropic, or other LangChain packages.",
            UserWarning,
            stacklevel=2,
        )
