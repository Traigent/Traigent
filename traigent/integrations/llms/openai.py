"""OpenAI SDK Integration for Traigent.

This module provides seamless integration with the OpenAI Python SDK,
enabling zero-code-change optimization of OpenAI applications through
automatic parameter override.

Key Features:
- Automatic parameter injection for OpenAI SDK calls
- Support for both sync and async OpenAI clients
- Streaming completions support
- Function/tool calling support
- Chat and completion APIs
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import warnings
from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import Any

from ...utils.logging import get_logger
from ..framework_override import (
    enable_framework_overrides,
    override_context,
    register_framework_mapping,
)

logger = get_logger(__name__)


class OpenAIIntegration:
    """Enhanced OpenAI SDK integration with Traigent optimization."""

    def __init__(self) -> None:
        """Initialize OpenAI SDK integration."""
        self.supported_clients: dict[str, Any] = {
            "openai.OpenAI": {
                "model": "model",
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "frequency_penalty": "frequency_penalty",
                "presence_penalty": "presence_penalty",
                "stop": "stop",
                "stream": "stream",
                "tools": "tools",
                "tool_choice": "tool_choice",
            },
            "openai.AsyncOpenAI": {
                "model": "model",
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "top_p": "top_p",
                "frequency_penalty": "frequency_penalty",
                "presence_penalty": "presence_penalty",
                "stop": "stop",
                "stream": "stream",
                "tools": "tools",
                "tool_choice": "tool_choice",
            },
        }
        self._register_mappings()

    def _register_mappings(self) -> None:
        """Register OpenAI SDK parameter mappings."""
        for client_class, mappings in self.supported_clients.items():
            register_framework_mapping(client_class, mappings)

    def _normalize_client_types(self, client_types: Iterable[str] | None) -> list[str]:
        """Validate and normalize requested client types."""

        if client_types is None:
            return list(self.supported_clients.keys())

        if isinstance(client_types, (str, bytes)) or not isinstance(
            client_types, Iterable
        ):
            raise TypeError("client_types must be an iterable of client type strings")

        normalized: list[str] = []
        for entry in client_types:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("client_types entries must be non-empty strings")
            if entry not in self.supported_clients:
                raise ValueError(
                    f"Unsupported OpenAI client type '{entry}'. Supported types: {sorted(self.supported_clients.keys())}"
                )
            if entry not in normalized:
                normalized.append(entry)

        return normalized

    def enable_openai_overrides(self, client_types: list[str] | None = None) -> None:
        """Enable OpenAI SDK parameter overrides.

        Args:
            client_types: Optional list of specific client types to override.
                         If None, all supported clients will be overridden.
        """
        normalized = self._normalize_client_types(client_types)

        enable_framework_overrides(normalized)
        logger.info(f"OpenAI SDK overrides enabled for: {', '.join(normalized)}")

    def get_supported_clients(self) -> list[str]:
        """Get list of supported OpenAI client types."""
        return list(self.supported_clients.keys())


# Global OpenAI integration instance
_openai_integration = OpenAIIntegration()


def enable_openai_optimization(client_types: list[str] | None = None) -> None:
    """Enable Traigent optimization for OpenAI SDK applications.

    This function enables automatic parameter override for OpenAI SDK clients,
    allowing seamless optimization without code changes.

    Args:
        client_types: Optional list of specific client types to override.
                     If None, all supported clients will be overridden.

    Example:
        ```python
        import traigent
        from traigent.integrations.openai_sdk import enable_openai_optimization
        import openai

        # Enable OpenAI optimization
        enable_openai_optimization()

        # Your existing OpenAI code works unchanged
        @traigent.optimize()
        def my_openai_app():
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello!"}]
            )
            return response.choices[0].message.content

        # Traigent will automatically test different models/parameters
        result = my_openai_app()
        ```
    """
    _openai_integration.enable_openai_overrides(client_types)


def get_supported_openai_clients() -> list[str]:
    """Get list of supported OpenAI client types.

    Returns:
        List of supported OpenAI client class names
    """
    return _openai_integration.get_supported_clients()


def enable_sync_openai() -> None:
    """Enable optimization for synchronous OpenAI client only."""
    enable_openai_optimization(["openai.OpenAI"])


def enable_async_openai() -> None:
    """Enable optimization for asynchronous OpenAI client only."""
    enable_openai_optimization(["openai.AsyncOpenAI"])


def openai_context(
    client_types: list[str] | None = None,
) -> AbstractContextManager[None]:
    """Context manager for temporary OpenAI optimization.

    Args:
        client_types: Optional list of client types to override.
                     If None, all supported clients will be overridden.

    Returns:
        Context manager for temporary OpenAI overrides

    Example:
        ```python
        from traigent.integrations.openai_sdk import openai_context
        import openai

        # Only override within this context
        with openai_context():
            client = openai.OpenAI()
            # This call will have parameters automatically overridden
            response = client.chat.completions.create(...)

        # Outside context, no overrides are applied
        client = openai.OpenAI()
        response = client.chat.completions.create(...)  # Normal behavior
        ```
    """
    normalized = _openai_integration._normalize_client_types(client_types)
    return override_context(normalized)


# Auto-detection helper
def auto_detect_openai() -> None:
    """Auto-detect and enable optimization for available OpenAI SDK.

    This function attempts to import the OpenAI SDK and enables optimization
    if it's available in the current environment.
    """
    try:
        import openai

        # Check which client types are available
        available_clients = []

        if hasattr(openai, "OpenAI"):
            available_clients.append("openai.OpenAI")

        if hasattr(openai, "AsyncOpenAI"):
            available_clients.append("openai.AsyncOpenAI")

        if available_clients:
            enable_openai_optimization(available_clients)
            logger.info(
                f"Auto-detected OpenAI SDK with {len(available_clients)} client types"
            )
        else:
            warnings.warn(
                "OpenAI SDK detected but no supported client types found. "
                "Make sure you have openai>=1.0.0 installed.",
                UserWarning,
                stacklevel=2,
            )

    except ImportError:
        warnings.warn(
            "OpenAI SDK not detected. Install with: pip install openai",
            UserWarning,
            stacklevel=2,
        )


# Streaming helpers
def enable_streaming_optimization() -> None:
    """Enable optimization specifically for streaming completions.

    This helper automatically sets stream=True in the Traigent configuration
    when optimizing OpenAI streaming calls.
    """
    from ..config.context import set_config
    from ..config.types import TraigentConfig

    # Enable OpenAI overrides
    enable_openai_optimization()

    # Set streaming in config
    config = TraigentConfig(custom_params={"stream": True})
    set_config(config)

    logger.info("OpenAI streaming optimization enabled")


# Tool/Function calling helpers
def enable_tools_optimization(tools: list[dict[str, Any]] | None = None) -> None:
    """Enable optimization for function/tool calling.

    Args:
        tools: Optional list of tool definitions to include in optimization

    Example:
        ```python
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]

        enable_tools_optimization(tools)
        ```
    """
    from ..config.context import set_config
    from ..config.types import TraigentConfig

    # Enable OpenAI overrides
    enable_openai_optimization()

    # Set tools in config
    config_params = {}
    if tools:
        config_params["tools"] = tools

    config = TraigentConfig(custom_params=config_params)
    set_config(config)

    logger.info("OpenAI tools/function calling optimization enabled")
