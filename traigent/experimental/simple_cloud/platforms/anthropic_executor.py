"""⚠️  EXPERIMENTAL: Naive Anthropic executor - NOT FOR PRODUCTION.

🚨 WARNING: This is a simplified, experimental implementation for local testing.
This is NOT the real TraiGent cloud implementation and does NOT represent
TraiGent's proprietary IP.

This module provides a basic wrapper around Anthropic's API for testing
framework override functionality and parameter mapping while the OptiGen
backend is under development.

For production use:
- Use @traigent.optimize(auto_override_frameworks=True)
- Use real TraiGent cloud services (via OptiGen backend)
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-AGENTS FUNC-INTEGRATIONS REQ-AGNT-013 REQ-INT-008 SYNC-OptimizationFlow

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any, cast

try:
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from traigent.agents.platforms.base_platform import BasePlatformExecutor
from traigent.agents.platforms.parameter_mapping import ParameterMapper
from traigent.utils.exceptions import (
    AgentExecutionError,
    ConfigurationError,
)
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


class AnthropicAgentExecutor(BasePlatformExecutor):
    """⚠️  EXPERIMENTAL: Naive Anthropic executor for local testing only.

    🚨 WARNING: This is NOT the real TraiGent cloud implementation!
    This is a simple wrapper for development/testing purposes.
    """

    # Supported Claude models
    SUPPORTED_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-5-haiku-latest",
        # Aliases
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
    ]

    # Model pricing (per 1K tokens)
    MODEL_COSTS = {
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(self, api_key: str | None = None, **kwargs) -> None:
        """Initialize Anthropic executor.

        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        # Get API key from environment (secure approach)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "Anthropic API key not found. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter. "
                "Never hardcode API keys in source code."
            )

        # Initialize client
        self.client = AsyncAnthropic(api_key=self.api_key)

        # Security: Never log actual API key
        masked_key = f"sk-...{self.api_key[-4:]}" if len(self.api_key) > 4 else "***"
        logger.info(f"Initialized AnthropicAgentExecutor with API key: {masked_key}")
        self.mapper = ParameterMapper("anthropic")

        logger.info("Initialized Anthropic executor")

    def __repr__(self) -> str:
        """Secure string representation that masks API key."""
        return "AnthropicAgentExecutor(api_key='***', platform='anthropic')"

    # ===== Core Implementation =====

    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using Claude.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (unified + anthropic-specific)

        Returns:
            The completion response
        """
        try:
            # Extract and translate parameters
            unified_params = self._extract_unified_params(kwargs)
            anthropic_kwargs = kwargs.get("anthropic_kwargs", {})

            # Translate to Anthropic format
            params = self.mapper.to_platform_params(unified_params, anthropic_kwargs)

            # Ensure required parameters
            if "model" not in params:
                params["model"] = "claude-3-sonnet-20240229"  # Default
            if "max_tokens_to_sample" not in params:
                params["max_tokens_to_sample"] = 1000  # Default

            # Handle system prompt
            messages = []
            if "system" in params:
                system_prompt = params.pop("system")
            else:
                system_prompt = None

            # Create message
            messages.append({"role": "user", "content": prompt})

            # Make API call
            response = await self.client.messages.create(
                messages=messages, system=system_prompt, **params
            )

            # Extract text from response
            return cast(str, response.content[0].text)

        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
            raise AgentExecutionError(f"Anthropic completion failed: {e}") from None

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream completion using Claude.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters

        Yields:
            Completion chunks as they arrive
        """
        try:
            # Extract and translate parameters
            unified_params = self._extract_unified_params(kwargs)
            anthropic_kwargs = kwargs.get("anthropic_kwargs", {})

            # Translate to Anthropic format
            params = self.mapper.to_platform_params(unified_params, anthropic_kwargs)

            # Ensure required parameters
            if "model" not in params:
                params["model"] = "claude-3-sonnet-20240229"
            if "max_tokens_to_sample" not in params:
                params["max_tokens_to_sample"] = 1000

            # Force streaming
            params["stream"] = True

            # Handle system prompt
            messages = []
            if "system" in params:
                system_prompt = params.pop("system")
            else:
                system_prompt = None

            # Create message
            messages.append({"role": "user", "content": prompt})

            # Make streaming API call
            async with self.client.messages.stream(
                messages=messages, system=system_prompt, **params
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise AgentExecutionError(f"Anthropic streaming failed: {e}") from None

    # ===== Capability Support =====

    def supports_streaming(self) -> bool:
        """Claude supports streaming."""
        return True

    def supports_batch(self) -> bool:
        """Claude doesn't have native batch API."""
        return False

    def supports_tools(self) -> bool:
        """Claude supports tool use (function calling)."""
        return True

    async def complete_with_tools(
        self, prompt: str, tools: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Complete with tool use support.

        Args:
            prompt: The input prompt
            tools: List of tool definitions
            **kwargs: Additional parameters

        Returns:
            Response including potential tool calls
        """
        try:
            # Extract and translate parameters
            unified_params = self._extract_unified_params(kwargs)
            anthropic_kwargs = kwargs.get("anthropic_kwargs", {})

            # Translate to Anthropic format
            params = self.mapper.to_platform_params(unified_params, anthropic_kwargs)

            # Ensure required parameters
            if "model" not in params:
                params["model"] = "claude-3-sonnet-20240229"
            if "max_tokens_to_sample" not in params:
                params["max_tokens_to_sample"] = 1000

            # Handle system prompt
            messages = []
            if "system" in params:
                system_prompt = params.pop("system")
            else:
                system_prompt = None

            # Create message
            messages.append({"role": "user", "content": prompt})

            # Add tools
            params["tools"] = tools

            # Make API call with tools
            response = await self.client.messages.create(
                messages=messages, system=system_prompt, **params
            )

            # Format response
            result: dict[str, Any] = {
                "content": response.content[0].text if response.content else "",
                "tool_calls": [],
            }

            # Check for tool calls in response
            for content in response.content:
                if hasattr(content, "type") and content.type == "tool_use":
                    result["tool_calls"].append(
                        {
                            "id": content.id,
                            "name": content.name,
                            "arguments": content.input,
                        }
                    )

            return result

        except Exception as e:
            logger.error(f"Anthropic tool completion failed: {e}")
            raise AgentExecutionError(
                f"Anthropic tool completion failed: {e}"
            ) from None

    # ===== Model Support =====

    def get_supported_models(self) -> list[str]:
        """Get list of supported Claude models."""
        return self.SUPPORTED_MODELS.copy()

    def validate_model(self, model: str) -> bool:
        """Validate if a model is supported."""
        # Handle aliases
        base_model = model.split("-20")[0] if "-20" in model else model
        return model in self.SUPPORTED_MODELS or base_model in self.SUPPORTED_MODELS

    # ===== Cost Estimation =====

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Estimate cost for Claude completion.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model identifier

        Returns:
            Estimated cost in USD
        """
        # Normalize model name
        base_model = model.split("-20")[0] if "-20" in model else model

        if base_model in self.MODEL_COSTS:
            costs = self.MODEL_COSTS[base_model]
            input_cost = (input_tokens / 1000) * costs["input"]
            output_cost = (output_tokens / 1000) * costs["output"]
            return input_cost + output_cost

        return 0.0

    # ===== Parameter Handling =====

    def _extract_unified_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract unified parameters from kwargs.

        Args:
            kwargs: All parameters passed

        Returns:
            Dictionary of unified parameters
        """
        unified = {}

        # List of known unified parameters
        unified_keys = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "top_k",
            "stop_sequences",
            "system_prompt",
            "stream",
        ]

        for key in unified_keys:
            if key in kwargs:
                unified[key] = kwargs[key]

        return unified

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, "client"):
            # Anthropic client doesn't need explicit cleanup
            pass
