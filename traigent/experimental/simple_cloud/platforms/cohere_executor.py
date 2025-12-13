"""Cohere platform executor implementation.

This module provides integration with Cohere's language models,
supporting Command and Command-R models with streaming capabilities.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility FUNC-AGENTS FUNC-INTEGRATIONS REQ-AGNT-013 REQ-INT-008 SYNC-OptimizationFlow

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any, cast

try:
    import cohere

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from traigent.utils.exceptions import (
    AgentExecutionError,
    ConfigurationError,
)
from traigent.utils.logging import get_logger

from .base_platform import BasePlatformExecutor
from .parameter_mapping import ParameterMapper

logger = get_logger(__name__)


class CohereAgentExecutor(BasePlatformExecutor):
    """Executor for Cohere language models."""

    # Supported Cohere models
    SUPPORTED_MODELS = [
        "command",
        "command-light",
        "command-nightly",
        "command-r",
        "command-r-plus",
    ]

    # Model pricing (per 1K tokens) - approximate
    MODEL_COSTS = {
        "command": {"input": 0.001, "output": 0.002},
        "command-light": {"input": 0.0004, "output": 0.0008},
        "command-r": {"input": 0.0005, "output": 0.0015},
        "command-r-plus": {"input": 0.003, "output": 0.015},
    }

    def __init__(self, api_key: str | None = None, **kwargs) -> None:
        """Initialize Cohere executor.

        Args:
            api_key: Cohere API key (or use COHERE_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        if not COHERE_AVAILABLE:
            raise ImportError(
                "cohere package not installed. Install with: pip install cohere"
            )

        # Get API key
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "Cohere API key not found. "
                "Set COHERE_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize client
        self.client = cohere.AsyncClient(api_key=self.api_key)
        self.mapper = ParameterMapper("cohere")

        logger.info("Initialized Cohere executor")

    # ===== Core Implementation =====

    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using Cohere.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (unified + cohere-specific)

        Returns:
            The completion response
        """
        try:
            # Extract and translate parameters
            unified_params = self._extract_unified_params(kwargs)
            cohere_kwargs = kwargs.get("cohere_kwargs", {})

            # Translate to Cohere format
            params = self.mapper.to_platform_params(unified_params, cohere_kwargs)

            # Ensure required parameters
            if "model" not in params:
                params["model"] = "command-r"  # Default to command-r
            if "max_tokens" not in params:
                params["max_tokens"] = 1000  # Default

            # Cohere uses 'message' instead of 'prompt' for chat
            params["message"] = prompt

            # Handle system prompt if provided
            if "system_prompt" in unified_params:
                params["preamble"] = unified_params["system_prompt"]

            # Remove parameters not used in chat endpoint
            params.pop("stream", None)

            # Make API call
            response = await self.client.chat(**params)

            # Extract text from response
            return cast(str, response.text)

        except Exception as e:
            logger.error(f"Cohere completion failed: {e}")
            raise AgentExecutionError(f"Cohere completion failed: {e}") from None

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream completion using Cohere.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters

        Yields:
            Completion chunks as they arrive
        """
        try:
            # Extract and translate parameters
            unified_params = self._extract_unified_params(kwargs)
            cohere_kwargs = kwargs.get("cohere_kwargs", {})

            # Translate to Cohere format
            params = self.mapper.to_platform_params(unified_params, cohere_kwargs)

            # Ensure required parameters
            if "model" not in params:
                params["model"] = "command-r"
            if "max_tokens" not in params:
                params["max_tokens"] = 1000

            # Cohere uses 'message' for chat
            params["message"] = prompt

            # Handle system prompt if provided
            if "system_prompt" in unified_params:
                params["preamble"] = unified_params["system_prompt"]

            # Force streaming
            params["stream"] = True

            # Make streaming API call
            async for event in self.client.chat_stream(**params):
                if event.event_type == "text-generation":
                    yield event.text

        except Exception as e:
            logger.error(f"Cohere streaming failed: {e}")
            raise AgentExecutionError(f"Cohere streaming failed: {e}") from None

    # ===== Capability Support =====

    def supports_streaming(self) -> bool:
        """Cohere supports streaming."""
        return True

    def supports_batch(self) -> bool:
        """Cohere doesn't have native batch API."""
        return False

    def supports_tools(self) -> bool:
        """Cohere supports tool use through connectors."""
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
            cohere_kwargs = kwargs.get("cohere_kwargs", {})

            # Translate to Cohere format
            params = self.mapper.to_platform_params(unified_params, cohere_kwargs)

            # Ensure required parameters
            if "model" not in params:
                params["model"] = "command-r"
            if "max_tokens" not in params:
                params["max_tokens"] = 1000

            params["message"] = prompt

            # Handle system prompt if provided
            if "system_prompt" in unified_params:
                params["preamble"] = unified_params["system_prompt"]

            # Convert tools to Cohere format
            cohere_tools = []
            for tool in tools:
                cohere_tool = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameter_definitions": {},
                }

                # Convert parameters
                if "parameters" in tool and "properties" in tool["parameters"]:
                    for param_name, param_def in tool["parameters"][
                        "properties"
                    ].items():
                        cohere_tool["parameter_definitions"][param_name] = {
                            "type": param_def.get("type", "string"),
                            "description": param_def.get("description", ""),
                            "required": param_name
                            in tool["parameters"].get("required", []),
                        }

                cohere_tools.append(cohere_tool)

            # Add tools to params
            params["tools"] = cohere_tools

            # Make API call with tools
            response = await self.client.chat(**params)

            # Format response
            result = {"content": response.text, "tool_calls": []}

            # Check for tool calls in response
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    result["tool_calls"].append(
                        {
                            "id": tool_call.get("id", ""),
                            "name": tool_call.get("name", ""),
                            "arguments": tool_call.get("parameters", {}),
                        }
                    )

            return result

        except Exception as e:
            logger.error(f"Cohere tool completion failed: {e}")
            raise AgentExecutionError(f"Cohere tool completion failed: {e}") from None

    # ===== Model Support =====

    def get_supported_models(self) -> list[str]:
        """Get list of supported Cohere models."""
        return self.SUPPORTED_MODELS.copy()

    def validate_model(self, model: str) -> bool:
        """Validate if a model is supported."""
        return model in self.SUPPORTED_MODELS

    # ===== Cost Estimation =====

    def estimate_completion_cost(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> float:
        """Estimate cost for Cohere completion.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model identifier

        Returns:
            Estimated cost in USD
        """
        # Normalize model name
        base_model = model.split("-")[0] if model != "command-r-plus" else model

        if base_model in self.MODEL_COSTS:
            costs = self.MODEL_COSTS[base_model]
            input_cost = (input_tokens / 1000) * costs["input"]
            output_cost = (output_tokens / 1000) * costs["output"]
            return input_cost + output_cost

        # Default pricing for unknown models
        if model in self.SUPPORTED_MODELS:
            # Use command pricing as default
            return self.estimate_completion_cost(input_tokens, output_tokens, "command")

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
            "frequency_penalty",
            "presence_penalty",
            "stop_sequences",
            "seed",
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
            # Cohere client doesn't need explicit cleanup
            pass
