"""Mock adapter for testing LLM integrations without API calls.

Provides a lightweight mock adapter pattern that keeps main plugin
logic clean by separating mock functionality into dedicated methods.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Environment variable names for each provider
MOCK_ENV_VARS = {
    "openai": "OPENAI_MOCK",
    "azure_openai": "AZURE_OPENAI_MOCK",
    "anthropic": "ANTHROPIC_MOCK",
    "gemini": "GEMINI_MOCK",
    "cohere": "COHERE_MOCK",
    "huggingface": "HUGGINGFACE_MOCK",
    "bedrock": "BEDROCK_MOCK",
    # Global override
    "traigent": "TRAIGENT_MOCK_MODE",
}


@dataclass
class MockResponse:
    """Generic mock response for LLM calls.

    This dataclass provides a simple mock response that can be adapted
    to provider-specific formats.
    """

    text: str = "This is a mock response for testing."
    model: str = "mock-model"
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30
    finish_reason: str = "stop"
    response_id: str = "mock-response-id"


class MockAdapter:
    """Lightweight mock adapter for testing without API calls.

    Usage in plugins:
        if MockAdapter.is_mock_enabled("openai"):
            return MockAdapter.get_mock_response("openai", **kwargs)

    This keeps mock logic separate from main plugin code.
    """

    @classmethod
    def is_mock_enabled(cls, provider: str) -> bool:
        """Check if mock mode is enabled for a provider.

        Checks both provider-specific and global mock env vars.

        Args:
            provider: Provider name (openai, anthropic, gemini, etc.)

        Returns:
            True if mock mode is enabled.
        """
        # Check global mock mode first
        global_mock = os.getenv("TRAIGENT_MOCK_MODE", "").lower()
        if global_mock in ("true", "1", "yes"):
            return True

        # Check provider-specific mock
        env_var = MOCK_ENV_VARS.get(provider.lower())
        if env_var:
            provider_mock = os.getenv(env_var, "").lower()
            if provider_mock in ("true", "1", "yes"):
                return True

        return False

    @classmethod
    def get_mock_response(
        cls,
        provider: str,
        response_text: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Get a provider-appropriate mock response object.

        Args:
            provider: Provider name.
            response_text: Custom response text (optional).
            model: Model name to include in response (optional).
            **kwargs: Additional kwargs (used to extract model if not provided).

        Returns:
            Mock response appropriate for the provider.
        """
        mock_data = MockResponse(
            text=response_text or MockResponse.text,
            model=model or kwargs.get("model", MockResponse.model),
        )

        # Route to provider-specific mock builder
        builders = {
            "openai": cls._build_openai_mock,
            "azure_openai": cls._build_openai_mock,  # Same format
            "anthropic": cls._build_anthropic_mock,
            "gemini": cls._build_gemini_mock,
            "cohere": cls._build_cohere_mock,
        }

        builder = builders.get(provider.lower(), cls._build_generic_mock)
        return builder(mock_data)

    @classmethod
    def _build_openai_mock(cls, data: MockResponse) -> dict[str, Any]:
        """Build OpenAI ChatCompletion-like response dict.

        Note: This returns a dict that mimics the structure.
        For full SDK compatibility, the caller may need to wrap this
        in an actual SDK response object.
        """
        return {
            "id": data.response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": data.text,
                    },
                    "finish_reason": data.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": data.prompt_tokens,
                "completion_tokens": data.completion_tokens,
                "total_tokens": data.total_tokens,
            },
        }

    @classmethod
    def _build_anthropic_mock(cls, data: MockResponse) -> dict[str, Any]:
        """Build Anthropic Message-like response dict."""
        return {
            "id": data.response_id,
            "type": "message",
            "role": "assistant",
            "model": data.model,
            "content": [
                {
                    "type": "text",
                    "text": data.text,
                }
            ],
            "stop_reason": data.finish_reason,
            "usage": {
                "input_tokens": data.prompt_tokens,
                "output_tokens": data.completion_tokens,
            },
        }

    @classmethod
    def _build_gemini_mock(cls, data: MockResponse) -> dict[str, Any]:
        """Build Gemini GenerateContentResponse-like dict."""
        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": data.text}],
                    },
                    "finish_reason": "STOP",
                    "index": 0,
                }
            ],
            "usage_metadata": {
                "prompt_token_count": data.prompt_tokens,
                "candidates_token_count": data.completion_tokens,
                "total_token_count": data.total_tokens,
            },
        }

    @classmethod
    def _build_cohere_mock(cls, data: MockResponse) -> dict[str, Any]:
        """Build Cohere response-like dict."""
        return {
            "id": data.response_id,
            "text": data.text,
            "generation_id": data.response_id,
            "meta": {
                "billed_units": {
                    "input_tokens": data.prompt_tokens,
                    "output_tokens": data.completion_tokens,
                }
            },
        }

    @classmethod
    def _build_generic_mock(cls, data: MockResponse) -> dict[str, Any]:
        """Build generic mock response dict."""
        return {
            "text": data.text,
            "model": data.model,
            "usage": {
                "prompt_tokens": data.prompt_tokens,
                "completion_tokens": data.completion_tokens,
                "total_tokens": data.total_tokens,
            },
        }


def with_mock_support(provider: str) -> Callable:
    """Decorator to add mock support to a function.

    Usage:
        @with_mock_support("openai")
        def create_completion(**kwargs):
            # Real implementation
            ...

    Args:
        provider: Provider name for mock lookup.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if MockAdapter.is_mock_enabled(provider):
                logger.debug(
                    f"Mock mode enabled for {provider}, returning mock response"
                )
                return MockAdapter.get_mock_response(provider, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    return decorator
