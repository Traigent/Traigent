"""Mock adapter for testing LLM integrations without API calls.

Provides a lightweight mock adapter pattern that keeps main plugin
logic clean by separating mock functionality into dedicated methods.

Security:
    Mock activation is NEVER controlled by environment variables. Mock
    responses are only produced when callers explicitly invoke
    ``MockAdapter.get_mock_response(...)`` (typically from test code that
    has already patched the LLM client at the integration boundary).
    ``is_mock_enabled`` always returns ``False`` to guarantee that no
    production code path can be silently swapped for fake responses by
    setting an env var such as ``TRAIGENT_MOCK_LLM`` or ``OPENAI_MOCK``.
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# Default mock delay in milliseconds (0 = no delay)
# Set TRAIGENT_MOCK_DELAY_MS to simulate realistic LLM latency in tests.
DEFAULT_MOCK_DELAY_MS = 0


def _get_mock_delay_ms() -> int:
    """Get mock delay from environment with safe parsing.

    Returns:
        Delay in milliseconds, or 0 if not set or invalid.
    """
    delay_str = os.getenv("TRAIGENT_MOCK_DELAY_MS", "")
    if not delay_str:
        return DEFAULT_MOCK_DELAY_MS
    try:
        # Strip common suffixes like "ms" for user-friendliness
        delay_str = delay_str.lower().rstrip("ms").strip()
        return max(0, int(delay_str))
    except ValueError:
        logger.warning(f"Invalid TRAIGENT_MOCK_DELAY_MS value '{delay_str}', using 0")
        return 0


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

    Mock activation is **never** auto-enabled via environment variables.
    Tests that need a mock response must call
    ``MockAdapter.get_mock_response(...)`` directly (e.g. from inside a
    ``unittest.mock.patch`` that replaces the real client method).

    ``is_mock_enabled`` is preserved as a stub that always returns
    ``False`` so any production interceptor that previously consulted it
    will now always take the real-LLM path. Callers wanting mock
    behaviour from tests must patch the underlying client instead.
    """

    _pending_tasks: ClassVar[set[asyncio.Task]] = set()

    @classmethod
    def is_mock_enabled(cls, provider: str) -> bool:
        """Always returns ``False``.

        Historically this consulted ``TRAIGENT_MOCK_LLM`` and a set of
        provider-specific ``*_MOCK`` environment variables. That env-toggle
        was removed because a stray env var in production caused real LLM
        calls to be silently replaced with canned mock text. Tests must
        patch the LLM client (or call ``get_mock_response`` directly)
        rather than relying on a global flag.

        Args:
            provider: Provider name (kept for signature compatibility).

        Returns:
            Always ``False``.
        """
        del provider  # signature compatibility only
        return False

    @classmethod
    def _apply_mock_delay(cls, provider: str) -> None:
        """Apply mock delay, using async sleep if in event loop.

        This method detects if it's running inside an async context and
        uses the appropriate sleep function to avoid blocking the event loop.

        Args:
            provider: Provider name for logging.
        """
        delay_ms = _get_mock_delay_ms()
        if delay_ms <= 0:
            return

        delay_sec = delay_ms / 1000.0
        logger.debug(f"Mock delay: {delay_ms}ms for {provider}")

        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - schedule async sleep
            # This won't block synchronous callers
            task = loop.create_task(asyncio.sleep(delay_sec))
            cls._pending_tasks.add(task)
            task.add_done_callback(cls._pending_tasks.discard)
            # Also do a minimal sync sleep to ensure some delay is visible
            time.sleep(delay_sec)
        except RuntimeError:
            # No running event loop - use sync sleep
            time.sleep(delay_sec)

    @classmethod
    def get_mock_response(
        cls,
        provider: str,
        response_text: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Get a provider-appropriate mock response object.

        This method is intended to be invoked **explicitly** from test
        code (e.g. inside a ``unittest.mock.patch`` of the real client).
        It is no longer reachable from any production code path via an
        environment toggle.

        Args:
            provider: Provider name.
            response_text: Custom response text (optional).
            model: Model name to include in response (optional).
            **kwargs: Additional kwargs (used to extract model if not provided).

        Returns:
            Mock response appropriate for the provider.
        """
        # Apply configurable delay to simulate realistic LLM latency
        # This is useful for testing parallel execution visibility in traces
        cls._apply_mock_delay(provider)

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
