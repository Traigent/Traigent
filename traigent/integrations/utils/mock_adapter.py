"""Mock adapter for testing LLM integrations without API calls.

Provides a lightweight mock adapter pattern that keeps main plugin
logic clean by separating mock functionality into dedicated methods.

Security:
    Mock activation in **production is impossible** — neither the
    in-code API (:func:`traigent.testing.enable_mock_mode_for_quickstart`)
    nor the legacy ``TRAIGENT_MOCK_LLM=true`` env var can flip mock
    mode on when ``ENVIRONMENT=production``. The in-code path raises
    ``RuntimeError``; the env-var path raises ``OSError`` at module
    import in :mod:`traigent.utils.env_config`. That's the surviving
    guard against the original prod incident — a stray env var
    silently swapping real LLM calls for canned mock text.

    Outside production, mock mode can be activated by either the
    in-code API (recommended) or the legacy env var (kept for
    backward compatibility with existing test fixtures and example
    scripts). Provider-specific ``*_MOCK`` env vars (e.g.
    ``OPENAI_MOCK``) are completely ignored everywhere — those were
    the worst offenders in the original incident.
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
    The single source of truth is
    :func:`traigent.testing.is_mock_mode_enabled`, which is only flipped
    on by an explicit in-code call to
    :func:`traigent.testing.enable_mock_mode_for_quickstart`. Tests that
    want a one-off mock response without flipping the global flag can
    still call :meth:`get_mock_response` directly from inside a
    ``unittest.mock.patch`` of the underlying client.
    """

    _pending_tasks: ClassVar[set[asyncio.Task]] = set()

    @classmethod
    def is_mock_enabled(cls, provider: str) -> bool:
        """Return ``True`` iff mock mode should intercept LLM calls.

        Delegates to :func:`traigent.utils.env_config.is_mock_llm`, which
        is the single source of truth for both LLM interceptors and the
        SDK's mock-aware behavior. The original prod incident — a stray
        env var swapping real calls for canned text — is prevented by
        :func:`is_mock_llm`'s production hard-block: in production
        environments the env-var path is rejected at import time, and
        only the in-code
        :func:`traigent.testing.enable_mock_mode_for_quickstart` opt-in
        can flip the flag. Tests and dev environments can still set
        ``TRAIGENT_MOCK_LLM=true`` for backward compatibility.

        Args:
            provider: Provider name (kept for signature compatibility).

        Returns:
            ``True`` if mock mode is on, ``False`` otherwise.
        """
        del provider  # signature compatibility only
        from traigent.utils.env_config import is_mock_llm

        return is_mock_llm()

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
