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
import re
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from traigent.utils.logging import configure_litellm_logging

_CHAT_COMPLETION_OBJECT = "chat.completion"

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


_STREAM_PIECE_RE = re.compile(r"\S+\s*|\s+")


class MockLiteLLMStream:
    """Offline stand-in for LiteLLM's ``CustomStreamWrapper`` in mock mode (#2801).

    Iterable both ways, like the real wrapper: ``for chunk in stream`` and
    ``async for chunk in stream``. Each chunk is a ``ModelResponseStream`` whose
    ``choices[0].delta.content`` holds a piece of the reply, so joining the
    pieces gives the reply text; a final chunk per choice carries
    ``finish_reason`` and, on the last one, the synthetic usage block.
    """

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self._position = 0

    def __iter__(self) -> "MockLiteLLMStream":
        return self

    def __next__(self) -> Any:
        if self._position >= len(self._chunks):
            raise StopIteration
        chunk = self._chunks[self._position]
        self._position += 1
        return chunk

    def __aiter__(self) -> "MockLiteLLMStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return self.__next__()
        except StopIteration:
            raise StopAsyncIteration from None


class MockAdapter:
    """Lightweight mock adapter for testing without API calls.

    Mock mode is never switched on implicitly. It is on only after an
    explicit in-code call to
    :func:`traigent.testing.enable_mock_mode_for_quickstart` (the
    recommended path) or, outside production, when the deprecated
    ``TRAIGENT_MOCK_LLM`` environment variable is set.
    :func:`traigent.utils.env_config.is_mock_llm` is the single source of
    truth for both. Tests that want a one-off mock response without
    flipping the global flag can still call :meth:`get_mock_response`
    directly from inside a ``unittest.mock.patch`` of the underlying
    client.
    """

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
        """Block for the configured mock latency (synchronous callers only).

        Async callers must ``await`` :meth:`await_mock_delay` instead: calling
        this from a running event loop blocks every other coroutine for the
        whole delay (#2809).

        Args:
            provider: Provider name for logging.
        """
        delay_ms = _get_mock_delay_ms()
        if delay_ms <= 0:
            return
        logger.debug(f"Mock delay: {delay_ms}ms for {provider}")
        time.sleep(delay_ms / 1000.0)

    @classmethod
    async def await_mock_delay(cls, provider: str) -> None:
        """Wait for the configured mock latency without blocking the event loop.

        Concurrent async mock calls overlap their delay instead of serializing.

        Args:
            provider: Provider name for logging.
        """
        delay_ms = _get_mock_delay_ms()
        if delay_ms <= 0:
            return
        logger.debug(f"Mock delay (async): {delay_ms}ms for {provider}")
        await asyncio.sleep(delay_ms / 1000.0)

    @classmethod
    def get_mock_response(
        cls,
        provider: str,
        response_text: str | None = None,
        model: str | None = None,
        apply_delay: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Get a provider-appropriate mock response object.

        Every mock interception point serves its response through here, once
        :func:`traigent.utils.env_config.is_mock_llm` says mock mode is on
        (the in-code opt-in, or the deprecated environment variable outside
        production). Test code may also call it directly (e.g. inside a
        ``unittest.mock.patch`` of the real client).

        Args:
            provider: Provider name.
            response_text: Custom response text (optional).
            model: Model name to include in response (optional).
            apply_delay: Block for the configured ``TRAIGENT_MOCK_DELAY_MS``
                before returning. Async callers pass ``False`` after
                ``await``-ing :meth:`await_mock_delay` themselves.
            **kwargs: Additional kwargs (used to extract model if not provided).

        Returns:
            Mock response appropriate for the provider.
        """
        # Apply configurable delay to simulate realistic LLM latency
        # This is useful for testing parallel execution visibility in traces
        if apply_delay:
            cls._apply_mock_delay(provider)

        # Every mock interception point serves through here: record that a mock
        # response reached a caller, for the run's provenance (#2516).
        from traigent.utils.mock_provenance import note_mock_response

        note_mock_response()

        mock_data = MockResponse(
            # ``is None`` (not falsy): a caller-supplied empty reply is a reply.
            text=MockResponse.text if response_text is None else response_text,
            model=model or kwargs.get("model", MockResponse.model),
        )

        # Route to provider-specific mock builder
        builders = {
            "openai": cls._build_openai_mock,
            "azure_openai": cls._build_openai_mock,  # Same format
            "anthropic": cls._build_anthropic_mock,
            "bedrock": cls._build_bedrock_mock,
            "gemini": cls._build_gemini_mock,
            "cohere": cls._build_cohere_mock,
            "litellm": cls._build_litellm_mock,
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
            "object": _CHAT_COMPLETION_OBJECT,
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
    def _build_bedrock_mock(cls, data: MockResponse) -> dict[str, Any]:
        """Build Bedrock Anthropic Messages API-like response dict."""
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
                "total_tokens": data.total_tokens,
            },
        }

    @classmethod
    def _build_litellm_mock(cls, data: MockResponse) -> Any:
        """Build a LiteLLM ModelResponse-like mock response."""
        try:
            import litellm
            from litellm import ModelResponse
        except Exception as exc:  # pragma: no cover - exercised only without litellm
            logger.debug("LiteLLM ModelResponse unavailable, using fallback: %s", exc)
            return SimpleNamespace(
                id=data.response_id,
                object=_CHAT_COMPLETION_OBJECT,
                created=int(time.time()),
                model=data.model,
                choices=[
                    SimpleNamespace(
                        index=0,
                        message=SimpleNamespace(
                            role="assistant",
                            content=data.text,
                        ),
                        finish_reason=data.finish_reason,
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=data.prompt_tokens,
                    completion_tokens=data.completion_tokens,
                    total_tokens=data.total_tokens,
                ),
            )

        configure_litellm_logging(litellm_module=litellm)
        return ModelResponse(
            id=data.response_id,
            model=data.model,
            object=_CHAT_COMPLETION_OBJECT,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": data.text,
                    },
                    "finish_reason": data.finish_reason,
                }
            ],
            usage={
                "prompt_tokens": data.prompt_tokens,
                "completion_tokens": data.completion_tokens,
                "total_tokens": data.total_tokens,
            },
        )

    @classmethod
    def expand_litellm_choices(cls, response: Any, n: int) -> Any:
        """Give a mock LiteLLM response ``n`` identical choices (the ``n=`` arg)."""
        if n <= 1:
            return response
        from litellm.types.utils import Choices, Message

        first = response.choices[0]
        for index in range(1, n):
            response.choices.append(
                Choices(
                    index=index,
                    message=Message(role="assistant", content=first.message.content),
                    finish_reason=first.finish_reason,
                )
            )
        return response

    @classmethod
    def build_litellm_stream(cls, response: Any) -> MockLiteLLMStream:
        """Turn a mock LiteLLM response into a stream of ``ModelResponseStream``."""
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        def chunk(index: int, delta: Any, finish_reason: str | None) -> Any:
            return ModelResponseStream(
                id=response.id,
                model=response.model,
                choices=[
                    StreamingChoices(
                        index=index, delta=delta, finish_reason=finish_reason
                    )
                ],
            )

        chunks: list[Any] = []
        for choice in response.choices:
            text = choice.message.content or ""
            pieces = _STREAM_PIECE_RE.findall(text)
            for position, piece in enumerate(pieces):
                role = "assistant" if position == 0 else None
                chunks.append(
                    chunk(choice.index, Delta(content=piece, role=role), None)
                )
            chunks.append(
                chunk(
                    choice.index,
                    Delta(content=None, role="assistant" if not pieces else None),
                    choice.finish_reason or "stop",
                )
            )
        if chunks:
            chunks[-1].usage = response.usage
        return MockLiteLLMStream(chunks)

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
