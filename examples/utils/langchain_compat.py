"""Compat helpers for optional LangChain dependencies used in examples."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

MOCK_MODE = str(os.environ.get("TRAIGENT_MOCK_LLM", "")).lower() in {"1", "true", "yes"}


class _StubResponse:
    def __init__(self, content: str):
        self.content = content
        self.response_metadata = {"response_time_ms": 0.0}
        self.usage_metadata = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }


class _StubChatOpenAI:  # type: ignore[override]
    """Minimal stub that returns mock responses for testing environments."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def invoke(self, messages: Any) -> _StubResponse:
        if isinstance(messages, str):
            payload = messages
        elif isinstance(messages, Sequence) and messages:
            last = messages[-1]
            payload = getattr(last, "content", str(last))
        else:
            payload = str(messages)

        # Generate contextual mock responses
        content = self._generate_mock_response(payload)
        return _StubResponse(content)

    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock responses based on prompt content."""
        prompt_lower = prompt.lower()

        # Calculator-style prompts
        if "calculate" in prompt_lower or "expression" in prompt_lower:
            return json.dumps(
                {"formula": "mock", "result": 42, "explanation": "Mock calculation"}
            )

        # Sentiment analysis
        if "sentiment" in prompt_lower:
            return "positive"

        # Extraction prompts
        if "extract" in prompt_lower or "json" in prompt_lower:
            return json.dumps({"extracted": "mock data"})

        # Default mock response
        return f"Mock response for: {prompt[:100]}"


if MOCK_MODE:
    ChatOpenAI = _StubChatOpenAI  # type: ignore[assignment]
else:
    try:  # pragma: no cover - optional dependency
        from langchain_openai import ChatOpenAI as _ChatOpenAI  # type: ignore

        ChatOpenAI = _ChatOpenAI  # type: ignore[assignment]
    except Exception:  # pragma: no cover - fallback stub
        ChatOpenAI = _StubChatOpenAI  # type: ignore[assignment]

try:  # pragma: no cover
    from langchain.schema import HumanMessage as _HumanMessage  # type: ignore
except Exception:  # pragma: no cover - fallback stub

    class HumanMessage:  # type: ignore[override]
        """Simple message container used by the stub ChatOpenAI."""

        def __init__(self, content: str):
            self.content = content

else:  # pragma: no cover - real dependency path
    HumanMessage = _HumanMessage  # type: ignore[assignment]


def extract_content(response: Any) -> str:
    """Return content text from a LangChain response or fallback to str."""

    return str(getattr(response, "content", response))


__all__ = ["ChatOpenAI", "HumanMessage", "extract_content"]
