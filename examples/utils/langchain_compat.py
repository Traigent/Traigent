"""Compat helpers for optional LangChain dependencies used in examples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:  # pragma: no cover - optional dependency
    from langchain_openai import ChatOpenAI as _ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover - fallback stub

    class _StubResponse:
        def __init__(self, content: str):
            self.content = content
            self.response_metadata = {"response_time_ms": 0.0}
            self.usage_metadata = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

    class ChatOpenAI:  # type: ignore[override]
        """Minimal stub that echoes prompts for offline environments."""

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
            content = payload or "Mock response"
            return _StubResponse(content)

else:  # pragma: no cover - real dependency path
    ChatOpenAI = _ChatOpenAI  # type: ignore[assignment]

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
