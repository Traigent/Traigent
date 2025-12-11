"""Fallback helpers for optional LangChain dependencies used in the examples.

This module attempts to import the real LangChain classes when available. When
those optional extras are missing (for instance during linting or documentation
builds), lightweight mock implementations are provided so the example scripts
can still run without raising import errors.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ChatAnthropic",
    "ChatOpenAI",
    "HumanMessage",
    "SystemMessage",
    "AIMessage",
    "Document",
    "BM25Retriever",
]


class _MockResponse:
    """Simple response object that mimics LangChain's message interface."""

    def __init__(self, content: str = "Mock response") -> None:
        self.content = content

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.content

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(content={self.content!r})"


class _BaseMockLLM:
    """Minimal stand-in for Chat-based LLM clients."""

    def __init__(self, **kwargs: Any) -> None:
        self.model = kwargs.get("model") or kwargs.get("model_name", "mock")
        self.model_name = kwargs.get("model_name", self.model)
        self.temperature = float(kwargs.get("temperature", 0.0))
        self.max_tokens = int(kwargs.get("max_tokens", 128))

    @staticmethod
    def _make_response() -> _MockResponse:
        return _MockResponse()

    def invoke(
        self, messages: Sequence[Any]
    ) -> _MockResponse:  # pragma: no cover - deterministic
        return self._make_response()

    async def ainvoke(
        self, messages: Sequence[Any]
    ) -> _MockResponse:  # pragma: no cover - deterministic
        return self._make_response()


def _load_chat_anthropic() -> type[_BaseMockLLM]:
    try:  # pragma: no cover - executed when dependency is installed
        from langchain_anthropic import ChatAnthropic as real_class  # type: ignore

        return real_class  # type: ignore[no-any-return, return-value]
    except ImportError:  # pragma: no cover - exercised in lint/mock environments

        class ChatAnthropic(_BaseMockLLM):
            """Fallback ChatAnthropic client used when LangChain isn't installed."""

        return ChatAnthropic


def _load_chat_openai() -> type[_BaseMockLLM]:
    try:  # pragma: no cover - executed when dependency is installed
        from langchain_openai import ChatOpenAI as real_class  # type: ignore

        return real_class  # type: ignore[no-any-return, return-value]
    except ImportError:  # pragma: no cover - exercised in lint/mock environments

        class ChatOpenAI(_BaseMockLLM):
            """Fallback ChatOpenAI client used when LangChain isn't installed."""

        return ChatOpenAI


@dataclass
class _BaseMessage:
    """Shared structure for mock LangChain message objects."""

    content: str = ""
    type: str = "generic"


def _load_messages() -> (
    tuple[type[_BaseMessage], type[_BaseMessage], type[_BaseMessage]]
):
    try:  # pragma: no cover - executed when dependency is installed
        from langchain_core.messages import AIMessage as real_ai  # type: ignore
        from langchain_core.messages import HumanMessage as real_human
        from langchain_core.messages import SystemMessage as real_system

        return real_human, real_system, real_ai  # type: ignore[return-value]
    except ImportError:  # pragma: no cover - exercised in lint/mock environments

        class HumanMessage(_BaseMessage):
            type: str = "human"

        class SystemMessage(_BaseMessage):
            type: str = "system"

        class AIMessage(_BaseMessage):
            type: str = "ai"

        return HumanMessage, SystemMessage, AIMessage


@dataclass
class _BaseDocument:
    """Minimal representation of a retriever document."""

    page_content: str = ""
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


def _load_document() -> type[_BaseDocument]:
    try:  # pragma: no cover - executed when dependency is installed
        from langchain_core.documents import Document as real_document  # type: ignore

        return real_document  # type: ignore[return-value]
    except ImportError:  # pragma: no cover - exercised in lint/mock environments

        class Document(_BaseDocument):
            """Fallback document used when LangChain isn't installed."""

        return Document


class _BaseRetriever:
    """Simplified retriever used as a fallback for BM25."""

    def __init__(self, **kwargs: Any) -> None:
        self.k = int(kwargs.get("k", 3))

    @classmethod
    def from_documents(
        cls, documents: Iterable[Any] | None, **kwargs: Any
    ) -> _BaseRetriever | None:
        return cls(**kwargs) if documents else None

    def get_relevant_documents(
        self, query: str
    ) -> list[_BaseDocument]:  # pragma: no cover - deterministic
        return []

    async def aget_relevant_documents(
        self, query: str
    ) -> list[_BaseDocument]:  # pragma: no cover - deterministic
        return []


def _load_bm25() -> type[_BaseRetriever]:
    try:  # pragma: no cover - executed when dependency is installed
        from langchain_community.retrievers import (
            BM25Retriever as real_retriever,  # type: ignore
        )

        return real_retriever  # type: ignore[no-any-return, return-value]
    except ImportError:  # pragma: no cover - exercised in lint/mock environments

        class BM25Retriever(_BaseRetriever):
            """Fallback BM25 retriever used when LangChain isn't installed."""

        return BM25Retriever


ChatAnthropic = _load_chat_anthropic()
ChatOpenAI = _load_chat_openai()
HumanMessage, SystemMessage, AIMessage = _load_messages()
Document = _load_document()
BM25Retriever = _load_bm25()
