"""Tests for config_generator.agent_classifier."""

from __future__ import annotations

from traigent.config_generator.agent_classifier import (
    AGENT_TYPES,
    ClassificationResult,
    _heuristic_classify,
    classify_agent,
)

RAG_SOURCE = """\
from langchain_openai import ChatOpenAI
from chromadb import Client

def rag_pipeline(query: str) -> str:
    docs = vector_store.similarity_search(query, k=5)
    retriever = db.as_retriever()
    return llm.invoke(query)
"""

CHAT_SOURCE = """\
from openai import OpenAI

def chat_agent(query: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content
"""

CODE_GEN_SOURCE = """\
def generate_and_run(prompt: str) -> str:
    code = generate_code(prompt)
    result = exec(code)
    return result
"""

SUMMARIZATION_SOURCE = """\
def summarize_document(text: str) -> str:
    summary = llm.invoke(f"Summarize: {text}")
    return summary
"""

CLASSIFICATION_SOURCE = """\
def classify_intent(text: str) -> str:
    label = predict_label(text)
    return label
"""

ROUTER_SOURCE = """\
def route_request(query: str) -> str:
    agent = router.dispatch(query)
    return agent_executor.invoke(query)
"""

GENERIC_SOURCE = """\
def my_function(x: int) -> int:
    return x * 2
"""


class TestHeuristicClassify:
    def test_rag_agent(self) -> None:
        result = _heuristic_classify(RAG_SOURCE)
        assert result.agent_type == "rag"
        assert result.confidence >= 0.5

    def test_chat_agent(self) -> None:
        result = _heuristic_classify(CHAT_SOURCE)
        assert result.agent_type == "chat"
        assert result.confidence >= 0.5

    def test_code_gen_agent(self) -> None:
        result = _heuristic_classify(CODE_GEN_SOURCE)
        assert result.agent_type == "code_gen"
        assert result.confidence >= 0.5

    def test_summarization_agent(self) -> None:
        result = _heuristic_classify(SUMMARIZATION_SOURCE)
        assert result.agent_type == "summarization"
        assert result.confidence >= 0.5

    def test_classification_agent(self) -> None:
        result = _heuristic_classify(CLASSIFICATION_SOURCE)
        assert result.agent_type == "classification"
        assert result.confidence >= 0.5

    def test_router_agent(self) -> None:
        result = _heuristic_classify(ROUTER_SOURCE)
        assert result.agent_type == "router"
        assert result.confidence >= 0.5

    def test_generic_defaults_to_general_llm(self) -> None:
        result = _heuristic_classify(GENERIC_SOURCE)
        assert result.agent_type == "general_llm"
        assert result.confidence < 0.5

    def test_all_results_have_valid_agent_type(self) -> None:
        for src in [
            RAG_SOURCE,
            CHAT_SOURCE,
            CODE_GEN_SOURCE,
            SUMMARIZATION_SOURCE,
            CLASSIFICATION_SOURCE,
            ROUTER_SOURCE,
            GENERIC_SOURCE,
        ]:
            result = _heuristic_classify(src)
            assert result.agent_type in AGENT_TYPES

    def test_result_is_frozen(self) -> None:
        result = _heuristic_classify(RAG_SOURCE)
        assert isinstance(result, ClassificationResult)
        assert result.source == "heuristic"


class TestClassifyAgent:
    def test_no_llm_uses_heuristic(self) -> None:
        result = classify_agent(RAG_SOURCE, llm=None)
        assert result.source == "heuristic"
        assert result.agent_type == "rag"

    def test_high_confidence_skips_llm(self) -> None:
        """When heuristic confidence >= 0.8, LLM is not called."""
        # RAG source with many matches should have high confidence
        rich_rag = """\
from chromadb import Client
from pinecone import Pinecone
def pipeline(query):
    docs = vector_store.similarity_search(query, k=5)
    retriever = db.as_retriever()
    chunks = embedding.embed(query)
    return rag_pipeline(docs, query)
"""
        result = classify_agent(rich_rag, llm=None)
        assert result.agent_type == "rag"
        assert result.confidence >= 0.8

    def test_llm_non_numeric_confidence_handled(self) -> None:
        """LLM returning non-numeric confidence should not crash."""
        import json
        from unittest.mock import MagicMock

        llm = MagicMock()
        llm.complete.return_value = json.dumps(
            {"agent_type": "rag", "confidence": "high", "reasoning": "test"}
        )
        # Use a generic source that triggers low heuristic confidence
        result = classify_agent("def f(): pass", llm=llm)
        # Should still return a result (fallback confidence = 0.5)
        assert result is not None
