"""Tests for config_generator.subsystems.safety_constraints."""

from __future__ import annotations

from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.subsystems.safety_constraints import (
    generate_safety_constraints,
)

RAG_SOURCE = """\
def rag_pipeline(query):
    docs = vector_store.similarity_search(query, k=5)
    retriever = db.as_retriever()
    return llm.invoke(query)
"""


class TestGenerateSafetyConstraints:
    def test_rag_source_returns_safety_constraints(self) -> None:
        constraints, classification = generate_safety_constraints(RAG_SOURCE)
        assert classification.agent_type == "rag"
        assert len(constraints) >= 1
        metrics = {c.metric_name for c in constraints}
        assert "faithfulness" in metrics

    def test_with_precomputed_classification(self) -> None:
        classification = ClassificationResult(
            agent_type="chat",
            confidence=0.9,
            source="heuristic",
            reasoning="test",
        )
        constraints, returned_class = generate_safety_constraints(
            "any source", classification=classification
        )
        assert returned_class is classification
        assert any(c.metric_name == "toxicity_score" for c in constraints)

    def test_unknown_agent_type_returns_empty(self) -> None:
        classification = ClassificationResult(
            agent_type="unknown",
            confidence=0.5,
            source="test",
            reasoning="test",
        )
        constraints, _ = generate_safety_constraints(
            "source", classification=classification
        )
        assert constraints == []

    def test_all_constraints_have_agent_type(self) -> None:
        constraints, classification = generate_safety_constraints(RAG_SOURCE)
        for c in constraints:
            assert c.agent_type == classification.agent_type
