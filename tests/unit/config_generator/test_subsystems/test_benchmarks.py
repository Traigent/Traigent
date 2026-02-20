"""Tests for config_generator.subsystems.benchmarks."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.llm_backend import BudgetExhausted
from traigent.config_generator.subsystems.benchmarks import generate_benchmarks


class TestGenerateBenchmarks:
    def test_default_returns_general_llm(self) -> None:
        benchmarks = generate_benchmarks("def f(): pass")
        assert len(benchmarks) == 1
        assert benchmarks[0].source == "catalog"

    def test_rag_classification(self) -> None:
        classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        benchmarks = generate_benchmarks("", classification=classification)
        assert len(benchmarks) == 1
        assert benchmarks[0].name == "RAG Question Answering"

    def test_code_gen_classification(self) -> None:
        classification = ClassificationResult(
            agent_type="code_gen", confidence=0.9, source="heuristic", reasoning="test"
        )
        benchmarks = generate_benchmarks("", classification=classification)
        assert len(benchmarks) == 1
        assert benchmarks[0].name == "Code Generation"

    def test_unknown_classification_falls_back_to_general(self) -> None:
        classification = ClassificationResult(
            agent_type="unknown_type",
            confidence=0.5,
            source="heuristic",
            reasoning="test",
        )
        benchmarks = generate_benchmarks("", classification=classification)
        assert len(benchmarks) == 1
        assert benchmarks[0].name == "General LLM Evaluation"

    def test_llm_enrichment_replaces_catalog(self) -> None:
        llm = MagicMock()
        examples = [
            {"input": {"prompt": "Hello"}, "output": "Hi there"},
            {"input": {"prompt": "What is 2+2?"}, "output": "4"},
            {"input": {"prompt": "Explain AI"}, "output": "AI is..."},
        ]
        llm.complete.return_value = json.dumps(examples)

        benchmarks = generate_benchmarks("def f(): pass", llm=llm)
        assert len(benchmarks) == 1
        assert benchmarks[0].source == "llm"
        assert len(benchmarks[0].sample_examples) == 3

    def test_llm_budget_exhausted_returns_catalog(self) -> None:
        llm = MagicMock()
        llm.complete.side_effect = BudgetExhausted(0.10)
        benchmarks = generate_benchmarks("def f(): pass", llm=llm)
        assert len(benchmarks) == 1
        assert benchmarks[0].source == "catalog"

    def test_llm_invalid_json_returns_catalog(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = "not valid json"
        benchmarks = generate_benchmarks("def f(): pass", llm=llm)
        assert len(benchmarks) == 1
        assert benchmarks[0].source == "catalog"

    def test_llm_empty_array_returns_catalog(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = "[]"
        benchmarks = generate_benchmarks("def f(): pass", llm=llm)
        assert len(benchmarks) == 1
        assert benchmarks[0].source == "catalog"

    def test_llm_markdown_fenced_response(self) -> None:
        llm = MagicMock()
        examples = [{"input": {"prompt": "test"}, "output": "result"}]
        llm.complete.return_value = "```json\n" + json.dumps(examples) + "\n```"

        benchmarks = generate_benchmarks("def f(): pass", llm=llm)
        assert len(benchmarks) == 1
        assert benchmarks[0].source == "llm"
        assert len(benchmarks[0].sample_examples) == 1
