"""Tests for config_generator.presets.benchmark_catalog."""

from __future__ import annotations

import pytest

from traigent.config_generator.presets.benchmark_catalog import (
    all_benchmark_types,
    get_benchmark_for_agent_type,
)


class TestGetBenchmarkForAgentType:
    @pytest.mark.parametrize(
        "agent_type",
        ["rag", "chat", "code_gen", "summarization", "classification", "general_llm"],
    )
    def test_known_agent_types_have_benchmarks(self, agent_type: str) -> None:
        benchmark = get_benchmark_for_agent_type(agent_type)
        assert benchmark is not None
        assert benchmark.name != ""
        assert benchmark.description != ""
        assert benchmark.source == "catalog"

    def test_unknown_agent_type_returns_none(self) -> None:
        assert get_benchmark_for_agent_type("unknown_type") is None

    def test_rag_benchmark_has_question_input(self) -> None:
        benchmark = get_benchmark_for_agent_type("rag")
        assert benchmark is not None
        assert "question" in benchmark.example_schema.get("input", {})

    def test_code_gen_benchmark_has_prompt_input(self) -> None:
        benchmark = get_benchmark_for_agent_type("code_gen")
        assert benchmark is not None
        assert "prompt" in benchmark.example_schema.get("input", {})

    def test_classification_benchmark_has_text_input(self) -> None:
        benchmark = get_benchmark_for_agent_type("classification")
        assert benchmark is not None
        assert "text" in benchmark.example_schema.get("input", {})

    def test_all_benchmarks_have_output_schema(self) -> None:
        for agent_type in all_benchmark_types():
            benchmark = get_benchmark_for_agent_type(agent_type)
            assert benchmark is not None
            assert "output" in benchmark.example_schema


class TestAllBenchmarkTypes:
    def test_returns_frozenset(self) -> None:
        result = all_benchmark_types()
        assert isinstance(result, frozenset)

    def test_contains_known_types(self) -> None:
        types = all_benchmark_types()
        assert "rag" in types
        assert "chat" in types
        assert "general_llm" in types

    def test_at_least_six_types(self) -> None:
        assert len(all_benchmark_types()) >= 6
