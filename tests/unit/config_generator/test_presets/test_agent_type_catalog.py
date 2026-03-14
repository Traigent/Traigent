"""Tests for config_generator.presets.agent_type_catalog."""

from __future__ import annotations

from traigent.config_generator.presets.agent_type_catalog import (
    all_agent_types,
    get_safety_presets,
)


class TestGetSafetyPresets:
    def test_rag_has_faithfulness(self) -> None:
        presets = get_safety_presets("rag")
        metrics = {p.metric_name for p in presets}
        assert "faithfulness" in metrics
        assert "hallucination_rate" in metrics

    def test_chat_has_toxicity(self) -> None:
        presets = get_safety_presets("chat")
        metrics = {p.metric_name for p in presets}
        assert "toxicity_score" in metrics
        assert "bias_score" in metrics

    def test_code_gen_has_safety_score(self) -> None:
        presets = get_safety_presets("code_gen")
        metrics = {p.metric_name for p in presets}
        assert "safety_score" in metrics

    def test_summarization_presets(self) -> None:
        presets = get_safety_presets("summarization")
        assert len(presets) >= 1
        metrics = {p.metric_name for p in presets}
        assert "faithfulness" in metrics

    def test_classification_has_bias(self) -> None:
        presets = get_safety_presets("classification")
        metrics = {p.metric_name for p in presets}
        assert "bias_score" in metrics

    def test_general_llm_has_hallucination(self) -> None:
        presets = get_safety_presets("general_llm")
        metrics = {p.metric_name for p in presets}
        assert "hallucination_rate" in metrics

    def test_unknown_type_returns_empty(self) -> None:
        assert get_safety_presets("unknown_type") == []

    def test_all_presets_have_valid_operators(self) -> None:
        for agent_type in all_agent_types():
            for preset in get_safety_presets(agent_type):
                assert preset.operator in (">=", "<=")
                assert 0.0 <= preset.threshold <= 1.0
                assert preset.agent_type == agent_type
                assert preset.source == "preset"
                assert preset.reasoning != ""


class TestAllAgentTypes:
    def test_returns_frozenset(self) -> None:
        types = all_agent_types()
        assert isinstance(types, frozenset)

    def test_contains_core_types(self) -> None:
        types = all_agent_types()
        assert "rag" in types
        assert "chat" in types
        assert "general_llm" in types

    def test_at_least_6_types(self) -> None:
        assert len(all_agent_types()) >= 6
