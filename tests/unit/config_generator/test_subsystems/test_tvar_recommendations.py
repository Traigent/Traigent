"""Tests for config_generator.subsystems.tvar_recommendations."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.llm_backend import BudgetExhausted
from traigent.config_generator.subsystems.tvar_recommendations import (
    generate_recommendations,
)
from traigent.config_generator.types import TVarSpec


def _make_tvar(name: str) -> TVarSpec:
    return TVarSpec(name=name, range_type="Range", range_kwargs={"low": 0, "high": 1})


class TestGenerateRecommendations:
    def test_rag_agent_recommends_prompting_strategy(self) -> None:
        classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        recs = generate_recommendations([], classification=classification)
        names = {r.name for r in recs}
        assert "prompting_strategy" in names

    def test_rag_agent_recommends_retriever_type(self) -> None:
        classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        recs = generate_recommendations([], classification=classification)
        names = {r.name for r in recs}
        assert "retriever_type" in names

    def test_existing_tvars_are_excluded(self) -> None:
        classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        tvars = [_make_tvar("prompting_strategy")]
        recs = generate_recommendations(tvars, classification=classification)
        names = {r.name for r in recs}
        assert "prompting_strategy" not in names

    def test_general_llm_default(self) -> None:
        recs = generate_recommendations([])
        names = {r.name for r in recs}
        assert "prompting_strategy" in names

    def test_classification_agent_recommends_few_shot(self) -> None:
        classification = ClassificationResult(
            agent_type="classification",
            confidence=0.9,
            source="heuristic",
            reasoning="test",
        )
        recs = generate_recommendations([], classification=classification)
        names = {r.name for r in recs}
        assert "few_shot_count" in names

    def test_all_recs_have_range_type(self) -> None:
        classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        recs = generate_recommendations([], classification=classification)
        for rec in recs:
            assert rec.range_type in ("Range", "IntRange", "LogRange", "Choices")

    def test_all_recs_have_category(self) -> None:
        classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        recs = generate_recommendations([], classification=classification)
        for rec in recs:
            assert rec.category != ""

    def test_all_recs_have_impact(self) -> None:
        classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        recs = generate_recommendations([], classification=classification)
        for rec in recs:
            assert rec.impact_estimate in ("high", "medium", "low")

    def test_unknown_agent_type_returns_empty(self) -> None:
        classification = ClassificationResult(
            agent_type="unknown_type",
            confidence=0.5,
            source="heuristic",
            reasoning="test",
        )
        recs = generate_recommendations([], classification=classification)
        assert recs == []

    def test_no_duplicate_recommendations(self) -> None:
        classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        recs = generate_recommendations([], classification=classification)
        names = [r.name for r in recs]
        assert len(names) == len(set(names))


class TestLLMEnrichment:
    def test_llm_adds_recommendations(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = json.dumps(
            [
                {
                    "name": "custom_param",
                    "range_type": "Range",
                    "kwargs": {"low": 0.0, "high": 1.0},
                    "category": "custom",
                    "reasoning": "LLM suggested",
                    "impact": "high",
                }
            ]
        )
        recs = generate_recommendations([], llm=llm, source_code="def f(): pass")
        names = {r.name for r in recs}
        assert "custom_param" in names

    def test_llm_does_not_duplicate_preset_recs(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = json.dumps(
            [
                {
                    "name": "prompting_strategy",
                    "range_type": "Choices",
                    "kwargs": {"values": ["direct"]},
                    "category": "prompting",
                    "reasoning": "duplicate",
                    "impact": "medium",
                }
            ]
        )
        recs = generate_recommendations([], llm=llm, source_code="def f(): pass")
        ps_recs = [r for r in recs if r.name == "prompting_strategy"]
        assert len(ps_recs) == 1  # only preset, not LLM duplicate

    def test_llm_does_not_duplicate_existing_tvars(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = json.dumps(
            [
                {
                    "name": "temperature",
                    "range_type": "Range",
                    "kwargs": {"low": 0.0, "high": 2.0},
                    "category": "model",
                    "reasoning": "dup",
                    "impact": "medium",
                }
            ]
        )
        tvars = [_make_tvar("temperature")]
        recs = generate_recommendations(tvars, llm=llm, source_code="def f(): pass")
        names = {r.name for r in recs}
        assert "temperature" not in names

    def test_llm_budget_exhausted(self) -> None:
        llm = MagicMock()
        llm.complete.side_effect = BudgetExhausted(0.10)
        recs = generate_recommendations([], llm=llm)
        # Should still return preset recommendations
        assert len(recs) >= 1

    def test_llm_invalid_json(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = "not json"
        recs = generate_recommendations([], llm=llm)
        assert len(recs) >= 1  # presets still returned

    def test_llm_invalid_range_type_filtered(self) -> None:
        llm = MagicMock()
        llm.complete.return_value = json.dumps(
            [
                {
                    "name": "bad_param",
                    "range_type": "InvalidType",
                    "kwargs": {},
                    "category": "test",
                    "reasoning": "bad",
                    "impact": "low",
                }
            ]
        )
        recs = generate_recommendations([], llm=llm, source_code="def f(): pass")
        names = {r.name for r in recs}
        assert "bad_param" not in names
