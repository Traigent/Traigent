"""Tests for config_generator.subsystems.tvar_recommendations."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import MagicMock

from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.llm_backend import BudgetExhausted
from traigent.config_generator.subsystems.tvar_recommendations import (
    generate_recommendations,
)
from traigent.config_generator.types import AutoConfigResult, TVarSpec

_READY_MADE_RECS = {
    "schema_context",
    "evidence_usage",
    "retrieval_k",
    "fewshot_selector",
    "generation_path",
    "fewshot_k",
    "candidate_count",
    "repair_policy",
}

_CODE_GEN_STRUCTURAL_RECS = _READY_MADE_RECS - {"retrieval_k"}


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

    def test_rag_agent_recommends_retriever(self) -> None:
        classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        recs = generate_recommendations([], classification=classification)
        names = {r.name for r in recs}
        assert "retriever" in names
        assert "retriever_type" not in names
        assert "reranker" in names
        assert "reranker_model" not in names

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

    def test_ready_made_structural_recommendations_present(self) -> None:
        rag_classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        code_gen_classification = ClassificationResult(
            agent_type="code_gen", confidence=0.9, source="heuristic", reasoning="test"
        )

        rag_recs = generate_recommendations([], classification=rag_classification)
        code_gen_recs = generate_recommendations(
            [], classification=code_gen_classification
        )
        recs_by_name = {r.name: r for r in [*rag_recs, *code_gen_recs]}

        assert _READY_MADE_RECS <= set(recs_by_name)
        for name in _READY_MADE_RECS:
            rec = recs_by_name[name]
            assert rec.evidence_refs, f"{name} missing evidence"
            assert rec.apply_guidance.strip(), f"{name} missing guidance"
            assert rec.range_type in ("Range", "IntRange", "LogRange", "Choices")

    def test_sql_recommendations_are_under_code_gen(self) -> None:
        rag_classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        code_gen_classification = ClassificationResult(
            agent_type="code_gen", confidence=0.9, source="heuristic", reasoning="test"
        )

        rag_names = {
            r.name for r in generate_recommendations([], classification=rag_classification)
        }
        code_gen_names = {
            r.name
            for r in generate_recommendations([], classification=code_gen_classification)
        }

        assert "retrieval_k" in rag_names
        assert _CODE_GEN_STRUCTURAL_RECS <= code_gen_names
        assert _CODE_GEN_STRUCTURAL_RECS.isdisjoint(rag_names)

    def test_ready_made_evidence_details(self) -> None:
        rag_classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        code_gen_classification = ClassificationResult(
            agent_type="code_gen", confidence=0.9, source="heuristic", reasoning="test"
        )
        recs = [
            *generate_recommendations([], classification=rag_classification),
            *generate_recommendations([], classification=code_gen_classification),
        ]
        recs_by_name = {r.name: r for r in recs}

        schema_context = recs_by_name["schema_context"]
        assert len(schema_context.evidence_refs) == 2
        # Public-safe provenance only: internal artifact paths / run IDs are
        # intentionally excluded from the public SDK contract.
        assert not hasattr(schema_context.evidence_refs[0], "artifact_path")
        assert not hasattr(schema_context.evidence_refs[0], "run_id")
        assert schema_context.evidence_refs[0].scope == "isolation"
        assert schema_context.evidence_refs[0].delta == 0.40
        assert schema_context.evidence_refs[1].candidate == "full_ddl_fk"
        assert schema_context.impact_estimate == "high"

        retrieval_k = recs_by_name["retrieval_k"]
        assert retrieval_k.evidence_refs[0].scope == "isolation"
        assert retrieval_k.evidence_refs[0].metric == "answer_em"
        assert retrieval_k.evidence_refs[0].baseline == 1
        assert retrieval_k.evidence_refs[0].candidate == 5
        assert retrieval_k.impact_estimate == "medium"

        fewshot_selector = recs_by_name["fewshot_selector"]
        assert fewshot_selector.evidence_refs[0].limitations == (
            "observational_not_causal",
        )
        assert fewshot_selector.impact_estimate == "medium"

        generation_path = recs_by_name["generation_path"]
        assert len(generation_path.evidence_refs) == 2
        assert all(ref.delta == 0.0 for ref in generation_path.evidence_refs)
        assert all(
            ref.limitations == ("low_or_zero_in_isolation", "gains_are_joint")
            for ref in generation_path.evidence_refs
        )
        assert generation_path.impact_estimate == "low"


class TestCLIJsonRecommendationStability:
    def test_recommendation_keys_unchanged(self) -> None:
        from traigent.cli.generate_config_command import _output_json

        classification = ClassificationResult(
            agent_type="code_gen", confidence=0.9, source="heuristic", reasoning="test"
        )
        rec = next(
            r
            for r in generate_recommendations([], classification=classification)
            if r.name == "schema_context"
        )
        assert rec.evidence_refs
        assert rec.apply_guidance

        buf = StringIO()
        with redirect_stdout(buf):
            _output_json(AutoConfigResult(recommendations=(rec,)))

        data = json.loads(buf.getvalue())
        assert set(data["recommendations"][0]) == {
            "name",
            "range_code",
            "category",
            "impact",
            "reasoning",
        }
        assert "evidence_refs" not in data["recommendations"][0]
        assert "apply_guidance" not in data["recommendations"][0]


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
