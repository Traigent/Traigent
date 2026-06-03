"""Tests for config_generator.subsystems.tvar_recommendations."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import MagicMock

from traigent.cloud.client import PriorsBundle
from traigent.config_generator.agent_classifier import ClassificationResult
from traigent.config_generator.catalog import (
    catalog_entries,
    entry_to_recommendation,
    load_catalog,
)
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

_CATALOG_ENTRY_KINDS = {
    "rag.retrieval_k.v1": "cardinality",
    "code_gen.schema_context.v1": "topology",
    "code_gen.evidence_usage.v1": "topology",
    "code_gen.fewshot_selector.v1": "topology",
    "code_gen.generation_path.v1": "topology",
    "code_gen.fewshot_k.v1": "cardinality",
    "code_gen.candidate_count.v1": "cardinality",
    "code_gen.repair_policy.v1": "topology",
}


def _make_tvar(name: str) -> TVarSpec:
    return TVarSpec(name=name, range_type="Range", range_kwargs={"low": 0, "high": 1})


def _classification(agent_type: str) -> ClassificationResult:
    return ClassificationResult(
        agent_type=agent_type,
        confidence=0.9,
        source="heuristic",
        reasoning="test",
    )


def _value_prior_row(
    tvar_name: str,
    values: list[dict],
    *,
    support_n: int = 100,
    confidence: float = 0.9,
) -> dict:
    return {
        "schema_version": "1.0.0",
        "tvar_name": tvar_name,
        "metric": "accuracy",
        "value_priors": values,
        "support_n": support_n,
        "confidence": confidence,
    }


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
        # Public-safe provenance only: no internal artifact paths / run IDs.
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

    def test_catalog_backed_recommendation_lists_are_stable(self) -> None:
        rag_classification = ClassificationResult(
            agent_type="rag", confidence=0.9, source="heuristic", reasoning="test"
        )
        code_gen_classification = ClassificationResult(
            agent_type="code_gen", confidence=0.9, source="heuristic", reasoning="test"
        )

        assert [
            rec.name
            for rec in generate_recommendations([], classification=rag_classification)
        ] == [
            "prompting_strategy",
            "retriever",
            "retrieval_k",
            "chunk_size",
            "reranker",
            "context_format",
        ]
        assert [
            rec.name
            for rec in generate_recommendations(
                [], classification=code_gen_classification
            )
        ] == [
            "prompting_strategy",
            "schema_context",
            "evidence_usage",
            "fewshot_selector",
            "generation_path",
            "fewshot_k",
            "candidate_count",
            "repair_policy",
        ]


class TestLearnedPriorAugmentation:
    def test_high_support_priors_reorder_and_annotate_recommended_values(
        self,
    ) -> None:
        # Backend returns value_priors already gated AND ordered by score desc;
        # the SDK consumes them in that server order without re-sorting.
        bundle = PriorsBundle(
            value_priors=(
                _value_prior_row(
                    "schema_context",
                    [
                        {
                            "value": "linked_top10",
                            "score": 0.91,
                            "support_n": 91,
                            "confidence": 0.87,
                        },
                        {
                            "value": "full_ddl_fk",
                            "score": 0.52,
                            "support_n": 75,
                            "confidence": 0.82,
                        },
                    ],
                ),
            ),
            correlations=(),
        )

        recs = generate_recommendations(
            [],
            classification=_classification("code_gen"),
            priors_bundle=bundle,
        )
        rec = next(rec for rec in recs if rec.name == "schema_context")

        assert rec.recommended_values == ("linked_top10", "full_ddl_fk")
        assert rec.range_kwargs["values"] == [
            "linked_top10",
            "full_ddl_fk",
            "none",
            "linked_top6",
        ]

    def test_sdk_consumes_server_gated_priors_without_regating(self) -> None:
        # Gating is server-side now: the SDK does NOT apply its own
        # support/confidence thresholds. Whatever the backend returns (already
        # gated) is consumed as-is. A backend that returned only this row means
        # it passed the backend gate, so the SDK applies it.
        classification = _classification("code_gen")
        server_bundle = PriorsBundle(
            value_priors=(
                _value_prior_row(
                    "schema_context",
                    [
                        {
                            "value": "linked_top10",
                            "score": 0.99,
                            "support_n": 2,
                            "confidence": 0.95,
                        }
                    ],
                    support_n=2,
                    confidence=0.95,
                ),
            ),
            correlations=(),
        )

        recs = generate_recommendations(
            [],
            classification=classification,
            priors_bundle=server_bundle,
        )
        rec = next(rec for rec in recs if rec.name == "schema_context")
        # SDK trusts the server's gating decision; the value is applied.
        assert rec.recommended_values == ("linked_top10",)

    def test_no_priors_keeps_phase_one_catalog_output(self) -> None:
        recs = generate_recommendations(
            [],
            classification=_classification("rag"),
            priors_bundle=PriorsBundle.empty(),
        )

        assert [rec.name for rec in recs] == [
            "prompting_strategy",
            "retriever",
            "retrieval_k",
            "chunk_size",
            "reranker",
            "context_format",
        ]
        assert all(not rec.recommended_values for rec in recs)


class TestTVarCatalog:
    def test_catalog_loads_and_validates_all_ready_made_entries(self) -> None:
        entries = load_catalog()

        assert len(entries) == 8
        assert {entry["entry_id"] for entry in entries} == set(_CATALOG_ENTRY_KINDS)
        for entry in entries:
            assert entry["kind"] == _CATALOG_ENTRY_KINDS[entry["entry_id"]]
            assert entry["status"] == "active"
            assert entry["schema_version"] == "1.0.0"
            assert entry["version"] == "1.0.0"
            assert entry["evidence_refs"]
            assert entry["apply_guidance"].strip()

        by_id = {entry["entry_id"]: entry for entry in entries}
        assert (
            by_id["code_gen.candidate_count.v1"]["effectuation_status"]
            == "executable"
        )
        assert (
            by_id["code_gen.candidate_count.v1"]["effectuation_strategy"]
            == "self_consistency"
        )
        assert {
            entry["entry_id"]
            for entry in entries
            if entry["effectuation_status"] == "manual_guidance"
        } == set(_CATALOG_ENTRY_KINDS) - {"code_gen.candidate_count.v1"}

    def test_catalog_filters_by_agent_type(self) -> None:
        assert [entry["entry_id"] for entry in catalog_entries("rag")] == [
            "rag.retrieval_k.v1"
        ]
        assert {entry["entry_id"] for entry in catalog_entries("code_gen")} == (
            set(_CATALOG_ENTRY_KINDS) - {"rag.retrieval_k.v1"}
        )

    def test_entry_to_recommendation_round_trips_catalog_fields(self) -> None:
        entry = next(
            entry
            for entry in load_catalog()
            if entry["entry_id"] == "code_gen.schema_context.v1"
        )

        rec = entry_to_recommendation(entry)

        assert rec.entry_id == entry["entry_id"]
        assert rec.name == entry["name"]
        assert rec.category == entry["category"]
        assert rec.reasoning == entry["reasoning"]
        assert rec.impact_estimate == entry["impact_estimate"]
        assert rec.apply_guidance == entry["apply_guidance"]
        assert len(rec.evidence_refs) == len(entry["evidence_refs"])
        # Public-safe provenance only (no internal artifact_path / run_id).
        assert not hasattr(rec.evidence_refs[0], "artifact_path")
        assert rec.evidence_refs[0].scope == entry["evidence_refs"][0]["scope"]

    def test_count_evidence_values_keep_public_types(self) -> None:
        entry = next(
            entry for entry in load_catalog() if entry["entry_id"] == "rag.retrieval_k.v1"
        )

        rec = entry_to_recommendation(entry)

        assert rec.evidence_refs[0].baseline == 1
        assert rec.evidence_refs[0].candidate == 5


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

    def test_recommendation_keys_unchanged_with_prior_hints(self) -> None:
        from traigent.cli.generate_config_command import _output_json

        bundle = PriorsBundle(
            value_priors=(
                _value_prior_row(
                    "schema_context",
                    [
                        {
                            "value": "linked_top10",
                            "score": 0.91,
                            "support_n": 91,
                            "confidence": 0.87,
                        }
                    ],
                ),
            ),
            correlations=(),
        )
        rec = next(
            r
            for r in generate_recommendations(
                [],
                classification=_classification("code_gen"),
                priors_bundle=bundle,
            )
            if r.name == "schema_context"
        )
        assert rec.recommended_values == ("linked_top10",)

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
        assert "recommended_values" not in data["recommendations"][0]


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
