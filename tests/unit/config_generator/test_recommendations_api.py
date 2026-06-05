"""Tests for the public TVAR recommendation catalog API."""

from __future__ import annotations

import copy
import json

import pytest

import traigent
from traigent.api.config_space import ConfigSpace
from traigent.api.functions import (
    list_recommendation_agent_types,
    recommend_configuration_space,
)
from traigent.config_generator import recommendations as recommendation_module
from traigent.config_generator.types import EvidenceRef

_EXPECTED_TOP_LEVEL_KEYS = [
    "schema_version",
    "catalog_version",
    "agent_type",
    "valid_agent_types",
    "filters",
    "caveat",
    "configuration_space",
    "recommendations",
]
_EXPECTED_RECOMMENDATION_KEYS = [
    "name",
    "range_type",
    "range_kwargs",
    "range_code",
    "suggested_values",
    "category",
    "kind",
    "impact",
    "confidence",
    "evidence_note",
    "effectuation_status",
    "effectuation_strategy",
    "apply_guidance",
    "catalog_entry_id",
]


def test_list_recommendation_agent_types_returns_public_catalog_types() -> None:
    assert list_recommendation_agent_types() == ("code_gen", "rag")


def test_recommend_configuration_space_valid_type_returns_structured_rows() -> None:
    result = recommend_configuration_space("rag")

    assert list(result) == _EXPECTED_TOP_LEVEL_KEYS
    assert result["schema_version"] == "1"
    assert result["catalog_version"] == "1.0.0"
    assert result["agent_type"] == "rag"
    assert "task-dependent" in result["caveat"]
    assert result["valid_agent_types"] == ["code_gen", "rag"]
    assert result["configuration_space"] == {"retrieval_k": {"low": 1, "high": 5}}
    config_space = ConfigSpace.from_decorator_args(
        configuration_space=result["configuration_space"]
    )
    assert set(config_space.tvars) == {"retrieval_k"}

    recommendations = result["recommendations"]
    assert len(recommendations) == 1
    retrieval_k = recommendations[0]
    assert list(retrieval_k) == _EXPECTED_RECOMMENDATION_KEYS
    assert retrieval_k["name"] == "retrieval_k"
    assert retrieval_k["range_type"] == "IntRange"
    assert retrieval_k["range_kwargs"] == {"low": 1, "high": 5}
    assert retrieval_k["impact"] == "medium"
    assert retrieval_k["confidence"] == "medium"
    assert retrieval_k["effectuation_status"] == "manual_guidance"
    assert retrieval_k["apply_guidance"]
    assert "answer_em delta +0.1" in retrieval_k["evidence_note"]


def test_recommend_configuration_space_unknown_type_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown agent_type 'planner'"):
        recommend_configuration_space("planner")


def test_recommend_configuration_space_impact_filter() -> None:
    result = recommend_configuration_space("code_gen", min_impact="high")

    names = {row["name"] for row in result["recommendations"]}
    assert names == {"schema_context"}
    assert set(result["configuration_space"]) == {"schema_context"}


def test_recommend_configuration_space_confidence_filter() -> None:
    result = recommend_configuration_space(
        "code_gen",
        min_impact="medium",
        min_confidence="medium",
    )

    names = {row["name"] for row in result["recommendations"]}
    assert names == {"schema_context", "evidence_usage"}


def test_recommend_configuration_space_filter_validation() -> None:
    with pytest.raises(ValueError, match="Unknown min_impact"):
        recommend_configuration_space("rag", min_impact="certain")
    with pytest.raises(ValueError, match="Unknown min_confidence"):
        recommend_configuration_space("rag", min_confidence="certain")


def test_recommend_configuration_space_json_round_trips() -> None:
    result = recommend_configuration_space("rag")

    assert json.loads(json.dumps(result)) == result


def test_negative_delta_evidence_is_not_positive_causal_evidence() -> None:
    negative_ref = EvidenceRef(
        scope="isolation",
        metric="answer_em",
        n=200,
        model="public/model",
        baseline="candidate_a",
        candidate="candidate_b",
        delta=-0.2,
    )

    assert recommendation_module._is_causal_positive_ref(negative_ref) is False
    assert recommendation_module._evidence_confidence((negative_ref,)) == "low"


def test_format_evidence_ref_projects_public_allowlist_only() -> None:
    note = recommendation_module._format_evidence_ref(
        {
            "scope": "isolation",
            "metric": "answer_em",
            "n": 10,
            "model": "public/model",
            "baseline": "1",
            "candidate": "5",
            "delta": 0.1,
            "limitations": ("single_slice",),
            "proprietary_tuning_signal": "internal-taxonomy-value",
        }
    )

    assert "answer_em delta +0.1" in note
    assert "internal-taxonomy-value" not in note
    assert "proprietary_tuning_signal" not in note


def test_unexpected_catalog_evidence_field_is_not_public(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = copy.deepcopy(recommendation_module._active_catalog_entries("rag")[0])
    entry["evidence_refs"][0]["proprietary_tuning_signal"] = "internal-taxonomy-value"

    def fake_catalog_entries(agent_type: str | None = None) -> list[dict[str, object]]:
        if agent_type in (None, "rag"):
            return [entry]
        return []

    monkeypatch.setattr(recommendation_module, "catalog_entries", fake_catalog_entries)

    payload = recommendation_module.recommend_configuration_space("rag")
    serialized = json.dumps(payload)

    assert "answer_em delta +0.1" in serialized
    assert "internal-taxonomy-value" not in serialized
    assert "proprietary_tuning_signal" not in serialized


def test_root_exports_public_query_api() -> None:
    old_name = "recommend" + "_config_space"

    assert "recommend_configuration_space" in traigent.__all__
    assert old_name not in traigent.__all__
    assert not hasattr(traigent, old_name)
    assert "list_recommendation_agent_types" in traigent.__all__
    assert traigent.list_recommendation_agent_types() == ("code_gen", "rag")
    assert traigent.recommend_configuration_space("rag")["agent_type"] == "rag"
