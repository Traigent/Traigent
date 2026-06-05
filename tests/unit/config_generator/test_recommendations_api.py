"""Tests for the public TVAR recommendation catalog API."""

from __future__ import annotations

import pytest

import traigent
from traigent.api.functions import (
    list_recommendation_agent_types,
    recommend_config_space,
)


def test_list_recommendation_agent_types_returns_public_catalog_types() -> None:
    assert list_recommendation_agent_types() == ("code_gen", "rag")


def test_recommend_config_space_valid_type_returns_structured_rows() -> None:
    result = recommend_config_space("rag")

    assert result["agent_type"] == "rag"
    assert "task-dependent" in result["caveat"]
    assert result["valid_agent_types"] == ["code_gen", "rag"]

    recommendations = result["recommendations"]
    assert len(recommendations) == 1
    retrieval_k = recommendations[0]
    assert retrieval_k["name"] == "retrieval_k"
    assert retrieval_k["range_type"] == "IntRange"
    assert retrieval_k["range_kwargs"] == {"low": 1, "high": 5}
    assert retrieval_k["impact"] == "medium"
    assert retrieval_k["confidence"] == "medium"
    assert retrieval_k["effectuation_status"] == "manual_guidance"
    assert retrieval_k["apply_guidance"]
    assert "answer_em delta +0.1" in retrieval_k["evidence_note"]


def test_recommend_config_space_unknown_type_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown agent_type 'planner'"):
        recommend_config_space("planner")


def test_recommend_config_space_impact_filter() -> None:
    result = recommend_config_space("code_gen", min_impact="high")

    names = {row["name"] for row in result["recommendations"]}
    assert names == {"schema_context"}


def test_recommend_config_space_confidence_filter() -> None:
    result = recommend_config_space(
        "code_gen",
        min_impact="medium",
        min_confidence="medium",
    )

    names = {row["name"] for row in result["recommendations"]}
    assert names == {"schema_context", "evidence_usage"}


def test_recommend_config_space_filter_validation() -> None:
    with pytest.raises(ValueError, match="Unknown min_impact"):
        recommend_config_space("rag", min_impact="certain")
    with pytest.raises(ValueError, match="Unknown min_confidence"):
        recommend_config_space("rag", min_confidence="certain")


def test_root_exports_public_query_api() -> None:
    assert "recommend_config_space" in traigent.__all__
    assert "list_recommendation_agent_types" in traigent.__all__
    assert traigent.list_recommendation_agent_types() == ("code_gen", "rag")
