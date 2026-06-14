"""Tests for metric/evaluator recommendation API helpers."""

from __future__ import annotations

import json

import pytest

from traigent.evaluators import (
    EVAL_RECOMMENDATION_CAVEAT,
    list_eval_recommendation_task_types,
    recommend_evaluator,
    recommend_metrics,
)


def test_list_eval_recommendation_task_types_returns_catalog_types() -> None:
    assert list_eval_recommendation_task_types() == ("code_gen", "general", "rag")


def test_recommend_metrics_filters_task_measure_type_and_confidence() -> None:
    payload = recommend_metrics(
        "rag",
        measure_types=["accuracy"],
        min_confidence="high",
    )

    assert payload["schema_version"] == "1"
    assert payload["catalog_version"] == "1.0.0"
    assert payload["task_type"] == "rag"
    assert payload["caveat"] == EVAL_RECOMMENDATION_CAVEAT
    assert "task-dependent" in payload["caveat"]
    assert payload["filters"] == {
        "measure_types": ["accuracy"],
        "min_confidence": "high",
    }
    assert {row["metric"]["name"] for row in payload["recommendations"]} == {
        "exact_match",
        "token_f1",
    }
    assert all(row["measure_type"] == "accuracy" for row in payload["recommendations"])
    assert all(row["confidence"] == "high" for row in payload["recommendations"])
    assert payload["metric_functions_stub"] == {
        "exact_match": "accuracy",
        "token_f1": None,
    }


def test_recommend_metrics_unknown_task_type_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown task_type 'planner'"):
        recommend_metrics("planner")
    with pytest.raises(ValueError, match="Valid evaluator recommendation task types"):
        recommend_metrics("planner")


def test_recommend_metrics_rejects_unknown_filters() -> None:
    with pytest.raises(ValueError, match="Unknown measure_types"):
        recommend_metrics("rag", measure_types=["magic"])
    with pytest.raises(ValueError, match="Unknown min_confidence"):
        recommend_metrics("rag", min_confidence="certain")


def test_recommend_metrics_json_round_trips() -> None:
    payload = recommend_metrics("general")

    assert json.loads(json.dumps(payload)) == payload


def test_recommend_evaluator_prefers_deterministic_entries() -> None:
    payload = recommend_evaluator("rag", prefer_deterministic=True)
    methods = [row["evaluation_method"] for row in payload["recommendations"]]

    first_non_deterministic = next(
        index for index, method in enumerate(methods) if method != "deterministic"
    )
    assert first_non_deterministic > 0
    assert all(
        method == "deterministic" for method in methods[:first_non_deterministic]
    )
    assert all("evaluator_binding" in row for row in payload["recommendations"])
    assert all("cost_note" in row for row in payload["recommendations"])


def test_recommend_evaluator_cost_tier_filter() -> None:
    payload = recommend_evaluator("general", max_cost_tier="low")

    assert payload["filters"]["max_cost_tier"] == "low"
    assert payload["recommendations"]
    assert {row["cost_tier"] for row in payload["recommendations"]} == {"low"}
