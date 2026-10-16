"""SDK accepts the Backend's nullable-cost response members (unknown is not $0)."""

from __future__ import annotations

from typing import Any

import pytest

from traigent.cloud.analytics_client import (
    AnalyticsClientError,
    _project_observability_cohort_summary,
    _project_observability_tool_analysis,
)
from traigent.core_metrics.dtos import (
    ProjectUsageTrendPointDTO,
    UsageSummaryDTO,
)
from traigent.evaluation.dtos import EvaluatorRunDTO


def _tool_item(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "normalized_tool_id": "search",
        "trace_count": 2,
        "attempt_count": 3,
        "success_count": 2,
        "failure_count": 1,
        "retry_count": 0,
        "fallback_count": 0,
        "failure_rate": 0.3,
        "retry_rate": 0.0,
        "fallback_rate": 0.0,
        "p50_latency_ms": 10,
        "p95_latency_ms": 20,
        "total_cost_usd": 0.5,
        "issue_ids": [],
    }
    item.update(overrides)
    return item


def _tool_payload(**item_overrides: Any) -> dict[str, Any]:
    return {
        "project_id": "p1",
        "start_time": "2026-07-01T00:00:00Z",
        "end_time": "2026-07-02T00:00:00Z",
        "items": [_tool_item(**item_overrides)],
        "generated_at": "2026-07-02T00:00:00Z",
    }


def _cohort(**metric_overrides: Any) -> dict[str, Any]:
    metric: dict[str, Any] = {
        "metric": "cost_usd",
        "sample_count": 3,
        "mean": 0.1,
        "median": 0.1,
        "p95": 0.2,
    }
    metric.update(metric_overrides)
    return {"trace_count": 5, "metrics": [metric]}


# ---- tool analysis -------------------------------------------------------


def test_tool_analysis_accepts_new_members_and_null_total_cost() -> None:
    out = _project_observability_tool_analysis(
        _tool_payload(
            total_cost_usd=None,
            cost_status="partial",
            priced_cost_usd=0.25,
            unpriced_attempt_count=1,
        )
    )
    item = out["items"][0]
    assert item["total_cost_usd"] is None
    assert item["cost_status"] == "partial"
    assert item["priced_cost_usd"] == 0.25
    assert item["unpriced_attempt_count"] == 1


def test_tool_analysis_new_members_are_optional() -> None:
    out = _project_observability_tool_analysis(_tool_payload())
    assert "cost_status" not in out["items"][0]
    assert out["items"][0]["total_cost_usd"] == 0.5


def test_tool_analysis_null_priced_cost_allowed() -> None:
    out = _project_observability_tool_analysis(
        _tool_payload(total_cost_usd=None, cost_status="unpriced", priced_cost_usd=None)
    )
    assert out["items"][0]["priced_cost_usd"] is None


def test_tool_analysis_total_cost_key_still_required() -> None:
    payload = _tool_payload()
    del payload["items"][0]["total_cost_usd"]
    with pytest.raises(AnalyticsClientError):
        _project_observability_tool_analysis(payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"unexpected_key": 1},
        {"cost_status": "free"},
        {"cost_status": 3},
        {"priced_cost_usd": -0.1},
        {"priced_cost_usd": "1.0"},
        {"total_cost_usd": -1},
        {"total_cost_usd": "1"},
        {"total_cost_usd": True},
        {"unpriced_attempt_count": -1},
        {"unpriced_attempt_count": 1.5},
        {"unpriced_attempt_count": True},
    ],
)
def test_tool_analysis_still_rejects_bad_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(AnalyticsClientError):
        _project_observability_tool_analysis(_tool_payload(**overrides))


# ---- cohort compare ------------------------------------------------------


def test_cohort_accepts_cost_status_and_unknown_count() -> None:
    out = _project_observability_cohort_summary(
        _cohort(cost_status="partial", unknown_count=2), what="reference"
    )
    metric = out["metrics"][0]
    assert metric["cost_status"] == "partial"
    assert metric["unknown_count"] == 2


def test_cohort_new_members_are_optional_and_null_stats_kept() -> None:
    out = _project_observability_cohort_summary(
        _cohort(mean=None, median=None, p95=None), what="reference"
    )
    assert out["metrics"][0]["mean"] is None
    assert "unknown_count" not in out["metrics"][0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"extra": 1},
        {"cost_status": "bogus"},
        {"unknown_count": -1},
        {"unknown_count": "2"},
        {"unknown_count": 1.0},
        {"unknown_count": True},
    ],
)
def test_cohort_still_rejects_bad_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(AnalyticsClientError):
        _project_observability_cohort_summary(_cohort(**overrides), what="reference")


def test_cohort_required_metric_member_still_required() -> None:
    cohort = _cohort()
    del cohort["metrics"][0]["p95"]
    with pytest.raises(AnalyticsClientError):
        _project_observability_cohort_summary(cohort, what="reference")


# ---- DTOs ----------------------------------------------------------------


def test_evaluator_run_null_cost_is_none_not_zero() -> None:
    base = {"id": "r1", "target_type": "experiment_run", "cost_usd": None}
    assert EvaluatorRunDTO.from_dict(base).cost_usd is None
    assert EvaluatorRunDTO.from_dict({**base, "cost_usd": 0.25}).cost_usd == 0.25
    assert EvaluatorRunDTO.from_dict({**base, "cost_usd": 0}).cost_usd == 0.0


def _usage_summary(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment_runs": 1,
        "configuration_runs": 2,
        "priced_configuration_runs": 1,
        "unpriced_configuration_runs": 1,
        "total_cost_usd": 1.5,
        "avg_cost_usd": 0.75,
        "cost_source_breakdown": {
            "observed_usage": 1,
            "recorded_metrics": 0,
            "catalog_fallback": 0,
            "unknown_unpriced": 1,
        },
        "total_tokens": 10,
        "avg_latency_ms": 1.0,
        "p95_latency_ms": 2.0,
    }
    payload.update(overrides)
    return payload


def test_usage_summary_null_costs_become_none() -> None:
    dto = UsageSummaryDTO.from_dict(
        _usage_summary(total_cost_usd=None, avg_cost_usd=None)
    )
    assert dto.total_cost_usd is None
    assert dto.avg_cost_usd is None
    assert UsageSummaryDTO.from_dict(_usage_summary()).total_cost_usd == 1.5


def test_trend_point_null_cost_is_none_and_zero_stays_zero() -> None:
    base = {
        "bucket_start": "2026-07-01",
        "bucket_label": "Jul 1",
        "cost_usd": None,
        "total_tokens": 1,
        "avg_latency_ms": None,
        "configuration_runs": 1,
    }
    assert ProjectUsageTrendPointDTO.from_dict(base).cost_usd is None
    assert ProjectUsageTrendPointDTO.from_dict({**base, "cost_usd": 0}).cost_usd == 0.0


def test_trend_point_missing_cost_key_still_raises() -> None:
    base = {
        "bucket_start": "2026-07-01",
        "bucket_label": "Jul 1",
        "total_tokens": 1,
        "avg_latency_ms": None,
        "configuration_runs": 1,
    }
    with pytest.raises(KeyError):
        ProjectUsageTrendPointDTO.from_dict(base)


def test_trend_point_wrong_type_still_raises() -> None:
    base = {
        "bucket_start": "2026-07-01",
        "bucket_label": "Jul 1",
        "cost_usd": "abc",
        "total_tokens": 1,
        "avg_latency_ms": None,
        "configuration_runs": 1,
    }
    with pytest.raises(ValueError):
        ProjectUsageTrendPointDTO.from_dict(base)
