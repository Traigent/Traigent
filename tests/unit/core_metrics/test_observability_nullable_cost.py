"""Observability dashboard DTOs and trace-search projection accept nullable cost."""

from __future__ import annotations

from typing import Any

import pytest

from traigent.cloud.analytics_client import (
    AnalyticsClientError,
    _project_observability_trace_search,
)
from traigent.core_metrics.dtos import (
    ObservabilityActivityTrendPointDTO,
    ObservabilitySummaryCardsDTO,
    ObservabilityTopTraceDTO,
)


def _cards(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sessions_in_range": 1,
        "traces_in_range": 2,
        "observations_in_range": 3,
        "bookmarked_traces_in_range": 0,
        "published_traces_in_range": 0,
        "commented_traces_in_range": 0,
        "total_cost_usd_in_range": 1.5,
        "total_tokens_in_range": 10,
    }
    payload.update(overrides)
    return payload


def _trace(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trace_id": "t1",
        "session_id": None,
        "name": "n",
        "status": "completed",
        "observation_count": 1,
        "total_cost_usd": 0.5,
        "total_tokens": 1,
        "total_latency_ms": 1,
        "is_bookmarked": False,
        "is_published": False,
        "started_at": None,
        "privacy_classification": "aggregate_safe",
    }
    payload.update(overrides)
    return payload


def test_old_numeric_summary_payload_still_reads() -> None:
    dto = ObservabilitySummaryCardsDTO.from_dict(_cards())
    assert dto.total_cost_usd_in_range == 1.5
    assert dto.total_cost_usd == 1.5
    assert dto.cost_status is None
    assert dto.priced_cost_usd is None
    assert dto.unpriced_trace_count is None


def test_priced_summary() -> None:
    dto = ObservabilitySummaryCardsDTO.from_dict(
        _cards(
            total_cost_usd=1.5,
            cost_status="priced",
            priced_cost_usd=1.5,
            unpriced_trace_count=0,
        )
    )
    assert (dto.total_cost_usd, dto.cost_status, dto.unpriced_trace_count) == (
        1.5,
        "priced",
        0,
    )


def test_partial_summary_keeps_null_total_and_lower_bound() -> None:
    dto = ObservabilitySummaryCardsDTO.from_dict(
        _cards(
            total_cost_usd=None,
            total_cost_usd_in_range=None,
            cost_status="partial",
            priced_cost_usd=0.75,
            unpriced_trace_count=1,
        )
    )
    assert dto.total_cost_usd is None
    assert dto.total_cost_usd_in_range is None
    assert dto.priced_cost_usd == 0.75
    assert dto.unpriced_trace_count == 1


def test_unpriced_summary_null_is_not_zero() -> None:
    dto = ObservabilitySummaryCardsDTO.from_dict(
        _cards(
            total_cost_usd=None,
            total_cost_usd_in_range=None,
            cost_status="unpriced",
            priced_cost_usd=0.0,
            unpriced_trace_count=2,
        )
    )
    assert dto.total_cost_usd is None
    assert dto.total_cost_usd_in_range is None
    assert dto.priced_cost_usd == 0.0


def test_not_applicable_summary() -> None:
    dto = ObservabilitySummaryCardsDTO.from_dict(
        _cards(
            total_cost_usd=None,
            total_cost_usd_in_range=None,
            cost_status="not_applicable",
            priced_cost_usd=0.0,
            unpriced_trace_count=0,
        )
    )
    assert dto.cost_status == "not_applicable"
    assert dto.total_cost_usd is None


def test_deprecated_alias_only_null_payload_reads() -> None:
    dto = ObservabilitySummaryCardsDTO.from_dict(_cards(total_cost_usd_in_range=None))
    assert dto.total_cost_usd_in_range is None
    assert dto.total_cost_usd is None


def test_trend_point_null_and_numeric() -> None:
    base = {
        "bucket_start": "2026-03-12T00:00:00+00:00",
        "bucket_label": "2026-03-12",
        "traces": 1,
        "observations": 1,
        "total_tokens": 1,
    }
    assert (
        ObservabilityActivityTrendPointDTO.from_dict(
            {**base, "total_cost_usd": None}
        ).total_cost_usd
        is None
    )
    assert (
        ObservabilityActivityTrendPointDTO.from_dict(
            {**base, "total_cost_usd": 0.42}
        ).total_cost_usd
        == 0.42
    )


def test_top_trace_variants() -> None:
    old = ObservabilityTopTraceDTO.from_dict(_trace())
    assert old.total_cost_usd == 0.5 and old.cost_status is None
    partial = ObservabilityTopTraceDTO.from_dict(
        _trace(
            total_cost_usd=None,
            cost_status="partial",
            priced_cost_usd=0.2,
            unpriced_observation_count=3,
        )
    )
    assert partial.total_cost_usd is None
    assert partial.priced_cost_usd == 0.2
    assert partial.unpriced_observation_count == 3
    unpriced = ObservabilityTopTraceDTO.from_dict(
        _trace(
            total_cost_usd=None,
            cost_status="unpriced",
            priced_cost_usd=None,
            unpriced_observation_count=None,
        )
    )
    assert unpriced.total_cost_usd is None and unpriced.priced_cost_usd is None


def test_negative_control_missing_required_cost_key_still_raises() -> None:
    payload = _cards()
    del payload["total_cost_usd_in_range"]
    # alias absent and no total_cost_usd -> unknown, not a crash: reads as None
    assert ObservabilitySummaryCardsDTO.from_dict(payload).total_cost_usd is None
    with pytest.raises(ValueError):
        ObservabilitySummaryCardsDTO.from_dict(_cards(total_cost_usd="abc"))
    with pytest.raises(KeyError):
        ObservabilitySummaryCardsDTO.from_dict(
            {k: v for k, v in _cards().items() if k != "traces_in_range"}
        )


def _search(item: dict[str, Any]) -> dict[str, Any]:
    return {"items": [item], "page": 1, "per_page": 10, "total": 1, "has_more": False}


@pytest.mark.parametrize(
    "extra",
    [
        {"total_cost_usd": 0.5, "cost_status": "priced"},
        {"total_cost_usd": 0.5},
        {
            "total_cost_usd": None,
            "cost_status": "partial",
            "priced_cost_usd": 0.1,
            "unpriced_observation_count": 2,
        },
        {"total_cost_usd": None, "cost_status": "unpriced"},
        {"total_cost_usd": None, "cost_status": "not_applicable"},
    ],
)
def test_trace_search_accepts_status_aware_cost(extra: dict[str, Any]) -> None:
    out = _project_observability_trace_search(_search({"id": "trace-1", **extra}))
    item = out["items"][0]
    for key, value in extra.items():
        assert item[key] == value


@pytest.mark.parametrize(
    "extra",
    [
        {"total_cost_usd": -1},
        {"total_cost_usd": "1.0"},
        {"total_cost_usd": True},
        {"total_cost_usd": None, "priced_cost_usd": -0.1},
        {"total_cost_usd": None, "cost_status": "bogus"},
        {"total_cost_usd": None, "unpriced_observation_count": -1},
    ],
)
def test_trace_search_negative_control_still_rejects(extra: dict[str, Any]) -> None:
    with pytest.raises(AnalyticsClientError):
        _project_observability_trace_search(_search({"id": "trace-1", **extra}))
