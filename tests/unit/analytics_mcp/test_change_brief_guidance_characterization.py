"""Snapshot characterization of ``_change_brief_guidance``.

The function turns already-projected evidence into bounded hypotheses and
recommendations. Each scenario below is compared with a recorded snapshot so
section order, truncation caps, type guards and exact wording stay pinned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from traigent.analytics_mcp.tools import _change_brief_guidance

SNAPSHOT = Path(__file__).parent / "fixtures" / "change_brief_guidance_snapshots.json"

_LINK_KEYS = ("resource_type", "resource_id", "resource_version")
_CONFORMANCE_KEYS = (
    "baseline_type",
    "baseline_variant_id",
    "sampled_trace_count",
    "conformance_rate",
    "alternate_trace_count",
    "alternate_rate",
    "alternate_variant_count",
    "sample_truncated",
)


def _completed(**extra: Any) -> dict[str, Any]:
    return {"analysis_status": "completed", **extra}


def _tool(i: int, failure: Any = 0, retry: Any = 0, fallback: Any = 0, **kw: Any):
    return {
        "normalized_tool_id": f"tool-{i}",
        "attempt_count": kw.get("attempt", 3),
        "failure_count": failure,
        "retry_count": retry,
        "fallback_count": fallback,
    }


def _aggregate_rec(i: int, category: Any = "tool_reliability") -> dict[str, Any]:
    return {
        "id": f"rec-{i}",
        "category": category,
        "priority": "high",
        "confidence": 0.8,
        "subject": f"subject-{i}",
        "evidence": {"n": i},
        "measurement": "compare error_rate",
    }


SLICE = {"trace_slice": {"items": [{"x": 1}]}}

SCENARIOS: dict[str, dict[str, Any]] = {
    "empty_evidence": {},
    "analysis_not_a_dict": {"trace_analysis": "oops", **SLICE},
    "analysis_incomplete_with_failure_code": {
        "trace_analysis": {"analysis_status": "failed", "failure_code": "E1"},
        **SLICE,
    },
    "completed_nothing_else": {"trace_analysis": _completed(), **SLICE},
    "tool_summaries_mixed": {
        "trace_analysis": _completed(
            tool_summaries=[
                "not-a-dict",
                _tool(1, failure=2),
                _tool(2),
                _tool(3, retry=1.9),
                _tool(4, fallback=True),
                _tool(5, failure=float("nan"), retry=float("inf")),
                _tool(6, failure=-4),
                _tool(7, failure="3"),
                _tool(8, fallback=1),
            ]
        ),
        **SLICE,
    },
    "tool_summaries_capped_at_20": {
        "trace_analysis": _completed(
            tool_summaries=[_tool(i, failure=1) for i in range(30)]
        ),
        **SLICE,
    },
    "tool_summaries_not_a_list": {
        "trace_analysis": _completed(tool_summaries={"a": 1}),
        **SLICE,
    },
    "repeat_groups": {
        "trace_analysis": _completed(
            repeat_groups=[{"id": "g1"}, "junk", {"id": "g2"}, {"name": "no-id"}]
        ),
        **SLICE,
    },
    "repeat_groups_capped": {
        "trace_analysis": _completed(
            repeat_groups=[{"id": f"g{i}"} for i in range(25)]
        ),
        **SLICE,
    },
    "repeat_groups_empty_ignored": {
        "trace_analysis": _completed(repeat_groups=[]),
        **SLICE,
    },
    "issue_ids": {
        "trace_analysis": _completed(issue_ids=[f"i{i}" for i in range(25)]),
        **SLICE,
    },
    "issue_ids_empty_ignored": {
        "trace_analysis": _completed(issue_ids=[]),
        **SLICE,
    },
    "lineage_links": {
        "trace_analysis": _completed(),
        "related_changes": {
            "links": [
                {
                    "resource_type": "prompt",
                    "resource_id": "p1",
                    "resource_version": "3",
                    "extra": "dropped",
                },
                "junk",
                {"resource_type": "release"},
            ]
        },
        **SLICE,
    },
    "lineage_not_a_dict": {
        "trace_analysis": _completed(),
        "related_changes": ["x"],
        **SLICE,
    },
    "lineage_empty_links": {
        "trace_analysis": _completed(),
        "related_changes": {"links": []},
        **SLICE,
    },
    "conformance_with_alternates": {
        "trace_analysis": _completed(),
        "analysis_insights": {
            "conformance": {
                "baseline_type": "dominant",
                "baseline_variant_id": "v1",
                "sampled_trace_count": 10,
                "conformance_rate": 0.7,
                "alternate_trace_count": 3,
                "alternate_rate": 0.3,
                "alternate_variant_count": 2,
                "sample_truncated": False,
                "ignored": "x",
            }
        },
        **SLICE,
    },
    "conformance_zero_alternates_ignored": {
        "trace_analysis": _completed(),
        "analysis_insights": {"conformance": {"alternate_trace_count": 0}},
        **SLICE,
    },
    "conformance_bool_alternates_ignored": {
        "trace_analysis": _completed(),
        "analysis_insights": {"conformance": {"alternate_trace_count": True}},
        **SLICE,
    },
    "aggregate_recommendations": {
        "trace_analysis": _completed(),
        "analysis_insights": {
            "recommendations": [
                _aggregate_rec(1, "tool_reliability"),
                "junk",
                _aggregate_rec(2, 7),
                _aggregate_rec(3, "unknown_category"),
                _aggregate_rec(4, "retry_policy"),
                _aggregate_rec(5, "tool_routing"),
                _aggregate_rec(6, "recurring_issue"),
                _aggregate_rec(7, "behavioral_variation"),
            ]
        },
        **SLICE,
    },
    "aggregate_recommendations_capped": {
        "trace_analysis": _completed(),
        "analysis_insights": {
            "recommendations": [_aggregate_rec(i) for i in range(30)]
        },
        **SLICE,
    },
    "insights_not_a_dict": {
        "trace_analysis": _completed(),
        "analysis_insights": ["x"],
        **SLICE,
    },
    "cohort_present": {
        "trace_analysis": _completed(),
        "cohort_comparison": {"deltas": []},
        **SLICE,
    },
    "cohort_not_a_dict": {
        "trace_analysis": _completed(),
        "cohort_comparison": "x",
        **SLICE,
    },
    "no_projection_items": {
        "trace_analysis": _completed(),
        "trace_slice": {"items": []},
    },
    "projection_not_a_dict": {"trace_analysis": _completed(), "trace_slice": [1]},
    "everything_and_caps": {
        "trace_analysis": {
            "analysis_status": "partial",
            "failure_code": "E9",
            "tool_summaries": [_tool(i, failure=1) for i in range(20)],
            "repeat_groups": [{"id": "g"}],
            "issue_ids": ["a", "b"],
        },
        "related_changes": {"links": [{"resource_type": "toolset"}]},
        "analysis_insights": {
            "conformance": {"alternate_trace_count": 2},
            "recommendations": [_aggregate_rec(i) for i in range(20)],
        },
        "cohort_comparison": {},
        "trace_slice": {"items": []},
    },
}


def _normalise(result: tuple[list[dict[str, Any]], list[dict[str, Any]]]) -> Any:
    return json.loads(
        json.dumps({"hypotheses": result[0], "recommendations": result[1]})
    )


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_change_brief_guidance_snapshot(name):
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))[name]
    assert _normalise(_change_brief_guidance(SCENARIOS[name])) == expected


def test_outputs_are_capped_at_25_each():
    hypotheses, recommendations = _change_brief_guidance(
        {
            "trace_analysis": {
                "analysis_status": "failed",
                "tool_summaries": [_tool(i, failure=1) for i in range(20)],
            },
            "analysis_insights": {
                "recommendations": [_aggregate_rec(i) for i in range(20)]
            },
        }
    )
    assert len(hypotheses) == 21
    assert len(recommendations) == 25
