"""LangChain handler reports the known spend as a floor when a call is unmeasured (SDK-40).

Parity with traigent-js ``toMeasuresDict``: ``{cost_unmeasured: 1,
cost_lower_bound: <priced part>}`` instead of a total, never $0.
"""

from __future__ import annotations

import pytest

from traigent.integrations.langchain.handler import (
    LLMCallMetrics,
    TraigentHandlerMetrics,
)
from traigent.utils.trial_costs import is_cost_unmeasured, known_spend_lower_bound


def _call(call_id: str, cost: float, *, unmeasured: bool = False) -> LLMCallMetrics:
    return LLMCallMetrics(
        call_id=call_id,
        node_name="grader",
        model="m",
        start_time=0.0,
        end_time=1.0,
        cost=cost,
        cost_unmeasured=unmeasured,
    )


def _metrics(*calls: LLMCallMetrics) -> TraigentHandlerMetrics:
    return TraigentHandlerMetrics(
        trace_id="t",
        llm_calls=list(calls),
        total_cost=sum(c.cost for c in calls),
        cost_unmeasured=any(c.cost_unmeasured for c in calls),
    )


def test_unmeasured_call_reports_priced_part_as_cost_lower_bound_not_total() -> None:
    measures = _metrics(
        _call("a", 0.25), _call("b", 0.0, unmeasured=True)
    ).to_measures_dict()
    assert measures["cost_unmeasured"] == 1.0
    assert measures["cost_lower_bound"] == pytest.approx(0.25)
    assert "total_cost" not in measures


def test_floor_key_carries_the_same_prefix_as_the_unmeasured_flag() -> None:
    measures = _metrics(
        _call("a", 0.25), _call("b", 0.0, unmeasured=True)
    ).to_measures_dict(prefix="langchain_")
    assert measures["langchain_cost_unmeasured"] == 1.0
    assert measures["langchain_cost_lower_bound"] == pytest.approx(0.25)
    assert "langchain_total_cost" not in measures


def test_unprefixed_measures_are_readable_by_trial_cost_consumers() -> None:
    measures = _metrics(
        _call("a", 0.25), _call("b", 0.0, unmeasured=True)
    ).to_measures_dict()
    assert is_cost_unmeasured(measures)
    assert known_spend_lower_bound(measures) == pytest.approx(0.25)


def test_fully_measured_calls_report_total_cost_and_no_floor() -> None:
    measures = _metrics(_call("a", 0.25), _call("b", 0.5)).to_measures_dict()
    assert measures["total_cost"] == pytest.approx(0.75)
    assert "cost_lower_bound" not in measures
    assert "cost_unmeasured" not in measures
