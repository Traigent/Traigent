"""PydanticAI metrics report the priced part as a floor when a run is unmeasured (SDK-40)."""

from __future__ import annotations

import pytest

from traigent.integrations.pydantic_ai._types import (
    AgentRunMetrics,
    PydanticAIHandlerMetrics,
)


def _unmeasured_mix() -> PydanticAIHandlerMetrics:
    return PydanticAIHandlerMetrics(
        runs=[
            AgentRunMetrics(cost=0.25),
            AgentRunMetrics(cost=0.0, cost_unmeasured=True),
        ]
    )


def test_unmeasured_run_reports_priced_part_as_cost_lower_bound_not_total() -> None:
    measures = _unmeasured_mix().to_measures_dict()
    assert measures["cost_unmeasured"] == 1.0
    assert measures["cost_lower_bound"] == pytest.approx(0.25)
    assert "total_cost" not in measures


def test_floor_key_carries_the_same_prefix_as_the_unmeasured_flag() -> None:
    measures = _unmeasured_mix().to_measures_dict(prefix="pydantic_ai_")
    assert measures["pydantic_ai_cost_unmeasured"] == 1.0
    assert measures["pydantic_ai_cost_lower_bound"] == pytest.approx(0.25)
    assert "pydantic_ai_total_cost" not in measures


def test_fully_measured_runs_report_total_cost_and_no_floor() -> None:
    metrics = PydanticAIHandlerMetrics(
        runs=[AgentRunMetrics(cost=0.25), AgentRunMetrics(cost=0.5)]
    )
    measures = metrics.to_measures_dict()
    assert measures["total_cost"] == pytest.approx(0.75)
    assert "cost_lower_bound" not in measures
    assert "cost_unmeasured" not in measures
