"""Characterization of CostOptimizationAI.simulate_optimization_impact."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from traigent.analytics.cost_optimization import (
    CostOptimizationAction,
    CostOptimizationAI,
    Priority,
    ResourceType,
    ResourceUsage,
)

BASE_KEYS = [
    "baseline_cost",
    "projected_cost",
    "total_savings",
    "total_estimated_savings",
    "savings_breakdown",
    "risk_assessment",
    "implementation_timeline",
    "cost_reduction_percent",
    "confidence_score",
]


def _usage(resource: ResourceType, cost: float) -> ResourceUsage:
    return ResourceUsage(
        resource_type=resource,
        current_usage=1.0,
        historical_average=1.0,
        peak_usage=1.0,
        unit_cost=1.0,
        total_cost=cost,
        utilization_percent=50.0,
    )


def _action(resource: ResourceType, absolute: float) -> CostOptimizationAction:
    return CostOptimizationAction(
        action_id="a",
        action_type="t",
        description="d",
        resource_type=resource,
        estimated_savings_percent=10.0,
        estimated_savings_absolute=absolute,
        implementation_effort="low",
        risk_level="low",
        priority=list(Priority)[0],
        detailed_steps=[],
        impact_analysis={},
        timeline_days=1,
    )


def _ai(costs: dict[ResourceType, list[float]] | None = None) -> CostOptimizationAI:
    ai = CostOptimizationAI()
    for resource, values in (costs or {}).items():
        for value in values:
            ai.usage_history[resource].append(_usage(resource, value))
    return ai


def test_empty_history_and_no_actions_full_result() -> None:
    result = _ai().simulate_optimization_impact([])
    assert list(result) == BASE_KEYS
    assert result == {
        "baseline_cost": 0.0,
        "projected_cost": 0,
        "total_savings": 0.0,
        "total_estimated_savings": 0.0,
        "savings_breakdown": {},
        "risk_assessment": "low",
        "implementation_timeline": {},
        "cost_reduction_percent": 0.0,
        "confidence_score": 0.8,
    }
    # max(0, 0.0) keeps the int 0
    assert type(result["projected_cost"]) is int


def test_baseline_uses_last_seven_entries_per_resource_and_days() -> None:
    ai = _ai(
        {
            ResourceType.COMPUTE: [1000.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0],
            ResourceType.STORAGE: [4.0, 6.0],
        }
    )
    # compute last 7: six 10s and a 20 -> 80/7 ; storage 5.0
    expected = (80.0 / 7) * 10 + 5.0 * 10
    result = ai.simulate_optimization_impact([], simulation_days=10)
    assert result["baseline_cost"] == pytest.approx(expected)
    assert result["projected_cost"] == pytest.approx(expected)
    assert result["total_savings"] == 0.0
    assert result["cost_reduction_percent"] == 0.0


def test_empty_resource_lists_are_skipped() -> None:
    ai = _ai({ResourceType.COMPUTE: [2.0]})
    ai.usage_history[ResourceType.STORAGE] = []
    result = ai.simulate_optimization_impact([], simulation_days=3)
    assert result["baseline_cost"] == 6.0


def test_default_simulation_days_is_thirty() -> None:
    result = _ai({ResourceType.COMPUTE: [1.0]}).simulate_optimization_impact([])
    assert result["baseline_cost"] == 30.0


def test_list_actions_scale_by_days_and_overwrite_same_resource() -> None:
    ai = _ai({ResourceType.COMPUTE: [10.0]})
    actions = [
        _action(ResourceType.COMPUTE, 1.0),
        _action(ResourceType.STORAGE, 2.0),
        _action(ResourceType.COMPUTE, 3.0),
    ]
    result = ai.simulate_optimization_impact(actions, simulation_days=5)
    assert result == {
        "baseline_cost": 50.0,
        "projected_cost": 50.0 - 30.0,
        "total_savings": 30.0,
        "total_estimated_savings": 30.0,
        "savings_breakdown": {"compute": 15.0, "storage": 10.0},
        "risk_assessment": "low",
        "implementation_timeline": {},
        "cost_reduction_percent": 60.0,
        "confidence_score": 0.8,
    }


def test_list_item_without_savings_attribute_is_skipped() -> None:
    ai = _ai({ResourceType.COMPUTE: [10.0]})
    items: list[Any] = [
        SimpleNamespace(resource_type=ResourceType.COMPUTE),
        "text",
        None,
        _action(ResourceType.NETWORK, 1.0),
    ]
    result = ai.simulate_optimization_impact(items, simulation_days=2)
    assert result["savings_breakdown"] == {"network": 2.0}
    assert result["total_savings"] == 2.0
    assert result["projected_cost"] == 18.0


def test_savings_exceeding_baseline_clamps_projected_to_int_zero() -> None:
    ai = _ai({ResourceType.COMPUTE: [1.0]})
    result = ai.simulate_optimization_impact(
        [_action(ResourceType.COMPUTE, 100.0)], simulation_days=2
    )
    assert result["projected_cost"] == 0
    assert type(result["projected_cost"]) is int
    assert result["total_savings"] == 200.0
    assert result["cost_reduction_percent"] == 10000.0


def test_zero_baseline_skips_reduction_percent() -> None:
    result = _ai().simulate_optimization_impact(
        [_action(ResourceType.COMPUTE, 5.0)], simulation_days=1
    )
    assert result["baseline_cost"] == 0.0
    assert result["total_savings"] == 5.0
    assert result["projected_cost"] == 0
    assert result["cost_reduction_percent"] == 0.0


def test_negative_baseline_skips_reduction_percent() -> None:
    ai = _ai({ResourceType.COMPUTE: [-4.0]})
    result = ai.simulate_optimization_impact(
        [_action(ResourceType.COMPUTE, 1.0)], simulation_days=2
    )
    assert result["baseline_cost"] == -8.0
    assert result["total_savings"] == 2.0
    assert result["projected_cost"] == 0
    assert result["cost_reduction_percent"] == 0.0


def test_dict_actions_use_baseline_percent_and_skip_non_numbers() -> None:
    ai = _ai({ResourceType.COMPUTE: [10.0]})
    actions: dict[str, Any] = {
        "a": 0.5,
        "b": 1,
        "c": "0.5",
        "d": None,
        "e": [0.1],
        "f": True,
    }
    result = ai.simulate_optimization_impact(actions, simulation_days=10)
    # baseline 100: a=50, b=100, f(True counts as an int)=100
    assert result["savings_breakdown"] == {"a": 50.0, "b": 100, "f": 100}
    assert result["total_savings"] == 250.0
    assert result["total_estimated_savings"] == 250.0
    assert result["projected_cost"] == 0
    assert result["cost_reduction_percent"] == 250.0


def test_dict_actions_accumulate_in_insertion_order() -> None:
    ai = _ai({ResourceType.COMPUTE: [10.0]})
    result = ai.simulate_optimization_impact({"z": 0.1, "a": 0.2}, simulation_days=10)
    assert list(result["savings_breakdown"]) == ["z", "a"]
    assert result["savings_breakdown"] == {"z": 10.0, "a": 20.0}
    assert result["projected_cost"] == 70.0


def test_dict_with_zero_baseline_records_zero_savings() -> None:
    result = _ai().simulate_optimization_impact({"x": 0.9})
    assert result["savings_breakdown"] == {"x": 0.0}
    assert result["total_savings"] == 0.0
    assert result["projected_cost"] == 0


@pytest.mark.parametrize("actions", [(), None, "abc", 5, {1, 2}])
def test_other_action_container_types_do_nothing(actions: Any) -> None:
    ai = _ai({ResourceType.COMPUTE: [10.0]})
    result = ai.simulate_optimization_impact(actions, simulation_days=2)
    assert result["savings_breakdown"] == {}
    assert result["total_savings"] == 0.0
    assert result["projected_cost"] == 20.0
    assert result["cost_reduction_percent"] == 0.0


def test_list_subclass_and_dict_subclass_are_accepted() -> None:
    class _L(list[Any]):
        pass

    class _D(dict[str, Any]):
        pass

    ai = _ai({ResourceType.COMPUTE: [10.0]})
    assert ai.simulate_optimization_impact(
        _L([_action(ResourceType.COMPUTE, 1.0)]), simulation_days=1
    )["savings_breakdown"] == {"compute": 1.0}
    assert ai.simulate_optimization_impact(_D(q=0.5), simulation_days=1)[
        "savings_breakdown"
    ] == {"q": 5.0}


def test_infinite_percent_gives_infinite_savings_and_zero_projection() -> None:
    ai = _ai({ResourceType.COMPUTE: [10.0]})
    result = ai.simulate_optimization_impact({"x": float("inf")}, simulation_days=1)
    assert result["total_savings"] == float("inf")
    assert result["projected_cost"] == 0
    assert result["cost_reduction_percent"] == float("inf")


def test_simulation_does_not_mutate_history_or_actions() -> None:
    ai = _ai({ResourceType.COMPUTE: [1.0, 2.0]})
    actions = [_action(ResourceType.COMPUTE, 1.0)]
    ai.simulate_optimization_impact(actions)
    assert [u.total_cost for u in ai.usage_history[ResourceType.COMPUTE]] == [1.0, 2.0]
    assert len(actions) == 1


def test_attribute_read_trace_for_list_actions() -> None:
    events: list[str] = []

    class _Traced:
        @property
        def estimated_savings_absolute(self) -> float:
            events.append("abs")
            return 2.0

        @property
        def resource_type(self) -> ResourceType:
            events.append("rt")
            return ResourceType.COMPUTE

    items: list[Any] = [_Traced()]
    result = _ai().simulate_optimization_impact(items, simulation_days=3)
    assert events == ["abs", "abs", "rt"]
    assert result["savings_breakdown"] == {"compute": 6.0}


def test_raising_savings_accessor_attribute_error_skips_other_errors_propagate() -> (
    None
):
    class _Missing:
        resource_type = ResourceType.COMPUTE

        @property
        def estimated_savings_absolute(self) -> float:
            raise AttributeError("gone")

    class _Broken:
        resource_type = ResourceType.COMPUTE

        @property
        def estimated_savings_absolute(self) -> float:
            raise RuntimeError("broken")

    ai = _ai({ResourceType.COMPUTE: [10.0]})
    items: list[Any] = [_Missing()]
    assert ai.simulate_optimization_impact(items)["savings_breakdown"] == {}
    broken: list[Any] = [_Broken()]
    with pytest.raises(RuntimeError, match="broken"):
        ai.simulate_optimization_impact(broken)


def test_raising_resource_type_accessor_propagates() -> None:
    class _Bad:
        estimated_savings_absolute = 1.0

        @property
        def resource_type(self) -> ResourceType:
            raise RuntimeError("rt")

    items: list[Any] = [_Bad()]
    with pytest.raises(RuntimeError, match="rt"):
        _ai().simulate_optimization_impact(items)


def test_non_numeric_savings_value_raises_type_error() -> None:
    items: list[Any] = [
        SimpleNamespace(
            estimated_savings_absolute="x", resource_type=ResourceType.COMPUTE
        )
    ]
    with pytest.raises(TypeError):
        _ai().simulate_optimization_impact(items)
