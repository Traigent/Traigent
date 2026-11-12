"""Analytics options that are not implemented must be rejected, not ignored (#3127)."""

import pytest

from traigent.analytics.cost_optimization import (
    OptimizationStrategy,
    ResourceOptimizer,
    ResourceType,
)
from traigent.analytics.intelligence import CostOptimizationAI
from traigent.analytics.predictive import (
    CostForecaster,
    ForecastPeriod,
    PerformanceForecaster,
)
from traigent.analytics.scheduling import SchedulingPolicy, SmartScheduler

HISTORY = [{"cost": 10}, {"cost": 12}, {"cost": 15}]


def test_cost_ai_rejects_days_back():
    with pytest.raises(ValueError, match="days_back"):
        CostOptimizationAI().analyze_cost_patterns(days_back=7)


def test_forecast_costs_rejects_confidence_level():
    with pytest.raises(ValueError, match="confidence_level"):
        CostForecaster().forecast_costs(HISTORY, confidence_level=0.5)


def test_forecast_costs_still_accepts_forecast_period():
    result = CostForecaster().forecast_costs(
        HISTORY, forecast_period=ForecastPeriod.MONTHLY
    )
    assert result.forecast_horizon_days == 30


def test_performance_forecast_rejects_forecast_period():
    with pytest.raises(ValueError, match="forecast_period"):
        PerformanceForecaster().forecast([], forecast_period=ForecastPeriod.DAILY)


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("get_optimization_recommendations", {"objectives": ["cost"]}),
        ("get_optimization_recommendations", {"constraints": {"x": 1}}),
        ("get_optimization_recommendations", {"current_usage": {"x": 1}}),
        ("get_optimization_recommendations", {"usage_history": [1]}),
        (
            "optimize_resource_allocation",
            {"resources": {"budget": 1}, "constraints": {"a": 1}},
        ),
        ("optimize_multi_cloud", {"workload": {"a": 1}}),
    ],
)
def test_cost_ai_rejects_ignored_options(method, kwargs):
    name = next(iter(k for k in kwargs if k != "resources"))
    with pytest.raises(ValueError, match=name):
        getattr(CostOptimizationAI(), method)(**kwargs)


def test_generate_actions_rejects_strategy_but_allows_default():
    opt = ResourceOptimizer()
    with pytest.raises(ValueError, match="strategy"):
        opt.generate_optimization_actions(
            ResourceType.COMPUTE, {}, strategy=OptimizationStrategy.AGGRESSIVE
        )
    opt.generate_optimization_actions(
        ResourceType.COMPUTE, {}, strategy=OptimizationStrategy.BALANCED
    )


def test_resource_configuration_rejects_cost_constraints():
    with pytest.raises(ValueError, match="cost_constraints"):
        ResourceOptimizer().optimize_resource_configuration(
            ResourceType.COMPUTE, {}, cost_constraints={"max_hourly_cost": 1.0}
        )


def test_schedule_optimization_rejects_time_constraints():
    with pytest.raises(ValueError, match="time_constraints"):
        SmartScheduler().schedule_optimization(
            optimization_request={"function_name": "f"},
            time_constraints={"max_delay_hours": 1},
        )


@pytest.mark.parametrize(
    "field", ["prefer_off_peak", "batch_similar_jobs", "preemption_allowed"]
)
def test_scheduling_policy_rejects_non_default_flags(field):
    default = getattr(SchedulingPolicy(), field)
    with pytest.raises(ValueError, match=field):
        SchedulingPolicy(**{field: not default})
