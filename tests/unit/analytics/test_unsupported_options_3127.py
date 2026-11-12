"""Analytics options that are not implemented must warn, not be silently ignored (#3127)."""

from datetime import UTC, datetime, timedelta

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

USAGE = [
    {
        "timestamp": datetime.now(UTC) - timedelta(hours=h),
        "cost": 10.0 + h,
        "resource_type": "compute",
        "usage": {"cpu_hours": 5.0, "memory_gb_hours": 2.0},
    }
    for h in range(24)
]
HISTORY = [{"cost": 10}, {"cost": 12}, {"cost": 15}]


def test_cost_ai_warns_on_days_back():
    with pytest.warns(UserWarning, match="days_back"):
        CostOptimizationAI().analyze_cost_patterns(USAGE, days_back=7)


def test_forecast_costs_warns_on_confidence_level():
    with pytest.warns(UserWarning, match="confidence_level"):
        CostForecaster().forecast_costs(HISTORY, confidence_level=0.5)


def test_forecast_costs_still_accepts_forecast_period():
    result = CostForecaster().forecast_costs(
        HISTORY, forecast_period=ForecastPeriod.MONTHLY
    )
    assert result.forecast_horizon_days == 30


def test_performance_forecast_warns_on_forecast_period():
    with pytest.warns(UserWarning, match="forecast_period"):
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
def test_cost_ai_warns_on_ignored_options(method, kwargs):
    name = next(iter(k for k in kwargs if k != "resources"))
    with pytest.warns(UserWarning, match=name):
        getattr(CostOptimizationAI(), method)(**kwargs)


def test_generate_actions_warns_on_strategy_but_allows_default():
    opt = ResourceOptimizer()
    with pytest.warns(UserWarning, match="strategy"):
        opt.generate_optimization_actions(
            ResourceType.COMPUTE, {}, strategy=OptimizationStrategy.AGGRESSIVE
        )
    opt.generate_optimization_actions(
        ResourceType.COMPUTE, {}, strategy=OptimizationStrategy.BALANCED
    )


def test_resource_configuration_warns_on_cost_constraints():
    with pytest.warns(UserWarning, match="cost_constraints"):
        ResourceOptimizer().optimize_resource_configuration(
            ResourceType.COMPUTE, {}, cost_constraints={"max_hourly_cost": 1.0}
        )


def test_schedule_optimization_warns_on_time_constraints():
    with pytest.warns(UserWarning, match="time_constraints"):
        SmartScheduler().schedule_optimization(
            optimization_request={"function_name": "f"},
            time_constraints={"max_delay_hours": 1},
        )


@pytest.mark.parametrize(
    "field", ["prefer_off_peak", "batch_similar_jobs", "preemption_allowed"]
)
def test_scheduling_policy_warns_on_non_default_flags(field):
    default = getattr(SchedulingPolicy(), field)
    with pytest.warns(UserWarning, match=field):
        SchedulingPolicy(**{field: not default})


def test_days_back_former_default_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CostOptimizationAI().analyze_cost_patterns(USAGE, days_back=30)


def test_scheduling_policy_later_assignment_warns():
    policy = SchedulingPolicy()
    with pytest.warns(UserWarning, match="preemption_allowed"):
        policy.preemption_allowed = True


def test_forecast_costs_former_default_confidence_level_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CostForecaster().forecast_costs(HISTORY, confidence_level=0.95)


def test_scheduler_warns_at_use_time_even_for_default_flags_once_per_policy():
    import warnings

    scheduler = SmartScheduler(SchedulingPolicy())
    assert scheduler.policy.preemption_allowed is False
    with pytest.warns(UserWarning, match="preemption_allowed"):
        scheduler.schedule_optimization(optimization_request={"function_name": "f"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scheduler.schedule_optimization(optimization_request={"function_name": "f"})
