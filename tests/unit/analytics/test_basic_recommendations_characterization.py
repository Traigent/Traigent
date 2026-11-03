"""Characterization for CostOptimizationAI._generate_basic_recommendations."""

from __future__ import annotations

from traigent.analytics.intelligence import CostOptimizationAI

WEEKLY = "Consider implementing weekend scaling to reduce workweek costs"
DAILY = (
    "Peak usage during business hours - consider off-hours scheduling for batch jobs"
)
SEASONAL = (
    "Holiday season cost spikes detected - plan capacity scaling for peak periods"
)


class _LoggingPattern(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log: list[tuple[str, str]] = []

    def get(self, key, default=None):
        self.log.append(("get", key))
        return super().get(key, default)

    def __contains__(self, key):
        self.log.append(("in", key))
        return super().__contains__(key)

    def __getitem__(self, key):
        self.log.append(("item", key))
        return super().__getitem__(key)


def _gen(patterns):
    return CostOptimizationAI()._generate_basic_recommendations(patterns)


def test_empty_patterns():
    assert _gen([]) == []


def test_typed_patterns_match_only_their_own_pattern_name():
    assert _gen(
        [
            {"type": "weekly", "pattern": "workweek_heavy"},
            {"type": "daily", "pattern": "business_hours"},
            {"type": "seasonal", "pattern": "holiday_peaks"},
        ]
    ) == [WEEKLY, DAILY, SEASONAL]
    assert (
        _gen(
            [
                {"type": "weekly", "pattern": "business_hours"},
                {"type": "daily", "pattern": "workweek_heavy"},
                {"type": "seasonal", "pattern": "workweek_heavy"},
                {"type": "other", "pattern": "workweek_heavy"},
                {"pattern": "holiday_peaks"},
            ]
        )
        == []
    )


def test_legacy_avg_cost_threshold_and_model_default():
    assert _gen(
        [{"avg_cost": 0.1}, {"avg_cost": 0.1001, "model": "m"}, {"avg_cost": 0.5}]
    ) == [
        "Consider optimizing m usage - avg cost $0.100",
        "Consider optimizing unknown usage - avg cost $0.500",
    ]


def test_cost_trend_increasing_uses_service_default():
    assert _gen(
        [
            {"cost_trend": "increasing"},
            {"cost_trend": "increasing", "model": "g"},
            {"cost_trend": "flat", "model": "g"},
        ]
    ) == [
        "Cost trend increasing for service - review usage patterns",
        "Cost trend increasing for g - review usage patterns",
    ]


def test_message_order_within_one_pattern_and_across_patterns():
    assert _gen(
        [
            {
                "type": "weekly",
                "pattern": "workweek_heavy",
                "avg_cost": 1.0,
                "model": "x",
                "cost_trend": "increasing",
            },
            {"type": "daily", "pattern": "business_hours"},
        ]
    ) == [
        WEEKLY,
        "Consider optimizing x usage - avg cost $1.000",
        "Cost trend increasing for x - review usage patterns",
        DAILY,
    ]


def test_lookup_sequences_on_stateful_mapping():
    p = _LoggingPattern(type="weekly", pattern="workweek_heavy")
    _gen([p])
    assert p.log == [
        ("get", "type"),
        ("get", "pattern"),
        ("in", "avg_cost"),
        ("get", "cost_trend"),
    ]
    q = _LoggingPattern(type="mystery", avg_cost=0.5, cost_trend="increasing")
    _gen([q])
    assert q.log == [
        ("get", "type"),
        ("in", "avg_cost"),
        ("item", "avg_cost"),
        ("get", "model"),
        ("item", "avg_cost"),
        ("get", "cost_trend"),
        ("get", "model"),
    ]
    r = _LoggingPattern(type="daily", pattern="nope", avg_cost=0.01)
    _gen([r])
    assert r.log == [
        ("get", "type"),
        ("get", "pattern"),
        ("in", "avg_cost"),
        ("item", "avg_cost"),
        ("get", "cost_trend"),
    ]
