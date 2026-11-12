"""Ratio properties tolerate a legal zero denominator (#3121)."""

from traigent.cloud.sessions import _SessionStateRegistry
from traigent.optimizers.random import RandomSearchOptimizer


def test_random_progress_with_zero_max_trials_is_complete():
    optimizer = RandomSearchOptimizer({"a": [1, 2]}, ["accuracy"], max_trials=0)
    assert optimizer.progress == 1.0


def test_random_progress_positive_max_trials_unchanged():
    optimizer = RandomSearchOptimizer({"a": [1, 2]}, ["accuracy"], max_trials=10)
    assert optimizer.progress == 0.0


def test_session_registry_statistics_with_zero_capacity():
    registry = _SessionStateRegistry(max_sessions=0)
    assert registry.get_statistics()["capacity_utilization"] == 1.0
