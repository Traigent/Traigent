from __future__ import annotations

import pytest

from traigent.core.metric_registry import MetricSpec
from traigent.metrics.registry import (
    clone_registry,
    get_registry,
    register_metric,
    reset_registry,
)


@pytest.fixture(autouse=True)
def reset_global_registry() -> None:
    """Ensure each test starts from a clean registry state."""
    reset_registry()
    yield
    reset_registry()


def test_register_metric_updates_global_registry() -> None:
    register_metric(MetricSpec(name="accuracy", aggregator="last"))

    registry = get_registry()
    assert registry.get("accuracy") is not None
    assert registry.aggregator_for("accuracy") == "last"


def test_clone_registry_carries_custom_specs() -> None:
    register_metric(MetricSpec(name="accuracy", aggregator="sum"))

    cloned = clone_registry()

    assert cloned.aggregator_for("accuracy") == "sum"
    assert cloned is not get_registry()


def test_clone_registry_is_isolated_from_mutations() -> None:
    cloned = clone_registry()
    cloned.register(MetricSpec(name="latency", aggregator="last"))

    assert cloned.get("latency") is not None
    assert get_registry().get("latency") is None


def test_reset_registry_with_custom_specs() -> None:
    custom_spec = MetricSpec(name="custom_metric", aggregator="mean")

    reset_registry(specs=[custom_spec])

    registry = get_registry()
    assert registry.get("custom_metric") is not None
    assert registry.aggregator_for("custom_metric") == "mean"
