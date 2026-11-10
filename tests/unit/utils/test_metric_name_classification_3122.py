"""Metric names are classified by whole tokens, not substrings (#3122)."""

from __future__ import annotations

import pytest

from traigent.analytics.anomaly import RegressionDetector
from traigent.core.stop_conditions import (
    _looks_like_continuous_metric,
    _looks_like_quality_metric,
)
from traigent.utils.metric_name_tokens import is_usd_cost_metric
from traigent.utils.results_table import _format_metric_value, _metric_unit


@pytest.mark.parametrize(
    "name", ["token_f1", "token_accuracy", "programs_passed", "error_free_rate"]
)
def test_quality_names_with_a_unit_like_word_stay_quality(name: str) -> None:
    assert _looks_like_quality_metric(name)
    assert not _looks_like_continuous_metric(name)


@pytest.mark.parametrize(
    "name",
    [
        "cost",
        "total_tokens",
        "latency_ms",
        "response_time",
        "cost_per_accuracy_point",
        "time_to_accuracy_s",
    ],
)
def test_unit_names_stay_continuous(name: str) -> None:
    assert _looks_like_continuous_metric(name)
    assert not _looks_like_quality_metric(name)


@pytest.mark.parametrize(
    "name",
    [
        "cost_count",
        "unknown_cost_count",
        "cost_savings_pct",
        "runtime_cost_rank",
        "accuracy_per_1k_cost",
        "cost_efficiency",
        "cost_ratio",
        "x_per_dollar_cost",
    ],
)
def test_cost_named_non_amounts_are_not_usd(name: str) -> None:
    assert not is_usd_cost_metric(name)
    assert _metric_unit(name) == "fraction"
    assert not _format_metric_value(name, 12.0).startswith("$")


@pytest.mark.parametrize(
    "name", ["cost", "total_cost", "avg_cost", "cost_per_call", "total_cost_usd"]
)
def test_cost_amounts_are_usd(name: str) -> None:
    assert is_usd_cost_metric(name)
    assert _format_metric_value(name, 12.0) == "$12.00000"


@pytest.mark.parametrize(
    ("name", "change", "regressed"),
    [
        ("cost_per_accuracy_point", 0.3, True),
        ("cost_per_accuracy_point", -0.3, False),
        ("time_to_accuracy_s", 0.3, True),
        ("time_to_accuracy_s", -0.3, False),
        ("error_free_rate", -0.3, True),
        ("error_free_rate", 0.3, False),
        ("accuracy", -0.3, True),
        ("accuracy", 0.3, False),
        ("total_cost", 0.3, True),
    ],
)
def test_regression_direction(name: str, change: float, regressed: bool) -> None:
    detector = RegressionDetector(sensitivity=0.1)
    assert detector._is_significant_regression(name, change) is regressed


def test_unmeasured_cost_backfill_only_omits_amount_objectives() -> None:
    from traigent.cloud.sync_manager import SyncManager

    measures: dict[str, float] = {}
    SyncManager._backfill_trial_objectives(
        measures, ["total_cost", "cost_efficiency"], 0.5, True
    )
    assert "total_cost" not in measures
    assert measures["cost_efficiency"] == 0.5
