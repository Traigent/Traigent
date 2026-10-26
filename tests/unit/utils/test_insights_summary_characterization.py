"""Characterization for the two post-hoc summaries in utils.insights."""

from __future__ import annotations

import json
import statistics
from typing import Any
from unittest.mock import call, patch

import pytest

from traigent.utils.insights import (
    _analyze_parameter_importance,
    _analyze_performance_summary,
)


class _TrackingMetrics(dict[str, Any]):
    def __init__(self, label: str, values: dict[str, Any], trace: list[tuple]) -> None:
        super().__init__(values)
        self.label = label
        self.trace = trace

    def get(self, key: str, default: Any = None) -> Any:
        self.trace.append(("metrics.get", self.label, key, default))
        return super().get(key, default)


class _TrackingTrial:
    def __init__(
        self,
        label: str,
        config: dict[str, Any],
        metrics: dict[str, Any],
        trace: list[tuple],
    ) -> None:
        self.label = label
        self._config = config
        self._metrics = _TrackingMetrics(label, metrics, trace)
        self.trace = trace

    @property
    def config(self) -> dict[str, Any]:
        self.trace.append(("config.read", self.label))
        return self._config

    @property
    def metrics(self) -> _TrackingMetrics:
        self.trace.append(("metrics.read", self.label))
        return self._metrics


def _expected_metric_trace(keys: list[tuple[str, str]]) -> list[tuple]:
    trace: list[tuple] = []
    for label, key in keys:
        trace.extend([("metrics.read", label), ("metrics.get", label, key, 0.0)])
    return trace


def test_performance_summary_pins_exact_output_and_metric_reads() -> None:
    trace: list[tuple] = []
    trials = [
        _TrackingTrial("one", {}, {"quality": 1.0, "latency": 0.2}, trace),
        _TrackingTrial("two", {}, {"quality": -1.0, "latency": 0.8}, trace),
        _TrackingTrial("three", {}, {"latency": 0.0}, trace),
    ]

    summary = _analyze_performance_summary(trials, ["quality", "latency", "quality"])

    assert json.dumps(summary, separators=(",", ":")) == (
        '{"total_trials":3,"successful_trials":3,"primary_objective":"quality",'
        '"best_score":1.0,"worst_score":-1.0,"average_score":0.0,'
        '"score_std":1.0,"improvement":0.0,"consistency":1.0,'
        '"multi_objective_analysis":{"quality":{"best":1.0,"average":0.0,'
        '"improvement":0.0},"latency":{"best":0.8,'
        '"average":0.33333333333333337,"improvement":0.0}}}'
    )
    assert trace == _expected_metric_trace(
        [(label, key) for key in ("quality",) for label in ("one", "two", "three")]
        + [
            (label, key)
            for key in ("quality", "latency", "quality")
            for label in ("one", "two", "three")
        ]
    )


def test_performance_summary_pins_empty_default_objective_and_repeated_stdev() -> None:
    assert _analyze_performance_summary([], []) == {}

    trace: list[tuple] = []
    trial = _TrackingTrial("one", {}, {"score": 0.0}, trace)
    assert _analyze_performance_summary([trial], []) == {
        "total_trials": 1,
        "successful_trials": 1,
        "primary_objective": "score",
        "best_score": 0.0,
        "worst_score": 0.0,
        "average_score": 0.0,
        "score_std": 0.0,
        "improvement": 0.0,
        "consistency": 1.0,
    }
    assert trace == _expected_metric_trace([("one", "score")])

    trials = [
        _TrackingTrial("one", {}, {"quality": 1.0}, []),
        _TrackingTrial("two", {}, {"quality": 2.0}, []),
        _TrackingTrial("three", {}, {"quality": 3.0}, []),
    ]
    with patch(
        "traigent.utils.insights.statistics.stdev", wraps=statistics.stdev
    ) as stdev:
        positive_summary = _analyze_performance_summary(trials, ["quality"])
    assert positive_summary["score_std"] == 1.0
    assert positive_summary["consistency"] == 0.5
    assert stdev.call_args_list == [call([1.0, 2.0, 3.0]), call([1.0, 2.0, 3.0])]


def test_parameter_importance_pins_output_order_and_config_metric_read_order() -> None:
    trace: list[tuple] = []
    configs = [
        {"color": "blue", "temperature": 0.1, "sparse": "x"},
        {"color": "red", "temperature": 0.1},
        {},
        {"color": "blue", "temperature": 0.9, "first_only": True},
    ]
    metrics = [
        {"quality": 1.0},
        {"quality": 1.0},
        {"quality": 0.8},
        {},
    ]
    labels = ["one", "two", "three", "four"]
    trials = [
        _TrackingTrial(label, config, metric, trace)
        for label, config, metric in zip(labels, configs, metrics, strict=True)
    ]

    importance = _analyze_parameter_importance(trials, ["quality"])

    assert list(importance) == list(set(key for config in configs for key in config))
    assert json.dumps(importance, sort_keys=True, separators=(",", ":")) == (
        '{"color":{"best_value":"blue","optimization_priority":"high",'
        '"performance_impact":0.5303300858899107,"value_distribution":'
        '{"best_value":"red","type":"categorical","unique_values":2,'
        '"value_performance":{"blue":0.5,"red":1.0}}},'
        '"first_only":{"best_value":true,"optimization_priority":"low",'
        '"performance_impact":0.0,"value_distribution":{"best_value":true,'
        '"type":"constant"}},"sparse":{"best_value":"x",'
        '"optimization_priority":"low","performance_impact":0.0,'
        '"value_distribution":{"best_value":"x","type":"constant"}},'
        '"temperature":{"best_value":0.1,"optimization_priority":"high",'
        '"performance_impact":0.9999999999999998,"value_distribution":'
        '{"best_value":0.1,"type":"numeric","unique_values":2,'
        '"value_performance":{"0.1":1.0,"0.9":0.0}}}}'
    )

    expected_trace: list[tuple] = [("config.read", label) for label in labels]
    parameter_order = list(set(key for config in configs for key in config))
    for parameter in parameter_order:
        for label, config in zip(labels, configs, strict=True):
            expected_trace.append(("config.read", label))
            if parameter in config:
                expected_trace.extend(
                    [
                        ("config.read", label),
                        ("metrics.read", label),
                        ("metrics.get", label, "quality", 0.0),
                    ]
                )
    assert trace == expected_trace


def test_parameter_importance_pins_priority_thresholds_ties_and_errors() -> None:
    trials = [
        _TrackingTrial(
            "one",
            {
                "above": "above-one",
                "at_high": "mid-one",
                "at_medium": "low-one",
                "tie": "first",
            },
            {"quality": 1.0},
            [],
        ),
        _TrackingTrial(
            "two",
            {
                "above": "above-two",
                "at_high": "mid-two",
                "at_medium": "low-two",
                "tie": "second",
            },
            {"quality": 1.0},
            [],
        ),
    ]
    impacts = {
        "above-one": 0.1000001,
        "mid-one": 0.1,
        "low-one": 0.05,
        "first": 0.2,
    }
    with patch(
        "traigent.utils.insights._calculate_parameter_impact",
        side_effect=lambda values, scores: impacts[values[0]],
    ):
        importance = _analyze_parameter_importance(trials, ["quality"])

    assert importance["above"]["optimization_priority"] == "high"
    assert importance["at_high"]["optimization_priority"] == "medium"
    assert importance["at_medium"]["optimization_priority"] == "low"
    assert importance["tie"]["best_value"] == "first"

    unhashable = [_TrackingTrial("bad", {"value": []}, {"quality": 1.0}, [])]
    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        _analyze_parameter_importance(unhashable, ["quality"])

    assert _analyze_parameter_importance([], ["quality"]) == {}
    assert _analyze_parameter_importance(trials, []) == {}
