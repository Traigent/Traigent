"""Golden characterization tests for semantic-saturation metric resolution."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from traigent.core.stop_conditions import SemanticSaturationStopCondition


def _trial(
    examples: list[Any] | None = None, metrics: dict[Any, Any] | None = None
) -> Any:
    return SimpleNamespace(
        metadata={"example_results": examples} if examples is not None else {},
        metrics=metrics,
    )


class _Boom:
    @property
    def metrics(self) -> Any:
        raise RuntimeError("boom")


def _schema(*names: str) -> Any:
    return SimpleNamespace(objectives=[SimpleNamespace(name=n) for n in names])


def test_quality_metrics_explicit_config_short_circuits() -> None:
    cond = SemanticSaturationStopCondition({"example_score_metrics": ["zzz"]})
    assert cond._resolve_quality_metrics([_trial([{"metrics": {"accuracy": 1}}])]) == (
        "zzz",
    )


def test_quality_metrics_selection_rules_and_sorting() -> None:
    examples = [
        {"metrics": {"accuracy": 0.5, "latency_ms": 1, "flag": True, "bin": 0}},
        {"metrics": {"flag": False, "bin": 1, "other": 0.3, "score_x": 2.0}},
        {"metrics": {"accuracy": "0.7", "nan": float("nan"), "s": "bad", 5: 1}},
        {"metrics": None},
        {"metrics": "notmapping"},
        SimpleNamespace(metrics={"bin": 1.0}),
        _Boom(),
    ]
    cond = SemanticSaturationStopCondition()
    result = cond._resolve_quality_metrics([_trial(examples), _trial(None)])
    # accuracy/score_x look like quality; flag and bin are binary; latency_ms is
    # continuous (excluded even though quality-like or binary); other is neither.
    assert result == ("accuracy", "bin", "flag", "score_x")


def test_quality_metrics_continuous_name_wins_over_quality_marker() -> None:
    cond = SemanticSaturationStopCondition()
    examples = [{"metrics": {"score_time": 1.0, "error_flag": 1, "quality": 3}}]
    assert cond._resolve_quality_metrics([_trial(examples)]) == ("quality",)


def test_quality_metrics_empty() -> None:
    cond = SemanticSaturationStopCondition()
    assert cond._resolve_quality_metrics([]) == ()


def test_continuous_objectives_explicit_filters_quality() -> None:
    cond = SemanticSaturationStopCondition(
        {"continuous_objectives": ["cost", "acc", "latency"]}
    )
    assert cond._resolve_continuous_objectives([], ["acc"]) == ("cost", "latency")


def test_continuous_objectives_from_schema() -> None:
    cond = SemanticSaturationStopCondition(
        objective_schema=_schema("accuracy", "cost", "score", "latency", "custom", "q")
    )
    # accuracy quality-listed; score quality-like not continuous -> skipped;
    # custom/cost/latency kept; "q" kept (no markers).
    assert cond._resolve_continuous_objectives([], ["accuracy"]) == (
        "cost",
        "latency",
        "custom",
        "q",
    )


def test_continuous_objectives_schema_all_filtered_falls_back_to_trials() -> None:
    cond = SemanticSaturationStopCondition(objective_schema=_schema("accuracy", "f1"))
    trials = [
        _trial(metrics={"cost": 1, "latency": "2", "accuracy": 1, "x": 1, 3: 1}),
        _trial(metrics={"total_tokens": True, "bad_time": None, "err_time": "z"}),
        _trial(metrics=None),
    ]
    assert cond._resolve_continuous_objectives(trials, ["accuracy"]) == (
        "cost",
        "latency",
        "total_tokens",
    )


def test_continuous_objectives_no_schema_uses_trials_and_skips_quality() -> None:
    cond = SemanticSaturationStopCondition()
    trials = [_trial(metrics={"cost": 1, "time_score": 1, "ms": 2})]
    assert cond._resolve_continuous_objectives(trials, ["time_score"]) == ("cost", "ms")
    assert cond._resolve_continuous_objectives([], []) == ()
