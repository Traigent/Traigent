"""#3117: list or dict valued config values do not crash importance analysis."""

from __future__ import annotations

from datetime import UTC, datetime

from traigent.api.types import TrialResult, TrialStatus
from traigent.utils.callbacks import (
    StatisticsCallback,
    _parameter_importance_for_values,
)
from traigent.utils.importance import ParameterImportanceAnalyzer


def _trials(make_stop) -> list[TrialResult]:
    return [
        TrialResult(
            trial_id=str(i),
            config={"stop": make_stop(i), "t": i % 3},
            metrics={"accuracy": float(i % 2)},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        )
        for i in range(40)
    ]


def test_all_methods_handle_list_and_dict_values() -> None:
    analyzer = ParameterImportanceAnalyzer("accuracy")
    for make_stop in (
        lambda i: ["END"] if i % 2 else ["STOP", "\n"],
        lambda i: {"k": i % 2},
    ):
        trials = _trials(make_stop)
        variance = analyzer.analyze_variance_based(trials)
        assert variance["stop"].importance_score == 1.0
        assert "t" in variance
        correlation = analyzer.analyze_correlation_based(trials)
        # Reported explicitly as unsupported, not silently omitted.
        assert correlation["stop"].importance_score is None
        assert correlation["stop"].not_measurable_reason
        assert "t" in correlation
        permutation = analyzer.analyze_permutation_based(trials)
        assert permutation["stop"].importance_score is not None


def test_statistics_callback_groups_unhashable_values() -> None:
    cb = StatisticsCallback()
    cb.stats["scores_by_trial"] = [1.0, 0.0, 1.0, 0.0]
    values = [["a"], ["b"], ["a"], ["b"]]
    assert _parameter_importance_for_values(cb, values) > 0
    assert (
        _parameter_importance_for_values(cb, [{"k": 1}, {"k": 2}, {"k": 1}, {"k": 2}])
        > 0
    )


def test_mixed_complex_and_scalar_values_are_not_measurable() -> None:
    trials = _trials(lambda i: ["END"] if i % 4 == 0 else ("a" if i % 2 else "b"))
    result = ParameterImportanceAnalyzer("accuracy").analyze_correlation_based(trials)
    assert result["stop"].importance_score is None
    assert result["stop"].not_measurable_reason
