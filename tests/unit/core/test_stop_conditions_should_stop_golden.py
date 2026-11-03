"""Golden characterization for MetricLimit / MaxSamples ``should_stop``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from traigent.api.types import TrialStatus
from traigent.core.stop_conditions import (
    MaxSamplesStopCondition,
    MetricLimitStopCondition,
)


def _t(
    tid: str = "t",
    metrics: Any = None,
    metadata: Any = None,
    status: Any = TrialStatus.COMPLETED,
    example_results: Any = None,
) -> Any:
    return SimpleNamespace(
        trial_id=tid,
        metrics=metrics,
        metadata=metadata,
        status=status,
        example_results=example_results,
    )


# ----------------------------- MetricLimit ---------------------------------


def _ml(limit: float = 10, metric: str = "m", **kw: Any) -> MetricLimitStopCondition:
    return MetricLimitStopCondition(limit=limit, metric_name=metric, **kw)


def test_ml_empty_returns_threshold_check_and_leaves_state() -> None:
    cond = _ml()
    assert cond.should_stop([]) is False
    assert cond._last_index == 0
    cond._running_total = 11.0
    assert cond.should_stop([]) is True


def test_ml_accumulates_and_stops_midway_marks_everything_seen() -> None:
    cond = _ml(limit=5)
    trials = [_t("a", {"m": 2}), _t("b", {"m": 3}), _t("c", {"m": 100})]
    assert cond.should_stop(trials) is True
    assert cond._running_total == 5.0  # third trial never read
    assert cond._last_index == 3


def test_ml_incremental_calls_only_read_new_trials() -> None:
    cond = _ml(limit=10)
    trials = [_t("a", {"m": 4})]
    assert cond.should_stop(trials) is False
    trials.append(_t("b", {"m": 7}))
    assert cond.should_stop(trials) is True
    assert cond._running_total == 11.0
    assert cond.should_stop(trials) is True  # no new -> threshold check


def test_ml_accepts_generator() -> None:
    cond = _ml(limit=1)
    assert cond.should_stop(iter([_t("a", {"m": 1})])) is True


def test_ml_pruned_skipped_unless_included() -> None:
    pruned = _t("p", {"m": 50}, status=TrialStatus.PRUNED)
    cond = _ml(limit=10, include_pruned=False)
    assert cond.should_stop([pruned]) is False
    assert cond._running_total == 0.0
    cond = _ml(limit=10, include_pruned=True)
    assert cond.should_stop([pruned]) is True


def test_ml_value_coercions_and_falsey_zero_is_measured() -> None:
    cond = _ml(limit=100)
    cond.should_stop([_t("a", {"m": "2.5"}), _t("b", {"m": True}), _t("c", {"m": 0})])
    assert cond._running_total == 3.5
    assert cond._unmeasured_trials == 0


def test_ml_none_metrics_and_missing_metric_mandatory_error() -> None:
    cond = _ml()
    with pytest.raises(ValueError, match="Mandatory metric 'm' missing for trial x"):
        cond.should_stop([_t("x", None)])
    assert cond._last_index == 0


def test_ml_non_numeric_raises_with_cause_and_partial_total_kept() -> None:
    cond = _ml(limit=100)
    with pytest.raises(ValueError, match="is not numeric: 'zz'") as ei:
        cond.should_stop([_t("a", {"m": 3}), _t("b", {"m": "zz"})])
    assert isinstance(ei.value.__cause__, ValueError)
    assert cond._running_total == 3.0
    assert cond._last_index == 0  # index not advanced on error
    with pytest.raises(ValueError):
        cond.should_stop([_t("a", {"m": 3}), _t("b", {"m": "zz"})])
    assert cond._running_total == 6.0  # re-reads from start (double counts)
    with pytest.raises(ValueError, match="not numeric"):
        _ml().should_stop([_t("n", {"m": [1]})])


def test_ml_total_cost_compat_chain() -> None:
    cond = _ml(limit=100, metric="total_cost")
    cond.should_stop(
        [
            _t("a", {"total_cost": 1}),
            _t("b", {"cost": 2}),
            _t("c", {}, {"total_example_cost": 4}),
            _t("d", {"cost": 0, "total_cost": None}, {"total_example_cost": 99}),
        ]
    )
    assert cond._running_total == 7.0
    assert cond._unmeasured_trials == 0


def test_ml_total_cost_fallback_not_used_for_other_metrics() -> None:
    cond = _ml(limit=100, metric="other")
    with pytest.raises(ValueError, match="Mandatory"):
        cond.should_stop([_t("a", {"cost": 5})])


@pytest.mark.parametrize("metric", ["cost", "total_cost", "input_cost", "output_cost"])
def test_ml_unmeasured_cost_trials_are_skipped_and_warned_once(metric: str) -> None:
    cond = _ml(limit=10, metric=metric)
    with patch("traigent.core.stop_conditions.logger") as log:
        out = cond.should_stop([_t("a", {}), _t("b", None, {}), _t("c", {metric: 4})])
    assert out is False
    assert cond._unmeasured_trials == 2
    assert cond._running_total == 4.0
    assert log.warning.call_count == 1
    assert log.warning.call_args.args[2] == "a"
    assert cond._last_index == 3


def test_ml_nan_never_stops() -> None:
    cond = _ml(limit=1)
    assert cond.should_stop([_t("a", {"m": float("nan")})]) is False
    assert cond._running_total != cond._running_total


def test_ml_reset() -> None:
    cond = _ml(limit=1)
    cond.should_stop([_t("a", {"m": 5})])
    cond.reset()
    assert (cond._running_total, cond._last_index, cond._unmeasured_trials) == (0, 0, 0)


# ----------------------------- MaxSamples ----------------------------------


def _ms(n: int | None = 10, **kw: Any) -> MaxSamplesStopCondition:
    return MaxSamplesStopCondition(max_samples=n, **kw)


def test_ms_none_limit_returns_false_without_reading() -> None:
    class Boom:
        def __iter__(self) -> Any:
            raise AssertionError("must not iterate")

    assert _ms(None).should_stop(Boom()) is False  # type: ignore[arg-type]


def test_ms_empty_and_threshold() -> None:
    cond = _ms(3)
    assert cond.should_stop([]) is False
    cond._total_attempted = 3
    assert cond.should_stop([]) is True


def test_ms_accumulates_sources_in_order() -> None:
    cond = _ms(100)
    cond.should_stop(
        [
            _t("a", {"examples_attempted": 2}),
            _t("b", None, {"examples_attempted": 3}),
            _t("c", None, None, example_results=[1, 2, 3, 4]),
            _t("d", {"examples_attempted": "7"}),
            _t("e", {"examples_attempted": "bad"}),  # unparsable -> skipped
            _t("f", {}),  # nothing -> skipped
        ]
    )
    assert cond._total_attempted == 16
    assert cond._last_index == 6


def test_ms_stops_midway_and_marks_all_seen() -> None:
    cond = _ms(5)
    trials = [_t("a", {"examples_attempted": 5}), _t("b", {"examples_attempted": 9})]
    assert cond.should_stop(trials) is True
    assert cond._total_attempted == 5
    assert cond._last_index == 2


def test_ms_resumed_adjustments() -> None:
    cond = _ms(1000)
    cond.should_stop(
        [
            _t("a", {"examples_attempted": 10}, {"resumed_from_prior_run": True}),
            _t("b", {"examples_attempted": 10}, {"resumed_examples_reused": 4}),
            _t("c", {"examples_attempted": 3}, {"resumed_examples_reused": 9}),
            _t("d", {"examples_attempted": 5}, {"resumed_examples_reused": 0}),
            _t("e", {"examples_attempted": 5}, {"resumed_examples_reused": -2}),
            _t("f", {"examples_attempted": 5}, {"resumed_examples_reused": "3"}),
            _t("g", {"examples_attempted": 5}, {"resumed_examples_reused": True}),
        ]
    )
    # 0 + 6 + 0 + 5 + 5 + 5 + (5-1)
    assert cond._total_attempted == 25


def test_ms_resumed_flag_zeroes_even_when_zero_attempted_counts() -> None:
    cond = _ms(1)
    # attempted == 0 is not None: it is added (0) and compared
    assert cond.should_stop([_t("a", {"examples_attempted": 0})]) is False
    assert cond._total_attempted == 0


def test_ms_pruned_and_incremental_and_generator() -> None:
    cond = _ms(5, include_pruned=False)
    pruned = _t("p", {"examples_attempted": 50}, status=TrialStatus.PRUNED)
    trials = [pruned, _t("a", {"examples_attempted": 2})]
    assert cond.should_stop(iter(trials)) is False
    assert cond._total_attempted == 2
    trials = [*trials, _t("b", {"examples_attempted": 3})]
    assert cond.should_stop(trials) is True
    assert cond._total_attempted == 5


def test_ms_negative_attempted_reduces_total() -> None:
    cond = _ms(100)
    cond.should_stop(
        [_t("a", {"examples_attempted": 4}), _t("b", {"examples_attempted": -1})]
    )
    assert cond._total_attempted == 3
