"""Characterization for ParameterImportanceAnalyzer._calculate_parameter_correlation_importance."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.utils.importance import ParameterImportanceAnalyzer


class _LoggingConfig(dict):
    log: list[tuple[str, str]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []

    def __contains__(self, key):
        self.log.append(("in", key))
        return super().__contains__(key)

    def __getitem__(self, key):
        self.log.append(("get", key))
        return super().__getitem__(key)


def _trials(values, key="p"):
    return [
        SimpleNamespace(config={key: v} if v is not _MISSING else {}) for v in values
    ]


_MISSING = object()


def _run(values, objectives=None):
    trials = _trials(values)
    objectives = objectives or [float(i) for i in range(len(values))]
    analyzer = ParameterImportanceAnalyzer()
    return analyzer._calculate_parameter_correlation_importance(trials, "p", objectives)


def test_too_few_values_returns_none():
    assert _run(list(range(9))) is None


def test_constant_values_return_none():
    assert _run([3] * 12) is None


def test_numeric_perfect_correlation():
    res = _run(list(range(12)))
    assert res.parameter_name == "p"
    assert res.method == "correlation_based"
    assert res.sample_size == 12
    assert res.importance_score == pytest.approx(1.0)
    assert res.confidence_interval[1] == 1.0
    assert res.confidence_interval[0] == pytest.approx(1.0 - 1.96 / 3.0)


def test_categorical_encoding_is_first_seen_order():
    values = ["b", "a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "c"]
    # b->0, a->1, c->2 so objectives track the encoding exactly
    expected = [0, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 2]
    objs = [{0: 0.0, 1: 1.0, 2: 2.0}[e] for e in expected]
    res = _run(values, objs)
    assert res.importance_score == pytest.approx(1.0)
    assert res.sample_size == 12


def test_bool_counts_as_numeric_and_other_types_are_skipped():
    values = [True, False] * 3 + [None, [1], {"a": 1}, 1.5, 2, "s", 7, 8, 9]
    res = _run(values)
    # None / list / dict skipped; 6 bool + 1.5, 2, "s", 7, 8, 9 -> 12 valid
    assert res.sample_size == 12


def test_missing_parameter_is_skipped_and_indices_align_objectives():
    values = [_MISSING, 1, 2, _MISSING, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    objectives = [100.0, 1, 2, 100.0, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    res = _run(values, objectives)
    assert res.sample_size == 11
    assert res.importance_score == pytest.approx(1.0)


def test_valid_indices_skip_dropped_trials():
    values = [1, None, 2, None, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    objectives = [1, 999, 2, -999, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    res = _run(values, objectives)
    assert res.sample_size == 12
    assert res.importance_score == pytest.approx(1.0)


def test_small_sample_uses_fallback_error_via_ten_values_boundary():
    res = _run(list(range(10)))
    assert res.sample_size == 10
    assert res.confidence_interval[0] == pytest.approx(1.0 - 1.96 / (7**0.5))


def test_config_lookup_is_membership_then_single_read_per_trial():
    configs = [_LoggingConfig({"p": float(i)}) for i in range(12)]
    configs.append(_LoggingConfig({"other": 1}))
    trials = [SimpleNamespace(config=c) for c in configs]
    analyzer = ParameterImportanceAnalyzer()
    res = analyzer._calculate_parameter_correlation_importance(
        trials, "p", [float(i) for i in range(13)]
    )
    assert res.sample_size == 12
    for c in configs[:12]:
        assert c.log == [("in", "p"), ("get", "p")]
    assert configs[12].log == [("in", "p")]
