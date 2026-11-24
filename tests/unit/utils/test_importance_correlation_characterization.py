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
    # No interval is invented (issue #2945)
    assert res.confidence_interval is None


def test_categorical_score_does_not_depend_on_first_seen_order():
    values = ["b", "a", "b", "c", "a", "b", "c", "a", "b", "c", "a", "c"]
    level = {"a": 1.0, "b": 0.0, "c": 5.0}
    objs = [level[v] for v in values]
    res = _run(values, objs)
    # Groups explain all variance: correlation ratio is 1.0 whatever the order
    assert res.importance_score == pytest.approx(1.0)
    assert res.sample_size == 12
    # Relabel so the first-seen order differs; the score must not change
    relabel = {"b": "c", "a": "b", "c": "a"}
    res2 = _run([relabel[v] for v in values], objs)
    assert res2.importance_score == pytest.approx(res.importance_score)


def test_bool_counts_as_numeric_and_other_types_are_skipped():
    values = [True, False] * 3 + [None, [1], {"a": 1}, 1.5, 2, 6, 7, 8, 9]
    res = _run(values)
    # None / list / dict skipped; 6 bool + 1.5, 2, 6, 7, 8, 9 -> 12 valid
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


def test_ten_values_boundary_is_measured():
    res = _run(list(range(10)))
    assert res.sample_size == 10
    assert res.confidence_interval is None


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
