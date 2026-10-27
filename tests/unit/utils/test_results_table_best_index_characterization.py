"""Characterization tests for results_table best-trial helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.utils.results_table import _find_best_trial_index, _trials_all_failed


def _t(trial_id, metrics, config=None, ok=True, metadata=None):
    return SimpleNamespace(
        trial_id=trial_id,
        metrics=metrics,
        config=config if config is not None else {"id": trial_id},
        is_successful=ok,
        metadata=metadata or {},
    )


class _Results:
    def __init__(self, weighted=None, raises=None, **attrs):
        self._weighted = weighted
        self._raises = raises
        self.calls = []
        self.metadata = attrs.pop("metadata", {})
        for k, v in attrs.items():
            setattr(self, k, v)

    def calculate_weighted_scores(self, objective_schema=None):
        self.calls.append(objective_schema)
        if self._raises:
            raise self._raises
        return self._weighted


INFO = [("accuracy", "maximize")]


def test_best_trial_id_in_metadata_wins():
    trials = [_t("a", {"accuracy": 0.9}), _t("b", {"accuracy": 0.1})]
    res = _Results(metadata={"best_trial_id": "b"}, weighted={"weighted_scores": []})
    assert _find_best_trial_index(res, trials, INFO, None) == 1
    assert res.calls == []  # short-circuit before weighted scoring


def test_best_trial_id_attribute_used_when_metadata_lacks_it():
    trials = [_t("a", {"accuracy": 0.9}), _t("b", {"accuracy": 0.1})]
    res = _Results(best_trial_id="b", weighted={"weighted_scores": []})
    assert _find_best_trial_index(res, trials, INFO, None) == 1


def test_non_dict_metadata_ignored():
    trials = [_t("a", {"accuracy": 0.9})]
    res = _Results(metadata="junk", weighted={"weighted_scores": []})
    assert _find_best_trial_index(res, trials, INFO, None) == 0


def test_no_ranking_eligible_reason_returns_none():
    trials = [_t("a", {"accuracy": 0.9})]
    res = _Results(
        best_config=None,
        reason_code="NO_RANKING_ELIGIBLE_TRIALS",
        weighted={"weighted_scores": []},
    )
    assert _find_best_trial_index(res, trials, INFO, None) is None
    assert res.calls == []


def test_reason_code_ignored_when_best_config_present():
    trials = [_t("a", {"accuracy": 0.9}, config={"x": 1})]
    res = _Results(
        best_config={"x": 1},
        reason_code="NO_RANKING_ELIGIBLE_TRIALS",
        weighted={"weighted_scores": []},
    )
    assert _find_best_trial_index(res, trials, INFO, None) == 0


def test_weighted_scores_choose_winner_and_receive_schema():
    t1, t2 = _t("a", {"accuracy": 0.9}), _t("b", {"accuracy": 0.1})
    res = _Results(weighted={"weighted_scores": [(t1, 0.2), (t2, 0.8)]})
    sentinel = object()
    assert _find_best_trial_index(res, [t1, t2], INFO, sentinel) == 1
    assert res.calls == [sentinel]


def test_weighted_failure_falls_back_to_unweighted_primary_metric():
    trials = [_t("a", {"accuracy": 0.2}), _t("b", {"accuracy": 0.9})]
    res = _Results(raises=RuntimeError("boom"))
    assert _find_best_trial_index(res, trials, INFO, None) == 1


def test_weighted_returning_none_scores_falls_back():
    trials = [_t("a", {"accuracy": 0.2}), _t("b", {"accuracy": 0.9})]
    res = _Results(weighted={"weighted_scores": None})
    assert _find_best_trial_index(res, trials, INFO, None) == 1


def test_band_primary_metric_uses_best_config_fallback_with_score_match():
    trials = [
        _t("a", {"lat": 1.0}, config={"m": 1}),
        _t("b", {"lat": 2.0}, config={"m": 1}),
    ]
    res = _Results(
        weighted={"weighted_scores": []}, best_config={"m": 1}, best_score=2.0
    )
    assert _find_best_trial_index(res, trials, [("lat", "band")], None) == 1


def test_fallback_without_score_returns_first_matching_config():
    trials = [
        _t("a", {"lat": 1.0}, config={"m": 0}),
        _t("b", {"lat": 2.0}, config={"m": 1}),
        _t("c", {"lat": 3.0}, config={"m": 1}),
    ]
    res = _Results(weighted={"weighted_scores": []}, best_config={"m": 1})
    assert _find_best_trial_index(res, trials, [("lat", "band")], None) == 1


def test_fallback_no_match_returns_none():
    trials = [_t("a", {"lat": 1.0}, config={"m": 0})]
    res = _Results(
        weighted={"weighted_scores": []}, best_config={"m": 9}, best_score=1.0
    )
    assert _find_best_trial_index(res, trials, [("lat", "band")], None) is None


def test_fallback_score_mismatch_returns_none():
    trials = [_t("a", {"lat": 1.0}, config={"m": 1})]
    res = _Results(
        weighted={"weighted_scores": []}, best_config={"m": 1}, best_score=5.0
    )
    assert _find_best_trial_index(res, trials, [("lat", "band")], None) is None


def test_fallback_skips_trial_without_primary_metric_value():
    trials = [_t("a", {}, config={"m": 1})]
    res = _Results(
        weighted={"weighted_scores": []}, best_config={"m": 1}, best_score=1.0
    )
    assert _find_best_trial_index(res, trials, [("lat", "band")], None) is None


def test_fallback_without_metric_names_returns_first_config_match():
    trials = [_t("a", {}, config={"m": 1})]
    res = _Results(
        weighted={"weighted_scores": []}, best_config={"m": 1}, best_score=1.0
    )
    assert _find_best_trial_index(res, trials, [], None) == 0


# --- _trials_all_failed ---------------------------------------------------


def _s(value, ok=True):
    return _t("x", {}, ok=ok, metadata={"successful_examples": value})


@pytest.mark.parametrize(
    "trials,expected",
    [
        ([], True),
        ([_s(0, ok=False)], True),
        ([_s(0)], True),
        ([_s(0.0)], True),
        ([_s(-1)], True),
        ([_s(2)], False),
        ([_s(0.5)], False),
        ([_s(True)], False),
        ([_s(False)], False),
        ([_s("3")], False),
        ([_s("0")], True),
        ([_s("0.0")], True),
        ([_s("abc")], False),
        ([_s([1])], False),
        ([_t("x", {})], False),
        ([_s(0), _s(0), _s(4)], False),
        ([_s(0), _s("zzz")], False),
        ([_s(4, ok=False), _s(0)], True),
    ],
)
def test_trials_all_failed(trials, expected):
    assert _trials_all_failed(trials) is expected
