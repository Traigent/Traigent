"""Characterization of wire_rows_from_session odd inputs and check ordering."""

from __future__ import annotations

import pytest

from traigent.reporting.example_map import wire_rows_from_session
from traigent.utils.exceptions import ValidationError


def _t(measures, rows="full", extra_meta=None, config=None):
    meta = {"measures": measures, "_traigent_local": {"rows": rows}}
    meta.update(extra_meta or {})
    trial = {"metadata": meta}
    if config is not None:
        trial["config"] = config
    return trial


def _m(i):
    return {"example_id": i}


def _sess(trials):
    return {"metadata": {"backend_session": True}, "trials": trials}


def test_trial_with_no_metadata_or_measures_is_skipped():
    s = _sess([{"metadata": None}, {"metadata": {"measures": None}}, _t([_m("example_1")])])
    assert wire_rows_from_session(s) == {"example_1": 1}


def test_trial_missing_metadata_key_is_skipped():
    s = _sess([{}, _t([_m("ex_319aca3f_2")])])
    assert wire_rows_from_session(s) == {"ex_319aca3f_2": 2}


def test_empty_measures_list_contributes_nothing_then_empty_raises():
    with pytest.raises(ValidationError, match="no per-example measures"):
        wire_rows_from_session(_sess([_t([])]))


@pytest.mark.parametrize(
    "session, text",
    [
        ({"trials": "x"}, "'trials'"),
        ({}, "'trials'"),
        (_sess([None]), "a trial is not an object"),
        (_sess([{"metadata": []}]), "trial metadata is not an object"),
        (_sess([{"metadata": {"measures": {}}}]), "measures is not a list"),
        (_sess([_t([{"example_id": 3}])]), "unrecognised example id"),
        (_sess([_t([None])]), "unrecognised example id"),
        (_sess([_t([{"no": 1}])]), "unrecognised example id"),
        (_sess([_t([_m("example_1\n")])]), "unrecognised example id"),
        (
            {"metadata": 3, "trials": [_t([_m("example_0")])]},
            "session metadata is not an object",
        ),
    ],
)
def test_malformed(session, text):
    with pytest.raises(ValidationError, match=text):
        wire_rows_from_session(session)


def test_invalid_id_checked_before_privacy_refusal():
    t = _t([_m("bad")], extra_meta={"_traigent_local": {"rows": "full", "privacy_mode": True}})
    with pytest.raises(ValidationError, match="unrecognised"):
        wire_rows_from_session(_sess([t]))


def test_privacy_refusal_connected_only_and_before_subset_check():
    meta = {"_traigent_local": {"privacy_mode": True}}  # no rows -> subset would raise
    t = _t([_m("example_0")], extra_meta=meta)
    with pytest.raises(ValidationError, match="privacy mode"):
        wire_rows_from_session(_sess([t]), "connected")


def test_privacy_only_when_exactly_true():
    t = _t([_m("example_0")], extra_meta={"_traigent_local": {"rows": "full", "privacy_mode": "yes"}})
    assert wire_rows_from_session(_sess([t]), "connected") == {"example_0": 0}


def test_unordered_without_assume_full_refused_and_with_accepted():
    t = _t([_m("example_2")], rows="unordered")
    with pytest.raises(ValidationError):
        wire_rows_from_session(_sess([t]))
    assert wire_rows_from_session(_sess([t]), assume_full_dataset=True) == {"example_2": 2}


def test_subset_maps_and_same_row_repeat_is_ok():
    t1 = _t([_m("example_0"), _m("example_1")], rows=[5, 3])
    t2 = _t([_m("example_0")], rows=[5])
    assert wire_rows_from_session(_sess([t1, t2])) == {"example_0": 5, "example_1": 3}


def test_conflict_and_outside_subset():
    with pytest.raises(ValidationError, match="different dataset rows"):
        wire_rows_from_session(
            _sess([_t([_m("example_0")], rows=[1]), _t([_m("example_0")], rows=[2])])
        )
    with pytest.raises(ValidationError, match="outside its trial subset"):
        wire_rows_from_session(_sess([_t([_m("example_1")], rows=[1])]))


def test_synced_skips_empty_config_trials():
    with pytest.raises(ValidationError, match="no per-example measures"):
        wire_rows_from_session(_sess([_t([_m("example_0")], config={})]), "synced")


def test_invalid_target_rejected():
    with pytest.raises(ValidationError):
        wire_rows_from_session(_sess([_t([_m("example_0")])]), "nope")
