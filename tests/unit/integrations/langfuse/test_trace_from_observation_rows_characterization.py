"""Golden characterization for LangfuseClient._trace_from_observation_rows."""

from __future__ import annotations

import logging

from traigent.integrations.langfuse.client import LangfuseClient

build = LangfuseClient._trace_from_observation_rows


def test_empty_rows_is_none():
    assert build("t", [], False) is None
    assert build("t", [], True, include_io=True) is None


def test_no_root_leaves_root_fields_unset(caplog):
    rows = [{"id": "a", "parentObservationId": "p", "name": "child"}]
    with caplog.at_level(logging.WARNING):
        out = build("t1", rows, True)
    assert out == {
        "id": "t1",
        "name": None,
        "metadata": None,
        "sessionId": None,
        "userId": None,
        "observations": rows,
        "observationsPartial": True,
    }
    assert "Trace t1 has no root observation." in caplog.text


def test_no_root_with_io_sets_none_io():
    out = build("t", [{"id": "a", "parentObservationId": "p"}], False, include_io=True)
    assert out["input"] is None and out["output"] is None


def test_physical_root_fields_and_defaults():
    rows = [
        {"id": "c", "parentObservationId": "r", "name": "child"},
        {
            "id": "r",
            "parentObservationId": None,
            "name": "rootname",
            "metadata": None,
            "sessionId": "s",
            "userId": "u",
        },
    ]
    out = build("t", rows, False)
    assert out["name"] == "rootname"
    assert out["metadata"] == {}
    assert (out["sessionId"], out["userId"]) == ("s", "u")
    assert out["observations"] is rows
    assert out["observationsPartial"] is False
    assert "input" not in out and "output" not in out


def test_trace_name_preferred_and_falsey_falls_back():
    rows = [{"id": "r", "parentObservationId": None, "traceName": "", "name": "n"}]
    assert build("t", rows, False)["name"] == "n"
    rows = [{"id": "r", "parentObservationId": None, "traceName": "tn", "name": "n"}]
    assert build("t", rows, False)["name"] == "tn"
    rows = [{"id": "r", "parentObservationId": None}]
    assert build("t", rows, False)["name"] is None


def test_logical_root_beats_physical_and_earliest_wins_with_warning(caplog):
    rows = [
        {"id": "phys", "parentObservationId": None, "name": "phys"},
        {"id": "b", "isRootObservation": True, "startTime": "2024-02", "name": "b"},
        {"id": "a", "isRootObservation": True, "startTime": "2024-01", "name": "a"},
        {"id": "z", "isRootObservation": True, "name": "z-no-start"},
    ]
    with caplog.at_level(logging.WARNING):
        out = build("t", rows, False)
    assert out["name"] == "a"
    assert "Trace t has 3 root observations; using the earliest (a)." in caplog.text


def test_tie_broken_by_id_and_missing_ids_sort_first():
    rows = [
        {"id": "y", "isRootObservation": True, "startTime": "x", "name": "y"},
        {"id": "x", "isRootObservation": True, "startTime": "x", "name": "x"},
    ]
    assert build("t", rows, False)["name"] == "x"
    rows = [
        {"isRootObservation": True, "startTime": "x", "name": "noid"},
        {"id": "a", "isRootObservation": True, "startTime": "x", "name": "a"},
    ]
    assert build("t", rows, False)["name"] == "noid"


def test_isRootObservation_must_be_exactly_true():
    rows = [{"id": "a", "isRootObservation": 1, "parentObservationId": "p"}]
    assert build("t", rows, False)["name"] is None


def test_include_io_decodes_json_strings_only():
    rows = [
        {
            "id": "r",
            "parentObservationId": None,
            "input": '{"a": 1}',
            "output": "not json",
        }
    ]
    out = build("t", rows, False, include_io=True)
    assert out["input"] == {"a": 1}
    assert out["output"] == "not json"
    rows = [{"id": "r", "parentObservationId": None, "input": {"k": 1}}]
    out = build("t", rows, False, include_io=True)
    assert out["input"] == {"k": 1} and out["output"] is None


def test_single_root_no_multi_root_warning(caplog):
    rows = [{"id": "r", "parentObservationId": None}]
    with caplog.at_level(logging.WARNING):
        build("t", rows, False)
    assert caplog.text == ""
