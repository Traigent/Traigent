"""Odd-input characterization for select_examples_by_id (argument handling)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.regression_dataset import (
    UnresolvedExampleIdError,
    select_examples_by_id,
)


def _ds(*pairs):
    return Dataset(
        examples=[EvaluationExample(input_data=i, expected_output=o) for i, o in pairs],
        name="n",
        description="d",
        metadata={"k": [1]},
    )


def _run(*records):
    return SimpleNamespace(
        trials=[SimpleNamespace(metadata={"example_results": list(records)})]
    )


def _rec(eid, inp, exp):
    return {"example_id": eid, "input_data": inp, "expected_output": exp}


def test_non_dataset_rejected_first():
    with pytest.raises(TypeError, match="dataset must be a Dataset, got list"):
        select_examples_by_id([], "", None)


def test_string_and_bytes_ids_rejected_before_emptiness():
    for bad in ("abc", b"abc", ""):
        with pytest.raises(TypeError, match="not a single string"):
            select_examples_by_id(_ds(), bad, _run())


def test_empty_ids_value_error_precedes_type_check():
    with pytest.raises(ValueError, match="example_ids is empty"):
        select_examples_by_id(_ds(), [], _run())


def test_non_string_id_rejected():
    with pytest.raises(TypeError, match="example ids must be strings, got 5"):
        select_examples_by_id(_ds(), ["a", 5], _run())


def test_non_string_id_rejected_even_if_duplicates_collapse():
    with pytest.raises(TypeError):
        select_examples_by_id(_ds(), [None, None], _run())


def test_run_without_trials_attribute_is_unverifiable():
    with pytest.raises(UnresolvedExampleIdError) as info:
        select_examples_by_id(_ds(({"q": 1}, "a")), ["x"], object())
    assert info.value.unverifiable == ["x"]


def test_run_with_none_trials_is_unverifiable():
    with pytest.raises(UnresolvedExampleIdError) as info:
        select_examples_by_id(_ds(({"q": 1}, "a")), ["x"], SimpleNamespace(trials=None))
    assert info.value.unverifiable == ["x"]


def test_dataset_row_without_canonical_form_raises_value_error():
    ds = _ds(({"q": 1}, "a"), ({1: "a", "b": 2}, "x"))
    with pytest.raises(ValueError, match="Dataset row 1 has no canonical JSON form"):
        select_examples_by_id(ds, ["x"], _run(_rec("x", {"q": 1}, "a")))


def test_duplicate_requested_ids_collapse_and_order_follows_source():
    ds = _ds(({"q": 0}, "0"), ({"q": 1}, "1"), ({"q": 2}, "2"))
    run = _run(_rec("b", {"q": 2}, "2"), _rec("a", {"q": 0}, "0"))
    out = select_examples_by_id(ds, ["b", "a", "b"], run)
    assert [e.input_data for e in out.examples] == [{"q": 0}, {"q": 2}]
    assert [e.metadata for e in out.examples] == [
        {"example_id": "example_0"},
        {"example_id": "example_2"},
    ]
    assert (out.name, out.description, out.metadata) == ("n", "d", {"k": [1]})
    assert out.metadata is not ds.metadata


def test_all_three_failure_kinds_reported_together():
    ds = _ds(({"q": 0}, "0"), ({"q": 0}, "0"), ({"q": 5}, "5"))
    run = _run(
        _rec("amb", {"q": 0}, "0"),
        _rec("gone", {"q": 9}, "9"),
        _rec("ok", {"q": 5}, "5"),
    )
    with pytest.raises(UnresolvedExampleIdError) as info:
        select_examples_by_id(ds, ["amb", "gone", "none", "ok"], run)
    assert info.value.ambiguous == ["amb"]
    assert info.value.unresolved == ["gone"]
    assert info.value.unverifiable == ["none"]


def test_shared_row_ids_appended_after_per_id_ambiguities():
    ds = _ds(({"q": 0}, "0"), ({"q": 0, "z": 1}, "0"))
    run = _run(
        _rec("p", {"q": 0}, "0"),
        _rec("q", {"q": 0}, "0"),
        _rec("m", {"q": 0}, "0"),
    )
    # one row matches content of p, q, m? no: rows differ, so each resolves to row 0.
    with pytest.raises(UnresolvedExampleIdError) as info:
        select_examples_by_id(ds, ["p", "q"], run)
    assert info.value.ambiguous == ["p", "q"]
