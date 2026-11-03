"""Characterization for regression_dataset._recorded_content."""

from __future__ import annotations

from types import SimpleNamespace

from traigent.utils.regression_dataset import _canonical, _recorded_content


def _trial(records, key="example_results"):
    return SimpleNamespace(metadata={key: records})


def _rec(eid, inp, exp=None):
    return {"example_id": eid, "input_data": inp, "expected_output": exp}


class _Rec:
    """Record object that logs attribute reads."""

    def __init__(self, **kw):
        self._kw = kw
        self.log: list[str] = []

    def __getattr__(self, name):
        if name.startswith("_") or name == "log":
            raise AttributeError(name)
        self.log.append(name)
        return self._kw.get(name)


class _LoggingMeta(dict):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log: list[str] = []

    def get(self, key, default=None):
        self.log.append(key)
        return super().get(key, default)


def test_empty_and_non_list_records_skipped():
    assert _recorded_content([]) == {}
    trials = [
        SimpleNamespace(),  # no metadata
        SimpleNamespace(metadata=None),
        SimpleNamespace(metadata="x"),
        SimpleNamespace(metadata=["example_results"]),
        _trial("not a list"),
        _trial(None),
        _trial({"a": 1}),
        _trial((_rec("a", 1),)),  # tuple is not a list
        _trial([], key="other"),
    ]
    assert _recorded_content(trials) == {}


def test_basic_collection_and_id_stringification():
    out = _recorded_content([_trial([_rec(1, "q", "a"), _rec("1", "q", "a")])])
    assert out == {"1": {_canonical("q", "a")}}


def test_none_example_id_skipped_but_falsey_ids_kept():
    out = _recorded_content([_trial([_rec(None, "q"), _rec(0, "q"), _rec("", "q")])])
    assert set(out) == {"0", ""}


def test_multiple_contents_across_trials_merge():
    out = _recorded_content(
        [_trial([_rec("a", "q1", "x")]), _trial([_rec("a", "q2", "x")])]
    )
    assert out == {"a": {_canonical("q1", "x"), _canonical("q2", "x")}}


def test_missing_input_marks_none_and_is_sticky():
    out = _recorded_content(
        [_trial([_rec("a", "q"), _rec("a", None), _rec("a", "q2"), _rec("b", None)])]
    )
    assert out == {"a": None, "b": None}


def test_none_after_missing_input_not_revived_by_later_record():
    out = _recorded_content([_trial([_rec("a", None)]), _trial([_rec("a", "q")])])
    assert out == {"a": None}


def test_uncanonicalizable_content_marks_none():
    bad = {1: "x", "a": "y"}  # mixed key types -> TypeError in sort_keys
    out = _recorded_content([_trial([_rec("a", "q"), _rec("a", bad), _rec("a", "z")])])
    assert out == {"a": None}


def test_falsey_input_data_is_content_not_missing():
    out = _recorded_content([_trial([_rec("a", 0), _rec("b", ""), _rec("c", [])])])
    assert all(isinstance(v, set) and len(v) == 1 for v in out.values())


def test_object_records_and_attribute_read_order():
    r = _Rec(example_id="a", input_data="q", expected_output="e")
    out = _recorded_content([_trial([r])])
    assert out == {"a": {_canonical("q", "e")}}
    assert r.log == ["example_id", "input_data", "expected_output"]
    r2 = _Rec(example_id="a", input_data=None)
    _recorded_content([_trial([r2])])
    assert r2.log == ["example_id", "input_data"]
    r3 = _Rec(example_id=None, input_data="q")
    _recorded_content([_trial([r3])])
    assert r3.log == ["example_id"]


def test_sticky_none_skips_expected_output_read():
    first = _rec("a", None)
    r = _Rec(example_id="a", input_data="q", expected_output="e")
    _recorded_content([_trial([first, r])])
    assert r.log == ["example_id", "input_data"]


def test_metadata_read_once_per_trial():
    meta = _LoggingMeta(example_results=[])
    _recorded_content([SimpleNamespace(metadata=meta)])
    assert meta.log == ["example_results"]


def test_accepts_generator_of_trials():
    out = _recorded_content(t for t in [_trial([_rec("a", "q")])])
    assert set(out) == {"a"}
