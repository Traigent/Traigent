"""Golden characterization for effectuation._iter_declared_knobs."""

from __future__ import annotations

from types import SimpleNamespace

from traigent.effectuation import _iter_declared_knobs as it


def test_none_and_scalars():
    assert it(None) == []
    assert it(5) == []
    assert it("name") == []
    assert it(b"x") == []


def test_single_knob_mapping_is_copied():
    src = {"name": "k", "kind": "value"}
    out = it(src)
    assert out == [src]
    assert out[0] is not src


def test_section_keys_in_order_and_none_sections_skipped():
    out = it(
        {
            "tvars": [{"name": "t"}],
            "knobs": [{"name": "k"}],
            "recommendations": None,
            "configuration_space": {"zzz_unknown": 1},
        }
    )
    assert out == [{"name": "k"}, {"name": "t"}]


def test_sections_nest_recursively():
    out = it({"knobs": {"tvars": [[{"name": "a"}, {"name": "b"}]]}})
    assert out == [{"name": "a"}, {"name": "b"}]


def test_empty_sections_fall_through_to_flat_entries():
    out = it({"knobs": [], "candidate_count": 3, "_x": 1})
    assert out == [
        {
            "name": "candidate_count",
            "kind": "cardinality",
            "value_set": 3,
            "effectuation_strategy": "self_consistency",
        }
    ]
    assert it({"knobs": []}) == []


def test_flat_entries_skip_underscore_and_non_str_keys_and_unknown():
    assert it({"_hidden": 1, 5: 2, "totally_unknown_name_xyz": 3}) == []


def test_sequences_flatten_and_skip_junk():
    out = it([{"name": "a"}, None, "s", [{"name": "b"}], 7])
    assert out == [{"name": "a"}, {"name": "b"}]
    assert it(()) == []
    assert it(({"name": "t"},)) == [{"name": "t"}]


def test_objects_with_section_attrs_then_name_attr():
    inner = SimpleNamespace(name="i")
    holder = SimpleNamespace(knobs=[inner], tvars=[{"name": "t"}])
    assert it(holder) == [inner, {"name": "t"}]
    named = SimpleNamespace(name="solo")
    assert it(named) == [named]
    assert it(SimpleNamespace(knobs=[])) == []
    # section attrs that yield nothing fall back to the name attribute
    both = SimpleNamespace(name="n", knobs=[])
    assert it(both) == [both]
    assert it(SimpleNamespace(other=1)) == []
