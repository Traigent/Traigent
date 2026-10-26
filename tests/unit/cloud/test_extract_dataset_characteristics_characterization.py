"""Characterization of SDKBackendBridge._extract_dataset_characteristics."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from traigent.cloud.backend_bridges import bridge


def _ex(inp=None, out=None, **kw):
    return NS(input_data=inp, expected_output=out, **kw)


def _go(ds):
    return bridge._extract_dataset_characteristics(ds)


def test_empty_object():
    assert _go(NS()) == {}


def test_name_is_included_even_when_none():
    assert _go(NS(name=None)) == {"name": None}


def test_empty_examples_adds_nothing():
    assert _go(NS(name="n", examples=[])) == {"name": "n"}
    assert _go(NS(examples=None)) == {}


def test_single_example_has_no_consistency_key():
    out = _go(NS(examples=[_ex({"a": 1}, "x")]))
    assert out == {"example_count": 1, "input_type": "dict", "output_type": "str"}


def test_consistent_and_inconsistent_inputs():
    same = [_ex("a", 1), _ex("b", 2), _ex("c", 3)]
    assert _go(NS(examples=same))["structure_consistent"] is True
    mixed = [_ex("a"), _ex("b"), _ex(5)]
    assert _go(NS(examples=mixed))["structure_consistent"] is False


def test_only_first_five_examples_sampled():
    exs = [_ex("a")] * 5 + [_ex(1)]
    assert _go(NS(examples=exs))["structure_consistent"] is True
    exs = [_ex("a")] * 4 + [_ex(1)]
    assert _go(NS(examples=exs))["structure_consistent"] is False


def test_examples_without_input_data_attribute_are_skipped_in_consistency():
    exs = [_ex("a"), NS(), NS(expected_output=1)]
    out = _go(NS(examples=exs))
    assert out["structure_consistent"] is True
    assert out["input_type"] == "str"


def test_first_example_without_input_data_with_second_having_it_is_unbound():
    exs = [NS(expected_output=1), _ex("a")]
    with pytest.raises(UnboundLocalError):
        _go(NS(examples=exs))


def test_first_example_without_input_data_and_none_after_is_fine():
    exs = [NS(expected_output=1), NS()]
    out = _go(NS(examples=exs))
    assert out == {
        "example_count": 2,
        "output_type": "int",
        "structure_consistent": True,
    }


def test_metadata_only_stable_keys_in_order():
    md = {"domain": "d", "extra": 1, "version": 2, "task_type": None}
    out = _go(NS(metadata=md))
    assert out == {"meta_version": 2, "meta_task_type": None, "meta_domain": "d"}
    assert list(out) == ["meta_version", "meta_task_type", "meta_domain"]


@pytest.mark.parametrize("md", [None, {}])
def test_falsy_metadata_ignored(md):
    assert _go(NS(metadata=md)) == {}


def test_key_order_name_examples_metadata():
    ds = NS(name="n", examples=[_ex("a", "b")], metadata={"source": "s"})
    assert list(_go(ds)) == [
        "name",
        "example_count",
        "input_type",
        "output_type",
        "meta_source",
    ]
