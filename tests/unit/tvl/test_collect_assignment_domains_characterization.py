"""Characterization of tvl.spec_loader._collect_assignment_domains."""

from __future__ import annotations

import pytest

from traigent.tvl import spec_loader as sl


def _view(domains):
    return {
        k: (v.raw_type, v.domain, v.legacy_parameter) for k, v in domains.items()
    }


def test_empty_and_missing():
    assert sl._collect_assignment_domains({}) == {}
    assert sl._collect_assignment_domains({"tvars": None}) == {}
    assert sl._collect_assignment_domains({"tvars": "x", "configuration_space": 3}) == {}


def test_tvars_dict_form():
    out = sl._collect_assignment_domains(
        {
            "tvars": {
                "a": {"type": "int", "domain": [1, 2]},
                "b": {"type": 5, "domain": None},
                "c": {},
                "d": "notadict",
                3: {"type": "int"},
            }
        }
    )
    assert _view(out) == {
        "a": ("int", [1, 2], False),
        "b": (None, None, False),
        "c": (None, None, False),
    }


def test_tvars_list_form():
    out = sl._collect_assignment_domains(
        {
            "tvars": [
                {"name": "a", "type": "float", "domain": {"range": [0, 1]}},
                "skip",
                {"name": "", "type": "int"},
                {"name": 4, "type": "int"},
                {"type": "int"},
                {"name": "b", "type": ["x"]},
                {"name": "a", "type": "int", "domain": [1]},
            ]
        }
    )
    assert _view(out) == {"a": ("int", [1], False), "b": (None, None, False)}
    assert list(out) == ["a", "b"]


def test_configuration_space_overrides_tvars_and_marks_legacy():
    out = sl._collect_assignment_domains(
        {
            "tvars": {"a": {"type": "int", "domain": [1]}},
            "configuration_space": {
                "a": {"type": "float", "default": 1},
                "z": "bad",
                7: {"type": "int"},
            },
        }
    )
    assert set(out) == {"a"}
    assert out["a"].legacy_parameter is True
    assert out["a"].raw_type == sl._legacy_parameter_type({"type": "float", "default": 1})
    assert out["a"].domain == {"type": "float", "default": 1}


@pytest.mark.parametrize("cfg", [[], None, "x"])
def test_non_dict_configuration_space_ignored(cfg):
    out = sl._collect_assignment_domains(
        {"tvars": [{"name": "a"}], "configuration_space": cfg}
    )
    assert list(out) == ["a"]
