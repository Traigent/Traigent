"""Golden characterization for spec_loader._parse_configuration_space and
_validate_configuration_space_domains."""

from __future__ import annotations

import pytest

from traigent.tvl.spec_loader import (
    _parse_configuration_space,
    _validate_configuration_space_domains,
)
from traigent.utils.exceptions import TVLValidationError


def _msg(section):
    with pytest.raises(TVLValidationError) as ei:
        _parse_configuration_space({"configuration_space": section})
    return str(ei.value)


@pytest.mark.parametrize("section", [None, {}, [], "x", 5])
def test_parse_requires_nonempty_mapping(section):
    assert _msg(section) == "configuration_space must be a non-empty mapping"
    with pytest.raises(TVLValidationError):
        _parse_configuration_space({})


def test_parse_name_and_definition_checks():
    assert _msg({1: {"type": "boolean"}}) == (
        "Configuration parameter names must be strings"
    )
    assert _msg({"a": []}) == "Parameter 'a' must be defined as a mapping"
    assert _msg({"a": {}}) == "Parameter 'a' must define 'type', 'values', or 'range'"
    assert _msg({"a": {"type": ""}}) == (
        "Parameter 'a' must define 'type', 'values', or 'range'"
    )
    assert _msg({"a": {"type": None, "unit": "ms"}}) == (
        "Parameter 'a' must define 'type', 'values', or 'range'"
    )


def test_parse_first_error_wins_in_order():
    assert _msg({"a": {"type": "boolean"}, 2: {}, "c": []}) == (
        "Configuration parameter names must be strings"
    )
    assert _msg({"a": [], 2: {}}) == "Parameter 'a' must be defined as a mapping"


def test_parse_non_string_type_raises_attribute_error():
    with pytest.raises(AttributeError):
        _parse_configuration_space({"configuration_space": {"a": {"type": 5}}})


def test_parse_inferred_types_and_normalisation():
    space, defaults, units = _parse_configuration_space(
        {
            "configuration_space": {
                "cat": {"values": [1, 2], "default": 2, "unit": "x"},
                "rng": {"range": [0, 3], "unit": 5},
                "typed_upper": {"type": "INTEGER", "range": {"min": 1.9, "max": 4}},
                "flt": {"type": "Float", "range": {"start": 0, "end": 1}, "default": 0},
                "b": {"type": "boolean", "default": None, "unit": "flag"},
                "d": {"type": "discrete", "values": ["a"]},
                "c": {"type": "continuous", "range": (1, 2)},
                "both": {"values": [1], "range": [0, 1]},
            }
        }
    )
    assert space == {
        "cat": [1, 2],
        "rng": (0.0, 3.0),
        "typed_upper": (1, 4),
        "flt": (0.0, 1.0),
        "b": [True, False],
        "d": ["a"],
        "c": (1.0, 2.0),
        "both": [1],
    }
    assert defaults == {"cat": 2, "flt": 0, "b": None}
    assert units == {"cat": "x", "b": "flag"}
    assert list(space) == ["cat", "rng", "typed_upper", "flt", "b", "d", "c", "both"]


def test_parse_bad_normalisation_errors():
    assert _msg({"a": {"type": "categorical", "values": []}}) == (
        "Parameter 'a' must define a non-empty 'values' list"
    )
    assert _msg({"a": {"type": "integer"}}) == (
        "Parameter 'a' requires a numeric range with min/max values"
    )
    assert _msg({"a": {"type": "weird"}}) == (
        "Unsupported parameter type 'weird' for 'a'"
    )
    # error from an earlier parameter wins over a later one
    assert _msg({"x": {"type": "weird"}, "y": {}}) == (
        "Unsupported parameter type 'weird' for 'x'"
    )


def test_validate_domains_issue_order_and_shapes():
    issues = _validate_configuration_space_domains(
        {
            1: {"type": "boolean"},
            "notmap": [],
            "notype": {},
            "weird": {"type": "weirdo"},
            "cat_ok": {"type": "categorical", "values": [1, 2], "default": 1},
            "cat_bad": {"values": [1, 2], "default": 9},
            "flt_bad": {"type": "float", "range": [0, 1], "default": 5},
            "int_bad": {"type": "int", "range": [0, 1], "default": 5},
            "bool_bad": {"type": "boolean", "default": "yes"},
            "bool_ok": {"type": "bool", "default": True},
            "bool_nodefault": {"type": "boolean"},
            "UP": {"type": "INTEGER", "range": [0, 3], "default": 2},
        }
    )
    assert issues[:4] == [
        "Configuration parameter name must be string, got int",
        "Parameter 'notmap' must be defined as a mapping",
        "Parameter 'notype' must define 'type', 'values', or 'range'",
        "Unsupported parameter type 'weirdo' for 'weird'",
    ]
    assert len(issues) == 8
    assert sum("cat_bad" in i for i in issues) >= 1
    assert sum("flt_bad" in i for i in issues) >= 1
    assert sum("int_bad" in i for i in issues) >= 1
    assert sum("bool_bad" in i for i in issues) >= 1
    assert not any(
        "cat_ok" in i or "bool_ok" in i or "bool_nodefault" in i or "UP" in i
        for i in issues
    )
    assert issues == _validate_configuration_space_domains(
        {
            1: {"type": "boolean"},
            "notmap": [],
            "notype": {},
            "weird": {"type": "weirdo"},
            "cat_ok": {"type": "categorical", "values": [1, 2], "default": 1},
            "cat_bad": {"values": [1, 2], "default": 9},
            "flt_bad": {"type": "float", "range": [0, 1], "default": 5},
            "int_bad": {"type": "int", "range": [0, 1], "default": 5},
            "bool_bad": {"type": "boolean", "default": "yes"},
            "bool_ok": {"type": "bool", "default": True},
            "bool_nodefault": {"type": "boolean"},
            "UP": {"type": "INTEGER", "range": [0, 3], "default": 2},
        }
    )


def test_validate_domains_type_precedence():
    # empty/non-str type falls back to values/range inference
    assert (
        _validate_configuration_space_domains({"a": {"type": "", "values": [1]}}) == []
    )
    assert (
        _validate_configuration_space_domains({"a": {"type": 7, "values": [1]}}) == []
    )
    assert _validate_configuration_space_domains({}) == []


class _NoDefaultLookup(dict):
    """Mapping whose ``.get("default", ...)`` must never be called."""

    def get(self, key, *args):
        if key == "default":
            raise AssertionError("default looked up via .get")
        return super().get(key, *args)


class _RecordingLookup(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[str] = []

    def get(self, key, *args):
        self.calls.append(key)
        return super().get(key, *args)


def test_validate_domains_boolean_does_not_use_get_default():
    assert (
        _validate_configuration_space_domains({"a": _NoDefaultLookup(type="boolean")})
        == []
    )


def test_validate_domains_unsupported_type_does_not_use_get_default():
    assert _validate_configuration_space_domains(
        {"a": _NoDefaultLookup(type="weird")}
    ) == ["Unsupported parameter type 'weird' for 'a'"]


def test_validate_domains_categorical_lookup_order_values_then_default():
    definition = _RecordingLookup(type="categorical", values=[1, 2], default=1)
    assert _validate_configuration_space_domains({"a": definition}) == []
    assert [c for c in definition.calls if c in {"values", "default"}] == [
        "values",
        "default",
    ]
