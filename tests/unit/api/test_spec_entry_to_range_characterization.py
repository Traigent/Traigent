"""Characterization of ConfigSpace._spec_entry_to_range (spec_dict contract)."""

from __future__ import annotations

import pytest

from traigent.api.config_space import ConfigSpace


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake(cls, name, spec_dict, range_type):
        calls.append((name, spec_dict, range_type))
        return "RANGE"

    monkeypatch.setattr(ConfigSpace, "_build_parameter_range", classmethod(fake))
    return calls


def _go(entry):
    return ConfigSpace._spec_entry_to_range("t", entry)


def test_list_domain(captured):
    assert _go({"domain": [1, 2], "default": 1}) == "RANGE"
    assert captured == [("t", {"values": [1, 2], "default": 1}, None)]


@pytest.mark.parametrize("domain", [None, "x", 5, ("a",)])
def test_non_list_non_mapping_domain(captured, domain):
    with pytest.raises(ValueError, match="domain must be a list or mapping"):
        _go({"domain": domain})
    with pytest.raises(ValueError, match="domain must be a list or mapping"):
        _go({})


def test_canonical_set_form(captured):
    _go({"domain": {"set": ["a", "b"]}, "default": "a"})
    assert captured[0][1] == {"values": ["a", "b"], "default": "a"}


def test_canonical_values_form(captured):
    _go({"domain": {"values": ["a"]}})
    assert captured[0][1] == {"values": ["a"], "default": None}


def test_set_takes_precedence_over_values_and_range_over_set(captured):
    _go({"domain": {"set": [1], "values": [2]}})
    assert captured[-1][1]["values"] == [1]
    _go({"domain": {"range": [0, 1], "set": [1]}})
    assert captured[-1][1]["low"] == 0


@pytest.mark.parametrize("key,msg", [("set", "set domain"), ("values", "values domain")])
@pytest.mark.parametrize("bad", [None, "ab", (1,), {"a": 1}])
def test_non_list_set_or_values(captured, key, msg, bad):
    with pytest.raises(ValueError, match=f"{msg} must contain a list"):
        _go({"domain": {key: bad}})


def test_explicit_enum_kind(captured):
    _go({"domain": {"kind": "enum", "values": ("a", "b")}, "default": "b"})
    assert captured[0][1] == {"values": ["a", "b"], "default": "b"}


def test_explicit_enum_kind_without_values_is_empty(captured):
    _go({"domain": {"kind": "enum"}})
    assert captured[0][1] == {"values": [], "default": None}


def test_range_form_with_resolution_log_and_range_type(captured):
    _go(
        {
            "domain": {"range": [1, 5], "resolution": 2, "log": 1},
            "default": 3,
            "x_traigent_parameter_range": "IntRange",
        }
    )
    assert captured[0] == (
        "t",
        {"low": 1, "high": 5, "step": 2, "log": True, "default": 3},
        "IntRange",
    )


def test_explicit_range_kind(captured):
    _go({"domain": {"kind": "range", "range": [0.1, 1.0]}})
    assert captured[0][1] == {
        "low": 0.1, "high": 1.0, "step": None, "log": False, "default": None,
    }


@pytest.mark.parametrize("rng", [None, [1], [1, 2, 3], (1, 2), "ab"])
def test_bad_range(captured, rng):
    with pytest.raises(ValueError, match="range domain must contain a 2-item list"):
        _go({"domain": {"range": rng}})
    with pytest.raises(ValueError, match="range domain must contain a 2-item list"):
        _go({"domain": {"kind": "range", "range": rng}})


@pytest.mark.parametrize("domain", [{}, {"kind": "weird"}, {"foo": 1}, {"kind": 3}])
def test_unsupported_kind(captured, domain):
    with pytest.raises(ValueError, match="unsupported domain kind"):
        _go({"domain": domain})


def test_unsupported_kind_message_names_kind(captured):
    with pytest.raises(ValueError, match="kind 'None'"):
        _go({"domain": {"foo": 1}})
    with pytest.raises(ValueError, match="kind 'weird'"):
        _go({"domain": {"kind": "weird", "range": [0, 1]}})


def test_unit_and_agent_only_when_strings(captured):
    _go({"domain": [1], "unit": "ms", "agent": "a"})
    assert captured[-1][1] == {"values": [1], "default": None, "unit": "ms", "agent": "a"}
    _go({"domain": [1], "unit": 3, "agent": None})
    assert captured[-1][1] == {"values": [1], "default": None}


def test_explicit_enum_resolved_kind_is_unsupported(captured):
    # A literal "enum_resolved" kind used to raise UnboundLocalError.
    with pytest.raises(ValueError, match="unsupported domain kind .enum_resolved."):
        _go({"domain": {"kind": "enum_resolved", "values": [1]}})
