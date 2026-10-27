"""Characterization of ConfigSpace._mapping_domain_to_spec_dict."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from traigent.api.config_space import ConfigSpace


class RecordingDomain(Mapping):
    """Mapping that records ``get`` and ``in`` probes in order."""

    def __init__(self, data):
        self._data = dict(data)
        self.log = []

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        self.log.append(("in", key))
        return key in self._data

    def get(self, key, default=None):
        self.log.append(("get", key))
        return self._data.get(key, default)


def _go(domain, entry=None, cls=ConfigSpace):
    func = ConfigSpace._mapping_domain_to_spec_dict.__func__
    return func(cls, "t", domain, {} if entry is None else entry)


def test_kind_none_without_known_forms_raises_with_none_kind():
    with pytest.raises(ValueError, match="unsupported domain kind 'None'"):
        _go({})
    with pytest.raises(ValueError, match="unsupported domain kind 'None'"):
        _go({"foo": 1})


def test_range_form_takes_priority_over_set_and_values():
    result = _go({"range": [0, 1], "set": [1], "values": [2]}, {"default": 0.5})
    assert result == {
        "low": 0,
        "high": 1,
        "step": None,
        "log": False,
        "default": 0.5,
    }


def test_set_takes_priority_over_values():
    assert _go({"set": [1], "values": [2]}, {"default": 1}) == {
        "values": [1],
        "default": 1,
    }


def test_values_form_alone():
    assert _go({"values": ["a", "b"]}) == {"values": ["a", "b"], "default": None}


def test_values_list_is_copied():
    members = ["a"]
    result = _go({"set": members})
    assert result["values"] == members
    assert result["values"] is not members


def test_empty_set_list_is_accepted_and_hides_values():
    assert _go({"set": [], "values": [1]}) == {"values": [], "default": None}


@pytest.mark.parametrize("bad", [None, "ab", (1,), {"a": 1}])
def test_invalid_set_raises_before_probing_values(bad):
    domain = RecordingDomain({"set": bad, "values": [1]})
    with pytest.raises(ValueError, match="set domain must contain a list"):
        _go(domain)
    assert ("in", "values") not in domain.log
    assert ("get", "values") not in domain.log


@pytest.mark.parametrize("bad", [None, "ab", (1,), {"a": 1}])
def test_invalid_values_raises(bad):
    with pytest.raises(ValueError, match="values domain must contain a list"):
        _go({"values": bad})


def test_probe_order_for_set_form():
    domain = RecordingDomain({"set": [1]})
    _go(domain)
    assert domain.log == [
        ("get", "kind"),
        ("in", "range"),
        ("in", "set"),
        ("get", "set"),
    ]


def test_probe_order_for_values_form():
    domain = RecordingDomain({"values": [1]})
    _go(domain)
    assert domain.log == [
        ("get", "kind"),
        ("in", "range"),
        ("in", "set"),
        ("in", "values"),
        ("get", "values"),
    ]


def test_probe_order_for_no_form():
    domain = RecordingDomain({})
    with pytest.raises(ValueError):
        _go(domain)
    assert domain.log == [
        ("get", "kind"),
        ("in", "range"),
        ("in", "set"),
        ("in", "values"),
    ]


def test_probe_order_for_range_form_skips_canonical_lookup():
    domain = RecordingDomain({"range": [0, 1], "set": [1]})
    _go(domain)
    assert ("in", "set") not in domain.log
    assert domain.log[:2] == [("get", "kind"), ("in", "range")]


@pytest.mark.parametrize("kind", ["enum", "range"])
def test_explicit_kind_skips_canonical_lookup(kind):
    domain = RecordingDomain({"kind": kind, "set": [1], "range": [0, 1]})
    calls = []

    class Cls:
        @staticmethod
        def _canonical_values_spec(*args):
            calls.append(args)
            return {"values": []}

    _go(domain, cls=Cls)
    assert calls == []


def test_explicit_enum_and_range_shapes_unchanged():
    assert _go({"kind": "enum", "values": ("a",)}, {"default": "a"}) == {
        "values": ["a"],
        "default": "a",
    }
    assert _go({"kind": "enum"}) == {"values": [], "default": None}
    assert _go(
        {"kind": "range", "range": [1, 5], "resolution": 2, "log": 1},
        {"default": 3},
    ) == {"low": 1, "high": 5, "step": 2, "log": True, "default": 3}


@pytest.mark.parametrize("rng", [None, [1], [1, 2, 3], (1, 2), "ab"])
def test_bad_range_raises(rng):
    with pytest.raises(ValueError, match="range domain must contain a 2-item list"):
        _go({"range": rng})


def test_unsupported_kind_names_kind():
    with pytest.raises(ValueError, match="unsupported domain kind 'weird'"):
        _go({"kind": "weird", "set": [1]})


def test_override_helper_call_order_and_arguments():
    calls = []

    class Cls:
        @staticmethod
        def _canonical_values_spec(name, domain, entry, key, label):
            calls.append((name, domain, entry, key, label))
            return None

    domain = {"foo": 1}
    entry = {"default": 1}
    with pytest.raises(ValueError, match="kind 'None'"):
        ConfigSpace._mapping_domain_to_spec_dict.__func__(Cls, "t", domain, entry)
    assert calls == [
        ("t", domain, entry, "set", "set"),
        ("t", domain, entry, "values", "values"),
    ]
    assert calls[0][1] is domain
    assert calls[0][2] is entry


def test_first_non_none_result_returned_and_stops_loop():
    calls = []
    sentinel = {"values": ["x"]}

    class Cls:
        @staticmethod
        def _canonical_values_spec(name, domain, entry, key, label):
            calls.append(key)
            return sentinel

    assert _go({"foo": 1}, cls=Cls) is sentinel
    assert calls == ["set"]


def test_second_key_result_used_when_first_is_none():
    calls = []

    class Cls:
        @staticmethod
        def _canonical_values_spec(name, domain, entry, key, label):
            calls.append(key)
            return {"values": [key]} if key == "values" else None

    assert _go({"foo": 1}, cls=Cls) == {"values": ["values"]}
    assert calls == ["set", "values"]


@pytest.mark.parametrize("falsey", [{}, [], 0, "", False])
def test_falsey_non_none_result_is_returned(falsey):
    calls = []

    class Cls:
        @staticmethod
        def _canonical_values_spec(name, domain, entry, key, label):
            calls.append(key)
            return falsey

    result = _go({"foo": 1}, cls=Cls)
    assert result is falsey
    assert calls == ["set"]


def test_duck_typed_cls_with_only_canonical_helper_works():
    class Cls:
        @staticmethod
        def _canonical_values_spec(name, domain, entry, key, label):
            return {"values": [key], "default": None} if key == "set" else None

    assert _go({}, cls=Cls) == {"values": ["set"], "default": None}


def test_helper_lookup_count_matches_loop_iterations():
    lookups = []

    class Meta(type):
        def __getattribute__(cls, attr):
            if attr == "_canonical_values_spec":
                lookups.append(attr)
            return super().__getattribute__(attr)

    class Cls(metaclass=Meta):
        @staticmethod
        def _canonical_values_spec(name, domain, entry, key, label):
            return None

    with pytest.raises(ValueError):
        _go({"foo": 1}, cls=Cls)
    assert lookups == ["_canonical_values_spec"] * 2


def test_helper_exception_halts_loop():
    calls = []

    class Cls:
        @staticmethod
        def _canonical_values_spec(name, domain, entry, key, label):
            calls.append(key)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _go({"foo": 1}, cls=Cls)
    assert calls == ["set"]


def test_helper_not_called_when_range_present():
    calls = []

    class Cls:
        @staticmethod
        def _canonical_values_spec(*args):
            calls.append(args)

    result = _go({"range": [0, 1]}, cls=Cls)
    assert result["low"] == 0
    assert calls == []
