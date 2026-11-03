"""Characterization tests for spec_loader._validate_domain_structure."""

from __future__ import annotations

import pytest

from traigent.tvl import spec_loader as sl

validate = sl._validate_domain_structure


def test_none_domain_non_bool_is_missing():
    assert validate("x", None, raw_type="int") == [
        "TVAR 'x' missing required 'domain' field"
    ]
    assert validate("x", None) == ["TVAR 'x' missing required 'domain' field"]


def test_none_domain_bool_with_and_without_default():
    assert validate("x", None, raw_type="bool") == []
    assert validate("x", None, raw_type="bool", default=True) == []
    out = validate("x", None, raw_type="bool", default="yes")
    assert out == sl._default_type_issues("x", "bool", "yes")


@pytest.mark.parametrize("domain", [["a", "b"], ("a", "b")])
def test_sequence_domain_delegates_to_choice_membership(domain):
    assert validate("x", domain, default="c", raw_type="str") == (
        sl._validate_choice_membership("x", domain, "c", "str")
    )
    assert validate("x", domain, default="a", raw_type="str") == []


def test_non_container_domain():
    assert validate("x", 5) == [
        "TVAR 'x' domain should be dict, list, or tuple, got int"
    ]
    assert validate("x", "abc") == [
        "TVAR 'x' domain should be dict, list, or tuple, got str"
    ]


def test_empty_list_domain_is_sequence_not_missing():
    assert validate("x", [], default="a") == sl._validate_choice_membership(
        "x", [], "a", None
    )


def test_dict_range():
    domain = {"range": [0, 10]}
    assert validate("x", domain, default=5, raw_type="int") == (
        sl._validate_range_membership("x", domain, 5, "int")
    )
    assert validate("x", domain, default=50, raw_type="int") == (
        sl._validate_range_membership("x", domain, 50, "int")
    )


def test_dict_type_range_without_range_key():
    domain = {"type": "range"}
    assert validate("x", domain) == sl._validate_range_membership(
        "x", domain, sl._MISSING, None
    )


def test_invalid_type_message_is_kept_before_other_issues():
    domain = {"type": "bogus", "range": [0, 1]}
    out = validate("x", domain)
    assert out[0].startswith("TVAR 'x' has invalid domain type 'bogus'. Valid: ")
    assert out[1:] == sl._validate_range_membership("x", domain, sl._MISSING, None)


def test_invalid_type_alone_also_reports_shape_issue():
    out = validate("x", {"type": "bogus"})
    assert len(out) == 2
    assert out[0].startswith("TVAR 'x' has invalid domain type 'bogus'")
    assert out[1] == (
        "TVAR 'x' domain must define one of range, set, values, "
        "registry, or (tuple types) components"
    )


def test_dict_set_and_values_and_choices():
    assert validate("x", {"set": ["a"]}, default="b", raw_type="str") == (
        sl._validate_choice_membership("x", ["a"], "b", "str")
    )
    assert validate("x", {"values": ["a"]}, default="b", raw_type="str") == (
        sl._validate_choice_membership("x", ["a"], "b", "str")
    )
    assert validate("x", {"type": "choices"}, default="b", raw_type="str") == (
        sl._validate_choice_membership("x", None, "b", "str")
    )


def test_set_takes_precedence_over_values():
    out = validate("x", {"set": ["a"], "values": ["b"]}, default="b", raw_type="str")
    assert out == sl._validate_choice_membership("x", ["a"], "b", "str")


def test_range_takes_precedence_over_set():
    domain = {"range": [0, 1], "set": ["a"]}
    assert validate("x", domain, default=0, raw_type="int") == (
        sl._validate_range_membership("x", domain, 0, "int")
    )


def test_registry_domain_is_accepted():
    assert validate("x", {"registry": "models"}) == []
    assert validate("x", {"type": "registry"}) == []


def test_tuple_components_only_for_tuple_type():
    domain = {"components": [{"name": "a", "domain": [1, 2]}]}
    assert validate("x", domain, raw_type="tuple") == (
        sl._validate_tuple_components("x", domain, sl._MISSING)
    )
    out = validate("x", domain, raw_type="int")
    assert out == [
        "TVAR 'x' domain must define one of range, set, values, "
        "registry, or (tuple types) components"
    ]
    assert validate("x", domain) == out


def test_empty_dict_reports_shape_issue():
    assert validate("x", {}) == [
        "TVAR 'x' domain must define one of range, set, values, "
        "registry, or (tuple types) components"
    ]
