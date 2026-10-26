"""Characterization tests pinning IntegrationPlugin.validate_config."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from traigent.integrations.base_plugin import (
    IntegrationPlugin,
    PluginMetadata,
    ValidationRule,
)
from traigent.utils.exceptions import TraigentError


class _Plugin(IntegrationPlugin):
    def __init__(self, rules):
        self._rules = rules
        super().__init__()

    def _get_metadata(self):
        return PluginMetadata(name="demo", version="1", supported_packages=[])

    def _get_default_mappings(self):
        return {}

    def _get_validation_rules(self):
        return self._rules

    def get_target_classes(self):
        return []

    def get_target_methods(self):
        return {}

    def check_even(self, name, value):
        return [] if value % 2 == 0 else [f"{name} must be even"]

    def check_many(self, name, value):
        return ["first", "second"]

    def check_none(self, name, value):
        return None


def _errors(rules, config):
    with pytest.raises(TraigentError) as ei:
        _Plugin(rules).validate_config(config)
    prefix = "Validation failed for demo: "
    msg = str(ei.value)
    assert msg.startswith(prefix)
    return msg[len(prefix):].split("; ")


def _ok(rules, config):
    assert _Plugin(rules).validate_config(config) is True


def test_valid_config_returns_true_and_no_rules_ok():
    _ok({}, SimpleNamespace())
    _ok({"t": ValidationRule(min_value=0, max_value=1)}, SimpleNamespace(t=0.5))


def test_required_missing_from_attr_and_custom_params():
    errs = _errors({"t": ValidationRule(required=True)}, SimpleNamespace(custom_params={}))
    assert errs == ["Required parameter 't' is missing"]


def test_value_taken_from_custom_params_when_attr_none_or_absent():
    rules = {"t": ValidationRule(required=True, min_value=1)}
    _ok(rules, SimpleNamespace(t=None, custom_params={"t": 2}))
    _ok(rules, SimpleNamespace(custom_params={"t": 2}))
    assert _errors(rules, SimpleNamespace(t=None, custom_params={"t": 0})) == [
        "Parameter 't' value 0 is below minimum 1"
    ]


def test_attr_wins_over_custom_params():
    rules = {"t": ValidationRule(min_value=1)}
    _ok(rules, SimpleNamespace(t=5, custom_params={"t": 0}))


def test_optional_missing_skipped_even_with_other_constraints():
    _ok({"t": ValidationRule(min_value=1, allowed_values=[1])}, SimpleNamespace())


def test_min_and_max_messages_both_reported_in_order():
    rules = {
        "a": ValidationRule(min_value=5),
        "b": ValidationRule(max_value=5),
    }
    assert _errors(rules, SimpleNamespace(a=1, b=9)) == [
        "Parameter 'a' value 1 is below minimum 5",
        "Parameter 'b' value 9 is above maximum 5",
    ]


def test_incompatible_type_reported_once_per_bound_check():
    rules = {"t": ValidationRule(min_value=1, max_value=2)}
    msg = "Parameter 't' value x has incompatible type (expected numeric)"
    assert _errors(rules, SimpleNamespace(t="x")) == [msg, msg]
    assert _errors({"t": ValidationRule(min_value=1)}, SimpleNamespace(t="x")) == [msg]


def test_allowed_values():
    rules = {"t": ValidationRule(allowed_values=["a", "b"])}
    _ok(rules, SimpleNamespace(t="a"))
    assert _errors(rules, SimpleNamespace(t="c")) == [
        "Parameter 't' value c not in allowed values: ['a', 'b']"
    ]


def test_custom_validator_errors_extended_and_none_ignored():
    rules = {"t": ValidationRule(custom_validator="check_even")}
    _ok(rules, SimpleNamespace(t=2))
    assert _errors(rules, SimpleNamespace(t=3)) == ["t must be even"]
    assert _errors({"t": ValidationRule(custom_validator="check_many")}, SimpleNamespace(t=1)) == [
        "first",
        "second",
    ]
    _ok({"t": ValidationRule(custom_validator="check_none")}, SimpleNamespace(t=1))


def test_missing_custom_validator_method_is_ignored():
    _ok({"t": ValidationRule(custom_validator="nope")}, SimpleNamespace(t=1))


def test_errors_accumulate_across_params_and_checks():
    rules = {
        "a": ValidationRule(required=True),
        "b": ValidationRule(min_value=5, allowed_values=[9], custom_validator="check_even"),
    }
    assert _errors(rules, SimpleNamespace(b=1)) == [
        "Required parameter 'a' is missing",
        "Parameter 'b' value 1 is below minimum 5",
        "Parameter 'b' value 1 not in allowed values: [9]",
        "b must be even",
    ]


def test_odd_falsy_zero_value_is_validated_not_missing():
    rules = {"t": ValidationRule(required=True, min_value=1)}
    assert _errors(rules, SimpleNamespace(t=0)) == [
        "Parameter 't' value 0 is below minimum 1"
    ]


def test_odd_custom_params_none_and_non_mapping_attr():
    rules = {"t": ValidationRule(required=True)}
    assert _errors(rules, SimpleNamespace(custom_params=None)) == [
        "Required parameter 't' is missing"
    ]
    with pytest.raises(AttributeError):
        _Plugin(rules).validate_config(SimpleNamespace(custom_params=["t"]))
