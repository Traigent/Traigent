"""Characterization tests pinning ConfigurationMapper._apply_parameter_mappings."""

from __future__ import annotations

import logging

import pytest

from traigent.agents.config_mapper import (
    ConfigurationMapper,
    ParameterMapping,
    PlatformMapping,
)
from traigent.cloud.models import AgentSpecification
from traigent.utils.exceptions import ConfigurationError


def _apply(mappings, config, spec=None):
    spec = spec or AgentSpecification(id="a", name="n")
    mapper = ConfigurationMapper.__new__(ConfigurationMapper)
    out = mapper._apply_parameter_mappings(
        spec, config, PlatformMapping(platform="p", parameter_mappings=mappings)
    )
    assert out is spec
    return spec


def test_model_parameters_created_and_set():
    spec = _apply([ParameterMapping("t", "temperature")], {"t": 0.3})
    assert spec.model_parameters == {"temperature": 0.3}


def test_model_parameters_extend_existing():
    base = AgentSpecification(id="a", name="n", model_parameters={"x": 1})
    spec = _apply([ParameterMapping("t", "temperature")], {"t": 0.3}, base)
    assert spec.model_parameters == {"x": 1, "temperature": 0.3}


def test_default_used_when_missing_and_none_skipped():
    spec = _apply(
        [
            ParameterMapping("a", "ka", default_value=5),
            ParameterMapping("b", "kb"),
            ParameterMapping("c", "kc", default_value=2),
        ],
        {"c": None},
    )
    assert spec.model_parameters == {"ka": 5}


def test_falsy_values_are_applied():
    spec = _apply([ParameterMapping("a", "ka")], {"a": 0})
    assert spec.model_parameters == {"ka": 0}


def test_transform_applied_and_validation_receives_transformed():
    seen = []
    spec = _apply(
        [
            ParameterMapping(
                "a", "ka", transform=lambda v: v * 2, validation=seen.append
            )
        ],
        {"a": 4},
    )
    assert spec.model_parameters == {"ka": 8}
    assert seen == [8]


def test_transform_configuration_error_propagates_unwrapped():
    err = ConfigurationError("direct")

    def bad(_v):
        raise err

    with pytest.raises(ConfigurationError) as ei:
        _apply([ParameterMapping("a", "ka", transform=bad)], {"a": 1})
    assert ei.value is err


def test_transform_value_error_wrapped_with_context():
    def bad(_v):
        raise ValueError("neg")

    with pytest.raises(ConfigurationError) as ei:
        _apply([ParameterMapping("a", "ka", transform=bad)], {"a": 1})
    assert "Transform 'bad' for parameter 'a' produced an invalid value: neg" in str(
        ei.value
    )
    assert isinstance(ei.value.__cause__, ValueError)
    assert ei.value.details == {
        "source_key": "a",
        "target_key": "ka",
        "transform": "bad",
    }


def test_transform_other_error_wrapped_unexpected():
    class Callable_:
        def __call__(self, _v):
            raise KeyError("k")

    with pytest.raises(ConfigurationError) as ei:
        _apply([ParameterMapping("a", "ka", transform=Callable_())], {"a": 1})
    assert "Transform 'Callable_' raised an unexpected error for 'a'" in str(ei.value)
    assert isinstance(ei.value.__cause__, KeyError)


def test_validation_value_error_wrapped_and_other_errors_propagate():
    def bad(_v):
        raise ValueError("nope")

    with pytest.raises(ConfigurationError, match="Validation failed for ka: nope"):
        _apply([ParameterMapping("a", "ka", validation=bad)], {"a": 1})

    def worse(_v):
        raise RuntimeError("raw")

    with pytest.raises(RuntimeError, match="raw"):
        _apply([ParameterMapping("a", "ka", validation=worse)], {"a": 1})


def test_custom_tools_list_and_scalar_stringified():
    spec = _apply(
        [
            ParameterMapping("a", "tools", target_section="custom_tools"),
            ParameterMapping("b", "tools", target_section="custom_tools"),
        ],
        {"a": ["x", 2], "b": 7},
    )
    assert spec.custom_tools == ["x", "2", "7"]


def test_custom_tools_extend_existing():
    base = AgentSpecification(id="a", name="n", custom_tools=["t0"])
    spec = _apply(
        [ParameterMapping("a", "tools", target_section="custom_tools")],
        {"a": "t1"},
        base,
    )
    assert spec.custom_tools == ["t0", "t1"]


def test_direct_attribute_allowlisted_set():
    spec = _apply(
        [ParameterMapping("a", "persona", target_section="direct")], {"a": "pirate"}
    )
    assert spec.persona == "pirate"


def test_direct_attribute_unknown_refused_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        spec = _apply(
            [
                ParameterMapping("a", "_secret", target_section="direct"),
                ParameterMapping("b", "tone", target_section="direct"),
            ],
            {"a": 1, "b": "warm"},
        )
    assert not hasattr(spec, "_secret")
    assert spec.tone == "warm"
    assert any("Refusing to set unknown agent specification" in r.message for r in caplog.records)


def test_empty_mappings_and_odd_non_dict_config_get():
    spec = _apply([], {})
    assert spec.model_parameters == {}
    with pytest.raises(AttributeError):
        _apply([ParameterMapping("a", "ka")], None)


def test_odd_transform_returning_none_still_sets_none():
    spec = _apply([ParameterMapping("a", "ka", transform=lambda v: None)], {"a": 1})
    assert spec.model_parameters == {"ka": None}
