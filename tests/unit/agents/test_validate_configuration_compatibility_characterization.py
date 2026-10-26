"""Characterization tests pinning ConfigurationMapper.validate_configuration_compatibility."""

from __future__ import annotations

from traigent.agents.config_mapper import (
    ConfigurationMapper,
    ParameterMapping,
    PlatformMapping,
)
from traigent.cloud.models import AgentSpecification


def _mapper(*mappings, platform="Plat"):
    mapper = ConfigurationMapper.__new__(ConfigurationMapper)
    mapper._platform_mappings = {
        platform.lower(): PlatformMapping(platform=platform, parameter_mappings=list(mappings))
    }
    return mapper


def _spec(platform):
    spec = AgentSpecification(id="a", name="n")
    spec.agent_platform = platform
    return spec


def _positive(value):
    if value <= 0:
        raise ValueError(f"bad {value}")


def test_unknown_platform_incompatible():
    mapper = _mapper()
    assert mapper.validate_configuration_compatibility(_spec("other"), {"x": [1]}) == {
        "compatible": False,
        "errors": ["No mapping available for platform: other"],
        "warnings": [],
        "supported_parameters": [],
    }


def test_none_or_empty_platform_uses_empty_name():
    mapper = _mapper()
    for platform in (None, ""):
        out = mapper.validate_configuration_compatibility(_spec(platform), {})
        assert out["errors"] == ["No mapping available for platform: "]


def test_platform_lookup_is_case_insensitive():
    mapper = _mapper(ParameterMapping("a", "ka"))
    out = mapper.validate_configuration_compatibility(_spec("PLAT"), {"a": [1]})
    assert out["compatible"] is True
    assert out["supported_parameters"] == ["a"]


def test_unmapped_parameter_warns_with_platform_name():
    mapper = _mapper(ParameterMapping("a", "ka"))
    out = mapper.validate_configuration_compatibility(
        _spec("Plat"), {"a": [1], "zzz": [2]}
    )
    assert out["supported_parameters"] == ["a"]
    assert out["warnings"] == ["Parameter 'zzz' has no mapping for platform 'plat'"]
    assert out["errors"] == [] and out["compatible"] is True


def test_list_space_validates_each_value_and_stops_at_first_failure():
    calls = []

    def validation(v):
        calls.append(v)
        _positive(v)

    mapper = _mapper(ParameterMapping("a", "ka", validation=validation))
    out = mapper.validate_configuration_compatibility(_spec("plat"), {"a": [1, -2, -3]})
    assert out["warnings"] == ["Parameter a: bad -2"]
    assert calls == [1, -2]
    assert out["supported_parameters"] == ["a"]


def test_tuple_and_scalar_spaces():
    mapper = _mapper(ParameterMapping("a", "ka", validation=_positive))
    tup = mapper.validate_configuration_compatibility(_spec("plat"), {"a": (1, -1)})
    assert tup["warnings"] == ["Parameter a: bad -1"]
    scalar = mapper.validate_configuration_compatibility(_spec("plat"), {"a": -5})
    assert scalar["warnings"] == ["Parameter a: bad -5"]
    ok = mapper.validate_configuration_compatibility(_spec("plat"), {"a": 5})
    assert ok["warnings"] == []


def test_type_error_from_validation_is_a_warning_other_errors_propagate():
    mapper = _mapper(ParameterMapping("a", "ka", validation=_positive))
    out = mapper.validate_configuration_compatibility(_spec("plat"), {"a": ["x"]})
    assert out["warnings"][0].startswith("Parameter a: ")

    def boom(_v):
        raise RuntimeError("raw")

    import pytest

    mapper2 = _mapper(ParameterMapping("a", "ka", validation=boom))
    with pytest.raises(RuntimeError):
        mapper2.validate_configuration_compatibility(_spec("plat"), {"a": [1]})


def test_first_matching_mapping_wins_and_parameters_keep_config_order():
    first = ParameterMapping("a", "k1", validation=_positive)
    second = ParameterMapping("a", "k2")
    mapper = _mapper(first, second)
    out = mapper.validate_configuration_compatibility(
        _spec("plat"), {"b": 1, "a": [-1]}
    )
    assert out["warnings"] == [
        "Parameter 'b' has no mapping for platform 'plat'",
        "Parameter a: bad -1",
    ]
    assert out["supported_parameters"] == ["a"]


def test_empty_config_space_is_compatible():
    out = _mapper().validate_configuration_compatibility(_spec("plat"), {})
    assert out == {
        "compatible": True,
        "errors": [],
        "warnings": [],
        "supported_parameters": [],
    }


def test_odd_none_config_space_raises():
    import pytest

    with pytest.raises(AttributeError):
        _mapper().validate_configuration_compatibility(_spec("plat"), None)
