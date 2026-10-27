"""Characterization of TraigentService._normalize_evaluation_kwarg_definition."""

from __future__ import annotations

import pytest

from traigent.hybrid.protocol import EvaluationKwargDefinition as EKD
from traigent.wrapper.service import TraigentService


@pytest.fixture
def svc():
    return TraigentService()


def _n(svc, spec):
    return svc._normalize_evaluation_kwarg_definition(spec)


def test_check_order_name_before_description_before_domain_before_default(svc):
    bad = EKD(name="1x", type="str", description=1, domain="d", default=[1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="identifier"):
        _n(svc, bad)
    bad = EKD(name="x", type="str", description=1, domain="d", default=[1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="description"):
        _n(svc, bad)
    bad = EKD(name="x", type="str", domain="d", default=[1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="domain must be a dict"):
        _n(svc, bad)
    bad = EKD(name="x", type="str", domain={"zz": 1}, default=[1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default must be a scalar"):
        _n(svc, bad)


def test_unknown_keys_reported_sorted_and_before_values_check(svc):
    spec = {"name": "x", "type": "str", "domain": {"values": [], "b": 1, "a": 2}}
    with pytest.raises(ValueError, match=r"unsupported keys: \['a', 'b'\]"):
        _n(svc, spec)


def test_values_checked_before_range_before_resolution_before_enum(svc):
    d = {"values": [], "range": [1], "resolution": "x"}
    with pytest.raises(ValueError, match="non-empty list"):
        _n(svc, {"name": "x", "type": "enum", "domain": d})
    d = {"values": ["a"], "range": [1], "resolution": "x"}
    with pytest.raises(ValueError, match="two-item"):
        _n(svc, {"name": "x", "type": "enum", "domain": d})
    d = {"values": ["a"], "range": [1, 2], "resolution": "x"}
    with pytest.raises(ValueError, match="resolution must be numeric"):
        _n(svc, {"name": "x", "type": "enum", "domain": d})


@pytest.mark.parametrize(
    "rng",
    [[1, 2, 3], "ab", (1, 2), [1, "2"], [None, 1], 5],
)
def test_range_rejections(svc, rng):
    with pytest.raises(ValueError, match="two-item numeric list"):
        _n(svc, {"name": "x", "type": "float", "domain": {"range": rng}})


def test_float_range_accepted_and_range_not_ordered_checked(svc):
    out = _n(svc, {"name": "x", "type": "float", "domain": {"range": [2.5, 1]}})
    assert out.domain == {"range": [2.5, 1]}


def test_empty_domain_becomes_none_and_input_domain_not_aliased(svc):
    assert _n(svc, {"name": "x", "type": "str", "domain": {}}).domain is None
    src = {"values": ["a"]}
    out = _n(svc, EKD(name="x", type="str", domain=src))
    assert out.domain == src and out.domain is not src


def test_enum_with_empty_domain_dict_raises(svc):
    with pytest.raises(ValueError, match="must declare domain.values"):
        _n(svc, {"name": "x", "type": "enum", "domain": {}})


@pytest.mark.parametrize(
    "default, typ", [(0, "int"), (False, "bool"), ("", "str"), (0.0, "float")]
)
def test_falsy_defaults_are_still_validated_when_not_none(svc, default, typ):
    out = _n(svc, {"name": "x", "type": typ, "default": default})
    assert out.default == default and type(out.default) is type(default)


def test_falsy_default_wrong_type_rejected(svc):
    with pytest.raises(ValueError, match="must be an int"):
        _n(svc, {"name": "x", "type": "int", "default": False})


def test_default_none_skips_value_validation(svc):
    out = _n(svc, {"name": "x", "type": "enum", "domain": {"values": ["a"]}})
    assert out.default is None


def test_complex_default_is_not_scalar(svc):
    with pytest.raises(ValueError, match="default must be a scalar"):
        _n(svc, {"name": "x", "type": "str", "default": 1j})


def test_definition_instance_passthrough_builds_new_instance(svc):
    spec = EKD(name="x", type="int", description="d", default=3)
    out = _n(svc, spec)
    assert out is not spec and out == spec


def test_dict_missing_name_raises_keyerror(svc):
    with pytest.raises(KeyError):
        _n(svc, {"type": "str"})
