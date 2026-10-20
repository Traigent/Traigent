"""#1612: canonical TVL 1.0 ``components`` (tuple product) domains load."""

import pytest

from traigent.tvl.models import parse_domain_spec
from traigent.tvl.spec_loader import TVLValidationError, load_tvl_spec

_SPEC = """
tvl:
  module: corp.validation.tuple_product_semantics
environment:
  snapshot_id: "2025-01-01T00:00:00Z"
evaluation_set:
  dataset: s3://datasets/validation/dev.parquet
tvars:
  - name: model_config
    type: tuple[enum[str], float]
    default: ["gpt-4o-mini", 0.5]
    domain:
      components:
        - ["gpt-4o-mini", "gpt-4o"]
        - { range: [0.0, 1.0], resolution: 0.5 }
  - name: retrieval_config
    type: tuple[int, int]
    domain:
      components:
        - { range: [1, 3] }
        - { set: [100, 200] }
objectives:
  - name: quality
    direction: maximize
"""


def test_components_domain_loads_as_cartesian_product(tmp_path):
    path = tmp_path / "t.tvl.yml"
    path.write_text(_SPEC)
    spec = load_tvl_spec(spec_path=str(path))
    cs = spec.configuration_space
    assert cs["model_config"] == [
        ("gpt-4o-mini", 0.0),
        ("gpt-4o-mini", 0.5),
        ("gpt-4o-mini", 1.0),
        ("gpt-4o", 0.0),
        ("gpt-4o", 0.5),
        ("gpt-4o", 1.0),
    ]
    assert len(cs["retrieval_config"]) == 6
    assert spec.default_config["model_config"] == ("gpt-4o-mini", 0.5)


def test_float_range_component_requires_resolution():
    with pytest.raises(ValueError, match="resolution"):
        parse_domain_spec("t", "tuple", {"components": [{"range": [0.0, 1.0]}]})


def test_registry_component_rejected_explicitly():
    with pytest.raises(ValueError, match="registry"):
        parse_domain_spec("t", "tuple", {"components": [{"registry": "r"}]})


def test_product_size_guard():
    big = {"range": [0, 200]}
    with pytest.raises(ValueError, match="exceeds limit"):
        parse_domain_spec("t", "tuple", {"components": [big, big]})


def test_default_outside_product_rejected(tmp_path):
    path = tmp_path / "t.tvl.yml"
    path.write_text(_SPEC.replace('["gpt-4o-mini", 0.5]', '["nope", 0.5]'))
    with pytest.raises(TVLValidationError, match="outside declared tuple domain"):
        load_tvl_spec(spec_path=str(path))


def test_components_on_non_tuple_type_still_rejected():
    with pytest.raises(ValueError):
        parse_domain_spec("t", "int", {"components": [[1, 2]]})


def _spec_with(range_line: str, default: str) -> str:
    return f"""
tvl:
  module: corp.validation.tuple_bounds
environment:
  snapshot_id: "2025-01-01T00:00:00Z"
evaluation_set:
  dataset: s3://datasets/validation/dev.parquet
tvars:
  - name: cfg
    type: tuple[enum[str], float]
    default: {default}
    domain:
      components:
        - ["a"]
        - {range_line}
objectives:
  - name: quality
    direction: maximize
"""


def test_non_divisible_resolution_stays_within_bounds(tmp_path):
    from traigent.tvl.models import _enumerate_tuple_components

    vals = _enumerate_tuple_components(
        "cfg", [["a"], {"range": [0.0, 1.0], "resolution": 0.6}]
    )
    assert [v[1] for v in vals] == [0.0, 0.6]


def test_divisible_resolution_includes_upper_bound_despite_float_error():
    from traigent.tvl.models import _enumerate_tuple_components

    vals = _enumerate_tuple_components(
        "cfg", [["a"], {"range": [0.0, 1.0], "resolution": 0.1}]
    )
    assert len(vals) == 11 and vals[-1][1] == 1.0


@pytest.mark.parametrize(
    "comp",
    [
        {"range": [0.0, float("inf")], "resolution": 0.5},
        {"range": [0.0, 1.0], "resolution": float("nan")},
        {"range": [0.0, 1.0], "resolution": float("inf")},
        {"range": [0.0, 1.0], "resolution": 0},
        {"range": [0.0, 1.0], "resolution": -0.5},
    ],
)
def test_non_finite_or_non_positive_rejected(comp):
    from traigent.tvl.models import _enumerate_tuple_components

    with pytest.raises(ValueError):
        _enumerate_tuple_components("cfg", [["a"], comp])


def test_out_of_range_default_rejected(tmp_path):
    path = tmp_path / "t.tvl.yml"
    path.write_text(_spec_with("{ range: [0.0, 1.0], resolution: 0.6 }", '["a", 1.2]'))
    with pytest.raises(TVLValidationError):
        load_tvl_spec(spec_path=str(path))
