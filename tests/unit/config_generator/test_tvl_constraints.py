"""Generated structural constraints must load with the SDK's own TVL loader (#2647)."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from traigent.config_generator.presets.constraint_templates import (
    get_matching_constraints,
)
from traigent.config_generator.tvl_constraints import translate_constraint_code
from traigent.config_generator.types import (
    AutoConfigResult,
    ObjectiveSpec,
    StructuralConstraintSpec,
    TVarSpec,
)
from traigent.tvl.spec_loader import load_tvl_spec

_TVARS = (
    TVarSpec(
        name="temperature",
        range_type="Range",
        range_kwargs={"low": 0.0, "high": 2.0},
    ),
    TVarSpec(
        name="top_p",
        range_type="Range",
        range_kwargs={"low": 0.1, "high": 1.0},
    ),
    TVarSpec(
        name="model",
        range_type="Choices",
        range_kwargs={"values": ["gpt-4o", "gpt-4o-mini"]},
    ),
    TVarSpec(
        name="max_tokens",
        range_type="IntRange",
        range_kwargs={"low": 64, "high": 4096},
    ),
    TVarSpec(
        name="chunk_size",
        range_type="IntRange",
        range_kwargs={"low": 100, "high": 1000},
    ),
    TVarSpec(
        name="chunk_overlap",
        range_type="IntRange",
        range_kwargs={"low": 0, "high": 500},
    ),
)


def _load(result: AutoConfigResult, tmp_path: Path):
    path = tmp_path / "gen.tvl.yml"
    path.write_text(yaml.safe_dump(result.to_tvl_spec(module_name="gen")))
    return load_tvl_spec(spec_path=path)


def _result(*constraints: StructuralConstraintSpec) -> AutoConfigResult:
    return AutoConfigResult(
        tvars=_TVARS,
        objectives=(
            ObjectiveSpec(name="accuracy", orientation="maximize", weight=1.0),
        ),
        structural_constraints=tuple(constraints),
    )


def _template(name_fragment: str) -> StructuralConstraintSpec:
    names = {t.name for t in _TVARS}
    for spec in get_matching_constraints(names):
        if name_fragment in spec.constraint_code:
            return spec
    raise AssertionError(name_fragment)


def _accepts(artifact, config: dict) -> bool:
    return all(c(config, None) for c in artifact.constraints)


_BASE = {
    "temperature": 0.7,
    "top_p": 0.9,
    "model": "gpt-4o-mini",
    "max_tokens": 512,
    "chunk_size": 500,
    "chunk_overlap": 50,
}


@pytest.mark.parametrize(
    ("fragment", "violation"),
    [
        ("top_p.lte(0.2)", {"temperature": 1.8, "top_p": 0.1}),
        ("model.is_in(['gpt-4o', 'gpt-4.1'])", {"model": "gpt-4o", "temperature": 1.5}),
        ("max_tokens.lte(2048)", {"model": "gpt-4o-mini", "max_tokens": 4000}),
        ("chunk_overlap", {"chunk_size": 200, "chunk_overlap": 300}),
    ],
)
def test_each_known_template_loads_and_rejects_violations(
    fragment: str, violation: dict, tmp_path: Path
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        artifact = _load(_result(_template(fragment)), tmp_path)
    assert len(artifact.constraints) == 1
    assert _accepts(artifact, _BASE)
    assert not _accepts(artifact, {**_BASE, **violation})


def test_implication_not_triggered_is_allowed(tmp_path: Path) -> None:
    artifact = _load(_result(_template("gpt-4.1-mini")), tmp_path)
    # guard model absent -> consequent not required
    assert _accepts(artifact, {**_BASE, "model": "gpt-4o", "max_tokens": 4000})


def test_full_generated_spec_round_trip(tmp_path: Path) -> None:
    all_templates = tuple(get_matching_constraints({t.name for t in _TVARS}))
    assert len(all_templates) == 4
    result = _result(*all_templates)
    spec = result.to_tvl_spec(module_name="gen")
    assert "header" not in spec and "safety" not in spec
    for entry in spec["constraints"]["structural"]:
        assert "code" not in entry
        assert "expr" in entry or ("when" in entry and "then" in entry)
    artifact = _load(result, tmp_path)
    assert len(artifact.constraints) == 4
    assert _accepts(artifact, _BASE)


def test_unknown_shape_warns_value_free_and_is_skipped(tmp_path: Path) -> None:
    secret = "sk-very-secret-value"
    unknown = StructuralConstraintSpec(
        description="custom",
        constraint_code=f"custom_check(temperature, '{secret}')",
    )
    with pytest.warns(UserWarning, match="structural_1") as record:
        artifact = _load(_result(_template("chunk_overlap"), unknown), tmp_path)
    assert secret not in str(record[0].message)
    assert "custom_check" not in str(record[0].message)
    # known one kept, unknown skipped, spec still loads
    assert len(artifact.constraints) == 1


def test_only_unknown_shapes_omit_constraints_section(tmp_path: Path) -> None:
    unknown = StructuralConstraintSpec(description="x", constraint_code="foo(bar)")
    with pytest.warns(UserWarning):
        spec = _result(unknown).to_tvl_spec()
    assert "structural" not in spec.get("constraints", {})
    with pytest.warns(UserWarning):
        _load(_result(unknown), tmp_path)


@pytest.mark.parametrize(
    "code",
    [
        "not valid python (",
        "lambda config, other: config['a'] < config['b']",
        "lambda config: config['a.b'] < config['c']",
        "lambda config: __import__('os')",
        "require(params.gte(1))",
        "implies(model.is_in('abc'), temperature.lte(1.0))",
        "require(temperature.gte(other))",
    ],
)
def test_unrecognised_code_returns_none(code: str) -> None:
    assert translate_constraint_code(code) is None


def test_translated_forms() -> None:
    assert translate_constraint_code(
        "lambda config: config['chunk_overlap'] < config['chunk_size']"
    ) == {"expr": "chunk_overlap < chunk_size"}
    assert translate_constraint_code(
        "require(~(temperature.gte(1.5) & top_p.lte(0.2)))"
    ) == {"expr": "not ((temperature >= 1.5) and (top_p <= 0.2))"}
