"""Report model: serialization, versioning, immutability, keys-only invariant."""

from __future__ import annotations

import dataclasses
import json

import pytest

import traigent
from traigent.contract import (
    EVALUATION_CONTRACT_VERSION,
    ContractFinding,
    EvaluationContractReport,
    EvaluatorBinding,
    ExampleCallShape,
    InjectionSummary,
    NormalizedExampleSummary,
    validate_evaluation_contract,
)

DATASET = [{"input": {"question": "q1"}, "output": "gold"}]


def _agent(question):
    return "x"


def _scorer(output, expected):
    return 1.0


def _report():
    return validate_evaluation_contract(
        func=_agent,
        dataset=DATASET,
        scoring_function=_scorer,
        config={"model": "gpt-4"},
    )


def test_to_dict_is_json_serializable():
    report = _report()
    payload = report.to_dict()
    # Round-trips through JSON without custom encoders.
    restored = json.loads(json.dumps(payload))
    assert restored == payload


def test_to_dict_top_level_keys():
    report = _report()
    payload = report.to_dict()
    assert set(payload) == {
        "ok",
        "sdk_version",
        "contract_version",
        "dataset_name",
        "dataset_source",
        "examples",
        "injection",
        "call_shapes",
        "evaluator_bindings",
        "unsupported",
        "findings",
    }


def test_contract_version_matches_constant():
    report = _report()
    assert report.contract_version == EVALUATION_CONTRACT_VERSION
    assert report.to_dict()["contract_version"] == EVALUATION_CONTRACT_VERSION


def test_sdk_version_matches_package():
    report = _report()
    assert report.sdk_version == traigent.__version__


@pytest.mark.parametrize(
    "cls",
    [
        ContractFinding,
        NormalizedExampleSummary,
        InjectionSummary,
        ExampleCallShape,
        EvaluatorBinding,
        EvaluationContractReport,
    ],
)
def test_report_dataclasses_are_frozen(cls):
    assert dataclasses.is_dataclass(cls)
    params = cls.__dataclass_params__
    assert params.frozen is True


def test_report_instance_is_immutable():
    report = _report()
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.ok = False  # type: ignore[misc]


def test_nested_to_dict_shapes():
    report = _report()
    payload = report.to_dict()
    assert isinstance(payload["examples"], list)
    assert isinstance(payload["examples"][0], dict)
    assert isinstance(payload["injection"], dict)
    assert isinstance(payload["call_shapes"][0], dict)
    assert isinstance(payload["evaluator_bindings"][0], dict)
    # tuples serialize to lists
    assert isinstance(payload["injection"]["config_keys"], list)
    assert isinstance(payload["findings"], list)


def test_finding_code_serializes_as_plain_string():
    # ContractCode is a StrEnum; the serialized value must be a bare string.
    report = validate_evaluation_contract(func=_agent, dataset=[])
    payload = report.to_dict()
    assert payload["findings"]
    for finding in payload["findings"]:
        assert isinstance(finding["code"], str)
        assert type(finding["code"]) is str


def test_keys_only_invariant_holds_for_full_report():
    report = _report()
    payload = report.to_dict()
    # The report never carries example input values, only shapes/keys/names.
    injection = payload["injection"]
    assert injection["config_keys"] == ["model"]
    # No field named to suggest values are carried.
    assert "config_values" not in injection
    assert "config" not in injection
