"""Dataset-normalization stage of the evaluation contract.

Proves the contract reuses the production ``Dataset`` loaders (input/input_data
aliases, the six expected-output aliases, metadata preservation, the
``from_jsonl`` path) and emits the right diagnostics for empty / missing-input
datasets -- all without executing the target function.
"""

from __future__ import annotations

import json

import pytest

from traigent.contract import ContractCode, validate_evaluation_contract

from ._support import error_codes, find_finding, finding_codes

# Six expected-output aliases, mirroring ``base._EXPECTED_OUTPUT_FIELDS``.
EXPECTED_OUTPUT_ALIASES = (
    "output",
    "expected",
    "expected_output",
    "answer",
    "target",
    "label",
)


def _agent(question):
    return "answer"


def test_input_alias_normalizes():
    report = validate_evaluation_contract(
        func=_agent, dataset=[{"input": {"question": "q1"}}]
    )
    assert report.ok
    assert report.examples[0].input_is_mapping is True
    assert report.examples[0].input_keys == ("question",)


def test_input_data_alias_normalizes():
    report = validate_evaluation_contract(
        func=_agent, dataset=[{"input_data": {"question": "q1"}}]
    )
    assert report.ok
    assert report.examples[0].input_keys == ("question",)


def test_input_takes_priority_over_input_data():
    # Production ``_coerce_dataset_example_mapping`` prefers "input"; the key it
    # did not consume stays as metadata.
    report = validate_evaluation_contract(
        func=_agent,
        dataset=[{"input": {"question": "primary"}, "input_data": {"question": "alt"}}],
    )
    assert report.examples[0].input_keys == ("question",)
    assert "input_data" in report.examples[0].metadata_keys


@pytest.mark.parametrize("alias", EXPECTED_OUTPUT_ALIASES)
def test_expected_output_aliases(alias):
    report = validate_evaluation_contract(
        func=_agent, dataset=[{"input": {"question": "q1"}, alias: "gold"}]
    )
    assert report.ok
    assert report.examples[0].has_expected_output is True
    # The expected-output key is consumed, not left in metadata.
    assert alias not in report.examples[0].metadata_keys


def test_metadata_preserved_as_keys():
    report = validate_evaluation_contract(
        func=_agent,
        dataset=[
            {
                "input": {"question": "q1"},
                "output": "gold",
                "difficulty": "hard",
                "source": "unit",
            }
        ],
    )
    assert report.examples[0].metadata_keys == ("difficulty", "source")


def test_scalar_input_summary():
    report = validate_evaluation_contract(
        func=_agent, dataset=[{"input": "a plain string"}]
    )
    assert report.examples[0].input_is_mapping is False
    assert report.examples[0].input_type == "str"
    assert report.examples[0].input_keys is None


def test_empty_dataset_is_error():
    report = validate_evaluation_contract(func=_agent, dataset=[])
    assert report.ok is False
    assert error_codes(report) == [ContractCode.DATASET_EMPTY]


def test_missing_input_key_fails_normalization():
    report = validate_evaluation_contract(func=_agent, dataset=[{"not_input": "x"}])
    assert report.ok is False
    finding = find_finding(report, ContractCode.DATASET_NORMALIZATION_FAILED)
    assert finding is not None
    assert finding.severity == "error"
    # Value-free: the message no longer embeds the raw ValidationError text; it
    # names the exception TYPE + source kind and points at the source, while the
    # actionable "input" hint lives in the finding's action.
    assert "Dataset normalization failed" in finding.message
    assert "inspect the dataset source" in finding.message
    assert finding.action is not None and "input" in finding.action.lower()


def test_empty_input_payload_warns():
    report = validate_evaluation_contract(func=_agent, dataset=[{"input": {}}])
    # A warning, not an error -- the report stays ok.
    assert report.ok is True
    finding = find_finding(report, ContractCode.DATASET_MISSING_INPUT)
    assert finding is not None
    assert finding.severity == "warning"
    assert finding.example_index == 0


def test_correlation_key_uses_example_id_metadata():
    report = validate_evaluation_contract(
        func=_agent,
        dataset=[{"input": {"q": "1"}, "example_id": "custom-key"}],
    )
    assert report.examples[0].correlation_key == "custom-key"


def test_max_examples_caps_per_example_work():
    dataset = [{"input": {"question": f"q{i}"}} for i in range(5)]
    report = validate_evaluation_contract(func=_agent, dataset=dataset, max_examples=2)
    assert len(report.examples) == 2
    assert len(report.call_shapes) == 2


def _write_jsonl(directory, rows):
    path = directory / "contract_data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_jsonl_path_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    path = _write_jsonl(
        tmp_path,
        [
            {"input": {"question": "q1"}, "output": "a1"},
            {"input": {"question": "q2"}, "output": "a2"},
        ],
    )
    report = validate_evaluation_contract(func=_agent, dataset=str(path))
    assert report.ok
    assert report.dataset_source == str(path)
    assert len(report.examples) == 2
    assert finding_codes(report) == []


def test_jsonl_path_uses_from_jsonl_not_reparsed(tmp_path, monkeypatch):
    """The contract must delegate to ``Dataset.from_jsonl``, not re-parse JSONL."""
    import traigent.evaluators.base as base

    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    path = _write_jsonl(tmp_path, [{"input": {"question": "q1"}, "output": "a1"}])

    original = base.Dataset.from_jsonl.__func__
    calls: list[str] = []

    def spy(cls, file_path):
        calls.append(file_path)
        return original(cls, file_path)

    monkeypatch.setattr(base.Dataset, "from_jsonl", classmethod(spy))

    report = validate_evaluation_contract(func=_agent, dataset=str(path))
    assert report.ok
    # Delegated exactly once, with the path the caller supplied.
    assert calls == [str(path)]
