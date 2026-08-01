"""End-to-end offline tests for the cold-start construction boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import traigent
from traigent.core.execution_budget import ExecutionBudget
from traigent.evaluators.base import Dataset
from traigent.generation.coldstart import (
    CallableOracle,
    ColdStartConfigurationError,
    ColdStartOptions,
    ColdStartOutcome,
    DiscoveryGap,
    GroundTruth,
    GroundTruthSource,
    ScenarioCandidate,
    ScoringContract,
    assert_optimizer_eligible,
    generate_eval_set,
)
from traigent.generation.coldstart.writer import canonical_json_bytes, sha256_bytes


REPO_ROOT = Path(__file__).resolve().parents[4]
_OPTIONS = ColdStartOptions(
    num_candidates=2,
    max_files=10,
    include_globs=("tests/unit/generation/coldstart/test_pipeline.py",),
)


def _target_must_not_run(number: int) -> int:
    raise AssertionError("cold-start construction must not execute the target")


def _untyped_target(number):  # noqa: ANN001
    raise AssertionError("cold-start construction must not execute the target")


def _unsupported_target() -> int:
    raise AssertionError("cold-start construction must not execute the target")


def _two_input_target_must_not_run(number: int, suffix: str) -> int:
    raise AssertionError("cold-start construction must not execute the target")


@traigent.optimize(
    eval_dataset=None,
    objectives=["accuracy"],
    configuration_space={"implementation": ["correct", "predictably-wrong"]},
    injection_mode="seamless",
    offline=True,
)
def _decorated_target_must_not_run(number: int) -> int:
    raise AssertionError("cold-start construction must not execute the target")


class _InputsOnlyGenerator:
    technique_id = "test.inputs_only.v1"

    def __init__(self, inputs: dict[str, object]) -> None:
        self._inputs = inputs

    def propose(self, system, count: int, seed: int):  # noqa: ANN001
        return [ScenarioCandidate("candidate-1", self._inputs)]


class _TwoInputsGenerator:
    technique_id = "test.two_inputs.v1"

    def propose(self, system, count: int, seed: int):  # noqa: ANN001
        return [
            ScenarioCandidate("candidate-1", {"number": 3}),
            ScenarioCandidate("candidate-2", {"number": 4}),
        ]


class _ModelGoldGenerator:
    technique_id = "test.model_gold.v1"

    def propose(self, system, count: int, seed: int):  # noqa: ANN001
        return [
            ScenarioCandidate(
                "candidate-1",
                {"number": 3},
                GroundTruth(
                    "model-made-label",
                    GroundTruthSource.MODEL_PROPOSED,
                    ScoringContract.EXACT_MATCH,
                ),
            )
        ]


class _SelfDeclaredOracleGoldGenerator:
    technique_id = "test.self_declared_oracle_gold.v1"

    def propose(self, system, count: int, seed: int):  # noqa: ANN001
        return [
            ScenarioCandidate(
                "candidate-1",
                {"number": 3},
                GroundTruth(
                    "self-declared-label",
                    GroundTruthSource.ORACLE_COMPUTED,
                    ScoringContract.EXACT_MATCH,
                ),
            )
        ]


class _EmptyGenerator:
    technique_id = "test.empty.v1"

    def propose(self, system, count: int, seed: int):  # noqa: ANN001
        return []


class _FailingOracle:
    oracle_id = "test.failing.v1"
    scoring_contract = ScoringContract.EXACT_MATCH

    def __init__(self) -> None:
        self.calls = 0

    def ground_truth(self, inputs):  # noqa: ANN001
        self.calls += 1
        raise RuntimeError("oracle failed")


def _square_oracle(inputs) -> int:  # noqa: ANN001
    number = inputs["number"]
    assert isinstance(number, int)
    return number * number


def _output_dir(tmp_path, monkeypatch) -> Path:  # noqa: ANN001
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    return tmp_path / "coldstart"


def test_pipeline_never_executes_target_and_writes_eligible_dataset(
    tmp_path, monkeypatch
) -> None:
    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(_square_oracle, oracle_id="test.square.v1"),
        generator=_InputsOnlyGenerator({"number": 3}),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )

    assert result.outcome is ColdStartOutcome.EVAL_SET
    assert result.tuning_path is not None
    assert result.contract_report is not None and result.contract_report.ok
    assert_optimizer_eligible(result.tuning_path)
    tuning = result.tuning_path.read_text(encoding="utf-8")
    assert '"example_id"' in tuning
    assert '"split":"tune"' in tuning
    assert "holdout" not in tuning


def test_decorated_target_is_not_executed_and_accepts_generated_override(
    tmp_path, monkeypatch
) -> None:
    result = generate_eval_set(
        func=_decorated_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(_square_oracle, oracle_id="test.square.v1"),
        generator=_InputsOnlyGenerator({"number": 3}),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )

    assert result.outcome is ColdStartOutcome.EVAL_SET
    assert result.tuning_path is not None
    generated = Dataset.from_jsonl(str(result.tuning_path))
    _decorated_target_must_not_run.set_eval_dataset_override(generated)
    assert _decorated_target_must_not_run._dataset_override is generated


def test_missing_oracle_is_discovery_only_without_tuning_file(
    tmp_path, monkeypatch
) -> None:
    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=None,
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.tuning_path is None
    assert not (result.audit_path.parent / "coldstart_tuning.jsonl").exists()
    assert result.gaps


def test_untyped_contract_is_discovery_only(tmp_path, monkeypatch) -> None:
    result = generate_eval_set(
        func=_untyped_target,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(_square_oracle),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.tuning_path is None
    assert result.gaps == (DiscoveryGap.UNTYPED_INPUT_CONTRACT,)


def test_unsupported_contract_is_discovery_only_with_its_own_gap(
    tmp_path, monkeypatch
) -> None:
    result = generate_eval_set(
        func=_unsupported_target,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(_square_oracle),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.tuning_path is None
    assert result.gaps == (DiscoveryGap.UNSUPPORTED_INPUT_CONTRACT,)


def test_static_inspection_failure_has_its_own_discovery_gap(
    tmp_path, monkeypatch
) -> None:
    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=tmp_path,
        oracle=CallableOracle(_square_oracle),
        generator=_InputsOnlyGenerator({"number": 3}),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.gaps == (DiscoveryGap.STATIC_INSPECTION_FAILED,)


def test_model_created_label_is_discarded_and_oracle_regrounds_input(
    tmp_path, monkeypatch
) -> None:
    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(_square_oracle),
        generator=_ModelGoldGenerator(),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )

    assert result.tuning_path is not None
    payload = result.tuning_path.read_text(encoding="utf-8")
    assert "model-made-label" not in payload
    assert '"ground_truth_source":"oracle_computed"' in payload


def test_self_declared_oracle_gold_cannot_bypass_the_injected_oracle(
    tmp_path, monkeypatch
) -> None:
    calls = 0

    def oracle(inputs):  # noqa: ANN001
        nonlocal calls
        calls += 1
        return inputs["number"] * inputs["number"]

    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(oracle),
        generator=_SelfDeclaredOracleGoldGenerator(),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )

    assert calls == 1
    assert result.tuning_path is not None
    payload = result.tuning_path.read_text(encoding="utf-8")
    assert "self-declared-label" not in payload
    assert '"expected_output":9' in payload


def test_empty_proposals_have_a_distinct_discovery_gap(tmp_path, monkeypatch) -> None:
    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(_square_oracle),
        generator=_EmptyGenerator(),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.gaps == (DiscoveryGap.NO_SCENARIOS_PROPOSED,)


def test_evaluation_contract_mismatch_has_a_distinct_discovery_gap(
    tmp_path, monkeypatch
) -> None:
    result = generate_eval_set(
        func=_two_input_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(lambda inputs: "safe"),
        generator=_InputsOnlyGenerator({"unexpected": 3}),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.gaps == (DiscoveryGap.EVALUATION_CONTRACT_MISMATCH,)


def test_oracle_failure_still_counts_the_oracle_call_in_the_budget(
    tmp_path, monkeypatch
) -> None:
    budget = ExecutionBudget(max_examples=10)
    oracle = _FailingOracle()
    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=oracle,
        generator=_InputsOnlyGenerator({"number": 3}),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
        budget=budget,
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert oracle.calls == 1
    assert budget.consumed_examples == 2


@pytest.mark.parametrize("empty_gold", ["", " \t\n "])
def test_blank_oracle_gold_is_no_gold_and_charges_actual_oracle_calls(
    tmp_path, monkeypatch, empty_gold: str
) -> None:
    calls = 0

    def oracle(inputs):  # noqa: ANN001
        nonlocal calls
        calls += 1
        return empty_gold

    budget = ExecutionBudget(max_examples=10)
    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(oracle),
        generator=_TwoInputsGenerator(),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
        budget=budget,
    )

    assert result.outcome is ColdStartOutcome.DISCOVERY_ONLY
    assert result.tuning_path is None
    assert result.gaps == (DiscoveryGap.NO_GOLD_DERIVABLE,)
    assert calls == 2
    assert budget.consumed_examples == 4


def test_audit_and_manifest_do_not_export_candidate_content(
    tmp_path, monkeypatch
) -> None:
    sentinel = "private-candidate-content-63c421"
    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(lambda inputs: "safe"),
        generator=_InputsOnlyGenerator({"number": sentinel}),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )

    assert result.tuning_path is not None
    assert sentinel in result.tuning_path.read_text(encoding="utf-8")
    assert sentinel not in result.audit_path.read_text(encoding="utf-8")
    assert sentinel not in result.manifest_path.read_text(encoding="utf-8")


def test_integrity_rejects_altered_dataset_and_unknown_cost_stays_unknown(
    tmp_path, monkeypatch
) -> None:
    budget = ExecutionBudget(max_examples=100)
    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(_square_oracle),
        generator=_InputsOnlyGenerator({"number": 3}),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
        budget=budget,
    )

    assert result.tuning_path is not None
    assert budget.cost_tracking == "untracked"
    result.tuning_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ColdStartConfigurationError):
        assert_optimizer_eligible(result.tuning_path)


def test_integrity_rejects_manifest_row_provenance_mismatch(
    tmp_path, monkeypatch
) -> None:
    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(_square_oracle),
        generator=_InputsOnlyGenerator({"number": 3}),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )
    assert result.tuning_path is not None
    row = json.loads(result.tuning_path.read_text(encoding="utf-8"))
    provenance = row["traigent_coldstart"]
    provenance["oracle_id"] = "forged.oracle.v1"
    unsigned_row = dict(row)
    unsigned_provenance = dict(provenance)
    unsigned_provenance.pop("row_digest")
    unsigned_row["traigent_coldstart"] = unsigned_provenance
    provenance["row_digest"] = sha256_bytes(canonical_json_bytes(unsigned_row))
    payload = canonical_json_bytes(row) + b"\n"
    result.tuning_path.write_bytes(payload)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_sha256"] = sha256_bytes(payload)
    result.manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ColdStartConfigurationError, match="construction-evidence"):
        assert_optimizer_eligible(result.tuning_path)


@pytest.mark.parametrize("empty_gold", ["", " \t\n "])
def test_integrity_rejects_recomputed_empty_gold_row(
    tmp_path, monkeypatch, empty_gold: str
) -> None:
    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(_square_oracle),
        generator=_InputsOnlyGenerator({"number": 3}),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )
    assert result.tuning_path is not None
    row = json.loads(result.tuning_path.read_text(encoding="utf-8"))
    row["expected_output"] = empty_gold
    provenance = row["traigent_coldstart"]
    unsigned_row = dict(row)
    unsigned_provenance = dict(provenance)
    unsigned_provenance.pop("row_digest")
    unsigned_row["traigent_coldstart"] = unsigned_provenance
    provenance["row_digest"] = sha256_bytes(canonical_json_bytes(unsigned_row))
    payload = canonical_json_bytes(row) + b"\n"
    result.tuning_path.write_bytes(payload)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_sha256"] = sha256_bytes(payload)
    result.manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(ColdStartConfigurationError, match="construction-evidence"):
        assert_optimizer_eligible(result.tuning_path)


def test_integrity_preserves_dataset_trusted_root_diagnostics(
    tmp_path, monkeypatch
) -> None:
    result = generate_eval_set(
        func=_target_must_not_run,
        repo_root=REPO_ROOT,
        oracle=CallableOracle(_square_oracle),
        generator=_InputsOnlyGenerator({"number": 3}),
        options=_OPTIONS,
        output_dir=_output_dir(tmp_path, monkeypatch),
    )
    assert result.tuning_path is not None
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(other_root))

    with pytest.raises(
        ColdStartConfigurationError, match="Dataset path must reside under"
    ):
        assert_optimizer_eligible(result.tuning_path)
