"""End-to-end offline tests for the cold-start construction boundary."""

from __future__ import annotations

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
    GroundTruth,
    GroundTruthSource,
    ScenarioCandidate,
    ScoringContract,
    assert_optimizer_eligible,
    generate_eval_set,
)


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
