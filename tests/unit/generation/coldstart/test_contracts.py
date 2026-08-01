"""Tests for the frozen cold-start public contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from traigent.generation.coldstart.contracts import (
    COLDSTART_SCHEMA_VERSION,
    CandidateState,
    ColdStartOptions,
    ColdStartOutcome,
    ColdStartResult,
    DiscoveryGap,
    GroundTruth,
    GroundTruthSource,
    Oracle,
    ParameterSpec,
    ScenarioCandidate,
    ScenarioGenerator,
    ScoringContract,
    SystemSpec,
)


def _system() -> SystemSpec:
    return SystemSpec(
        callable_name="answer",
        module_name="example",
        parameters=(ParameterSpec("question", "str", True),),
        return_annotation="str",
        files=(),
        fingerprint="a" * 64,
    )


def test_contract_enums_are_strict_string_values() -> None:
    assert COLDSTART_SCHEMA_VERSION == "traigent.coldstart.v1"
    assert ColdStartOutcome.EVAL_SET == "eval_set"
    assert GroundTruthSource.ORACLE_COMPUTED == "oracle_computed"
    assert ScoringContract.JSON_SUBSET == "json_subset"

    with pytest.raises(ValueError):
        DiscoveryGap("missing_oracle")


def test_artifacts_are_frozen_and_mapping_fields_cannot_be_mutated() -> None:
    inputs = {"question": "What is two plus two?"}
    candidate = ScenarioCandidate("candidate-1", inputs)
    inputs["question"] = "mutated"

    assert candidate.inputs["question"] == "What is two plus two?"
    with pytest.raises(TypeError):
        candidate.inputs["other"] = "value"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        candidate.state = CandidateState.ELIGIBLE  # type: ignore[misc]

    result = ColdStartResult(
        outcome=ColdStartOutcome.DISCOVERY_ONLY,
        tuning_path=None,
        audit_path=Path("audit.jsonl"),
        manifest_path=Path("manifest.json"),
        gaps=(DiscoveryGap.NO_ORACLE,),
        counts={CandidateState.PROPOSED: 1},
    )
    with pytest.raises(TypeError):
        result.counts[CandidateState.ELIGIBLE] = 1  # type: ignore[index]

    with pytest.raises(ValueError, match="paths"):
        ColdStartResult(  # type: ignore[arg-type]
            outcome=ColdStartOutcome.DISCOVERY_ONLY,
            tuning_path=None,
            audit_path="audit.jsonl",
            manifest_path=Path("manifest.json"),
        )


def test_runtime_checkable_protocols_accept_conforming_structural_types() -> None:
    class Generator:
        technique_id = "static"

        def propose(
            self, system: SystemSpec, count: int, seed: int
        ) -> list[ScenarioCandidate]:
            return [ScenarioCandidate("candidate-1", {"question": "four"})]

    class LocalOracle:
        oracle_id = "local"
        scoring_contract = ScoringContract.EXACT_MATCH

        def ground_truth(self, inputs: dict[str, object]) -> GroundTruth:
            return GroundTruth(
                expected_output=inputs["question"],
                source=GroundTruthSource.ORACLE_COMPUTED,
                scoring_contract=self.scoring_contract,
            )

    assert isinstance(Generator(), ScenarioGenerator)
    assert isinstance(LocalOracle(), Oracle)
    assert Generator().propose(_system(), 1, 7)[0].state is CandidateState.PROPOSED


def test_options_are_strict_and_apply_safety_bounds() -> None:
    assert ColdStartOptions().include_globs == ("*.py",)

    with pytest.raises(ValueError):
        ColdStartOptions(num_candidates=0)
    with pytest.raises(ValueError):
        ColdStartOptions(max_files=0)
    with pytest.raises(ValueError):
        ColdStartOptions(max_file_bytes=0)
    with pytest.raises(ValueError):
        ColdStartOptions(include_globs=())
    with pytest.raises(ValueError):
        ColdStartOptions(unknown=True)
