"""Tests for the fixed, atomic cold-start artifact writer."""

from __future__ import annotations

import os

import pytest

from traigent.evaluators.base import Dataset
from traigent.generation.coldstart.writer import (
    ColdStartArtifactError,
    jsonl_bytes,
    sha256_bytes,
    write_coldstart_artifacts,
)
from traigent.utils.exceptions import ValidationError


def _tuning_row() -> dict[str, object]:
    return {
        "input": {"number": 4},
        "expected_output": 16,
        "example_id": "coldstart_example_4",
        "traigent_coldstart": {
            "schema_version": "traigent.coldstart.v1",
            "ground_truth_source": "oracle_computed",
            "scoring_contract": "exact_match",
            "oracle_id": "square",
            "generator_id": "contract_grounded",
            "seed": 7,
            "system_fingerprint": "system-digest",
            "split": "tune",
            "row_digest": "not-checked-by-loader",
        },
    }


def _manifest(row: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "traigent.coldstart.v1",
        "outcome": "eval_set",
        "dataset_path": "coldstart_tuning.jsonl",
        "dataset_sha256": sha256_bytes(jsonl_bytes([row])),
        "holdout_prohibited": True,
    }


def test_writer_emits_loader_compatible_tuning_and_unloadable_audit(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    row = _tuning_row()
    paths = write_coldstart_artifacts(
        output_dir=tmp_path,
        tuning_rows=[row],
        audit_rows=[
            {
                "artifact": "coldstart_audit",
                "candidate_digest": "a" * 64,
                "state": "admitted",
            }
        ],
        manifest=_manifest(row),
    )

    assert paths.tuning_path is not None
    dataset = Dataset.from_jsonl(str(paths.tuning_path))
    assert dataset.examples[0].input_data == {"number": 4}
    assert dataset.examples[0].expected_output == 16
    assert dataset.examples[0].metadata["traigent_coldstart"]["split"] == "tune"
    with pytest.raises(ValidationError):
        Dataset.from_jsonl(str(paths.audit_path))


def test_writer_refuses_discovery_directory_with_stale_tuning_file(tmp_path) -> None:
    (tmp_path / "coldstart_tuning.jsonl").write_text('{"input": {"x": 1}}\n')

    with pytest.raises(ColdStartArtifactError, match="Discovery-only"):
        write_coldstart_artifacts(
            output_dir=tmp_path,
            tuning_rows=None,
            audit_rows=[{"artifact": "coldstart_audit"}],
            manifest={
                "outcome": "discovery_only",
                "dataset_path": None,
                "dataset_sha256": None,
                "holdout_prohibited": True,
            },
        )


def test_writer_refuses_symlink_target(tmp_path) -> None:
    destination = tmp_path / "destination.jsonl"
    destination.write_text("existing")
    target = tmp_path / "coldstart_tuning.jsonl"
    try:
        os.symlink(destination, target)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    row = _tuning_row()
    with pytest.raises(ColdStartArtifactError, match="symlink"):
        write_coldstart_artifacts(
            output_dir=tmp_path,
            tuning_rows=[row],
            audit_rows=[{"artifact": "coldstart_audit"}],
            manifest=_manifest(row),
        )


@pytest.mark.parametrize("expected_output", [None, "", " \t\n "])
def test_writer_refuses_absent_or_blank_expected_output(
    tmp_path, expected_output: object
) -> None:
    row = _tuning_row()
    row["expected_output"] = expected_output
    with pytest.raises(ColdStartArtifactError, match="non-empty expected_output"):
        write_coldstart_artifacts(
            output_dir=tmp_path,
            tuning_rows=[row],
            audit_rows=[{"artifact": "coldstart_audit"}],
            manifest=_manifest(row),
        )


def test_writer_refuses_mapping_expected_output(tmp_path) -> None:
    row = _tuning_row()
    row["expected_output"] = {"answer": 16}

    with pytest.raises(ColdStartArtifactError, match="mapping expected_output"):
        write_coldstart_artifacts(
            output_dir=tmp_path,
            tuning_rows=[row],
            audit_rows=[{"artifact": "coldstart_audit"}],
            manifest=_manifest(row),
        )
