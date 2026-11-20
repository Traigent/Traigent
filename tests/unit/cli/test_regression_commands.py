"""Tests for ``traigent regression add`` (committed regression set, slice 1).

No LLM calls and no network: results are built in memory and saved locally;
the egress harness would flag any socket use.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from tests.egress_canary.harness import capture_egress
from traigent.api.types import (
    ExampleResult,
    OptimizationResult,
    OptimizationStatus,
    TrialResult,
    TrialStatus,
)
from traigent.cli.main import cli
from traigent.evaluators.base import Dataset
from traigent.utils.persistence import PersistenceManager
from traigent.utils.regression_set import load_regression_dataset

ROWS: list[dict[str, Any]] = [
    {"input": {"q": "synthetic question zero"}, "output": "a0"},
    {"input": {"q": "synthetic question one"}, "output": "a1"},
    {"input": {"q": "synthetic question two"}, "output": "a2"},
    {"input": {"q": "synthetic question three"}, "output": "a3"},
]


def _example(index: int, accuracy: float, rows: list[dict[str, Any]]) -> dict:
    return ExampleResult(
        example_id=f"example_{index}",
        input_data=rows[index]["input"],
        expected_output=rows[index]["output"],
        actual_output="x",
        metrics={"accuracy": accuracy},
        execution_time=0.1,
        success=True,
    ).to_dict()


def _result(failing: set[int], rows: list[dict[str, Any]] = ROWS) -> OptimizationResult:
    per_ex = {i: (0.0 if i in failing else 1.0) for i in range(len(rows))}
    trial = TrialResult(
        trial_id="t0",
        config={"model": "a"},
        metrics={"accuracy": 0.5},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={"example_results": [_example(i, a, rows) for i, a in per_ex.items()]},
    )
    return OptimizationResult(
        trials=[trial],
        best_config={"model": "a"},
        best_score=0.5,
        optimization_id="opt-reg",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="GridSearchOptimizer",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={"function_name": "f"},
    )


def _write_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _write_dataset(tmp_path / "data.jsonl", ROWS)
    pm = PersistenceManager(str(tmp_path / "results"))
    pm.save_result(_result({1, 2}), name="run_a")
    pm.save_result(_result({2, 3}), name="run_b")
    return tmp_path


def _add(ws: Path, run: str, *extra: str):
    return CliRunner().invoke(
        cli,
        [
            "regression",
            "add",
            run,
            "--dataset",
            str(ws / "data.jsonl"),
            "--dir",
            str(ws / "reg"),
            "--storage-dir",
            str(ws / "results"),
            *extra,
        ],
    )


def _rows(ws: Path) -> list[dict[str, Any]]:
    text = (ws / "reg" / "regression.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _by_id(ws: Path) -> dict[str, dict[str, Any]]:
    return {r["metadata"]["example_id"]: r for r in _rows(ws)}


def test_group_lists_only_add() -> None:
    result = CliRunner().invoke(cli, ["regression", "--help"])
    assert result.exit_code == 0
    section = result.output.split("Commands:")[1]
    names = [line.split()[0] for line in section.splitlines() if line.strip()]
    assert names == ["add"]
    assert "committed to your" in " ".join(result.output.split())


def test_add_help_states_content_and_commit_intent() -> None:
    result = CliRunner().invoke(cli, ["regression", "add", "--help"])
    text = " ".join(result.output.split())
    assert "holds your example content" in text
    assert "committed to your own repository" in text
    assert "--on-id-conflict" in text


def test_first_add_writes_seen_once_rows_with_plan_fields(ws: Path) -> None:
    result = _add(ws, "run_a")
    assert result.exit_code == 0, result.output
    assert "Added 2 row(s), 0 new provenance entries" in result.output
    rows = _by_id(ws)
    assert sorted(rows) == ["example_1", "example_2"]
    row = rows["example_1"]
    assert row["input"] == ROWS[1]["input"] and row["output"] == "a1"
    meta = row["metadata"]
    assert meta["content_digest"].startswith("sha256:")
    reg = meta["regression"]
    assert reg["label"] == "seen_once"
    assert len(reg["provenance"]) == 1
    prov = reg["provenance"][0]
    assert prov["run"] == "run_a" and prov["metric"] == "accuracy"
    assert len(prov["config_digest"]) == 12 and prov["date"]


def test_overlapping_runs_give_one_row_per_example_with_two_provenance(
    ws: Path,
) -> None:
    assert _add(ws, "run_a").exit_code == 0
    result = _add(ws, "run_b")
    assert result.exit_code == 0, result.output
    assert "Added 1 row(s), 1 new provenance entry" in result.output
    rows = _by_id(ws)
    assert sorted(rows) == ["example_1", "example_2", "example_3"]
    runs = {
        k: [p["run"] for p in v["metadata"]["regression"]["provenance"]]
        for k, v in rows.items()
    }
    assert runs == {
        "example_1": ["run_a"],
        "example_2": ["run_a", "run_b"],
        "example_3": ["run_b"],
    }


def test_reload_through_dataset_loader_and_label_filter(ws: Path) -> None:
    assert _add(ws, "run_a").exit_code == 0
    reg_dir = ws / "reg"
    plain = Dataset.from_jsonl(str(reg_dir / "regression.jsonl"))
    assert len(plain) == 2
    loaded = load_regression_dataset(reg_dir)
    assert [e.input_data for e in loaded] == [ROWS[1]["input"], ROWS[2]["input"]]
    assert isinstance(loaded, Dataset)
    # Relabel one row on disk the way a later slice would, then filter.
    path = reg_dir / "regression.jsonl"
    rows = _rows(ws)
    rows[0]["metadata"]["regression"]["label"] = "reliable"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    only = load_regression_dataset(reg_dir, labels=("reliable",))
    assert [e.metadata["example_id"] for e in only] == ["example_1"]


def _drift_dataset_and_result(ws: Path) -> None:
    """The same positional ids now name different content."""
    drifted = [dict(r) for r in ROWS]
    drifted[2] = {"input": {"q": "drifted two"}, "output": "d2"}
    _write_dataset(ws / "data.jsonl", drifted)
    PersistenceManager(str(ws / "results")).save_result(
        _result({2, 3}, drifted), name="run_drift"
    )


def test_drifted_digest_refuses_exits_nonzero_and_leaves_file_identical(
    ws: Path,
) -> None:
    assert _add(ws, "run_a").exit_code == 0
    path = ws / "reg" / "regression.jsonl"
    before = path.read_bytes()
    _drift_dataset_and_result(ws)
    result = _add(ws, "run_drift")
    assert result.exit_code != 0
    assert "example_2" in result.output
    assert "drifted two" not in result.output
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".*.tmp"))


def test_on_id_conflict_replace_overwrites_the_stored_row(ws: Path) -> None:
    assert _add(ws, "run_a").exit_code == 0
    _drift_dataset_and_result(ws)
    result = _add(ws, "run_drift", "--on-id-conflict", "replace")
    assert result.exit_code == 0, result.output
    rows = _by_id(ws)
    assert sorted(rows) == ["example_1", "example_2", "example_3"]
    assert rows["example_2"]["input"] == {"q": "drifted two"}
    assert [
        p["run"] for p in rows["example_2"]["metadata"]["regression"]["provenance"]
    ] == ["run_drift"]


def test_on_id_conflict_keep_both_stores_both_contents(ws: Path) -> None:
    assert _add(ws, "run_a").exit_code == 0
    _drift_dataset_and_result(ws)
    result = _add(ws, "run_drift", "--on-id-conflict", "keep-both")
    assert result.exit_code == 0, result.output
    rows = _rows(ws)
    inputs = [r["input"]["q"] for r in rows]
    assert "synthetic question two" in inputs and "drifted two" in inputs
    ids = [r["metadata"]["example_id"] for r in rows]
    assert len(ids) == len(set(ids)) == 4
    # The file still loads (ids unique).
    assert len(Dataset.from_jsonl(str(ws / "reg" / "regression.jsonl"))) == 4


def test_unknown_run_exits_nonzero_like_export_failures(ws: Path) -> None:
    result = _add(ws, "no_such_run")
    assert result.exit_code != 0
    assert "Result 'no_such_run' not found" in result.output
    assert not (ws / "reg" / "regression.jsonl").exists()


def test_corrupt_existing_file_is_an_error_and_not_overwritten(ws: Path) -> None:
    reg = ws / "reg"
    reg.mkdir()
    (reg / "regression.jsonl").write_text("{not json\n", encoding="utf-8")
    result = _add(ws, "run_a")
    assert result.exit_code != 0
    assert "line 1" in result.output
    assert (reg / "regression.jsonl").read_text(encoding="utf-8") == "{not json\n"


def test_add_makes_no_network_request(ws: Path) -> None:
    with capture_egress() as cap:
        result = _add(ws, "run_a")
    assert result.exit_code == 0, result.output
    assert len(_rows(ws)) == 2
    assert cap.requests == []
    assert cap.guard_violations == []
    assert cap.dns_lookups == []
