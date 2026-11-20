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
from traigent.utils.hashing import generate_config_hash
from traigent.utils.persistence import PersistenceManager
from traigent.utils.regression_set import load_regression_dataset

ROWS: list[dict[str, Any]] = [
    {"input": {"q": "synthetic question zero"}, "output": "a0"},
    {"input": {"q": "synthetic question one"}, "output": "a1"},
    {"input": {"q": "synthetic question two"}, "output": "a2"},
    {"input": {"q": "synthetic question three"}, "output": "a3"},
]


def _example(
    index: int,
    accuracy: float,
    rows: list[dict[str, Any]],
    ids: list[str] | None = None,
    error: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict:
    return ExampleResult(
        example_id=ids[index] if ids else f"example_{index}",
        input_data=rows[index]["input"],
        expected_output=rows[index]["output"],
        actual_output="x",
        metrics=metrics if metrics is not None else {"accuracy": accuracy},
        execution_time=0.1,
        success=error is None,
        error_message=error,
    ).to_dict()


def _result(
    failing: set[int],
    rows: list[dict[str, Any]] = ROWS,
    ids: list[str] | None = None,
    oid: str = "opt-reg",
    objectives: list[str] | None = None,
    errors: dict[int, str] | None = None,
    scores: dict[int, float] | None = None,
) -> OptimizationResult:
    per_ex = {i: (0.0 if i in failing else 1.0) for i in range(len(rows))}
    per_ex.update(scores or {})
    trial = TrialResult(
        trial_id="t0",
        config={"model": "a"},
        metrics={"accuracy": 0.5},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={
            "example_results": [
                _example(i, a, rows, ids, (errors or {}).get(i))
                for i, a in per_ex.items()
            ]
        },
    )
    return OptimizationResult(
        trials=[trial],
        best_config={"model": "a"},
        best_score=0.5,
        optimization_id=oid,
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=objectives or ["accuracy"],
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
    pm.save_result(_result({1, 2}, oid="opt-a"), name="run_a")
    pm.save_result(_result({2, 3}, oid="opt-b"), name="run_b")
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
    assert "Review the file before you commit" in " ".join(result.output.split())


def test_help_says_content_is_unredacted_and_makes_no_redaction_claim() -> None:
    for argv in (["regression", "--help"], ["regression", "add", "--help"]):
        result = CliRunner().invoke(cli, argv)
        text = " ".join(result.output.split())
        assert "exactly as it is in your dataset" in text
        assert "nothing in it is redacted" in text.lower()
        assert "nothing is uploaded" in text.lower()
        assert "review the file before you commit" in text.lower()
        assert "pattern redaction" not in text.lower()
        assert "redacted" not in text.lower().replace("nothing in it is redacted", "")
    assert "--on-id-conflict" in " ".join(
        CliRunner().invoke(cli, ["regression", "add", "--help"]).output.split()
    )


def test_written_rows_are_raw_not_redacted(ws: Path) -> None:
    rows = [dict(r) for r in ROWS]
    rows[1] = {"input": {"q": "mail alice@example.com"}, "output": "a1"}
    _write_dataset(ws / "data.jsonl", rows)
    PersistenceManager(str(ws / "results")).save_result(
        _result({1}, rows), name="run_raw"
    )
    assert _add(ws, "run_raw").exit_code == 0
    assert _rows(ws)[0]["input"] == {"q": "mail alice@example.com"}


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
    assert reg["schema_version"] == 1
    assert prov["run"] == "run_a" and prov["run_id"] == "opt-a"
    assert prov["imported"]
    assert prov["mode"] == "best_config" and prov["threshold"] == 0.5
    assert prov["config_digest"] == generate_config_hash({"model": "a"})
    assert (prov["failed_by"], prov["metric_field"]) == ("score", "accuracy")
    assert "metric" not in prov and "date" not in prov


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
        _result({2, 3}, drifted, oid="opt-drift"), name="run_drift"
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
    entries = rows["example_2"]["metadata"]["regression"]["provenance"]
    # the replaced content's history is kept and marked, then the new entry
    assert [p["run"] for p in entries] == ["run_a", "run_drift"]
    assert "superseded_content_digest" in entries[0]
    assert "superseded_content_digest" not in entries[1]


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


def _custom_run(ws: Path, rows: list[dict[str, Any]], ids: list[str]) -> None:
    _write_dataset(ws / "data.jsonl", rows)
    PersistenceManager(str(ws / "results")).save_result(
        _result({2, 3}, rows, ids), name="run_ids"
    )


def _assert_reloads_with_failed_rows(ws: Path, expected: dict[str, str]) -> None:
    reloaded = Dataset.from_jsonl(ws / "reg" / "regression.jsonl")
    assert {
        e.metadata["example_id"]: e.expected_output for e in reloaded.examples
    } == expected


def test_custom_example_ids_are_stored_as_the_run_recorded_them(ws: Path) -> None:
    rows = [{**r, "metadata": {"example_id": f"case-{i}"}} for i, r in enumerate(ROWS)]
    _custom_run(ws, rows, [f"case-{i}" for i in range(4)])
    result = _add(ws, "run_ids")
    assert result.exit_code == 0, result.output
    _assert_reloads_with_failed_rows(ws, {"case-2": "a2", "case-3": "a3"})


@pytest.mark.parametrize("second_absent", ["", None])
def test_absent_ids_use_the_original_dataset_position(
    ws: Path, second_absent: str | None
) -> None:
    """Failing rows 2 and 3 carry an empty/None/missing id; the run used the
    positional id of the original dataset (not the position in the subset)."""
    rows = [dict(r) for r in ROWS]
    rows[2] = {**rows[2], "metadata": {"example_id": ""}}
    rows[3] = {**rows[3], "metadata": {"example_id": second_absent}}
    _custom_run(ws, rows, [f"example_{i}" for i in range(4)])
    result = _add(ws, "run_ids")
    assert result.exit_code == 0, result.output
    _assert_reloads_with_failed_rows(ws, {"example_2": "a2", "example_3": "a3"})


def test_add_refuses_row_edited_by_hand_and_file_is_identical(ws: Path) -> None:
    assert _add(ws, "run_a").exit_code == 0
    path = ws / "reg" / "regression.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["output"] = "zzz-sentinel-edit"
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    before = path.read_bytes()
    result = _add(ws, "run_b")
    assert result.exit_code != 0
    assert "line 1" in result.output and "example_1" in result.output
    assert "zzz-sentinel-edit" not in result.output
    assert path.read_bytes() == before


def _prov(ws: Path, example_id: str) -> list[dict[str, Any]]:
    return _by_id(ws)[example_id]["metadata"]["regression"]["provenance"]


def test_cost_first_run_records_the_field_that_decided_not_the_objective(
    ws: Path,
) -> None:
    PersistenceManager(str(ws / "results")).save_result(
        _result({1}, oid="opt-cost", objectives=["cost", "accuracy"]), name="run_cost"
    )
    assert _add(ws, "run_cost").exit_code == 0
    (prov,) = _prov(ws, "example_1")
    assert prov["metric_field"] == "accuracy" and prov["failed_by"] == "score"
    assert "cost" not in prov.values()


def test_execution_error_is_recorded_as_error_not_as_a_score(ws: Path) -> None:
    PersistenceManager(str(ws / "results")).save_result(
        _result({1}, oid="opt-err", errors={1: "boom"}, scores={1: 1.0}),
        name="run_err",
    )
    assert _add(ws, "run_err").exit_code == 0
    (prov,) = _prov(ws, "example_1")
    assert (prov["failed_by"], prov["metric_field"]) == ("error", "error")
    assert "boom" not in json.dumps(_rows(ws))


def test_same_saved_run_under_another_name_adds_no_provenance(ws: Path) -> None:
    assert _add(ws, "run_a").exit_code == 0
    before = (ws / "reg" / "regression.jsonl").read_bytes()
    pm = PersistenceManager(str(ws / "results"))
    pm.save_result(_result({1, 2}, oid="opt-a"), name="run_a_again")
    result = _add(ws, "run_a_again")
    assert result.exit_code == 0, result.output
    assert "0 new provenance entries" in result.output
    assert (ws / "reg" / "regression.jsonl").read_bytes() == before


def test_all_configs_mode_records_null_config_digest_and_mode(ws: Path) -> None:
    assert _add(ws, "run_a", "--mode", "all_configs").exit_code == 0
    (prov,) = _prov(ws, "example_1")
    assert prov["mode"] == "all_configs" and prov["config_digest"] is None


def test_custom_threshold_changes_what_fails_and_is_recorded(ws: Path) -> None:
    PersistenceManager(str(ws / "results")).save_result(
        _result({3}, oid="opt-thr", scores={1: 0.6, 2: 0.8}), name="run_thr"
    )
    assert _add(ws, "run_thr", "--threshold", "0.7").exit_code == 0
    rows = _by_id(ws)
    assert sorted(rows) == ["example_1", "example_3"]  # 0.6 < 0.7, 0.8 passes
    assert _prov(ws, "example_1")[0]["threshold"] == 0.7


def test_success_says_the_file_is_unredacted_and_to_review_it(ws: Path) -> None:
    result = _add(ws, "run_a")
    assert result.exit_code == 0, result.output
    assert "unredacted example content" in result.output
    assert (
        "metadata" in result.output and "Review it before you commit" in result.output
    )


def test_success_names_the_gitignore_it_wrote_only_the_first_time(ws: Path) -> None:
    first = _add(ws, "run_a")
    assert first.exit_code == 0, first.output
    assert "lock file" in first.output and ".gitignore" in first.output
    assert (ws / "reg" / ".gitignore").read_text() == ".regression.jsonl.lock\n"
    second = _add(ws, "run_b")
    assert second.exit_code == 0, second.output
    assert ".gitignore" not in second.output


def test_errors_do_not_echo_exception_text(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "sentinel-example-text"

    def boom(*_a: Any, **_k: Any) -> Any:
        raise ValueError(sentinel)

    from traigent.utils import regression_dataset

    monkeypatch.setattr(regression_dataset, "select_examples_with_run_ids", boom)
    result = _add(ws, "run_a")
    assert result.exit_code != 0 and sentinel not in result.output
    assert "ValueError" in result.output

    monkeypatch.setattr(Dataset, "from_jsonl", boom)
    result = _add(ws, "run_a")
    assert result.exit_code != 0 and sentinel not in result.output
    assert "ValueError" in result.output
    assert not (ws / "reg").exists()


def test_zero_failures_exits_zero_and_creates_nothing(ws: Path) -> None:
    PersistenceManager(str(ws / "results")).save_result(
        _result(set(), oid="opt-clean"), name="run_clean"
    )
    result = _add(ws, "run_clean")
    assert result.exit_code == 0, result.output
    assert "No failing examples" in result.output
    assert not (ws / "reg").exists()


def test_zero_failures_leaves_an_existing_file_untouched(ws: Path) -> None:
    assert _add(ws, "run_a").exit_code == 0
    path = ws / "reg" / "regression.jsonl"
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns
    PersistenceManager(str(ws / "results")).save_result(
        _result(set(), oid="opt-clean"), name="run_clean"
    )
    assert _add(ws, "run_clean").exit_code == 0
    assert (path.read_bytes(), path.stat().st_mtime_ns) == (before, mtime)


def test_dataset_option_accepts_json_as_its_help_says(ws: Path) -> None:
    (ws / "data.json").write_text(json.dumps(ROWS), encoding="utf-8")
    result = CliRunner().invoke(
        cli,
        [
            "regression",
            "add",
            "run_a",
            "--dataset",
            str(ws / "data.json"),
            "--dir",
            str(ws / "reg"),
            "--storage-dir",
            str(ws / "results"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert sorted(_by_id(ws)) == ["example_1", "example_2"]
