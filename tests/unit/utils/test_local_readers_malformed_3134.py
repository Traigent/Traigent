"""#3134: local-file readers survive damaged files instead of raising raw errors."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from traigent.analytics.meta_learning import (
    AlgorithmType,
    OptimizationHistory,
    OptimizationRecord,
    OptimizationStatus,
)
from traigent.api.types import (
    TrialResult,
    TrialStatus,
)
from traigent.core import ci_approval
from traigent.utils.file_versioning import RunVersionInfo
from traigent.utils.optimization_analyzer import OptimizationAnalyzer
from traigent.utils.optimization_logger import OptimizationLogger
from traigent.utils.outcome_matrix import load_outcome_matrix


def _write_run(base: Path, exp: str, run: str, session: Any, trials: str = "") -> Path:
    run_dir = base / "experiments" / exp / "runs" / run
    (run_dir / "meta").mkdir(parents=True)
    (run_dir / "trials").mkdir()
    (run_dir / "metrics").mkdir()
    if isinstance(session, bytes):
        (run_dir / "meta" / "session_v2.json").write_bytes(session)
    else:
        (run_dir / "meta" / "session_v2.json").write_text(
            session if isinstance(session, str) else json.dumps(session)
        )
    if trials:
        (run_dir / "trials" / "trials_v2.jsonl").write_text(trials)
    return run_dir


GOOD = {"session_id": "s", "start_time": "2025-01-01T00:00:00", "status": "ok"}


@pytest.mark.parametrize(
    "bad",
    ['{"session_id": "x", "star', "[]", b"\xff\xfe\x00bad"],
    ids=["torn", "list", "binary"],
)
def test_list_experiments_skips_damaged_session(tmp_path: Path, bad: Any) -> None:
    _write_run(tmp_path, "bad", "r1", bad)
    _write_run(tmp_path, "good", "r1", GOOD)
    df = OptimizationAnalyzer(base_path=tmp_path, file_version="2").list_experiments()
    assert "good" in df["experiment"].tolist()


@pytest.mark.parametrize("index", ['{"experiments": {"e": {"runs": [{"x":', "[]"])
def test_damaged_index_falls_back_to_directories(tmp_path: Path, index: str) -> None:
    _write_run(tmp_path, "good", "r1", GOOD)
    (tmp_path / "index.json").write_text(index)
    df = OptimizationAnalyzer(base_path=tmp_path, file_version="2").list_experiments()
    assert df["experiment"].tolist() == ["good"]


def test_index_entry_without_session_id_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "experiments": {
                    "e": {
                        "runs": [
                            {"run_id": "bad"},
                            {
                                "run_id": "ok",
                                "session_id": "s",
                                "timestamp": "2025-01-01T00:00:00",
                                "path": "p",
                            },
                        ]
                    }
                }
            }
        )
    )
    df = OptimizationAnalyzer(base_path=tmp_path, file_version="2").list_experiments()
    assert df["run_ids"].tolist() == [["ok"]]


def test_load_run_and_export_skip_bad_trial_lines(tmp_path: Path) -> None:
    trials = (
        json.dumps({"trial_id": "a", "config": {"m": 1}, "metrics": {"acc": 1}})
        + "\n[1,2]\n"
        + json.dumps({"trial_id": "b", "config": None, "metrics": None})
        + '\n{"trial_id": "torn'
    )
    _write_run(tmp_path, "e", "r1", GOOD, trials)
    an = OptimizationAnalyzer(base_path=tmp_path, file_version="2")
    data = an.load_run("e", "r1")
    assert [t["trial_id"] for t in data["trials"]] == ["a", "b"]
    (tmp_path / "out").mkdir()
    out = an.export_for_analysis("e", format="csv", output_dir=tmp_path / "out")
    assert out is not None and Path(out).exists()


def test_compare_runs_tolerates_wrong_typed_summary(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "e", "r1", GOOD)
    (run_dir / "metrics" / "metrics_summary_v2.json").write_text("[]")
    df = OptimizationAnalyzer(base_path=tmp_path, file_version="2").compare_runs("e")
    assert df["run_id"].tolist() == ["r1"]
    (run_dir / "metrics" / "metrics_summary_v2.json").write_text(
        json.dumps({"best_metrics": None})
    )
    df = OptimizationAnalyzer(base_path=tmp_path, file_version="2").compare_runs("e")
    assert df["run_id"].tolist() == ["r1"]


# --- checkpoint / version info ------------------------------------------------


def _logger_with_checkpoint(tmp_path: Path) -> OptimizationLogger:
    log = OptimizationLogger(
        experiment_name="exp",
        session_id="sess12345678",
        execution_mode="local",
        base_path=tmp_path,
    )
    trial = TrialResult(
        trial_id="t1",
        config={"m": 1},
        metrics={"accuracy": 0.9},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={},
    )
    log.save_checkpoint(optimizer_state={}, trials_history=[trial], trial_count=1)
    return log


@pytest.mark.parametrize(
    "content", ["[]", "null", "{}", '{"checkpoint_file": 5}', "{bad"]
)
def test_load_checkpoint_damaged_latest_raises_value_error(
    tmp_path: Path, content: str
) -> None:
    log = _logger_with_checkpoint(tmp_path)
    latest = next((log.run_path / "checkpoints").glob("latest_checkpoint_v*.json"))
    latest.write_text(content)
    with pytest.raises(ValueError):
        OptimizationLogger.load_checkpoint("exp", log.run_id, base_path=tmp_path)


def test_load_checkpoint_damaged_checkpoint_raises_value_error(tmp_path: Path) -> None:
    log = _logger_with_checkpoint(tmp_path)
    cp = next((log.run_path / "checkpoints").glob("checkpoint_00001_v*.json"))
    cp.write_text("[1]")
    with pytest.raises(ValueError):
        OptimizationLogger.load_checkpoint("exp", log.run_id, base_path=tmp_path)


def test_version_info_wrong_type_raises_value_error(tmp_path: Path) -> None:
    info = RunVersionInfo(tmp_path)
    info.version_file.parent.mkdir(parents=True, exist_ok=True)
    info.version_file.write_text(json.dumps({"traigent_version": 1}))
    with pytest.raises(ValueError):
        info.check_compatibility("1.0.0")
    info.version_file.write_text("[]")
    with pytest.raises(ValueError):
        info.check_compatibility("1.0.0")


# --- approval token (fail closed) ---------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "null",
        "[]",
        '"str"',
        '{"approved_by": "a", "expires_at": 12345}',
        '{"approved_by": "a", "expires_at": null}',
        '{"approver": "a", "expires_iso": 1, "nonce": "n", "signature": 5}',
        '{"approver": "a", "expires_iso": null, "nonce": "n", "signature": "x"}',
        '{"approved_by":',
    ],
)
def test_malformed_approval_token_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    monkeypatch.setenv("TRAIGENT_APPROVAL_SECRET", "s3cret")
    token = tmp_path / "approval.token"
    token.write_text(content)
    assert ci_approval._check_token_file_approval(token, tmp_path) is False


def test_non_utf8_approval_token_is_rejected(tmp_path: Path) -> None:
    token = tmp_path / "approval.token"
    token.write_bytes(b"\xff\xfe\x00")
    assert ci_approval._check_token_file_approval(token, tmp_path) is False


# --- outcome matrix -------------------------------------------------------------


@pytest.mark.parametrize("content", [b"[1]", b"null", b"\xff\xfe\x00", b'{"a":'])
def test_load_outcome_matrix_damaged_raises_value_error(
    tmp_path: Path, content: bytes
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "outcome_matrix_v2.json").write_bytes(content)
    with pytest.raises(ValueError):
        load_outcome_matrix(tmp_path)


# --- history --------------------------------------------------------------------


def _record(i: int) -> OptimizationRecord:
    return OptimizationRecord(
        optimization_id=f"o{i}",
        function_name="f",
        algorithm=AlgorithmType.GRID,
        configuration_space={"p": [1]},
        objectives=["accuracy"],
        dataset_size=1,
        best_score=0.5,
        total_trials=1,
        duration_seconds=1,
        status=OptimizationStatus.COMPLETED,
        timestamp=datetime.now(UTC),
    )


@pytest.mark.parametrize("bad", ['{"optimization_id": "x', "[1]", '{"a": 1}'])
def test_history_continues_past_a_bad_line(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "h.jsonl"
    hist = OptimizationHistory(storage_path=str(path))
    hist.add_record(_record(1))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(bad + "\n")
    hist.add_record(_record(2))
    reloaded = OptimizationHistory(storage_path=str(path))
    assert [r.optimization_id for r in reloaded.records] == ["o1", "o2"]


def test_history_invalid_utf8_and_deep_nesting_lines_are_counted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "h.jsonl"
    hist = OptimizationHistory(storage_path=str(path))
    hist.add_record(_record(1))
    good = path.read_bytes().splitlines()[0]
    bad_utf8 = good.replace(b'"f"', b'"f\xff"')
    assert bad_utf8 != good
    with open(path, "ab") as fh:
        fh.write(bad_utf8 + b"\n")
        fh.write(b"[" * 100000 + b"\n")
    hist.add_record(_record(2))
    reloaded = OptimizationHistory(storage_path=str(path))
    assert [r.optimization_id for r in reloaded.records] == ["o1", "o2"]
    assert reloaded._skipped_lines == 2


def test_trial_jsonl_non_utf8_line_is_skipped(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "e", "r1", GOOD)
    first = json.dumps({"trial_id": "a"}).encode()
    last = json.dumps({"trial_id": "c"}).encode()
    (run_dir / "trials" / "trials_v2.jsonl").write_bytes(
        first + b'\n{"trial_id": "b\xff"}\n' + last + b"\n"
    )
    data = OptimizationAnalyzer(base_path=tmp_path, file_version="2").load_run(
        "e", "r1"
    )
    assert [t["trial_id"] for t in data["trials"]] == ["a", "c"]


@pytest.mark.parametrize(
    "index",
    [
        '{"experiments": []}',
        '{"experiments": "x"}',
        '{"experiments": {"e": {"runs": [{"run_id": "bad"}]}}}',
        '{"experiments": {"e": 5}}',
    ],
)
def test_index_with_unusable_experiments_rebuilds(tmp_path: Path, index: str) -> None:
    _write_run(tmp_path, "good", "r1", GOOD)
    (tmp_path / "index.json").write_text(index)
    df = OptimizationAnalyzer(base_path=tmp_path, file_version="2").list_experiments()
    assert df["experiment"].tolist() == ["good"]
