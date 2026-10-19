"""CLI failure paths exit non-zero and write only where asked.

Covers Traigent#2661 (``optimize FILE`` kwargs), #2662 (relative ``export -o``
written against the caller's cwd), #2663 (``validate-config`` exit codes) and
#2665 (``export`` failure exit codes).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.cli.main import cli
from traigent.utils.persistence import PersistenceManager


@pytest.fixture
def stored(tmp_path: Path) -> Path:
    """A storage dir holding one saved result named ``run1``."""
    store = tmp_path / "store"
    result = OptimizationResult(
        trials=[],
        best_config={"model": "a"},
        best_score=1.0,
        optimization_id="opt-1",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=__import__("datetime").datetime.now(),
    )
    PersistenceManager(str(store)).save_result(result, "run1")
    return store


# ---- #2661 -----------------------------------------------------------------
def test_optimize_file_runs_decorated_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "res"))
    monkeypatch.delenv("TRAIGENT_API_KEY", raising=False)
    (tmp_path / "d.jsonl").write_text(
        '{"input":{"q":"1"},"output":"aa"}\n{"input":{"q":"2"},"output":"aa"}\n'
    )
    app = tmp_path / "app_2661.py"
    app.write_text(
        "import traigent\n"
        '@traigent.optimize(configuration_space={"v": ["aa","bb"]}, '
        'objectives=["accuracy"], eval_dataset="d.jsonl", offline=True, '
        'max_trials=2, algorithm="grid")\n'
        "def agent(q: str) -> str:\n"
        '    return traigent.get_config()["v"]\n'
    )
    result = CliRunner().invoke(cli, ["optimize", str(app)])
    assert "Unknown keyword" not in result.output, result.output
    assert result.exit_code == 0, result.output


def test_optimize_kwargs_do_not_include_cli_only_verbose() -> None:
    from traigent.cli.main import _build_optimize_kwargs

    kwargs = _build_optimize_kwargs("grid", 2, 10.0, True, False, None, None, None)
    assert "verbose" not in kwargs


# ---- #2662 / #2665 ---------------------------------------------------------
def test_export_relative_output_written_under_cwd(
    stored: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    result = CliRunner().invoke(
        cli, ["export", "run1", "-d", str(stored), "-o", "cfg/x.json"]
    )
    assert result.exit_code == 0, result.output
    written = work / "cfg" / "x.json"
    assert written.is_file()
    assert json.loads(written.read_text())["config"] == {"model": "a"}
    assert str(written) in result.output.replace("\n", "")


def test_export_directory_target_exits_nonzero(
    stored: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "adir").mkdir()
    result = CliRunner().invoke(
        cli, ["export", "run1", "-d", str(stored), "-o", "adir"]
    )
    assert result.exit_code != 0, result.output


def test_export_missing_result_exits_nonzero(
    stored: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["export", "nope", "-d", str(stored)])
    assert result.exit_code != 0, result.output
    assert "nope" in result.output


def test_export_rejected_path_exits_nonzero_and_creates_no_dirs(
    stored: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli, ["export", "run1", "-d", str(stored), "-o", "/etc/newdir/x.json"]
    )
    assert result.exit_code != 0, result.output
    assert not Path("/etc/newdir").exists()


def test_export_valid_path_still_exits_zero(
    stored: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli, ["export", "run1", "-d", str(stored), "-o", "ok.json"]
    )
    assert result.exit_code == 0, result.output


# ---- #2663 -----------------------------------------------------------------
def test_validate_config_invalid_json_exits_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"a":')
    result = CliRunner().invoke(cli, ["validate-config", str(bad)])
    assert result.exit_code != 0, result.output
    assert "Invalid JSON" in result.output


def test_validate_config_wrong_schema_exits_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"configuration_space": {}, "objectives": []}')
    result = CliRunner().invoke(cli, ["validate-config", str(bad)])
    assert result.exit_code != 0, result.output


def test_validate_config_non_object_json_exits_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    result = CliRunner().invoke(cli, ["validate-config", str(bad)])
    assert result.exit_code != 0, result.output


def test_validate_config_unreadable_file_exits_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe\x00bad")
    result = CliRunner().invoke(cli, ["validate-config", str(bad)])
    assert result.exit_code != 0, result.output


def test_validate_config_valid_input_exits_zero(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text('{"configuration_space": {"v": ["aa", "bb"]}}')
    result = CliRunner().invoke(cli, ["validate-config", str(good)])
    assert result.exit_code == 0, result.output


# ---- secrecy canaries --------------------------------------------------------
def test_export_not_found_message_does_not_echo_storage_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    secret_dir = tmp_path / "SECRETCANARY-store"
    secret_dir.mkdir()
    result = CliRunner().invoke(cli, ["export", "nope", "-d", str(secret_dir)])
    assert result.exit_code != 0
    assert "SECRETCANARY" not in result.output


def test_path_validation_error_does_not_echo_resolved_path(tmp_path: Path) -> None:
    from traigent.utils.validation import Validators

    secret = tmp_path / "SECRETCANARY" / "data.jsonl"
    result = Validators.validate_path(
        secret, "dataset", allowed_base_dirs=[tmp_path / "other"]
    )
    assert not result.is_valid
    assert "SECRETCANARY" not in result.get_feedback()


def test_rerank_applies_selector_eligibility_on_a_tie(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partially covered v4 listed before fully covered v5 must not win (#2660)."""
    from datetime import UTC, datetime

    from traigent.api.types import TrialResult, TrialStatus

    def comp(present: int) -> dict:
        return {
            "schema_version": "1.0",
            "primary_objective": "accuracy",
            "evaluation_mode": "evaluated",
            "total_examples": 8,
            "examples_with_primary_metric": present,
            "coverage_ratio": present / 8,
            "derivation_path": "none",
            "ranking_eligible": present == 8,
            "warning_codes": [] if present == 8 else ["MCI-002"],
            "per_metric_coverage": {
                "accuracy": {"present": present, "total": 8, "ratio": present / 8}
            },
        }

    trials = [
        TrialResult(
            trial_id=tid,
            config={"variant": tid},
            metrics={"accuracy": 0.625},
            status=TrialStatus.COMPLETED,
            duration=1.0,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            metadata={"comparability": comp(present)},
        )
        for tid, present in (("v4", 5), ("v5", 8))
    ]
    result = OptimizationResult(
        trials=trials,
        best_config={"variant": "v5"},
        best_score=0.625,
        optimization_id="opt-tie",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(),
        metadata={"session_summary": {"ranking": {"comparability_mode": "warn"}}},
    )
    store = tmp_path / "store"
    PersistenceManager(str(store)).save_result(result, "tie")
    monkeypatch.chdir(tmp_path)
    out = CliRunner().invoke(
        cli, ["results", "rerank", "tie", "-w", "accuracy=1.0", "-d", str(store)]
    )
    assert out.exit_code == 0, out.output
    assert "v5" in out.output
    assert "v4" not in out.output.split("New Best Configuration")[1].split("Best")[0]


def _partial_winner_result(mode: str | None):
    from datetime import UTC, datetime

    from traigent.api.types import TrialResult, TrialStatus

    comp = {
        "schema_version": "1.0",
        "primary_objective": "accuracy",
        "evaluation_mode": "evaluated",
        "total_examples": 8,
        "examples_with_primary_metric": 5,
        "coverage_ratio": 5 / 8,
        "derivation_path": "none",
        "ranking_eligible": False,
        "warning_codes": ["MCI-002"],
        "per_metric_coverage": {"accuracy": {"present": 5, "total": 8, "ratio": 5 / 8}},
    }
    trial = TrialResult(
        trial_id="v1",
        config={"variant": "v1"},
        metrics={"accuracy": 0.9},
        status=TrialStatus.COMPLETED,
        duration=1.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        metadata={"comparability": comp},
    )
    metadata = (
        {"session_summary": {"ranking": {"comparability_mode": mode}}} if mode else {}
    )
    return OptimizationResult(
        trials=[trial],
        best_config={"variant": "v1"},
        best_score=0.9,
        optimization_id="opt-legacy",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(),
        metadata=metadata,
    )


@pytest.mark.parametrize("mode,kept", [("legacy", True), (None, True), ("warn", False)])
def test_rerank_uses_the_recorded_eligibility_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str | None, kept: bool
) -> None:
    store = tmp_path / "store"
    PersistenceManager(str(store)).save_result(_partial_winner_result(mode), "r")
    monkeypatch.chdir(tmp_path)
    out = CliRunner().invoke(
        cli, ["results", "rerank", "r", "-w", "accuracy=1.0", "-d", str(store)]
    )
    assert out.exit_code == 0, out.output
    assert ("New Best Configuration" in out.output) is kept


def test_cancelled_finalize_clears_a_stale_stored_best() -> None:
    from traigent.api.types import OptimizationStatus as S
    from traigent.core.backend_session_manager import BackendSessionManager

    calls: list[dict] = []

    class _Storage:
        def finalize_session(self, session_id, status, **kw):  # noqa: ANN001, ANN003, ANN202
            calls.append({"status": status, **kw})

    mgr = BackendSessionManager.__new__(BackendSessionManager)
    mgr._backend_client = None
    mgr._local_storage = lambda: _Storage()  # type: ignore[method-assign]
    mgr._finalize_local_session_unguarded(
        "sid", S.CANCELLED, best_config=None, best_score=None, apply_best=True
    )
    assert calls == [
        {
            "status": "cancelled",
            "best_config": None,
            "best_score": None,
            "apply_best": True,
        }
    ]
