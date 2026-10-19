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
