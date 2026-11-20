"""``traigent results show`` prints the per-example consistency line."""

from __future__ import annotations

from datetime import datetime

import pytest
from click.testing import CliRunner

from traigent.api.types import OptimizationResult, OptimizationStatus
from traigent.cli.main import cli
from traigent.utils.example_consistency import (
    build_consistency_report,
    summarize_example_outcomes,
)
from traigent.utils.persistence import PersistenceManager


def _result(report: object = None) -> OptimizationResult:
    return OptimizationResult(
        trials=[],
        best_config={"model": "m"},
        best_score=0.9,
        optimization_id="opt_cli_consistency",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=["accuracy"],
        algorithm="grid",
        timestamp=datetime.now(),
        example_consistency_report=report,  # type: ignore[arg-type]
    )


def _show(store, name: str) -> str:
    outcome = CliRunner().invoke(
        cli, ["results", "show", name, "--storage-dir", str(store)]
    )
    assert outcome.exit_code == 0, outcome.output
    return outcome.output


def test_show_prints_one_consistency_line_when_labels_exist(tmp_path) -> None:
    reps = [
        {"a": "pass", "b": "fail", "c": "pass"},
        {"a": "pass", "b": "fail", "c": "fail"},
        {"a": "pass", "b": "fail", "c": "pass"},
    ]
    stored = build_consistency_report(
        summarize_example_outcomes(reps), reps=3, config_hash="abc"
    )
    store = tmp_path / "store"
    PersistenceManager(str(store)).save_result(_result(stored), name="with_labels")

    output = _show(store, "with_labels")

    lines = [ln for ln in output.splitlines() if ln.startswith("Consistency:")]
    assert lines == [
        "Consistency: 1 consistent_pass, 1 consistent_fail, 1 flaky (winner, 3 reps)"
    ]


def test_show_prints_nothing_without_labels(tmp_path) -> None:
    store = tmp_path / "store"
    PersistenceManager(str(store)).save_result(_result(), name="no_labels")

    output = _show(store, "no_labels")

    assert "Consistency" not in output
    assert "Summary" in output


@pytest.mark.parametrize("junk", ["not a dict", 7, {"examples": "bad"}])
def test_malformed_stored_value_prints_nothing(tmp_path, junk) -> None:
    store = tmp_path / "store"
    PersistenceManager(str(store)).save_result(_result(junk), name="junk")

    assert "Consistency" not in _show(store, "junk")
