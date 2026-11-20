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

    lines = [ln for ln in output.splitlines() if ln.startswith("Repeat outcomes")]
    assert lines == [
        "Repeat outcomes (winner, 3 reps): 1 all_pass, 1 all_fail, 1 mixed, "
        "0 errored; all_pass at 3/3 means a 95% interval of 0.44-1.00 on the "
        "pass rate"
    ]


def test_show_prints_nothing_without_labels(tmp_path) -> None:
    store = tmp_path / "store"
    PersistenceManager(str(store)).save_result(_result(), name="no_labels")

    output = _show(store, "no_labels")

    assert "Repeat outcomes" not in output
    assert "Summary" in output


def _entry(**overrides: object) -> dict:
    entry = {
        "label": "all_pass",
        "attempted": 3,
        "passes": 3,
        "fails": 0,
        "errored": 0,
        "pass_rate_low": 0.439,
        "pass_rate_high": 1.0,
    }
    entry.update(overrides)
    return entry


def _report(**overrides: object) -> dict:
    report = {
        "counts": {},
        "examples": {"a": _entry()},
        "reps": 3,
        "config_hash": "abc",
    }
    report.update(overrides)
    return report


_MALFORMED = [
    pytest.param("not a dict", id="string"),
    pytest.param(7, id="int"),
    pytest.param([], id="list"),
    pytest.param({"examples": "bad"}, id="examples-not-a-mapping"),
    pytest.param(_report(examples={"a": 5}), id="entry-not-a-mapping"),
    pytest.param(_report(examples={"a": [1, 2]}), id="entry-is-a-list"),
    pytest.param(_report(examples={"a": _entry(passes="3")}), id="count-is-a-string"),
    pytest.param(_report(examples={"a": _entry(fails=1.5)}), id="count-is-a-float"),
    pytest.param(_report(examples={"a": _entry(passes=True)}), id="count-is-a-bool"),
    pytest.param(_report(examples={"a": _entry(errored=-1)}), id="count-is-negative"),
    pytest.param(_report(examples={"a": _entry(attempted=None)}), id="count-missing"),
    pytest.param(_report(examples={"a": _entry(label="flaky")}), id="unknown-label"),
    pytest.param(_report(examples={"a": _entry(pass_rate_low=7)}), id="rate-too-big"),
    pytest.param(_report(reps=True), id="reps-is-a-bool"),
    pytest.param(_report(reps="3"), id="reps-is-a-string"),
    pytest.param(_report(reps=-1), id="reps-is-negative"),
    pytest.param(_report(config_hash=5), id="hash-is-not-a-string"),
]


@pytest.mark.parametrize("junk", _MALFORMED)
def test_malformed_stored_report_is_a_note_not_a_crash(tmp_path, junk) -> None:
    store = tmp_path / "store"
    PersistenceManager(str(store)).save_result(_result(junk), name="junk")

    output = _show(store, "junk")  # exit code 0 is asserted inside

    assert "the stored report is unreadable" in output
    assert "Summary" in output


@pytest.mark.parametrize("junk", _MALFORMED)
def test_accessor_reads_a_malformed_report_as_no_report(junk) -> None:
    report = _result(junk).example_consistency()

    assert report["examples"] == {}
    assert report["reps"] == 0
