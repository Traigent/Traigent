"""Results commands: not-found exits non-zero and listed local names resolve.

Regression tests for #2507 and #2643.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.storage.local_storage import LocalStorageManager


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def local_store(tmp_path):
    store = tmp_path / "store"
    storage = LocalStorageManager(str(store))
    session_id = storage.create_session(
        "local_func", optimization_config={"algorithm": "random"}
    )
    storage.add_trial_result(session_id, {"temperature": 0.2}, 0.5)
    storage.add_trial_result(session_id, {"temperature": 0.7}, 0.91)
    storage.finalize_session(session_id)
    return store, session_id


@pytest.mark.parametrize(
    "args",
    [
        ["results", "show", "no_such"],
        ["results", "compare", "no_such1", "no_such2"],
        ["results", "rerank", "no_such", "-w", "accuracy=1"],
        ["export", "no_such"],
        ["plot", "no_such"],
    ],
)
def test_not_found_exits_nonzero(runner, tmp_path, args):
    result = runner.invoke(cli, [*args, "-d", str(tmp_path / "empty")])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
    assert "traigent results list" in result.output
    assert "'traigent results' " not in result.output


def test_every_listed_name_round_trips_show_and_export(runner, local_store, tmp_path):
    store, session_id = local_store
    listed = runner.invoke(cli, ["results", "list", "-d", str(store)])
    name = f"local:{session_id}"
    assert name in listed.output.replace("\n", "")

    shown = runner.invoke(cli, ["results", "show", name, "-d", str(store)])
    assert shown.exit_code == 0, shown.output
    assert "not found" not in shown.output.lower()

    out = tmp_path / "cand.json"
    exported = runner.invoke(cli, ["export", name, "-d", str(store), "-o", str(out)])
    assert exported.exit_code == 0, exported.output
    data = json.loads(out.read_text())
    assert data["config"] == {"temperature": 0.7}


def test_default_storage_dir_matches_results_list(
    runner, local_store, tmp_path, monkeypatch
):
    """A name printed by `results list` resolves without -d (#2507)."""
    store, session_id = local_store
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    monkeypatch.chdir(workdir)  # no ./.traigent here
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(store))
    name = f"local:{session_id}"
    listed = runner.invoke(cli, ["results", "list"])
    assert name in listed.output.replace("\n", "")
    for args in (
        ["results", "show", name],
        ["results", "compare", name, name],
        ["results", "rerank", name, "-w", "score=1"],
        ["export", name, "-o", str(tmp_path / "out.json")],
        ["plot", name, "-p", "pareto"],
    ):
        res = runner.invoke(cli, args)
        assert "not found" not in res.output.lower(), (args, res.output)


@pytest.mark.parametrize("plot_type", ["progress", "pareto", "report"])
def test_plot_works_for_completed_local_session(runner, local_store, plot_type):
    """The local-session adapter must be plottable (#2507 review)."""
    store, session_id = local_store
    res = runner.invoke(
        cli,
        ["plot", f"local:{session_id}", "-d", str(store), "-p", plot_type],
    )
    assert res.exit_code == 0, res.output
    assert "Error generating plot" not in res.output


def test_report_plot_handles_missing_duration(local_store, monkeypatch):
    from traigent.cli import main as cli_main

    store, session_id = local_store
    result = cli_main._load_cli_result(f"local:{session_id}", str(store))
    result.duration = None
    from traigent.visualization.plots import create_quick_plot

    assert "Duration: N/A" in create_quick_plot(result, "report")
