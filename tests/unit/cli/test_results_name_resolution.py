"""End-to-end CLI tests: every name ``results list`` prints must be usable.

``traigent results list`` prints local sessions as ``local:<session_id>``.
``results show``, ``export``, ``plot`` (and ``compare`` / ``rerank``) must
resolve those names through one shared resolver, fail with a non-zero exit code
when a name cannot be resolved, and never read outside the results directory.
All data is synthetic and lives in a temporary results directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.storage.local_storage import LocalStorageManager


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_session(store: Path, function_name: str, scores: list[float]) -> str:
    storage = LocalStorageManager(str(store))
    session_id = storage.create_session(
        function_name, optimization_config={"algorithm": "random"}
    )
    for index, score in enumerate(scores):
        storage.add_trial_result(session_id, {"temperature": 0.1 * (index + 1)}, score)
    storage.finalize_session(session_id)
    return session_id


@pytest.fixture
def store(tmp_path: Path) -> Path:
    path = tmp_path / "results"
    _make_session(path, "triage_agent", [0.5, 0.9, 0.7])
    return path


def _printed_names(runner: CliRunner, store: Path) -> list[str]:
    output = runner.invoke(cli, ["results", "list", "-d", str(store)]).output
    return re.findall(r"local:[A-Za-z0-9_\-]+", output)


class TestListedNamesRoundTrip:
    def test_listed_name_is_accepted_by_show_export_and_plot(
        self, runner: CliRunner, store: Path, tmp_path: Path
    ) -> None:
        names = _printed_names(runner, store)
        assert len(names) == 1
        name = names[0]

        shown = runner.invoke(cli, ["results", "show", name, "-d", str(store)])
        assert shown.exit_code == 0, shown.output
        assert "not found" not in shown.output.lower()
        assert "triage_agent" in shown.output

        out = tmp_path / "cand.json"
        exported = runner.invoke(
            cli, ["export", name, "-d", str(store), "-o", str(out)]
        )
        assert exported.exit_code == 0, exported.output
        data = json.loads(out.read_text())
        assert data["config"] == {"temperature": pytest.approx(0.2)}
        assert data["function_name"] == "triage_agent"
        assert data["metrics"]["score"] == pytest.approx(0.9)

        plotted = runner.invoke(cli, ["plot", name, "-d", str(store)])
        assert plotted.exit_code == 0, plotted.output
        assert "not found" not in plotted.output.lower()

    def test_bare_session_id_and_unique_prefix_are_accepted(
        self, runner: CliRunner, store: Path
    ) -> None:
        name = _printed_names(runner, store)[0]
        session_id = name.removeprefix("local:")
        for ident in (session_id, name[: len("local:") + 12], session_id[:12]):
            result = runner.invoke(cli, ["results", "show", ident, "-d", str(store)])
            assert result.exit_code == 0, (ident, result.output)
            assert "not found" not in result.output.lower(), ident
            assert "triage_agent" in result.output, ident

    def test_export_default_output_name_has_no_colon(
        self,
        runner: CliRunner,
        store: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        name = _printed_names(runner, store)[0]
        result = runner.invoke(cli, ["export", name, "-d", str(store)])
        assert result.exit_code == 0, result.output
        written = list((tmp_path / "configs").glob("*.json"))
        assert len(written) == 1
        assert ":" not in written[0].name


class TestNotFoundExitsNonZero:
    @pytest.mark.parametrize(
        "argv",
        [
            ["results", "show"],
            ["export"],
            ["plot"],
        ],
    )
    def test_unknown_name_exits_nonzero_and_names_it(
        self, runner: CliRunner, store: Path, argv: list[str]
    ) -> None:
        result = runner.invoke(cli, [*argv, "local:does-not-exist", "-d", str(store)])
        assert result.exit_code != 0
        assert "local:does-not-exist" in result.output
        assert "traigent results list" in result.output

    def test_export_unknown_name_writes_no_file(
        self, runner: CliRunner, store: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "cand.json"
        result = runner.invoke(
            cli, ["export", "nope", "-d", str(store), "-o", str(out)]
        )
        assert result.exit_code != 0
        assert not out.exists()

    def test_export_without_best_config_exits_nonzero(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        empty = tmp_path / "results"
        session_id = LocalStorageManager(str(empty)).create_session("f")
        out = tmp_path / "cand.json"
        result = runner.invoke(
            cli,
            ["export", f"local:{session_id}", "-d", str(empty), "-o", str(out)],
        )
        assert result.exit_code != 0
        assert not out.exists()


class TestAmbiguousPrefix:
    def test_ambiguous_prefix_is_rejected_with_candidates(
        self, runner: CliRunner, store: Path
    ) -> None:
        second = _make_session(store, "triage_agent", [0.4])
        first = _printed_names(runner, store)
        assert len(first) == 2
        ids = [n.removeprefix("local:") for n in first]
        common = ""
        for a, b in zip(*ids, strict=False):
            if a != b:
                break
            common += a
        assert common, "synthetic sessions should share a timestamp prefix"

        result = runner.invoke(
            cli, ["results", "show", f"local:{common}", "-d", str(store)]
        )
        assert result.exit_code != 0
        assert "ambiguous" in result.output.lower()
        for session_id in ids:
            assert session_id in result.output
        assert second in result.output


class TestPathHandling:
    def test_traversal_in_name_cannot_read_outside_results_dir(
        self, runner: CliRunner, store: Path, tmp_path: Path
    ) -> None:
        # A syntactically valid session file *outside* the results dir.
        outside = tmp_path / "outside"
        outside_id = _make_session(outside, "secret_agent", [1.0])
        target = outside / "sessions" / outside_id

        for ident in (
            f"local:../../outside/sessions/{outside_id}",
            f"local:{target}",
            f"../outside/sessions/{outside_id}",
        ):
            result = runner.invoke(cli, ["results", "show", ident, "-d", str(store)])
            assert result.exit_code != 0, (ident, result.output)
            assert "Best Configuration" not in result.output
            assert "Summary" not in result.output
